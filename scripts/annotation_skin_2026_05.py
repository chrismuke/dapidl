"""Skin slides annotation analysis — mirror of annotation_run_2026_05.py for breast.

Methods:
- CellTypist Adult_Human_Skin.pkl (no majority_voting; skin-specific reference)
- scType custom_default (universal markers)
- SingleR Blueprint (universal reference)
+ 2-method (CT+scType, CT+SingleR, scType+SingleR) and 3-method confidence-weighted consensus.

GROUND TRUTH for skin:
- Read `label2` column directly from the per-slide zarr table (NOT a parquet — STHELAR
  ships breast metadata as parquets but skin stays in the zarr table_nuclei).
- Map label2 -> COARSE-5 + MEDIUM-12 via SKIN_LABEL2_TO_* in scripts/skin_dapi_lmdb.py.

OUTPUTS:
    pipeline_output/annotation_skin_2026_05/
        per_slide/{slide}.json          predictions cache
        per_slide/{slide}_gt.json       GT cache
        coarse_metrics.parquet          per slide × method × COARSE-5
        medium_metrics.parquet          per slide × method × MEDIUM-12
        consensus_results.parquet       all 2/3-method consensus combinations
        summary.md                      headline table
"""
from __future__ import annotations
import gc
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import polars as pl
from loguru import logger
from sklearn.metrics import f1_score, precision_recall_fscore_support

from annotation_benchmark_2026_03 import (
    preprocess_adata,
    run_sctype, run_singler,
    get_default_markers,
)
from annotation_run_2026_05 import (
    run_celltypist_no_mv, run_methods_on_slide,
    consensus_confidence_weighted,
    map_to_tier, macro_f1, per_class_f1,
)
from dapidl.ontology.cl_mapper import get_mapper
from dapidl.ontology.training_tiers import COARSE_NAMES, MEDIUM_NAMES
from skin_dapi_lmdb import (
    SKIN_LABEL2_TO_COARSE, SKIN_LABEL2_TO_MEDIUM,
)

# ---------------------------------------------------------------------------
# COARSE/MEDIUM label sets — skin uses 5-class with Neural for Melanocyte.
# ---------------------------------------------------------------------------
COARSE_5 = ["Endothelial", "Epithelial", "Immune", "Stromal", "Neural"]
MEDIUM_12 = list(MEDIUM_NAMES)

STHELAR_BASE = Path("/mnt/work/datasets/STHELAR/sdata_slides")
SLIDES = ["skin_s1", "skin_s2", "skin_s3", "skin_s4"]

OUT_DIR = Path("/mnt/work/git/dapidl/pipeline_output/annotation_skin_2026_05")
PER_SLIDE_DIR = OUT_DIR / "per_slide"
PER_SLIDE_DIR.mkdir(parents=True, exist_ok=True)

# OOM safeguard: subsample to 100k cells if cells*genes > 1e9 (skin slides are
# small, so this rarely triggers — but keep parity with breast runner).
MAX_CELLS_X_GENES = 1_000_000_000
SUBSAMPLE_TO = 100_000


def load_skin_adata(slide_name: str) -> ad.AnnData:
    """Load skin slide raw counts from zarr table_nuclei.

    Probes both layouts that exist in STHELAR's distribution:
    - skin_s1/s2: `sdata_<slide>.zarr/sdata_<slide>.zarr/tables/table_nuclei` (nested)
    - skin_s3/s4: `sdata_<slide>.zarr/tables/table_nuclei` (single-level)
    """
    candidates = [
        STHELAR_BASE / f"sdata_{slide_name}.zarr" / f"sdata_{slide_name}.zarr"
            / "tables" / "table_nuclei",
        STHELAR_BASE / f"sdata_{slide_name}.zarr" / "tables" / "table_nuclei",
    ]
    table_path = next((p for p in candidates if p.exists()), None)
    if table_path is None:
        raise FileNotFoundError(
            f"{slide_name}: no table_nuclei found in {[str(p) for p in candidates]}")
    logger.info(f"Loading STHELAR {slide_name} from {table_path}")
    adata = ad.read_zarr(str(table_path))

    # Use raw counts as X
    if "count" in adata.layers:
        adata.X = adata.layers["count"]
        del adata.layers["count"]
    if "log_norm" in adata.layers:
        del adata.layers["log_norm"]

    if "label2" not in adata.obs.columns:
        raise SystemExit(f"{slide_name}: missing label2 column")

    # Drop cells where label2 is None / less10 (we want benchmarkable cells)
    l2 = adata.obs["label2"].astype(str).values
    mappable = np.array([
        SKIN_LABEL2_TO_COARSE.get(v) is not None for v in l2
    ])
    n_before = len(adata)
    adata = adata[mappable].copy()
    logger.info(f"  {slide_name}: {n_before:,} -> {len(adata):,} cells "
                f"(dropped {n_before - len(adata):,} unmappable/less10)")
    adata.obs["dataset"] = slide_name
    return adata


def get_gt_skin(adata) -> tuple[np.ndarray, np.ndarray]:
    l2 = adata.obs["label2"].astype(str).values
    gt_coarse = np.array(
        [SKIN_LABEL2_TO_COARSE.get(v) or "Unknown" for v in l2],
        dtype=object)
    gt_medium = np.array(
        [SKIN_LABEL2_TO_MEDIUM.get(v) or "Unknown" for v in l2],
        dtype=object)
    return gt_coarse, gt_medium


def maybe_subsample(adata, slide):
    n_cells, n_genes = adata.shape
    if n_cells * n_genes <= MAX_CELLS_X_GENES:
        return adata, None
    rng = np.random.default_rng(42)
    keep = rng.choice(n_cells, SUBSAMPLE_TO, replace=False)
    keep_sorted = np.sort(keep)
    logger.info(f"  {slide}: subsampling {n_cells:,} -> {SUBSAMPLE_TO:,} "
                f"(cells*genes={n_cells * n_genes / 1e9:.2f}e9 > 1e9)")
    return adata[keep_sorted].copy(), keep_sorted


def run_one_slide(slide: str, methods_to_run, mapper):
    """Run methods, return (slide_results, gt_coarse, gt_medium)."""
    cache_path = PER_SLIDE_DIR / f"{slide}.json"
    gt_path = PER_SLIDE_DIR / f"{slide}_gt.json"

    if cache_path.exists() and gt_path.exists():
        logger.info(f"=== {slide}: cached, loading ===")
        with open(cache_path) as f:
            slide_results = json.load(f)
        for m in slide_results:
            slide_results[m]["raw_preds"] = np.array(
                slide_results[m]["raw_preds"], dtype=object)
            if slide_results[m].get("conf") is not None:
                slide_results[m]["conf"] = np.array(
                    slide_results[m]["conf"], dtype=np.float32)
        with open(gt_path) as f:
            gt_data = json.load(f)
        gt_coarse = np.array(gt_data["coarse"], dtype=object)
        gt_medium = np.array(gt_data["medium"], dtype=object)
        return slide_results, gt_coarse, gt_medium

    logger.info(f"=== {slide}: running ===")
    adata_raw = load_skin_adata(slide)
    adata_raw, keep_idx = maybe_subsample(adata_raw, slide)

    gt_coarse, gt_medium = get_gt_skin(adata_raw)

    adata_pp = preprocess_adata(adata_raw)
    slide_results = run_methods_on_slide(adata_pp, adata_raw, methods_to_run)

    # Persist GT + predictions
    cache_data = {}
    for m, vals in slide_results.items():
        cache_data[m] = {
            "raw_preds": vals["raw_preds"].tolist(),
            "conf": (vals["conf"].tolist()
                     if vals.get("conf") is not None else None),
        }
    with open(cache_path, "w") as f:
        json.dump(cache_data, f)
    with open(gt_path, "w") as f:
        json.dump({"coarse": gt_coarse.tolist(),
                   "medium": gt_medium.tolist()}, f)

    del adata_pp, adata_raw
    gc.collect()
    return slide_results, gt_coarse, gt_medium


def evaluate_method(slide, method, raw_preds, gt_coarse, gt_medium, mapper,
                    coarse_rows, medium_rows):
    pred_coarse = map_to_tier(raw_preds, "coarse", mapper)
    pred_medium = map_to_tier(raw_preds, "medium", mapper)

    f1_c = macro_f1(gt_coarse, pred_coarse, COARSE_5)
    f1_m = macro_f1(gt_medium, pred_medium, MEDIUM_12)
    pc_c = per_class_f1(gt_coarse, pred_coarse, COARSE_5)
    pc_m = per_class_f1(gt_medium, pred_medium, MEDIUM_12)

    coarse_rows.append({
        "slide": slide, "method": method,
        "macro_f1": f1_c,
        **{f"f1_{k}": v for k, v in pc_c.items()},
    })
    medium_rows.append({
        "slide": slide, "method": method,
        "macro_f1": f1_m,
        **{f"f1_{k}": v for k, v in pc_m.items()},
    })
    logger.info(f"  {method:50s} coarse_F1={f1_c:.3f}  medium_F1={f1_m:.3f}")


def evaluate_consensus(slide, methods_in_consensus, slide_results, gt_coarse,
                        gt_medium, mapper, consensus_rows):
    """Consensus over a chosen set of methods."""
    raw_list = [slide_results[m]["raw_preds"] for m in methods_in_consensus]
    conf_list = [slide_results[m].get("conf") for m in methods_in_consensus]

    coarse_preds_list = [map_to_tier(rp, "coarse", mapper) for rp in raw_list]
    medium_preds_list = [map_to_tier(rp, "medium", mapper) for rp in raw_list]

    cons_coarse = consensus_confidence_weighted(coarse_preds_list, conf_list)
    cons_medium = consensus_confidence_weighted(medium_preds_list, conf_list)

    f1_c = macro_f1(gt_coarse, cons_coarse, COARSE_5)
    f1_m = macro_f1(gt_medium, cons_medium, MEDIUM_12)

    consensus_rows.append({
        "slide": slide,
        "n_methods": len(methods_in_consensus),
        "methods": "+".join(sorted(methods_in_consensus)),
        "coarse_f1": f1_c,
        "medium_f1": f1_m,
    })


def main():
    mapper = get_mapper()

    sctype_markers = {}
    try:
        m = get_default_markers()
        if m:
            sctype_markers["custom_default"] = m
    except Exception as e:
        logger.warning(f"sctype markers unavailable: {e}")

    methods_to_run = []
    methods_to_run.append(("celltypist", "Adult_Human_Skin.pkl"))
    if "custom_default" in sctype_markers:
        methods_to_run.append(("sctype",
                               (sctype_markers["custom_default"], "custom_default")))
    methods_to_run.append(("singler", "blueprint"))

    coarse_rows = []
    medium_rows = []
    consensus_rows = []

    for slide in SLIDES:
        try:
            slide_results, gt_coarse, gt_medium = run_one_slide(
                slide, methods_to_run, mapper)
        except Exception as e:
            logger.error(f"{slide}: FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue

        if not slide_results:
            logger.warning(f"  {slide}: no method produced predictions")
            continue

        # 1) Per-method evaluation
        for method, vals in slide_results.items():
            evaluate_method(slide, method, vals["raw_preds"],
                            gt_coarse, gt_medium, mapper,
                            coarse_rows, medium_rows)

        # 2) Consensus over all 2- and 3-method subsets
        method_names = list(slide_results.keys())
        for k in (2, 3):
            for combo in combinations(method_names, k):
                if len(combo) > len(method_names):
                    continue
                evaluate_consensus(slide, list(combo), slide_results,
                                   gt_coarse, gt_medium, mapper,
                                   consensus_rows)

    # ---------------------------------------------------------------------
    # Persist tables
    # ---------------------------------------------------------------------
    if coarse_rows:
        pl.DataFrame(coarse_rows).write_parquet(OUT_DIR / "coarse_metrics.parquet")
    if medium_rows:
        pl.DataFrame(medium_rows).write_parquet(OUT_DIR / "medium_metrics.parquet")
    if consensus_rows:
        pl.DataFrame(consensus_rows).write_parquet(
            OUT_DIR / "consensus_results.parquet")

    # ---------------------------------------------------------------------
    # Summary markdown
    # ---------------------------------------------------------------------
    lines = ["# Skin annotation results — 2026-05-05", ""]
    if coarse_rows:
        df = pl.DataFrame(coarse_rows)
        lines.append("## Per-method × per-slide (COARSE-5)")
        lines.append("")
        lines.append("| Slide | Method | macro_F1 |")
        lines.append("|---|---|---|")
        for r in df.iter_rows(named=True):
            lines.append(f"| {r['slide']} | {r['method']} | {r['macro_f1']:.3f} |")
        lines.append("")
        lines.append("## Method ranking (mean across 4 slides, COARSE-5)")
        lines.append("")
        ranked = (df.group_by("method")
                    .agg(pl.col("macro_f1").mean().alias("mean_f1"))
                    .sort("mean_f1", descending=True))
        lines.append("| Method | Mean F1 |")
        lines.append("|---|---|")
        for r in ranked.iter_rows(named=True):
            lines.append(f"| {r['method']} | {r['mean_f1']:.3f} |")
        lines.append("")
    if consensus_rows:
        cdf = pl.DataFrame(consensus_rows)
        ranked = (cdf.group_by(["methods", "n_methods"])
                     .agg(pl.col("coarse_f1").mean().alias("mean_coarse"),
                          pl.col("medium_f1").mean().alias("mean_medium"))
                     .sort("mean_coarse", descending=True))
        lines.append("## Consensus ranking (mean across 4 slides)")
        lines.append("")
        lines.append("| Methods | n | Coarse F1 | Medium F1 |")
        lines.append("|---|---|---|---|")
        for r in ranked.iter_rows(named=True):
            lines.append(f"| {r['methods']} | {r['n_methods']} | "
                         f"{r['mean_coarse']:.3f} | {r['mean_medium']:.3f} |")
        lines.append("")
    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    logger.info(f"wrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
