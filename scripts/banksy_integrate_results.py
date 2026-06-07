"""Integrate BANKSY+scType predictions into the annotation_run_2026_05 outputs.

Reads per-slide BANKSY JSON files emitted by banksy_breast_worker.py and:
1. Appends a BANKSY method row to coarse_metrics.parquet + medium_metrics.parquet
   (per slide × method × tier).
2. Adds BANKSY-inclusive consensus rows to consensus_results.parquet
   (2-method: CT+BANKSY, scType+BANKSY, SR+BANKSY; 3-method: CT+scType+BANKSY,
   CT+SR+BANKSY, scType+SR+BANKSY; 4-method: CT+scType+SR+BANKSY).
3. Refreshes summary.md with BANKSY rankings highlighted.

Idempotent — re-running this overwrites the BANKSY-suffixed rows but doesn't
duplicate them.
"""
from __future__ import annotations
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl
from loguru import logger

from annotation_run_2026_05 import (
    consensus_confidence_weighted, map_to_tier, macro_f1, per_class_f1,
)
from dapidl.ontology.cl_mapper import get_mapper
from dapidl.ontology.training_tiers import COARSE_NAMES, MEDIUM_NAMES
from tier_macro import present_class_macro  # GT-present-class macro (absent -> N/A)

OUT_DIR = Path("/mnt/work/git/dapidl/pipeline_output/annotation_run_2026_05")
PER_SLIDE = OUT_DIR / "per_slide"
SLIDES = ["rep1", "rep2", "breast_s0", "breast_s1", "breast_s3", "breast_s6"]
COARSE_4 = ["Endothelial", "Epithelial", "Immune", "Stromal"]
MEDIUM_12 = list(MEDIUM_NAMES)


def load_existing_methods(slide):
    """Load existing 3-method cache + GT from per_slide JSON."""
    cache_path = PER_SLIDE / f"{slide}.json"
    gt_path = PER_SLIDE / f"{slide}_gt.json"
    if not cache_path.exists() or not gt_path.exists():
        return None, None, None
    with open(cache_path) as f:
        sd = json.load(f)
    for m in sd:
        sd[m]["raw_preds"] = np.array(sd[m]["raw_preds"], dtype=object)
        if sd[m].get("conf") is not None:
            sd[m]["conf"] = np.array(sd[m]["conf"], dtype=np.float32)
    with open(gt_path) as f:
        g = json.load(f)
    return (sd,
            np.array(g["coarse"], dtype=object),
            np.array(g["medium"], dtype=object))


def load_banksy_preds(slide):
    p = PER_SLIDE / f"{slide}_banksy.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    if "raw_preds" not in data:
        logger.warning(f"  {slide}: BANKSY result has no preds (error: {data.get('error')})")
        return None
    return {
        "raw_preds": np.array(data["raw_preds"], dtype=object),
        "conf": np.array(data["conf"], dtype=np.float32),
        "method": data["method"],
    }


def eval_method(method_name, raw_preds, gt_coarse, gt_medium, mapper):
    """Compute present-class macro + per-class F1 for one method.

    Macro averages over GT-present classes only (absent classes -> N/A), so a
    method isn't penalized for a class the GT can't measure (e.g. STHELAR
    Tangram GT lacks Endothelial on s1/s3/s6). The full coarse-4 macro is kept
    as ``coarse_f1_all`` for reference."""
    pc = map_to_tier(raw_preds, "coarse", mapper)
    pm = map_to_tier(raw_preds, "medium", mapper)
    coarse_pc = per_class_f1(gt_coarse, pc, COARSE_4)
    medium_pc = per_class_f1(gt_medium, pm, MEDIUM_12)
    c_macro, c_per_na, c_present = present_class_macro(gt_coarse, coarse_pc, COARSE_4)
    m_macro, _, _ = present_class_macro(gt_medium, medium_pc, MEDIUM_12)
    return {
        "coarse_f1": c_macro,
        "coarse_f1_all": macro_f1(gt_coarse, pc, COARSE_4),
        "medium_f1": m_macro,
        "coarse_per_class": c_per_na,
        "n_present": len(c_present),
        "n_eval": len(gt_coarse),
    }


def main():
    """Rebuild the metrics parquets from per_slide JSONs (existing methods +
    BANKSY where available). Writes fresh — does NOT append, because the
    previous append_unique had a (slide AND method) bug that destructively
    dropped existing rows."""
    mapper = get_mapper()

    coarse_rows = []
    medium_rows = []
    consensus_rows = []

    for slide in SLIDES:
        slide_results, gt_coarse, gt_medium = load_existing_methods(slide)
        if slide_results is None:
            logger.warning(f"{slide}: missing cache, skipping")
            continue
        banksy = load_banksy_preds(slide)

        # Optionally splice BANKSY in if cell counts match
        if banksy is not None:
            n_existing = len(next(iter(slide_results.values()))["raw_preds"])
            n_banksy = len(banksy["raw_preds"])
            if n_existing != n_banksy:
                logger.warning(f"{slide}: BANKSY length mismatch "
                               f"({n_banksy} vs {n_existing}); skipping BANKSY only")
            else:
                slide_results[banksy["method"]] = {"raw_preds": banksy["raw_preds"],
                                                    "conf": banksy["conf"]}

        # Compute per-method metrics for *every* method present (including BANKSY
        # if it spliced in successfully). This rebuilds the parquet from scratch.
        for m_name, m_data in slide_results.items():
            ev = eval_method(m_name, m_data["raw_preds"], gt_coarse, gt_medium, mapper)
            coarse_rows.append({"slide": slide, "method": m_name,
                                "f1_macro": ev["coarse_f1"],
                                "f1_macro_all4": ev["coarse_f1_all"],
                                "n_present": ev["n_present"],
                                "n_eval": ev["n_eval"],
                                **{f"f1_{k}": v for k, v in ev["coarse_per_class"].items()}})
            medium_rows.append({"slide": slide, "method": m_name,
                                "f1_macro": ev["medium_f1"],
                                "n_eval": ev["n_eval"]})
            logger.info(f"  {slide:>10}: {m_name:<35} coarse_F1={ev['coarse_f1']:.3f} medium_F1={ev['medium_f1']:.3f}")

        # Compute consensus over all 2/3/4-method combinations on this slide.
        # Note: emits ALL combos, not just BANKSY-inclusive ones — we're
        # rebuilding the full table.
        all_methods = list(slide_results.keys())
        for k in (2, 3, 4):
            if k > len(all_methods):
                continue
            for combo in combinations(all_methods, k):
                raws = [slide_results[m]["raw_preds"] for m in combo]
                confs = [slide_results[m].get("conf") for m in combo]
                cps_c = [map_to_tier(r, "coarse", mapper) for r in raws]
                cps_m = [map_to_tier(r, "medium", mapper) for r in raws]
                cc = consensus_confidence_weighted(cps_c, confs)
                cm = consensus_confidence_weighted(cps_m, confs)
                combo_str = "+".join(sorted(combo))
                consensus_rows.append({
                    "slide": slide, "n_methods": len(combo),
                    "combo": combo_str, "tier": "coarse",
                    "f1_macro": present_class_macro(
                        gt_coarse, per_class_f1(gt_coarse, cc, COARSE_4), COARSE_4)[0],
                })
                consensus_rows.append({
                    "slide": slide, "n_methods": len(combo),
                    "combo": combo_str, "tier": "medium",
                    "f1_macro": present_class_macro(
                        gt_medium, per_class_f1(gt_medium, cm, MEDIUM_12), MEDIUM_12)[0],
                })

    # Write fresh parquets (full rebuild, not append). Use diagonal_relaxed in
    # case some slides have BANKSY rows with extra per-class columns and others
    # don't.
    def write_parquet(path, rows):
        if not rows:
            logger.warning(f"  skip {path}: no rows")
            return
        df = pl.from_dicts(rows, infer_schema_length=None)
        df.write_parquet(path)
        logger.info(f"  wrote {path}: {len(df):,} rows ({len(df.columns)} cols)")

    write_parquet(OUT_DIR / "coarse_metrics.parquet", coarse_rows)
    write_parquet(OUT_DIR / "medium_metrics.parquet", medium_rows)
    write_parquet(OUT_DIR / "consensus_results.parquet", consensus_rows)

    # Refresh summary.md tail with BANKSY-inclusive ranking
    if coarse_rows:
        df = pl.read_parquet(OUT_DIR / "coarse_metrics.parquet")
        ranked = (df.group_by("method")
                    .agg(pl.col("f1_macro").mean().alias("mean_f1"))
                    .sort("mean_f1", descending=True))
        cdf = pl.read_parquet(OUT_DIR / "consensus_results.parquet")
        # Pivot tier to columns so we can show coarse vs medium side-by-side
        cranked = (cdf.group_by(["combo", "n_methods", "tier"])
                      .agg(pl.col("f1_macro").mean().alias("mean_f1"))
                      .sort(["tier", "mean_f1"], descending=[False, True]))
        out = ["# Annotation results — BANKSY-INCLUSIVE refresh", ""]
        out.append("## Single-method ranking (mean F1 across breast slides, COARSE-4)")
        out.append("| Method | Mean F1 |")
        out.append("|---|---|")
        for r in ranked.iter_rows(named=True):
            out.append(f"| {r['method']} | {r['mean_f1']:.3f} |")
        out.append("")
        out.append("## Consensus ranking (mean across slides, by tier)")
        out.append("| Combo | n | Tier | Mean F1 |")
        out.append("|---|---|---|---|")
        for r in cranked.iter_rows(named=True):
            out.append(f"| {r['combo']} | {r['n_methods']} | "
                       f"{r['tier']} | {r['mean_f1']:.3f} |")
        out.append("")
        (OUT_DIR / "summary_with_banksy.md").write_text("\n".join(out))
        logger.info(f"  wrote {OUT_DIR / 'summary_with_banksy.md'}")


if __name__ == "__main__":
    main()
