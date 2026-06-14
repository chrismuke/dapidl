#!/usr/bin/env python3
"""Run all annotation methods on all breast datasets — comprehensive benchmark.

Wrapper around `scripts/annotation_benchmark_2026_03.py` that:
- Runs every CPU-bound method (CellTypist, SingleR, scType, SCINA, decoupler, popV)
  on all 6 breast datasets: rep1, rep2, STHELAR breast_s0, s1, s3, s6.
- Saves per-cell predictions (coarse) to NPZ for downstream consensus combination.
- Saves per-dataset metrics to parquet.

Usage:
    uv run python scripts/breast_full_annotation.py [--datasets rep1,rep2,...]

Output:
    pipeline_output/breast_annotation_full/
        predictions_<dataset>.npz       # per-cell coarse predictions per method
        metrics_<dataset>.json          # per-method metrics (no per-cell preds)
        master_metrics.parquet          # all (dataset, method) → metrics rows
        master_metrics.md               # human-readable table
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import psutil
from loguru import logger

# Dynamically import the existing benchmark module
SCRIPT_DIR = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "ab2026_03",
    SCRIPT_DIR / "annotation_benchmark_2026_03.py",
)
ab = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules["ab2026_03"] = ab
spec.loader.exec_module(ab)  # type: ignore[union-attr]

OUT_DIR = Path("/mnt/work/git/dapidl/pipeline_output/breast_annotation_full")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATASETS = ["rep1", "rep2", "breast_s0", "breast_s1", "breast_s3", "breast_s6"]


def _estimate_peak_ram_gb(n_cells: int, n_genes: int) -> float:
    """Estimate peak RAM in GB for full evaluate pipeline.

    Calibrated 2026-05-02 from observed crash: at 200k cells × 8264 genes the
    full sweep (5+ celltypist models + singler + scType + …) hit 56 GB before
    crashing. preprocess_adata densifies X (n*g*4B) AND keeps a copy in
    layers["raw"] (n*g*4B) → 2×. Each celltypist model's `predictions.to_adata()`
    creates ANOTHER copy that isn't released between models → ~3-4× more for
    the multi-model loop. Plus transient kNN graphs (~1×) per model.
    Empirical fit: peak ≈ 8 × n_cells × n_genes × 4B.
    """
    return 8.0 * n_cells * n_genes * 4 / 1024**3


def _check_ram_budget(needed_gb: float, headroom_gb: float = 8.0) -> None:
    """Pre-flight RAM check. Raise if not enough free memory + headroom.

    Why: systemd-oomd kills the entire user session (tmux + all Claude
    processes) when memory pressure exceeds 90% > 50% threshold for >20s.
    Better to fail fast than have the session nuked.
    """
    avail_gb = psutil.virtual_memory().available / 1024**3
    if needed_gb + headroom_gb > avail_gb:
        raise MemoryError(
            f"Estimated peak RAM {needed_gb:.1f} GB + {headroom_gb:.0f} GB headroom "
            f"exceeds available {avail_gb:.1f} GB. Free memory or pass --max-cells N "
            f"to subsample. (systemd-oomd kills the entire user session at 90% "
            f"pressure — better to fail fast.)"
        )
    logger.info(
        f"RAM pre-flight OK: need ~{needed_gb:.1f} GB peak, "
        f"have {avail_gb:.1f} GB available, {headroom_gb:.0f} GB headroom"
    )


def _stratified_subsample(adata, max_cells: int, label_col: str, seed: int = 42):
    """Class-stratified subsample to at most max_cells, preserving class balance.

    Why stratified: Epithelial dominates (~60% of breast tumor cells); a uniform
    subsample of 150k cells would still give >90k Epithelial and <100 Mast.
    Stratified subsample maintains relative proportions while guaranteeing rare
    classes have at least min_per_class cells for stable F1.
    """
    rng = np.random.default_rng(seed)

    if len(adata) <= max_cells:
        return adata

    labels = np.asarray(adata.obs[label_col].values)
    classes, counts = np.unique(labels, return_counts=True)
    n_classes = len(classes)

    # Allocate per-class quota proportional to original counts; but guarantee
    # min(50, original_count) cells per class so rare-class metrics stay stable.
    quotas = (counts / counts.sum() * max_cells).astype(int)
    min_per_class = 50
    for i, c in enumerate(counts):
        quotas[i] = max(quotas[i], min(min_per_class, c))
    # Final clamp to budget; trim from largest class first if over.
    while quotas.sum() > max_cells:
        idx = int(np.argmax(quotas))
        quotas[idx] -= 1

    keep_idx = []
    for cls, q in zip(classes, quotas):
        cls_idx = np.where(labels == cls)[0]
        keep_idx.append(rng.choice(cls_idx, size=min(q, len(cls_idx)), replace=False))
    keep = np.sort(np.concatenate(keep_idx))

    sub = adata[keep].copy()
    logger.info(
        f"Stratified subsample: {len(adata):,} → {len(sub):,} cells "
        f"({n_classes} classes preserved, min={min(quotas)}, max={max(quotas)})"
    )
    return sub


def evaluate_one(dataset: str, max_cells: int | None = None) -> dict:
    """Run all methods on one dataset, save predictions + metrics."""
    logger.info(f"\n{'='*80}\nLOADING {dataset}\n{'='*80}")
    if dataset.startswith("rep"):
        adata_raw = ab.load_xenium_adata(dataset)
    elif dataset.startswith("breast_s"):
        adata_raw = ab.load_sthelar_adata(dataset)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Subsample BEFORE preprocess_adata so both densification + kNN scale down.
    if max_cells is not None and len(adata_raw) > max_cells:
        gt_col = "gt_coarse" if "gt_coarse" in adata_raw.obs.columns else None
        if gt_col is None:
            # fall back to whichever obs column carries the broad label
            for c in ["coarse", "label1", "celltype_coarse"]:
                if c in adata_raw.obs.columns:
                    gt_col = c
                    break
        if gt_col is None:
            raise RuntimeError(
                f"{dataset}: cannot find a class column for stratified subsample"
            )
        adata_raw = _stratified_subsample(adata_raw, max_cells, gt_col)

    # Pre-flight RAM check using the (possibly subsampled) adata size
    needed = _estimate_peak_ram_gb(len(adata_raw), adata_raw.n_vars)
    _check_ram_budget(needed)

    adata_pp = ab.preprocess_adata(adata_raw)
    logger.info(f"{dataset}: {len(adata_pp)} cells, {adata_pp.n_vars} genes")

    # Align adata_raw to the cells that survived preprocess_adata's filter_cells.
    # Without this, popv (which operates on adata_raw) returns predictions with
    # the original cell count, while gt_coarse (taken from adata_pp) has the
    # filtered count → broadcast mismatch in compute_metrics.
    if len(adata_raw) != len(adata_pp):
        adata_raw = adata_raw[adata_pp.obs_names].copy()
        logger.info(f"  aligned adata_raw → {len(adata_raw)} cells (matches adata_pp)")

    # Run the existing per-method evaluator
    t0 = time.time()
    results = ab.evaluate_dataset(dataset, adata_raw, adata_pp)
    elapsed_total = time.time() - t0

    # Extract per-cell predictions to NPZ
    preds_npz: dict[str, np.ndarray] = {}
    confidence_npz: dict[str, np.ndarray] = {}
    method_metrics: dict[str, dict] = {}

    gt_coarse = adata_pp.obs["gt_coarse"].values
    preds_npz["__ground_truth__"] = np.asarray(gt_coarse)

    for method, r in results.items():
        if "error" in r:
            method_metrics[method] = {"error": r["error"]}
            continue
        raw = r.get("raw_result")
        if raw is None or raw.get("coarse_predictions") is None:
            continue
        # Save per-cell coarse predictions
        preds_npz[method] = np.asarray(raw["coarse_predictions"])
        # Save confidence if available (popV/CellTypist have these)
        if "confidence" in raw and raw["confidence"] is not None:
            confidence_npz[f"{method}_conf"] = np.asarray(raw["confidence"])
        # Strip raw_result for metrics save
        method_metrics[method] = {
            k: v for k, v in r.items()
            if k != "raw_result" and not isinstance(v, np.ndarray)
        }

    # Save NPZ (predictions per cell)
    npz_path = OUT_DIR / f"predictions_{dataset}.npz"
    np.savez_compressed(npz_path, **preds_npz, **confidence_npz)
    logger.info(f"Saved {len(preds_npz)-1} method predictions to {npz_path}")

    # Save metrics JSON
    metrics_path = OUT_DIR / f"metrics_{dataset}.json"
    metrics_path.write_text(json.dumps(method_metrics, indent=2, default=str))
    logger.info(f"Saved metrics to {metrics_path}")

    return {
        "dataset": dataset,
        "n_cells": len(adata_pp),
        "n_methods": len(preds_npz) - 1,
        "elapsed_s": round(elapsed_total, 1),
        "metrics_path": str(metrics_path),
        "predictions_path": str(npz_path),
        "method_metrics": method_metrics,
    }


def aggregate_master_table(results: list[dict]) -> None:
    """Build master metrics parquet + markdown."""
    rows = []
    for r in results:
        ds = r["dataset"]
        for method, m in r["method_metrics"].items():
            if "error" in m:
                rows.append({
                    "dataset": ds, "method": method,
                    "macro_f1": None, "accuracy": None,
                    "weighted_f1": None, "error": m["error"],
                })
                continue
            row = {
                "dataset": ds,
                "method": method,
                "macro_f1": m.get("f1_macro"),
                "accuracy": m.get("accuracy"),
                "weighted_f1": m.get("f1_weighted"),
                "runtime_s": m.get("runtime_s"),
                "error": None,
            }
            # Per-class F1
            for c in ab.COARSE_CLASSES:
                row[f"f1_{c}"] = m.get("per_class", {}).get(c, {}).get("f1")
            rows.append(row)

    df = pl.DataFrame(rows)
    df.write_parquet(OUT_DIR / "master_metrics.parquet")
    df.write_csv(OUT_DIR / "master_metrics.csv")

    # Markdown
    lines = [
        "# Breast Annotation — Master Metrics (all methods × all breast datasets)",
        "",
        f"**Datasets**: {df['dataset'].n_unique()} | **Methods**: {df['method'].n_unique()} | "
        f"**Runs**: {len(df)}",
        "",
        "## Top by macro F1 per dataset",
        "",
    ]
    for ds in sorted(df["dataset"].unique().to_list()):
        sub = df.filter(
            (pl.col("dataset") == ds) & pl.col("macro_f1").is_not_null()
        ).sort("macro_f1", descending=True)
        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| Method | Macro F1 | Accuracy | Endo | Epi | Imm | Str | s |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in sub.iter_rows(named=True):
            def _f(v):
                return f"{v:.3f}" if v is not None else "—"
            lines.append(
                f"| `{r['method']}` | {_f(r['macro_f1'])} | {_f(r['accuracy'])} | "
                f"{_f(r['f1_Endothelial'])} | {_f(r['f1_Epithelial'])} | "
                f"{_f(r['f1_Immune'])} | {_f(r['f1_Stromal'])} | "
                f"{r['runtime_s'] if r['runtime_s'] else '—'} |"
            )
        lines.append("")

    (OUT_DIR / "master_metrics.md").write_text("\n".join(lines))
    logger.info(f"Wrote master_metrics.{{parquet,csv,md}} to {OUT_DIR}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                    help="Comma-separated list of breast datasets")
    ap.add_argument(
        "--max-cells", type=int, default=100_000,
        help="Class-stratified cap per dataset to keep RAM safe on this 62 GB box. "
             "0 disables. Default 100k → ~25 GB peak (calibrated 2026-05-02 from "
             "observed crash at 200k → 56 GB).",
    )
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    max_cells = args.max_cells if args.max_cells > 0 else None

    logger.info(
        f"Running annotation benchmark on {len(datasets)} datasets: {datasets} "
        f"(max_cells={'∞' if max_cells is None else max_cells})"
    )
    all_results = []
    for ds in datasets:
        try:
            r = evaluate_one(ds, max_cells=max_cells)
            all_results.append(r)
            logger.info(f"  ✓ {ds}: {r['n_methods']} methods, {r['elapsed_s']}s")
        except Exception as e:
            logger.error(f"  ✗ {ds} FAILED: {e}")
            import traceback
            traceback.print_exc()
        gc.collect()

    if all_results:
        aggregate_master_table(all_results)
    logger.info(f"\nDone. Outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
