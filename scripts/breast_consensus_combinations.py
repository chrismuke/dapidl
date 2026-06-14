#!/usr/bin/env python3
"""Evaluate consensus combinations of 2-8 methods on breast datasets.

Loads per-cell predictions from `pipeline_output/breast_annotation_full/predictions_*.npz`
and computes naive_majority consensus for every combination of K methods (K = 2..8)
drawn from the top-N performing methods.

Output:
    pipeline_output/breast_annotation_full/
        consensus_combos_<dataset>.parquet   # all combination results
        consensus_top50.md                   # top-50 per dataset
        consensus_all.parquet                # cross-dataset
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from sklearn.metrics import f1_score, accuracy_score

DIR = Path("/mnt/work/git/dapidl/pipeline_output/breast_annotation_full")

COARSE_CLASSES = ["Endothelial", "Epithelial", "Immune", "Stromal"]


def naive_majority(preds_array: np.ndarray) -> np.ndarray:
    """preds_array: (n_methods, n_cells) string preds. Returns (n_cells,)."""
    n_methods, n_cells = preds_array.shape
    out = np.empty(n_cells, dtype=preds_array.dtype)
    for j in range(n_cells):
        votes = preds_array[:, j]
        clean = [v for v in votes if v and v != "Unknown"]
        if not clean:
            out[j] = "Unknown"
            continue
        c = Counter(clean)
        max_count = max(c.values())
        winners = [k for k, v in c.items() if v == max_count]
        out[j] = winners[0]
    return out


def evaluate_combo(
    methods: tuple[str, ...],
    preds_dict: dict[str, np.ndarray],
    gt: np.ndarray,
) -> dict:
    """Evaluate one combination of methods using naive majority."""
    preds_array = np.stack([preds_dict[m] for m in methods], axis=0)
    consensus = naive_majority(preds_array)

    valid = consensus != "Unknown"
    if valid.sum() == 0:
        return {
            "methods": "+".join(methods), "k": len(methods),
            "n_evaluated": 0,
            "macro_f1": None, "accuracy": None, "weighted_f1": None,
            "endo_f1": None, "epi_f1": None, "imm_f1": None, "str_f1": None,
        }
    yt = gt[valid]
    yp = consensus[valid]

    acc = float(accuracy_score(yt, yp))
    macro = float(f1_score(yt, yp, labels=COARSE_CLASSES, average="macro", zero_division=0))
    weighted = float(f1_score(yt, yp, labels=COARSE_CLASSES, average="weighted", zero_division=0))
    per_class_f1 = f1_score(yt, yp, labels=COARSE_CLASSES, average=None, zero_division=0)

    return {
        "methods": "+".join(methods),
        "k": len(methods),
        "n_evaluated": int(valid.sum()),
        "macro_f1": macro,
        "accuracy": acc,
        "weighted_f1": weighted,
        "endo_f1": float(per_class_f1[0]),
        "epi_f1": float(per_class_f1[1]),
        "imm_f1": float(per_class_f1[2]),
        "str_f1": float(per_class_f1[3]),
    }


def evaluate_dataset_combos(dataset: str, top_n: int, k_range: range) -> pl.DataFrame:
    """Run all combinations of K (K in k_range) drawn from top_n best methods."""
    npz_path = DIR / f"predictions_{dataset}.npz"
    if not npz_path.exists():
        logger.warning(f"  {dataset}: no NPZ at {npz_path}, skipping")
        return pl.DataFrame()
    npz = np.load(npz_path)
    gt = npz["__ground_truth__"]
    method_keys = [k for k in npz.keys()
                   if k != "__ground_truth__" and not k.endswith("_conf")]
    if len(method_keys) < 2:
        logger.warning(f"  {dataset}: only {len(method_keys)} methods, need ≥2")
        return pl.DataFrame()

    preds_dict: dict[str, np.ndarray] = {m: npz[m] for m in method_keys}

    # Rank methods by individual macro F1 to pick top-N
    individual_f1 = {}
    for m in method_keys:
        valid = preds_dict[m] != "Unknown"
        if valid.sum() == 0:
            individual_f1[m] = 0.0
            continue
        try:
            individual_f1[m] = float(f1_score(
                gt[valid], preds_dict[m][valid],
                labels=COARSE_CLASSES, average="macro", zero_division=0,
            ))
        except Exception:
            individual_f1[m] = 0.0

    ranked = sorted(individual_f1.items(), key=lambda x: -x[1])[:top_n]
    top_methods = [m for m, _ in ranked]
    logger.info(f"  {dataset}: top-{top_n} methods (individual F1):")
    for m, f in ranked:
        logger.info(f"    {f:.4f}  {m}")

    rows = []
    # K=1 baseline (individual methods)
    for m in top_methods:
        r = evaluate_combo((m,), preds_dict, gt)
        r["dataset"] = dataset
        r["k"] = 1
        rows.append(r)

    # K=2..max_k
    for k in k_range:
        if k > len(top_methods):
            break
        n_combos = sum(1 for _ in combinations(top_methods, k))
        logger.info(f"  {dataset} K={k}: {n_combos} combinations")
        for combo in combinations(top_methods, k):
            r = evaluate_combo(combo, preds_dict, gt)
            r["dataset"] = dataset
            rows.append(r)

    df = pl.DataFrame(rows)
    df.write_parquet(DIR / f"consensus_combos_{dataset}.parquet")
    logger.info(f"  Saved {len(df)} rows to consensus_combos_{dataset}.parquet")
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="rep1,rep2,breast_s0,breast_s1,breast_s3,breast_s6")
    ap.add_argument("--top-n", type=int, default=10,
                    help="Number of top individual methods to draw combinations from")
    ap.add_argument("--max-k", type=int, default=8,
                    help="Maximum combination size")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    k_range = range(2, args.max_k + 1)

    all_dfs = []
    for ds in datasets:
        logger.info(f"\n=== {ds} ===")
        df = evaluate_dataset_combos(ds, args.top_n, k_range)
        if len(df):
            all_dfs.append(df)

    if not all_dfs:
        logger.error("No results — aborting")
        return

    big = pl.concat(all_dfs)
    big.write_parquet(DIR / "consensus_all.parquet")

    # Top-50 per dataset markdown
    lines = ["# Consensus Combination Leaderboard", ""]
    for ds in sorted(big["dataset"].unique().to_list()):
        lines.append(f"## {ds}")
        lines.append("")
        sub = big.filter(
            (pl.col("dataset") == ds) & pl.col("macro_f1").is_not_null()
        ).sort("macro_f1", descending=True).head(50)
        lines.append("| Rank | K | Methods | Macro F1 | Acc | Endo | Epi | Imm | Str |")
        lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|")
        for i, r in enumerate(sub.iter_rows(named=True), 1):
            def _f(v):
                return f"{v:.3f}" if v is not None else "—"
            lines.append(
                f"| {i} | {r['k']} | `{r['methods']}` | {_f(r['macro_f1'])} | "
                f"{_f(r['accuracy'])} | {_f(r['endo_f1'])} | {_f(r['epi_f1'])} | "
                f"{_f(r['imm_f1'])} | {_f(r['str_f1'])} |"
            )
        lines.append("")

    # Cross-dataset best-K averages
    lines.append("## Best K-method consensus averaged across datasets (K=1..8)")
    lines.append("")
    lines.append("| K | Best combo | Mean macro F1 | n_datasets |")
    lines.append("|---:|---|---:|---:|")
    for k in sorted(big["k"].unique().to_list()):
        sub_k = big.filter((pl.col("k") == k) & pl.col("macro_f1").is_not_null())
        if not len(sub_k):
            continue
        avg = sub_k.group_by("methods").agg(
            pl.col("macro_f1").mean().alias("mean_f1"),
            pl.col("dataset").n_unique().alias("n_datasets"),
        ).filter(pl.col("n_datasets") >= 2).sort("mean_f1", descending=True).head(1)
        if len(avg):
            r = avg.row(0, named=True)
            lines.append(
                f"| {k} | `{r['methods']}` | {r['mean_f1']:.4f} | {r['n_datasets']} |"
            )

    (DIR / "consensus_top50.md").write_text("\n".join(lines))
    logger.info(f"Wrote consensus_top50.md with {len(big)} total rows")


if __name__ == "__main__":
    main()
