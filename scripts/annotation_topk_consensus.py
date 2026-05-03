"""Top-3 single + 2-method + 3-method consensus from existing predictions.

GROUND TRUTH SOURCES (per user requirement):
- xenium rep1, rep2  -> Janesick supervised GT (in npz as __ground_truth__)
- STHELAR breast_s0/s1/s3/s6 -> cells_label2 from parquet, mapped via
  STHELAR_LABEL2_TO_COARSE in derive_label2_labels.py (matches the LMDB labels
  used for DAPI training; this is OUR ontology binding).

This makes the annotation benchmark a defense of our actual training labels.
"""
from __future__ import annotations
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, precision_recall_fscore_support

PRED_DIR = Path("/mnt/work/git/dapidl/pipeline_output/breast_annotation_full")
OUT_DIR = Path("/mnt/work/git/dapidl/pipeline_output/annotation_topk_2026_05")
PARQUET_DIR = Path("/mnt/work/datasets/STHELAR/summary_all_labels_per_slide")

SLIDES = ["rep1", "rep2", "breast_s0", "breast_s1", "breast_s3", "breast_s6"]
COARSE_NAMES = ["Endothelial", "Epithelial", "Immune", "Stromal"]


def get_ground_truth(slide):
    """Return (gt_array, source_label) for one slide.

    For STHELAR breast_s*, prefer cells_label2 -> COARSE (matches our DAPI
    training labels). For xenium rep, use the npz __ground_truth__ (Janesick
    supervised). Cells whose label maps to None are returned as 'Unknown'.
    """
    from derive_label2_labels import STHELAR_LABEL2_TO_COARSE
    npz = PRED_DIR / f"predictions_{slide}.npz"
    if not npz.exists():
        return None, None
    z = np.load(npz, allow_pickle=True)
    janesick_gt = z["__ground_truth__"]

    if slide.startswith("rep"):
        return janesick_gt, "janesick_supervised"

    pq = PARQUET_DIR / f"{slide}_cell_metadata.parquet"
    if not pq.exists():
        return janesick_gt, "label1_fallback"
    meta = pl.read_parquet(pq)
    l2 = meta["cells_label2"].to_list()
    coarse = [STHELAR_LABEL2_TO_COARSE.get(v) or "Unknown" for v in l2]
    coarse_arr = np.array(coarse, dtype=object)
    if len(coarse_arr) != len(janesick_gt):
        print(f"  {slide}: parquet/npz length mismatch "
              f"({len(coarse_arr)} vs {len(janesick_gt)}) - falling back")
        return janesick_gt, "label1_fallback"
    return coarse_arr, "cells_label2_coarse"


def consensus_majority(preds_list):
    if len(preds_list) == 1:
        return preds_list[0]
    n = len(preds_list[0])
    out = np.empty(n, dtype=object)
    stacked = np.stack(preds_list, axis=0)
    for i in range(n):
        col = stacked[:, i]
        vals, counts = np.unique(col, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def macro_f1(gt, pred, labels):
    mask = np.isin(gt, labels)
    if mask.sum() == 0:
        return 0.0
    return float(f1_score(gt[mask], pred[mask], labels=labels,
                          average="macro", zero_division=0))


def per_class_f1(gt, pred, labels):
    mask = np.isin(gt, labels)
    if mask.sum() == 0:
        return {c: 0.0 for c in labels}
    _, _, f1, _ = precision_recall_fscore_support(
        gt[mask], pred[mask], labels=labels, zero_division=0)
    return dict(zip(labels, [float(x) for x in f1]))


def per_method_table():
    rows = []
    for slide in SLIDES:
        gt, src = get_ground_truth(slide)
        if gt is None:
            continue
        z = np.load(PRED_DIR / f"predictions_{slide}.npz", allow_pickle=True)
        for k in z.keys():
            if k == "__ground_truth__" or k.endswith("_conf"):
                continue
            pred = z[k]
            f1 = macro_f1(gt, pred, COARSE_NAMES)
            n_eval = int(np.isin(gt, COARSE_NAMES).sum())
            row = {"slide": slide, "gt_source": src, "method": k,
                   "f1_macro": f1, "n_eval": n_eval}
            row.update({f"f1_{c}": v for c, v in
                        per_class_f1(gt, pred, COARSE_NAMES).items()})
            rows.append(row)
    return pl.DataFrame(rows)


def top_k_singles(per_method, k=3):
    return (per_method.group_by("method")
            .agg(pl.col("f1_macro").mean().alias("mean_f1"),
                 pl.col("f1_macro").std().alias("std_f1"),
                 pl.col("f1_macro").len().alias("n_slides"))
            .sort("mean_f1", descending=True)
            .head(k))


def top_k_consensus(k_methods, k_top=3):
    method_preds = {}
    method_gt = {}
    method_gt_src = {}
    for slide in SLIDES:
        gt, src = get_ground_truth(slide)
        if gt is None:
            continue
        method_gt[slide] = gt
        method_gt_src[slide] = src
        z = np.load(PRED_DIR / f"predictions_{slide}.npz", allow_pickle=True)
        for k in z.keys():
            if k == "__ground_truth__" or k.endswith("_conf"):
                continue
            method_preds.setdefault(k, {})[slide] = z[k]

    all_slides = set(SLIDES) & set(method_gt)
    methods_full = sorted(m for m, d in method_preds.items()
                          if set(d) == all_slides)
    print(f"Methods on all {len(all_slides)} slides: {len(methods_full)}")
    print(f"  -> {methods_full}")
    print(f"  GT sources: {method_gt_src}")

    rows = []
    for n, combo in enumerate(combinations(methods_full, k_methods), 1):
        scores = []
        for slide in sorted(all_slides):
            preds = [method_preds[m][slide] for m in combo]
            cons = consensus_majority(preds)
            f1 = macro_f1(method_gt[slide], cons, COARSE_NAMES)
            scores.append(f1)
        rows.append({"combo": "+".join(combo), "n_methods": k_methods,
                     "mean_f1": float(np.mean(scores)),
                     "std_f1": float(np.std(scores))})
        if n % 50 == 0:
            print(f"  {n} combos scored, best so far: "
                  f"{max(r['mean_f1'] for r in rows):.3f}")
    return (pl.DataFrame(rows).sort("mean_f1", descending=True).head(k_top))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== per-method F1 per slide (GT = cells_label2 for STHELAR, Janesick for rep) ===")
    pm = per_method_table()
    pm.write_parquet(OUT_DIR / "per_method_per_slide.parquet")
    pm.write_csv(OUT_DIR / "per_method_per_slide.csv")
    print(pm.sort(["slide", "f1_macro"], descending=[False, True]))

    print("\n=== TOP 3 SINGLE METHODS ===")
    top1 = top_k_singles(pm, k=3)
    top1.write_csv(OUT_DIR / "top3_single.csv")
    print(top1)

    print("\n=== TOP 3 TWO-METHOD CONSENSUS ===")
    top2 = top_k_consensus(k_methods=2, k_top=3)
    top2.write_csv(OUT_DIR / "top3_pair_consensus.csv")
    print(top2)

    print("\n=== TOP 3 THREE-METHOD CONSENSUS ===")
    top3 = top_k_consensus(k_methods=3, k_top=3)
    top3.write_csv(OUT_DIR / "top3_triple_consensus.csv")
    print(top3)

    md = ["# Annotation Top-K Consensus -- DAPIDL deck companion\n"]
    md.append(f"GT sources: STHELAR breast_s* -> cells_label2 -> COARSE; "
              f"xenium rep1/rep2 -> Janesick supervised\n")
    md.append("\n## Top 3 single methods (mean F1 across 6 slides)\n")
    md.append(top1.to_pandas().to_markdown(index=False))
    md.append("\n\n## Top 3 two-method consensus (majority vote)\n")
    md.append(top2.to_pandas().to_markdown(index=False))
    md.append("\n\n## Top 3 three-method consensus\n")
    md.append(top3.to_pandas().to_markdown(index=False))
    (OUT_DIR / "summary.md").write_text("\n".join(md))
    print(f"\nwrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
