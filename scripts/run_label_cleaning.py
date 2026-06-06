#!/usr/bin/env python3
"""Validate cleanlab label-cleaning on breast Xenium rep1, graded against Janesick GT.

Generates WEAK labels (CellTypist) for rep1 cells, runs cleanlab on them, and measures
whether the flagged label-issues correspond to the TRUE errors (weak != GT). Restricted
to the 3 broad classes {Epithelial, Immune, Stromal}; Endothelial is BANKSY's job
(sub-project 2). This is the spec's core validation (detection precision/recall vs GT).

    uv run python scripts/run_label_cleaning.py --rep rep1 --max-cells 5000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))

from breast_dapi_lmdb import (  # noqa: E402
    JANESICK17_TO_COARSE,
    _load_xenium_supervised_gt,
)

from dapidl.data import label_cleaning as lc  # noqa: E402
from dapidl.data.annotation import map_to_broad_category  # noqa: E402
from dapidl.data.xenium import XeniumDataReader  # noqa: E402

XENIUM_BASE = Path("/mnt/work/datasets/raw/xenium")
BROAD = {"Epithelial", "Immune", "Stromal"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="rep1", choices=["rep1", "rep2"])
    ap.add_argument("--max-cells", type=int, default=5000)
    ap.add_argument("--model", default="Cells_Adult_Breast.pkl")
    ap.add_argument("--out", type=Path, default=Path("pipeline_output/label_cleaning"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    reader = XeniumDataReader(XENIUM_BASE / f"xenium-breast-tumor-{args.rep}" / "outs")
    expr, genes, cell_ids = reader.load_expression_matrix()
    cell_ids = np.asarray([str(c) for c in cell_ids])
    logger.info(f"{args.rep}: {expr.shape[0]:,} cells x {expr.shape[1]} genes")

    # Janesick GT -> broad (drop Endothelial / hybrids / None)
    gt_fine = _load_xenium_supervised_gt(args.rep)  # barcode -> 17-class
    gt_broad = {}
    for bc, fine in gt_fine.items():
        coarse = JANESICK17_TO_COARSE.get(fine)
        if coarse in BROAD:
            gt_broad[str(bc)] = coarse

    # Keep cells that have a broad GT; subsample for a tractable run
    keep = np.array([i for i, c in enumerate(cell_ids) if c in gt_broad])
    rng = np.random.default_rng(args.seed)
    if args.max_cells and len(keep) > args.max_cells:
        keep = np.sort(rng.choice(keep, args.max_cells, replace=False))
    expr, ids = expr[keep], cell_ids[keep]
    logger.info(f"cells with broad GT (subsampled): {len(ids):,}")

    # WEAK labels via CellTypist (log1p-CPM normalized)
    import anndata as ad
    import celltypist
    import scanpy as sc

    adata = ad.AnnData(X=expr.astype(np.float32))
    adata.var_names = list(genes)
    adata.obs_names = list(ids)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    res = celltypist.annotate(adata, model=args.model, majority_voting=False)
    weak_fine = res.predicted_labels["predicted_labels"].to_numpy()
    weak_broad = np.array([map_to_broad_category(c) for c in weak_fine], dtype=object)

    # Restrict to cells whose weak label is also broad
    mask = np.array([w in BROAD for w in weak_broad])
    xn = np.asarray(adata.X[mask])
    ids_m = ids[mask]
    y_weak = weak_broad[mask]
    y_gt = np.array([gt_broad[c] for c in ids_m], dtype=object)
    logger.info(f"cells with broad weak+GT: {len(ids_m):,}")

    # cleanlab on the WEAK labels
    probs = lc.cv_pred_probs(xn, y_weak, n_folds=5, seed=args.seed)
    is_issue, quality = lc.find_label_issues(y_weak, probs)

    # Grade: do flagged issues match the TRUE errors (weak != GT)?
    true_err = y_weak != y_gt
    tp = int((is_issue & true_err).sum())
    fp = int((is_issue & ~true_err).sum())
    fn = int((~is_issue & true_err).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    err_rate = float(true_err.mean())
    flag_rate = float(is_issue.mean())

    report = (
        f"# cleanlab label-cleaning — {args.rep} (vs Janesick GT, 3 broad classes)\n\n"
        f"- cells graded: {len(ids_m):,}\n"
        f"- true weak-label error rate (weak != GT): {err_rate:.1%}\n"
        f"- cleanlab flag rate: {flag_rate:.1%}\n\n"
        f"## Detection of true errors\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| precision | {precision:.3f} |\n| recall | {recall:.3f} |\n| F1 | {f1:.3f} |\n"
        f"| TP | {tp} |\n| FP | {fp} |\n| FN | {fn} |\n\n"
        f"Interpretation: precision = of cells cleanlab flags, fraction truly wrong vs GT; "
        f"recall = of truly-wrong cells, fraction cleanlab catches.\n"
    )
    (args.out / f"{args.rep}_cleanlab_report.md").write_text(report)
    pl.DataFrame({
        "row_idx": np.arange(len(ids_m)),
        "cell_id": ids_m, "weak": y_weak, "gt": y_gt,
        "label_quality": quality, "broken": is_issue,
    }).write_parquet(args.out / f"{args.rep}_label_issues.parquet")
    print(report)
    logger.info(f"wrote {args.out}/{args.rep}_cleanlab_report.md + _label_issues.parquet")


if __name__ == "__main__":
    main()
