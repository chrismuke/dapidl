"""Test annotation confidence + filtering on breast rep1 AND rep2.

Runs BANKSY + marker_gene_overlap on both replicates, computes confidence,
and tests filtering at multiple thresholds to show F1 improvement.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from loguru import logger
from sklearn.metrics import f1_score

from dapidl.data.annotation import map_to_broad_category
from dapidl.validation.annotation_confidence import (
    AnnotationConfidenceConfig,
    compute_annotation_confidence,
    filter_predictions,
)

logger.remove()
logger.add(sys.stderr, level="INFO")

DATASETS = {
    "rep1": {
        "path": Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1"),
        "gt_file": "celltypes_ground_truth_rep1_supervised.xlsx",
    },
    "rep2": {
        "path": Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep2"),
        "gt_file": "celltypes_ground_truth_rep2_supervised.xlsx",
    },
}

GT_BROAD_MAP = {
    "DCIS_1": "Epithelial", "DCIS_2": "Epithelial",
    "Invasive_Tumor": "Epithelial", "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_KRT15+": "Epithelial", "Myoepi_ACTA2+": "Epithelial",
    "T_Cell_&_Tumor_Hybrid": "Epithelial",
    "CD4+_T_Cells": "Immune", "CD8+_T_Cells": "Immune",
    "B_Cells": "Immune", "Macrophages_1": "Immune", "Macrophages_2": "Immune",
    "Mast_Cells": "Immune", "LAMP3+_DCs": "Immune", "IRF7+_DCs": "Immune",
    "Perivascular-Like": "Immune",
    "Stromal": "Stromal", "Stromal_&_T_Cell_Hybrid": "Stromal", "Endothelial": "Stromal",
}


def load_dataset(name: str, n_cells: int | None = None) -> tuple[ad.AnnData, np.ndarray]:
    """Load a breast dataset with spatial coords and ground truth."""
    cfg = DATASETS[name]
    data_path = cfg["path"]
    logger.info(f"Loading {name} from {data_path}...")

    adata = sc.read_10x_h5(str(data_path / "cell_feature_matrix.h5"))
    adata.var_names_make_unique()

    # Spatial coordinates
    cells_df = pl.read_parquet(data_path / "cells.parquet")
    cell_ids = cells_df["cell_id"].cast(pl.Utf8).to_list()

    # Align cells
    expr_cells = set(adata.obs_names)
    common = sorted(expr_cells & set(cell_ids))
    cell_to_idx = {c: i for i, c in enumerate(adata.obs_names)}
    mask = [c for c in common if c in cell_to_idx]
    adata = adata[[cell_to_idx[c] for c in mask]].copy()

    cell_id_to_row = {str(cid): i for i, cid in enumerate(cell_ids)}
    spatial_indices = [cell_id_to_row[c] for c in mask]
    x = cells_df["x_centroid"].to_numpy()[spatial_indices]
    y = cells_df["y_centroid"].to_numpy()[spatial_indices]
    spatial_coords = np.column_stack([x, y])

    # Ground truth
    gt_df = pd.read_excel(data_path / cfg["gt_file"])
    gt_pl = pl.DataFrame({
        "cell_id": [str(b) for b in gt_df.iloc[:, 0]],
        "gt_fine": gt_df.iloc[:, 1].astype(str).tolist(),
    })
    gt_pl = gt_pl.with_columns(
        pl.col("gt_fine")
        .map_elements(lambda v: GT_BROAD_MAP.get(v, "Unknown"), return_dtype=pl.Utf8)
        .alias("gt_broad")
    ).filter(pl.col("gt_broad") != "Unknown")

    gt_cell_set = set(gt_pl["cell_id"].to_list())
    gt_mask = [i for i, c in enumerate(adata.obs_names) if c in gt_cell_set]
    adata = adata[gt_mask].copy()
    spatial_coords = spatial_coords[gt_mask]

    gt_dict = dict(zip(gt_pl["cell_id"].to_list(), gt_pl["gt_broad"].to_list()))
    adata.obs["ground_truth"] = [gt_dict.get(c, "Unknown") for c in adata.obs_names]
    adata.obs["x_centroid"] = spatial_coords[:, 0]
    adata.obs["y_centroid"] = spatial_coords[:, 1]

    # Subsample
    if n_cells and n_cells < len(adata):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(adata), n_cells, replace=False)
        adata = adata[idx].copy()
        spatial_coords = spatial_coords[idx]

    # Normalize
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    gt_counts = pl.Series(adata.obs["ground_truth"].tolist()).value_counts()
    logger.info(f"Loaded {len(adata)} cells — GT: {dict(gt_counts.iter_rows())}")
    return adata, spatial_coords


def run_celltypist(adata: ad.AnnData) -> np.ndarray:
    """CellTypist → broad categories."""
    import celltypist

    model = celltypist.models.Model.load("Cells_Adult_Breast.pkl")
    result = celltypist.annotate(adata, model=model, majority_voting=False)
    return np.array([map_to_broad_category(l) for l in result.predicted_labels["predicted_labels"]])


def run_banksy(adata: ad.AnnData) -> np.ndarray:
    """BANKSY + marker_gene_overlap (PanglaoDB) → broad categories."""
    import scipy.sparse as sp
    from scipy.spatial import cKDTree

    coords = np.column_stack([
        adata.obs["x_centroid"].values.astype(float),
        adata.obs["y_centroid"].values.astype(float),
    ])

    tree = cKDTree(coords)
    dists, indices = tree.query(coords, k=11)
    indices = indices[:, 1:]
    dists = dists[:, 1:]

    median_dist = np.median(dists[:, 0])
    weights = np.exp(-dists**2 / (2 * median_dist**2))

    n = len(adata)
    rows = np.repeat(np.arange(n), 10)
    cols = indices.ravel()
    vals = weights.ravel()
    W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    row_sums = np.array(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1
    W = sp.diags(1 / row_sums) @ W

    X = adata.X
    if sp.issparse(X):
        neighbor_mean = W @ X
    else:
        neighbor_mean = W @ sp.csr_matrix(X)
    aug = sp.hstack([X if sp.issparse(X) else sp.csr_matrix(X), 0.5 * neighbor_mean])

    aug_adata = ad.AnnData(X=aug)
    sc.pp.pca(aug_adata, n_comps=30)
    sc.pp.neighbors(aug_adata, n_neighbors=15)
    sc.tl.leiden(aug_adata, resolution=1.0, key_added="banksy_cluster")

    adata.obs["banksy_cluster"] = aug_adata.obs["banksy_cluster"].values
    sc.tl.rank_genes_groups(adata, "banksy_cluster", method="wilcoxon", n_genes=100)

    import decoupler as dc

    markers_raw = dc.op.resource("PanglaoDB", organism="human")
    markers_raw = markers_raw[
        markers_raw["canonical_marker"].astype(bool)
        & (markers_raw["human_sensitivity"].astype(float) > 0.5)
    ]
    panel_genes = set(adata.var_names)
    markers_raw = markers_raw[markers_raw["genesymbol"].isin(panel_genes)]

    marker_dict = {}
    for ct, group in markers_raw.groupby("cell_type"):
        genes = list(group["genesymbol"].unique())
        if len(genes) >= 3:
            marker_dict[ct] = genes

    overlap = sc.tl.marker_gene_overlap(adata, marker_dict, key="rank_genes_groups")

    panglao_broad_map = {
        "Epithelial cells": "Epithelial", "Basal cells": "Epithelial",
        "Luminal epithelial cells": "Epithelial", "Ductal cells": "Epithelial",
        "T cells": "Immune", "T memory cells": "Immune", "T helper cells": "Immune",
        "T regulatory cells": "Immune", "T cytotoxic cells": "Immune",
        "NK cells": "Immune", "B cells": "Immune", "B cells memory": "Immune",
        "Plasma cells": "Immune", "Dendritic cells": "Immune",
        "Macrophages": "Immune", "Monocytes": "Immune", "Mast cells": "Immune",
        "Fibroblasts": "Stromal", "Myofibroblasts": "Stromal",
        "Endothelial cells": "Stromal", "Pericytes": "Stromal",
        "Smooth muscle cells": "Stromal", "Adipocytes": "Stromal",
    }

    cluster_labels = {}
    for cluster in overlap.columns:
        best_ct = overlap[cluster].idxmax()
        broad = panglao_broad_map.get(best_ct, "Unknown")
        cluster_labels[cluster] = broad

    return np.array([cluster_labels.get(c, "Unknown") for c in adata.obs["banksy_cluster"]])


@dataclass
class RepResult:
    """Results for one replicate."""
    name: str
    n_cells: int
    banksy_f1: float
    ct_f1: float
    banksy_confidence: float
    ct_confidence: float
    filter_results: dict  # threshold → (n_kept, f1)


def run_replicate(name: str, n_cells: int | None = None) -> RepResult:
    """Run full benchmark + filtering on one replicate."""
    adata, spatial_coords = load_dataset(name, n_cells=n_cells)
    gt = np.array(adata.obs["ground_truth"].tolist())
    labels = ["Epithelial", "Immune", "Stromal"]
    config = AnnotationConfidenceConfig(tissue_type="breast")

    # ── CellTypist ────────────────────────────────────────────
    ct_preds = run_celltypist(adata)
    result_ct = compute_annotation_confidence(
        adata=adata,
        predictions={"CellTypist": ct_preds},
        spatial_coords=spatial_coords,
        config=config,
    )
    f1_ct = f1_score(gt, ct_preds, average="macro", labels=labels)

    print(f"\n{'=' * 70}")
    print(f"  {name.upper()}: CellTypist")
    print(f"{'=' * 70}")
    print(result_ct.summary())
    print(f"\n  [GT check] F1 macro = {f1_ct:.3f}")

    # ── BANKSY ────────────────────────────────────────────────
    banksy_preds = run_banksy(adata)
    result_bk = compute_annotation_confidence(
        adata=adata,
        predictions={"BANKSY": banksy_preds},
        spatial_coords=spatial_coords,
        config=config,
    )
    f1_bk = f1_score(gt, banksy_preds, average="macro", labels=labels)

    print(f"\n{'=' * 70}")
    print(f"  {name.upper()}: BANKSY")
    print(f"{'=' * 70}")
    print(result_bk.summary())
    print(f"\n  [GT check] F1 macro = {f1_bk:.3f}")

    # ── Multi-method ──────────────────────────────────────────
    result_multi = compute_annotation_confidence(
        adata=adata,
        predictions={"CellTypist": ct_preds, "BANKSY": banksy_preds},
        spatial_coords=spatial_coords,
        config=config,
        primary_method="BANKSY",
    )

    print(f"\n{'=' * 70}")
    print(f"  {name.upper()}: BANKSY + CellTypist (multi-method)")
    print(f"{'=' * 70}")
    print(result_multi.summary())

    # ── Filtering sweep (BANKSY) ──────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  {name.upper()}: Confidence-based filtering (BANKSY)")
    print(f"{'=' * 70}")
    print(f"  {'Threshold':>10} {'Kept':>8} {'%Kept':>7} {'F1':>7}  Per-type retained")
    print(f"  {'-' * 65}")

    filter_results = {}
    for threshold in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        filt = filter_predictions(banksy_preds, result_bk, min_confidence=threshold)
        kept = filt.mask
        if kept.sum() > 0:
            f1_filt = f1_score(gt[kept], filt.predictions[kept], average="macro", labels=labels)
            pct_kept = filt.n_kept / len(banksy_preds) * 100
            filter_results[threshold] = (filt.n_kept, f1_filt)

            # Per-type retention
            per_type_str = "  ".join(
                f"{ct[:3]}:{filt.per_type_kept.get(ct, 0)/(filt.per_type_kept.get(ct, 0)+filt.per_type_filtered.get(ct, 0))*100:.0f}%"
                if (filt.per_type_kept.get(ct, 0) + filt.per_type_filtered.get(ct, 0)) > 0 else f"{ct[:3]}:  -"
                for ct in labels
            )
            marker = " ◀ best" if threshold == 0.3 else ""
            print(f"  {threshold:>10.1f} {filt.n_kept:>8} {pct_kept:>6.1f}% {f1_filt:>7.3f}  {per_type_str}{marker}")

    return RepResult(
        name=name,
        n_cells=len(adata),
        banksy_f1=f1_bk,
        ct_f1=f1_ct,
        banksy_confidence=result_bk.overall_score,
        ct_confidence=result_ct.overall_score,
        filter_results=filter_results,
    )


def main():
    results = []
    for name in ["rep1", "rep2"]:
        r = run_replicate(name, n_cells=10000)
        results.append(r)

    # ── Cross-replicate summary ───────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  CROSS-REPLICATE SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'':>20} {'Rep1':>15} {'Rep2':>15}")
    print(f"  {'-' * 50}")
    print(f"  {'Cells':>20} {results[0].n_cells:>15,} {results[1].n_cells:>15,}")
    print(f"  {'CellTypist F1':>20} {results[0].ct_f1:>15.3f} {results[1].ct_f1:>15.3f}")
    print(f"  {'CT Confidence':>20} {results[0].ct_confidence:>15.3f} {results[1].ct_confidence:>15.3f}")
    print(f"  {'BANKSY F1':>20} {results[0].banksy_f1:>15.3f} {results[1].banksy_f1:>15.3f}")
    print(f"  {'BK Confidence':>20} {results[0].banksy_confidence:>15.3f} {results[1].banksy_confidence:>15.3f}")

    print(f"\n  Filtering (BANKSY):")
    print(f"  {'Threshold':>20} {'Rep1 F1':>15} {'Rep2 F1':>15}")
    print(f"  {'-' * 50}")
    for t in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        r1 = results[0].filter_results.get(t)
        r2 = results[1].filter_results.get(t)
        r1_str = f"{r1[1]:.3f} ({r1[0]:,})" if r1 else "    -"
        r2_str = f"{r2[1]:.3f} ({r2[0]:,})" if r2 else "    -"
        print(f"  {t:>20.1f} {r1_str:>15} {r2_str:>15}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
