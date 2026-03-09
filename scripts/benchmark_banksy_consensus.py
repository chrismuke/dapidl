#!/usr/bin/env python3
"""
Comprehensive benchmark: BANKSY + marker_gene_overlap as a new annotation method
in ALL consensus combinations with existing methods.

Tests on breast rep1 and rep2 datasets against ground truth.

Methods:
  1. BANKSY + marker_gene_overlap (PanglaoDB) — NEW unsupervised spatial method
  2. CellTypist (Breast model)
  3. CellTypist (Immune model)
  4. scType (marker-based)
  5. SingleR (Blueprint + HPCA combined)
  6. CellAssign (scvi-tools, probabilistic)

Then: all 2^N-1 consensus combinations via majority voting, plus
spatial smoothing of BANKSY predictions.

CRITICAL ordering:
  - ALL Leiden/BANKSY clustering runs BEFORE any rpy2/SingleR calls
    (R initialization corrupts igraph/leidenalg C extensions → segfault)

Usage:
    uv run python scripts/benchmark_banksy_consensus.py
    uv run python scripts/benchmark_banksy_consensus.py --datasets rep1
    uv run python scripts/benchmark_banksy_consensus.py --sample-size 5000
"""

import itertools
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Disable GPU — all methods are CPU-only
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from loguru import logger
from scipy.sparse import csr_matrix, diags, issparse
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
)
from sklearn.neighbors import NearestNeighbors

# ── Configuration ─────────────────────────────────────────────────

DATASETS = {
    "rep1": {
        "h5_path": Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1/cell_feature_matrix.h5"),
        "cells_path": Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1/cells.parquet"),
        "gt_path": Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1/celltypes_ground_truth_rep1_supervised.xlsx"),
        "name": "Breast Rep1",
    },
    "rep2": {
        "h5_path": Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep2/cell_feature_matrix.h5"),
        "cells_path": Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep2/cells.parquet"),
        "gt_path": Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep2/celltypes_ground_truth_rep2_supervised.xlsx"),
        "name": "Breast Rep2",
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
    "Stromal": "Stromal", "Stromal_&_T_Cell_Hybrid": "Stromal",
    "Endothelial": "Stromal",
}

PANGLAO_BROAD_MAP = {
    # Epithelial
    "Epithelial cells": "Epithelial", "Basal cells": "Epithelial",
    "Luminal epithelial cells": "Epithelial", "Ductal cells": "Epithelial",
    "Myoepithelial cells": "Epithelial", "Keratinocytes": "Epithelial",
    "Alveolar cells": "Epithelial", "Clara cells": "Epithelial",
    "Goblet cells": "Epithelial", "Enterocytes": "Epithelial",
    "Tuft cells": "Epithelial", "Paneth cells": "Epithelial",
    # Immune
    "T cells": "Immune", "T memory cells": "Immune", "T helper cells": "Immune",
    "T regulatory cells": "Immune", "T cytotoxic cells": "Immune",
    "NK cells": "Immune", "NKT cells": "Immune",
    "B cells": "Immune", "B cells memory": "Immune",
    "Plasma cells": "Immune", "Plasmacytoid dendritic cells": "Immune",
    "Dendritic cells": "Immune", "Macrophages": "Immune",
    "Monocytes": "Immune", "Mast cells": "Immune",
    "Neutrophils": "Immune", "Eosinophils": "Immune", "Basophils": "Immune",
    "Langerhans cells": "Immune", "Kupffer cells": "Immune",
    "Microglia": "Immune", "Granulocytes": "Immune",
    # Stromal
    "Fibroblasts": "Stromal", "Myofibroblasts": "Stromal",
    "Endothelial cells": "Stromal", "Pericytes": "Stromal",
    "Smooth muscle cells": "Stromal", "Mesenchymal stem cells": "Stromal",
    "Adipocytes": "Stromal", "Stellate cells": "Stromal",
    "Lymphatic endothelial cells": "Stromal",
}

# Marker genes for CellAssign broad categories
CELLASSIGN_MARKERS = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "KRT7", "MUC1"],
    "Immune": ["PTPRC", "CD3D", "CD3E", "CD4", "CD8A", "CD14", "CD68",
               "MS4A1", "CD19", "NKG7", "GNLY", "TPSAB1"],
    "Stromal": ["VIM", "FAP", "PDGFRA", "PDGFRB", "ACTA2", "COL1A1",
                "PECAM1", "VWF", "CLDN5", "CDH5"],
}


# ── Data Loading ──────────────────────────────────────────────────

def load_data(
    config: dict, sample_size: int | None = None
) -> tuple[ad.AnnData, np.ndarray, pl.DataFrame]:
    """Load expression, spatial coords, and ground truth. Returns normalized adata."""
    logger.info(f"Loading {config['name']}...")

    adata = sc.read_10x_h5(str(config["h5_path"]))
    adata.var_names_make_unique()

    cells_df = pl.read_parquet(config["cells_path"])
    cell_ids = cells_df["cell_id"].cast(pl.Utf8).to_list()

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
    gt_df = pd.read_excel(config["gt_path"])
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

    if sample_size and sample_size < adata.n_obs:
        logger.info(f"Sampling {sample_size} from {adata.n_obs}")
        idx = np.random.choice(adata.n_obs, sample_size, replace=False)
        adata = adata[idx].copy()
        spatial_coords = spatial_coords[idx]

    adata.layers["counts"] = adata.X.copy()

    n_genes = np.array((adata.X > 0).sum(axis=1)).flatten()
    keep_mask = n_genes >= 5
    n_filtered = (~keep_mask).sum()
    if n_filtered > 0:
        logger.info(f"Filtered {n_filtered} cells with <5 genes")
        adata = adata[keep_mask].copy()
        spatial_coords = spatial_coords[keep_mask]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    logger.info(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    return adata, spatial_coords, gt_pl


def get_gt_labels(adata: ad.AnnData, gt_df: pl.DataFrame) -> list[str]:
    """Aligned ground truth labels for adata obs order."""
    gt_map = dict(zip(gt_df["cell_id"].to_list(), gt_df["gt_broad"].to_list()))
    return [gt_map.get(c, "Unknown") for c in adata.obs_names]


# ── Evaluation ────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    method: str
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    per_class_f1: dict = field(default_factory=dict)
    kappa: float = 0.0
    n_cells: int = 0
    runtime_seconds: float = 0.0


def evaluate(y_true: list, y_pred: list, method: str, runtime: float = 0.0) -> BenchmarkResult:
    """Compute accuracy, F1, kappa against ground truth."""
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t != "Unknown" and p != "Unknown"]
    if not pairs:
        return BenchmarkResult(method=method)

    true, pred = zip(*pairs)
    labels = sorted(set(true) | set(pred))

    acc = accuracy_score(true, pred)
    macro_f1 = f1_score(true, pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(true, pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(true, pred)

    report = classification_report(true, pred, output_dict=True, zero_division=0)
    per_class = {label: round(report[label]["f1-score"], 3) for label in labels if label in report}

    return BenchmarkResult(
        method=method,
        accuracy=round(acc, 4),
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        per_class_f1=per_class,
        kappa=round(kappa, 4),
        n_cells=len(pairs),
        runtime_seconds=round(runtime, 1),
    )


# ── BANKSY + marker_gene_overlap ──────────────────────────────────

def get_panglao_markers(adata: ad.AnnData) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """Pull PanglaoDB markers via decoupler and return (markers_df, broad_dict)."""
    import decoupler as dc

    logger.info("Pulling PanglaoDB markers via decoupler/OmniPath...")
    markers = dc.op.resource("PanglaoDB", organism="human")

    markers = markers[
        markers["canonical_marker"].astype(bool)
        & (markers["human_sensitivity"].astype(float) > 0.5)
    ]
    markers = markers[~markers.duplicated(["cell_type", "genesymbol"])]

    panel_genes = set(adata.var_names)
    markers_in_panel = markers[markers["genesymbol"].isin(panel_genes)]

    logger.info(f"PanglaoDB: {len(markers)} total markers -> {len(markers_in_panel)} in Xenium panel")

    # Build broad dict
    broad_markers: dict[str, set[str]] = {"Epithelial": set(), "Immune": set(), "Stromal": set()}
    for _, row in markers_in_panel.iterrows():
        ct = row["cell_type"]
        gene = row["genesymbol"]
        broad = PANGLAO_BROAD_MAP.get(ct)
        if broad:
            broad_markers[broad].add(gene)

    for cat, genes in broad_markers.items():
        logger.info(f"  {cat}: {len(genes)} markers")

    return markers_in_panel, broad_markers


def run_banksy_marker_overlap(
    adata: ad.AnnData,
    spatial_coords: np.ndarray,
    panglao_broad_markers: dict[str, set[str]],
    num_neighbours: int = 10,
    resolution: float = 1.0,
) -> list[str]:
    """BANKSY augmentation + Leiden clustering + marker_gene_overlap labeling."""
    import logging as _logging
    _logging.getLogger("banksy").setLevel(_logging.WARNING)
    from banksy.main import generate_spatial_weights_fixed_nbrs

    logger.info(f"BANKSY: k={num_neighbours}, res={resolution}")

    # Spatial weight graph (sparse)
    W, _, _ = generate_spatial_weights_fixed_nbrs(
        spatial_coords.astype(np.float64), m=0,
        num_neighbours=num_neighbours, decay_type="scaled_gaussian",
    )
    if not issparse(W):
        W = csr_matrix(W)
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    W_norm = diags(1.0 / row_sums) @ W

    # Augment expression with neighbor means
    X = adata.X
    neighbor_mean = W_norm @ X
    if issparse(neighbor_mean):
        neighbor_mean = neighbor_mean.toarray()
    X_dense = X.toarray() if issparse(X) else np.array(X)
    X_aug = np.concatenate([X_dense, neighbor_mean], axis=1)

    gene_names = list(adata.var_names) + [f"nbr_{g}" for g in adata.var_names]
    adata_aug = ad.AnnData(X=X_aug)
    adata_aug.obs = adata.obs.copy()
    adata_aug.obs_names = adata.obs_names.tolist()
    adata_aug.var_names = gene_names

    # PCA + neighbors + Leiden on augmented features
    sc.pp.pca(adata_aug, n_comps=min(50, adata_aug.n_vars - 1))
    sc.pp.neighbors(adata_aug, n_neighbors=15)
    sc.tl.leiden(adata_aug, resolution=resolution, flavor="igraph", n_iterations=2, directed=False)

    n_clusters = len(adata_aug.obs["leiden"].unique())
    logger.info(f"BANKSY produced {n_clusters} clusters")

    # Copy cluster assignments to original adata for DEG analysis
    adata.obs["banksy_leiden"] = adata_aug.obs["leiden"].values

    # Rank genes per cluster (on original expression, NOT augmented)
    adata_degs = adata.copy()
    sc.tl.rank_genes_groups(adata_degs, groupby="banksy_leiden", method="wilcoxon")

    # marker_gene_overlap: rows=cell types, cols=clusters
    overlap_df = sc.tl.marker_gene_overlap(
        adata_degs,
        reference_markers=panglao_broad_markers,
        method="jaccard",
        top_n_markers=50,
    )
    logger.info(f"Marker overlap (Jaccard):\n{overlap_df.to_string()}")

    cluster_to_type = overlap_df.idxmax(axis=0).to_dict()
    logger.info(f"Cluster assignments: {cluster_to_type}")

    clusters = adata.obs["banksy_leiden"].tolist()
    return [cluster_to_type.get(c, "Unknown") for c in clusters]


# ── Spatial Smoothing ─────────────────────────────────────────────

def spatial_smooth(predictions: list[str], spatial_coords: np.ndarray, k: int = 30) -> list[str]:
    """KNN spatial majority vote smoothing."""
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
    nn.fit(spatial_coords)
    _, indices = nn.kneighbors(spatial_coords)

    smoothed = []
    for i in range(len(predictions)):
        neighbor_preds = [predictions[j] for j in indices[i]]
        vote = Counter(neighbor_preds).most_common(1)[0][0]
        smoothed.append(vote)
    return smoothed


# ── Annotation Methods ────────────────────────────────────────────

def run_celltypist(adata: ad.AnnData, model_name: str) -> list[str]:
    """CellTypist -> broad category predictions."""
    import celltypist
    from celltypist import models as ct_models
    from dapidl.pipeline.components.annotators.mapping import map_to_broad_category

    ct_models.download_models(model=[model_name])
    model = ct_models.Model.load(model=model_name)
    predictions = celltypist.annotate(adata.copy(), model=model, majority_voting=False)
    labels = predictions.predicted_labels.predicted_labels.tolist()
    return [map_to_broad_category(l) for l in labels]


def run_sctype(adata: ad.AnnData) -> list[str]:
    """scType -> broad category predictions."""
    from dapidl.pipeline.base import AnnotationConfig
    from dapidl.pipeline.components.annotators.sctype import ScTypeAnnotator

    config = AnnotationConfig(fine_grained=True)
    annotator = ScTypeAnnotator(config)
    result = annotator.annotate(config=config, adata=adata.copy())
    df = result.annotations_df
    pred_map = dict(zip(df["cell_id"].to_list(), df["broad_category"].to_list()))
    return [pred_map.get(c, "Unknown") for c in adata.obs_names]


def run_singler(adata: ad.AnnData, reference: str = "blueprint") -> list[str]:
    """SingleR -> broad category predictions."""
    from dapidl.pipeline.base import AnnotationConfig
    from dapidl.pipeline.components.annotators.singler import SingleRAnnotator, is_singler_available

    if not is_singler_available():
        raise RuntimeError("SingleR not available (requires R + SingleR + celldex)")

    config = AnnotationConfig()
    # Set singler_reference as a dynamic attribute (not in dataclass fields)
    config.singler_reference = reference  # type: ignore[attr-defined]

    annotator = SingleRAnnotator(config)
    result = annotator.annotate(config=config, adata=adata.copy())
    df = result.annotations_df
    pred_map = dict(zip(df["cell_id"].to_list(), df["broad_category"].to_list()))
    return [pred_map.get(c, "Unknown") for c in adata.obs_names]


def run_cellassign(adata: ad.AnnData) -> list[str]:
    """CellAssign (scvi-tools) -> broad category predictions. CPU only."""
    import scvi

    logger.info("Running CellAssign (scvi-tools, CPU)...")

    # Build binary marker matrix
    panel_genes = set(adata.var_names)
    cell_types = sorted(CELLASSIGN_MARKERS.keys())
    all_marker_genes = sorted({g for genes in CELLASSIGN_MARKERS.values() for g in genes if g in panel_genes})

    if len(all_marker_genes) < 3:
        raise RuntimeError(f"Only {len(all_marker_genes)} marker genes in panel — too few for CellAssign")

    marker_matrix = pd.DataFrame(
        np.zeros((len(all_marker_genes), len(cell_types)), dtype=np.int32),
        index=all_marker_genes,
        columns=cell_types,
    )
    for ct, genes in CELLASSIGN_MARKERS.items():
        for g in genes:
            if g in all_marker_genes:
                marker_matrix.loc[g, ct] = 1

    logger.info(f"CellAssign marker matrix: {marker_matrix.shape[0]} genes x {marker_matrix.shape[1]} types")
    logger.info(f"Markers per type: {dict(marker_matrix.sum(axis=0))}")

    # Subset adata to marker genes and use raw counts
    adata_ca = adata[:, all_marker_genes].copy()
    if "counts" in adata.layers:
        adata_ca.X = adata.layers["counts"][:, [list(adata.var_names).index(g) for g in all_marker_genes]].copy()
    else:
        # Fallback: expm1 to undo log1p
        X = adata_ca.X.toarray() if issparse(adata_ca.X) else np.array(adata_ca.X)
        adata_ca.X = np.expm1(X * np.log(10001))  # approximate raw counts

    # Size factors from total counts
    lib_size = np.array(adata_ca.X.sum(axis=1)).flatten()
    lib_size[lib_size == 0] = 1.0
    adata_ca.obs["size_factor"] = lib_size / np.median(lib_size)

    # Setup and train CellAssign
    scvi.external.CellAssign.setup_anndata(adata_ca, size_factor_key="size_factor")
    model = scvi.external.CellAssign(adata_ca, marker_matrix)
    model.train(max_epochs=100, accelerator="cpu", early_stopping=True)

    # Get predictions
    predictions = model.predict()  # DataFrame: cells x cell_types (probabilities)
    pred_labels = predictions.idxmax(axis=1).tolist()

    logger.info(f"CellAssign predictions: {Counter(pred_labels)}")
    return pred_labels


# ── Consensus Voting ──────────────────────────────────────────────

def majority_vote(method_preds: dict[str, list[str]], method_names: list[str]) -> list[str]:
    """Unweighted majority vote across selected methods. Ties go to first method."""
    n_cells = len(next(iter(method_preds.values())))
    result = []
    for i in range(n_cells):
        votes = [method_preds[m][i] for m in method_names if method_preds[m][i] != "Unknown"]
        if votes:
            result.append(Counter(votes).most_common(1)[0][0])
        else:
            result.append("Unknown")
    return result


# ── Main Benchmark ────────────────────────────────────────────────

def run_benchmark(dataset_keys: list[str], sample_size: int | None = None):
    """Run full benchmark on all datasets."""
    all_results: dict[str, list[dict]] = {}

    for ds_key in dataset_keys:
        if ds_key not in DATASETS:
            logger.warning(f"Unknown dataset: {ds_key}")
            continue

        config = DATASETS[ds_key]
        logger.info(f"\n{'='*80}")
        logger.info(f"  DATASET: {config['name']}")
        logger.info(f"{'='*80}")

        adata, spatial_coords, gt_df = load_data(config, sample_size=sample_size)
        gt_labels = get_gt_labels(adata, gt_df)

        results: list[BenchmarkResult] = []
        method_preds: dict[str, list[str]] = {}

        # ────────────────────────────────────────────────────────
        # PHASE 1: BANKSY + Leiden (MUST run before R/SingleR)
        # ────────────────────────────────────────────────────────

        logger.info("\n>>> [1/6] BANKSY + marker_gene_overlap (PanglaoDB)...")
        t0 = time.time()
        try:
            _, panglao_broad = get_panglao_markers(adata)
            banksy_preds = run_banksy_marker_overlap(
                adata, spatial_coords, panglao_broad,
                num_neighbours=10, resolution=1.0,
            )
            runtime = time.time() - t0
            results.append(evaluate(gt_labels, banksy_preds, "BANKSY_MarkerOverlap", runtime))
            method_preds["BANKSY"] = banksy_preds
            logger.info(f"  BANKSY done in {runtime:.1f}s — {Counter(banksy_preds)}")

            # Also evaluate spatially smoothed BANKSY
            t0s = time.time()
            banksy_smoothed = spatial_smooth(banksy_preds, spatial_coords, k=30)
            results.append(evaluate(
                gt_labels, banksy_smoothed,
                "BANKSY_MarkerOverlap_Smooth_k30",
                time.time() - t0s,
            ))
            method_preds["BANKSY_Smooth"] = banksy_smoothed
        except Exception as e:
            logger.error(f"BANKSY failed: {e}")
            import traceback; traceback.print_exc()

        # ────────────────────────────────────────────────────────
        # PHASE 2: CellTypist (no R dependency)
        # ────────────────────────────────────────────────────────

        logger.info("\n>>> [2/6] CellTypist (Breast)...")
        t0 = time.time()
        try:
            ct_breast = run_celltypist(adata, "Cells_Adult_Breast.pkl")
            results.append(evaluate(gt_labels, ct_breast, "CellTypist_Breast", time.time() - t0))
            method_preds["CT_Breast"] = ct_breast
        except Exception as e:
            logger.error(f"CellTypist Breast failed: {e}")
            import traceback; traceback.print_exc()

        logger.info("\n>>> [3/6] CellTypist (Immune)...")
        t0 = time.time()
        try:
            ct_immune = run_celltypist(adata, "Immune_All_High.pkl")
            results.append(evaluate(gt_labels, ct_immune, "CellTypist_Immune", time.time() - t0))
            method_preds["CT_Immune"] = ct_immune
        except Exception as e:
            logger.error(f"CellTypist Immune failed: {e}")
            import traceback; traceback.print_exc()

        # ────────────────────────────────────────────────────────
        # PHASE 3: scType (no R dependency)
        # ────────────────────────────────────────────────────────

        logger.info("\n>>> [4/6] scType...")
        t0 = time.time()
        try:
            sct_preds = run_sctype(adata)
            results.append(evaluate(gt_labels, sct_preds, "scType", time.time() - t0))
            method_preds["scType"] = sct_preds
        except Exception as e:
            logger.error(f"scType failed: {e}")
            import traceback; traceback.print_exc()

        # ────────────────────────────────────────────────────────
        # PHASE 4: CellAssign (no R dependency)
        # ────────────────────────────────────────────────────────

        logger.info("\n>>> [5/6] CellAssign (scvi-tools)...")
        t0 = time.time()
        try:
            ca_preds = run_cellassign(adata)
            results.append(evaluate(gt_labels, ca_preds, "CellAssign", time.time() - t0))
            method_preds["CellAssign"] = ca_preds
        except Exception as e:
            logger.warning(f"CellAssign failed (skipping): {e}")
            import traceback; traceback.print_exc()

        # ────────────────────────────────────────────────────────
        # PHASE 5: SingleR — LAST (R corrupts igraph/leidenalg)
        # ────────────────────────────────────────────────────────

        logger.info("\n>>> [6/6] SingleR (Blueprint + HPCA)...")
        t0 = time.time()
        try:
            sr_bp = run_singler(adata, "blueprint")
            results.append(evaluate(gt_labels, sr_bp, "SingleR_Blueprint", time.time() - t0))
            method_preds["SR_Blueprint"] = sr_bp
        except Exception as e:
            logger.error(f"SingleR Blueprint failed: {e}")
            import traceback; traceback.print_exc()

        t0 = time.time()
        try:
            sr_hpca = run_singler(adata, "hpca")
            results.append(evaluate(gt_labels, sr_hpca, "SingleR_HPCA", time.time() - t0))
            method_preds["SR_HPCA"] = sr_hpca
        except Exception as e:
            logger.error(f"SingleR HPCA failed: {e}")
            import traceback; traceback.print_exc()

        # ────────────────────────────────────────────────────────
        # PHASE 6: ALL 2^N - 1 consensus combinations
        # ────────────────────────────────────────────────────────

        available_methods = list(method_preds.keys())
        # Exclude the smoothed BANKSY from consensus combinations (it's a variant, not a base method)
        base_methods = [m for m in available_methods if m != "BANKSY_Smooth"]
        n_methods = len(base_methods)

        logger.info(f"\n>>> Consensus voting: {n_methods} methods -> {2**n_methods - 1} combinations")
        logger.info(f"    Methods: {base_methods}")

        consensus_results: list[BenchmarkResult] = []

        for r in range(2, n_methods + 1):  # combos of size 2..N (singles already evaluated)
            for combo in itertools.combinations(base_methods, r):
                combo_name = "+".join(combo)
                t0 = time.time()
                try:
                    consensus_preds = majority_vote(method_preds, list(combo))
                    res = evaluate(gt_labels, consensus_preds, f"Consensus({combo_name})", time.time() - t0)
                    consensus_results.append(res)
                except Exception as e:
                    logger.error(f"Consensus {combo_name} failed: {e}")

        results.extend(consensus_results)

        # ────────────────────────────────────────────────────────
        # Print results
        # ────────────────────────────────────────────────────────

        print_results(config["name"], results, base_methods)

        # Convert to dicts for JSON serialization
        all_results[ds_key] = [asdict(r) for r in results]

    # Save JSON
    out_path = Path("/tmp/banksy_consensus_results.json")
    out_path.write_text(json.dumps(all_results, indent=2))
    logger.info(f"\nResults saved to {out_path}")

    return all_results


def print_results(dataset_name: str, results: list[BenchmarkResult], base_methods: list[str]):
    """Print formatted results, highlighting BANKSY impact."""
    print(f"\n{'='*120}")
    print(f"  {dataset_name} -- ALL RESULTS (sorted by Macro F1)")
    print(f"{'='*120}")
    print(f"{'Method':<55} {'Acc':>7} {'F1_M':>7} {'F1_W':>7} {'Epi':>7} {'Imm':>7} {'Str':>7} {'k':>7} {'Time':>7}")
    print("-" * 120)

    sorted_results = sorted(results, key=lambda x: -x.macro_f1)
    for r in sorted_results:
        epi = r.per_class_f1.get("Epithelial", 0)
        imm = r.per_class_f1.get("Immune", 0)
        stro = r.per_class_f1.get("Stromal", 0)
        # Marker for BANKSY-containing results
        marker = " *" if "BANKSY" in r.method else "  "
        print(
            f"{marker}{r.method:<53} {r.accuracy:>7.3f} {r.macro_f1:>7.3f} {r.weighted_f1:>7.3f} "
            f"{epi:>7.3f} {imm:>7.3f} {stro:>7.3f} {r.kappa:>7.3f} {r.runtime_seconds:>6.1f}s"
        )

    # ── Summary sections ──────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  INDIVIDUAL METHOD RANKING")
    print(f"{'='*120}")
    individual = [r for r in sorted_results if not r.method.startswith("Consensus(")]
    for r in individual:
        epi = r.per_class_f1.get("Epithelial", 0)
        imm = r.per_class_f1.get("Immune", 0)
        stro = r.per_class_f1.get("Stromal", 0)
        print(f"  {r.method:<50} F1={r.macro_f1:.3f}  (Epi={epi:.3f} Imm={imm:.3f} Str={stro:.3f})")

    print(f"\n{'='*120}")
    print(f"  TOP 20 CONSENSUS COMBINATIONS")
    print(f"{'='*120}")
    consensus = sorted(
        [r for r in results if r.method.startswith("Consensus(")],
        key=lambda x: -x.macro_f1,
    )
    for i, r in enumerate(consensus[:20]):
        epi = r.per_class_f1.get("Epithelial", 0)
        imm = r.per_class_f1.get("Immune", 0)
        stro = r.per_class_f1.get("Stromal", 0)
        has_banksy = "*" if "BANKSY" in r.method else " "
        print(
            f"  {has_banksy} {i+1:>2}. {r.method:<65} F1={r.macro_f1:.3f}  "
            f"Epi={epi:.3f} Imm={imm:.3f} Str={stro:.3f}"
        )

    # ── BANKSY Impact Analysis ────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  BANKSY IMPACT ANALYSIS (combo WITH BANKSY vs same combo WITHOUT)")
    print(f"{'='*120}")

    # For each non-BANKSY combo, find the same combo + BANKSY
    non_banksy_methods = [m for m in base_methods if m != "BANKSY"]
    result_map = {r.method: r for r in results}

    impact_rows = []
    for r_size in range(1, len(non_banksy_methods) + 1):
        for combo in itertools.combinations(non_banksy_methods, r_size):
            without_name = "+".join(combo) if len(combo) > 1 else combo[0]
            without_key = f"Consensus({without_name})" if len(combo) > 1 else without_name

            with_combo = ("BANKSY",) + combo
            with_name = "+".join(with_combo)
            with_key = f"Consensus({with_name})"

            r_without = result_map.get(without_key)
            r_with = result_map.get(with_key)

            if r_without and r_with:
                delta = r_with.macro_f1 - r_without.macro_f1
                impact_rows.append((without_name, r_without.macro_f1, r_with.macro_f1, delta))

    if impact_rows:
        impact_rows.sort(key=lambda x: -x[3])  # Sort by delta descending
        print(f"  {'Base Combination':<55} {'Without':>9} {'With':>9} {'Delta':>9}")
        print(f"  {'-'*85}")
        for base, f1_without, f1_with, delta in impact_rows:
            sign = "+" if delta >= 0 else ""
            print(f"  {base:<55} {f1_without:>9.3f} {f1_with:>9.3f} {sign}{delta:>8.3f}")

        avg_delta = np.mean([d for _, _, _, d in impact_rows])
        print(f"\n  Average BANKSY impact on F1: {'+' if avg_delta >= 0 else ''}{avg_delta:.3f}")
        best = max(impact_rows, key=lambda x: x[3])
        print(f"  Best BANKSY boost: +{best[3]:.3f} on '{best[0]}'")
        worst = min(impact_rows, key=lambda x: x[3])
        print(f"  Worst BANKSY impact: {'+' if worst[3] >= 0 else ''}{worst[3]:.3f} on '{worst[0]}'")
    else:
        print("  (No paired comparisons available)")

    # ── Best overall ──────────────────────────────────────────
    best_overall = sorted_results[0] if sorted_results else None
    if best_overall:
        print(f"\n  BEST OVERALL: {best_overall.method}  ->  Macro F1 = {best_overall.macro_f1:.4f}")


# ── Entry Point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BANKSY consensus benchmark")
    parser.add_argument("--datasets", "-d", nargs="+", choices=list(DATASETS.keys()),
                        default=["rep1", "rep2"])
    parser.add_argument("--sample-size", "-s", type=int, default=None)
    args = parser.parse_args()

    t_start = time.time()
    run_benchmark(args.datasets, args.sample_size)
    total = time.time() - t_start
    logger.info(f"\nTotal benchmark time: {total/60:.1f} min")
