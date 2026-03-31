#!/usr/bin/env python3
"""
COMPLETE STHELAR annotation pipeline (Giraud-Sauveur et al. 2026).

Replicates ALL automated steps from the paper on breast_s0:
  1. Tangram with DISCO breast reference -> initial per-cell labels
  2. Leiden clustering at multiple resolutions (0.2, 0.4, 0.6)
  3. Wilcoxon DGE per cluster -> automated cluster labeling
  4. scVI validation (independent VAE per slide)
  5. Confidence scoring (KNN entropy + RNA depth)
  6. Standardize to 10 STHELAR categories + evaluate against GT

Data paths:
  Spatial: table_cells (raw counts in layers["count"], log-norm in X)
  Reference: DISCO breast v2.1 (174K cells, 42 types)
  GT: table_combined.final_label_combined

Usage:
    uv run python scripts/sthelar_annotation_pipeline.py
"""

import json
import logging
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("/mnt/work/git/dapidl/pipeline_output/sthelar_pipeline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DISCO cell type -> STHELAR 10-category mapping
# ============================================================
DISCO_TO_STHELAR = {
    # Epithelial (12 types)
    "SFN mammary luminal progenitor": "Epithelial",
    "KRT6B mammary basal cell": "Epithelial",
    "KRT17 mammary luminal cell": "Epithelial",
    "PIP mammary luminal cell": "Epithelial",
    "SAA2 mammary luminal progenitor": "Epithelial",
    "CXCL14 mammary basal cell": "Epithelial",
    "Secretoglobin mammary luminal progenitor": "Epithelial",
    "SCGB3A1 mammary luminal progenitor": "Epithelial",
    "Secretoglobin mammary luminal cell": "Epithelial",
    "CCSER1 mammary basal cell": "Epithelial",
    "Cycling mammary luminal progenitor": "Epithelial",
    "Lactocyte": "Epithelial",
    # Fibroblast / Myofibroblast (6 types)
    "APOD+PTGDS+ fibroblast": "Fibroblast_Myofibroblast",
    "CFD+MGP+ fibroblast": "Fibroblast_Myofibroblast",
    "CDH19+LAMA2+ fibroblast": "Fibroblast_Myofibroblast",
    "MFAP5+IGFBP6+ fibroblast": "Fibroblast_Myofibroblast",
    "GPC3+ fibroblast": "Fibroblast_Myofibroblast",
    "BNC2+ZFPM2+ fibroblast": "Fibroblast_Myofibroblast",
    # Blood vessel (9 types)
    "Capillary EC": "Blood_vessel",
    "Venous EC": "Blood_vessel",
    "Arterial EC": "Blood_vessel",
    "Lymphatic EC": "Blood_vessel",
    "Pericyte": "Blood_vessel",
    "CXCL+ pericyte": "Blood_vessel",
    "CCL19/21 pericyte": "Blood_vessel",
    "Vascular smooth muscle cell": "Blood_vessel",
    "CREB+MT1A+ vascular smooth muscle cell": "Blood_vessel",
    # T / NK cells (5 types)
    "CD4 T cell": "T_NK",
    "GZMB CD8 T cell": "T_NK",
    "GZMK CD8 T cell": "T_NK",
    "Treg cell": "T_NK",
    "NK cell": "T_NK",
    "ILC": "T_NK",
    # B / Plasma cells (2 types)
    "B cell": "B_Plasma",
    "Plasma cell": "B_Plasma",
    # Myeloid (6 types)
    "M1 macrophage": "Myeloid",
    "Macrophage": "Myeloid",
    "LYVE1 macrophage": "Myeloid",
    "Monocyte": "Myeloid",
    "Dendritic cell": "Myeloid",
    "pDC": "Myeloid",
    # Specialized
    "Mast cell": "Specialized",
}

# Marker genes for each STHELAR category (for Wilcoxon DGE validation)
CATEGORY_MARKERS = {
    "Epithelial": {
        "positive": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "KRT7", "MUC1",
                      "KRT14", "KRT5", "KRT17", "KRT6B", "GATA3"],
        "negative": ["PTPRC", "PECAM1", "COL1A1"],
    },
    "Blood_vessel": {
        "positive": ["PECAM1", "VWF", "CLDN5", "KDR", "CDH5", "ACTA2", "PDGFRB",
                      "RGS5", "MCAM", "AQP1", "CAV1", "CD93"],
        "negative": ["EPCAM", "PTPRC"],
    },
    "Fibroblast_Myofibroblast": {
        "positive": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "FAP", "PDGFRA",
                      "VIM", "FN1", "POSTN", "CTHRC1"],
        "negative": ["EPCAM", "PTPRC", "PECAM1"],
    },
    "Myeloid": {
        "positive": ["CD68", "CD163", "CSF1R", "LYZ", "CD14", "S100A8", "S100A9",
                      "ITGAX", "AIF1", "FCER1G"],
        "negative": ["CD3D", "MS4A1", "EPCAM"],
    },
    "B_Plasma": {
        "positive": ["CD19", "CD79A", "CD79B", "MS4A1", "PAX5", "SDC1", "MZB1",
                      "JCHAIN", "IGHG1", "BANK1"],
        "negative": ["CD3D", "CD68", "EPCAM"],
    },
    "T_NK": {
        "positive": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC", "NKG7", "GNLY",
                      "NCAM1", "GZMB", "CD8A", "IL7R"],
        "negative": ["MS4A1", "CD68", "EPCAM"],
    },
    "Specialized": {
        "positive": ["KIT", "TPSAB1", "CPA3", "TPSB2", "HDC", "HPGD"],
        "negative": ["CD3D", "CD68", "EPCAM", "PECAM1"],
    },
    "Melanocyte": {
        "positive": ["MLANA", "PMEL", "TYR", "TYRP1", "DCT", "SOX10", "MITF"],
        "negative": ["EPCAM", "PTPRC", "COL1A1"],
    },
    "Glioblastoma": {
        "positive": ["GFAP", "SOX2", "NES", "OLIG2"],
        "negative": ["EPCAM", "PTPRC"],
    },
}

STHELAR_CATEGORIES = [
    "Epithelial", "Blood_vessel", "Fibroblast_Myofibroblast",
    "Myeloid", "B_Plasma", "T_NK", "Melanocyte",
    "Glioblastoma", "Specialized", "Other",
]


# ============================================================
# STEP 0: Load data
# ============================================================

def load_data():
    """Load DISCO reference and STHELAR spatial data with GT labels."""
    log.info("=" * 70)
    log.info("STEP 0: Loading data")
    log.info("=" * 70)

    # Load DISCO reference
    t0 = time.time()
    log.info("Loading DISCO breast atlas...")
    ref = sc.read_h5ad("/mnt/work/datasets/DISCO/disco_breast_v2.1.h5ad")
    log.info(f"  Reference: {ref.n_obs:,} cells, {ref.n_vars:,} genes ({time.time()-t0:.1f}s)")

    # Load spatial data
    t0 = time.time()
    log.info("Loading STHELAR breast_s0 spatial data (table_cells)...")
    adata = ad.read_zarr(
        "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s0.zarr/"
        "sdata_breast_s0.zarr/tables/table_cells"
    )
    log.info(f"  Spatial: {adata.n_obs:,} cells, {adata.n_vars} genes ({time.time()-t0:.1f}s)")

    # Swap X to raw counts for Tangram (must be raw counts, not log-norm)
    log.info("Setting X to raw counts from layers['count']...")
    adata.layers["log_norm_original"] = adata.X.copy()
    adata.X = adata.layers["count"].copy()
    if sp.issparse(adata.X):
        adata.X = adata.X.astype(np.float32)

    # Verify raw counts
    if sp.issparse(adata.X):
        xmax = adata.X.data.max()
    else:
        xmax = float(np.max(adata.X))
    log.info(f"  X max after swap: {xmax} (should be integer-valued raw counts)")

    # Load GT labels from table_combined
    log.info("Loading ground truth labels from table_combined...")
    gt = ad.read_zarr(
        "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s0.zarr/"
        "sdata_breast_s0.zarr/tables/table_combined"
    )
    adata.obs["gt_label"] = gt.obs["final_label_combined"].values
    adata.obs["gt_tangram"] = gt.obs["ct_tangram"].values  # Paper's original Tangram labels
    adata.obs["transcript_cat"] = gt.obs["transcript_cat"].values

    log.info(f"  GT label distribution:")
    for lbl, cnt in adata.obs["gt_label"].value_counts().items():
        log.info(f"    {lbl:<30s}: {cnt:>8,} ({100*cnt/adata.n_obs:.1f}%)")

    return ref, adata


# ============================================================
# STEP 1: Tangram annotation
# ============================================================

def step1_tangram(adata, ref):
    """Run Tangram annotation with DISCO breast reference via sopa."""
    log.info("=" * 70)
    log.info("STEP 1: Tangram annotation with DISCO breast reference")
    log.info("=" * 70)

    from sopa.utils.annotation import tangram_annotate
    import torch

    # Clear GPU cache before Tangram
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    gene_overlap = len(set(ref.var_names) & set(adata.var_names))
    log.info(f"  Reference cell types: {ref.obs['cell_type'].nunique()}")
    log.info(f"  Spatial cells: {adata.n_obs:,}")
    log.info(f"  Gene overlap: {gene_overlap}")
    log.info(f"  GPU free: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    # Use bag_size=5000 to reduce per-chunk memory (10K x 10K mapping matrix is ~20GB)
    t0 = time.time()
    tangram_annotate(
        adata,
        ref,
        cell_type_key="cell_type",
        bag_size=5_000,
        max_obs_reference=10_000,
        density_prior="rna_count_based",
        clip_percentile=0.95,
    )
    elapsed = time.time() - t0
    log.info(f"Tangram completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Tangram places results in adata.obs["cell_type"]
    tangram_key = "cell_type"
    if tangram_key in adata.obs.columns:
        adata.obs["tangram_fine"] = adata.obs[tangram_key].astype(str)
        adata.obs["tangram_sthelar"] = adata.obs["tangram_fine"].map(DISCO_TO_STHELAR).fillna("Other")
        log.info(f"  Tangram unique types: {adata.obs['tangram_fine'].nunique()}")
        log.info(f"  Tangram -> STHELAR distribution:")
        for lbl, cnt in adata.obs["tangram_sthelar"].value_counts().items():
            log.info(f"    {lbl:<30s}: {cnt:>8,} ({100*cnt/adata.n_obs:.1f}%)")
    else:
        log.error(f"Tangram did not produce '{tangram_key}' column. Available: {list(adata.obs.columns)}")
        sys.exit(1)

    return adata, elapsed


# ============================================================
# STEP 2: Leiden clustering refinement
# ============================================================

def step2_leiden(adata, resolutions=(0.2, 0.4, 0.6)):
    """Leiden clustering at multiple resolutions following STHELAR protocol."""
    log.info("=" * 70)
    log.info("STEP 2: Leiden clustering refinement")
    log.info("=" * 70)

    t0 = time.time()

    # Work on a copy with raw counts -> normalize fresh
    adata_proc = adata.copy()

    # Filter cells with <10 transcripts (STHELAR threshold)
    if "transcript_counts" in adata_proc.obs.columns:
        total_counts = adata_proc.obs["transcript_counts"].values.astype(float)
    elif sp.issparse(adata_proc.X):
        total_counts = np.array(adata_proc.X.sum(axis=1)).flatten()
    else:
        total_counts = adata_proc.X.sum(axis=1)

    mask = total_counts >= 10
    n_before = adata_proc.n_obs
    adata_proc = adata_proc[mask].copy()
    log.info(f"  Filtered: {n_before - adata_proc.n_obs:,} cells with <10 transcripts "
             f"({adata_proc.n_obs:,} remaining)")

    # Normalize, log1p, scale (STHELAR exact protocol)
    log.info("  Normalizing (target_sum=1e4), log1p, scale(max_value=10)...")
    sc.pp.normalize_total(adata_proc, target_sum=1e4)
    sc.pp.log1p(adata_proc)

    # Store log-normalized for DGE (Wilcoxon needs unscaled log-norm)
    adata_proc.layers["log_norm"] = adata_proc.X.copy()

    sc.pp.scale(adata_proc, max_value=10)

    # Clean up NaN/inf from scaling
    if sp.issparse(adata_proc.X):
        adata_proc.X = np.asarray(adata_proc.X.toarray(), dtype=np.float32)
    adata_proc.X = np.nan_to_num(np.asarray(adata_proc.X, dtype=np.float32))

    # Remove zero-variance genes (PCA fails on them)
    gene_var = np.var(adata_proc.X, axis=0)
    keep = gene_var > 0
    log.info(f"  Keeping {keep.sum()} / {len(keep)} genes with nonzero variance")
    adata_proc = adata_proc[:, keep].copy()

    # PCA: STHELAR uses n_pcs=16, svd_solver=arpack, no HVG selection
    n_pcs = min(16, adata_proc.n_vars - 1)
    log.info(f"  PCA with {n_pcs} components (svd_solver=arpack)...")
    sc.tl.pca(adata_proc, n_comps=n_pcs, svd_solver="arpack")

    # Neighbors: STHELAR uses n_neighbors=10
    log.info(f"  Computing neighbors (n_neighbors=10, n_pcs={n_pcs})...")
    sc.pp.neighbors(adata_proc, n_neighbors=10, n_pcs=n_pcs)

    # Leiden at multiple resolutions
    leiden_info = {}
    for res in resolutions:
        key = f"leiden_r{res}"
        log.info(f"  Leiden resolution={res}...")
        sc.tl.leiden(adata_proc, resolution=res, key_added=key)
        n_clusters = adata_proc.obs[key].nunique()
        log.info(f"    -> {n_clusters} clusters")
        leiden_info[res] = {"key": key, "n_clusters": n_clusters}

    # Pick optimal resolution: STHELAR targets 10-20 clusters
    best_res = None
    for res in sorted(resolutions, reverse=True):
        n_cl = leiden_info[res]["n_clusters"]
        if 10 <= n_cl <= 20:
            best_res = res
            break
    if best_res is None:
        # Pick closest to 15
        best_res = min(resolutions, key=lambda r: abs(leiden_info[r]["n_clusters"] - 15))
    best_key = f"leiden_r{best_res}"
    log.info(f"  Selected resolution: {best_res} ({leiden_info[best_res]['n_clusters']} clusters)")

    # Copy results back to original adata
    for res in resolutions:
        key = f"leiden_r{res}"
        adata.obs[key] = "filtered"
        adata.obs.loc[adata_proc.obs_names, key] = adata_proc.obs[key].values

    elapsed = time.time() - t0
    log.info(f"Leiden clustering completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return adata, adata_proc, best_key, best_res, leiden_info, elapsed


# ============================================================
# STEP 3: Wilcoxon DGE + automated cluster labeling
# ============================================================

def step3_dge_labeling(adata, adata_proc, cluster_key):
    """Wilcoxon DGE per cluster + automated label assignment."""
    log.info("=" * 70)
    log.info(f"STEP 3: Wilcoxon DGE + cluster labeling ({cluster_key})")
    log.info("=" * 70)

    t0 = time.time()

    # Use log-normalized (unscaled) data for Wilcoxon DGE
    a = adata_proc.copy()
    a.X = a.layers["log_norm"].copy()
    if sp.issparse(a.X):
        a.X = np.asarray(a.X.toarray(), dtype=np.float32)
    a.X = np.asarray(a.X, dtype=np.float32)

    log.info("  Running Wilcoxon rank-sum test per cluster...")
    sc.tl.rank_genes_groups(a, cluster_key, method="wilcoxon", use_raw=False,
                            corr_method="benjamini-hochberg")

    gene_names = list(a.var_names)
    expr = a.X

    cluster_labels = {}
    for cluster in sorted(a.obs[cluster_key].unique()):
        mask = (a.obs[cluster_key] == cluster).values
        n_cells = int(mask.sum())

        # Top DEGs
        try:
            top_genes = [str(a.uns["rank_genes_groups"]["names"][i][cluster])
                         for i in range(min(30, len(a.uns["rank_genes_groups"]["names"])))]
            top_genes = [g for g in top_genes if g != "nan"]
        except (KeyError, IndexError):
            top_genes = []

        # Tangram majority vote for this cluster
        tangram_in_cluster = adata.obs.loc[a.obs_names[mask], "tangram_fine"]
        tangram_counts = Counter(tangram_in_cluster)
        tangram_majority = tangram_counts.most_common(1)[0][0] if tangram_counts else "Unknown"
        tangram_sthelar = DISCO_TO_STHELAR.get(tangram_majority, "Other")

        # Tangram majority fraction
        total_in_cluster = sum(tangram_counts.values())
        tangram_frac = tangram_counts.most_common(1)[0][1] / total_in_cluster if total_in_cluster > 0 else 0

        # Score cluster against marker gene sets
        cl_expr = expr[mask]
        marker_scores = {}
        for cat, markers in CATEGORY_MARKERS.items():
            pos = [g for g in markers["positive"] if g in gene_names]
            neg = [g for g in markers.get("negative", []) if g in gene_names]
            if not pos:
                marker_scores[cat] = 0.0
                continue
            pos_idx = [gene_names.index(g) for g in pos]
            score = float(cl_expr[:, pos_idx].mean())
            if neg:
                neg_idx = [gene_names.index(g) for g in neg]
                score -= 0.5 * float(cl_expr[:, neg_idx].mean())
            marker_scores[cat] = score

        marker_best = max(marker_scores, key=marker_scores.get)
        marker_best_score = marker_scores[marker_best]

        # Decision rule (STHELAR paper: markers override when they disagree)
        if tangram_sthelar == marker_best:
            final_label = tangram_sthelar
            decision = "agree"
        elif marker_best_score > 0.5:
            final_label = marker_best
            decision = "markers_override"
        elif tangram_sthelar != "Other":
            final_label = tangram_sthelar
            decision = "tangram_fallback"
        else:
            final_label = marker_best if marker_best_score > 0 else "Other"
            decision = "low_confidence"

        cluster_labels[cluster] = {
            "final_label": final_label,
            "tangram_majority": tangram_majority,
            "tangram_sthelar": tangram_sthelar,
            "tangram_frac": round(tangram_frac, 3),
            "marker_best": marker_best,
            "marker_score": round(marker_best_score, 3),
            "decision": decision,
            "n_cells": n_cells,
            "top_degs": top_genes[:10],
            "top3_tangram": dict(Counter(tangram_in_cluster).most_common(3)),
        }

        log.info(f"  Cluster {cluster:>3s} ({n_cells:>6,} cells): "
                 f"{final_label:<25s} [{decision:>16s}] "
                 f"tangram={tangram_sthelar:<15s} markers={marker_best}({marker_best_score:.2f})")

    # Assign labels to cells
    label_map = {cl: info["final_label"] for cl, info in cluster_labels.items()}
    adata.obs["pipeline_label"] = adata.obs[cluster_key].map(label_map).fillna("Other")

    # Filtered cells -> "Other"
    adata.obs.loc[adata.obs[cluster_key] == "filtered", "pipeline_label"] = "Other"

    elapsed = time.time() - t0
    log.info(f"\nCluster labeling completed in {elapsed:.1f}s")

    # Report distribution
    log.info("  Pipeline label distribution:")
    for lbl, cnt in adata.obs["pipeline_label"].value_counts().items():
        log.info(f"    {lbl:<30s}: {cnt:>8,} ({100*cnt/adata.n_obs:.1f}%)")

    return adata, cluster_labels, elapsed


# ============================================================
# STEP 4: scVI validation
# ============================================================

def step4_scvi_validation(adata, adata_proc, cluster_key):
    """Train scVI model for independent latent representation validation."""
    log.info("=" * 70)
    log.info("STEP 4: scVI validation (independent VAE)")
    log.info("=" * 70)

    import scvi

    t0 = time.time()

    # Work on the filtered cells with raw counts
    a = adata_proc.copy()
    # Restore raw counts from original adata
    raw_counts = adata[a.obs_names].layers["count"].copy()
    if sp.issparse(raw_counts):
        raw_counts = raw_counts.toarray()
    a.layers["count"] = np.asarray(raw_counts, dtype=np.float32)

    log.info(f"  Setting up scVI on {a.n_obs:,} cells x {a.n_vars} genes...")
    scvi.model.SCVI.setup_anndata(a, layer="count")

    model = scvi.model.SCVI(a, n_latent=10, n_layers=2)
    log.info("  Training scVI model (GPU, early stopping)...")

    model.train(
        accelerator="gpu",
        early_stopping=True,
        max_epochs=100,
        batch_size=256,
        train_size=0.9,
    )

    # Get latent representation
    log.info("  Extracting latent representation...")
    latent = model.get_latent_representation()
    a.obsm["X_scVI"] = latent
    log.info(f"  scVI latent shape: {latent.shape}")

    # Compute neighbors in scVI space
    log.info("  Computing neighbors in scVI space...")
    sc.pp.neighbors(a, use_rep="X_scVI", n_neighbors=10)

    # Leiden on scVI space
    sc.tl.leiden(a, resolution=0.4, key_added="leiden_scvi")
    n_scvi_clusters = a.obs["leiden_scvi"].nunique()
    log.info(f"  scVI Leiden clusters: {n_scvi_clusters}")

    # Compute ARI between pipeline labels and scVI clusters
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    pipeline_labels = adata.obs.loc[a.obs_names, "pipeline_label"].values
    scvi_clusters = a.obs["leiden_scvi"].values

    ari = adjusted_rand_score(pipeline_labels, scvi_clusters)
    nmi = normalized_mutual_info_score(pipeline_labels, scvi_clusters)

    log.info(f"  Pipeline labels vs scVI clusters:")
    log.info(f"    ARI: {ari:.4f}")
    log.info(f"    NMI: {nmi:.4f}")

    # Also check: for each scVI cluster, is there a dominant pipeline label?
    cluster_purity = []
    for cl in sorted(a.obs["leiden_scvi"].unique()):
        cl_mask = a.obs["leiden_scvi"] == cl
        cl_labels = adata.obs.loc[a.obs_names[cl_mask], "pipeline_label"]
        majority_frac = cl_labels.value_counts().iloc[0] / len(cl_labels)
        cluster_purity.append(majority_frac)
    mean_purity = float(np.mean(cluster_purity))
    log.info(f"    Mean cluster purity: {mean_purity:.4f}")

    # Store scVI latent in main adata
    adata.obsm["X_scVI"] = np.zeros((adata.n_obs, latent.shape[1]))
    obs_idx = np.isin(adata.obs_names, a.obs_names)
    adata.obsm["X_scVI"][obs_idx] = latent

    elapsed = time.time() - t0
    log.info(f"scVI validation completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    scvi_results = {
        "n_latent_dims": int(latent.shape[1]),
        "n_scvi_clusters": n_scvi_clusters,
        "ari_vs_pipeline": round(float(ari), 4),
        "nmi_vs_pipeline": round(float(nmi), 4),
        "mean_cluster_purity": round(mean_purity, 4),
        "training_epochs": len(model.history["elbo_train"]),
        "final_elbo": float(model.history["elbo_train"].iloc[-1].values[0]),
    }

    return adata, scvi_results, elapsed


# ============================================================
# STEP 5: Confidence scoring
# ============================================================

def step5_confidence_scoring(adata, adata_proc, cluster_key):
    """Compute per-cell confidence scores using KNN entropy + RNA depth."""
    log.info("=" * 70)
    log.info("STEP 5: Confidence scoring (KNN entropy + RNA depth)")
    log.info("=" * 70)

    t0 = time.time()

    # Initialize confidence arrays
    adata.obs["knn_entropy"] = np.nan
    adata.obs["knn_confidence"] = np.nan
    adata.obs["rna_depth_quantile"] = np.nan
    adata.obs["combined_confidence"] = np.nan

    # Work on filtered cells that have the kNN graph
    valid_mask = adata.obs[cluster_key] != "filtered"
    valid_obs = adata.obs_names[valid_mask]
    log.info(f"  Computing confidence for {valid_mask.sum():,} cells (excluding {(~valid_mask).sum():,} filtered)")

    # Get the KNN graph from adata_proc (connectivities from Leiden step)
    if "connectivities" not in adata_proc.obsp:
        log.warning("  No KNN graph found. Re-computing neighbors...")
        sc.pp.neighbors(adata_proc, n_neighbors=10, n_pcs=16)

    conn = adata_proc.obsp["connectivities"]
    labels = adata.obs.loc[adata_proc.obs_names, "pipeline_label"].values

    # KNN entropy: for each cell, Shannon entropy of label distribution among neighbors
    log.info("  Computing KNN label entropy...")
    n_cells = conn.shape[0]
    entropy_arr = np.zeros(n_cells, dtype=np.float32)
    unique_labels = sorted(set(labels))
    n_labels = len(unique_labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    # Process in chunks for memory efficiency
    chunk_size = 10000
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        if start % 50000 == 0:
            log.info(f"    Processing cells {start:,} - {end:,} / {n_cells:,}")

        for i in range(start, end):
            # Get neighbors of cell i
            row = conn[i]
            if sp.issparse(row):
                neighbor_idx = row.indices
            else:
                neighbor_idx = np.nonzero(row)[0]

            if len(neighbor_idx) == 0:
                entropy_arr[i] = 0.0
                continue

            # Count label distribution among neighbors
            neighbor_labels = labels[neighbor_idx]
            counts = Counter(neighbor_labels)
            total = sum(counts.values())

            # Shannon entropy
            entropy = 0.0
            for count in counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)

            entropy_arr[i] = entropy

    # Normalize entropy: confidence = 1 - (entropy / max_possible_entropy)
    max_entropy = np.log2(n_labels) if n_labels > 1 else 1.0
    knn_confidence = 1.0 - (entropy_arr / max_entropy)
    knn_confidence = np.clip(knn_confidence, 0.0, 1.0)

    adata.obs.loc[adata_proc.obs_names, "knn_entropy"] = entropy_arr
    adata.obs.loc[adata_proc.obs_names, "knn_confidence"] = knn_confidence

    log.info(f"  KNN confidence: mean={knn_confidence.mean():.3f}, "
             f"median={np.median(knn_confidence):.3f}, "
             f">0.5: {(knn_confidence > 0.5).sum():,} ({100*(knn_confidence > 0.5).mean():.1f}%)")

    # RNA depth score: log1p(transcript_counts), rank-based quantile
    log.info("  Computing RNA depth scores...")
    if "transcript_counts" in adata.obs.columns:
        tc = adata.obs["transcript_counts"].values.astype(float)
    elif sp.issparse(adata.X):
        tc = np.array(adata.X.sum(axis=1)).flatten()
    else:
        tc = adata.X.sum(axis=1)

    log_tc = np.log1p(tc)
    # Rank-based quantile (within this slide)
    from scipy.stats import rankdata
    ranks = rankdata(log_tc, method="average")
    rna_quantile = ranks / len(ranks)

    adata.obs["rna_depth_quantile"] = rna_quantile

    # Combined confidence: geometric mean of KNN confidence and RNA depth quantile
    # Only for valid (non-filtered) cells
    combined = np.zeros(adata.n_obs, dtype=np.float32)
    valid_idx = np.isin(adata.obs_names, adata_proc.obs_names)

    knn_conf_full = adata.obs["knn_confidence"].values.astype(float)
    rna_q = adata.obs["rna_depth_quantile"].values.astype(float)

    # Geometric mean, handling NaN
    combined[valid_idx] = np.sqrt(
        np.clip(knn_conf_full[valid_idx], 0, 1) *
        np.clip(rna_q[valid_idx], 0, 1)
    )
    combined[~valid_idx] = 0.0

    adata.obs["combined_confidence"] = combined

    log.info(f"  Combined confidence: mean={combined[valid_idx].mean():.3f}, "
             f"median={np.median(combined[valid_idx]):.3f}")

    # Confidence thresholds
    thresholds = [0.3, 0.5, 0.7, 0.8]
    for thr in thresholds:
        n_above = (combined > thr).sum()
        log.info(f"    Confidence > {thr}: {n_above:,} ({100*n_above/adata.n_obs:.1f}%)")

    elapsed = time.time() - t0
    log.info(f"Confidence scoring completed in {elapsed:.1f}s")

    confidence_results = {
        "knn_confidence_mean": round(float(knn_confidence.mean()), 4),
        "knn_confidence_median": round(float(np.median(knn_confidence)), 4),
        "rna_depth_quantile_mean": round(float(rna_quantile.mean()), 4),
        "combined_confidence_mean": round(float(combined[valid_idx].mean()), 4),
        "combined_confidence_median": round(float(np.median(combined[valid_idx])), 4),
        "pct_above_0.3": round(float(100 * (combined > 0.3).sum() / adata.n_obs), 2),
        "pct_above_0.5": round(float(100 * (combined > 0.5).sum() / adata.n_obs), 2),
        "pct_above_0.7": round(float(100 * (combined > 0.7).sum() / adata.n_obs), 2),
        "pct_above_0.8": round(float(100 * (combined > 0.8).sum() / adata.n_obs), 2),
    }

    return adata, confidence_results, elapsed


# ============================================================
# STEP 6: Evaluate against GT
# ============================================================

def step6_evaluate(adata, gt_column="gt_label"):
    """Evaluate pipeline predictions against STHELAR ground truth."""
    log.info("=" * 70)
    log.info("STEP 6: Evaluation against ground truth")
    log.info("=" * 70)

    gt = adata.obs[gt_column].astype(str).values
    pred = adata.obs["pipeline_label"].astype(str).values

    # Filter out Less10 and Other from GT
    eval_mask = ~np.isin(gt, ["Less10", "Other", "Unknown"])
    gt_eval = gt[eval_mask]
    pred_eval = pred[eval_mask]

    classes = sorted(set(gt_eval) | set(pred_eval))
    acc = accuracy_score(gt_eval, pred_eval)
    f1_macro = f1_score(gt_eval, pred_eval, average="macro", zero_division=0, labels=classes)
    f1_weighted = f1_score(gt_eval, pred_eval, average="weighted", zero_division=0, labels=classes)

    log.info(f"  Cells evaluated: {eval_mask.sum():,} / {adata.n_obs:,}")
    log.info(f"  Accuracy:     {acc:.4f}")
    log.info(f"  F1 (macro):   {f1_macro:.4f}")
    log.info(f"  F1 (weighted): {f1_weighted:.4f}")

    # Per-class metrics
    prec, rec, f1, sup = precision_recall_fscore_support(
        gt_eval, pred_eval, labels=classes, zero_division=0
    )
    log.info(f"\n  {'Class':<30s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Support':>8s}")
    log.info(f"  {'-'*60}")
    per_class = {}
    for i, c in enumerate(classes):
        if sup[i] > 0:
            log.info(f"  {c:<30s} {prec[i]:>7.3f} {rec[i]:>7.3f} {f1[i]:>7.3f} {sup[i]:>8,}")
        per_class[c] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(sup[i]),
        }

    # Confusion matrix
    labels_for_cm = sorted(set(gt_eval))
    cm = confusion_matrix(gt_eval, pred_eval, labels=labels_for_cm)
    log.info(f"\n  Confusion matrix (rows=GT, cols=Pred):")
    log.info(f"  Labels: {labels_for_cm}")
    for i, row in enumerate(cm):
        log.info(f"  {labels_for_cm[i]:>25s}: {row}")

    # --- Also evaluate raw Tangram (before Leiden refinement) ---
    log.info(f"\n  --- Raw Tangram evaluation (no Leiden refinement) ---")
    tangram_pred = adata.obs["tangram_sthelar"].astype(str).values
    tangram_eval = tangram_pred[eval_mask]
    tangram_acc = accuracy_score(gt_eval, tangram_eval)
    tangram_f1_macro = f1_score(gt_eval, tangram_eval, average="macro", zero_division=0)
    tangram_f1_weighted = f1_score(gt_eval, tangram_eval, average="weighted", zero_division=0)
    log.info(f"  Tangram raw: Acc={tangram_acc:.4f}, F1_macro={tangram_f1_macro:.4f}, F1_weighted={tangram_f1_weighted:.4f}")

    # --- Evaluate with confidence filtering ---
    confidence_filtered_metrics = {}
    for thr in [0.3, 0.5, 0.7]:
        conf = adata.obs["combined_confidence"].values
        conf_mask = eval_mask & (conf > thr)
        if conf_mask.sum() < 100:
            continue
        gt_c = gt[conf_mask]
        pred_c = pred[conf_mask]
        acc_c = accuracy_score(gt_c, pred_c)
        f1_c = f1_score(gt_c, pred_c, average="macro", zero_division=0)
        retention = conf_mask.sum() / eval_mask.sum()
        log.info(f"  Confidence > {thr}: Acc={acc_c:.4f}, F1_macro={f1_c:.4f}, "
                 f"retention={100*retention:.1f}% ({conf_mask.sum():,} cells)")
        confidence_filtered_metrics[f"conf_gt_{thr}"] = {
            "accuracy": round(float(acc_c), 4),
            "f1_macro": round(float(f1_c), 4),
            "retention": round(float(retention), 4),
            "n_cells": int(conf_mask.sum()),
        }

    metrics = {
        "n_evaluated": int(eval_mask.sum()),
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels_for_cm,
            "matrix": cm.tolist(),
        },
        "tangram_raw": {
            "accuracy": round(float(tangram_acc), 4),
            "f1_macro": round(float(tangram_f1_macro), 4),
            "f1_weighted": round(float(tangram_f1_weighted), 4),
        },
        "confidence_filtered": confidence_filtered_metrics,
    }

    return metrics


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    log.info("#" * 70)
    log.info("#  COMPLETE STHELAR ANNOTATION PIPELINE")
    log.info("#  Slide: breast_s0 | Reference: DISCO breast v2.1")
    log.info("#  Steps: Tangram -> Leiden -> DGE -> scVI -> Confidence -> Eval")
    log.info("#" * 70)

    total_t0 = time.time()
    timings = {}

    # Step 0: Load data
    ref, adata = load_data()

    # Step 1: Tangram annotation
    adata, tangram_elapsed = step1_tangram(adata, ref)
    timings["tangram"] = tangram_elapsed

    # Free reference memory
    del ref
    import gc
    gc.collect()

    # Step 2: Leiden clustering
    adata, adata_proc, best_key, best_res, leiden_info, leiden_elapsed = step2_leiden(adata)
    timings["leiden"] = leiden_elapsed

    # Step 3: DGE + cluster labeling
    adata, cluster_labels, dge_elapsed = step3_dge_labeling(adata, adata_proc, best_key)
    timings["dge_labeling"] = dge_elapsed

    # Step 4: scVI validation
    adata, scvi_results, scvi_elapsed = step4_scvi_validation(adata, adata_proc, best_key)
    timings["scvi"] = scvi_elapsed

    # Step 5: Confidence scoring
    adata, confidence_results, conf_elapsed = step5_confidence_scoring(adata, adata_proc, best_key)
    timings["confidence"] = conf_elapsed

    # Step 6: Evaluate
    metrics = step6_evaluate(adata)
    timings["evaluation"] = 0.0  # negligible

    # ---- ASSEMBLE FINAL RESULTS ----
    total_elapsed = time.time() - total_t0
    timings["total"] = total_elapsed

    results = {
        "pipeline": "STHELAR_complete",
        "slide": "breast_s0",
        "reference": "DISCO_breast_v2.1",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "steps": {
            "1_tangram": {
                "method": "sopa.tangram_annotate",
                "bag_size": 10000,
                "max_obs_reference": 10000,
                "density_prior": "rna_count_based",
                "n_unique_types": int(adata.obs["tangram_fine"].nunique()),
                "elapsed_s": round(tangram_elapsed, 1),
            },
            "2_leiden": {
                "resolutions_tested": [0.2, 0.4, 0.6],
                "selected_resolution": best_res,
                "n_clusters": leiden_info[best_res]["n_clusters"],
                "all_resolutions": {
                    str(r): {"n_clusters": info["n_clusters"]}
                    for r, info in leiden_info.items()
                },
                "pca_n_comps": 16,
                "n_neighbors": 10,
                "elapsed_s": round(leiden_elapsed, 1),
            },
            "3_dge_labeling": {
                "method": "wilcoxon",
                "cluster_annotations": cluster_labels,
                "elapsed_s": round(dge_elapsed, 1),
            },
            "4_scvi_validation": scvi_results | {"elapsed_s": round(scvi_elapsed, 1)},
            "5_confidence": confidence_results | {"elapsed_s": round(conf_elapsed, 1)},
        },
        "metrics": metrics,
        "timings": {k: round(v, 1) for k, v in timings.items()},
        "label_distribution": adata.obs["pipeline_label"].value_counts().to_dict(),
    }

    # Save results
    output_file = OUTPUT_DIR / "complete_pipeline_breast_s0.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {output_file}")

    # Print final summary
    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)
    log.info(f"Total pipeline time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    for step, t in timings.items():
        if step != "total":
            log.info(f"  {step:<20s}: {t:>8.1f}s ({t/60:.1f} min)")

    log.info(f"\nFull pipeline metrics:")
    log.info(f"  Accuracy:     {metrics['accuracy']:.4f}")
    log.info(f"  F1 (macro):   {metrics['f1_macro']:.4f}")
    log.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")

    log.info(f"\nRaw Tangram (baseline):")
    log.info(f"  Accuracy:     {metrics['tangram_raw']['accuracy']:.4f}")
    log.info(f"  F1 (macro):   {metrics['tangram_raw']['f1_macro']:.4f}")

    log.info(f"\nscVI validation:")
    log.info(f"  ARI vs pipeline:    {scvi_results['ari_vs_pipeline']:.4f}")
    log.info(f"  NMI vs pipeline:    {scvi_results['nmi_vs_pipeline']:.4f}")
    log.info(f"  Cluster purity:     {scvi_results['mean_cluster_purity']:.4f}")

    log.info(f"\nConfidence scoring:")
    log.info(f"  Mean combined:  {confidence_results['combined_confidence_mean']:.4f}")
    log.info(f"  % > 0.5:       {confidence_results['pct_above_0.5']:.1f}%")
    log.info(f"  % > 0.7:       {confidence_results['pct_above_0.7']:.1f}%")

    if metrics.get("confidence_filtered"):
        log.info(f"\nConfidence-filtered metrics:")
        for k, v in metrics["confidence_filtered"].items():
            log.info(f"  {k}: Acc={v['accuracy']:.4f}, F1={v['f1_macro']:.4f}, "
                     f"retention={100*v['retention']:.1f}%")

    return results


if __name__ == "__main__":
    main()
