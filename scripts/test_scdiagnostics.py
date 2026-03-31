#!/usr/bin/env python3
"""Test scDiagnostics R package for annotation quality assessment.

Runs key scDiagnostics functions on Xenium breast rep1 data:
1. detectAnomaly() - Isolation forest on PCA space per cell type
2. calculateCategorizationEntropy() - Annotation confidence via entropy
3. calculateHotellingPValue() - Cluster-reference alignment test
4. calculateNearestNeighborProbabilities() - KNN-based dataset membership

Also implements Python equivalents for comparison:
- sklearn IsolationForest on PCA space
- Shannon entropy of KNN label distribution

Results saved to pipeline_output/annotation_benchmark_2026_03/scdiagnostics_results.json
"""

import gc
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from loguru import logger
from scipy.sparse import issparse

# Fix libstdc++ before rpy2
sys.path.insert(0, "/mnt/work/git/dapidl")
from dapidl.pipeline.components.annotators.singler import _fix_libstdcxx

_fix_libstdcxx()

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path("pipeline_output/annotation_benchmark_2026_03")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING (from annotation_benchmark_2026_03.py)
# ──────────────────────────────────────────────────────────────────────────────

XENIUM_BASE = Path("/mnt/work/datasets/raw/xenium")

GT_TO_COARSE = {
    "B_Cells": "Immune",
    "CD4+_T_Cells": "Immune",
    "CD8+_T_Cells": "Immune",
    "DCIS_1": "Epithelial",
    "DCIS_2": "Epithelial",
    "Endothelial": "Endothelial",
    "IRF7+_DCs": "Immune",
    "Invasive_Tumor": "Epithelial",
    "LAMP3+_DCs": "Immune",
    "Macrophages_1": "Immune",
    "Macrophages_2": "Immune",
    "Mast_Cells": "Immune",
    "Myoepi_ACTA2+": "Epithelial",
    "Myoepi_KRT15+": "Epithelial",
    "Perivascular-Like": "Stromal",
    "Prolif_Invasive_Tumor": "Epithelial",
    "Stromal": "Stromal",
    "Unlabeled": "Unknown",
    "Hybrid": "Unknown",
}


def load_xenium_adata(rep: str) -> ad.AnnData:
    """Load Xenium breast cancer replicate with ground truth."""
    if rep == "rep1":
        base = XENIUM_BASE / "xenium-breast-tumor-rep1"
        gt_file = base / "celltypes_ground_truth_rep1_supervised.xlsx"
    else:
        raise ValueError(f"Unknown rep: {rep}")

    outs = base / "outs" if (base / "outs").exists() else base

    h5_path = outs / "cell_feature_matrix.h5"
    logger.info(f"Loading {rep} from {h5_path}")
    adata = sc.read_10x_h5(str(h5_path))
    adata.var_names_make_unique()

    cells_path = outs / "cells.parquet"
    if cells_path.exists():
        cells_df = pd.read_parquet(cells_path)
        cells_df["cell_id"] = cells_df["cell_id"].astype(str)
        cells_df.index = cells_df["cell_id"]
        common = adata.obs_names.intersection(cells_df.index)
        adata = adata[common].copy()
        for col in ["x_centroid", "y_centroid"]:
            if col in cells_df.columns:
                adata.obs[col] = cells_df.loc[adata.obs_names, col].values

    if gt_file.exists():
        logger.info(f"Loading ground truth from {gt_file}")
        gt_df = pd.read_excel(gt_file)
        barcode_col = next(
            (c for c in gt_df.columns if "barcode" in c.lower()), gt_df.columns[0]
        )
        cluster_col = next(
            (
                c
                for c in gt_df.columns
                if c.lower() in ("cluster", "cell_type", "celltype", "ground_truth")
            ),
            gt_df.columns[1],
        )
        gt_df[barcode_col] = gt_df[barcode_col].astype(str)
        gt_df = gt_df.set_index(barcode_col)
        common_gt = adata.obs_names.intersection(gt_df.index)
        adata = adata[common_gt].copy()
        adata.obs["gt_fine"] = gt_df.loc[adata.obs_names, cluster_col].values
        adata.obs["gt_coarse"] = (
            adata.obs["gt_fine"].map(GT_TO_COARSE).fillna("Unknown")
        )
        mask = adata.obs["gt_coarse"] != "Unknown"
        n_before = len(adata)
        adata = adata[mask].copy()
        logger.info(
            f"  {rep}: {n_before} -> {len(adata)} cells (removed {n_before - len(adata)} Unknown/Hybrid)"
        )

    adata.obs["dataset"] = rep
    return adata


def preprocess_adata(adata: ad.AnnData) -> ad.AnnData:
    """Standard preprocessing for annotation methods."""
    a = adata.copy()
    if issparse(a.X):
        a.X = a.X.toarray()
    a.layers["raw"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


# ──────────────────────────────────────────────────────────────────────────────
# ANNOTATE WITH CELLTYPIST (to get predicted labels + probability scores)
# ──────────────────────────────────────────────────────────────────────────────


def annotate_with_celltypist(adata: ad.AnnData) -> ad.AnnData:
    """Add CellTypist predictions to adata.

    Uses majority_voting=False to avoid segfault on large datasets
    (neighborhood graph construction crashes at 158K+ cells).
    """
    import celltypist
    from celltypist import models as ct_models

    model_name = "Cells_Adult_Breast.pkl"
    logger.info(f"Running CellTypist with {model_name} (no majority voting)")
    try:
        model = ct_models.Model.load(model_name)
    except Exception:
        ct_models.download_models(model=model_name, force_update=False)
        model = ct_models.Model.load(model_name)

    predictions = celltypist.annotate(adata, model=model, majority_voting=False)
    result = predictions.to_adata()

    # Copy predictions back (predicted_labels instead of majority_voting)
    adata.obs["celltypist_label"] = result.obs["predicted_labels"].values
    adata.obs["celltypist_conf"] = result.obs["conf_score"].values

    # Store probability matrix for entropy calculation
    if hasattr(predictions, "probability_matrix"):
        prob_matrix = predictions.probability_matrix
        adata.obsm["celltypist_probs"] = prob_matrix.values
        adata.uns["celltypist_classes"] = list(prob_matrix.columns)
        logger.info(
            f"  CellTypist: {len(prob_matrix.columns)} classes, "
            f"mean conf={adata.obs['celltypist_conf'].mean():.3f}"
        )

    return adata


# ──────────────────────────────────────────────────────────────────────────────
# R-BASED scDiagnostics
# ──────────────────────────────────────────────────────────────────────────────


def adata_to_sce(adata: ad.AnnData, cell_type_col: str = "gt_coarse") -> object:
    """Convert AnnData to SingleCellExperiment via R.

    We do manual conversion rather than zellkonverter to avoid
    potential basilisk/reticulate conflicts. Uses rpy2 assign for all
    large objects to avoid string-length limits.
    """
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    base = importr("base")
    sce_pkg = importr("SingleCellExperiment")
    s4vectors = importr("S4Vectors")

    # Get expression matrix (cells x genes in Python -> genes x cells in R)
    if issparse(adata.X):
        expr = np.array(adata.X.toarray(), dtype=np.float64)
    else:
        expr = np.array(adata.X, dtype=np.float64)

    # R wants genes x cells -- flatten in column-major (Fortran) order for R
    expr_flat = expr.T.flatten(order="C")  # transpose then flatten row-by-row = R's column-major
    expr_r = ro.r["matrix"](
        ro.FloatVector(expr_flat.tolist()),
        nrow=expr.shape[1],
        ncol=expr.shape[0],
    )
    ro.r.assign("expr_mat", expr_r)

    # Assign row/column names via rpy2 vectors
    gene_names = ro.StrVector(adata.var_names.astype(str).tolist())
    cell_names = ro.StrVector(adata.obs_names.astype(str).tolist())
    ro.r.assign("gene_names_vec", gene_names)
    ro.r.assign("cell_names_vec", cell_names)
    ro.r("rownames(expr_mat) <- gene_names_vec")
    ro.r("colnames(expr_mat) <- cell_names_vec")

    # Build colData
    cell_types = adata.obs[cell_type_col].values.astype(str)
    ro.r.assign("cell_types_vec", ro.StrVector(cell_types.tolist()))

    # Create SCE with PCA in reducedDims (required by scDiagnostics)
    ro.r(
        """
        library(SingleCellExperiment)
        library(scater)

        col_df <- S4Vectors::DataFrame(
            cell_type = cell_types_vec,
            row.names = cell_names_vec
        )
        sce <- SingleCellExperiment(
            assays = list(logcounts = expr_mat),
            colData = col_df
        )

        # Run PCA via scater (scDiagnostics requires reducedDims(sce)$PCA)
        sce <- scater::runPCA(sce, ncomponents = 20, assay.type = "logcounts")
    """
    )

    sce = ro.r("sce")
    logger.info(f"  SCE created with PCA: {ro.r('dim(reducedDim(sce, \"PCA\"))')[0]} cells x {ro.r('dim(reducedDim(sce, \"PCA\"))')[1]} PCs")
    return sce


def run_detect_anomaly(sce, cell_types: list[str]) -> dict:
    """Run scDiagnostics detectAnomaly on SCE object.

    Uses isolation forest on PCA space per cell type to find outlier cells.
    """
    import rpy2.robjects as ro

    ro.r("library(scDiagnostics)")
    ro.r.assign("sce_data", sce)

    results = {}
    for ct in cell_types:
        logger.info(f"  detectAnomaly for '{ct}'...")
        try:
            ro.r.assign("ct_name", ct)
            ro.r(
                """
                anomaly_result <- detectAnomaly(
                    reference_data = sce_data,
                    ref_cell_type_col = "cell_type",
                    cell_types = ct_name,
                    pc_subset = 1:10,
                    n_tree = 500,
                    anomaly_treshold = 0.6
                )
            """
            )

            # Extract results -- directly under cell type, not under $anomaly
            ro.r(
                """
                ct_result <- anomaly_result[[ct_name]]
                anomaly_scores <- ct_result$reference_anomaly_scores
                anomaly_labels <- ct_result$reference_anomaly
            """
            )

            scores = np.array(list(ro.r("anomaly_scores")))
            labels = list(ro.r("anomaly_labels"))

            n_anomalous = sum(1 for lb in labels if lb == "anomaly")
            n_total = len(labels)

            results[ct] = {
                "n_cells": n_total,
                "n_anomalous": n_anomalous,
                "anomaly_rate": round(n_anomalous / n_total, 4) if n_total > 0 else 0,
                "mean_anomaly_score": round(float(np.mean(scores)), 4),
                "median_anomaly_score": round(float(np.median(scores)), 4),
                "std_anomaly_score": round(float(np.std(scores)), 4),
            }
            logger.info(
                f"    {ct}: {n_anomalous}/{n_total} anomalous "
                f"({results[ct]['anomaly_rate']:.1%}), "
                f"mean score={results[ct]['mean_anomaly_score']:.3f}"
            )

        except Exception as e:
            logger.warning(f"    detectAnomaly failed for {ct}: {e}")
            results[ct] = {"error": str(e)}

    return results


def run_categorization_entropy(prob_matrix: np.ndarray) -> dict:
    """Run scDiagnostics calculateCategorizationEntropy on probability matrix."""
    import rpy2.robjects as ro

    ro.r("library(scDiagnostics)")

    # prob_matrix is cells x classes
    n_cells, n_classes = prob_matrix.shape
    logger.info(
        f"  calculateCategorizationEntropy: {n_cells} cells, {n_classes} classes"
    )

    # Convert to R matrix (classes x cells for scDiagnostics)
    prob_r = ro.r["matrix"](
        ro.FloatVector(prob_matrix.T.flatten().tolist()),
        nrow=n_classes,
        ncol=n_cells,
    )
    ro.r.assign("prob_mat", prob_r)

    try:
        # calculateCategorizationEntropy returns a plain numeric vector of entropy values
        ro.r(
            """
            entropy_vec <- calculateCategorizationEntropy(
                prob_mat,
                inverse_normal_transform = FALSE,
                plot = FALSE,
                verbose = FALSE
            )
        """
        )

        entropies = np.array(list(ro.r("entropy_vec")))
        # Max possible entropy for n_classes categories is log(n_classes)
        max_entropy = float(np.log(n_classes))

        result = {
            "n_cells": n_cells,
            "n_classes": n_classes,
            "max_possible_entropy": round(max_entropy, 4),
            "mean_entropy": round(float(np.mean(entropies)), 4),
            "median_entropy": round(float(np.median(entropies)), 4),
            "std_entropy": round(float(np.std(entropies)), 4),
            "min_entropy": round(float(np.min(entropies)), 4),
            "max_entropy_observed": round(float(np.max(entropies)), 4),
            "pct_high_confidence": round(
                float(np.mean(entropies < max_entropy * 0.3)) * 100, 1
            ),
            "pct_low_confidence": round(
                float(np.mean(entropies > max_entropy * 0.7)) * 100, 1
            ),
            "entropy_percentiles": {
                "p10": round(float(np.percentile(entropies, 10)), 4),
                "p25": round(float(np.percentile(entropies, 25)), 4),
                "p50": round(float(np.percentile(entropies, 50)), 4),
                "p75": round(float(np.percentile(entropies, 75)), 4),
                "p90": round(float(np.percentile(entropies, 90)), 4),
            },
        }
        logger.info(
            f"    Mean entropy: {result['mean_entropy']:.4f} / {max_entropy:.4f} max, "
            f"high conf: {result['pct_high_confidence']:.1f}%, "
            f"low conf: {result['pct_low_confidence']:.1f}%"
        )

    except Exception as e:
        logger.warning(f"    calculateCategorizationEntropy failed: {e}")
        result = {"error": str(e)}

    return result


def run_hotelling_test(
    query_sce, ref_sce, query_col: str, ref_col: str, cell_types: list[str]
) -> dict:
    """Run Hotelling T-squared test between query and reference datasets."""
    import rpy2.robjects as ro

    ro.r("library(scDiagnostics)")

    ro.r.assign("query_sce", query_sce)
    ro.r.assign("ref_sce", ref_sce)

    results = {}
    for ct in cell_types:
        logger.info(f"  Hotelling T^2 test for '{ct}'...")
        try:
            ro.r.assign("ct_name", ct)
            ro.r(
                f"""
                hotelling_result <- calculateHotellingPValue(
                    query_data = query_sce,
                    reference_data = ref_sce,
                    query_cell_type_col = "{query_col}",
                    ref_cell_type_col = "{ref_col}",
                    cell_types = ct_name,
                    pc_subset = 1:5,
                    n_permutation = 100
                )
            """
            )

            # calculateHotellingPValue returns a named numeric vector
            p_value = float(ro.r("hotelling_result")[0])

            results[ct] = {
                "p_value": round(p_value, 6),
                "significant_005": bool(p_value < 0.05),
                "significant_001": bool(p_value < 0.01),
            }
            logger.info(
                f"    {ct}: p={p_value:.4f} "
                f"({'SIGNIFICANT' if p_value < 0.05 else 'not significant'})"
            )

        except Exception as e:
            logger.warning(f"    Hotelling test failed for {ct}: {e}")
            results[ct] = {"error": str(e)}

    return results


def run_knn_probabilities(
    query_sce, ref_sce, query_col: str, ref_col: str, cell_types: list[str]
) -> dict:
    """Run KNN-based dataset membership probabilities."""
    import rpy2.robjects as ro

    ro.r("library(scDiagnostics)")

    ro.r.assign("query_sce", query_sce)
    ro.r.assign("ref_sce", ref_sce)

    results = {}
    for ct in cell_types:
        logger.info(f"  KNN probabilities for '{ct}'...")
        try:
            ro.r.assign("ct_name", ct)
            ro.r(
                f"""
                knn_result <- calculateNearestNeighborProbabilities(
                    query_data = query_sce,
                    reference_data = ref_sce,
                    query_cell_type_col = "{query_col}",
                    ref_cell_type_col = "{ref_col}",
                    cell_types = ct_name,
                    pc_subset = 1:5,
                    n_neighbor = 20
                )
            """
            )

            # Extract probabilities
            ro.r(
                """
                ct_probs <- knn_result[[ct_name]]
                query_probs <- ct_probs$query
            """
            )
            probs = np.array(list(ro.r("query_probs")))

            results[ct] = {
                "n_cells": len(probs),
                "mean_ref_probability": round(float(np.mean(probs)), 4),
                "median_ref_probability": round(float(np.median(probs)), 4),
                "pct_high_ref_match": round(float(np.mean(probs > 0.5)) * 100, 1),
                "pct_low_ref_match": round(float(np.mean(probs < 0.3)) * 100, 1),
            }
            logger.info(
                f"    {ct}: mean ref prob={results[ct]['mean_ref_probability']:.3f}, "
                f"high match: {results[ct]['pct_high_ref_match']:.1f}%"
            )

        except Exception as e:
            logger.warning(f"    KNN probabilities failed for {ct}: {e}")
            results[ct] = {"error": str(e)}

    return results


# ──────────────────────────────────────────────────────────────────────────────
# PYTHON-BASED EQUIVALENTS (sklearn/scipy)
# ──────────────────────────────────────────────────────────────────────────────


def python_isolation_forest(adata: ad.AnnData, cell_type_col: str) -> dict:
    """Python equivalent of detectAnomaly using sklearn IsolationForest."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest

    logger.info("Running Python IsolationForest on PCA space...")

    # PCA on expression
    X = adata.X if not issparse(adata.X) else adata.X.toarray()
    n_pcs = min(20, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_pcs)
    X_pca = pca.fit_transform(X)

    cell_types = adata.obs[cell_type_col].unique()
    results = {}

    for ct in sorted(cell_types):
        mask = adata.obs[cell_type_col] == ct
        X_ct = X_pca[mask]

        if len(X_ct) < 10:
            results[ct] = {"error": f"Too few cells ({len(X_ct)})"}
            continue

        iso = IsolationForest(n_estimators=500, contamination="auto", random_state=42)
        labels = iso.fit_predict(X_ct)
        scores = iso.score_samples(X_ct)

        n_anomalous = int(np.sum(labels == -1))
        results[ct] = {
            "n_cells": int(len(X_ct)),
            "n_anomalous": n_anomalous,
            "anomaly_rate": round(n_anomalous / len(X_ct), 4),
            "mean_anomaly_score": round(float(np.mean(scores)), 4),
            "median_anomaly_score": round(float(np.median(scores)), 4),
        }
        logger.info(
            f"  {ct}: {n_anomalous}/{len(X_ct)} anomalous "
            f"({results[ct]['anomaly_rate']:.1%})"
        )

    return results


def python_knn_entropy(adata: ad.AnnData, cell_type_col: str, k: int = 20) -> dict:
    """Shannon entropy of KNN label distribution per cell.

    For each cell, find K nearest neighbors in PCA space and compute
    Shannon entropy of their label distribution. High entropy = neighbors
    have diverse labels = less confident annotation.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    logger.info(f"Running Python KNN entropy (k={k}) on PCA space...")

    X = adata.X if not issparse(adata.X) else adata.X.toarray()
    n_pcs = min(20, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_pcs)
    X_pca = pca.fit_transform(X)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X_pca)
    _, indices = nn.kneighbors(X_pca)

    labels = adata.obs[cell_type_col].values
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    max_entropy = np.log(n_labels)

    entropies = np.zeros(len(adata))
    for i in range(len(adata)):
        neighbor_labels = labels[indices[i, 1:]]  # skip self
        counts = Counter(neighbor_labels)
        total = sum(counts.values())
        probs = np.array([counts.get(lb, 0) / total for lb in unique_labels])
        probs = probs[probs > 0]
        entropies[i] = -np.sum(probs * np.log(probs))

    # Per cell-type summary
    per_type = {}
    for ct in sorted(np.unique(labels)):
        mask = labels == ct
        ct_entropies = entropies[mask]
        per_type[ct] = {
            "n_cells": int(np.sum(mask)),
            "mean_entropy": round(float(np.mean(ct_entropies)), 4),
            "median_entropy": round(float(np.median(ct_entropies)), 4),
            "normalized_entropy": round(
                float(np.mean(ct_entropies)) / max_entropy, 4
            )
            if max_entropy > 0
            else 0,
            "pct_pure_neighborhood": round(
                float(np.mean(ct_entropies < 0.1)) * 100, 1
            ),
            "pct_mixed_neighborhood": round(
                float(np.mean(ct_entropies > max_entropy * 0.5)) * 100, 1
            ),
        }

    result = {
        "global": {
            "k": k,
            "n_cells": len(adata),
            "n_labels": n_labels,
            "max_possible_entropy": round(max_entropy, 4),
            "mean_entropy": round(float(np.mean(entropies)), 4),
            "median_entropy": round(float(np.median(entropies)), 4),
            "pct_pure_neighborhood": round(
                float(np.mean(entropies < 0.1)) * 100, 1
            ),
            "pct_mixed_neighborhood": round(
                float(np.mean(entropies > max_entropy * 0.5)) * 100, 1
            ),
        },
        "per_cell_type": per_type,
    }

    logger.info(
        f"  Global: mean entropy={result['global']['mean_entropy']:.4f}/{max_entropy:.4f}, "
        f"pure: {result['global']['pct_pure_neighborhood']:.1f}%, "
        f"mixed: {result['global']['pct_mixed_neighborhood']:.1f}%"
    )

    return result


def python_hotelling_t2(
    adata: ad.AnnData, cell_type_col: str, n_pcs: int = 10
) -> dict:
    """Hotelling T-squared test between each cell type and rest.

    Tests whether each cell type's PCA distribution differs significantly
    from the overall distribution -- a proxy for cluster separability.
    """
    from sklearn.decomposition import PCA

    logger.info("Running Python Hotelling T^2 test...")

    X = adata.X if not issparse(adata.X) else adata.X.toarray()
    n_pcs_actual = min(n_pcs, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_pcs_actual)
    X_pca = pca.fit_transform(X)

    labels = adata.obs[cell_type_col].values
    results = {}

    for ct in sorted(np.unique(labels)):
        mask = labels == ct
        X_in = X_pca[mask]
        X_out = X_pca[~mask]

        n1 = len(X_in)
        n2 = len(X_out)
        if n1 < n_pcs_actual + 2 or n2 < n_pcs_actual + 2:
            results[ct] = {"error": f"Too few cells (n1={n1}, n2={n2})"}
            continue

        # Mean difference
        mean_diff = np.mean(X_in, axis=0) - np.mean(X_out, axis=0)

        # Pooled covariance
        cov_in = np.cov(X_in.T)
        cov_out = np.cov(X_out.T)
        S_pooled = ((n1 - 1) * cov_in + (n2 - 1) * cov_out) / (n1 + n2 - 2)

        # T-squared statistic
        try:
            S_inv = np.linalg.inv(S_pooled)
            t2 = (n1 * n2) / (n1 + n2) * mean_diff @ S_inv @ mean_diff

            # Convert to F statistic
            p = n_pcs_actual
            n = n1 + n2
            f_stat = t2 * (n - p - 1) / (p * (n - 2))

            from scipy import stats

            p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)

            results[ct] = {
                "n_cells": int(n1),
                "t2_statistic": round(float(t2), 2),
                "f_statistic": round(float(f_stat), 2),
                "p_value": round(float(p_value), 6),
                "significant_005": bool(p_value < 0.05),
                "significant_001": bool(p_value < 0.01),
            }
            logger.info(
                f"  {ct}: T2={t2:.1f}, F={f_stat:.1f}, "
                f"p={p_value:.2e} ({'***' if p_value < 0.001 else 'ns'})"
            )
        except np.linalg.LinAlgError:
            results[ct] = {"error": "Singular covariance matrix"}

    return results


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────


def main():
    t_start = time.time()
    all_results = {
        "metadata": {
            "dataset": "xenium_breast_rep1",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "scDiagnostics annotation quality assessment",
        }
    }

    # 1. Load and preprocess data
    logger.info("=" * 70)
    logger.info("STEP 1: Loading Xenium breast rep1 data")
    logger.info("=" * 70)
    adata = load_xenium_adata("rep1")
    adata_pp = preprocess_adata(adata)

    logger.info(f"Data shape: {adata_pp.shape}")
    logger.info(f"Cell types (gt_coarse): {dict(Counter(adata_pp.obs['gt_coarse']))}")

    # Subsample for R-based methods (158K cells is too many for some operations)
    MAX_CELLS_R = 20_000
    if len(adata_pp) > MAX_CELLS_R:
        logger.info(
            f"Subsampling to {MAX_CELLS_R} cells for R-based scDiagnostics (stratified)"
        )
        np.random.seed(42)
        # Stratified subsample
        indices = []
        for ct in adata_pp.obs["gt_coarse"].unique():
            ct_idx = np.where(adata_pp.obs["gt_coarse"] == ct)[0]
            n_sample = min(
                len(ct_idx),
                max(100, int(MAX_CELLS_R * len(ct_idx) / len(adata_pp))),
            )
            indices.extend(np.random.choice(ct_idx, n_sample, replace=False))
        adata_sub = adata_pp[sorted(indices)].copy()
        logger.info(
            f"Subsampled: {len(adata_sub)} cells, "
            f"types: {dict(Counter(adata_sub.obs['gt_coarse']))}"
        )
    else:
        adata_sub = adata_pp

    all_results["metadata"]["n_cells_total"] = len(adata_pp)
    all_results["metadata"]["n_cells_subsample"] = len(adata_sub)

    # 2. Run CellTypist on SUBSAMPLE only (avoids segfault on 158K cells)
    logger.info("=" * 70)
    logger.info("STEP 2: Running CellTypist annotation on subsample")
    logger.info("=" * 70)
    adata_sub = annotate_with_celltypist(adata_sub)

    # Compute PCA for downstream use
    logger.info("Computing PCA...")
    sc.pp.highly_variable_genes(adata_sub, n_top_genes=2000, flavor="seurat_v3", layer="raw")
    sc.pp.pca(adata_sub, n_comps=30)

    # 3. R-based scDiagnostics
    logger.info("=" * 70)
    logger.info("STEP 3: R-based scDiagnostics")
    logger.info("=" * 70)

    cell_types = sorted(adata_sub.obs["gt_coarse"].unique().tolist())
    logger.info(f"Cell types for analysis: {cell_types}")

    # 3a. Convert to SCE
    logger.info("Converting to SingleCellExperiment...")
    sce = adata_to_sce(adata_sub, cell_type_col="gt_coarse")
    logger.info("SCE created successfully")

    # 3b. detectAnomaly (isolation forest per cell type in PCA space)
    logger.info("-" * 50)
    logger.info("3a. detectAnomaly (isolation forest per cell type)")
    logger.info("-" * 50)
    anomaly_results = run_detect_anomaly(sce, cell_types)
    all_results["scdiagnostics_detectAnomaly"] = anomaly_results

    # 3c. calculateCategorizationEntropy (on CellTypist probability matrix)
    logger.info("-" * 50)
    logger.info("3b. calculateCategorizationEntropy (CellTypist probabilities)")
    logger.info("-" * 50)
    if "celltypist_probs" in adata_sub.obsm:
        entropy_results = run_categorization_entropy(adata_sub.obsm["celltypist_probs"])
        all_results["scdiagnostics_categorizationEntropy"] = entropy_results
    else:
        logger.warning("No CellTypist probability matrix available")
        all_results["scdiagnostics_categorizationEntropy"] = {
            "error": "No probability matrix"
        }

    # 3d. Hotelling test (split data: use first half as "reference", second as "query")
    logger.info("-" * 50)
    logger.info("3c. Hotelling T^2 test (random split: reference vs query)")
    logger.info("-" * 50)
    np.random.seed(42)
    n = len(adata_sub)
    perm = np.random.permutation(n)
    half = n // 2
    adata_ref = adata_sub[perm[:half]].copy()
    adata_query = adata_sub[perm[half:]].copy()
    logger.info(f"Split: reference={len(adata_ref)}, query={len(adata_query)}")

    sce_ref = adata_to_sce(adata_ref, cell_type_col="gt_coarse")
    sce_query = adata_to_sce(adata_query, cell_type_col="gt_coarse")

    # Find shared cell types
    ref_types = set(adata_ref.obs["gt_coarse"].unique())
    query_types = set(adata_query.obs["gt_coarse"].unique())
    shared_types = sorted(ref_types & query_types)
    logger.info(f"Shared cell types: {shared_types}")

    hotelling_results = run_hotelling_test(
        sce_query, sce_ref, "cell_type", "cell_type", shared_types
    )
    all_results["scdiagnostics_hotelling"] = hotelling_results

    # 3e. KNN probabilities
    logger.info("-" * 50)
    logger.info("3d. KNN nearest neighbor probabilities")
    logger.info("-" * 50)
    knn_results = run_knn_probabilities(
        sce_query, sce_ref, "cell_type", "cell_type", shared_types
    )
    all_results["scdiagnostics_knn_probabilities"] = knn_results

    # Clean up R objects
    gc.collect()

    # 4. Python-based equivalents
    logger.info("=" * 70)
    logger.info("STEP 4: Python-based equivalents")
    logger.info("=" * 70)

    # Use subsample for consistency
    logger.info("-" * 50)
    logger.info("4a. Python IsolationForest")
    logger.info("-" * 50)
    py_anomaly = python_isolation_forest(adata_sub, "gt_coarse")
    all_results["python_isolation_forest"] = py_anomaly

    logger.info("-" * 50)
    logger.info("4b. Python KNN entropy")
    logger.info("-" * 50)
    py_knn_entropy = python_knn_entropy(adata_sub, "gt_coarse", k=20)
    all_results["python_knn_entropy"] = py_knn_entropy

    logger.info("-" * 50)
    logger.info("4c. Python Hotelling T^2")
    logger.info("-" * 50)
    py_hotelling = python_hotelling_t2(adata_sub, "gt_coarse")
    all_results["python_hotelling_t2"] = py_hotelling

    # 5. Summary comparison
    logger.info("=" * 70)
    logger.info("STEP 5: Summary")
    logger.info("=" * 70)

    summary = {
        "anomaly_detection": {},
        "entropy_assessment": {},
        "cluster_separability": {},
        "knn_neighborhood": {},
    }

    # Compare anomaly rates
    for ct in cell_types:
        r_rate = anomaly_results.get(ct, {}).get("anomaly_rate", None)
        py_rate = py_anomaly.get(ct, {}).get("anomaly_rate", None)
        summary["anomaly_detection"][ct] = {
            "r_anomaly_rate": r_rate,
            "python_anomaly_rate": py_rate,
        }

    # Entropy summary
    if "error" not in all_results.get("scdiagnostics_categorizationEntropy", {}):
        ent = all_results["scdiagnostics_categorizationEntropy"]
        summary["entropy_assessment"] = {
            "mean_entropy": ent["mean_entropy"],
            "max_possible": ent["max_possible_entropy"],
            "normalized_mean": round(
                ent["mean_entropy"] / ent["max_possible_entropy"], 4
            )
            if ent["max_possible_entropy"] > 0
            else 0,
            "pct_high_confidence": ent["pct_high_confidence"],
            "pct_low_confidence": ent["pct_low_confidence"],
            "interpretation": (
                "HIGH confidence"
                if ent["pct_high_confidence"] > 70
                else (
                    "MODERATE confidence"
                    if ent["pct_high_confidence"] > 40
                    else "LOW confidence"
                )
            ),
        }

    # Hotelling summary
    for ct in shared_types:
        r_p = hotelling_results.get(ct, {}).get("p_value", None)
        py_p = py_hotelling.get(ct, {}).get("p_value", None)
        summary["cluster_separability"][ct] = {
            "r_hotelling_p": r_p,
            "python_hotelling_p": py_p,
            "well_separated": (py_p is not None and py_p < 0.001),
        }

    # KNN summary
    for ct in shared_types:
        knn_r = knn_results.get(ct, {})
        py_knn = py_knn_entropy.get("per_cell_type", {}).get(ct, {})
        summary["knn_neighborhood"][ct] = {
            "r_mean_ref_prob": knn_r.get("mean_ref_probability"),
            "python_mean_entropy": py_knn.get("mean_entropy"),
            "python_pct_pure": py_knn.get("pct_pure_neighborhood"),
        }

    all_results["summary"] = summary

    elapsed = time.time() - t_start
    all_results["metadata"]["elapsed_seconds"] = round(elapsed, 1)

    # Print summary table
    logger.info("")
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(
        f"{'Cell Type':<16} {'Anomaly(R)':<12} {'Anomaly(Py)':<12} "
        f"{'KNN Entropy':<12} {'Hotelling p':<12}"
    )
    logger.info("-" * 64)
    for ct in cell_types:
        r_anom = anomaly_results.get(ct, {}).get("anomaly_rate", "N/A")
        py_anom = py_anomaly.get(ct, {}).get("anomaly_rate", "N/A")
        knn_ent = (
            py_knn_entropy.get("per_cell_type", {}).get(ct, {}).get("mean_entropy", "N/A")
        )
        hot_p = py_hotelling.get(ct, {}).get("p_value", "N/A")
        logger.info(
            f"{ct:<16} {r_anom!s:<12} {py_anom!s:<12} {knn_ent!s:<12} {hot_p!s:<12}"
        )

    # Save
    out_file = OUTPUT_DIR / "scdiagnostics_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_file}")
    logger.info(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
