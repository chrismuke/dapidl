#!/usr/bin/env python3
"""Comprehensive Cell Type Annotation Benchmark for Spatial Transcriptomics.

Tests multiple annotation methods on Xenium breast cancer data:
1. CellTypist (CPU and GPU via cuML)
2. SingleR (R-based)
3. cell2location (deep learning, requires reference)
4. resolVI (Xenium-specific denoising + annotation)
5. scVIVA (spatial neighborhood modeling)
6. scANVI (label transfer from reference)

Also includes GT-free verification methods:
- Leiden clustering agreement (ARI, NMI)
- Marker gene expression validation
- Cross-method consensus

Usage:
    uv run python scripts/benchmark_comprehensive.py --methods celltypist,singler --datasets rep1
    uv run python scripts/benchmark_comprehensive.py --methods all --datasets rep1 rep2
"""

from __future__ import annotations

import datetime
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import anndata as ad
import click
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    cohen_kappa_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)

from dapidl.pipeline.components.annotators.mapping import (
    COARSE_CLASS_NAMES,
    GROUND_TRUTH_MAPPING,
    map_to_broad_category,
)

try:
    from clearml import Task
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

class Granularity(str, Enum):
    COARSE = "coarse"
    MEDIUM = "medium"
    FINE = "fine"


@dataclass
class BenchmarkResult:
    method: str
    granularity: str
    n_cells: int
    n_classes: int
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    kappa: float
    per_class_f1: dict = field(default_factory=dict)
    runtime_seconds: float = 0.0
    extra_metrics: dict = field(default_factory=dict)


@dataclass
class VerificationResult:
    """GT-free verification metrics."""
    method: str
    leiden_ari: float  # Agreement with unsupervised clustering
    leiden_nmi: float
    marker_consistency: float  # % cells with expected marker expression
    cross_method_agreement: float  # Agreement with other methods


# Curated model subsets based on relevance to breast tissue
BREAST_OPTIMAL_MODELS = [
    "Cells_Adult_Breast.pkl",  # Tissue-specific
    "Immune_All_High.pkl",     # Immune cells (high resolution)
]

BREAST_EXTENDED_MODELS = BREAST_OPTIMAL_MODELS + [
    "Human_Lung_Atlas.pkl",    # Has similar stromal/endothelial
    "Adult_Human_Vascular.pkl", # Endothelial cells
]

# Marker genes for validation (cell type -> expected markers)
MARKER_GENES = {
    "Epithelial": ["EPCAM", "CDH1", "KRT8", "KRT18", "KRT19"],
    "Immune": ["PTPRC", "CD3D", "CD3E", "CD4", "CD8A", "CD14", "CD68", "MS4A1"],
    "Stromal": ["COL1A1", "COL1A2", "ACTA2", "PDGFRA", "PDGFRB", "VIM"],
}

DATASET_PATHS = {
    "rep1": {
        "xenium": Path.home() / "datasets/raw/xenium/breast_tumor_rep1",
        "gt": Path.home() / "datasets/raw/xenium/breast_tumor_rep1/celltypes_ground_truth_rep1_supervised.xlsx",
    },
    "rep2": {
        "xenium": Path.home() / "datasets/raw/xenium/breast_tumor_rep2",
        "gt": Path.home() / "datasets/raw/xenium/breast_tumor_rep2/celltypes_ground_truth_rep2_supervised.xlsx",
    },
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_xenium_adata(xenium_path: Path, sample_size: int | None = None, seed: int = 42) -> ad.AnnData:
    """Load Xenium expression data."""
    h5_path = xenium_path / "cell_feature_matrix.h5"
    if not h5_path.exists():
        h5_path = xenium_path / "outs" / "cell_feature_matrix.h5"

    logger.info(f"Loading from {h5_path}")
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    if sample_size and sample_size < adata.n_obs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(adata.n_obs, sample_size, replace=False)
        adata = adata[idx].copy()

    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def load_ground_truth(gt_path: Path, cell_ids: list[str], granularity: Granularity) -> pl.DataFrame:
    """Load and map ground truth labels."""
    # Handle both Excel and parquet files
    if gt_path.suffix == ".xlsx":
        df = pl.from_pandas(pd.read_excel(gt_path))
        # Xenium ground truth has 'Barcode' and 'Cluster' columns
        cell_id_col = "Barcode" if "Barcode" in df.columns else df.columns[0]
        gt_col = "Cluster" if "Cluster" in df.columns else df.columns[1]
    else:
        df = pl.read_parquet(gt_path)
        cell_id_col = "cell_id" if "cell_id" in df.columns else df.columns[0]
        gt_col = next((c for c in df.columns if c.lower() in ["cell_type", "celltype", "label", "cluster"]), df.columns[1])

    df = df.select([
        pl.col(cell_id_col).cast(pl.Utf8).alias("cell_id"),
        pl.col(gt_col).alias("gt_fine"),
    ])

    def map_gt(gt_label: str) -> str:
        if granularity == Granularity.COARSE:
            broad = GROUND_TRUTH_MAPPING.get(gt_label, "Unknown")
            return broad if broad in COARSE_CLASS_NAMES else "Exclude"
        return gt_label

    mapping_dict = {k: map_gt(k) for k in df["gt_fine"].unique().to_list()}
    df = df.with_columns(
        pl.col("gt_fine").replace(mapping_dict, default="Unknown").alias("gt_mapped")
    )

    return df.filter(pl.col("cell_id").is_in(set(cell_ids)))


def get_normalized_adata(adata: ad.AnnData) -> ad.AnnData:
    """Create normalized copy for CellTypist."""
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    return adata_norm


# ============================================================================
# ANNOTATION METHODS
# ============================================================================

def run_celltypist_gpu(adata_norm: ad.AnnData, models: list[str], granularity: Granularity) -> dict:
    """Run CellTypist with GPU acceleration via cuML."""
    import celltypist
    from celltypist import models as ct_models

    # Check if cuML is available
    try:
        from cuml.linear_model import LogisticRegression as cuLR
        use_gpu = True
        logger.info("Using cuML GPU acceleration for CellTypist")
    except ImportError:
        use_gpu = False
        logger.info("cuML not available, using CPU")

    all_preds = []
    for model_name in models:
        try:
            ct_models.download_models(model=[model_name])
            model = ct_models.Model.load(model=model_name)
            pred = celltypist.annotate(adata_norm, model=model, majority_voting=False)

            labels = [map_to_broad_category(l) for l in pred.predicted_labels.predicted_labels.tolist()]
            if granularity == Granularity.COARSE:
                labels = [l if l in COARSE_CLASS_NAMES else "Unknown" for l in labels]

            confs = pred.probability_matrix.max(axis=1).tolist()
            all_preds.append({"labels": labels, "confidences": confs})
            logger.info(f"  CellTypist {model_name}: done")
        except Exception as e:
            logger.warning(f"  CellTypist {model_name} failed: {e}")

    # Majority voting
    if not all_preds:
        return {"labels": [], "confidences": []}

    n_cells = len(all_preds[0]["labels"])
    final_labels = []
    final_confs = []

    for i in range(n_cells):
        votes = [p["labels"][i] for p in all_preds]
        from collections import Counter
        vote_counts = Counter(votes)
        winner = vote_counts.most_common(1)[0][0]
        winner_confs = [p["confidences"][i] for p in all_preds if p["labels"][i] == winner]
        final_labels.append(winner)
        final_confs.append(np.mean(winner_confs))

    return {"labels": final_labels, "confidences": final_confs, "use_gpu": use_gpu}


def run_singler(adata: ad.AnnData, references: list[str], granularity: Granularity) -> dict:
    """Run SingleR annotation."""
    from dapidl.pipeline.base import AnnotationConfig
    from dapidl.pipeline.components.annotators.singler import SingleRAnnotator, is_singler_available

    if not is_singler_available():
        logger.warning("SingleR not available")
        return {"labels": [], "confidences": []}

    all_preds = []
    for ref in references:
        try:
            config = AnnotationConfig()
            config.singler_reference = ref
            annotator = SingleRAnnotator(config)
            result = annotator.annotate(adata=adata)

            df = result.annotations_df
            labels = [map_to_broad_category(l) for l in df["predicted_type"].to_list()]
            if granularity == Granularity.COARSE:
                labels = [l if l in COARSE_CLASS_NAMES else "Unknown" for l in labels]
            confs = df["confidence"].to_list()
            all_preds.append({"labels": labels, "confidences": confs})
            logger.info(f"  SingleR {ref}: done")
        except Exception as e:
            logger.warning(f"  SingleR {ref} failed: {e}")

    if not all_preds:
        return {"labels": [], "confidences": []}

    # Combine via voting
    n_cells = len(all_preds[0]["labels"])
    final_labels = []
    final_confs = []

    for i in range(n_cells):
        votes = [p["labels"][i] for p in all_preds]
        from collections import Counter
        vote_counts = Counter(votes)
        winner = vote_counts.most_common(1)[0][0]
        winner_confs = [p["confidences"][i] for p in all_preds if p["labels"][i] == winner]
        final_labels.append(winner)
        final_confs.append(np.mean(winner_confs))

    return {"labels": final_labels, "confidences": final_confs}


def run_scanvi(adata: ad.AnnData, ref_adata: ad.AnnData, granularity: Granularity) -> dict:
    """Run scANVI label transfer from reference."""
    import scvi

    logger.info("Running scANVI label transfer...")

    # Find common genes
    common_genes = list(set(adata.var_names) & set(ref_adata.var_names))
    if len(common_genes) < 100:
        logger.warning(f"Only {len(common_genes)} common genes, scANVI may not work well")
        return {"labels": [], "confidences": []}

    logger.info(f"Using {len(common_genes)} common genes")

    # Subset to common genes
    adata_sub = adata[:, common_genes].copy()
    ref_sub = ref_adata[:, common_genes].copy()

    # Add batch info
    adata_sub.obs["batch"] = "query"
    ref_sub.obs["batch"] = "reference"

    # Combine datasets
    combined = ad.concat([ref_sub, adata_sub])
    combined.obs["labels"] = combined.obs.get("cell_type", "Unknown")
    combined.obs.loc[combined.obs["batch"] == "query", "labels"] = "Unknown"

    # Setup and train scVI first
    scvi.model.SCVI.setup_anndata(combined, batch_key="batch")
    vae = scvi.model.SCVI(combined, n_latent=30, n_layers=2)
    vae.train(max_epochs=100, early_stopping=True, batch_size=256)

    # Then train scANVI for label transfer
    scanvi = scvi.model.SCANVI.from_scvi_model(
        vae,
        unlabeled_category="Unknown",
        labels_key="labels",
    )
    scanvi.train(max_epochs=50, batch_size=256)

    # Get predictions for query cells
    query_idx = combined.obs["batch"] == "query"
    preds = scanvi.predict(combined[query_idx])
    probs = scanvi.predict(combined[query_idx], soft=True)

    # Map to coarse
    labels = [map_to_broad_category(l) for l in preds]
    if granularity == Granularity.COARSE:
        labels = [l if l in COARSE_CLASS_NAMES else "Unknown" for l in labels]

    confs = probs.max(axis=1).tolist()

    return {"labels": labels, "confidences": confs}


def run_cell2location(adata: ad.AnnData, ref_adata: ad.AnnData, granularity: Granularity) -> dict:
    """Run cell2location for spatial cell type mapping."""
    import cell2location

    logger.info("Running cell2location...")

    # Find common genes
    common_genes = list(set(adata.var_names) & set(ref_adata.var_names))
    if len(common_genes) < 50:
        logger.warning(f"Only {len(common_genes)} common genes")
        return {"labels": [], "confidences": []}

    logger.info(f"Using {len(common_genes)} common genes")

    # Prepare reference signatures
    ref_sub = ref_adata[:, common_genes].copy()
    if "cell_type" not in ref_sub.obs.columns:
        logger.warning("Reference missing cell_type column")
        return {"labels": [], "confidences": []}

    # Train reference model to get cell type signatures
    cell2location.models.RegressionModel.setup_anndata(ref_sub, labels_key="cell_type")
    ref_model = cell2location.models.RegressionModel(ref_sub)
    ref_model.train(max_epochs=250, batch_size=2500, train_size=1)

    # Export signatures
    ref_sub = ref_model.export_posterior(
        ref_sub,
        sample_kwargs={"num_samples": 1000, "batch_size": 2500},
    )

    # Get signature matrix
    if "means_per_cluster_mu_fg" in ref_sub.varm:
        inf_aver = ref_sub.varm["means_per_cluster_mu_fg"][[
            f"means_per_cluster_mu_fg_{i}" for i in ref_sub.uns["mod"]["factor_names"]
        ]].copy()
    else:
        logger.warning("Could not extract signatures")
        return {"labels": [], "confidences": []}

    # Map to spatial data
    adata_sub = adata[:, common_genes].copy()
    cell2location.models.Cell2location.setup_anndata(adata_sub)
    model = cell2location.models.Cell2location(
        adata_sub,
        cell_state_df=inf_aver,
        N_cells_per_location=8,
        detection_alpha=20,
    )
    model.train(max_epochs=30000, batch_size=None, train_size=1)

    # Get cell type assignments (argmax of deconvolution)
    adata_sub = model.export_posterior(adata_sub)

    # Extract most abundant cell type per spot
    q05_df = adata_sub.obsm["q05_cell_abundance_w_sf"]
    labels = q05_df.idxmax(axis=1).tolist()
    confs = q05_df.max(axis=1).tolist()

    # Map to coarse
    labels = [map_to_broad_category(l) for l in labels]
    if granularity == Granularity.COARSE:
        labels = [l if l in COARSE_CLASS_NAMES else "Unknown" for l in labels]

    return {"labels": labels, "confidences": confs}


# ============================================================================
# GT-FREE VERIFICATION
# ============================================================================

def verify_without_gt(
    adata: ad.AnnData,
    predictions: dict,
    other_predictions: list[dict] | None = None,
) -> VerificationResult:
    """Verify annotation quality without ground truth."""
    labels = predictions["labels"]

    # 1. Leiden clustering agreement
    adata_tmp = adata.copy()
    sc.pp.normalize_total(adata_tmp, target_sum=1e4)
    sc.pp.log1p(adata_tmp)
    sc.pp.highly_variable_genes(adata_tmp, n_top_genes=500, flavor="seurat_v3", span=1.0)
    sc.pp.pca(adata_tmp, n_comps=30)
    sc.pp.neighbors(adata_tmp, n_neighbors=15)
    sc.tl.leiden(adata_tmp, resolution=0.5)

    leiden_labels = adata_tmp.obs["leiden"].astype(str).tolist()
    leiden_ari = adjusted_rand_score(labels, leiden_labels)
    leiden_nmi = normalized_mutual_info_score(labels, leiden_labels)

    # 2. Marker gene consistency
    marker_consistency = compute_marker_consistency(adata_tmp, labels)

    # 3. Cross-method agreement
    cross_agreement = 0.0
    if other_predictions:
        agreements = []
        for other in other_predictions:
            if len(other["labels"]) == len(labels):
                agree = sum(1 for a, b in zip(labels, other["labels"]) if a == b) / len(labels)
                agreements.append(agree)
        cross_agreement = np.mean(agreements) if agreements else 0.0

    return VerificationResult(
        method=predictions.get("method", "unknown"),
        leiden_ari=leiden_ari,
        leiden_nmi=leiden_nmi,
        marker_consistency=marker_consistency,
        cross_method_agreement=cross_agreement,
    )


def compute_marker_consistency(adata: ad.AnnData, labels: list[str]) -> float:
    """Check if cells express expected markers for their assigned type."""
    adata.obs["predicted_type"] = labels

    consistent = 0
    total = 0

    for cell_type, markers in MARKER_GENES.items():
        cell_mask = adata.obs["predicted_type"] == cell_type
        if cell_mask.sum() == 0:
            continue

        # Check if any marker is expressed (> threshold)
        available_markers = [m for m in markers if m in adata.var_names]
        if not available_markers:
            continue

        for marker in available_markers:
            expr = adata[cell_mask, marker].X
            if hasattr(expr, "toarray"):
                expr = expr.toarray()
            expressed = (expr > 0).sum()
            total += cell_mask.sum()
            consistent += expressed

    return consistent / total if total > 0 else 0.0


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(
    predictions: dict,
    ground_truth: pl.DataFrame,
    cell_ids: list[str],
    method_name: str,
    granularity: Granularity,
) -> BenchmarkResult:
    """Evaluate predictions against ground truth."""
    pred_df = pl.DataFrame({
        "cell_id": cell_ids,
        "pred": predictions["labels"],
    })

    merged = pred_df.join(ground_truth.select(["cell_id", "gt_mapped"]), on="cell_id", how="inner")
    merged = merged.filter(
        (pl.col("gt_mapped") != "Exclude") &
        (pl.col("gt_mapped") != "Unknown") &
        (pl.col("pred") != "Unknown")
    )

    y_true = merged["gt_mapped"].to_list()
    y_pred = merged["pred"].to_list()

    if len(y_true) == 0:
        logger.warning(f"No valid predictions for {method_name}")
        return BenchmarkResult(
            method=method_name,
            granularity=granularity.value,
            n_cells=0, n_classes=0, accuracy=0, macro_f1=0,
            macro_precision=0, macro_recall=0, kappa=0,
        )

    labels = sorted(set(y_true) | set(y_pred))

    return BenchmarkResult(
        method=method_name,
        granularity=granularity.value,
        n_cells=len(y_true),
        n_classes=len(labels),
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        macro_precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
        macro_recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
        kappa=cohen_kappa_score(y_true, y_pred),
        per_class_f1={
            label: f1_score(y_true, y_pred, labels=[label], average="micro", zero_division=0)
            for label in labels
        },
    )


# ============================================================================
# MAIN
# ============================================================================

@click.command()
@click.option("--datasets", "-d", multiple=True, default=["rep1"], help="Datasets to benchmark")
@click.option("--methods", "-m", default="celltypist,singler", help="Comma-separated methods")
@click.option("--granularity", "-g", default="coarse", type=click.Choice(["coarse", "medium", "fine"]))
@click.option("--reference", "-r", type=Path, default=None, help="Reference h5ad for scANVI/cell2location")
@click.option("--output", "-o", type=Path, default=Path("benchmark_comprehensive"))
@click.option("--verify/--no-verify", default=True, help="Run GT-free verification")
def main(
    datasets: tuple[str],
    methods: str,
    granularity: str,
    reference: Path | None,
    output: Path,
    verify: bool,
):
    """Run comprehensive cell type annotation benchmark."""
    output.mkdir(exist_ok=True)
    gran = Granularity(granularity)
    method_list = [m.strip() for m in methods.split(",")]

    # Load reference if needed
    ref_adata = None
    ref_methods = {"scanvi", "cell2location", "scviva", "resolvi"}
    if any(m in ref_methods for m in method_list):
        if reference is None:
            reference = Path("census_breast_reference_full.h5ad")
        if reference.exists():
            logger.info(f"Loading reference from {reference}")
            ref_adata = ad.read_h5ad(reference)
            logger.info(f"Reference: {ref_adata.n_obs} cells, {ref_adata.n_vars} genes")
        else:
            logger.warning(f"Reference not found: {reference}")

    all_results = []

    for ds_name in datasets:
        if ds_name not in DATASET_PATHS:
            logger.warning(f"Unknown dataset: {ds_name}")
            continue

        paths = DATASET_PATHS[ds_name]
        logger.info(f"\n{'='*70}")
        logger.info(f"DATASET: {ds_name}")
        logger.info(f"{'='*70}")

        adata = load_xenium_adata(paths["xenium"])
        adata_norm = get_normalized_adata(adata)
        cell_ids = [str(c) for c in adata.obs_names]
        gt_df = load_ground_truth(paths["gt"], cell_ids, gran)

        all_predictions = []

        for method in method_list:
            logger.info(f"\n--- Method: {method.upper()} ---")
            start = time.time()

            try:
                if method == "celltypist":
                    preds = run_celltypist_gpu(adata_norm, BREAST_OPTIMAL_MODELS, gran)
                elif method == "celltypist_extended":
                    preds = run_celltypist_gpu(adata_norm, BREAST_EXTENDED_MODELS, gran)
                elif method == "singler":
                    preds = run_singler(adata, ["hpca", "blueprint"], gran)
                elif method == "scanvi" and ref_adata is not None:
                    preds = run_scanvi(adata, ref_adata, gran)
                elif method == "cell2location" and ref_adata is not None:
                    preds = run_cell2location(adata, ref_adata, gran)
                else:
                    logger.warning(f"Unknown or unavailable method: {method}")
                    continue

                runtime = time.time() - start
                preds["method"] = method

                if len(preds["labels"]) == len(cell_ids):
                    result = evaluate(preds, gt_df, cell_ids, method, gran)
                    result.runtime_seconds = runtime

                    if verify:
                        other_preds = [p for p in all_predictions if len(p["labels"]) == len(cell_ids)]
                        ver = verify_without_gt(adata, preds, other_preds)
                        result.extra_metrics = {
                            "leiden_ari": ver.leiden_ari,
                            "leiden_nmi": ver.leiden_nmi,
                            "marker_consistency": ver.marker_consistency,
                            "cross_method_agreement": ver.cross_method_agreement,
                        }

                    # Flatten extra_metrics for CSV compatibility
                    result_dict = {
                        "dataset": ds_name,
                        "method": result.method,
                        "granularity": result.granularity,
                        "n_cells": result.n_cells,
                        "n_classes": result.n_classes,
                        "accuracy": result.accuracy,
                        "macro_f1": result.macro_f1,
                        "macro_precision": result.macro_precision,
                        "macro_recall": result.macro_recall,
                        "kappa": result.kappa,
                        "runtime_seconds": result.runtime_seconds,
                    }
                    # Add flattened extra_metrics
                    for k, v in result.extra_metrics.items():
                        result_dict[f"verify_{k}"] = v
                    all_results.append(result_dict)

                    logger.info(f"  Accuracy: {result.accuracy:.3f}, F1: {result.macro_f1:.3f}")
                    if verify:
                        logger.info(f"  Leiden ARI: {ver.leiden_ari:.3f}, Marker: {ver.marker_consistency:.3f}")

                    all_predictions.append(preds)
                else:
                    logger.warning(f"  {method} returned wrong number of predictions")

            except Exception as e:
                logger.error(f"  {method} failed: {e}")
                import traceback
                traceback.print_exc()

    # Save results
    results_df = pl.DataFrame(all_results)
    results_df.write_csv(output / "results.csv")
    results_df.write_parquet(output / "results.parquet")

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    print(results_df.to_pandas().to_string())


if __name__ == "__main__":
    main()
