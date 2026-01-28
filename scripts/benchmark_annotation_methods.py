#!/usr/bin/env python3
"""
Comprehensive Benchmark of Cell Type Annotation Methods with ClearML Tracking.

Tests ALL combinations of annotation methods:
- SingleR single models (HPCA, Blueprint, Monaco, Novershtern)
- SingleR all models ensemble
- CellTypist single models (Breast, Immune_High, Immune_Low)
- CellTypist tissue-specific ensemble
- CellTypist universal ensemble (10 models)
- CellTypist all models
- PopV-style combined (CellTypist + SingleR)

Features:
- Ground truth comparison
- Confusion matrices
- UMAP visualization (GT vs predicted)
- ClearML experiment tracking
- Xenium Explorer CSV export

Usage:
    # Run all methods on both datasets
    uv run python scripts/benchmark_annotation_methods.py

    # Test specific methods
    uv run python scripts/benchmark_annotation_methods.py --methods singler_hpca celltypist_breast

    # Sample for faster testing
    uv run python scripts/benchmark_annotation_methods.py --sample-size 5000

    # Skip ClearML (local only)
    uv run python scripts/benchmark_annotation_methods.py --no-clearml
"""

import sys
import json
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

# Try to import ClearML
try:
    from clearml import Task
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    logger.warning("ClearML not available - results won't be logged")


class AnnotationMethod(str, Enum):
    """Available annotation methods."""
    # SingleR single references
    SINGLER_HPCA = "singler_hpca"
    SINGLER_BLUEPRINT = "singler_blueprint"
    SINGLER_MONACO = "singler_monaco"
    SINGLER_NOVERSHTERN = "singler_novershtern"
    SINGLER_ALL = "singler_all"  # All 4 references combined

    # CellTypist single
    CELLTYPIST_BREAST = "celltypist_breast"
    CELLTYPIST_IMMUNE_HIGH = "celltypist_immune_high"
    CELLTYPIST_IMMUNE_LOW = "celltypist_immune_low"

    # CellTypist ensembles
    CELLTYPIST_TISSUE = "celltypist_tissue"  # Breast + Immune
    CELLTYPIST_UNIVERSAL = "celltypist_universal"  # 10 models
    CELLTYPIST_ALL = "celltypist_all"  # All available human models

    # Combined (PopV-style)
    POPV_MINIMAL = "popv_minimal"  # 2 CT + 2 SR
    POPV_STANDARD = "popv_standard"  # 5 CT + 2 SR
    POPV_COMPREHENSIVE = "popv_comprehensive"  # 10 CT + 4 SR


# CellTypist model sets
CELLTYPIST_MODELS = {
    "breast": ["Cells_Adult_Breast.pkl"],
    "immune_high": ["Immune_All_High.pkl"],
    "immune_low": ["Immune_All_Low.pkl"],
    "tissue": [
        "Cells_Adult_Breast.pkl",
        "Immune_All_High.pkl",
    ],
    "universal": [
        "Cells_Adult_Breast.pkl",
        "Immune_All_High.pkl",
        "Immune_All_Low.pkl",
        "Human_Lung_Atlas.pkl",
        "Healthy_Human_Liver.pkl",
        "Cells_Intestinal_Tract.pkl",
        "Pan_Fetal_Human.pkl",
        "Developing_Human_Brain.pkl",
        "Adult_Human_MTG.pkl",
        "Human_Cell_Landscape.pkl",
    ],
}

# SingleR reference names
SINGLER_REFERENCES = ["hpca", "blueprint", "monaco", "novershtern"]

# Dataset configuration
DATASETS = {
    "rep1": {
        "xenium_path": Path.home() / "datasets/raw/xenium/breast_tumor_rep1/outs",
        "gt_path": Path.home() / "datasets/raw/xenium/breast_tumor_rep1/celltypes_ground_truth_rep1_supervised.xlsx",
        "name": "Xenium Breast Rep1",
    },
    "rep2": {
        "xenium_path": Path.home() / "datasets/raw/xenium/breast_tumor_rep2/outs",
        "gt_path": Path.home() / "datasets/raw/xenium/breast_tumor_rep2/celltypes_ground_truth_rep2_supervised.xlsx",
        "name": "Xenium Breast Rep2",
    },
}

# Ground truth to broad category mapping
GT_BROAD_MAP = {
    "DCIS_1": "Epithelial",
    "DCIS_2": "Epithelial",
    "Invasive_Tumor": "Epithelial",
    "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_KRT15+": "Epithelial",
    "Myoepi_ACTA2+": "Epithelial",
    "T_Cell_&_Tumor_Hybrid": "Epithelial",
    "CD4+_T_Cells": "Immune",
    "CD8+_T_Cells": "Immune",
    "B_Cells": "Immune",
    "Macrophages_1": "Immune",
    "Macrophages_2": "Immune",
    "Mast_Cells": "Immune",
    "LAMP3+_DCs": "Immune",
    "IRF7+_DCs": "Immune",
    "Perivascular-Like": "Immune",
    "Stromal": "Stromal",
    "Stromal_&_T_Cell_Hybrid": "Stromal",
    "Endothelial": "Stromal",  # Merge into Stromal for 3-class
}


@dataclass
class MethodResult:
    """Results from a single annotation method."""
    method: str
    predictions: pl.DataFrame
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    per_class_f1: dict = field(default_factory=dict)
    precision: float = 0.0
    recall: float = 0.0
    kappa: float = 0.0
    mcc: float = 0.0
    confusion_matrix: np.ndarray = None
    class_labels: list = field(default_factory=list)
    runtime_seconds: float = 0.0


def load_xenium_adata(xenium_path: Path, sample_size: int | None = None) -> ad.AnnData:
    """Load Xenium data as AnnData."""
    import h5py
    from scipy.sparse import csc_matrix

    h5_path = xenium_path / "cell_feature_matrix.h5"
    logger.info(f"Loading Xenium data from {h5_path}")

    with h5py.File(h5_path, "r") as f:
        data = f["matrix/data"][:]
        indices = f["matrix/indices"][:]
        indptr = f["matrix/indptr"][:]
        shape = f["matrix/shape"][:]
        barcodes = [b.decode() for b in f["matrix/barcodes"][:]]
        genes = [g.decode() for g in f["matrix/features/name"][:]]

    X = csc_matrix((data, indices, indptr), shape=shape).T
    adata = ad.AnnData(X=X)
    adata.obs_names = barcodes
    adata.var_names = genes
    adata.obs["cell_id"] = barcodes

    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")

    # Sample if requested
    if sample_size and sample_size < adata.n_obs:
        logger.info(f"Sampling {sample_size} from {adata.n_obs} cells")
        indices = np.random.choice(adata.n_obs, sample_size, replace=False)
        adata = adata[indices].copy()

    return adata


def load_ground_truth(gt_path: Path) -> pl.DataFrame:
    """Load ground truth annotations."""
    logger.info(f"Loading ground truth from {gt_path}")
    pd_df = pd.read_excel(gt_path)

    df = pl.DataFrame({
        "cell_id": [str(b) for b in pd_df["Barcode"]],
        "gt_type": pd_df["Cluster"].astype(str).tolist(),
    })

    # Map to broad categories
    df = df.with_columns(
        pl.col("gt_type")
        .map_elements(lambda x: GT_BROAD_MAP.get(x, "Unknown"), return_dtype=pl.Utf8)
        .alias("gt_broad")
    )

    logger.info(f"Loaded {len(df)} cells, {df['gt_type'].n_unique()} types")
    return df


def map_to_broad_category(label: str) -> str:
    """Map fine-grained cell type to broad category."""
    from dapidl.pipeline.components.annotators.mapping import map_to_broad_category as _map
    return _map(label)


def run_singler(adata: ad.AnnData, reference: str) -> pl.DataFrame:
    """Run SingleR with specified reference via rpy2."""
    from dapidl.pipeline.components.annotators.singler import (
        SingleRAnnotator,
        is_singler_available,
    )
    from dapidl.pipeline.base import AnnotationConfig

    if not is_singler_available():
        raise RuntimeError("SingleR not available. Install R with SingleR and celldex packages.")

    logger.info(f"Running SingleR ({reference})...")

    config = AnnotationConfig()
    config.singler_reference = reference

    annotator = SingleRAnnotator(config)
    result = annotator.annotate(adata=adata)

    df = result.annotations_df.select([
        pl.col("cell_id"),
        pl.col("predicted_type"),
        pl.col("broad_category").alias("pred_broad"),
        pl.col("confidence"),
    ])

    return df


def run_singler_ensemble(adata: ad.AnnData, references: list[str]) -> pl.DataFrame:
    """Run multiple SingleR references and combine via voting."""
    all_preds = []

    for ref in references:
        try:
            df = run_singler(adata, ref)
            all_preds.append({
                "source": f"singler_{ref}",
                "labels": df["pred_broad"].to_list(),
                "confidences": df["confidence"].to_list(),
            })
        except Exception as e:
            logger.warning(f"SingleR ({ref}) failed: {e}")

    if not all_preds:
        raise RuntimeError("All SingleR references failed")

    return _combine_predictions(adata, all_preds)


def run_celltypist(adata: ad.AnnData, models: list[str]) -> pl.DataFrame:
    """Run CellTypist with specified models."""
    import celltypist
    from celltypist import models as ct_models

    logger.info(f"Running CellTypist ({len(models)} models)...")

    # Normalize data
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)

    all_preds = []

    for model_name in models:
        try:
            ct_models.download_models(model=[model_name])
            model = ct_models.Model.load(model=model_name)

            predictions = celltypist.annotate(
                adata_norm, model=model, majority_voting=False
            )

            all_preds.append({
                "source": f"celltypist_{model_name}",
                "labels": [
                    map_to_broad_category(l)
                    for l in predictions.predicted_labels.predicted_labels.tolist()
                ],
                "confidences": predictions.probability_matrix.max(axis=1).tolist(),
            })

        except Exception as e:
            logger.warning(f"CellTypist {model_name} failed: {e}")

    if not all_preds:
        raise RuntimeError("All CellTypist models failed")

    return _combine_predictions(adata, all_preds)


def run_combined(
    adata: ad.AnnData,
    celltypist_models: list[str],
    singler_references: list[str],
) -> pl.DataFrame:
    """Run combined CellTypist + SingleR (PopV-style)."""
    logger.info(
        f"Running combined: {len(celltypist_models)} CT + {len(singler_references)} SR"
    )

    all_preds = []

    # CellTypist predictions
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)

    import celltypist
    from celltypist import models as ct_models

    for model_name in celltypist_models:
        try:
            ct_models.download_models(model=[model_name])
            model = ct_models.Model.load(model=model_name)
            predictions = celltypist.annotate(
                adata_norm, model=model, majority_voting=False
            )
            all_preds.append({
                "source": f"celltypist_{model_name}",
                "labels": [
                    map_to_broad_category(l)
                    for l in predictions.predicted_labels.predicted_labels.tolist()
                ],
                "confidences": predictions.probability_matrix.max(axis=1).tolist(),
            })
        except Exception as e:
            logger.warning(f"CellTypist {model_name} failed: {e}")

    # SingleR predictions
    for ref in singler_references:
        try:
            df = run_singler(adata, ref)
            all_preds.append({
                "source": f"singler_{ref}",
                "labels": df["pred_broad"].to_list(),
                "confidences": df["confidence"].to_list(),
            })
        except Exception as e:
            logger.warning(f"SingleR ({ref}) failed: {e}")

    if not all_preds:
        raise RuntimeError("All methods failed")

    return _combine_predictions(adata, all_preds)


def _combine_predictions(adata: ad.AnnData, all_preds: list[dict]) -> pl.DataFrame:
    """Combine predictions via unweighted majority voting (PopV-style)."""
    n_cells = len(all_preds[0]["labels"])
    final_labels = []
    final_confidences = []

    for i in range(n_cells):
        # Unweighted majority vote (popV style - beats confidence-weighted)
        votes = [p["labels"][i] for p in all_preds]
        vote_counts = Counter(votes)
        winner = vote_counts.most_common(1)[0][0]

        # Mean confidence of winning votes
        winner_confs = [
            p["confidences"][i] for p in all_preds if p["labels"][i] == winner
        ]
        confidence = np.mean(winner_confs)

        final_labels.append(winner)
        final_confidences.append(confidence)

    return pl.DataFrame({
        "cell_id": [str(c) for c in adata.obs_names],
        "predicted_type": final_labels,
        "pred_broad": final_labels,
        "confidence": final_confidences,
    })


def run_method(adata: ad.AnnData, method: AnnotationMethod) -> pl.DataFrame:
    """Run a specific annotation method."""
    start_time = time.time()

    if method == AnnotationMethod.SINGLER_HPCA:
        df = run_singler(adata, "hpca")
    elif method == AnnotationMethod.SINGLER_BLUEPRINT:
        df = run_singler(adata, "blueprint")
    elif method == AnnotationMethod.SINGLER_MONACO:
        df = run_singler(adata, "monaco")
    elif method == AnnotationMethod.SINGLER_NOVERSHTERN:
        df = run_singler(adata, "novershtern")
    elif method == AnnotationMethod.SINGLER_ALL:
        df = run_singler_ensemble(adata, SINGLER_REFERENCES)

    elif method == AnnotationMethod.CELLTYPIST_BREAST:
        df = run_celltypist(adata, CELLTYPIST_MODELS["breast"])
    elif method == AnnotationMethod.CELLTYPIST_IMMUNE_HIGH:
        df = run_celltypist(adata, CELLTYPIST_MODELS["immune_high"])
    elif method == AnnotationMethod.CELLTYPIST_IMMUNE_LOW:
        df = run_celltypist(adata, CELLTYPIST_MODELS["immune_low"])
    elif method == AnnotationMethod.CELLTYPIST_TISSUE:
        df = run_celltypist(adata, CELLTYPIST_MODELS["tissue"])
    elif method == AnnotationMethod.CELLTYPIST_UNIVERSAL:
        df = run_celltypist(adata, CELLTYPIST_MODELS["universal"])
    elif method == AnnotationMethod.CELLTYPIST_ALL:
        # Get all available human models
        from celltypist import models as ct_models
        all_models = [m for m in ct_models.get_all_models() if "Human" in m][:20]
        df = run_celltypist(adata, all_models)

    elif method == AnnotationMethod.POPV_MINIMAL:
        df = run_combined(
            adata,
            celltypist_models=["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
            singler_references=["hpca", "blueprint"],
        )
    elif method == AnnotationMethod.POPV_STANDARD:
        df = run_combined(
            adata,
            celltypist_models=[
                "Cells_Adult_Breast.pkl",
                "Immune_All_High.pkl",
                "Immune_All_Low.pkl",
                "Human_Lung_Atlas.pkl",
                "Healthy_Human_Liver.pkl",
            ],
            singler_references=["hpca", "blueprint"],
        )
    elif method == AnnotationMethod.POPV_COMPREHENSIVE:
        df = run_combined(
            adata,
            celltypist_models=CELLTYPIST_MODELS["universal"],
            singler_references=SINGLER_REFERENCES,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed = time.time() - start_time
    logger.info(f"  Completed in {elapsed:.1f}s")

    return df


def evaluate_predictions(
    predictions: pl.DataFrame,
    ground_truth: pl.DataFrame,
    method: str,
) -> MethodResult:
    """Evaluate predictions against ground truth."""
    # Join predictions with ground truth
    merged = predictions.join(
        ground_truth.select(["cell_id", "gt_broad"]),
        on="cell_id",
        how="inner",
    )

    # Filter out Unknown
    merged = merged.filter(
        (pl.col("gt_broad") != "Unknown") & (pl.col("pred_broad") != "Unknown")
    )

    y_true = merged["gt_broad"].to_list()
    y_pred = merged["pred_broad"].to_list()

    labels = sorted(set(y_true) | set(y_pred))

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Per-class F1
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class_f1 = {
        label: report[label]["f1-score"] for label in labels if label in report
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return MethodResult(
        method=method,
        predictions=merged,
        accuracy=acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_f1=per_class_f1,
        precision=precision,
        recall=recall,
        kappa=kappa,
        mcc=mcc,
        confusion_matrix=cm,
        class_labels=labels,
    )


def create_confusion_matrix_plot(
    result: MethodResult,
    output_path: Path,
    normalize: bool = True,
) -> Path:
    """Create confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))

    cm = result.confusion_matrix.astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=result.class_labels,
        yticklabels=result.class_labels,
        ax=ax,
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title(
        f"Confusion Matrix: {result.method}\nAccuracy: {result.accuracy:.3f}, Macro F1: {result.macro_f1:.3f}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrix: {output_path}")
    return output_path


def create_umap_comparison(
    adata: ad.AnnData,
    result: MethodResult,
    output_path: Path,
) -> Path:
    """Create UMAP visualization comparing GT vs predicted."""
    logger.info("Computing UMAP...")

    # Prepare data
    adata_vis = adata.copy()
    sc.pp.normalize_total(adata_vis, target_sum=1e4)
    sc.pp.log1p(adata_vis)

    # Highly variable genes
    if adata_vis.n_vars > 2000:
        sc.pp.highly_variable_genes(adata_vis, n_top_genes=2000, subset=True)

    # PCA + UMAP
    sc.pp.pca(adata_vis, n_comps=min(50, adata_vis.n_vars - 1))
    sc.pp.neighbors(adata_vis, n_neighbors=15)
    sc.tl.umap(adata_vis, min_dist=0.1)

    # Add annotations
    pred_df = result.predictions
    gt_map = dict(zip(pred_df["cell_id"].to_list(), pred_df["gt_broad"].to_list()))
    pred_map = dict(zip(pred_df["cell_id"].to_list(), pred_df["pred_broad"].to_list()))

    adata_vis.obs["Ground Truth"] = [
        gt_map.get(str(c), "Unknown") for c in adata_vis.obs_names
    ]
    adata_vis.obs["Predicted"] = [
        pred_map.get(str(c), "Unknown") for c in adata_vis.obs_names
    ]

    # Filter to cells with both annotations
    mask = (adata_vis.obs["Ground Truth"] != "Unknown") & (
        adata_vis.obs["Predicted"] != "Unknown"
    )
    adata_vis = adata_vis[mask].copy()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sc.pl.umap(adata_vis, color="Ground Truth", ax=axes[0], show=False, title="Ground Truth")
    sc.pl.umap(adata_vis, color="Predicted", ax=axes[1], show=False, title=f"Predicted ({result.method})")

    plt.suptitle(
        f"Accuracy: {result.accuracy:.3f}, Macro F1: {result.macro_f1:.3f}",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved UMAP comparison: {output_path}")
    return output_path


def export_xenium_csv(result: MethodResult, output_path: Path) -> Path:
    """Export predictions as CSV for Xenium Explorer import."""
    # Format for Xenium Explorer: Barcode, CellType column
    df = result.predictions.select([
        pl.col("cell_id").alias("Barcode"),
        pl.col("pred_broad").alias(f"CellType_{result.method}"),
        pl.col("predicted_type").alias(f"DetailedType_{result.method}")
        if "predicted_type" in result.predictions.columns
        else pl.col("pred_broad").alias(f"DetailedType_{result.method}"),
        pl.col("confidence").alias(f"Confidence_{result.method}")
        if "confidence" in result.predictions.columns
        else pl.lit(1.0).alias(f"Confidence_{result.method}"),
    ])

    df.write_csv(output_path)
    logger.info(f"Exported Xenium CSV: {output_path}")
    return output_path


def run_benchmark(
    datasets: list[str],
    methods: list[AnnotationMethod],
    output_dir: Path,
    sample_size: int | None = None,
    use_clearml: bool = True,
    generate_umap: bool = True,
    export_csv: bool = True,
) -> dict[str, dict[str, MethodResult]]:
    """Run complete benchmark."""
    # Initialize ClearML
    task = None
    if use_clearml and CLEARML_AVAILABLE:
        task = Task.init(
            project_name="DAPIDL/Annotation-Benchmark",
            task_name=f"Comprehensive_Benchmark_{len(methods)}_methods",
        )
        task.set_parameters({
            "datasets": datasets,
            "methods": [m.value for m in methods],
            "sample_size": sample_size,
        })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset_key in datasets:
        if dataset_key not in DATASETS:
            logger.warning(f"Unknown dataset: {dataset_key}")
            continue

        config = DATASETS[dataset_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"DATASET: {config['name']}")
        logger.info(f"{'='*60}")

        # Load data
        adata = load_xenium_adata(config["xenium_path"], sample_size=sample_size)
        ground_truth = load_ground_truth(config["gt_path"])

        # Filter to common cells
        adata_cells = set(str(c) for c in adata.obs_names)
        gt_cells = set(ground_truth["cell_id"].to_list())
        common_cells = adata_cells & gt_cells

        mask = [str(c) in common_cells for c in adata.obs_names]
        adata = adata[mask].copy()
        ground_truth = ground_truth.filter(pl.col("cell_id").is_in(list(common_cells)))

        logger.info(f"Common cells: {len(common_cells)}")

        dataset_results = {}
        dataset_dir = output_dir / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for method in methods:
            logger.info(f"\n--- Method: {method.value} ---")

            try:
                start = time.time()
                predictions = run_method(adata, method)
                runtime = time.time() - start

                # Evaluate
                result = evaluate_predictions(predictions, ground_truth, method.value)
                result.runtime_seconds = runtime
                dataset_results[method.value] = result

                logger.info(f"  Accuracy: {result.accuracy:.3f}")
                logger.info(f"  Macro F1: {result.macro_f1:.3f}")
                logger.info(f"  Kappa: {result.kappa:.3f}")
                logger.info(f"  Per-class F1: {result.per_class_f1}")

                # Create method output directory
                method_dir = dataset_dir / method.value
                method_dir.mkdir(parents=True, exist_ok=True)

                # Confusion matrix
                cm_path = method_dir / "confusion_matrix.png"
                create_confusion_matrix_plot(result, cm_path)
                if task:
                    task.upload_artifact(f"{dataset_key}/{method.value}/confusion_matrix", cm_path)

                # UMAP (optional - slow)
                if generate_umap:
                    umap_path = method_dir / "umap_comparison.png"
                    try:
                        create_umap_comparison(adata, result, umap_path)
                        if task:
                            task.upload_artifact(f"{dataset_key}/{method.value}/umap", umap_path)
                    except Exception as e:
                        logger.warning(f"UMAP failed: {e}")

                # Xenium CSV export
                if export_csv:
                    csv_path = method_dir / f"{method.value}_xenium.csv"
                    export_xenium_csv(result, csv_path)
                    if task:
                        task.upload_artifact(f"{dataset_key}/{method.value}/xenium_csv", csv_path)

                # Log metrics to ClearML
                if task:
                    clearml_logger = task.get_logger()
                    clearml_logger.report_scalar(
                        title=dataset_key,
                        series=method.value,
                        value=result.macro_f1,
                        iteration=0,
                    )
                    clearml_logger.report_confusion_matrix(
                        title=f"Confusion Matrix - {method.value}",
                        series=dataset_key,
                        matrix=result.confusion_matrix,
                        xlabels=result.class_labels,
                        ylabels=result.class_labels,
                        iteration=0,
                    )

            except Exception as e:
                logger.error(f"Method {method.value} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        all_results[dataset_key] = dataset_results

    # Save summary JSON
    summary = {}
    for dataset, results in all_results.items():
        summary[dataset] = {
            method: {
                "accuracy": r.accuracy,
                "macro_f1": r.macro_f1,
                "weighted_f1": r.weighted_f1,
                "precision": r.precision,
                "recall": r.recall,
                "kappa": r.kappa,
                "mcc": r.mcc,
                "per_class_f1": r.per_class_f1,
                "runtime_seconds": r.runtime_seconds,
            }
            for method, r in results.items()
        }

    summary_path = output_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSummary saved to: {summary_path}")

    if task:
        task.upload_artifact("benchmark_summary", summary_path)

    # Print summary table
    print_summary_table(all_results)

    if task:
        task.close()

    return all_results


def print_summary_table(results: dict[str, dict[str, MethodResult]]) -> None:
    """Print formatted summary table."""
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    for dataset, methods in results.items():
        print(f"\n{dataset}:")
        print("-" * 95)
        print(
            f"{'Method':<30} {'Acc':>8} {'F1':>8} {'Epi':>8} {'Imm':>8} {'Str':>8} {'Kappa':>8} {'Time':>8}"
        )
        print("-" * 95)

        # Sort by macro F1
        for method, result in sorted(
            methods.items(), key=lambda x: -x[1].macro_f1
        ):
            epi = result.per_class_f1.get("Epithelial", 0)
            imm = result.per_class_f1.get("Immune", 0)
            strom = result.per_class_f1.get("Stromal", 0)

            print(
                f"{method:<30} {result.accuracy:>8.3f} {result.macro_f1:>8.3f} "
                f"{epi:>8.3f} {imm:>8.3f} {strom:>8.3f} {result.kappa:>8.3f} {result.runtime_seconds:>7.1f}s"
            )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark of annotation methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        "-d",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=["rep1", "rep2"],
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--methods",
        "-m",
        nargs="+",
        choices=[m.value for m in AnnotationMethod],
        default=None,
        help="Methods to test (default: all)",
    )
    parser.add_argument(
        "--sample-size",
        "-s",
        type=int,
        default=None,
        help="Sample size per dataset (for faster testing)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory",
    )
    parser.add_argument(
        "--no-clearml",
        action="store_true",
        help="Disable ClearML tracking",
    )
    parser.add_argument(
        "--no-umap",
        action="store_true",
        help="Skip UMAP generation (faster)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip Xenium CSV export",
    )

    args = parser.parse_args()

    # Default to all methods if not specified
    if args.methods:
        methods = [AnnotationMethod(m) for m in args.methods]
    else:
        methods = [
            # SingleR single
            AnnotationMethod.SINGLER_HPCA,
            AnnotationMethod.SINGLER_BLUEPRINT,
            # CellTypist single
            AnnotationMethod.CELLTYPIST_BREAST,
            AnnotationMethod.CELLTYPIST_IMMUNE_HIGH,
            # Ensembles
            AnnotationMethod.CELLTYPIST_TISSUE,
            AnnotationMethod.SINGLER_ALL,
            # Combined
            AnnotationMethod.POPV_MINIMAL,
            AnnotationMethod.POPV_STANDARD,
        ]

    run_benchmark(
        datasets=args.datasets,
        methods=methods,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        use_clearml=not args.no_clearml,
        generate_umap=not args.no_umap,
        export_csv=not args.no_csv,
    )


if __name__ == "__main__":
    main()
