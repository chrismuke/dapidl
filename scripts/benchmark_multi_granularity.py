#!/usr/bin/env python3
"""Multi-Granularity Benchmark of Cell Type Annotation Methods.

This script evaluates annotation methods at three granularity levels:
- COARSE: 3 classes (Epithelial, Immune, Stromal)
- MEDIUM: 10 classes (T_Cell, B_Cell, Myeloid, etc.)
- FINE: 17 classes (original ground truth labels)

Methods tested:
- CellTypist (breast, tissue, universal models)
- SingleR (HPCA, Blueprint references)
- PopV-style ensemble (minimal: 4 voters, standard: 7 voters)

Output artifacts:
- Confusion matrices (PNG)
- Classification reports (JSON)
- ClearML experiment tracking

Usage:
    # Run all granularities on both datasets
    uv run python scripts/benchmark_multi_granularity_v2.py

    # Run specific granularity
    uv run python scripts/benchmark_multi_granularity_v2.py --granularity medium

    # Sample for faster testing
    uv run python scripts/benchmark_multi_granularity_v2.py --sample-size 10000

Dependencies:
    - CellTypist (models downloaded automatically)
    - SingleR (via rpy2, requires R installation)
    - ClearML (optional, for experiment tracking)
"""

from __future__ import annotations

import datetime
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path

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
    precision_score,
    recall_score,
)

# Import canonical mappings from dapidl
from dapidl.pipeline.components.annotators.mapping import (
    CELL_TYPE_HIERARCHY,
    COARSE_CLASS_NAMES,
    FINEGRAINED_CLASS_NAMES,
    GROUND_TRUTH_MAPPING,
    map_to_broad_category,
)

# ClearML
try:
    from clearml import Task

    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False


# ============================================================================
# GRANULARITY DEFINITIONS
# ============================================================================


class Granularity(str, Enum):
    """Annotation granularity levels."""

    COARSE = "coarse"  # 3 classes: Epithelial, Immune, Stromal
    MEDIUM = "medium"  # 10 classes: T_Cell, B_Cell, Myeloid, etc.
    FINE = "fine"  # 17 classes: original ground truth labels


MEDIUM_CLASS_NAMES = [
    "Epithelial_Luminal",
    "Epithelial_Basal",
    "Epithelial_Tumor",
    "T_Cell",
    "B_Cell",
    "Myeloid",
    "NK_Cell",
    "Stromal_Fibroblast",
    "Stromal_Pericyte",
    "Endothelial",
]


# ============================================================================
# GROUND TRUTH MAPPINGS (extend canonical mapping.py)
# ============================================================================

# Ground truth -> Medium granularity
GT_TO_MEDIUM: dict[str, str] = {
    "DCIS_1": "Epithelial_Tumor",
    "DCIS_2": "Epithelial_Tumor",
    "Invasive_Tumor": "Epithelial_Tumor",
    "Prolif_Invasive_Tumor": "Epithelial_Tumor",
    "Myoepi_KRT15+": "Epithelial_Basal",
    "Myoepi_ACTA2+": "Epithelial_Basal",
    "CD4+_T_Cells": "T_Cell",
    "CD8+_T_Cells": "T_Cell",
    "B_Cells": "B_Cell",
    "Macrophages_1": "Myeloid",
    "Macrophages_2": "Myeloid",
    "Mast_Cells": "Myeloid",
    "LAMP3+_DCs": "Myeloid",
    "IRF7+_DCs": "Myeloid",
    "Perivascular-Like": "Stromal_Pericyte",  # FIXED: Was incorrectly "Immune"
    "Stromal": "Stromal_Fibroblast",
    "Endothelial": "Endothelial",
    # Hybrid cells - exclude from evaluation
    "T_Cell_&_Tumor_Hybrid": "Hybrid",  # FIXED: Was incorrectly "Epithelial_Tumor"
    "Stromal_&_T_Cell_Hybrid": "Hybrid",  # FIXED: Was incorrectly "Stromal_Fibroblast"
}


# ============================================================================
# CELLTYPIST AND SINGLER MAPPINGS
# ============================================================================

# CellTypist predictions -> Medium granularity
CELLTYPIST_TO_MEDIUM: dict[str, str] = {
    # Luminal
    "LummHR": "Epithelial_Luminal",
    "LummHR-SCGB": "Epithelial_Luminal",
    "LummHR-active": "Epithelial_Luminal",
    "LummHR-major": "Epithelial_Luminal",
    "Lumsec": "Epithelial_Luminal",
    "Lumsec-HLA": "Epithelial_Luminal",
    "Lumsec-KIT": "Epithelial_Luminal",
    "Lumsec-lac": "Epithelial_Luminal",
    "Lumsec-major": "Epithelial_Luminal",
    # Basal
    "Lumsec-basal": "Epithelial_Basal",
    "Lumsec-myo": "Epithelial_Basal",
    "basal": "Epithelial_Basal",
    # Tumor (proliferating)
    "Lumsec-prol": "Epithelial_Tumor",
    # T cells
    "CD4": "T_Cell",
    "CD8": "T_Cell",
    "T_prol": "T_Cell",
    "GD": "T_Cell",
    "NKT": "T_Cell",
    "Treg": "T_Cell",
    # NK
    "NK": "NK_Cell",
    "NK-ILCs": "NK_Cell",
    # B cells
    "b_naive": "B_Cell",
    "bmem_switched": "B_Cell",
    "bmem_unswitched": "B_Cell",
    "plasma": "B_Cell",
    "plasma_IgA": "B_Cell",
    "plasma_IgG": "B_Cell",
    # Myeloid
    "Macro": "Myeloid",
    "Mono": "Myeloid",
    "cDC": "Myeloid",
    "mDC": "Myeloid",
    "pDC": "Myeloid",
    "mye-prol": "Myeloid",
    "Mast": "Myeloid",
    "Neutrophil": "Myeloid",
    # Stromal
    "Fibro": "Stromal_Fibroblast",
    "Fibro-SFRP4": "Stromal_Fibroblast",
    "Fibro-major": "Stromal_Fibroblast",
    "Fibro-matrix": "Stromal_Fibroblast",
    "Fibro-prematrix": "Stromal_Fibroblast",
    "adipocytes": "Stromal_Fibroblast",
    "pericytes": "Stromal_Pericyte",
    # Endothelial
    "Endo": "Endothelial",
    "Endo-vas": "Endothelial",
    "Endo-lymph": "Endothelial",
}

# SingleR predictions -> Medium granularity
SINGLER_TO_MEDIUM: dict[str, str] = {
    "Epithelial cells": "Epithelial_Luminal",
    "Keratinocytes": "Epithelial_Basal",
    "CD4+ T-cells": "T_Cell",
    "CD8+ T-cells": "T_Cell",
    "B-cells": "B_Cell",
    "NK cells": "NK_Cell",
    "Macrophages": "Myeloid",
    "Monocytes": "Myeloid",
    "DC": "Myeloid",
    "Neutrophils": "Myeloid",
    "Eosinophils": "Myeloid",
    "Fibroblasts": "Stromal_Fibroblast",
    "Adipocytes": "Stromal_Fibroblast",
    "Smooth muscle": "Stromal_Pericyte",
    "Endothelial cells": "Endothelial",
}


# ============================================================================
# RESULT DATACLASS
# ============================================================================


@dataclass
class BenchmarkResult:
    """Structured result from a benchmark run."""

    granularity: str
    n_evaluated: int
    n_classes: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    precision: float
    recall: float
    kappa: float
    per_class_f1: dict[str, float] = field(default_factory=dict)
    per_class_support: dict[str, int] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    runtime_seconds: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if k != "confusion_matrix"  # Skip large arrays
        }


# ============================================================================
# DATASET CONFIG
# ============================================================================

DATASETS = {
    "rep1": {
        "xenium_path": Path.home() / "datasets/raw/xenium/breast_tumor_rep1/outs",
        "gt_path": Path.home()
        / "datasets/raw/xenium/breast_tumor_rep1/celltypes_ground_truth_rep1_supervised.xlsx",
        "name": "Xenium Breast Rep1",
    },
    "rep2": {
        "xenium_path": Path.home() / "datasets/raw/xenium/breast_tumor_rep2/outs",
        "gt_path": Path.home()
        / "datasets/raw/xenium/breast_tumor_rep2/celltypes_ground_truth_rep2_supervised.xlsx",
        "name": "Xenium Breast Rep2",
    },
}

CELLTYPIST_MODELS = {
    "breast": ["Cells_Adult_Breast.pkl"],
    "immune_high": ["Immune_All_High.pkl"],
    "tissue": ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
    "universal": [
        "Cells_Adult_Breast.pkl",
        "Immune_All_High.pkl",
        "Immune_All_Low.pkl",
        "Human_Lung_Atlas.pkl",
        "Healthy_Human_Liver.pkl",
    ],
}


# ============================================================================
# MAPPING FUNCTIONS
# ============================================================================


def map_gt_to_granularity(label: str, granularity: Granularity) -> str:
    """Map ground truth label to specified granularity.

    Args:
        label: Ground truth cell type label
        granularity: Target granularity level

    Returns:
        Mapped label, or "Exclude" for hybrid cells at COARSE/MEDIUM
    """
    if granularity == Granularity.FINE:
        return label

    # Use canonical mapping for COARSE
    if granularity == Granularity.COARSE:
        mapped = GROUND_TRUTH_MAPPING.get(label, "Unknown")
        # Exclude hybrid cells
        if mapped in ("Hybrid", "Unlabeled"):
            return "Exclude"
        return mapped

    # MEDIUM granularity
    mapped = GT_TO_MEDIUM.get(label, "Unknown")
    if mapped == "Hybrid":
        return "Exclude"
    return mapped


def map_prediction_to_granularity(
    label: str, source: str, granularity: Granularity
) -> str:
    """Map a prediction to the specified granularity.

    Args:
        label: Predicted cell type
        source: Prediction source ("celltypist" or "singler")
        granularity: Target granularity level

    Returns:
        Mapped label
    """
    if granularity == Granularity.FINE:
        return label

    # COARSE: use canonical map_to_broad_category
    if granularity == Granularity.COARSE:
        return map_to_broad_category(label)

    # MEDIUM: use specific mappings
    if source == "celltypist":
        if label in CELLTYPIST_TO_MEDIUM:
            return CELLTYPIST_TO_MEDIUM[label]
    elif source == "singler":
        if label in SINGLER_TO_MEDIUM:
            return SINGLER_TO_MEDIUM[label]

    # Fallback: map to coarse, then infer medium
    broad = map_to_broad_category(label)
    default_medium = {
        "Epithelial": "Epithelial_Luminal",
        "Immune": "Myeloid",
        "Stromal": "Stromal_Fibroblast",
    }
    return default_medium.get(broad, "Unknown")


# ============================================================================
# DATA LOADING
# ============================================================================


def load_xenium_adata(
    xenium_path: Path, sample_size: int | None = None, seed: int = 42
) -> ad.AnnData:
    """Load Xenium expression data from HDF5 file.

    Args:
        xenium_path: Path to Xenium output directory (containing cell_feature_matrix.h5)
        sample_size: If provided, randomly sample this many cells for faster testing
        seed: Random seed for reproducibility

    Returns:
        AnnData object with expression matrix (cells x genes)

    Raises:
        FileNotFoundError: If cell_feature_matrix.h5 not found
    """
    import h5py
    from scipy.sparse import csc_matrix

    h5_path = xenium_path / "cell_feature_matrix.h5"
    logger.info(f"Loading from {h5_path}")

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

    if sample_size and sample_size < adata.n_obs:
        logger.info(f"Sampling {sample_size} from {adata.n_obs} (seed={seed})")
        rng = np.random.default_rng(seed)
        idx = rng.choice(adata.n_obs, sample_size, replace=False)
        adata = adata[idx].copy()

    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def load_ground_truth(gt_path: Path, granularity: Granularity) -> pl.DataFrame:
    """Load and map ground truth labels.

    Args:
        gt_path: Path to Excel file with ground truth labels
        granularity: Target granularity for mapping

    Returns:
        DataFrame with columns: cell_id, gt_fine, gt_mapped
    """
    logger.info(f"Loading ground truth from {gt_path}")
    pd_df = pd.read_excel(gt_path)  # Polars Excel support is limited

    df = pl.DataFrame(
        {
            "cell_id": [str(b) for b in pd_df["Barcode"]],
            "gt_fine": pd_df["Cluster"].astype(str).tolist(),
        }
    )

    # Efficient mapping using polars replace
    if granularity == Granularity.COARSE:
        # Include Exclude for hybrid cells
        mapping_dict = {
            k: ("Exclude" if v in ("Hybrid", "Unlabeled") else v)
            for k, v in GROUND_TRUTH_MAPPING.items()
        }
    elif granularity == Granularity.MEDIUM:
        mapping_dict = {
            k: ("Exclude" if v == "Hybrid" else v) for k, v in GT_TO_MEDIUM.items()
        }
    else:
        mapping_dict = {k: k for k in FINEGRAINED_CLASS_NAMES}

    df = df.with_columns(
        pl.col("gt_fine").replace(mapping_dict, default="Unknown").alias("gt_mapped")
    )

    n_unique = df.filter(pl.col("gt_mapped") != "Exclude")["gt_mapped"].n_unique()
    logger.info(f"Loaded {len(df)} cells, {n_unique} classes at {granularity.value}")
    return df


def get_normalized_adata(adata: ad.AnnData) -> ad.AnnData:
    """Get normalized AnnData for CellTypist (log1p normalized).

    Args:
        adata: Raw count matrix

    Returns:
        Normalized AnnData copy
    """
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    return adata_norm


# ============================================================================
# ANNOTATION FUNCTIONS
# ============================================================================


def run_celltypist(
    adata: ad.AnnData,
    adata_norm: ad.AnnData,
    models: list[str],
    granularity: Granularity,
) -> pl.DataFrame:
    """Run CellTypist annotation.

    Args:
        adata: Raw AnnData (for cell IDs)
        adata_norm: Normalized AnnData for CellTypist
        models: List of CellTypist model names
        granularity: Target granularity

    Returns:
        DataFrame with predictions
    """
    import celltypist
    from celltypist import models as ct_models

    logger.info(f"Running CellTypist ({len(models)} models)...")

    all_preds = []
    for model_name in models:
        try:
            ct_models.download_models(model=[model_name])
            model = ct_models.Model.load(model=model_name)
            pred = celltypist.annotate(adata_norm, model=model, majority_voting=False)

            all_preds.append(
                {
                    "labels": [
                        map_prediction_to_granularity(l, "celltypist", granularity)
                        for l in pred.predicted_labels.predicted_labels.tolist()
                    ],
                    "fine_labels": pred.predicted_labels.predicted_labels.tolist(),
                    "confidences": pred.probability_matrix.max(axis=1).tolist(),
                }
            )
        except Exception as e:
            logger.warning(f"CellTypist {model_name} failed: {e}")

    return _combine_predictions(adata, all_preds, granularity)


def run_singler(
    adata: ad.AnnData, reference: str, granularity: Granularity
) -> pl.DataFrame:
    """Run SingleR annotation.

    Args:
        adata: AnnData object
        reference: SingleR reference ("hpca" or "blueprint")
        granularity: Target granularity

    Returns:
        DataFrame with predictions
    """
    from dapidl.pipeline.base import AnnotationConfig
    from dapidl.pipeline.components.annotators.singler import (
        SingleRAnnotator,
        is_singler_available,
    )

    if not is_singler_available():
        raise RuntimeError("SingleR not available")

    logger.info(f"Running SingleR ({reference})...")

    config = AnnotationConfig()
    config.singler_reference = reference
    annotator = SingleRAnnotator(config)
    result = annotator.annotate(adata=adata)

    df = result.annotations_df

    # Map to granularity
    mapped = [
        map_prediction_to_granularity(l, "singler", granularity)
        for l in df["predicted_type"].to_list()
    ]

    return pl.DataFrame(
        {
            "cell_id": df["cell_id"].to_list(),
            "pred_mapped": mapped,
            "pred_fine": df["predicted_type"].to_list(),
            "confidence": df["confidence"].to_list(),
        }
    )


def run_combined(
    adata: ad.AnnData,
    adata_norm: ad.AnnData,
    celltypist_models: list[str],
    singler_references: list[str],
    granularity: Granularity,
) -> pl.DataFrame:
    """Run combined CellTypist + SingleR ensemble (PopV-style).

    Args:
        adata: Raw AnnData
        adata_norm: Normalized AnnData for CellTypist
        celltypist_models: List of CellTypist model names
        singler_references: List of SingleR references
        granularity: Target granularity

    Returns:
        DataFrame with majority-vote predictions
    """
    logger.info(
        f"Running combined: {len(celltypist_models)} CT + {len(singler_references)} SR"
    )

    all_preds = []

    # CellTypist
    import celltypist
    from celltypist import models as ct_models

    for model_name in celltypist_models:
        try:
            ct_models.download_models(model=[model_name])
            model = ct_models.Model.load(model=model_name)
            pred = celltypist.annotate(adata_norm, model=model, majority_voting=False)

            all_preds.append(
                {
                    "labels": [
                        map_prediction_to_granularity(l, "celltypist", granularity)
                        for l in pred.predicted_labels.predicted_labels.tolist()
                    ],
                    "fine_labels": pred.predicted_labels.predicted_labels.tolist(),
                    "confidences": pred.probability_matrix.max(axis=1).tolist(),
                }
            )
        except Exception as e:
            logger.warning(f"CellTypist {model_name} failed: {e}")

    # SingleR
    for ref in singler_references:
        try:
            df = run_singler(adata, ref, granularity)
            all_preds.append(
                {
                    "labels": df["pred_mapped"].to_list(),
                    "fine_labels": df["pred_fine"].to_list(),
                    "confidences": df["confidence"].to_list(),
                }
            )
        except Exception as e:
            logger.warning(f"SingleR ({ref}) failed: {e}")

    return _combine_predictions(adata, all_preds, granularity)


def _combine_predictions(
    adata: ad.AnnData, all_preds: list[dict], granularity: Granularity
) -> pl.DataFrame:
    """Combine predictions via majority voting.

    Args:
        adata: AnnData for cell IDs
        all_preds: List of prediction dicts with 'labels', 'fine_labels', 'confidences'
        granularity: Target granularity (unused, for signature consistency)

    Returns:
        DataFrame with combined predictions

    Raises:
        RuntimeError: If no predictions provided
        ValueError: If predictions have inconsistent lengths
    """
    if not all_preds:
        raise RuntimeError("No predictions available")

    n_cells = len(all_preds[0]["labels"])

    # Validate consistent lengths
    for i, p in enumerate(all_preds):
        if len(p["labels"]) != n_cells:
            raise ValueError(
                f"Prediction {i} has {len(p['labels'])} labels, expected {n_cells}"
            )

    final_labels = []
    final_fine = []
    final_conf = []

    for i in range(n_cells):
        votes = [p["labels"][i] for p in all_preds]
        vote_counts = Counter(votes)
        winner = vote_counts.most_common(1)[0][0]

        # Get fine label from first predictor that voted for winner
        fine_label = winner
        for p in all_preds:
            if p["labels"][i] == winner:
                fine_label = p.get("fine_labels", [winner] * n_cells)[i]
                break

        # Average confidence of winning voters
        winner_confs = [
            p["confidences"][i] for p in all_preds if p["labels"][i] == winner
        ]
        conf = np.mean(winner_confs) if winner_confs else 0.0

        final_labels.append(winner)
        final_fine.append(fine_label)
        final_conf.append(conf)

    return pl.DataFrame(
        {
            "cell_id": [str(c) for c in adata.obs_names],
            "pred_mapped": final_labels,
            "pred_fine": final_fine,
            "confidence": final_conf,
        }
    )


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate(
    predictions: pl.DataFrame, ground_truth: pl.DataFrame, granularity: Granularity
) -> BenchmarkResult:
    """Evaluate predictions against ground truth.

    Args:
        predictions: DataFrame with pred_mapped column
        ground_truth: DataFrame with gt_mapped column
        granularity: Granularity level for reporting

    Returns:
        BenchmarkResult with all metrics
    """
    merged = predictions.join(
        ground_truth.select(["cell_id", "gt_mapped"]),
        on="cell_id",
        how="inner",
    )

    # Filter excluded cells (Hybrid, Unknown)
    merged = merged.filter(
        (pl.col("gt_mapped") != "Exclude")
        & (pl.col("gt_mapped") != "Unknown")
        & (pl.col("pred_mapped") != "Unknown")
    )

    y_true = merged["gt_mapped"].to_list()
    y_pred = merged["pred_mapped"].to_list()

    if not y_true:
        return BenchmarkResult(
            granularity=granularity.value,
            n_evaluated=0,
            n_classes=0,
            accuracy=0,
            macro_f1=0,
            weighted_f1=0,
            precision=0,
            recall=0,
            kappa=0,
            error="No valid predictions after filtering",
        )

    labels = sorted(set(y_true) | set(y_pred))

    # Calculate metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return BenchmarkResult(
        granularity=granularity.value,
        n_evaluated=len(y_true),
        n_classes=len(labels),
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        weighted_f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
        kappa=cohen_kappa_score(y_true, y_pred),
        per_class_f1={
            label: report[label]["f1-score"] for label in labels if label in report
        },
        per_class_support={
            label: report[label]["support"] for label in labels if label in report
        },
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        labels=labels,
    )


def create_confusion_matrix_plot(
    result: BenchmarkResult, output_path: Path, title: str
) -> None:
    """Create and save confusion matrix visualization.

    Args:
        result: BenchmarkResult with confusion matrix
        output_path: Path to save PNG
        title: Plot title
    """
    cm = np.array(result.confusion_matrix)
    labels = result.labels

    # Normalize
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"{title}\nAcc: {result.accuracy:.3f}, F1: {result.macro_f1:.3f}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    del fig
    logger.info(f"Saved: {output_path}")


# ============================================================================
# MAIN BENCHMARK
# ============================================================================


def run_benchmark(
    datasets: list[str],
    granularities: list[Granularity],
    output_dir: Path,
    sample_size: int | None = None,
    seed: int = 42,
    use_clearml: bool = True,
) -> dict[str, dict[str, BenchmarkResult]]:
    """Run multi-granularity benchmark.

    Args:
        datasets: Dataset keys ("rep1", "rep2")
        granularities: Granularity levels to test
        output_dir: Output directory for artifacts
        sample_size: Optional sample size for faster testing
        seed: Random seed for reproducibility
        use_clearml: Whether to use ClearML tracking

    Returns:
        Nested dict: dataset_granularity -> method -> BenchmarkResult
    """
    task = None
    if use_clearml and CLEARML_AVAILABLE:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        task = Task.init(
            project_name="DAPIDL/Annotation-Benchmark",
            task_name=f"MultiGranularity_Benchmark_{timestamp}",
        )
        task.set_parameters(
            {
                "datasets": datasets,
                "granularities": [g.value for g in granularities],
                "sample_size": sample_size,
                "seed": seed,
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, BenchmarkResult]] = {}

    for dataset_key in datasets:
        config = DATASETS[dataset_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"DATASET: {config['name']}")
        logger.info(f"{'='*60}")

        adata = load_xenium_adata(config["xenium_path"], sample_size=sample_size, seed=seed)

        # Pre-normalize once for all CellTypist runs
        adata_norm = get_normalized_adata(adata)

        for granularity in granularities:
            logger.info(f"\n--- Granularity: {granularity.value.upper()} ---")

            ground_truth = load_ground_truth(config["gt_path"], granularity)

            # Filter to common cells using vectorized operation
            adata_cell_ids = np.array([str(c) for c in adata.obs_names])
            gt_cells = set(ground_truth["cell_id"].to_list())
            mask = np.isin(adata_cell_ids, list(gt_cells))
            adata_filtered = adata[mask].copy()
            adata_norm_filtered = adata_norm[mask].copy()
            gt_filtered = ground_truth.filter(
                pl.col("cell_id").is_in(adata_cell_ids[mask].tolist())
            )

            logger.info(f"Common cells: {mask.sum()}")

            gran_results: dict[str, BenchmarkResult] = {}
            gran_dir = output_dir / dataset_key / granularity.value
            gran_dir.mkdir(parents=True, exist_ok=True)

            # Define methods using functools.partial for safety
            from functools import partial

            methods = [
                (
                    "celltypist_breast",
                    partial(
                        run_celltypist,
                        adata_filtered,
                        adata_norm_filtered,
                        CELLTYPIST_MODELS["breast"],
                        granularity,
                    ),
                ),
                (
                    "celltypist_tissue",
                    partial(
                        run_celltypist,
                        adata_filtered,
                        adata_norm_filtered,
                        CELLTYPIST_MODELS["tissue"],
                        granularity,
                    ),
                ),
                (
                    "singler_hpca",
                    partial(run_singler, adata_filtered, "hpca", granularity),
                ),
                (
                    "singler_blueprint",
                    partial(run_singler, adata_filtered, "blueprint", granularity),
                ),
                (
                    "popv_minimal",
                    partial(
                        run_combined,
                        adata_filtered,
                        adata_norm_filtered,
                        CELLTYPIST_MODELS["tissue"],
                        ["hpca", "blueprint"],
                        granularity,
                    ),
                ),
                (
                    "popv_standard",
                    partial(
                        run_combined,
                        adata_filtered,
                        adata_norm_filtered,
                        CELLTYPIST_MODELS["universal"],
                        ["hpca", "blueprint"],
                        granularity,
                    ),
                ),
            ]

            for method_name, method_fn in methods:
                logger.info(f"\nMethod: {method_name}")
                try:
                    start = time.time()
                    predictions = method_fn()
                    runtime = time.time() - start

                    result = evaluate(predictions, gt_filtered, granularity)
                    result.runtime_seconds = runtime
                    gran_results[method_name] = result

                    logger.info(f"  Accuracy: {result.accuracy:.3f}")
                    logger.info(f"  Macro F1: {result.macro_f1:.3f}")
                    logger.info(f"  Classes: {result.n_classes}")

                    # Confusion matrix
                    cm_path = gran_dir / f"{method_name}_confusion.png"
                    create_confusion_matrix_plot(
                        result, cm_path, f"{method_name} - {granularity.value}"
                    )

                    if task:
                        task.upload_artifact(
                            f"{dataset_key}/{granularity.value}/{method_name}", cm_path
                        )

                except Exception as e:
                    logger.error(f"Method {method_name} failed: {e}")
                    import traceback

                    traceback.print_exc()
                    gran_results[method_name] = BenchmarkResult(
                        granularity=granularity.value,
                        n_evaluated=0,
                        n_classes=0,
                        accuracy=0,
                        macro_f1=0,
                        weighted_f1=0,
                        precision=0,
                        recall=0,
                        kappa=0,
                        error=str(e),
                    )

            result_key = f"{dataset_key}_{granularity.value}"
            all_results[result_key] = gran_results

    # Save summary
    summary_path = output_dir / "multi_granularity_summary.json"
    serializable_results = {
        key: {method: result.to_dict() for method, result in methods.items()}
        for key, methods in all_results.items()
    }
    with open(summary_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"\nSummary saved: {summary_path}")
    print_summary(all_results)

    if task:
        task.upload_artifact("summary", summary_path)
        task.close()

    return all_results


def print_summary(results: dict[str, dict[str, BenchmarkResult]]) -> None:
    """Print summary table of results."""
    print("\n" + "=" * 120)
    print("MULTI-GRANULARITY BENCHMARK SUMMARY")
    print("=" * 120)

    for result_key, methods in sorted(results.items()):
        print(f"\n{result_key}:")
        print("-" * 100)

        print(
            f"{'Method':<20} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Kappa':>8} {'Classes':>8}"
        )
        print("-" * 100)

        for method, result in sorted(
            methods.items(), key=lambda x: -x[1].macro_f1
        ):
            if result.error:
                print(f"{method:<20} ERROR: {result.error}")
                continue
            print(
                f"{method:<20} "
                f"{result.accuracy:>8.3f} "
                f"{result.macro_f1:>8.3f} "
                f"{result.precision:>8.3f} "
                f"{result.recall:>8.3f} "
                f"{result.kappa:>8.3f} "
                f"{result.n_classes:>8}"
            )

        # Per-class breakdown for best method
        valid_methods = [(m, r) for m, r in methods.items() if not r.error]
        if valid_methods:
            best_name, best = max(valid_methods, key=lambda x: x[1].macro_f1)
            print(f"\nBest method: {best_name} (F1: {best.macro_f1:.3f})")
            print("Per-class F1:")
            for cls, f1 in sorted(best.per_class_f1.items(), key=lambda x: -x[1]):
                support = best.per_class_support.get(cls, 0)
                print(f"  {cls:<25} F1: {f1:.3f}  (n={support})")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-granularity annotation benchmark")
    parser.add_argument("--datasets", "-d", nargs="+", default=["rep1", "rep2"])
    parser.add_argument(
        "--granularity",
        "-g",
        nargs="+",
        choices=["coarse", "medium", "fine"],
        default=["coarse", "medium", "fine"],
    )
    parser.add_argument("--sample-size", "-s", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=Path("benchmark_multi_granularity_v2")
    )
    parser.add_argument("--no-clearml", action="store_true")

    args = parser.parse_args()

    granularities = [Granularity(g) for g in args.granularity]

    run_benchmark(
        datasets=args.datasets,
        granularities=granularities,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        use_clearml=not args.no_clearml,
    )


if __name__ == "__main__":
    main()
