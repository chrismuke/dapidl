#!/usr/bin/env python3
"""
Comprehensive Fine-Grained Cell Type Annotation Benchmark.

Tests ALL combinations of annotation methods on fine-grained classification.
With N methods, tests 2^N - 1 combinations (from single methods to all combined).

Usage:
    # Run all combinations with 4 core methods
    python benchmark_finegrained_all_combinations.py --methods singler celltypist sctype scina

    # Run specific combination sizes only
    python benchmark_finegrained_all_combinations.py --methods singler celltypist sctype scina --combo-sizes 1 2 3

    # Run all 7 methods (127 combinations)
    python benchmark_finegrained_all_combinations.py --all-methods
"""

import sys
from pathlib import Path
from itertools import combinations
from typing import Literal
from collections import defaultdict
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
import pandas as pd
import numpy as np
import scanpy as sc
import h5py
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
)
import subprocess
import tempfile

# Try to import ClearML
try:
    from clearml import Task, Logger
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    logger.warning("ClearML not available - results won't be logged")


# =============================================================================
# Dataset and Ground Truth Configuration
# =============================================================================

DATASETS = {
    "rep1": {
        "xenium_path": Path("/home/chrism/datasets/raw/xenium/breast_tumor_rep1/outs"),
        "gt_path": Path("/home/chrism/.clearml/cache/storage_manager/datasets/ds_ee53f76b00c0400d83a303408b5ea8e2/celltypes_ground_truth_rep1_supervised.xlsx"),
        "name": "Xenium Breast Rep1",
    },
    "rep2": {
        "xenium_path": Path("/home/chrism/datasets/raw/xenium/breast_tumor_rep2/outs"),
        "gt_path": Path("/home/chrism/.clearml/cache/storage_manager/datasets/ds_d73638c8e809416a8b94697d20b2d971/celltypes_ground_truth_rep2_supervised.xlsx"),
        "name": "Xenium Breast Rep2",
    },
}

# Fine-grained ground truth labels (20 types)
GT_FINEGRAINED_TYPES = [
    "Stromal", "Invasive_Tumor", "DCIS_1", "DCIS_2", "Macrophages_1",
    "Endothelial", "CD4+_T_Cells", "Myoepi_ACTA2+", "CD8+_T_Cells", "B_Cells",
    "Prolif_Invasive_Tumor", "Myoepi_KRT15+", "Macrophages_2", "Perivascular-Like",
    "Stromal_&_T_Cell_Hybrid", "T_Cell_&_Tumor_Hybrid", "IRF7+_DCs", "LAMP3+_DCs",
    "Mast_Cells", "Unlabeled"
]

# Mapping from various annotator outputs to ground truth fine-grained types
FINEGRAINED_MAPPINGS = {
    # CellTypist mappings
    "Luminal epithelial cell of mammary gland": "Invasive_Tumor",
    "Basal cell": "Myoepi_KRT15+",
    "T cell": "CD4+_T_Cells",
    "CD4-positive, alpha-beta T cell": "CD4+_T_Cells",
    "CD8-positive, alpha-beta T cell": "CD8+_T_Cells",
    "B cell": "B_Cells",
    "Macrophage": "Macrophages_1",
    "Classical monocyte": "Macrophages_1",
    "Non-classical monocyte": "Macrophages_2",
    "Dendritic cell": "IRF7+_DCs",
    "Mast cell": "Mast_Cells",
    "Fibroblast": "Stromal",
    "Endothelial cell": "Endothelial",
    "Smooth muscle cell": "Stromal",
    "Pericyte": "Perivascular-Like",
    "Adipocyte": "Stromal",
    "NK cell": "CD8+_T_Cells",  # Often confused
    "Plasma cell": "B_Cells",

    # SingleR mappings
    "Epithelial cells": "Invasive_Tumor",
    "Keratinocytes": "Myoepi_KRT15+",
    "CD4+ T-cells": "CD4+_T_Cells",
    "CD8+ T-cells": "CD8+_T_Cells",
    "B-cells": "B_Cells",
    "Macrophages": "Macrophages_1",
    "Monocytes": "Macrophages_1",
    "DC": "IRF7+_DCs",
    "NK cells": "CD8+_T_Cells",
    "Fibroblasts": "Stromal",
    "Smooth muscle": "Stromal",
    "Endothelial cells": "Endothelial",
    "Adipocytes": "Stromal",
    "Neutrophils": "Macrophages_1",
    "Eosinophils": "Macrophages_1",

    # scType mappings
    "Epithelial": "Invasive_Tumor",
    "T cells": "CD4+_T_Cells",
    "CD4+ T cells": "CD4+_T_Cells",
    "CD8+ T cells": "CD8+_T_Cells",
    "Regulatory T cells": "CD4+_T_Cells",
    "B cells": "B_Cells",
    "Plasma cells": "B_Cells",
    "Macrophages": "Macrophages_1",
    "Monocytes": "Macrophages_1",
    "Dendritic cells": "IRF7+_DCs",
    "Mast cells": "Mast_Cells",
    "NK cells": "CD8+_T_Cells",
    "Fibroblasts": "Stromal",
    "Myofibroblasts": "Myoepi_ACTA2+",
    "Pericytes": "Perivascular-Like",
    "Adipocytes": "Stromal",
    "Endothelial cells": "Endothelial",
    "Lymphatic endothelial": "Endothelial",

    # SCINA mappings (similar to scType)
    "T_cells": "CD4+_T_Cells",
    "CD4_T_cells": "CD4+_T_Cells",
    "CD8_T_cells": "CD8+_T_Cells",
    "Tregs": "CD4+_T_Cells",
    "B_cells": "B_Cells",
    "Plasma_cells": "B_Cells",
    "Dendritic_cells": "IRF7+_DCs",
    "Mast_cells": "Mast_Cells",
    "NK_cells": "CD8+_T_Cells",
    "Myofibroblasts": "Myoepi_ACTA2+",
    "Lymphatic_endothelial": "Endothelial",
}


def map_to_finegrained(label: str) -> str:
    """Map annotator output to ground truth fine-grained type."""
    if label in GT_FINEGRAINED_TYPES:
        return label
    if label in FINEGRAINED_MAPPINGS:
        return FINEGRAINED_MAPPINGS[label]

    # Fuzzy matching
    label_lower = label.lower()

    # Epithelial subtypes
    if any(x in label_lower for x in ["epithelial", "luminal", "tumor", "dcis"]):
        return "Invasive_Tumor"
    if any(x in label_lower for x in ["myoepithelial", "basal"]):
        return "Myoepi_KRT15+"

    # Immune subtypes
    if "cd4" in label_lower or ("t cell" in label_lower and "cd8" not in label_lower):
        return "CD4+_T_Cells"
    if "cd8" in label_lower:
        return "CD8+_T_Cells"
    if any(x in label_lower for x in ["b cell", "b-cell"]):
        return "B_Cells"
    if any(x in label_lower for x in ["macrophage", "monocyte"]):
        return "Macrophages_1"
    if any(x in label_lower for x in ["dendritic", "dc"]):
        return "IRF7+_DCs"
    if "mast" in label_lower:
        return "Mast_Cells"
    if "nk" in label_lower or "natural killer" in label_lower:
        return "CD8+_T_Cells"
    if "plasma" in label_lower:
        return "B_Cells"

    # Stromal
    if any(x in label_lower for x in ["fibroblast", "stromal", "smooth muscle", "adipocyte"]):
        return "Stromal"
    if "pericyte" in label_lower or "perivascular" in label_lower:
        return "Perivascular-Like"

    # Endothelial
    if "endothelial" in label_lower:
        return "Endothelial"

    return "Unlabeled"


# =============================================================================
# Data Loading
# =============================================================================

def load_xenium_data(xenium_path: Path) -> sc.AnnData:
    """Load Xenium expression data."""
    h5_path = xenium_path / "cell_feature_matrix.h5"
    logger.info(f"Loading Xenium data from {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        data = f['matrix/data'][:]
        indices = f['matrix/indices'][:]
        indptr = f['matrix/indptr'][:]
        shape = f['matrix/shape'][:]
        barcodes = [b.decode() for b in f['matrix/barcodes'][:]]
        genes = [g.decode() for g in f['matrix/features/name'][:]]

    from scipy.sparse import csc_matrix
    X = csc_matrix((data, indices, indptr), shape=shape).T
    adata = sc.AnnData(X=X)
    adata.obs_names = barcodes
    adata.var_names = genes
    adata.obs["cell_id"] = barcodes

    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def load_ground_truth(gt_path: Path) -> pl.DataFrame:
    """Load fine-grained ground truth annotations."""
    logger.info(f"Loading ground truth from {gt_path}")
    df = pd.read_excel(gt_path)
    gt_df = pl.from_pandas(df)

    gt_df = gt_df.rename({"Barcode": "cell_id", "Cluster": "gt_type"})
    gt_df = gt_df.with_columns(pl.col("cell_id").cast(pl.Utf8))

    # Filter out Unlabeled for evaluation
    gt_df = gt_df.filter(pl.col("gt_type") != "Unlabeled")

    return gt_df


# =============================================================================
# Individual Method Runners
# =============================================================================

def run_singler(adata: sc.AnnData, output_dir: Path) -> pl.DataFrame | None:
    """Run SingleR annotation."""
    try:
        r_script = f'''
library(SingleR)
library(celldex)
library(Matrix)

expr <- as.matrix(read.csv("{output_dir}/expr_matrix.csv", row.names=1, check.names=FALSE))
ref <- BlueprintEncodeData()
common_genes <- intersect(rownames(expr), rownames(ref))

if (length(common_genes) < 50) {{
    stop("Too few common genes")
}}

expr_sub <- expr[common_genes, , drop=FALSE]
ref_sub <- ref[common_genes, , drop=FALSE]
results <- SingleR(test = expr_sub, ref = ref_sub, labels = ref_sub$label.main)

write.csv(data.frame(
    cell_id = colnames(expr_sub),
    predicted_type = results$labels,
    confidence = apply(results$scores, 1, max)
), "{output_dir}/singler_results.csv", row.names = FALSE)
'''
        expr_df = pd.DataFrame(
            adata.X.toarray().T if hasattr(adata.X, 'toarray') else adata.X.T,
            index=adata.var_names,
            columns=adata.obs_names
        )
        expr_df.to_csv(output_dir / "expr_matrix.csv")

        script_path = output_dir / "run_singler.R"
        with open(script_path, 'w') as f:
            f.write(r_script)

        logger.info("Running SingleR...")
        result = subprocess.run(
            ["Rscript", str(script_path)],
            capture_output=True, text=True, timeout=600, cwd=str(output_dir)
        )

        if result.returncode == 0:
            df = pl.read_csv(output_dir / "singler_results.csv")
            df = df.with_columns([
                pl.col("cell_id").cast(pl.Utf8),
                pl.col("predicted_type").map_elements(map_to_finegrained, return_dtype=pl.Utf8).alias("pred_finegrained"),
            ])
            return df.select(["cell_id", "predicted_type", "pred_finegrained", "confidence"])
        else:
            logger.error(f"SingleR failed: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"SingleR error: {e}")
        return None


def run_celltypist(adata: sc.AnnData) -> pl.DataFrame:
    """Run CellTypist with multiple models."""
    import celltypist
    from celltypist import models as ct_models

    models = ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl", "Healthy_Human_Liver.pkl"]

    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)

    all_preds = []
    all_confs = []

    for model_name in models:
        try:
            ct_models.download_models(model=[model_name])
            model = ct_models.Model.load(model=model_name)
            predictions = celltypist.annotate(adata_norm, model=model, majority_voting=False)
            all_preds.append(predictions.predicted_labels.predicted_labels.tolist())
            all_confs.append(predictions.probability_matrix.max(axis=1).tolist())
        except Exception as e:
            logger.warning(f"CellTypist {model_name} failed: {e}")

    # Confidence-weighted voting
    cell_ids = list(adata.obs_names)
    final_preds = []
    final_confs = []

    for i in range(len(cell_ids)):
        votes = defaultdict(float)
        for j in range(len(all_preds)):
            pred = all_preds[j][i]
            conf = all_confs[j][i]
            fg = map_to_finegrained(pred)
            if fg != "Unlabeled":
                votes[fg] += conf

        if votes:
            best = max(votes, key=votes.get)
            final_preds.append(best)
            final_confs.append(votes[best] / len(all_preds))
        else:
            final_preds.append("Unlabeled")
            final_confs.append(0.0)

    return pl.DataFrame({
        "cell_id": [str(c) for c in cell_ids],
        "predicted_type": final_preds,
        "pred_finegrained": final_preds,
        "confidence": final_confs,
    })


def run_sctype(adata: sc.AnnData) -> pl.DataFrame:
    """Run scType annotation."""
    from dapidl.pipeline.components.annotators.sctype import ScTypeAnnotator

    annotator = ScTypeAnnotator()
    result = annotator.annotate(adata=adata)

    df = result.annotations_df.with_columns(
        pl.col("predicted_type").map_elements(map_to_finegrained, return_dtype=pl.Utf8).alias("pred_finegrained")
    )

    return df.select([
        pl.col("cell_id"),
        pl.col("predicted_type"),
        pl.col("pred_finegrained"),
        pl.col("confidence"),
    ])


def run_scina(adata: sc.AnnData) -> pl.DataFrame:
    """Run SCINA annotation."""
    from dapidl.pipeline.components.annotators.scina import SCINAAnnotator

    annotator = SCINAAnnotator()
    result = annotator.annotate(adata=adata)

    df = result.annotations_df.with_columns(
        pl.col("predicted_type").map_elements(map_to_finegrained, return_dtype=pl.Utf8).alias("pred_finegrained")
    )

    return df.select([
        pl.col("cell_id"),
        pl.col("predicted_type"),
        pl.col("pred_finegrained"),
        pl.col("confidence"),
    ])


def run_popv(adata: sc.AnnData) -> pl.DataFrame | None:
    """Run PopV annotation."""
    try:
        from dapidl.pipeline.components.annotators.popv import PopVAnnotator

        annotator = PopVAnnotator()
        result = annotator.annotate(adata=adata)

        if result.annotations_df.height == 0:
            return None

        df = result.annotations_df.with_columns(
            pl.col("predicted_type").map_elements(map_to_finegrained, return_dtype=pl.Utf8).alias("pred_finegrained")
        )

        return df.select([
            pl.col("cell_id"),
            pl.col("predicted_type"),
            pl.col("pred_finegrained"),
            pl.col("confidence") if "confidence" in df.columns else pl.lit(0.7).alias("confidence"),
        ])
    except Exception as e:
        logger.error(f"PopV error: {e}")
        return None


METHOD_RUNNERS = {
    "singler": lambda adata, temp_dir: run_singler(adata, temp_dir),
    "celltypist": lambda adata, temp_dir: run_celltypist(adata),
    "sctype": lambda adata, temp_dir: run_sctype(adata),
    "scina": lambda adata, temp_dir: run_scina(adata),
    "popv": lambda adata, temp_dir: run_popv(adata),
}


# =============================================================================
# Ensemble and Metrics
# =============================================================================

def run_ensemble(predictions: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Combine predictions using confidence-weighted voting."""
    methods = list(predictions.keys())
    if not methods:
        raise ValueError("No predictions to ensemble")

    base_df = predictions[methods[0]].select(["cell_id"])

    for method, df in predictions.items():
        conf_col = "confidence" if "confidence" in df.columns else None
        base_df = base_df.join(
            df.select([
                pl.col("cell_id"),
                pl.col("pred_finegrained").alias(f"{method}_pred"),
                pl.col("confidence").alias(f"{method}_conf") if conf_col else pl.lit(0.7).alias(f"{method}_conf"),
            ]),
            on="cell_id", how="left"
        )

    pred_cols = [f"{m}_pred" for m in methods]
    conf_cols = [f"{m}_conf" for m in methods]

    def weighted_vote(row):
        votes = defaultdict(float)
        for pred_col, conf_col in zip(pred_cols, conf_cols):
            pred = row[pred_col]
            conf = row[conf_col] if row[conf_col] else 0.5
            if pred and pred != "Unlabeled":
                votes[pred] += conf
        if not votes:
            return "Unlabeled"
        return max(votes, key=votes.get)

    base_df = base_df.with_columns(
        pl.struct(pred_cols + conf_cols).map_elements(weighted_vote, return_dtype=pl.Utf8).alias("pred_finegrained")
    )

    return base_df.select(["cell_id", "pred_finegrained"])


def calculate_metrics(y_true: list, y_pred: list) -> dict:
    """Calculate comprehensive metrics."""
    valid = [(t, p) for t, p in zip(y_true, y_pred) if t != "Unlabeled" and p != "Unlabeled"]
    if not valid:
        return {"error": "No valid predictions", "n_cells": 0}

    y_true_valid, y_pred_valid = zip(*valid)
    labels = sorted(set(y_true_valid) | set(y_pred_valid))

    metrics = {
        "n_cells": len(y_true_valid),
        "n_classes": len(set(y_true_valid)),
        "accuracy": accuracy_score(y_true_valid, y_pred_valid),
        "macro_f1": f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true_valid, y_pred_valid, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true_valid, y_pred_valid, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true_valid, y_pred_valid),
        "mcc": matthews_corrcoef(y_true_valid, y_pred_valid),
    }

    # Per-class F1
    for label in labels:
        label_true = [1 if t == label else 0 for t in y_true_valid]
        label_pred = [1 if p == label else 0 for p in y_pred_valid]
        if sum(label_true) > 0:
            metrics[f"{label}_f1"] = f1_score(label_true, label_pred, zero_division=0)
            metrics[f"{label}_support"] = sum(label_true)

    return metrics


def log_to_clearml(task: Task, combo_name: str, dataset_name: str, metrics: dict):
    """Log metrics to ClearML."""
    clearml_logger = task.get_logger()

    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            clearml_logger.report_scalar(
                title=f"{dataset_name}/metrics",
                series=f"{combo_name}/{key}",
                value=value,
                iteration=0
            )


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_combination(
    methods: list[str],
    adata: sc.AnnData,
    gt_df: pl.DataFrame,
    temp_dir: Path,
    dataset_key: str,
) -> dict:
    """Run a single combination of methods."""
    combo_name = "+".join(sorted(methods))
    logger.info(f"  Running: {combo_name}")

    predictions = {}

    for method in methods:
        if method not in METHOD_RUNNERS:
            logger.warning(f"    Unknown method: {method}")
            continue

        try:
            result = METHOD_RUNNERS[method](adata, temp_dir)
            if result is not None:
                predictions[method] = result
                logger.info(f"    ✓ {method}: {result.height} cells")
            else:
                logger.warning(f"    ✗ {method}: No results")
        except Exception as e:
            logger.error(f"    ✗ {method}: {e}")

    if not predictions:
        return {"error": "No successful methods", "combo": combo_name}

    # Get predictions (ensemble if multiple methods)
    if len(predictions) == 1:
        method_name = list(predictions.keys())[0]
        pred_df = predictions[method_name].select(["cell_id", "pred_finegrained"])
    else:
        pred_df = run_ensemble(predictions)

    # Join with ground truth
    eval_df = pred_df.join(gt_df, on="cell_id", how="inner")
    eval_df = eval_df.filter(
        (pl.col("gt_type") != "Unlabeled") &
        (pl.col("pred_finegrained") != "Unlabeled")
    )

    if eval_df.height == 0:
        return {"error": "No overlapping cells", "combo": combo_name}

    y_true = eval_df["gt_type"].to_list()
    y_pred = eval_df["pred_finegrained"].to_list()

    metrics = calculate_metrics(y_true, y_pred)
    metrics["combo"] = combo_name
    metrics["n_methods"] = len(methods)
    metrics["methods"] = methods

    return metrics


def run_all_combinations(
    available_methods: list[str],
    datasets: list[str],
    combo_sizes: list[int] | None = None,
    project_name: str = "DAPIDL/Finegrained-Benchmark",
) -> dict:
    """Run all method combinations."""

    # Generate combinations
    all_combinations = []
    max_size = len(available_methods)
    sizes = combo_sizes or list(range(1, max_size + 1))

    for size in sizes:
        if size > max_size:
            continue
        for combo in combinations(available_methods, size):
            all_combinations.append(list(combo))

    logger.info(f"Testing {len(all_combinations)} combinations across {sizes} sizes")

    # Initialize ClearML task
    task = None
    if CLEARML_AVAILABLE:
        task = Task.init(
            project_name=project_name,
            task_name=f"Finegrained_All_Combinations_{len(available_methods)}methods",
            task_type=Task.TaskTypes.testing,
        )
        task.set_parameters({
            "methods": available_methods,
            "datasets": datasets,
            "total_combinations": len(all_combinations),
        })

    all_results = []

    for dataset_key in datasets:
        if dataset_key not in DATASETS:
            continue

        dataset_config = DATASETS[dataset_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset_config['name']}")
        logger.info(f"{'='*60}")

        # Load data once
        adata = load_xenium_data(dataset_config["xenium_path"])
        gt_df = load_ground_truth(dataset_config["gt_path"])

        temp_dir = Path(tempfile.mkdtemp())

        for i, combo in enumerate(all_combinations):
            logger.info(f"\nCombination {i+1}/{len(all_combinations)}")

            start_time = time.time()
            metrics = run_combination(combo, adata, gt_df, temp_dir, dataset_key)
            elapsed = time.time() - start_time

            metrics["dataset"] = dataset_key
            metrics["elapsed_seconds"] = elapsed
            all_results.append(metrics)

            if "error" not in metrics:
                logger.info(f"    Accuracy: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}")

                if task:
                    log_to_clearml(task, metrics["combo"], dataset_key, metrics)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE BENCHMARK RESULTS")
    logger.info("="*80)

    # Filter successful results
    successful = [r for r in all_results if "error" not in r]

    if successful:
        summary_df = pl.DataFrame([
            {
                "dataset": r["dataset"],
                "combo": r["combo"],
                "n_methods": r["n_methods"],
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "weighted_f1": r["weighted_f1"],
                "cohen_kappa": r["cohen_kappa"],
                "n_cells": r["n_cells"],
            }
            for r in successful
        ])

        # Sort by macro_f1
        summary_df = summary_df.sort(["dataset", "macro_f1"], descending=[False, True])
        print("\nTop 20 combinations by Macro F1:")
        print(summary_df.head(20))

        # Best per combo size
        print("\n\nBest combination per size:")
        for size in sorted(set(r["n_methods"] for r in successful)):
            size_results = [r for r in successful if r["n_methods"] == size]
            best = max(size_results, key=lambda x: x["macro_f1"])
            print(f"  {size} method(s): {best['combo']} - F1={best['macro_f1']:.4f}, Acc={best['accuracy']:.4f}")

        if task:
            task.upload_artifact("summary", summary_df.to_pandas())
            task.upload_artifact("all_results", pd.DataFrame(successful))

    if task:
        task.close()

    return {"results": all_results, "successful": successful}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive fine-grained annotation benchmark")
    parser.add_argument("--methods", "-m", nargs="+", default=["singler", "celltypist", "sctype", "scina"],
                       help="Methods to test")
    parser.add_argument("--all-methods", action="store_true", help="Use all available methods")
    parser.add_argument("--datasets", "-d", nargs="+", default=["rep1"],
                       choices=list(DATASETS.keys()), help="Datasets to evaluate")
    parser.add_argument("--combo-sizes", nargs="+", type=int, help="Only test these combination sizes")
    parser.add_argument("--project", default="DAPIDL/Finegrained-Benchmark", help="ClearML project")

    args = parser.parse_args()

    if args.all_methods:
        methods = list(METHOD_RUNNERS.keys())
    else:
        methods = args.methods

    logger.info(f"Methods: {methods}")
    logger.info(f"Datasets: {args.datasets}")

    run_all_combinations(
        available_methods=methods,
        datasets=args.datasets,
        combo_sizes=args.combo_sizes,
        project_name=args.project,
    )


if __name__ == "__main__":
    main()
