#!/usr/bin/env python3
"""
Benchmark Cell Type Annotation Methods with ClearML Tracking.

Compares SingleR, CellTypist, scType, and ensemble combinations
on spatial transcriptomics datasets with ground truth.

Usage:
    # Single method
    python benchmark_annotation_methods.py --methods celltypist --dataset rep1

    # Multiple methods
    python benchmark_annotation_methods.py --methods singler celltypist sctype --dataset rep1

    # All ensemble combinations
    python benchmark_annotation_methods.py --all-combinations --dataset rep1 rep2
"""

import sys
from pathlib import Path
from itertools import combinations
from typing import Literal

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
from collections import defaultdict
import subprocess
import tempfile

# Try to import ClearML
try:
    from clearml import Task, Logger
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    logger.warning("ClearML not available - results won't be logged")


# Dataset configurations
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

# Ground truth label mapping
GT_TO_BROAD = {
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
    "Endothelial": "Endothelial",
}


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
    """Load and process ground truth annotations."""
    logger.info(f"Loading ground truth from {gt_path}")
    df = pd.read_excel(gt_path)
    gt_df = pl.from_pandas(df)

    # Standardize column names
    gt_df = gt_df.rename({"Barcode": "cell_id", "Cluster": "gt_type"})
    gt_df = gt_df.with_columns(pl.col("cell_id").cast(pl.Utf8))

    # Map to broad categories
    gt_df = gt_df.with_columns(
        pl.col("gt_type").map_elements(
            lambda x: GT_TO_BROAD.get(x, "Unknown"),
            return_dtype=pl.Utf8
        ).alias("gt_broad")
    )

    return gt_df


def run_singler(adata: sc.AnnData, output_dir: Path) -> pl.DataFrame | None:
    """Run SingleR annotation via R script."""
    try:
        # Create R script - SingleR expects genes x cells matrix
        r_script = f'''
library(SingleR)
library(celldex)
library(Matrix)

# Load expression from CSV (genes x cells format)
expr <- as.matrix(read.csv("{output_dir}/expr_matrix.csv", row.names=1, check.names=FALSE))
cat("Expression matrix dimensions:", dim(expr), "\\n")

# Load reference
ref <- BlueprintEncodeData()
cat("Reference genes:", length(rownames(ref)), "\\n")

# Find common genes
common_genes <- intersect(rownames(expr), rownames(ref))
cat("Common genes:", length(common_genes), "\\n")

if (length(common_genes) < 50) {{
    stop(paste("Too few common genes:", length(common_genes)))
}}

# Subset to common genes
expr_sub <- expr[common_genes, , drop=FALSE]
ref_sub <- ref[common_genes, , drop=FALSE]

# Run SingleR
results <- SingleR(test = expr_sub, ref = ref_sub, labels = ref_sub$label.main)

# Save results
write.csv(data.frame(
    cell_id = colnames(expr_sub),
    singler_label = results$labels,
    singler_pruned = results$pruned.labels
), "{output_dir}/singler_results.csv", row.names = FALSE)
'''
        # Save expression matrix for R (genes x cells, transposed)
        expr_df = pd.DataFrame(
            adata.X.toarray().T if hasattr(adata.X, 'toarray') else adata.X.T,
            index=adata.var_names,  # genes as rows
            columns=adata.obs_names  # cells as columns
        )
        expr_df.to_csv(output_dir / "expr_matrix.csv")
        logger.info(f"Saved expression matrix: {expr_df.shape[0]} genes x {expr_df.shape[1]} cells")

        # Save and run R script
        script_path = output_dir / "run_singler.R"
        with open(script_path, 'w') as f:
            f.write(r_script)

        logger.info("Running SingleR...")
        result = subprocess.run(
            ["Rscript", str(script_path)],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(output_dir)
        )

        if result.returncode == 0:
            df = pl.read_csv(output_dir / "singler_results.csv")
            # Map to broad categories
            singler_broad_map = {
                "Epithelial cells": "Epithelial",
                "Keratinocytes": "Epithelial",
                "B-cells": "Immune",
                "CD4+ T-cells": "Immune",
                "CD8+ T-cells": "Immune",
                "NK cells": "Immune",
                "Macrophages": "Immune",
                "Monocytes": "Immune",
                "DC": "Immune",
                "Neutrophils": "Immune",
                "Eosinophils": "Immune",
                "Fibroblasts": "Stromal",
                "Adipocytes": "Stromal",
                "Smooth muscle": "Stromal",
                "Endothelial cells": "Endothelial",
            }
            df = df.with_columns(
                pl.col("singler_label").map_elements(
                    lambda x: singler_broad_map.get(x, "Unknown"),
                    return_dtype=pl.Utf8
                ).alias("pred_broad")
            )
            df = df.with_columns(pl.col("cell_id").cast(pl.Utf8))
            return df
        else:
            logger.error(f"SingleR failed: {result.stderr}")
            return None

    except Exception as e:
        logger.error(f"SingleR error: {e}")
        return None


def run_celltypist(adata: sc.AnnData, models: list[str] = None) -> pl.DataFrame:
    """Run CellTypist with multiple models."""
    import celltypist
    from celltypist import models as ct_models

    if models is None:
        # Default models for breast tissue
        models = [
            "Cells_Adult_Breast.pkl",
            "Immune_All_High.pkl",
            "Adult_Human_Skin.pkl",
            "Healthy_Human_Liver.pkl",
        ]

    # Normalize data
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)

    all_preds = []
    all_confs = []

    for model_name in models:
        try:
            ct_models.download_models(model=[model_name])
            model = ct_models.Model.load(model=model_name)

            predictions = celltypist.annotate(
                adata_norm,
                model=model,
                majority_voting=False,
            )

            preds = predictions.predicted_labels.predicted_labels.tolist()
            confs = predictions.probability_matrix.max(axis=1).tolist()

            all_preds.append(preds)
            all_confs.append(confs)

        except Exception as e:
            logger.warning(f"CellTypist {model_name} failed: {e}")
            continue

    # Ensemble: confidence-weighted voting
    cell_ids = list(adata.obs_names)
    n_cells = len(cell_ids)

    broad_map = {
        "Epithelial cell": "Epithelial",
        "Luminal epithelial cell": "Epithelial",
        "Basal cell": "Epithelial",
        "T cell": "Immune",
        "B cell": "Immune",
        "Macrophage": "Immune",
        "Monocyte": "Immune",
        "NK cell": "Immune",
        "Dendritic cell": "Immune",
        "Mast cell": "Immune",
        "Fibroblast": "Stromal",
        "Smooth muscle cell": "Stromal",
        "Adipocyte": "Stromal",
        "Pericyte": "Stromal",
        "Endothelial cell": "Endothelial",
        "Vascular endothelial cell": "Endothelial",
    }

    def map_to_broad(label):
        if label in broad_map:
            return broad_map[label]
        label_lower = label.lower()
        if any(x in label_lower for x in ["epithelial", "keratinocyte", "luminal", "basal"]):
            return "Epithelial"
        if any(x in label_lower for x in ["t cell", "b cell", "macrophage", "monocyte", "dendritic", "nk cell", "mast", "immune"]):
            return "Immune"
        if any(x in label_lower for x in ["fibroblast", "stromal", "smooth muscle", "pericyte", "adipocyte"]):
            return "Stromal"
        if any(x in label_lower for x in ["endothelial", "vascular"]):
            return "Endothelial"
        return "Unknown"

    final_preds = []
    final_confs = []

    for cell_idx in range(n_cells):
        votes = defaultdict(float)
        for model_idx in range(len(all_preds)):
            pred = all_preds[model_idx][cell_idx]
            conf = all_confs[model_idx][cell_idx]
            broad = map_to_broad(pred)
            if broad != "Unknown":
                votes[broad] += conf

        if votes:
            best = max(votes, key=votes.get)
            final_preds.append(best)
            final_confs.append(votes[best] / len(all_preds))
        else:
            final_preds.append("Unknown")
            final_confs.append(0.0)

    return pl.DataFrame({
        "cell_id": [str(c) for c in cell_ids],
        "pred_broad": final_preds,
        "confidence": final_confs,
    })


def run_sctype(adata: sc.AnnData) -> pl.DataFrame:
    """Run scType marker-based annotation."""
    from dapidl.pipeline.components.annotators.sctype import ScTypeAnnotator

    annotator = ScTypeAnnotator()
    result = annotator.annotate(adata=adata)

    return result.annotations_df.select([
        pl.col("cell_id"),
        pl.col("broad_category").alias("pred_broad"),
        pl.col("confidence"),
    ])


def run_ensemble(
    predictions: dict[str, pl.DataFrame],
    strategy: Literal["majority", "confidence_weighted"] = "confidence_weighted",
) -> pl.DataFrame:
    """Combine predictions from multiple methods.

    Args:
        predictions: Dict of method_name -> DataFrame with cell_id, pred_broad, confidence
        strategy: How to combine predictions

    Returns:
        DataFrame with ensemble predictions
    """
    methods = list(predictions.keys())
    if not methods:
        raise ValueError("No predictions to ensemble")

    # Start with first method's cell_ids
    base_df = predictions[methods[0]].select(["cell_id"])

    # Join all predictions
    for method, df in predictions.items():
        base_df = base_df.join(
            df.select([
                pl.col("cell_id"),
                pl.col("pred_broad").alias(f"{method}_pred"),
                pl.col("confidence").alias(f"{method}_conf") if "confidence" in df.columns else pl.lit(1.0).alias(f"{method}_conf"),
            ]),
            on="cell_id",
            how="left"
        )

    # Ensemble prediction
    pred_cols = [f"{m}_pred" for m in methods]
    conf_cols = [f"{m}_conf" for m in methods]

    if strategy == "majority":
        # Simple majority vote
        def majority_vote(row):
            votes = [row[c] for c in pred_cols if row[c] and row[c] != "Unknown"]
            if not votes:
                return "Unknown"
            from collections import Counter
            return Counter(votes).most_common(1)[0][0]

        base_df = base_df.with_columns(
            pl.struct(pred_cols).map_elements(
                majority_vote,
                return_dtype=pl.Utf8
            ).alias("pred_broad")
        )

    else:  # confidence_weighted
        def weighted_vote(row):
            votes = defaultdict(float)
            for pred_col, conf_col in zip(pred_cols, conf_cols):
                pred = row[pred_col]
                conf = row[conf_col] if row[conf_col] else 0.5
                if pred and pred != "Unknown":
                    votes[pred] += conf
            if not votes:
                return "Unknown"
            return max(votes, key=votes.get)

        base_df = base_df.with_columns(
            pl.struct(pred_cols + conf_cols).map_elements(
                weighted_vote,
                return_dtype=pl.Utf8
            ).alias("pred_broad")
        )

    return base_df.select(["cell_id", "pred_broad"])


def calculate_metrics(y_true: list, y_pred: list, labels: list = None) -> dict:
    """Calculate comprehensive classification metrics."""
    if labels is None:
        labels = sorted(set(y_true + y_pred) - {"Unknown"})

    # Filter out Unknown
    valid = [(t, p) for t, p in zip(y_true, y_pred) if t != "Unknown" and p != "Unknown"]
    if not valid:
        return {"error": "No valid predictions"}

    y_true_valid, y_pred_valid = zip(*valid)

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

    # Per-class metrics
    for label in labels:
        label_true = [1 if t == label else 0 for t in y_true_valid]
        label_pred = [1 if p == label else 0 for p in y_pred_valid]

        if sum(label_true) > 0:
            metrics[f"{label}_f1"] = f1_score(label_true, label_pred, zero_division=0)
            metrics[f"{label}_precision"] = precision_score(label_true, label_pred, zero_division=0)
            metrics[f"{label}_recall"] = recall_score(label_true, label_pred, zero_division=0)
            metrics[f"{label}_support"] = sum(label_true)

    # Confusion matrix
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_labels"] = labels

    return metrics


def log_to_clearml(
    task: Task,
    method_name: str,
    dataset_name: str,
    metrics: dict,
):
    """Log metrics to ClearML."""
    clearml_logger = task.get_logger()

    # Scalar metrics
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            clearml_logger.report_scalar(
                title=f"{dataset_name}/{method_name}",
                series=key,
                value=value,
                iteration=0
            )

    # Confusion matrix
    if "confusion_matrix" in metrics:
        cm = np.array(metrics["confusion_matrix"])
        labels = metrics["confusion_labels"]

        clearml_logger.report_confusion_matrix(
            title=f"Confusion Matrix - {method_name}",
            series=dataset_name,
            matrix=cm,
            xlabels=labels,
            ylabels=labels,
            iteration=0
        )


def run_benchmark(
    methods: list[str],
    datasets: list[str],
    project_name: str = "DAPIDL/Annotation-Benchmark",
    task_name: str = None,
) -> dict:
    """Run benchmark on specified methods and datasets.

    Args:
        methods: List of methods to run (singler, celltypist, sctype)
        datasets: List of dataset keys (rep1, rep2)
        project_name: ClearML project name
        task_name: ClearML task name

    Returns:
        Dictionary of results
    """
    # Initialize ClearML
    task = None
    if CLEARML_AVAILABLE:
        method_str = "+".join(sorted(methods))
        task = Task.init(
            project_name=project_name,
            task_name=task_name or f"Benchmark_{method_str}",
            task_type=Task.TaskTypes.testing,
        )
        task.set_parameters({
            "methods": methods,
            "datasets": datasets,
        })

    all_results = {}

    for dataset_key in datasets:
        if dataset_key not in DATASETS:
            logger.warning(f"Unknown dataset: {dataset_key}")
            continue

        dataset_config = DATASETS[dataset_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {dataset_config['name']}")
        logger.info(f"{'='*60}")

        # Load data
        adata = load_xenium_data(dataset_config["xenium_path"])
        gt_df = load_ground_truth(dataset_config["gt_path"])

        # Create temp directory for intermediate files
        temp_dir = Path(tempfile.mkdtemp())

        # Run each method
        predictions = {}

        if "singler" in methods:
            logger.info("\n--- Running SingleR ---")
            singler_df = run_singler(adata, temp_dir)
            if singler_df is not None:
                predictions["singler"] = singler_df.select([
                    pl.col("cell_id"),
                    pl.col("pred_broad"),
                    pl.lit(0.8).alias("confidence"),  # SingleR doesn't provide confidence
                ])

        if "celltypist" in methods:
            logger.info("\n--- Running CellTypist ---")
            celltypist_df = run_celltypist(adata)
            predictions["celltypist"] = celltypist_df

        if "sctype" in methods:
            logger.info("\n--- Running scType ---")
            sctype_df = run_sctype(adata)
            predictions["sctype"] = sctype_df

        # Evaluate individual methods
        results = {}

        for method_name, pred_df in predictions.items():
            logger.info(f"\n--- Evaluating {method_name} ---")

            # Join with ground truth
            eval_df = pred_df.join(gt_df, on="cell_id", how="inner")
            eval_df = eval_df.filter(
                (pl.col("gt_broad") != "Unknown") &
                (pl.col("pred_broad") != "Unknown")
            )

            y_true = eval_df["gt_broad"].to_list()
            y_pred = eval_df["pred_broad"].to_list()

            metrics = calculate_metrics(y_true, y_pred)
            results[method_name] = metrics

            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
            logger.info(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}")

            if task:
                log_to_clearml(task, method_name, dataset_key, metrics)

        # Evaluate ensemble if multiple methods
        if len(predictions) > 1:
            logger.info("\n--- Evaluating Ensemble ---")
            ensemble_df = run_ensemble(predictions)

            eval_df = ensemble_df.join(gt_df, on="cell_id", how="inner")
            eval_df = eval_df.filter(
                (pl.col("gt_broad") != "Unknown") &
                (pl.col("pred_broad") != "Unknown")
            )

            y_true = eval_df["gt_broad"].to_list()
            y_pred = eval_df["pred_broad"].to_list()

            metrics = calculate_metrics(y_true, y_pred)
            ensemble_name = "ensemble_" + "+".join(sorted(predictions.keys()))
            results[ensemble_name] = metrics

            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")

            if task:
                log_to_clearml(task, ensemble_name, dataset_key, metrics)

        all_results[dataset_key] = results

    # Summary
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*60)

    summary_rows = []
    for dataset_key, results in all_results.items():
        for method, metrics in results.items():
            if "error" not in metrics:
                summary_rows.append({
                    "dataset": dataset_key,
                    "method": method,
                    "accuracy": metrics.get("accuracy", 0),
                    "macro_f1": metrics.get("macro_f1", 0),
                    "cohen_kappa": metrics.get("cohen_kappa", 0),
                    "n_cells": metrics.get("n_cells", 0),
                })

    if summary_rows:
        summary_df = pl.DataFrame(summary_rows)
        print(summary_df.sort(["dataset", "macro_f1"], descending=[False, True]))
    else:
        logger.warning("No successful method runs to summarize")
        summary_df = pl.DataFrame()

    if task:
        task.upload_artifact("summary", summary_df.to_pandas())
        task.close()

    return all_results


def run_all_combinations(datasets: list[str], project_name: str = "DAPIDL/Annotation-Benchmark"):
    """Run all possible method combinations."""
    base_methods = ["singler", "celltypist", "sctype"]

    # Generate all combinations (1, 2, 3 methods)
    all_combinations = []
    for r in range(1, len(base_methods) + 1):
        for combo in combinations(base_methods, r):
            all_combinations.append(list(combo))

    logger.info(f"Running {len(all_combinations)} method combinations:")
    for combo in all_combinations:
        logger.info(f"  - {'+'.join(combo)}")

    all_results = {}

    for combo in all_combinations:
        combo_name = "+".join(combo)
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Running combination: {combo_name}")
        logger.info(f"{'#'*60}")

        results = run_benchmark(
            methods=combo,
            datasets=datasets,
            project_name=project_name,
            task_name=f"Benchmark_{combo_name}",
        )
        all_results[combo_name] = results

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark cell type annotation methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        choices=["singler", "celltypist", "sctype"],
        default=["celltypist"],
        help="Methods to run"
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=["rep1"],
        help="Datasets to evaluate"
    )
    parser.add_argument(
        "--all-combinations",
        action="store_true",
        help="Run all possible method combinations"
    )
    parser.add_argument(
        "--project",
        default="DAPIDL/Annotation-Benchmark",
        help="ClearML project name"
    )

    args = parser.parse_args()

    if args.all_combinations:
        run_all_combinations(args.datasets, args.project)
    else:
        run_benchmark(args.methods, args.datasets, args.project)


if __name__ == "__main__":
    main()
