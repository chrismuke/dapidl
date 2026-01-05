#!/usr/bin/env python3
"""
Cell Ontology-Standardized Annotation Benchmark.

Maps BOTH predictions AND ground truth to Cell Ontology standard names,
ensuring we compare apples to apples across different annotation methods.

This reduces the 20 fine-grained GT types to ~12 CL-standardized categories.
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
# Cell Ontology Standardization
# =============================================================================

# CL standard names (our canonical vocabulary)
CL_STANDARD_TYPES = [
    "Epithelial",           # CL:0000066 - all epithelial including tumor
    "CD4+ T cell",          # CL:0000624
    "CD8+ T cell",          # CL:0000625
    "B cell",               # CL:0000236
    "Macrophage",           # CL:0000235
    "Dendritic cell",       # CL:0000451
    "Mast cell",            # CL:0000097
    "NK cell",              # CL:0000623
    "Fibroblast",           # CL:0000057
    "Myofibroblast",        # CL:0000186
    "Pericyte",             # CL:0000669
    "Endothelial",          # CL:0000115
    "Unknown",              # Unmapped
]

# Map ground truth labels to CL standard
GT_TO_CL = {
    # Epithelial (all tumor types)
    "Invasive_Tumor": "Epithelial",
    "DCIS_1": "Epithelial",
    "DCIS_2": "Epithelial",
    "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_ACTA2+": "Myofibroblast",
    "Myoepi_KRT15+": "Epithelial",  # Basal epithelial

    # T cells
    "CD4+_T_Cells": "CD4+ T cell",
    "CD8+_T_Cells": "CD8+ T cell",

    # B cells
    "B_Cells": "B cell",

    # Myeloid
    "Macrophages_1": "Macrophage",
    "Macrophages_2": "Macrophage",
    "IRF7+_DCs": "Dendritic cell",
    "LAMP3+_DCs": "Dendritic cell",
    "Mast_Cells": "Mast cell",

    # Stromal
    "Stromal": "Fibroblast",
    "Perivascular-Like": "Pericyte",

    # Endothelial
    "Endothelial": "Endothelial",

    # Hybrids (assign to dominant component)
    "Stromal_&_T_Cell_Hybrid": "Fibroblast",
    "T_Cell_&_Tumor_Hybrid": "Epithelial",

    # Unknown
    "Unlabeled": "Unknown",
}

# Map annotator outputs to CL standard
ANNOTATOR_TO_CL = {
    # === SingleR (BlueprintEncode) ===
    "Epithelial cells": "Epithelial",
    "Keratinocytes": "Epithelial",
    "CD4+ T-cells": "CD4+ T cell",
    "CD8+ T-cells": "CD8+ T cell",
    "T-cells": "CD4+ T cell",  # Default T cells to CD4
    "B-cells": "B cell",
    "Macrophages": "Macrophage",
    "Monocytes": "Macrophage",
    "DC": "Dendritic cell",
    "NK cells": "NK cell",
    "Fibroblasts": "Fibroblast",
    "Smooth muscle": "Fibroblast",
    "Endothelial cells": "Endothelial",
    "Adipocytes": "Fibroblast",
    "Neutrophils": "Macrophage",
    "Eosinophils": "Macrophage",

    # === CellTypist ===
    "Luminal epithelial cell of mammary gland": "Epithelial",
    "Basal cell": "Epithelial",
    "T cell": "CD4+ T cell",
    "CD4-positive, alpha-beta T cell": "CD4+ T cell",
    "CD8-positive, alpha-beta T cell": "CD8+ T cell",
    "B cell": "B cell",
    "Macrophage": "Macrophage",
    "Classical monocyte": "Macrophage",
    "Non-classical monocyte": "Macrophage",
    "Dendritic cell": "Dendritic cell",
    "Mast cell": "Mast cell",
    "Fibroblast": "Fibroblast",
    "Endothelial cell": "Endothelial",
    "Smooth muscle cell": "Fibroblast",
    "Pericyte": "Pericyte",
    "Adipocyte": "Fibroblast",
    "NK cell": "NK cell",
    "Plasma cell": "B cell",

    # === scType ===
    "Epithelial": "Epithelial",
    "T cells": "CD4+ T cell",
    "CD4+ T cells": "CD4+ T cell",
    "CD8+ T cells": "CD8+ T cell",
    "Regulatory T cells": "CD4+ T cell",
    "B cells": "B cell",
    "Plasma cells": "B cell",
    "Macrophages": "Macrophage",
    "Monocytes": "Macrophage",
    "Dendritic cells": "Dendritic cell",
    "Mast cells": "Mast cell",
    "NK cells": "NK cell",
    "Fibroblasts": "Fibroblast",
    "Myofibroblasts": "Myofibroblast",
    "Pericytes": "Pericyte",
    "Adipocytes": "Fibroblast",
    "Endothelial cells": "Endothelial",
    "Lymphatic endothelial": "Endothelial",

    # === SCINA ===
    "T_cells": "CD4+ T cell",
    "CD4_T_cells": "CD4+ T cell",
    "CD8_T_cells": "CD8+ T cell",
    "Tregs": "CD4+ T cell",
    "B_cells": "B cell",
    "Plasma_cells": "B cell",
    "Dendritic_cells": "Dendritic cell",
    "Mast_cells": "Mast cell",
    "NK_cells": "NK cell",
    "Endothelial": "Endothelial",
    "Lymphatic_endothelial": "Endothelial",
}


def standardize_to_cl(label: str, mapping_dict: dict = None) -> str:
    """Standardize any label to Cell Ontology standard name."""
    # Check direct mapping first
    if mapping_dict and label in mapping_dict:
        return mapping_dict[label]

    # Check annotator mapping
    if label in ANNOTATOR_TO_CL:
        return ANNOTATOR_TO_CL[label]

    # Check GT mapping
    if label in GT_TO_CL:
        return GT_TO_CL[label]

    # Already a CL standard type?
    if label in CL_STANDARD_TYPES:
        return label

    # Handle Unknown variants
    if label.lower() in ["unknown", "unlabeled", "unassigned", "na", "nan"]:
        return "Unknown"

    # Fuzzy matching
    label_lower = label.lower()

    if any(x in label_lower for x in ["epithelial", "luminal", "tumor", "dcis", "cancer"]):
        return "Epithelial"
    if "cd4" in label_lower:
        return "CD4+ T cell"
    if "cd8" in label_lower:
        return "CD8+ T cell"
    if "t cell" in label_lower or "t-cell" in label_lower:
        return "CD4+ T cell"  # Default
    if "b cell" in label_lower or "b-cell" in label_lower or "plasma" in label_lower:
        return "B cell"
    if "macrophage" in label_lower or "monocyte" in label_lower:
        return "Macrophage"
    if "dendritic" in label_lower or label_lower == "dc":
        return "Dendritic cell"
    if "mast" in label_lower:
        return "Mast cell"
    if "nk" in label_lower or "natural killer" in label_lower:
        return "NK cell"
    if "myofibroblast" in label_lower:
        return "Myofibroblast"
    if "fibroblast" in label_lower or "stromal" in label_lower:
        return "Fibroblast"
    if "pericyte" in label_lower or "perivascular" in label_lower:
        return "Pericyte"
    if "endothelial" in label_lower:
        return "Endothelial"

    return "Unknown"


# =============================================================================
# Dataset Configuration
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
    """Load and CL-standardize ground truth annotations."""
    logger.info(f"Loading ground truth from {gt_path}")
    df = pd.read_excel(gt_path)
    gt_df = pl.from_pandas(df)

    gt_df = gt_df.rename({"Barcode": "cell_id", "Cluster": "gt_type_original"})
    gt_df = gt_df.with_columns(pl.col("cell_id").cast(pl.Utf8))

    # Standardize GT to CL
    gt_df = gt_df.with_columns(
        pl.col("gt_type_original").map_elements(
            lambda x: standardize_to_cl(x, GT_TO_CL),
            return_dtype=pl.Utf8
        ).alias("gt_type")
    )

    # Filter out Unknown for evaluation
    gt_df = gt_df.filter(pl.col("gt_type") != "Unknown")

    # Log distribution
    dist = gt_df.group_by("gt_type").count().sort("count", descending=True)
    logger.info(f"CL-standardized GT distribution:\n{dist}")

    return gt_df


# =============================================================================
# Individual Method Runners
# =============================================================================

def run_singler(adata: sc.AnnData) -> pl.DataFrame:
    """Run SingleR annotation via R subprocess."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save expression (genes x cells for R)
        expr_df = pd.DataFrame(
            adata.X.toarray().T if hasattr(adata.X, 'toarray') else adata.X.T,
            index=adata.var_names,
            columns=adata.obs_names
        )
        expr_path = temp_path / "expr_matrix.csv"
        expr_df.to_csv(expr_path)

        results_path = temp_path / "singler_results.csv"

        r_script = f'''
library(SingleR)
library(celldex)

expr <- as.matrix(read.csv("{expr_path}", row.names=1, check.names=FALSE))
ref <- BlueprintEncodeData()
common <- intersect(rownames(expr), rownames(ref))

if (length(common) >= 50) {{
    results <- SingleR(test = expr[common,], ref = ref[common,], labels = ref$label.main)
    output <- data.frame(
        cell_id = colnames(expr),
        predicted_type = results$labels,
        confidence = 1 - results$delta.next,
        stringsAsFactors = FALSE
    )
}} else {{
    output <- data.frame(
        cell_id = colnames(expr),
        predicted_type = rep("Unknown", ncol(expr)),
        confidence = rep(0.0, ncol(expr)),
        stringsAsFactors = FALSE
    )
}}
write.csv(output, "{results_path}", row.names = FALSE)
'''

        script_path = temp_path / "run_singler.R"
        with open(script_path, 'w') as f:
            f.write(r_script)

        logger.info("Running SingleR...")
        result = subprocess.run(
            ["Rscript", str(script_path)],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(temp_path)
        )

        if result.returncode != 0:
            logger.error(f"SingleR failed: {result.stderr[:500]}")
            return pl.DataFrame()

        results_df = pd.read_csv(results_path)
        return pl.from_pandas(results_df)


def run_celltypist(adata: sc.AnnData) -> pl.DataFrame:
    """Run CellTypist annotation."""
    import celltypist
    from celltypist import models

    adata_copy = adata.copy()
    sc.pp.normalize_total(adata_copy, target_sum=1e4)
    sc.pp.log1p(adata_copy)

    model_names = ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl", "Healthy_Human_Liver.pkl"]
    all_predictions = []

    for model_name in model_names:
        try:
            model = models.Model.load(model=model_name)
            predictions = celltypist.annotate(adata_copy, model=model, majority_voting=False)
            pred_df = pl.DataFrame({
                "cell_id": adata.obs_names.tolist(),
                "predicted_type": predictions.predicted_labels["predicted_labels"].tolist(),
                "confidence": predictions.probability_matrix.max(axis=1).tolist(),
            })
            all_predictions.append(pred_df)
        except Exception as e:
            logger.warning(f"CellTypist {model_name} failed: {e}")

    if not all_predictions:
        return pl.DataFrame()

    # Use first model's predictions (Breast-specific)
    return all_predictions[0]


def run_sctype(adata: sc.AnnData) -> pl.DataFrame:
    """Run scType annotation."""
    from dapidl.pipeline.components.annotators.sctype import ScTypeAnnotator
    from dapidl.pipeline.base import AnnotationConfig

    annotator = ScTypeAnnotator(config=AnnotationConfig(fine_grained=True))
    result = annotator.annotate(adata=adata)

    return result.annotations_df.select(["cell_id", "predicted_type", "confidence"])


def run_scina(adata: sc.AnnData) -> pl.DataFrame:
    """Run SCINA annotation."""
    from dapidl.pipeline.components.annotators.scina import SCINAAnnotator
    from dapidl.pipeline.base import AnnotationConfig

    annotator = SCINAAnnotator(config=AnnotationConfig(fine_grained=True))
    result = annotator.annotate(adata=adata)

    return result.annotations_df.select(["cell_id", "predicted_type", "confidence"])


METHOD_RUNNERS = {
    "singler": run_singler,
    "celltypist": run_celltypist,
    "sctype": run_sctype,
    "scina": run_scina,
}


# =============================================================================
# Combination Runner
# =============================================================================

def run_combination(
    methods: list[str],
    adata: sc.AnnData,
    method_cache: dict,
) -> pl.DataFrame:
    """Run a combination of methods and return CL-standardized ensemble predictions."""

    logger.info(f"  Running: {'+'.join(methods)}")

    all_predictions = []

    for method in methods:
        if method in method_cache:
            pred_df = method_cache[method]
        else:
            runner = METHOD_RUNNERS.get(method)
            if runner is None:
                logger.warning(f"Unknown method: {method}")
                continue
            pred_df = runner(adata)
            method_cache[method] = pred_df

        if pred_df.height == 0:
            logger.warning(f"    {method}: no predictions")
            continue

        logger.info(f"    ✓ {method}: {pred_df.height} cells")
        all_predictions.append((method, pred_df))

    if not all_predictions:
        return pl.DataFrame()

    # Single method - just standardize to CL
    if len(all_predictions) == 1:
        method, pred_df = all_predictions[0]
        return pred_df.with_columns([
            pl.col("cell_id").cast(pl.Utf8),
            pl.col("predicted_type").map_elements(
                lambda x: standardize_to_cl(x),
                return_dtype=pl.Utf8
            ).alias("predicted_type_cl")
        ])

    # Ensemble - confidence-weighted voting at CL level
    cell_ids = all_predictions[0][1]["cell_id"].to_list()

    # Collect CL-standardized votes
    cell_cl_votes = defaultdict(lambda: defaultdict(float))

    for method, pred_df in all_predictions:
        pred_dict = dict(zip(
            pred_df["cell_id"].to_list(),
            zip(pred_df["predicted_type"].to_list(), pred_df["confidence"].to_list())
        ))

        for cell_id in cell_ids:
            if cell_id in pred_dict:
                pred_type, conf = pred_dict[cell_id]
                cl_type = standardize_to_cl(pred_type)
                cell_cl_votes[cell_id][cl_type] += conf

    # Pick highest-weighted CL type
    results = []
    for cell_id in cell_ids:
        votes = cell_cl_votes[cell_id]
        if votes:
            best_cl = max(votes.items(), key=lambda x: x[1])
            results.append({
                "cell_id": cell_id,
                "predicted_type_cl": best_cl[0],
                "confidence": best_cl[1] / len(all_predictions),
            })
        else:
            results.append({
                "cell_id": cell_id,
                "predicted_type_cl": "Unknown",
                "confidence": 0.0,
            })

    result_df = pl.DataFrame(results)
    # Ensure cell_id is string to match ground truth
    return result_df.with_columns(pl.col("cell_id").cast(pl.Utf8))


# =============================================================================
# Main Benchmark
# =============================================================================

def run_all_combinations(
    methods: list[str],
    dataset_keys: list[str],
    combo_sizes: list[int] | None = None,
    project_name: str = "DAPIDL/CL-Standardized-Benchmark",
):
    """Run all method combinations with CL standardization."""

    # Generate all combinations
    all_combos = []
    sizes = combo_sizes or list(range(1, len(methods) + 1))

    for size in sizes:
        for combo in combinations(methods, size):
            all_combos.append(list(combo))

    logger.info(f"Testing {len(all_combos)} combinations across {sizes} sizes")

    # Initialize ClearML
    task = None
    if CLEARML_AVAILABLE:
        task = Task.init(
            project_name=project_name,
            task_name=f"CL-Standardized Benchmark {'+'.join(dataset_keys)}",
            reuse_last_task_id=True,
        )

    all_results = []

    for dataset_key in dataset_keys:
        dataset = DATASETS[dataset_key]

        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset['name']}")
        logger.info(f"{'='*60}")

        # Load data
        adata = load_xenium_data(dataset["xenium_path"])
        gt_df = load_ground_truth(dataset["gt_path"])

        # Cache method results
        method_cache = {}

        for i, combo in enumerate(all_combos):
            logger.info(f"\nCombination {i+1}/{len(all_combos)}")

            # Run combination
            pred_df = run_combination(combo, adata, method_cache)

            if pred_df.height == 0:
                logger.warning(f"  No predictions for {combo}")
                continue

            # Ensure we have CL-standardized predictions and cell_id is string
            pred_df = pred_df.with_columns(pl.col("cell_id").cast(pl.Utf8))
            if "predicted_type_cl" not in pred_df.columns:
                pred_df = pred_df.with_columns(
                    pl.col("predicted_type").map_elements(
                        lambda x: standardize_to_cl(x),
                        return_dtype=pl.Utf8
                    ).alias("predicted_type_cl")
                )

            # Join with GT
            eval_df = gt_df.join(
                pred_df.select(["cell_id", "predicted_type_cl"]),
                on="cell_id",
                how="inner"
            )

            if eval_df.height == 0:
                continue

            # Calculate metrics
            y_true = eval_df["gt_type"].to_list()
            y_pred = eval_df["predicted_type_cl"].to_list()

            acc = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            kappa = cohen_kappa_score(y_true, y_pred)

            combo_name = "+".join(combo)

            logger.info(f"    Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}")

            result = {
                "dataset": dataset_key,
                "combo": combo_name,
                "n_methods": len(combo),
                "accuracy": acc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "cohen_kappa": kappa,
                "n_cells": eval_df.height,
            }
            all_results.append(result)

            # Log to ClearML
            if task:
                Logger.current_logger().report_scalar(
                    title=f"{dataset_key}_accuracy",
                    series=combo_name,
                    value=acc,
                    iteration=len(combo)
                )
                Logger.current_logger().report_scalar(
                    title=f"{dataset_key}_macro_f1",
                    series=combo_name,
                    value=macro_f1,
                    iteration=len(combo)
                )

    # Summary
    if all_results:
        results_df = pl.DataFrame(all_results)

        logger.info(f"\n{'='*80}")
        logger.info("CL-STANDARDIZED BENCHMARK RESULTS")
        logger.info(f"{'='*80}")

        # Sort by macro_f1
        sorted_df = results_df.sort("macro_f1", descending=True)
        print("\nTop 20 combinations by Macro F1:")
        print(sorted_df.head(20))

        # Best per size
        print("\n\nBest combination per size:")
        for size in sorted(results_df["n_methods"].unique().to_list()):
            size_df = results_df.filter(pl.col("n_methods") == size)
            best = size_df.sort("macro_f1", descending=True).head(1)
            if best.height > 0:
                row = best.row(0, named=True)
                print(f"  {size} method(s): {row['combo']} - F1={row['macro_f1']:.4f}, Acc={row['accuracy']:.4f}")

        # Save results
        results_df.write_csv("cl_standardized_benchmark_results.csv")

        if task:
            task.upload_artifact("results_csv", "cl_standardized_benchmark_results.csv")

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CL-Standardized Annotation Benchmark")
    parser.add_argument("--methods", nargs="+", default=["singler", "celltypist", "sctype", "scina"])
    parser.add_argument("--datasets", nargs="+", default=["rep1", "rep2"])
    parser.add_argument("--combo-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--project", default="DAPIDL/CL-Standardized-Benchmark")

    args = parser.parse_args()

    logger.info(f"Methods: {args.methods}")
    logger.info(f"Datasets: {args.datasets}")

    run_all_combinations(
        methods=args.methods,
        dataset_keys=args.datasets,
        combo_sizes=args.combo_sizes,
        project_name=args.project,
    )


if __name__ == "__main__":
    main()
