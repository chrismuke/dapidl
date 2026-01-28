#!/usr/bin/env python
"""Benchmark annotation methods on all datasets with Cell Ontology standardization.

This script:
1. Runs annotation pipeline on each dataset
2. Compares predictions to ground truth using CL standardization
3. Generates comprehensive benchmark reports
4. Optionally logs results to ClearML

Usage:
    # Benchmark all datasets with GT
    uv run python scripts/benchmark_all_datasets.py --all

    # Benchmark specific dataset
    uv run python scripts/benchmark_all_datasets.py --dataset breast_tumor_rep1

    # Log results to ClearML
    uv run python scripts/benchmark_all_datasets.py --all --clearml

    # Quick test (subset of cells)
    uv run python scripts/benchmark_all_datasets.py --dataset breast_tumor_rep1 --n-cells 10000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scanpy as sc
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dapidl.ontology import get_mapper

# =============================================================================
# Configuration
# =============================================================================

RAW_XENIUM_DIR = Path.home() / "datasets" / "raw" / "xenium"
OUTPUT_DIR = Path("benchmark_results") / "all_datasets"

# Datasets with manually supervised ground truth
GOLD_STANDARD_DATASETS = {
    "breast_tumor_rep1": {
        "path": RAW_XENIUM_DIR / "breast_tumor_rep1",
        "gt_file": "celltypes_ground_truth_rep1_supervised.xlsx",
        "gt_col": "Cluster",
        "cell_id_col": "Barcode",
        "tissue": "breast",
    },
    "breast_tumor_rep2": {
        "path": RAW_XENIUM_DIR / "breast_tumor_rep2",
        "gt_file": "celltypes_ground_truth_rep2_supervised.xlsx",
        "gt_col": "Cluster",
        "cell_id_col": "Barcode",
        "tissue": "breast",
    },
    "mouse_brain": {
        "path": RAW_XENIUM_DIR / "mouse_brain",
        "gt_file": "celltypes_ground_truth_mouse_brain.xlsx",
        "gt_col": "cell_type",
        "cell_id_col": "cell_id",
        "tissue": "brain",
        "species": "mouse",
        "celltypist_models": ["Mouse_Whole_Brain.pkl", "Mouse_Isocortex_Hippocampus.pkl"],
    },
}

# Hierarchy levels to evaluate
HIERARCHY_LEVELS = ["broad", "coarse", "fine"]


# =============================================================================
# Data Loading
# =============================================================================


def load_xenium_expression(dataset_path: Path, n_cells: int | None = None) -> sc.AnnData:
    """Load Xenium expression data as AnnData."""
    import h5py
    from scipy.sparse import csc_matrix

    # Find the cell feature matrix
    cfm_path = None
    for candidate in [
        dataset_path / "cell_feature_matrix.h5",
        dataset_path / "outs" / "cell_feature_matrix.h5",
    ]:
        if candidate.exists():
            cfm_path = candidate
            break

    if cfm_path is None:
        raise FileNotFoundError(f"cell_feature_matrix.h5 not found in {dataset_path}")

    logger.info(f"Loading expression from {cfm_path}")

    with h5py.File(cfm_path, "r") as f:
        matrix = f["matrix"]
        data = matrix["data"][:]
        indices = matrix["indices"][:]
        indptr = matrix["indptr"][:]
        shape = matrix["shape"][:]
        features = [x.decode() for x in matrix["features"]["name"][:]]
        barcodes = [x.decode() for x in matrix["barcodes"][:]]

    X = csc_matrix((data, indices, indptr), shape=shape).T.toarray()
    adata = sc.AnnData(X=X, obs={"cell_id": barcodes}, var={"gene": features})
    adata.var_names = features
    adata.obs_names = barcodes

    # Subsample if requested
    if n_cells and n_cells < len(adata):
        np.random.seed(42)
        indices = np.random.choice(len(adata), n_cells, replace=False)
        adata = adata[indices].copy()
        logger.info(f"Subsampled to {n_cells} cells")

    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    logger.info(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
    return adata


def load_ground_truth(gt_path: Path, gt_col: str, cell_id_col: str) -> dict[str, str]:
    """Load ground truth annotations."""
    import pandas as pd

    if gt_path.suffix in [".xlsx", ".xls"]:
        gt_df = pd.read_excel(gt_path)
    elif gt_path.suffix == ".csv":
        gt_df = pd.read_csv(gt_path)
    elif gt_path.suffix == ".parquet":
        gt_df = pl.read_parquet(gt_path).to_pandas()
    else:
        raise ValueError(f"Unsupported format: {gt_path.suffix}")

    # Convert to dict
    gt_dict = dict(zip(gt_df[cell_id_col].astype(str), gt_df[gt_col].astype(str)))

    logger.info(f"Loaded {len(gt_dict)} ground truth labels from {gt_path.name}")
    return gt_dict


def load_pipeline_annotations(dataset_path: Path) -> dict[str, str] | None:
    """Load existing pipeline annotations."""
    for candidate in [
        dataset_path / "pipeline_outputs" / "annotation" / "annotations.parquet",
        dataset_path / "outs" / "pipeline_outputs" / "annotation" / "annotations.parquet",
    ]:
        if candidate.exists():
            df = pl.read_parquet(candidate)
            cell_col = "cell_id" if "cell_id" in df.columns else df.columns[0]
            label_col = "predicted_type" if "predicted_type" in df.columns else df.columns[1]
            return dict(zip(df[cell_col].cast(str).to_list(), df[label_col].cast(str).to_list()))
    return None


# =============================================================================
# Annotation Methods
# =============================================================================


def run_celltypist(adata: sc.AnnData, model_name: str = "Immune_All_High.pkl") -> list[str]:
    """Run CellTypist annotation."""
    import celltypist
    from celltypist import models

    models.download_models(force_update=False, model=model_name)
    predictions = celltypist.annotate(adata, model=model_name, majority_voting=False)
    return predictions.predicted_labels["predicted_labels"].tolist()


def run_sctype(adata: sc.AnnData, tissue: str = "breast") -> list[str]:
    """Run scType marker-based annotation."""
    from dapidl.pipeline.components.annotators.sctype import ScTypeAnnotator

    annotator = ScTypeAnnotator(tissue=tissue)
    result = annotator.annotate(None, adata=adata)
    return result.annotations_df["predicted_type"].to_list()


def run_singler(adata: sc.AnnData, reference: str = "BlueprintEncodeData") -> list[str]:
    """Run SingleR annotation (requires R)."""
    try:
        from dapidl.pipeline.components.annotators.singler import SingleRAnnotator

        annotator = SingleRAnnotator(reference=reference)
        result = annotator.annotate(None, adata=adata)
        return result.annotations_df["predicted_type"].to_list()
    except Exception as e:
        logger.warning(f"SingleR failed: {e}")
        return ["FAILED"] * len(adata)


# =============================================================================
# CL-Standardized Evaluation
# =============================================================================


def evaluate_with_cl(
    predictions: list[str],
    ground_truth: list[str],
    mapper=None,
    level: str = "coarse",
) -> dict[str, Any]:
    """Evaluate predictions using Cell Ontology standardization."""
    if mapper is None:
        mapper = get_mapper()

    # Map both to CL IDs
    pred_results = [mapper.map_with_info(p) for p in predictions]
    gt_results = [mapper.map_with_info(g) for g in ground_truth]

    pred_cl = [r.cl_id for r in pred_results]
    gt_cl = [r.cl_id for r in gt_results]

    # Roll up to hierarchy level
    pred_level = [mapper.get_hierarchy_level(p, level) for p in pred_cl]
    gt_level = [mapper.get_hierarchy_level(g, level) for g in gt_cl]

    # Filter unmapped
    valid_mask = [(p != "Unknown" and g != "Unknown") for p, g in zip(pred_level, gt_level)]
    pred_valid = [p for p, v in zip(pred_level, valid_mask) if v]
    gt_valid = [g for g, v in zip(gt_level, valid_mask) if v]

    if len(pred_valid) == 0:
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "n_valid": 0,
            "n_total": len(predictions),
            "unmapped_rate_pred": 1.0,
            "unmapped_rate_gt": 1.0,
        }

    return {
        "accuracy": accuracy_score(gt_valid, pred_valid),
        "f1_macro": f1_score(gt_valid, pred_valid, average="macro", zero_division=0),
        "f1_weighted": f1_score(gt_valid, pred_valid, average="weighted", zero_division=0),
        "n_valid": len(pred_valid),
        "n_total": len(predictions),
        "unmapped_rate_pred": sum(1 for r in pred_results if r.cl_id == "UNMAPPED") / len(predictions),
        "unmapped_rate_gt": sum(1 for r in gt_results if r.cl_id == "UNMAPPED") / len(ground_truth),
        "per_class": classification_report(gt_valid, pred_valid, output_dict=True, zero_division=0),
    }


# =============================================================================
# Benchmark Runner
# =============================================================================


def benchmark_dataset(
    dataset_name: str,
    dataset_config: dict,
    methods: list[str] | None = None,
    n_cells: int | None = None,
) -> dict[str, Any]:
    """Run benchmark on a single dataset."""
    species = dataset_config.get("species", "human")
    tissue = dataset_config.get("tissue", "breast")

    # Use species-specific default methods
    if methods is None:
        if species == "mouse":
            # Use CellTypist models from config for mouse
            ct_models = dataset_config.get("celltypist_models", ["Mouse_Whole_Brain.pkl"])
            methods = [f"celltypist_{m.replace('.pkl', '')}" for m in ct_models]
        else:
            methods = ["celltypist_high", "celltypist_breast", "sctype"]

    dataset_path = dataset_config["path"]
    results = {"dataset": dataset_name, "methods": {}}

    # Load expression data
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking {dataset_name} (species={species}, tissue={tissue})")
    logger.info(f"{'='*60}")

    try:
        adata = load_xenium_expression(dataset_path, n_cells=n_cells)
    except Exception as e:
        logger.error(f"Failed to load expression: {e}")
        return {"dataset": dataset_name, "error": str(e)}

    # Load ground truth
    gt_path = dataset_path / dataset_config["gt_file"]
    if not gt_path.exists():
        # Try outs directory
        gt_path = dataset_path / "outs" / dataset_config["gt_file"]
    if not gt_path.exists():
        logger.error(f"Ground truth not found: {gt_path}")
        return {"dataset": dataset_name, "error": "Ground truth not found"}

    gt_dict = load_ground_truth(gt_path, dataset_config["gt_col"], dataset_config["cell_id_col"])

    # Align GT with data
    cell_ids = list(adata.obs_names)
    ground_truth = [gt_dict.get(cid, "Unknown") for cid in cell_ids]
    unknown_count = sum(1 for gt in ground_truth if gt == "Unknown")
    logger.info(f"Ground truth: {len(ground_truth) - unknown_count} matched, {unknown_count} unknown")

    # Initialize mapper
    mapper = get_mapper()

    # Define methods - include both static and dynamic CellTypist models
    method_fns = {
        "celltypist_high": lambda: run_celltypist(adata, "Immune_All_High.pkl"),
        "celltypist_low": lambda: run_celltypist(adata, "Immune_All_Low.pkl"),
        "celltypist_breast": lambda: run_celltypist(adata, "Cells_Adult_Breast.pkl"),
        "sctype": lambda: run_sctype(adata, tissue),
        "singler_blueprint": lambda: run_singler(adata, "BlueprintEncodeData"),
    }

    # Add mouse-specific CellTypist models dynamically
    if species == "mouse":
        ct_models = dataset_config.get("celltypist_models", [])
        for model in ct_models:
            method_key = f"celltypist_{model.replace('.pkl', '')}"
            # Use default argument to capture current model value
            method_fns[method_key] = lambda m=model: run_celltypist(adata, m)

    # Run each method
    for method_name in methods:
        if method_name not in method_fns:
            logger.warning(f"Unknown method: {method_name}")
            continue

        logger.info(f"\nRunning {method_name}...")

        try:
            predictions = method_fns[method_name]()

            method_results = {}
            for level in HIERARCHY_LEVELS:
                result = evaluate_with_cl(predictions, ground_truth, mapper, level=level)
                method_results[level] = result

                logger.info(f"  {level}: Acc={result['accuracy']:.3f}, F1={result['f1_macro']:.3f}")

            results["methods"][method_name] = method_results

        except Exception as e:
            logger.error(f"  Method failed: {e}")
            results["methods"][method_name] = {"error": str(e)}

    return results


def benchmark_all_datasets(
    methods: list[str] | None = None,
    n_cells: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Benchmark all datasets with ground truth."""
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset_name, config in GOLD_STANDARD_DATASETS.items():
        results = benchmark_dataset(dataset_name, config, methods=methods, n_cells=n_cells)
        all_results[dataset_name] = results

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:

        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(all_results), f, indent=2)

    logger.info(f"\nResults saved to {results_path}")

    # Generate comparison table
    print_comparison_table(all_results)

    return all_results


def print_comparison_table(all_results: dict) -> None:
    """Print a comparison table."""
    print("\n" + "=" * 100)
    print("BENCHMARK COMPARISON TABLE (CL-Standardized)")
    print("=" * 100)

    print(f"{'Dataset':<25} {'Method':<20} {'Level':<10} {'Accuracy':>10} {'F1 Macro':>10} {'Unmapped':>10}")
    print("-" * 100)

    for dataset_name, dataset_results in all_results.items():
        if "error" in dataset_results:
            print(f"{dataset_name:<25} ERROR: {dataset_results['error']}")
            continue

        for method_name, method_results in dataset_results.get("methods", {}).items():
            if "error" in method_results:
                print(f"{dataset_name:<25} {method_name:<20} ERROR: {method_results['error']}")
                continue

            for level, metrics in method_results.items():
                if isinstance(metrics, dict) and "accuracy" in metrics:
                    print(
                        f"{dataset_name:<25} {method_name:<20} {level:<10} "
                        f"{metrics['accuracy']:>10.3f} {metrics['f1_macro']:>10.3f} "
                        f"{metrics['unmapped_rate_pred']:>10.1%}"
                    )

    print("=" * 100)


# =============================================================================
# ClearML Integration
# =============================================================================


def log_to_clearml(results: dict, task_name: str = "benchmark_all_datasets") -> None:
    """Log results to ClearML."""
    try:
        from clearml import Task

        task = Task.init(project_name="DAPIDL/Benchmarks", task_name=task_name, task_type="testing")

        # Log scalars
        for dataset_name, dataset_results in results.items():
            if "error" in dataset_results:
                continue

            for method_name, method_results in dataset_results.get("methods", {}).items():
                if "error" in method_results:
                    continue

                for level, metrics in method_results.items():
                    if isinstance(metrics, dict) and "accuracy" in metrics:
                        task.get_logger().report_scalar(
                            title=f"{dataset_name}/{level}",
                            series=method_name,
                            value=metrics["accuracy"],
                            iteration=0,
                        )
                        task.get_logger().report_scalar(
                            title=f"{dataset_name}/{level}_f1",
                            series=method_name,
                            value=metrics["f1_macro"],
                            iteration=0,
                        )

        # Log full results as artifact
        task.upload_artifact("benchmark_results", results)

        task.close()
        logger.info(f"Logged results to ClearML task: {task.id}")

    except Exception as e:
        logger.error(f"Failed to log to ClearML: {e}")


# =============================================================================
# CLI
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark annotation methods with CL standardization")
    parser.add_argument("--all", action="store_true", help="Benchmark all datasets")
    parser.add_argument("--dataset", type=str, help="Benchmark specific dataset")
    parser.add_argument("--methods", type=str, nargs="+", help="Methods to benchmark")
    parser.add_argument("--n-cells", type=int, help="Subsample to N cells")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--clearml", action="store_true", help="Log results to ClearML")
    args = parser.parse_args()

    # Allow methods=None so species-specific defaults are used
    methods = args.methods  # None if not specified, allowing dataset-specific selection

    if args.all:
        results = benchmark_all_datasets(methods=methods, n_cells=args.n_cells, output_dir=args.output)
    elif args.dataset:
        if args.dataset not in GOLD_STANDARD_DATASETS:
            logger.error(f"Unknown dataset: {args.dataset}")
            logger.info(f"Available: {list(GOLD_STANDARD_DATASETS.keys())}")
            sys.exit(1)

        results = benchmark_dataset(
            args.dataset, GOLD_STANDARD_DATASETS[args.dataset], methods=methods, n_cells=args.n_cells
        )
        results = {args.dataset: results}
        print_comparison_table(results)
    else:
        parser.print_help()
        sys.exit(1)

    if args.clearml:
        log_to_clearml(results)


if __name__ == "__main__":
    main()
