#!/usr/bin/env python
"""Comprehensive cell type annotation benchmark with Cell Ontology standardization.

This script evaluates ALL implemented annotation methods using CL standardization
to ensure vocabulary-independent evaluation.

Usage:
    uv run python scripts/benchmark_all_with_cl.py \
        --dataset ~/datasets/raw/xenium/breast_tumor_rep1 \
        --n-cells 30000

Output:
    - benchmark_results/cl_standardized/results.json
    - benchmark_results/cl_standardized/comparison_table.csv
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
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dapidl.ontology import get_mapper, CLMapper
from dapidl.ontology.cl_mapper import MappingResult


# =============================================================================
# CL-Standardized Evaluator
# =============================================================================


class CLStandardizedEvaluator:
    """Evaluate annotations with Cell Ontology standardization.

    This ensures vocabulary-independent evaluation by mapping all predictions
    and ground truth to standardized CL IDs before comparison.
    """

    def __init__(self, mapper: CLMapper | None = None):
        self.mapper = mapper or get_mapper()

    def evaluate(
        self,
        predictions: list[str],
        ground_truth: list[str],
        level: str = "coarse",
    ) -> dict[str, Any]:
        """Evaluate predictions against ground truth using CL standardization.

        Args:
            predictions: List of predicted cell type labels (any vocabulary)
            ground_truth: List of ground truth labels (any vocabulary)
            level: Hierarchy level - "broad" (5), "coarse" (~15), "fine" (~75)

        Returns:
            Dictionary with accuracy, F1 scores, per-class metrics, and mapping stats
        """
        assert len(predictions) == len(ground_truth), "Length mismatch"

        # Map both to CL IDs
        pred_results = [self.mapper.map_with_info(p) for p in predictions]
        gt_results = [self.mapper.map_with_info(g) for g in ground_truth]

        # Get CL IDs
        pred_cl = [r.cl_id for r in pred_results]
        gt_cl = [r.cl_id for r in gt_results]

        # Roll up to target hierarchy level
        pred_level = [self.mapper.get_hierarchy_level(p, level) for p in pred_cl]
        gt_level = [self.mapper.get_hierarchy_level(g, level) for g in gt_cl]

        # Filter out UNMAPPED for metrics (but report rate)
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
                "unmapped_rate_pred": sum(1 for r in pred_results if r.cl_id == "UNMAPPED") / len(predictions),
                "unmapped_rate_gt": sum(1 for r in gt_results if r.cl_id == "UNMAPPED") / len(ground_truth),
            }

        # Compute metrics on valid cells
        accuracy = accuracy_score(gt_valid, pred_valid)
        f1_macro = f1_score(gt_valid, pred_valid, average="macro", zero_division=0)
        f1_weighted = f1_score(gt_valid, pred_valid, average="weighted", zero_division=0)

        # Per-class report
        report = classification_report(gt_valid, pred_valid, output_dict=True, zero_division=0)

        # Mapping statistics
        pred_methods = {}
        for r in pred_results:
            method = r.method.value
            pred_methods[method] = pred_methods.get(method, 0) + 1

        gt_methods = {}
        for r in gt_results:
            method = r.method.value
            gt_methods[method] = gt_methods.get(method, 0) + 1

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "n_valid": len(pred_valid),
            "n_total": len(predictions),
            "unmapped_rate_pred": sum(1 for r in pred_results if r.cl_id == "UNMAPPED") / len(predictions),
            "unmapped_rate_gt": sum(1 for r in gt_results if r.cl_id == "UNMAPPED") / len(ground_truth),
            "per_class": report,
            "pred_mapping_methods": pred_methods,
            "gt_mapping_methods": gt_methods,
        }


# =============================================================================
# Data Loading
# =============================================================================


def load_xenium_data(dataset_path: Path, n_cells: int | None = None) -> sc.AnnData:
    """Load Xenium data and create AnnData object."""
    import tifffile

    dataset_path = Path(dataset_path)

    # Load cells
    cells_path = dataset_path / "cells.parquet"
    if not cells_path.exists():
        cells_path = dataset_path / "cells.csv.gz"
    cells_df = pl.read_parquet(cells_path) if cells_path.suffix == ".parquet" else pl.read_csv(cells_path)

    # Load expression matrix
    cell_feature_matrix_path = dataset_path / "cell_feature_matrix.h5"
    if cell_feature_matrix_path.exists():
        import h5py
        with h5py.File(cell_feature_matrix_path, "r") as f:
            # Standard 10x format
            matrix = f["matrix"]
            data = matrix["data"][:]
            indices = matrix["indices"][:]
            indptr = matrix["indptr"][:]
            shape = matrix["shape"][:]
            features = [x.decode() for x in matrix["features"]["name"][:]]
            barcodes = [x.decode() for x in matrix["barcodes"][:]]

        from scipy.sparse import csc_matrix
        X = csc_matrix((data, indices, indptr), shape=shape).T.toarray()
        adata = sc.AnnData(X=X, obs={"cell_id": barcodes}, var={"gene": features})
        adata.var_names = features
        adata.obs_names = barcodes
    else:
        raise FileNotFoundError(f"No cell_feature_matrix.h5 found at {dataset_path}")

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


def load_ground_truth(gt_path: Path, cell_ids: list[str] | None = None) -> dict[str, str]:
    """Load ground truth annotations."""
    gt_path = Path(gt_path)

    if gt_path.suffix == ".csv":
        gt_df = pl.read_csv(gt_path)
    elif gt_path.suffix == ".parquet":
        gt_df = pl.read_parquet(gt_path)
    elif gt_path.suffix in [".xls", ".xlsx"]:
        import pandas as pd
        gt_df = pl.from_pandas(pd.read_excel(gt_path))
    else:
        raise ValueError(f"Unsupported format: {gt_path.suffix}")

    # Find cell ID and label columns
    cell_id_col = None
    for col in ["cell_id", "Cell_ID", "barcode", "Barcode"]:
        if col in gt_df.columns:
            cell_id_col = col
            break
    if cell_id_col is None:
        # Use first column or index
        cell_id_col = gt_df.columns[0]

    label_col = None
    for col in ["Cluster", "celltype_final", "cell_type", "annotation", "label"]:
        if col in gt_df.columns:
            label_col = col
            break
    if label_col is None:
        raise ValueError(f"Could not find label column in {gt_df.columns}")

    # Convert to dict
    gt_dict = dict(zip(
        gt_df[cell_id_col].cast(str).to_list(),
        gt_df[label_col].cast(str).to_list()
    ))

    logger.info(f"Loaded {len(gt_dict)} ground truth labels from column '{label_col}'")

    # Filter to requested cell IDs
    if cell_ids:
        gt_dict = {k: v for k, v in gt_dict.items() if k in cell_ids}
        logger.info(f"Filtered to {len(gt_dict)} matching cells")

    return gt_dict


# =============================================================================
# Annotator Wrappers
# =============================================================================


def run_celltypist(adata: sc.AnnData, model_name: str) -> list[str]:
    """Run CellTypist annotation."""
    import celltypist
    from celltypist import models

    models.download_models(force_update=False, model=model_name)
    model = models.Model.load(model=model_name)

    # Find overlapping genes
    model_genes = set(model.classifier.features)
    query_genes = set(adata.var_names)
    overlap = model_genes & query_genes
    logger.info(f"CellTypist {model_name}: {len(overlap)}/{len(model_genes)} genes overlap")

    # Run prediction
    predictions = celltypist.annotate(adata, model=model_name, majority_voting=False)
    return predictions.predicted_labels["predicted_labels"].tolist()


def run_sctype(adata: sc.AnnData) -> list[str]:
    """Run scType marker-based annotation."""
    from dapidl.pipeline.components.annotators.sctype import ScTypeAnnotator

    annotator = ScTypeAnnotator()
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


def run_scina(adata: sc.AnnData) -> list[str]:
    """Run SCINA marker-based annotation."""
    try:
        from dapidl.pipeline.components.annotators.scina import SCINAAnnotator
        annotator = SCINAAnnotator()
        result = annotator.annotate(None, adata=adata)
        return result.annotations_df["predicted_type"].to_list()
    except Exception as e:
        logger.warning(f"SCINA failed: {e}")
        return ["FAILED"] * len(adata)


# =============================================================================
# Main Benchmark
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark annotation methods with CL standardization")
    parser.add_argument("--dataset", type=Path, default=Path.home() / "datasets/raw/xenium/breast_tumor_rep1")
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--n-cells", type=int, default=30000)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/cl_standardized"))
    parser.add_argument("--levels", type=str, default="broad,coarse,fine")
    args = parser.parse_args()

    # Setup output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading dataset from {args.dataset}")
    adata = load_xenium_data(args.dataset, n_cells=args.n_cells)

    # Find ground truth
    gt_path = args.ground_truth
    if gt_path is None:
        # Try common locations
        for candidate in [
            args.dataset / "10x_supervised_annot.csv",
            args.dataset / "ground_truth.csv",
            args.dataset / "annotations.parquet",
        ]:
            if candidate.exists():
                gt_path = candidate
                break

    if gt_path is None or not gt_path.exists():
        logger.error(f"Ground truth not found. Tried: {gt_path}")
        sys.exit(1)

    # Load ground truth
    gt_dict = load_ground_truth(gt_path, cell_ids=list(adata.obs_names))

    # Align ground truth with data
    cell_ids = list(adata.obs_names)
    ground_truth = [gt_dict.get(cid, "Unknown") for cid in cell_ids]
    unknown_count = sum(1 for gt in ground_truth if gt == "Unknown")
    logger.info(f"Ground truth: {len(ground_truth) - unknown_count} matched, {unknown_count} unknown")

    # Define methods to benchmark
    METHODS = {
        "celltypist_high": lambda adata: run_celltypist(adata, "Immune_All_High.pkl"),
        "celltypist_low": lambda adata: run_celltypist(adata, "Immune_All_Low.pkl"),
        "celltypist_breast": lambda adata: run_celltypist(adata, "Cells_Adult_Breast.pkl"),
        "sctype": lambda adata: run_sctype(adata),
        # "singler_blueprint": lambda adata: run_singler(adata, "BlueprintEncodeData"),
        # "scina": lambda adata: run_scina(adata),
    }

    levels = args.levels.split(",")
    evaluator = CLStandardizedEvaluator()

    # Run all methods
    all_results = {}
    for method_name, method_fn in METHODS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {method_name}...")
        logger.info(f"{'='*60}")

        try:
            predictions = method_fn(adata)

            for level in levels:
                logger.info(f"  Evaluating at {level} level...")
                result = evaluator.evaluate(predictions, ground_truth, level=level)

                key = f"{method_name}_{level}"
                all_results[key] = result

                logger.info(f"    Accuracy: {result['accuracy']:.3f}")
                logger.info(f"    F1 Macro: {result['f1_macro']:.3f}")
                logger.info(f"    Unmapped (pred): {result['unmapped_rate_pred']:.1%}")
                logger.info(f"    Unmapped (gt): {result['unmapped_rate_gt']:.1%}")

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            for level in levels:
                all_results[f"{method_name}_{level}"] = {"error": str(e)}

    # Save results
    results_path = args.output / "results.json"
    with open(results_path, "w") as f:
        # Convert numpy types for JSON serialization
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

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE (CL-Standardized)")
    print("=" * 80)
    print(f"{'Method':<25} {'Level':<10} {'Accuracy':>10} {'F1 Macro':>10} {'Unmapped':>10}")
    print("-" * 80)

    for key, result in sorted(all_results.items()):
        if "error" in result:
            continue
        parts = key.rsplit("_", 1)
        method = parts[0]
        level = parts[1] if len(parts) > 1 else "?"
        print(f"{method:<25} {level:<10} {result['accuracy']:>10.3f} {result['f1_macro']:>10.3f} {result['unmapped_rate_pred']:>10.1%}")

    print("=" * 80)


if __name__ == "__main__":
    main()
