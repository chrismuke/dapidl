#!/usr/bin/env python3
"""Comprehensive CellTypist Model Benchmarking Script.

Tests ALL pretrained CellTypist models with ALL option combinations against
the ground truth annotations from the Xenium breast cancer dataset.

Experimental Matrix:
- 58 CellTypist pretrained models
- 2 majority_voting options: True/False
- 2 mode options: 'best match' / 'prob match'
- 3 over_clustering options (when majority_voting=True): None, 'auto', Leiden(resolution=0.5/1.0/2.0)

Combinations:
- majority_voting=False: 2 modes = 2 configs
- majority_voting=True: 2 modes × 4 over_clustering = 8 configs
Total per model: 10 configs
Total experiments: 58 × 10 = 580 experiments
"""

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import celltypist
import numpy as np
import pandas as pd
import scanpy as sc
from celltypist import models
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# Paths
XENIUM_PATH = Path("/home/chrism/datasets/xenium_breast_tumor/outs")
GROUND_TRUTH_PATH = Path("/home/chrism/datasets/xenium_breast_tumor/Cell_Barcode_Type_Matrices.xlsx")
OUTPUT_DIR = Path("/home/chrism/git/dapidl/benchmark_results")
OBSIDIAN_PATH = Path("/home/chrism/obsidian/llmbrain/DAPIDL")

# Ground truth mapping from annotation.py
GROUND_TRUTH_MAPPING = {
    "DCIS_1": "Epithelial",
    "DCIS_2": "Epithelial",
    "Invasive_Tumor": "Epithelial",
    "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_ACTA2+": "Epithelial",
    "Myoepi_KRT15+": "Epithelial",
    "B_Cells": "Immune",
    "CD4+_T_Cells": "Immune",
    "CD8+_T_Cells": "Immune",
    "Macrophages_1": "Immune",
    "Macrophages_2": "Immune",
    "IRF7+_DCs": "Immune",
    "LAMP3+_DCs": "Immune",
    "Mast_Cells": "Immune",
    "Stromal": "Stromal",
    "Endothelial": "Stromal",
    "Perivascular-Like": "Stromal",
    "Stromal_&_T_Cell_Hybrid": "Hybrid",
    "T_Cell_&_Tumor_Hybrid": "Hybrid",
    "Unlabeled": "Unlabeled",
}

# Mapping from CellTypist cell types to broad categories
CELL_TYPE_HIERARCHY = {
    "Epithelial": [
        "LummHR", "LummHR-SCGB", "LummHR-active", "LummHR-major",
        "Lumsec", "Lumsec-HLA", "Lumsec-KIT", "Lumsec-basal",
        "Lumsec-lac", "Lumsec-major", "Lumsec-myo", "Lumsec-prol",
        "basal", "epithelial", "luminal", "mammary", "ductal",
        "myoepithelial", "glandular",
    ],
    "Immune": [
        "CD4", "CD8", "T_prol", "GD", "NKT", "Treg",
        "b_naive", "bmem", "plasma", "Macro", "Mono",
        "cDC", "mDC", "pDC", "mye", "NK", "Mast", "Neutrophil",
        "B cell", "T cell", "Macrophage", "Dendritic", "DC",
        "Monocyte", "lymphocyte", "immune",
    ],
    "Stromal": [
        "Fibro", "Fibroblast", "pericyte", "Pericyte",
        "vsmc", "Smooth muscle", "stromal", "mesenchymal",
        "adipocyte", "stellate",
    ],
    "Endothelial": [
        "Vas", "Endothelial", "Lymph", "vascular", "capillary",
        "arterial", "venous",
    ],
}


def map_to_broad_category(cell_type: str) -> str:
    """Map a detailed cell type to a broad category."""
    cell_type_lower = cell_type.lower()
    for broad_cat, keywords in CELL_TYPE_HIERARCHY.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if cell_type_lower == keyword_lower or cell_type_lower.startswith(keyword_lower):
                return broad_cat
    return "Unknown"


def load_xenium_expression() -> ad.AnnData:
    """Load Xenium expression data and create AnnData object."""
    logger.info("Loading Xenium expression data...")

    # Load expression matrix
    import scipy.io
    import h5py

    h5_path = XENIUM_PATH / "cell_feature_matrix.h5"
    with h5py.File(h5_path, 'r') as f:
        # 10x H5 format
        matrix = f['matrix']
        data = matrix['data'][:]
        indices = matrix['indices'][:]
        indptr = matrix['indptr'][:]
        shape = matrix['shape'][:]

        # Get gene names
        features = matrix['features']
        gene_names = [x.decode('utf-8') for x in features['name'][:]]

        # Get barcodes
        barcodes = [x.decode('utf-8') for x in matrix['barcodes'][:]]

    # Create sparse matrix
    from scipy.sparse import csc_matrix
    expr_matrix = csc_matrix((data, indices, indptr), shape=shape).T  # cells x genes

    # Create AnnData
    adata = ad.AnnData(X=expr_matrix)
    adata.var_names = gene_names
    adata.obs_names = barcodes

    # Load cell centroids
    cells_df = pd.read_parquet(XENIUM_PATH / "cells.parquet")

    # Match cell IDs
    adata.obs['cell_id'] = [b.split('-')[0] if '-' in b else b for b in barcodes]

    logger.info(f"Loaded AnnData: {adata.shape[0]} cells x {adata.shape[1]} genes")

    return adata


def load_ground_truth() -> pd.DataFrame:
    """Load ground truth annotations from Excel file."""
    logger.info(f"Loading ground truth from: {GROUND_TRUTH_PATH}")

    gt_df = pd.read_excel(
        GROUND_TRUTH_PATH,
        sheet_name="Xenium R1 Fig1-5 (supervised)",
    )

    logger.info(f"Loaded {len(gt_df)} cells from ground truth")
    logger.info(f"Columns: {list(gt_df.columns)}")

    # Map to broad categories
    gt_df["broad_category"] = gt_df["Cluster"].map(GROUND_TRUTH_MAPPING)

    # Filter out Hybrid and Unlabeled
    gt_df = gt_df[~gt_df["broad_category"].isin(["Hybrid", "Unlabeled"])]

    logger.info(f"After filtering: {len(gt_df)} cells")
    logger.info(f"Category distribution:\n{gt_df['broad_category'].value_counts()}")

    return gt_df


def run_celltypist_benchmark(
    adata: ad.AnnData,
    model_name: str,
    majority_voting: bool,
    mode: str,
    over_clustering: str | None = None,
    leiden_clusters: dict[str, pd.Series] | None = None,
) -> dict[str, Any]:
    """Run CellTypist with specific configuration and return predictions.

    Args:
        adata: Normalized AnnData object (with pre-computed neighbor graph)
        model_name: Name of CellTypist model
        majority_voting: Whether to use majority voting
        mode: 'best match' or 'prob match'
        over_clustering: Over-clustering strategy for majority voting
            - None: Let CellTypist auto-compute clustering (default when mv=False)
            - 'leiden_0.5', 'leiden_1.0', 'leiden_2.0': Use pre-computed Leiden clustering
        leiden_clusters: Pre-computed Leiden clusters dict {name: Series}
    """

    result = {
        "model_name": model_name,
        "majority_voting": majority_voting,
        "mode": mode,
        "over_clustering": over_clustering,
        "status": "failed",
        "error": None,
        "predictions": None,
        "run_time_seconds": 0,
    }

    try:
        start_time = time.time()

        # Download and load model
        models.download_models(model=model_name, force_update=False)
        model = models.Model.load(model=model_name)

        # Prepare over_clustering parameter
        oc_param = None
        if majority_voting and over_clustering:
            if over_clustering.startswith('leiden_') and leiden_clusters:
                # Use pre-computed Leiden clustering
                oc_param = leiden_clusters.get(over_clustering)
                if oc_param is None:
                    raise ValueError(f"Pre-computed clusters not found for {over_clustering}")

        # Run annotation
        if majority_voting:
            predictions = celltypist.annotate(
                adata,
                model=model,
                majority_voting=True,
                mode=mode,
                over_clustering=oc_param,
            )
        else:
            predictions = celltypist.annotate(
                adata,
                model=model,
                majority_voting=False,
                mode=mode,
            )

        # Extract results
        pred_labels = predictions.predicted_labels
        label_col = "majority_voting" if majority_voting and "majority_voting" in pred_labels.columns else "predicted_labels"

        result["predictions"] = pred_labels[label_col].values
        result["confidence"] = predictions.probability_matrix.max(axis=1).values
        result["status"] = "success"
        result["run_time_seconds"] = time.time() - start_time
        result["n_cell_types"] = len(model.cell_types)

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error running {model_name}: {e}")

    return result


def calculate_metrics(
    predictions: np.ndarray,
    ground_truth: pd.Series,
    cell_ids_pred: list,
    cell_ids_gt: list,
) -> dict[str, Any]:
    """Calculate classification metrics comparing predictions to ground truth."""

    # Create mapping from barcode to ground truth
    gt_map = dict(zip(cell_ids_gt, ground_truth))

    # Match predictions to ground truth
    matched_pred = []
    matched_gt = []
    matched_pred_broad = []
    matched_gt_broad = []

    for i, cell_id in enumerate(cell_ids_pred):
        # Extract numeric part for matching
        cell_num = cell_id.split('-')[0] if '-' in cell_id else cell_id

        if cell_num in gt_map:
            pred = predictions[i]
            gt = gt_map[cell_num]

            matched_pred.append(pred)
            matched_gt.append(gt)

            # Map to broad categories
            pred_broad = map_to_broad_category(str(pred))
            gt_broad = GROUND_TRUTH_MAPPING.get(gt, "Unknown")

            matched_pred_broad.append(pred_broad)
            matched_gt_broad.append(gt_broad)

    if len(matched_pred) == 0:
        return {"error": "No matching cells found"}

    metrics = {
        "n_matched_cells": len(matched_pred),
        "n_total_pred": len(predictions),
        "coverage": len(matched_pred) / len(predictions),
    }

    # Broad category metrics (main comparison)
    try:
        metrics["broad_accuracy"] = accuracy_score(matched_gt_broad, matched_pred_broad)
        metrics["broad_f1_macro"] = f1_score(matched_gt_broad, matched_pred_broad, average='macro', zero_division=0)
        metrics["broad_f1_weighted"] = f1_score(matched_gt_broad, matched_pred_broad, average='weighted', zero_division=0)
        metrics["broad_precision_macro"] = precision_score(matched_gt_broad, matched_pred_broad, average='macro', zero_division=0)
        metrics["broad_recall_macro"] = recall_score(matched_gt_broad, matched_pred_broad, average='macro', zero_division=0)

        # Per-class metrics
        report = classification_report(matched_gt_broad, matched_pred_broad, output_dict=True, zero_division=0)
        metrics["broad_per_class"] = report

    except Exception as e:
        metrics["broad_error"] = str(e)

    return metrics


def save_results_to_obsidian(all_results: list[dict], output_path: Path):
    """Save benchmark results to Obsidian markdown file."""

    # Sort by broad_f1_macro
    sorted_results = sorted(
        [r for r in all_results if r.get("metrics", {}).get("broad_f1_macro") is not None],
        key=lambda x: x.get("metrics", {}).get("broad_f1_macro", 0),
        reverse=True
    )

    md_content = f"""# CellTypist Model Benchmark Results

*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

## Summary

- **Total experiments**: {len(all_results)}
- **Successful runs**: {len([r for r in all_results if r.get("status") == "success"])}
- **Failed runs**: {len([r for r in all_results if r.get("status") == "failed"])}

## Top 20 Models by Broad Category F1 Score

| Rank | Model | Majority Voting | Mode | F1 (macro) | Accuracy | Precision | Recall |
|------|-------|-----------------|------|------------|----------|-----------|--------|
"""

    for i, result in enumerate(sorted_results[:20], 1):
        metrics = result.get("metrics", {})
        md_content += f"| {i} | {result['model_name']} | {result['majority_voting']} | {result['mode']} | {metrics.get('broad_f1_macro', 'N/A'):.4f} | {metrics.get('broad_accuracy', 'N/A'):.4f} | {metrics.get('broad_precision_macro', 'N/A'):.4f} | {metrics.get('broad_recall_macro', 'N/A'):.4f} |\n"

    md_content += """

## Best Configuration per Model

| Model | Best Config | F1 Score | Notes |
|-------|-------------|----------|-------|
"""

    # Group by model
    model_best = {}
    for result in sorted_results:
        model = result['model_name']
        f1 = result.get("metrics", {}).get("broad_f1_macro", 0)
        if model not in model_best or f1 > model_best[model].get("metrics", {}).get("broad_f1_macro", 0):
            model_best[model] = result

    for model, result in sorted(model_best.items(), key=lambda x: x[1].get("metrics", {}).get("broad_f1_macro", 0), reverse=True)[:30]:
        metrics = result.get("metrics", {})
        config = f"mv={result['majority_voting']}, mode={result['mode']}"
        md_content += f"| {model} | {config} | {metrics.get('broad_f1_macro', 'N/A'):.4f} | {result.get('n_cell_types', 'N/A')} cell types |\n"

    md_content += """

## Option Analysis

### Majority Voting Impact

"""

    # Analyze majority voting impact
    mv_true_f1 = [r.get("metrics", {}).get("broad_f1_macro", 0) for r in all_results if r.get("majority_voting") and r.get("metrics", {}).get("broad_f1_macro")]
    mv_false_f1 = [r.get("metrics", {}).get("broad_f1_macro", 0) for r in all_results if not r.get("majority_voting") and r.get("metrics", {}).get("broad_f1_macro")]

    if mv_true_f1 and mv_false_f1:
        md_content += f"- **Majority Voting = True**: Mean F1 = {np.mean(mv_true_f1):.4f} (std: {np.std(mv_true_f1):.4f})\n"
        md_content += f"- **Majority Voting = False**: Mean F1 = {np.mean(mv_false_f1):.4f} (std: {np.std(mv_false_f1):.4f})\n"

    md_content += """

### Mode Impact

"""

    best_match_f1 = [r.get("metrics", {}).get("broad_f1_macro", 0) for r in all_results if r.get("mode") == "best match" and r.get("metrics", {}).get("broad_f1_macro")]
    prob_match_f1 = [r.get("metrics", {}).get("broad_f1_macro", 0) for r in all_results if r.get("mode") == "prob match" and r.get("metrics", {}).get("broad_f1_macro")]

    if best_match_f1 and prob_match_f1:
        md_content += f"- **Mode = best match**: Mean F1 = {np.mean(best_match_f1):.4f} (std: {np.std(best_match_f1):.4f})\n"
        md_content += f"- **Mode = prob match**: Mean F1 = {np.mean(prob_match_f1):.4f} (std: {np.std(prob_match_f1):.4f})\n"

    md_content += """

## Failed Experiments

"""

    failed = [r for r in all_results if r.get("status") == "failed"]
    if failed:
        for result in failed[:20]:
            md_content += f"- **{result['model_name']}** (mv={result['majority_voting']}, mode={result['mode']}): {result.get('error', 'Unknown error')}\n"
    else:
        md_content += "No failed experiments.\n"

    md_content += f"""

---
*Benchmark completed with {len(all_results)} total experiments*
*Data: Xenium FFPE Human Breast Cancer Rep1*
*Ground truth: Cell_Barcode_Type_Matrices.xlsx (supervised annotations)*
"""

    with open(output_path, 'w') as f:
        f.write(md_content)

    logger.info(f"Results saved to: {output_path}")


def main():
    """Run comprehensive CellTypist benchmark."""

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    logger.info("=" * 60)
    logger.info("CellTypist Comprehensive Benchmark")
    logger.info("=" * 60)

    # Load expression data
    adata = load_xenium_expression()

    # Normalize for CellTypist (log1p to 10k counts)
    logger.info("Normalizing expression data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Load ground truth
    gt_df = load_ground_truth()

    # Pre-compute neighbor graph and Leiden clustering (saves ~2 min per experiment)
    logger.info("Pre-computing neighbor graph and Leiden clustering...")
    sc.pp.neighbors(adata, n_neighbors=15)

    # Pre-compute Leiden at different resolutions
    leiden_clusters = {}
    for res in [0.5, 1.0, 2.0]:
        logger.info(f"  Computing Leiden at resolution {res}...")
        sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}')
        leiden_clusters[f'leiden_{res}'] = adata.obs[f'leiden_{res}'].copy()
    logger.info("  Pre-computation done!")

    # Get all available models
    all_models_df = models.models_description()
    all_model_names = list(all_models_df['model'])  # Model names are in 'model' column, not index
    logger.info(f"Found {len(all_model_names)} CellTypist models")

    # Define experimental matrix
    # When majority_voting=False: no over_clustering
    # When majority_voting=True: test various over_clustering options
    mode_options = ["best match", "prob match"]
    # Note: 'auto' doesn't work as expected - None with mv=True auto-computes clustering
    over_clustering_options = [None, 'leiden_0.5', 'leiden_1.0', 'leiden_2.0']

    # Build all configurations
    configs = []
    for model_name in all_model_names:
        for mode in mode_options:
            # Without majority voting (no over_clustering)
            configs.append({
                "model": model_name,
                "mv": False,
                "mode": mode,
                "oc": None
            })
            # With majority voting (test all over_clustering options)
            for oc in over_clustering_options:
                configs.append({
                    "model": model_name,
                    "mv": True,
                    "mode": mode,
                    "oc": oc
                })

    logger.info(f"Total experiments to run: {len(configs)}")
    logger.info(f"  - Models: {len(all_model_names)}")
    logger.info(f"  - Modes: {len(mode_options)}")
    logger.info(f"  - MV=False configs: {len(mode_options)}")
    logger.info(f"  - MV=True configs: {len(mode_options) * len(over_clustering_options)}")

    # Run benchmarks
    all_results = []
    completed = 0

    for config in configs:
        completed += 1
        model_name = config["model"]
        mv = config["mv"]
        mode = config["mode"]
        oc = config["oc"]

        logger.info(f"\n[{completed}/{len(configs)}] {model_name} (mv={mv}, mode={mode}, oc={oc})")

        # Run CellTypist (with pre-computed Leiden clusters for speedup)
        result = run_celltypist_benchmark(adata, model_name, mv, mode, oc, leiden_clusters)

        # Calculate metrics if successful
        if result["status"] == "success" and result["predictions"] is not None:
            metrics = calculate_metrics(
                result["predictions"],
                gt_df["Cluster"],
                list(adata.obs_names),
                list(gt_df["Barcode"].astype(str)),
            )
            result["metrics"] = metrics

            f1 = metrics.get("broad_f1_macro", "N/A")
            acc = metrics.get("broad_accuracy", "N/A")
            logger.info(f"  -> F1: {f1:.4f}, Accuracy: {acc:.4f}" if isinstance(f1, float) else f"  -> {metrics}")

        all_results.append(result)

        # Save intermediate results every 20 experiments
        if completed % 20 == 0:
            intermediate_path = OUTPUT_DIR / f"results_intermediate_{timestamp}.json"
            with open(intermediate_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"Intermediate results saved to: {intermediate_path}")

    # Save final results
    final_json_path = OUTPUT_DIR / f"celltypist_benchmark_{timestamp}.json"
    with open(final_json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Final results saved to: {final_json_path}")

    # Save to Obsidian
    obsidian_path = OBSIDIAN_PATH / f"CellTypist-Benchmark-{timestamp}.md"
    save_results_to_obsidian(all_results, obsidian_path)

    # Print summary
    successful = [r for r in all_results if r.get("status") == "success"]
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total experiments: {len(all_results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(all_results) - len(successful)}")

    if successful:
        best = max(successful, key=lambda x: x.get("metrics", {}).get("broad_f1_macro", 0))
        logger.info(f"\nBest configuration:")
        logger.info(f"  Model: {best['model_name']}")
        logger.info(f"  Majority Voting: {best['majority_voting']}")
        logger.info(f"  Mode: {best['mode']}")
        logger.info(f"  F1 (macro): {best.get('metrics', {}).get('broad_f1_macro', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
