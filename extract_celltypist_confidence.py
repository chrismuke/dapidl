#!/usr/bin/env python3
"""
Extract confidence scores from all CellTypist models in parallel.
Creates a comprehensive table with predictions and confidence scores for each model.
"""

import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from loguru import logger

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Paths
XENIUM_PATH = Path("/home/chrism/datasets/xenium_breast_tumor/outs")
GROUND_TRUTH_PATH = Path("/home/chrism/datasets/xenium_breast_tumor/Cell_Barcode_Type_Matrices.xlsx")
OUTPUT_DIR = Path("/home/chrism/git/dapidl/benchmark_results/confidence_analysis")

# Ground truth mapping
GT_TO_BROAD = {
    "DCIS_1": "Epithelial", "DCIS_2": "Epithelial",
    "Invasive_Tumor": "Epithelial", "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_ACTA2+": "Epithelial", "Myoepi_KRT15+": "Epithelial",
    "B_Cells": "Immune", "CD4+_T_Cells": "Immune", "CD8+_T_Cells": "Immune",
    "Macrophages_1": "Immune", "Macrophages_2": "Immune", "Mast_Cells": "Immune",
    "IRF7+_DCs": "Immune", "LAMP3+_DCs": "Immune",
    "Stromal": "Stromal", "Endothelial": "Stromal", "Perivascular-Like": "Stromal",
}


def run_single_model(args: tuple) -> dict[str, Any] | None:
    """Run a single CellTypist model and extract confidence scores.

    This function is designed to be called in a separate process.
    """
    model_name, h5_path = args

    try:
        # Import inside function to avoid pickling issues
        import celltypist
        from celltypist import models

        # Load data fresh in each process (shared memory would be complex)
        adata = sc.read_10x_h5(h5_path)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Load model and run prediction
        model = models.Model.load(model_name)
        pred = celltypist.annotate(
            adata,
            model=model,
            majority_voting=False,
            mode='best match'
        )

        # Extract results
        prob_matrix = pred.probability_matrix
        predictions = pred.predicted_labels['predicted_labels'].values
        max_confidence = prob_matrix.max(axis=1).values

        # Get top 3 predictions for each cell
        top3_labels = []
        top3_probs = []
        for i in range(len(prob_matrix)):
            sorted_idx = np.argsort(prob_matrix.iloc[i].values)[::-1][:3]
            top3_labels.append([prob_matrix.columns[j] for j in sorted_idx])
            top3_probs.append([prob_matrix.iloc[i].values[j] for j in sorted_idx])

        return {
            'model_name': model_name,
            'predictions': predictions,
            'max_confidence': max_confidence,
            'top3_labels': top3_labels,
            'top3_probs': top3_probs,
            'n_cell_types': len(prob_matrix.columns),
            'cell_types': list(prob_matrix.columns),
        }

    except Exception as e:
        logger.error(f"Error running {model_name}: {e}")
        return None


def load_ground_truth() -> pd.DataFrame:
    """Load ground truth annotations."""
    gt = pd.read_excel(
        GROUND_TRUTH_PATH,
        sheet_name="Xenium R1 Fig1-5 (supervised)"
    )
    gt['cell_id'] = gt['Barcode'].astype(int)
    gt['gt_cluster'] = gt['Cluster']
    gt['gt_broad'] = gt['Cluster'].map(GT_TO_BROAD)
    return gt[['cell_id', 'gt_cluster', 'gt_broad']]


def get_human_models() -> list[str]:
    """Get list of human-compatible CellTypist models (exclude mouse)."""
    from celltypist import models

    all_models = models.models_description()
    human_models = []

    for _, row in all_models.iterrows():
        model_name = row['model']
        # Skip mouse models
        if 'Mouse' in model_name or 'mouse' in model_name:
            continue
        human_models.append(model_name)

    return human_models


def main():
    """Main function to extract confidence from all models."""
    import celltypist
    from celltypist import models as ct_models

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("CellTypist Confidence Score Extraction")
    logger.info("=" * 70)

    # Get list of models
    logger.info("Getting list of human-compatible models...")
    model_list = get_human_models()
    logger.info(f"Found {len(model_list)} human models")

    # Download all models first (sequential to avoid race conditions)
    logger.info("Ensuring all models are downloaded...")
    ct_models.download_models(force_update=False)

    # Load ground truth
    logger.info("Loading ground truth...")
    gt_df = load_ground_truth()

    # Prepare arguments for parallel execution
    h5_path = str(XENIUM_PATH / "cell_feature_matrix.h5")
    args_list = [(model_name, h5_path) for model_name in model_list]

    # Run in parallel
    n_workers = min(10, len(model_list))
    logger.info(f"Running {len(model_list)} models with {n_workers} parallel workers...")

    results = []
    failed_models = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_model, args): args[0] for args in args_list}

        for i, future in enumerate(as_completed(futures)):
            model_name = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    logger.info(f"[{i+1}/{len(model_list)}] ✓ {model_name} ({result['n_cell_types']} cell types)")
                else:
                    failed_models.append(model_name)
                    logger.warning(f"[{i+1}/{len(model_list)}] ✗ {model_name} (failed)")
            except Exception as e:
                failed_models.append(model_name)
                logger.error(f"[{i+1}/{len(model_list)}] ✗ {model_name}: {e}")

    logger.info(f"\nCompleted: {len(results)} successful, {len(failed_models)} failed")

    # Build combined DataFrame
    logger.info("Building combined results table...")

    # Start with cell IDs and ground truth
    n_cells = 167780  # Known from data
    combined_df = pd.DataFrame({
        'cell_id': range(1, n_cells + 1)
    })
    combined_df = combined_df.merge(gt_df, on='cell_id', how='left')

    # Add results from each model
    for result in results:
        model_short = result['model_name'].replace('.pkl', '')
        combined_df[f'{model_short}_pred'] = result['predictions']
        combined_df[f'{model_short}_conf'] = result['max_confidence']
        # Add top 3 as separate columns
        combined_df[f'{model_short}_top2'] = [x[1] if len(x) > 1 else '' for x in result['top3_labels']]
        combined_df[f'{model_short}_top2_conf'] = [x[1] if len(x) > 1 else 0 for x in result['top3_probs']]
        combined_df[f'{model_short}_top3'] = [x[2] if len(x) > 2 else '' for x in result['top3_labels']]
        combined_df[f'{model_short}_top3_conf'] = [x[2] if len(x) > 2 else 0 for x in result['top3_probs']]

    # Save to Excel (might be large, use parquet as backup)
    logger.info("Saving results...")

    # Save as parquet (efficient)
    parquet_path = OUTPUT_DIR / "celltypist_all_confidence.parquet"
    combined_df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved parquet: {parquet_path}")

    # Save summary Excel (just predictions and confidence, not top2/top3)
    summary_cols = ['cell_id', 'gt_cluster', 'gt_broad']
    for result in results:
        model_short = result['model_name'].replace('.pkl', '')
        summary_cols.extend([f'{model_short}_pred', f'{model_short}_conf'])

    summary_df = combined_df[summary_cols]
    excel_path = OUTPUT_DIR / "celltypist_confidence_summary.xlsx"

    # Excel has row limit of ~1M, we have 167K so it's fine
    logger.info(f"Saving Excel summary ({len(summary_cols)} columns)...")
    summary_df.to_excel(excel_path, index=False, engine='openpyxl')
    logger.info(f"Saved Excel: {excel_path}")

    # Save model metadata
    meta_data = []
    for result in results:
        meta_data.append({
            'model_name': result['model_name'],
            'n_cell_types': result['n_cell_types'],
            'cell_types': ', '.join(result['cell_types']),
            'mean_confidence': np.mean(result['max_confidence']),
            'median_confidence': np.median(result['max_confidence']),
            'high_conf_pct': np.mean(result['max_confidence'] > 0.5) * 100,
        })

    meta_df = pd.DataFrame(meta_data)
    meta_df = meta_df.sort_values('mean_confidence', ascending=False)
    meta_path = OUTPUT_DIR / "model_confidence_summary.csv"
    meta_df.to_csv(meta_path, index=False)
    logger.info(f"Saved model summary: {meta_path}")

    # Print summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 70)
    logger.info(f"\nTotal cells: {n_cells:,}")
    logger.info(f"Models run: {len(results)}")
    logger.info(f"Failed models: {len(failed_models)}")
    if failed_models:
        logger.info(f"  Failed: {failed_models}")

    logger.info("\nTop 10 models by mean confidence:")
    for _, row in meta_df.head(10).iterrows():
        logger.info(f"  {row['model_name']}: {row['mean_confidence']:.4f} (median: {row['median_confidence']:.4f})")

    logger.info("\nOutput files:")
    logger.info(f"  - {parquet_path}")
    logger.info(f"  - {excel_path}")
    logger.info(f"  - {meta_path}")


if __name__ == "__main__":
    main()
