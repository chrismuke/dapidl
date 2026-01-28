#!/usr/bin/env python3
"""Compare tissue-specific vs universal annotation for training.

This script:
1. Annotates a dataset with both tissue-specific and universal approaches
2. Creates LMDB patches for each
3. Trains models on both and compares F1 at coarse/medium/fine levels
"""

import json
import time
from pathlib import Path
from collections import Counter

import anndata as ad
import numpy as np
import polars as pl
import scanpy as sc
from loguru import logger

from dapidl.data.xenium import XeniumDataReader
from dapidl.pipeline.components.annotators.universal_ensemble import (
    CORE_HUMAN_MODELS,
    filter_available_models,
)
from dapidl.pipeline.components.annotators.popv_ensemble import (
    PopVEnsembleConfig,
    PopVStyleEnsembleAnnotator,
    VotingStrategy,
    GranularityLevel,
)


def load_and_normalize_xenium(xenium_path: Path) -> ad.AnnData:
    """Load Xenium data and normalize for CellTypist."""
    reader = XeniumDataReader(xenium_path)
    expr_matrix, gene_names, cell_ids = reader.load_expression_matrix()

    adata = ad.AnnData(X=expr_matrix)
    adata.var_names = gene_names
    adata.obs_names = [str(c) for c in cell_ids]

    # Normalize for CellTypist
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    logger.info(f"Loaded {adata.n_obs} cells with {adata.n_vars} genes")
    return adata


def annotate_tissue_specific(adata: ad.AnnData) -> pl.DataFrame:
    """Annotate with tissue-specific models (breast + immune)."""
    tissue_models = ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"]
    tissue_models = filter_available_models(tissue_models)

    config = PopVEnsembleConfig(
        celltypist_models=tissue_models,
        include_singler_hpca=True,
        include_singler_blueprint=True,
        voting_strategy=VotingStrategy.ONTOLOGY_HIERARCHICAL,
        granularity=GranularityLevel.FINE,  # Get all levels
    )

    annotator = PopVStyleEnsembleAnnotator(config)
    result = annotator.annotate(adata)

    return result.annotations_df


def annotate_universal(adata: ad.AnnData) -> pl.DataFrame:
    """Annotate with universal core models (10 models)."""
    core_models = filter_available_models(CORE_HUMAN_MODELS)

    config = PopVEnsembleConfig(
        celltypist_models=core_models,
        include_singler_hpca=True,
        include_singler_blueprint=True,
        voting_strategy=VotingStrategy.ONTOLOGY_HIERARCHICAL,
        granularity=GranularityLevel.FINE,  # Get all levels
    )

    annotator = PopVStyleEnsembleAnnotator(config)
    result = annotator.annotate(adata)

    return result.annotations_df


def compare_annotations(
    tissue_df: pl.DataFrame,
    universal_df: pl.DataFrame,
) -> dict:
    """Compare annotation distributions at all granularity levels."""
    results = {}

    for level in ["coarse", "medium", "fine"]:
        col = f"predicted_type_{level}"

        if col not in tissue_df.columns or col not in universal_df.columns:
            continue

        tissue_dist = dict(tissue_df[col].value_counts().iter_rows())
        universal_dist = dict(universal_df[col].value_counts().iter_rows())

        # Agreement
        tissue_labels = tissue_df[col].to_list()
        universal_labels = universal_df[col].to_list()
        agreement = sum(1 for t, u in zip(tissue_labels, universal_labels) if t == u) / len(tissue_labels)

        results[level] = {
            "tissue_specific": {
                "n_classes": len(tissue_dist),
                "distribution": tissue_dist,
                "unknown_count": tissue_dist.get("Unknown", 0),
            },
            "universal": {
                "n_classes": len(universal_dist),
                "distribution": universal_dist,
                "unknown_count": universal_dist.get("Unknown", 0),
            },
            "agreement_rate": agreement,
        }

        logger.info(f"\n{level.upper()} level:")
        logger.info(f"  Tissue-specific: {len(tissue_dist)} classes, {tissue_dist.get('Unknown', 0)} Unknown")
        logger.info(f"  Universal: {len(universal_dist)} classes, {universal_dist.get('Unknown', 0)} Unknown")
        logger.info(f"  Agreement: {agreement:.1%}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare annotation strategies for training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="~/datasets/raw/xenium/breast_tumor_rep1",
        help="Path to Xenium dataset",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of cells to sample (0 for all)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    # Load data
    logger.info(f"Loading dataset: {dataset_path}")
    adata = load_and_normalize_xenium(dataset_path)

    # Sample if requested
    if args.sample_size > 0 and adata.n_obs > args.sample_size:
        np.random.seed(42)
        idx = np.random.choice(adata.n_obs, args.sample_size, replace=False)
        adata = adata[idx].copy()
        logger.info(f"Sampled {args.sample_size} cells")

    # Run both annotation strategies
    logger.info("\n" + "=" * 60)
    logger.info("TISSUE-SPECIFIC ANNOTATION")
    logger.info("=" * 60)
    start = time.time()
    tissue_df = annotate_tissue_specific(adata)
    tissue_time = time.time() - start
    logger.info(f"Time: {tissue_time:.1f}s")

    logger.info("\n" + "=" * 60)
    logger.info("UNIVERSAL ANNOTATION")
    logger.info("=" * 60)
    start = time.time()
    universal_df = annotate_universal(adata)
    universal_time = time.time() - start
    logger.info(f"Time: {universal_time:.1f}s")

    # Compare
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)
    comparison = compare_annotations(tissue_df, universal_df)

    # Summary table
    print("\n" + "=" * 80)
    print("ANNOTATION COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Level':<10} {'Metric':<20} {'Tissue-Specific':>18} {'Universal':>18}")
    print("-" * 66)

    for level in ["coarse", "medium", "fine"]:
        if level not in comparison:
            continue
        data = comparison[level]
        print(f"{level:<10} {'Classes':>20} {data['tissue_specific']['n_classes']:>18} {data['universal']['n_classes']:>18}")
        print(f"{'':<10} {'Unknown cells':>20} {data['tissue_specific']['unknown_count']:>18} {data['universal']['unknown_count']:>18}")
        print(f"{'':<10} {'Agreement':>20} {'-':>18} {data['agreement_rate']:>17.1%}")
        print("-" * 66)

    print(f"\n{'Annotation time (s)':<30} {tissue_time:>18.1f} {universal_time:>18.1f}")

    # Per-class breakdown at coarse level
    print("\n" + "-" * 80)
    print("COARSE CLASS DISTRIBUTION")
    print("-" * 80)

    if "coarse" in comparison:
        all_classes = set(comparison["coarse"]["tissue_specific"]["distribution"].keys()) | \
                     set(comparison["coarse"]["universal"]["distribution"].keys())

        print(f"{'Class':<20} {'Tissue-Specific':>18} {'Universal':>18} {'Diff':>12}")
        print("-" * 68)
        for cls in sorted(all_classes):
            t_count = comparison["coarse"]["tissue_specific"]["distribution"].get(cls, 0)
            u_count = comparison["coarse"]["universal"]["distribution"].get(cls, 0)
            diff = u_count - t_count
            print(f"{cls:<20} {t_count:>18} {u_count:>18} {diff:>+12}")

    # Save results
    output = {
        "dataset": str(dataset_path),
        "sample_size": adata.n_obs,
        "tissue_specific_time": tissue_time,
        "universal_time": universal_time,
        "comparison": comparison,
    }

    with open("annotation_training_comparison.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("\nResults saved to annotation_training_comparison.json")

    # Save annotations for training
    tissue_df.write_parquet("annotations_tissue_specific.parquet")
    universal_df.write_parquet("annotations_universal.parquet")
    logger.info("Annotations saved to parquet files")


if __name__ == "__main__":
    main()
