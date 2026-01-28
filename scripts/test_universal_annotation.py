#!/usr/bin/env python3
"""Test universal annotation vs tissue-specific annotation.

Compares:
1. Current: 2 tissue-specific models (Cells_Adult_Breast.pkl + Immune_All_High.pkl)
2. Universal: 10 core human models + SingleR
3. Universal All: 40+ human models + SingleR (optional, slow)
"""

import json
from pathlib import Path
from collections import Counter
import time

import anndata as ad
import numpy as np
from loguru import logger
import polars as pl

from dapidl.pipeline.components.annotators.universal_ensemble import (
    CORE_HUMAN_MODELS,
    ALL_HUMAN_CELLTYPIST_MODELS,
    filter_available_models,
)
from dapidl.pipeline.components.annotators.popv_ensemble import (
    PopVEnsembleConfig,
    PopVStyleEnsembleAnnotator,
    VotingStrategy,
    GranularityLevel,
)


def load_xenium_adata(xenium_path: Path) -> ad.AnnData:
    """Load AnnData from Xenium output directory."""
    from dapidl.data.xenium import XeniumDataReader
    import pandas as pd

    reader = XeniumDataReader(xenium_path)
    expr_matrix, gene_names, cell_ids = reader.load_expression_matrix()

    # Create AnnData object
    adata = ad.AnnData(X=expr_matrix)
    adata.var_names = gene_names
    adata.obs_names = [str(c) for c in cell_ids]

    logger.info(f"Loaded {adata.n_obs} cells with {adata.n_vars} genes")
    return adata


def run_annotation_comparison(
    adata: ad.AnnData,
    sample_size: int = 5000,
) -> dict:
    """Compare different annotation strategies."""

    # Sample for speed
    if adata.n_obs > sample_size:
        np.random.seed(42)
        idx = np.random.choice(adata.n_obs, sample_size, replace=False)
        adata_sample = adata[idx].copy()
        logger.info(f"Sampled {sample_size} cells for comparison")
    else:
        adata_sample = adata.copy()

    results = {}

    # 1. Tissue-specific (current approach)
    logger.info("\n" + "=" * 60)
    logger.info("1. TISSUE-SPECIFIC (current approach)")
    logger.info("=" * 60)

    tissue_models = ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"]
    tissue_models = filter_available_models(tissue_models)

    if tissue_models:
        config_tissue = PopVEnsembleConfig(
            celltypist_models=tissue_models,
            include_singler_hpca=True,
            include_singler_blueprint=True,
            voting_strategy=VotingStrategy.ONTOLOGY_HIERARCHICAL,
            granularity=GranularityLevel.COARSE,
        )

        start = time.time()
        annotator = PopVStyleEnsembleAnnotator(config_tissue)
        result_tissue = annotator.annotate(adata_sample)
        time_tissue = time.time() - start

        # Extract results from EnsembleResult dataclass
        df_tissue = result_tissue.annotations_df
        coarse_col = "predicted_type_coarse"
        conf_col = "confidence" if "confidence" in df_tissue.columns else None

        results["tissue_specific"] = {
            "models": tissue_models,
            "n_models": len(tissue_models),
            "time_seconds": time_tissue,
            "coarse_distribution": dict(df_tissue[coarse_col].value_counts().iter_rows()),
            "mean_confidence": float(df_tissue[conf_col].mean()) if conf_col and conf_col in df_tissue.columns else 0.0,
        }
        logger.info(f"Time: {time_tissue:.1f}s, Models: {len(tissue_models)}")
        logger.info(f"Distribution: {results['tissue_specific']['coarse_distribution']}")

    # 2. Universal Core (10 models)
    logger.info("\n" + "=" * 60)
    logger.info("2. UNIVERSAL CORE (10 models)")
    logger.info("=" * 60)

    core_models = filter_available_models(CORE_HUMAN_MODELS)
    logger.info(f"Using {len(core_models)} core models: {core_models[:5]}...")

    config_core = PopVEnsembleConfig(
        celltypist_models=core_models,
        include_singler_hpca=True,
        include_singler_blueprint=True,
        voting_strategy=VotingStrategy.ONTOLOGY_HIERARCHICAL,
        granularity=GranularityLevel.COARSE,
    )

    start = time.time()
    annotator = PopVStyleEnsembleAnnotator(config_core)
    result_core = annotator.annotate(adata_sample)
    time_core = time.time() - start

    # Extract results from EnsembleResult dataclass
    df_core = result_core.annotations_df
    coarse_col = "predicted_type_coarse"
    conf_col = "confidence" if "confidence" in df_core.columns else None

    results["universal_core"] = {
        "models": core_models,
        "n_models": len(core_models),
        "time_seconds": time_core,
        "coarse_distribution": dict(df_core[coarse_col].value_counts().iter_rows()),
        "mean_confidence": float(df_core[conf_col].mean()) if conf_col and conf_col in df_core.columns else 0.0,
    }
    logger.info(f"Time: {time_core:.1f}s, Models: {len(core_models)}")
    logger.info(f"Distribution: {results['universal_core']['coarse_distribution']}")

    # 3. Compare predictions
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)

    if "tissue_specific" in results:
        # Agreement rate
        tissue_pred = df_tissue[coarse_col].to_list()
        core_pred = df_core[coarse_col].to_list()

        agreement = sum(1 for t, c in zip(tissue_pred, core_pred) if t == c) / len(tissue_pred)
        logger.info(f"Agreement rate (tissue vs core): {agreement:.1%}")

        # Per-class comparison
        for cat in set(tissue_pred) | set(core_pred):
            t_count = sum(1 for p in tissue_pred if p == cat)
            c_count = sum(1 for p in core_pred if p == cat)
            diff = c_count - t_count
            logger.info(f"  {cat}: tissue={t_count}, core={c_count} ({diff:+d})")

        results["comparison"] = {
            "agreement_rate": agreement,
            "tissue_vs_core_diff": {
                cat: sum(1 for p in core_pred if p == cat) - sum(1 for p in tissue_pred if p == cat)
                for cat in set(tissue_pred) | set(core_pred)
            },
        }

    # 4. Confidence analysis
    logger.info("\n--- Confidence Analysis ---")
    if "tissue_specific" in results:
        logger.info(f"Tissue-specific mean confidence: {results['tissue_specific']['mean_confidence']:.3f}")
    logger.info(f"Universal core mean confidence: {results['universal_core']['mean_confidence']:.3f}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare annotation strategies")
    parser.add_argument(
        "--dataset",
        type=str,
        default="~/datasets/raw/xenium/breast_tumor_rep1",
        help="Path to Xenium dataset",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of cells to sample",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    logger.info(f"Loading dataset: {dataset_path}")
    adata = load_xenium_adata(dataset_path)

    logger.info(f"\nComparing annotation strategies on {adata.n_obs} cells...")
    results = run_annotation_comparison(adata, sample_size=args.sample_size)

    # Save results
    output_path = Path("annotation_comparison_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
