#!/usr/bin/env python3
"""Comprehensive ensemble test using ALL CellTypist models and ALL SingleR references.

This script:
1. Uses ALL 47 human CellTypist models
2. Uses ALL 4 SingleR references (HPCA, Blueprint, Monaco, Novershtern)
3. Runs on multiple datasets
4. Compares with tissue-specific and universal (10-model) approaches
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
from dapidl.pipeline.components.annotators.mapping import map_to_broad_category


# All human CellTypist models (47 total)
ALL_HUMAN_MODELS = [
    "Adult_COVID19_PBMC.pkl",
    "Adult_Human_MTG.pkl",
    "Adult_Human_PancreaticIslet.pkl",
    "Adult_Human_PrefrontalCortex.pkl",
    "Adult_Human_Skin.pkl",
    "Adult_Human_Vascular.pkl",
    "Adult_cHSPCs_Illumina.pkl",
    "Adult_cHSPCs_Ultima.pkl",
    "Autopsy_COVID19_Lung.pkl",
    "COVID19_HumanChallenge_Blood.pkl",
    "COVID19_Immune_Landscape.pkl",
    "Cells_Adult_Breast.pkl",
    "Cells_Fetal_Lung.pkl",
    "Cells_Human_Tonsil.pkl",
    "Cells_Intestinal_Tract.pkl",
    "Cells_Lung_Airway.pkl",
    "Developing_Human_Brain.pkl",
    "Developing_Human_Gonads.pkl",
    "Developing_Human_Hippocampus.pkl",
    "Developing_Human_Organs.pkl",
    "Developing_Human_Thymus.pkl",
    "Fetal_Human_AdrenalGlands.pkl",
    "Fetal_Human_Pancreas.pkl",
    "Fetal_Human_Pituitary.pkl",
    "Fetal_Human_Retina.pkl",
    "Fetal_Human_Skin.pkl",
    "Healthy_Adult_Heart.pkl",
    "Healthy_COVID19_PBMC.pkl",
    "Healthy_Human_Liver.pkl",
    "Human_AdultAged_Hippocampus.pkl",
    "Human_Colorectal_Cancer.pkl",
    "Human_Developmental_Retina.pkl",
    "Human_Embryonic_YolkSac.pkl",
    "Human_Endometrium_Atlas.pkl",
    "Human_IPF_Lung.pkl",
    "Human_Longitudinal_Hippocampus.pkl",
    "Human_Lung_Atlas.pkl",
    "Human_PF_Lung.pkl",
    "Human_Placenta_Decidua.pkl",
    "Immune_All_High.pkl",
    "Immune_All_Low.pkl",
    "Lethal_COVID19_Lung.pkl",
    "Nuclei_Human_InnerEar.pkl",
    "Nuclei_Lung_Airway.pkl",
    "PaediatricAdult_COVID19_Airway.pkl",
    "PaediatricAdult_COVID19_PBMC.pkl",
    "Pan_Fetal_Human.pkl",
]

# All SingleR references
ALL_SINGLER_REFS = ["hpca", "blueprint", "monaco", "novershtern"]

# Tissue-specific models (original approach)
TISSUE_SPECIFIC_MODELS = ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"]

# Core universal models (10 models)
CORE_UNIVERSAL_MODELS = [
    "Immune_All_High.pkl",
    "Immune_All_Low.pkl",
    "Cells_Adult_Breast.pkl",
    "Human_Lung_Atlas.pkl",
    "Cells_Intestinal_Tract.pkl",
    "Healthy_Human_Liver.pkl",
    "Adult_Human_Skin.pkl",
    "Cells_Human_Tonsil.pkl",
    "Adult_Human_Vascular.pkl",
    "Developing_Human_Organs.pkl",
]

# All datasets to test
DATASETS = {
    "xenium_breast_rep1": Path("~/datasets/raw/xenium/breast_tumor_rep1").expanduser(),
    "xenium_breast_rep2": Path("~/datasets/raw/xenium/breast_tumor_rep2").expanduser(),
    "xenium_lung": Path("~/datasets/raw/xenium/lung_2fov").expanduser(),
    "xenium_ovarian": Path("~/datasets/raw/xenium/ovarian_cancer").expanduser(),
    "xenium_lymph_node": Path("~/datasets/raw/xenium/lymph_node_normal").expanduser(),
    "xenium_colon_cancer": Path("~/datasets/raw/xenium/colon_cancer_colon-panel").expanduser(),
    "xenium_liver_cancer": Path("~/datasets/raw/xenium/liver_cancer_multi-tissue-panel").expanduser(),
    "merscope_breast": Path("~/datasets/raw/merscope/breast").expanduser(),
}


def filter_available_models(models: list[str]) -> list[str]:
    """Filter models to only those available."""
    import celltypist
    available = []
    for model in models:
        try:
            celltypist.models.download_models(model=model, force_update=False)
            available.append(model)
        except Exception:
            logger.warning(f"Model not available: {model}")
    return available


def load_and_normalize(dataset_path: Path, platform: str = "xenium") -> ad.AnnData:
    """Load data and normalize for CellTypist."""
    if platform == "xenium":
        reader = XeniumDataReader(dataset_path)
        expr_matrix, gene_names, cell_ids = reader.load_expression_matrix()
    else:
        # MERSCOPE
        from dapidl.data.merscope import MerscopeDataReader
        reader = MerscopeDataReader(dataset_path)
        expr_matrix, gene_names, cell_ids = reader.load_expression_matrix()

    adata = ad.AnnData(X=expr_matrix)
    adata.var_names = gene_names
    adata.obs_names = [str(c) for c in cell_ids]

    # Normalize for CellTypist
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


def run_celltypist_models(adata: ad.AnnData, models: list[str]) -> pl.DataFrame:
    """Run multiple CellTypist models and collect predictions."""
    import celltypist

    all_predictions = {"cell_id": list(adata.obs_names)}

    for model_name in models:
        safe_name = model_name.replace(".pkl", "").replace(".", "_")
        try:
            logger.info(f"Running CellTypist: {model_name}")
            model = celltypist.models.Model.load(model=model_name)
            predictions = celltypist.annotate(adata, model=model, majority_voting=False)

            labels = predictions.predicted_labels["predicted_labels"].tolist()
            # Map to broad categories
            broad = [map_to_broad_category(l) for l in labels]

            all_predictions[f"ct_{safe_name}"] = labels
            all_predictions[f"ct_{safe_name}_broad"] = broad

        except Exception as e:
            logger.warning(f"Failed to run {model_name}: {e}")
            # Fill with Unknown
            all_predictions[f"ct_{safe_name}"] = ["Unknown"] * adata.n_obs
            all_predictions[f"ct_{safe_name}_broad"] = ["Unknown"] * adata.n_obs

    return pl.DataFrame(all_predictions)


def run_singler_references(adata: ad.AnnData, references: list[str]) -> pl.DataFrame:
    """Run all SingleR references."""
    from dapidl.pipeline.components.annotators.singler import (
        is_singler_available,
        SINGLER_REFERENCES,
    )

    if not is_singler_available():
        logger.warning("SingleR not available - skipping")
        return pl.DataFrame({"cell_id": list(adata.obs_names)})

    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    singler = importr("SingleR")
    celldex = importr("celldex")

    all_predictions = {"cell_id": list(adata.obs_names)}

    # Prepare expression matrix once
    expr = adata.X
    if hasattr(expr, "toarray"):
        expr = expr.toarray()
    genes = list(adata.var_names)

    for ref_name in references:
        if ref_name not in SINGLER_REFERENCES:
            logger.warning(f"Unknown SingleR reference: {ref_name}")
            continue

        try:
            logger.info(f"Running SingleR: {ref_name}")
            ref_func_name = SINGLER_REFERENCES[ref_name]
            ref_func = getattr(celldex, ref_func_name)
            ref_data = ref_func()

            # Get reference gene names and find intersection
            ref_genes = list(ro.r.rownames(ref_data))
            common_genes = list(set(genes) & set(ref_genes))
            logger.info(f"  Common genes: {len(common_genes)}")

            if len(common_genes) < 50:
                logger.warning(f"  Low gene overlap, skipping {ref_name}")
                all_predictions[f"sr_{ref_name}"] = ["Unknown"] * adata.n_obs
                all_predictions[f"sr_{ref_name}_broad"] = ["Unknown"] * adata.n_obs
                continue

            # Subset to common genes
            gene_indices = [genes.index(g) for g in common_genes]

            # Create matrix with dimnames using list (not dict with empty keys)
            cell_names = [str(cid) for cid in adata.obs_names]
            dimnames = ro.r.list(ro.StrVector(common_genes), ro.StrVector(cell_names))
            expr_subset = ro.r.matrix(
                ro.FloatVector(expr[:, gene_indices].T.flatten()),
                nrow=len(common_genes),
                ncol=expr.shape[0],
                dimnames=dimnames,
            )

            ref_subset = ro.r["["](ref_data, ro.StrVector(common_genes), True)
            labels = ro.r("function(x) x$label.main")(ref_data)

            # Run SingleR
            results = singler.SingleR(
                test=expr_subset,
                ref=ref_subset,
                labels=labels,
                de_method="classic",
            )

            pred_labels = list(ro.r("function(x) x$labels")(results))
            broad = [map_to_broad_category(l) for l in pred_labels]

            all_predictions[f"sr_{ref_name}"] = pred_labels
            all_predictions[f"sr_{ref_name}_broad"] = broad

        except Exception as e:
            logger.warning(f"Failed to run SingleR {ref_name}: {e}")
            all_predictions[f"sr_{ref_name}"] = ["Unknown"] * adata.n_obs
            all_predictions[f"sr_{ref_name}_broad"] = ["Unknown"] * adata.n_obs

    return pl.DataFrame(all_predictions)


def consensus_vote(predictions_df: pl.DataFrame, strategy: str = "majority") -> pl.DataFrame:
    """Compute consensus vote from all predictions.

    Args:
        predictions_df: DataFrame with all predictions (columns ending in _broad)
        strategy: 'majority' or 'weighted'

    Returns:
        DataFrame with consensus predictions
    """
    broad_cols = [c for c in predictions_df.columns if c.endswith("_broad")]

    if not broad_cols:
        logger.warning("No broad category columns found")
        return predictions_df

    # For each cell, compute majority vote
    consensus = []
    confidence = []

    rows = predictions_df.select(broad_cols).to_numpy()

    for row in rows:
        # Count votes, excluding Unknown
        votes = Counter([v for v in row if v != "Unknown"])

        if not votes:
            consensus.append("Unknown")
            confidence.append(0.0)
        else:
            winner, count = votes.most_common(1)[0]
            total_valid = sum(votes.values())
            consensus.append(winner)
            confidence.append(count / len(broad_cols))  # Fraction of all models that agree

    result_df = predictions_df.with_columns([
        pl.Series("consensus_broad", consensus),
        pl.Series("consensus_confidence", confidence),
    ])

    return result_df


def evaluate_on_dataset(
    dataset_name: str,
    dataset_path: Path,
    sample_size: int = 5000,
    platform: str = "xenium",
) -> dict:
    """Evaluate all annotation strategies on a single dataset."""

    logger.info(f"\n{'='*60}")
    logger.info(f"DATASET: {dataset_name}")
    logger.info(f"{'='*60}")

    if not dataset_path.exists():
        logger.warning(f"Dataset not found: {dataset_path}")
        return {"error": "not_found"}

    # Load and sample
    adata = load_and_normalize(dataset_path, platform)

    if sample_size > 0 and adata.n_obs > sample_size:
        np.random.seed(42)
        idx = np.random.choice(adata.n_obs, sample_size, replace=False)
        adata = adata[idx].copy()
        logger.info(f"Sampled {sample_size} from {adata.n_obs} cells")
    else:
        logger.info(f"Using all {adata.n_obs} cells")

    results = {
        "dataset": dataset_name,
        "n_cells": adata.n_obs,
        "strategies": {},
    }

    # Strategy 1: Tissue-specific (2 models + 2 SingleR)
    logger.info("\n--- Strategy 1: Tissue-Specific (2 CT + 2 SR) ---")
    tissue_models = filter_available_models(TISSUE_SPECIFIC_MODELS)
    ct_tissue = run_celltypist_models(adata, tissue_models)
    sr_tissue = run_singler_references(adata, ["hpca", "blueprint"])
    tissue_df = ct_tissue.join(sr_tissue, on="cell_id", how="left")
    tissue_consensus = consensus_vote(tissue_df)

    tissue_dist = dict(tissue_consensus["consensus_broad"].value_counts().iter_rows())
    results["strategies"]["tissue_specific"] = {
        "n_models": len(tissue_models) + 2,
        "distribution": tissue_dist,
        "unknown_count": tissue_dist.get("Unknown", 0),
        "mean_confidence": tissue_consensus["consensus_confidence"].mean(),
    }

    # Strategy 2: Universal (10 models + 2 SingleR)
    logger.info("\n--- Strategy 2: Universal (10 CT + 2 SR) ---")
    universal_models = filter_available_models(CORE_UNIVERSAL_MODELS)
    ct_universal = run_celltypist_models(adata, universal_models)
    sr_universal = run_singler_references(adata, ["hpca", "blueprint"])
    universal_df = ct_universal.join(sr_universal, on="cell_id", how="left")
    universal_consensus = consensus_vote(universal_df)

    universal_dist = dict(universal_consensus["consensus_broad"].value_counts().iter_rows())
    results["strategies"]["universal_10"] = {
        "n_models": len(universal_models) + 2,
        "distribution": universal_dist,
        "unknown_count": universal_dist.get("Unknown", 0),
        "mean_confidence": universal_consensus["consensus_confidence"].mean(),
    }

    # Strategy 3: COMPREHENSIVE (ALL 47 CT + ALL 4 SR)
    logger.info("\n--- Strategy 3: COMPREHENSIVE (47 CT + 4 SR) ---")
    all_models = filter_available_models(ALL_HUMAN_MODELS)
    ct_all = run_celltypist_models(adata, all_models)
    sr_all = run_singler_references(adata, ALL_SINGLER_REFS)
    all_df = ct_all.join(sr_all, on="cell_id", how="left")
    all_consensus = consensus_vote(all_df)

    all_dist = dict(all_consensus["consensus_broad"].value_counts().iter_rows())
    results["strategies"]["comprehensive"] = {
        "n_models": len(all_models) + len(ALL_SINGLER_REFS),
        "distribution": all_dist,
        "unknown_count": all_dist.get("Unknown", 0),
        "mean_confidence": all_consensus["consensus_confidence"].mean(),
    }

    # Save detailed predictions
    output_dir = Path(f"comprehensive_ensemble_results/{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tissue_consensus.write_parquet(output_dir / "tissue_specific.parquet")
    universal_consensus.write_parquet(output_dir / "universal_10.parquet")
    all_consensus.write_parquet(output_dir / "comprehensive.parquet")

    # Print summary
    logger.info(f"\n--- Summary for {dataset_name} ---")
    for strategy, data in results["strategies"].items():
        logger.info(f"{strategy}: {data['n_models']} models, Unknown={data['unknown_count']}, Conf={data['mean_confidence']:.3f}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive ensemble annotation test")
    parser.add_argument("--datasets", nargs="*", help="Specific datasets to run (default: all)")
    parser.add_argument("--sample-size", type=int, default=5000, help="Cells to sample per dataset")
    args = parser.parse_args()

    # Select datasets
    if args.datasets:
        datasets = {k: v for k, v in DATASETS.items() if k in args.datasets}
    else:
        datasets = DATASETS

    logger.info(f"Testing {len(datasets)} datasets with sample size {args.sample_size}")

    all_results = {}

    for name, path in datasets.items():
        platform = "merscope" if "merscope" in name else "xenium"
        try:
            result = evaluate_on_dataset(name, path, args.sample_size, platform)
            all_results[name] = result
        except Exception as e:
            logger.error(f"Failed on {name}: {e}")
            all_results[name] = {"error": str(e)}

    # Save combined results
    output_path = Path("comprehensive_ensemble_results/summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print final summary table
    print("\n" + "=" * 100)
    print("COMPREHENSIVE ENSEMBLE COMPARISON SUMMARY")
    print("=" * 100)

    header = f"{'Dataset':<25} {'Strategy':<20} {'Models':<8} {'Unknown':<10} {'Confidence':<12}"
    print(f"\n{header}")
    print("-" * 75)

    for dataset, result in all_results.items():
        if "error" in result:
            print(f"{dataset:<25} ERROR: {result['error']}")
            continue

        for strategy, data in result["strategies"].items():
            row = f"{dataset:<25} {strategy:<20} {data['n_models']:<8} {data['unknown_count']:<10} {data['mean_confidence']:<12.3f}"
            print(row)
        print("-" * 75)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
