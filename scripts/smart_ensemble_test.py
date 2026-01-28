#!/usr/bin/env python3
"""Smart ensemble with weighted voting based on model relevance.

The naive "all models" approach is biased because:
- Many CellTypist models are immune-focused (COVID, PBMC, blood)
- Immune models outvote tissue-specific models in simple majority voting

This script implements weighted voting strategies:
1. EQUAL: All models get equal weight (baseline - shows bias)
2. TISSUE_WEIGHTED: Models matching the tissue get higher weight
3. CONFIDENCE_WEIGHTED: Weight by prediction confidence
4. ENTROPY_WEIGHTED: Models with more certain predictions get higher weight
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

import anndata as ad
import numpy as np
import polars as pl
import scanpy as sc
from loguru import logger

from dapidl.data.xenium import XeniumDataReader
from dapidl.pipeline.components.annotators.mapping import map_to_broad_category


# Model categories for tissue-weighted voting
MODEL_CATEGORIES = {
    # Immune-focused models (should get lower weight for solid tumors)
    "immune": [
        "Immune_All_High.pkl", "Immune_All_Low.pkl",
        "Adult_COVID19_PBMC.pkl", "COVID19_HumanChallenge_Blood.pkl",
        "COVID19_Immune_Landscape.pkl", "Healthy_COVID19_PBMC.pkl",
        "Adult_cHSPCs_Illumina.pkl", "Adult_cHSPCs_Ultima.pkl",
        "PaediatricAdult_COVID19_PBMC.pkl",
        "Developing_Human_Thymus.pkl",
    ],
    # Tissue/organ-specific models (good for solid tumors)
    "tissue": [
        "Cells_Adult_Breast.pkl", "Healthy_Human_Liver.pkl",
        "Human_Lung_Atlas.pkl", "Cells_Lung_Airway.pkl",
        "Adult_Human_Skin.pkl", "Fetal_Human_Skin.pkl",
        "Cells_Intestinal_Tract.pkl", "Human_Colorectal_Cancer.pkl",
        "Adult_Human_Vascular.pkl", "Healthy_Adult_Heart.pkl",
        "Adult_Human_PancreaticIslet.pkl", "Fetal_Human_Pancreas.pkl",
        "Human_Endometrium_Atlas.pkl", "Human_Placenta_Decidua.pkl",
        "Developing_Human_Organs.pkl", "Pan_Fetal_Human.pkl",
    ],
    # Brain/neural models
    "neural": [
        "Adult_Human_MTG.pkl", "Adult_Human_PrefrontalCortex.pkl",
        "Developing_Human_Brain.pkl", "Developing_Human_Hippocampus.pkl",
        "Human_AdultAged_Hippocampus.pkl", "Human_Longitudinal_Hippocampus.pkl",
    ],
    # Lung-disease models
    "lung_disease": [
        "Autopsy_COVID19_Lung.pkl", "Lethal_COVID19_Lung.pkl",
        "Human_IPF_Lung.pkl", "Human_PF_Lung.pkl",
        "PaediatricAdult_COVID19_Airway.pkl", "Nuclei_Lung_Airway.pkl",
    ],
    # Tonsil/lymphoid
    "lymphoid": [
        "Cells_Human_Tonsil.pkl",
    ],
    # Other specialized
    "other": [
        "Fetal_Human_AdrenalGlands.pkl", "Fetal_Human_Pituitary.pkl",
        "Fetal_Human_Retina.pkl", "Human_Developmental_Retina.pkl",
        "Cells_Fetal_Lung.pkl", "Human_Embryonic_YolkSac.pkl",
        "Developing_Human_Gonads.pkl", "Nuclei_Human_InnerEar.pkl",
    ],
}

# Tissue-specific model weights
TISSUE_WEIGHTS = {
    "breast": {
        "immune": 0.3,      # Downweight immune-focused models
        "tissue": 1.0,      # Full weight for tissue models
        "neural": 0.1,      # Very low for neural
        "lung_disease": 0.2,
        "lymphoid": 0.5,
        "other": 0.3,
    },
    "lung": {
        "immune": 0.5,
        "tissue": 0.8,
        "neural": 0.1,
        "lung_disease": 1.0,  # High weight for lung disease models
        "lymphoid": 0.5,
        "other": 0.3,
    },
    "lymph_node": {
        "immune": 1.0,      # Full weight for immune
        "tissue": 0.5,
        "neural": 0.1,
        "lung_disease": 0.3,
        "lymphoid": 1.0,    # Full weight for lymphoid
        "other": 0.3,
    },
    "colon": {
        "immune": 0.4,
        "tissue": 1.0,
        "neural": 0.1,
        "lung_disease": 0.2,
        "lymphoid": 0.5,
        "other": 0.3,
    },
    "default": {
        "immune": 0.5,
        "tissue": 1.0,
        "neural": 0.3,
        "lung_disease": 0.3,
        "lymphoid": 0.5,
        "other": 0.5,
    },
}


def get_model_category(model_name: str) -> str:
    """Get the category for a model."""
    for category, models in MODEL_CATEGORIES.items():
        if model_name in models:
            return category
    return "other"


def get_model_weight(model_name: str, tissue_type: str) -> float:
    """Get the weight for a model based on tissue type."""
    category = get_model_category(model_name)
    weights = TISSUE_WEIGHTS.get(tissue_type, TISSUE_WEIGHTS["default"])
    return weights.get(category, 0.5)


def filter_available_models(models: list[str]) -> list[str]:
    """Filter models to only those available."""
    import celltypist
    available = []
    for model in models:
        try:
            celltypist.models.download_models(model=model, force_update=False)
            available.append(model)
        except Exception:
            pass
    return available


def load_and_normalize(dataset_path: Path, platform: str = "xenium") -> ad.AnnData:
    """Load data and normalize for CellTypist."""
    if platform == "xenium":
        reader = XeniumDataReader(dataset_path)
        expr_matrix, gene_names, cell_ids = reader.load_expression_matrix()
    else:
        from dapidl.data.merscope import MerscopeDataReader
        reader = MerscopeDataReader(dataset_path)
        expr_matrix, gene_names, cell_ids = reader.load_expression_matrix()

    adata = ad.AnnData(X=expr_matrix)
    adata.var_names = gene_names
    adata.obs_names = [str(c) for c in cell_ids]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


def run_all_celltypist(adata: ad.AnnData, models: list[str]) -> dict[str, list]:
    """Run all CellTypist models and return predictions with confidences."""
    import celltypist

    results = {}

    for model_name in models:
        safe_name = model_name.replace(".pkl", "").replace(".", "_")
        try:
            model = celltypist.models.Model.load(model=model_name)
            predictions = celltypist.annotate(adata, model=model, majority_voting=False)

            labels = predictions.predicted_labels["predicted_labels"].tolist()
            probs = predictions.probability_matrix
            confidences = probs.max(axis=1).tolist()

            broad = [map_to_broad_category(l) for l in labels]

            results[model_name] = {
                "labels": labels,
                "broad": broad,
                "confidences": confidences,
            }

        except Exception as e:
            logger.warning(f"Failed {model_name}: {e}")

    return results


def weighted_consensus(
    predictions: dict[str, dict],
    tissue_type: str,
    voting_strategy: str = "tissue_weighted",
) -> tuple[list[str], list[float]]:
    """Compute weighted consensus vote.

    Args:
        predictions: Dict of model_name -> {labels, broad, confidences}
        tissue_type: Type of tissue for weighting
        voting_strategy: 'equal', 'tissue_weighted', or 'confidence_weighted'

    Returns:
        (consensus_labels, consensus_confidences)
    """
    if not predictions:
        return [], []

    n_cells = len(next(iter(predictions.values()))["broad"])
    consensus = []
    confidence = []

    for i in range(n_cells):
        votes = defaultdict(float)

        for model_name, data in predictions.items():
            pred = data["broad"][i]
            if pred == "Unknown":
                continue

            if voting_strategy == "equal":
                weight = 1.0
            elif voting_strategy == "tissue_weighted":
                weight = get_model_weight(model_name, tissue_type)
            elif voting_strategy == "confidence_weighted":
                weight = data["confidences"][i]
            elif voting_strategy == "combined":
                tissue_weight = get_model_weight(model_name, tissue_type)
                conf_weight = data["confidences"][i]
                weight = tissue_weight * conf_weight
            else:
                weight = 1.0

            votes[pred] += weight

        if not votes:
            consensus.append("Unknown")
            confidence.append(0.0)
        else:
            winner = max(votes, key=votes.get)
            total_weight = sum(votes.values())
            consensus.append(winner)
            confidence.append(votes[winner] / total_weight)

    return consensus, confidence


def evaluate_voting_strategies(
    dataset_name: str,
    dataset_path: Path,
    tissue_type: str,
    sample_size: int = 5000,
    platform: str = "xenium",
) -> dict:
    """Evaluate different voting strategies on a dataset."""

    logger.info(f"\n{'='*60}")
    logger.info(f"DATASET: {dataset_name} (tissue: {tissue_type})")
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
    logger.info(f"Using {adata.n_obs} cells")

    # Get all available human models
    all_models = [m for cat in MODEL_CATEGORIES.values() for m in cat]
    available_models = filter_available_models(all_models)
    logger.info(f"Running {len(available_models)} models...")

    # Run all models
    predictions = run_all_celltypist(adata, available_models)
    logger.info(f"Got predictions from {len(predictions)} models")

    results = {
        "dataset": dataset_name,
        "tissue_type": tissue_type,
        "n_cells": adata.n_obs,
        "n_models": len(predictions),
        "strategies": {},
    }

    # Test different voting strategies
    for strategy in ["equal", "tissue_weighted", "confidence_weighted", "combined"]:
        logger.info(f"Computing {strategy} consensus...")
        consensus, conf = weighted_consensus(predictions, tissue_type, strategy)

        dist = Counter(consensus)
        results["strategies"][strategy] = {
            "distribution": dict(dist),
            "unknown_count": dist.get("Unknown", 0),
            "mean_confidence": np.mean(conf) if conf else 0,
            "epithelial_pct": dist.get("Epithelial", 0) / len(consensus) * 100,
            "immune_pct": dist.get("Immune", 0) / len(consensus) * 100,
            "stromal_pct": dist.get("Stromal", 0) / len(consensus) * 100,
        }

    # Print summary
    logger.info(f"\n--- Voting Strategy Comparison for {dataset_name} ---")
    print(f"\n{'Strategy':<20} {'Epithelial':<12} {'Immune':<12} {'Stromal':<12} {'Confidence':<12}")
    print("-" * 68)
    for strategy, data in results["strategies"].items():
        print(f"{strategy:<20} {data['epithelial_pct']:>10.1f}% {data['immune_pct']:>10.1f}% {data['stromal_pct']:>10.1f}% {data['mean_confidence']:>10.3f}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Smart weighted ensemble voting")
    parser.add_argument("--sample-size", type=int, default=5000)
    args = parser.parse_args()

    # Datasets with tissue types
    datasets = [
        ("xenium_breast_rep1", Path("~/datasets/raw/xenium/breast_tumor_rep1").expanduser(), "breast"),
        ("xenium_breast_rep2", Path("~/datasets/raw/xenium/breast_tumor_rep2").expanduser(), "breast"),
        ("xenium_lung", Path("~/datasets/raw/xenium/lung_2fov").expanduser(), "lung"),
        ("xenium_lymph_node", Path("~/datasets/raw/xenium/lymph_node_normal").expanduser(), "lymph_node"),
        ("xenium_colon_cancer", Path("~/datasets/raw/xenium/colon_cancer_colon-panel").expanduser(), "colon"),
    ]

    all_results = {}

    for name, path, tissue in datasets:
        if not path.exists():
            logger.warning(f"Skipping {name}: not found")
            continue
        result = evaluate_voting_strategies(name, path, tissue, args.sample_size)
        all_results[name] = result

    # Save results
    output_path = Path("smart_ensemble_results/summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Final comparison table
    print("\n" + "=" * 100)
    print("SMART ENSEMBLE VOTING - FINAL COMPARISON")
    print("=" * 100)
    print(f"\n{'Dataset':<25} {'Strategy':<20} {'Epithelial':<12} {'Immune':<12} {'Stromal':<12}")
    print("-" * 81)

    for dataset, result in all_results.items():
        if "error" in result:
            continue
        for strategy in ["equal", "tissue_weighted", "combined"]:
            data = result["strategies"][strategy]
            marker = "***" if strategy == "tissue_weighted" else "   "
            print(f"{dataset:<25} {strategy:<20} {data['epithelial_pct']:>10.1f}% {data['immune_pct']:>10.1f}% {data['stromal_pct']:>10.1f}%{marker}")
        print("-" * 81)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
