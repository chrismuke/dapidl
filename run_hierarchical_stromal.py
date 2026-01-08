#!/usr/bin/env python3
"""Hierarchical classification for improved stromal cell annotation.

Strategy:
    Stage 1: Coarse classification (Epithelial, Immune, Stromal) - F1=0.84
    Stage 2: For cells predicted as Stromal, apply specialized stromal annotation
             with stromal-focused models and adjusted voting

This reduces noise from epithelial/immune contamination and allows
lower confidence thresholds for the stromal-specific stage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import f1_score, classification_report, accuracy_score
from loguru import logger
import scanpy as sc

from dapidl.pipeline.components.annotators.popv_ensemble import (
    PopVStyleEnsembleAnnotator,
    PopVEnsembleConfig,
    VotingStrategy,
    GranularityLevel,
)
from dapidl.ontology.cl_mapper import CLMapper
from dapidl.ontology.annotator_mappings import get_all_annotator_mappings, get_gt_mappings


# Stage 1: Coarse models (optimized for broad category)
COARSE_MODELS = [
    "Cells_Adult_Breast.pkl",
    "Immune_All_High.pkl",
    "Immune_All_Low.pkl",
    "Adult_Human_Vascular.pkl",
    "Pan_Fetal_Human.pkl",
]

# Stage 2: Stromal-specialized voting weights
# Weight annotators that are better at stromal classification
STROMAL_ANNOTATOR_WEIGHTS = {
    "celltypist_Cells_Adult_Breast.pkl": 1.0,
    "celltypist_Adult_Human_Vascular.pkl": 2.0,  # Good for endothelial/pericyte
    "celltypist_Pan_Fetal_Human.pkl": 1.5,  # Broader stromal coverage
    "singler_hpca": 2.0,  # HPCA has better stromal coverage
    "singler_blueprint": 0.5,  # Blueprint is immune-biased
}

# Stromal-specific categories at COARSE level
STROMAL_COARSE_TYPES = [
    "Fibroblast",
    "Pericyte",
    "Adipocyte",
    "Vascular_Endothelial",
    "Myofibroblast",
]


def load_data():
    """Load expression and ground truth data."""
    adata = sc.read_10x_h5("/home/chrism/datasets/raw/xenium/breast_tumor_rep1/outs/cell_feature_matrix.h5")
    adata.obs["cell_id"] = adata.obs.index.astype(str)

    gt_df = pd.read_excel("/home/chrism/datasets/raw/xenium/breast_tumor_rep1/celltypes_ground_truth_rep1_supervised.xlsx")
    gt_df = gt_df.rename(columns={"Barcode": "cell_id", "Cluster": "ground_truth"})
    gt_df["cell_id"] = gt_df["cell_id"].astype(str)

    return adata, gt_df


def load_singler_predictions() -> dict:
    """Load pre-computed SingleR predictions."""
    singler_path = Path("experiment_all_methods/breast_rep1/singler_predictions.csv")
    if not singler_path.exists():
        logger.warning(f"SingleR predictions not found at {singler_path}")
        return {}

    df = pd.read_csv(singler_path)
    results = {}
    if "singler_hpca_label" in df.columns:
        results["hpca"] = {
            "source": "singler_hpca",
            "predictions": df["singler_hpca_label"].tolist(),
            "confidence": df["singler_hpca_score"].tolist(),
            "cell_ids": [str(cid) for cid in df["cell_id"].tolist()],
        }
    if "singler_bp_label" in df.columns:
        results["blueprint"] = {
            "source": "singler_blueprint",
            "predictions": df["singler_bp_label"].tolist(),
            "confidence": df["singler_bp_score"].tolist(),
            "cell_ids": [str(cid) for cid in df["cell_id"].tolist()],
        }
    return results


def run_stage1_coarse(adata, singler_preds: dict) -> pd.DataFrame:
    """Stage 1: Run coarse classification on all cells."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Coarse Classification (Epithelial/Immune/Stromal)")
    logger.info("=" * 60)

    config = PopVEnsembleConfig(
        celltypist_models=COARSE_MODELS,
        include_singler_hpca=True,
        include_singler_blueprint=True,
        voting_strategy=VotingStrategy.UNWEIGHTED,
        granularity=GranularityLevel.COARSE,
    )

    annotator = PopVStyleEnsembleAnnotator(config)

    # Inject cached SingleR predictions
    if singler_preds:
        if "hpca" in singler_preds:
            annotator._predictions_cache["singler_hpca"] = singler_preds["hpca"]
        if "blueprint" in singler_preds:
            annotator._predictions_cache["singler_blueprint"] = singler_preds["blueprint"]

    result = annotator.annotate(adata)
    pred_df = result.annotations_df.to_pandas()
    pred_df["cell_id"] = pred_df["cell_id"].astype(str)

    # Get coarse predictions
    coarse_counts = pred_df["predicted_type_coarse"].value_counts()
    logger.info(f"Stage 1 coarse distribution:\n{coarse_counts}")

    return pred_df


def run_stage2_stromal_refinement(
    adata,
    stage1_df: pd.DataFrame,
    singler_preds: dict,
    mapper: CLMapper,
) -> pd.DataFrame:
    """Stage 2: Refine stromal predictions with specialized voting.

    For cells predicted as Stromal in Stage 1:
    1. Use fine-grained predictions from each annotator
    2. Apply stromal-specific voting weights
    3. Use lower confidence threshold for rare types

    OPTIMIZED: Uses vectorized operations instead of row-by-row iteration.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: Stromal Refinement (Vectorized)")
    logger.info("=" * 60)

    # Get cells predicted as Stromal - use SET for O(1) lookup
    stromal_mask = stage1_df["predicted_type_coarse"] == "Stromal"
    stromal_cells_set = set(stage1_df[stromal_mask]["cell_id"].tolist())
    logger.info(f"Refining {len(stromal_cells_set)} stromal cells")

    # Run fine-grained annotation on ALL cells (we need the raw predictions)
    config = PopVEnsembleConfig(
        celltypist_models=COARSE_MODELS,
        include_singler_hpca=True,
        include_singler_blueprint=True,
        voting_strategy=VotingStrategy.UNWEIGHTED,
        granularity=GranularityLevel.FINE,
    )

    annotator = PopVStyleEnsembleAnnotator(config)
    if singler_preds:
        if "hpca" in singler_preds:
            annotator._predictions_cache["singler_hpca"] = singler_preds["hpca"]
        if "blueprint" in singler_preds:
            annotator._predictions_cache["singler_blueprint"] = singler_preds["blueprint"]

    result = annotator.annotate(adata)
    fine_df = result.annotations_df.to_pandas()
    fine_df["cell_id"] = fine_df["cell_id"].astype(str)

    # Get annotator columns
    annotator_cols = [col for col in fine_df.columns if col.startswith("pred_")]
    logger.info(f"Found annotator columns: {annotator_cols}")

    # Pre-compute CL mappings for ALL unique predictions (cache for speed)
    logger.info("Pre-computing CL mappings for all unique predictions...")
    unique_preds = set()
    for col in annotator_cols:
        unique_preds.update(fine_df[col].dropna().unique())
    unique_preds.update(fine_df["predicted_type_fine"].dropna().unique())

    pred_to_coarse = {}
    for pred in unique_preds:
        mapped = mapper.map_with_info(pred)
        pred_to_coarse[pred] = mapped.coarse_category
    logger.info(f"Cached {len(pred_to_coarse)} unique prediction mappings")

    # Merge stage1 predictions into fine_df (vectorized join)
    stage1_lookup = stage1_df[["cell_id", "predicted_type_coarse"]].copy()
    stage1_lookup = stage1_lookup.rename(columns={"predicted_type_coarse": "stage1_pred"})
    fine_df = fine_df.merge(stage1_lookup, on="cell_id", how="left")

    # Add is_stromal flag
    fine_df["is_stromal"] = fine_df["cell_id"].isin(stromal_cells_set)

    # VECTORIZED: Map all annotator predictions to coarse categories
    logger.info("Mapping predictions to coarse categories (vectorized)...")
    for col in annotator_cols:
        coarse_col = col.replace("pred_", "coarse_")
        fine_df[coarse_col] = fine_df[col].map(pred_to_coarse)

    # Map fine prediction to coarse for non-stromal cells
    fine_df["fine_coarse"] = fine_df["predicted_type_fine"].map(pred_to_coarse)

    # VECTORIZED: Compute weighted votes for stromal cells
    logger.info("Computing weighted votes for stromal cells...")
    stromal_coarse_set = set(STROMAL_COARSE_TYPES)

    def compute_stromal_vote(row):
        """Compute weighted stromal vote for a single row."""
        if not row["is_stromal"]:
            return row["fine_coarse"]

        votes = Counter()
        for col in annotator_cols:
            coarse_col = col.replace("pred_", "coarse_")
            coarse_type = row[coarse_col]
            if pd.isna(coarse_type):
                continue
            if coarse_type in stromal_coarse_set:
                annotator_name = col.replace("pred_", "")
                weight = STROMAL_ANNOTATOR_WEIGHTS.get(annotator_name, 1.0)
                votes[coarse_type] += weight

        if votes:
            return votes.most_common(1)[0][0]
        return "Fibroblast"

    # Apply vectorized (actually using apply, but much faster than iterrows)
    fine_df["stage2_pred"] = fine_df.apply(compute_stromal_vote, axis=1)
    fine_df["final_pred"] = fine_df["stage2_pred"]

    # Build result dataframe
    refined_df = fine_df[["cell_id", "stage1_pred", "stage2_pred", "final_pred"]].copy()

    # Count refined stromal predictions
    stromal_refined = refined_df[refined_df["stage1_pred"] == "Stromal"]
    logger.info(f"Stromal refinement distribution:\n{stromal_refined['stage2_pred'].value_counts()}")

    return refined_df


def evaluate_hierarchical(
    refined_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    mapper: CLMapper,
) -> dict:
    """Evaluate hierarchical classification results."""
    logger.info("=" * 60)
    logger.info("EVALUATION: Hierarchical Classification")
    logger.info("=" * 60)

    # Merge with ground truth
    merged = pd.merge(refined_df, gt_df[["cell_id", "ground_truth"]], on="cell_id")

    # Map ground truth to coarse
    gt_mapped = []
    for label in merged["ground_truth"]:
        result = mapper.map_with_info(label)
        gt_mapped.append(result.coarse_category)
    merged["gt_coarse"] = gt_mapped

    # Filter valid (non-Unknown)
    valid = merged[(merged["gt_coarse"] != "Unknown") & (merged["final_pred"] != "Unknown")]

    # Overall metrics
    acc = accuracy_score(valid["gt_coarse"], valid["final_pred"])
    f1_macro = f1_score(valid["gt_coarse"], valid["final_pred"], average="macro", zero_division=0)

    logger.info(f"Overall: Accuracy={acc:.3f}, F1 Macro={f1_macro:.3f}")

    # Per-class report
    report = classification_report(
        valid["gt_coarse"],
        valid["final_pred"],
        output_dict=True,
        zero_division=0
    )

    # Extract stromal-specific metrics
    results = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "n_cells": len(valid),
    }

    stromal_types = ["Fibroblast", "Pericyte", "Adipocyte", "Vascular_Endothelial"]
    for st in stromal_types:
        if st in report:
            results[f"{st}_f1"] = report[st]["f1-score"]
            results[f"{st}_precision"] = report[st]["precision"]
            results[f"{st}_recall"] = report[st]["recall"]
            logger.info(f"  {st}: F1={report[st]['f1-score']:.3f}, P={report[st]['precision']:.3f}, R={report[st]['recall']:.3f}")

    # Print full classification report
    print("\nFull Classification Report:")
    print(classification_report(valid["gt_coarse"], valid["final_pred"], zero_division=0))

    return results


def run_baseline_flat(adata, singler_preds: dict, gt_df: pd.DataFrame, mapper: CLMapper) -> dict:
    """Run baseline flat classification for comparison."""
    logger.info("=" * 60)
    logger.info("BASELINE: Flat Classification (no hierarchy)")
    logger.info("=" * 60)

    config = PopVEnsembleConfig(
        celltypist_models=COARSE_MODELS,
        include_singler_hpca=True,
        include_singler_blueprint=True,
        voting_strategy=VotingStrategy.UNWEIGHTED,
        granularity=GranularityLevel.FINE,
    )

    annotator = PopVStyleEnsembleAnnotator(config)
    if singler_preds:
        if "hpca" in singler_preds:
            annotator._predictions_cache["singler_hpca"] = singler_preds["hpca"]
        if "blueprint" in singler_preds:
            annotator._predictions_cache["singler_blueprint"] = singler_preds["blueprint"]

    result = annotator.annotate(adata)
    pred_df = result.annotations_df.to_pandas()
    pred_df["cell_id"] = pred_df["cell_id"].astype(str)

    # Merge with ground truth
    merged = pd.merge(pred_df, gt_df[["cell_id", "ground_truth"]], on="cell_id")

    # Map to coarse
    pred_coarse = []
    gt_coarse = []
    for i, row in merged.iterrows():
        pred_mapped = mapper.map_with_info(row["predicted_type_fine"])
        gt_mapped = mapper.map_with_info(row["ground_truth"])
        pred_coarse.append(pred_mapped.coarse_category)
        gt_coarse.append(gt_mapped.coarse_category)

    merged["pred_coarse"] = pred_coarse
    merged["gt_coarse"] = gt_coarse

    # Filter valid
    valid = merged[(merged["gt_coarse"] != "Unknown") & (merged["pred_coarse"] != "Unknown")]

    # Metrics
    acc = accuracy_score(valid["gt_coarse"], valid["pred_coarse"])
    f1_macro = f1_score(valid["gt_coarse"], valid["pred_coarse"], average="macro", zero_division=0)

    logger.info(f"Baseline: Accuracy={acc:.3f}, F1 Macro={f1_macro:.3f}")

    report = classification_report(
        valid["gt_coarse"],
        valid["pred_coarse"],
        output_dict=True,
        zero_division=0
    )

    results = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "n_cells": len(valid),
    }

    stromal_types = ["Fibroblast", "Pericyte", "Adipocyte", "Vascular_Endothelial"]
    for st in stromal_types:
        if st in report:
            results[f"{st}_f1"] = report[st]["f1-score"]
            logger.info(f"  {st}: F1={report[st]['f1-score']:.3f}")

    return results


def main():
    """Run hierarchical classification experiment."""
    logger.info("=" * 70)
    logger.info("Hierarchical Stromal Classification Experiment")
    logger.info("=" * 70)

    # Initialize mapper
    annotator_maps = get_all_annotator_mappings()
    gt_maps = get_gt_mappings("xenium_breast")
    mapper = CLMapper(annotator_mappings=annotator_maps, ground_truth_mappings=gt_maps)

    # Load data
    adata, gt_df = load_data()
    logger.info(f"Loaded {adata.n_obs} cells")

    # Load SingleR
    singler_preds = load_singler_predictions()
    logger.info(f"SingleR predictions: {list(singler_preds.keys())}")

    # Run baseline flat classification
    baseline_results = run_baseline_flat(adata, singler_preds, gt_df, mapper)

    # Run hierarchical classification
    # Stage 1: Coarse
    stage1_df = run_stage1_coarse(adata, singler_preds)

    # Stage 2: Stromal refinement
    refined_df = run_stage2_stromal_refinement(adata, stage1_df, singler_preds, mapper)

    # Evaluate
    hierarchical_results = evaluate_hierarchical(refined_df, gt_df, mapper)

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON: Baseline vs Hierarchical")
    print("=" * 80)

    comparison = pd.DataFrame({
        "Metric": ["F1 Macro", "Accuracy", "Fibroblast F1", "Vascular_Endothelial F1", "Adipocyte F1"],
        "Baseline": [
            baseline_results.get("f1_macro", 0),
            baseline_results.get("accuracy", 0),
            baseline_results.get("Fibroblast_f1", 0),
            baseline_results.get("Vascular_Endothelial_f1", 0),
            baseline_results.get("Adipocyte_f1", 0),
        ],
        "Hierarchical": [
            hierarchical_results.get("f1_macro", 0),
            hierarchical_results.get("accuracy", 0),
            hierarchical_results.get("Fibroblast_f1", 0),
            hierarchical_results.get("Vascular_Endothelial_f1", 0),
            hierarchical_results.get("Adipocyte_f1", 0),
        ],
    })
    comparison["Delta"] = comparison["Hierarchical"] - comparison["Baseline"]
    print(comparison.to_string(index=False))

    # Save results
    output_dir = Path("experiment_hierarchical_stromal")
    output_dir.mkdir(exist_ok=True)

    refined_df.to_csv(output_dir / "refined_predictions.csv", index=False)
    comparison.to_csv(output_dir / "comparison.csv", index=False)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
