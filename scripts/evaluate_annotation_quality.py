#!/usr/bin/env python3
"""
GT-Free Annotation Quality Evaluation

Evaluate cell type annotation quality WITHOUT ground truth using:
1. Marker gene expression validation
2. Spatial coherence (neighborhood consistency)
3. Cross-method agreement
4. Biological plausibility (expected proportions)

Usage:
    uv run python scripts/evaluate_annotation_quality.py -x /path/to/xenium -o results/
"""

import click
import numpy as np
import polars as pl
import scanpy as sc
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
from scipy.spatial import KDTree
from scipy.stats import entropy
from collections import Counter
import json


# Canonical markers for breast tissue
CANONICAL_MARKERS = {
    "Epithelial": ["EPCAM", "CDH1", "KRT8", "KRT18", "KRT19", "KRT7", "MUC1"],
    "Immune": ["PTPRC", "CD3D", "CD3E", "CD4", "CD8A", "CD14", "CD68", "MS4A1", "CD19", "CD79A"],
    "Stromal": ["COL1A1", "COL1A2", "ACTA2", "PDGFRA", "PDGFRB", "VIM", "FAP", "DCN"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "KDR"],
    "T_Cell": ["CD3D", "CD3E", "CD4", "CD8A", "CD8B"],
    "B_Cell": ["MS4A1", "CD19", "CD79A", "CD79B"],
    "Macrophage": ["CD68", "CD163", "CSF1R", "MARCO"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA"],
}

# Expected proportions for breast tumor (literature-based)
EXPECTED_PROPORTIONS_BREAST = {
    "Epithelial": (0.30, 0.70),  # 30-70%
    "Immune": (0.10, 0.40),      # 10-40%
    "Stromal": (0.10, 0.35),     # 10-35%
    "Endothelial": (0.02, 0.15), # 2-15%
    "Unknown": (0.00, 0.20),     # Should be low
}


@dataclass
class AnnotationQualityScore:
    """Quality scores for an annotation method."""
    method: str
    marker_score: float        # 0-1, higher = better marker expression
    spatial_coherence: float   # 0-1, higher = more spatially coherent
    cross_method_agreement: float  # 0-1, agreement with other methods
    proportion_plausibility: float # 0-1, how plausible are proportions
    unknown_rate: float        # 0-1, lower = better (fewer unknowns)
    overall_score: float       # Weighted combination
    details: dict


def load_xenium_adata(xenium_path: Path) -> sc.AnnData:
    """Load Xenium data into AnnData."""
    h5_path = xenium_path / "cell_feature_matrix.h5"
    adata = sc.read_10x_h5(h5_path)

    # Load cell coordinates
    cells_path = xenium_path / "cells.parquet"
    if cells_path.exists():
        cells_df = pl.read_parquet(cells_path)
        # Match cell IDs
        adata.obs["cell_id"] = adata.obs.index
        adata.obsm["spatial"] = cells_df.select(["x_centroid", "y_centroid"]).to_numpy()

    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


def calculate_marker_score(
    adata: sc.AnnData,
    predictions: np.ndarray,
    class_names: list[str],
) -> tuple[float, dict]:
    """
    Calculate marker gene expression score.

    For each predicted cell type, check if cells express expected markers.
    """
    scores = {}
    details = {}

    for cell_type in class_names:
        if cell_type in ["Unknown", "unknown", "Unlabeled"]:
            continue

        # Find canonical markers for this cell type
        markers = None
        for canonical_type, canonical_markers in CANONICAL_MARKERS.items():
            if canonical_type.lower() in cell_type.lower() or cell_type.lower() in canonical_type.lower():
                markers = canonical_markers
                break

        if markers is None:
            continue

        # Find which markers are in the gene panel
        available_markers = [m for m in markers if m in adata.var_names]
        if not available_markers:
            continue

        # Get cells predicted as this type
        mask = predictions == cell_type
        if mask.sum() == 0:
            continue

        # Calculate mean expression of markers in predicted cells vs all cells
        marker_expr_predicted = adata[mask, available_markers].X.mean()
        marker_expr_all = adata[:, available_markers].X.mean()

        # Score: how much higher is expression in predicted cells?
        if marker_expr_all > 0:
            enrichment = marker_expr_predicted / marker_expr_all
            score = min(enrichment / 2.0, 1.0)  # Normalize to 0-1
        else:
            score = 0.5

        scores[cell_type] = score
        details[cell_type] = {
            "markers_found": available_markers,
            "enrichment": float(enrichment) if marker_expr_all > 0 else None,
            "n_cells": int(mask.sum()),
        }

    overall_score = np.mean(list(scores.values())) if scores else 0.0
    return overall_score, details


def calculate_spatial_coherence(
    coords: np.ndarray,
    predictions: np.ndarray,
    k_neighbors: int = 10,
) -> tuple[float, dict]:
    """
    Calculate spatial coherence of predictions.

    Cell types should form spatially coherent regions, not random patterns.
    Uses local entropy of cell type labels.
    """
    if coords is None or len(coords) == 0:
        return 0.5, {"error": "No spatial coordinates"}

    # Build KD-tree for spatial queries
    tree = KDTree(coords)

    # Sample cells for efficiency (full calculation too slow)
    n_sample = min(10000, len(predictions))
    sample_idx = np.random.choice(len(predictions), n_sample, replace=False)

    local_entropies = []
    class_names = list(set(predictions))
    n_classes = len(class_names)

    for idx in sample_idx:
        # Get k nearest neighbors
        _, neighbor_idx = tree.query(coords[idx], k=k_neighbors + 1)
        neighbor_idx = neighbor_idx[1:]  # Exclude self

        # Get labels of neighbors
        neighbor_labels = predictions[neighbor_idx]

        # Calculate local entropy
        counts = Counter(neighbor_labels)
        probs = np.array([counts.get(c, 0) / k_neighbors for c in class_names])
        local_entropy = entropy(probs + 1e-10)  # Add small value to avoid log(0)
        local_entropies.append(local_entropy)

    mean_entropy = np.mean(local_entropies)
    max_entropy = np.log(n_classes)  # Maximum possible entropy

    # Coherence = 1 - normalized entropy (higher = more coherent)
    coherence = 1.0 - (mean_entropy / max_entropy) if max_entropy > 0 else 0.5

    return coherence, {
        "mean_local_entropy": float(mean_entropy),
        "max_entropy": float(max_entropy),
        "n_sampled": n_sample,
        "k_neighbors": k_neighbors,
    }


def calculate_cross_method_agreement(
    predictions_dict: dict[str, np.ndarray],
    target_method: str,
) -> tuple[float, dict]:
    """
    Calculate agreement between target method and other methods.
    """
    if len(predictions_dict) < 2:
        return 0.5, {"error": "Need at least 2 methods for comparison"}

    target_preds = predictions_dict[target_method]
    other_methods = [k for k in predictions_dict if k != target_method]

    agreements = []
    details = {}

    for other_method in other_methods:
        other_preds = predictions_dict[other_method]

        # Simple agreement: % of cells with same prediction
        agreement = (target_preds == other_preds).mean()
        agreements.append(agreement)
        details[other_method] = float(agreement)

    return np.mean(agreements), details


def calculate_proportion_plausibility(
    predictions: np.ndarray,
    expected_proportions: dict[str, tuple[float, float]] = EXPECTED_PROPORTIONS_BREAST,
) -> tuple[float, dict]:
    """
    Check if predicted proportions are biologically plausible.
    """
    counts = Counter(predictions)
    total = len(predictions)

    proportions = {k: v / total for k, v in counts.items()}

    scores = []
    details = {}

    for cell_type, (min_prop, max_prop) in expected_proportions.items():
        # Find matching predicted type
        matching_types = [k for k in proportions if cell_type.lower() in k.lower()]
        if not matching_types:
            continue

        actual_prop = sum(proportions.get(t, 0) for t in matching_types)

        # Score based on how close to expected range
        if min_prop <= actual_prop <= max_prop:
            score = 1.0
        elif actual_prop < min_prop:
            score = max(0, actual_prop / min_prop)
        else:  # actual_prop > max_prop
            score = max(0, 1 - (actual_prop - max_prop) / max_prop)

        scores.append(score)
        details[cell_type] = {
            "actual": float(actual_prop),
            "expected_range": (min_prop, max_prop),
            "score": float(score),
        }

    return np.mean(scores) if scores else 0.5, details


def run_annotation_methods(adata: sc.AnnData) -> dict[str, np.ndarray]:
    """Run multiple annotation methods and return predictions."""
    import celltypist
    from celltypist import models
    from dapidl.pipeline.components.annotators.singler import SingleRAnnotator

    predictions = {}

    # CellTypist - Breast model
    logger.info("Running CellTypist (Cells_Adult_Breast)...")
    try:
        model = models.Model.load("Cells_Adult_Breast.pkl")
        result = celltypist.annotate(adata, model=model, majority_voting=False)
        # Map to coarse categories
        preds = result.predicted_labels["predicted_labels"].values
        coarse_preds = map_to_coarse(preds)
        predictions["celltypist_breast"] = coarse_preds
    except Exception as e:
        logger.warning(f"CellTypist failed: {e}")

    # CellTypist - Immune model
    logger.info("Running CellTypist (Immune_All_High)...")
    try:
        model = models.Model.load("Immune_All_High.pkl")
        result = celltypist.annotate(adata, model=model, majority_voting=False)
        preds = result.predicted_labels["predicted_labels"].values
        coarse_preds = map_to_coarse(preds)
        predictions["celltypist_immune"] = coarse_preds
    except Exception as e:
        logger.warning(f"CellTypist Immune failed: {e}")

    # SingleR - HPCA
    logger.info("Running SingleR (HPCA)...")
    try:
        from dapidl.pipeline.config import AnnotationConfig
        config = AnnotationConfig(singler_reference="hpca", fine_grained=False)
        annotator = SingleRAnnotator(config=config)
        result = annotator.annotate(adata=adata)
        predictions["singler_hpca"] = result["broad_category"].to_numpy()
    except Exception as e:
        logger.warning(f"SingleR HPCA failed: {e}")
        import traceback
        traceback.print_exc()

    # SingleR - Blueprint
    logger.info("Running SingleR (Blueprint)...")
    try:
        config = AnnotationConfig(singler_reference="blueprint", fine_grained=False)
        annotator = SingleRAnnotator(config=config)
        result = annotator.annotate(adata=adata)
        predictions["singler_blueprint"] = result["broad_category"].to_numpy()
    except Exception as e:
        logger.warning(f"SingleR Blueprint failed: {e}")
        import traceback
        traceback.print_exc()

    return predictions


def map_to_coarse(predictions: np.ndarray) -> np.ndarray:
    """Map fine-grained predictions to coarse categories."""
    COARSE_MAPPING = {
        # Epithelial
        "epithelial": "Epithelial", "luminal": "Epithelial", "basal": "Epithelial",
        "tumor": "Epithelial", "cancer": "Epithelial", "carcinoma": "Epithelial",
        "keratinocyte": "Epithelial", "secretory": "Epithelial",
        # Immune
        "immune": "Immune", "t cell": "Immune", "b cell": "Immune", "nk": "Immune",
        "macrophage": "Immune", "monocyte": "Immune", "dendritic": "Immune",
        "mast": "Immune", "neutrophil": "Immune", "lymphocyte": "Immune",
        "plasma": "Immune", "myeloid": "Immune",
        # Stromal
        "stromal": "Stromal", "fibroblast": "Stromal", "pericyte": "Stromal",
        "smooth muscle": "Stromal", "mesenchymal": "Stromal", "caf": "Stromal",
        # Endothelial
        "endothelial": "Endothelial", "vascular": "Endothelial",
    }

    def map_single(pred: str) -> str:
        pred_lower = pred.lower()
        for pattern, category in COARSE_MAPPING.items():
            if pattern in pred_lower:
                return category
        return "Unknown"

    return np.array([map_single(p) for p in predictions])


def evaluate_method(
    adata: sc.AnnData,
    predictions: np.ndarray,
    method_name: str,
    all_predictions: dict[str, np.ndarray],
) -> AnnotationQualityScore:
    """Evaluate a single annotation method."""

    class_names = list(set(predictions))

    # 1. Marker score
    marker_score, marker_details = calculate_marker_score(
        adata, predictions, class_names
    )

    # 2. Spatial coherence
    coords = adata.obsm.get("spatial", None)
    spatial_score, spatial_details = calculate_spatial_coherence(
        coords, predictions
    )

    # 3. Cross-method agreement
    agreement_score, agreement_details = calculate_cross_method_agreement(
        all_predictions, method_name
    )

    # 4. Proportion plausibility
    proportion_score, proportion_details = calculate_proportion_plausibility(
        predictions
    )

    # 5. Unknown rate
    unknown_rate = (predictions == "Unknown").mean()
    unknown_score = 1.0 - unknown_rate

    # Overall score (weighted)
    weights = {
        "marker": 0.30,
        "spatial": 0.25,
        "agreement": 0.15,
        "proportion": 0.15,
        "unknown": 0.15,
    }

    overall = (
        weights["marker"] * marker_score +
        weights["spatial"] * spatial_score +
        weights["agreement"] * agreement_score +
        weights["proportion"] * proportion_score +
        weights["unknown"] * unknown_score
    )

    return AnnotationQualityScore(
        method=method_name,
        marker_score=marker_score,
        spatial_coherence=spatial_score,
        cross_method_agreement=agreement_score,
        proportion_plausibility=proportion_score,
        unknown_rate=unknown_rate,
        overall_score=overall,
        details={
            "marker": marker_details,
            "spatial": spatial_details,
            "agreement": agreement_details,
            "proportion": proportion_details,
        }
    )


@click.command()
@click.option("-x", "--xenium-path", type=click.Path(exists=True), required=True)
@click.option("-o", "--output", type=click.Path(), default="annotation_quality_results")
@click.option("--methods", type=str, default="all", help="Methods to evaluate (comma-separated or 'all')")
def main(xenium_path: str, output: str, methods: str):
    """Evaluate annotation quality without ground truth."""

    xenium_path = Path(xenium_path)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Xenium data from {xenium_path}")
    adata = load_xenium_adata(xenium_path)
    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")

    # Run annotation methods
    logger.info("Running annotation methods...")
    predictions = run_annotation_methods(adata)

    if not predictions:
        logger.error("No annotation methods succeeded")
        return

    # Evaluate each method
    logger.info("Evaluating annotation quality...")
    results = []

    for method_name, preds in predictions.items():
        logger.info(f"Evaluating {method_name}...")
        score = evaluate_method(adata, preds, method_name, predictions)
        results.append(score)

        logger.info(f"  Marker Score: {score.marker_score:.3f}")
        logger.info(f"  Spatial Coherence: {score.spatial_coherence:.3f}")
        logger.info(f"  Cross-Method Agreement: {score.cross_method_agreement:.3f}")
        logger.info(f"  Proportion Plausibility: {score.proportion_plausibility:.3f}")
        logger.info(f"  Unknown Rate: {score.unknown_rate:.3f}")
        logger.info(f"  OVERALL SCORE: {score.overall_score:.3f}")

    # Sort by overall score
    results.sort(key=lambda x: x.overall_score, reverse=True)

    # Print summary
    print("\n" + "="*80)
    print("ANNOTATION QUALITY RANKING (No Ground Truth)")
    print("="*80)
    print(f"{'Method':<25} {'Overall':>8} {'Marker':>8} {'Spatial':>8} {'Agree':>8} {'Prop':>8} {'Unk%':>8}")
    print("-"*80)

    for r in results:
        print(f"{r.method:<25} {r.overall_score:>8.3f} {r.marker_score:>8.3f} "
              f"{r.spatial_coherence:>8.3f} {r.cross_method_agreement:>8.3f} "
              f"{r.proportion_plausibility:>8.3f} {r.unknown_rate:>8.1%}")

    print("-"*80)
    print(f"\nRECOMMENDED METHOD: {results[0].method} (score: {results[0].overall_score:.3f})")

    # Save results
    results_dict = {
        "ranking": [
            {
                "rank": i + 1,
                "method": r.method,
                "overall_score": r.overall_score,
                "marker_score": r.marker_score,
                "spatial_coherence": r.spatial_coherence,
                "cross_method_agreement": r.cross_method_agreement,
                "proportion_plausibility": r.proportion_plausibility,
                "unknown_rate": r.unknown_rate,
                "details": r.details,
            }
            for i, r in enumerate(results)
        ],
        "recommended_method": results[0].method,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
    }

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    results_dict = convert_numpy(results_dict)

    with open(output_path / "quality_scores.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
