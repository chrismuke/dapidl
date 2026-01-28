#!/usr/bin/env python3
"""
Comprehensive GT-Free Cell Type Annotation Validation

Combines multiple validation methods:
1. Marker gene validation (canonical markers from CellMarker/PanglaoDB)
2. Leiden clustering (transcriptomic structure)
3. UMAP visualization + spatial coherence
4. Cross-method consensus (popV-style)
5. Proportion plausibility

Usage:
    uv run python scripts/run_comprehensive_validation.py -d rep1 -o results/validation
    uv run python scripts/run_comprehensive_validation.py -d rep1 rep2 -o results/validation
"""

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import click
import numpy as np
import polars as pl
import scanpy as sc
from loguru import logger
from scipy.spatial import KDTree
from scipy.stats import entropy
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Import our validation modules
from dapidl.validation.marker_validation import validate_with_markers, BREAST_MARKERS


# Dataset paths
DATASET_PATHS = {
    "rep1": {
        "xenium": Path.home() / "datasets/raw/xenium/breast_tumor_rep1",
        "platform": "xenium",
    },
    "rep2": {
        "xenium": Path.home() / "datasets/raw/xenium/breast_tumor_rep2",
        "platform": "xenium",
    },
    "merscope": {
        "merscope": Path.home() / "datasets/raw/merscope/breast",
        "platform": "merscope",
    },
}

# Expected proportions for breast tumor (literature-based)
EXPECTED_PROPORTIONS = {
    "Epithelial": (0.30, 0.70),
    "Immune": (0.10, 0.40),
    "Stromal": (0.10, 0.35),
    "Endothelial": (0.02, 0.15),
    "Unknown": (0.00, 0.25),
}

# Coarse mapping for various cell type predictions
COARSE_MAPPING = {
    # Epithelial
    "epithelial": "Epithelial", "luminal": "Epithelial", "basal": "Epithelial",
    "tumor": "Epithelial", "cancer": "Epithelial", "carcinoma": "Epithelial",
    "keratinocyte": "Epithelial", "secretory": "Epithelial", "dcis": "Epithelial",
    "invasive": "Epithelial", "myoepithelial": "Epithelial",
    # Immune
    "immune": "Immune", "t cell": "Immune", "b cell": "Immune", "nk": "Immune",
    "macrophage": "Immune", "monocyte": "Immune", "dendritic": "Immune",
    "mast": "Immune", "neutrophil": "Immune", "lymphocyte": "Immune",
    "plasma": "Immune", "myeloid": "Immune", "cd4": "Immune", "cd8": "Immune",
    # Stromal
    "stromal": "Stromal", "fibroblast": "Stromal", "pericyte": "Stromal",
    "smooth muscle": "Stromal", "mesenchymal": "Stromal", "caf": "Stromal",
    "adipocyte": "Stromal",
    # Endothelial
    "endothelial": "Endothelial", "vascular": "Endothelial", "lymphatic": "Endothelial",
}


@dataclass
class ValidationResult:
    """Complete validation result for one method."""
    method: str
    dataset: str

    # Marker validation
    marker_score: float = 0.0
    marker_details: dict = field(default_factory=dict)

    # Leiden clustering
    leiden_ari: float = 0.0
    leiden_nmi: float = 0.0
    leiden_resolution: float = 0.5

    # Spatial coherence
    spatial_coherence: float = 0.0
    mean_local_entropy: float = 0.0

    # Proportion plausibility
    proportion_score: float = 0.0
    proportions: dict = field(default_factory=dict)

    # Coverage
    unknown_rate: float = 0.0
    n_cells: int = 0
    n_classes: int = 0

    # Cross-method agreement (filled later)
    cross_method_agreement: float = 0.0

    # Overall score
    overall_score: float = 0.0

    def compute_overall(self, weights: dict | None = None):
        """Compute weighted overall score."""
        if weights is None:
            weights = {
                "marker": 0.30,
                "leiden": 0.20,
                "spatial": 0.15,
                "proportion": 0.15,
                "unknown": 0.10,
                "agreement": 0.10,
            }

        self.overall_score = (
            weights["marker"] * self.marker_score +
            weights["leiden"] * max(0, self.leiden_ari) +  # ARI can be negative
            weights["spatial"] * self.spatial_coherence +
            weights["proportion"] * self.proportion_score +
            weights["unknown"] * (1.0 - self.unknown_rate) +
            weights["agreement"] * self.cross_method_agreement
        )


def load_xenium_data(xenium_path: Path) -> sc.AnnData:
    """Load Xenium data into AnnData."""
    logger.info(f"Loading Xenium data from {xenium_path}")

    h5_path = xenium_path / "cell_feature_matrix.h5"
    adata = sc.read_10x_h5(h5_path)

    # Load cell coordinates
    cells_path = xenium_path / "cells.parquet"
    if cells_path.exists():
        cells_df = pl.read_parquet(cells_path)
        coords = cells_df.select(["x_centroid", "y_centroid"]).to_numpy()
        adata.obsm["spatial"] = coords

    # Basic preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def load_merscope_data(merscope_path: Path, max_cells: int = 200000) -> sc.AnnData:
    """Load MERSCOPE data into AnnData.

    Args:
        merscope_path: Path to MERSCOPE output directory
        max_cells: Maximum cells to load (MERSCOPE has ~700K cells)
    """
    from dapidl.data.merscope import MerscopeDataReader

    logger.info(f"Loading MERSCOPE data from {merscope_path}")

    reader = MerscopeDataReader(merscope_path)
    expression, gene_names, cell_ids = reader.load_expression_matrix()

    # Subsample if too many cells
    if len(cell_ids) > max_cells:
        logger.info(f"Subsampling from {len(cell_ids)} to {max_cells} cells")
        idx = np.random.choice(len(cell_ids), max_cells, replace=False)
        expression = expression[idx]
        cell_ids = cell_ids[idx]

    # Create AnnData
    adata = sc.AnnData(X=expression)
    adata.obs_names = [str(cid) for cid in cell_ids]
    adata.var_names = gene_names

    # Load coordinates for subsampled cells
    cells_df = reader.cells_df
    coords = cells_df.filter(pl.col("cell_id").is_in(cell_ids)).select(
        ["center_x", "center_y"]
    ).to_numpy()
    if len(coords) == len(cell_ids):
        adata.obsm["spatial"] = coords
    else:
        logger.warning("Could not match cell IDs for spatial coordinates")

    # Basic preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def map_to_coarse(predictions: np.ndarray) -> np.ndarray:
    """Map predictions to coarse categories."""
    def map_single(pred: str) -> str:
        pred_lower = pred.lower().replace("_", " ").replace("-", " ")
        for pattern, category in COARSE_MAPPING.items():
            if pattern in pred_lower:
                return category
        return "Unknown"

    return np.array([map_single(p) for p in predictions])


def run_annotation_methods(adata: sc.AnnData) -> dict[str, np.ndarray]:
    """Run multiple annotation methods."""
    import celltypist
    from celltypist import models

    predictions = {}

    # CellTypist - Breast model
    logger.info("Running CellTypist (Cells_Adult_Breast)...")
    try:
        model = models.Model.load("Cells_Adult_Breast.pkl")
        result = celltypist.annotate(adata, model=model, majority_voting=False)
        preds = result.predicted_labels["predicted_labels"].values
        predictions["celltypist_breast"] = preds
        predictions["celltypist_breast_coarse"] = map_to_coarse(preds)
    except Exception as e:
        logger.warning(f"CellTypist Breast failed: {e}")

    # CellTypist - Immune model
    logger.info("Running CellTypist (Immune_All_High)...")
    try:
        model = models.Model.load("Immune_All_High.pkl")
        result = celltypist.annotate(adata, model=model, majority_voting=False)
        preds = result.predicted_labels["predicted_labels"].values
        predictions["celltypist_immune"] = preds
        predictions["celltypist_immune_coarse"] = map_to_coarse(preds)
    except Exception as e:
        logger.warning(f"CellTypist Immune failed: {e}")

    # CellTypist - Pan Tissue
    logger.info("Running CellTypist (Pan_Tissue)...")
    try:
        model = models.Model.load("Immune_All_Low.pkl")
        result = celltypist.annotate(adata, model=model, majority_voting=False)
        preds = result.predicted_labels["predicted_labels"].values
        predictions["celltypist_immune_low"] = preds
        predictions["celltypist_immune_low_coarse"] = map_to_coarse(preds)
    except Exception as e:
        logger.warning(f"CellTypist Pan Tissue failed: {e}")

    # SingleR via our annotator
    logger.info("Running SingleR (HPCA + Blueprint)...")
    try:
        from dapidl.pipeline.components.annotators.singler import SingleRAnnotator

        annotator = SingleRAnnotator()
        # HPCA reference
        result_hpca = annotator._run_singler(adata, "hpca")
        preds_hpca = result_hpca["broad_category"].to_numpy()
        predictions["singler_hpca"] = preds_hpca
        predictions["singler_hpca_coarse"] = preds_hpca  # Already coarse

        # Blueprint reference
        result_bp = annotator._run_singler(adata, "blueprint")
        preds_bp = result_bp["broad_category"].to_numpy()
        predictions["singler_blueprint"] = preds_bp
        predictions["singler_blueprint_coarse"] = preds_bp  # Already coarse
    except Exception as e:
        logger.warning(f"SingleR failed: {e}")
        import traceback
        traceback.print_exc()

    return predictions


def compute_leiden_validation(
    adata: sc.AnnData,
    predictions: np.ndarray,
    resolutions: list[float] = [0.3, 0.5, 0.8, 1.0],
    max_cells: int = 50000,
) -> tuple[float, float, float]:
    """Compare predictions with Leiden clustering.

    NOTE: Temporarily disabled due to memory issues with PCA computation.
    Returns default values until fixed.
    """
    logger.warning("Leiden validation temporarily disabled due to memory issues. Skipping.")
    return 0.0, 0.0, 0.5


def compute_spatial_coherence(
    coords: np.ndarray,
    predictions: np.ndarray,
    k_neighbors: int = 10,
    n_sample: int = 10000,
) -> tuple[float, float]:
    """Compute spatial coherence of predictions."""
    if coords is None:
        return 0.5, 0.0

    # Build KD-tree
    tree = KDTree(coords)

    # Sample cells
    n_sample = min(n_sample, len(predictions))
    sample_idx = np.random.choice(len(predictions), n_sample, replace=False)

    local_entropies = []
    class_names = list(set(predictions))
    n_classes = len(class_names)

    for idx in sample_idx:
        _, neighbor_idx = tree.query(coords[idx], k=k_neighbors + 1)
        neighbor_idx = neighbor_idx[1:]  # Exclude self

        neighbor_labels = predictions[neighbor_idx]
        counts = Counter(neighbor_labels)
        probs = np.array([counts.get(c, 0) / k_neighbors for c in class_names])
        local_entropy = entropy(probs + 1e-10)
        local_entropies.append(local_entropy)

    mean_entropy = np.mean(local_entropies)
    max_entropy = np.log(n_classes)

    # Coherence = 1 - normalized entropy
    coherence = 1.0 - (mean_entropy / max_entropy) if max_entropy > 0 else 0.5

    return coherence, mean_entropy


def compute_proportion_plausibility(
    predictions: np.ndarray,
    expected: dict[str, tuple[float, float]] = EXPECTED_PROPORTIONS,
) -> tuple[float, dict]:
    """Check if proportions are biologically plausible."""
    counts = Counter(predictions)
    total = len(predictions)
    proportions = {k: v / total for k, v in counts.items()}

    scores = []
    for cell_type, (min_p, max_p) in expected.items():
        matching = [k for k in proportions if cell_type.lower() in k.lower()]
        actual = sum(proportions.get(t, 0) for t in matching)

        if min_p <= actual <= max_p:
            score = 1.0
        elif actual < min_p:
            score = max(0, actual / min_p) if min_p > 0 else 0
        else:
            score = max(0, 1 - (actual - max_p) / max_p) if max_p > 0 else 0

        scores.append(score)

    return np.mean(scores) if scores else 0.5, proportions


def compute_cross_method_agreement(
    all_predictions: dict[str, np.ndarray],
    target_method: str,
) -> float:
    """Compute agreement with other methods (popV-style)."""
    if len(all_predictions) < 2:
        return 0.5

    target = all_predictions[target_method]
    others = [v for k, v in all_predictions.items() if k != target_method and len(v) == len(target)]

    if not others:
        return 0.5

    agreements = [(target == other).mean() for other in others]
    return np.mean(agreements)


def validate_method(
    adata: sc.AnnData,
    predictions: np.ndarray,
    method_name: str,
    dataset_name: str,
    all_coarse_predictions: dict[str, np.ndarray],
) -> ValidationResult:
    """Run full validation for one method."""
    logger.info(f"Validating {method_name}...")

    result = ValidationResult(
        method=method_name,
        dataset=dataset_name,
        n_cells=len(predictions),
        n_classes=len(set(predictions)),
    )

    # Map to coarse for some metrics
    coarse_preds = map_to_coarse(predictions) if "_coarse" not in method_name else predictions

    # 1. Marker validation
    logger.info("  Running marker validation...")
    marker_result = validate_with_markers(adata, predictions, tissue_type="breast")
    result.marker_score = marker_result["weighted_marker_score"]
    result.marker_details = marker_result

    # 2. Leiden clustering
    logger.info("  Running Leiden validation...")
    result.leiden_ari, result.leiden_nmi, result.leiden_resolution = compute_leiden_validation(
        adata, coarse_preds
    )

    # 3. Spatial coherence
    logger.info("  Computing spatial coherence...")
    coords = adata.obsm.get("spatial", None)
    result.spatial_coherence, result.mean_local_entropy = compute_spatial_coherence(
        coords, coarse_preds
    )

    # 4. Proportion plausibility
    logger.info("  Checking proportion plausibility...")
    result.proportion_score, result.proportions = compute_proportion_plausibility(coarse_preds)

    # 5. Unknown rate
    unknown_mask = np.isin(predictions, ["Unknown", "unknown", "Unlabeled", "unassigned"])
    result.unknown_rate = unknown_mask.mean()

    # 6. Cross-method agreement
    if all_coarse_predictions:
        result.cross_method_agreement = compute_cross_method_agreement(
            all_coarse_predictions,
            method_name.replace("_coarse", "") + "_coarse" if "_coarse" not in method_name else method_name
        )

    # Compute overall score
    result.compute_overall()

    logger.info(f"  {method_name}: Overall={result.overall_score:.3f} "
                f"(marker={result.marker_score:.3f}, leiden={result.leiden_ari:.3f}, "
                f"spatial={result.spatial_coherence:.3f}, unk={result.unknown_rate:.1%})")

    return result


def run_validation_for_dataset(dataset_name: str, output_dir: Path) -> list[ValidationResult]:
    """Run full validation for one dataset."""
    logger.info(f"\n{'='*70}")
    logger.info(f"VALIDATING DATASET: {dataset_name}")
    logger.info(f"{'='*70}")

    paths = DATASET_PATHS.get(dataset_name)
    if not paths:
        logger.error(f"Unknown dataset: {dataset_name}")
        return []

    # Load data based on platform
    platform = paths.get("platform", "xenium")
    if platform == "merscope":
        adata = load_merscope_data(paths["merscope"])
    else:
        adata = load_xenium_data(paths["xenium"])

    # Run annotation methods
    predictions = run_annotation_methods(adata)

    if not predictions:
        logger.error("No annotation methods succeeded")
        return []

    # Get coarse predictions for cross-method agreement
    coarse_predictions = {k: v for k, v in predictions.items() if "_coarse" in k}

    # Validate each method
    results = []
    for method_name, preds in predictions.items():
        if "_coarse" in method_name:
            continue  # Skip coarse versions, we map internally

        result = validate_method(
            adata, preds, method_name, dataset_name, coarse_predictions
        )
        results.append(result)

    # Sort by overall score
    results.sort(key=lambda x: x.overall_score, reverse=True)

    return results


def print_summary(results: list[ValidationResult], dataset_name: str):
    """Print validation summary."""
    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY: {dataset_name}")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'Overall':>8} {'Marker':>8} {'Leiden':>8} "
          f"{'Spatial':>8} {'Prop':>8} {'Unk%':>8}")
    print("-"*80)

    for r in results:
        print(f"{r.method:<25} {r.overall_score:>8.3f} {r.marker_score:>8.3f} "
              f"{r.leiden_ari:>8.3f} {r.spatial_coherence:>8.3f} "
              f"{r.proportion_score:>8.3f} {r.unknown_rate:>8.1%}")

    print("-"*80)
    print(f"\nRECOMMENDED: {results[0].method} (score: {results[0].overall_score:.3f})")
    print()


@click.command()
@click.option("-d", "--datasets", multiple=True, default=["rep1"],
              help="Datasets to validate (rep1, rep2)")
@click.option("-o", "--output", type=click.Path(), default="benchmark_results/validation",
              help="Output directory")
def main(datasets: tuple[str, ...], output: str):
    """Run comprehensive GT-free validation."""
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset in datasets:
        results = run_validation_for_dataset(dataset, output_path)
        all_results[dataset] = results

        # Print summary
        print_summary(results, dataset)

        # Save results
        results_dict = [asdict(r) for r in results]

        with open(output_path / f"validation_{dataset}.json", "w") as f:
            json.dump(results_dict, f, indent=2, default=str)

        # Save CSV
        df = pl.DataFrame([
            {
                "dataset": r.dataset,
                "method": r.method,
                "overall_score": r.overall_score,
                "marker_score": r.marker_score,
                "leiden_ari": r.leiden_ari,
                "leiden_nmi": r.leiden_nmi,
                "spatial_coherence": r.spatial_coherence,
                "proportion_score": r.proportion_score,
                "unknown_rate": r.unknown_rate,
                "n_cells": r.n_cells,
                "n_classes": r.n_classes,
            }
            for r in results
        ])
        df.write_csv(output_path / f"validation_{dataset}.csv")

    logger.info(f"\nResults saved to {output_path}")

    # Print cross-dataset comparison if multiple datasets
    if len(datasets) > 1:
        print("\n" + "="*80)
        print("CROSS-DATASET COMPARISON")
        print("="*80)

        for method in ["celltypist_breast", "celltypist_immune", "singler_hpca", "singler_blueprint"]:
            scores = []
            for dataset, results in all_results.items():
                for r in results:
                    if r.method == method:
                        scores.append((dataset, r.overall_score))

            if scores:
                print(f"\n{method}:")
                for ds, score in scores:
                    print(f"  {ds}: {score:.3f}")


if __name__ == "__main__":
    main()
