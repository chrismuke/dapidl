"""Automatic CellTypist model selection without ground truth.

This module implements automatic model selection using:
1. Confidence score distribution
2. Marker gene enrichment (AUCell-inspired)
3. Spatial coherence validation
4. Proportion plausibility checks

The AutoModelSelector can:
- Score individual models
- Select the best model(s) for a tissue
- Build weighted consensus from top models

Usage:
    from dapidl.pipeline.components.annotators.auto_selector import AutoModelSelector

    selector = AutoModelSelector(tissue_type="breast_tumor")
    best_models = selector.select_models(adata, n_models=3)
    consensus = selector.build_consensus(adata, models=best_models)
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

TISSUE_MODELS = {
    "breast_tumor": [
        "Cells_Adult_Breast.pkl",
        "Human_Colorectal_Cancer.pkl",
        "Immune_All_High.pkl",
        "Pan_Fetal_Human.pkl",
        "Adult_Human_Vascular.pkl",
        "Healthy_Human_Liver.pkl",
        "Fetal_Human_AdrenalGlands.pkl",
    ],
    "lymph_node": [
        "Immune_All_High.pkl",
        "Immune_All_Low.pkl",
        "Cells_Human_Tonsil.pkl",
        "Pan_Fetal_Human.pkl",
        "COVID19_Immune_Landscape.pkl",
    ],
    "liver": [
        "Healthy_Human_Liver.pkl",
        "Immune_All_High.pkl",
        "Pan_Fetal_Human.pkl",
        "Human_Colorectal_Cancer.pkl",
    ],
    "lung": [
        "Human_Lung_Atlas.pkl",
        "Cells_Lung_Airway.pkl",
        "Immune_All_High.pkl",
        "Pan_Fetal_Human.pkl",
    ],
    "skin": [
        "Adult_Human_Skin.pkl",
        "Fetal_Human_Skin.pkl",
        "Immune_All_High.pkl",
        "Pan_Fetal_Human.pkl",
    ],
    "brain": [
        "Developing_Human_Brain.pkl",
        "Adult_Human_MTG.pkl",
        "Adult_Human_PrefrontalCortex.pkl",
    ],
    "generic": [
        "Immune_All_High.pkl",
        "Pan_Fetal_Human.pkl",
        "Human_Colorectal_Cancer.pkl",
        "Healthy_Human_Liver.pkl",
        "Human_Lung_Atlas.pkl",
    ],
}

# Expected proportions by tissue (for plausibility checks)
TISSUE_PROPORTIONS = {
    "breast_tumor": {
        "Epithelial": (0.30, 0.70),
        "Immune": (0.10, 0.40),
        "Stromal": (0.10, 0.30),
    },
    "lymph_node": {
        "Epithelial": (0.00, 0.10),
        "Immune": (0.70, 0.95),
        "Stromal": (0.05, 0.20),
    },
    "liver": {
        "Epithelial": (0.60, 0.80),
        "Immune": (0.10, 0.25),
        "Stromal": (0.05, 0.20),
    },
    "lung": {
        "Epithelial": (0.40, 0.70),
        "Immune": (0.15, 0.35),
        "Stromal": (0.10, 0.25),
    },
    "generic": {
        "Epithelial": (0.30, 0.60),
        "Immune": (0.10, 0.40),
        "Stromal": (0.10, 0.30),
    },
}

# Canonical marker genes for validation
CANONICAL_MARKERS = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "MUC1"],
    "Immune": ["PTPRC", "CD3D", "CD3E", "CD19", "MS4A1", "CD68", "CD163", "NKG7"],
    "Stromal": ["COL1A1", "COL1A2", "DCN", "LUM", "PECAM1", "VWF", "ACTA2"],
    "T_cell": ["CD3D", "CD3E", "CD4", "CD8A", "CD8B"],
    "B_cell": ["CD19", "MS4A1", "CD79A", "CD79B"],
    "Macrophage": ["CD68", "CD163", "CSF1R", "MARCO"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "FAP"],
}

# Pattern-based mapping to broad categories
BROAD_CATEGORY_PATTERNS = {
    "Epithelial": [
        "epithelial", "keratinocyte", "hepatocyte", "ductal", "luminal",
        "basal", "alveolar", "secretory", "goblet", "enterocyte", "colonocyte",
        "cholangiocyte", "acinar", "ciliated", "club", "tuft", "paneth",
        "absorptive", "transit", "stem", "progenitor", "myoepithelial",
        "lumm", "lums",  # CellTypist breast
    ],
    "Immune": [
        "t cell", "b cell", "nk cell", "macrophage", "monocyte", "dendritic",
        "mast", "neutrophil", "eosinophil", "plasma", "lymphocyte", "immune",
        "myeloid", "cd4", "cd8", "treg", "th1", "th2", "memory", "naive",
        "effector", "regulatory", "helper", "cytotoxic", "innate", "adaptive",
        "granulocyte", "basophil", "langerhans", "microglia", "kupffer",
        "follicular", "germinal", "marginal zone", "plasmablast",
    ],
    "Stromal": [
        "fibroblast", "endothelial", "pericyte", "smooth muscle", "stromal",
        "vascular", "mesenchymal", "adipocyte", "stellate", "myofibroblast",
        "caf", "cancer-associated", "lymphatic", "blood vessel", "capillary",
        "venous", "arterial", "tip cell", "stalk cell",
    ],
}


def map_to_broad(cell_type: str) -> str:
    """Map fine-grained cell type to broad category."""
    cell_type_lower = cell_type.lower()
    for broad, patterns in BROAD_CATEGORY_PATTERNS.items():
        if any(p in cell_type_lower for p in patterns):
            return broad
    return "Other"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ModelScore:
    """Score for a single CellTypist model."""

    model_name: str
    # Confidence metrics
    mean_confidence: float = 0.0
    median_confidence: float = 0.0
    pct_high_confidence: float = 0.0  # % above 0.5
    # Marker validation
    marker_enrichment_score: float = 0.0
    marker_details: dict[str, float] = field(default_factory=dict)
    # Spatial coherence
    spatial_coherence_score: float = 0.0
    # Proportion plausibility
    proportion_score: float = 0.0
    predicted_proportions: dict[str, float] = field(default_factory=dict)
    # Composite
    composite_score: float = 0.0
    # Metadata
    n_cell_types: int = 0
    error: str | None = None


@dataclass
class ConsensusResult:
    """Result from consensus annotation."""

    annotations_df: pl.DataFrame
    model_scores: list[ModelScore]
    best_model: str
    stats: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Marker Gene Validation (AUCell-inspired)
# =============================================================================


def compute_marker_enrichment(
    adata: Any,
    predictions: np.ndarray,
    available_genes: set[str],
) -> tuple[float, dict[str, float]]:
    """Compute marker gene enrichment score (AUCell-inspired).

    For each predicted cell type, check if marker genes are enriched.
    Uses ranking-based approach for robustness to normalization.

    Returns:
        (enrichment_score, details_dict)
    """
    import numpy as np

    # Map predictions to broad categories
    broad_preds = np.array([map_to_broad(p) for p in predictions])

    enrichments = []
    details = {}

    for broad_cat in ["Epithelial", "Immune", "Stromal"]:
        markers = CANONICAL_MARKERS.get(broad_cat, [])
        present_markers = [m for m in markers if m in available_genes]

        if not present_markers:
            continue

        # Get cells predicted as this type
        mask = broad_preds == broad_cat
        if mask.sum() < 10:
            continue

        for marker in present_markers:
            try:
                # Get expression
                if hasattr(adata.X, "toarray"):
                    expr = adata[:, marker].X.toarray().flatten()
                else:
                    expr = np.array(adata[:, marker].X).flatten()

                expr_in_type = expr[mask]
                expr_in_others = expr[~mask]

                # Fold change with pseudocount
                fc = (np.mean(expr_in_type) + 0.1) / (np.mean(expr_in_others) + 0.1)
                enrichments.append(fc)
                details[f"{broad_cat}_{marker}"] = float(fc)

            except Exception:
                continue

    if not enrichments:
        return 50.0, details  # Neutral score if no markers

    # Score: % of markers with >1.5x enrichment
    good_enrichment = np.mean([e > 1.5 for e in enrichments]) * 100
    mean_fc = np.mean(enrichments)

    # Combine
    score = 0.7 * good_enrichment + 0.3 * min(mean_fc * 10, 100)

    return float(score), details


def compute_spatial_coherence(
    adata: Any,
    predictions: np.ndarray,
    n_neighbors: int = 10,
    n_sample: int = 5000,
) -> float:
    """Compute spatial coherence score.

    Cells of the same type should cluster spatially.
    """
    from scipy.spatial import cKDTree

    # Find spatial coordinates
    x_col = y_col = None
    for xc, yc in [("x_centroid", "y_centroid"), ("X", "Y"), ("centroid_x", "centroid_y")]:
        if xc in adata.obs.columns:
            x_col, y_col = xc, yc
            break

    if x_col is None:
        return 50.0  # Neutral if no spatial data

    try:
        coords = adata.obs[[x_col, y_col]].values
        tree = cKDTree(coords)

        # Map to broad categories
        broad_preds = np.array([map_to_broad(p) for p in predictions])

        # Sample for speed
        n_sample = min(n_sample, len(adata))
        sample_idx = np.random.choice(len(adata), n_sample, replace=False)

        same_type_fractions = []
        for idx in sample_idx:
            _, neighbor_idx = tree.query(coords[idx], k=n_neighbors + 1)
            neighbor_idx = neighbor_idx[1:]  # Exclude self
            same_type = np.mean(broad_preds[neighbor_idx] == broad_preds[idx])
            same_type_fractions.append(same_type)

        # Compare to random
        n_categories = len(np.unique(broad_preds))
        random_baseline = 1.0 / max(n_categories, 1)
        spatial_enrichment = np.mean(same_type_fractions) / max(random_baseline, 0.01)

        # Convert to 0-100 score
        score = min(spatial_enrichment * 25, 100)
        return float(score)

    except Exception as e:
        logger.warning(f"Spatial coherence failed: {e}")
        return 50.0


def compute_proportion_score(
    predictions: np.ndarray,
    expected_proportions: dict[str, tuple[float, float]],
) -> float:
    """Score based on biological plausibility of proportions."""
    broad_preds = np.array([map_to_broad(p) for p in predictions])

    # Compute observed proportions
    unique, counts = np.unique(broad_preds, return_counts=True)
    observed = dict(zip(unique, counts / len(broad_preds)))

    scores = []
    for cell_type, (low, high) in expected_proportions.items():
        obs_prop = observed.get(cell_type, 0)

        if low <= obs_prop <= high:
            scores.append(100)
        elif obs_prop < low:
            distance = low - obs_prop
            scores.append(max(0, 100 - distance * 200))
        else:
            distance = obs_prop - high
            scores.append(max(0, 100 - distance * 200))

    return float(np.mean(scores)) if scores else 50.0


# =============================================================================
# Model Scoring
# =============================================================================


def score_model(
    adata: Any,
    model_name: str,
    expected_proportions: dict[str, tuple[float, float]],
) -> ModelScore:
    """Score a single CellTypist model on quality metrics."""
    import celltypist
    from celltypist import models

    try:
        import scanpy as sc

        # Load model
        model = models.Model.load(model_name)

        # Normalize for CellTypist
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)

        # Run prediction
        result = celltypist.annotate(
            adata_norm,
            model=model,
            majority_voting=True,
        )

        predictions = result.predicted_labels["majority_voting"].values
        prob_matrix = result.probability_matrix
        max_conf = prob_matrix.max(axis=1).values

        # Confidence scores
        mean_conf = float(np.mean(max_conf))
        median_conf = float(np.median(max_conf))
        pct_high = float(np.mean(max_conf > 0.5) * 100)

        # Marker enrichment
        available_genes = set(adata.var_names)
        marker_score, marker_details = compute_marker_enrichment(
            adata, predictions, available_genes
        )

        # Spatial coherence
        spatial_score = compute_spatial_coherence(adata, predictions)

        # Proportion plausibility
        proportion_score = compute_proportion_score(predictions, expected_proportions)

        # Predicted proportions
        broad_preds = np.array([map_to_broad(p) for p in predictions])
        unique, counts = np.unique(broad_preds, return_counts=True)
        pred_proportions = {k: float(v / len(broad_preds)) for k, v in zip(unique, counts)}

        # Composite score (weighted by empirical importance)
        composite = (
            0.40 * pct_high +  # Confidence (strongest predictor)
            0.30 * marker_score +  # Marker enrichment
            0.15 * spatial_score +  # Spatial coherence
            0.15 * proportion_score  # Proportion plausibility
        )

        return ModelScore(
            model_name=model_name,
            mean_confidence=mean_conf,
            median_confidence=median_conf,
            pct_high_confidence=pct_high,
            marker_enrichment_score=marker_score,
            marker_details=marker_details,
            spatial_coherence_score=spatial_score,
            proportion_score=proportion_score,
            predicted_proportions=pred_proportions,
            composite_score=composite,
            n_cell_types=len(np.unique(predictions)),
        )

    except Exception as e:
        logger.warning(f"Failed to score {model_name}: {e}")
        return ModelScore(model_name=model_name, error=str(e))


# =============================================================================
# AutoModelSelector Class
# =============================================================================


class AutoModelSelector:
    """Automatic CellTypist model selection without ground truth.

    Uses confidence scores, marker enrichment, and spatial coherence
    to select the best model(s) for a given tissue type.

    Example:
        selector = AutoModelSelector(tissue_type="breast_tumor")

        # Select best models
        models = selector.select_models(adata, n_models=3)

        # Build consensus
        result = selector.build_consensus(adata, models=models)
    """

    def __init__(
        self,
        tissue_type: str = "generic",
        candidate_models: list[str] | None = None,
        sample_size: int = 20000,
        n_workers: int = 4,
    ):
        """Initialize the model selector.

        Args:
            tissue_type: One of the predefined tissue types
            candidate_models: Override default candidates
            sample_size: Sample size for scoring (for speed)
            n_workers: Number of parallel workers
        """
        self.tissue_type = tissue_type
        self.sample_size = sample_size
        self.n_workers = n_workers

        if candidate_models:
            self.candidate_models = candidate_models
        else:
            self.candidate_models = TISSUE_MODELS.get(
                tissue_type, TISSUE_MODELS["generic"]
            )

        self.expected_proportions = TISSUE_PROPORTIONS.get(
            tissue_type, TISSUE_PROPORTIONS["generic"]
        )

    def select_models(
        self,
        adata: Any,
        n_models: int = 3,
    ) -> list[ModelScore]:
        """Select the best models for the given data.

        Args:
            adata: AnnData object with expression data
            n_models: Number of models to select

        Returns:
            List of ModelScore objects, sorted by composite_score
        """
        import scanpy as sc

        logger.info(f"AutoModelSelector: Scoring {len(self.candidate_models)} models")
        logger.info(f"Tissue type: {self.tissue_type}")

        # Sample if needed
        if len(adata) > self.sample_size:
            logger.info(f"Sampling {self.sample_size:,} cells from {len(adata):,}")
            sample_idx = np.random.choice(len(adata), self.sample_size, replace=False)
            adata_sample = adata[sample_idx].copy()
        else:
            adata_sample = adata

        # Score all models
        scores = []
        for i, model_name in enumerate(self.candidate_models):
            logger.info(f"[{i+1}/{len(self.candidate_models)}] Scoring {model_name}...")
            score = score_model(adata_sample, model_name, self.expected_proportions)

            if score.error is None:
                scores.append(score)
                logger.info(
                    f"  → Composite: {score.composite_score:.1f} | "
                    f"Conf: {score.pct_high_confidence:.1f}% | "
                    f"Markers: {score.marker_enrichment_score:.1f}"
                )
            else:
                logger.warning(f"  → Failed: {score.error}")

        # Sort by composite score
        scores.sort(key=lambda x: x.composite_score, reverse=True)

        # Log results
        logger.info("\n=== Model Selection Results ===")
        for i, s in enumerate(scores[:n_models], 1):
            logger.info(f"{i}. {s.model_name}: {s.composite_score:.1f}")

        return scores[:n_models]

    def build_consensus(
        self,
        adata: Any,
        models: list[ModelScore] | list[str] | None = None,
        min_agreement: float = 0.5,
        confidence_weight: bool = True,
    ) -> ConsensusResult:
        """Build consensus annotation from multiple models.

        Args:
            adata: AnnData object with expression data
            models: Models to use (from select_models or model names)
            min_agreement: Minimum agreement threshold
            confidence_weight: Weight votes by model confidence

        Returns:
            ConsensusResult with annotations and statistics
        """
        import celltypist
        from celltypist import models as ct_models
        import scanpy as sc

        # Determine models to use
        if models is None:
            model_names = self.candidate_models[:5]
        elif isinstance(models[0], ModelScore):
            model_names = [m.model_name for m in models]
        else:
            model_names = models

        logger.info(f"Building consensus from {len(model_names)} models: {model_names}")

        # Normalize once
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)

        # Run all models
        results = []
        for model_name in model_names:
            try:
                model = ct_models.Model.load(model_name)
                pred = celltypist.annotate(
                    adata_norm, model=model, majority_voting=True
                )
                predictions = pred.predicted_labels["majority_voting"].values
                confidence = pred.probability_matrix.max(axis=1).values
                broad_preds = np.array([map_to_broad(p) for p in predictions])

                results.append({
                    "model": model_name,
                    "predictions": predictions,
                    "broad": broad_preds,
                    "confidence": confidence,
                })
                logger.info(f"  ✓ {model_name}")
            except Exception as e:
                logger.warning(f"  ✗ {model_name}: {e}")

        if len(results) < 2:
            raise ValueError("Need at least 2 successful models for consensus")

        # Build voting matrix
        n_cells = len(adata)
        broad_categories = ["Epithelial", "Immune", "Stromal", "Other"]
        votes = np.zeros((n_cells, len(broad_categories)))
        confidence_sum = np.zeros((n_cells, len(broad_categories)))

        for r in results:
            for i, cat in enumerate(broad_categories):
                mask = r["broad"] == cat
                if confidence_weight:
                    votes[mask, i] += r["confidence"][mask]
                    confidence_sum[mask, i] += r["confidence"][mask]
                else:
                    votes[mask, i] += 1
                    confidence_sum[mask, i] += r["confidence"][mask]

        # Normalize
        if confidence_weight:
            total_conf = sum(r["confidence"] for r in results)
            votes_normalized = votes / (total_conf[:, np.newaxis] + 1e-10)
        else:
            votes_normalized = votes / len(results)

        # Get consensus
        consensus_idx = np.argmax(votes_normalized, axis=1)
        consensus_broad = np.array([broad_categories[i] for i in consensus_idx])
        consensus_score = np.max(votes_normalized, axis=1)

        # Count raw agreement
        raw_votes = np.zeros((n_cells, len(broad_categories)))
        for r in results:
            for i, cat in enumerate(broad_categories):
                raw_votes[r["broad"] == cat, i] += 1
        n_models_agree = np.max(raw_votes, axis=1).astype(int)

        # Average confidence of agreeing models
        consensus_confidence = np.zeros(n_cells)
        for i in range(n_cells):
            cat = consensus_broad[i]
            cat_idx = broad_categories.index(cat)
            if raw_votes[i, cat_idx] > 0:
                consensus_confidence[i] = confidence_sum[i, cat_idx] / raw_votes[i, cat_idx]

        # Fine-grained consensus
        fine_predictions = []
        for i in range(n_cells):
            cat = consensus_broad[i]
            fine_votes = {}
            for r in results:
                if r["broad"][i] == cat:
                    fine = r["predictions"][i]
                    weight = r["confidence"][i] if confidence_weight else 1
                    fine_votes[fine] = fine_votes.get(fine, 0) + weight
            if fine_votes:
                fine_predictions.append(max(fine_votes, key=fine_votes.get))
            else:
                fine_predictions.append("Unknown")

        # Build result DataFrame
        annotations_df = pl.DataFrame({
            "cell_id": range(len(adata)),
            "consensus_broad": consensus_broad,
            "consensus_fine": fine_predictions,
            "consensus_score": consensus_score,
            "consensus_confidence": consensus_confidence,
            "n_models_agree": n_models_agree,
            "is_high_confidence": (consensus_score >= min_agreement) &
                                   (consensus_confidence >= 0.3),
        })

        # Add per-model predictions
        for r in results:
            model_short = r["model"].replace(".pkl", "")
            # Convert to list/array to handle Categorical types from CellTypist
            preds = list(r["predictions"]) if hasattr(r["predictions"], "__iter__") else r["predictions"]
            broad = list(r["broad"]) if hasattr(r["broad"], "__iter__") else r["broad"]
            annotations_df = annotations_df.with_columns([
                pl.Series(f"{model_short}_pred", preds),
                pl.Series(f"{model_short}_broad", broad),
                pl.Series(f"{model_short}_conf", r["confidence"]),
            ])

        # Statistics
        n_high_conf = (annotations_df["is_high_confidence"]).sum()
        stats = {
            "n_cells": n_cells,
            "n_models": len(results),
            "n_high_confidence": n_high_conf,
            "pct_high_confidence": n_high_conf / n_cells * 100,
            "broad_distribution": dict(zip(
                *np.unique(consensus_broad, return_counts=True)
            )),
        }

        # Get model scores if we have them
        model_scores = []
        if isinstance(models, list) and len(models) > 0 and isinstance(models[0], ModelScore):
            model_scores = models

        return ConsensusResult(
            annotations_df=annotations_df,
            model_scores=model_scores,
            best_model=model_names[0] if model_names else "",
            stats=stats,
        )


# =============================================================================
# Pipeline Integration
# =============================================================================


def get_auto_selector_config() -> dict[str, Any]:
    """Return configuration schema for auto model selection."""
    return {
        "tissue_type": {
            "type": "string",
            "enum": list(TISSUE_MODELS.keys()),
            "default": "generic",
            "description": "Tissue type for model selection",
        },
        "n_models": {
            "type": "integer",
            "default": 3,
            "minimum": 1,
            "maximum": 10,
            "description": "Number of models for consensus",
        },
        "min_agreement": {
            "type": "number",
            "default": 0.5,
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Minimum agreement for high-confidence",
        },
        "sample_size": {
            "type": "integer",
            "default": 20000,
            "description": "Sample size for model scoring",
        },
    }
