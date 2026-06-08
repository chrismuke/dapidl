"""Consensus-based cell type annotation with automatic model selection.

This annotator implements:
1. Automatic model selection (when strategy="auto")
2. Multi-model consensus annotation
3. Confidence-weighted voting
4. High-confidence filtering for training data

Usage in pipeline:
    annotator: "consensus"
    strategy: "auto"  # or "fixed" with specific models
    tissue_type: "breast_tumor"
    n_models: 3
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from dapidl.pipeline.base import AnnotationConfig, AnnotationResult
from dapidl.pipeline.components.annotators.mapping import (
    get_class_names,
)
from dapidl.pipeline.registry import register_annotator


@dataclass
class ConsensusConfig:
    """Extended configuration for consensus annotation."""

    # Model selection
    strategy: str = "auto"  # "auto" or "fixed"
    tissue_type: str = "generic"
    model_names: list[str] | None = None  # For fixed strategy
    n_models: int = 3  # Number of models for consensus

    # Consensus settings
    min_agreement: float = 0.5
    confidence_weight: bool = True
    min_confidence: float = 0.3

    # Output settings
    fine_grained: bool = False
    include_per_model: bool = True  # Include per-model predictions

    # Performance
    sample_size: int = 20000  # For auto-selection scoring
    n_workers: int = 4


@register_annotator
class ConsensusAnnotator:
    """Consensus annotator with automatic model selection.

    This annotator provides the highest-quality labels by:
    1. Automatically selecting best models for the tissue (if strategy="auto")
    2. Running multiple models
    3. Building confidence-weighted consensus
    4. Filtering to high-confidence cells

    Example:
        annotator = get_annotator("consensus", config)
        result = annotator.annotate(config, adata=adata)

        # High-confidence cells for training
        hc = result.annotations_df.filter(pl.col("is_high_confidence"))
    """

    name = "consensus"

    def __init__(self, config: AnnotationConfig | ConsensusConfig | None = None):
        """Initialize the consensus annotator."""
        self.config = config
        self._selector = None

    @property
    def selector(self):
        """Lazy-load the AutoModelSelector."""
        if self._selector is None:
            from dapidl.pipeline.components.annotators.auto_selector import (
                AutoModelSelector,
            )

            cfg = self._get_consensus_config()
            self._selector = AutoModelSelector(
                tissue_type=cfg.tissue_type,
                candidate_models=cfg.model_names,
                sample_size=cfg.sample_size,
                n_workers=cfg.n_workers,
            )
        return self._selector

    def _get_consensus_config(self) -> ConsensusConfig:
        """Convert AnnotationConfig to ConsensusConfig."""
        if isinstance(self.config, ConsensusConfig):
            return self.config

        cfg = self.config or AnnotationConfig()
        return ConsensusConfig(
            strategy=getattr(cfg, "strategy", "auto"),
            tissue_type=getattr(cfg, "tissue_type", "generic"),
            model_names=getattr(cfg, "model_names", None),
            n_models=getattr(cfg, "n_models", 3),
            min_agreement=getattr(cfg, "min_agreement", 0.5),
            confidence_weight=getattr(cfg, "confidence_weight", True),
            min_confidence=getattr(cfg, "confidence_threshold", 0.3),
            fine_grained=getattr(cfg, "fine_grained", False),
            include_per_model=getattr(cfg, "include_per_model", True),
            sample_size=getattr(cfg, "sample_size", 20000),
            n_workers=getattr(cfg, "n_workers", 4),
        )

    def annotate(
        self,
        config: AnnotationConfig | ConsensusConfig | None = None,
        adata: Any | None = None,
        expression_path: Path | None = None,
        cells_df: pl.DataFrame | None = None,
    ) -> AnnotationResult:
        """Annotate cells using consensus of multiple models.

        Args:
            config: Annotation configuration
            adata: AnnData object with expression data
            expression_path: Path to expression file
            cells_df: Cell metadata DataFrame

        Returns:
            AnnotationResult with consensus annotations
        """
        import scanpy as sc

        if config is not None:
            self.config = config

        cfg = self._get_consensus_config()

        # Load data if needed
        if adata is None and expression_path is not None:
            logger.info(f"Loading expression from {expression_path}")
            if str(expression_path).endswith(".h5ad"):
                adata = sc.read_h5ad(expression_path)
            elif str(expression_path).endswith(".h5"):
                adata = sc.read_10x_h5(expression_path)
            else:
                raise ValueError(f"Unsupported format: {expression_path}")

        if adata is None:
            raise ValueError("Must provide adata or expression_path")

        # Add spatial coordinates from cells_df if available
        if cells_df is not None:
            for col in ["x_centroid", "y_centroid"]:
                if col in cells_df.columns:
                    adata.obs[col] = cells_df[col].to_numpy()

        logger.info(f"ConsensusAnnotator: {len(adata):,} cells, strategy={cfg.strategy}")

        # Select models
        if cfg.strategy == "auto":
            logger.info("Auto-selecting best models...")
            model_scores = self.selector.select_models(adata, n_models=cfg.n_models)
            model_names = [m.model_name for m in model_scores]
        else:
            model_names = cfg.model_names or self.selector.candidate_models[:cfg.n_models]
            model_scores = []

        logger.info(f"Using models: {model_names}")

        # Build consensus
        from dapidl.pipeline.components.annotators.auto_selector import AutoModelSelector

        selector = AutoModelSelector(
            tissue_type=cfg.tissue_type,
            candidate_models=model_names,
        )

        consensus_result = selector.build_consensus(
            adata,
            models=model_names,
            min_agreement=cfg.min_agreement,
            confidence_weight=cfg.confidence_weight,
        )

        # Build annotation result
        annotations_df = consensus_result.annotations_df

        # Use the appropriate column for class mapping
        type_col = "consensus_fine" if cfg.fine_grained else "consensus_broad"

        # Rename for standard interface
        annotations_df = annotations_df.with_columns([
            pl.col(type_col).alias("predicted_type"),
            pl.col("consensus_broad").alias("broad_category"),
            pl.col("consensus_confidence").alias("confidence"),
        ])

        # Filter out "Other" for class mapping
        valid_types = annotations_df.filter(
            pl.col("broad_category").is_in(["Epithelial", "Immune", "Stromal"])
        )

        # Build class mapping
        class_names = get_class_names(cfg.fine_grained)
        if not cfg.fine_grained:
            class_names = ["Epithelial", "Immune", "Stromal"]

        class_mapping = {name: i for i, name in enumerate(class_names)}
        index_to_class = dict(enumerate(class_names))

        # Statistics
        n_high_conf = annotations_df.filter(pl.col("is_high_confidence")).height
        stats = {
            "n_cells": len(annotations_df),
            "n_high_confidence": n_high_conf,
            "pct_high_confidence": n_high_conf / len(annotations_df) * 100,
            "n_models_used": len(model_names),
            "models_used": model_names,
            "strategy": cfg.strategy,
            "tissue_type": cfg.tissue_type,
            **consensus_result.stats,
        }

        # Log summary
        logger.info("Annotation complete:")
        logger.info(f"  Total cells: {len(annotations_df):,}")
        logger.info(f"  High-confidence: {n_high_conf:,} ({stats['pct_high_confidence']:.1f}%)")

        dist = annotations_df.group_by("broad_category").count()
        for row in dist.iter_rows(named=True):
            logger.info(f"  {row['broad_category']}: {row['count']:,}")

        return AnnotationResult(
            annotations_df=annotations_df,
            class_mapping=class_mapping,
            index_to_class=index_to_class,
            stats=stats,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def annotate_with_auto_consensus(
    adata: Any,
    tissue_type: str = "generic",
    n_models: int = 3,
    min_agreement: float = 0.5,
) -> AnnotationResult:
    """Convenience function for auto-consensus annotation.

    Args:
        adata: AnnData object
        tissue_type: Tissue type for model selection
        n_models: Number of models for consensus
        min_agreement: Minimum agreement threshold

    Returns:
        AnnotationResult with consensus annotations
    """
    config = ConsensusConfig(
        strategy="auto",
        tissue_type=tissue_type,
        n_models=n_models,
        min_agreement=min_agreement,
    )
    annotator = ConsensusAnnotator(config)
    return annotator.annotate(adata=adata)


def get_high_confidence_cells(
    result: AnnotationResult,
    min_agreement: int = 3,
    min_confidence: float = 0.3,
) -> pl.DataFrame:
    """Filter annotation result to high-confidence cells.

    Args:
        result: AnnotationResult from consensus annotation
        min_agreement: Minimum number of agreeing models
        min_confidence: Minimum confidence score

    Returns:
        Filtered DataFrame with high-confidence cells
    """
    df = result.annotations_df

    return df.filter(
        (pl.col("n_models_agree") >= min_agreement) &
        (pl.col("consensus_confidence") >= min_confidence) &
        (pl.col("broad_category").is_in(["Epithelial", "Immune", "Stromal"]))
    )


# =============================================================================
# Naive majority-vote baseline (string-only, no Cell-Ontology hierarchy)
# =============================================================================

def naive_majority_vote(
    method_predictions: dict[str, list[str]],
    cell_ids: list[str] | None = None,
    confidence_scores: dict[str, list[float]] | None = None,
) -> pl.DataFrame:
    """Aggregate per-method cell-type predictions by exact-string majority vote.

    This is the **counterfactual baseline** for popV's ontology-aware ensemble:
    treats labels as opaque strings, no parent-class agreement, no depth tie-break.
    Same input voters as popV → directly measures the lift from CL hierarchy.

    Args:
        method_predictions: {method_name: [str_label_per_cell, ...]} — all lists
            must be the same length and aligned by cell index.
        cell_ids: Optional list of cell IDs (length = n_cells). Defaults to range(n).
        confidence_scores: Optional {method_name: [confidence_per_cell]} for
            confidence-weighted tie-breaking. If omitted, ties resolve to
            first-seen label.

    Returns:
        Polars DataFrame with columns:
            cell_id, predicted_type, n_votes, n_methods, agreement_fraction,
            tied (bool), per-method labels (one column per method).

    Note:
        This function is intentionally simple. The point is to make the
        algorithmic difference vs popV's `_ontology_hierarchical_vote` explicit
        and testable.
    """
    from collections import Counter

    methods = list(method_predictions.keys())
    if not methods:
        raise ValueError("method_predictions must contain at least one method")

    n_cells = len(method_predictions[methods[0]])
    for m, preds in method_predictions.items():
        if len(preds) != n_cells:
            raise ValueError(
                f"All methods must agree on n_cells; {m} has {len(preds)} != {n_cells}"
            )
    if cell_ids is None:
        cell_ids = [str(i) for i in range(n_cells)]
    elif len(cell_ids) != n_cells:
        raise ValueError(f"cell_ids has {len(cell_ids)} entries, expected {n_cells}")

    rows: list[dict] = []
    for i in range(n_cells):
        votes = [method_predictions[m][i] for m in methods]
        # Strip None/empty/"Unknown" from the vote pool
        clean_votes = [v for v in votes if v and v != "Unknown"]
        if not clean_votes:
            rows.append({
                "cell_id": cell_ids[i],
                "predicted_type": "Unknown",
                "n_votes": 0,
                "n_methods": len(methods),
                "agreement_fraction": 0.0,
                "tied": False,
                **{f"vote_{m}": method_predictions[m][i] for m in methods},
            })
            continue

        counts = Counter(clean_votes)
        max_count = max(counts.values())
        winners = [label for label, c in counts.items() if c == max_count]

        if len(winners) == 1:
            chosen = winners[0]
            tied = False
        elif confidence_scores is not None:
            # Tie-break by max confidence among methods voting for each tied label
            best_label = winners[0]
            best_conf = -1.0
            for label in winners:
                conf_for_label = max(
                    confidence_scores[m][i]
                    for m in methods
                    if method_predictions[m][i] == label
                )
                if conf_for_label > best_conf:
                    best_conf = conf_for_label
                    best_label = label
            chosen = best_label
            tied = True
        else:
            # First-seen tie-break (deterministic w.r.t. method ordering)
            chosen = winners[0]
            tied = True

        rows.append({
            "cell_id": cell_ids[i],
            "predicted_type": chosen,
            "n_votes": int(max_count),
            "n_methods": len(clean_votes),
            "agreement_fraction": float(max_count) / len(clean_votes),
            "tied": tied,
            **{f"vote_{m}": method_predictions[m][i] for m in methods},
        })

    return pl.DataFrame(rows)


def compare_aggregation_strategies(
    method_predictions: dict[str, list[str]],
    ground_truth: list[str],
    confidence_scores: dict[str, list[float]] | None = None,
) -> pl.DataFrame:
    """Side-by-side comparison: per-method, naive majority, and confidence-weighted majority.

    PopV ONTOLOGY_HIERARCHICAL is implemented in popv_ensemble.py — call it
    separately on the same `method_predictions` to add a third row.

    Returns a polars DataFrame with one row per strategy and columns:
        strategy, accuracy, macro_f1, weighted_f1.
    """
    from sklearn.metrics import accuracy_score, f1_score

    rows: list[dict] = []
    # Each individual method
    for m, preds in method_predictions.items():
        rows.append({
            "strategy": m,
            "accuracy": accuracy_score(ground_truth, preds),
            "macro_f1": f1_score(ground_truth, preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(ground_truth, preds, average="weighted", zero_division=0),
        })

    # Naive majority (no confidence)
    nm = naive_majority_vote(method_predictions)
    rows.append({
        "strategy": "naive_majority",
        "accuracy": accuracy_score(ground_truth, nm["predicted_type"].to_list()),
        "macro_f1": f1_score(ground_truth, nm["predicted_type"].to_list(),
                             average="macro", zero_division=0),
        "weighted_f1": f1_score(ground_truth, nm["predicted_type"].to_list(),
                                average="weighted", zero_division=0),
    })

    # Naive majority with confidence tie-break
    if confidence_scores is not None:
        nmc = naive_majority_vote(
            method_predictions, confidence_scores=confidence_scores
        )
        rows.append({
            "strategy": "naive_majority_confidence_tie",
            "accuracy": accuracy_score(ground_truth, nmc["predicted_type"].to_list()),
            "macro_f1": f1_score(ground_truth, nmc["predicted_type"].to_list(),
                                 average="macro", zero_division=0),
            "weighted_f1": f1_score(ground_truth, nmc["predicted_type"].to_list(),
                                    average="weighted", zero_division=0),
        })

    return pl.DataFrame(rows)
