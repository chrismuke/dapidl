"""Evaluation utilities using label harmonization.

Provides functions to evaluate cell type predictions against ground truth
using the harmonization system for meaningful comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from dapidl.harmonization.hierarchy import BREAST_HIERARCHY, CellTypeHierarchy
from dapidl.harmonization.mapper import (
    HarmonizationResult,
    LabelHarmonizer,
    LabelMapping,
)

logger = logging.getLogger(__name__)


def evaluate_predictions(
    predictions: Sequence[str],
    ground_truth: Sequence[str],
    prediction_source: str | None = None,
    ground_truth_source: str = "xenium_breast",
    harmonizer: LabelHarmonizer | None = None,
) -> HarmonizationResult:
    """Evaluate predictions against ground truth using harmonization.

    Maps both predictions and ground truth to a common hierarchy and
    computes metrics at broad, mid, and fine levels.

    Args:
        predictions: Predicted cell type labels
        ground_truth: Ground truth cell type labels
        prediction_source: Name of prediction source for mapping lookup
            (e.g., "celltypist_breast", "popv_immune")
        ground_truth_source: Name of ground truth source (default: "xenium_breast")
        harmonizer: Pre-configured LabelHarmonizer (uses default if None)

    Returns:
        HarmonizationResult with metrics at each hierarchy level

    Example:
        result = evaluate_predictions(
            predictions=["CD4-Tem", "Macro-m1", "Fibro-major"],
            ground_truth=["CD4+_T_Cells", "Macrophages_1", "Stromal"],
            prediction_source="celltypist_breast",
        )
        print(result.summary())
        print(f"Broad level accuracy: {result.metrics['broad']['accuracy']:.3f}")
    """
    if harmonizer is None:
        harmonizer = LabelHarmonizer()

    return harmonizer.compare(
        source_labels=predictions,
        target_labels=ground_truth,
        source_name=prediction_source,
        target_name=ground_truth_source,
    )


def evaluate_annotations_df(
    predictions_df: pl.DataFrame,
    ground_truth_df: pl.DataFrame,
    pred_label_col: str = "predicted_type",
    gt_label_col: str = "predicted_type",
    cell_id_col: str = "cell_id",
    prediction_source: str | None = None,
    ground_truth_source: str = "xenium_breast",
    harmonizer: LabelHarmonizer | None = None,
) -> tuple[HarmonizationResult, pl.DataFrame]:
    """Evaluate predictions DataFrame against ground truth DataFrame.

    Joins dataframes on cell_id and evaluates only matching cells.

    Args:
        predictions_df: DataFrame with predictions (must have cell_id_col and pred_label_col)
        ground_truth_df: DataFrame with ground truth (must have cell_id_col and gt_label_col)
        pred_label_col: Column name for predicted labels
        gt_label_col: Column name for ground truth labels
        cell_id_col: Column name for cell identifiers
        prediction_source: Name of prediction source for mapping
        ground_truth_source: Name of ground truth source
        harmonizer: Pre-configured LabelHarmonizer

    Returns:
        Tuple of (HarmonizationResult, joined_df with harmonized labels)
    """
    if harmonizer is None:
        harmonizer = LabelHarmonizer()

    # Rename columns to avoid conflicts during join
    pred_col_renamed = "pred_label_orig"
    gt_col_renamed = "gt_label_orig"

    pred_for_join = predictions_df.select([
        pl.col(cell_id_col),
        pl.col(pred_label_col).alias(pred_col_renamed),
    ])
    gt_for_join = ground_truth_df.select([
        pl.col(cell_id_col),
        pl.col(gt_label_col).alias(gt_col_renamed),
    ])

    # Join on cell_id
    joined = pred_for_join.join(gt_for_join, on=cell_id_col, how="inner")

    n_total_pred = len(predictions_df)
    n_total_gt = len(ground_truth_df)
    n_matched = len(joined)

    logger.info(f"Matched {n_matched} cells for evaluation")
    logger.info(f"  Predictions: {n_total_pred} total, {n_total_pred - n_matched} unmatched")
    logger.info(f"  Ground truth: {n_total_gt} total, {n_total_gt - n_matched} unmatched")

    if n_matched == 0:
        raise ValueError("No matching cells between predictions and ground truth")

    # Extract labels
    pred_labels = joined[pred_col_renamed].to_list()
    gt_labels = joined[gt_col_renamed].to_list()

    # Evaluate
    result = evaluate_predictions(
        predictions=pred_labels,
        ground_truth=gt_labels,
        prediction_source=prediction_source,
        ground_truth_source=ground_truth_source,
        harmonizer=harmonizer,
    )

    # Add harmonized labels to joined dataframe
    for level in ["broad", "mid", "fine"]:
        joined = joined.with_columns([
            pl.Series(f"pred_{level}", result.harmonized_source[level]),
            pl.Series(f"gt_{level}", result.harmonized_target[level]),
            pl.Series(
                f"match_{level}",
                [
                    s == t
                    for s, t in zip(
                        result.harmonized_source[level],
                        result.harmonized_target[level],
                    )
                ],
            ),
        ])

    return result, joined


def print_evaluation_report(
    result: HarmonizationResult,
    title: str = "Cell Type Prediction Evaluation",
) -> None:
    """Print a formatted evaluation report.

    Args:
        result: HarmonizationResult from evaluate_predictions
        title: Report title
    """
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}\n")

    print(f"Total samples: {len(result.source_labels)}")

    for level in ["broad", "mid", "fine"]:
        if level in result.metrics:
            m = result.metrics[level]
            print(f"\n{level.upper()} Level:")
            print(f"  Accuracy:    {m['accuracy']:.3f}")
            print(f"  Macro F1:    {m['f1_macro']:.3f}")
            print(f"  Weighted F1: {m['f1_weighted']:.3f}")
            print(f"  Evaluated:   {m['n_samples']} cells")
            print(f"  Excluded:    {m['n_excluded']} cells (unmapped)")

    if result.unmapped_source:
        print(f"\nUnmapped prediction labels ({len(result.unmapped_source)}):")
        for label in sorted(result.unmapped_source)[:10]:
            print(f"  - {label}")
        if len(result.unmapped_source) > 10:
            print(f"  ... and {len(result.unmapped_source) - 10} more")

    if result.unmapped_target:
        print(f"\nUnmapped ground truth labels ({len(result.unmapped_target)}):")
        for label in sorted(result.unmapped_target)[:10]:
            print(f"  - {label}")
        if len(result.unmapped_target) > 10:
            print(f"  ... and {len(result.unmapped_target) - 10} more")

    print(f"\n{'=' * 60}\n")


def create_mapping_from_annotations(
    source_name: str,
    labels: Sequence[str],
    description: str = "",
    harmonizer: LabelHarmonizer | None = None,
) -> LabelMapping:
    """Create a mapping file from a set of annotations.

    Attempts to automatically map labels to the hierarchy using
    alias matching and normalization. Reports unmapped labels.

    Args:
        source_name: Name for the mapping (e.g., "my_model")
        labels: Unique labels from the annotation source
        description: Description of the source
        harmonizer: Pre-configured harmonizer

    Returns:
        LabelMapping with automatic mappings and unmapped labels

    Example:
        from celltypist import models
        model = models.Model.load("My_Model.pkl")
        mapping = create_mapping_from_annotations(
            source_name="my_model",
            labels=model.cell_types,
            description="My custom CellTypist model",
        )
        # Review and manually fix unmapped labels
        save_mapping(mapping)
    """
    if harmonizer is None:
        harmonizer = LabelHarmonizer()

    mapping = LabelMapping(
        source_name=source_name,
        description=description or f"Auto-generated mapping for {source_name}",
    )

    unique_labels = sorted(set(labels))

    for label in unique_labels:
        hierarchy_label, found = harmonizer.map_label(label, level="fine")
        if found and hierarchy_label:
            mapping.add(label, hierarchy_label)
        else:
            mapping.unmapped_labels.append(label)

    logger.info(f"Auto-mapping results for {source_name}:")
    logger.info(f"  Mapped: {len(mapping.entries)} labels")
    logger.info(f"  Unmapped: {len(mapping.unmapped_labels)} labels")

    if mapping.unmapped_labels:
        logger.warning(f"Unmapped labels: {mapping.unmapped_labels}")

    return mapping
