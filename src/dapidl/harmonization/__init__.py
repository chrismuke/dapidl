""".. deprecated::
    Superseded by :mod:`dapidl.ontology` (CL-anchored coarse/medium/fine tiers,
    the single source of truth for training + evaluation; ~26 importers). This
    module's broad/mid/fine string-tree hierarchy carries no Cell Ontology IDs and
    is retained only for the legacy ``dapidl compare-labels`` CLI command. Do NOT
    use it in new code — use ``dapidl.ontology`` instead (review 2026-05-29 §4).

Cell type label harmonization across annotation sources.

This module provides tools to harmonize cell type labels from different
annotation sources (CellTypist, popV, ground truth) for meaningful comparison.

Key concepts:
- Hierarchy: Multi-level cell type tree (Broad → Mid → Fine)
- Mapping: Translation tables between source labels and hierarchy
- Harmonization: Converting labels to a common reference for comparison

Example:
    from dapidl.harmonization import LabelHarmonizer, evaluate_predictions

    harmonizer = LabelHarmonizer()

    # Map ground truth and predictions to common hierarchy
    gt_mapped, _ = harmonizer.map_labels(ground_truth_labels, source="xenium_breast")
    pred_mapped, _ = harmonizer.map_labels(predictions, source="celltypist_breast")

    # Evaluate predictions against ground truth at multiple levels
    result = evaluate_predictions(
        predictions=predictions,
        ground_truth=ground_truth_labels,
        prediction_source="celltypist_breast",
    )
    print(f"Broad accuracy: {result.metrics['broad']['accuracy']:.3f}")
    print(f"Mid accuracy: {result.metrics['mid']['accuracy']:.3f}")
"""

from dapidl.harmonization.hierarchy import (
    CellTypeHierarchy,
    BREAST_HIERARCHY,
)
from dapidl.harmonization.mapper import (
    LabelHarmonizer,
    HarmonizationResult,
    LabelMapping,
    load_mapping,
    save_mapping,
    get_available_mappings,
)
from dapidl.harmonization.evaluation import (
    evaluate_predictions,
    evaluate_annotations_df,
    print_evaluation_report,
    create_mapping_from_annotations,
)

__all__ = [
    # Hierarchy
    "CellTypeHierarchy",
    "BREAST_HIERARCHY",
    # Mapper
    "LabelHarmonizer",
    "HarmonizationResult",
    "LabelMapping",
    "load_mapping",
    "save_mapping",
    "get_available_mappings",
    # Evaluation
    "evaluate_predictions",
    "evaluate_annotations_df",
    "print_evaluation_report",
    "create_mapping_from_annotations",
]

import warnings as _warnings

# Non-fatal signal (no pytest -W error in this repo): nudge new code to the
# CL-anchored dapidl.ontology. compare-labels keeps working unchanged.
_warnings.warn(
    "dapidl.harmonization is deprecated (broad/mid/fine, no CL IDs); use the "
    "CL-anchored dapidl.ontology instead. Retained only for `dapidl compare-labels`.",
    DeprecationWarning,
    stacklevel=2,
)
