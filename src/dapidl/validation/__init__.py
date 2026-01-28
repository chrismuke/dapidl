"""Validation utilities for DAPIDL.

GT-FREE Validation Methods:
1. Marker gene validation - check canonical marker expression
2. Leiden clustering - compare with unsupervised transcriptomic clusters
3. UMAP visualization - check spatial coherence in embedding space
4. Morphology validation - independent validation using pretrained backbones
5. Cross-method consensus - popV-style agreement scoring
"""

from dapidl.validation.cross_modal import (
    ValidationMetrics,
    cluster_morphology_embeddings,
    compute_confidence_tiers,
    compute_dapi_agreement,
    compute_leiden_metrics,
    extract_morphology_embeddings,
    extract_pretrained_features,
    quick_validate,
    unsupervised_morphology_validation,
)

from dapidl.validation.marker_validation import (
    BREAST_MARKERS,
    MarkerValidationResult,
    compute_marker_scores,
    validate_with_markers,
)

__all__ = [
    # Cross-modal validation
    "ValidationMetrics",
    "compute_leiden_metrics",
    "compute_dapi_agreement",
    "extract_morphology_embeddings",
    "extract_pretrained_features",
    "cluster_morphology_embeddings",
    "compute_confidence_tiers",
    "quick_validate",
    "unsupervised_morphology_validation",
    # Marker validation
    "BREAST_MARKERS",
    "MarkerValidationResult",
    "compute_marker_scores",
    "validate_with_markers",
]
