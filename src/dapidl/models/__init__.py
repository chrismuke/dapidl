"""Model architectures."""

from dapidl.models.backbone import SingleChannelAdapter, create_backbone
from dapidl.models.classifier import CellTypeClassifier
from dapidl.models.domain_adaptation import (
    AdaptiveInference,
    adapt_batch_norm,
    compute_domain_shift_metrics,
    create_adaptation_loader,
)
from dapidl.models.ensemble import MultiScaleEnsemble
from dapidl.models.hierarchical import (
    HierarchicalClassifier,
    HierarchicalOutput,
    HierarchyConfig,
)
from dapidl.models.multitask import (
    MultiTaskCellTypeClassifier,
    MultiTaskClassifier,
    SegmentationDecoder,
)

__all__ = [
    "create_backbone",
    "SingleChannelAdapter",
    "CellTypeClassifier",
    "MultiScaleEnsemble",
    "MultiTaskClassifier",
    "MultiTaskCellTypeClassifier",
    "SegmentationDecoder",
    "HierarchicalClassifier",
    "HierarchyConfig",
    "HierarchicalOutput",
    # Domain adaptation
    "adapt_batch_norm",
    "create_adaptation_loader",
    "compute_domain_shift_metrics",
    "AdaptiveInference",
]
