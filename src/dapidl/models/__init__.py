"""Model architectures."""

from dapidl.models.backbone import create_backbone, SingleChannelAdapter
from dapidl.models.classifier import CellTypeClassifier
from dapidl.models.ensemble import MultiScaleEnsemble
from dapidl.models.multitask import (
    MultiTaskClassifier,
    MultiTaskCellTypeClassifier,
    SegmentationDecoder,
)
from dapidl.models.hierarchical import (
    HierarchicalClassifier,
    HierarchyConfig,
    HierarchicalOutput,
)
from dapidl.models.domain_adaptation import (
    adapt_batch_norm,
    create_adaptation_loader,
    compute_domain_shift_metrics,
    AdaptiveInference,
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
