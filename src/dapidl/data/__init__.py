"""Data loading and processing modules."""

from dapidl.data.annotation import CellTypeAnnotator, map_to_broad_category
from dapidl.data.cleaning import (
    clean_dataset_spatial,
    compute_spatial_coherence,
    filter_spatially_inconsistent,
)
from dapidl.data.dataset import DAPIDLDataset, create_data_splits, create_dataloaders
from dapidl.data.hierarchical_dataset import (
    HierarchicalDataset,
    HierarchicalLabels,
    create_hierarchical_data_splits,
    create_hierarchical_dataloaders,
)
from dapidl.data.merscope import MerscopeDataReader, create_reader, detect_platform
from dapidl.data.multi_tissue_dataset import (
    MultiTissueConfig,
    MultiTissueDataset,
    TissueDatasetConfig,
    create_multi_tissue_dataloaders,
    create_multi_tissue_splits,
)
from dapidl.data.patches import PatchExtractor
from dapidl.data.sthelar import SthelarDataReader
from dapidl.data.transforms import (
    InvalidNormalizationStatsError,
    compute_dataset_stats,
    get_train_transforms,
    get_val_transforms,
    validate_normalization_stats,
)
from dapidl.data.xenium import XeniumDataReader

__all__ = [
    # Data readers
    "XeniumDataReader",
    "MerscopeDataReader",
    "SthelarDataReader",
    "detect_platform",
    "create_reader",
    # Annotation
    "CellTypeAnnotator",
    "map_to_broad_category",
    # Patch extraction
    "PatchExtractor",
    # Datasets
    "DAPIDLDataset",
    "create_data_splits",
    "create_dataloaders",
    # Transforms
    "get_train_transforms",
    "get_val_transforms",
    "compute_dataset_stats",
    "validate_normalization_stats",
    "InvalidNormalizationStatsError",
    # Data cleaning
    "compute_spatial_coherence",
    "filter_spatially_inconsistent",
    "clean_dataset_spatial",
    # Hierarchical
    "HierarchicalDataset",
    "HierarchicalLabels",
    "create_hierarchical_data_splits",
    "create_hierarchical_dataloaders",
    # Multi-tissue
    "MultiTissueDataset",
    "MultiTissueConfig",
    "TissueDatasetConfig",
    "create_multi_tissue_splits",
    "create_multi_tissue_dataloaders",
]
