"""Data loading and processing modules."""

from dapidl.data.xenium import XeniumDataReader
from dapidl.data.merscope import MerscopeDataReader, detect_platform, create_reader
from dapidl.data.annotation import CellTypeAnnotator, map_to_broad_category
from dapidl.data.patches import PatchExtractor
from dapidl.data.dataset import DAPIDLDataset, create_data_splits, create_dataloaders
from dapidl.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    compute_dataset_stats,
    validate_normalization_stats,
    InvalidNormalizationStatsError,
)
from dapidl.data.cleaning import (
    compute_spatial_coherence,
    filter_spatially_inconsistent,
    clean_dataset_spatial,
)
from dapidl.data.hierarchical_dataset import (
    HierarchicalDataset,
    HierarchicalLabels,
    create_hierarchical_data_splits,
    create_hierarchical_dataloaders,
)
from dapidl.data.multi_tissue_dataset import (
    MultiTissueDataset,
    MultiTissueConfig,
    TissueDatasetConfig,
    create_multi_tissue_splits,
    create_multi_tissue_dataloaders,
)

__all__ = [
    # Data readers
    "XeniumDataReader",
    "MerscopeDataReader",
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
