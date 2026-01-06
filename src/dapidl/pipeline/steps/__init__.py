"""Pipeline step implementations.

Each step is a standalone ClearML Task with UI-configurable parameters:
- DataLoaderStep: Load raw data from ClearML Dataset
- SegmentationStep: Detect nuclei using configurable method
- AnnotationStep: Assign cell type labels (single method)
- EnsembleAnnotationStep: Ensemble annotation with multiple methods
- CLStandardizationStep: Standardize annotations to Cell Ontology
- PatchExtractionStep: Create LMDB patches for training
- LMDBCreationStep: Create LMDB with skip logic and lineage
- TrainingStep: Train classification model
- HierarchicalTrainingStep: Train multi-head hierarchical classifier
- CrossValidationStep: Cross-modal validation without ground truth
- CrossPlatformTransferStep: Test model transfer between platforms
- DocumentationStep: Generate docs and export to Obsidian
"""

# Steps are imported lazily to avoid circular imports
# Use: from dapidl.pipeline.steps.data_loader import DataLoaderStep

__all__ = [
    "DataLoaderStep",
    "SegmentationStep",
    "AnnotationStep",
    "EnsembleAnnotationStep",
    "EnsembleAnnotationConfig",
    "CLStandardizationStep",
    "CLStandardizationConfig",
    "PatchExtractionStep",
    "LMDBCreationStep",
    "LMDBCreationConfig",
    "TrainingStep",
    "HierarchicalTrainingStep",
    "HierarchicalTrainingConfig",
    "UniversalDAPITrainingStep",
    "UniversalTrainingConfig",
    "TissueDatasetSpec",
    "CrossValidationStep",
    "CrossPlatformTransferStep",
    "DocumentationStep",
]


def __getattr__(name: str):
    """Lazy import of step classes."""
    if name == "DataLoaderStep":
        from dapidl.pipeline.steps.data_loader import DataLoaderStep
        return DataLoaderStep
    elif name == "SegmentationStep":
        from dapidl.pipeline.steps.segmentation import SegmentationStep
        return SegmentationStep
    elif name == "AnnotationStep":
        from dapidl.pipeline.steps.annotation import AnnotationStep
        return AnnotationStep
    elif name == "EnsembleAnnotationStep":
        from dapidl.pipeline.steps.ensemble_annotation import EnsembleAnnotationStep
        return EnsembleAnnotationStep
    elif name == "EnsembleAnnotationConfig":
        from dapidl.pipeline.steps.ensemble_annotation import EnsembleAnnotationConfig
        return EnsembleAnnotationConfig
    elif name == "CLStandardizationStep":
        from dapidl.pipeline.steps.cl_standardization import CLStandardizationStep
        return CLStandardizationStep
    elif name == "CLStandardizationConfig":
        from dapidl.pipeline.steps.cl_standardization import CLStandardizationConfig
        return CLStandardizationConfig
    elif name == "PatchExtractionStep":
        from dapidl.pipeline.steps.patch_extraction import PatchExtractionStep
        return PatchExtractionStep
    elif name == "LMDBCreationStep":
        from dapidl.pipeline.steps.lmdb_creation import LMDBCreationStep
        return LMDBCreationStep
    elif name == "LMDBCreationConfig":
        from dapidl.pipeline.steps.lmdb_creation import LMDBCreationConfig
        return LMDBCreationConfig
    elif name == "TrainingStep":
        from dapidl.pipeline.steps.training import TrainingStep
        return TrainingStep
    elif name == "HierarchicalTrainingStep":
        from dapidl.pipeline.steps.hierarchical_training import HierarchicalTrainingStep
        return HierarchicalTrainingStep
    elif name == "HierarchicalTrainingConfig":
        from dapidl.pipeline.steps.hierarchical_training import HierarchicalTrainingConfig
        return HierarchicalTrainingConfig
    elif name == "UniversalDAPITrainingStep":
        from dapidl.pipeline.steps.universal_training import UniversalDAPITrainingStep
        return UniversalDAPITrainingStep
    elif name == "UniversalTrainingConfig":
        from dapidl.pipeline.steps.universal_training import UniversalTrainingConfig
        return UniversalTrainingConfig
    elif name == "TissueDatasetSpec":
        from dapidl.pipeline.steps.universal_training import TissueDatasetSpec
        return TissueDatasetSpec
    elif name == "CrossValidationStep":
        from dapidl.pipeline.steps.cross_validation import CrossValidationStep
        return CrossValidationStep
    elif name == "CrossPlatformTransferStep":
        from dapidl.pipeline.steps.cross_platform_transfer import CrossPlatformTransferStep
        return CrossPlatformTransferStep
    elif name == "DocumentationStep":
        from dapidl.pipeline.steps.documentation import DocumentationStep
        return DocumentationStep
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
