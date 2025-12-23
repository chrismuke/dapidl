"""Pipeline step implementations.

Each step is a standalone ClearML Task with UI-configurable parameters:
- DataLoaderStep: Load raw data from ClearML Dataset
- SegmentationStep: Detect nuclei using configurable method
- AnnotationStep: Assign cell type labels
- PatchExtractionStep: Create LMDB patches for training
- TrainingStep: Train classification model
"""

# Steps are imported lazily to avoid circular imports
# Use: from dapidl.pipeline.steps.data_loader import DataLoaderStep

__all__ = [
    "DataLoaderStep",
    "SegmentationStep",
    "AnnotationStep",
    "PatchExtractionStep",
    "TrainingStep",
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
    elif name == "PatchExtractionStep":
        from dapidl.pipeline.steps.patch_extraction import PatchExtractionStep
        return PatchExtractionStep
    elif name == "TrainingStep":
        from dapidl.pipeline.steps.training import TrainingStep
        return TrainingStep
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
