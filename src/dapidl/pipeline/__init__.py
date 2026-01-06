"""ClearML Pipeline for Spatial Transcriptomics Processing.

This module provides a modular, UI-configurable ClearML pipeline for DAPIDL that
processes spatial transcriptomics data (Xenium/MERSCOPE) through:

1. Data Loading - Load raw data from ClearML Dataset
2. Segmentation - Nucleus detection (Cellpose or native)
3. Annotation - Cell type labeling (CellTypist, ground truth, popV)
4. Patch Extraction - Create LMDB patches for training
5. Training - Train classification model

Each step is a standalone ClearML Task with UI-configurable parameters.
Components (segmenters, annotators) are swappable via registry.

Example:
    ```python
    from dapidl.pipeline import PipelineConfig, create_pipeline

    config = PipelineConfig(
        dataset_id="your-dataset-id",
        segmenter="cellpose",
        annotator="celltypist",
        epochs=50,
    )

    pipeline = create_pipeline(config)
    pipeline.run_locally()  # Or pipeline.run() for ClearML execution
    ```
"""

from dapidl.pipeline.base import (
    AnnotationConfig,
    AnnotationResult,
    PipelineStep,
    SegmentationConfig,
    SegmentationResult,
    StepArtifacts,
)
from dapidl.pipeline.controller import (
    DAPIDLPipelineController,
    PipelineConfig,
    create_pipeline,
)
from dapidl.pipeline.universal_controller import (
    UniversalDAPIPipelineController,
    UniversalPipelineConfig,
    TissueConfig,
    create_universal_pipeline,
)
from dapidl.pipeline.enhanced_controller import (
    EnhancedDAPIDLPipelineController,
    EnhancedPipelineResult,
    create_step_base_tasks,
)
from dapidl.pipeline.gui_pipeline_config import GUIPipelineConfig
from dapidl.pipeline.registry import (
    get_annotator,
    get_segmenter,
    list_annotators,
    list_segmenters,
    register_annotator,
    register_segmenter,
)

# Import components to trigger registration
# This must happen after registry imports
from dapidl.pipeline.components import segmenters, annotators  # noqa: F401

__all__ = [
    # Controller
    "PipelineConfig",
    "DAPIDLPipelineController",
    "create_pipeline",
    # Universal Controller
    "UniversalPipelineConfig",
    "UniversalDAPIPipelineController",
    "TissueConfig",
    "create_universal_pipeline",
    # Enhanced Controller (GUI-configurable)
    "GUIPipelineConfig",
    "EnhancedDAPIDLPipelineController",
    "EnhancedPipelineResult",
    "create_step_base_tasks",
    # Base classes
    "PipelineStep",
    "StepArtifacts",
    "SegmentationConfig",
    "SegmentationResult",
    "AnnotationConfig",
    "AnnotationResult",
    # Registry functions
    "register_segmenter",
    "register_annotator",
    "get_segmenter",
    "get_annotator",
    "list_segmenters",
    "list_annotators",
]
