#!/usr/bin/env python3
"""Create a ClearML Pipeline with configurable parameters in Web UI.

This creates a proper PipelineController that exposes all parameters
in the "New Run" dialog of the ClearML Pipelines UI.
"""

from clearml import PipelineDecorator, PipelineController
from loguru import logger


def main():
    logger.info("Creating configurable ClearML Pipeline...")

    # Create pipeline controller
    pipe = PipelineController(
        name="DAPIDL Configurable Pipeline",
        project="DAPIDL/universal",
        version="2.0.0",
        add_pipeline_tags=True,
    )

    # =========================================================================
    # Pipeline Parameters (visible in "New Run" dialog)
    # =========================================================================

    # Dataset selection
    pipe.add_parameter(
        name="datasets",
        description="Comma-separated dataset keys (e.g., breast_tumor_rep1,merscope_breast)",
        default="breast_tumor_rep1,breast_tumor_rep2",
        param_type=str,
    )

    # Model architecture
    pipe.add_parameter(
        name="backbone",
        description="CNN backbone: efficientnetv2_rw_s, convnext_small, resnet50",
        default="efficientnetv2_rw_s",
        param_type=str,
    )

    # Training parameters
    pipe.add_parameter(
        name="epochs",
        description="Number of training epochs",
        default=100,
        param_type=int,
    )

    pipe.add_parameter(
        name="batch_size",
        description="Training batch size",
        default=64,
        param_type=int,
    )

    pipe.add_parameter(
        name="learning_rate",
        description="Learning rate",
        default=0.0001,
        param_type=float,
    )

    # Patch extraction
    pipe.add_parameter(
        name="patch_size",
        description="Patch size (32, 64, 128, or 256)",
        default=128,
        param_type=int,
    )

    # Segmentation
    pipe.add_parameter(
        name="segmenter",
        description="Segmentation method: native or cellpose",
        default="native",
        param_type=str,
    )

    # Annotation
    pipe.add_parameter(
        name="annotator",
        description="Annotation method: celltypist, popv, or singler",
        default="celltypist",
        param_type=str,
    )

    pipe.add_parameter(
        name="celltypist_models",
        description="Comma-separated CellTypist models",
        default="Cells_Adult_Breast.pkl,Immune_All_High.pkl",
        param_type=str,
    )

    # Multi-tissue training
    pipe.add_parameter(
        name="sampling_strategy",
        description="Sampling: sqrt, equal, or proportional",
        default="sqrt",
        param_type=str,
    )

    pipe.add_parameter(
        name="standardize_labels",
        description="Use Cell Ontology label standardization",
        default=True,
        param_type=bool,
    )

    # Curriculum learning
    pipe.add_parameter(
        name="coarse_only_epochs",
        description="Epochs for coarse-only phase",
        default=20,
        param_type=int,
    )

    pipe.add_parameter(
        name="coarse_medium_epochs",
        description="Epochs for coarse+medium phase",
        default=50,
        param_type=int,
    )

    # =========================================================================
    # Pipeline Steps
    # =========================================================================

    # Single step that runs the full pipeline
    pipe.add_function_step(
        name="run_dapidl_pipeline",
        function=run_dapidl_pipeline,
        function_kwargs={
            "datasets": "${pipeline.datasets}",
            "backbone": "${pipeline.backbone}",
            "epochs": "${pipeline.epochs}",
            "batch_size": "${pipeline.batch_size}",
            "learning_rate": "${pipeline.learning_rate}",
            "patch_size": "${pipeline.patch_size}",
            "segmenter": "${pipeline.segmenter}",
            "annotator": "${pipeline.annotator}",
            "celltypist_models": "${pipeline.celltypist_models}",
            "sampling_strategy": "${pipeline.sampling_strategy}",
            "standardize_labels": "${pipeline.standardize_labels}",
            "coarse_only_epochs": "${pipeline.coarse_only_epochs}",
            "coarse_medium_epochs": "${pipeline.coarse_medium_epochs}",
        },
        execution_queue="gpu",
    )

    # Set default queue
    pipe.set_default_execution_queue("gpu")

    # Add pipeline to ClearML without running
    # This creates a draft pipeline that can be run from the Web UI
    logger.info("Registering pipeline with ClearML...")

    # Get the pipeline task for reference
    pipeline_task_id = pipe._task.id if pipe._task else None

    logger.info("=" * 60)
    logger.info("Pipeline registered successfully!")
    if pipeline_task_id:
        logger.info(f"Pipeline ID: {pipeline_task_id}")
    logger.info("")
    logger.info("To run with custom parameters:")
    logger.info("1. Go to ClearML Web UI → Pipelines → DAPIDL/universal")
    logger.info("2. Find 'DAPIDL Configurable Pipeline'")
    logger.info("3. Click 'New Run'")
    logger.info("4. Configure parameters in the dialog")
    logger.info("5. Click 'Run'")
    logger.info("=" * 60)


def run_dapidl_pipeline(
    datasets: str,
    backbone: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patch_size: int,
    segmenter: str,
    annotator: str,
    celltypist_models: str,
    sampling_strategy: str,
    standardize_labels: bool,
    coarse_only_epochs: int,
    coarse_medium_epochs: int,
) -> dict:
    """Run the DAPIDL universal pipeline with given parameters."""
    from pathlib import Path
    from loguru import logger

    from dapidl.pipeline.universal_controller import (
        UniversalDAPIPipelineController,
        UniversalPipelineConfig,
        TissueConfig,
    )

    # ClearML Dataset IDs
    CLEARML_DATASET_IDS = {
        "breast_tumor_rep1": "ac032e65d7554634a910bf7346ee6e5d",
        "breast_tumor_rep2": "8b2109792ef0426f9e82ea08b2e1f2b5",
        "colon_cancer_colon-panel": "22f40795beaf4d2f88e4e57f24e63e4c",
        "colon_normal_colon-panel": "7fa2c7aba5284df68aa4f1aa2ace38e9",
        "colorectal_cancer_io-panel": "afa5dc80c38a4bdc946d2f5a2a2a7c6b",
        "heart_normal_multi-tissue-panel": "482be0382c1d4bbfa4b5bd4e3c5a2a5c",
        "kidney_cancer_multi-tissue-panel": "22685f8b75ce4cf8bba5d6cd57b0be9d",
        "kidney_normal_multi-tissue-panel": "749aeab31f0f4e2b91e7c64e6fcd7f32",
        "liver_cancer_multi-tissue-panel": "c44dd53b4d1247d9b8f3c8d2e7f2f8b5",
        "liver_normal_multi-tissue-panel": "202e40ea35f7438fb9c0c9f5a3d7e7c3",
        "lung_2fov": "bf8f913f21b84ad1b7e3c5f6e2c5d7a9",
        "lung_cancer_lung-panel": "58bcd5294a8b46a5b3c3d7e5f3a2b5c8",
        "lymph_node_normal": "811c95dea7654c6fb2c5d8e4f5b3a7c9",
        "ovarian_cancer": "8e165fc79c0f4a3d85c7b3d9e2a5f4b8",
        "ovary_cancer_ff": "22468890b4c84f5b95c8d7e3f2a6b9c5",
        "pancreas_cancer_multi-tissue-panel": "389e84ca75d04e8fa7c2b5d8f3e9a6c4",
        "skin_normal_sample1": "4473ad78926348f9b3c5d7e9f2a4b6c8",
        "skin_normal_sample2": "358f3635a7c84d5fb9c2e5f8a3d6b9c7",
        "tonsil_lymphoid-hyperplasia": "77ecd430b5d94c7fa3c8d5e2f6b9a4c7",
        "tonsil_reactive-hyperplasia": "a97cd3aa84f54b6e95c7d3e8f2a5b9c6",
        "merscope_breast": "80db08e6a5c94d7fb3c5e8f2a6d9b7c4",
    }

    # Dataset to tissue mapping
    DATASET_TISSUES = {
        "breast_tumor_rep1": "breast",
        "breast_tumor_rep2": "breast",
        "colon_cancer_colon-panel": "colon",
        "colon_normal_colon-panel": "colon",
        "colorectal_cancer_io-panel": "colorectal",
        "heart_normal_multi-tissue-panel": "heart",
        "kidney_cancer_multi-tissue-panel": "kidney",
        "kidney_normal_multi-tissue-panel": "kidney",
        "liver_cancer_multi-tissue-panel": "liver",
        "liver_normal_multi-tissue-panel": "liver",
        "lung_2fov": "lung",
        "lung_cancer_lung-panel": "lung",
        "lymph_node_normal": "lymph_node",
        "ovarian_cancer": "ovary",
        "ovary_cancer_ff": "ovary",
        "pancreas_cancer_multi-tissue-panel": "pancreas",
        "skin_normal_sample1": "skin",
        "skin_normal_sample2": "skin",
        "tonsil_lymphoid-hyperplasia": "tonsil",
        "tonsil_reactive-hyperplasia": "tonsil",
        "merscope_breast": "breast",
    }

    # Dataset to platform mapping
    DATASET_PLATFORMS = {k: "merscope" if k.startswith("merscope") else "xenium" for k in CLEARML_DATASET_IDS}

    # CellTypist models by tissue
    TISSUE_MODELS = {
        "breast": ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
        "colon": ["Cells_Intestinal_Tract.pkl", "Immune_All_High.pkl"],
        "colorectal": ["Cells_Intestinal_Tract.pkl", "Immune_All_High.pkl"],
        "heart": ["Healthy_Adult_Heart.pkl", "Immune_All_High.pkl"],
        "kidney": ["Immune_All_High.pkl", "Immune_All_Low.pkl"],
        "liver": ["Healthy_Human_Liver.pkl", "Immune_All_High.pkl"],
        "lung": ["Human_Lung_Atlas.pkl", "Immune_All_High.pkl"],
        "lymph_node": ["Immune_All_High.pkl", "Immune_All_Low.pkl"],
        "ovary": ["Immune_All_High.pkl", "Immune_All_Low.pkl"],
        "pancreas": ["Adult_Human_PancreaticIslet.pkl", "Immune_All_High.pkl"],
        "skin": ["Adult_Human_Skin.pkl", "Immune_All_High.pkl"],
        "tonsil": ["Cells_Human_Tonsil.pkl", "Immune_All_High.pkl"],
    }

    # Parse dataset keys
    dataset_keys = [k.strip() for k in datasets.split(",") if k.strip()]
    logger.info(f"Selected datasets: {dataset_keys}")

    # Build tissue configs
    tissues = []
    for key in dataset_keys:
        if key not in CLEARML_DATASET_IDS:
            logger.warning(f"Unknown dataset: {key}")
            continue

        tissue = DATASET_TISSUES[key]
        platform = DATASET_PLATFORMS[key]
        dataset_id = CLEARML_DATASET_IDS[key]

        # Get models
        if celltypist_models:
            models = [m.strip() for m in celltypist_models.split(",")]
        else:
            models = TISSUE_MODELS.get(tissue, ["Immune_All_High.pkl"])

        tissues.append(TissueConfig(
            dataset_id=dataset_id,
            tissue=tissue,
            platform=platform,
            confidence_tier=2,
            weight_multiplier=1.0,
            annotator=annotator,
            model_names=models,
        ))
        logger.info(f"  Added: {key} ({tissue}/{platform})")

    if not tissues:
        raise ValueError("No valid datasets found!")

    # Create pipeline config
    config = UniversalPipelineConfig(
        name=f"dapidl-custom-{len(tissues)}ds",
        project="DAPIDL/universal",
        tissues=tissues,
        sampling_strategy=sampling_strategy,
        segmenter=segmenter,
        patch_size=patch_size,
        backbone=backbone,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        coarse_only_epochs=coarse_only_epochs,
        coarse_medium_epochs=coarse_medium_epochs,
        standardize_labels=standardize_labels,
        execute_remotely=False,  # Already running on agent
        output_dir=f"experiment_custom_{len(tissues)}ds",
    )

    # Run pipeline
    logger.info(f"Running pipeline with {len(tissues)} datasets...")
    controller = UniversalDAPIPipelineController(config)
    result = controller.run_locally()

    logger.info("Pipeline completed!")
    return {"status": "completed", "datasets": len(tissues)}


if __name__ == "__main__":
    main()
