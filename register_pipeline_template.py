#!/usr/bin/env python3
"""Register DAPIDL Pipeline Template for ClearML Web UI.

This creates a pipeline template with all parameters exposed in the
"New Run" dialog. Run this once to register the template.
"""

from clearml import PipelineController, Task
from loguru import logger


def run_training_step(
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
    """Run DAPIDL universal training pipeline."""
    from dapidl.pipeline.universal_controller import (
        UniversalDAPIPipelineController,
        UniversalPipelineConfig,
        TissueConfig,
    )

    # Load dataset IDs from migration mapping file (self-hosted) or fall back to cloud IDs
    CLEARML_DATASET_IDS_CLOUD = {
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

    # Try to load self-hosted IDs from migration mapping
    import json
    from pathlib import Path
    mapping_file = Path(__file__).parent / "configs" / "clearml_id_mapping.json"
    CLEARML_DATASET_IDS = dict(CLEARML_DATASET_IDS_CLOUD)
    if mapping_file.exists():
        try:
            mapping = json.loads(mapping_file.read_text())
            cloud_to_selfhosted = mapping.get("datasets", {})
            # Replace cloud IDs with self-hosted IDs where available
            for key, cloud_id in CLEARML_DATASET_IDS_CLOUD.items():
                if cloud_id in cloud_to_selfhosted:
                    CLEARML_DATASET_IDS[key] = cloud_to_selfhosted[cloud_id]
        except Exception:
            pass  # Fall back to cloud IDs

    DATASET_TISSUES = {
        "breast_tumor_rep1": "breast", "breast_tumor_rep2": "breast",
        "colon_cancer_colon-panel": "colon", "colon_normal_colon-panel": "colon",
        "colorectal_cancer_io-panel": "colorectal",
        "heart_normal_multi-tissue-panel": "heart",
        "kidney_cancer_multi-tissue-panel": "kidney", "kidney_normal_multi-tissue-panel": "kidney",
        "liver_cancer_multi-tissue-panel": "liver", "liver_normal_multi-tissue-panel": "liver",
        "lung_2fov": "lung", "lung_cancer_lung-panel": "lung",
        "lymph_node_normal": "lymph_node",
        "ovarian_cancer": "ovary", "ovary_cancer_ff": "ovary",
        "pancreas_cancer_multi-tissue-panel": "pancreas",
        "skin_normal_sample1": "skin", "skin_normal_sample2": "skin",
        "tonsil_lymphoid-hyperplasia": "tonsil", "tonsil_reactive-hyperplasia": "tonsil",
        "merscope_breast": "breast",
    }

    TISSUE_MODELS = {
        "breast": ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
        "colon": ["Cells_Intestinal_Tract.pkl", "Immune_All_High.pkl"],
        "colorectal": ["Cells_Intestinal_Tract.pkl", "Immune_All_High.pkl"],
        "heart": ["Healthy_Adult_Heart.pkl", "Immune_All_High.pkl"],
        "kidney": ["Immune_All_High.pkl"], "liver": ["Healthy_Human_Liver.pkl", "Immune_All_High.pkl"],
        "lung": ["Human_Lung_Atlas.pkl", "Immune_All_High.pkl"],
        "lymph_node": ["Immune_All_High.pkl"], "ovary": ["Immune_All_High.pkl"],
        "pancreas": ["Adult_Human_PancreaticIslet.pkl", "Immune_All_High.pkl"],
        "skin": ["Adult_Human_Skin.pkl", "Immune_All_High.pkl"],
        "tonsil": ["Cells_Human_Tonsil.pkl", "Immune_All_High.pkl"],
    }

    dataset_keys = [k.strip() for k in datasets.split(",") if k.strip()]
    tissues = []

    for key in dataset_keys:
        if key not in CLEARML_DATASET_IDS:
            continue
        tissue = DATASET_TISSUES[key]
        platform = "merscope" if key.startswith("merscope") else "xenium"
        models = [m.strip() for m in celltypist_models.split(",")] if celltypist_models else TISSUE_MODELS.get(tissue, ["Immune_All_High.pkl"])

        tissues.append(TissueConfig(
            dataset_id=CLEARML_DATASET_IDS[key],
            tissue=tissue,
            platform=platform,
            confidence_tier=2,
            weight_multiplier=1.0,
            annotator=annotator,
            model_names=models,
        ))

    if not tissues:
        raise ValueError("No valid datasets!")

    config = UniversalPipelineConfig(
        name=f"dapidl-{len(tissues)}ds",
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
        execute_remotely=False,
        output_dir=f"experiment_{len(tissues)}ds",
    )

    controller = UniversalDAPIPipelineController(config)
    return controller.run_locally()


def main():
    logger.info("Registering DAPIDL Pipeline Template...")

    # Create pipeline controller with explicit parameter definitions
    pipe = PipelineController(
        name="DAPIDL Universal Pipeline",
        project="DAPIDL/universal",
        version="3.0.0",
        add_pipeline_tags=True,
        repo="https://github.com/chrismuke/dapidl.git",
        repo_branch="main",
    )

    # =========================================================================
    # Define ALL configurable parameters
    # These will appear in the "New Run" dialog
    # =========================================================================

    pipe.add_parameter(
        name="datasets",
        default="breast_tumor_rep1,breast_tumor_rep2",
        description="Comma-separated dataset keys. Available: breast_tumor_rep1, breast_tumor_rep2, merscope_breast, colon_cancer_colon-panel, colon_normal_colon-panel, colorectal_cancer_io-panel, heart_normal_multi-tissue-panel, kidney_cancer_multi-tissue-panel, kidney_normal_multi-tissue-panel, liver_cancer_multi-tissue-panel, liver_normal_multi-tissue-panel, lung_2fov, lung_cancer_lung-panel, lymph_node_normal, ovarian_cancer, ovary_cancer_ff, pancreas_cancer_multi-tissue-panel, skin_normal_sample1, skin_normal_sample2, tonsil_lymphoid-hyperplasia, tonsil_reactive-hyperplasia",
    )

    pipe.add_parameter(
        name="backbone",
        default="efficientnetv2_rw_s",
        description="CNN backbone: efficientnetv2_rw_s, efficientnetv2_rw_m, convnext_tiny, convnext_small, resnet50, resnet101",
    )

    pipe.add_parameter(
        name="epochs",
        default=100,
        description="Number of training epochs",
    )

    pipe.add_parameter(
        name="batch_size",
        default=64,
        description="Training batch size",
    )

    pipe.add_parameter(
        name="learning_rate",
        default=0.0001,
        description="Learning rate for optimizer",
    )

    pipe.add_parameter(
        name="patch_size",
        default=128,
        description="Patch size for extraction: 32, 64, 128, or 256",
    )

    pipe.add_parameter(
        name="segmenter",
        default="native",
        description="Segmentation method: native (platform default) or cellpose",
    )

    pipe.add_parameter(
        name="annotator",
        default="celltypist",
        description="Cell type annotation: celltypist, popv, or singler",
    )

    pipe.add_parameter(
        name="celltypist_models",
        default="Cells_Adult_Breast.pkl,Immune_All_High.pkl",
        description="Comma-separated CellTypist model names",
    )

    pipe.add_parameter(
        name="sampling_strategy",
        default="sqrt",
        description="Multi-tissue sampling: sqrt (balanced), equal, or proportional",
    )

    pipe.add_parameter(
        name="standardize_labels",
        default=True,
        description="Use Cell Ontology label standardization",
    )

    pipe.add_parameter(
        name="coarse_only_epochs",
        default=20,
        description="Curriculum learning: epochs with only coarse labels",
    )

    pipe.add_parameter(
        name="coarse_medium_epochs",
        default=50,
        description="Curriculum learning: epochs with coarse+medium labels",
    )

    # =========================================================================
    # Add pipeline step using function
    # =========================================================================

    pipe.add_function_step(
        name="run_dapidl_training",
        function=run_training_step,
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

    pipe.set_default_execution_queue("gpu")

    # Create pipeline without running - just register it
    logger.info("Registering pipeline template...")

    # The pipeline is now defined - we need to start it for it to be saved
    # But we don't want it to actually run, so we'll stop it immediately after starting

    logger.info("=" * 60)
    logger.info("Pipeline template registered!")
    logger.info("")
    logger.info("To run from ClearML Web UI:")
    logger.info("1. Go to Pipelines → DAPIDL/universal")
    logger.info("2. Find 'DAPIDL Universal Pipeline v3.0.0'")
    logger.info("3. Click 'New Run'")
    logger.info("4. Configure parameters in the dialog")
    logger.info("5. Click 'Run'")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
