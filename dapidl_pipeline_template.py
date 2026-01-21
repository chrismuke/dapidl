#!/usr/bin/env python3
"""DAPIDL Pipeline Template using PipelineDecorator.

This creates a reusable pipeline template that exposes all parameters
in the ClearML Web UI "New Run" dialog.

Run once to register the template:
    python dapidl_pipeline_template.py

Then use from ClearML Web UI:
1. Go to Pipelines → DAPIDL/universal
2. Click "New Run" on "DAPIDL Universal Pipeline"
3. Configure parameters and run
"""

from clearml import PipelineDecorator


@PipelineDecorator.component(
    return_values=["result"],
    execution_queue="gpu",
)
def run_universal_pipeline(
    datasets: str = "breast_tumor_rep1,breast_tumor_rep2",
    backbone: str = "efficientnetv2_rw_s",
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.0001,
    patch_size: int = 128,
    segmenter: str = "native",
    annotator: str = "celltypist",
    celltypist_models: str = "Cells_Adult_Breast.pkl,Immune_All_High.pkl",
    sampling_strategy: str = "sqrt",
    standardize_labels: bool = True,
    coarse_only_epochs: int = 20,
    coarse_medium_epochs: int = 50,
) -> dict:
    """Run the DAPIDL universal pipeline.

    Args:
        datasets: Comma-separated dataset keys
        backbone: CNN backbone (efficientnetv2_rw_s, convnext_small, resnet50)
        epochs: Training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        patch_size: Patch size (32, 64, 128, 256)
        segmenter: Segmentation method (native, cellpose)
        annotator: Annotation method (celltypist, popv, singler)
        celltypist_models: Comma-separated CellTypist model names
        sampling_strategy: Multi-tissue sampling (sqrt, equal, proportional)
        standardize_labels: Use Cell Ontology standardization
        coarse_only_epochs: Curriculum phase 1 epochs
        coarse_medium_epochs: Curriculum phase 2 epochs
    """
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

    # Parse datasets
    dataset_keys = [k.strip() for k in datasets.split(",") if k.strip()]
    logger.info(f"Selected {len(dataset_keys)} datasets: {dataset_keys}")

    # Build tissue configs
    tissues = []
    for key in dataset_keys:
        if key not in CLEARML_DATASET_IDS:
            logger.warning(f"Unknown dataset: {key}")
            continue

        tissue = DATASET_TISSUES[key]
        platform = "merscope" if key.startswith("merscope") else "xenium"
        dataset_id = CLEARML_DATASET_IDS[key]

        models = ([m.strip() for m in celltypist_models.split(",")]
                  if celltypist_models else TISSUE_MODELS.get(tissue, ["Immune_All_High.pkl"]))

        tissues.append(TissueConfig(
            dataset_id=dataset_id,
            tissue=tissue,
            platform=platform,
            confidence_tier=2,
            weight_multiplier=1.0,
            annotator=annotator,
            model_names=models,
        ))

    if not tissues:
        raise ValueError("No valid datasets found!")

    # Create config
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

    # Run
    controller = UniversalDAPIPipelineController(config)
    result = controller.run_locally()

    return {"status": "completed", "datasets": len(tissues)}


@PipelineDecorator.pipeline(
    name="DAPIDL Universal Pipeline",
    project="DAPIDL/universal",
    version="2.0.0",
    default_queue="gpu",
)
def dapidl_pipeline(
    datasets: str = "breast_tumor_rep1,breast_tumor_rep2",
    backbone: str = "efficientnetv2_rw_s",
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.0001,
    patch_size: int = 128,
    segmenter: str = "native",
    annotator: str = "celltypist",
    celltypist_models: str = "Cells_Adult_Breast.pkl,Immune_All_High.pkl",
    sampling_strategy: str = "sqrt",
    standardize_labels: bool = True,
    coarse_only_epochs: int = 20,
    coarse_medium_epochs: int = 50,
):
    """DAPIDL Universal Multi-Tissue Pipeline.

    Train a universal cell type classifier on multiple spatial transcriptomics datasets.

    Available datasets:
        breast_tumor_rep1, breast_tumor_rep2, merscope_breast
        colon_cancer_colon-panel, colon_normal_colon-panel
        colorectal_cancer_io-panel, heart_normal_multi-tissue-panel
        kidney_cancer_multi-tissue-panel, kidney_normal_multi-tissue-panel
        liver_cancer_multi-tissue-panel, liver_normal_multi-tissue-panel
        lung_2fov, lung_cancer_lung-panel, lymph_node_normal
        ovarian_cancer, ovary_cancer_ff, pancreas_cancer_multi-tissue-panel
        skin_normal_sample1, skin_normal_sample2
        tonsil_lymphoid-hyperplasia, tonsil_reactive-hyperplasia

    Available backbones:
        efficientnetv2_rw_s, efficientnetv2_rw_m, convnext_tiny, convnext_small, resnet50
    """
    result = run_universal_pipeline(
        datasets=datasets,
        backbone=backbone,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patch_size=patch_size,
        segmenter=segmenter,
        annotator=annotator,
        celltypist_models=celltypist_models,
        sampling_strategy=sampling_strategy,
        standardize_labels=standardize_labels,
        coarse_only_epochs=coarse_only_epochs,
        coarse_medium_epochs=coarse_medium_epochs,
    )
    return result


if __name__ == "__main__":
    # Run locally to register the pipeline template
    PipelineDecorator.run_locally()

    # This will execute once with defaults and register the template
    print("Registering DAPIDL Universal Pipeline template...")
    print("This will run once with default parameters to create the template.")
    print()
    print("After registration, go to ClearML Web UI:")
    print("  1. Pipelines → DAPIDL/universal")
    print("  2. Click 'New Run' on 'DAPIDL Universal Pipeline'")
    print("  3. Configure all parameters in the dialog")
    print("  4. Click 'Run'")
    print()

    # Run with minimal test to register
    dapidl_pipeline(
        datasets="lung_2fov",  # Smallest dataset for quick registration
        epochs=1,  # Minimal epochs for registration
    )
