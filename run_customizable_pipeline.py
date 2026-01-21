#!/usr/bin/env python3
"""Customizable Universal DAPIDL Pipeline with ClearML parameter selection.

This script provides a user-friendly interface for running the DAPIDL pipeline
with customizable parameters visible in ClearML Web UI:

- Dataset selection (choose 1 to N from available raw datasets)
- Backbone architecture selection
- Annotation method selection (CellTypist models, SingleR)
- Training hyperparameters
- Segmentation options

Usage:
    # Local execution with defaults
    python run_customizable_pipeline.py --local

    # Create ClearML task for parameter editing in Web UI
    python run_customizable_pipeline.py --create-task

    # Run with specific parameters
    python run_customizable_pipeline.py --local \
        --datasets breast_tumor_rep1,merscope_breast \
        --backbone convnext_small \
        --epochs 50
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Any

from clearml import Task
from loguru import logger

from dapidl.pipeline.universal_controller import (
    UniversalDAPIPipelineController,
    UniversalPipelineConfig,
    TissueConfig,
)


# ============================================================================
# Available Options
# ============================================================================

# All available raw datasets (name -> (path, tissue, platform))
AVAILABLE_DATASETS = {
    # Xenium Breast
    "xenium_breast_rep1": ("~/datasets/raw/xenium/breast_tumor_rep1", "breast", "xenium"),
    "xenium_breast_rep2": ("~/datasets/raw/xenium/breast_tumor_rep2", "breast", "xenium"),
    # Xenium Colon
    "xenium_colon_cancer": ("~/datasets/raw/xenium/colon_cancer_colon-panel", "colon", "xenium"),
    "xenium_colon_normal": ("~/datasets/raw/xenium/colon_normal_colon-panel", "colon", "xenium"),
    "xenium_colorectal_cancer": ("~/datasets/raw/xenium/colorectal_cancer_io-panel", "colorectal", "xenium"),
    # Xenium Heart
    "xenium_heart_normal": ("~/datasets/raw/xenium/heart_normal_multi-tissue-panel", "heart", "xenium"),
    # Xenium Kidney
    "xenium_kidney_cancer": ("~/datasets/raw/xenium/kidney_cancer_multi-tissue-panel", "kidney", "xenium"),
    "xenium_kidney_normal": ("~/datasets/raw/xenium/kidney_normal_multi-tissue-panel", "kidney", "xenium"),
    # Xenium Liver
    "xenium_liver_cancer": ("~/datasets/raw/xenium/liver_cancer_multi-tissue-panel", "liver", "xenium"),
    "xenium_liver_normal": ("~/datasets/raw/xenium/liver_normal_multi-tissue-panel", "liver", "xenium"),
    # Xenium Lung
    "xenium_lung_2fov": ("~/datasets/raw/xenium/lung_2fov", "lung", "xenium"),
    "xenium_lung_cancer": ("~/datasets/raw/xenium/lung_cancer_lung-panel", "lung", "xenium"),
    # Xenium Lymph Node
    "xenium_lymph_node": ("~/datasets/raw/xenium/lymph_node_normal", "lymph_node", "xenium"),
    # Xenium Ovary
    "xenium_ovarian_cancer": ("~/datasets/raw/xenium/ovarian_cancer", "ovary", "xenium"),
    "xenium_ovary_cancer_ff": ("~/datasets/raw/xenium/ovary_cancer_ff", "ovary", "xenium"),
    # Xenium Pancreas
    "xenium_pancreas_cancer": ("~/datasets/raw/xenium/pancreas_cancer_multi-tissue-panel", "pancreas", "xenium"),
    # Xenium Skin
    "xenium_skin_sample1": ("~/datasets/raw/xenium/skin_normal_sample1", "skin", "xenium"),
    "xenium_skin_sample2": ("~/datasets/raw/xenium/skin_normal_sample2", "skin", "xenium"),
    # Xenium Tonsil
    "xenium_tonsil_lymphoid": ("~/datasets/raw/xenium/tonsil_lymphoid-hyperplasia", "tonsil", "xenium"),
    "xenium_tonsil_reactive": ("~/datasets/raw/xenium/tonsil_reactive-hyperplasia", "tonsil", "xenium"),
    # MERSCOPE
    "merscope_breast": ("~/datasets/raw/merscope/breast", "breast", "merscope"),
}

# Available backbones
AVAILABLE_BACKBONES = [
    "efficientnetv2_rw_s",  # Default, good balance
    "efficientnetv2_rw_m",  # Larger, more accurate
    "convnext_tiny",        # Modern architecture
    "convnext_small",       # Larger ConvNeXt
    "resnet50",             # Classic baseline
    "resnet101",            # Larger ResNet
    "vit_small_patch16_224", # Vision Transformer
]

# Available CellTypist models by tissue
CELLTYPIST_MODELS = {
    "breast": ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl", "Immune_All_Low.pkl"],
    "colon": ["Cells_Intestinal_Tract.pkl", "Immune_All_High.pkl"],
    "colorectal": ["Cells_Intestinal_Tract.pkl", "Immune_All_High.pkl"],
    "heart": ["Healthy_Adult_Heart.pkl", "Immune_All_High.pkl"],
    "kidney": ["Immune_All_High.pkl", "Immune_All_Low.pkl"],
    "liver": ["Healthy_Human_Liver.pkl", "Immune_All_High.pkl"],
    "lung": ["Human_Lung_Atlas.pkl", "Cells_Lung_Airway.pkl", "Immune_All_High.pkl"],
    "lymph_node": ["Immune_All_High.pkl", "Immune_All_Low.pkl"],
    "ovary": ["Immune_All_High.pkl", "Immune_All_Low.pkl"],
    "pancreas": ["Adult_Human_PancreaticIslet.pkl", "Immune_All_High.pkl"],
    "skin": ["Adult_Human_Skin.pkl", "Immune_All_High.pkl"],
    "tonsil": ["Cells_Human_Tonsil.pkl", "Immune_All_High.pkl"],
}

# All available CellTypist models
ALL_CELLTYPIST_MODELS = sorted(set(
    model for models in CELLTYPIST_MODELS.values() for model in models
))

# Available segmenters
AVAILABLE_SEGMENTERS = ["native", "cellpose"]

# Available annotation strategies
AVAILABLE_STRATEGIES = ["consensus", "single", "ensemble"]

# Sampling strategies
AVAILABLE_SAMPLING = ["sqrt", "equal", "proportional"]


# ============================================================================
# Parameter Configuration
# ============================================================================

@dataclass
class CustomizablePipelineConfig:
    """Configuration exposed to ClearML Web UI for user customization."""

    # Dataset Selection (comma-separated list of dataset keys)
    # Users can select any subset of available datasets
    datasets: str = "xenium_breast_rep1,merscope_breast"

    # Model Architecture
    backbone: str = "efficientnetv2_rw_s"

    # Annotation Settings
    annotator: str = "celltypist"  # celltypist, singler, popv
    celltypist_models: str = "Cells_Adult_Breast.pkl,Immune_All_High.pkl"
    use_singler_hpca: bool = True
    use_singler_blueprint: bool = True
    annotation_strategy: str = "consensus"  # consensus, single, ensemble

    # Segmentation
    segmenter: str = "native"  # native, cellpose
    cellpose_diameter: int = 40
    cellpose_flow_threshold: float = 0.4

    # Patch Extraction
    patch_size: int = 128  # 32, 64, 128, 256

    # Training Hyperparameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.0001

    # Curriculum Learning
    coarse_only_epochs: int = 20
    coarse_medium_epochs: int = 50

    # Sampling Strategy
    sampling_strategy: str = "sqrt"  # sqrt, equal, proportional

    # Confidence Tier Weights
    tier1_weight: float = 1.0   # Ground truth
    tier2_weight: float = 0.8   # Consensus
    tier3_weight: float = 0.5   # Predicted

    # Cell Ontology
    standardize_labels: bool = True

    # Execution
    execute_remotely: bool = False
    gpu_queue: str = "gpu"

    # Output
    output_dir: str = "experiment_customizable"


def connect_parameters_to_clearml(task: Task, config: CustomizablePipelineConfig) -> CustomizablePipelineConfig:
    """Connect configuration to ClearML task for Web UI editing.

    This creates editable parameters in the ClearML Web UI.
    """
    # Create a dictionary of parameters with helpful descriptions
    params = {
        "Dataset Selection": {
            "datasets": config.datasets,
            "_datasets_help": f"Comma-separated list. Available: {', '.join(sorted(AVAILABLE_DATASETS.keys()))}",
        },
        "Model Architecture": {
            "backbone": config.backbone,
            "_backbone_options": ", ".join(AVAILABLE_BACKBONES),
        },
        "Annotation Settings": {
            "annotator": config.annotator,
            "celltypist_models": config.celltypist_models,
            "_celltypist_options": ", ".join(ALL_CELLTYPIST_MODELS),
            "use_singler_hpca": config.use_singler_hpca,
            "use_singler_blueprint": config.use_singler_blueprint,
            "annotation_strategy": config.annotation_strategy,
            "_strategy_options": ", ".join(AVAILABLE_STRATEGIES),
        },
        "Segmentation": {
            "segmenter": config.segmenter,
            "_segmenter_options": ", ".join(AVAILABLE_SEGMENTERS),
            "cellpose_diameter": config.cellpose_diameter,
            "cellpose_flow_threshold": config.cellpose_flow_threshold,
        },
        "Patch Extraction": {
            "patch_size": config.patch_size,
            "_patch_size_options": "32, 64, 128, 256",
        },
        "Training": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "coarse_only_epochs": config.coarse_only_epochs,
            "coarse_medium_epochs": config.coarse_medium_epochs,
        },
        "Sampling": {
            "sampling_strategy": config.sampling_strategy,
            "_sampling_options": ", ".join(AVAILABLE_SAMPLING),
            "tier1_weight": config.tier1_weight,
            "tier2_weight": config.tier2_weight,
            "tier3_weight": config.tier3_weight,
        },
        "Output": {
            "standardize_labels": config.standardize_labels,
            "output_dir": config.output_dir,
        },
    }

    # Connect all parameters to ClearML
    connected = task.connect(params)

    # Update config from connected parameters (allows Web UI editing)
    config.datasets = connected["Dataset Selection"]["datasets"]
    config.backbone = connected["Model Architecture"]["backbone"]
    config.annotator = connected["Annotation Settings"]["annotator"]
    config.celltypist_models = connected["Annotation Settings"]["celltypist_models"]
    config.use_singler_hpca = connected["Annotation Settings"]["use_singler_hpca"]
    config.use_singler_blueprint = connected["Annotation Settings"]["use_singler_blueprint"]
    config.annotation_strategy = connected["Annotation Settings"]["annotation_strategy"]
    config.segmenter = connected["Segmentation"]["segmenter"]
    config.cellpose_diameter = connected["Segmentation"]["cellpose_diameter"]
    config.cellpose_flow_threshold = connected["Segmentation"]["cellpose_flow_threshold"]
    config.patch_size = connected["Patch Extraction"]["patch_size"]
    config.epochs = connected["Training"]["epochs"]
    config.batch_size = connected["Training"]["batch_size"]
    config.learning_rate = connected["Training"]["learning_rate"]
    config.coarse_only_epochs = connected["Training"]["coarse_only_epochs"]
    config.coarse_medium_epochs = connected["Training"]["coarse_medium_epochs"]
    config.sampling_strategy = connected["Sampling"]["sampling_strategy"]
    config.tier1_weight = connected["Sampling"]["tier1_weight"]
    config.tier2_weight = connected["Sampling"]["tier2_weight"]
    config.tier3_weight = connected["Sampling"]["tier3_weight"]
    config.standardize_labels = connected["Output"]["standardize_labels"]
    config.output_dir = connected["Output"]["output_dir"]

    return config


def create_tissue_configs(config: CustomizablePipelineConfig) -> list[TissueConfig]:
    """Create TissueConfig objects from user-selected datasets."""
    tissues = []

    # Parse selected datasets
    selected = [d.strip() for d in config.datasets.split(",") if d.strip()]

    for dataset_key in selected:
        if dataset_key not in AVAILABLE_DATASETS:
            logger.warning(f"Unknown dataset: {dataset_key}, skipping")
            continue

        path_str, tissue, platform = AVAILABLE_DATASETS[dataset_key]
        path = Path(path_str).expanduser()

        if not path.exists():
            logger.warning(f"Dataset path not found: {path}, skipping")
            continue

        # Determine CellTypist models for this tissue
        if config.annotator == "celltypist":
            # Use user-specified models or defaults for tissue
            if config.celltypist_models:
                models = [m.strip() for m in config.celltypist_models.split(",")]
            else:
                models = CELLTYPIST_MODELS.get(tissue, ["Immune_All_High.pkl"])
        else:
            models = []

        # Determine confidence tier (breast_rep1 has ground truth)
        confidence_tier = 1 if "rep1" in dataset_key else 2

        tissues.append(TissueConfig(
            local_path=str(path),
            tissue=tissue,
            platform=platform,
            confidence_tier=confidence_tier,
            weight_multiplier=1.0,
            annotator=config.annotator,
            model_names=models,
        ))

        logger.info(f"Added dataset: {dataset_key} ({tissue}/{platform})")

    return tissues


def run_pipeline(config: CustomizablePipelineConfig, local: bool = True) -> dict[str, Any]:
    """Run the customizable pipeline with the given configuration."""

    tissues = create_tissue_configs(config)

    if not tissues:
        logger.error("No valid datasets selected!")
        return {"error": "No valid datasets"}

    # Create UniversalPipelineConfig
    pipeline_config = UniversalPipelineConfig(
        name=f"dapidl-custom-{len(tissues)}datasets",
        project="DAPIDL/universal",
        tissues=tissues,
        sampling_strategy=config.sampling_strategy,
        segmenter=config.segmenter,
        diameter=config.cellpose_diameter,
        flow_threshold=config.cellpose_flow_threshold,
        patch_size=config.patch_size,
        backbone=config.backbone,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        coarse_only_epochs=config.coarse_only_epochs,
        coarse_medium_epochs=config.coarse_medium_epochs,
        tier1_weight=config.tier1_weight,
        tier2_weight=config.tier2_weight,
        tier3_weight=config.tier3_weight,
        standardize_labels=config.standardize_labels,
        execute_remotely=not local,
        gpu_queue=config.gpu_queue,
        output_dir=config.output_dir,
    )

    logger.info("=" * 60)
    logger.info("Customizable DAPIDL Pipeline")
    logger.info("=" * 60)
    logger.info(f"Selected datasets: {len(tissues)}")
    for t in tissues:
        path_name = Path(t.local_path).name if t.local_path else "unknown"
        logger.info(f"  - {t.tissue}/{t.platform}: {path_name}")
    logger.info(f"Backbone: {config.backbone}")
    logger.info(f"Annotator: {config.annotator}")
    if config.annotator == "celltypist":
        logger.info(f"CellTypist models: {config.celltypist_models}")
    logger.info(f"Segmenter: {config.segmenter}")
    logger.info(f"Patch size: {config.patch_size}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Sampling: {config.sampling_strategy}")
    logger.info("=" * 60)

    # Create and run controller
    controller = UniversalDAPIPipelineController(pipeline_config)

    if local:
        result = controller.run_locally()
        logger.info("Pipeline completed!")
        return result
    else:
        pipeline_id = controller.run(wait=False)
        logger.info(f"Pipeline started: {pipeline_id}")
        return {"pipeline_id": pipeline_id}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Customizable DAPIDL Pipeline with ClearML Web UI support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available Datasets:
  {chr(10).join(f'  {k}' for k in sorted(AVAILABLE_DATASETS.keys()))}

Available Backbones:
  {', '.join(AVAILABLE_BACKBONES)}

Available CellTypist Models:
  {', '.join(ALL_CELLTYPIST_MODELS)}

Examples:
  # Run with default settings (2 breast datasets)
  python run_customizable_pipeline.py --local

  # Run with specific datasets
  python run_customizable_pipeline.py --local \\
      --datasets xenium_breast_rep1,xenium_lung_2fov,merscope_breast

  # Run with different backbone
  python run_customizable_pipeline.py --local --backbone convnext_small

  # Create ClearML task for Web UI parameter editing
  python run_customizable_pipeline.py --create-task
""",
    )

    parser.add_argument("--local", "-l", action="store_true",
                       help="Run locally without ClearML agents")
    parser.add_argument("--create-task", action="store_true",
                       help="Create ClearML task and exit (for Web UI editing)")
    parser.add_argument("--datasets", "-d", type=str, default=None,
                       help="Comma-separated list of dataset keys")
    parser.add_argument("--backbone", "-b", type=str, default=None,
                       help="Model backbone architecture")
    parser.add_argument("--epochs", "-e", type=int, default=None,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size for training")
    parser.add_argument("--patch-size", type=int, default=None,
                       help="Patch size (32, 64, 128, 256)")
    parser.add_argument("--segmenter", type=str, default=None,
                       choices=AVAILABLE_SEGMENTERS,
                       help="Segmentation method")
    parser.add_argument("--annotator", type=str, default=None,
                       help="Annotation method (celltypist, singler, popv)")
    parser.add_argument("--celltypist-models", type=str, default=None,
                       help="Comma-separated CellTypist model names")
    parser.add_argument("--sampling", type=str, default=None,
                       choices=AVAILABLE_SAMPLING,
                       help="Sampling strategy")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                       help="Output directory for results")

    args = parser.parse_args()

    # Create base configuration
    config = CustomizablePipelineConfig()

    # Override with CLI arguments
    if args.datasets:
        config.datasets = args.datasets
    if args.backbone:
        config.backbone = args.backbone
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.patch_size:
        config.patch_size = args.patch_size
    if args.segmenter:
        config.segmenter = args.segmenter
    if args.annotator:
        config.annotator = args.annotator
    if args.celltypist_models:
        config.celltypist_models = args.celltypist_models
    if args.sampling:
        config.sampling_strategy = args.sampling
    if args.output_dir:
        config.output_dir = args.output_dir

    # Initialize ClearML Task
    task = Task.init(
        project_name="DAPIDL/universal",
        task_name="Customizable Pipeline",
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
    )

    # Connect parameters to ClearML for Web UI editing
    config = connect_parameters_to_clearml(task, config)

    if args.create_task:
        logger.info("ClearML task created. Edit parameters in Web UI, then clone and run.")
        logger.info(f"Task URL: https://app.clear.ml/projects/*/experiments/{task.id}")
        task.close()
        return

    # Run the pipeline
    result = run_pipeline(config, local=args.local or not config.execute_remotely)

    return result


if __name__ == "__main__":
    main()
