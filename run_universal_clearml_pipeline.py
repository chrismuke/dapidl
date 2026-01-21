#!/usr/bin/env python3
"""Run universal pipeline as a proper ClearML Pipeline.

This script launches the universal multi-tissue pipeline using ClearML Pipeline
execution, which provides:
- Visibility in ClearML Web UI for non-technical users
- Step-by-step execution with caching
- Reproducibility and audit trail
- Remote execution on GPU queues

Prerequisites:
1. Base tasks must be created: python create_clearml_base_tasks.py
2. ClearML agents must be running on 'default' and 'gpu' queues
3. Raw datasets must be registered (optional - can use local paths)
"""

from pathlib import Path
from loguru import logger

from dapidl.pipeline.universal_controller import (
    UniversalDAPIPipelineController,
    UniversalPipelineConfig,
    TissueConfig,
)

# ClearML Dataset IDs (registered raw datasets)
# Map: directory_name -> ClearML dataset ID
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
    # MERSCOPE
    "merscope_breast": "80db08e6a5c94d7fb3c5e8f2a6d9b7c4",
}

# Local paths (fallback if not using ClearML datasets)
RAW_XENIUM = Path("~/datasets/raw/xenium").expanduser()
RAW_MERSCOPE = Path("~/datasets/raw/merscope").expanduser()

# Tissue to CellTypist model mapping
TISSUE_MODELS = {
    "breast": ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
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

# Dataset configurations (same as run_universal_all_tissues_clearml.py)
XENIUM_DATASETS = [
    ("breast_tumor_rep1", "breast", 1, 1.0),
    ("breast_tumor_rep2", "breast", 2, 1.0),
    ("colon_cancer_colon-panel", "colon", 2, 1.0),
    ("colon_normal_colon-panel", "colon", 2, 1.0),
    ("colorectal_cancer_io-panel", "colorectal", 2, 1.0),
    ("heart_normal_multi-tissue-panel", "heart", 2, 1.0),
    ("kidney_cancer_multi-tissue-panel", "kidney", 2, 1.0),
    ("kidney_normal_multi-tissue-panel", "kidney", 2, 1.0),
    ("liver_cancer_multi-tissue-panel", "liver", 2, 1.0),
    ("liver_normal_multi-tissue-panel", "liver", 2, 1.0),
    ("lung_2fov", "lung", 2, 0.5),
    ("lung_cancer_lung-panel", "lung", 2, 1.0),
    ("lymph_node_normal", "lymph_node", 2, 1.0),
    ("ovarian_cancer", "ovary", 2, 1.0),
    ("ovary_cancer_ff", "ovary", 2, 1.0),
    ("pancreas_cancer_multi-tissue-panel", "pancreas", 2, 1.0),
    ("skin_normal_sample1", "skin", 2, 1.0),
    ("skin_normal_sample2", "skin", 2, 1.0),
    ("tonsil_lymphoid-hyperplasia", "tonsil", 2, 1.0),
    ("tonsil_reactive-hyperplasia", "tonsil", 2, 1.0),
]

MERSCOPE_DATASETS = [
    ("breast", "breast", 2, 1.0),
]


def create_tissue_configs(use_clearml_datasets: bool = True) -> list[TissueConfig]:
    """Create TissueConfig objects for all available datasets.

    Args:
        use_clearml_datasets: If True, use ClearML dataset IDs (for remote execution).
                             If False, use local paths (for local execution).
    """
    tissues = []

    for dir_name, tissue, confidence, weight in XENIUM_DATASETS:
        models = TISSUE_MODELS.get(tissue, ["Immune_All_High.pkl"])

        if use_clearml_datasets:
            # Use ClearML dataset ID for remote execution
            dataset_id = CLEARML_DATASET_IDS.get(dir_name)
            if not dataset_id:
                logger.warning(f"No ClearML dataset ID for: {dir_name}")
                continue

            tissues.append(TissueConfig(
                dataset_id=dataset_id,
                tissue=tissue,
                platform="xenium",
                confidence_tier=confidence,
                weight_multiplier=weight,
                annotator="celltypist",
                model_names=models,
            ))
        else:
            # Use local path for local execution
            path = RAW_XENIUM / dir_name
            if not path.exists():
                logger.warning(f"Xenium dataset not found: {path}")
                continue

            tissues.append(TissueConfig(
                local_path=str(path),
                tissue=tissue,
                platform="xenium",
                confidence_tier=confidence,
                weight_multiplier=weight,
                annotator="celltypist",
                model_names=models,
            ))

    for dir_name, tissue, confidence, weight in MERSCOPE_DATASETS:
        models = TISSUE_MODELS.get(tissue, ["Immune_All_High.pkl"])
        merscope_key = f"merscope_{dir_name}"

        if use_clearml_datasets:
            # Use ClearML dataset ID for remote execution
            dataset_id = CLEARML_DATASET_IDS.get(merscope_key)
            if not dataset_id:
                logger.warning(f"No ClearML dataset ID for: {merscope_key}")
                continue

            tissues.append(TissueConfig(
                dataset_id=dataset_id,
                tissue=tissue,
                platform="merscope",
                confidence_tier=confidence,
                weight_multiplier=weight,
                annotator="celltypist",
                model_names=models,
            ))
        else:
            # Use local path for local execution
            path = RAW_MERSCOPE / dir_name
            if not path.exists():
                logger.warning(f"MERSCOPE dataset not found: {path}")
                continue

            tissues.append(TissueConfig(
                local_path=str(path),
                tissue=tissue,
                platform="merscope",
                confidence_tier=confidence,
                weight_multiplier=weight,
                annotator="celltypist",
                model_names=models,
            ))

    return tissues


def get_all_dataset_keys() -> str:
    """Get comma-separated list of all available dataset keys."""
    xenium_keys = [name for name, _, _, _ in XENIUM_DATASETS]
    merscope_keys = [f"merscope_{name}" for name, _, _, _ in MERSCOPE_DATASETS]
    return ",".join(xenium_keys + merscope_keys)


def create_tissue_configs_from_keys(
    dataset_keys: str,
    use_clearml_datasets: bool = True,
) -> list[TissueConfig]:
    """Create TissueConfig objects from comma-separated dataset keys."""
    from clearml import Dataset

    tissues = []
    keys = [k.strip() for k in dataset_keys.split(",") if k.strip()]

    # Build lookup maps
    xenium_map = {name: (tissue, confidence, weight) for name, tissue, confidence, weight in XENIUM_DATASETS}
    merscope_map = {f"merscope_{name}": (tissue, confidence, weight) for name, tissue, confidence, weight in MERSCOPE_DATASETS}

    for key in keys:
        # Check Xenium datasets
        if key in xenium_map:
            tissue, confidence, weight = xenium_map[key]
            models = TISSUE_MODELS.get(tissue, ["Immune_All_High.pkl"])

            if use_clearml_datasets:
                dataset_id = CLEARML_DATASET_IDS.get(key)
                if not dataset_id:
                    logger.warning(f"No ClearML dataset ID for: {key}")
                    continue
                tissues.append(TissueConfig(
                    dataset_id=dataset_id,
                    tissue=tissue,
                    platform="xenium",
                    confidence_tier=confidence,
                    weight_multiplier=weight,
                    annotator="celltypist",
                    model_names=models,
                ))
            else:
                path = RAW_XENIUM / key
                if not path.exists():
                    logger.warning(f"Xenium dataset not found: {path}")
                    continue
                tissues.append(TissueConfig(
                    local_path=str(path),
                    tissue=tissue,
                    platform="xenium",
                    confidence_tier=confidence,
                    weight_multiplier=weight,
                    annotator="celltypist",
                    model_names=models,
                ))

        # Check MERSCOPE datasets
        elif key in merscope_map:
            tissue, confidence, weight = merscope_map[key]
            models = TISSUE_MODELS.get(tissue, ["Immune_All_High.pkl"])
            dir_name = key.replace("merscope_", "")

            if use_clearml_datasets:
                dataset_id = CLEARML_DATASET_IDS.get(key)
                if not dataset_id:
                    logger.warning(f"No ClearML dataset ID for: {key}")
                    continue
                tissues.append(TissueConfig(
                    dataset_id=dataset_id,
                    tissue=tissue,
                    platform="merscope",
                    confidence_tier=confidence,
                    weight_multiplier=weight,
                    annotator="celltypist",
                    model_names=models,
                ))
            else:
                path = RAW_MERSCOPE / dir_name
                if not path.exists():
                    logger.warning(f"MERSCOPE dataset not found: {path}")
                    continue
                tissues.append(TissueConfig(
                    local_path=str(path),
                    tissue=tissue,
                    platform="merscope",
                    confidence_tier=confidence,
                    weight_multiplier=weight,
                    annotator="celltypist",
                    model_names=models,
                ))
        else:
            logger.warning(f"Unknown dataset key: {key}")

    return tissues


def main():
    import argparse
    from clearml import Task

    parser = argparse.ArgumentParser(
        description="Run Universal DAPIDL Pipeline on ClearML"
    )
    parser.add_argument(
        "--local", "-l",
        action="store_true",
        help="Run locally without ClearML agents (for debugging)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Create pipeline but don't start execution",
    )
    parser.add_argument(
        "--datasets", "-d",
        type=str,
        default=None,
        help="Comma-separated dataset keys (default: all datasets)",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--backbone", "-b",
        type=str,
        default="efficientnetv2_rw_s",
        help="CNN backbone (default: efficientnetv2_rw_s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Patch size (default: 128)",
    )
    parser.add_argument(
        "--segmenter",
        type=str,
        default="native",
        choices=["native", "cellpose"],
        help="Segmentation method (default: native)",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="sqrt",
        choices=["sqrt", "equal", "proportional"],
        help="Sampling strategy (default: sqrt)",
    )
    parser.add_argument(
        "--queue",
        type=str,
        default="gpu",
        help="ClearML GPU queue name (default: gpu)",
    )
    args = parser.parse_args()

    # Initialize ClearML Task for parameter tracking
    if not args.local:
        task = Task.init(
            project_name="DAPIDL/universal",
            task_name="dapidl-universal-all-tissues",
            task_type=Task.TaskTypes.controller,
            reuse_last_task_id=False,
        )

        # Define all configurable parameters (visible in ClearML Web UI)
        pipeline_params = {
            # Dataset selection - comma-separated list of dataset keys
            "datasets": args.datasets or get_all_dataset_keys(),

            # Model architecture
            "backbone": args.backbone,

            # Training parameters
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": 1e-4,

            # Patch extraction
            "patch_size": args.patch_size,

            # Segmentation
            "segmenter": args.segmenter,

            # Multi-tissue training
            "sampling_strategy": args.sampling,
            "standardize_labels": True,

            # Curriculum learning
            "coarse_only_epochs": 20,
            "coarse_medium_epochs": 50,

            # Confidence tier weights
            "tier1_weight": 1.0,
            "tier2_weight": 0.8,
            "tier3_weight": 0.5,

            # Execution
            "gpu_queue": args.queue,
        }

        # Connect to ClearML (makes them editable in Web UI)
        pipeline_params = task.connect(pipeline_params, name="Pipeline Configuration")

        # Add dataset options as comment for reference
        task.set_comment(f"""
Universal DAPIDL Pipeline - Configurable Parameters

Available datasets (comma-separated in 'datasets' parameter):
{chr(10).join(f'  - {k}' for k in get_all_dataset_keys().split(','))}

Available backbones:
  - efficientnetv2_rw_s (default)
  - efficientnetv2_rw_m
  - convnext_tiny
  - convnext_small
  - resnet50
  - resnet101

Segmenters: native, cellpose
Sampling: sqrt, equal, proportional
""")

        # Use connected parameters
        dataset_keys = pipeline_params["datasets"]
        epochs = pipeline_params["epochs"]
        backbone = pipeline_params["backbone"]
        batch_size = pipeline_params["batch_size"]
        patch_size = pipeline_params["patch_size"]
        segmenter = pipeline_params["segmenter"]
        sampling = pipeline_params["sampling_strategy"]
        queue = pipeline_params["gpu_queue"]
    else:
        # Local mode - use CLI args directly
        dataset_keys = args.datasets or get_all_dataset_keys()
        epochs = args.epochs
        backbone = args.backbone
        batch_size = args.batch_size
        patch_size = args.patch_size
        segmenter = args.segmenter
        sampling = args.sampling
        queue = args.queue

    # Create tissue configs from selected datasets
    tissues = create_tissue_configs_from_keys(dataset_keys, use_clearml_datasets=not args.local)

    if not tissues:
        logger.error("No datasets found!")
        return None

    # Pipeline configuration
    config = UniversalPipelineConfig(
        name="dapidl-universal-all-tissues",
        project="DAPIDL/universal",
        tissues=tissues,

        # Sampling
        sampling_strategy=sampling,

        # Segmentation
        segmenter=segmenter,

        # Patches
        patch_size=patch_size,
        output_format="lmdb",

        # Training
        backbone=backbone,
        epochs=epochs,
        batch_size=batch_size,

        # Curriculum learning
        coarse_only_epochs=20,
        coarse_medium_epochs=50,

        # Cell Ontology
        standardize_labels=True,

        # Execution mode
        execute_remotely=not args.local,
        gpu_queue=queue,

        # Output
        output_dir="experiment_universal_clearml_pipeline",
    )

    logger.info("=" * 60)
    logger.info("Universal Multi-Tissue Pipeline (ClearML Pipeline)")
    logger.info("=" * 60)
    logger.info(f"Total datasets: {len(tissues)}")
    logger.info(f"Execution mode: {'LOCAL' if args.local else 'CLEARML PIPELINE'}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Backbone: {args.backbone}")
    if not args.local:
        logger.info(f"GPU Queue: {args.queue}")
    logger.info("=" * 60)

    # Create controller
    controller = UniversalDAPIPipelineController(config)

    if args.dry_run:
        # Just create pipeline, don't run
        controller.create_pipeline()
        logger.info("Pipeline created but not started (--dry-run)")
        return None

    if args.local:
        # Run locally (same as before)
        result = controller.run_locally()
        logger.info("Local pipeline execution complete!")
        return result
    else:
        # Run as ClearML Pipeline
        logger.info("Starting ClearML Pipeline execution...")
        logger.info("View progress in ClearML Web UI: https://app.clear.ml/")
        pipeline_id = controller.run(wait=False)
        logger.info(f"Pipeline started: {pipeline_id}")
        logger.info("Pipeline is running in background. Monitor in ClearML Web UI.")
        return pipeline_id


if __name__ == "__main__":
    main()
