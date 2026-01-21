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

# Raw dataset paths
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


def create_tissue_configs() -> list[TissueConfig]:
    """Create TissueConfig objects for all available datasets."""
    tissues = []

    for dir_name, tissue, confidence, weight in XENIUM_DATASETS:
        path = RAW_XENIUM / dir_name
        if not path.exists():
            logger.warning(f"Xenium dataset not found: {path}")
            continue

        models = TISSUE_MODELS.get(tissue, ["Immune_All_High.pkl"])

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
        path = RAW_MERSCOPE / dir_name
        if not path.exists():
            logger.warning(f"MERSCOPE dataset not found: {path}")
            continue

        models = TISSUE_MODELS.get(tissue, ["Immune_All_High.pkl"])

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


def main():
    import argparse

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
        "--queue",
        type=str,
        default="gpu",
        help="ClearML GPU queue name (default: gpu)",
    )
    args = parser.parse_args()

    tissues = create_tissue_configs()

    if not tissues:
        logger.error("No datasets found!")
        return None

    # Pipeline configuration
    config = UniversalPipelineConfig(
        name="dapidl-universal-all-tissues",
        project="DAPIDL/universal",
        tissues=tissues,

        # Sampling
        sampling_strategy="sqrt",

        # Segmentation
        segmenter="native",

        # Patches
        patch_size=128,
        output_format="lmdb",

        # Training
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=64,

        # Curriculum learning
        coarse_only_epochs=20,
        coarse_medium_epochs=50,

        # Cell Ontology
        standardize_labels=True,

        # Execution mode
        execute_remotely=not args.local,
        gpu_queue=args.queue,

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
