"""Run universal pipeline on ClearML with ALL available raw datasets.

Includes:
- 20 Xenium datasets (various tissues)
- 1 MERSCOPE dataset (breast)

Uses appropriate CellTypist models for each tissue type.
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
# Each tissue gets appropriate models for its cell types
TISSUE_MODELS = {
    # Breast
    "breast": ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],

    # Colon/Colorectal
    "colon": ["Cells_Intestinal_Tract.pkl", "Immune_All_High.pkl"],
    "colorectal": ["Cells_Intestinal_Tract.pkl", "Immune_All_High.pkl"],

    # Heart
    "heart": ["Healthy_Adult_Heart.pkl", "Immune_All_High.pkl"],

    # Kidney
    "kidney": ["Immune_All_High.pkl", "Immune_All_Low.pkl"],  # No kidney-specific model

    # Liver
    "liver": ["Healthy_Human_Liver.pkl", "Immune_All_High.pkl"],

    # Lung
    "lung": ["Human_Lung_Atlas.pkl", "Cells_Lung_Airway.pkl", "Immune_All_High.pkl"],

    # Lymph node
    "lymph_node": ["Immune_All_High.pkl", "Immune_All_Low.pkl"],

    # Ovary
    "ovary": ["Immune_All_High.pkl", "Immune_All_Low.pkl"],  # No ovary-specific model

    # Pancreas
    "pancreas": ["Adult_Human_PancreaticIslet.pkl", "Immune_All_High.pkl"],

    # Skin
    "skin": ["Adult_Human_Skin.pkl", "Immune_All_High.pkl"],

    # Tonsil
    "tonsil": ["Cells_Human_Tonsil.pkl", "Immune_All_High.pkl"],
}

# Dataset configurations
# Format: (directory_name, tissue_type, confidence_tier, weight)
XENIUM_DATASETS = [
    # Breast (ground truth available for rep1)
    ("breast_tumor_rep1", "breast", 1, 1.0),
    ("breast_tumor_rep2", "breast", 2, 1.0),

    # Colon
    ("colon_cancer_colon-panel", "colon", 2, 1.0),
    ("colon_normal_colon-panel", "colon", 2, 1.0),
    # NOTE: colorectal_cancer_io-panel uses io-panel (different gene panel)
    ("colorectal_cancer_io-panel", "colorectal", 2, 1.0),

    # Heart
    ("heart_normal_multi-tissue-panel", "heart", 2, 1.0),

    # Kidney
    ("kidney_cancer_multi-tissue-panel", "kidney", 2, 1.0),
    ("kidney_normal_multi-tissue-panel", "kidney", 2, 1.0),

    # Liver
    ("liver_cancer_multi-tissue-panel", "liver", 2, 1.0),
    ("liver_normal_multi-tissue-panel", "liver", 2, 1.0),

    # Lung
    ("lung_2fov", "lung", 2, 0.5),  # Small test dataset
    ("lung_cancer_lung-panel", "lung", 2, 1.0),

    # Lymph node (NOTE: Previous run had partially corrupt patches)
    ("lymph_node_normal", "lymph_node", 2, 1.0),

    # Ovary
    ("ovarian_cancer", "ovary", 2, 1.0),  # Xenium Prime, 5K panel
    ("ovary_cancer_ff", "ovary", 2, 1.0),

    # Pancreas
    ("pancreas_cancer_multi-tissue-panel", "pancreas", 2, 1.0),

    # Skin
    ("skin_normal_sample1", "skin", 2, 1.0),
    ("skin_normal_sample2", "skin", 2, 1.0),

    # Tonsil
    ("tonsil_lymphoid-hyperplasia", "tonsil", 2, 1.0),
    ("tonsil_reactive-hyperplasia", "tonsil", 2, 1.0),
]

MERSCOPE_DATASETS = [
    ("breast", "breast", 2, 1.0),
]


def create_tissue_configs():
    """Create TissueConfig objects for all available datasets."""
    tissues = []

    # Add Xenium datasets
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
        logger.info(f"Added Xenium {tissue}: {dir_name}")

    # Add MERSCOPE datasets
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
        logger.info(f"Added MERSCOPE {tissue}: {dir_name}")

    return tissues


def main():
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
        segmenter="native",  # Use native segmentation (faster)

        # Patches
        patch_size=128,
        output_format="lmdb",

        # Training
        backbone="efficientnetv2_rw_s",
        epochs=100,  # More epochs for larger dataset
        batch_size=64,

        # Curriculum learning
        coarse_only_epochs=20,
        coarse_medium_epochs=50,

        # Cell Ontology
        standardize_labels=True,

        # Run locally (ClearML base tasks need to be created first)
        execute_remotely=False,
        gpu_queue="gpu",  # ClearML queue (for future remote runs)

        # Output
        output_dir="experiment_universal_all_tissues",
    )

    logger.info("=" * 60)
    logger.info("Universal Multi-Tissue Pipeline (ClearML)")
    logger.info("=" * 60)
    logger.info(f"Total datasets: {len(tissues)}")

    # Count by platform
    xenium_count = sum(1 for t in tissues if t.platform == "xenium")
    merscope_count = sum(1 for t in tissues if t.platform == "merscope")
    logger.info(f"  Xenium: {xenium_count}")
    logger.info(f"  MERSCOPE: {merscope_count}")

    # Count by tissue
    tissue_counts = {}
    for t in tissues:
        tissue_counts[t.tissue] = tissue_counts.get(t.tissue, 0) + 1
    logger.info("By tissue:")
    for tissue, count in sorted(tissue_counts.items()):
        logger.info(f"  {tissue}: {count}")

    logger.info(f"Output: {config.output_dir}")
    logger.info(f"ClearML Queue: {config.gpu_queue}")
    logger.info("=" * 60)

    # Create and run controller locally
    controller = UniversalDAPIPipelineController(config)
    result = controller.run_locally()

    logger.info("Pipeline complete!")
    return result


if __name__ == "__main__":
    main()
