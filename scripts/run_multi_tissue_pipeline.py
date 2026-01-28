"""Run complete universal pipeline with 3 Xenium + 1 MERSCOPE from raw data.

Tests:
- Full pipeline from raw data to training
- Caching of prepared datasets
- Per-dataset normalization
- Cell Ontology standardization across platforms
"""
from pathlib import Path
from loguru import logger

from dapidl.pipeline.universal_controller import (
    UniversalDAPIPipelineController,
    UniversalPipelineConfig,
    TissueConfig,
)

# Raw dataset paths
# Note: breast_tumor_rep1 has an 'outs/' subdirectory (older format)
# Other Xenium datasets have files at root (newer format)
# The data loader handles both structures automatically
RAW_XENIUM_BREAST = Path("~/datasets/raw/xenium/breast_tumor_rep1").expanduser()
RAW_XENIUM_LUNG = Path("~/datasets/raw/xenium/lung_cancer_lung-panel").expanduser()
RAW_XENIUM_LYMPH = Path("~/datasets/raw/xenium/lymph_node_normal").expanduser()
RAW_MERSCOPE_BREAST = Path("~/datasets/raw/merscope/breast").expanduser()

# Verify paths exist
for name, path in [
    ("Xenium Breast", RAW_XENIUM_BREAST),
    ("Xenium Lung", RAW_XENIUM_LUNG),
    ("Xenium Lymph Node", RAW_XENIUM_LYMPH),
    ("MERSCOPE Breast", RAW_MERSCOPE_BREAST),
]:
    if not path.exists():
        logger.warning(f"{name} not found at {path}")
    else:
        logger.info(f"Found {name} at {path}")

def main():
    # Create tissue configurations
    # NOTE: lung and lymph_node datasets have corrupt patches (all zeros)
    #       from a previous pipeline run. Skipping for now.
    tissues = [
        # Xenium breast (main dataset with ground truth annotations)
        TissueConfig(
            local_path=str(RAW_XENIUM_BREAST),
            tissue="breast",
            platform="xenium",
            confidence_tier=1,  # Ground truth available
            weight_multiplier=1.0,
            annotator="celltypist",
            model_names=["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
        ),
        # # Xenium lung cancer - SKIPPED: corrupt patches (all zeros)
        # TissueConfig(
        #     local_path=str(RAW_XENIUM_LUNG),
        #     tissue="lung",
        #     platform="xenium",
        #     confidence_tier=2,  # CellTypist consensus
        #     weight_multiplier=1.0,
        #     annotator="celltypist",
        #     model_names=["Cells_Lung_Airway.pkl", "Immune_All_High.pkl"],
        # ),
        # # Xenium lymph node - SKIPPED: corrupt patches (all zeros)
        # TissueConfig(
        #     local_path=str(RAW_XENIUM_LYMPH),
        #     tissue="lymph_node",
        #     platform="xenium",
        #     confidence_tier=2,
        #     weight_multiplier=1.0,
        #     annotator="celltypist",
        #     model_names=["Immune_All_High.pkl", "Immune_All_Low.pkl"],  # Immune-focused models
        # ),
        # MERSCOPE breast
        TissueConfig(
            local_path=str(RAW_MERSCOPE_BREAST),
            tissue="breast",
            platform="merscope",
            confidence_tier=2,
            weight_multiplier=1.0,
            annotator="celltypist",
            model_names=["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
        ),
    ]

    # Pipeline configuration
    config = UniversalPipelineConfig(
        name="dapidl-universal-4tissue",
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
        epochs=50,
        batch_size=64,
        
        # Curriculum learning
        coarse_only_epochs=16,
        coarse_medium_epochs=30,
        
        # Cell Ontology
        standardize_labels=True,
        
        # Run locally
        execute_remotely=False,
        
        # Output
        output_dir="experiment_universal_4tissue",
    )

    logger.info("=" * 60)
    logger.info("Universal Multi-Tissue Pipeline")
    logger.info("=" * 60)
    logger.info(f"Datasets: {len(tissues)}")
    for t in tissues:
        logger.info(f"  - {t.tissue}/{t.platform}: {t.local_path}")
    logger.info(f"Output: {config.output_dir}")
    logger.info("=" * 60)

    # Create and run controller locally
    controller = UniversalDAPIPipelineController(config)
    result = controller.run_locally()
    
    logger.info("Pipeline complete!")
    return result


if __name__ == "__main__":
    main()
