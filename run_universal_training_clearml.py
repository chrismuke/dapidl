#!/usr/bin/env python
"""Run universal training with ClearML tracking on pre-processed LMDB dataset."""

from pathlib import Path
from loguru import logger

from dapidl.pipeline.steps.universal_training import (
    UniversalDAPITrainingStep,
    UniversalTrainingConfig,
    TissueDatasetSpec,
)
from dapidl.pipeline.base import StepArtifacts


def main():
    # Configuration - Combined Xenium + MERSCOPE breast datasets
    xenium_path = "/home/chrism/datasets/derived/xenium-breast-xenium-finegrained-p128"
    merscope_path = "/home/chrism/datasets/raw/merscope/breast/pipeline_outputs/patches"
    output_dir = Path("experiment_universal_clearml")
    output_dir.mkdir(exist_ok=True)

    # Create step config
    config = UniversalTrainingConfig(
        backbone="efficientnetv2_rw_s",  # timm model name
        epochs=50,
        batch_size=64,
        learning_rate=1e-4,
        weight_decay=1e-5,
        patience=15,
        num_workers=4,
        output_dir=str(output_dir),
        # Curriculum learning settings
        coarse_weight=1.0,
        medium_weight=0.5,
        fine_weight=0.3,
        consistency_weight=0.1,
        coarse_only_epochs=15,      # Phase 1: epochs 1-15 (coarse only)
        coarse_medium_epochs=30,    # Phase 2: epochs 16-30 (coarse + medium)
        warmup_epochs=5,
        # Sampling - sqrt for balanced platform representation
        sampling_strategy="sqrt",
        # Class weighting
        tier1_weight=1.0,
        tier2_weight=0.8,
        tier3_weight=0.5,
        min_samples_per_class=20,
        standardize_labels=True,
        # Combined Xenium + MERSCOPE datasets
        datasets=[
            TissueDatasetSpec(
                path=xenium_path,
                tissue="breast",
                platform="xenium",
                confidence_tier=2,  # Consensus annotations
                weight_multiplier=1.0,
            ),
            TissueDatasetSpec(
                path=merscope_path,
                tissue="breast",
                platform="merscope",
                confidence_tier=2,  # Consensus annotations
                weight_multiplier=1.0,
            ),
        ],
    )

    # Create step
    step = UniversalDAPITrainingStep(config)

    # Create input artifacts (empty for direct LMDB dataset)
    artifacts = StepArtifacts(inputs={}, outputs={})

    logger.info("Starting universal training with ClearML tracking...")
    logger.info(f"Datasets: Xenium ({xenium_path}) + MERSCOPE ({merscope_path})")
    logger.info(f"Output: {output_dir}")

    # Execute training
    result_artifacts = step.execute(artifacts)

    # Extract metrics from artifacts
    test_metrics = result_artifacts.outputs.get("test_metrics", {})
    tissue_metrics = result_artifacts.outputs.get("tissue_metrics", {})

    logger.info("Training complete!")
    logger.info(f"Test metrics: {test_metrics}")
    logger.info(f"Tissue metrics: {tissue_metrics}")


if __name__ == "__main__":
    main()
