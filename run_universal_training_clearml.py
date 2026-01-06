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
    # Configuration
    dataset_path = "/mnt/work/datasets/derived/xenium-breast-xenium-finegrained-p128"
    output_dir = Path("experiment_universal_v2")
    output_dir.mkdir(exist_ok=True)

    # Create step config
    config = UniversalTrainingConfig(
        backbone="efficientnetv2_rw_s",  # timm model name
        epochs=30,
        batch_size=64,  # Full batch size (GPU now free)
        learning_rate=1e-4,
        weight_decay=1e-5,
        patience=10,
        num_workers=4,
        output_dir=str(output_dir),
        # Curriculum learning settings
        coarse_weight=1.0,
        medium_weight=0.5,
        fine_weight=0.3,
        consistency_weight=0.1,
        coarse_only_epochs=10,      # Phase 1: coarse only
        coarse_medium_epochs=10,    # Phase 2: coarse + medium (relative, so epochs 11-20)
        warmup_epochs=5,            # Longer warmup with quadratic curve
        # Sampling
        sampling_strategy="sqrt",
        # Class weighting
        tier1_weight=1.0,
        tier2_weight=0.8,
        tier3_weight=0.5,
        min_samples_per_class=20,
        standardize_labels=True,
        # Add dataset
        datasets=[
            TissueDatasetSpec(
                path=dataset_path,
                tissue="breast",
                platform="xenium",
                confidence_tier=1,
                weight_multiplier=1.0,
            )
        ],
    )

    # Create step
    step = UniversalDAPITrainingStep(config)

    # Create input artifacts (empty for direct LMDB dataset)
    artifacts = StepArtifacts(inputs={}, outputs={})

    logger.info("Starting universal training with ClearML tracking...")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output: {output_dir}")

    # Execute training
    test_metrics, tissue_metrics = step.execute(artifacts)

    logger.info("Training complete!")
    logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
