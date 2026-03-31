#!/usr/bin/env python3
"""Evaluate already-trained baseline vs filtered models on rep2.

Uses trained models from /mnt/work/experiments/confidence_filtering_comparison/.
Creates LMDB for rep2 (needed for DAPI model inference in cross-validation).
"""

import json
import sys
from pathlib import Path

from loguru import logger

REP2_PATH = "/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep2/outs"
OUTPUT_ROOT = Path("/mnt/work/experiments/confidence_filtering_comparison")
PATCH_SIZE = 64

ANNOTATION_METHODS = [
    {"name": "celltypist", "params": {"model": "Cells_Adult_Breast.pkl"}},
    {"name": "celltypist", "params": {"model": "Immune_All_High.pkl"}},
    {"name": "celltypist", "params": {"model": "Immune_All_Low.pkl"}},
    {"name": "singler", "params": {"reference": "blueprint"}},
    {"name": "singler", "params": {"reference": "hpca"}},
]


def prepare_rep2():
    """Load data, segment, annotate, and create LMDB for rep2. Returns merged outputs."""
    from dapidl.pipeline.base import StepArtifacts
    from dapidl.pipeline.steps import DataLoaderStep, SegmentationStep
    from dapidl.pipeline.steps.data_loader import DataLoaderConfig
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationConfig,
        EnsembleAnnotationStep,
        MethodSpec,
    )
    from dapidl.pipeline.steps.lmdb_creation import LMDBCreationConfig, LMDBCreationStep
    from dapidl.pipeline.steps.segmentation import SegmentationStepConfig

    # Data loading
    data_config = DataLoaderConfig(platform="xenium", local_path=REP2_PATH)
    data_loader = DataLoaderStep(data_config)
    data_artifacts = data_loader.execute(StepArtifacts())

    # Segmentation
    seg_config = SegmentationStepConfig(
        segmenter="native",
        platform=data_artifacts.outputs.get("platform", "xenium"),
    )
    segmentation = SegmentationStep(seg_config)
    seg_artifacts = segmentation.execute(data_artifacts)

    # Annotation
    ensemble_config = EnsembleAnnotationConfig(
        methods=[MethodSpec.from_dict(m) for m in ANNOTATION_METHODS],
        min_agreement=2,
        confidence_threshold=0.5,
        use_confidence_weighting=False,
        fine_grained=True,
    )
    ensemble_step = EnsembleAnnotationStep(ensemble_config)
    annot_artifacts = ensemble_step.execute(
        StepArtifacts(inputs={}, outputs=seg_artifacts.outputs)
    )

    # LMDB creation for rep2 (needed for DAPI model inference)
    lmdb_config = LMDBCreationConfig(
        patch_size=PATCH_SIZE,
        normalization_method="adaptive",
        create_clearml_dataset=False,
    )
    lmdb_step = LMDBCreationStep(lmdb_config)
    # Mark no confidence filtering so LMDB path is deterministic
    lmdb_inputs = {**annot_artifacts.outputs, "confidence_filtering_skipped": True}
    lmdb_artifacts = lmdb_step.execute(
        StepArtifacts(inputs={}, outputs=lmdb_inputs)
    )

    # Merge all outputs
    lmdb_path = lmdb_artifacts.outputs.get("lmdb_path")
    merged = {
        **seg_artifacts.outputs,
        **annot_artifacts.outputs,
        **lmdb_artifacts.outputs,
        "patches_path": lmdb_path,  # cross_validation expects this key
        "tissue": "breast",
    }
    return merged


def evaluate_on_rep2(model_path: str, name: str, rep2_outputs: dict):
    """Evaluate a trained model on rep2 data."""
    from dapidl.pipeline.base import StepArtifacts
    from dapidl.pipeline.steps.cross_validation import CrossValidationConfig, CrossValidationStep

    logger.info(f"\n{'=' * 70}")
    logger.info(f"EVALUATING [{name}] on Rep2")
    logger.info(f"{'=' * 70}")

    merged = {
        **rep2_outputs,
        "model_path": model_path,
    }

    cv_config = CrossValidationConfig(
        run_leiden_check=True,
        run_dapi_check=True,
        run_consensus_check=False,
        output_dir=str(OUTPUT_ROOT / f"rep2_validation_{name}"),
    )
    cv_step = CrossValidationStep(cv_config)
    cv_artifacts = cv_step.execute(StepArtifacts(inputs={}, outputs=merged))

    validation_results = cv_artifacts.outputs.get("validation_results", {})
    logger.info(f"[{name}] Rep2 validation: {json.dumps(validation_results, indent=2, default=str)}")
    return validation_results


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    baseline_model = str(OUTPUT_ROOT / "baseline" / "best_model.pt")
    filtered_model = str(OUTPUT_ROOT / "filtered" / "best_model.pt")

    # Prepare rep2 data (shared between both evaluations)
    logger.info("Preparing Rep2 data (load + segment + annotate + LMDB)")
    rep2_outputs = prepare_rep2()

    # Evaluate both models
    logger.info("Evaluating both models on Rep2")
    baseline_eval = evaluate_on_rep2(baseline_model, "baseline", rep2_outputs)
    filtered_eval = evaluate_on_rep2(filtered_model, "filtered", rep2_outputs)

    # Print comparison
    print("\n" + "=" * 80)
    print("  REP2 EVALUATION COMPARISON")
    print("=" * 80)
    print(f"  {'Metric':<30} {'Baseline':>15} {'Filtered':>15} {'Delta':>10}")
    print("-" * 80)

    for key in ["dapi_agreement", "leiden_ari", "leiden_nmi"]:
        b_val = baseline_eval.get(key)
        f_val = filtered_eval.get(key)
        if b_val is not None and f_val is not None:
            delta = f_val - b_val
            print(f"  {'Rep2 ' + key:<30} {b_val:>15.4f} {f_val:>15.4f} {delta:>+10.4f}")

    print("=" * 80)

    # Save
    results = {"baseline": baseline_eval, "filtered": filtered_eval}
    with open(OUTPUT_ROOT / "rep2_eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved to {OUTPUT_ROOT / 'rep2_eval_results.json'}")


if __name__ == "__main__":
    main()
