#!/usr/bin/env python3
"""Compare training with vs without confidence filtering.

Runs on breast rep1 with best annotator combination (3 CellTypist + 2 SingleR),
then evaluates both models on rep2.

Shares annotation step (runs once) → forks into parallel training:
  A) baseline:  annotation → LMDB p64 → train 100 epochs
  B) filtered:  annotation → confidence filtering → LMDB p64 → train 100 epochs

After both complete, evaluates on rep2 using cross-validation step.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

# Paths
REP1_PATH = "/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1/outs"
REP2_PATH = "/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep2/outs"
OUTPUT_ROOT = Path("/mnt/work/experiments/confidence_filtering_comparison")

# Training config
PATCH_SIZE = 64
EPOCHS = 100
BATCH_SIZE = 32  # Reduced to fit 2 concurrent training runs on 24GB GPU
BACKBONE = "efficientnetv2_rw_s"

# Best annotator combination: 3 CellTypist + 2 SingleR (unweighted)
ANNOTATION_METHODS = [
    {"name": "celltypist", "params": {"model": "Cells_Adult_Breast.pkl"}},
    {"name": "celltypist", "params": {"model": "Immune_All_High.pkl"}},
    {"name": "celltypist", "params": {"model": "Immune_All_Low.pkl"}},
    {"name": "singler", "params": {"reference": "blueprint"}},
    {"name": "singler", "params": {"reference": "hpca"}},
]


def run_shared_steps():
    """Run data loading, segmentation, annotation (shared between both runs)."""
    from dapidl.pipeline.base import StepArtifacts
    from dapidl.pipeline.steps import DataLoaderStep, SegmentationStep
    from dapidl.pipeline.steps.data_loader import DataLoaderConfig
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationConfig,
        EnsembleAnnotationStep,
        MethodSpec,
    )
    from dapidl.pipeline.steps.segmentation import SegmentationStepConfig

    logger.info("=" * 70)
    logger.info("SHARED STEPS: Data Loading + Segmentation + Annotation")
    logger.info("=" * 70)

    # Step 1: Data Loader
    logger.info("Step 1: Data Loader")
    data_config = DataLoaderConfig(
        platform="xenium",
        local_path=REP1_PATH,
    )
    data_loader = DataLoaderStep(data_config)
    data_artifacts = data_loader.execute(StepArtifacts())

    # Step 2: Segmentation
    logger.info("Step 2: Segmentation")
    seg_config = SegmentationStepConfig(
        segmenter="native",
        platform=data_artifacts.outputs.get("platform", "xenium"),
    )
    segmentation = SegmentationStep(seg_config)
    seg_artifacts = segmentation.execute(data_artifacts)

    # Step 3: Annotation (ensemble: 3 CellTypist + 2 SingleR)
    logger.info("Step 3: Ensemble Annotation (3 CellTypist + 2 SingleR)")
    ensemble_config = EnsembleAnnotationConfig(
        methods=[MethodSpec.from_dict(m) for m in ANNOTATION_METHODS],
        min_agreement=2,
        confidence_threshold=0.5,
        use_confidence_weighting=False,  # Unweighted is better per benchmark
        fine_grained=True,
    )
    ensemble_step = EnsembleAnnotationStep(ensemble_config)
    annot_artifacts = ensemble_step.execute(
        StepArtifacts(inputs={}, outputs=seg_artifacts.outputs)
    )

    # Merge outputs
    merged = {
        **seg_artifacts.outputs,
        **annot_artifacts.outputs,
        "tissue": "breast",
        "dataset_id": "rep1",
    }

    logger.info(f"Annotation complete: {len(ANNOTATION_METHODS)} methods")
    return merged


def run_training_branch(name: str, merged_outputs: dict, apply_filtering: bool, output_dir: Path):
    """Run LMDB creation + training for one branch."""
    from dapidl.pipeline.base import StepArtifacts
    from dapidl.pipeline.steps.lmdb_creation import LMDBCreationConfig, LMDBCreationStep
    from dapidl.pipeline.steps.universal_training import (
        TissueDatasetSpec,
        UniversalDAPITrainingStep,
        UniversalTrainingConfig,
    )

    logger.info("=" * 70)
    logger.info(f"BRANCH [{name}]: {'WITH' if apply_filtering else 'WITHOUT'} confidence filtering")
    logger.info("=" * 70)

    outputs = dict(merged_outputs)
    # Mark whether confidence filtering is applied — this affects
    # the LMDB output path to prevent baseline/filtered collisions
    outputs["confidence_filtering_skipped"] = not apply_filtering
    artifacts = StepArtifacts(inputs={}, outputs=outputs)

    # Step 3.5: Confidence Filtering (only for filtered branch)
    if apply_filtering:
        from dapidl.pipeline.steps.confidence_filtering import (
            ConfidenceFilteringConfig,
            ConfidenceFilteringStep,
        )

        logger.info(f"[{name}] Step 3.5: Confidence Filtering")
        cf_config = ConfidenceFilteringConfig(
            enabled=True,
            tissue_type="breast",
            min_confidence=0.4,
            use_panglao_markers=True,
            spatial_k=15,
        )
        cf_step = ConfidenceFilteringStep(cf_config)
        cf_artifacts = cf_step.execute(artifacts)
        artifacts = StepArtifacts(inputs={}, outputs=cf_artifacts.outputs)

    # Step 4: LMDB Creation
    logger.info(f"[{name}] Step 4: LMDB Creation (p{PATCH_SIZE})")
    lmdb_config = LMDBCreationConfig(
        patch_size=PATCH_SIZE,
        normalization_method="adaptive",
        create_clearml_dataset=False,
    )
    lmdb_step = LMDBCreationStep(lmdb_config)
    lmdb_artifacts = lmdb_step.execute(artifacts)

    lmdb_path = lmdb_artifacts.outputs.get("lmdb_path")
    logger.info(f"[{name}] LMDB created at {lmdb_path}")

    # Step 5: Training
    logger.info(f"[{name}] Step 5: Training ({EPOCHS} epochs, batch_size={BATCH_SIZE})")
    train_config = UniversalTrainingConfig(
        backbone=BACKBONE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-4,
        sampling_strategy="sqrt",
        output_dir=str(output_dir),
        datasets=[
            TissueDatasetSpec(
                path=lmdb_path,
                tissue="breast",
                platform="xenium",
                confidence_tier=1,
            )
        ],
    )
    training_step = UniversalDAPITrainingStep(train_config)
    training_artifacts = training_step.execute(
        StepArtifacts(inputs={}, outputs={"dataset_configs": [train_config.datasets[0]]})
    )

    model_path = training_artifacts.outputs.get("model_path")
    test_metrics = training_artifacts.outputs.get("test_metrics", {})

    logger.info(f"[{name}] Training complete!")
    logger.info(f"[{name}] Model: {model_path}")
    logger.info(f"[{name}] Test metrics: {json.dumps(test_metrics, indent=2, default=str)}")

    return {
        "name": name,
        "model_path": model_path,
        "test_metrics": test_metrics,
        "lmdb_path": lmdb_path,
        "lmdb_stats": lmdb_artifacts.outputs.get("extraction_stats", {}),
        "confidence_stats": artifacts.outputs.get("confidence_stats", None),
    }


def evaluate_on_rep2(model_path: str, name: str):
    """Evaluate a trained model on rep2 data."""
    from dapidl.pipeline.base import StepArtifacts
    from dapidl.pipeline.steps import DataLoaderStep, SegmentationStep
    from dapidl.pipeline.steps.annotation import AnnotationStep, AnnotationStepConfig
    from dapidl.pipeline.steps.cross_validation import CrossValidationConfig, CrossValidationStep
    from dapidl.pipeline.steps.data_loader import DataLoaderConfig
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationConfig,
        EnsembleAnnotationStep,
        MethodSpec,
    )
    from dapidl.pipeline.steps.segmentation import SegmentationStepConfig

    logger.info(f"\n{'=' * 70}")
    logger.info(f"EVALUATING [{name}] on Rep2")
    logger.info(f"{'=' * 70}")

    # Load rep2 data
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

    # Annotation on rep2 (same methods as training)
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

    # Cross-validation: use DAPI model to predict on rep2
    merged = {
        **seg_artifacts.outputs,
        **annot_artifacts.outputs,
        "model_path": model_path,
        "tissue": "breast",
    }

    cv_config = CrossValidationConfig(
        run_leiden_check=True,
        run_dapi_check=True,
        run_consensus_check=False,
    )
    cv_step = CrossValidationStep(cv_config)
    cv_artifacts = cv_step.execute(StepArtifacts(inputs={}, outputs=merged))

    validation_results = cv_artifacts.outputs.get("validation_results", {})
    logger.info(f"[{name}] Rep2 validation: {json.dumps(validation_results, indent=2, default=str)}")
    return validation_results


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(OUTPUT_ROOT / "comparison.log", level="DEBUG")

    start_time = time.time()

    # ── Phase 1: Shared annotation (single process) ──
    logger.info("PHASE 1: Shared annotation on Rep1")
    merged_outputs = run_shared_steps()

    annotation_time = time.time() - start_time
    logger.info(f"Annotation completed in {annotation_time:.0f}s")

    # ── Phase 2: Sequential training ──
    # NOTE: Running sequentially instead of threaded because DataLoader
    # workers fork() from threads, corrupting Numba/TBB state and causing
    # random training failures. Sequential also avoids GPU memory contention.
    logger.info("\nPHASE 2: Sequential training (baseline then filtered)")

    baseline_dir = OUTPUT_ROOT / "baseline"
    filtered_dir = OUTPUT_ROOT / "filtered"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)

    # Run baseline first
    try:
        baseline_result = run_training_branch("baseline", merged_outputs, False, baseline_dir)
    except Exception as e:
        logger.error(f"[baseline] FAILED: {e}")
        baseline_result = {"error": str(e)}

    # Run filtered second
    try:
        filtered_result = run_training_branch("filtered", merged_outputs, True, filtered_dir)
    except Exception as e:
        logger.error(f"[filtered] FAILED: {e}")
        filtered_result = {"error": str(e)}

    if "error" in baseline_result or "error" in filtered_result:
        logger.error("One or both training runs failed!")
        for name, res in [("baseline", baseline_result), ("filtered", filtered_result)]:
            if "error" in res:
                logger.error(f"  [{name}]: {res['error']}")
        sys.exit(1)

    training_time = time.time() - start_time - annotation_time
    logger.info(f"Both training runs completed in {training_time:.0f}s")

    # ── Phase 3: Evaluate on Rep2 ──
    logger.info("\nPHASE 3: Evaluation on Rep2")

    baseline_eval = evaluate_on_rep2(baseline_result["model_path"], "baseline")
    filtered_eval = evaluate_on_rep2(filtered_result["model_path"], "filtered")

    # ── Summary ──
    total_time = time.time() - start_time

    summary = {
        "config": {
            "patch_size": PATCH_SIZE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "backbone": BACKBONE,
            "annotation_methods": [m["name"] + ":" + str(m["params"]) for m in ANNOTATION_METHODS],
        },
        "baseline": {
            "lmdb_patches": baseline_result["lmdb_stats"].get("n_patches", "?"),
            "lmdb_classes": baseline_result["lmdb_stats"].get("n_classes", "?"),
            "train_metrics": baseline_result["test_metrics"],
            "rep2_validation": baseline_eval,
        },
        "filtered": {
            "confidence_stats": filtered_result.get("confidence_stats"),
            "lmdb_patches": filtered_result["lmdb_stats"].get("n_patches", "?"),
            "lmdb_classes": filtered_result["lmdb_stats"].get("n_classes", "?"),
            "train_metrics": filtered_result["test_metrics"],
            "rep2_validation": filtered_eval,
        },
        "timing": {
            "annotation_seconds": annotation_time,
            "training_seconds": training_time,
            "total_seconds": total_time,
        },
    }

    # Save results
    results_path = OUTPUT_ROOT / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print comparison table
    print("\n" + "=" * 80)
    print("  CONFIDENCE FILTERING COMPARISON RESULTS")
    print("=" * 80)
    print(f"  Annotation: {len(ANNOTATION_METHODS)} methods | Patch: {PATCH_SIZE}px | Epochs: {EPOCHS}")
    print(f"  Training data: Rep1 (167k cells) | Evaluation: Rep2 (118k cells)")
    print("-" * 80)
    print(f"  {'Metric':<30} {'Baseline':>15} {'Filtered':>15} {'Delta':>10}")
    print("-" * 80)

    b_patches = baseline_result["lmdb_stats"].get("n_patches", 0)
    f_patches = filtered_result["lmdb_stats"].get("n_patches", 0)
    print(f"  {'Training patches':<30} {b_patches:>15,} {f_patches:>15,} {f_patches - b_patches:>+10,}")

    if filtered_result.get("confidence_stats"):
        cs = filtered_result["confidence_stats"]
        print(f"  {'Retention rate':<30} {'100%':>15} {cs['retention_rate']:>14.1%} {cs['retention_rate'] - 1:>+9.1%}")
        print(f"  {'Cells filtered':<30} {'0':>15} {cs['n_filtered']:>15,}")
        print(f"  {'Overall confidence':<30} {'-':>15} {cs['overall_score']:>15.3f}")

    # Rep2 evaluation comparison
    for key in ["dapi_agreement", "leiden_ari", "leiden_nmi"]:
        b_val = baseline_eval.get(key)
        f_val = filtered_eval.get(key)
        if b_val is not None and f_val is not None:
            delta = f_val - b_val
            print(f"  {'Rep2 ' + key:<30} {b_val:>15.4f} {f_val:>15.4f} {delta:>+10.4f}")

    print("=" * 80)
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"  Results saved to: {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
