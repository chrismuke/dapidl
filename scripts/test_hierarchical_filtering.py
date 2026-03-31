#!/usr/bin/env python3
"""Test three-pass hierarchical confidence filtering end-to-end.

Pipeline:
  Rep1: annotation (hierarchical) -> confidence filtering -> LMDB -> hierarchical training
  Rep2: LMDB patches -> model inference -> compare vs Janesick ground truth

Measures model accuracy at coarse (Epithelial/Immune/Stromal) and fine-grained levels.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger

# -- Paths --
REP1_PATH = "/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1/outs"
REP2_PATH = "/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep2/outs"
REP2_GT_PATH = "/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep2/celltypes_ground_truth_rep2_supervised.xlsx"
OUTPUT_ROOT = Path("/mnt/work/experiments/hierarchical_filtering_test")
PATCH_SIZE = 64
EPOCHS = 100
BATCH_SIZE = 64
BACKBONE = "efficientnetv2_rw_s"

ANNOTATION_METHODS = [
    {"name": "celltypist", "params": {"model": "Cells_Adult_Breast.pkl"}},
    {"name": "celltypist", "params": {"model": "Immune_All_High.pkl"}},
    {"name": "celltypist", "params": {"model": "Immune_All_Low.pkl"}},
    {"name": "singler", "params": {"reference": "blueprint"}},
    {"name": "singler", "params": {"reference": "hpca"}},
]

# Janesick ground truth -> coarse category mapping
GT_TO_COARSE = {
    "DCIS_1": "Epithelial",
    "DCIS_2": "Epithelial",
    "Invasive_Tumor": "Epithelial",
    "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_ACTA2+": "Epithelial",
    "Myoepi_KRT15+": "Epithelial",
    "B_Cells": "Immune",
    "CD4+_T_Cells": "Immune",
    "CD8+_T_Cells": "Immune",
    "Macrophages_1": "Immune",
    "Macrophages_2": "Immune",
    "IRF7+_DCs": "Immune",
    "LAMP3+_DCs": "Immune",
    "Mast_Cells": "Immune",
    "Stromal": "Stromal",
    "Endothelial": "Endothelial",
    "Perivascular-Like": "Stromal",
}


def run_rep1_pipeline():
    """Run full pipeline on rep1 with hierarchical filtering enabled."""
    from dapidl.pipeline.base import StepArtifacts
    from dapidl.pipeline.steps import DataLoaderStep, SegmentationStep
    from dapidl.pipeline.steps.confidence_filtering import (
        ConfidenceFilteringConfig,
        ConfidenceFilteringStep,
    )
    from dapidl.pipeline.steps.data_loader import DataLoaderConfig
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationConfig,
        EnsembleAnnotationStep,
        MethodSpec,
    )
    from dapidl.pipeline.steps.lmdb_creation import LMDBCreationConfig, LMDBCreationStep
    from dapidl.pipeline.steps.segmentation import SegmentationStepConfig

    logger.info("=" * 70)
    logger.info("PHASE 1: Rep1 Pipeline (hierarchical filtering)")
    logger.info("=" * 70)

    # Step 1: Data loading
    logger.info("Step 1: Data loading")
    data_config = DataLoaderConfig(platform="xenium", local_path=REP1_PATH)
    data_loader = DataLoaderStep(data_config)
    data_artifacts = data_loader.execute(StepArtifacts())

    # Step 2: Segmentation
    logger.info("Step 2: Segmentation (native)")
    seg_config = SegmentationStepConfig(
        segmenter="native",
        platform=data_artifacts.outputs.get("platform", "xenium"),
    )
    segmentation = SegmentationStep(seg_config)
    seg_artifacts = segmentation.execute(data_artifacts)

    # Step 3: Ensemble annotation WITH hierarchical filtering
    logger.info("Step 3: Ensemble annotation (hierarchical_filtering=True)")
    ensemble_config = EnsembleAnnotationConfig(
        methods=[MethodSpec.from_dict(m) for m in ANNOTATION_METHODS],
        min_agreement=2,
        confidence_threshold=0.5,
        use_confidence_weighting=False,
        fine_grained=True,
        # Enable three-pass hierarchical filtering
        hierarchical_filtering=True,
        fine_agreement_threshold=4,
        medium_agreement_threshold=3,
        coarse_agreement_threshold=2,
    )
    ensemble_step = EnsembleAnnotationStep(ensemble_config)
    annot_artifacts = ensemble_step.execute(
        StepArtifacts(inputs={}, outputs=seg_artifacts.outputs)
    )

    # Check confidence_level in output
    annot_path = annot_artifacts.outputs.get("annotations_path")
    if annot_path:
        annot_df = pl.read_parquet(annot_path)
        logger.info(f"Annotations: {annot_df.shape[0]} cells, columns: {annot_df.columns}")
        if "confidence_level" in annot_df.columns:
            dist = annot_df["confidence_level"].value_counts().sort("confidence_level")
            logger.info(f"Confidence level distribution:\n{dist}")
        else:
            logger.warning("No confidence_level column in annotations!")

    # Step 4: Confidence filtering (hierarchical adjustment)
    logger.info("Step 4: Confidence filtering (hierarchical)")
    cf_config = ConfidenceFilteringConfig(
        enabled=True,
        tissue_type="breast",
        min_confidence=0.4,
        use_panglao_markers=True,
        spatial_k=15,
    )
    cf_step = ConfidenceFilteringStep(cf_config)
    cf_artifacts = cf_step.execute(
        StepArtifacts(inputs={}, outputs=annot_artifacts.outputs)
    )

    # Log filtering stats
    cf_stats = cf_artifacts.outputs.get("confidence_stats", {})
    logger.info(f"Confidence filtering stats: {json.dumps(cf_stats, indent=2, default=str)}")

    # Step 5: Dataset creation (LMDB)
    logger.info("Step 5: Dataset creation (LMDB)")
    lmdb_config = LMDBCreationConfig(
        patch_size=PATCH_SIZE,
        normalization_method="adaptive",
        create_clearml_dataset=False,
        output_format="lmdb",
    )
    lmdb_step = LMDBCreationStep(lmdb_config)
    lmdb_artifacts = lmdb_step.execute(cf_artifacts)

    lmdb_path = lmdb_artifacts.outputs.get("lmdb_path")
    logger.info(f"LMDB created at: {lmdb_path}")

    # Check confidence_levels.npy was saved
    if lmdb_path:
        cl_path = Path(lmdb_path) / "confidence_levels.npy"
        if cl_path.exists():
            cl = np.load(cl_path)
            unique, counts = np.unique(cl, return_counts=True)
            logger.info(f"confidence_levels.npy: {dict(zip(unique.tolist(), counts.tolist()))}")
        else:
            logger.warning(f"confidence_levels.npy not found at {cl_path}")

    merged = {
        **seg_artifacts.outputs,
        **annot_artifacts.outputs,
        **cf_artifacts.outputs,
        **lmdb_artifacts.outputs,
        "tissue": "breast",
    }
    # Training step expects 'patches_path', LMDB step outputs 'lmdb_path'
    if "lmdb_path" in merged and "patches_path" not in merged:
        merged["patches_path"] = merged["lmdb_path"]
    return merged


def train_hierarchical_model(rep1_outputs: dict):
    """Train hierarchical curriculum model on rep1 LMDB."""
    from dapidl.pipeline.steps.hierarchical_training import (
        HierarchicalTrainingConfig,
        HierarchicalTrainingStep,
    )
    from dapidl.pipeline.base import StepArtifacts

    logger.info("=" * 70)
    logger.info("PHASE 2: Hierarchical Training (curriculum learning)")
    logger.info("=" * 70)

    output_dir = OUTPUT_ROOT / "training"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = HierarchicalTrainingConfig(
        backbone=BACKBONE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-4,
        coarse_only_epochs=20,
        coarse_medium_epochs=50,
        warmup_epochs=5,
        coarse_weight=1.0,
        medium_weight=0.5,
        fine_weight=0.3,
        consistency_weight=0.1,
        label_smoothing=0.1,
        use_focal=False,
        max_weight_ratio=10.0,
        patience=15,
        use_wandb=False,
        output_dir=str(output_dir),
        upload_to_s3=False,
    )
    training_step = HierarchicalTrainingStep(train_config)
    train_artifacts = training_step.execute(
        StepArtifacts(inputs={}, outputs=rep1_outputs)
    )

    model_path = train_artifacts.outputs.get("model_path")
    test_metrics = train_artifacts.outputs.get("test_metrics", {})
    logger.info(f"Model saved: {model_path}")
    logger.info(f"Test metrics: {json.dumps(test_metrics, indent=2, default=str)}")

    return model_path, test_metrics


def create_rep2_lmdb():
    """Create LMDB patches for rep2 (needed for model inference)."""
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

    logger.info("=" * 70)
    logger.info("PHASE 3a: Rep2 LMDB creation")
    logger.info("=" * 70)

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

    # Annotation (needed for LMDB labels)
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

    # Dataset creation (LMDB)
    lmdb_config = LMDBCreationConfig(
        patch_size=PATCH_SIZE,
        normalization_method="adaptive",
        create_clearml_dataset=False,
        output_format="lmdb",
    )
    lmdb_step = LMDBCreationStep(lmdb_config)
    lmdb_inputs = {**annot_artifacts.outputs, "confidence_filtering_skipped": True}
    lmdb_artifacts = lmdb_step.execute(
        StepArtifacts(inputs={}, outputs=lmdb_inputs)
    )

    lmdb_path = lmdb_artifacts.outputs.get("lmdb_path")
    logger.info(f"Rep2 LMDB: {lmdb_path}")

    return lmdb_path, annot_artifacts.outputs


def run_inference(model, loader, device):
    """Run model inference, return per-level predictions."""
    all_coarse_preds = []
    all_medium_preds = []
    all_fine_preds = []

    with torch.no_grad():
        for images, _labels in loader:
            images = images.to(device)
            output = model(images)
            all_coarse_preds.append(output.coarse_logits.argmax(1).cpu().numpy())
            if output.medium_logits is not None:
                all_medium_preds.append(output.medium_logits.argmax(1).cpu().numpy())
            if output.fine_logits is not None:
                all_fine_preds.append(output.fine_logits.argmax(1).cpu().numpy())

    return (
        np.concatenate(all_coarse_preds),
        np.concatenate(all_medium_preds) if all_medium_preds else None,
        np.concatenate(all_fine_preds) if all_fine_preds else None,
    )


def evaluate_on_rep2_ground_truth(model_path: str, rep2_lmdb_path: str):
    """Evaluate trained model on rep2 against Janesick ground truth.

    Returns dict with coarse/medium/fine accuracy and F1 scores.
    """
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    from dapidl.data.hierarchical_dataset import HierarchicalDataset, HierarchicalLabels
    from dapidl.data.transforms import get_val_transforms
    from dapidl.models.hierarchical import HierarchicalClassifier

    logger.info("=" * 70)
    logger.info("PHASE 3b: Rep2 Ground Truth Evaluation")
    logger.info("=" * 70)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalClassifier.from_checkpoint(model_path)
    model = model.to(device)
    model.set_active_heads({"coarse", "medium", "fine"})
    model.train(False)

    hierarchy_config = model.hierarchy_config
    logger.info(
        f"Model hierarchy: coarse={hierarchy_config.num_coarse} "
        f"({hierarchy_config.coarse_names}), "
        f"medium={hierarchy_config.num_medium}, "
        f"fine={hierarchy_config.num_fine}"
    )

    # Load rep2 LMDB dataset — HierarchicalDataset auto-loads class_mapping.json
    # and builds hierarchy from CL ontology
    lmdb_path = Path(rep2_lmdb_path)
    dataset = HierarchicalDataset(
        data_path=lmdb_path,
        transform=get_val_transforms(),
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True
    )

    # Load ground truth
    gt_pd = pd.read_excel(REP2_GT_PATH, sheet_name="Xenium R2 Fig1-5 (supervised)")
    gt_df = pl.from_pandas(gt_pd)
    gt_df = gt_df.rename({"Barcode": "cell_id", "Cluster": "gt_type"})
    # Filter out Unlabeled and Hybrid types
    gt_df = gt_df.filter(pl.col("gt_type").is_in(list(GT_TO_COARSE.keys())))
    gt_df = gt_df.with_columns(
        pl.col("gt_type").replace_strict(GT_TO_COARSE).alias("gt_coarse")
    )
    logger.info(f"Ground truth: {gt_df.shape[0]} cells with valid labels")

    # Reconstruct cell_id -> LMDB index mapping by replaying LMDB creation filtering.
    # LMDB doesn't store cell_ids, so we reconstruct from annotations + cells.parquet
    # using the same edge/class filtering logic as LMDBCreationStep._create_lmdb().
    annot_parquet = Path("/mnt/work/datasets/pipeline_outputs/xenium-breast-tumor-rep2/ensemble_annotation/annotations.parquet")
    cells_parquet = Path(REP2_PATH) / "cells.parquet"
    annot_df = pl.read_parquet(annot_parquet)
    cells_df = pl.read_parquet(cells_parquet)
    annot_df = annot_df.with_columns(pl.col("cell_id").cast(pl.Int64))
    cells_df = cells_df.with_columns(pl.col("cell_id").cast(pl.Int64))
    id_df = annot_df.join(cells_df.select(["cell_id", "x_centroid", "y_centroid"]), on="cell_id", how="left")

    # Edge filter: same defaults as LMDBCreationConfig (patch_size=64, edge_margin_px=64)
    import tifffile
    with tifffile.TiffFile(str(Path(REP2_PATH) / "morphology_focus.ome.tif")) as tif:
        img_h, img_w = tif.pages[0].shape
    half_patch = PATCH_SIZE // 2
    edge_margin = 64  # LMDBCreationConfig default
    id_df = id_df.filter(
        (pl.col("x_centroid") >= half_patch + edge_margin)
        & (pl.col("x_centroid") < img_w - half_patch - edge_margin)
        & (pl.col("y_centroid") >= half_patch + edge_margin)
        & (pl.col("y_centroid") < img_h - half_patch - edge_margin)
    )
    # Class filter: only cells with predicted_type in class_mapping
    class_mapping = json.load(open(lmdb_path / "class_mapping.json"))
    id_df = id_df.filter(pl.col("predicted_type").is_in(list(class_mapping.keys())))
    cell_ids = id_df["cell_id"].to_list()
    logger.info(f"Reconstructed cell_id mapping: {len(cell_ids)} cells (LMDB has {len(dataset)} patches)")

    # Run inference
    logger.info("Running inference on rep2 patches...")
    coarse_preds, medium_preds, fine_preds = run_inference(model, loader, device)
    logger.info(f"Predictions: {len(coarse_preds)} patches")

    # Map prediction indices to names
    coarse_pred_names = [hierarchy_config.coarse_names[i] for i in coarse_preds]

    # Truncate cell_ids to match prediction count (should be identical)
    if len(cell_ids) != len(coarse_preds):
        logger.warning(f"Cell ID count ({len(cell_ids)}) != prediction count ({len(coarse_preds)}), truncating")
        cell_ids = cell_ids[:len(coarse_preds)]

    pred_df = pl.DataFrame({
        "cell_id": cell_ids,
        "pred_coarse": coarse_pred_names,
    })

    if medium_preds is not None:
        medium_pred_names = [hierarchy_config.medium_names[i] for i in medium_preds]
        pred_df = pred_df.with_columns(pl.Series("pred_medium", medium_pred_names))

    if fine_preds is not None:
        # class_mapping was loaded above during cell_id reconstruction
        idx_to_class = {v: k for k, v in class_mapping.items()}
        fine_pred_names = [idx_to_class.get(i, "Unknown") for i in fine_preds]
        pred_df = pred_df.with_columns(pl.Series("pred_fine", fine_pred_names))

    # Join predictions with ground truth
    pred_df = pred_df.with_columns(pl.col("cell_id").cast(gt_df["cell_id"].dtype))
    merged = pred_df.join(gt_df, on="cell_id", how="inner")
    logger.info(f"Matched {merged.shape[0]} cells between predictions and ground truth")

    if merged.shape[0] == 0:
        logger.error("No cells matched! Check cell_id format.")
        logger.info(f"Pred cell_ids sample: {pred_df['cell_id'].head(5).to_list()}")
        logger.info(f"GT cell_ids sample: {gt_df['cell_id'].head(5).to_list()}")
        return {}

    # -- Coarse evaluation --
    # Map model coarse categories to match ground truth conventions
    # Model may have: Endothelial, Epithelial, Immune, Stromal, Neural, Unknown
    # GT has: Endothelial, Epithelial, Immune, Stromal
    coarse_remap = {}
    for name in hierarchy_config.coarse_names:
        name_lower = name.lower()
        if name_lower == "unknown":
            coarse_remap[name] = "Unknown"
        elif name_lower == "neural":
            coarse_remap[name] = "Unknown"  # No neural cells in GT
        else:
            coarse_remap[name] = name

    merged = merged.with_columns(
        pl.col("pred_coarse").replace_strict(coarse_remap, default="Unknown").alias("pred_coarse_mapped")
    )

    # Filter out Unknown predictions for fair comparison
    coarse_eval = merged.filter(
        (pl.col("pred_coarse_mapped") != "Unknown")
        & (pl.col("gt_coarse") != "Unknown")
    )

    gt_coarse = coarse_eval["gt_coarse"].to_list()
    pred_coarse = coarse_eval["pred_coarse_mapped"].to_list()

    coarse_acc = accuracy_score(gt_coarse, pred_coarse)
    coarse_f1 = f1_score(gt_coarse, pred_coarse, average="macro", zero_division=0)
    coarse_f1_weighted = f1_score(gt_coarse, pred_coarse, average="weighted", zero_division=0)

    logger.info(f"\n{'=' * 70}")
    logger.info("COARSE-LEVEL EVALUATION (Endothelial / Epithelial / Immune / Stromal)")
    logger.info(f"{'=' * 70}")
    logger.info(f"Accuracy: {coarse_acc:.4f}")
    logger.info(f"Macro F1: {coarse_f1:.4f}")
    logger.info(f"Weighted F1: {coarse_f1_weighted:.4f}")
    logger.info(f"\n{classification_report(gt_coarse, pred_coarse, zero_division=0)}")

    # Confusion matrix
    labels_sorted = sorted(set(gt_coarse + pred_coarse))
    cm = confusion_matrix(gt_coarse, pred_coarse, labels=labels_sorted)
    logger.info("Confusion matrix (rows=GT, cols=Pred):")
    logger.info(f"Labels: {labels_sorted}")
    for i, row in enumerate(cm):
        logger.info(f"  {labels_sorted[i]:>15s}: {row}")

    results = {
        "coarse": {
            "accuracy": float(coarse_acc),
            "macro_f1": float(coarse_f1),
            "weighted_f1": float(coarse_f1_weighted),
            "n_cells": len(gt_coarse),
            "classification_report": classification_report(
                gt_coarse, pred_coarse, output_dict=True, zero_division=0
            ),
        },
    }

    # -- Fine-grained evaluation (where possible) --
    if "pred_fine" in merged.columns:
        logger.info(f"\n{'=' * 70}")
        logger.info("FINE-GRAINED EVALUATION")
        logger.info(f"{'=' * 70}")

        fine_dist = merged["pred_fine"].value_counts().sort("count", descending=True)
        logger.info(f"Model fine prediction distribution:\n{fine_dist.head(25)}")

        gt_fine_dist = merged["gt_type"].value_counts().sort("count", descending=True)
        logger.info(f"Ground truth fine distribution:\n{gt_fine_dist.head(25)}")

        results["fine_distributions"] = {
            "model": dict(fine_dist.iter_rows()),
            "ground_truth": dict(gt_fine_dist.iter_rows()),
        }

    # Save results
    results_path = OUTPUT_ROOT / "rep2_ground_truth_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")

    return results


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(OUTPUT_ROOT / "experiment.log", level="DEBUG")

    start_time = time.time()

    # -- Phase 1: Rep1 pipeline with hierarchical filtering --
    logger.info("Starting hierarchical filtering experiment")
    rep1_outputs = run_rep1_pipeline()

    annotation_time = time.time() - start_time
    logger.info(f"Phase 1 complete in {annotation_time / 60:.1f} minutes")

    # -- Phase 2: Hierarchical training --
    model_path, train_metrics = train_hierarchical_model(rep1_outputs)

    training_time = time.time() - start_time - annotation_time
    logger.info(f"Phase 2 complete in {training_time / 60:.1f} minutes")

    # -- Phase 3: Rep2 evaluation against ground truth --
    rep2_lmdb_path, rep2_annot_outputs = create_rep2_lmdb()
    gt_results = evaluate_on_rep2_ground_truth(model_path, rep2_lmdb_path)

    total_time = time.time() - start_time

    # -- Summary --
    summary = {
        "config": {
            "patch_size": PATCH_SIZE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "backbone": BACKBONE,
            "hierarchical_filtering": True,
            "fine_agreement_threshold": 4,
            "medium_agreement_threshold": 3,
            "coarse_agreement_threshold": 2,
        },
        "training": {
            "model_path": model_path,
            "test_metrics": train_metrics,
        },
        "rep2_ground_truth": gt_results,
        "timing": {
            "pipeline_minutes": annotation_time / 60,
            "training_minutes": training_time / 60,
            "total_minutes": total_time / 60,
        },
    }

    results_path = OUTPUT_ROOT / "experiment_summary.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print final summary
    print("\n" + "=" * 80)
    print("  HIERARCHICAL FILTERING EXPERIMENT RESULTS")
    print("=" * 80)
    print(f"  Pipeline: Rep1 (167K cells) -> hierarchical filtering -> curriculum training")
    n_cells = gt_results.get("coarse", {}).get("n_cells", "?")
    print(f"  Evaluation: Rep2 vs Janesick ground truth ({n_cells} cells)")
    print("-" * 80)

    if "coarse" in gt_results:
        c = gt_results["coarse"]
        print(f"  Coarse Accuracy:    {c['accuracy']:.4f}")
        print(f"  Coarse Macro F1:    {c['macro_f1']:.4f}")
        print(f"  Coarse Weighted F1: {c['weighted_f1']:.4f}")

        if "classification_report" in c:
            cr = c["classification_report"]
            print("-" * 80)
            print(f"  {'Category':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
            print("-" * 80)
            for cat in ["Epithelial", "Immune", "Stromal"]:
                if cat in cr:
                    r = cr[cat]
                    print(
                        f"  {cat:<20} {r['precision']:>10.3f} {r['recall']:>10.3f} "
                        f"{r['f1-score']:>10.3f} {r['support']:>10.0f}"
                    )

    print("-" * 80)
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"  Results: {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
