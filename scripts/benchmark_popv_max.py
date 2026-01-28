#!/usr/bin/env python3
"""Expanded PopV Benchmark with ALL Human CellTypist Models and ALL SingleR References.

This script tests the maximum ensemble configuration:
- 47 human CellTypist models
- 5 SingleR references (hpca, blueprint, dice, monaco, novershtern)
- Total: 52 voters in majority voting

Configurations tested:
- popv_standard: 5 CellTypist + 2 SingleR (7 voters) - baseline
- popv_extended: 10 CellTypist + 5 SingleR (15 voters) - extended
- popv_max: 47 CellTypist + 5 SingleR (52 voters) - maximum

Usage:
    uv run python scripts/benchmark_popv_max.py --datasets rep1 rep2 --granularity coarse medium fine
"""

from __future__ import annotations

import datetime
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from dapidl.pipeline.components.annotators.mapping import (
    COARSE_CLASS_NAMES,
    FINEGRAINED_CLASS_NAMES,
    GROUND_TRUTH_MAPPING,
    map_to_broad_category,
)

try:
    from clearml import Task
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# All human CellTypist models (47 total)
ALL_HUMAN_CELLTYPIST_MODELS = [
    "Adult_COVID19_PBMC.pkl",
    "Adult_Human_MTG.pkl",
    "Adult_Human_PancreaticIslet.pkl",
    "Adult_Human_PrefrontalCortex.pkl",
    "Adult_Human_Skin.pkl",
    "Adult_Human_Vascular.pkl",
    "Adult_cHSPCs_Illumina.pkl",
    "Adult_cHSPCs_Ultima.pkl",
    "Autopsy_COVID19_Lung.pkl",
    "COVID19_HumanChallenge_Blood.pkl",
    "COVID19_Immune_Landscape.pkl",
    "Cells_Adult_Breast.pkl",
    "Cells_Fetal_Lung.pkl",
    "Cells_Human_Tonsil.pkl",
    "Cells_Intestinal_Tract.pkl",
    "Cells_Lung_Airway.pkl",
    "Developing_Human_Brain.pkl",
    "Developing_Human_Gonads.pkl",
    "Developing_Human_Hippocampus.pkl",
    "Developing_Human_Organs.pkl",
    "Developing_Human_Thymus.pkl",
    "Fetal_Human_AdrenalGlands.pkl",
    "Fetal_Human_Pancreas.pkl",
    "Fetal_Human_Pituitary.pkl",
    "Fetal_Human_Retina.pkl",
    "Fetal_Human_Skin.pkl",
    "Healthy_Adult_Heart.pkl",
    "Healthy_COVID19_PBMC.pkl",
    "Healthy_Human_Liver.pkl",
    "Human_AdultAged_Hippocampus.pkl",
    "Human_Colorectal_Cancer.pkl",
    "Human_Developmental_Retina.pkl",
    "Human_Embryonic_YolkSac.pkl",
    "Human_Endometrium_Atlas.pkl",
    "Human_IPF_Lung.pkl",
    "Human_Longitudinal_Hippocampus.pkl",
    "Human_Lung_Atlas.pkl",
    "Human_PF_Lung.pkl",
    "Human_Placenta_Decidua.pkl",
    "Immune_All_High.pkl",
    "Immune_All_Low.pkl",
    "Lethal_COVID19_Lung.pkl",
    "Nuclei_Human_InnerEar.pkl",
    "Nuclei_Lung_Airway.pkl",
    "PaediatricAdult_COVID19_Airway.pkl",
    "PaediatricAdult_COVID19_PBMC.pkl",
    "Pan_Fetal_Human.pkl",
]

# All SingleR references (5 total)
ALL_SINGLER_REFERENCES = ["hpca", "blueprint", "dice", "monaco", "novershtern"]

# Standard PopV configuration (baseline)
POPV_STANDARD_CT = [
    "Cells_Adult_Breast.pkl",
    "Immune_All_High.pkl",
    "Immune_All_Low.pkl",
    "Human_Lung_Atlas.pkl",
    "Healthy_Human_Liver.pkl",
]
POPV_STANDARD_SR = ["hpca", "blueprint"]

# Extended PopV configuration (more diverse)
POPV_EXTENDED_CT = [
    "Cells_Adult_Breast.pkl",
    "Immune_All_High.pkl",
    "Immune_All_Low.pkl",
    "Human_Lung_Atlas.pkl",
    "Healthy_Human_Liver.pkl",
    "Adult_Human_Skin.pkl",
    "Adult_Human_Vascular.pkl",
    "Cells_Intestinal_Tract.pkl",
    "Human_Colorectal_Cancer.pkl",
    "Healthy_COVID19_PBMC.pkl",
]
POPV_EXTENDED_SR = ALL_SINGLER_REFERENCES


class Granularity(str, Enum):
    COARSE = "coarse"
    MEDIUM = "medium"
    FINE = "fine"


# Medium granularity mappings
MEDIUM_CLASS_NAMES = [
    "Epithelial_Luminal", "Epithelial_Basal", "Epithelial_Tumor",
    "T_Cell", "B_Cell", "Myeloid", "NK_Cell",
    "Stromal_Fibroblast", "Stromal_Pericyte", "Endothelial",
]

GT_TO_MEDIUM = {
    "DCIS_1": "Epithelial_Tumor", "DCIS_2": "Epithelial_Tumor",
    "Invasive_Tumor": "Epithelial_Tumor", "Prolif_Invasive_Tumor": "Epithelial_Tumor",
    "Myoepi_KRT15+": "Epithelial_Basal", "Myoepi_ACTA2+": "Epithelial_Basal",
    "CD4+_T_Cells": "T_Cell", "CD8+_T_Cells": "T_Cell",
    "B_Cells": "B_Cell",
    "Macrophages_1": "Myeloid", "Macrophages_2": "Myeloid",
    "Mast_Cells": "Myeloid", "LAMP3+_DCs": "Myeloid", "IRF7+_DCs": "Myeloid",
    "Perivascular-Like": "Stromal_Pericyte",
    "Stromal": "Stromal_Fibroblast",
    "Endothelial": "Endothelial",
    "T_Cell_&_Tumor_Hybrid": "Hybrid",
    "Stromal_&_T_Cell_Hybrid": "Hybrid",
}


@dataclass
class BenchmarkResult:
    granularity: str
    n_evaluated: int
    n_classes: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    precision: float
    recall: float
    kappa: float
    n_voters: int = 0
    per_class_f1: dict[str, float] = field(default_factory=dict)
    per_class_support: dict[str, int] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    runtime_seconds: float | None = None
    models_used: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "confusion_matrix"}


# Dataset configuration
DATASETS = {
    "rep1": {
        "xenium_path": Path.home() / "datasets/raw/xenium/breast_tumor_rep1/outs",
        "gt_path": Path.home() / "datasets/raw/xenium/breast_tumor_rep1/celltypes_ground_truth_rep1_supervised.xlsx",
        "name": "Xenium Breast Rep1",
    },
    "rep2": {
        "xenium_path": Path.home() / "datasets/raw/xenium/breast_tumor_rep2/outs",
        "gt_path": Path.home() / "datasets/raw/xenium/breast_tumor_rep2/celltypes_ground_truth_rep2_supervised.xlsx",
        "name": "Xenium Breast Rep2",
    },
}


def map_prediction(label: str, granularity: Granularity) -> str:
    """Map prediction to target granularity."""
    if granularity == Granularity.FINE:
        return label
    if granularity == Granularity.COARSE:
        return map_to_broad_category(label)
    # MEDIUM
    broad = map_to_broad_category(label)
    defaults = {"Epithelial": "Epithelial_Luminal", "Immune": "Myeloid", "Stromal": "Stromal_Fibroblast"}
    return defaults.get(broad, "Unknown")


def map_gt(label: str, granularity: Granularity) -> str:
    """Map ground truth to target granularity."""
    if granularity == Granularity.FINE:
        return label
    if granularity == Granularity.COARSE:
        mapped = GROUND_TRUTH_MAPPING.get(label, "Unknown")
        return "Exclude" if mapped in ("Hybrid", "Unlabeled") else mapped
    # MEDIUM
    mapped = GT_TO_MEDIUM.get(label, "Unknown")
    return "Exclude" if mapped == "Hybrid" else mapped


def load_xenium_adata(xenium_path: Path, sample_size: int | None = None, seed: int = 42) -> ad.AnnData:
    """Load Xenium expression data."""
    import h5py
    from scipy.sparse import csc_matrix

    h5_path = xenium_path / "cell_feature_matrix.h5"
    logger.info(f"Loading from {h5_path}")

    with h5py.File(h5_path, "r") as f:
        data = f["matrix/data"][:]
        indices = f["matrix/indices"][:]
        indptr = f["matrix/indptr"][:]
        shape = f["matrix/shape"][:]
        barcodes = [b.decode() for b in f["matrix/barcodes"][:]]
        genes = [g.decode() for g in f["matrix/features/name"][:]]

    X = csc_matrix((data, indices, indptr), shape=shape).T
    adata = ad.AnnData(X=X)
    adata.obs_names = barcodes
    adata.var_names = genes

    if sample_size and sample_size < adata.n_obs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(adata.n_obs, sample_size, replace=False)
        adata = adata[idx].copy()

    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def load_ground_truth(gt_path: Path, granularity: Granularity) -> pl.DataFrame:
    """Load ground truth."""
    pd_df = pd.read_excel(gt_path)
    df = pl.DataFrame({
        "cell_id": [str(b) for b in pd_df["Barcode"]],
        "gt_fine": pd_df["Cluster"].astype(str).tolist(),
    })

    if granularity == Granularity.COARSE:
        mapping = {k: ("Exclude" if v in ("Hybrid", "Unlabeled") else v) for k, v in GROUND_TRUTH_MAPPING.items()}
    elif granularity == Granularity.MEDIUM:
        mapping = {k: ("Exclude" if v == "Hybrid" else v) for k, v in GT_TO_MEDIUM.items()}
    else:
        mapping = {k: k for k in FINEGRAINED_CLASS_NAMES}

    df = df.with_columns(pl.col("gt_fine").replace(mapping, default="Unknown").alias("gt_mapped"))
    return df


def run_celltypist_models(
    adata_norm: ad.AnnData,
    models_list: list[str],
    granularity: Granularity,
) -> list[dict]:
    """Run multiple CellTypist models and return predictions."""
    import celltypist
    from celltypist import models as ct_models

    all_preds = []
    for model_name in models_list:
        try:
            ct_models.download_models(model=[model_name])
            model = ct_models.Model.load(model=model_name)
            pred = celltypist.annotate(adata_norm, model=model, majority_voting=False)

            labels = [map_prediction(l, granularity) for l in pred.predicted_labels.predicted_labels.tolist()]
            confs = pred.probability_matrix.max(axis=1).tolist()

            all_preds.append({
                "source": f"celltypist:{model_name}",
                "labels": labels,
                "confidences": confs,
            })
            logger.info(f"  CellTypist {model_name}: done")
        except Exception as e:
            logger.warning(f"  CellTypist {model_name} failed: {e}")

    return all_preds


def run_singler_references(
    adata: ad.AnnData,
    references: list[str],
    granularity: Granularity,
) -> list[dict]:
    """Run multiple SingleR references and return predictions."""
    from dapidl.pipeline.base import AnnotationConfig
    from dapidl.pipeline.components.annotators.singler import SingleRAnnotator, is_singler_available

    if not is_singler_available():
        logger.warning("SingleR not available")
        return []

    all_preds = []
    for ref in references:
        try:
            config = AnnotationConfig()
            config.singler_reference = ref
            annotator = SingleRAnnotator(config)
            result = annotator.annotate(adata=adata)

            df = result.annotations_df
            labels = [map_prediction(l, granularity) for l in df["predicted_type"].to_list()]
            confs = df["confidence"].to_list()

            all_preds.append({
                "source": f"singler:{ref}",
                "labels": labels,
                "confidences": confs,
            })
            logger.info(f"  SingleR {ref}: done")
        except Exception as e:
            logger.warning(f"  SingleR {ref} failed: {e}")

    return all_preds


def combine_predictions(cell_ids: list[str], all_preds: list[dict]) -> pl.DataFrame:
    """Combine predictions via majority voting."""
    if not all_preds:
        raise RuntimeError("No predictions")

    n_cells = len(all_preds[0]["labels"])
    final_labels = []
    final_confs = []

    for i in range(n_cells):
        votes = [p["labels"][i] for p in all_preds]
        vote_counts = Counter(votes)
        winner = vote_counts.most_common(1)[0][0]

        winner_confs = [p["confidences"][i] for p in all_preds if p["labels"][i] == winner]
        conf = np.mean(winner_confs) if winner_confs else 0.0

        final_labels.append(winner)
        final_confs.append(conf)

    return pl.DataFrame({
        "cell_id": cell_ids,
        "pred_mapped": final_labels,
        "confidence": final_confs,
    })


def evaluate(predictions: pl.DataFrame, ground_truth: pl.DataFrame, granularity: Granularity) -> BenchmarkResult:
    """Evaluate predictions."""
    merged = predictions.join(ground_truth.select(["cell_id", "gt_mapped"]), on="cell_id", how="inner")
    merged = merged.filter(
        (pl.col("gt_mapped") != "Exclude") & (pl.col("gt_mapped") != "Unknown") & (pl.col("pred_mapped") != "Unknown")
    )

    y_true = merged["gt_mapped"].to_list()
    y_pred = merged["pred_mapped"].to_list()

    if not y_true:
        return BenchmarkResult(granularity=granularity.value, n_evaluated=0, n_classes=0,
                               accuracy=0, macro_f1=0, weighted_f1=0, precision=0, recall=0, kappa=0,
                               error="No valid predictions")

    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return BenchmarkResult(
        granularity=granularity.value,
        n_evaluated=len(y_true),
        n_classes=len(labels),
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        weighted_f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
        kappa=cohen_kappa_score(y_true, y_pred),
        per_class_f1={l: report[l]["f1-score"] for l in labels if l in report},
        per_class_support={l: report[l]["support"] for l in labels if l in report},
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        labels=labels,
    )


def create_confusion_plot(result: BenchmarkResult, path: Path, title: str):
    """Create confusion matrix plot."""
    cm = np.array(result.confusion_matrix)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=result.labels, yticklabels=result.labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"{title}\nAcc: {result.accuracy:.3f}, F1: {result.macro_f1:.3f}, Voters: {result.n_voters}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def run_popv_benchmark(
    adata: ad.AnnData,
    adata_norm: ad.AnnData,
    cell_ids: list[str],
    ground_truth: pl.DataFrame,
    granularity: Granularity,
    ct_models: list[str],
    sr_refs: list[str],
    config_name: str,
    output_dir: Path,
) -> BenchmarkResult:
    """Run PopV benchmark with specified configuration."""
    logger.info(f"\n=== {config_name}: {len(ct_models)} CellTypist + {len(sr_refs)} SingleR ===")

    start = time.time()
    all_preds = []

    # Run CellTypist
    ct_preds = run_celltypist_models(adata_norm, ct_models, granularity)
    all_preds.extend(ct_preds)

    # Run SingleR
    sr_preds = run_singler_references(adata, sr_refs, granularity)
    all_preds.extend(sr_preds)

    runtime = time.time() - start

    if not all_preds:
        return BenchmarkResult(
            granularity=granularity.value, n_evaluated=0, n_classes=0,
            accuracy=0, macro_f1=0, weighted_f1=0, precision=0, recall=0, kappa=0,
            error="No predictions succeeded"
        )

    # Combine via majority vote
    predictions = combine_predictions(cell_ids, all_preds)
    result = evaluate(predictions, ground_truth, granularity)
    result.runtime_seconds = runtime
    result.n_voters = len(all_preds)
    result.models_used = [p["source"] for p in all_preds]

    logger.info(f"  Accuracy: {result.accuracy:.3f}, F1: {result.macro_f1:.3f}, Voters: {result.n_voters}")

    # Save confusion matrix
    cm_path = output_dir / f"{config_name}_confusion.png"
    create_confusion_plot(result, cm_path, f"{config_name} - {granularity.value}")

    return result


def run_benchmark(
    datasets: list[str],
    granularities: list[Granularity],
    output_dir: Path,
    sample_size: int | None = None,
    use_clearml: bool = True,
):
    """Run full expanded PopV benchmark."""
    task = None
    if use_clearml and CLEARML_AVAILABLE:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        task = Task.init(
            project_name="DAPIDL/Annotation-Benchmark",
            task_name=f"PopV_Max_Benchmark_{timestamp}",
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset_key in datasets:
        config = DATASETS[dataset_key]
        logger.info(f"\n{'='*70}")
        logger.info(f"DATASET: {config['name']}")
        logger.info(f"{'='*70}")

        adata = load_xenium_adata(config["xenium_path"], sample_size=sample_size)
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)

        cell_ids = [str(c) for c in adata.obs_names]

        for granularity in granularities:
            logger.info(f"\n--- Granularity: {granularity.value.upper()} ---")

            ground_truth = load_ground_truth(config["gt_path"], granularity)
            gt_cells = set(ground_truth["cell_id"].to_list())
            mask = [c in gt_cells for c in cell_ids]
            adata_f = adata[mask].copy()
            adata_norm_f = adata_norm[mask].copy()
            cell_ids_f = [c for c, m in zip(cell_ids, mask) if m]
            gt_f = ground_truth.filter(pl.col("cell_id").is_in(cell_ids_f))

            gran_dir = output_dir / dataset_key / granularity.value
            gran_dir.mkdir(parents=True, exist_ok=True)

            results = {}

            # Configuration 1: popv_standard (7 voters)
            results["popv_standard"] = run_popv_benchmark(
                adata_f, adata_norm_f, cell_ids_f, gt_f, granularity,
                POPV_STANDARD_CT, POPV_STANDARD_SR, "popv_standard", gran_dir
            )

            # Configuration 2: popv_extended (15 voters)
            results["popv_extended"] = run_popv_benchmark(
                adata_f, adata_norm_f, cell_ids_f, gt_f, granularity,
                POPV_EXTENDED_CT, POPV_EXTENDED_SR, "popv_extended", gran_dir
            )

            # Configuration 3: popv_max (52 voters)
            results["popv_max"] = run_popv_benchmark(
                adata_f, adata_norm_f, cell_ids_f, gt_f, granularity,
                ALL_HUMAN_CELLTYPIST_MODELS, ALL_SINGLER_REFERENCES, "popv_max", gran_dir
            )

            result_key = f"{dataset_key}_{granularity.value}"
            all_results[result_key] = results

            if task:
                for name, res in results.items():
                    task.upload_artifact(f"{result_key}/{name}", gran_dir / f"{name}_confusion.png")

    # Summary
    summary_path = output_dir / "popv_max_summary.json"
    with open(summary_path, "w") as f:
        json.dump({k: {m: r.to_dict() for m, r in v.items()} for k, v in all_results.items()}, f, indent=2)

    print_summary(all_results)

    if task:
        task.upload_artifact("summary", summary_path)
        task.close()

    return all_results


def print_summary(results: dict):
    """Print summary table."""
    print("\n" + "=" * 100)
    print("EXPANDED PopV BENCHMARK SUMMARY")
    print("=" * 100)

    for result_key, methods in sorted(results.items()):
        print(f"\n{result_key}:")
        print("-" * 90)
        print(f"{'Config':<20} {'Voters':>8} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Time':>10}")
        print("-" * 90)

        for config, res in sorted(methods.items(), key=lambda x: -x[1].macro_f1):
            if res.error:
                print(f"{config:<20} ERROR: {res.error}")
                continue
            runtime = f"{res.runtime_seconds:.1f}s" if res.runtime_seconds else "N/A"
            print(
                f"{config:<20} {res.n_voters:>8} {res.accuracy:>8.3f} {res.macro_f1:>8.3f} "
                f"{res.precision:>8.3f} {res.recall:>8.3f} {runtime:>10}"
            )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Expanded PopV benchmark")
    parser.add_argument("--datasets", "-d", nargs="+", default=["rep1", "rep2"])
    parser.add_argument("--granularity", "-g", nargs="+", choices=["coarse", "medium", "fine"],
                        default=["coarse", "medium", "fine"])
    parser.add_argument("--sample-size", "-s", type=int, default=None)
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("benchmark_popv_max"))
    parser.add_argument("--no-clearml", action="store_true")

    args = parser.parse_args()
    granularities = [Granularity(g) for g in args.granularity]

    run_benchmark(
        datasets=args.datasets,
        granularities=granularities,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        use_clearml=not args.no_clearml,
    )


if __name__ == "__main__":
    main()
