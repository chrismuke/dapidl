#!/usr/bin/env python3
"""Parallelized PopV Benchmark with ALL Human CellTypist Models and ALL SingleR References.

This script tests the maximum ensemble configuration with parallelization:
- CellTypist models run in parallel using ProcessPoolExecutor
- SingleR references run in parallel using separate processes
- Rep1 and Rep2 can run in parallel

Usage:
    uv run python scripts/benchmark_popv_parallel.py --datasets rep1 rep2 --granularity coarse --workers 8
"""

from __future__ import annotations

import datetime
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import anndata as ad
import click
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
    "Human_IPF_Lung.pkl",
    "Human_Lung_Atlas.pkl",
    "Human_PF_Lung.pkl",
    "Immune_All_AddPIP.pkl",
    "Immune_All_High.pkl",
    "Immune_All_Low.pkl",
    "Normal_Human_Liver.pkl",
    "PaediatricAdult_COVID19_Airway.pkl",
    "PaediatricAdult_COVID19_PBMC.pkl",
    "Pan_Fetal_Human.pkl",
]

# All SingleR references (5 total)
ALL_SINGLER_REFERENCES = ["hpca", "blueprint", "dice", "monaco", "novershtern"]

# Curated breast-relevant subsets
STANDARD_CELLTYPIST_MODELS = [
    "Cells_Adult_Breast.pkl",
    "Immune_All_High.pkl",
    "Immune_All_Low.pkl",
    "Human_Lung_Atlas.pkl",
    "Healthy_Human_Liver.pkl",
]
STANDARD_SINGLER_REFERENCES = ["hpca", "blueprint"]

EXTENDED_CELLTYPIST_MODELS = STANDARD_CELLTYPIST_MODELS + [
    "Adult_Human_Skin.pkl",
    "Adult_Human_Vascular.pkl",
    "Cells_Intestinal_Tract.pkl",
    "Human_Colorectal_Cancer.pkl",
    "Healthy_COVID19_PBMC.pkl",
]


class Granularity(str, Enum):
    COARSE = "coarse"
    MEDIUM = "medium"
    FINE = "fine"


MEDIUM_CLASS_NAMES = [
    "Epithelial_Luminal", "Epithelial_Basal", "Epithelial_Tumor",
    "T_Cell", "B_Cell", "Myeloid", "NK_Cell",
    "Stromal_Fibroblast", "Stromal_Pericyte", "Endothelial",
]


@dataclass
class BenchmarkResult:
    granularity: str
    n_evaluated: int
    n_classes: int
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    kappa: float
    per_class_f1: dict = field(default_factory=dict)


# ============================================================================
# PREDICTION MAPPING
# ============================================================================

def map_prediction(pred_label: str, granularity: Granularity) -> str:
    """Map predicted label to target granularity."""
    broad = map_to_broad_category(pred_label)
    if granularity == Granularity.COARSE:
        return broad if broad in COARSE_CLASS_NAMES else "Unknown"
    elif granularity == Granularity.MEDIUM:
        # For medium, use ground truth mapping if available, otherwise broad category
        if pred_label in GROUND_TRUTH_MAPPING:
            return _map_to_medium(pred_label)
        return broad if broad in COARSE_CLASS_NAMES else "Unknown"
    else:
        return pred_label


def _map_to_medium(gt_label: str) -> str:
    """Map ground truth label to medium granularity."""
    # Map Xenium ground truth labels to medium-level categories
    medium_mapping = {
        "B_Cells": "B_Cell",
        "CD4+_T_Cells": "T_Cell",
        "CD8+_T_Cells": "T_Cell",
        "DCIS_1": "Epithelial_Tumor",
        "DCIS_2": "Epithelial_Tumor",
        "Endothelial": "Endothelial",
        "IRF7+_DCs": "Myeloid",
        "Invasive_Tumor": "Epithelial_Tumor",
        "LAMP3+_DCs": "Myeloid",
        "Macrophages_1": "Myeloid",
        "Macrophages_2": "Myeloid",
        "Mast_Cells": "Myeloid",
        "Myoepi_ACTA2+": "Epithelial_Basal",
        "Myoepi_KRT15+": "Epithelial_Basal",
        "Perivascular-Like": "Stromal_Pericyte",
        "Prolif_Invasive_Tumor": "Epithelial_Tumor",
        "Stromal": "Stromal_Fibroblast",
        "Stromal_&_T_Cell_Hybrid": "Exclude",
        "T_Cell_&_Tumor_Hybrid": "Exclude",
        "Unlabeled": "Exclude",
    }
    return medium_mapping.get(gt_label, "Unknown")


def map_ground_truth(gt_label: str, granularity: Granularity) -> str:
    """Map ground truth label to target granularity."""
    if granularity == Granularity.COARSE:
        broad = GROUND_TRUTH_MAPPING.get(gt_label, "Unknown")
        return broad if broad in COARSE_CLASS_NAMES else "Exclude"
    elif granularity == Granularity.MEDIUM:
        return _map_to_medium(gt_label)
    else:
        return gt_label


# ============================================================================
# PARALLEL CELLTYPIST RUNNER
# ============================================================================

def _run_single_celltypist(
    model_name: str,
    adata_path: str,
    granularity_str: str,
) -> dict | None:
    """Run a single CellTypist model (worker function for multiprocessing)."""
    import warnings
    warnings.filterwarnings("ignore")

    import anndata as ad
    import celltypist
    from celltypist import models as ct_models

    granularity = Granularity(granularity_str)

    try:
        adata = ad.read_h5ad(adata_path)
        ct_models.download_models(model=[model_name])
        model = ct_models.Model.load(model=model_name)
        pred = celltypist.annotate(adata, model=model, majority_voting=False)

        labels = [map_prediction(l, granularity) for l in pred.predicted_labels.predicted_labels.tolist()]
        confs = pred.probability_matrix.max(axis=1).tolist()

        return {
            "source": f"celltypist:{model_name}",
            "labels": labels,
            "confidences": confs,
        }
    except Exception as e:
        logger.warning(f"CellTypist {model_name} failed: {e}")
        return None


def run_celltypist_parallel(
    adata_norm: ad.AnnData,
    models_list: list[str],
    granularity: Granularity,
    n_workers: int = 6,
    temp_dir: Path | None = None,
) -> list[dict]:
    """Run multiple CellTypist models in parallel."""
    if temp_dir is None:
        temp_dir = Path("/tmp/celltypist_parallel")
    temp_dir.mkdir(exist_ok=True)

    # Save adata to temp file for workers
    temp_adata = temp_dir / f"adata_{granularity.value}.h5ad"
    adata_norm.write_h5ad(temp_adata)

    logger.info(f"Running {len(models_list)} CellTypist models with {n_workers} workers...")
    start = time.time()

    all_preds = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _run_single_celltypist,
                model_name,
                str(temp_adata),
                granularity.value,
            ): model_name
            for model_name in models_list
        }

        for future in as_completed(futures):
            model_name = futures[future]
            try:
                result = future.result()
                if result:
                    all_preds.append(result)
                    logger.info(f"  CellTypist {model_name}: done")
            except Exception as e:
                logger.warning(f"  CellTypist {model_name} failed: {e}")

    elapsed = time.time() - start
    logger.info(f"CellTypist parallel complete: {len(all_preds)}/{len(models_list)} models in {elapsed:.1f}s")

    # Cleanup
    temp_adata.unlink(missing_ok=True)

    return all_preds


def run_celltypist_sequential(
    adata_norm: ad.AnnData,
    models_list: list[str],
    granularity: Granularity,
) -> list[dict]:
    """Run CellTypist models sequentially (fallback)."""
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


# ============================================================================
# SINGLER RUNNER (Sequential - R has issues with multiprocessing)
# ============================================================================

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


# ============================================================================
# CORE BENCHMARK LOGIC
# ============================================================================

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

    labels = sorted(set(y_true) | set(y_pred))

    return BenchmarkResult(
        granularity=granularity.value,
        n_evaluated=len(y_true),
        n_classes=len(labels),
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        macro_precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
        macro_recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
        kappa=cohen_kappa_score(y_true, y_pred),
        per_class_f1={
            label: f1_score(y_true, y_pred, labels=[label], average="micro", zero_division=0)
            for label in labels
        },
    )


def load_xenium_adata(xenium_path: Path, sample_size: int | None = None, seed: int = 42) -> ad.AnnData:
    """Load Xenium expression data."""
    h5_path = xenium_path / "cell_feature_matrix.h5"
    if not h5_path.exists():
        h5_path = xenium_path / "outs" / "cell_feature_matrix.h5"

    logger.info(f"Loading from {h5_path}")
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    if sample_size and sample_size < adata.n_obs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(adata.n_obs, sample_size, replace=False)
        adata = adata[idx].copy()

    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def load_ground_truth(gt_path: Path, cell_ids: list[str], granularity: Granularity) -> pl.DataFrame:
    """Load and map ground truth labels."""
    df = pl.read_parquet(gt_path)

    cell_id_col = "cell_id" if "cell_id" in df.columns else df.columns[0]
    gt_col = next((c for c in df.columns if c.lower() in ["cell_type", "celltype", "label"]), df.columns[1])

    df = df.select([
        pl.col(cell_id_col).cast(pl.Utf8).alias("cell_id"),
        pl.col(gt_col).alias("gt_fine"),
    ])

    mapping_dict = {k: map_ground_truth(k, granularity) for k in df["gt_fine"].unique().to_list()}
    df = df.with_columns(
        pl.col("gt_fine").replace(mapping_dict, default="Unknown").alias("gt_mapped")
    )

    cell_ids_set = set(cell_ids)
    df = df.filter(pl.col("cell_id").is_in(cell_ids_set))

    return df


def get_normalized_adata(adata: ad.AnnData) -> ad.AnnData:
    """Create normalized copy for CellTypist."""
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    return adata_norm


def run_popv_benchmark(
    adata: ad.AnnData,
    adata_norm: ad.AnnData,
    ground_truth: pl.DataFrame,
    granularity: Granularity,
    config_name: str,
    ct_models: list[str],
    sr_refs: list[str],
    n_workers: int = 6,
    use_parallel: bool = True,
) -> tuple[BenchmarkResult, int]:
    """Run a single PopV configuration benchmark."""
    logger.info(f"\n=== {config_name}: {len(ct_models)} CellTypist + {len(sr_refs)} SingleR ===")

    cell_ids = [str(c) for c in adata.obs_names]

    # Run CellTypist (parallel or sequential)
    if use_parallel and len(ct_models) > 3:
        ct_preds = run_celltypist_parallel(adata_norm, ct_models, granularity, n_workers)
    else:
        ct_preds = run_celltypist_sequential(adata_norm, ct_models, granularity)

    # Run SingleR (always sequential due to R limitations)
    sr_preds = run_singler_references(adata, sr_refs, granularity)

    # Combine predictions
    all_preds = ct_preds + sr_preds
    n_voters = len(all_preds)

    if not all_preds:
        raise RuntimeError("No successful predictions")

    predictions = combine_predictions(cell_ids, all_preds)
    result = evaluate(predictions, ground_truth, granularity)

    logger.info(f"  Accuracy: {result.accuracy:.3f}, F1: {result.macro_f1:.3f}, Voters: {n_voters}")

    return result, n_voters


# ============================================================================
# DATASET PATHS
# ============================================================================

DATASET_PATHS = {
    "rep1": {
        "xenium": Path.home() / "datasets/raw/xenium/breast_tumor_rep1",
        "gt": Path.home() / "datasets/raw/xenium/breast_tumor_rep1/outs/analysis/annotation/supervised/cell_types.parquet",
    },
    "rep2": {
        "xenium": Path.home() / "datasets/raw/xenium/breast_tumor_rep2",
        "gt": Path.home() / "datasets/raw/xenium/breast_tumor_rep2/outs/analysis/annotation/supervised/cell_types.parquet",
    },
}


@click.command()
@click.option("--datasets", "-d", multiple=True, default=["rep1"], help="Datasets to benchmark")
@click.option("--granularity", "-g", multiple=True, default=["coarse"], help="Granularity levels")
@click.option("--workers", "-w", default=6, help="Number of parallel workers for CellTypist")
@click.option("--no-parallel", is_flag=True, help="Disable parallel processing")
@click.option("--output", "-o", type=Path, default=Path("benchmark_popv_parallel"))
def main(datasets: tuple[str], granularity: tuple[str], workers: int, no_parallel: bool, output: Path):
    """Run parallelized PopV benchmark."""
    output.mkdir(exist_ok=True)

    if CLEARML_AVAILABLE:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        task = Task.init(
            project_name="dapidl/benchmarks",
            task_name=f"PopV_Parallel_Benchmark_{timestamp}",
            auto_connect_frameworks=False,
        )

    use_parallel = not no_parallel
    logger.info(f"Parallel mode: {use_parallel}, Workers: {workers}")

    all_results = {}

    for ds_name in datasets:
        if ds_name not in DATASET_PATHS:
            logger.warning(f"Unknown dataset: {ds_name}")
            continue

        paths = DATASET_PATHS[ds_name]
        logger.info(f"\n{'='*70}")
        logger.info(f"DATASET: {ds_name}")
        logger.info(f"{'='*70}")

        # Load data
        adata = load_xenium_adata(paths["xenium"])
        adata_norm = get_normalized_adata(adata)
        cell_ids = [str(c) for c in adata.obs_names]

        for gran_str in granularity:
            gran = Granularity(gran_str)
            logger.info(f"\n--- Granularity: {gran.value.upper()} ---")

            gt_df = load_ground_truth(paths["gt"], cell_ids, gran)

            ds_results = {}

            # Configuration 1: Standard (7 voters)
            result, n_voters = run_popv_benchmark(
                adata, adata_norm, gt_df, gran,
                "popv_standard",
                STANDARD_CELLTYPIST_MODELS,
                STANDARD_SINGLER_REFERENCES,
                workers, use_parallel,
            )
            ds_results["popv_standard"] = {"result": result, "voters": n_voters}

            # Configuration 2: Extended (15 voters)
            result, n_voters = run_popv_benchmark(
                adata, adata_norm, gt_df, gran,
                "popv_extended",
                EXTENDED_CELLTYPIST_MODELS,
                ALL_SINGLER_REFERENCES,
                workers, use_parallel,
            )
            ds_results["popv_extended"] = {"result": result, "voters": n_voters}

            # Configuration 3: Max (52 voters)
            result, n_voters = run_popv_benchmark(
                adata, adata_norm, gt_df, gran,
                "popv_max",
                ALL_HUMAN_CELLTYPIST_MODELS,
                ALL_SINGLER_REFERENCES,
                workers, use_parallel,
            )
            ds_results["popv_max"] = {"result": result, "voters": n_voters}

            all_results[f"{ds_name}_{gran.value}"] = ds_results

    # Save results summary
    summary = {}
    for key, configs in all_results.items():
        summary[key] = {
            name: {
                "voters": data["voters"],
                "accuracy": data["result"].accuracy,
                "f1": data["result"].macro_f1,
            }
            for name, data in configs.items()
        }

    with open(output / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)
    for key, configs in all_results.items():
        logger.info(f"\n{key}:")
        for name, data in configs.items():
            r = data["result"]
            logger.info(f"  {name}: Acc={r.accuracy:.3f}, F1={r.macro_f1:.3f}, Voters={data['voters']}")


if __name__ == "__main__":
    main()
