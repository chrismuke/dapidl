"""Test all CPU-based annotation methods on lung-2fov dataset.

Runs CellTypist, SingleR, scType, and SCINA individually,
then reports prediction distributions, confidence stats,
and checks ensemble compatibility.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import polars as pl
import scanpy as sc
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATASET_PATH = Path("/mnt/work/datasets/raw/xenium/xenium-lung-2fov")
H5_PATH = DATASET_PATH / "cell_feature_matrix.h5"
CELLS_PATH = DATASET_PATH / "cells.parquet"

RESULTS: dict[str, dict] = {}


def load_adata() -> ad.AnnData:
    """Load and preprocess expression data for annotation."""
    logger.info("Loading expression matrix...")
    adata = sc.read_10x_h5(str(H5_PATH))
    adata.var_names_make_unique()

    # Basic preprocessing (required by most annotators)
    sc.pp.filter_cells(adata, min_genes=10)

    # Add cell_id to obs (required by annotators, normally done by pipeline steps)
    adata.obs["cell_id"] = adata.obs_names

    # Keep raw counts for annotators that normalize internally
    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    logger.info(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    return adata


def summarize_result(name: str, predictions: list | np.ndarray,
                     confidences: list | np.ndarray | None = None,
                     elapsed: float = 0.0) -> dict:
    """Compute and display summary stats for an annotation method."""
    preds = pl.Series("pred", predictions)
    dist = preds.value_counts().sort("count", descending=True)

    result = {
        "method": name,
        "n_cells": len(predictions),
        "n_classes": preds.n_unique(),
        "elapsed_seconds": round(elapsed, 1),
        "distribution": {
            row[0]: row[1] for row in dist.iter_rows()
        },
    }

    if confidences is not None:
        confs = np.array(confidences, dtype=float)
        confs = confs[~np.isnan(confs)]
        result["confidence"] = {
            "mean": round(float(np.mean(confs)), 4),
            "median": round(float(np.median(confs)), 4),
            "std": round(float(np.std(confs)), 4),
            "min": round(float(np.min(confs)), 4),
            "max": round(float(np.max(confs)), 4),
            "pct_above_0.5": round(float(np.mean(confs > 0.5) * 100), 1),
            "pct_above_0.8": round(float(np.mean(confs > 0.8) * 100), 1),
        }

    logger.info(f"\n{'='*60}")
    logger.info(f"  {name}")
    logger.info(f"{'='*60}")
    logger.info(f"  Cells: {result['n_cells']}, Classes: {result['n_classes']}, Time: {result['elapsed_seconds']}s")
    for label, count in list(result["distribution"].items())[:10]:
        pct = count / result["n_cells"] * 100
        logger.info(f"  {label:30s} {count:6d} ({pct:5.1f}%)")
    if confidences is not None:
        logger.info(f"  Confidence: mean={result['confidence']['mean']:.3f}, "
                     f"median={result['confidence']['median']:.3f}, "
                     f">0.5={result['confidence']['pct_above_0.5']}%, "
                     f">0.8={result['confidence']['pct_above_0.8']}%")

    RESULTS[name] = result
    return result


def test_celltypist(adata: ad.AnnData) -> None:
    """Test CellTypist annotation."""
    logger.info("\n>>> Testing CellTypist...")
    from dapidl.pipeline.components.annotators.celltypist import CellTypistAnnotator
    from dapidl.pipeline.base import AnnotationConfig

    # Use a lung-appropriate model
    config = AnnotationConfig(
        model_names=["Human_Lung_Atlas.pkl"],
        strategy="single",
        majority_voting=True,
        fine_grained=True,
    )

    annotator = CellTypistAnnotator(config)
    t0 = time.time()
    result = annotator.annotate(config=config, adata=adata.copy())
    elapsed = time.time() - t0

    df = result.annotations_df
    preds = df["predicted_type"].to_list()
    confs = df["confidence"].to_numpy() if "confidence" in df.columns else None

    summarize_result("CellTypist (Human_Lung_Atlas)", preds, confs, elapsed)


def test_celltypist_multi(adata: ad.AnnData) -> None:
    """Test CellTypist with multiple models (consensus)."""
    logger.info("\n>>> Testing CellTypist multi-model consensus...")
    from dapidl.pipeline.components.annotators.celltypist import CellTypistAnnotator
    from dapidl.pipeline.base import AnnotationConfig

    config = AnnotationConfig(
        model_names=["Human_Lung_Atlas.pkl", "Immune_All_High.pkl", "Immune_All_Low.pkl"],
        strategy="consensus",
        majority_voting=True,
        fine_grained=True,
    )

    annotator = CellTypistAnnotator(config)
    t0 = time.time()
    result = annotator.annotate(config=config, adata=adata.copy())
    elapsed = time.time() - t0

    df = result.annotations_df
    preds = df["predicted_type"].to_list()
    confs = df["confidence"].to_numpy() if "confidence" in df.columns else None

    summarize_result("CellTypist (3-model consensus)", preds, confs, elapsed)


def test_singler(adata: ad.AnnData) -> None:
    """Test SingleR annotation."""
    logger.info("\n>>> Testing SingleR...")
    try:
        from dapidl.pipeline.components.annotators.singler import (
            SingleRAnnotator,
            is_singler_available,
        )
    except ImportError as e:
        logger.warning(f"SingleR import failed: {e}")
        RESULTS["SingleR"] = {"error": str(e)}
        return

    if not is_singler_available():
        logger.warning("SingleR not available (R packages missing)")
        RESULTS["SingleR"] = {"error": "R packages not installed"}
        return

    from dapidl.pipeline.base import AnnotationConfig

    # Test with Blueprint (best performer)
    # SingleR reads reference from config.singler_reference
    config = AnnotationConfig(fine_grained=True)
    config.singler_reference = "blueprint"

    annotator = SingleRAnnotator(config)
    t0 = time.time()
    result = annotator.annotate(
        config=config,
        adata=adata.copy(),
    )
    elapsed = time.time() - t0

    df = result.annotations_df
    preds = df["predicted_type"].to_list()
    confs = df["confidence"].to_numpy() if "confidence" in df.columns else None

    summarize_result("SingleR (Blueprint)", preds, confs, elapsed)


def test_sctype(adata: ad.AnnData) -> None:
    """Test scType annotation."""
    logger.info("\n>>> Testing scType...")
    from dapidl.pipeline.components.annotators.sctype import ScTypeAnnotator
    from dapidl.pipeline.base import AnnotationConfig

    config = AnnotationConfig(fine_grained=True)

    annotator = ScTypeAnnotator(config)
    t0 = time.time()
    result = annotator.annotate(config=config, adata=adata.copy())
    elapsed = time.time() - t0

    df = result.annotations_df
    preds = df["predicted_type"].to_list()
    confs = df["confidence"].to_numpy() if "confidence" in df.columns else None

    summarize_result("scType", preds, confs, elapsed)


def test_scina(adata: ad.AnnData) -> None:
    """Test SCINA annotation."""
    logger.info("\n>>> Testing SCINA...")
    from dapidl.pipeline.components.annotators.scina import SCINAAnnotator
    from dapidl.pipeline.base import AnnotationConfig

    config = AnnotationConfig(fine_grained=True)

    annotator = SCINAAnnotator(config)
    t0 = time.time()
    result = annotator.annotate(config=config, adata=adata.copy())
    elapsed = time.time() - t0

    df = result.annotations_df
    preds = df["predicted_type"].to_list()
    confs = df["confidence"].to_numpy() if "confidence" in df.columns else None

    summarize_result("SCINA", preds, confs, elapsed)


def print_comparison() -> None:
    """Print side-by-side comparison of all methods."""
    logger.info(f"\n{'='*70}")
    logger.info("  COMPARISON SUMMARY")
    logger.info(f"{'='*70}")

    # Table header
    header = f"{'Method':35s} {'Classes':>8s} {'Time':>8s} {'Conf μ':>8s} {'>0.5':>6s} {'>0.8':>6s}"
    logger.info(header)
    logger.info("-" * 70)

    for name, r in RESULTS.items():
        if "error" in r:
            logger.info(f"{name:35s} ERROR: {r['error']}")
            continue
        conf = r.get("confidence", {})
        logger.info(
            f"{name:35s} {r['n_classes']:>8d} {r['elapsed_seconds']:>7.1f}s "
            f"{conf.get('mean', 0):>8.3f} {conf.get('pct_above_0.5', 0):>5.1f}% "
            f"{conf.get('pct_above_0.8', 0):>5.1f}%"
        )

    # Ensemble compatibility check
    logger.info(f"\n{'='*70}")
    logger.info("  ENSEMBLE COMPATIBILITY")
    logger.info(f"{'='*70}")
    for name, r in RESULTS.items():
        if "error" in r:
            compat = "UNAVAILABLE"
        elif r.get("confidence") and r["confidence"]["mean"] > 0:
            compat = "YES — produces predictions + confidence"
        else:
            compat = "PARTIAL — predictions only, no confidence"
        logger.info(f"  {name:35s} {compat}")


if __name__ == "__main__":
    adata = load_adata()

    test_celltypist(adata)
    test_celltypist_multi(adata)
    test_singler(adata)
    test_sctype(adata)
    test_scina(adata)

    print_comparison()

    # Save results
    out_path = Path("/tmp/annotation_test_results.json")
    # Convert numpy types for JSON serialization
    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    out_path.write_text(json.dumps(RESULTS, indent=2, default=_serialize))
    logger.info(f"\nResults saved to {out_path}")
