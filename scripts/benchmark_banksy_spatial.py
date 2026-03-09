#!/usr/bin/env python3
"""
Benchmark BANKSY spatial augmentation + all CPU annotation methods against
breast cancer ground truth (rep1 + rep2).

Tests:
1. Baseline annotation methods (CellTypist, SingleR, scType, SCINA)
2. BANKSY spatial augmentation + Leiden clustering
3. Spatial neighbor smoothing of baseline predictions
4. BANKSY-augmented Leiden + label transfer from reference annotations

Usage:
    uv run python scripts/benchmark_banksy_spatial.py
    uv run python scripts/benchmark_banksy_spatial.py --sample-size 5000
    uv run python scripts/benchmark_banksy_spatial.py --datasets rep1
"""

import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from loguru import logger
from scipy.sparse import issparse
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
)
from sklearn.neighbors import NearestNeighbors

# Dataset configuration — actual paths on this machine
DATASETS = {
    "rep1": {
        "h5_path": Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1/cell_feature_matrix.h5"),
        "cells_path": Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1/cells.parquet"),
        "gt_path": Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1/celltypes_ground_truth_rep1_supervised.xlsx"),
        "name": "Breast Rep1",
    },
    "rep2": {
        "h5_path": Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep2/cell_feature_matrix.h5"),
        "cells_path": Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep2/cells.parquet"),
        "gt_path": Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep2/celltypes_ground_truth_rep2_supervised.xlsx"),
        "name": "Breast Rep2",
    },
}

# Ground truth to broad category mapping
GT_BROAD_MAP = {
    "DCIS_1": "Epithelial",
    "DCIS_2": "Epithelial",
    "Invasive_Tumor": "Epithelial",
    "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_KRT15+": "Epithelial",
    "Myoepi_ACTA2+": "Epithelial",
    "T_Cell_&_Tumor_Hybrid": "Epithelial",
    "CD4+_T_Cells": "Immune",
    "CD8+_T_Cells": "Immune",
    "B_Cells": "Immune",
    "Macrophages_1": "Immune",
    "Macrophages_2": "Immune",
    "Mast_Cells": "Immune",
    "LAMP3+_DCs": "Immune",
    "IRF7+_DCs": "Immune",
    "Perivascular-Like": "Immune",
    "Stromal": "Stromal",
    "Stromal_&_T_Cell_Hybrid": "Stromal",
    "Endothelial": "Stromal",
}


@dataclass
class BenchmarkResult:
    """Result from a single benchmark method."""
    method: str
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    per_class_f1: dict = field(default_factory=dict)
    kappa: float = 0.0
    n_cells: int = 0
    runtime_seconds: float = 0.0


def load_data(config: dict, sample_size: int | None = None) -> tuple[ad.AnnData, np.ndarray, pl.DataFrame]:
    """Load expression matrix, spatial coordinates, and ground truth."""
    logger.info(f"Loading {config['name']}...")

    # Load expression
    adata = sc.read_10x_h5(str(config["h5_path"]))
    adata.var_names_make_unique()

    # Load spatial coordinates
    cells_df = pl.read_parquet(config["cells_path"])
    cell_ids = cells_df["cell_id"].cast(pl.Utf8).to_list()

    # Align cells between expression and spatial
    expr_cells = set(adata.obs_names)
    spatial_cells = set(cell_ids)
    common = sorted(expr_cells & spatial_cells)

    # Reorder adata to match cells_df order
    cell_to_idx = {c: i for i, c in enumerate(adata.obs_names)}
    mask = [c for c in common if c in cell_to_idx]
    adata = adata[[cell_to_idx[c] for c in mask]].copy()

    # Extract spatial coordinates in same order
    cell_id_to_row = {str(cid): i for i, cid in enumerate(cell_ids)}
    spatial_indices = [cell_id_to_row[c] for c in mask]
    x_coords = cells_df["x_centroid"].to_numpy()[spatial_indices]
    y_coords = cells_df["y_centroid"].to_numpy()[spatial_indices]
    spatial_coords = np.column_stack([x_coords, y_coords])

    # Load ground truth
    gt_df = pd.read_excel(config["gt_path"])
    gt_pl = pl.DataFrame({
        "cell_id": [str(b) for b in gt_df.iloc[:, 0]],
        "gt_fine": gt_df.iloc[:, 1].astype(str).tolist(),
    })
    gt_pl = gt_pl.with_columns(
        pl.col("gt_fine")
        .map_elements(lambda x: GT_BROAD_MAP.get(x, "Unknown"), return_dtype=pl.Utf8)
        .alias("gt_broad")
    ).filter(pl.col("gt_broad") != "Unknown")

    # Filter adata to cells with ground truth
    gt_cell_set = set(gt_pl["cell_id"].to_list())
    gt_mask = [i for i, c in enumerate(adata.obs_names) if c in gt_cell_set]
    adata = adata[gt_mask].copy()
    spatial_coords = spatial_coords[gt_mask]

    # Add cell_id to obs
    adata.obs["cell_id"] = adata.obs_names.tolist()

    # Sample if needed
    if sample_size and sample_size < adata.n_obs:
        logger.info(f"Sampling {sample_size} from {adata.n_obs} cells")
        idx = np.random.choice(adata.n_obs, sample_size, replace=False)
        adata = adata[idx].copy()
        spatial_coords = spatial_coords[idx]

    # Store raw counts
    adata.layers["counts"] = adata.X.copy()

    # Filter cells with too few genes — keep spatial_coords in sync
    n_genes = np.array((adata.X > 0).sum(axis=1)).flatten()
    keep_mask = n_genes >= 5
    n_filtered = (~keep_mask).sum()
    if n_filtered > 0:
        logger.info(f"Filtered {n_filtered} cells with <5 genes")
        adata = adata[keep_mask].copy()
        spatial_coords = spatial_coords[keep_mask]

    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    logger.info(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes, {len(gt_pl)} GT labels")
    assert len(spatial_coords) == adata.n_obs, f"spatial_coords ({len(spatial_coords)}) != adata ({adata.n_obs})"
    return adata, spatial_coords, gt_pl


def evaluate(y_true: list, y_pred: list, method: str, runtime: float = 0.0) -> BenchmarkResult:
    """Evaluate predictions against ground truth."""
    # Filter paired predictions
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t != "Unknown" and p != "Unknown"]
    if not pairs:
        return BenchmarkResult(method=method)

    true, pred = zip(*pairs)
    labels = sorted(set(true) | set(pred))

    acc = accuracy_score(true, pred)
    macro_f1 = f1_score(true, pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(true, pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(true, pred)

    report = classification_report(true, pred, output_dict=True, zero_division=0)
    per_class = {l: round(report[l]["f1-score"], 3) for l in labels if l in report}

    return BenchmarkResult(
        method=method,
        accuracy=round(acc, 4),
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        per_class_f1=per_class,
        kappa=round(kappa, 4),
        n_cells=len(pairs),
        runtime_seconds=round(runtime, 1),
    )


def get_gt_labels(adata: ad.AnnData, gt_df: pl.DataFrame) -> list[str]:
    """Get ground truth broad labels aligned with adata obs order."""
    gt_map = dict(zip(gt_df["cell_id"].to_list(), gt_df["gt_broad"].to_list()))
    return [gt_map.get(c, "Unknown") for c in adata.obs_names]


# ── Annotation Methods ──────────────────────────────────────────────

def run_celltypist(adata: ad.AnnData, model_name: str = "Cells_Adult_Breast.pkl") -> list[str]:
    """Run CellTypist and return broad category predictions."""
    import celltypist
    from celltypist import models as ct_models
    from dapidl.pipeline.components.annotators.mapping import map_to_broad_category

    ct_models.download_models(model=[model_name])
    model = ct_models.Model.load(model=model_name)
    predictions = celltypist.annotate(adata.copy(), model=model, majority_voting=False)
    labels = predictions.predicted_labels.predicted_labels.tolist()
    return [map_to_broad_category(l) for l in labels]


def run_singler(adata: ad.AnnData, reference: str = "blueprint") -> list[str]:
    """Run SingleR and return broad category predictions."""
    from dapidl.pipeline.components.annotators.singler import SingleRAnnotator, is_singler_available
    from dapidl.pipeline.base import AnnotationConfig

    if not is_singler_available():
        raise RuntimeError("SingleR not available")

    config = AnnotationConfig()
    config.singler_reference = reference

    annotator = SingleRAnnotator(config)
    result = annotator.annotate(config=config, adata=adata.copy())
    df = result.annotations_df

    # Map cell_id → prediction, aligned to adata order
    pred_map = dict(zip(df["cell_id"].to_list(), df["broad_category"].to_list()))
    return [pred_map.get(c, "Unknown") for c in adata.obs_names]


def run_sctype(adata: ad.AnnData) -> list[str]:
    """Run scType and return broad category predictions."""
    from dapidl.pipeline.components.annotators.sctype import ScTypeAnnotator
    from dapidl.pipeline.base import AnnotationConfig

    config = AnnotationConfig(fine_grained=True)
    annotator = ScTypeAnnotator(config)
    result = annotator.annotate(config=config, adata=adata.copy())
    df = result.annotations_df

    pred_map = dict(zip(df["cell_id"].to_list(), df["broad_category"].to_list()))
    return [pred_map.get(c, "Unknown") for c in adata.obs_names]


def run_scina(adata: ad.AnnData) -> list[str]:
    """Run SCINA and return broad category predictions."""
    from dapidl.pipeline.components.annotators.scina import SCINAAnnotator
    from dapidl.pipeline.base import AnnotationConfig

    config = AnnotationConfig(fine_grained=True)
    annotator = SCINAAnnotator(config)
    result = annotator.annotate(config=config, adata=adata.copy())
    df = result.annotations_df

    pred_map = dict(zip(df["cell_id"].to_list(), df["broad_category"].to_list()))
    return [pred_map.get(c, "Unknown") for c in adata.obs_names]


# ── BANKSY Methods ──────────────────────────────────────────────────

def banksy_augment(adata: ad.AnnData, spatial_coords: np.ndarray,
                   num_neighbours: int = 15, decay_type: str = "scaled_gaussian") -> ad.AnnData:
    """Apply BANKSY spatial augmentation to expression data.

    Creates a new AnnData with [expression | neighbor_mean] features.
    """
    import logging as _logging
    _logging.getLogger("banksy").setLevel(_logging.WARNING)

    from banksy.main import generate_spatial_weights_fixed_nbrs

    logger.info(f"BANKSY augmentation: {num_neighbours} neighbors, {decay_type} decay")

    # Generate spatial weight graph
    weight_graph, dist_graph, _ = generate_spatial_weights_fixed_nbrs(
        spatial_coords.astype(np.float64),
        m=0,  # No azimuthal component
        num_neighbours=num_neighbours,
        decay_type=decay_type,
    )

    # Get expression matrix (keep sparse if possible)
    X = adata.X

    # Normalize weight graph rows to sum to 1 (stay sparse!)
    from scipy.sparse import diags, issparse as sp_issparse
    W = weight_graph
    if not sp_issparse(W):
        from scipy.sparse import csr_matrix
        W = csr_matrix(W)
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    W_norm = diags(1.0 / row_sums) @ W  # sparse row normalization

    # Sparse × (sparse or dense) → neighbor mean expression
    neighbor_mean = W_norm @ X  # (n_cells, n_genes), stays sparse if X is sparse
    if sp_issparse(neighbor_mean):
        neighbor_mean = neighbor_mean.toarray()
    X_dense = X.toarray() if sp_issparse(X) else np.array(X)

    # Concatenate original + neighbor features
    X_augmented = np.concatenate([X_dense, neighbor_mean], axis=1)

    # Create new AnnData
    gene_names = list(adata.var_names) + [f"nbr_{g}" for g in adata.var_names]
    adata_aug = ad.AnnData(X=X_augmented)
    adata_aug.obs = adata.obs.copy()
    adata_aug.obs_names = adata.obs_names.tolist()
    adata_aug.var_names = gene_names

    logger.info(f"Augmented: {X.shape[1]} → {X_augmented.shape[1]} features")
    return adata_aug


def banksy_leiden_cluster(adata: ad.AnnData, spatial_coords: np.ndarray,
                          num_neighbours: int = 15, resolution: float = 0.5,
                          gt_labels: list[str] | None = None) -> list[str]:
    """BANKSY augmentation + Leiden clustering + label transfer from GT.

    Uses majority ground truth label per cluster to assign cell types.
    This is an ORACLE approach — shows ceiling of spatial clustering quality.
    """
    # Augment
    adata_aug = banksy_augment(adata, spatial_coords, num_neighbours=num_neighbours)

    # Standard scanpy pipeline on augmented features
    sc.pp.pca(adata_aug, n_comps=min(50, adata_aug.n_vars - 1))
    sc.pp.neighbors(adata_aug, n_neighbors=15)
    sc.tl.leiden(adata_aug, resolution=resolution, flavor="igraph", n_iterations=2, directed=False)

    clusters = adata_aug.obs["leiden"].tolist()

    if gt_labels is None:
        return clusters

    # Assign majority GT label to each cluster
    cluster_labels = {}
    for cluster_id in set(clusters):
        cluster_gt = [gt for c, gt in zip(clusters, gt_labels) if c == cluster_id and gt != "Unknown"]
        if cluster_gt:
            cluster_labels[cluster_id] = Counter(cluster_gt).most_common(1)[0][0]
        else:
            cluster_labels[cluster_id] = "Unknown"

    n_clusters = len(set(clusters))
    logger.info(f"BANKSY Leiden: {n_clusters} clusters at resolution={resolution}")

    return [cluster_labels.get(c, "Unknown") for c in clusters]


def spatial_smooth_predictions(predictions: list[str], spatial_coords: np.ndarray,
                                k: int = 15) -> list[str]:
    """Smooth predictions using spatial KNN majority voting.

    For each cell, take majority vote among its K nearest spatial neighbors.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
    nn.fit(spatial_coords)
    _, indices = nn.kneighbors(spatial_coords)

    smoothed = []
    for i in range(len(predictions)):
        neighbor_preds = [predictions[j] for j in indices[i]]  # includes self
        vote = Counter(neighbor_preds).most_common(1)[0][0]
        smoothed.append(vote)

    return smoothed


def leiden_cluster_baseline(adata: ad.AnnData, resolution: float = 0.5,
                            gt_labels: list[str] | None = None) -> list[str]:
    """Standard Leiden clustering (no spatial) + label transfer."""
    adata_tmp = adata.copy()
    sc.pp.pca(adata_tmp, n_comps=min(50, adata_tmp.n_vars - 1))
    sc.pp.neighbors(adata_tmp, n_neighbors=15)
    sc.tl.leiden(adata_tmp, resolution=resolution, flavor="igraph", n_iterations=2, directed=False)

    clusters = adata_tmp.obs["leiden"].tolist()

    if gt_labels is None:
        return clusters

    cluster_labels = {}
    for cluster_id in set(clusters):
        cluster_gt = [gt for c, gt in zip(clusters, gt_labels) if c == cluster_id and gt != "Unknown"]
        if cluster_gt:
            cluster_labels[cluster_id] = Counter(cluster_gt).most_common(1)[0][0]
        else:
            cluster_labels[cluster_id] = "Unknown"

    n_clusters = len(set(clusters))
    logger.info(f"Baseline Leiden: {n_clusters} clusters at resolution={resolution}")

    return [cluster_labels.get(c, "Unknown") for c in clusters]


# ── Main Benchmark ──────────────────────────────────────────────────

def run_benchmark(dataset_keys: list[str], sample_size: int | None = None,
                  output_dir: Path = Path("/tmp/banksy_benchmark")):
    """Run the full benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for ds_key in dataset_keys:
        if ds_key not in DATASETS:
            logger.warning(f"Unknown dataset: {ds_key}")
            continue

        config = DATASETS[ds_key]
        logger.info(f"\n{'='*70}")
        logger.info(f"  DATASET: {config['name']}")
        logger.info(f"{'='*70}")

        adata, spatial_coords, gt_df = load_data(config, sample_size=sample_size)
        gt_labels = get_gt_labels(adata, gt_df)

        results: list[BenchmarkResult] = []
        baseline_preds: dict[str, list[str]] = {}

        # ── 1. Unsupervised clustering (BEFORE R to avoid segfault) ──

        # Standard Leiden (expression only, oracle label transfer)
        logger.info("\n>>> Leiden clustering (expression only, oracle)...")
        for res in [0.3, 0.5, 1.0]:
            t0 = time.time()
            try:
                leiden_preds = leiden_cluster_baseline(adata, resolution=res, gt_labels=gt_labels)
                results.append(evaluate(gt_labels, leiden_preds, f"Leiden_r{res}", time.time() - t0))
            except Exception as e:
                logger.error(f"Leiden r={res} failed: {e}")

        # ── 2. BANKSY spatial methods (before R) ─────────────────

        # BANKSY Leiden clustering (oracle label transfer)
        logger.info("\n>>> BANKSY Leiden clustering (oracle label transfer)...")
        for res in [0.3, 0.5, 1.0]:
            for k in [10, 15, 30]:
                t0 = time.time()
                try:
                    banksy_preds = banksy_leiden_cluster(
                        adata, spatial_coords,
                        num_neighbours=k, resolution=res,
                        gt_labels=gt_labels,
                    )
                    results.append(evaluate(
                        gt_labels, banksy_preds,
                        f"BANKSY_Leiden_k{k}_r{res}",
                        time.time() - t0,
                    ))
                except Exception as e:
                    logger.error(f"BANKSY Leiden k={k} r={res} failed: {e}")

        # ── 3. Baseline annotation methods ───────────────────────

        # CellTypist (Breast model)
        logger.info("\n>>> CellTypist (Breast)...")
        t0 = time.time()
        try:
            ct_preds = run_celltypist(adata, "Cells_Adult_Breast.pkl")
            results.append(evaluate(gt_labels, ct_preds, "CellTypist_Breast", time.time() - t0))
            baseline_preds["ct_breast"] = ct_preds
        except Exception as e:
            logger.error(f"CellTypist Breast failed: {e}")

        # CellTypist (Immune High)
        logger.info("\n>>> CellTypist (Immune_High)...")
        t0 = time.time()
        try:
            ct_imm_preds = run_celltypist(adata, "Immune_All_High.pkl")
            results.append(evaluate(gt_labels, ct_imm_preds, "CellTypist_Immune", time.time() - t0))
            baseline_preds["ct_immune"] = ct_imm_preds
        except Exception as e:
            logger.error(f"CellTypist Immune failed: {e}")

        # scType
        logger.info("\n>>> scType...")
        t0 = time.time()
        try:
            sct_preds = run_sctype(adata)
            results.append(evaluate(gt_labels, sct_preds, "scType", time.time() - t0))
            baseline_preds["sctype"] = sct_preds
        except Exception as e:
            logger.error(f"scType failed: {e}")

        # SCINA
        logger.info("\n>>> SCINA...")
        t0 = time.time()
        try:
            scina_preds = run_scina(adata)
            results.append(evaluate(gt_labels, scina_preds, "SCINA", time.time() - t0))
            baseline_preds["scina"] = scina_preds
        except Exception as e:
            logger.error(f"SCINA failed: {e}")

        # SingleR (Blueprint) — runs LAST because rpy2/R corrupts process state
        logger.info("\n>>> SingleR (Blueprint)...")
        t0 = time.time()
        try:
            sr_preds = run_singler(adata, "blueprint")
            results.append(evaluate(gt_labels, sr_preds, "SingleR_Blueprint", time.time() - t0))
            baseline_preds["sr_blueprint"] = sr_preds
        except Exception as e:
            logger.error(f"SingleR Blueprint failed: {e}")

        # SingleR (HPCA)
        logger.info("\n>>> SingleR (HPCA)...")
        t0 = time.time()
        try:
            sr_hpca_preds = run_singler(adata, "hpca")
            results.append(evaluate(gt_labels, sr_hpca_preds, "SingleR_HPCA", time.time() - t0))
            baseline_preds["sr_hpca"] = sr_hpca_preds
        except Exception as e:
            logger.error(f"SingleR HPCA failed: {e}")

        # ── 4. PopV-style majority voting ensembles ──────────────

        ensemble_preds = None
        if "ct_breast" in baseline_preds and "sr_blueprint" in baseline_preds:
            logger.info("\n>>> Ensemble: majority vote...")
            ensemble_voters = [baseline_preds["ct_breast"], baseline_preds["sr_blueprint"]]
            if "sr_hpca" in baseline_preds:
                ensemble_voters.append(baseline_preds["sr_hpca"])
            if "ct_immune" in baseline_preds:
                ensemble_voters.append(baseline_preds["ct_immune"])

            ensemble_preds = []
            for i in range(len(gt_labels)):
                votes = [v[i] for v in ensemble_voters]
                ensemble_preds.append(Counter(votes).most_common(1)[0][0])
            results.append(evaluate(gt_labels, ensemble_preds, f"Ensemble_{len(ensemble_voters)}methods"))

        # ── 5. Spatial smoothing of baseline predictions ─────────

        logger.info("\n>>> Spatial smoothing of baseline predictions...")
        for method_name, preds in baseline_preds.items():
            for k in [10, 15, 30]:
                t0 = time.time()
                try:
                    smoothed = spatial_smooth_predictions(preds, spatial_coords, k=k)
                    results.append(evaluate(
                        gt_labels, smoothed,
                        f"Spatial_k{k}_{method_name}",
                        time.time() - t0,
                    ))
                except Exception as e:
                    logger.error(f"Spatial smooth {method_name} k={k} failed: {e}")

        # ── 6. Spatial smoothing of ensemble ─────────────────────

        if ensemble_preds is not None:
            for k in [10, 15, 30]:
                t0 = time.time()
                try:
                    smoothed_ens = spatial_smooth_predictions(ensemble_preds, spatial_coords, k=k)
                    results.append(evaluate(
                        gt_labels, smoothed_ens,
                        f"Spatial_k{k}_Ensemble",
                        time.time() - t0,
                    ))
                except Exception as e:
                    logger.error(f"Spatial smooth ensemble k={k} failed: {e}")

        # ── Print results ────────────────────────────────────────

        all_results[ds_key] = results
        print_results(ds_key, results)

    # Save JSON summary
    summary = {}
    for ds_key, results in all_results.items():
        summary[ds_key] = [
            {
                "method": r.method,
                "accuracy": r.accuracy,
                "macro_f1": r.macro_f1,
                "weighted_f1": r.weighted_f1,
                "per_class_f1": r.per_class_f1,
                "kappa": r.kappa,
                "n_cells": r.n_cells,
                "runtime_seconds": r.runtime_seconds,
            }
            for r in results
        ]

    summary_path = output_dir / "banksy_benchmark_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"\nResults saved to {summary_path}")


def print_results(dataset: str, results: list[BenchmarkResult]):
    """Print formatted results table."""
    print(f"\n{'='*100}")
    print(f"  {dataset.upper()} — BENCHMARK RESULTS (sorted by Macro F1)")
    print(f"{'='*100}")
    print(f"{'Method':<35} {'Acc':>7} {'F1_M':>7} {'F1_W':>7} {'Epi':>7} {'Imm':>7} {'Str':>7} {'κ':>7} {'Time':>7}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: -x.macro_f1):
        epi = r.per_class_f1.get("Epithelial", 0)
        imm = r.per_class_f1.get("Immune", 0)
        stro = r.per_class_f1.get("Stromal", 0)
        print(
            f"{r.method:<35} {r.accuracy:>7.3f} {r.macro_f1:>7.3f} {r.weighted_f1:>7.3f} "
            f"{epi:>7.3f} {imm:>7.3f} {stro:>7.3f} {r.kappa:>7.3f} {r.runtime_seconds:>6.1f}s"
        )

    # Category summary
    print(f"\n  Category Breakdown:")
    categories = {
        "Baseline": [r for r in results if not r.method.startswith(("BANKSY", "Spatial", "Leiden", "Ensemble"))],
        "Ensemble": [r for r in results if r.method.startswith("Ensemble")],
        "Leiden (no spatial)": [r for r in results if r.method.startswith("Leiden")],
        "BANKSY Leiden": [r for r in results if r.method.startswith("BANKSY")],
        "Spatial Smoothing": [r for r in results if r.method.startswith("Spatial")],
    }
    for cat, cat_results in categories.items():
        if cat_results:
            best = max(cat_results, key=lambda x: x.macro_f1)
            print(f"  Best {cat:<25}: {best.method:<35} F1={best.macro_f1:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BANKSY spatial benchmark")
    parser.add_argument("--datasets", "-d", nargs="+", choices=list(DATASETS.keys()),
                        default=["rep1", "rep2"])
    parser.add_argument("--sample-size", "-s", type=int, default=None)
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("/tmp/banksy_benchmark"))
    args = parser.parse_args()

    run_benchmark(args.datasets, args.sample_size, args.output_dir)
