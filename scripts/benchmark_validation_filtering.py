#!/usr/bin/env python3
"""
Benchmark: annotation methods with GT-free confidence filtering.

Compares unfiltered vs filtered F1 for:
- 4 single methods (BANKSY, CellTypist Breast, CellTypist Immune, scType)
- 5 consensus combinations (best from previous benchmark + new candidates)

Each is evaluated:
  1. Unfiltered (all cells)
  2. Filtered @ 0.3 (recommended default)
  3. Filtered @ 0.5 (conservative)

CRITICAL: All Leiden/BANKSY clustering BEFORE SingleR/R calls (segfault risk).
SingleR excluded from this benchmark (too slow, F1=0.28 not worth it).

Usage:
    uv run python scripts/benchmark_validation_filtering.py
    uv run python scripts/benchmark_validation_filtering.py --sample-size 10000
"""

import itertools
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from loguru import logger
from scipy.sparse import csr_matrix, diags, issparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors

from dapidl.validation.annotation_confidence import (
    AnnotationConfidenceConfig,
    compute_annotation_confidence,
    filter_predictions,
)

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<5} | {message}")

# ── Dataset Config ────────────────────────────────────────────────

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

GT_BROAD_MAP = {
    "DCIS_1": "Epithelial", "DCIS_2": "Epithelial",
    "Invasive_Tumor": "Epithelial", "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_KRT15+": "Epithelial", "Myoepi_ACTA2+": "Epithelial",
    "T_Cell_&_Tumor_Hybrid": "Epithelial",
    "CD4+_T_Cells": "Immune", "CD8+_T_Cells": "Immune",
    "B_Cells": "Immune", "Macrophages_1": "Immune", "Macrophages_2": "Immune",
    "Mast_Cells": "Immune", "LAMP3+_DCs": "Immune", "IRF7+_DCs": "Immune",
    "Perivascular-Like": "Immune",
    "Stromal": "Stromal", "Stromal_&_T_Cell_Hybrid": "Stromal", "Endothelial": "Stromal",
}

LABELS = ["Epithelial", "Immune", "Stromal"]


# ── Data Loading ──────────────────────────────────────────────────

def load_data(config: dict, sample_size: int | None = None):
    """Load expression, spatial coords, ground truth."""
    logger.info(f"Loading {config['name']}...")

    adata = sc.read_10x_h5(str(config["h5_path"]))
    adata.var_names_make_unique()

    cells_df = pl.read_parquet(config["cells_path"])
    cell_ids = cells_df["cell_id"].cast(pl.Utf8).to_list()

    expr_cells = set(adata.obs_names)
    common = sorted(expr_cells & set(cell_ids))
    cell_to_idx = {c: i for i, c in enumerate(adata.obs_names)}
    mask = [c for c in common if c in cell_to_idx]
    adata = adata[[cell_to_idx[c] for c in mask]].copy()

    cell_id_to_row = {str(cid): i for i, cid in enumerate(cell_ids)}
    spatial_indices = [cell_id_to_row[c] for c in mask]
    x = cells_df["x_centroid"].to_numpy()[spatial_indices]
    y = cells_df["y_centroid"].to_numpy()[spatial_indices]
    spatial_coords = np.column_stack([x, y])

    # Ground truth
    gt_df = pd.read_excel(config["gt_path"])
    gt_pl = pl.DataFrame({
        "cell_id": [str(b) for b in gt_df.iloc[:, 0]],
        "gt_fine": gt_df.iloc[:, 1].astype(str).tolist(),
    })
    gt_pl = gt_pl.with_columns(
        pl.col("gt_fine")
        .map_elements(lambda v: GT_BROAD_MAP.get(v, "Unknown"), return_dtype=pl.Utf8)
        .alias("gt_broad")
    ).filter(pl.col("gt_broad") != "Unknown")

    gt_cell_set = set(gt_pl["cell_id"].to_list())
    gt_mask = [i for i, c in enumerate(adata.obs_names) if c in gt_cell_set]
    adata = adata[gt_mask].copy()
    spatial_coords = spatial_coords[gt_mask]

    if sample_size and sample_size < adata.n_obs:
        rng = np.random.default_rng(42)
        idx = rng.choice(adata.n_obs, sample_size, replace=False)
        adata = adata[idx].copy()
        spatial_coords = spatial_coords[idx]

    adata.layers["counts"] = adata.X.copy()

    # Filter very low quality cells
    n_genes = np.array((adata.X > 0).sum(axis=1)).flatten()
    keep = n_genes >= 5
    if (~keep).sum() > 0:
        adata = adata[keep].copy()
        spatial_coords = spatial_coords[keep]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # GT labels aligned to adata
    gt_map = dict(zip(gt_pl["cell_id"].to_list(), gt_pl["gt_broad"].to_list()))
    gt_labels = np.array([gt_map.get(c, "Unknown") for c in adata.obs_names])

    adata.obs["x_centroid"] = spatial_coords[:, 0]
    adata.obs["y_centroid"] = spatial_coords[:, 1]

    logger.info(f"Loaded {adata.n_obs} cells, GT: {Counter(gt_labels)}")
    return adata, spatial_coords, gt_labels


# ── Annotation Methods ────────────────────────────────────────────

def run_banksy(adata: ad.AnnData, spatial_coords: np.ndarray) -> np.ndarray:
    """BANKSY + marker_gene_overlap (PanglaoDB)."""
    import decoupler as dc
    from banksy.main import generate_spatial_weights_fixed_nbrs

    logger.info("BANKSY: k=10, res=1.0")
    W, _, _ = generate_spatial_weights_fixed_nbrs(
        spatial_coords.astype(np.float64), m=0,
        num_neighbours=10, decay_type="scaled_gaussian",
    )
    if not issparse(W):
        W = csr_matrix(W)
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    W_norm = diags(1.0 / row_sums) @ W

    X = adata.X
    neighbor_mean = W_norm @ X
    if issparse(neighbor_mean):
        neighbor_mean = neighbor_mean.toarray()
    X_dense = X.toarray() if issparse(X) else np.array(X)
    X_aug = np.concatenate([X_dense, neighbor_mean], axis=1)

    gene_names = list(adata.var_names) + [f"nbr_{g}" for g in adata.var_names]
    adata_aug = ad.AnnData(X=X_aug)
    adata_aug.obs = adata.obs.copy()
    adata_aug.obs_names = adata.obs_names.tolist()
    adata_aug.var_names = gene_names

    sc.pp.pca(adata_aug, n_comps=min(50, adata_aug.n_vars - 1))
    sc.pp.neighbors(adata_aug, n_neighbors=15)
    sc.tl.leiden(adata_aug, resolution=1.0, flavor="igraph", n_iterations=2, directed=False)

    n_clusters = len(adata_aug.obs["leiden"].unique())
    logger.info(f"BANKSY: {n_clusters} clusters")

    adata.obs["banksy_leiden"] = adata_aug.obs["leiden"].values

    # DEG analysis on original expression
    adata_degs = adata.copy()
    sc.tl.rank_genes_groups(adata_degs, groupby="banksy_leiden", method="wilcoxon")

    # PanglaoDB markers
    markers = dc.op.resource("PanglaoDB", organism="human")
    markers = markers[
        markers["canonical_marker"].astype(bool)
        & (markers["human_sensitivity"].astype(float) > 0.5)
    ]
    markers = markers[~markers.duplicated(["cell_type", "genesymbol"])]
    panel_genes = set(adata.var_names)
    markers = markers[markers["genesymbol"].isin(panel_genes)]

    # Build broad marker dict
    panglao_broad_map = {
        "Epithelial cells": "Epithelial", "Basal cells": "Epithelial",
        "Luminal epithelial cells": "Epithelial", "Ductal cells": "Epithelial",
        "Myoepithelial cells": "Epithelial", "Keratinocytes": "Epithelial",
        "Goblet cells": "Epithelial", "Enterocytes": "Epithelial",
        "Clara cells": "Epithelial", "Alveolar cells": "Epithelial",
        "T cells": "Immune", "T memory cells": "Immune", "T helper cells": "Immune",
        "T regulatory cells": "Immune", "T cytotoxic cells": "Immune",
        "NK cells": "Immune", "NKT cells": "Immune",
        "B cells": "Immune", "B cells memory": "Immune",
        "Plasma cells": "Immune", "Dendritic cells": "Immune",
        "Macrophages": "Immune", "Monocytes": "Immune", "Mast cells": "Immune",
        "Neutrophils": "Immune", "Basophils": "Immune",
        "Fibroblasts": "Stromal", "Myofibroblasts": "Stromal",
        "Endothelial cells": "Stromal", "Pericytes": "Stromal",
        "Smooth muscle cells": "Stromal", "Adipocytes": "Stromal",
        "Lymphatic endothelial cells": "Stromal",
    }

    broad_markers: dict[str, set[str]] = {}
    for _, row in markers.iterrows():
        broad = panglao_broad_map.get(row["cell_type"])
        if broad:
            broad_markers.setdefault(broad, set()).add(row["genesymbol"])

    # marker_gene_overlap
    overlap_df = sc.tl.marker_gene_overlap(
        adata_degs, reference_markers=broad_markers,
        method="jaccard", top_n_markers=50,
    )
    cluster_to_type = overlap_df.idxmax(axis=0).to_dict()
    clusters = adata.obs["banksy_leiden"].tolist()

    return np.array([cluster_to_type.get(c, "Unknown") for c in clusters])


def run_celltypist(adata: ad.AnnData, model_name: str) -> np.ndarray:
    """CellTypist → broad categories."""
    import celltypist
    from celltypist import models as ct_models
    from dapidl.pipeline.components.annotators.mapping import map_to_broad_category

    ct_models.download_models(model=[model_name])
    model = ct_models.Model.load(model=model_name)
    predictions = celltypist.annotate(adata.copy(), model=model, majority_voting=False)
    labels = predictions.predicted_labels.predicted_labels.tolist()
    return np.array([map_to_broad_category(l) for l in labels])


def run_sctype(adata: ad.AnnData) -> np.ndarray:
    """scType → broad categories."""
    from dapidl.pipeline.base import AnnotationConfig
    from dapidl.pipeline.components.annotators.sctype import ScTypeAnnotator

    config = AnnotationConfig(fine_grained=True)
    annotator = ScTypeAnnotator(config)
    result = annotator.annotate(config=config, adata=adata.copy())
    df = result.annotations_df
    pred_map = dict(zip(df["cell_id"].to_list(), df["broad_category"].to_list()))
    return np.array([pred_map.get(c, "Unknown") for c in adata.obs_names])


def majority_vote(method_preds: dict[str, np.ndarray], method_names: list[str]) -> np.ndarray:
    """Unweighted majority vote. Ties go to first method."""
    n_cells = len(next(iter(method_preds.values())))
    result = []
    for i in range(n_cells):
        votes = [method_preds[m][i] for m in method_names if method_preds[m][i] != "Unknown"]
        if votes:
            result.append(Counter(votes).most_common(1)[0][0])
        else:
            result.append("Unknown")
    return np.array(result)


# ── Evaluation ────────────────────────────────────────────────────

@dataclass
class MethodResult:
    """Results for one method (single or consensus)."""
    name: str
    n_methods: int  # 1 for single, 2+ for consensus
    # Unfiltered
    f1_unfiltered: float = 0.0
    acc_unfiltered: float = 0.0
    per_class_unfiltered: dict = field(default_factory=dict)
    confidence_score: float = 0.0
    # Filtered at various thresholds
    filtered: dict = field(default_factory=dict)  # threshold → {f1, acc, n_kept, pct_kept, per_class}


def evaluate_with_filtering(
    name: str,
    preds: np.ndarray,
    gt: np.ndarray,
    all_preds: dict[str, np.ndarray],
    adata: ad.AnnData,
    spatial_coords: np.ndarray,
    config: AnnotationConfidenceConfig,
    primary_method: str | None = None,
    thresholds: list[float] | None = None,
) -> MethodResult:
    """Evaluate a method unfiltered + at multiple filtering thresholds."""
    thresholds = thresholds or [0.3, 0.5]
    result = MethodResult(name=name, n_methods=len(all_preds))

    # Unfiltered evaluation
    valid = (gt != "Unknown") & (preds != "Unknown")
    if valid.sum() == 0:
        return result

    result.f1_unfiltered = float(f1_score(gt[valid], preds[valid], average="macro", labels=LABELS, zero_division=0))
    result.acc_unfiltered = float(accuracy_score(gt[valid], preds[valid]))
    for label in LABELS:
        result.per_class_unfiltered[label] = float(
            f1_score(gt[valid], preds[valid], average=None, labels=[label], zero_division=0)[0]
        )

    # Compute confidence
    conf_result = compute_annotation_confidence(
        adata=adata,
        predictions=all_preds,
        spatial_coords=spatial_coords,
        config=config,
        primary_method=primary_method,
    )
    result.confidence_score = conf_result.overall_score

    # Filter at each threshold
    for t in thresholds:
        filt = filter_predictions(preds, conf_result, min_confidence=t)
        kept = filt.mask & (gt != "Unknown") & (filt.predictions != "Unknown")
        if kept.sum() > 0:
            f1_filt = float(f1_score(gt[kept], filt.predictions[kept], average="macro", labels=LABELS, zero_division=0))
            acc_filt = float(accuracy_score(gt[kept], filt.predictions[kept]))
            per_class = {}
            for label in LABELS:
                per_class[label] = float(
                    f1_score(gt[kept], filt.predictions[kept], average=None, labels=[label], zero_division=0)[0]
                )
            result.filtered[t] = {
                "f1": f1_filt,
                "acc": acc_filt,
                "n_kept": int(kept.sum()),
                "pct_kept": float(kept.sum() / valid.sum() * 100),
                "per_class": per_class,
            }

    return result


# ── Main ──────────────────────────────────────────────────────────

def run_benchmark(dataset_keys: list[str], sample_size: int | None = None):
    """Run full benchmark with validation filtering."""

    all_dataset_results: dict[str, list[MethodResult]] = {}

    for ds_key in dataset_keys:
        config = DATASETS[ds_key]
        logger.info(f"\n{'='*80}")
        logger.info(f"  DATASET: {config['name']}")
        logger.info(f"{'='*80}")

        adata, spatial_coords, gt = load_data(config, sample_size=sample_size)
        conf_config = AnnotationConfidenceConfig(tissue_type="breast")

        method_preds: dict[str, np.ndarray] = {}
        results: list[MethodResult] = []

        # ── Phase 1: BANKSY (must run before R) ──────────────────
        logger.info("\n>>> [1/4] BANKSY + marker_gene_overlap...")
        t0 = time.time()
        try:
            banksy_preds = run_banksy(adata, spatial_coords)
            method_preds["BANKSY"] = banksy_preds
            logger.info(f"  Done in {time.time()-t0:.1f}s — {Counter(banksy_preds.tolist())}")
        except Exception as e:
            logger.error(f"BANKSY failed: {e}")
            import traceback; traceback.print_exc()

        # ── Phase 2: CellTypist ──────────────────────────────────
        logger.info("\n>>> [2/4] CellTypist (Breast)...")
        t0 = time.time()
        try:
            ct_breast = run_celltypist(adata, "Cells_Adult_Breast.pkl")
            method_preds["CT_Breast"] = ct_breast
            logger.info(f"  Done in {time.time()-t0:.1f}s — {Counter(ct_breast.tolist())}")
        except Exception as e:
            logger.error(f"CellTypist Breast failed: {e}")

        logger.info("\n>>> [3/4] CellTypist (Immune)...")
        t0 = time.time()
        try:
            ct_immune = run_celltypist(adata, "Immune_All_High.pkl")
            method_preds["CT_Immune"] = ct_immune
            logger.info(f"  Done in {time.time()-t0:.1f}s — {Counter(ct_immune.tolist())}")
        except Exception as e:
            logger.error(f"CellTypist Immune failed: {e}")

        # ── Phase 3: scType ──────────────────────────────────────
        logger.info("\n>>> [4/4] scType...")
        t0 = time.time()
        try:
            sctype_preds = run_sctype(adata)
            method_preds["scType"] = sctype_preds
            logger.info(f"  Done in {time.time()-t0:.1f}s — {Counter(sctype_preds.tolist())}")
        except Exception as e:
            logger.error(f"scType failed: {e}")

        # ── Evaluate single methods ──────────────────────────────
        logger.info("\n>>> Evaluating single methods with filtering...")
        for name, preds in method_preds.items():
            r = evaluate_with_filtering(
                name=name, preds=preds, gt=gt,
                all_preds={name: preds},  # Single method
                adata=adata, spatial_coords=spatial_coords,
                config=conf_config,
            )
            results.append(r)

        # ── Consensus combinations ───────────────────────────────
        CONSENSUS_COMBOS = [
            # Best from previous benchmark
            ("BANKSY+CT_Breast+scType", ["BANKSY", "CT_Breast", "scType"]),
            # New candidates
            ("BANKSY+CT_Immune+scType", ["BANKSY", "CT_Immune", "scType"]),
            ("BANKSY+CT_Breast+CT_Immune", ["BANKSY", "CT_Breast", "CT_Immune"]),
            ("BANKSY+CT_Breast+CT_Immune+scType", ["BANKSY", "CT_Breast", "CT_Immune", "scType"]),
            # CellTypist-only consensus (no BANKSY)
            ("CT_Breast+CT_Immune+scType", ["CT_Breast", "CT_Immune", "scType"]),
        ]

        logger.info("\n>>> Evaluating consensus combinations with filtering...")
        for combo_name, combo_methods in CONSENSUS_COMBOS:
            # Check all methods available
            available = [m for m in combo_methods if m in method_preds]
            if len(available) < len(combo_methods):
                missing = set(combo_methods) - set(available)
                logger.warning(f"Skipping {combo_name}: missing {missing}")
                continue

            consensus_preds = majority_vote(method_preds, combo_methods)

            # For confidence: pass all component methods
            combo_preds_dict = {m: method_preds[m] for m in combo_methods}
            # Primary = first method (BANKSY if present, else first)
            primary = combo_methods[0]

            r = evaluate_with_filtering(
                name=combo_name, preds=consensus_preds, gt=gt,
                all_preds=combo_preds_dict,
                adata=adata, spatial_coords=spatial_coords,
                config=conf_config,
                primary_method=primary,
            )
            results.append(r)

        all_dataset_results[ds_key] = results

        # ── Print results table ──────────────────────────────────
        print(f"\n{'='*100}")
        print(f"  {config['name']}: UNFILTERED vs FILTERED RESULTS")
        print(f"{'='*100}")
        print(f"  {'Method':<35} {'Conf':>5} {'F1':>6} {'F1@0.3':>7} {'%kept':>6} {'F1@0.5':>7} {'%kept':>6}  {'Epi':>5} {'Imm':>5} {'Str':>5}")
        print(f"  {'-'*95}")

        # Sort: singles first, then consensus, by unfiltered F1 desc
        singles = sorted([r for r in results if r.n_methods == 1], key=lambda r: -r.f1_unfiltered)
        consensus = sorted([r for r in results if r.n_methods > 1], key=lambda r: -r.f1_unfiltered)

        for section_name, section in [("SINGLE METHODS", singles), ("CONSENSUS", consensus)]:
            print(f"  {section_name}")
            for r in section:
                f1_03 = r.filtered.get(0.3, {}).get("f1", 0)
                pct_03 = r.filtered.get(0.3, {}).get("pct_kept", 0)
                f1_05 = r.filtered.get(0.5, {}).get("f1", 0)
                pct_05 = r.filtered.get(0.5, {}).get("pct_kept", 0)
                epi = r.per_class_unfiltered.get("Epithelial", 0)
                imm = r.per_class_unfiltered.get("Immune", 0)
                stro = r.per_class_unfiltered.get("Stromal", 0)
                print(f"  {r.name:<35} {r.confidence_score:>5.3f} {r.f1_unfiltered:>6.3f} "
                      f"{f1_03:>7.3f} {pct_03:>5.1f}% {f1_05:>7.3f} {pct_05:>5.1f}%  "
                      f"{epi:>.3f} {imm:>.3f} {stro:>.3f}")
            print()

    # ── Cross-replicate summary ──────────────────────────────────
    if len(all_dataset_results) > 1:
        print(f"\n{'='*100}")
        print(f"  CROSS-REPLICATE SUMMARY")
        print(f"{'='*100}")
        print(f"  {'Method':<35} {'Rep1 F1':>7} {'@0.3':>6} {'@0.5':>6} {'Rep2 F1':>8} {'@0.3':>6} {'@0.5':>6}")
        print(f"  {'-'*75}")

        # Collect unique method names across both datasets
        method_names = []
        for ds_results in all_dataset_results.values():
            for r in ds_results:
                if r.name not in method_names:
                    method_names.append(r.name)

        for name in method_names:
            cols = []
            for ds_key in ["rep1", "rep2"]:
                ds_results = all_dataset_results.get(ds_key, [])
                r = next((r for r in ds_results if r.name == name), None)
                if r:
                    f1_03 = r.filtered.get(0.3, {}).get("f1", 0)
                    f1_05 = r.filtered.get(0.5, {}).get("f1", 0)
                    cols.append(f"{r.f1_unfiltered:>7.3f} {f1_03:>6.3f} {f1_05:>6.3f}")
                else:
                    cols.append(f"{'  -':>7} {'  -':>6} {'  -':>6}")
            print(f"  {name:<35} {cols[0]} {cols[1]}")

        print(f"{'='*100}")

    # Save JSON results
    output_path = Path("/tmp/benchmark_validation_filtering.json")
    json_results = {}
    for ds_key, ds_results in all_dataset_results.items():
        json_results[ds_key] = []
        for r in ds_results:
            json_results[ds_key].append({
                "name": r.name,
                "n_methods": r.n_methods,
                "confidence": r.confidence_score,
                "f1_unfiltered": r.f1_unfiltered,
                "acc_unfiltered": r.acc_unfiltered,
                "per_class_unfiltered": r.per_class_unfiltered,
                "filtered": r.filtered,
            })
    output_path.write_text(json.dumps(json_results, indent=2))
    logger.info(f"Results saved to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["rep1", "rep2"])
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()
    run_benchmark(args.datasets, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
