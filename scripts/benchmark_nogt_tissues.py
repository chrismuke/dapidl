#!/usr/bin/env python3
"""
No-GT benchmark: BANKSY+CT_Breast+scType consensus with confidence filtering
on lung, kidney, and liver datasets (no ground truth available).

Uses the standard consensus pipeline validated on breast (F1=0.869):
  1. Run 3 methods: BANKSY, CellTypist (Breast), scType
  2. Consensus via majority vote
  3. Confidence filtering at 0.3 and 0.5 thresholds
  4. Report confidence scores + proportions + cell type distributions

Since no GT exists, we evaluate via:
  - Overall confidence score (marker enrichment + spatial coherence + consensus)
  - Proportion plausibility for each tissue type
  - Per-cell type confidence breakdown
  - Filter statistics (how many cells pass each threshold)

Usage:
    uv run python scripts/benchmark_nogt_tissues.py
    uv run python scripts/benchmark_nogt_tissues.py --sample-size 20000
    uv run python scripts/benchmark_nogt_tissues.py --datasets lung
"""

import json
import os
import sys
import time
from collections import Counter
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

from dapidl.validation.annotation_confidence import (
    AnnotationConfidenceConfig,
    compute_annotation_confidence,
    filter_predictions,
)

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<5} | {message}")


# ── Dataset Config ────────────────────────────────────────────────

DATASETS = {
    "lung": {
        "path": Path("/mnt/work/datasets/raw/xenium/xenium-lung-cancer"),
        "tissue_type": "lung",
        "name": "Lung Cancer",
    },
    "kidney": {
        "path": Path("/mnt/work/datasets/raw/xenium/xenium-kidney-normal"),
        "tissue_type": "kidney",
        "name": "Kidney Normal",
    },
    "liver": {
        "path": Path("/mnt/work/datasets/raw/xenium/xenium-liver-normal"),
        "tissue_type": "liver",
        "name": "Liver Normal",
    },
}


# ── Data Loading ──────────────────────────────────────────────────

def load_data(config: dict, sample_size: int | None = None):
    """Load expression + spatial coords (no GT)."""
    data_path = config["path"]
    logger.info(f"Loading {config['name']} from {data_path}...")

    adata = sc.read_10x_h5(str(data_path / "cell_feature_matrix.h5"))
    adata.var_names_make_unique()

    cells_df = pl.read_parquet(data_path / "cells.parquet")
    cell_ids = cells_df["cell_id"].cast(pl.Utf8).to_list()

    # Align expression and spatial
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

    if sample_size and sample_size < adata.n_obs:
        rng = np.random.default_rng(42)
        idx = rng.choice(adata.n_obs, sample_size, replace=False)
        adata = adata[idx].copy()
        spatial_coords = spatial_coords[idx]

    adata.layers["counts"] = adata.X.copy()

    # Filter low-quality cells
    n_genes = np.array((adata.X > 0).sum(axis=1)).flatten()
    keep = n_genes >= 5
    if (~keep).sum() > 0:
        logger.info(f"Filtered {(~keep).sum()} cells with <5 genes")
        adata = adata[keep].copy()
        spatial_coords = spatial_coords[keep]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata.obs["x_centroid"] = spatial_coords[:, 0]
    adata.obs["y_centroid"] = spatial_coords[:, 1]

    logger.info(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
    return adata, spatial_coords


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

    # DEG on original expression
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

    panglao_broad_map = {
        "Epithelial cells": "Epithelial", "Basal cells": "Epithelial",
        "Luminal epithelial cells": "Epithelial", "Ductal cells": "Epithelial",
        "Myoepithelial cells": "Epithelial", "Keratinocytes": "Epithelial",
        "Goblet cells": "Epithelial", "Enterocytes": "Epithelial",
        "Clara cells": "Epithelial", "Alveolar cells": "Epithelial",
        "Hepatocytes": "Epithelial", "Cholangiocytes": "Epithelial",
        "Podocytes": "Epithelial", "Proximal tubular cells": "Epithelial",
        "T cells": "Immune", "T memory cells": "Immune", "T helper cells": "Immune",
        "T regulatory cells": "Immune", "T cytotoxic cells": "Immune",
        "NK cells": "Immune", "NKT cells": "Immune",
        "B cells": "Immune", "B cells memory": "Immune",
        "Plasma cells": "Immune", "Dendritic cells": "Immune",
        "Macrophages": "Immune", "Monocytes": "Immune", "Mast cells": "Immune",
        "Neutrophils": "Immune", "Basophils": "Immune",
        "Kupffer cells": "Immune",
        "Fibroblasts": "Stromal", "Myofibroblasts": "Stromal",
        "Endothelial cells": "Stromal", "Pericytes": "Stromal",
        "Smooth muscle cells": "Stromal", "Adipocytes": "Stromal",
        "Lymphatic endothelial cells": "Stromal",
        "Stellate cells": "Stromal",
    }

    broad_markers: dict[str, set[str]] = {}
    for _, row in markers.iterrows():
        broad = panglao_broad_map.get(row["cell_type"])
        if broad:
            broad_markers.setdefault(broad, set()).add(row["genesymbol"])

    logger.info(f"PanglaoDB broad markers in panel: {sum(len(v) for v in broad_markers.values())} "
                f"across {len(broad_markers)} types")

    overlap_df = sc.tl.marker_gene_overlap(
        adata_degs, reference_markers=broad_markers,
        method="jaccard", top_n_markers=50,
    )
    cluster_to_type = overlap_df.idxmax(axis=0).to_dict()
    logger.info(f"Cluster assignments: {cluster_to_type}")

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
    """Unweighted majority vote."""
    n_cells = len(next(iter(method_preds.values())))
    result = []
    for i in range(n_cells):
        votes = [method_preds[m][i] for m in method_names if method_preds[m][i] != "Unknown"]
        if votes:
            result.append(Counter(votes).most_common(1)[0][0])
        else:
            result.append("Unknown")
    return np.array(result)


# ── Main ──────────────────────────────────────────────────────────

def run_tissue(ds_key: str, sample_size: int | None = None):
    """Run standard consensus pipeline on one tissue."""
    config = DATASETS[ds_key]
    tissue = config["tissue_type"]

    print(f"\n{'='*80}")
    print(f"  {config['name'].upper()} ({tissue}) — No Ground Truth")
    print(f"{'='*80}")

    adata, spatial_coords = load_data(config, sample_size=sample_size)
    conf_config = AnnotationConfidenceConfig(tissue_type=tissue)

    method_preds: dict[str, np.ndarray] = {}

    # ── Phase 1: BANKSY (must be first, before any R) ────────
    logger.info("\n>>> [1/3] BANKSY + marker_gene_overlap...")
    t0 = time.time()
    try:
        banksy_preds = run_banksy(adata, spatial_coords)
        method_preds["BANKSY"] = banksy_preds
        logger.info(f"  Done in {time.time()-t0:.1f}s — {Counter(banksy_preds.tolist())}")
    except Exception as e:
        logger.error(f"BANKSY failed: {e}")
        import traceback; traceback.print_exc()

    # ── Phase 2: CellTypist ──────────────────────────────────
    logger.info("\n>>> [2/3] CellTypist (Breast)...")
    t0 = time.time()
    try:
        ct_preds = run_celltypist(adata, "Cells_Adult_Breast.pkl")
        method_preds["CT_Breast"] = ct_preds
        logger.info(f"  Done in {time.time()-t0:.1f}s — {Counter(ct_preds.tolist())}")
    except Exception as e:
        logger.error(f"CellTypist failed: {e}")

    # ── Phase 3: scType ──────────────────────────────────────
    logger.info("\n>>> [3/3] scType...")
    t0 = time.time()
    try:
        sct_preds = run_sctype(adata)
        method_preds["scType"] = sct_preds
        logger.info(f"  Done in {time.time()-t0:.1f}s — {Counter(sct_preds.tolist())}")
    except Exception as e:
        logger.error(f"scType failed: {e}")

    # ── Consensus vote ───────────────────────────────────────
    consensus_methods = ["BANKSY", "CT_Breast", "scType"]
    available = [m for m in consensus_methods if m in method_preds]
    if len(available) < 2:
        logger.error(f"Only {len(available)} methods available — cannot form consensus")
        return None

    consensus_preds = majority_vote(method_preds, available)
    method_preds["Consensus"] = consensus_preds

    # ── Confidence for each method ───────────────────────────
    results = {}
    for name, preds in [
        ("BANKSY", method_preds.get("BANKSY")),
        ("CT_Breast", method_preds.get("CT_Breast")),
        ("scType", method_preds.get("scType")),
    ]:
        if preds is None:
            continue
        conf = compute_annotation_confidence(
            adata=adata, predictions={name: preds},
            spatial_coords=spatial_coords, config=conf_config,
        )
        results[name] = {"confidence": conf, "preds": preds}

    # Consensus confidence (multi-method)
    combo_dict = {m: method_preds[m] for m in available}
    consensus_conf = compute_annotation_confidence(
        adata=adata, predictions=combo_dict,
        spatial_coords=spatial_coords, config=conf_config,
        primary_method=available[0],  # BANKSY if available
    )
    results["Consensus"] = {"confidence": consensus_conf, "preds": consensus_preds}

    # ── Filtering ────────────────────────────────────────────
    filt_03 = filter_predictions(consensus_preds, consensus_conf, min_confidence=0.3)
    filt_05 = filter_predictions(consensus_preds, consensus_conf, min_confidence=0.5)

    # ── Print Results ────────────────────────────────────────
    print(f"\n  Cell Distribution:")
    print(f"  {'Method':<20} {'Epi':>8} {'Imm':>8} {'Str':>8} {'Other':>8} {'Total':>8}")
    print(f"  {'-'*52}")
    for name in ["BANKSY", "CT_Breast", "scType", "Consensus"]:
        if name not in results:
            continue
        preds = results[name]["preds"]
        c = Counter(preds.tolist())
        epi = c.get("Epithelial", 0)
        imm = c.get("Immune", 0)
        stro = c.get("Stromal", 0)
        other = sum(v for k, v in c.items() if k not in ("Epithelial", "Immune", "Stromal"))
        print(f"  {name:<20} {epi:>8} {imm:>8} {stro:>8} {other:>8} {len(preds):>8}")

    # Filtered distributions
    c03 = Counter(filt_03.predictions[filt_03.mask].tolist())
    c05 = Counter(filt_05.predictions[filt_05.mask].tolist())
    for label, c, filt in [("Consensus @0.3", c03, filt_03), ("Consensus @0.5", c05, filt_05)]:
        epi = c.get("Epithelial", 0)
        imm = c.get("Immune", 0)
        stro = c.get("Stromal", 0)
        other = sum(v for k, v in c.items() if k not in ("Epithelial", "Immune", "Stromal"))
        print(f"  {label:<20} {epi:>8} {imm:>8} {stro:>8} {other:>8} {filt.n_kept:>8}")

    print(f"\n  Confidence Scores:")
    print(f"  {'Method':<20} {'Overall':>8} {'Marker':>8} {'Spatial':>8} {'Consensus':>10} {'Plausible':>10}")
    print(f"  {'-'*64}")
    for name in ["BANKSY", "CT_Breast", "scType", "Consensus"]:
        if name not in results:
            continue
        conf = results[name]["confidence"]
        cons_str = f"{conf.overall_consensus_score:.3f}" if not np.isnan(conf.overall_consensus_score) else "    -"
        plaus = "Yes" if conf.proportion_plausible else "NO"
        print(f"  {name:<20} {conf.overall_score:>8.3f} {conf.overall_marker_score:>8.3f} "
              f"{conf.overall_spatial_coherence:>8.3f} {cons_str:>10} {plaus:>10}")

    # Full confidence report for consensus
    print(f"\n  --- Consensus Confidence Report ---")
    print(consensus_conf.summary())

    # Filtering summary
    print(f"\n  --- Filtering Summary ---")
    print(f"  Unfiltered: {len(consensus_preds)} cells")
    print(f"  @0.3: {filt_03.n_kept} kept ({filt_03.n_kept/len(consensus_preds)*100:.1f}%), "
          f"{filt_03.n_filtered} filtered")
    print(filt_03.summary())
    print(f"\n  @0.5: {filt_05.n_kept} kept ({filt_05.n_kept/len(consensus_preds)*100:.1f}%), "
          f"{filt_05.n_filtered} filtered")
    print(filt_05.summary())

    return {
        "tissue": tissue,
        "name": config["name"],
        "n_cells": len(consensus_preds),
        "confidence": consensus_conf.overall_score,
        "marker_score": consensus_conf.overall_marker_score,
        "spatial_coherence": consensus_conf.overall_spatial_coherence,
        "consensus_score": float(consensus_conf.overall_consensus_score) if not np.isnan(consensus_conf.overall_consensus_score) else None,
        "proportions_plausible": consensus_conf.proportion_plausible,
        "filt_03_kept": filt_03.n_kept,
        "filt_03_pct": filt_03.n_kept / len(consensus_preds) * 100,
        "filt_05_kept": filt_05.n_kept,
        "filt_05_pct": filt_05.n_kept / len(consensus_preds) * 100,
        "distribution": dict(Counter(consensus_preds.tolist())),
        "warnings": consensus_conf.warnings,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["lung", "kidney", "liver"])
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()

    tissue_results = []
    for ds_key in args.datasets:
        if ds_key not in DATASETS:
            logger.warning(f"Unknown dataset: {ds_key}")
            continue
        r = run_tissue(ds_key, sample_size=args.sample_size)
        if r:
            tissue_results.append(r)

    # ── Cross-tissue summary ─────────────────────────────────
    if tissue_results:
        print(f"\n{'='*90}")
        print(f"  CROSS-TISSUE SUMMARY (No Ground Truth)")
        print(f"{'='*90}")
        print(f"  {'Tissue':<15} {'Cells':>8} {'Conf':>6} {'Marker':>7} {'Spat':>6} {'Cons':>6} "
              f"{'Plaus':>6} {'@0.3%':>6} {'@0.5%':>6}  Distribution")
        print(f"  {'-'*85}")
        for r in tissue_results:
            cons_str = f"{r['consensus_score']:.3f}" if r['consensus_score'] else "  -"
            plaus = "Yes" if r["proportions_plausible"] else "NO"
            dist = r["distribution"]
            epi_pct = dist.get("Epithelial", 0) / r["n_cells"] * 100
            imm_pct = dist.get("Immune", 0) / r["n_cells"] * 100
            str_pct = dist.get("Stromal", 0) / r["n_cells"] * 100
            dist_str = f"E:{epi_pct:.0f}% I:{imm_pct:.0f}% S:{str_pct:.0f}%"
            print(f"  {r['name']:<15} {r['n_cells']:>8,} {r['confidence']:>6.3f} {r['marker_score']:>7.3f} "
                  f"{r['spatial_coherence']:>6.3f} {cons_str:>6} {plaus:>6} "
                  f"{r['filt_03_pct']:>5.1f}% {r['filt_05_pct']:>5.1f}%  {dist_str}")

        # Breast reference for comparison
        print(f"  {'-'*85}")
        print(f"  {'Breast (ref)':<15} {'159K':>8} {'0.713':>6} {'0.636':>7} {'0.578':>6} {'0.577':>6} "
              f"{'Yes':>6} {'98.8%':>6} {'85.5%':>6}  E:47% I:28% S:24%")
        print(f"{'='*90}")

        if any(not r["proportions_plausible"] for r in tissue_results):
            print("\n  WARNING: Some tissues have implausible proportions.")
            print("  This may indicate the breast-trained CellTypist model is not appropriate.")
            for r in tissue_results:
                if not r["proportions_plausible"]:
                    print(f"    {r['name']}: {', '.join(r['warnings'][:3])}")

    # Save JSON
    output_path = Path("/tmp/benchmark_nogt_tissues.json")
    output_path.write_text(json.dumps(tissue_results, indent=2, default=str))
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
