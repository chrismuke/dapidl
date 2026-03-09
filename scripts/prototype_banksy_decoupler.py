#!/usr/bin/env python3
"""
Prototype: BANKSY clustering + decoupler ORA annotation with PanglaoDB markers.

Compares three cluster-labeling strategies (no ground truth needed):
1. decoupler ORA with PanglaoDB markers
2. scanpy.tl.marker_gene_overlap with PanglaoDB markers
3. Our existing z-score marker enrichment (DEFAULT_MARKERS from sctype.py)

All evaluated against Janesick ground truth on breast rep1.

Usage:
    uv run python scripts/prototype_banksy_decoupler.py
    uv run python scripts/prototype_banksy_decoupler.py --sample-size 5000
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from loguru import logger
from scipy.sparse import csr_matrix, diags, issparse
from sklearn.metrics import accuracy_score, classification_report, f1_score

# ── Config ─────────────────────────────────────────────────────────
DATASET = {
    "h5_path": Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1/cell_feature_matrix.h5"),
    "cells_path": Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1/cells.parquet"),
    "gt_path": Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1/celltypes_ground_truth_rep1_supervised.xlsx"),
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
    "Stromal": "Stromal", "Stromal_&_T_Cell_Hybrid": "Stromal",
    "Endothelial": "Stromal",
}

# PanglaoDB cell type → broad category mapping
PANGLAO_BROAD_MAP = {
    # Epithelial
    "Epithelial cells": "Epithelial", "Basal cells": "Epithelial",
    "Luminal epithelial cells": "Epithelial", "Ductal cells": "Epithelial",
    "Myoepithelial cells": "Epithelial", "Alveolar cells": "Epithelial",
    "Keratinocytes": "Epithelial", "Clara cells": "Epithelial",
    "Goblet cells": "Epithelial", "Enterocytes": "Epithelial",
    "Tuft cells": "Epithelial", "Paneth cells": "Epithelial",
    # Immune
    "T cells": "Immune", "T memory cells": "Immune", "T helper cells": "Immune",
    "T regulatory cells": "Immune", "T cytotoxic cells": "Immune",
    "NK cells": "Immune", "NKT cells": "Immune",
    "B cells": "Immune", "B cells memory": "Immune",
    "Plasma cells": "Immune", "Plasmacytoid dendritic cells": "Immune",
    "Dendritic cells": "Immune", "Macrophages": "Immune",
    "Monocytes": "Immune", "Mast cells": "Immune",
    "Neutrophils": "Immune", "Eosinophils": "Immune", "Basophils": "Immune",
    "Langerhans cells": "Immune", "Kupffer cells": "Immune",
    "Microglia": "Immune", "Granulocytes": "Immune",
    # Stromal
    "Fibroblasts": "Stromal", "Myofibroblasts": "Stromal",
    "Endothelial cells": "Stromal", "Pericytes": "Stromal",
    "Smooth muscle cells": "Stromal", "Mesenchymal stem cells": "Stromal",
    "Adipocytes": "Stromal", "Stellate cells": "Stromal",
    "Lymphatic endothelial cells": "Stromal",
}

# Our existing scType DEFAULT_MARKERS → broad
SCTYPE_MARKERS = {
    "Epithelial": {
        "positive": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "MUC1", "KRT7"],
        "negative": ["VIM", "PTPRC", "PECAM1"],
    },
    "Immune": {
        "positive": ["PTPRC", "CD3D", "CD3E", "CD4", "CD8A", "CD14", "CD68",
                      "MS4A1", "CD19", "NKG7", "GNLY", "TPSAB1", "MS4A2"],
        "negative": ["EPCAM", "KRT8", "KRT18", "PECAM1"],
    },
    "Stromal": {
        "positive": ["VIM", "FAP", "PDGFRA", "PDGFRB", "ACTA2", "COL1A1",
                      "COL1A2", "PECAM1", "VWF", "CLDN5", "CDH5"],
        "negative": ["EPCAM", "PTPRC", "CD3D"],
    },
}


# ── Data Loading ───────────────────────────────────────────────────

def load_data(sample_size: int | None = None):
    """Load expression matrix, spatial coordinates, and ground truth."""
    logger.info("Loading Breast Rep1...")
    adata = sc.read_10x_h5(str(DATASET["h5_path"]))
    adata.var_names_make_unique()

    cells_df = pl.read_parquet(DATASET["cells_path"])
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

    gt_df = pd.read_excel(DATASET["gt_path"])
    gt_pl = pl.DataFrame({
        "cell_id": [str(b) for b in gt_df.iloc[:, 0]],
        "gt_fine": gt_df.iloc[:, 1].astype(str).tolist(),
    })
    gt_pl = gt_pl.with_columns(
        pl.col("gt_fine")
        .map_elements(lambda x: GT_BROAD_MAP.get(x, "Unknown"), return_dtype=pl.Utf8)
        .alias("gt_broad")
    ).filter(pl.col("gt_broad") != "Unknown")

    gt_cell_set = set(gt_pl["cell_id"].to_list())
    gt_mask = [i for i, c in enumerate(adata.obs_names) if c in gt_cell_set]
    adata = adata[gt_mask].copy()
    spatial_coords = spatial_coords[gt_mask]

    if sample_size and sample_size < adata.n_obs:
        idx = np.random.choice(adata.n_obs, sample_size, replace=False)
        adata = adata[idx].copy()
        spatial_coords = spatial_coords[idx]

    adata.layers["counts"] = adata.X.copy()

    n_genes = np.array((adata.X > 0).sum(axis=1)).flatten()
    keep_mask = n_genes >= 5
    n_filtered = (~keep_mask).sum()
    if n_filtered > 0:
        logger.info(f"Filtered {n_filtered} cells with <5 genes")
        adata = adata[keep_mask].copy()
        spatial_coords = spatial_coords[keep_mask]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    gt_map = dict(zip(gt_pl["cell_id"].to_list(), gt_pl["gt_broad"].to_list()))
    gt_labels = [gt_map.get(c, "Unknown") for c in adata.obs_names]

    logger.info(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    return adata, spatial_coords, gt_labels


# ── BANKSY Augmentation & Clustering ──────────────────────────────

def banksy_cluster(adata, spatial_coords, num_neighbours=10, resolution=1.0):
    """BANKSY augmentation + Leiden → returns adata_aug with 'leiden' in obs."""
    import logging as _logging
    _logging.getLogger("banksy").setLevel(_logging.WARNING)
    from banksy.main import generate_spatial_weights_fixed_nbrs

    logger.info(f"BANKSY: k={num_neighbours}, res={resolution}")

    W, _, _ = generate_spatial_weights_fixed_nbrs(
        spatial_coords.astype(np.float64), m=0,
        num_neighbours=num_neighbours, decay_type="scaled_gaussian",
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
    sc.tl.leiden(adata_aug, resolution=resolution, flavor="igraph", n_iterations=2, directed=False)

    n_clusters = len(adata_aug.obs["leiden"].unique())
    logger.info(f"BANKSY produced {n_clusters} clusters")

    # Copy cluster labels back to original adata for DEG analysis
    adata.obs["banksy_leiden"] = adata_aug.obs["leiden"].values

    return adata, adata_aug


# ── Marker Retrieval ──────────────────────────────────────────────

def get_panglao_markers(adata):
    """Pull PanglaoDB markers via decoupler and filter to panel genes."""
    import decoupler as dc

    logger.info("Pulling PanglaoDB markers via decoupler/OmniPath...")
    markers = dc.op.resource("PanglaoDB", organism="human")

    # Filter: canonical markers, good sensitivity
    markers = markers[
        markers["canonical_marker"].astype(bool)
        & (markers["human_sensitivity"].astype(float) > 0.5)
    ]
    markers = markers[~markers.duplicated(["cell_type", "genesymbol"])]

    # Filter to genes in our Xenium panel
    panel_genes = set(adata.var_names)
    markers_in_panel = markers[markers["genesymbol"].isin(panel_genes)]

    logger.info(f"PanglaoDB: {len(markers)} total markers → {len(markers_in_panel)} in Xenium panel")
    logger.info(f"Cell types with markers in panel: {markers_in_panel['cell_type'].nunique()}")

    # Show which cell types have markers
    type_counts = markers_in_panel.groupby("cell_type")["genesymbol"].count().sort_values(ascending=False)
    logger.info(f"Top cell types by marker count:\n{type_counts.head(20).to_string()}")

    return markers_in_panel


def panglao_to_broad_dict(markers_df):
    """Convert PanglaoDB markers DataFrame to {broad_category: set(genes)} for overlap scoring."""
    broad_markers = {"Epithelial": set(), "Immune": set(), "Stromal": set()}

    for _, row in markers_df.iterrows():
        ct = row["cell_type"]
        gene = row["genesymbol"]
        broad = PANGLAO_BROAD_MAP.get(ct)
        if broad:
            broad_markers[broad].add(gene)

    for cat, genes in broad_markers.items():
        logger.info(f"  {cat}: {len(genes)} markers — {sorted(genes)[:10]}...")

    return broad_markers


# ── Cluster Labeling Methods ──────────────────────────────────────

def label_clusters_oracle(adata, gt_labels):
    """Oracle: majority GT label per cluster."""
    clusters = adata.obs["banksy_leiden"].tolist()
    cluster_labels = {}
    for cid in set(clusters):
        gt_in_cluster = [gt for c, gt in zip(clusters, gt_labels) if c == cid and gt != "Unknown"]
        if gt_in_cluster:
            cluster_labels[cid] = Counter(gt_in_cluster).most_common(1)[0][0]
        else:
            cluster_labels[cid] = "Unknown"
    return [cluster_labels.get(c, "Unknown") for c in clusters]


def label_clusters_decoupler_ora(adata, panglao_markers_df):
    """decoupler ORA: over-representation analysis of cluster DEGs against PanglaoDB markers."""
    import decoupler as dc

    # Map PanglaoDB cell types to broad categories for enrichment
    markers_with_broad = panglao_markers_df.copy()
    markers_with_broad["broad"] = markers_with_broad["cell_type"].map(PANGLAO_BROAD_MAP)
    markers_broad = markers_with_broad.dropna(subset=["broad"])

    # Create gene set for broad categories
    broad_net = markers_broad[["broad", "genesymbol"]].drop_duplicates()
    broad_net.columns = ["source", "target"]
    broad_net["weight"] = 1.0

    logger.info(f"Broad gene set: {broad_net.groupby('source')['target'].count().to_dict()}")

    # Run ORA per cell using the broad marker sets
    logger.info("Running decoupler ORA on expression matrix...")
    adata_ora = adata.copy()
    dc.mt.ora(
        data=adata_ora,
        net=broad_net,
        tmin=3,
        verbose=True,
    )

    # Get per-cell ORA scores
    ora_scores = adata_ora.obsm["ora_estimate"]
    logger.info(f"ORA scores shape: {ora_scores.shape}, columns: {list(ora_scores.columns)}")

    # Assign each cell to highest-scoring broad category
    predictions = ora_scores.idxmax(axis=1).tolist()

    dist = Counter(predictions)
    logger.info(f"decoupler ORA predictions: {dict(dist)}")

    return predictions


def label_clusters_marker_overlap(adata, panglao_broad_markers):
    """scanpy.tl.marker_gene_overlap: Jaccard similarity between cluster DEGs and reference markers."""
    adata_degs = adata.copy()
    sc.tl.rank_genes_groups(adata_degs, groupby="banksy_leiden", method="wilcoxon")

    # marker_gene_overlap needs reference_markers as {cell_type: set(genes)}
    overlap_df = sc.tl.marker_gene_overlap(
        adata_degs,
        reference_markers=panglao_broad_markers,
        method="jaccard",
        top_n_markers=50,
    )

    logger.info(f"Marker overlap (Jaccard):\n{overlap_df.to_string()}")

    # overlap_df: rows=cell types, cols=clusters → idxmax(axis=0) gives best cell type per cluster
    cluster_to_type = overlap_df.idxmax(axis=0).to_dict()
    logger.info(f"Cluster assignments: {cluster_to_type}")

    clusters = adata.obs["banksy_leiden"].tolist()
    return [cluster_to_type.get(c, "Unknown") for c in clusters]


def label_clusters_zscore(adata, markers_dict=None):
    """Z-score marker enrichment (our existing approach from the benchmark)."""
    if markers_dict is None:
        markers_dict = SCTYPE_MARKERS

    clusters = adata.obs["banksy_leiden"].tolist()
    cluster_ids = sorted(set(clusters))

    X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
    gene_names = list(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Global stats
    global_mean = X.mean(axis=0)
    global_std = X.std(axis=0)
    global_std[global_std == 0] = 1.0

    cluster_labels = {}
    for cid in cluster_ids:
        cell_mask = [i for i, c in enumerate(clusters) if c == cid]
        cluster_expr = X[cell_mask]
        cluster_mean = cluster_expr.mean(axis=0)

        best_type = "Unknown"
        best_score = -np.inf

        for cell_type, marker_info in markers_dict.items():
            pos_genes = [g for g in marker_info["positive"] if g in gene_to_idx]
            neg_genes = [g for g in marker_info.get("negative", []) if g in gene_to_idx]

            if not pos_genes:
                continue

            # Z-score enrichment
            pos_zscores = [(cluster_mean[gene_to_idx[g]] - global_mean[gene_to_idx[g]]) / global_std[gene_to_idx[g]]
                           for g in pos_genes]
            neg_zscores = [(cluster_mean[gene_to_idx[g]] - global_mean[gene_to_idx[g]]) / global_std[gene_to_idx[g]]
                           for g in neg_genes]

            score = np.mean(pos_zscores)
            if neg_zscores:
                score -= 0.5 * np.mean(neg_zscores)

            if score > best_score:
                best_score = score
                best_type = cell_type

        cluster_labels[cid] = best_type

    logger.info(f"Z-score cluster assignments: {cluster_labels}")
    return [cluster_labels.get(c, "Unknown") for c in clusters]


def label_clusters_zscore_panglao(adata, panglao_broad_markers):
    """Z-score enrichment using PanglaoDB markers (same algorithm, different markers)."""
    # Convert PanglaoDB broad markers to scType-style format
    markers_dict = {}
    for broad, genes in panglao_broad_markers.items():
        markers_dict[broad] = {
            "positive": list(genes),
            "negative": [],
        }

    return label_clusters_zscore(adata, markers_dict)


# ── Evaluation ────────────────────────────────────────────────────

def evaluate(gt_labels, predictions, method_name):
    """Evaluate predictions against ground truth."""
    pairs = [(t, p) for t, p in zip(gt_labels, predictions) if t != "Unknown" and p != "Unknown"]
    if not pairs:
        logger.warning(f"{method_name}: No valid pairs!")
        return None

    true, pred = zip(*pairs)
    acc = accuracy_score(true, pred)
    macro_f1 = f1_score(true, pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(true, pred, average="weighted", zero_division=0)

    report = classification_report(true, pred, output_dict=True, zero_division=0)
    per_class = {}
    for label in sorted(set(true) | set(pred)):
        if label in report:
            per_class[label] = round(report[label]["f1-score"], 3)

    logger.info(f"\n{'─'*60}")
    logger.info(f"  {method_name}")
    logger.info(f"  Accuracy: {acc:.4f}  Macro F1: {macro_f1:.4f}  Weighted F1: {weighted_f1:.4f}")
    logger.info(f"  Per-class: {per_class}")
    logger.info(f"  n_cells: {len(pairs)} (of {len(gt_labels)})")

    # Distribution
    gt_dist = Counter(true)
    pred_dist = Counter(pred)
    logger.info(f"  GT dist: {dict(gt_dist)}")
    logger.info(f"  Pred dist: {dict(pred_dist)}")

    return {
        "method": method_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_f1": per_class,
        "n_cells": len(pairs),
    }


# ── Main ──────────────────────────────────────────────────────────

def main(sample_size=None):
    logger.info("=" * 70)
    logger.info("  BANKSY + decoupler Prototype")
    logger.info("=" * 70)

    # 1. Load data
    adata, spatial_coords, gt_labels = load_data(sample_size)

    # 2. BANKSY clustering (k=10, res=1.0 — best from previous benchmark)
    t0 = time.time()
    adata, adata_aug = banksy_cluster(adata, spatial_coords, num_neighbours=10, resolution=1.0)
    banksy_time = time.time() - t0
    logger.info(f"BANKSY clustering took {banksy_time:.1f}s")

    # 3. Get PanglaoDB markers
    panglao_markers = get_panglao_markers(adata)
    panglao_broad = panglao_to_broad_dict(panglao_markers)

    # 4. Run all labeling methods
    results = []

    # 4a. Oracle (ceiling)
    logger.info("\n>>> Method 1: Oracle (GT majority vote per cluster)")
    t0 = time.time()
    oracle_preds = label_clusters_oracle(adata, gt_labels)
    r = evaluate(gt_labels, oracle_preds, "Oracle (BANKSY k10 r1.0)")
    if r:
        r["runtime"] = round(time.time() - t0, 1)
        results.append(r)

    # 4b. decoupler ORA
    logger.info("\n>>> Method 2: decoupler ORA (PanglaoDB markers)")
    t0 = time.time()
    try:
        ora_preds = label_clusters_decoupler_ora(adata, panglao_markers)
        r = evaluate(gt_labels, ora_preds, "decoupler ORA (PanglaoDB)")
        if r:
            r["runtime"] = round(time.time() - t0, 1)
            results.append(r)
    except Exception as e:
        logger.error(f"decoupler ORA failed: {e}")
        import traceback; traceback.print_exc()

    # 4c. scanpy marker_gene_overlap
    logger.info("\n>>> Method 3: scanpy marker_gene_overlap (PanglaoDB markers)")
    t0 = time.time()
    try:
        overlap_preds = label_clusters_marker_overlap(adata, panglao_broad)
        r = evaluate(gt_labels, overlap_preds, "marker_gene_overlap (PanglaoDB)")
        if r:
            r["runtime"] = round(time.time() - t0, 1)
            results.append(r)
    except Exception as e:
        logger.error(f"marker_gene_overlap failed: {e}")
        import traceback; traceback.print_exc()

    # 4d. Z-score with our DEFAULT_MARKERS
    logger.info("\n>>> Method 4: Z-score enrichment (DEFAULT_MARKERS)")
    t0 = time.time()
    zscore_default_preds = label_clusters_zscore(adata)
    r = evaluate(gt_labels, zscore_default_preds, "Z-score (DEFAULT_MARKERS)")
    if r:
        r["runtime"] = round(time.time() - t0, 1)
        results.append(r)

    # 4e. Z-score with PanglaoDB markers
    logger.info("\n>>> Method 5: Z-score enrichment (PanglaoDB markers)")
    t0 = time.time()
    zscore_panglao_preds = label_clusters_zscore_panglao(adata, panglao_broad)
    r = evaluate(gt_labels, zscore_panglao_preds, "Z-score (PanglaoDB)")
    if r:
        r["runtime"] = round(time.time() - t0, 1)
        results.append(r)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  RESULTS SUMMARY — BANKSY k=10 r=1.0 + cluster labeling")
    print("=" * 90)
    print(f"{'Method':<40} {'Acc':>7} {'F1_M':>7} {'F1_W':>7} {'Epi':>7} {'Imm':>7} {'Str':>7}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: -x["macro_f1"]):
        epi = r["per_class_f1"].get("Epithelial", 0)
        imm = r["per_class_f1"].get("Immune", 0)
        stro = r["per_class_f1"].get("Stromal", 0)
        print(f"{r['method']:<40} {r['accuracy']:>7.3f} {r['macro_f1']:>7.3f} {r['weighted_f1']:>7.3f} "
              f"{epi:>7.3f} {imm:>7.3f} {stro:>7.3f}")

    # Save results
    out_path = Path("/tmp/banksy_decoupler_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", "-s", type=int, default=None)
    args = parser.parse_args()
    main(args.sample_size)
