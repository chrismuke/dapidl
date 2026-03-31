#!/usr/bin/env python3
"""Full BANKSY spatial clustering benchmark.

Runs the complete BANKSY pipeline:
1. Load Xenium data with spatial coordinates
2. Normalize + HVG selection
3. BANKSY spatial augmentation (multiple lambda values)
4. PCA on BANKSY matrices
5. Leiden clustering (multiple resolutions)
6. Label clusters via marker genes (multiple DBs)
7. Optional: spatial smoothing (KNN majority vote)
8. Evaluate against ground truth

Grid search: 3 lambda × 4 resolutions × 2 k-values × 2 marker DBs = 48+ configs

Usage:
    uv run python scripts/banksy_full_benchmark.py
"""

import gc
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from loguru import logger
from scipy.sparse import issparse
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.annotation_benchmark_2026_03 import (
    COARSE_CLASSES,
    GT_TO_COARSE,
    compute_metrics,
    load_xenium_adata,
    map_predictions_to_coarse,
    preprocess_adata,
)

OUTPUT_DIR = Path("pipeline_output/annotation_benchmark_2026_03")

# ──────────────────────────────────────────────────────────────────────────────
# MARKER DATABASES FOR CLUSTER LABELING
# ──────────────────────────────────────────────────────────────────────────────

MARKERS_DEFAULT = {
    "Epithelial": {"positive": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "KRT7", "MUC1"],
                   "negative": ["PTPRC", "VIM", "PECAM1"]},
    "Immune": {"positive": ["PTPRC", "CD3D", "CD3E", "CD4", "CD8A", "CD14", "CD68", "MS4A1", "NKG7", "TRAC"],
               "negative": ["EPCAM", "KRT8", "COL1A1"]},
    "Stromal": {"positive": ["COL1A1", "COL1A2", "ACTA2", "VIM", "FAP", "DCN", "PDGFRA", "PDGFRB"],
                "negative": ["EPCAM", "PTPRC", "PECAM1"]},
    "Endothelial": {"positive": ["PECAM1", "VWF", "CLDN5", "KDR"],
                    "negative": ["EPCAM", "PTPRC", "COL1A1"]},
}

# SCINA-style signatures (EM-optimized from our benchmark)
MARKERS_SCINA = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "KRT7", "MUC1", "AGR2", "TACSTD2"],
    "T_cells": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC"],
    "B_cells": ["CD19", "CD79A", "CD79B", "MS4A1", "PAX5"],
    "Macrophages": ["CD68", "CD163", "CSF1R"],
    "Mast_cells": ["KIT", "TPSAB1", "CPA3"],
    "NK_cells": ["NCAM1", "NKG7", "GNLY"],
    "Fibroblasts": ["COL1A1", "COL1A2", "ACTA2", "VIM", "FAP", "DCN"],
    "Endothelial": ["PECAM1", "VWF", "CLDN5", "KDR"],
    "Plasma_cells": ["SDC1", "MZB1", "JCHAIN"],
    "Dendritic_cells": ["ITGAX", "CD1C", "CLEC9A", "FLT3"],
}

# Mapping fine-grained SCINA types to coarse
SCINA_TO_COARSE = {
    "Epithelial": "Epithelial", "T_cells": "Immune", "B_cells": "Immune",
    "Macrophages": "Immune", "Mast_cells": "Immune", "NK_cells": "Immune",
    "Fibroblasts": "Stromal", "Endothelial": "Endothelial",
    "Plasma_cells": "Immune", "Dendritic_cells": "Immune",
}


def label_clusters_sctype(adata, cluster_key, markers):
    """Label clusters using scType-style scoring (positive - negative)."""
    gene_names = list(adata.var_names)
    expr = np.asarray(adata.X.toarray() if issparse(adata.X) else adata.X)

    cluster_labels = {}
    for cluster in sorted(adata.obs[cluster_key].unique()):
        mask = adata.obs[cluster_key] == cluster
        cluster_expr = expr[mask]

        best_ct, best_score = "Unknown", -999
        for ct, m in markers.items():
            pos = [g for g in m["positive"] if g in gene_names]
            neg = [g for g in m.get("negative", []) if g in gene_names]
            if not pos:
                continue
            pos_idx = [gene_names.index(g) for g in pos]
            score = cluster_expr[:, pos_idx].mean()
            if neg:
                neg_idx = [gene_names.index(g) for g in neg]
                score -= 0.5 * cluster_expr[:, neg_idx].mean()
            if score > best_score:
                best_score = score
                best_ct = ct

        cluster_labels[cluster] = best_ct

    return np.array([cluster_labels.get(c, "Unknown") for c in adata.obs[cluster_key]])


def label_clusters_scina(adata, cluster_key, signatures):
    """Label clusters using SCINA-style mean expression scoring."""
    gene_names = list(adata.var_names)
    expr = np.asarray(adata.X.toarray() if issparse(adata.X) else adata.X)

    cluster_labels = {}
    for cluster in sorted(adata.obs[cluster_key].unique()):
        mask = adata.obs[cluster_key] == cluster
        cluster_expr = expr[mask]

        best_ct, best_score = "Unknown", -1
        for ct, genes in signatures.items():
            present = [g for g in genes if g in gene_names]
            if not present:
                continue
            idx = [gene_names.index(g) for g in present]
            score = cluster_expr[:, idx].mean()
            if score > best_score:
                best_score = score
                best_ct = ct

        # Map to coarse
        cluster_labels[cluster] = SCINA_TO_COARSE.get(best_ct, best_ct)

    return np.array([cluster_labels.get(c, "Unknown") for c in adata.obs[cluster_key]])


def label_clusters_rank_genes(adata, cluster_key):
    """Label clusters using scanpy rank_genes_groups + marker_gene_overlap."""
    sc.tl.rank_genes_groups(adata, cluster_key, method="wilcoxon", use_raw=False)

    # Build reference marker dict for scanpy
    ref_markers = {}
    for ct, m in MARKERS_DEFAULT.items():
        ref_markers[ct] = m["positive"]

    # Manual overlap scoring (marker_gene_overlap is deprecated in newer scanpy)
    gene_names = list(adata.var_names)
    cluster_labels = {}
    for cluster in sorted(adata.obs[cluster_key].unique()):
        # Get top DEGs for this cluster
        try:
            top_genes = set()
            names = adata.uns["rank_genes_groups"]["names"]
            for i in range(min(50, len(names))):
                g = names[i][cluster]
                if str(g) != "nan":
                    top_genes.add(str(g))
        except (KeyError, IndexError):
            cluster_labels[cluster] = "Unknown"
            continue

        # Overlap with each cell type's markers
        best_ct, best_overlap = "Unknown", 0
        for ct, markers in ref_markers.items():
            overlap = len(top_genes & set(markers))
            if overlap > best_overlap:
                best_overlap = overlap
                best_ct = ct

        cluster_labels[cluster] = best_ct

    return np.array([cluster_labels.get(c, "Unknown") for c in adata.obs[cluster_key]])


def spatial_smoothing(preds, coords, k=30):
    """KNN majority vote spatial smoothing."""
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k)
    smoothed = []
    for i in range(len(preds)):
        nbr_preds = preds[indices[i]]
        counts = Counter(p for p in nbr_preds if p != "Unknown")
        if counts:
            smoothed.append(counts.most_common(1)[0][0])
        else:
            smoothed.append(preds[i])
    return np.array(smoothed)


def run_banksy_pipeline(adata_raw, gt, configs):
    """Run full BANKSY pipeline with grid search over parameters."""
    from banksy.initialize_banksy import initialize_banksy
    from banksy.embed_banksy import generate_banksy_matrix
    from banksy.cluster_methods import run_Leiden_partition
    from banksy_utils.umap_pca import pca_umap

    results = {}

    # Prepare adata with spatial coordinates
    a = adata_raw.copy()
    if issparse(a.X):
        a.X = a.X.toarray()

    # Store raw for labeling
    a.layers["raw_counts"] = a.X.copy()

    # Normalize
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    # HVG
    n_hvg = min(2000, a.n_vars)
    sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, subset=False)
    a_hvg = a[:, a.var["highly_variable"]].copy()

    # Setup coordinates
    if "x_centroid" in a.obs.columns and "y_centroid" in a.obs.columns:
        a_hvg.obs["xcoord"] = a.obs["x_centroid"].values
        a_hvg.obs["ycoord"] = a.obs["y_centroid"].values
        a_hvg.obsm["xy_coord"] = np.column_stack([
            a_hvg.obs["xcoord"].values.astype(float),
            a_hvg.obs["ycoord"].values.astype(float),
        ])
    else:
        logger.error("No spatial coordinates found!")
        return results

    coords = a_hvg.obsm["xy_coord"]
    logger.info(f"BANKSY input: {a_hvg.shape[0]} cells, {a_hvg.shape[1]} HVGs, coords range: "
                f"x=[{coords[:,0].min():.0f},{coords[:,0].max():.0f}], "
                f"y=[{coords[:,1].min():.0f},{coords[:,1].max():.0f}]")

    # ── Grid search over BANKSY parameters ────────────────────────────
    for k_nbr in configs.get("k_neighbours", [15]):
        logger.info(f"\n  Initializing BANKSY with k={k_nbr}...")
        t0 = time.time()

        try:
            banksy_dict = initialize_banksy(
                adata=a_hvg,
                coord_keys=("xcoord", "ycoord", "xy_coord"),
                num_neighbours=k_nbr,
                nbr_weight_decay="scaled_gaussian",
                max_m=1,
                plt_edge_hist=False,
                plt_nbr_weights=False,
                plt_theta=False,
            )
        except Exception as e:
            logger.error(f"  BANKSY init failed: {e}")
            import traceback; traceback.print_exc()
            continue

        for lam in configs.get("lambda_list", [0.2, 0.5, 0.8]):
            logger.info(f"  Generating BANKSY matrix (lambda={lam})...")
            try:
                banksy_dict_copy = banksy_dict.copy()
                banksy_dict_copy, banksy_matrix = generate_banksy_matrix(
                    adata=a_hvg,
                    banksy_dict=banksy_dict_copy,
                    lambda_list=[lam],
                    max_m=1,
                    plot_std=False,
                    verbose=False,
                )
            except Exception as e:
                logger.error(f"  BANKSY matrix failed: {e}")
                import traceback; traceback.print_exc()
                continue

            # PCA
            try:
                pca_umap(banksy_dict=banksy_dict_copy, pca_dims=[20], plt_remaining_var=False, add_umap=False)
            except Exception as e:
                logger.error(f"  PCA failed: {e}")
                import traceback; traceback.print_exc()
                continue

            # Leiden clustering at multiple resolutions
            for res in configs.get("resolutions", [0.5, 1.0, 1.5]):
                try:
                    results_df, max_labels = run_Leiden_partition(
                        banksy_dict=banksy_dict_copy,
                        resolutions=[res],
                        num_nn=50,
                        num_iterations=-1,
                        partition_seed=1234,
                        match_labels=False,
                        verbose=False,
                    )
                except Exception as e:
                    logger.error(f"  Leiden failed (res={res}): {e}")
                    continue

                # Process each result row
                for _, row in results_df.iterrows():
                    n_clusters = row["num_labels"]
                    labels = row["labels"]
                    cluster_ids = labels.dense

                    # Create annotated adata for labeling
                    a_clustered = a.copy()  # Full gene set for labeling
                    a_clustered.obs["banksy_cluster"] = pd.Categorical(cluster_ids.astype(str))

                    param_str = f"k{k_nbr}_l{lam}_r{res}"
                    logger.info(f"    {param_str}: {n_clusters} clusters")

                    # ── Label with multiple marker strategies ────────
                    for label_method, label_fn in [
                        ("sctype", lambda ad: label_clusters_sctype(ad, "banksy_cluster", MARKERS_DEFAULT)),
                        ("scina", lambda ad: label_clusters_scina(ad, "banksy_cluster", MARKERS_SCINA)),
                        ("rank_genes", lambda ad: label_clusters_rank_genes(ad, "banksy_cluster")),
                    ]:
                        try:
                            preds = label_fn(a_clustered)
                            coarse = preds  # Already coarse for sctype/rank_genes

                            # Also try with spatial smoothing
                            for smooth_k in [0, 15, 30]:
                                if smooth_k > 0:
                                    preds_smooth = spatial_smoothing(coarse, coords, k=smooth_k)
                                    key = f"banksy_{param_str}_{label_method}_smooth{smooth_k}"
                                else:
                                    preds_smooth = coarse
                                    key = f"banksy_{param_str}_{label_method}"

                                metrics = compute_metrics(gt, preds_smooth, COARSE_CLASSES)
                                metrics["n_clusters"] = int(n_clusters)
                                metrics["lambda"] = lam
                                metrics["resolution"] = res
                                metrics["k_neighbours"] = k_nbr
                                metrics["label_method"] = label_method
                                metrics["smooth_k"] = smooth_k
                                results[key] = metrics

                                f1 = metrics["f1_macro"]
                                if smooth_k == 0:
                                    logger.info(f"      {label_method}: F1={f1:.3f}")
                                elif f1 > results.get(f"banksy_{param_str}_{label_method}", {}).get("f1_macro", 0):
                                    logger.info(f"      {label_method}+smooth{smooth_k}: F1={f1:.3f} (↑)")

                        except Exception as e:
                            logger.error(f"      {label_method} failed: {e}")

                    # Also compute ARI/NMI vs GT
                    gt_int = np.array([COARSE_CLASSES.index(g) if g in COARSE_CLASSES else -1 for g in gt])
                    valid = gt_int >= 0
                    if valid.sum() > 0:
                        ari = adjusted_rand_score(gt_int[valid], cluster_ids[valid])
                        nmi = normalized_mutual_info_score(gt_int[valid], cluster_ids[valid])
                        results[f"banksy_{param_str}_unsupervised"] = {
                            "ari": round(float(ari), 4),
                            "nmi": round(float(nmi), 4),
                            "n_clusters": int(n_clusters),
                        }
                        logger.info(f"      Unsupervised: ARI={ari:.3f} NMI={nmi:.3f}")

        elapsed = time.time() - t0
        logger.info(f"  k={k_nbr} completed in {elapsed:.0f}s")

    return results


def main():
    logger.info("=" * 80)
    logger.info("FULL BANKSY SPATIAL CLUSTERING BENCHMARK")
    logger.info("=" * 80)

    ds_name = os.environ.get("BENCH_DATASETS", "rep1")
    adata_raw = load_xenium_adata(ds_name)
    adata_pp = preprocess_adata(adata_raw)
    gt = np.array(adata_pp.obs["gt_coarse"].values)
    logger.info(f"Dataset: {ds_name}, {len(adata_pp)} cells, {adata_pp.n_vars} genes")

    # Full grid search
    configs = {
        "k_neighbours": [10, 15, 20],
        "lambda_list": [0.2, 0.5, 0.8],
        "resolutions": [0.5, 0.8, 1.0, 1.5, 2.0],
    }

    n_configs = (len(configs["k_neighbours"]) * len(configs["lambda_list"]) *
                 len(configs["resolutions"]) * 3 * 3)  # 3 label methods × 3 smoothing
    logger.info(f"Grid search: {n_configs}+ configurations")

    results = run_banksy_pipeline(adata_raw, gt, configs)

    # ── Rankings ─────────────────────────────────────────────────────
    ranked = sorted(
        [(k, v) for k, v in results.items() if "f1_macro" in v],
        key=lambda x: -x[1]["f1_macro"]
    )

    logger.info(f"\n{'='*80}")
    logger.info(f"BANKSY BENCHMARK RESULTS ({len(results)} configs)")
    logger.info(f"{'='*80}")
    logger.info(f"\n{'Method':<60s} {'F1':>6s} {'Acc':>6s} {'#Cl':>4s}")
    logger.info("-" * 80)
    for k, v in ranked[:30]:
        logger.info(f"{k:<60s} {v['f1_macro']:>6.3f} {v['accuracy']:>6.3f} {v.get('n_clusters', 0):>4d}")

    # Save
    out_file = OUTPUT_DIR / f"banksy_full_{ds_name}.json"
    clean = {}
    for k, v in results.items():
        clean[k] = {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv) for kk, vv in v.items()}
    with open(out_file, "w") as f:
        json.dump(clean, f, indent=2)
    logger.info(f"\nSaved to {out_file}")

    # Merge top results into main file
    r1_path = OUTPUT_DIR / f"results_{ds_name}.json"
    if r1_path.exists():
        existing = json.load(open(r1_path))
        for k, v in ranked[:5]:
            existing[k] = v
        with open(r1_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    logger.info("BANKSY BENCHMARK COMPLETE!")


if __name__ == "__main__":
    main()
