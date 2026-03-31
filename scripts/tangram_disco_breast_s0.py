"""
STHELAR Tangram pipeline: annotate breast_s0 using DISCO breast atlas.

Steps:
1. Load DISCO breast atlas (raw counts) and STHELAR spatial data (raw counts from layers['count'])
2. Run Tangram via sopa (bag_size=10000, GPU-accelerated)
3. Leiden clustering refinement (res 0.2, 0.4, 0.6)
4. DGE-based cluster labeling using Tangram majority vote
5. Map to STHELAR 8 categories
6. Evaluate against GT (final_label_combined)
"""

import json
import logging
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("/mnt/work/git/dapidl/pipeline_output/sthelar_pipeline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DISCO cell type -> STHELAR category mapping
# ============================================================
DISCO_TO_STHELAR = {
    # Epithelial
    "SFN mammary luminal progenitor": "Epithelial",
    "KRT6B mammary basal cell": "Epithelial",
    "KRT17 mammary luminal cell": "Epithelial",
    "PIP mammary luminal cell": "Epithelial",
    "SAA2 mammary luminal progenitor": "Epithelial",
    "CXCL14 mammary basal cell": "Epithelial",
    "Secretoglobin mammary luminal progenitor": "Epithelial",
    "SCGB3A1 mammary luminal progenitor": "Epithelial",
    "Secretoglobin mammary luminal cell": "Epithelial",
    "CCSER1 mammary basal cell": "Epithelial",
    "Cycling mammary luminal progenitor": "Epithelial",
    "Lactocyte": "Epithelial",
    # Fibroblast / Myofibroblast
    "APOD+PTGDS+ fibroblast": "Fibroblast_Myofibroblast",
    "CFD+MGP+ fibroblast": "Fibroblast_Myofibroblast",
    "CDH19+LAMA2+ fibroblast": "Fibroblast_Myofibroblast",
    "MFAP5+IGFBP6+ fibroblast": "Fibroblast_Myofibroblast",
    "GPC3+ fibroblast": "Fibroblast_Myofibroblast",
    "BNC2+ZFPM2+ fibroblast": "Fibroblast_Myofibroblast",
    # Blood vessel (endothelial + pericyte + smooth muscle)
    "Capillary EC": "Blood_vessel",
    "Venous EC": "Blood_vessel",
    "Arterial EC": "Blood_vessel",
    "Lymphatic EC": "Blood_vessel",
    "Pericyte": "Blood_vessel",
    "CXCL+ pericyte": "Blood_vessel",
    "CCL19/21 pericyte": "Blood_vessel",
    "Vascular smooth muscle cell": "Blood_vessel",
    "CREB+MT1A+ vascular smooth muscle cell": "Blood_vessel",
    # T / NK cells
    "CD4 T cell": "T_NK",
    "GZMB CD8 T cell": "T_NK",
    "GZMK CD8 T cell": "T_NK",
    "Treg cell": "T_NK",
    "NK cell": "T_NK",
    "ILC": "T_NK",
    # B / Plasma cells
    "B cell": "B_Plasma",
    "Plasma cell": "B_Plasma",
    # Myeloid
    "M1 macrophage": "Myeloid",
    "Macrophage": "Myeloid",
    "LYVE1 macrophage": "Myeloid",
    "Monocyte": "Myeloid",
    "Dendritic cell": "Myeloid",
    # Specialized
    "Mast cell": "Specialized",
    "pDC": "Myeloid",
}

STHELAR_CATEGORIES = [
    "Epithelial",
    "Fibroblast_Myofibroblast",
    "Blood_vessel",
    "T_NK",
    "B_Plasma",
    "Myeloid",
    "Specialized",
    "Less10",
]


def load_data():
    """Load DISCO reference and STHELAR spatial data."""
    log.info("Loading DISCO breast atlas...")
    t0 = time.time()
    ref = sc.read_h5ad("/mnt/work/datasets/DISCO/disco_breast_v2.1.h5ad")
    log.info(f"  Reference: {ref.n_obs} cells, {ref.n_vars} genes (loaded in {time.time()-t0:.1f}s)")

    log.info("Loading STHELAR breast_s0 spatial data...")
    t0 = time.time()
    adata = ad.read_zarr(
        "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s0.zarr/"
        "sdata_breast_s0.zarr/tables/table_cells"
    )
    log.info(f"  Spatial: {adata.n_obs} cells, {adata.n_vars} genes (loaded in {time.time()-t0:.1f}s)")

    # Swap X to raw counts from 'count' layer
    log.info("Setting spatial X to raw counts from layers['count']...")
    adata.X = adata.layers["count"].copy()
    if sp.issparse(adata.X):
        adata.X = adata.X.astype(np.float32)

    # Load GT labels from table_combined
    log.info("Loading ground truth labels from table_combined...")
    gt = ad.read_zarr(
        "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s0.zarr/"
        "sdata_breast_s0.zarr/tables/table_combined"
    )
    adata.obs["gt_label"] = gt.obs["final_label_combined"].values

    # Verify raw counts
    if sp.issparse(adata.X):
        xmax = adata.X.data.max()
    else:
        xmax = adata.X.max()
    log.info(f"  Spatial X max: {xmax} (should be integer-valued raw counts)")

    return ref, adata


def run_tangram(adata, ref):
    """Run Tangram annotation via sopa."""
    from sopa.utils.annotation import tangram_annotate

    log.info("=" * 60)
    log.info("STEP 1: Running Tangram annotation")
    log.info(f"  Reference cell types: {ref.obs['cell_type'].nunique()}")
    log.info(f"  Spatial cells: {adata.n_obs}")
    log.info(f"  Gene overlap: {len(set(ref.var_names) & set(adata.var_names))}")
    log.info("=" * 60)

    t0 = time.time()
    tangram_annotate(
        adata,
        ref,
        cell_type_key="cell_type",
        bag_size=10_000,
        max_obs_reference=10_000,
        density_prior="rna_count_based",
        clip_percentile=0.95,
    )
    elapsed = time.time() - t0
    log.info(f"Tangram completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Check what Tangram produced
    tangram_key = "cell_type"
    if tangram_key in adata.obs.columns:
        log.info(f"Tangram cell_type predictions:")
        log.info(f"\n{adata.obs[tangram_key].value_counts()}")
    else:
        log.warning(f"Expected '{tangram_key}' in adata.obs, got: {list(adata.obs.columns)}")
        # Check obsm for probability matrices
        for key in adata.obsm:
            if "tangram" in key.lower():
                log.info(f"  obsm['{key}']: {adata.obsm[key].shape}")

    # Save intermediate
    log.info("Saving Tangram predictions...")
    tangram_results = {}
    if tangram_key in adata.obs.columns:
        tangram_results["tangram_raw_predictions"] = adata.obs[tangram_key].value_counts().to_dict()
    tangram_results["elapsed_seconds"] = elapsed
    with open(OUTPUT_DIR / "tangram_intermediate.json", "w") as f:
        json.dump(tangram_results, f, indent=2)

    return adata, elapsed


def leiden_refinement(adata):
    """
    Leiden clustering refinement following STHELAR protocol:
    - Filter cells <10 transcripts
    - Normalize, log1p, scale, PCA, neighbors
    - Leiden at multiple resolutions
    - DGE per cluster
    - Label clusters using Tangram majority vote
    """
    log.info("=" * 60)
    log.info("STEP 2: Leiden clustering refinement")
    log.info("=" * 60)

    t0 = time.time()

    # Work on a copy for clustering (keep original X as raw counts)
    adata_proc = adata.copy()

    # Filter cells with <10 transcripts
    if "transcript_counts" in adata_proc.obs.columns:
        mask = adata_proc.obs["transcript_counts"] >= 10
        n_before = adata_proc.n_obs
        adata_proc = adata_proc[mask].copy()
        log.info(f"  Filtered {n_before - adata_proc.n_obs} cells with <10 transcripts ({adata_proc.n_obs} remaining)")
    else:
        # Calculate from X
        if sp.issparse(adata_proc.X):
            total_counts = np.array(adata_proc.X.sum(axis=1)).flatten()
        else:
            total_counts = adata_proc.X.sum(axis=1)
        mask = total_counts >= 10
        n_before = adata_proc.n_obs
        adata_proc = adata_proc[mask].copy()
        log.info(f"  Filtered {n_before - adata_proc.n_obs} cells with <10 transcripts ({adata_proc.n_obs} remaining)")

    # Normalize and process
    log.info("  Normalizing and preprocessing...")
    sc.pp.normalize_total(adata_proc, target_sum=1e4)
    sc.pp.log1p(adata_proc)
    sc.pp.scale(adata_proc, max_value=10)

    log.info("  Running PCA (n_pcs=16)...")
    sc.tl.pca(adata_proc, n_comps=16)

    log.info("  Computing neighbors (n=10)...")
    sc.pp.neighbors(adata_proc, n_neighbors=10, n_pcs=16)

    # Leiden at multiple resolutions
    resolutions = [0.2, 0.4, 0.6]
    results_by_res = {}

    for res in resolutions:
        log.info(f"  Running Leiden clustering (resolution={res})...")
        key = f"leiden_{res}"
        sc.tl.leiden(adata_proc, resolution=res, key_added=key)
        n_clusters = adata_proc.obs[key].nunique()
        log.info(f"    Found {n_clusters} clusters")

        # Wilcoxon DGE per cluster
        log.info(f"    Running Wilcoxon DGE...")
        sc.tl.rank_genes_groups(adata_proc, groupby=key, method="wilcoxon")

        # Get top markers per cluster
        dge_results = {}
        for cluster in sorted(adata_proc.obs[key].unique()):
            try:
                top_genes = sc.get.rank_genes_groups_df(adata_proc, group=cluster).head(10)
                dge_results[cluster] = top_genes["names"].tolist()
            except Exception as e:
                log.warning(f"    DGE failed for cluster {cluster}: {e}")
                dge_results[cluster] = []

        # Majority vote: assign cluster label from Tangram predictions
        tangram_key = "cell_type"
        if tangram_key in adata_proc.obs.columns:
            cluster_labels = {}
            cluster_tangram_dist = {}
            for cluster in sorted(adata_proc.obs[key].unique()):
                cluster_mask = adata_proc.obs[key] == cluster
                tangram_preds = adata_proc.obs.loc[cluster_mask, tangram_key]
                majority = tangram_preds.value_counts()
                cluster_labels[cluster] = majority.index[0]
                cluster_tangram_dist[cluster] = {
                    "majority": majority.index[0],
                    "majority_frac": float(majority.iloc[0] / majority.sum()),
                    "top3": {k: int(v) for k, v in majority.head(3).items()},
                    "n_cells": int(cluster_mask.sum()),
                }

            # Assign labels
            adata_proc.obs[f"tangram_leiden_{res}"] = adata_proc.obs[key].map(cluster_labels)

            # Map to STHELAR categories
            adata_proc.obs[f"sthelar_leiden_{res}"] = adata_proc.obs[f"tangram_leiden_{res}"].map(
                DISCO_TO_STHELAR
            ).fillna("Unknown")

            results_by_res[res] = {
                "n_clusters": n_clusters,
                "cluster_labels": cluster_labels,
                "cluster_details": cluster_tangram_dist,
                "dge_markers": dge_results,
            }
        else:
            log.warning(f"  No Tangram predictions found in obs['{tangram_key}']")

    elapsed = time.time() - t0
    log.info(f"Leiden refinement completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return adata_proc, results_by_res, elapsed


def evaluate(adata_proc, results_by_res, tangram_elapsed, leiden_elapsed):
    """Evaluate predictions against GT."""
    log.info("=" * 60)
    log.info("STEP 3: Evaluation against ground truth")
    log.info("=" * 60)

    all_results = {
        "dataset": "STHELAR breast_s0",
        "reference": "DISCO breast_v2.1",
        "method": "Tangram + Leiden refinement",
        "n_cells_spatial": int(adata_proc.n_obs),
        "tangram_elapsed_s": tangram_elapsed,
        "leiden_elapsed_s": leiden_elapsed,
    }

    # -- Evaluate raw Tangram predictions (mapped to STHELAR categories)
    tangram_key = "cell_type"
    if tangram_key in adata_proc.obs.columns:
        log.info("Evaluating raw Tangram predictions (mapped to STHELAR)...")
        adata_proc.obs["tangram_sthelar"] = adata_proc.obs[tangram_key].map(DISCO_TO_STHELAR).fillna("Unknown")

        gt = adata_proc.obs["gt_label"]
        pred = adata_proc.obs["tangram_sthelar"]

        # Filter out Less10 from evaluation
        eval_mask = gt != "Less10"
        gt_eval = gt[eval_mask]
        pred_eval = pred[eval_mask]

        acc = accuracy_score(gt_eval, pred_eval)
        f1_macro = f1_score(gt_eval, pred_eval, average="macro", zero_division=0)
        f1_weighted = f1_score(gt_eval, pred_eval, average="weighted", zero_division=0)

        log.info(f"  Raw Tangram -> STHELAR mapping:")
        log.info(f"    Accuracy:   {acc:.4f}")
        log.info(f"    F1 (macro): {f1_macro:.4f}")
        log.info(f"    F1 (weighted): {f1_weighted:.4f}")

        report = classification_report(gt_eval, pred_eval, zero_division=0, output_dict=True)
        log.info(f"\n{classification_report(gt_eval, pred_eval, zero_division=0)}")

        # Confusion matrix
        labels = sorted(set(gt_eval.unique()) | set(pred_eval.unique()))
        cm = confusion_matrix(gt_eval, pred_eval, labels=labels)
        log.info(f"Confusion matrix (rows=GT, cols=Pred):")
        log.info(f"Labels: {labels}")
        for i, row in enumerate(cm):
            log.info(f"  {labels[i]:30s}: {row}")

        all_results["tangram_raw"] = {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "per_class": {
                k: {
                    "precision": float(v["precision"]),
                    "recall": float(v["recall"]),
                    "f1-score": float(v["f1-score"]),
                    "support": int(v["support"]),
                }
                for k, v in report.items()
                if k not in ["accuracy", "macro avg", "weighted avg"]
            },
            "confusion_matrix": {
                "labels": labels,
                "matrix": cm.tolist(),
            },
            "prediction_distribution": pred_eval.value_counts().to_dict(),
        }

    # -- Evaluate Leiden-refined predictions
    for res in [0.2, 0.4, 0.6]:
        leiden_col = f"sthelar_leiden_{res}"
        if leiden_col not in adata_proc.obs.columns:
            continue

        log.info(f"\nEvaluating Leiden-refined predictions (resolution={res})...")
        pred = adata_proc.obs[leiden_col]
        gt = adata_proc.obs["gt_label"]

        eval_mask = gt != "Less10"
        gt_eval = gt[eval_mask]
        pred_eval = pred[eval_mask]

        acc = accuracy_score(gt_eval, pred_eval)
        f1_macro = f1_score(gt_eval, pred_eval, average="macro", zero_division=0)
        f1_weighted = f1_score(gt_eval, pred_eval, average="weighted", zero_division=0)

        log.info(f"  Leiden res={res}:")
        log.info(f"    Accuracy:   {acc:.4f}")
        log.info(f"    F1 (macro): {f1_macro:.4f}")
        log.info(f"    F1 (weighted): {f1_weighted:.4f}")

        report = classification_report(gt_eval, pred_eval, zero_division=0, output_dict=True)
        log.info(f"\n{classification_report(gt_eval, pred_eval, zero_division=0)}")

        labels = sorted(set(gt_eval.unique()) | set(pred_eval.unique()))
        cm = confusion_matrix(gt_eval, pred_eval, labels=labels)

        all_results[f"leiden_res_{res}"] = {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "per_class": {
                k: {
                    "precision": float(v["precision"]),
                    "recall": float(v["recall"]),
                    "f1-score": float(v["f1-score"]),
                    "support": int(v["support"]),
                }
                for k, v in report.items()
                if k not in ["accuracy", "macro avg", "weighted avg"]
            },
            "confusion_matrix": {
                "labels": labels,
                "matrix": cm.tolist(),
            },
            "prediction_distribution": pred_eval.value_counts().to_dict(),
            "leiden_details": results_by_res.get(res, {}),
        }
        # Clean cluster_details for JSON (remove non-serializable)
        if f"leiden_res_{res}" in all_results:
            details = all_results[f"leiden_res_{res}"].get("leiden_details", {})
            if "cluster_details" in details:
                for cluster, info in details["cluster_details"].items():
                    info["majority_frac"] = float(info["majority_frac"])

    return all_results


def main():
    log.info("=" * 60)
    log.info("STHELAR Tangram Pipeline: DISCO breast -> breast_s0")
    log.info("=" * 60)

    total_t0 = time.time()

    # Load data
    ref, adata = load_data()

    # Run Tangram
    adata, tangram_elapsed = run_tangram(adata, ref)

    # Save intermediate checkpoint (Tangram predictions only)
    log.info("Saving Tangram predictions checkpoint...")
    tangram_pred_col = "cell_type"
    if tangram_pred_col in adata.obs.columns:
        adata.obs[[tangram_pred_col, "gt_label"]].to_csv(OUTPUT_DIR / "tangram_predictions.csv")

    # Leiden refinement
    adata_proc, results_by_res, leiden_elapsed = leiden_refinement(adata)

    # Evaluate
    all_results = evaluate(adata_proc, results_by_res, tangram_elapsed, leiden_elapsed)
    all_results["total_elapsed_s"] = time.time() - total_t0

    # Save final results
    output_file = OUTPUT_DIR / "tangram_disco_breast_s0.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nResults saved to {output_file}")
    log.info(f"Total pipeline time: {all_results['total_elapsed_s']:.1f}s ({all_results['total_elapsed_s']/60:.1f} min)")

    # Print summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    if "tangram_raw" in all_results:
        r = all_results["tangram_raw"]
        log.info(f"Raw Tangram:    Acc={r['accuracy']:.4f}  F1_macro={r['f1_macro']:.4f}  F1_weighted={r['f1_weighted']:.4f}")
    for res in [0.2, 0.4, 0.6]:
        key = f"leiden_res_{res}"
        if key in all_results:
            r = all_results[key]
            log.info(f"Leiden {res}:  Acc={r['accuracy']:.4f}  F1_macro={r['f1_macro']:.4f}  F1_weighted={r['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
