#!/usr/bin/env python3
"""
STHELAR pipeline RESUME script.

Reloads data + applies Tangram from the previous full run, then re-runs
steps 2-6 without Tangram (saving ~40 minutes). Uses improved cluster
labeling that doesn't over-assign Fibroblast.

This also fixes the memory issue that caused the previous run to hang.
"""

import gc
import json
import logging
import time
import warnings
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd  # noqa: TID251 — scanpy/anndata boundary
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("/mnt/work/git/dapidl/pipeline_output/sthelar_pipeline")

# DISCO -> STHELAR category mapping (same as main script)
DISCO_TO_STHELAR = {
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
    "APOD+PTGDS+ fibroblast": "Fibroblast_Myofibroblast",
    "CFD+MGP+ fibroblast": "Fibroblast_Myofibroblast",
    "CDH19+LAMA2+ fibroblast": "Fibroblast_Myofibroblast",
    "MFAP5+IGFBP6+ fibroblast": "Fibroblast_Myofibroblast",
    "GPC3+ fibroblast": "Fibroblast_Myofibroblast",
    "BNC2+ZFPM2+ fibroblast": "Fibroblast_Myofibroblast",
    "Capillary EC": "Blood_vessel",
    "Venous EC": "Blood_vessel",
    "Arterial EC": "Blood_vessel",
    "Lymphatic EC": "Blood_vessel",
    "Pericyte": "Blood_vessel",
    "CXCL+ pericyte": "Blood_vessel",
    "CCL19/21 pericyte": "Blood_vessel",
    "Vascular smooth muscle cell": "Blood_vessel",
    "CREB+MT1A+ vascular smooth muscle cell": "Blood_vessel",
    "CD4 T cell": "T_NK",
    "GZMB CD8 T cell": "T_NK",
    "GZMK CD8 T cell": "T_NK",
    "Treg cell": "T_NK",
    "NK cell": "T_NK",
    "ILC": "T_NK",
    "B cell": "B_Plasma",
    "Plasma cell": "B_Plasma",
    "M1 macrophage": "Myeloid",
    "Macrophage": "Myeloid",
    "LYVE1 macrophage": "Myeloid",
    "Monocyte": "Myeloid",
    "Dendritic cell": "Myeloid",
    "pDC": "Myeloid",
    "Mast cell": "Specialized",
}

# Marker genes with STRICT specificity scoring
# Key improvement: use score = mean(pos) - mean(neg) with equal weighting
CATEGORY_MARKERS = {
    "Epithelial": {
        "positive": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "KRT7", "MUC1",
                      "KRT14", "KRT5", "KRT17"],
        "negative": ["PTPRC", "PECAM1", "COL1A1", "CD3D", "CD3E", "CD68", "MS4A1"],
    },
    "Blood_vessel": {
        "positive": ["PECAM1", "VWF", "CLDN5", "KDR", "CDH5", "RGS5", "MCAM",
                      "AQP1", "CAV1", "CD93"],
        "negative": ["EPCAM", "PTPRC", "CD3D", "KRT18", "COL1A1"],
    },
    "Fibroblast_Myofibroblast": {
        "positive": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "FAP", "PDGFRA",
                      "POSTN", "CTHRC1"],
        "negative": ["EPCAM", "PTPRC", "PECAM1", "CD3D", "CD3E", "CD68", "MS4A1",
                      "KRT18", "KRT19"],
    },
    "Myeloid": {
        "positive": ["CD68", "CD163", "CSF1R", "LYZ", "CD14", "S100A8", "S100A9",
                      "AIF1", "FCER1G"],
        "negative": ["CD3D", "CD3E", "MS4A1", "EPCAM", "KRT18", "PECAM1", "COL1A1"],
    },
    "B_Plasma": {
        "positive": ["CD79A", "CD79B", "MS4A1", "SDC1", "MZB1", "JCHAIN",
                      "IGHG1", "BANK1"],
        "negative": ["CD3D", "CD3E", "CD68", "EPCAM", "KRT18", "PECAM1", "COL1A1"],
    },
    "T_NK": {
        "positive": ["CD3D", "CD3E", "CD3G", "TRAC", "NKG7", "GNLY", "GZMB",
                      "CD8A", "IL7R", "CD2"],
        "negative": ["MS4A1", "CD68", "EPCAM", "KRT18", "PECAM1", "COL1A1"],
    },
    "Specialized": {
        "positive": ["KIT", "TPSAB1", "CPA3", "TPSB2", "HDC"],
        "negative": ["CD3D", "CD3E", "CD68", "EPCAM", "PECAM1", "MS4A1", "COL1A1"],
    },
}


def load_data_with_tangram():
    """Load spatial data + apply Tangram labels from previous run's table_combined."""
    log.info("Loading STHELAR breast_s0 data...")
    t0 = time.time()

    adata = ad.read_zarr(
        "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s0.zarr/"
        "sdata_breast_s0.zarr/tables/table_cells"
    )
    log.info(f"  Loaded {adata.n_obs:,} cells, {adata.n_vars} genes ({time.time()-t0:.1f}s)")

    # Use raw counts as X
    adata.X = adata.layers["count"].copy()
    if sp.issparse(adata.X):
        adata.X = adata.X.astype(np.float32)

    # Load GT + Tangram labels from table_combined
    gt = ad.read_zarr(
        "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s0.zarr/"
        "sdata_breast_s0.zarr/tables/table_combined"
    )
    adata.obs["gt_label"] = gt.obs["final_label_combined"].values
    adata.obs["tangram_fine"] = gt.obs["ct_tangram"].values  # Paper's Tangram labels
    adata.obs["tangram_sthelar"] = adata.obs["tangram_fine"].map(DISCO_TO_STHELAR).fillna("Other")

    log.info(f"  Tangram types: {adata.obs['tangram_fine'].nunique()}")
    log.info("  GT distribution:")
    for lbl, cnt in adata.obs["gt_label"].value_counts().items():
        log.info(f"    {lbl:<30s}: {cnt:>8,}")

    del gt
    gc.collect()
    return adata


def run_leiden(adata, resolutions=(0.2, 0.4, 0.6)):
    """Leiden clustering at multiple resolutions (STHELAR protocol)."""
    log.info("=" * 70)
    log.info("STEP 2: Leiden clustering")
    log.info("=" * 70)
    t0 = time.time()

    a = adata.copy()

    # Filter <10 transcripts
    if "transcript_counts" in a.obs.columns:
        tc = a.obs["transcript_counts"].values.astype(float)
    else:
        tc = np.array(a.X.sum(axis=1)).flatten() if sp.issparse(a.X) else a.X.sum(axis=1)
    mask = tc >= 10
    a = a[mask].copy()
    log.info(f"  {mask.sum():,} cells pass >= 10 transcript filter")

    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.layers["log_norm"] = a.X.copy()
    sc.pp.scale(a, max_value=10)

    if sp.issparse(a.X):
        a.X = np.asarray(a.X.toarray(), dtype=np.float32)
    a.X = np.nan_to_num(np.asarray(a.X, dtype=np.float32))

    gene_var = np.var(a.X, axis=0)
    keep = gene_var > 0
    a = a[:, keep].copy()

    n_pcs = min(16, a.n_vars - 1)
    log.info(f"  PCA({n_pcs}), neighbors(10), Leiden...")
    sc.tl.pca(a, n_comps=n_pcs, svd_solver="arpack")
    sc.pp.neighbors(a, n_neighbors=10, n_pcs=n_pcs)

    leiden_info = {}
    for res in resolutions:
        key = f"leiden_r{res}"
        sc.tl.leiden(a, resolution=res, key_added=key)
        n_cl = a.obs[key].nunique()
        leiden_info[res] = n_cl
        log.info(f"  res={res}: {n_cl} clusters")
        adata.obs[key] = "filtered"
        adata.obs.loc[a.obs_names, key] = a.obs[key].values

    # Pick best resolution (10-20 clusters)
    best_res = min(resolutions, key=lambda r: abs(leiden_info[r] - 15))
    best_key = f"leiden_r{best_res}"

    elapsed = time.time() - t0
    log.info(f"  Selected: {best_key} ({leiden_info[best_res]} clusters, {elapsed:.0f}s)")

    # Save kNN graph for confidence scoring
    knn_conn = a.obsp["connectivities"].copy()
    knn_obs = a.obs_names.copy()

    return adata, a, best_key, best_res, leiden_info, knn_conn, knn_obs, elapsed


def run_dge_labeling(adata, adata_proc, cluster_key):
    """Wilcoxon DGE + IMPROVED cluster labeling.

    Key fix: use net_specificity = mean(pos) - mean(neg) with FULL negative list.
    Also use Tangram majority fraction as tiebreaker -- if Tangram agrees with >30%
    majority, trust it over weak marker evidence.
    """
    log.info("=" * 70)
    log.info(f"STEP 3: DGE + cluster labeling ({cluster_key})")
    log.info("=" * 70)
    t0 = time.time()

    a = adata_proc.copy()
    a.X = a.layers["log_norm"].copy()
    if sp.issparse(a.X):
        a.X = np.asarray(a.X.toarray(), dtype=np.float32)
    a.X = np.asarray(a.X, dtype=np.float32)

    sc.tl.rank_genes_groups(a, cluster_key, method="wilcoxon", use_raw=False,
                            corr_method="benjamini-hochberg")

    gene_names = list(a.var_names)
    expr = a.X

    cluster_labels = {}
    for cluster in sorted(a.obs[cluster_key].unique()):
        mask = (a.obs[cluster_key] == cluster).values
        n_cells = int(mask.sum())

        # Top DEGs
        try:
            top_genes = [str(a.uns["rank_genes_groups"]["names"][i][cluster])
                         for i in range(min(30, len(a.uns["rank_genes_groups"]["names"])))]
            top_genes = [g for g in top_genes if g != "nan"]
        except (KeyError, IndexError):
            top_genes = []

        # Tangram majority vote
        tangram_in_cluster = adata.obs.loc[a.obs_names[mask], "tangram_fine"]
        tangram_counts = Counter(tangram_in_cluster)
        tangram_majority = tangram_counts.most_common(1)[0][0] if tangram_counts else "Unknown"
        tangram_sthelar = DISCO_TO_STHELAR.get(tangram_majority, "Other")
        total = sum(tangram_counts.values())
        tangram_frac = tangram_counts.most_common(1)[0][1] / total if total > 0 else 0

        # Also compute STHELAR-level majority (aggregate all Tangram labels to STHELAR)
        sthelar_counts = Counter()
        for lbl, cnt in tangram_counts.items():
            sthelar_counts[DISCO_TO_STHELAR.get(lbl, "Other")] += cnt
        sthelar_majority = sthelar_counts.most_common(1)[0][0] if sthelar_counts else "Other"
        sthelar_majority_frac = sthelar_counts.most_common(1)[0][1] / total if total > 0 else 0

        # Marker gene scoring with BALANCED pos/neg
        cl_expr = expr[mask]
        marker_scores = {}
        for cat, markers in CATEGORY_MARKERS.items():
            pos = [g for g in markers["positive"] if g in gene_names]
            neg = [g for g in markers["negative"] if g in gene_names]
            if not pos:
                marker_scores[cat] = -999.0
                continue
            pos_idx = [gene_names.index(g) for g in pos]
            neg_idx = [gene_names.index(g) for g in neg]
            pos_score = float(cl_expr[:, pos_idx].mean())
            neg_score = float(cl_expr[:, neg_idx].mean()) if neg_idx else 0.0
            # Net specificity: positive evidence minus negative evidence (equally weighted)
            marker_scores[cat] = pos_score - neg_score

        marker_best = max(marker_scores, key=marker_scores.get)
        marker_best_score = marker_scores[marker_best]

        # IMPROVED decision rule:
        # 1. If Tangram STHELAR majority > 40% AND markers agree -> high confidence
        # 2. If Tangram STHELAR majority > 40% AND markers disagree:
        #    - Trust markers ONLY if marker_best_score > 1.5 (strong marker evidence)
        #    - Otherwise trust Tangram (it has cell-level resolution, markers are cluster-level)
        # 3. If Tangram is weak (<40%) -> trust markers if score > 0.5

        if sthelar_majority == marker_best:
            final_label = sthelar_majority
            decision = "agree"
        elif sthelar_majority_frac > 0.4 and marker_best_score > 1.5:
            # Strong marker evidence overrides Tangram
            final_label = marker_best
            decision = "markers_override_strong"
        elif sthelar_majority_frac > 0.4:
            # Tangram has clear majority, markers are weak -> trust Tangram
            final_label = sthelar_majority
            decision = "tangram_majority"
        elif marker_best_score > 0.5:
            final_label = marker_best
            decision = "markers_override_weak_tangram"
        elif sthelar_majority != "Other":
            final_label = sthelar_majority
            decision = "tangram_fallback"
        else:
            final_label = marker_best if marker_best_score > 0 else "Other"
            decision = "low_confidence"

        cluster_labels[cluster] = {
            "final_label": final_label,
            "tangram_majority": tangram_majority,
            "tangram_sthelar": tangram_sthelar,
            "sthelar_majority": sthelar_majority,
            "sthelar_majority_frac": round(sthelar_majority_frac, 3),
            "tangram_frac": round(tangram_frac, 3),
            "marker_best": marker_best,
            "marker_score": round(marker_best_score, 3),
            "decision": decision,
            "n_cells": n_cells,
            "top_degs": top_genes[:10],
        }

        log.info(f"  Cluster {cluster:>3s} ({n_cells:>6,}): "
                 f"{final_label:<25s} [{decision:>25s}] "
                 f"tangram={sthelar_majority}({sthelar_majority_frac:.0%}) "
                 f"markers={marker_best}({marker_best_score:.2f})")

    label_map = {cl: info["final_label"] for cl, info in cluster_labels.items()}
    adata.obs["pipeline_label"] = adata.obs[cluster_key].map(label_map).fillna("Other")
    adata.obs.loc[adata.obs[cluster_key] == "filtered", "pipeline_label"] = "Other"

    elapsed = time.time() - t0
    log.info(f"  Completed in {elapsed:.0f}s")
    log.info("  Label distribution:")
    for lbl, cnt in adata.obs["pipeline_label"].value_counts().items():
        log.info(f"    {lbl:<30s}: {cnt:>8,} ({100*cnt/adata.n_obs:.1f}%)")

    return adata, cluster_labels, elapsed


def run_scvi(adata, adata_proc, cluster_key):
    """scVI validation (memory-efficient)."""
    log.info("=" * 70)
    log.info("STEP 4: scVI validation")
    log.info("=" * 70)
    import scvi
    import torch

    t0 = time.time()
    torch.cuda.empty_cache()

    obs_names = adata_proc.obs_names
    var_names = adata_proc.var_names

    # Minimal AnnData with raw counts
    raw = adata[obs_names].layers["count"]
    if sp.issparse(raw):
        raw = raw.toarray()
    raw = np.asarray(raw, dtype=np.float32)
    gene_mask = np.isin(adata.var_names, var_names)
    raw_sub = raw[:, gene_mask]

    a = ad.AnnData(X=raw_sub, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))
    a.layers["count"] = raw_sub.copy()
    del raw, raw_sub
    gc.collect()

    scvi.model.SCVI.setup_anndata(a, layer="count")
    model = scvi.model.SCVI(a, n_latent=10, n_layers=2)
    log.info(f"  Training scVI on {a.n_obs:,} cells...")

    model.train(accelerator="gpu", early_stopping=True, early_stopping_patience=10,
                max_epochs=100, batch_size=256, train_size=0.9)

    latent = model.get_latent_representation()
    n_epochs = len(model.history["elbo_train"])
    final_elbo = float(model.history["elbo_train"].iloc[-1].values[0])
    log.info(f"  Trained {n_epochs} epochs, ELBO={final_elbo:.1f}, latent={latent.shape}")

    del model, a
    torch.cuda.empty_cache()
    gc.collect()

    # Cluster in scVI space
    a_scvi = ad.AnnData(X=latent, obs=pd.DataFrame(index=obs_names))
    a_scvi.obsm["X_scVI"] = latent
    sc.pp.neighbors(a_scvi, use_rep="X_scVI", n_neighbors=10)
    sc.tl.leiden(a_scvi, resolution=0.4, key_added="leiden_scvi")

    pipeline_labels = adata.obs.loc[obs_names, "pipeline_label"].values
    scvi_clusters = a_scvi.obs["leiden_scvi"].values

    ari = adjusted_rand_score(pipeline_labels, scvi_clusters)
    nmi = normalized_mutual_info_score(pipeline_labels, scvi_clusters)

    purity = []
    for cl in sorted(a_scvi.obs["leiden_scvi"].unique()):
        m = a_scvi.obs["leiden_scvi"] == cl
        lbls = adata.obs.loc[obs_names[m], "pipeline_label"]
        purity.append(lbls.value_counts().iloc[0] / len(lbls))

    n_scvi_cl = a_scvi.obs["leiden_scvi"].nunique()
    mean_purity = float(np.mean(purity))

    del a_scvi, latent
    gc.collect()

    elapsed = time.time() - t0
    log.info(f"  ARI={ari:.4f}, NMI={nmi:.4f}, purity={mean_purity:.4f}, {n_scvi_cl} clusters")
    log.info(f"  Completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return {
        "n_latent_dims": 10, "n_scvi_clusters": n_scvi_cl,
        "ari_vs_pipeline": round(float(ari), 4),
        "nmi_vs_pipeline": round(float(nmi), 4),
        "mean_cluster_purity": round(mean_purity, 4),
        "training_epochs": n_epochs,
        "final_elbo": round(final_elbo, 1),
    }, elapsed


def run_confidence(adata, knn_conn, knn_obs, cluster_key):
    """Confidence scoring: KNN entropy + RNA depth."""
    log.info("=" * 70)
    log.info("STEP 5: Confidence scoring")
    log.info("=" * 70)
    t0 = time.time()
    from scipy.stats import rankdata

    labels = adata.obs.loc[knn_obs, "pipeline_label"].values
    unique_labels = sorted(set(labels))
    max_entropy = np.log2(len(unique_labels)) if len(unique_labels) > 1 else 1.0

    n_cells = knn_conn.shape[0]
    entropy_arr = np.zeros(n_cells, dtype=np.float32)
    log.info(f"  Computing KNN entropy for {n_cells:,} cells...")

    for i in range(n_cells):
        row = knn_conn[i]
        nb_idx = row.indices if sp.issparse(row) else np.nonzero(row)[0]
        if len(nb_idx) == 0:
            continue
        counts = Counter(labels[nb_idx])
        total = sum(counts.values())
        entropy = sum(-c/total * np.log2(c/total) for c in counts.values() if c > 0)
        entropy_arr[i] = entropy

    knn_conf = 1.0 - (entropy_arr / max_entropy)
    knn_conf = np.clip(knn_conf, 0.0, 1.0)

    adata.obs["knn_entropy"] = np.nan
    adata.obs["knn_confidence"] = np.nan
    adata.obs.loc[knn_obs, "knn_entropy"] = entropy_arr
    adata.obs.loc[knn_obs, "knn_confidence"] = knn_conf

    log.info(f"  KNN confidence: mean={knn_conf.mean():.3f}, "
             f"median={np.median(knn_conf):.3f}, >0.5: {(knn_conf > 0.5).sum():,}")

    # RNA depth
    if "transcript_counts" in adata.obs.columns:
        tc = adata.obs["transcript_counts"].values.astype(float)
    else:
        tc = np.array(adata.X.sum(axis=1)).flatten() if sp.issparse(adata.X) else adata.X.sum(axis=1)
    rna_q = rankdata(np.log1p(tc), method="average") / len(tc)
    adata.obs["rna_depth_quantile"] = rna_q

    # Combined: geometric mean
    valid_idx = np.isin(adata.obs_names, knn_obs)
    combined = np.zeros(adata.n_obs, dtype=np.float32)
    knn_full = adata.obs["knn_confidence"].values.astype(float)
    combined[valid_idx] = np.sqrt(np.nan_to_num(knn_full[valid_idx], nan=0) * rna_q[valid_idx])
    adata.obs["combined_confidence"] = combined

    elapsed = time.time() - t0
    log.info(f"  Combined: mean={combined[valid_idx].mean():.3f}, >0.5: {(combined > 0.5).sum():,}")

    thresholds = [0.3, 0.5, 0.7, 0.8]
    for thr in thresholds:
        n = (combined > thr).sum()
        log.info(f"    > {thr}: {n:,} ({100*n/adata.n_obs:.1f}%)")

    log.info(f"  Completed in {elapsed:.0f}s")
    return adata, {
        "knn_confidence_mean": round(float(knn_conf.mean()), 4),
        "knn_confidence_median": round(float(np.median(knn_conf)), 4),
        "combined_confidence_mean": round(float(combined[valid_idx].mean()), 4),
        "pct_above_0.3": round(float(100 * (combined > 0.3).sum() / adata.n_obs), 2),
        "pct_above_0.5": round(float(100 * (combined > 0.5).sum() / adata.n_obs), 2),
        "pct_above_0.7": round(float(100 * (combined > 0.7).sum() / adata.n_obs), 2),
    }, elapsed


def run_evaluation(adata):
    """Evaluate pipeline labels against GT."""
    log.info("=" * 70)
    log.info("STEP 6: Evaluation")
    log.info("=" * 70)

    gt = adata.obs["gt_label"].astype(str).values
    pred = adata.obs["pipeline_label"].astype(str).values

    eval_mask = ~np.isin(gt, ["Less10", "Other", "Unknown"])
    gt_e, pred_e = gt[eval_mask], pred[eval_mask]

    classes = sorted(set(gt_e) | set(pred_e))
    acc = accuracy_score(gt_e, pred_e)
    f1_m = f1_score(gt_e, pred_e, average="macro", zero_division=0, labels=classes)
    f1_w = f1_score(gt_e, pred_e, average="weighted", zero_division=0, labels=classes)

    log.info(f"  Cells: {eval_mask.sum():,}")
    log.info(f"  Accuracy:     {acc:.4f}")
    log.info(f"  F1 (macro):   {f1_m:.4f}")
    log.info(f"  F1 (weighted): {f1_w:.4f}")

    prec, rec, f1, sup = precision_recall_fscore_support(gt_e, pred_e, labels=classes, zero_division=0)
    per_class = {}
    log.info(f"\n  {'Class':<30s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'N':>8s}")
    log.info(f"  {'-'*60}")
    for i, c in enumerate(classes):
        if sup[i] > 0:
            log.info(f"  {c:<30s} {prec[i]:>7.3f} {rec[i]:>7.3f} {f1[i]:>7.3f} {sup[i]:>8,}")
        per_class[c] = {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                        "f1": round(float(f1[i]), 4), "support": int(sup[i])}

    # Raw Tangram baseline
    tang_pred = adata.obs["tangram_sthelar"].astype(str).values[eval_mask]
    tang_acc = accuracy_score(gt_e, tang_pred)
    tang_f1 = f1_score(gt_e, tang_pred, average="macro", zero_division=0)
    log.info(f"\n  Tangram raw baseline: Acc={tang_acc:.4f}, F1_macro={tang_f1:.4f}")

    # Confidence-filtered
    conf_metrics = {}
    for thr in [0.3, 0.5, 0.7]:
        c = adata.obs["combined_confidence"].values
        cm = eval_mask & (c > thr)
        if cm.sum() < 100:
            continue
        a_ = accuracy_score(gt[cm], pred[cm])
        f_ = f1_score(gt[cm], pred[cm], average="macro", zero_division=0)
        ret = cm.sum() / eval_mask.sum()
        log.info(f"  conf>{thr}: Acc={a_:.4f}, F1={f_:.4f}, retention={100*ret:.1f}%")
        conf_metrics[f"conf_gt_{thr}"] = {
            "accuracy": round(float(a_), 4), "f1_macro": round(float(f_), 4),
            "retention": round(float(ret), 4), "n_cells": int(cm.sum()),
        }

    cm_labels = sorted(set(gt_e))
    cm_matrix = confusion_matrix(gt_e, pred_e, labels=cm_labels)

    return {
        "n_evaluated": int(eval_mask.sum()),
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_m), 4),
        "f1_weighted": round(float(f1_w), 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": cm_labels, "matrix": cm_matrix.tolist()},
        "tangram_raw": {"accuracy": round(float(tang_acc), 4), "f1_macro": round(float(tang_f1), 4)},
        "confidence_filtered": conf_metrics,
    }


def main():
    log.info("#" * 70)
    log.info("# STHELAR COMPLETE PIPELINE (resume: uses paper's Tangram labels)")
    log.info("#" * 70)

    total_t0 = time.time()
    timings = {}

    # Load data with paper's Tangram labels
    adata = load_data_with_tangram()

    # Step 2: Leiden
    adata, adata_proc, best_key, best_res, leiden_info, knn_conn, knn_obs, t = run_leiden(adata)
    timings["leiden"] = t

    # Step 3: DGE + labeling
    adata, cluster_labels, t = run_dge_labeling(adata, adata_proc, best_key)
    timings["dge_labeling"] = t

    # Step 4: scVI
    scvi_results, t = run_scvi(adata, adata_proc, best_key)
    timings["scvi"] = t

    del adata_proc
    gc.collect()

    # Step 5: Confidence
    adata, conf_results, t = run_confidence(adata, knn_conn, knn_obs, best_key)
    timings["confidence"] = t
    del knn_conn
    gc.collect()

    # Step 6: Evaluate
    metrics = run_evaluation(adata)

    total = time.time() - total_t0
    timings["total"] = total

    results = {
        "pipeline": "STHELAR_complete",
        "slide": "breast_s0",
        "reference": "DISCO_breast_v2.1 (paper's Tangram labels from ct_tangram)",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "steps": {
            "1_tangram": {
                "method": "paper's ct_tangram from table_combined",
                "n_unique_types": int(adata.obs["tangram_fine"].nunique()),
                "note": "Using STHELAR paper's pre-computed Tangram labels",
            },
            "2_leiden": {
                "resolutions_tested": [0.2, 0.4, 0.6],
                "selected_resolution": best_res,
                "n_clusters": leiden_info[best_res],
                "all_resolutions": {str(r): n for r, n in leiden_info.items()},
                "elapsed_s": round(timings["leiden"], 1),
            },
            "3_dge_labeling": {
                "method": "wilcoxon",
                "cluster_annotations": cluster_labels,
                "elapsed_s": round(timings["dge_labeling"], 1),
            },
            "4_scvi_validation": scvi_results | {"elapsed_s": round(timings["scvi"], 1)},
            "5_confidence": conf_results | {"elapsed_s": round(timings["confidence"], 1)},
        },
        "metrics": metrics,
        "timings": {k: round(v, 1) for k, v in timings.items()},
        "label_distribution": adata.obs["pipeline_label"].value_counts().to_dict(),
    }

    out = OUTPUT_DIR / "complete_pipeline_breast_s0.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {out}")

    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)
    log.info(f"Total: {total:.0f}s ({total/60:.1f} min)")
    for k, v in timings.items():
        if k != "total":
            log.info(f"  {k:<20s}: {v:>6.0f}s")
    log.info(f"\nPipeline: Acc={metrics['accuracy']:.4f}, F1_macro={metrics['f1_macro']:.4f}")
    log.info(f"Tangram raw: Acc={metrics['tangram_raw']['accuracy']:.4f}, "
             f"F1_macro={metrics['tangram_raw']['f1_macro']:.4f}")
    log.info(f"scVI: ARI={scvi_results['ari_vs_pipeline']:.4f}, "
             f"NMI={scvi_results['nmi_vs_pipeline']:.4f}")


if __name__ == "__main__":
    main()
