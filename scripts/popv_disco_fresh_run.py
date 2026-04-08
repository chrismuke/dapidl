#!/usr/bin/env python
"""
Fresh popV+DISCO retrain run on STHELAR breast_s0.
Re-runs the full Process_Query + annotate_data pipeline (not from cache).
Evaluates at label1 and label2 levels.
"""

import gc
import json
import os
import time
import warnings
from collections import Counter
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT = Path("/mnt/work/git/dapidl")
STHELAR_BASE = Path(
    "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s0.zarr/"
    "sdata_breast_s0.zarr/tables"
)
DISCO_PATH = Path("/mnt/work/datasets/DISCO/disco_breast_v2.1.h5ad")
OUT_DIR = PROJECT / "pipeline_output" / "sthelar_pipeline"

POPV_TO_LABEL1 = {
    "luminal epithelial cell of mammary gland": "Epithelial",
    "basal cell": "Epithelial",
    "progenitor cell of mammary luminal epithelium": "Epithelial",
    "epithelial cell": "Epithelial",
    "endothelial cell": "Blood_vessel",
    "vascular associated smooth muscle cell": "Blood_vessel",
    "pericyte": "Blood_vessel",
    "fibroblast of breast": "Fibroblast",
    "fibroblast": "Fibroblast",
    "macrophage": "Monocyte/Macrophage",
    "monocyte": "Monocyte/Macrophage",
    "dendritic cell": "Monocyte/Macrophage",
    "CD4-positive, alpha-beta T cell": "T",
    "CD8-positive, alpha-beta T cell": "T",
    "T cell": "T",
    "regulatory T cell": "T",
    "mature NK T cell": "T",
    "natural killer cell": "T",
    "innate lymphoid cell": "T",
    "B cell": "B",
    "plasma cell": "Plasma",
    "mast cell": "Mast",
    "basophil": "Monocyte/Macrophage",
    "adipocyte": "Adipocyte",
    "unknown": "less10",
    "unassigned": "less10",
}

POPV_TO_LABEL2 = {
    "luminal epithelial cell of mammary gland": "Mammary_luminal_cell",
    "basal cell": "Mammary_basal_cell_(=myoepithelial)",
    "progenitor cell of mammary luminal epithelium": "Mammary_luminal_cell",
    "epithelial cell": "Mammary_luminal_cell",
    "endothelial cell": "Endothelial_Pericyte_Smooth_muscle",
    "vascular associated smooth muscle cell": "Endothelial_Pericyte_Smooth_muscle",
    "pericyte": "Endothelial_Pericyte_Smooth_muscle",
    "fibroblast of breast": "CAF",
    "fibroblast": "CAF",
    "macrophage": "Monocyte/Macrophage",
    "monocyte": "Monocyte/Macrophage",
    "dendritic cell": "Monocyte/Macrophage",
    "CD4-positive, alpha-beta T cell": "T",
    "CD8-positive, alpha-beta T cell": "T",
    "T cell": "T",
    "regulatory T cell": "T",
    "mature NK T cell": "T",
    "natural killer cell": "T",
    "innate lymphoid cell": "T",
    "B cell": "B",
    "plasma cell": "Plasma",
    "mast cell": "Mast",
    "basophil": "Mast",
    "adipocyte": "Adipocyte",
    "unknown": "less10",
    "unassigned": "less10",
}

DISCO_TO_CL = {
    "SFN mammary luminal progenitor": "luminal epithelial cell of mammary gland",
    "KRT6B mammary basal cell": "basal cell",
    "KRT17 mammary luminal cell": "luminal epithelial cell of mammary gland",
    "PIP mammary luminal cell": "luminal epithelial cell of mammary gland",
    "SAA2 mammary luminal progenitor": "luminal epithelial cell of mammary gland",
    "CXCL14 mammary basal cell": "basal cell",
    "Secretoglobin mammary luminal progenitor": "luminal epithelial cell of mammary gland",
    "SCGB3A1 mammary luminal progenitor": "luminal epithelial cell of mammary gland",
    "Secretoglobin mammary luminal cell": "luminal epithelial cell of mammary gland",
    "CCSER1 mammary basal cell": "basal cell",
    "Cycling mammary luminal progenitor": "luminal epithelial cell of mammary gland",
    "Lactocyte": "luminal epithelial cell of mammary gland",
    "APOD+PTGDS+ fibroblast": "fibroblast of breast",
    "CFD+MGP+ fibroblast": "fibroblast of breast",
    "CDH19+LAMA2+ fibroblast": "fibroblast of breast",
    "MFAP5+IGFBP6+ fibroblast": "fibroblast of breast",
    "GPC3+ fibroblast": "fibroblast of breast",
    "BNC2+ZFPM2+ fibroblast": "fibroblast of breast",
    "Capillary EC": "endothelial cell",
    "Venous EC": "endothelial cell",
    "Arterial EC": "endothelial cell",
    "Lymphatic EC": "endothelial cell",
    "Vascular smooth muscle cell": "vascular associated smooth muscle cell",
    "CREB+MT1A+ vascular smooth muscle cell": "vascular associated smooth muscle cell",
    "CXCL+ pericyte": "pericyte",
    "CCL19/21 pericyte": "pericyte",
    "Pericyte": "pericyte",
    "M1 macrophage": "macrophage",
    "Macrophage": "macrophage",
    "LYVE1 macrophage": "macrophage",
    "Monocyte": "monocyte",
    "Dendritic cell": "dendritic cell",
    "pDC": "dendritic cell",
    "GZMB CD8 T cell": "CD8-positive, alpha-beta T cell",
    "CD4 T cell": "CD4-positive, alpha-beta T cell",
    "GZMK CD8 T cell": "CD8-positive, alpha-beta T cell",
    "Treg cell": "regulatory T cell",
    "NK cell": "natural killer cell",
    "ILC": "innate lymphoid cell",
    "B cell": "B cell",
    "Plasma cell": "plasma cell",
    "Mast cell": "mast cell",
}


def elapsed(start):
    dt = time.time() - start
    return f"{dt:.1f}s" if dt < 60 else f"{dt / 60:.1f}min"


def evaluate(y_true, y_pred, level_name, method_name):
    mask = ~pd.Series(y_true).isin(["less10", "Other", "unknown", ""])
    y_t = np.array(y_true)[mask]
    y_p = np.array(y_pred)[mask]
    acc = accuracy_score(y_t, y_p)
    f1_mac = f1_score(y_t, y_p, average="macro", zero_division=0)
    f1_wt = f1_score(y_t, y_p, average="weighted", zero_division=0)
    report = classification_report(y_t, y_p, output_dict=True, zero_division=0)
    per_class = {}
    for cls in sorted(set(y_t)):
        if cls in report:
            per_class[cls] = {
                "f1": round(report[cls]["f1-score"], 4),
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "support": int(report[cls]["support"]),
            }
    return {
        "method": method_name, "level": level_name,
        "n_cells": int(mask.sum()), "accuracy": round(acc, 4),
        "f1_macro": round(f1_mac, 4), "f1_weighted": round(f1_wt, 4),
        "per_class": per_class,
    }


def map_to_labels(preds, mapping):
    def _map(label):
        s = str(label)
        if s in mapping:
            return mapping[s]
        sl = s.lower()
        if "epithelial" in sl or "luminal" in sl or "basal" in sl or "mammary" in sl:
            return mapping.get("epithelial cell", "Epithelial")
        if "fibroblast" in sl or "stromal" in sl:
            return mapping.get("fibroblast", "Fibroblast")
        if "endothel" in sl or "pericyte" in sl or "vascular" in sl:
            return mapping.get("endothelial cell", "Blood_vessel")
        if "macrophage" in sl or "monocyte" in sl or "dendritic" in sl:
            return mapping.get("macrophage", "Monocyte/Macrophage")
        if "t cell" in sl or "cd4" in sl or "cd8" in sl or "nk" in sl:
            return mapping.get("T cell", "T")
        if "b cell" in sl:
            return mapping.get("B cell", "B")
        if "plasma" in sl:
            return mapping.get("plasma cell", "Plasma")
        if "mast" in sl:
            return mapping.get("mast cell", "Mast")
        return "less10"
    return pd.Series(preds).map(_map)


def main():
    import anndata as ad
    t_total = time.time()

    # ---- Load STHELAR ----
    print("=" * 70)
    print("Loading STHELAR breast_s0")
    print("=" * 70)
    adata = ad.read_zarr(str(STHELAR_BASE / "table_cells"))
    adata.X = adata.layers["count"].copy()
    print(f"  Shape: {adata.shape[0]:,} x {adata.shape[1]}")
    print(f"  label1: {dict(adata.obs['label1'].value_counts())}")

    # ---- Load DISCO reference ----
    print("\n  Loading DISCO breast v2.1...")
    ref = sc.read_h5ad(str(DISCO_PATH))
    ref.obs["popv_labels"] = ref.obs["cell_type"].map(DISCO_TO_CL).fillna("unknown").astype(str)
    ref.obs["batch_key"] = ref.obs["sample_id"].astype(str)
    n_mapped = (ref.obs["popv_labels"] != "unknown").sum()
    print(f"  Ref: {ref.shape}, {n_mapped}/{ref.shape[0]} mapped")

    # Ensure raw counts
    if issparse(ref.X):
        x_max = float(ref.X[:100, :100].toarray().max())
    else:
        x_max = float(ref.X[:100, :100].max())
    if x_max < 20:
        for layer in ["counts", "raw_counts"]:
            if layer in ref.layers:
                ref.X = ref.layers[layer].copy()
                print(f"  Set ref.X to layers['{layer}']")
                break
    ref.layers = {}

    # ---- Subsample 50K ----
    np.random.seed(42)
    n_target = 50_000
    idx = np.random.choice(adata.shape[0], size=n_target, replace=False)
    idx.sort()
    adata_sub = adata[idx].copy()
    print(f"\n  Subsample: {adata_sub.shape}")

    # Prepare query
    adata_query = adata_sub.copy()
    if "count" in adata_query.layers:
        adata_query.X = adata_query.layers["count"].copy()
    adata_query.layers = {}
    adata_query.obsm = {}
    adata_query.uns = {}

    common_genes = sorted(set(adata_query.var_names) & set(ref.var_names))
    print(f"  Common genes: {len(common_genes)}")
    adata_query = adata_query[:, common_genes].copy()
    ref = ref[:, common_genes].copy()

    keep_cols = {"cell_id"}
    for col in list(adata_query.obs.columns):
        if col not in keep_cols:
            del adata_query.obs[col]

    # ---- Process_Query (retrain) ----
    print(f"\n  Running popV Process_Query (retrain)...")
    t_pq = time.time()

    import popv
    from popv.annotation import annotate_data
    from popv.preprocessing import Process_Query
    print(f"  popV {popv.__version__}")

    save_path = str(OUT_DIR / "popv_disco_fresh" / "trained_models")
    os.makedirs(save_path, exist_ok=True)

    ontology_dir = (
        Path.home()
        / ".cache/huggingface/hub/datasets--popV--ontology"
        / "snapshots/2da43b6e227e76e67b4f32f028886e6308f56246"
    )

    pq = Process_Query(
        query_adata=adata_query,
        ref_adata=ref,
        ref_labels_key="popv_labels",
        ref_batch_key="batch_key",
        cl_obo_folder=str(ontology_dir),
        query_batch_key=None,
        prediction_mode="retrain",
        unknown_celltype_label="unknown",
        n_samples_per_label=300,
        save_path_trained_models=save_path,
        pretrained_scvi_path=None,
        hvg=4000,
    )
    concatenated = pq.adata
    print(f"  Preprocessed: {concatenated.shape} [{elapsed(t_pq)}]")

    del ref, adata_query
    gc.collect()

    # ---- annotate_data ----
    print(f"\n  Running popV annotation (all 8 methods)...")
    t_annot = time.time()
    popv_output = str(OUT_DIR / "popv_disco_fresh" / "popv_output")
    os.makedirs(popv_output, exist_ok=True)
    annotate_data(concatenated, methods=None, save_path=popv_output)
    print(f"  Done [{elapsed(t_annot)}]")

    # Extract query results
    if "_dataset" in concatenated.obs.columns:
        result_sub = concatenated[concatenated.obs["_dataset"] == "query"].copy()
    elif "_is_ref" in concatenated.obs.columns:
        result_sub = concatenated[~concatenated.obs["_is_ref"].astype(bool)].copy()
    else:
        result_sub = concatenated[-adata_sub.shape[0]:].copy()
    print(f"  Query results: {result_sub.shape[0]:,} cells")

    consensus_col = None
    for c in ["popv_prediction", "popv_majority_vote_prediction"]:
        if c in result_sub.obs.columns:
            consensus_col = c
            break
    print(f"  Consensus column: {consensus_col}")
    print(f"  Predictions:")
    for lbl, cnt in result_sub.obs[consensus_col].value_counts().items():
        print(f"    {lbl}: {cnt:,}")

    # ---- Direct evaluation on subsample ----
    print("\n--- DIRECT EVALUATION (48K subsample, no KNN) ---")
    # Match result obs_names to subsample
    result_obs = set(result_sub.obs_names.astype(str))
    matched_idx, matched_labels = [], []
    for i, obs_name in enumerate(adata_sub.obs_names.astype(str)):
        if obs_name in result_obs:
            pos = list(result_sub.obs_names.astype(str)).index(obs_name)
            matched_idx.append(idx[i])
            matched_labels.append(result_sub.obs[consensus_col].iloc[pos])
    matched_idx = np.array(matched_idx)
    matched_labels = np.array(matched_labels)
    print(f"  Matched: {len(matched_idx)} cells")

    gt_sub = adata.obs["label1"].values[matched_idx].astype(str)
    pred_sub = map_to_labels(matched_labels, POPV_TO_LABEL1).values

    mask = (gt_sub != "less10") & (pred_sub != "less10")
    f1_direct = f1_score(gt_sub[mask], pred_sub[mask], average="macro", zero_division=0)
    acc_direct = accuracy_score(gt_sub[mask], pred_sub[mask])
    print(f"  Direct subsample: F1={f1_direct:.4f}, Acc={acc_direct:.4f}")

    # ---- KNN transfer to full dataset ----
    print("\n--- KNN TRANSFER (48K → 577K) ---")
    t_knn = time.time()

    adata_tmp = adata.copy()
    sc.pp.normalize_total(adata_tmp, target_sum=1e4)
    sc.pp.log1p(adata_tmp)
    X = adata_tmp.X.toarray() if issparse(adata_tmp.X) else np.asarray(adata_tmp.X)
    del adata_tmp
    gc.collect()

    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)
    del X
    gc.collect()

    knn = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)
    knn.fit(X_pca[matched_idx], matched_labels)
    full_preds = knn.predict(X_pca)
    del X_pca
    gc.collect()
    print(f"  KNN done [{elapsed(t_knn)}]")

    # ---- Evaluate at label1 ----
    gt_l1 = adata.obs["label1"].astype(str).values
    pred_l1 = map_to_labels(full_preds, POPV_TO_LABEL1).values
    r_l1 = evaluate(gt_l1, pred_l1, "label1", "popv_disco_fresh")
    print(f"\n  popV+DISCO FRESH @ label1: F1={r_l1['f1_macro']:.4f}, Acc={r_l1['accuracy']:.4f}")
    for cls, m in sorted(r_l1["per_class"].items()):
        print(f"    {cls}: F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")

    # ---- Evaluate at label2 ----
    gt_l2 = adata.obs["label2"].astype(str).values
    pred_l2 = map_to_labels(full_preds, POPV_TO_LABEL2).values
    r_l2 = evaluate(gt_l2, pred_l2, "label2", "popv_disco_fresh")
    print(f"\n  popV+DISCO FRESH @ label2: F1={r_l2['f1_macro']:.4f}, Acc={r_l2['accuracy']:.4f}")

    # ---- Save ----
    out = {
        "method": "popv_disco_fresh_retrain",
        "reference": "DISCO_breast_v2.1",
        "direct_subsample": {"f1_macro": f1_direct, "accuracy": acc_direct, "n_cells": len(matched_idx)},
        "results_label1": r_l1,
        "results_label2": r_l2,
        "elapsed_s": round(time.time() - t_total, 1),
    }
    out_path = OUT_DIR / "popv_disco_fresh_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")
    print(f"  TOTAL: {elapsed(t_total)}")


if __name__ == "__main__":
    main()
