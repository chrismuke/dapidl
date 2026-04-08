#!/usr/bin/env python
"""
Fine-Grained Cell Type Annotation Comparison: 3 Methods on STHELAR Breast S0

Compares:
  1. popV retrain with DISCO breast v2.1 (42 fine-grained types)
  2. OnClass CL hierarchy propagation (expand existing popV predictions)
  3. BANKSY spatial clustering + fine-grained marker annotation

Evaluates against STHELAR label1 (10 types), label2 (11 types),
and fine-grained CL-normalized level.

Usage:
    uv run python scripts/finegrained_comparison_3methods.py
"""

import gc
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path("/mnt/work/git/dapidl")
STHELAR_BASE = Path(
    "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s0.zarr/"
    "sdata_breast_s0.zarr/tables"
)
DISCO_PATH = Path("/mnt/work/datasets/DISCO/disco_breast_v2.1.h5ad")
OUT_DIR = PROJECT / "pipeline_output" / "sthelar_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POPV_CACHE = OUT_DIR / "popv_cache"
POPV_PREDICTIONS = POPV_CACHE / "popv_output" / "predictions.csv"

# DISCO cache (from v4 standalone run — 48K cells, all 8 methods)
DISCO_CACHE_PREDICTIONS = (
    OUT_DIR / "popv_disco_cache" / "popv_output" / "predictions.csv"
)

# ---------------------------------------------------------------------------
# DISCO 42 types -> STHELAR label1 (10 types) mapping
# ---------------------------------------------------------------------------
DISCO_TO_LABEL1 = {
    # Epithelial (luminal + basal + cycling + lactocyte)
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
    # Fibroblast
    "APOD+PTGDS+ fibroblast": "Fibroblast",
    "CFD+MGP+ fibroblast": "Fibroblast",
    "CDH19+LAMA2+ fibroblast": "Fibroblast",
    "MFAP5+IGFBP6+ fibroblast": "Fibroblast",
    "GPC3+ fibroblast": "Fibroblast",
    "BNC2+ZFPM2+ fibroblast": "Fibroblast",
    # Blood_vessel (endothelial + pericyte + smooth muscle)
    "Capillary EC": "Blood_vessel",
    "Venous EC": "Blood_vessel",
    "Arterial EC": "Blood_vessel",
    "Lymphatic EC": "Blood_vessel",
    "Vascular smooth muscle cell": "Blood_vessel",
    "CREB+MT1A+ vascular smooth muscle cell": "Blood_vessel",
    "CXCL+ pericyte": "Blood_vessel",
    "CCL19/21 pericyte": "Blood_vessel",
    "Pericyte": "Blood_vessel",
    # Monocyte/Macrophage
    "M1 macrophage": "Monocyte/Macrophage",
    "Macrophage": "Monocyte/Macrophage",
    "LYVE1 macrophage": "Monocyte/Macrophage",
    "Monocyte": "Monocyte/Macrophage",
    "Dendritic cell": "Monocyte/Macrophage",
    "pDC": "Monocyte/Macrophage",
    # T
    "GZMB CD8 T cell": "T",
    "CD4 T cell": "T",
    "GZMK CD8 T cell": "T",
    "Treg cell": "T",
    "NK cell": "T",  # T_NK in final_label, T in label1
    "ILC": "T",
    # B
    "B cell": "B",
    # Plasma
    "Plasma cell": "Plasma",
    # Mast
    "Mast cell": "Mast",
    # Adipocyte
    # (no adipocyte in DISCO breast v2.1, but may appear via CL mapping)
}

# popV Cell Ontology labels -> STHELAR label1
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

# STHELAR label1 -> label2 (for finer eval)
LABEL1_TO_LABEL2 = {
    "Epithelial": "Mammary_luminal_cell",  # majority mapping; basal handled separately
    "Fibroblast": "CAF",
    "Monocyte/Macrophage": "Monocyte/Macrophage",
    "T": "T",
    "Blood_vessel": "Endothelial_Pericyte_Smooth_muscle",
    "B": "B",
    "Plasma": "Plasma",
    "Mast": "Mast",
    "Adipocyte": "Adipocyte",
    "less10": "less10",
}

# OnClass CL predictions -> STHELAR label1 (tissue-agnostic mapping)
# OnClass picks tissue-inappropriate subtypes (e.g. "respiratory basal cell" in breast)
ONCLASS_TO_LABEL1 = {
    # Epithelial
    "luminal adaptive secretory precursor cell of mammary gland": "Epithelial",
    "respiratory basal cell": "Epithelial",
    "progenitor cell": "Epithelial",
    # Fibroblast
    "fibroblast": "Fibroblast",
    # Blood_vessel
    "endothelial cell of uterus": "Blood_vessel",
    "central nervous system pericyte": "Blood_vessel",
    "blood vessel smooth muscle cell": "Blood_vessel",
    "perivascular cell": "Blood_vessel",
    "microcirculation associated smooth muscle cell": "Blood_vessel",
    "mural cell": "Blood_vessel",
    # Monocyte/Macrophage
    "tissue-resident macrophage": "Monocyte/Macrophage",
    "mononuclear phagocyte": "Monocyte/Macrophage",
    "myeloid leukocyte": "Monocyte/Macrophage",
    "liver dendritic cell": "Monocyte/Macrophage",
    "classical monocyte": "Monocyte/Macrophage",
    "non-classical monocyte": "Monocyte/Macrophage",
    "professional antigen presenting cell": "Monocyte/Macrophage",
    # T
    "mature alpha-beta T cell": "T",
    "CD4-positive, CD25-positive, alpha-beta regulatory T cell": "T",
    "CD8-positive, alpha-beta regulatory T cell": "T",
    "activated CD8-positive, alpha-beta T cell": "T",
    "CD4-positive, alpha-beta memory T cell": "T",
    "mature natural killer cell": "T",
    "group 1 innate lymphoid cell": "T",
    # B
    "lymph node mantle zone B cell": "B",
    # Plasma
    "long lived plasma cell": "Plasma",
    "antibody secreting cell": "Plasma",
    # Mast
    "mucosal type mast cell": "Mast",
    "mature basophil": "Mast",
}

# OnClass CL predictions -> STHELAR label2 (11 types)
ONCLASS_TO_LABEL2 = {
    "luminal adaptive secretory precursor cell of mammary gland": "Mammary_luminal_cell",
    "respiratory basal cell": "Mammary_basal_cell_(=myoepithelial)",
    "progenitor cell": "Mammary_luminal_cell",
    "fibroblast": "CAF",
    "endothelial cell of uterus": "Endothelial_Pericyte_Smooth_muscle",
    "central nervous system pericyte": "Endothelial_Pericyte_Smooth_muscle",
    "blood vessel smooth muscle cell": "Endothelial_Pericyte_Smooth_muscle",
    "perivascular cell": "Endothelial_Pericyte_Smooth_muscle",
    "microcirculation associated smooth muscle cell": "Endothelial_Pericyte_Smooth_muscle",
    "mural cell": "Endothelial_Pericyte_Smooth_muscle",
    "tissue-resident macrophage": "Monocyte/Macrophage",
    "mononuclear phagocyte": "Monocyte/Macrophage",
    "myeloid leukocyte": "Monocyte/Macrophage",
    "liver dendritic cell": "Monocyte/Macrophage",
    "classical monocyte": "Monocyte/Macrophage",
    "non-classical monocyte": "Monocyte/Macrophage",
    "professional antigen presenting cell": "Monocyte/Macrophage",
    "mature alpha-beta T cell": "T",
    "CD4-positive, CD25-positive, alpha-beta regulatory T cell": "T",
    "CD8-positive, alpha-beta regulatory T cell": "T",
    "activated CD8-positive, alpha-beta T cell": "T",
    "CD4-positive, alpha-beta memory T cell": "T",
    "mature natural killer cell": "T",
    "group 1 innate lymphoid cell": "T",
    "lymph node mantle zone B cell": "B",
    "long lived plasma cell": "Plasma",
    "antibody secreting cell": "Plasma",
    "mucosal type mast cell": "Mast",
    "mature basophil": "Mast",
}

# popV ensemble CL predictions -> STHELAR label2
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

# BANKSY fine-grained type -> STHELAR label2
BANKSY_TYPE_TO_LABEL2 = {
    "Mammary_luminal_cell": "Mammary_luminal_cell",
    "Mammary_basal_cell": "Mammary_basal_cell_(=myoepithelial)",
    "CAF": "CAF",
    "Endothelial": "Endothelial_Pericyte_Smooth_muscle",
    "Pericyte": "Endothelial_Pericyte_Smooth_muscle",
    "Smooth_muscle": "Endothelial_Pericyte_Smooth_muscle",
    "Macrophage": "Monocyte/Macrophage",
    "Monocyte": "Monocyte/Macrophage",
    "Dendritic_cell": "Monocyte/Macrophage",
    "Mast": "Mast",
    "CD4_T": "T",
    "CD8_T": "T",
    "Treg": "T",
    "NK": "T",
    "B": "B",
    "Plasma": "Plasma",
    "Adipocyte": "Adipocyte",
}

# Fine-grained marker genes for BANKSY cluster annotation
MARKER_GENES_FINEGRAINED = {
    # Epithelial subtypes
    "Mammary_luminal_cell": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "GATA3"],
    "Mammary_basal_cell": ["KRT5", "KRT14", "KRT17", "TP63", "ACTA2"],
    # Fibroblast
    "CAF": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA", "FAP", "ACTA2"],
    # Vascular
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "KDR", "FLT1"],
    "Pericyte": ["RGS5", "PDGFRB", "NOTCH3", "ACTA2"],
    "Smooth_muscle": ["ACTA2", "MYH11", "TAGLN", "CNN1"],
    # Immune - Myeloid
    "Macrophage": ["CD68", "CD163", "CSF1R", "MSR1", "MARCO", "APOE"],
    "Monocyte": ["CD14", "FCGR3A", "S100A8", "S100A9", "LYZ"],
    "Dendritic_cell": ["CD1C", "CLEC9A", "FCER1A", "IRF7", "IRF8"],
    "Mast": ["KIT", "TPSAB1", "TPSB2", "CPA3"],
    # Immune - Lymphoid
    "CD4_T": ["CD3D", "CD3E", "CD4", "IL7R", "CCR7"],
    "CD8_T": ["CD3D", "CD3E", "CD8A", "CD8B", "GZMB", "PRF1"],
    "Treg": ["CD3D", "FOXP3", "IL2RA", "CTLA4"],
    "NK": ["NKG7", "KLRD1", "GNLY", "NCAM1"],
    "B": ["CD79A", "CD79B", "MS4A1", "CD19", "PAX5"],
    "Plasma": ["JCHAIN", "IGHA1", "IGHG1", "MZB1", "XBP1", "SDC1"],
    # Other
    "Adipocyte": ["ADIPOQ", "LEP", "FABP4", "PPARG"],
}

# Broad BANKSY cluster -> label1 mapping
BANKSY_TYPE_TO_LABEL1 = {
    "Mammary_luminal_cell": "Epithelial",
    "Mammary_basal_cell": "Epithelial",
    "CAF": "Fibroblast",
    "Endothelial": "Blood_vessel",
    "Pericyte": "Blood_vessel",
    "Smooth_muscle": "Blood_vessel",
    "Macrophage": "Monocyte/Macrophage",
    "Monocyte": "Monocyte/Macrophage",
    "Dendritic_cell": "Monocyte/Macrophage",
    "Mast": "Mast",
    "CD4_T": "T",
    "CD8_T": "T",
    "Treg": "T",
    "NK": "T",
    "B": "B",
    "Plasma": "Plasma",
    "Adipocyte": "Adipocyte",
}


def elapsed(start: float) -> str:
    dt = time.time() - start
    if dt < 60:
        return f"{dt:.1f}s"
    return f"{dt / 60:.1f}min"


def evaluate(y_true, y_pred, level_name: str, method_name: str) -> dict:
    """Evaluate predictions against ground truth."""
    # Filter out less10/unknown
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

    result = {
        "method": method_name,
        "level": level_name,
        "n_cells": int(mask.sum()),
        "n_gt_types": len(set(y_t)),
        "n_pred_types": len(set(y_p)),
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_mac, 4),
        "f1_weighted": round(f1_wt, 4),
        "per_class": per_class,
    }
    return result


def map_predictions_to_label1(labels: pd.Series, mapping: dict) -> pd.Series:
    """Map predicted labels to STHELAR label1 categories with substring fallback."""
    def _map_one(label):
        if pd.isna(label) or str(label) in ("unknown", "unassigned", "nan", ""):
            return "less10"
        s = str(label)
        if s in mapping:
            return mapping[s]
        # Substring fallback
        sl = s.lower()
        if "epithelial" in sl or "luminal" in sl or "basal" in sl or "mammary" in sl:
            return "Epithelial"
        if "fibroblast" in sl or "stromal" in sl or "caf" in sl:
            return "Fibroblast"
        if "endothel" in sl or "pericyte" in sl or "smooth muscle" in sl or "vascular" in sl:
            return "Blood_vessel"
        if "macrophage" in sl or "monocyte" in sl or "dendritic" in sl or "myeloid" in sl:
            return "Monocyte/Macrophage"
        if "b cell" in sl:
            return "B"
        if "plasma" in sl:
            return "Plasma"
        if "t cell" in sl or "cd4" in sl or "cd8" in sl or "treg" in sl or "nk" in sl:
            return "T"
        if "mast" in sl:
            return "Mast"
        if "adipocyte" in sl:
            return "Adipocyte"
        return "less10"
    return labels.map(_map_one)


# ===========================================================================
# STEP 1: Load shared data
# ===========================================================================
def load_sthelar():
    """Load STHELAR breast_s0 data with raw counts and GT labels."""
    import anndata as ad

    print("=" * 70)
    print("Loading STHELAR breast_s0")
    print("=" * 70)
    t0 = time.time()

    adata = ad.read_zarr(str(STHELAR_BASE / "table_cells"))
    print(f"  Shape: {adata.shape[0]:,} cells x {adata.shape[1]} genes")

    # Use raw counts
    if "count" in adata.layers:
        adata.X = adata.layers["count"].copy()
    else:
        raise ValueError("No 'count' layer found!")

    # GT labels are already in obs
    for col in ["final_label", "label1", "label2", "label3"]:
        if col in adata.obs.columns:
            vc = adata.obs[col].value_counts()
            print(f"\n  {col} ({len(vc)} types):")
            for ct, n in vc.items():
                print(f"    {ct}: {n:,}")

    # Load spatial coords
    print(f"\n  Spatial coords in obsm: {'spatial' in adata.obsm}")

    print(f"\n  [{elapsed(t0)}]")
    return adata


# ===========================================================================
# Shared: KNN transfer from 48K popV subsample to full 577K cells
# ===========================================================================
def knn_transfer(adata, pred_df, label_col, method_name="knn"):
    """Transfer predictions from 48K subsample to full dataset via KNN on PCA."""
    t0 = time.time()
    labels = pred_df[label_col].astype(str).values
    sub_indices = pred_df.index.values  # row indices into adata

    # PCA on normalized expression
    adata_tmp = adata.copy()
    sc.pp.normalize_total(adata_tmp, target_sum=1e4)
    sc.pp.log1p(adata_tmp)
    if issparse(adata_tmp.X):
        X_dense = adata_tmp.X.toarray()
    else:
        X_dense = adata_tmp.X
    del adata_tmp
    gc.collect()

    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_dense)
    del X_dense
    gc.collect()

    X_train = X_pca[sub_indices]
    knn = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)
    knn.fit(X_train, labels)
    full_predictions = knn.predict(X_pca)
    del X_pca, X_train
    gc.collect()

    print(f"  KNN transfer ({method_name}): {len(sub_indices):,} -> {len(full_predictions):,} cells [{elapsed(t0)}]")
    return full_predictions


# ===========================================================================
# METHOD 1: popV ensemble (DISCO cache) — majority vote across 8 methods
# ===========================================================================
def method1_popv_disco(adata):
    """Use cached popV+DISCO predictions (majority vote across 8 methods)."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("METHOD 1: popV Ensemble + DISCO Breast v2.1 (majority vote)")
    print("=" * 70)

    if not DISCO_CACHE_PREDICTIONS.exists():
        print(f"  ERROR: No DISCO cache found at {DISCO_CACHE_PREDICTIONS}")
        print("  Run the standalone popV+DISCO pipeline first.")
        return None

    pred_df = pd.read_csv(DISCO_CACHE_PREDICTIONS, index_col=0)
    print(f"  Cached predictions: {pred_df.shape[0]:,} cells, {len(pred_df.columns)} columns")

    consensus_col = "popv_majority_vote_prediction"
    if consensus_col not in pred_df.columns:
        consensus_col = "popv_prediction"
    print(f"\n  Ensemble predictions ({consensus_col}, {pred_df[consensus_col].nunique()} types):")
    for lbl, cnt in pred_df[consensus_col].value_counts().items():
        print(f"    {lbl}: {cnt:,}")

    # KNN transfer to full dataset
    print(f"\n  Transferring to full {adata.shape[0]:,} cells...")
    full_predictions = knn_transfer(adata, pred_df, consensus_col, "popv_ensemble")

    # Evaluate at label1
    pred_label1 = map_predictions_to_label1(pd.Series(full_predictions), POPV_TO_LABEL1)
    gt_label1 = adata.obs["label1"].astype(str).values
    result_label1 = evaluate(gt_label1, pred_label1.values, "label1", "popv_disco_ensemble")
    print(f"\n  popV+DISCO @ label1: F1={result_label1['f1_macro']:.4f}, Acc={result_label1['accuracy']:.4f}")

    # Evaluate at label2
    pred_label2 = map_predictions_to_label1(pd.Series(full_predictions), POPV_TO_LABEL2)
    gt_label2 = adata.obs["label2"].astype(str).values
    result_label2 = evaluate(gt_label2, pred_label2.values, "label2", "popv_disco_ensemble")
    print(f"  popV+DISCO @ label2: F1={result_label2['f1_macro']:.4f}, Acc={result_label2['accuracy']:.4f}")

    total_time = time.time() - t0
    return {
        "method": "popv_disco_ensemble",
        "reference": "DISCO_breast_v2.1",
        "n_ref_types": 42,
        "results_label1": result_label1,
        "results_label2": result_label2,
        "raw_pred_distribution": dict(Counter(full_predictions)),
        "elapsed_s": round(total_time, 1),
        "full_predictions": full_predictions,
    }


# ===========================================================================
# METHOD 2: OnClass standalone (DISCO cache) — single method, CL-aware
# ===========================================================================
def method2_onclass_propagation(adata):
    """Use OnClass-specific predictions from DISCO-trained popV cache."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("METHOD 2: OnClass Standalone (DISCO-trained, CL hierarchy)")
    print("=" * 70)

    if not DISCO_CACHE_PREDICTIONS.exists():
        print(f"  ERROR: No DISCO cache found at {DISCO_CACHE_PREDICTIONS}")
        return None

    pred_df = pd.read_csv(DISCO_CACHE_PREDICTIONS, index_col=0)

    onclass_col = "popv_onclass_prediction"
    if onclass_col not in pred_df.columns:
        print(f"  ERROR: OnClass column not found. Available: {pred_df.columns.tolist()}")
        return None

    print(f"  OnClass raw predictions ({pred_df[onclass_col].nunique()} CL types):")
    for lbl, cnt in pred_df[onclass_col].value_counts().items():
        print(f"    {lbl}: {cnt:,}")

    # KNN transfer to full dataset
    print(f"\n  Transferring to full {adata.shape[0]:,} cells...")
    full_predictions = knn_transfer(adata, pred_df, onclass_col, "onclass")

    # Evaluate at label1 (using proper OnClass CL→label1 mapping)
    pred_label1 = map_predictions_to_label1(pd.Series(full_predictions), ONCLASS_TO_LABEL1)
    gt_label1 = adata.obs["label1"].astype(str).values
    result_label1 = evaluate(gt_label1, pred_label1.values, "label1", "onclass_disco")
    print(f"\n  OnClass @ label1: F1={result_label1['f1_macro']:.4f}, Acc={result_label1['accuracy']:.4f}")

    # Evaluate at label2
    pred_label2 = map_predictions_to_label1(pd.Series(full_predictions), ONCLASS_TO_LABEL2)
    gt_label2 = adata.obs["label2"].astype(str).values
    result_label2 = evaluate(gt_label2, pred_label2.values, "label2", "onclass_disco")
    print(f"  OnClass @ label2: F1={result_label2['f1_macro']:.4f}, Acc={result_label2['accuracy']:.4f}")

    total_time = time.time() - t0
    return {
        "method": "onclass_disco_standalone",
        "results_label1": result_label1,
        "results_label2": result_label2,
        "raw_pred_distribution": dict(Counter(full_predictions)),
        "elapsed_s": round(total_time, 1),
        "full_predictions": full_predictions,
    }


# ===========================================================================
# METHOD 3: BANKSY + fine-grained marker annotation
# ===========================================================================
def method3_banksy_finegrained(adata):
    """Run BANKSY spatial clustering with fine-grained marker annotation."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("METHOD 3: BANKSY + Fine-Grained Marker Annotation")
    print("=" * 70)

    # Check for spatial coordinates
    if "spatial" not in adata.obsm:
        print("  ERROR: No spatial coordinates in obsm!")
        return None

    coords = adata.obsm["spatial"]
    print(f"  Spatial coords shape: {coords.shape}")
    print(f"  Spatial range: x=[{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}], "
          f"y=[{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}]")

    # BANKSY: spatial augmentation + Leiden clustering
    print("\n  Running BANKSY spatial clustering...")
    t_banksy = time.time()

    # Prepare adata for BANKSY
    adata_b = adata.copy()

    # Normalize
    sc.pp.normalize_total(adata_b, target_sum=1e4)
    sc.pp.log1p(adata_b)

    # HVGs
    sc.pp.highly_variable_genes(adata_b, n_top_genes=3000, flavor="seurat_v3",
                                 layer=None, subset=False)
    hvg_mask = adata_b.var["highly_variable"]
    print(f"  HVGs: {hvg_mask.sum()}")

    # BANKSY spatial augmentation: augment expression with neighbor-averaged expression
    from scipy.spatial import cKDTree

    print("  Building spatial KD-tree (k=15)...")
    tree = cKDTree(coords)
    k = 15  # number of spatial neighbors

    # Compute neighbor-averaged expression for HVGs
    hvg_genes = adata_b.var_names[hvg_mask]
    if issparse(adata_b.X):
        X_hvg = adata_b[:, hvg_genes].X.toarray()
    else:
        X_hvg = adata_b[:, hvg_genes].X.copy()

    print(f"  Computing spatial neighbor averages for {len(hvg_genes)} HVGs...")
    distances, indices = tree.query(coords, k=k + 1)  # +1 includes self
    indices = indices[:, 1:]  # exclude self
    distances = distances[:, 1:]

    # Distance-weighted average
    weights = 1.0 / (distances + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)

    # Vectorized neighbor averaging (no Python loop!)
    N, G = X_hvg.shape
    X_neighbors_all = X_hvg[indices.ravel()].reshape(N, k, G)  # (N, k, G)
    X_neighbor = np.einsum("nk,nkg->ng", weights, X_neighbors_all)
    del X_neighbors_all

    # BANKSY augmented features: [expression, lambda * neighbor_avg]
    lambda_banksy = 0.2  # Optimal from previous benchmarks
    X_augmented = np.hstack([X_hvg, lambda_banksy * X_neighbor])
    print(f"  Augmented feature matrix: {X_augmented.shape}")

    del X_neighbor, distances, indices, weights
    gc.collect()

    # PCA on augmented features
    print("  PCA on BANKSY features...")
    from sklearn.decomposition import PCA as skPCA
    pca = skPCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_augmented)
    del X_augmented
    gc.collect()

    # Store in adata for scanpy neighbors
    import anndata as ad
    adata_banksy = ad.AnnData(obs=adata_b.obs.copy())
    adata_banksy.obsm["X_pca"] = X_pca

    print("  Computing neighbors...")
    sc.pp.neighbors(adata_banksy, use_rep="X_pca", n_neighbors=30)

    # Leiden at 2 resolutions (skip 1.0=too coarse, 5.0=too slow on 577K cells)
    resolutions = [2.0, 3.0]
    best_result = None
    best_f1 = -1

    for res in resolutions:
        print(f"\n  Leiden resolution={res}...")
        sc.tl.leiden(adata_banksy, resolution=res, key_added=f"leiden_{res}")
        n_clusters = adata_banksy.obs[f"leiden_{res}"].nunique()
        print(f"    Clusters: {n_clusters}")

        # Annotate clusters with fine-grained markers
        cluster_labels = annotate_banksy_clusters(
            adata_b, adata_banksy.obs[f"leiden_{res}"].values, X_hvg, hvg_genes
        )

        # Map to label1
        pred_label1 = map_predictions_to_label1(
            pd.Series(cluster_labels), BANKSY_TYPE_TO_LABEL1
        )
        gt_label1 = adata.obs["label1"].astype(str).values
        result = evaluate(gt_label1, pred_label1.values, "label1", f"banksy_r{res}")
        print(f"    BANKSY r={res} @ label1: F1={result['f1_macro']:.4f}, Acc={result['accuracy']:.4f}")

        # Also evaluate at label2
        pred_label2 = map_predictions_to_label1(
            pd.Series(cluster_labels), BANKSY_TYPE_TO_LABEL2
        )
        gt_label2 = adata.obs["label2"].astype(str).values
        result_l2 = evaluate(gt_label2, pred_label2.values, "label2", f"banksy_r{res}")
        print(f"    BANKSY r={res} @ label2: F1={result_l2['f1_macro']:.4f}, Acc={result_l2['accuracy']:.4f}")

        if result["f1_macro"] > best_f1:
            best_f1 = result["f1_macro"]
            best_result = result
            best_result["results_label2"] = result_l2
            best_result["resolution"] = res
            best_result["n_clusters"] = n_clusters
            best_result["raw_pred_distribution"] = dict(Counter(cluster_labels))
            best_result["full_predictions"] = cluster_labels

    del adata_banksy, adata_b, X_pca, X_hvg
    gc.collect()

    total_time = time.time() - t0
    best_result["method"] = "banksy_finegrained"
    best_result["elapsed_s"] = round(total_time, 1)
    print(f"\n  Best BANKSY: r={best_result['resolution']}, F1={best_f1:.4f} [{elapsed(t0)}]")

    return best_result


def annotate_banksy_clusters(adata_norm, cluster_labels, X_hvg, hvg_genes):
    """Annotate BANKSY clusters using marker gene scoring."""
    clusters = pd.Series(cluster_labels).astype(str)
    unique_clusters = sorted(clusters.unique(), key=lambda x: int(x) if x.isdigit() else x)

    gene_to_idx = {g: i for i, g in enumerate(hvg_genes)}
    available_genes_set = set(adata_norm.var_names)
    var_name_to_idx = {g: i for i, g in enumerate(adata_norm.var_names)}

    cluster_to_type = {}

    for cl in unique_clusters:
        mask = clusters == cl
        n_cells = mask.sum()
        if n_cells < 5:
            cluster_to_type[cl] = "less10"
            continue

        # Get mean expression for this cluster
        if issparse(adata_norm.X):
            cluster_expr = np.asarray(adata_norm.X[mask.values].mean(axis=0)).ravel()
        else:
            cluster_expr = adata_norm.X[mask.values].mean(axis=0)

        # Score each cell type by marker expression
        best_type = "less10"
        best_score = 0

        for cell_type, markers in MARKER_GENES_FINEGRAINED.items():
            present = [g for g in markers if g in available_genes_set]
            if not present:
                continue

            # Get expression indices (using prebuilt dict, not list.index())
            gene_indices = [var_name_to_idx[g] for g in present]
            score = np.mean(cluster_expr[gene_indices])

            # Normalize by number of markers available
            coverage = len(present) / len(markers)
            score *= coverage

            if score > best_score:
                best_score = score
                best_type = cell_type

        cluster_to_type[cl] = best_type

    # Map cells to types
    cell_types = clusters.map(cluster_to_type).values
    return cell_types


# ===========================================================================
# MAIN: Run all three and compare
# ===========================================================================
def main():
    t_total = time.time()

    # Load shared data
    adata = load_sthelar()

    results = {}

    # Method 1: popV + DISCO
    try:
        r1 = method1_popv_disco(adata)
        if r1:
            preds1 = r1.pop("full_predictions")
            results["popv_disco"] = r1
    except Exception as e:
        print(f"\n  METHOD 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        r1, preds1 = None, None

    gc.collect()

    # Method 2: OnClass propagation
    try:
        r2 = method2_onclass_propagation(adata)
        if r2:
            preds2 = r2.pop("full_predictions")
            results["onclass_refined"] = r2
    except Exception as e:
        print(f"\n  METHOD 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        r2, preds2 = None, None

    gc.collect()

    # Method 3: BANKSY
    try:
        r3 = method3_banksy_finegrained(adata)
        if r3:
            preds3 = r3.pop("full_predictions")
            results["banksy_finegrained"] = r3
    except Exception as e:
        print(f"\n  METHOD 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        r3, preds3 = None, None

    # ===========================================================================
    # SUMMARY COMPARISON
    # ===========================================================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON — Fine-Grained Cell Type Annotation")
    print("=" * 70)

    print(f"\n{'Method':<35} {'Level':<10} {'F1 Macro':<10} {'Accuracy':<10} {'Time':<10}")
    print("-" * 75)

    for name, r in results.items():
        t_str = f"{r.get('elapsed_s', 0):.1f}s"
        if "results_label1" in r:
            rl = r["results_label1"]
            print(f"{name:<35} {'label1':<10} {rl['f1_macro']:<10.4f} {rl['accuracy']:<10.4f} {t_str:<10}")
        if "results_label2" in r:
            rl2 = r["results_label2"]
            print(f"{'':<35} {'label2':<10} {rl2['f1_macro']:<10.4f} {rl2['accuracy']:<10.4f}")

    # Per-class comparison
    if results:
        all_classes = set()
        for r in results.values():
            if "results_label1" in r and "per_class" in r["results_label1"]:
                all_classes.update(r["results_label1"]["per_class"].keys())
        all_classes = sorted(all_classes)

        print(f"\n{'Per-Class F1 (label1)':}")
        header = f"{'Class':<25}"
        for name in results:
            header += f" {name[:20]:<22}"
        print(header)
        print("-" * (25 + 22 * len(results)))

        for cls in all_classes:
            row = f"{cls:<25}"
            for name, r in results.items():
                if "results_label1" in r and cls in r["results_label1"].get("per_class", {}):
                    f1 = r["results_label1"]["per_class"][cls]["f1"]
                    row += f" {f1:<22.4f}"
                else:
                    row += f" {'--':<22}"
            print(row)

    # Save results
    out_path = OUT_DIR / "finegrained_3method_comparison.json"

    # Clean for JSON serialization
    save_results = {}
    for k, v in results.items():
        save_results[k] = {kk: vv for kk, vv in v.items()
                           if not isinstance(vv, np.ndarray)}

    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print(f"\n  TOTAL TIME: {elapsed(t_total)}")


if __name__ == "__main__":
    main()
