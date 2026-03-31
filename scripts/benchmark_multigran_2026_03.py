#!/usr/bin/env python3
"""Multi-granularity annotation benchmark: coarse (4), medium (~9), fine (17).

Tests the winning methods at all granularity levels on rep1.
"""

import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import celltypist
import numpy as np
import pandas as pd
import scanpy as sc
from celltypist import models as ct_models
from loguru import logger
from scipy.sparse import issparse
from scipy.spatial import cKDTree
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.annotation_benchmark_2026_03 import load_xenium_adata, preprocess_adata
from scripts.banksy_full_benchmark import MARKERS_DEFAULT

OUTPUT_DIR = Path("pipeline_output/annotation_benchmark_2026_03")

from dapidl.pipeline.components.annotators.mapping import map_to_broad_category

# ── GRANULARITY DEFINITIONS ──────────────────────────────────────────────────

GT_FINE_TO_COARSE = {
    "B_Cells": "Immune", "CD4+_T_Cells": "Immune", "CD8+_T_Cells": "Immune",
    "DCIS_1": "Epithelial", "DCIS_2": "Epithelial", "Endothelial": "Endothelial",
    "IRF7+_DCs": "Immune", "Invasive_Tumor": "Epithelial", "LAMP3+_DCs": "Immune",
    "Macrophages_1": "Immune", "Macrophages_2": "Immune", "Mast_Cells": "Immune",
    "Myoepi_ACTA2+": "Epithelial", "Myoepi_KRT15+": "Epithelial",
    "Perivascular-Like": "Stromal", "Prolif_Invasive_Tumor": "Epithelial", "Stromal": "Stromal",
}
GT_FINE_TO_MEDIUM = {
    "B_Cells": "B_Cell", "CD4+_T_Cells": "T_Cell", "CD8+_T_Cells": "T_Cell",
    "DCIS_1": "Epithelial_Tumor", "DCIS_2": "Epithelial_Tumor",
    "Endothelial": "Endothelial", "IRF7+_DCs": "Myeloid",
    "Invasive_Tumor": "Epithelial_Tumor", "LAMP3+_DCs": "Myeloid",
    "Macrophages_1": "Myeloid", "Macrophages_2": "Myeloid", "Mast_Cells": "Myeloid",
    "Myoepi_ACTA2+": "Epithelial_Basal", "Myoepi_KRT15+": "Epithelial_Basal",
    "Perivascular-Like": "Stromal_Pericyte", "Prolif_Invasive_Tumor": "Epithelial_Tumor",
    "Stromal": "Stromal_Fibroblast",
}

COARSE_CLASSES = sorted(set(GT_FINE_TO_COARSE.values()))
MEDIUM_CLASSES = sorted(set(GT_FINE_TO_MEDIUM.values()))
FINE_CLASSES = sorted(GT_FINE_TO_COARSE.keys())


def map_pred_to_medium(pred):
    p = pred.lower()
    if "t cell" in p or "cd4" in p or "cd8" in p or "treg" in p: return "T_Cell"
    if "b cell" in p or "b_cell" in p: return "B_Cell"
    if "plasma" in p: return "B_Cell"
    if "macrophage" in p or "monocyte" in p or "dendritic" in p or "mast" in p or "nk " in p or "nk_" in p: return "Myeloid"
    if "myoepithelial" in p or "basal" in p or "krt15" in p or "krt14" in p: return "Epithelial_Basal"
    if "epithelial" in p or "luminal" in p or "tumor" in p or "dcis" in p or "cancer" in p or "mammary" in p: return "Epithelial_Tumor"
    if "fibroblast" in p or "stromal" in p or "smooth muscle" in p or "col1a" in p: return "Stromal_Fibroblast"
    if "pericyte" in p or "perivascular" in p: return "Stromal_Pericyte"
    if "endothelial" in p or "vascular" in p or "pecam" in p: return "Endothelial"
    coarse = map_to_broad_category(pred)
    return {"Epithelial": "Epithelial_Tumor", "Immune": "T_Cell",
            "Stromal": "Stromal_Fibroblast", "Endothelial": "Endothelial"}.get(coarse, "Unknown")


def compute_metrics_at(gt_level, preds, classes):
    mask = (gt_level != "Unknown") & (preds != "Unknown")
    yt, yp = gt_level[mask], preds[mask]
    if len(yt) == 0: return {"f1_macro": 0, "accuracy": 0}
    acc = accuracy_score(yt, yp)
    f1m = f1_score(yt, yp, average="macro", zero_division=0, labels=classes)
    prec, rec, f1, sup = precision_recall_fscore_support(yt, yp, labels=classes, zero_division=0)
    pc = {c: {"f1": round(float(f1[i]),3), "support": int(sup[i])} for i, c in enumerate(classes)}
    return {"f1_macro": round(float(f1m),4), "accuracy": round(float(acc),4), "per_class": pc}


# ── FINE-GRAINED MARKERS ────────────────────────────────────────────────────

MARKERS_FINE = {
    "T_Cell": {"positive": ["CD3D","CD3E","TRAC"], "negative": ["CD14","MS4A1"]},
    "CD4_T": {"positive": ["CD3D","CD4","IL7R"], "negative": ["CD8A"]},
    "CD8_T": {"positive": ["CD3D","CD8A","CD8B","GZMB"], "negative": ["CD4"]},
    "B_Cell": {"positive": ["CD19","CD79A","MS4A1"], "negative": ["CD3D"]},
    "Macrophage": {"positive": ["CD68","CD163","CSF1R"], "negative": ["CD3D"]},
    "DC": {"positive": ["ITGAX","CD1C","CLEC9A","FLT3"], "negative": ["CD14"]},
    "Mast": {"positive": ["KIT","TPSAB1","CPA3"], "negative": ["CD3D"]},
    "NK": {"positive": ["NCAM1","NKG7","GNLY"], "negative": ["CD3D"]},
    "Epithelial": {"positive": ["EPCAM","KRT8","KRT18","KRT19"], "negative": ["ACTA2"]},
    "Myoepithelial": {"positive": ["KRT14","KRT5","ACTA2","TP63"], "negative": ["KRT8"]},
    "Fibroblast": {"positive": ["COL1A1","COL1A2","DCN","FAP"], "negative": ["EPCAM","PTPRC"]},
    "Pericyte": {"positive": ["ACTA2","PDGFRB","RGS5"], "negative": ["EPCAM","COL1A1"]},
    "Endothelial": {"positive": ["PECAM1","VWF","CLDN5","KDR"], "negative": ["EPCAM","COL1A1"]},
    "Plasma": {"positive": ["SDC1","MZB1","JCHAIN"], "negative": ["MS4A1"]},
}
FINE_TO_MED = {
    "T_Cell": "T_Cell", "CD4_T": "T_Cell", "CD8_T": "T_Cell",
    "B_Cell": "B_Cell", "Plasma": "B_Cell",
    "Macrophage": "Myeloid", "DC": "Myeloid", "Mast": "Myeloid", "NK": "Myeloid",
    "Epithelial": "Epithelial_Tumor", "Myoepithelial": "Epithelial_Basal",
    "Fibroblast": "Stromal_Fibroblast", "Pericyte": "Stromal_Pericyte",
    "Endothelial": "Endothelial",
}
FINE_TO_COARSE = {
    "T_Cell": "Immune", "CD4_T": "Immune", "CD8_T": "Immune",
    "B_Cell": "Immune", "Plasma": "Immune",
    "Macrophage": "Immune", "DC": "Immune", "Mast": "Immune", "NK": "Immune",
    "Epithelial": "Epithelial", "Myoepithelial": "Epithelial",
    "Fibroblast": "Stromal", "Pericyte": "Stromal",
    "Endothelial": "Endothelial",
}


def score_cells(adata, markers):
    """Score each cell with marker signatures."""
    gene_names = list(adata.var_names)
    expr = np.asarray(adata.X.toarray() if issparse(adata.X) else adata.X)
    scores = {}
    for ct, m in markers.items():
        pos = [g for g in m["positive"] if g in gene_names]
        neg = [g for g in m.get("negative", []) if g in gene_names]
        if not pos: scores[ct] = np.zeros(len(adata)); continue
        sc_ = np.asarray(expr[:, [gene_names.index(g) for g in pos]].mean(axis=1)).ravel()
        if neg: sc_ -= 0.5 * np.asarray(expr[:, [gene_names.index(g) for g in neg]].mean(axis=1)).ravel()
        scores[ct] = sc_
    mat = np.column_stack([scores[ct] for ct in scores])
    names = list(scores.keys())
    return np.array([names[i] for i in mat.argmax(axis=1)])


def label_banksy_clusters_fine(adata, cluster_ids, markers):
    """Label BANKSY clusters with fine markers."""
    gene_names = list(adata.var_names)
    expr = np.asarray(adata.X)
    labels = {}
    for cl in set(cluster_ids.astype(str)):
        mask = cluster_ids.astype(str) == cl
        cl_expr = expr[mask]
        best, best_sc = "Unknown", -999
        for ct, m in markers.items():
            pos = [g for g in m["positive"] if g in gene_names]
            neg = [g for g in m.get("negative", []) if g in gene_names]
            if not pos: continue
            sc_ = cl_expr[:, [gene_names.index(g) for g in pos]].mean()
            if neg: sc_ -= 0.5 * cl_expr[:, [gene_names.index(g) for g in neg]].mean()
            if sc_ > best_sc: best_sc, best = sc_, ct
        labels[cl] = best
    return np.array([labels.get(str(c), "Unknown") for c in cluster_ids])


def main():
    logger.info("=" * 80)
    logger.info("MULTI-GRANULARITY BENCHMARK (coarse / medium / fine)")
    logger.info("=" * 80)

    adata_raw = load_xenium_adata("rep1")
    adata_pp = preprocess_adata(adata_raw)
    gt_fine_raw = np.array(adata_raw.obs["gt_fine"].values)
    valid = np.isin(gt_fine_raw, FINE_CLASSES)
    adata_pp = adata_pp[valid].copy()
    adata_raw_v = adata_raw[valid].copy()
    gt_fine = gt_fine_raw[valid]
    gt_medium = np.array([GT_FINE_TO_MEDIUM[g] for g in gt_fine])
    gt_coarse = np.array([GT_FINE_TO_COARSE[g] for g in gt_fine])
    logger.info(f"Cells: {valid.sum()}, Coarse={len(COARSE_CLASSES)}, Medium={len(MEDIUM_CLASSES)}, Fine={len(FINE_CLASSES)}")

    R = {}

    # ── 1. CellTypist models ────────────────────────────────────────
    for model_name in ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl", "Immune_All_Low.pkl",
                       "Developing_Human_Organs.pkl", "Pan_Fetal_Human.pkl",
                       "Healthy_Human_Liver.pkl", "Human_Lung_Atlas.pkl"]:
        short = model_name.replace(".pkl","")
        logger.info(f"  CT: {short}")
        try:
            fp = celltypist.annotate(adata_pp, model=ct_models.Model.load(model_name), majority_voting=False).to_adata().obs["predicted_labels"].astype(str).values
            cp = np.array([map_to_broad_category(p) for p in fp])
            mp = np.array([map_pred_to_medium(p) for p in fp])
            for lv, gt_lv, pr, cl in [("coarse",gt_coarse,cp,COARSE_CLASSES),("medium",gt_medium,mp,MEDIUM_CLASSES),("fine",gt_fine,fp,FINE_CLASSES)]:
                R[f"ct_{short}_{lv}"] = compute_metrics_at(gt_lv, pr, cl)
                logger.info(f"    {lv}: F1={R[f'ct_{short}_{lv}']['f1_macro']:.3f}")
        except Exception as e:
            logger.error(f"    FAIL: {e}")

    # ── 2. SCINA fine markers ────────────────────────────────────────
    logger.info(f"  SCINA fine markers")
    sf = score_cells(adata_pp, MARKERS_FINE)
    sm = np.array([FINE_TO_MED.get(p,"Unknown") for p in sf])
    scc = np.array([FINE_TO_COARSE.get(p,"Unknown") for p in sf])
    for lv, gt_lv, pr, cl in [("coarse",gt_coarse,scc,COARSE_CLASSES),("medium",gt_medium,sm,MEDIUM_CLASSES),("fine",gt_fine,sf,FINE_CLASSES)]:
        R[f"scina_fine_{lv}"] = compute_metrics_at(gt_lv, pr, cl)
        logger.info(f"    {lv}: F1={R[f'scina_fine_{lv}']['f1_macro']:.3f}")

    # ── 3. BANKSY (best config: k=10 λ=0.2) at multiple resolutions ─
    logger.info(f"  BANKSY spatial clustering...")
    from banksy.initialize_banksy import initialize_banksy
    from banksy.embed_banksy import generate_banksy_matrix
    from banksy.cluster_methods import run_Leiden_partition
    from banksy_utils.umap_pca import pca_umap

    a = adata_raw_v.copy()
    if issparse(a.X): a.X = a.X.toarray()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=min(2000,a.n_vars), subset=False)
    ah = a[:, a.var["highly_variable"]].copy()
    ah.obs["xcoord"] = a.obs["x_centroid"].values
    ah.obs["ycoord"] = a.obs["y_centroid"].values
    ah.obsm["xy_coord"] = np.column_stack([ah.obs["xcoord"].values.astype(float), ah.obs["ycoord"].values.astype(float)])

    bd = initialize_banksy(adata=ah, coord_keys=("xcoord","ycoord","xy_coord"),
                           num_neighbours=10, nbr_weight_decay="scaled_gaussian", max_m=1,
                           plt_edge_hist=False, plt_nbr_weights=False, plt_theta=False)
    bd, _ = generate_banksy_matrix(adata=ah, banksy_dict=bd, lambda_list=[0.2], max_m=1, plot_std=False, verbose=False)
    pca_umap(banksy_dict=bd, pca_dims=[20], plt_remaining_var=False, add_umap=False)

    for res in [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]:
        logger.info(f"    BANKSY r={res}...")
        rdf, _ = run_Leiden_partition(banksy_dict=bd, resolutions=[res], num_nn=50,
                                      num_iterations=-1, partition_seed=1234, match_labels=False, verbose=False)
        for _, row in rdf.iterrows():
            cids = row["labels"].dense
            ncl = row["num_labels"]

            # Fine markers labeling
            fl = label_banksy_clusters_fine(a, cids, MARKERS_FINE)
            ml = np.array([FINE_TO_MED.get(p,"Unknown") for p in fl])
            ccl = np.array([FINE_TO_COARSE.get(p,"Unknown") for p in fl])

            # scType coarse labeling
            sctype_cl = label_banksy_clusters_fine(a, cids, MARKERS_DEFAULT)

            for lv, gt_lv, pr, cl, tag in [
                ("coarse",gt_coarse,ccl,COARSE_CLASSES,"fine_mk"),
                ("coarse",gt_coarse,sctype_cl,COARSE_CLASSES,"sctype"),
                ("medium",gt_medium,ml,MEDIUM_CLASSES,"fine_mk"),
                ("fine",gt_fine,fl,FINE_CLASSES,"fine_mk"),
            ]:
                k = f"banksy_r{res}_{tag}_{lv}"
                R[k] = compute_metrics_at(gt_lv, pr, cl)
                R[k]["n_clusters"] = int(ncl)
                logger.info(f"      r={res} {ncl}cl {tag} {lv}: F1={R[k]['f1_macro']:.3f}")

    # ── SUMMARY ──────────────────────────────────────────────────────
    logger.info(f"\n{'='*100}")
    for level in ["coarse", "medium", "fine"]:
        lr = sorted([(k,v) for k,v in R.items() if k.endswith(f"_{level}") and "f1_macro" in v],
                    key=lambda x: -x[1]["f1_macro"])
        logger.info(f"\n{'='*40} {level.upper()} {'='*40}")
        logger.info(f"{'Method':<55s} {'F1':>6s} {'Acc':>6s}")
        logger.info("-"*70)
        for k,v in lr[:15]:
            logger.info(f"{k:<55s} {v['f1_macro']:>6.3f} {v.get('accuracy',0):>6.3f}")

    with open(OUTPUT_DIR / "multi_granularity_rep1.json", "w") as f:
        json.dump({k: {kk: (float(vv) if isinstance(vv,(np.floating,np.integer)) else vv) for kk,vv in v.items()}
                   for k,v in R.items()}, f, indent=2)
    logger.info(f"\nSaved to {OUTPUT_DIR}/multi_granularity_rep1.json")


if __name__ == "__main__":
    main()
