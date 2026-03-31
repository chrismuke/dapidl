#!/usr/bin/env python3
"""Cell Ontology-normalized annotation benchmark.

Maps ALL predictions and GT to CL IDs, then evaluates at multiple hierarchy levels:
- CL_EXACT: Exact CL ID match
- CL_PARENT: Match at parent level (e.g., CL:0002325 luminal ≈ CL:0000066 epithelial)
- BROAD: 4-class (via CL hierarchy)
- MEDIUM: ~10-class (via CL hierarchy)
"""

import json, sys, time, warnings
from collections import Counter, defaultdict
from pathlib import Path

import celltypist, numpy as np, scanpy as sc
from celltypist import models as ct_models
from loguru import logger
from scipy.sparse import issparse
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.annotation_benchmark_2026_03 import load_xenium_adata, preprocess_adata
from scripts.benchmark_multigran_2026_03 import MARKERS_FINE

OUTPUT_DIR = Path("pipeline_output/annotation_benchmark_2026_03")

from dapidl.ontology.cl_mapper import CLMapper
from dapidl.ontology.cl_database import CL_TO_BROAD_CATEGORY, CL_TO_COARSE_CATEGORY, get_broad_category
from dapidl.ontology.annotator_mappings import (
    XENIUM_BREAST_GT_TO_CL, get_all_annotator_mappings,
)

# CL hierarchy rollup for evaluation
# Level 1: Broad (4 classes)
CL_TO_BROAD = {
    "CL:0000066": "Epithelial", "CL:0002325": "Epithelial", "CL:0000185": "Epithelial",
    "CL:0000646": "Epithelial", "CL:0002327": "Epithelial",
    "CL:0000624": "Immune", "CL:0000625": "Immune", "CL:0000236": "Immune",
    "CL:0000235": "Immune", "CL:0000784": "Immune", "CL:0001056": "Immune",
    "CL:0000451": "Immune", "CL:0000097": "Immune", "CL:0000623": "Immune",
    "CL:0000786": "Immune", "CL:0000084": "Immune", "CL:0000576": "Immune",
    "CL:0000057": "Stromal", "CL:0000669": "Stromal", "CL:0000192": "Stromal",
    "CL:0000186": "Stromal", "CL:0000136": "Stromal", "CL:0000499": "Stromal",
    "CL:0000115": "Endothelial", "CL:0002138": "Endothelial",
}

# Level 2: Medium (~12 classes)
CL_TO_MEDIUM = {
    "CL:0000066": "Epithelial", "CL:0002325": "Luminal_Epithelial", "CL:0000185": "Myoepithelial",
    "CL:0000646": "Basal_Epithelial", "CL:0002327": "Luminal_Epithelial",
    "CL:0000624": "CD4_T_Cell", "CL:0000625": "CD8_T_Cell", "CL:0000084": "T_Cell",
    "CL:0000236": "B_Cell", "CL:0000786": "Plasma_Cell",
    "CL:0000235": "Macrophage", "CL:0000576": "Monocyte",
    "CL:0000784": "Dendritic_Cell", "CL:0001056": "Dendritic_Cell", "CL:0000451": "Dendritic_Cell",
    "CL:0000097": "Mast_Cell", "CL:0000623": "NK_Cell",
    "CL:0000057": "Fibroblast", "CL:0000669": "Pericyte",
    "CL:0000192": "Smooth_Muscle", "CL:0000186": "Myofibroblast",
    "CL:0000136": "Adipocyte", "CL:0000499": "Stromal",
    "CL:0000115": "Endothelial", "CL:0002138": "Lymphatic_Endothelial",
}


def map_labels_to_cl(labels, mapper):
    """Map array of labels to CL IDs."""
    return np.array([mapper.map(str(l)) for l in labels])


def cl_to_level(cl_ids, level_map, default="Unknown"):
    """Roll up CL IDs to a hierarchy level."""
    return np.array([level_map.get(cl, default) for cl in cl_ids])


def evaluate_at_level(gt_level, pred_level, level_name):
    """Compute metrics at a given hierarchy level."""
    mask = (gt_level != "Unknown") & (pred_level != "Unknown") & (gt_level != "UNMAPPED") & (pred_level != "UNMAPPED")
    yt, yp = gt_level[mask], pred_level[mask]
    if len(yt) == 0:
        return {"f1_macro": 0, "accuracy": 0, "n_cells": 0, "level": level_name}
    classes = sorted(set(yt) | set(yp) - {"Unknown", "UNMAPPED"})
    f1m = f1_score(yt, yp, average="macro", zero_division=0, labels=classes)
    acc = accuracy_score(yt, yp)
    prec, rec, f1, sup = precision_recall_fscore_support(yt, yp, labels=classes, zero_division=0)
    pc = {c: {"f1": round(float(f1[i]),3), "p": round(float(prec[i]),3),
              "r": round(float(rec[i]),3), "n": int(sup[i])} for i, c in enumerate(classes)}
    return {"f1_macro": round(float(f1m),4), "accuracy": round(float(acc),4),
            "n_cells": int(len(yt)), "level": level_name, "per_class": pc,
            "n_classes_gt": len(set(yt)), "n_classes_pred": len(set(yp))}


def main():
    logger.info("=" * 80)
    logger.info("CELL ONTOLOGY-NORMALIZED BENCHMARK")
    logger.info("=" * 80)

    # Initialize mapper with ALL curated mappings
    mapper = CLMapper(
        annotator_mappings=get_all_annotator_mappings(),
        ground_truth_mappings=XENIUM_BREAST_GT_TO_CL,
    )

    adata_raw = load_xenium_adata("rep1")
    adata_pp = preprocess_adata(adata_raw)
    gt_fine = np.array(adata_raw.obs["gt_fine"].values)

    # Filter valid cells
    valid = ~np.isin(gt_fine, ["Unlabeled", "Hybrid", "Stromal_&_T_Cell_Hybrid", "T_Cell_&_Tumor_Hybrid"])
    adata_pp = adata_pp[valid].copy()
    adata_raw_v = adata_raw[valid].copy()
    gt_fine = gt_fine[valid]
    logger.info(f"Cells: {len(adata_pp)}")

    # Map GT to CL IDs
    gt_cl = map_labels_to_cl(gt_fine, mapper)
    gt_broad = cl_to_level(gt_cl, CL_TO_BROAD)
    gt_medium = cl_to_level(gt_cl, CL_TO_MEDIUM)
    logger.info(f"GT CL IDs: {len(set(gt_cl))} unique, Broad: {len(set(gt_broad))}, Medium: {len(set(gt_medium))}")
    logger.info(f"GT unmapped: {(gt_cl == 'UNMAPPED').sum()}")

    results = {}

    # ── CellTypist models ────────────────────────────────────────────
    ct_models_list = ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl", "Immune_All_Low.pkl",
                      "Developing_Human_Organs.pkl", "Pan_Fetal_Human.pkl",
                      "Healthy_Human_Liver.pkl", "Human_Lung_Atlas.pkl"]
    for model_name in ct_models_list:
        short = model_name.replace(".pkl","")
        logger.info(f"\n  CellTypist: {short}")
        try:
            model = ct_models.Model.load(model_name)
            preds = celltypist.annotate(adata_pp, model=model, majority_voting=False).to_adata()
            fine_preds = preds.obs["predicted_labels"].astype(str).values

            pred_cl = map_labels_to_cl(fine_preds, mapper)
            pred_broad = cl_to_level(pred_cl, CL_TO_BROAD)
            pred_medium = cl_to_level(pred_cl, CL_TO_MEDIUM)

            unmapped_pct = (pred_cl == "UNMAPPED").sum() / len(pred_cl) * 100
            logger.info(f"    Unmapped: {unmapped_pct:.1f}%")

            for level_name, gt_lv, pred_lv in [
                ("CL_exact", gt_cl, pred_cl),
                ("CL_broad", gt_broad, pred_broad),
                ("CL_medium", gt_medium, pred_medium),
            ]:
                m = evaluate_at_level(gt_lv, pred_lv, level_name)
                results[f"ct_{short}_{level_name}"] = m
                logger.info(f"    {level_name}: F1={m['f1_macro']:.3f} Acc={m['accuracy']:.3f} ({m['n_classes_gt']}gt/{m.get('n_classes_pred',0)}pred)")
        except Exception as e:
            logger.error(f"    FAIL: {e}")

    # ── SCINA ────────────────────────────────────────────────────────
    logger.info(f"\n  SCINA fine markers")
    gene_names = list(adata_pp.var_names)
    expr = np.asarray(adata_pp.X.toarray() if issparse(adata_pp.X) else adata_pp.X)
    scores = {}
    for ct, m in MARKERS_FINE.items():
        pos = [g for g in m["positive"] if g in gene_names]
        neg = [g for g in m.get("negative", []) if g in gene_names]
        if not pos: scores[ct] = np.zeros(len(adata_pp)); continue
        sc_ = np.asarray(expr[:, [gene_names.index(g) for g in pos]].mean(axis=1)).ravel()
        if neg: sc_ -= 0.5 * np.asarray(expr[:, [gene_names.index(g) for g in neg]].mean(axis=1)).ravel()
        scores[ct] = sc_
    mat = np.column_stack([scores[ct] for ct in scores])
    ct_names = list(scores.keys())
    scina_fine = np.array([ct_names[i] for i in mat.argmax(axis=1)])
    scina_cl = map_labels_to_cl(scina_fine, mapper)
    scina_broad = cl_to_level(scina_cl, CL_TO_BROAD)
    scina_medium = cl_to_level(scina_cl, CL_TO_MEDIUM)

    for level_name, gt_lv, pred_lv in [
        ("CL_exact", gt_cl, scina_cl), ("CL_broad", gt_broad, scina_broad), ("CL_medium", gt_medium, scina_medium),
    ]:
        m = evaluate_at_level(gt_lv, pred_lv, level_name)
        results[f"scina_{level_name}"] = m
        logger.info(f"    {level_name}: F1={m['f1_macro']:.3f} Acc={m['accuracy']:.3f}")

    # ── BANKSY + CL-normalized labeling ──────────────────────────────
    logger.info(f"\n  BANKSY k=10 λ=0.2 + CL-normalized cluster labeling")
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

    for res in [0.5, 1.0, 2.0, 3.0, 5.0]:
        rdf, _ = run_Leiden_partition(banksy_dict=bd, resolutions=[res], num_nn=50,
                                      num_iterations=-1, partition_seed=1234, match_labels=False, verbose=False)
        for _, row in rdf.iterrows():
            cids = row["labels"].dense
            ncl = row["num_labels"]

            # Label each cluster with fine markers → CL IDs
            a_cl = a.copy()
            a_cl.obs["cluster"] = cids.astype(str)
            gene_names_a = list(a_cl.var_names)
            expr_a = np.asarray(a_cl.X)

            cluster_cl = {}
            for cl_id in sorted(set(cids.astype(str))):
                mask = a_cl.obs["cluster"] == cl_id
                cl_expr = expr_a[mask]
                best, best_sc = "Unknown", -999
                for ct, m in MARKERS_FINE.items():
                    pos = [g for g in m["positive"] if g in gene_names_a]
                    neg = [g for g in m.get("negative", []) if g in gene_names_a]
                    if not pos: continue
                    sc_ = cl_expr[:, [gene_names_a.index(g) for g in pos]].mean()
                    if neg: sc_ -= 0.5 * cl_expr[:, [gene_names_a.index(g) for g in neg]].mean()
                    if sc_ > best_sc: best_sc, best = sc_, ct
                # Map marker label → CL ID
                cl_mapped = mapper.map(best)
                cluster_cl[cl_id] = cl_mapped

            banksy_cl = np.array([cluster_cl.get(str(c), "UNMAPPED") for c in cids])
            banksy_broad = cl_to_level(banksy_cl, CL_TO_BROAD)
            banksy_medium = cl_to_level(banksy_cl, CL_TO_MEDIUM)

            for level_name, gt_lv, pred_lv in [
                ("CL_exact", gt_cl, banksy_cl), ("CL_broad", gt_broad, banksy_broad), ("CL_medium", gt_medium, banksy_medium),
            ]:
                m = evaluate_at_level(gt_lv, pred_lv, level_name)
                m["n_clusters"] = int(ncl)
                results[f"banksy_r{res}_{level_name}"] = m
                logger.info(f"    r={res} {ncl}cl {level_name}: F1={m['f1_macro']:.3f} Acc={m['accuracy']:.3f} ({m.get('n_classes_gt',0)}gt/{m.get('n_classes_pred',0)}pred)")

    # ── SUMMARY ──────────────────────────────────────────────────────
    logger.info(f"\n{'='*100}")
    for level in ["CL_broad", "CL_medium", "CL_exact"]:
        lr = sorted([(k,v) for k,v in results.items() if level in k and "f1_macro" in v],
                    key=lambda x: -x[1]["f1_macro"])
        n_gt = lr[0][1].get("n_classes_gt", "?") if lr else "?"
        n_pred = lr[0][1].get("n_classes_pred", "?") if lr else "?"
        logger.info(f"\n{'='*30} {level} ({n_gt} GT classes) {'='*30}")
        logger.info(f"{'Method':<55s} {'F1':>6s} {'Acc':>6s}")
        logger.info("-"*70)
        for k, v in lr[:12]:
            logger.info(f"{k:<55s} {v['f1_macro']:>6.3f} {v.get('accuracy',0):>6.3f}")

    # Save
    out = OUTPUT_DIR / "cl_normalized_rep1.json"
    with open(out, "w") as f:
        json.dump({k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                       for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
    logger.info(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
