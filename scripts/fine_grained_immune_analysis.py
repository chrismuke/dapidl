#!/usr/bin/env python3
"""Comprehensive fine-grained cell type detection analysis across all datasets.

Evaluates immune subtype detection on:
- Rep1 (158K cells, 17 GT types, expert annotations)
- Rep2 (114K cells, 17 GT types, expert annotations)
- STHELAR breast_s0, s1, s3, s6 (365K-893K cells, 39 Tangram types)

Methods: CellTypist (Breast + Immune_High), SCINA fine markers
Output: per-type detection rates, confusion analysis, visualizations
"""

import gc
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import celltypist
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from celltypist import models as ct_models
from loguru import logger
from scipy.sparse import issparse
from sklearn.metrics import f1_score

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from dapidl.pipeline.components.annotators.mapping import map_to_broad_category

OUTPUT = Path("pipeline_output/annotation_benchmark_2026_03/fine_grained")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# CELL ONTOLOGY MAPPINGS
# ──────────────────────────────────────────────────────────────────────────────

# Janesick GT → CL
JANESICK_CL = {
    "CD4+_T_Cells":         ("CL:0000624", "CD4-positive, alpha-beta T cell", "Immune", "T_Cell"),
    "CD8+_T_Cells":         ("CL:0000625", "CD8-positive, alpha-beta T cell", "Immune", "T_Cell"),
    "B_Cells":              ("CL:0000236", "B cell", "Immune", "B_Cell"),
    "Macrophages_1":        ("CL:0000235", "macrophage", "Immune", "Myeloid"),
    "Macrophages_2":        ("CL:0000235", "macrophage", "Immune", "Myeloid"),
    "IRF7+_DCs":            ("CL:0000784", "plasmacytoid dendritic cell", "Immune", "Myeloid"),
    "LAMP3+_DCs":           ("CL:0000451", "dendritic cell", "Immune", "Myeloid"),
    "Mast_Cells":           ("CL:0000097", "mast cell", "Immune", "Myeloid"),
    "Invasive_Tumor":       ("CL:0001064", "malignant cell", "Epithelial", "Epithelial_Tumor"),
    "Prolif_Invasive_Tumor":("CL:0001064", "malignant cell", "Epithelial", "Epithelial_Tumor"),
    "DCIS_1":               ("CL:0001064", "malignant cell", "Epithelial", "Epithelial_Tumor"),
    "DCIS_2":               ("CL:0001064", "malignant cell", "Epithelial", "Epithelial_Tumor"),
    "Myoepi_ACTA2+":        ("CL:0000185", "myoepithelial cell", "Epithelial", "Epithelial_Basal"),
    "Myoepi_KRT15+":        ("CL:0000185", "myoepithelial cell", "Epithelial", "Epithelial_Basal"),
    "Stromal":              ("CL:0000057", "fibroblast", "Stromal", "Stromal"),
    "Perivascular-Like":    ("CL:0000669", "pericyte", "Stromal", "Stromal"),
    "Endothelial":          ("CL:0000115", "endothelial cell", "Endothelial", "Endothelial"),
}

# STHELAR Tangram → CL (breast cell types)
from dapidl.data.sthelar_reader import TANGRAM_TO_COARSE

TANGRAM_CL = {
    # T cells
    "CD4 T cell": ("CL:0000624", "CD4+ T cell", "Immune", "T_Cell"),
    "CD8 T cell": ("CL:0000625", "CD8+ T cell", "Immune", "T_Cell"),
    "T_prol": ("CL:0000084", "T cell (prolif)", "Immune", "T_Cell"),
    "GD": ("CL:0000798", "gamma-delta T cell", "Immune", "T_Cell"),
    "NKT": ("CL:0000814", "NKT cell", "Immune", "T_Cell"),
    "Treg": ("CL:0000815", "regulatory T cell", "Immune", "T_Cell"),
    # B / Plasma
    "B cell": ("CL:0000236", "B cell", "Immune", "B_Cell"),
    "Plasma IgA": ("CL:0000987", "plasma cell (IgA)", "Immune", "B_Cell"),
    "Plasma IgG": ("CL:0000985", "plasma cell (IgG)", "Immune", "B_Cell"),
    # Myeloid
    "M1 macrophage": ("CL:0000863", "M1 macrophage", "Immune", "Myeloid"),
    "M2 macrophage": ("CL:0000890", "M2 macrophage", "Immune", "Myeloid"),
    "Monocyte": ("CL:0000576", "monocyte", "Immune", "Myeloid"),
    "cDC1": ("CL:0002394", "cDC1", "Immune", "Myeloid"),
    "cDC2": ("CL:0002399", "cDC2", "Immune", "Myeloid"),
    "Mast": ("CL:0000097", "mast cell", "Immune", "Myeloid"),
    "NK": ("CL:0000623", "NK cell", "Immune", "NK"),
    # Epithelial
    "LummHR-major": ("CL:0002325", "luminal epithelial (HR+)", "Epithelial", "Epithelial_Luminal"),
    "LummHR-active": ("CL:0002325", "luminal epithelial (active)", "Epithelial", "Epithelial_Luminal"),
    "LummHR-SCGB": ("CL:0002325", "luminal epithelial (SCGB+)", "Epithelial", "Epithelial_Luminal"),
    "Lumsec-major": ("CL:0002326", "luminal secretory", "Epithelial", "Epithelial_Luminal"),
    "Lumsec-lac": ("CL:0002326", "luminal secretory (lac)", "Epithelial", "Epithelial_Luminal"),
    "Lumsec-HLA": ("CL:0002326", "luminal secretory (HLA)", "Epithelial", "Epithelial_Luminal"),
    "Lumsec-KIT": ("CL:0002326", "luminal secretory (KIT)", "Epithelial", "Epithelial_Luminal"),
    "Lumsec-prol": ("CL:0002326", "luminal secretory (prolif)", "Epithelial", "Epithelial_Luminal"),
    "Lumsec-basal": ("CL:0000646", "basal epithelial", "Epithelial", "Epithelial_Basal"),
    "Lumsec-myo": ("CL:0000185", "myoepithelial", "Epithelial", "Epithelial_Basal"),
    "basal": ("CL:0000646", "basal cell", "Epithelial", "Epithelial_Basal"),
    # Stromal
    "CXCL+ fibroblast": ("CL:0000057", "fibroblast (CXCL+)", "Stromal", "Stromal"),
    "COL1+ fibroblast": ("CL:0000057", "fibroblast (COL1+)", "Stromal", "Stromal"),
    "COL4+ fibroblast": ("CL:0000057", "fibroblast (COL4+)", "Stromal", "Stromal"),
    "Myofibroblast": ("CL:0000186", "myofibroblast", "Stromal", "Stromal"),
    "Pericyte": ("CL:0000669", "pericyte", "Stromal", "Stromal"),
    "Smooth muscle": ("CL:0000192", "smooth muscle cell", "Stromal", "Stromal"),
    "Adipocyte": ("CL:0000136", "adipocyte", "Stromal", "Stromal"),
    # Endothelial
    "Arterial EC": ("CL:0002543", "arterial endothelial", "Endothelial", "Endothelial"),
    "Capillary EC": ("CL:0002144", "capillary endothelial", "Endothelial", "Endothelial"),
    "Venous EC": ("CL:0002543", "venous endothelial", "Endothelial", "Endothelial"),
    "Lymphatic EC": ("CL:0002138", "lymphatic endothelial", "Endothelial", "Endothelial"),
}

# ──────────────────────────────────────────────────────────────────────────────
# SCINA FINE MARKERS
# ──────────────────────────────────────────────────────────────────────────────

MARKERS_FINE = {
    "CD4+ T cell": {"pos": ["CD3D","CD3E","CD4","IL7R"], "neg": ["CD8A","CD14"]},
    "CD8+ T cell": {"pos": ["CD3D","CD3E","CD8A","CD8B","GZMB"], "neg": ["CD4"]},
    "Treg": {"pos": ["CD3D","CD4","FOXP3","IL2RA","CTLA4"], "neg": ["CD8A"]},
    "B cell": {"pos": ["CD19","CD79A","CD79B","MS4A1","PAX5"], "neg": ["CD3D"]},
    "Plasma cell": {"pos": ["SDC1","MZB1","JCHAIN","XBP1"], "neg": ["MS4A1"]},
    "Macrophage": {"pos": ["CD68","CD163","CSF1R","MARCO"], "neg": ["CD3D","CD19"]},
    "Monocyte": {"pos": ["CD14","FCGR3A","LYZ","S100A8"], "neg": ["CD3D"]},
    "cDC": {"pos": ["ITGAX","CD1C","CLEC9A","FLT3","HLA-DRA"], "neg": ["CD14"]},
    "pDC": {"pos": ["IRF7","LILRA4","CLEC4C","IL3RA"], "neg": ["CD14","ITGAX"]},
    "Mast cell": {"pos": ["KIT","TPSAB1","CPA3","MS4A2","FCER1A"], "neg": ["CD3D"]},
    "NK cell": {"pos": ["NCAM1","NKG7","GNLY","KLRD1","KLRF1"], "neg": ["CD3D","CD19"]},
    "Epithelial": {"pos": ["EPCAM","KRT8","KRT18","KRT19","CDH1"], "neg": ["PTPRC","VIM"]},
    "Myoepithelial": {"pos": ["KRT14","KRT5","ACTA2","TP63"], "neg": ["KRT8","KRT18"]},
    "Fibroblast": {"pos": ["COL1A1","COL1A2","DCN","FAP","PDGFRA"], "neg": ["EPCAM","PTPRC"]},
    "Pericyte": {"pos": ["ACTA2","PDGFRB","RGS5","NOTCH3"], "neg": ["EPCAM","COL1A1"]},
    "Endothelial": {"pos": ["PECAM1","VWF","CLDN5","KDR"], "neg": ["EPCAM","COL1A1"]},
}


def run_scina(adata, markers=MARKERS_FINE):
    """Score cells with SCINA-style markers."""
    gene_names = list(adata.var_names)
    expr = np.asarray(adata.X.toarray() if issparse(adata.X) else adata.X)
    scores = {}
    for ct, m in markers.items():
        pos = [g for g in m["pos"] if g in gene_names]
        neg = [g for g in m["neg"] if g in gene_names]
        if not pos:
            scores[ct] = np.zeros(len(adata))
            continue
        s = np.asarray(expr[:, [gene_names.index(g) for g in pos]].mean(axis=1)).ravel()
        if neg:
            s -= 0.5 * np.asarray(expr[:, [gene_names.index(g) for g in neg]].mean(axis=1)).ravel()
        scores[ct] = s
    mat = np.column_stack([scores[ct] for ct in scores])
    names = list(scores.keys())
    preds = np.array([names[i] for i in mat.argmax(axis=1)])
    conf = mat.max(axis=1)
    return preds, conf


def run_celltypist_model(adata, model_name):
    """Run CellTypist, return fine-grained predictions."""
    try:
        model = ct_models.Model.load(model_name)
    except Exception:
        ct_models.download_models(model=model_name, force_update=False)
        model = ct_models.Model.load(model_name)
    result = celltypist.annotate(adata, model=model, majority_voting=False).to_adata()
    preds = result.obs["predicted_labels"].astype(str).values
    conf = result.obs["conf_score"].values if "conf_score" in result.obs.columns else np.ones(len(result))
    return preds, conf


def load_xenium_rep(rep):
    """Load Xenium rep with GT."""
    from scripts.annotation_benchmark_2026_03 import load_xenium_adata, preprocess_adata
    adata_raw = load_xenium_adata(rep)
    adata_pp = preprocess_adata(adata_raw)
    # Remove hybrids
    valid = ~adata_pp.obs["gt_fine"].isin(["Unlabeled", "Hybrid", "Stromal_&_T_Cell_Hybrid", "T_Cell_&_Tumor_Hybrid"])
    return adata_pp[valid].copy(), JANESICK_CL


def load_sthelar_slide(slide_name):
    """Load STHELAR slide with Tangram GT."""
    import anndata as ad
    zarr_path = Path(f"/mnt/work/datasets/STHELAR/sdata_slides/sdata_{slide_name}.zarr/sdata_{slide_name}.zarr")
    table = ad.read_zarr(str(zarr_path / "tables" / "table_combined"))

    # Normalize
    sc.pp.normalize_total(table, target_sum=1e4)
    sc.pp.log1p(table)

    # Map tangram labels
    if "ct_tangram" in table.obs.columns:
        table.obs["gt_fine"] = table.obs["ct_tangram"].astype(str)
        # Filter to known types
        known = set(TANGRAM_CL.keys())
        valid = table.obs["gt_fine"].isin(known)
        table = table[valid].copy()

    return table, TANGRAM_CL


def analyze_dataset(adata, cl_map, ds_name):
    """Run all methods and analyze per-type detection."""
    gt_fine = np.array(adata.obs["gt_fine"].values)
    logger.info(f"\n{'='*80}")
    logger.info(f"DATASET: {ds_name} ({len(adata)} cells, {len(set(gt_fine))} types)")
    logger.info(f"{'='*80}")

    # Run methods
    logger.info("  Running CellTypist Breast...")
    ct_breast, ct_breast_conf = run_celltypist_model(adata, "Cells_Adult_Breast.pkl")
    logger.info("  Running CellTypist Immune_High...")
    ct_immune, ct_immune_conf = run_celltypist_model(adata, "Immune_All_High.pkl")
    logger.info("  Running SCINA...")
    scina, scina_conf = run_scina(adata)

    results = []
    for gt_type in sorted(set(gt_fine)):
        if gt_type not in cl_map:
            continue
        info = cl_map[gt_type]
        cl_id, cl_name, broad, medium = info
        mask = gt_fine == gt_type
        n = mask.sum()
        if n < 5:
            continue

        row = {
            "dataset": ds_name,
            "gt_type": gt_type,
            "cl_id": cl_id,
            "cl_name": cl_name,
            "broad": broad,
            "medium": medium,
            "n_cells": int(n),
        }

        # Analyze each method
        for method_name, preds, conf in [
            ("ct_breast", ct_breast, ct_breast_conf),
            ("ct_immune", ct_immune, ct_immune_conf),
            ("scina", scina, scina_conf),
        ]:
            p = preds[mask]
            c = conf[mask]
            top3 = Counter(p).most_common(3)
            broad_preds = np.array([map_to_broad_category(x) for x in p])
            broad_correct = (broad_preds == broad).mean()
            row[f"{method_name}_top1"] = top3[0][0]
            row[f"{method_name}_top1_pct"] = round(top3[0][1] / n * 100, 1)
            row[f"{method_name}_broad_correct"] = round(broad_correct * 100, 1)
            row[f"{method_name}_mean_conf"] = round(float(c.mean()), 3)
            if len(top3) > 1:
                row[f"{method_name}_top2"] = top3[1][0]
                row[f"{method_name}_top2_pct"] = round(top3[1][1] / n * 100, 1)

        # Best detection rate across methods
        row["best_broad"] = max(row["ct_breast_broad_correct"], row["ct_immune_broad_correct"], row["scina_broad_correct"])
        row["best_method"] = ["ct_breast", "ct_immune", "scina"][
            np.argmax([row["ct_breast_broad_correct"], row["ct_immune_broad_correct"], row["scina_broad_correct"]])]

        results.append(row)

    return results


def make_visualizations(all_results):
    """Generate comprehensive plots."""
    df = pd.DataFrame(all_results)

    # ── 1. Per-type detection heatmap across datasets ──────────────────
    datasets = sorted(df["dataset"].unique())
    immune_types = df[df["broad"] == "Immune"]["gt_type"].unique()
    all_types = sorted(df["gt_type"].unique(), key=lambda x: (
        df[df["gt_type"]==x].iloc[0]["broad"],
        df[df["gt_type"]==x].iloc[0]["medium"],
        x
    ))

    fig, ax = plt.subplots(figsize=(max(14, len(datasets)*2.5), max(10, len(all_types)*0.4)))
    matrix = np.full((len(all_types), len(datasets)), np.nan)
    for i, ct in enumerate(all_types):
        for j, ds in enumerate(datasets):
            rows = df[(df["gt_type"] == ct) & (df["dataset"] == ds)]
            if len(rows) > 0:
                matrix[i, j] = rows.iloc[0]["best_broad"]

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(all_types)))
    # Color-code y labels by broad category
    colors = {"Immune": "#1565C0", "Epithelial": "#2E7D32", "Stromal": "#E65100", "Endothelial": "#6A1B9A"}
    labels = []
    for ct in all_types:
        r = df[df["gt_type"]==ct].iloc[0]
        labels.append(f"{ct} ({r['cl_id']})")
    ax.set_yticklabels(labels, fontsize=7)
    for i, ct in enumerate(all_types):
        r = df[df["gt_type"]==ct].iloc[0]
        ax.get_yticklabels()[i].set_color(colors.get(r["broad"], "black"))

    for i in range(len(all_types)):
        for j in range(len(datasets)):
            if not np.isnan(matrix[i, j]):
                v = matrix[i, j]
                color = "white" if v < 50 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Best Broad Detection %", shrink=0.8)
    ax.set_title("Cell Type Detection Rate Across Datasets (Best Method)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT / "01_detection_heatmap_all.png", dpi=200)
    plt.close()

    # ── 2. Immune subtypes: method comparison per dataset ─────────────
    immune_df = df[df["broad"] == "Immune"].copy()
    if len(immune_df) > 0:
        fig, axes = plt.subplots(1, min(3, len(datasets)), figsize=(min(18, len(datasets)*6), 8), squeeze=False)
        for j, ds in enumerate(datasets[:3]):
            ax = axes[0, j]
            ds_data = immune_df[immune_df["dataset"] == ds].sort_values("n_cells", ascending=True)
            if len(ds_data) == 0:
                continue
            y = range(len(ds_data))
            w = 0.25
            for k, method in enumerate(["ct_breast", "ct_immune", "scina"]):
                col = f"{method}_broad_correct"
                if col in ds_data.columns:
                    vals = ds_data[col].values
                    bars = ax.barh([yi + k*w for yi in y], vals, w, label=method.replace("ct_", "CT_"),
                                    color=["#FF9800", "#2196F3", "#4CAF50"][k], alpha=0.8)
            ax.set_yticks([yi + w for yi in y])
            ax.set_yticklabels([f"{r['gt_type']} (n={r['n_cells']})" for _, r in ds_data.iterrows()], fontsize=7)
            ax.set_xlabel("Broad Category Correct %")
            ax.set_title(f"Immune Subtypes: {ds}", fontsize=11)
            ax.set_xlim(0, 105)
            ax.legend(fontsize=8)
            ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT / "02_immune_subtypes_by_dataset.png", dpi=200)
        plt.close()

    # ── 3. Tier list visualization ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("off")

    # Aggregate across datasets
    agg = df.groupby(["gt_type", "cl_id", "cl_name", "broad"]).agg(
        total_cells=("n_cells", "sum"),
        mean_detect=("best_broad", "mean"),
        n_datasets=("dataset", "nunique"),
    ).reset_index().sort_values("mean_detect", ascending=False)

    tier_colors = {"HIGH": "#C8E6C9", "MEDIUM": "#FFF9C4", "LOW": "#FFE0B2", "FAIL": "#FFCDD2"}
    headers = ["Cell Type", "CL ID", "CL Name", "Broad", "Cells", "Datasets", "Detection", "Tier"]
    rows = []
    for _, r in agg.iterrows():
        d = r["mean_detect"]
        tier = "HIGH" if d > 80 else "MEDIUM" if d > 60 else "LOW" if d > 40 else "FAIL"
        rows.append([r["gt_type"][:25], r["cl_id"], r["cl_name"][:30], r["broad"],
                      f"{r['total_cells']:,.0f}", f"{r['n_datasets']:.0f}", f"{d:.0f}%", tier])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.auto_set_column_width(range(len(headers)))
    table.scale(1, 1.4)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=8)
    ncols = len(headers)
    for i, row in enumerate(rows):
        tier = row[-1]
        table[i+1, ncols-1].set_facecolor(tier_colors.get(tier, "white"))
        table[i+1, ncols-1].set_text_props(fontweight="bold")

    ax.set_title("Cell Type Detection Tier List\n(Aggregated across all datasets, best method per type)",
                  fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT / "03_tier_list.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── 4. Confusion: what gets confused with what ────────────────────
    # For immune types on rep1
    rep1_immune = df[(df["dataset"] == "rep1") & (df["broad"] == "Immune")]
    if len(rep1_immune) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, method in zip(axes, ["ct_breast", "ct_immune", "scina"]):
            types = rep1_immune["gt_type"].values
            y = range(len(types))
            top1_pcts = rep1_immune[f"{method}_top1_pct"].values
            top1_names = rep1_immune[f"{method}_top1"].values
            top2_col = f"{method}_top2_pct"
            top2_pcts = rep1_immune[top2_col].values if top2_col in rep1_immune.columns else np.zeros(len(types))

            ax.barh(y, top1_pcts, color="#2196F3", alpha=0.8, label="Top prediction")
            ax.barh(y, top2_pcts, left=top1_pcts, color="#FF9800", alpha=0.6, label="2nd prediction")
            ax.set_yticks(y)
            ax.set_yticklabels([f"{t} (n={n})" for t, n in zip(types, rep1_immune["n_cells"].values)], fontsize=7)
            for i in range(len(types)):
                ax.text(2, i, f"{top1_names[i][:20]}", fontsize=6, va="center", color="white", fontweight="bold")
            ax.set_xlabel("% of cells")
            ax.set_title(method.replace("ct_", "CT_"), fontsize=11)
            ax.set_xlim(0, 105)
            ax.legend(fontsize=7)
        fig.suptitle("Immune Subtype Predictions: Rep1", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT / "04_immune_confusion_rep1.png", dpi=200)
        plt.close()

    logger.info(f"Saved {len(list(OUTPUT.glob('*.png')))} plots to {OUTPUT}/")


def main():
    all_results = []

    # ── Xenium Rep1 + Rep2 ───────────────────────────────────────────
    for rep in ["rep1", "rep2"]:
        try:
            adata, cl_map = load_xenium_rep(rep)
            results = analyze_dataset(adata, cl_map, rep)
            all_results.extend(results)
            del adata; gc.collect()
        except Exception as e:
            logger.error(f"{rep} failed: {e}")
            import traceback; traceback.print_exc()

    # ── STHELAR breast slides ────────────────────────────────────────
    for slide in ["breast_s0", "breast_s1", "breast_s3", "breast_s6"]:
        try:
            adata, cl_map = load_sthelar_slide(slide)
            results = analyze_dataset(adata, cl_map, slide)
            all_results.extend(results)
            del adata; gc.collect()
        except Exception as e:
            logger.error(f"{slide} failed: {e}")
            import traceback; traceback.print_exc()

    # ── Summary ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)

    logger.info(f"\n{'='*100}")
    logger.info("AGGREGATE RESULTS ACROSS ALL DATASETS")
    logger.info(f"{'='*100}")

    agg = df.groupby(["gt_type", "cl_id", "cl_name", "broad", "medium"]).agg(
        total_cells=("n_cells", "sum"),
        mean_detect=("best_broad", "mean"),
        best_method=("best_method", lambda x: Counter(x).most_common(1)[0][0]),
        n_datasets=("dataset", "nunique"),
    ).reset_index().sort_values("mean_detect", ascending=False)

    logger.info(f"\n{'Type':<28s} {'CL ID':<14s} {'Broad':<12s} {'Cells':>8s} {'Detect':>7s} {'Best':>12s} {'DS':>3s}")
    logger.info("-" * 95)
    for _, r in agg.iterrows():
        d = r["mean_detect"]
        tier = "HIGH" if d > 80 else "MED" if d > 60 else "LOW" if d > 40 else "FAIL"
        logger.info(f"{r['gt_type']:<28s} {r['cl_id']:<14s} {r['broad']:<12s} {r['total_cells']:>8,.0f} "
                     f"{d:>6.1f}% {r['best_method']:>12s} {r['n_datasets']:>3.0f}  [{tier}]")

    # Save
    with open(OUTPUT / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    df.to_csv(OUTPUT / "all_results.csv", index=False)

    # Visualizations
    logger.info("\nGenerating visualizations...")
    make_visualizations(all_results)

    logger.info("ANALYSIS COMPLETE!")


if __name__ == "__main__":
    main()
