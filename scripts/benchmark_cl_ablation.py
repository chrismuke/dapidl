#!/usr/bin/env python3
"""
CL Standardization Ablation Study.

Tests whether Cell Ontology as an intermediate representation improves
annotation accuracy vs ad-hoc string matching.

Both approaches map to the SAME 20 fine-grained GT classes.
"""

import sys
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
import pandas as pd
import numpy as np
import scanpy as sc
import h5py
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import subprocess
import tempfile

# =============================================================================
# Ground Truth Classes (20 fine-grained)
# =============================================================================

GT_FINEGRAINED_TYPES = [
    "Stromal", "Invasive_Tumor", "DCIS_1", "DCIS_2", "Macrophages_1",
    "Endothelial", "CD4+_T_Cells", "Myoepi_ACTA2+", "CD8+_T_Cells", "B_Cells",
    "Prolif_Invasive_Tumor", "Myoepi_KRT15+", "Macrophages_2", "Perivascular-Like",
    "Stromal_&_T_Cell_Hybrid", "T_Cell_&_Tumor_Hybrid", "IRF7+_DCs", "LAMP3+_DCs",
    "Mast_Cells", "Unlabeled"
]

# =============================================================================
# Approach A: Ad-hoc String Matching (Original)
# =============================================================================

ADHOC_MAPPINGS = {
    # SingleR outputs
    "Epithelial cells": "Invasive_Tumor",
    "Keratinocytes": "Myoepi_KRT15+",
    "CD4+ T-cells": "CD4+_T_Cells",
    "CD8+ T-cells": "CD8+_T_Cells",
    "T-cells": "CD4+_T_Cells",
    "B-cells": "B_Cells",
    "Macrophages": "Macrophages_1",
    "Monocytes": "Macrophages_1",
    "DC": "IRF7+_DCs",
    "NK cells": "CD8+_T_Cells",
    "Fibroblasts": "Stromal",
    "Smooth muscle": "Stromal",
    "Endothelial cells": "Endothelial",
    "Adipocytes": "Stromal",

    # CellTypist outputs
    "Luminal epithelial cell of mammary gland": "Invasive_Tumor",
    "Basal cell": "Myoepi_KRT15+",
    "T cell": "CD4+_T_Cells",
    "CD4-positive, alpha-beta T cell": "CD4+_T_Cells",
    "CD8-positive, alpha-beta T cell": "CD8+_T_Cells",
    "B cell": "B_Cells",
    "Macrophage": "Macrophages_1",
    "Classical monocyte": "Macrophages_1",
    "Non-classical monocyte": "Macrophages_1",
    "Dendritic cell": "IRF7+_DCs",
    "Mast cell": "Mast_Cells",
    "Fibroblast": "Stromal",
    "Endothelial cell": "Endothelial",
    "Smooth muscle cell": "Stromal",
    "Pericyte": "Perivascular-Like",
    "Adipocyte": "Stromal",
    "NK cell": "CD8+_T_Cells",
    "Plasma cell": "B_Cells",

    # scType outputs
    "Epithelial": "Invasive_Tumor",
    "T cells": "CD4+_T_Cells",
    "CD4+ T cells": "CD4+_T_Cells",
    "CD8+ T cells": "CD8+_T_Cells",
    "Regulatory T cells": "CD4+_T_Cells",
    "B cells": "B_Cells",
    "Plasma cells": "B_Cells",
    "Macrophages": "Macrophages_1",
    "Monocytes": "Macrophages_1",
    "Dendritic cells": "IRF7+_DCs",
    "Mast cells": "Mast_Cells",
    "NK cells": "CD8+_T_Cells",
    "Fibroblasts": "Stromal",
    "Myofibroblasts": "Myoepi_ACTA2+",
    "Pericytes": "Perivascular-Like",
    "Adipocytes": "Stromal",
    "Endothelial cells": "Endothelial",
    "Lymphatic endothelial": "Endothelial",

    # SCINA outputs
    "T_cells": "CD4+_T_Cells",
    "CD4_T_cells": "CD4+_T_Cells",
    "CD8_T_cells": "CD8+_T_Cells",
    "Tregs": "CD4+_T_Cells",
    "B_cells": "B_Cells",
    "Plasma_cells": "B_Cells",
    "Dendritic_cells": "IRF7+_DCs",
    "Mast_cells": "Mast_Cells",
    "NK_cells": "CD8+_T_Cells",
    "Endothelial": "Endothelial",
    "Lymphatic_endothelial": "Endothelial",
}


def map_adhoc(label: str) -> str:
    """Map using ad-hoc string matching (original approach)."""
    # Direct GT match
    if label in GT_FINEGRAINED_TYPES:
        return label

    # Direct mapping
    if label in ADHOC_MAPPINGS:
        return ADHOC_MAPPINGS[label]

    # Handle Unknown
    if label.lower() in ["unknown", "unlabeled", "unassigned", "na", "nan"]:
        return "Unlabeled"

    # Fuzzy matching
    label_lower = label.lower()

    if any(x in label_lower for x in ["epithelial", "luminal", "tumor", "dcis", "cancer"]):
        return "Invasive_Tumor"
    if any(x in label_lower for x in ["myoepithelial", "basal"]):
        return "Myoepi_KRT15+"
    if "cd4" in label_lower:
        return "CD4+_T_Cells"
    if "cd8" in label_lower:
        return "CD8+_T_Cells"
    if "t cell" in label_lower or "t-cell" in label_lower:
        return "CD4+_T_Cells"
    if "b cell" in label_lower or "b-cell" in label_lower or "plasma" in label_lower:
        return "B_Cells"
    if "macrophage" in label_lower or "monocyte" in label_lower:
        return "Macrophages_1"
    if "dendritic" in label_lower or label_lower == "dc":
        return "IRF7+_DCs"
    if "mast" in label_lower:
        return "Mast_Cells"
    if "nk" in label_lower or "natural killer" in label_lower:
        return "CD8+_T_Cells"
    if any(x in label_lower for x in ["fibroblast", "stromal", "smooth muscle", "adipocyte"]):
        return "Stromal"
    if "pericyte" in label_lower or "perivascular" in label_lower:
        return "Perivascular-Like"
    if "myofibroblast" in label_lower:
        return "Myoepi_ACTA2+"
    if "endothelial" in label_lower:
        return "Endothelial"

    return "Unlabeled"


# =============================================================================
# Approach B: CL Intermediate Mapping
# =============================================================================

# Step 1: Annotator → CL
ANNOTATOR_TO_CL = {
    # SingleR
    "Epithelial cells": "CL:0000066",
    "Keratinocytes": "CL:0000646",
    "CD4+ T-cells": "CL:0000624",
    "CD8+ T-cells": "CL:0000625",
    "T-cells": "CL:0000084",
    "B-cells": "CL:0000236",
    "Macrophages": "CL:0000235",
    "Monocytes": "CL:0000576",
    "DC": "CL:0000451",
    "NK cells": "CL:0000623",
    "Fibroblasts": "CL:0000057",
    "Smooth muscle": "CL:0000192",
    "Endothelial cells": "CL:0000115",
    "Adipocytes": "CL:0000136",

    # CellTypist
    "Luminal epithelial cell of mammary gland": "CL:0002325",
    "Basal cell": "CL:0000646",
    "T cell": "CL:0000084",
    "CD4-positive, alpha-beta T cell": "CL:0000624",
    "CD8-positive, alpha-beta T cell": "CL:0000625",
    "B cell": "CL:0000236",
    "Macrophage": "CL:0000235",
    "Classical monocyte": "CL:0000576",
    "Non-classical monocyte": "CL:0000576",
    "Dendritic cell": "CL:0000451",
    "Mast cell": "CL:0000097",
    "Fibroblast": "CL:0000057",
    "Endothelial cell": "CL:0000115",
    "Smooth muscle cell": "CL:0000192",
    "Pericyte": "CL:0000669",
    "Adipocyte": "CL:0000136",
    "NK cell": "CL:0000623",
    "Plasma cell": "CL:0000786",

    # scType
    "Epithelial": "CL:0000066",
    "T cells": "CL:0000084",
    "CD4+ T cells": "CL:0000624",
    "CD8+ T cells": "CL:0000625",
    "Regulatory T cells": "CL:0000815",
    "B cells": "CL:0000236",
    "Plasma cells": "CL:0000786",
    "Macrophages": "CL:0000235",
    "Monocytes": "CL:0000576",
    "Dendritic cells": "CL:0000451",
    "Mast cells": "CL:0000097",
    "NK cells": "CL:0000623",
    "Fibroblasts": "CL:0000057",
    "Myofibroblasts": "CL:0000186",
    "Pericytes": "CL:0000669",
    "Adipocytes": "CL:0000136",
    "Endothelial cells": "CL:0000115",
    "Lymphatic endothelial": "CL:0002138",

    # SCINA
    "T_cells": "CL:0000084",
    "CD4_T_cells": "CL:0000624",
    "CD8_T_cells": "CL:0000625",
    "Tregs": "CL:0000815",
    "B_cells": "CL:0000236",
    "Plasma_cells": "CL:0000786",
    "Dendritic_cells": "CL:0000451",
    "Mast_cells": "CL:0000097",
    "NK_cells": "CL:0000623",
    "Endothelial": "CL:0000115",
    "Lymphatic_endothelial": "CL:0002138",
}

# Step 2: CL → GT (fine-grained)
CL_TO_GT = {
    "CL:0000066": "Invasive_Tumor",      # Epithelial
    "CL:0002325": "Invasive_Tumor",      # Luminal epithelial
    "CL:0002327": "Invasive_Tumor",      # Mammary epithelial
    "CL:0000646": "Myoepi_KRT15+",       # Basal cell
    "CL:0000185": "Myoepi_ACTA2+",       # Myoepithelial

    "CL:0000084": "CD4+_T_Cells",        # Generic T cell
    "CL:0000624": "CD4+_T_Cells",        # CD4+ T cell
    "CL:0000625": "CD8+_T_Cells",        # CD8+ T cell
    "CL:0000815": "CD4+_T_Cells",        # Treg

    "CL:0000236": "B_Cells",             # B cell
    "CL:0000786": "B_Cells",             # Plasma cell

    "CL:0000235": "Macrophages_1",       # Macrophage
    "CL:0000576": "Macrophages_1",       # Monocyte
    "CL:0000451": "IRF7+_DCs",           # Dendritic cell
    "CL:0000097": "Mast_Cells",          # Mast cell
    "CL:0000623": "CD8+_T_Cells",        # NK cell (functional similarity)

    "CL:0000057": "Stromal",             # Fibroblast
    "CL:0000192": "Stromal",             # Smooth muscle
    "CL:0000136": "Stromal",             # Adipocyte
    "CL:0000186": "Myoepi_ACTA2+",       # Myofibroblast
    "CL:0000669": "Perivascular-Like",   # Pericyte

    "CL:0000115": "Endothelial",         # Endothelial
    "CL:0002138": "Endothelial",         # Lymphatic endothelial
}


def map_via_cl(label: str) -> str:
    """Map using Cell Ontology as intermediate representation."""
    # Direct GT match
    if label in GT_FINEGRAINED_TYPES:
        return label

    # Handle Unknown first
    if label.lower() in ["unknown", "unlabeled", "unassigned", "na", "nan"]:
        return "Unlabeled"

    # Try CL mapping
    cl_id = ANNOTATOR_TO_CL.get(label)
    if cl_id and cl_id in CL_TO_GT:
        return CL_TO_GT[cl_id]

    # Fallback to fuzzy matching (same as ad-hoc)
    label_lower = label.lower()

    if any(x in label_lower for x in ["epithelial", "luminal", "tumor", "dcis", "cancer"]):
        return "Invasive_Tumor"
    if any(x in label_lower for x in ["myoepithelial", "basal"]):
        return "Myoepi_KRT15+"
    if "cd4" in label_lower:
        return "CD4+_T_Cells"
    if "cd8" in label_lower:
        return "CD8+_T_Cells"
    if "t cell" in label_lower or "t-cell" in label_lower:
        return "CD4+_T_Cells"
    if "b cell" in label_lower or "b-cell" in label_lower or "plasma" in label_lower:
        return "B_Cells"
    if "macrophage" in label_lower or "monocyte" in label_lower:
        return "Macrophages_1"
    if "dendritic" in label_lower or label_lower == "dc":
        return "IRF7+_DCs"
    if "mast" in label_lower:
        return "Mast_Cells"
    if "nk" in label_lower or "natural killer" in label_lower:
        return "CD8+_T_Cells"
    if any(x in label_lower for x in ["fibroblast", "stromal", "smooth muscle", "adipocyte"]):
        return "Stromal"
    if "pericyte" in label_lower or "perivascular" in label_lower:
        return "Perivascular-Like"
    if "myofibroblast" in label_lower:
        return "Myoepi_ACTA2+"
    if "endothelial" in label_lower:
        return "Endothelial"

    return "Unlabeled"


# =============================================================================
# Data Loading
# =============================================================================

DATASETS = {
    "rep1": {
        "xenium_path": Path("/home/chrism/datasets/raw/xenium/breast_tumor_rep1/outs"),
        "gt_path": Path("/home/chrism/.clearml/cache/storage_manager/datasets/ds_ee53f76b00c0400d83a303408b5ea8e2/celltypes_ground_truth_rep1_supervised.xlsx"),
        "name": "Xenium Breast Rep1",
    },
    "rep2": {
        "xenium_path": Path("/home/chrism/datasets/raw/xenium/breast_tumor_rep2/outs"),
        "gt_path": Path("/home/chrism/.clearml/cache/storage_manager/datasets/ds_d73638c8e809416a8b94697d20b2d971/celltypes_ground_truth_rep2_supervised.xlsx"),
        "name": "Xenium Breast Rep2",
    },
}


def load_xenium_data(xenium_path: Path) -> sc.AnnData:
    """Load Xenium expression data."""
    h5_path = xenium_path / "cell_feature_matrix.h5"

    with h5py.File(h5_path, 'r') as f:
        data = f['matrix/data'][:]
        indices = f['matrix/indices'][:]
        indptr = f['matrix/indptr'][:]
        shape = f['matrix/shape'][:]
        barcodes = [b.decode() for b in f['matrix/barcodes'][:]]
        genes = [g.decode() for g in f['matrix/features/name'][:]]

    from scipy.sparse import csc_matrix
    X = csc_matrix((data, indices, indptr), shape=shape).T
    adata = sc.AnnData(X=X)
    adata.obs_names = barcodes
    adata.var_names = genes
    adata.obs["cell_id"] = barcodes

    return adata


def load_ground_truth(gt_path: Path) -> pl.DataFrame:
    """Load ground truth annotations."""
    df = pd.read_excel(gt_path)
    gt_df = pl.from_pandas(df)
    gt_df = gt_df.rename({"Barcode": "cell_id", "Cluster": "gt_type"})
    gt_df = gt_df.with_columns(pl.col("cell_id").cast(pl.Utf8))
    gt_df = gt_df.filter(pl.col("gt_type") != "Unlabeled")
    return gt_df


# =============================================================================
# Method Runners
# =============================================================================

def run_singler(adata: sc.AnnData) -> pl.DataFrame:
    """Run SingleR annotation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        expr_df = pd.DataFrame(
            adata.X.toarray().T if hasattr(adata.X, 'toarray') else adata.X.T,
            index=adata.var_names,
            columns=adata.obs_names
        )
        expr_path = temp_path / "expr_matrix.csv"
        expr_df.to_csv(expr_path)
        results_path = temp_path / "singler_results.csv"

        r_script = f'''
library(SingleR)
library(celldex)
expr <- as.matrix(read.csv("{expr_path}", row.names=1, check.names=FALSE))
ref <- BlueprintEncodeData()
common <- intersect(rownames(expr), rownames(ref))
if (length(common) >= 50) {{
    results <- SingleR(test = expr[common,], ref = ref[common,], labels = ref$label.main)
    output <- data.frame(
        cell_id = colnames(expr),
        predicted_type = results$labels,
        confidence = 1 - results$delta.next,
        stringsAsFactors = FALSE
    )
}} else {{
    output <- data.frame(
        cell_id = colnames(expr),
        predicted_type = rep("Unknown", ncol(expr)),
        confidence = rep(0.0, ncol(expr)),
        stringsAsFactors = FALSE
    )
}}
write.csv(output, "{results_path}", row.names = FALSE)
'''

        script_path = temp_path / "run_singler.R"
        with open(script_path, 'w') as f:
            f.write(r_script)

        result = subprocess.run(
            ["Rscript", str(script_path)],
            capture_output=True, text=True, timeout=600, cwd=str(temp_path)
        )

        if result.returncode != 0:
            logger.error(f"SingleR failed: {result.stderr[:500]}")
            return pl.DataFrame()

        return pl.from_pandas(pd.read_csv(results_path))


def run_celltypist(adata: sc.AnnData) -> pl.DataFrame:
    """Run CellTypist annotation."""
    import celltypist
    from celltypist import models

    adata_copy = adata.copy()
    sc.pp.normalize_total(adata_copy, target_sum=1e4)
    sc.pp.log1p(adata_copy)

    model = models.Model.load(model="Cells_Adult_Breast.pkl")
    predictions = celltypist.annotate(adata_copy, model=model, majority_voting=False)

    return pl.DataFrame({
        "cell_id": adata.obs_names.tolist(),
        "predicted_type": predictions.predicted_labels["predicted_labels"].tolist(),
        "confidence": predictions.probability_matrix.max(axis=1).tolist(),
    })


def run_sctype(adata: sc.AnnData) -> pl.DataFrame:
    """Run scType annotation."""
    from dapidl.pipeline.components.annotators.sctype import ScTypeAnnotator
    from dapidl.pipeline.base import AnnotationConfig

    annotator = ScTypeAnnotator(config=AnnotationConfig(fine_grained=True))
    result = annotator.annotate(adata=adata)
    return result.annotations_df.select(["cell_id", "predicted_type", "confidence"])


def run_scina(adata: sc.AnnData) -> pl.DataFrame:
    """Run SCINA annotation."""
    from dapidl.pipeline.components.annotators.scina import SCINAAnnotator
    from dapidl.pipeline.base import AnnotationConfig

    annotator = SCINAAnnotator(config=AnnotationConfig(fine_grained=True))
    result = annotator.annotate(adata=adata)
    return result.annotations_df.select(["cell_id", "predicted_type", "confidence"])


METHOD_RUNNERS = {
    "singler": run_singler,
    "celltypist": run_celltypist,
    "sctype": run_sctype,
    "scina": run_scina,
}


# =============================================================================
# Main Ablation Test
# =============================================================================

def run_ablation(dataset_key: str = "rep1"):
    """Run ablation study comparing ad-hoc vs CL mapping."""

    dataset = DATASETS[dataset_key]
    logger.info(f"Loading {dataset['name']}...")

    adata = load_xenium_data(dataset["xenium_path"])
    gt_df = load_ground_truth(dataset["gt_path"])

    logger.info(f"Loaded {adata.n_obs} cells, {gt_df.height} with ground truth")

    results = []

    for method_name, runner in METHOD_RUNNERS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {method_name}...")

        pred_df = runner(adata)
        if pred_df.height == 0:
            logger.warning(f"{method_name} returned no predictions")
            continue

        pred_df = pred_df.with_columns(pl.col("cell_id").cast(pl.Utf8))

        # Apply both mapping approaches
        pred_df = pred_df.with_columns([
            pl.col("predicted_type").map_elements(
                map_adhoc, return_dtype=pl.Utf8
            ).alias("mapped_adhoc"),
            pl.col("predicted_type").map_elements(
                map_via_cl, return_dtype=pl.Utf8
            ).alias("mapped_cl"),
        ])

        # Join with GT
        eval_df = gt_df.join(
            pred_df.select(["cell_id", "predicted_type", "mapped_adhoc", "mapped_cl"]),
            on="cell_id",
            how="inner"
        )

        if eval_df.height == 0:
            logger.warning(f"No matching cells for {method_name}")
            continue

        y_true = eval_df["gt_type"].to_list()
        y_adhoc = eval_df["mapped_adhoc"].to_list()
        y_cl = eval_df["mapped_cl"].to_list()

        # Calculate metrics for both approaches
        acc_adhoc = accuracy_score(y_true, y_adhoc)
        f1_adhoc = f1_score(y_true, y_adhoc, average="macro", zero_division=0)

        acc_cl = accuracy_score(y_true, y_cl)
        f1_cl = f1_score(y_true, y_cl, average="macro", zero_division=0)

        # Count differences
        differences = sum(1 for a, c in zip(y_adhoc, y_cl) if a != c)
        diff_pct = 100 * differences / len(y_adhoc)

        logger.info(f"  Ad-hoc:  Acc={acc_adhoc:.4f}, F1={f1_adhoc:.4f}")
        logger.info(f"  CL:      Acc={acc_cl:.4f}, F1={f1_cl:.4f}")
        logger.info(f"  Δ Acc:   {(acc_cl - acc_adhoc)*100:+.2f}%")
        logger.info(f"  Δ F1:    {(f1_cl - f1_adhoc)*100:+.2f}%")
        logger.info(f"  Different predictions: {differences} ({diff_pct:.1f}%)")

        results.append({
            "method": method_name,
            "dataset": dataset_key,
            "n_cells": eval_df.height,
            "acc_adhoc": acc_adhoc,
            "f1_adhoc": f1_adhoc,
            "acc_cl": acc_cl,
            "f1_cl": f1_cl,
            "acc_delta": acc_cl - acc_adhoc,
            "f1_delta": f1_cl - f1_adhoc,
            "n_different": differences,
            "pct_different": diff_pct,
        })

    # Summary
    print("\n" + "="*80)
    print("ABLATION RESULTS: Ad-hoc vs CL Mapping")
    print("="*80)

    results_df = pl.DataFrame(results)
    print(results_df)

    # Average improvement
    avg_acc_delta = results_df["acc_delta"].mean()
    avg_f1_delta = results_df["f1_delta"].mean()

    print(f"\nAverage improvement from CL mapping:")
    print(f"  Accuracy: {avg_acc_delta*100:+.2f}%")
    print(f"  Macro F1: {avg_f1_delta*100:+.2f}%")

    if avg_f1_delta > 0:
        print("\n✓ CL standardization HELPS - improves mapping accuracy")
    elif avg_f1_delta < 0:
        print("\n✗ CL standardization HURTS - ad-hoc mapping is better")
    else:
        print("\n= CL standardization has NO EFFECT")

    return results_df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rep1", choices=["rep1", "rep2"])
    args = parser.parse_args()

    run_ablation(args.dataset)


if __name__ == "__main__":
    main()
