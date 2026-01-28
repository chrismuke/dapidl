#!/usr/bin/env python3
"""
Test hybrid ensemble at multiple granularity levels:
- COARSE: 4 categories (Epithelial, Immune, Stromal, Endothelial)
- MEDIUM: 10-12 categories (T_Cell, B_Cell, Macrophage, Tumor, Myoepithelial, etc.)
- FINE: 20 categories (full ground truth labels)
"""

from pathlib import Path
import json
import sys

import celltypist
from celltypist import models
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score

from dapidl.pipeline.components.annotators.sctype import ScTypeAnnotator


# =============================================================================
# GRANULARITY MAPPINGS
# =============================================================================

# Ground truth -> MEDIUM level mapping
GT_TO_MEDIUM = {
    # Epithelial - Tumor
    "Invasive_Tumor": "Tumor",
    "DCIS_1": "DCIS",
    "DCIS_2": "DCIS",
    "Prolif_Invasive_Tumor": "Tumor",
    # Epithelial - Myoepithelial
    "Myoepi_ACTA2+": "Myoepithelial",
    "Myoepi_KRT15+": "Myoepithelial",
    # Immune - T cells
    "CD4+_T_Cells": "T_Cell",
    "CD8+_T_Cells": "T_Cell",
    # Immune - Other
    "B_Cells": "B_Cell",
    "Macrophages_1": "Macrophage",
    "Macrophages_2": "Macrophage",
    "IRF7+_DCs": "Dendritic_Cell",
    "LAMP3+_DCs": "Dendritic_Cell",
    "Mast_Cells": "Mast_Cell",
    # Stromal
    "Stromal": "Fibroblast",
    "Perivascular-Like": "Perivascular",
    "Stromal_&_T_Cell_Hybrid": "Fibroblast",
    # Endothelial
    "Endothelial": "Endothelial",
    # Unknown/Hybrid
    "Unlabeled": "Unknown",
    "T_Cell_&_Tumor_Hybrid": "Unknown",
}

# Ground truth -> COARSE level mapping
GT_TO_COARSE = {
    "Invasive_Tumor": "Epithelial",
    "DCIS_1": "Epithelial",
    "DCIS_2": "Epithelial",
    "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_ACTA2+": "Epithelial",
    "Myoepi_KRT15+": "Epithelial",
    "CD4+_T_Cells": "Immune",
    "CD8+_T_Cells": "Immune",
    "B_Cells": "Immune",
    "Macrophages_1": "Immune",
    "Macrophages_2": "Immune",
    "IRF7+_DCs": "Immune",
    "LAMP3+_DCs": "Immune",
    "Mast_Cells": "Immune",
    "Stromal": "Stromal",
    "Perivascular-Like": "Stromal",
    "Stromal_&_T_Cell_Hybrid": "Stromal",
    "Endothelial": "Endothelial",
    "Unlabeled": "Unknown",
    "T_Cell_&_Tumor_Hybrid": "Unknown",
}

# CellTypist prediction -> MEDIUM level mapping
CELLTYPIST_TO_MEDIUM = {
    # T cells
    "T cells": "T_Cell",
    "CD4+ T cells": "T_Cell",
    "CD8+ T cells": "T_Cell",
    "Tregs": "T_Cell",
    "Tem/Trm cytotoxic T cells": "T_Cell",
    "Tem/Effector helper T cells": "T_Cell",
    "Tcm/Naive helper T cells": "T_Cell",
    "Tcm/Naive cytotoxic T cells": "T_Cell",
    "Proliferative T cells": "T_Cell",
    "gdT": "T_Cell",
    "MAIT cells": "T_Cell",
    "NKT cells": "T_Cell",
    # B cells
    "B cells": "B_Cell",
    "Memory B cells": "B_Cell",
    "Naive B cells": "B_Cell",
    "Pro-B cells": "B_Cell",
    "Pre-B cells": "B_Cell",
    "Plasma cells": "B_Cell",
    "Plasmablasts": "B_Cell",
    # Myeloid
    "Macrophages": "Macrophage",
    "Classical monocytes": "Macrophage",
    "Non-classical monocytes": "Macrophage",
    "Intermediate monocytes": "Macrophage",
    "Monocytes": "Macrophage",
    "Kupffer cells": "Macrophage",
    "cDC1": "Dendritic_Cell",
    "cDC2": "Dendritic_Cell",
    "DC": "Dendritic_Cell",
    "pDC": "Dendritic_Cell",
    "DC1": "Dendritic_Cell",
    "DC2": "Dendritic_Cell",
    "DC3": "Dendritic_Cell",
    "Mast cells": "Mast_Cell",
    # NK
    "NK cells": "NK_Cell",
    # Stromal
    "Fibroblasts": "Fibroblast",
    "Smooth muscle cells": "Fibroblast",
    "Pericytes": "Perivascular",
    "Adipocytes": "Fibroblast",
    # Epithelial
    "Epithelial cells": "Tumor",  # Best guess for spatial
    # Endothelial
    "Endothelial cells": "Endothelial",
}

# scType prediction -> MEDIUM level mapping
SCTYPE_TO_MEDIUM = {
    "T cells": "T_Cell",
    "CD4+ T cells": "T_Cell",
    "CD8+ T cells": "T_Cell",
    "Regulatory T cells": "T_Cell",
    "B cells": "B_Cell",
    "Plasma cells": "B_Cell",
    "Macrophages": "Macrophage",
    "Monocytes": "Macrophage",
    "Dendritic cells": "Dendritic_Cell",
    "NK cells": "NK_Cell",
    "Mast cells": "Mast_Cell",
    "Fibroblasts": "Fibroblast",
    "Myofibroblasts": "Fibroblast",
    "Pericytes": "Perivascular",
    "Adipocytes": "Fibroblast",
    "Epithelial": "Tumor",
    "Endothelial cells": "Endothelial",
}


def load_xenium_data(xenium_path: Path, max_cells: int | None = None) -> tuple[sc.AnnData, np.ndarray | None]:
    """Load Xenium data into AnnData format."""
    import h5py
    import scipy.sparse as sp

    logger.info(f"Loading Xenium data from {xenium_path}")

    h5_path = xenium_path / "cell_feature_matrix.h5"
    with h5py.File(h5_path, "r") as f:
        matrix = f["matrix"]
        data = matrix["data"][:]
        indices = matrix["indices"][:]
        indptr = matrix["indptr"][:]
        shape = matrix["shape"][:]
        features = [x.decode() for x in matrix["features"]["name"][:]]

    X = sp.csr_matrix((data, indices, indptr), shape=(shape[1], shape[0]))

    selected_idx = None
    if max_cells and X.shape[0] > max_cells:
        selected_idx = np.sort(np.random.choice(X.shape[0], max_cells, replace=False))
        X = X[selected_idx]
        cell_ids = [f"cell_{i}" for i in selected_idx]
    else:
        cell_ids = [f"cell_{i}" for i in range(X.shape[0])]

    adata = sc.AnnData(X=X.toarray().astype("float32"))
    adata.obs_names = cell_ids
    adata.var_names = features

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    return adata, selected_idx


def load_ground_truth(xenium_path: Path, selected_indices: np.ndarray | None = None) -> dict[str, dict]:
    """Load ground truth at all granularity levels."""
    gt_file = list(xenium_path.glob("celltypes_ground_truth_*_supervised.xlsx"))[0]
    gt_df = pd.read_excel(gt_file)

    gt_dict = {"fine": {}, "medium": {}, "coarse": {}}

    if selected_indices is not None:
        indices_set = set(selected_indices)
        for i, row in gt_df.iterrows():
            if i in indices_set:
                cell_id = f"cell_{i}"
                fine_label = row["Cluster"]
                gt_dict["fine"][cell_id] = fine_label
                gt_dict["medium"][cell_id] = GT_TO_MEDIUM.get(fine_label, "Unknown")
                gt_dict["coarse"][cell_id] = GT_TO_COARSE.get(fine_label, "Unknown")
    else:
        for i, row in gt_df.iterrows():
            cell_id = f"cell_{i}"
            fine_label = row["Cluster"]
            gt_dict["fine"][cell_id] = fine_label
            gt_dict["medium"][cell_id] = GT_TO_MEDIUM.get(fine_label, "Unknown")
            gt_dict["coarse"][cell_id] = GT_TO_COARSE.get(fine_label, "Unknown")

    logger.info(f"Loaded {len(gt_dict['fine'])} ground truth labels")
    return gt_dict


def run_celltypist(adata: sc.AnnData, model_name: str) -> dict[str, dict]:
    """Run CellTypist and return predictions at all granularity levels."""
    logger.info(f"Running CellTypist {model_name}...")
    model = models.Model.load(model=model_name)
    predictions = celltypist.annotate(adata, model=model, majority_voting=False)

    results = {"fine": {}, "medium": {}, "coarse": {}}
    for cell_id, pred in zip(adata.obs_names, predictions.predicted_labels["predicted_labels"]):
        results["fine"][cell_id] = pred
        results["medium"][cell_id] = CELLTYPIST_TO_MEDIUM.get(pred, "Unknown")
        results["coarse"][cell_id] = map_celltypist_to_coarse(pred)
    return results


def run_sctype(adata: sc.AnnData) -> dict[str, dict]:
    """Run scType and return predictions at all granularity levels."""
    logger.info("Running scType...")
    annotator = ScTypeAnnotator()
    result = annotator.annotate(adata=adata)

    results = {"fine": {}, "medium": {}, "coarse": {}}
    for row in result.annotations_df.iter_rows(named=True):
        cell_id = row["cell_id"]
        fine_label = row["predicted_type"]
        results["fine"][cell_id] = fine_label
        results["medium"][cell_id] = SCTYPE_TO_MEDIUM.get(fine_label, "Unknown")
        results["coarse"][cell_id] = row["broad_category"]
    return results


def map_celltypist_to_coarse(label: str) -> str:
    """Map CellTypist predictions to coarse categories."""
    label_lower = label.lower()

    if any(x in label_lower for x in ["epithelial", "luminal", "basal", "keratinocyte"]):
        return "Epithelial"

    immune_patterns = [
        "t cell", "b cell", "macrophage", "monocyte", "dendritic", "dc",
        "nk", "plasma", "mast", "neutrophil", "eosinophil", "basophil",
        "granulocyte", "megakaryocyte", "erythroid", "progenitor"
    ]
    if any(p in label_lower for p in immune_patterns):
        return "Immune"

    if any(x in label_lower for x in ["fibroblast", "stromal", "smooth muscle", "adipocyte", "pericyte"]):
        return "Stromal"

    if any(x in label_lower for x in ["endothelial", "vascular"]):
        return "Endothelial"

    return "Unknown"


def hybrid_ensemble(ct_high: dict, ct_low: dict, sctype: dict, cell_ids: list, level: str) -> dict[str, str]:
    """
    Hybrid ensemble using cell-type-specific routing.

    Strategy varies by granularity level:
    - COARSE: Route based on broad category expertise
    - MEDIUM: Route based on cell type family
    - FINE: More conservative, prefer scType for consistency
    """
    results = {}

    for cell_id in cell_ids:
        ct_h = ct_high[level].get(cell_id, "Unknown")
        ct_l = ct_low[level].get(cell_id, "Unknown")
        sc = sctype[level].get(cell_id, "Unknown")

        preds = {ct_h, ct_l, sc}

        if level == "coarse":
            # Use coarse-level routing (proven strategy)
            if "Stromal" in preds:
                results[cell_id] = sc if sc == "Stromal" else (ct_h if ct_h == "Stromal" else sc)
            elif "Endothelial" in preds:
                results[cell_id] = ct_l if ct_l == "Endothelial" else sc
            elif "Epithelial" in preds:
                results[cell_id] = ct_h if ct_h == "Epithelial" else sc
            else:
                results[cell_id] = sc
        elif level == "medium":
            # Medium level: trust scType for stromal subtypes, CellTypist for immune subtypes
            stromal_types = {"Fibroblast", "Perivascular"}
            immune_types = {"T_Cell", "B_Cell", "Macrophage", "Dendritic_Cell", "Mast_Cell", "NK_Cell"}

            if sc in stromal_types:
                results[cell_id] = sc
            elif ct_h in immune_types or ct_l in immune_types:
                # Prefer CellTypist for immune subtypes (better training data)
                results[cell_id] = ct_h if ct_h in immune_types else ct_l
            elif sc == "Tumor" or ct_h == "Tumor":
                results[cell_id] = "Tumor"
            else:
                results[cell_id] = sc if sc != "Unknown" else ct_h
        else:
            # Fine level: Use majority voting, prefer scType on ties
            votes = [ct_h, ct_l, sc]
            from collections import Counter
            vote_counts = Counter(v for v in votes if v != "Unknown")
            if vote_counts:
                most_common = vote_counts.most_common(1)[0]
                results[cell_id] = most_common[0]
            else:
                results[cell_id] = sc

    return results


def evaluate_method(predictions: dict, ground_truth: dict, method_name: str, level: str) -> dict:
    """Evaluate predictions against ground truth."""
    common_cells = set(predictions.keys()) & set(ground_truth.keys())

    y_true = [ground_truth[c] for c in common_cells]
    y_pred = [predictions[c] for c in common_cells]

    # Get unique labels
    labels = sorted(set(y_true) | set(y_pred))

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    # Per-class F1
    per_class = {}
    for label in labels:
        f1 = f1_score(y_true, y_pred, labels=[label], average="macro", zero_division=0)
        support = sum(1 for y in y_true if y == label)
        per_class[label] = {"f1": f1, "support": support}

    unknown_rate = sum(1 for p in y_pred if p == "Unknown") / len(y_pred)

    return {
        "method": method_name,
        "level": level,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "unknown_rate": unknown_rate,
        "n_classes": len(labels),
        "per_class": per_class,
    }


def main():
    """Run multi-granularity evaluation."""
    max_cells = int(sys.argv[1]) if len(sys.argv) > 1 else None

    datasets = {
        "rep1": Path.home() / "datasets/raw/xenium/breast_tumor_rep1",
        "rep2": Path.home() / "datasets/raw/xenium/breast_tumor_rep2",
    }

    all_results = {}

    for name, path in datasets.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"DATASET: {name}")
        logger.info(f"{'='*80}")

        # Load data
        adata, selected_idx = load_xenium_data(path, max_cells)
        gt = load_ground_truth(path, selected_idx)
        cell_ids = list(adata.obs_names)

        # Run methods
        ct_high = run_celltypist(adata, "Immune_All_High.pkl")
        ct_low = run_celltypist(adata, "Immune_All_Low.pkl")
        sctype = run_sctype(adata)

        dataset_results = {}

        for level in ["coarse", "medium", "fine"]:
            logger.info(f"\n--- {level.upper()} LEVEL ---")

            # Run hybrid ensemble
            hybrid = hybrid_ensemble(ct_high, ct_low, sctype, cell_ids, level)

            # Evaluate all methods
            results = []
            methods = {
                "CellTypist_High": ct_high[level],
                "CellTypist_Low": ct_low[level],
                "scType": sctype[level],
                "Hybrid": hybrid,
            }

            for method_name, preds in methods.items():
                result = evaluate_method(preds, gt[level], method_name, level)
                results.append(result)
                logger.info(f"{method_name}: Acc={result['accuracy']:.3f}, F1={result['f1_macro']:.3f}, "
                           f"Classes={result['n_classes']}, Unk={result['unknown_rate']:.1%}")

            dataset_results[level] = results

        all_results[name] = dataset_results

    # Print summary
    print("\n" + "="*100)
    print("SUMMARY - MULTI-GRANULARITY EVALUATION")
    print("="*100)

    for level in ["coarse", "medium", "fine"]:
        print(f"\n## {level.upper()} LEVEL")
        print("-" * 90)

        for name in datasets.keys():
            print(f"\n### {name.upper()}")
            results = all_results[name][level]
            results_sorted = sorted(results, key=lambda x: x["f1_macro"], reverse=True)

            print(f"{'Method':<20} {'Acc':>8} {'F1 Macro':>10} {'F1 Wgt':>10} {'Classes':>8} {'Unk%':>8}")
            print("-" * 70)
            for r in results_sorted:
                print(f"{r['method']:<20} {r['accuracy']:>8.3f} {r['f1_macro']:>10.3f} "
                      f"{r['f1_weighted']:>10.3f} {r['n_classes']:>8} {r['unknown_rate']*100:>7.1f}%")

    # Save results
    output_dir = Path("benchmark_results/multigranularity")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        # Convert per_class to serializable format
        serializable = {}
        for name, dataset_results in all_results.items():
            serializable[name] = {}
            for level, results in dataset_results.items():
                serializable[name][level] = []
                for r in results:
                    r_copy = r.copy()
                    # Keep only top 10 classes for readability
                    if r_copy.get("per_class"):
                        top_classes = sorted(r_copy["per_class"].items(),
                                           key=lambda x: x[1]["support"], reverse=True)[:10]
                        r_copy["per_class_top10"] = dict(top_classes)
                        del r_copy["per_class"]
                    serializable[name][level].append(r_copy)
        json.dump(serializable, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
