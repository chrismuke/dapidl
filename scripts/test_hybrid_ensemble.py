#!/usr/bin/env python3
"""
Test hybrid ensemble combining CellTypist (best for Epithelial/Endothelial)
with scType (best for Stromal/Immune).

Based on benchmark findings:
- Epithelial: CellTypist_Immune_All_High F1=0.96 vs scType F1=0.91
- Stromal: scType F1=0.69 vs CellTypist F1=0.09 (7x better!)
- Immune: scType F1=0.72 vs CellTypist_Immune_All_Low F1=0.71
- Endothelial: CellTypist_Immune_All_Low F1=0.56 vs scType F1=0.28
"""

from pathlib import Path
import json
import sys

import celltypist
from celltypist import models
import numpy as np
import polars as pl
import scanpy as sc
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from dapidl.pipeline.components.annotators.sctype import ScTypeAnnotator


def load_xenium_data(xenium_path: Path, max_cells: int | None = None) -> tuple[sc.AnnData, np.ndarray | None]:
    """Load Xenium data into AnnData format.

    Returns:
        Tuple of (adata, selected_indices) where selected_indices is None if no subsampling.
    """
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


def load_ground_truth(xenium_path: Path, selected_indices: np.ndarray | None = None) -> dict[str, str]:
    """Load ground truth labels.

    Args:
        xenium_path: Path to Xenium output directory
        selected_indices: If provided, only load GT for these cell indices
    """
    import pandas as pd

    # Try different GT file patterns
    gt_patterns = [
        "celltypes_ground_truth_*_supervised.xlsx",
        "*ground_truth*.xlsx",
        "*groundtruth*.xlsx",
    ]

    gt_file = None
    for pattern in gt_patterns:
        files = list(xenium_path.glob(pattern))
        if files:
            gt_file = files[0]
            break

    if gt_file is None:
        raise FileNotFoundError(f"No ground truth file found in {xenium_path}")

    gt_df = pd.read_excel(gt_file)

    # Determine which column has cell types
    if "Cluster" in gt_df.columns:
        label_col = "Cluster"
    elif "celltype_final" in gt_df.columns:
        label_col = "celltype_final"
    elif "cell_type" in gt_df.columns:
        label_col = "cell_type"
    else:
        raise ValueError(f"Cannot find cell type column in {gt_file}. Columns: {gt_df.columns.tolist()}")

    # Build cell_id -> cell_type mapping
    gt_dict = {}

    if selected_indices is not None:
        # Only load for selected indices
        indices_set = set(selected_indices)
        for i, row in gt_df.iterrows():
            if i in indices_set:
                cell_id = f"cell_{i}"
                cell_type = row[label_col]
                gt_dict[cell_id] = map_gt_to_coarse(cell_type)
    else:
        for i, row in gt_df.iterrows():
            cell_id = f"cell_{i}"
            cell_type = row[label_col]
            gt_dict[cell_id] = map_gt_to_coarse(cell_type)

    logger.info(f"Loaded {len(gt_dict)} ground truth labels")
    return gt_dict


def map_gt_to_coarse(label: str) -> str:
    """Map ground truth labels to coarse categories."""
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return "Unknown"

    label_lower = str(label).lower()

    # Unknown patterns
    if "unknown" in label_lower or label == "unlabeled":
        return "Unknown"

    # Epithelial patterns (including tumor cells which are epithelial)
    epithelial_patterns = [
        "epithelial", "luminal", "basal", "invasive_tumor", "dcis",
        "tumor", "cancer", "carcinoma", "adenocarcinoma", "lumsec", "lumhr"
    ]
    if any(p in label_lower for p in epithelial_patterns):
        return "Epithelial"

    # Immune patterns
    immune_patterns = [
        "t_cell", "b_cell", "macrophage", "monocyte", "dendritic", "dc_",
        "nk_cell", "plasma", "mast", "neutrophil", "eosinophil", "basophil",
        "lymphocyte", "immune", "cd4", "cd8", "treg", "nkt"
    ]
    if any(p in label_lower for p in immune_patterns):
        return "Immune"

    # Stromal patterns
    stromal_patterns = [
        "fibroblast", "stromal", "mesenchymal", "caf", "myofibroblast",
        "smooth_muscle", "adipocyte", "pericyte"
    ]
    if any(p in label_lower for p in stromal_patterns):
        return "Stromal"

    # Endothelial patterns
    endothelial_patterns = ["endothelial", "vascular", "vessel", "lymphatic_vessel", "perivascular"]
    if any(p in label_lower for p in endothelial_patterns):
        return "Endothelial"

    return "Unknown"


def run_celltypist_high(adata: sc.AnnData) -> dict[str, str]:
    """Run CellTypist Immune_All_High."""
    logger.info("Running CellTypist Immune_All_High...")
    model = models.Model.load(model="Immune_All_High.pkl")
    predictions = celltypist.annotate(adata, model=model, majority_voting=False)

    results = {}
    for cell_id, pred in zip(adata.obs_names, predictions.predicted_labels["predicted_labels"]):
        results[cell_id] = map_celltypist_to_coarse(pred)
    return results


def run_celltypist_low(adata: sc.AnnData) -> dict[str, str]:
    """Run CellTypist Immune_All_Low."""
    logger.info("Running CellTypist Immune_All_Low...")
    model = models.Model.load(model="Immune_All_Low.pkl")
    predictions = celltypist.annotate(adata, model=model, majority_voting=False)

    results = {}
    for cell_id, pred in zip(adata.obs_names, predictions.predicted_labels["predicted_labels"]):
        results[cell_id] = map_celltypist_to_coarse(pred)
    return results


def run_sctype(adata: sc.AnnData) -> dict[str, str]:
    """Run scType annotation."""
    logger.info("Running scType...")
    annotator = ScTypeAnnotator()
    result = annotator.annotate(adata=adata)

    results = {}
    for row in result.annotations_df.iter_rows(named=True):
        results[row["cell_id"]] = row["broad_category"]
    return results


def map_celltypist_to_coarse(label: str) -> str:
    """Map CellTypist predictions to coarse categories."""
    label_lower = label.lower()

    # Epithelial
    if any(x in label_lower for x in ["epithelial", "luminal", "basal", "keratinocyte"]):
        return "Epithelial"

    # Immune - CellTypist has extensive immune subtypes
    immune_patterns = [
        "t cell", "b cell", "macrophage", "monocyte", "dendritic", "dc",
        "nk", "plasma", "mast", "neutrophil", "eosinophil", "basophil",
        "granulocyte", "megakaryocyte", "erythroid", "progenitor"
    ]
    if any(p in label_lower for p in immune_patterns):
        return "Immune"

    # Stromal
    if any(x in label_lower for x in ["fibroblast", "stromal", "smooth muscle", "adipocyte", "pericyte"]):
        return "Stromal"

    # Endothelial
    if any(x in label_lower for x in ["endothelial", "vascular"]):
        return "Endothelial"

    return "Unknown"


def hybrid_ensemble_v1(ct_high: dict, ct_low: dict, sctype: dict, cell_ids: list) -> dict[str, str]:
    """
    Hybrid ensemble v1: Use best method per cell type.

    Strategy:
    - If both methods agree -> use that
    - If CellTypist says Epithelial and scType says something else -> trust CellTypist (F1=0.96)
    - If scType says Stromal and CellTypist says something else -> trust scType (F1=0.69 vs 0.09)
    - If CellTypist_Low says Endothelial -> trust it (F1=0.56 vs 0.28)
    - For Immune: prefer scType slightly (F1=0.72 vs 0.71)
    """
    results = {}

    for cell_id in cell_ids:
        ct_h = ct_high.get(cell_id, "Unknown")
        ct_l = ct_low.get(cell_id, "Unknown")
        sc = sctype.get(cell_id, "Unknown")

        # If all agree
        if ct_h == ct_l == sc:
            results[cell_id] = ct_h
            continue

        # If two agree
        votes = [ct_h, ct_l, sc]
        for v in ["Epithelial", "Immune", "Stromal", "Endothelial"]:
            if votes.count(v) >= 2:
                results[cell_id] = v
                break
        else:
            # No majority - use method-specific expertise
            if sc == "Stromal":
                # Trust scType for Stromal (7x better!)
                results[cell_id] = "Stromal"
            elif ct_h == "Epithelial" or ct_l == "Epithelial":
                # Trust CellTypist for Epithelial
                results[cell_id] = "Epithelial"
            elif ct_l == "Endothelial":
                # Trust CellTypist_Low for Endothelial
                results[cell_id] = "Endothelial"
            elif sc == "Immune":
                # Trust scType for Immune
                results[cell_id] = "Immune"
            else:
                # Fallback to majority or scType
                results[cell_id] = sc

    return results


def hybrid_ensemble_v2(ct_high: dict, ct_low: dict, sctype: dict, cell_ids: list) -> dict[str, str]:
    """
    Hybrid ensemble v2: Cell-type-specific routing.

    For each cell:
    1. Get predictions from all methods
    2. Route based on what methods predict:
       - If any says Stromal -> check scType (it's 7x better)
       - If only Epithelial predictions -> trust CellTypist_High
       - If Endothelial in play -> trust CellTypist_Low
    """
    results = {}

    for cell_id in cell_ids:
        ct_h = ct_high.get(cell_id, "Unknown")
        ct_l = ct_low.get(cell_id, "Unknown")
        sc = sctype.get(cell_id, "Unknown")

        preds = {ct_h, ct_l, sc}

        # Routing logic based on benchmarked strengths
        if "Stromal" in preds:
            # If any method thinks Stromal, check scType (7x better)
            if sc == "Stromal":
                results[cell_id] = "Stromal"
            elif ct_h == ct_l == "Stromal":
                results[cell_id] = "Stromal"
            else:
                # Conflict - scType is right about Stromal, others aren't
                # But if scType says something else, it might be misclassifying stromal
                results[cell_id] = sc if sc != "Unknown" else ct_h
        elif "Endothelial" in preds:
            # CellTypist_Low is best for Endothelial
            if ct_l == "Endothelial":
                results[cell_id] = "Endothelial"
            elif ct_h == "Endothelial":
                results[cell_id] = "Endothelial"
            else:
                results[cell_id] = sc
        elif "Epithelial" in preds:
            # CellTypist is best for Epithelial
            if ct_h == "Epithelial" or ct_l == "Epithelial":
                results[cell_id] = "Epithelial"
            else:
                results[cell_id] = sc
        elif "Immune" in preds:
            # scType slightly better for Immune
            if sc == "Immune":
                results[cell_id] = "Immune"
            elif ct_h == "Immune" or ct_l == "Immune":
                results[cell_id] = "Immune"
            else:
                results[cell_id] = "Unknown"
        else:
            # All unknown
            results[cell_id] = "Unknown"

    return results


def evaluate_method(predictions: dict, ground_truth: dict, method_name: str) -> dict:
    """Evaluate predictions against ground truth."""
    common_cells = set(predictions.keys()) & set(ground_truth.keys())

    y_true = [ground_truth[c] for c in common_cells]
    y_pred = [predictions[c] for c in common_cells]

    labels = ["Epithelial", "Immune", "Stromal", "Endothelial", "Unknown"]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    # Per-class F1
    per_class = {}
    for label in labels:
        f1 = f1_score(y_true, y_pred, labels=[label], average="macro", zero_division=0)
        support = sum(1 for y in y_true if y == label)
        per_class[label] = {"f1": f1, "support": support}

    # Unknown rate
    unknown_rate = sum(1 for p in y_pred if p == "Unknown") / len(y_pred)

    return {
        "method": method_name,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "unknown_rate": unknown_rate,
        "per_class": per_class,
    }


def main():
    """Run hybrid ensemble evaluation."""
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

        # Run individual methods
        ct_high = run_celltypist_high(adata)
        ct_low = run_celltypist_low(adata)
        sctype = run_sctype(adata)

        # Run hybrid ensembles
        hybrid_v1 = hybrid_ensemble_v1(ct_high, ct_low, sctype, cell_ids)
        hybrid_v2 = hybrid_ensemble_v2(ct_high, ct_low, sctype, cell_ids)

        # Evaluate all methods
        results = []
        methods = {
            "CellTypist_High": ct_high,
            "CellTypist_Low": ct_low,
            "scType": sctype,
            "Hybrid_v1": hybrid_v1,
            "Hybrid_v2": hybrid_v2,
        }

        for method_name, preds in methods.items():
            result = evaluate_method(preds, gt, method_name)
            results.append(result)
            logger.info(f"{method_name}: Acc={result['accuracy']:.3f}, F1={result['f1_macro']:.3f}")

        all_results[name] = results

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - HYBRID ENSEMBLE EVALUATION")
    print("="*80)

    for name, results in all_results.items():
        print(f"\n## {name.upper()}")
        print(f"{'Method':<20} {'Acc':>8} {'F1':>8} {'Epi':>8} {'Imm':>8} {'Str':>8} {'End':>8} {'Unk%':>8}")
        print("-" * 88)

        # Sort by F1
        results_sorted = sorted(results, key=lambda x: x["f1_macro"], reverse=True)

        for r in results_sorted:
            print(f"{r['method']:<20} {r['accuracy']:>8.3f} {r['f1_macro']:>8.3f} "
                  f"{r['per_class']['Epithelial']['f1']:>8.3f} "
                  f"{r['per_class']['Immune']['f1']:>8.3f} "
                  f"{r['per_class']['Stromal']['f1']:>8.3f} "
                  f"{r['per_class']['Endothelial']['f1']:>8.3f} "
                  f"{r['unknown_rate']*100:>7.1f}%")

    # Save results
    output_dir = Path("benchmark_results/hybrid_ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
