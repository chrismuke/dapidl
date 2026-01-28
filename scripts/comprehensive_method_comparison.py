#!/usr/bin/env python3
"""Comprehensive comparison of all cell type annotation methods against ground truth.

Tests: CellTypist, SingleR, scType, scANVI, PopV-style ensemble
Datasets: Xenium breast rep1, rep2, combined
"""

from collections import defaultdict
from pathlib import Path

import celltypist
import h5py
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import scipy.sparse as sp
from celltypist import models
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Import our annotators
from dapidl.pipeline.components.annotators.sctype import ScTypeAnnotator, DEFAULT_MARKERS
from dapidl.pipeline.components.annotators.singler import SingleRAnnotator, is_singler_available

# Paths
REP1_PATH = Path.home() / "datasets/raw/xenium/breast_tumor_rep1"
REP2_PATH = Path.home() / "datasets/raw/xenium/breast_tumor_rep2"
GT_REP1 = REP1_PATH / "celltypes_ground_truth_rep1_supervised.xlsx"
GT_REP2 = REP2_PATH / "celltypes_ground_truth_rep2_supervised.xlsx"


def map_gt_to_coarse(label: str) -> str:
    """Map ground truth labels to coarse categories."""
    label_lower = label.lower()

    # Epithelial patterns
    if any(x in label_lower for x in ["invasive_tumor", "dcis", "prolif_invasive", "tumor"]):
        return "Epithelial"
    if "myoepi" in label_lower:
        return "Epithelial"

    # Immune patterns
    if any(x in label_lower for x in ["t_cell", "b_cell", "macrophage", "dc", "mast", "nk"]):
        return "Immune"
    if "cd4" in label_lower or "cd8" in label_lower:
        return "Immune"
    if "irf7" in label_lower or "lamp3" in label_lower:
        return "Immune"

    # Stromal patterns
    if label_lower == "stromal" or "fibroblast" in label_lower:
        return "Stromal"
    if "perivascular" in label_lower:
        return "Stromal"

    # Endothelial
    if "endothelial" in label_lower:
        return "Endothelial"

    # Hybrid/Unknown
    if "hybrid" in label_lower or "unlabeled" in label_lower:
        return "Unknown"

    return "Unknown"


def map_pred_to_coarse(label: str) -> str:
    """Map prediction labels to coarse categories."""
    label_lower = label.lower()

    if "unknown" in label_lower or "unassigned" in label_lower:
        return "Unknown"

    # Epithelial
    if any(x in label_lower for x in ["epithelial", "luminal", "basal", "keratinocyte", "mammary"]):
        return "Epithelial"

    # Immune
    if any(x in label_lower for x in [
        "t cell", "b cell", "macrophage", "monocyte", "dendritic", "dc",
        "nk", "plasma", "mast", "neutrophil", "immune", "lymphocyte"
    ]):
        return "Immune"

    # Stromal
    if any(x in label_lower for x in ["fibroblast", "stromal", "smooth muscle", "pericyte", "adipocyte", "mesenchymal"]):
        return "Stromal"

    # Endothelial
    if any(x in label_lower for x in ["endothelial", "vascular"]):
        return "Endothelial"

    return "Unknown"


def load_xenium_data(xenium_path: Path, max_cells: int | None = None) -> sc.AnnData:
    """Load Xenium data into AnnData format."""
    logger.info(f"Loading Xenium data from {xenium_path}")

    h5_path = xenium_path / "cell_feature_matrix.h5"
    with h5py.File(h5_path, "r") as f:
        matrix = f["matrix"]
        data = matrix["data"][:]
        indices = matrix["indices"][:]
        indptr = matrix["indptr"][:]
        shape = matrix["shape"][:]
        features = [x.decode() for x in matrix["features"]["name"][:]]
        barcodes = [x.decode() for x in f["matrix/barcodes"][:]]

    X = sp.csr_matrix((data, indices, indptr), shape=(shape[1], shape[0]))

    # Create AnnData
    adata = sc.AnnData(X=X.toarray().astype("float32"))
    adata.obs_names = barcodes
    adata.var_names = features
    adata.obs["cell_id"] = barcodes

    # Subsample if needed
    if max_cells and adata.n_obs > max_cells:
        idx = np.random.choice(adata.n_obs, max_cells, replace=False)
        adata = adata[idx].copy()
        logger.info(f"Subsampled to {max_cells} cells")

    # Basic preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.uns["log1p"] = {}

    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def load_ground_truth(gt_path: Path) -> dict[str, str]:
    """Load ground truth labels."""
    df = pd.read_excel(gt_path)
    barcode_col = df.columns[0]
    label_col = df.columns[1]

    gt_dict = {}
    for _, row in df.iterrows():
        barcode = str(row[barcode_col])
        label = str(row[label_col])
        gt_dict[barcode] = label

    logger.info(f"Loaded {len(gt_dict)} ground truth labels from {gt_path.name}")
    return gt_dict


def run_celltypist(adata: sc.AnnData, model_name: str) -> dict[str, str]:
    """Run CellTypist annotation."""
    logger.info(f"Running CellTypist {model_name}...")
    model = models.Model.load(model=model_name)
    predictions = celltypist.annotate(adata, model=model, majority_voting=True)

    results = {}
    pred_labels = predictions.predicted_labels["majority_voting"].tolist()
    for barcode, label in zip(adata.obs_names, pred_labels):
        results[barcode] = label

    return results


def run_sctype(adata: sc.AnnData) -> dict[str, str]:
    """Run scType annotation."""
    logger.info("Running scType annotation...")
    annotator = ScTypeAnnotator()
    result = annotator.annotate(adata=adata)

    results = {}
    for row in result.annotations_df.iter_rows(named=True):
        results[row["cell_id"]] = row["predicted_type"]

    return results


def run_singler(adata: sc.AnnData, reference: str = "hpca") -> dict[str, str]:
    """Run SingleR annotation."""
    if not is_singler_available():
        logger.warning("SingleR not available")
        return {}

    logger.info(f"Running SingleR with {reference} reference...")

    from dapidl.pipeline.components.annotators.singler import (
        SINGLER_REFERENCES,
        _fix_libstdcxx,
    )
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    _fix_libstdcxx()

    singler = importr("SingleR")
    celldex = importr("celldex")

    # Get reference data
    ref_func_name = SINGLER_REFERENCES[reference]
    ref_func = getattr(celldex, ref_func_name)
    ref_data = ref_func()

    # Prepare expression
    expr = adata.X.copy()
    genes = list(adata.var_names)

    # Get common genes
    ref_genes = list(ro.r.rownames(ref_data))
    common_genes = list(set(genes) & set(ref_genes))
    logger.info(f"Common genes: {len(common_genes)}/{len(genes)}")

    if len(common_genes) < 50:
        logger.warning("Too few common genes for SingleR")
        return {}

    # Subset and run
    gene_idx = [genes.index(g) for g in common_genes]
    expr_subset = expr[:, gene_idx]

    cell_names = list(adata.obs_names)
    dimnames = ro.r.list(ro.StrVector(common_genes), ro.StrVector(cell_names))
    expr_r = ro.r.matrix(
        ro.FloatVector(expr_subset.T.flatten()),
        nrow=len(common_genes),
        ncol=adata.n_obs,
        dimnames=dimnames,
    )

    ref_subset = ro.r["["](ref_data, ro.StrVector(common_genes), True)
    labels = ro.r("function(x) x$label.main")(ref_data)

    results_r = singler.SingleR(
        test=expr_r,
        ref=ref_subset,
        labels=labels,
        de_method="classic",
    )

    pred_labels = list(ro.r("function(x) x$labels")(results_r))

    results = {}
    for barcode, label in zip(cell_names, pred_labels):
        results[barcode] = label

    return results


def evaluate_method(
    predictions: dict[str, str],
    ground_truth: dict[str, str],
    method_name: str,
) -> dict:
    """Evaluate predictions against ground truth."""
    # Get common cells
    common_cells = set(predictions.keys()) & set(ground_truth.keys())
    if not common_cells:
        logger.warning(f"No common cells for {method_name}")
        return {}

    # Coarse mapping
    y_true = []
    y_pred = []
    for cell in common_cells:
        gt_label = ground_truth[cell]
        pred_label = predictions[cell]

        gt_coarse = map_gt_to_coarse(gt_label)
        pred_coarse = map_pred_to_coarse(pred_label)

        y_true.append(gt_coarse)
        y_pred.append(pred_coarse)

    # Calculate metrics
    labels = ["Epithelial", "Immune", "Stromal", "Endothelial", "Unknown"]
    valid_labels = [l for l in labels if l in set(y_true) | set(y_pred)]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=valid_labels, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=valid_labels, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=valid_labels)

    # Per-class stats
    class_results = {}
    for i, label in enumerate(valid_labels):
        class_results[label] = {
            "f1": f1_per_class[i],
            "support": sum(1 for x in y_true if x == label),
            "predicted": sum(1 for x in y_pred if x == label),
        }

    # Unknown rate
    unknown_rate = sum(1 for x in y_pred if x == "Unknown") / len(y_pred)

    results = {
        "method": method_name,
        "n_cells": len(common_cells),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "unknown_rate": unknown_rate,
        "per_class": class_results,
        "confusion_matrix": cm.tolist(),
        "labels": valid_labels,
    }

    logger.info(f"{method_name}: Acc={acc:.3f}, F1={f1_macro:.3f}, Unknown={unknown_rate:.1%}")
    for label, stats in class_results.items():
        if stats["support"] > 0:
            logger.info(f"  {label}: F1={stats['f1']:.3f} (n={stats['support']})")

    return results


def run_comprehensive_comparison(dataset_name: str, xenium_path: Path, gt_path: Path, max_cells: int | None = None):
    """Run comprehensive comparison for a dataset."""
    logger.info(f"\n{'='*80}")
    logger.info(f"DATASET: {dataset_name}")
    logger.info(f"{'='*80}")

    # Load data
    adata = load_xenium_data(xenium_path, max_cells=max_cells)
    ground_truth = load_ground_truth(gt_path)

    results = []

    # 1. CellTypist models
    celltypist_models = [
        "Immune_All_High.pkl",
        "Immune_All_Low.pkl",
        "Cells_Adult_Breast.pkl",
    ]

    for model_name in celltypist_models:
        try:
            preds = run_celltypist(adata, model_name)
            result = evaluate_method(preds, ground_truth, f"CellTypist_{model_name.replace('.pkl', '')}")
            results.append(result)
        except Exception as e:
            logger.error(f"CellTypist {model_name} failed: {e}")

    # 2. scType
    try:
        preds = run_sctype(adata)
        result = evaluate_method(preds, ground_truth, "scType")
        results.append(result)
    except Exception as e:
        logger.error(f"scType failed: {e}")

    # 3. SingleR
    if is_singler_available():
        for ref in ["hpca", "blueprint"]:
            try:
                preds = run_singler(adata, ref)
                if preds:
                    result = evaluate_method(preds, ground_truth, f"SingleR_{ref}")
                    results.append(result)
            except Exception as e:
                logger.error(f"SingleR {ref} failed: {e}")

    # 4. Multi-model ensemble (CellTypist only)
    try:
        logger.info("Running CellTypist ensemble...")
        all_preds = {}
        for model_name in celltypist_models:
            model = models.Model.load(model=model_name)
            predictions = celltypist.annotate(adata, model=model, majority_voting=True)
            pred_labels = predictions.predicted_labels["majority_voting"].tolist()
            for barcode, label in zip(adata.obs_names, pred_labels):
                if barcode not in all_preds:
                    all_preds[barcode] = []
                all_preds[barcode].append(map_pred_to_coarse(label))

        # Majority vote
        ensemble_preds = {}
        for barcode, votes in all_preds.items():
            from collections import Counter
            counts = Counter(votes)
            # Prefer non-Unknown
            valid_votes = [(v, c) for v, c in counts.items() if v != "Unknown"]
            if valid_votes:
                ensemble_preds[barcode] = max(valid_votes, key=lambda x: x[1])[0]
            else:
                ensemble_preds[barcode] = counts.most_common(1)[0][0]

        # Map back to original format for evaluation
        ensemble_preds_full = {k: v for k, v in ensemble_preds.items()}
        result = evaluate_method(ensemble_preds_full, ground_truth, "CellTypist_Ensemble")
        results.append(result)
    except Exception as e:
        logger.error(f"Ensemble failed: {e}")

    return results


def analyze_complementarity(all_results: dict[str, list[dict]]):
    """Analyze which methods are good at which cell types."""
    print("\n" + "=" * 80)
    print("COMPLEMENTARITY ANALYSIS")
    print("=" * 80)

    # Collect per-class F1 across methods
    class_scores = defaultdict(lambda: defaultdict(list))

    for dataset, results in all_results.items():
        for result in results:
            if "per_class" not in result:
                continue
            method = result["method"]
            for cell_type, stats in result["per_class"].items():
                if stats["support"] > 0:
                    class_scores[cell_type][method].append({
                        "f1": stats["f1"],
                        "dataset": dataset,
                    })

    # Find best method for each cell type
    print("\n## Best Method Per Cell Type")
    print("-" * 60)

    for cell_type in ["Epithelial", "Immune", "Stromal", "Endothelial"]:
        if cell_type not in class_scores:
            continue

        print(f"\n### {cell_type}")
        method_avg = {}
        for method, scores in class_scores[cell_type].items():
            avg_f1 = np.mean([s["f1"] for s in scores])
            method_avg[method] = avg_f1

        # Sort by avg F1
        sorted_methods = sorted(method_avg.items(), key=lambda x: x[1], reverse=True)
        for method, f1 in sorted_methods[:5]:
            print(f"  {method}: F1={f1:.3f}")


def main():
    """Run comprehensive comparison."""
    import sys

    max_cells = int(sys.argv[1]) if len(sys.argv) > 1 else None

    all_results = {}

    # Run on rep1
    results_rep1 = run_comprehensive_comparison(
        "rep1", REP1_PATH, GT_REP1, max_cells=max_cells
    )
    all_results["rep1"] = results_rep1

    # Run on rep2
    results_rep2 = run_comprehensive_comparison(
        "rep2", REP2_PATH, GT_REP2, max_cells=max_cells
    )
    all_results["rep2"] = results_rep2

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n## Rep1 Results")
    print(f"{'Method':<35} {'Acc':<8} {'F1':<8} {'Epi':<8} {'Imm':<8} {'Str':<8} {'Unk%':<8}")
    print("-" * 90)
    for r in sorted(results_rep1, key=lambda x: x.get("f1_macro", 0), reverse=True):
        epi = r.get("per_class", {}).get("Epithelial", {}).get("f1", 0)
        imm = r.get("per_class", {}).get("Immune", {}).get("f1", 0)
        stro = r.get("per_class", {}).get("Stromal", {}).get("f1", 0)
        print(f"{r['method']:<35} {r['accuracy']:.3f}   {r['f1_macro']:.3f}   {epi:.3f}   {imm:.3f}   {stro:.3f}   {r['unknown_rate']:.1%}")

    print("\n## Rep2 Results")
    print(f"{'Method':<35} {'Acc':<8} {'F1':<8} {'Epi':<8} {'Imm':<8} {'Str':<8} {'Unk%':<8}")
    print("-" * 90)
    for r in sorted(results_rep2, key=lambda x: x.get("f1_macro", 0), reverse=True):
        epi = r.get("per_class", {}).get("Epithelial", {}).get("f1", 0)
        imm = r.get("per_class", {}).get("Immune", {}).get("f1", 0)
        stro = r.get("per_class", {}).get("Stromal", {}).get("f1", 0)
        print(f"{r['method']:<35} {r['accuracy']:.3f}   {r['f1_macro']:.3f}   {epi:.3f}   {imm:.3f}   {stro:.3f}   {r['unknown_rate']:.1%}")

    # Complementarity analysis
    analyze_complementarity(all_results)

    # Save results
    output_dir = Path("benchmark_results/comprehensive_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
