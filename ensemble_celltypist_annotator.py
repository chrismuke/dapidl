#!/usr/bin/env python3
"""
Ensemble CellTypist Annotator for Improved Stromal Detection

Strategy based on benchmark findings:
- Primary model: Immune_All_High.pkl (F1=0.5003) - best for Epithelial + Immune
- Stromal specialist: Fetal_Human_AdrenalGlands.pkl (F1=0.723 for Stromal)

The ensemble combines both models to achieve better detection across all cell types.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import anndata as ad
import celltypist
import numpy as np
import pandas as pd
import scanpy as sc
from celltypist import models
from loguru import logger

# Ground truth mapping from fine-grained to broad categories
GROUND_TRUTH_MAPPING = {
    # Epithelial (from supervised sheet in Cell_Barcode_Type_Matrices.xlsx)
    "DCIS_1": "Epithelial",
    "DCIS_2": "Epithelial",
    "Invasive_Tumor": "Epithelial",
    "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_ACTA2+": "Epithelial",
    "Myoepi_KRT15+": "Epithelial",
    # Immune
    "B_Cells": "Immune",
    "CD4+_T_Cells": "Immune",
    "CD8+_T_Cells": "Immune",
    "Macrophages_1": "Immune",
    "Macrophages_2": "Immune",
    "Mast_Cells": "Immune",
    "IRF7+_DCs": "Immune",
    "LAMP3+_DCs": "Immune",
    # Stromal
    "Stromal": "Stromal",
    "Endothelial": "Stromal",
    "Perivascular-Like": "Stromal",
    # Ambiguous/hybrid cells - exclude from evaluation
    "Stromal_&_T_Cell_Hybrid": None,  # ambiguous
    "T_Cell_&_Tumor_Hybrid": None,  # ambiguous
    "Unlabeled": None,  # unlabeled cells
}

# CellTypist to broad category mapping for Immune_All_High model
IMMUNE_MODEL_MAPPING = {
    # Immune cells -> Immune
    "B cells": "Immune",
    "B-cell lineage": "Immune",
    "CD4+ T cells": "Immune",
    "CD8+ T cells": "Immune",
    "T cells": "Immune",
    "NK cells": "Immune",
    "Monocytes": "Immune",
    "Macrophages": "Immune",
    "Dendritic cells": "Immune",
    "Mast cells": "Immune",
    "Plasma cells": "Immune",
    "ILC": "Immune",
    "Granulocytes": "Immune",
    "Neutrophils": "Immune",
    "Eosinophils": "Immune",
    "Basophils": "Immune",
    # Epithelial indicators
    "Epithelial cells": "Epithelial",
    "Tumor cells": "Epithelial",
    # Everything else defaults to checking secondary model
}

# CellTypist to broad category mapping for Fetal_Human_AdrenalGlands model
ADRENAL_MODEL_MAPPING = {
    # Stromal cells
    "Fibroblasts": "Stromal",
    "Stromal cells": "Stromal",
    "Mesenchymal cells": "Stromal",
    "Smooth muscle cells": "Stromal",
    "Endothelial cells": "Stromal",
    "Pericytes": "Stromal",
    "Capsular cells": "Stromal",
    # Immune cells
    "Immune cells": "Immune",
    "Macrophages": "Immune",
    "T cells": "Immune",
    # Epithelial/steroidogenic (often misclassified)
    "Adrenocortical cells": "Epithelial",
    "Chromaffin cells": "Epithelial",
}


def map_predictions_to_broad(
    predictions: pd.Series,
    mapping: dict[str, str],
    default: str = "Unknown"
) -> pd.Series:
    """Map CellTypist predictions to broad categories."""

    def map_single(pred: str) -> str:
        # Direct match
        if pred in mapping:
            return mapping[pred]

        # Partial match (for hierarchical labels like "B cells memory")
        pred_lower = pred.lower()
        for key, value in mapping.items():
            if key.lower() in pred_lower or pred_lower in key.lower():
                return value

        # Keyword-based fallback
        if any(kw in pred_lower for kw in ["b cell", "t cell", "nk ", "monocyte", "macrophage",
                                            "dendritic", "mast", "plasma", "neutro", "eosino",
                                            "baso", "lympho", "immune"]):
            return "Immune"
        if any(kw in pred_lower for kw in ["epithelial", "tumor", "carcinoma", "dcis"]):
            return "Epithelial"
        if any(kw in pred_lower for kw in ["fibroblast", "stromal", "endothelial", "pericyte",
                                            "smooth muscle", "mesenchym", "vascular"]):
            return "Stromal"

        return default

    return predictions.apply(map_single)


class EnsembleCellTypistAnnotator:
    """
    Ensemble annotator combining multiple CellTypist models for improved accuracy.

    Strategy:
    1. Run primary model (Immune_All_High) for Epithelial + Immune detection
    2. Run stromal specialist (Fetal_Human_AdrenalGlands) for Stromal detection
    3. Combine predictions with confidence-weighted voting
    """

    def __init__(
        self,
        primary_model: str = "Immune_All_High.pkl",
        stromal_model: str = "Fetal_Human_AdrenalGlands.pkl",
        leiden_resolution: float = 0.5,
    ):
        self.primary_model_name = primary_model
        self.stromal_model_name = stromal_model
        self.leiden_resolution = leiden_resolution

        # Load models
        logger.info(f"Loading primary model: {primary_model}")
        models.download_models(model=primary_model, force_update=False)
        self.primary_model = models.Model.load(model=primary_model)

        logger.info(f"Loading stromal specialist model: {stromal_model}")
        models.download_models(model=stromal_model, force_update=False)
        self.stromal_model = models.Model.load(model=stromal_model)

    def annotate(
        self,
        adata: ad.AnnData,
        majority_voting: bool = True,
        stromal_confidence_threshold: float = 0.3,
    ) -> pd.DataFrame:
        """
        Annotate cells using ensemble strategy.

        Args:
            adata: Normalized AnnData object (log1p normalized to 10k)
            majority_voting: Whether to use majority voting (recommended)
            stromal_confidence_threshold: Minimum confidence for stromal override

        Returns:
            DataFrame with columns: predicted_labels, broad_category, confidence, model_used
        """
        start_time = time.time()

        # Prepare over-clustering if using majority voting
        oc_param = None
        if majority_voting:
            logger.info(f"Computing Leiden clustering at resolution {self.leiden_resolution}...")
            if 'neighbors' not in adata.uns:
                sc.pp.neighbors(adata, n_neighbors=15)
            sc.tl.leiden(adata, resolution=self.leiden_resolution, key_added='ensemble_clustering')
            oc_param = adata.obs['ensemble_clustering']

        # Run primary model
        logger.info("Running primary model (Immune_All_High)...")
        primary_result = celltypist.annotate(
            adata,
            model=self.primary_model,
            majority_voting=majority_voting,
            mode="best match",
            over_clustering=oc_param,
        )

        primary_labels = primary_result.predicted_labels
        label_col = "majority_voting" if majority_voting and "majority_voting" in primary_labels.columns else "predicted_labels"
        primary_preds = primary_labels[label_col]
        primary_conf = primary_result.probability_matrix.max(axis=1)

        # Run stromal specialist model
        logger.info("Running stromal specialist model (Fetal_Human_AdrenalGlands)...")
        stromal_result = celltypist.annotate(
            adata,
            model=self.stromal_model,
            majority_voting=majority_voting,
            mode="best match",
            over_clustering=oc_param,
        )

        stromal_labels = stromal_result.predicted_labels
        stromal_col = "majority_voting" if majority_voting and "majority_voting" in stromal_labels.columns else "predicted_labels"
        stromal_preds = stromal_labels[stromal_col]
        stromal_conf = stromal_result.probability_matrix.max(axis=1)

        # Map to broad categories
        logger.info("Mapping predictions to broad categories...")
        primary_broad = map_predictions_to_broad(primary_preds, IMMUNE_MODEL_MAPPING)
        stromal_broad = map_predictions_to_broad(stromal_preds, ADRENAL_MODEL_MAPPING)

        # Ensemble combination strategy
        logger.info("Combining ensemble predictions...")
        n_cells = len(adata)
        final_labels = []
        final_broad = []
        final_conf = []
        model_used = []

        for i in range(n_cells):
            primary_cat = primary_broad.iloc[i]
            stromal_cat = stromal_broad.iloc[i]
            primary_c = primary_conf.iloc[i]
            stromal_c = stromal_conf.iloc[i]

            # Decision logic:
            # 1. If stromal model says Stromal with decent confidence, trust it
            # 2. Otherwise, use primary model prediction

            if stromal_cat == "Stromal" and stromal_c >= stromal_confidence_threshold:
                # Stromal specialist detected stromal cell
                final_labels.append(stromal_preds.iloc[i])
                final_broad.append("Stromal")
                final_conf.append(stromal_c)
                model_used.append("stromal_specialist")
            elif primary_cat == "Stromal":
                # Primary model detected stromal (rare but trust it)
                final_labels.append(primary_preds.iloc[i])
                final_broad.append("Stromal")
                final_conf.append(primary_c)
                model_used.append("primary")
            else:
                # Use primary model for Epithelial/Immune
                final_labels.append(primary_preds.iloc[i])
                final_broad.append(primary_cat)
                final_conf.append(primary_c)
                model_used.append("primary")

        # Create result DataFrame
        result_df = pd.DataFrame({
            'cell_id': adata.obs_names,
            'predicted_labels': final_labels,
            'broad_category': final_broad,
            'confidence': final_conf,
            'model_used': model_used,
            'primary_prediction': primary_preds.values,
            'primary_broad': primary_broad.values,
            'primary_confidence': primary_conf.values,
            'stromal_prediction': stromal_preds.values,
            'stromal_broad': stromal_broad.values,
            'stromal_confidence': stromal_conf.values,
        })

        elapsed = time.time() - start_time
        logger.info(f"Ensemble annotation completed in {elapsed:.1f}s")

        # Summary statistics
        broad_counts = result_df['broad_category'].value_counts()
        model_counts = result_df['model_used'].value_counts()

        logger.info("Ensemble results:")
        logger.info(f"  Total cells: {n_cells:,}")
        for cat, count in broad_counts.items():
            logger.info(f"  {cat}: {count:,} ({100*count/n_cells:.1f}%)")
        logger.info("Model usage:")
        for model, count in model_counts.items():
            logger.info(f"  {model}: {count:,} ({100*count/n_cells:.1f}%)")

        return result_df


def load_and_normalize_xenium(xenium_path: str | Path) -> ad.AnnData:
    """Load and normalize Xenium expression data for CellTypist."""
    import h5py
    import scipy.sparse as sp

    xenium_path = Path(xenium_path)

    # Load expression matrix
    h5_path = xenium_path / "outs" / "cell_feature_matrix.h5"
    if not h5_path.exists():
        h5_path = xenium_path / "cell_feature_matrix.h5"

    logger.info(f"Loading expression from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        matrix_group = f['matrix']
        data = matrix_group['data'][:]
        indices = matrix_group['indices'][:]
        indptr = matrix_group['indptr'][:]
        shape = matrix_group['shape'][:]

        features = matrix_group['features']
        gene_names = [x.decode() for x in features['name'][:]]

        barcodes = matrix_group['barcodes']
        cell_ids = [x.decode() for x in barcodes[:]]

    # Create sparse matrix (genes x cells) -> transpose to (cells x genes)
    X = sp.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()

    adata = ad.AnnData(X)
    adata.obs_names = cell_ids
    adata.var_names = gene_names

    logger.info(f"Loaded: {adata.n_obs:,} cells x {adata.n_vars} genes")

    # Normalize for CellTypist
    logger.info("Normalizing (10k counts + log1p)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


def evaluate_ensemble(
    result_df: pd.DataFrame,
    ground_truth_path: str | Path,
) -> dict[str, Any]:
    """Evaluate ensemble predictions against ground truth."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

    # Load ground truth from supervised sheet
    gt_df = pd.read_excel(
        ground_truth_path,
        sheet_name="Xenium R1 Fig1-5 (supervised)",
    )
    gt_df['Barcode'] = gt_df['Barcode'].astype(str)

    # Map ground truth to broad categories using 'Cluster' column
    gt_df['broad_category'] = gt_df['Cluster'].map(GROUND_TRUTH_MAPPING)
    gt_df = gt_df[gt_df['broad_category'].notna()]

    # Match predictions to ground truth
    gt_map = dict(zip(gt_df['Barcode'], gt_df['broad_category']))

    matched_pred = []
    matched_gt = []

    for _, row in result_df.iterrows():
        cell_id = str(row['cell_id'])
        if cell_id in gt_map:
            matched_pred.append(row['broad_category'])
            matched_gt.append(gt_map[cell_id])

    if not matched_pred:
        logger.warning("No cells matched between predictions and ground truth!")
        return {}

    # Calculate metrics
    labels = ['Epithelial', 'Immune', 'Stromal']

    metrics = {
        'n_matched': len(matched_pred),
        'accuracy': accuracy_score(matched_gt, matched_pred),
        'f1_macro': f1_score(matched_gt, matched_pred, labels=labels, average='macro', zero_division=0),
        'f1_weighted': f1_score(matched_gt, matched_pred, labels=labels, average='weighted', zero_division=0),
        'precision_macro': precision_score(matched_gt, matched_pred, labels=labels, average='macro', zero_division=0),
        'recall_macro': recall_score(matched_gt, matched_pred, labels=labels, average='macro', zero_division=0),
    }

    # Per-class metrics
    report = classification_report(matched_gt, matched_pred, labels=labels, output_dict=True, zero_division=0)
    metrics['per_class'] = {
        label: {
            'precision': report[label]['precision'],
            'recall': report[label]['recall'],
            'f1': report[label]['f1-score'],
            'support': report[label]['support'],
        }
        for label in labels if label in report
    }

    return metrics


def main():
    """Run ensemble annotation on Xenium breast cancer data."""
    import json

    # Paths
    xenium_path = Path("/home/chrism/datasets/xenium_breast_tumor")
    ground_truth_path = xenium_path / "Cell_Barcode_Type_Matrices.xlsx"
    output_dir = Path("benchmark_results/ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Ensemble CellTypist Annotation")
    logger.info("=" * 60)

    # Load and normalize data
    adata = load_and_normalize_xenium(xenium_path)

    # Create ensemble annotator
    annotator = EnsembleCellTypistAnnotator(
        primary_model="Immune_All_High.pkl",
        stromal_model="Fetal_Human_AdrenalGlands.pkl",
        leiden_resolution=0.5,
    )

    # Run annotation
    result_df = annotator.annotate(
        adata,
        majority_voting=True,
        stromal_confidence_threshold=0.3,
    )

    # Save predictions
    result_path = output_dir / "ensemble_predictions.csv"
    result_df.to_csv(result_path, index=False)
    logger.info(f"Predictions saved to: {result_path}")

    # Evaluate against ground truth
    logger.info("\nEvaluating against ground truth...")
    metrics = evaluate_ensemble(result_df, ground_truth_path)

    if metrics:
        logger.info("\n" + "=" * 60)
        logger.info("ENSEMBLE RESULTS")
        logger.info("=" * 60)
        logger.info(f"Matched cells: {metrics['n_matched']:,}")
        logger.info(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"Precision (macro): {metrics['precision_macro']:.4f}")
        logger.info(f"Recall (macro): {metrics['recall_macro']:.4f}")

        logger.info("\nPer-Class Performance:")
        for cls, cls_metrics in metrics['per_class'].items():
            logger.info(f"  {cls:12}: P={cls_metrics['precision']:.3f} R={cls_metrics['recall']:.3f} F1={cls_metrics['f1']:.3f}")

        # Save metrics
        metrics_path = output_dir / "ensemble_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"\nMetrics saved to: {metrics_path}")

        # Compare to single model baseline
        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON TO SINGLE MODEL")
        logger.info("=" * 60)
        logger.info("Single model (Immune_All_High.pkl): F1 = 0.5003")
        logger.info(f"Ensemble approach:                  F1 = {metrics['f1_macro']:.4f}")
        improvement = (metrics['f1_macro'] - 0.5003) / 0.5003 * 100
        logger.info(f"Improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
