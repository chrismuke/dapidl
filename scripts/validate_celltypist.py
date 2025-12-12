"""Validate CellTypist predictions against ground truth annotations."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import warnings

warnings.filterwarnings("ignore")

# Ground truth cell type to broad category mapping
GROUND_TRUTH_MAPPING = {
    # Epithelial/Tumor cells
    "DCIS_1": "Epithelial",
    "DCIS_2": "Epithelial",
    "Invasive_Tumor": "Epithelial",
    "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_ACTA2+": "Epithelial",
    "Myoepi_KRT15+": "Epithelial",
    # Immune cells
    "B_Cells": "Immune",
    "CD4+_T_Cells": "Immune",
    "CD8+_T_Cells": "Immune",
    "Macrophages_1": "Immune",
    "Macrophages_2": "Immune",
    "IRF7+_DCs": "Immune",
    "LAMP3+_DCs": "Immune",
    "Mast_Cells": "Immune",
    # Stromal cells
    "Stromal": "Stromal",
    "Endothelial": "Stromal",
    "Perivascular-Like": "Stromal",
    # Hybrid/Unlabeled - exclude from analysis
    "Stromal_&_T_Cell_Hybrid": "Hybrid",
    "T_Cell_&_Tumor_Hybrid": "Hybrid",
    "Unlabeled": "Unlabeled",
}


def load_ground_truth(excel_path: str, sheet_name: str = "Xenium R1 Fig1-5 (supervised)") -> pd.DataFrame:
    """Load ground truth annotations from Excel file."""
    print(f"Loading ground truth from: {excel_path}")
    print(f"Sheet: {sheet_name}")

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    print(f"Loaded {len(df)} cells")
    print(f"Columns: {list(df.columns)}")

    # Rename columns for consistency
    df = df.rename(columns={"Barcode": "cell_id", "Cluster": "ground_truth_type"})

    # Map to broad categories
    df["ground_truth_broad"] = df["ground_truth_type"].map(GROUND_TRUTH_MAPPING)

    print("\nGround truth distribution:")
    print(df["ground_truth_type"].value_counts())

    print("\nBroad category distribution:")
    print(df["ground_truth_broad"].value_counts())

    return df


def run_celltypist_annotation(xenium_path: str, model_name: str = "Cells_Adult_Breast.pkl"):
    """Run CellTypist annotation on Xenium data."""
    import sys
    sys.path.insert(0, "/home/chrism/git/dapidl/src")

    from dapidl.data.xenium import XeniumDataReader
    from dapidl.data.annotation import CellTypeAnnotator

    print(f"\nRunning CellTypist with model: {model_name}")

    reader = XeniumDataReader(xenium_path)
    annotator = CellTypeAnnotator(model_names=[model_name])

    annotations = annotator.annotate_from_reader(reader)

    print(f"Annotated {len(annotations)} cells")
    print(f"Columns: {annotations.columns}")

    return annotations


def compare_predictions(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Compare CellTypist predictions to ground truth."""
    # Merge on cell_id
    merged = ground_truth.merge(
        predictions.to_pandas() if hasattr(predictions, 'to_pandas') else predictions,
        on="cell_id",
        how="inner"
    )

    print(f"\nMerged {len(merged)} cells")

    # Exclude hybrid and unlabeled cells
    valid_mask = ~merged["ground_truth_broad"].isin(["Hybrid", "Unlabeled"])
    merged_valid = merged[valid_mask].copy()
    print(f"Valid cells (excluding Hybrid/Unlabeled): {len(merged_valid)}")

    # Get prediction column name
    pred_col = "broad_category" if "broad_category" in merged_valid.columns else "broad_category_1"

    y_true = merged_valid["ground_truth_broad"]
    y_pred = merged_valid[pred_col]

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"CELLTYPIST VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
    labels = ["Epithelial", "Immune", "Stromal"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for label in labels:
        mask = y_true == label
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == label).mean()
            print(f"  {label}: {class_acc:.4f} ({mask.sum()} samples)")

    # Analyze misclassifications
    print("\nMisclassification Analysis:")
    for true_label in labels:
        for pred_label in labels:
            if true_label != pred_label:
                mask = (y_true == true_label) & (y_pred == pred_label)
                if mask.sum() > 0:
                    pct = mask.sum() / (y_true == true_label).sum() * 100
                    print(f"  {true_label} → {pred_label}: {mask.sum()} cells ({pct:.1f}%)")

    # Fine-grained analysis: which ground truth types are misclassified?
    print("\nFine-grained misclassification (ground truth type → predicted broad):")
    misclassified = merged_valid[y_true != y_pred]
    if len(misclassified) > 0:
        cross_tab = pd.crosstab(
            misclassified["ground_truth_type"],
            misclassified[pred_col],
            margins=True
        )
        print(cross_tab.head(25))

    return {
        "accuracy": accuracy,
        "n_samples": len(merged_valid),
        "confusion_matrix": cm,
    }


def main():
    # Paths
    excel_path = "/home/chrism/datasets/xenium_breast_tumor/Cell_Barcode_Type_Matrices.xlsx"
    xenium_path = "/home/chrism/datasets/xenium_breast_tumor"

    # Load ground truth
    ground_truth = load_ground_truth(excel_path)

    # Run CellTypist annotation
    predictions = run_celltypist_annotation(xenium_path, "Cells_Adult_Breast.pkl")

    # Compare
    results = compare_predictions(ground_truth, predictions)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"CellTypist accuracy on ground truth: {results['accuracy']:.2%}")


if __name__ == "__main__":
    main()
