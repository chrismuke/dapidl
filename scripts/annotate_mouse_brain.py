#!/usr/bin/env python3
"""Annotate Xenium mouse brain dataset with CellTypist and create ground truth.

Uses multiple CellTypist mouse brain models to create consensus annotations:
- Mouse_Whole_Brain.pkl: General mouse brain reference
- Mouse_Isocortex_Hippocampus.pkl: Specific to cortex and hippocampus

The consensus annotations serve as "pseudo ground truth" for benchmarking.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import celltypist
import h5py
import numpy as np
import polars as pl
from loguru import logger
from scipy.sparse import csc_matrix

# CellTypist models for mouse brain
MOUSE_BRAIN_MODELS = [
    "Mouse_Whole_Brain.pkl",
    "Mouse_Isocortex_Hippocampus.pkl",
]


def load_xenium_expression(xenium_path: Path) -> ad.AnnData:
    """Load expression matrix from Xenium output."""
    h5_path = xenium_path / "cell_feature_matrix.h5"

    if not h5_path.exists():
        raise FileNotFoundError(f"Expression matrix not found: {h5_path}")

    logger.info(f"Loading expression from {h5_path}")

    with h5py.File(h5_path, "r") as f:
        data = f["matrix/data"][:]
        indices = f["matrix/indices"][:]
        indptr = f["matrix/indptr"][:]
        shape = f["matrix/shape"][:]
        barcodes = [b.decode() for b in f["matrix/barcodes"][:]]
        genes = [g.decode() for g in f["matrix/features/name"][:]]

    # Create sparse matrix (cells x genes)
    X = csc_matrix((data, indices, indptr), shape=shape).T

    adata = ad.AnnData(X=X)
    adata.obs_names = barcodes
    adata.var_names = genes

    # Basic QC
    adata.obs["n_counts"] = np.array(adata.X.sum(axis=1)).flatten()
    adata.obs["n_genes"] = np.array((adata.X > 0).sum(axis=1)).flatten()

    logger.info(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
    logger.info(f"Total transcripts: {adata.obs['n_counts'].sum():,.0f}")

    # Normalize for CellTypist (log1p normalized to 10000 counts per cell)
    import scanpy as sc
    logger.info("Normalizing expression data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


def annotate_with_celltypist(
    adata: ad.AnnData,
    models: list[str] | None = None,
) -> pl.DataFrame:
    """Annotate cells with CellTypist models and create consensus."""
    models = models or MOUSE_BRAIN_MODELS

    # Download models if needed
    logger.info(f"Using CellTypist models: {models}")
    celltypist.models.download_models(model=models)

    all_predictions = []

    for model_name in models:
        logger.info(f"Running {model_name}...")
        model = celltypist.models.Model.load(model=model_name)

        predictions = celltypist.annotate(
            adata,
            model=model,
            majority_voting=True,
        )

        # Get predictions
        pred_labels = predictions.predicted_labels["majority_voting"].values
        pred_conf = predictions.probability_matrix.max(axis=1).values

        all_predictions.append({
            "model": model_name,
            "labels": pred_labels,
            "confidence": pred_conf,
        })

        # Log distribution
        unique, counts = np.unique(pred_labels, return_counts=True)
        top_5 = sorted(zip(unique, counts), key=lambda x: -x[1])[:5]
        logger.info(f"  Top 5 types: {dict(top_5)}")

    # Create consensus from multiple models
    n_cells = adata.n_obs
    cell_ids = adata.obs_names.tolist()

    annotations_data = []
    for i in range(n_cells):
        # Get predictions from all models
        model_predictions = [p["labels"][i] for p in all_predictions]
        model_confidences = [p["confidence"][i] for p in all_predictions]

        # Simple consensus: majority vote
        from collections import Counter
        vote_counts = Counter(model_predictions)
        consensus_label, consensus_count = vote_counts.most_common(1)[0]

        # Confidence = agreement ratio * average confidence
        agreement = consensus_count / len(model_predictions)
        avg_conf = np.mean([c for p, c in zip(model_predictions, model_confidences) if p == consensus_label])
        final_conf = agreement * avg_conf

        annotations_data.append({
            "cell_id": cell_ids[i],
            "predicted_type": consensus_label,
            "confidence": float(final_conf),
            "agreement": float(agreement),
            # Store individual model predictions
            **{f"model_{j}": p["labels"][i] for j, p in enumerate(all_predictions)},
        })

    df = pl.DataFrame(annotations_data)

    # Add broad category mapping for mouse brain
    df = df.with_columns(
        pl.col("predicted_type").map_elements(
            map_mouse_brain_to_broad,
            return_dtype=pl.Utf8,
        ).alias("broad_category")
    )

    return df


def map_mouse_brain_to_broad(cell_type: str) -> str:
    """Map mouse brain cell types to broad categories."""
    ct_lower = cell_type.lower()

    # Neurons
    if any(x in ct_lower for x in ["neuron", "excitatory", "inhibitory", "pyramidal",
                                     "interneuron", "granule", "purkinje", "dopamin"]):
        return "Neuron"

    # Glia - Astrocytes
    if any(x in ct_lower for x in ["astrocyte", "astro"]):
        return "Astrocyte"

    # Glia - Oligodendrocytes
    if any(x in ct_lower for x in ["oligodendrocyte", "oligo", "opc", "myelin"]):
        return "Oligodendrocyte"

    # Glia - Microglia
    if any(x in ct_lower for x in ["microglia", "macrophage"]):
        return "Microglia"

    # Endothelial
    if any(x in ct_lower for x in ["endothelial", "vascular", "pericyte"]):
        return "Vascular"

    # Ependymal
    if any(x in ct_lower for x in ["ependymal", "choroid"]):
        return "Ependymal"

    # Immune
    if any(x in ct_lower for x in ["immune", "t cell", "b cell", "lymphocyte"]):
        return "Immune"

    return "Other"


def save_ground_truth(
    df: pl.DataFrame,
    output_path: Path,
    min_confidence: float = 0.5,
    min_agreement: float = 0.5,
) -> None:
    """Save high-confidence annotations as ground truth."""
    # Filter by confidence and agreement
    high_conf_df = df.filter(
        (pl.col("confidence") >= min_confidence) &
        (pl.col("agreement") >= min_agreement)
    )

    logger.info(f"High-confidence cells: {high_conf_df.height}/{df.height} "
                f"({100*high_conf_df.height/df.height:.1f}%)")

    # Save full annotations
    df.write_parquet(output_path / "celltypist_consensus_annotations.parquet")

    # Save ground truth Excel format (like breast datasets)
    gt_df = high_conf_df.select([
        pl.col("cell_id").alias("cell_id"),
        pl.col("predicted_type").alias("cell_type"),
    ])

    # Convert to pandas for Excel export
    gt_pd = gt_df.to_pandas()
    excel_path = output_path / "celltypes_ground_truth_mouse_brain.xlsx"
    gt_pd.to_excel(excel_path, index=False)
    logger.info(f"Saved ground truth Excel: {excel_path}")

    # Summary statistics
    logger.info("\n=== Ground Truth Summary ===")
    logger.info(f"Total cells: {df.height}")
    logger.info(f"High-confidence cells: {high_conf_df.height}")

    broad_dist = dict(
        high_conf_df.group_by("broad_category")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .iter_rows()
    )
    logger.info(f"Broad category distribution: {broad_dist}")

    fine_dist = dict(
        high_conf_df.group_by("predicted_type")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(15)
        .iter_rows()
    )
    logger.info(f"Top 15 cell types: {fine_dist}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate Xenium mouse brain with CellTypist"
    )
    parser.add_argument(
        "xenium_path",
        type=Path,
        help="Path to Xenium output directory",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output directory (default: xenium_path)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for ground truth (default: 0.5)",
    )
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=0.5,
        help="Minimum model agreement for ground truth (default: 0.5)",
    )

    args = parser.parse_args()

    xenium_path = args.xenium_path
    output_path = args.output or xenium_path
    output_path.mkdir(parents=True, exist_ok=True)

    # Load expression data
    adata = load_xenium_expression(xenium_path)

    # Annotate with CellTypist
    annotations_df = annotate_with_celltypist(adata)

    # Save ground truth
    save_ground_truth(
        annotations_df,
        output_path,
        min_confidence=args.min_confidence,
        min_agreement=args.min_agreement,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
