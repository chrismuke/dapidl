"""scANVI-based cell type annotation component.

scANVI (single-cell ANnotation using Variational Inference) is a semi-supervised
deep learning model that transfers cell type labels from reference to query data.

Reference: Xu et al. Molecular Systems Biology 2021
https://github.com/scverse/scvi-tools
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scanpy as sc
from loguru import logger

from dapidl.pipeline.base import AnnotationConfig, AnnotationResult
from dapidl.pipeline.components.annotators.mapping import (
    get_class_names,
    map_to_broad_category,
)
from dapidl.pipeline.registry import register_annotator


# Mapping scANVI predictions to broad categories
SCANVI_TO_BROAD = {
    # Common labels from reference datasets
    "Epithelial cell": "Epithelial",
    "Luminal epithelial cell": "Epithelial",
    "Basal cell": "Epithelial",
    "Myoepithelial cell": "Epithelial",
    "Keratinocyte": "Epithelial",
    "T cell": "Immune",
    "CD4+ T cell": "Immune",
    "CD8+ T cell": "Immune",
    "B cell": "Immune",
    "Plasma cell": "Immune",
    "Macrophage": "Immune",
    "Monocyte": "Immune",
    "Dendritic cell": "Immune",
    "NK cell": "Immune",
    "Mast cell": "Immune",
    "Neutrophil": "Immune",
    "Fibroblast": "Stromal",
    "Smooth muscle cell": "Stromal",
    "Pericyte": "Stromal",
    "Adipocyte": "Stromal",
    "Endothelial cell": "Endothelial",
    "Vascular endothelial cell": "Endothelial",
    "Lymphatic endothelial cell": "Endothelial",
}


def map_scanvi_to_broad(label: str) -> str:
    """Map scANVI prediction to broad category."""
    if label in SCANVI_TO_BROAD:
        return SCANVI_TO_BROAD[label]

    # Try fuzzy matching
    label_lower = label.lower()
    if any(x in label_lower for x in ["epithelial", "luminal", "basal", "myoepithelial", "keratinocyte"]):
        return "Epithelial"
    if any(x in label_lower for x in ["t cell", "b cell", "macrophage", "monocyte", "dendritic", "nk", "mast", "neutrophil", "immune", "plasma"]):
        return "Immune"
    if any(x in label_lower for x in ["fibroblast", "smooth muscle", "pericyte", "adipocyte", "stromal"]):
        return "Stromal"
    if any(x in label_lower for x in ["endothelial", "vascular", "lymphatic"]):
        return "Endothelial"

    return map_to_broad_category(label)


@register_annotator
class ScANVIAnnotator:
    """Cell type annotation using scANVI semi-supervised transfer learning.

    scANVI learns a joint embedding of reference and query data, then uses
    the reference labels to classify query cells.
    """

    name = "scanvi"

    def __init__(
        self,
        config: AnnotationConfig | None = None,
        reference_path: Path | str | None = None,
        label_key: str = "cell_type",
        batch_key: str | None = "batch",
        n_latent: int = 30,
        n_layers: int = 2,
        max_epochs_scvi: int = 100,
        max_epochs_scanvi: int = 50,
    ):
        """Initialize the scANVI annotator.

        Args:
            config: Annotation configuration
            reference_path: Path to reference AnnData (h5ad)
            label_key: Column in reference.obs with cell type labels
            batch_key: Column for batch correction (None to disable)
            n_latent: Latent dimension size
            n_layers: Number of hidden layers
            max_epochs_scvi: Max epochs for unsupervised pretraining
            max_epochs_scanvi: Max epochs for semi-supervised training
        """
        self.config = config or AnnotationConfig()
        self.reference_path = Path(reference_path) if reference_path else None
        self.label_key = label_key
        self.batch_key = batch_key
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.max_epochs_scvi = max_epochs_scvi
        self.max_epochs_scanvi = max_epochs_scanvi

    def annotate(
        self,
        config: AnnotationConfig | None = None,
        adata: Any | None = None,
        expression_path: Path | None = None,
        cells_df: pl.DataFrame | None = None,
        reference_adata: Any | None = None,
    ) -> AnnotationResult:
        """Annotate cells using scANVI transfer learning.

        Args:
            config: Override config for this call
            adata: Query AnnData object (required)
            expression_path: Path to query expression (alternative to adata)
            cells_df: Cell metadata DataFrame (not used)
            reference_adata: Pre-loaded reference AnnData (alternative to reference_path)

        Returns:
            Annotation results with cell types and confidence
        """
        import scvi

        cfg = config or self.config

        # Load query data
        if adata is None:
            if expression_path is None:
                raise ValueError("ScANVIAnnotator requires either adata or expression_path")
            adata = self._load_expression(expression_path)

        # Load reference data
        if reference_adata is None:
            if self.reference_path is None:
                raise ValueError("ScANVIAnnotator requires reference data")
            reference_adata = self._load_expression(self.reference_path)

        logger.info(f"Query: {adata.n_obs} cells, Reference: {reference_adata.n_obs} cells")

        # Ensure we have the label key
        if self.label_key not in reference_adata.obs:
            raise ValueError(f"Reference must have '{self.label_key}' in obs")

        # Find common genes
        common_genes = list(set(adata.var_names) & set(reference_adata.var_names))
        logger.info(f"Using {len(common_genes)} common genes")

        if len(common_genes) < 100:
            raise ValueError(f"Too few common genes ({len(common_genes)}), need at least 100")

        # Subset to common genes
        adata_sub = adata[:, common_genes].copy()
        ref_sub = reference_adata[:, common_genes].copy()

        # Add batch info
        adata_sub.obs["_batch"] = "query"
        ref_sub.obs["_batch"] = "reference"

        # Mark unlabeled
        adata_sub.obs[self.label_key] = "Unknown"

        # Concatenate
        combined = sc.concat([ref_sub, adata_sub], join="outer")
        combined.obs_names_make_unique()

        logger.info(f"Combined dataset: {combined.n_obs} cells")

        # Normalize if needed
        if combined.X.max() > 100:  # Likely counts
            sc.pp.normalize_total(combined, target_sum=1e4)
            sc.pp.log1p(combined)

        # Select highly variable genes
        sc.pp.highly_variable_genes(
            combined,
            n_top_genes=2000,
            batch_key="_batch",
            flavor="seurat_v3",
            subset=True,
        )

        logger.info(f"Using {combined.n_vars} highly variable genes")

        # Setup scVI model
        scvi.model.SCVI.setup_anndata(
            combined,
            batch_key="_batch",
            labels_key=self.label_key,
        )

        # Train scVI (unsupervised)
        logger.info("Training scVI (unsupervised)...")
        scvi_model = scvi.model.SCVI(
            combined,
            n_latent=self.n_latent,
            n_layers=self.n_layers,
        )
        scvi_model.train(
            max_epochs=self.max_epochs_scvi,
            early_stopping=True,
            early_stopping_patience=10,
        )

        # Train scANVI (semi-supervised)
        logger.info("Training scANVI (semi-supervised)...")
        scanvi_model = scvi.model.SCANVI.from_scvi_model(
            scvi_model,
            unlabeled_category="Unknown",
        )
        scanvi_model.train(
            max_epochs=self.max_epochs_scanvi,
            early_stopping=True,
            early_stopping_patience=10,
        )

        # Predict on query cells
        logger.info("Predicting cell types...")
        predictions = scanvi_model.predict()

        # Get query predictions only
        query_mask = combined.obs["_batch"] == "query"
        query_predictions = predictions[query_mask]
        query_cell_ids = combined.obs_names[query_mask]

        # Get confidence (probability of predicted class)
        proba = scanvi_model.predict(soft=True)
        query_proba = proba[query_mask]
        confidence = query_proba.max(axis=1)

        # Build annotations DataFrame
        annotations_data = []
        for cid, pred, conf in zip(query_cell_ids, query_predictions, confidence):
            broad_cat = map_scanvi_to_broad(pred)
            annotations_data.append({
                "cell_id": str(cid),
                "predicted_type": pred,
                "broad_category": broad_cat,
                "confidence": float(conf),
            })

        annotations_df = pl.DataFrame(annotations_data)

        # Filter out Unknown if configured
        include_unknown = getattr(cfg, "include_unknown", True)
        if not include_unknown:
            annotations_df = annotations_df.filter(pl.col("broad_category") != "Unknown")

        # Build class mapping
        class_names = get_class_names(cfg.fine_grained)
        class_mapping = {name: i for i, name in enumerate(class_names)}
        index_to_class = {i: name for i, name in enumerate(class_names)}

        # Calculate statistics
        n_annotated = annotations_df.height
        class_dist = (
            annotations_df.group_by("broad_category")
            .count()
            .to_pandas()
            .set_index("broad_category")["count"]
            .to_dict()
        )

        logger.info(f"scANVI annotation complete: {n_annotated} cells annotated")
        logger.info(f"  Class distribution: {class_dist}")

        return AnnotationResult(
            annotations_df=annotations_df,
            class_mapping=class_mapping,
            index_to_class=index_to_class,
            stats={
                "n_annotated": n_annotated,
                "class_distribution": class_dist,
                "n_common_genes": len(common_genes),
                "n_hvg_used": combined.n_vars,
            },
        )

    def _load_expression(self, path: Path) -> Any:
        """Load expression data from file."""
        import anndata as ad

        path = Path(path)

        if path.suffix in [".h5", ".h5ad"]:
            return ad.read_h5ad(path)
        elif path.suffix == ".zarr" or path.is_dir():
            return ad.read_zarr(path)
        else:
            raise ValueError(f"Unsupported expression format: {path}")


def download_reference(
    reference_name: str = "tabula_sapiens_breast",
    cache_dir: Path | None = None,
) -> Path:
    """Download a pre-defined reference dataset.

    Args:
        reference_name: Name of reference to download
        cache_dir: Where to cache downloaded files

    Returns:
        Path to downloaded reference h5ad file
    """
    import pooch

    REFERENCES = {
        "tabula_sapiens_breast": {
            "url": "https://figshare.com/ndownloader/files/37612753",
            "known_hash": None,  # Update with actual hash
            "filename": "tabula_sapiens_breast.h5ad",
        },
        # Add more references as needed
    }

    if reference_name not in REFERENCES:
        raise ValueError(f"Unknown reference: {reference_name}. Available: {list(REFERENCES.keys())}")

    ref_info = REFERENCES[reference_name]

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "dapidl" / "references"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    path = cache_dir / ref_info["filename"]
    if path.exists():
        logger.info(f"Using cached reference: {path}")
        return path

    logger.info(f"Downloading reference: {reference_name}")
    pooch.retrieve(
        url=ref_info["url"],
        known_hash=ref_info["known_hash"],
        path=cache_dir,
        fname=ref_info["filename"],
    )

    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="scANVI cell type annotation")
    parser.add_argument("query_path", type=Path, help="Path to query expression data")
    parser.add_argument("--reference", "-r", type=Path, required=True, help="Path to reference h5ad")
    parser.add_argument("--output", "-o", type=Path, help="Output parquet path")
    parser.add_argument("--label-key", default="cell_type", help="Column with cell type labels")

    args = parser.parse_args()

    annotator = ScANVIAnnotator(reference_path=args.reference, label_key=args.label_key)

    import anndata as ad
    adata = ad.read_h5ad(args.query_path)

    result = annotator.annotate(adata=adata)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.annotations_df.write_parquet(args.output)
        logger.info(f"Saved annotations to {args.output}")

    print(result.annotations_df.head(10))
