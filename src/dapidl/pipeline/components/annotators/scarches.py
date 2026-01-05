"""scArches-based cell type annotation component.

scArches enables architectural surgery on pretrained models to map query data
to existing reference atlases. It supports multiple underlying models like
scVI, scANVI, totalVI, etc.

Reference: Lotfollahi et al. Nature Biotechnology 2022
https://github.com/theislab/scarches
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

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


# Label to broad category mapping
SCARCHES_TO_BROAD = {
    "Epithelial": "Epithelial",
    "Epithelial cell": "Epithelial",
    "Luminal epithelial": "Epithelial",
    "Basal epithelial": "Epithelial",
    "Myoepithelial": "Epithelial",
    "T cell": "Immune",
    "B cell": "Immune",
    "Macrophage": "Immune",
    "Monocyte": "Immune",
    "Dendritic cell": "Immune",
    "NK cell": "Immune",
    "Mast cell": "Immune",
    "Plasma cell": "Immune",
    "Fibroblast": "Stromal",
    "Smooth muscle": "Stromal",
    "Pericyte": "Stromal",
    "Adipocyte": "Stromal",
    "Endothelial": "Endothelial",
    "Vascular endothelial": "Endothelial",
}


def map_to_broad(label: str) -> str:
    """Map label to broad category."""
    if label in SCARCHES_TO_BROAD:
        return SCARCHES_TO_BROAD[label]

    label_lower = label.lower()
    if any(x in label_lower for x in ["epithelial", "luminal", "basal", "myoepithelial"]):
        return "Epithelial"
    if any(x in label_lower for x in ["t cell", "b cell", "macrophage", "monocyte", "dendritic", "nk", "mast", "plasma", "immune"]):
        return "Immune"
    if any(x in label_lower for x in ["fibroblast", "smooth muscle", "pericyte", "adipocyte", "stromal"]):
        return "Stromal"
    if any(x in label_lower for x in ["endothelial", "vascular"]):
        return "Endothelial"

    return map_to_broad_category(label)


@register_annotator
class ScArchesAnnotator:
    """Cell type annotation using scArches reference mapping.

    scArches performs architectural surgery on pretrained models to enable
    mapping query data to reference atlases without full retraining.
    """

    name = "scarches"

    def __init__(
        self,
        config: AnnotationConfig | None = None,
        model_path: Path | str | None = None,
        reference_path: Path | str | None = None,
        label_key: str = "cell_type",
        batch_key: str = "batch",
        model_type: Literal["scvi", "scanvi", "totalvi"] = "scanvi",
        n_epochs_surgery: int = 50,
    ):
        """Initialize the scArches annotator.

        Args:
            config: Annotation configuration
            model_path: Path to pretrained model directory
            reference_path: Path to reference AnnData (for training new model)
            label_key: Column in reference.obs with cell type labels
            batch_key: Column for batch correction
            model_type: Type of underlying model
            n_epochs_surgery: Epochs for surgery training
        """
        self.config = config or AnnotationConfig()
        self.model_path = Path(model_path) if model_path else None
        self.reference_path = Path(reference_path) if reference_path else None
        self.label_key = label_key
        self.batch_key = batch_key
        self.model_type = model_type
        self.n_epochs_surgery = n_epochs_surgery

    def annotate(
        self,
        config: AnnotationConfig | None = None,
        adata: Any | None = None,
        expression_path: Path | None = None,
        cells_df: pl.DataFrame | None = None,
        reference_adata: Any | None = None,
    ) -> AnnotationResult:
        """Annotate cells using scArches reference mapping.

        Args:
            config: Override config for this call
            adata: Query AnnData object (required)
            expression_path: Path to query expression (alternative to adata)
            cells_df: Cell metadata DataFrame (not used)
            reference_adata: Pre-loaded reference AnnData

        Returns:
            Annotation results with cell types and confidence
        """
        import scvi
        import scarches as sca

        cfg = config or self.config

        # Load query data
        if adata is None:
            if expression_path is None:
                raise ValueError("ScArchesAnnotator requires either adata or expression_path")
            adata = self._load_expression(expression_path)

        logger.info(f"Query data: {adata.n_obs} cells x {adata.n_vars} genes")

        # Option 1: Use pretrained model
        if self.model_path and self.model_path.exists():
            return self._annotate_with_pretrained(adata, cfg)

        # Option 2: Train new model on reference
        if reference_adata is None:
            if self.reference_path is None:
                raise ValueError("ScArchesAnnotator requires either model_path or reference_path")
            reference_adata = self._load_expression(self.reference_path)

        logger.info(f"Reference data: {reference_adata.n_obs} cells x {reference_adata.n_vars} genes")

        # Find common genes
        common_genes = list(set(adata.var_names) & set(reference_adata.var_names))
        logger.info(f"Using {len(common_genes)} common genes")

        if len(common_genes) < 100:
            raise ValueError(f"Too few common genes ({len(common_genes)})")

        # Subset to common genes
        query = adata[:, common_genes].copy()
        ref = reference_adata[:, common_genes].copy()

        # Prepare for scArches
        query.obs[self.batch_key] = "query"
        query.obs[self.label_key] = "Unknown"

        if self.batch_key not in ref.obs:
            ref.obs[self.batch_key] = "reference"

        # Normalize if needed
        for data in [ref, query]:
            if data.X.max() > 100:
                sc.pp.normalize_total(data, target_sum=1e4)
                sc.pp.log1p(data)

        # Select HVGs
        sc.pp.highly_variable_genes(ref, n_top_genes=2000, batch_key=self.batch_key, subset=True)
        query = query[:, ref.var_names].copy()

        logger.info(f"Using {ref.n_vars} highly variable genes")

        # Setup and train reference model
        if self.model_type == "scanvi":
            scvi.model.SCVI.setup_anndata(ref, batch_key=self.batch_key)
            ref_model = scvi.model.SCVI(ref, n_layers=2, n_latent=30)
            ref_model.train(max_epochs=100, early_stopping=True)

            scvi.model.SCANVI.setup_anndata(ref, batch_key=self.batch_key, labels_key=self.label_key)
            ref_scanvi = scvi.model.SCANVI.from_scvi_model(ref_model, unlabeled_category="Unknown")
            ref_scanvi.train(max_epochs=50, early_stopping=True)

            # Surgery on query
            logger.info("Performing architectural surgery...")
            query_model = sca.models.SCANVI.load_query_data(
                query,
                ref_scanvi,
                freeze_dropout=True,
            )
            query_model.train(max_epochs=self.n_epochs_surgery, early_stopping=True)

            # Predict
            predictions = query_model.predict()
            proba = query_model.predict(soft=True)

        else:  # scvi
            scvi.model.SCVI.setup_anndata(ref, batch_key=self.batch_key)
            ref_model = scvi.model.SCVI(ref, n_layers=2, n_latent=30)
            ref_model.train(max_epochs=100, early_stopping=True)

            # Surgery on query
            logger.info("Performing architectural surgery...")
            query_model = sca.models.SCVI.load_query_data(
                query,
                ref_model,
                freeze_dropout=True,
            )
            query_model.train(max_epochs=self.n_epochs_surgery, early_stopping=True)

            # Use KNN for label transfer
            ref_latent = ref_model.get_latent_representation()
            query_latent = query_model.get_latent_representation()

            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=15, metric="euclidean")
            knn.fit(ref_latent, ref.obs[self.label_key].values)
            predictions = knn.predict(query_latent)
            proba = knn.predict_proba(query_latent)

        # Get confidence
        if hasattr(proba, "max"):
            confidence = proba.max(axis=1)
        else:
            confidence = np.ones(len(predictions)) * 0.8

        # Build annotations
        cell_ids = adata.obs_names.tolist()
        annotations_data = []

        for cid, pred, conf in zip(cell_ids, predictions, confidence):
            broad = map_to_broad(pred)
            annotations_data.append({
                "cell_id": str(cid),
                "predicted_type": pred,
                "broad_category": broad,
                "confidence": float(conf),
            })

        annotations_df = pl.DataFrame(annotations_data)

        include_unknown = getattr(cfg, "include_unknown", True)
        if not include_unknown:
            annotations_df = annotations_df.filter(pl.col("broad_category") != "Unknown")

        # Build class mapping
        class_names = get_class_names(cfg.fine_grained)
        class_mapping = {name: i for i, name in enumerate(class_names)}
        index_to_class = {i: name for i, name in enumerate(class_names)}

        # Stats
        n_annotated = annotations_df.height
        class_dist = (
            annotations_df.group_by("broad_category")
            .count()
            .to_pandas()
            .set_index("broad_category")["count"]
            .to_dict()
        )

        logger.info(f"scArches annotation complete: {n_annotated} cells annotated")
        logger.info(f"  Class distribution: {class_dist}")

        return AnnotationResult(
            annotations_df=annotations_df,
            class_mapping=class_mapping,
            index_to_class=index_to_class,
            stats={
                "n_annotated": n_annotated,
                "class_distribution": class_dist,
                "model_type": self.model_type,
            },
        )

    def _annotate_with_pretrained(self, adata: Any, cfg: AnnotationConfig) -> AnnotationResult:
        """Use pretrained scArches model for annotation."""
        import scvi
        import scarches as sca

        logger.info(f"Loading pretrained model from {self.model_path}")

        # Load the appropriate model type
        if self.model_type == "scanvi":
            model = sca.models.SCANVI.load_query_data(adata, self.model_path, freeze_dropout=True)
        else:
            model = sca.models.SCVI.load_query_data(adata, self.model_path, freeze_dropout=True)

        model.train(max_epochs=self.n_epochs_surgery, early_stopping=True)

        if self.model_type == "scanvi":
            predictions = model.predict()
            proba = model.predict(soft=True)
            confidence = proba.max(axis=1)
        else:
            # Need KNN for scVI
            latent = model.get_latent_representation()
            predictions = ["Unknown"] * len(latent)
            confidence = np.ones(len(latent)) * 0.5

        # Build annotations
        cell_ids = adata.obs_names.tolist()
        annotations_data = []

        for cid, pred, conf in zip(cell_ids, predictions, confidence):
            broad = map_to_broad(pred)
            annotations_data.append({
                "cell_id": str(cid),
                "predicted_type": pred,
                "broad_category": broad,
                "confidence": float(conf),
            })

        annotations_df = pl.DataFrame(annotations_data)

        include_unknown = getattr(cfg, "include_unknown", True)
        if not include_unknown:
            annotations_df = annotations_df.filter(pl.col("broad_category") != "Unknown")

        class_names = get_class_names(cfg.fine_grained)
        class_mapping = {name: i for i, name in enumerate(class_names)}
        index_to_class = {i: name for i, name in enumerate(class_names)}

        return AnnotationResult(
            annotations_df=annotations_df,
            class_mapping=class_mapping,
            index_to_class=index_to_class,
            stats={"n_annotated": annotations_df.height},
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="scArches cell type annotation")
    parser.add_argument("query_path", type=Path, help="Path to query expression data")
    parser.add_argument("--reference", "-r", type=Path, help="Path to reference h5ad")
    parser.add_argument("--model", "-m", type=Path, help="Path to pretrained model")
    parser.add_argument("--output", "-o", type=Path, help="Output parquet path")

    args = parser.parse_args()

    annotator = ScArchesAnnotator(
        reference_path=args.reference,
        model_path=args.model,
    )

    import anndata as ad
    adata = ad.read_h5ad(args.query_path)

    result = annotator.annotate(adata=adata)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.annotations_df.write_parquet(args.output)

    print(result.annotations_df.head(10))
