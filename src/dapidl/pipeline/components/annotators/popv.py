"""PopV-based cell type annotation component.

This module provides cell type annotation using popV ensemble prediction
with multiple HuggingFace models from the Tabula Sapiens collection.

PopV combines predictions from multiple specialized models:
- Mammary: Breast-specific cell types
- Immune: All immune cell types
- Epithelium: Epithelial cells
- Stromal: Fibroblasts and stromal cells

Requires: pip install popv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from dapidl.pipeline.base import AnnotationConfig, AnnotationResult
from dapidl.pipeline.components.annotators.mapping import (
    get_class_names,
    map_to_broad_category,
)
from dapidl.pipeline.registry import register_annotator


def is_popv_available() -> bool:
    """Check if popV is installed and available."""
    try:
        import popv  # noqa: F401
        return True
    except ImportError:
        return False


@register_annotator
class PopVAnnotator:
    """Cell type annotation using popV ensemble prediction.

    Uses multiple HuggingFace models from the Tabula Sapiens collection
    to annotate cells with comprehensive cell type labels.
    """

    name = "popv"

    def __init__(self, config: AnnotationConfig | None = None):
        """Initialize the popV annotator.

        Args:
            config: Annotation configuration
        """
        self.config = config or AnnotationConfig()

        if not is_popv_available():
            raise ImportError(
                "popV is not installed. Install with: pip install popv"
            )

    def annotate(
        self,
        config: AnnotationConfig | None = None,
        adata: Any | None = None,
        expression_path: Path | None = None,
        cells_df: pl.DataFrame | None = None,
    ) -> AnnotationResult:
        """Annotate cells with type labels using popV.

        Args:
            config: Override config for this call
            adata: AnnData object with expression data (required)
            expression_path: Path to expression matrix (h5, h5ad)
            cells_df: Cell metadata DataFrame (not used)

        Returns:
            Annotation results with cell types and confidence
        """
        from popv.hub import HubModel

        cfg = config or self.config

        # Need either adata or expression_path
        if adata is None and expression_path is None:
            raise ValueError(
                "PopVAnnotator requires either adata or expression_path"
            )

        # Load adata if needed
        if adata is None and expression_path is not None:
            adata = self._load_expression(expression_path)

        logger.info("Running popV multi-model ensemble prediction...")

        # Available popV HubModels for human tissue annotation
        popv_models = [
            ("popV/tabula_sapiens_Mammary", "mammary"),
            ("popV/tabula_sapiens_Immune", "immune"),
            ("popV/tabula_sapiens_Epithelium", "epithelium"),
            ("popV/tabula_sapiens_Stromal", "stromal"),
        ]

        # Collect predictions from all models
        all_predictions: dict[str, dict[str, str]] = {}
        all_scores: dict[str, dict[str, float]] = {}
        cell_ids = adata.obs["cell_id"].values if "cell_id" in adata.obs else adata.obs_names

        # Initialize
        for cid in cell_ids:
            all_predictions[cid] = {}
            all_scores[cid] = {}

        # Check if genes are already Ensembl IDs
        sample_genes = list(adata.var_names[:5])
        genes_are_ensembl = all(str(g).startswith("ENSG") for g in sample_genes)
        gene_symbols_param = None if genes_are_ensembl else "feature_name"

        if gene_symbols_param:
            logger.info("Gene symbols detected - popV will map to Ensembl IDs")

        for repo_name, model_key in popv_models:
            try:
                logger.info(f"Loading popV model: {repo_name}")
                hub_model = HubModel.pull_from_huggingface_hub(repo_name)

                # Run prediction
                logger.info(f"Running prediction with {model_key}...")
                result = hub_model.predict(
                    adata,
                    gene_symbols=gene_symbols_param,
                )

                # Extract predictions (popV adds columns to adata.obs)
                pred_col = f"popv_{model_key}_prediction"
                score_col = f"popv_{model_key}_score"

                if pred_col in result.obs.columns:
                    for i, cid in enumerate(cell_ids):
                        all_predictions[cid][model_key] = result.obs[pred_col].iloc[i]
                        if score_col in result.obs.columns:
                            all_scores[cid][model_key] = result.obs[score_col].iloc[i]
                        else:
                            all_scores[cid][model_key] = 1.0

                logger.info(f"Completed {model_key} predictions")

            except Exception as e:
                logger.warning(f"Error with model {repo_name}: {e}")
                continue

        # Combine predictions using voting
        annotations_data = []
        for cid in cell_ids:
            preds = all_predictions.get(cid, {})
            scores = all_scores.get(cid, {})

            if not preds:
                continue

            # Use mammary prediction if available, else most confident
            if "mammary" in preds and preds["mammary"]:
                best_pred = preds["mammary"]
                best_score = scores.get("mammary", 1.0)
            else:
                # Find prediction with highest score
                best_pred = None
                best_score = 0.0
                for model, pred in preds.items():
                    if pred and scores.get(model, 0.0) > best_score:
                        best_pred = pred
                        best_score = scores[model]

            if best_pred:
                broad_cat = map_to_broad_category(best_pred)
                annotations_data.append({
                    "cell_id": cid,
                    "predicted_type": best_pred,
                    "broad_category": broad_cat,
                    "confidence": best_score,
                    **{f"popv_{m}": preds.get(m, "") for m in ["mammary", "immune", "epithelium", "stromal"]},
                })

        annotations_df = pl.DataFrame(annotations_data)

        # Filter out Unknown
        annotations_df = annotations_df.filter(
            pl.col("broad_category") != "Unknown"
        )

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

        return AnnotationResult(
            annotations_df=annotations_df,
            class_mapping=class_mapping,
            index_to_class=index_to_class,
            stats={
                "n_annotated": n_annotated,
                "class_distribution": class_dist,
                "models_used": [m[1] for m in popv_models],
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
