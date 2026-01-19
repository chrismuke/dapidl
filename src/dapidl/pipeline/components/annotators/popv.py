"""PopV-based cell type annotation component.

This module provides cell type annotation using the full popV ensemble prediction
pipeline with 8+ methods and ontology-based voting.

PopV combines predictions from multiple annotation methods:
- Random Forest (RF)
- Support Vector Machine (SVM)
- XGBoost
- CellTypist
- OnClass (Cell Ontology-aware)
- scVI + kNN
- scANVI (semi-supervised VAE)
- BBKNN + kNN
- Scanorama + kNN
- Harmony + kNN

Requires: pip install popv

References:
- Ergen et al. (2024) Nature Genetics
- https://github.com/YosefLab/popV
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


# =============================================================================
# PopV Configuration
# =============================================================================


@dataclass
class PopVConfig:
    """Configuration for popV annotation.

    Attributes:
        reference: Reference dataset to use ('tabula_sapiens' or 'custom')
        reference_organ: Organ-specific reference ('auto', 'Mammary', 'Lung', etc.)
        custom_reference_path: Path to custom reference h5ad file
        methods: List of methods to use (None = all available)
        mode: Annotation mode ('fast', 'inference', 'retrain')
        min_consensus_score: Minimum consensus for high-confidence (0-8)
        use_ontology_voting: Use Cell Ontology hierarchy for voting
        n_top_genes: Number of highly variable genes to use
        batch_key: Key for batch correction (optional)
        include_method_predictions: Include per-method prediction columns
        include_ontology_parent: Include common ancestor column
        save_full_results: Save full popV h5ad output
    """

    # Reference selection
    reference: str = "tabula_sapiens"
    reference_organ: str = "auto"
    custom_reference_path: str | None = None

    # Method selection
    methods: list[str] | None = None  # None = all available

    # Mode selection
    mode: str = "fast"  # fast (5min), inference (30min), retrain (1hr)

    # Consensus settings
    min_consensus_score: int = 6  # 6/8 = ~90% accuracy
    use_ontology_voting: bool = True

    # Processing settings
    n_top_genes: int = 2000
    batch_key: str | None = None

    # Output settings
    include_method_predictions: bool = True
    include_ontology_parent: bool = True
    save_full_results: bool = False

    # Fine-grained output
    fine_grained: bool = True  # Use popV predictions vs broad categories


# Available Tabula Sapiens organs
TABULA_SAPIENS_ORGANS = [
    "Bladder",
    "Blood",
    "Bone_Marrow",
    "Eye",
    "Fat",
    "Heart",
    "Kidney",
    "Large_Intestine",
    "Liver",
    "Lung",
    "Mammary",
    "Muscle",
    "Pancreas",
    "Prostate",
    "Salivary_Gland",
    "Skin",
    "Small_Intestine",
    "Spleen",
    "Thymus",
    "Tongue",
    "Trachea",
    "Uterus",
    "Vasculature",
]

# Marker genes for tissue auto-detection
TISSUE_MARKERS = {
    "Mammary": ["EPCAM", "KRT18", "KRT19", "ESR1", "PGR", "MKI67"],
    "Lung": ["SFTPC", "SCGB1A1", "AGER", "HOPX", "NKX2-1"],
    "Liver": ["ALB", "APOB", "CYP3A4", "HNF4A", "SERPINA1"],
    "Brain": ["RBFOX3", "MAP2", "GFAP", "MBP", "SYP"],
    "Kidney": ["AQP1", "SLC12A1", "UMOD", "NPHS1", "PODXL"],
    "Heart": ["TNNT2", "MYH7", "ACTN2", "RYR2", "TTN"],
    "Blood": ["PTPRC", "CD3E", "CD19", "CD14", "NCAM1"],
    "Skin": ["KRT14", "KRT1", "KRT10", "COL1A1", "DCN"],
}


# =============================================================================
# PopV Annotator
# =============================================================================


@register_annotator
class PopVAnnotator:
    """Cell type annotation using full popV ensemble prediction.

    Uses the complete popV pipeline with 8+ annotation methods and
    ontology-based voting for robust cross-tissue annotation.

    Features:
    - 8+ annotation methods (RF, SVM, XGBoost, CellTypist, OnClass, scVI, scANVI, etc.)
    - Cell Ontology-based voting for consensus
    - Pretrained Tabula Sapiens references for 20+ organs
    - Consensus scores for uncertainty quantification
    - ClearML and WandB integration support
    """

    name = "popv"

    def __init__(self, config: PopVConfig | AnnotationConfig | None = None):
        """Initialize the popV annotator.

        Args:
            config: PopV or annotation configuration
        """
        if isinstance(config, AnnotationConfig):
            # Convert AnnotationConfig to PopVConfig
            self.config = self._convert_config(config)
        else:
            self.config = config or PopVConfig()

        if not is_popv_available():
            raise ImportError(
                "popV is not installed. Install with: pip install popv\n"
                "For full functionality: pip install 'dapidl[popv]'"
            )

    def _convert_config(self, config: AnnotationConfig) -> PopVConfig:
        """Convert AnnotationConfig to PopVConfig."""
        return PopVConfig(
            fine_grained=config.fine_grained,
            min_consensus_score=6 if config.confidence_threshold >= 0.75 else 5,
        )

    def annotate(
        self,
        config: PopVConfig | AnnotationConfig | None = None,
        adata: Any | None = None,
        expression_path: Path | None = None,
        cells_df: pl.DataFrame | None = None,
    ) -> AnnotationResult:
        """Annotate cells with type labels using full popV pipeline.

        Args:
            config: Override config for this call
            adata: AnnData object with expression data
            expression_path: Path to expression matrix (h5, h5ad)
            cells_df: Cell metadata DataFrame (not used by popV)

        Returns:
            AnnotationResult with:
            - annotations_df: Polars DataFrame with predictions
            - class_mapping: Cell type to index mapping
            - index_to_class: Index to cell type mapping
            - stats: Annotation statistics including consensus scores
        """
        import time

        start_time = time.time()

        # Get config
        if config is not None:
            if isinstance(config, AnnotationConfig):
                cfg = self._convert_config(config)
            else:
                cfg = config
        else:
            cfg = self.config

        # Load adata if needed
        if adata is None:
            if expression_path is None:
                raise ValueError(
                    "PopVAnnotator requires either adata or expression_path"
                )
            adata = self._load_expression(expression_path)

        logger.info(f"Running popV annotation on {adata.n_obs} cells...")

        # Detect tissue type if auto
        organ = cfg.reference_organ
        if organ == "auto":
            organ = self._detect_tissue(adata)
            logger.info(f"Auto-detected tissue type: {organ}")

        # Run popV annotation
        annotated_adata = self._run_popv_full(adata, organ, cfg)

        # Extract results to DAPIDL format
        result = self._extract_results(annotated_adata, cfg)

        elapsed = time.time() - start_time
        logger.info(
            f"PopV annotation complete: {result.n_annotated} cells, "
            f"{result.stats.get('n_high_confidence', 0)} high-confidence, "
            f"elapsed {elapsed:.1f}s"
        )

        return result

    def _detect_tissue(self, adata: Any) -> str:
        """Auto-detect tissue type from marker gene expression.

        Args:
            adata: AnnData object with gene expression

        Returns:
            Detected organ name (default: 'Mammary')
        """
        import numpy as np

        genes_in_data = set(adata.var_names)
        best_organ = "Mammary"
        best_score = 0

        for organ, markers in TISSUE_MARKERS.items():
            # Count how many markers are present and expressed
            present_markers = [m for m in markers if m in genes_in_data]
            if not present_markers:
                continue

            # Calculate mean expression of markers
            try:
                marker_idx = [
                    list(adata.var_names).index(m) for m in present_markers
                ]
                if hasattr(adata.X, "toarray"):
                    expr = adata.X[:, marker_idx].toarray()
                else:
                    expr = adata.X[:, marker_idx]

                # Score = fraction of cells expressing markers * marker coverage
                expressing_frac = np.mean(expr > 0)
                coverage = len(present_markers) / len(markers)
                score = expressing_frac * coverage

                if score > best_score:
                    best_score = score
                    best_organ = organ
            except Exception:
                continue

        logger.debug(f"Tissue detection scores: best={best_organ} (score={best_score:.3f})")
        return best_organ

    def _run_popv_full(
        self,
        query_adata: Any,
        organ: str,
        config: PopVConfig,
    ) -> Any:
        """Run the full popV annotation pipeline.

        Args:
            query_adata: Query AnnData object
            organ: Organ name for reference selection
            config: PopV configuration

        Returns:
            Annotated AnnData with popV results in .obs
        """
        import popv

        # Map organ to Tabula Sapiens name
        ts_organ = self._map_organ_name(organ)
        logger.info(f"Using Tabula Sapiens reference: {ts_organ}")

        # Check if using custom reference
        if config.reference == "custom" and config.custom_reference_path:
            import anndata as ad

            ref_adata = ad.read_h5ad(config.custom_reference_path)
            logger.info(f"Using custom reference: {config.custom_reference_path}")
        else:
            # Use popV's pretrained pipeline
            ref_adata = None

        # Determine mode and methods
        mode = config.mode
        methods = config.methods

        logger.info(f"PopV mode: {mode}, methods: {methods or 'all'}")

        # Run popV preprocessing
        logger.info("Running popV preprocessing...")
        try:
            # Process query with reference
            pq = popv.preprocessing.Process_Query(
                query_adata,
                ref_adata,
                save_path="./popv_cache",
                pretrained_scvi_path=None,  # Will use HuggingFace models
                prediction_mode=mode,
                n_samples_per_label=300 if mode == "retrain" else None,
                unknown_celltype_label="unknown",
            )

            # Get processed adata
            processed_adata = pq.adata

            logger.info("Running popV annotation with all methods...")

            # Run annotation
            popv.annotation.annotate_data(
                processed_adata,
                methods=methods,
                save_path=None,
            )

            # Run ontology voting if enabled
            if config.use_ontology_voting:
                logger.info("Running ontology-based voting...")
                try:
                    popv.annotation.ontology_vote_onclass(processed_adata)
                except Exception as e:
                    logger.warning(f"Ontology voting failed: {e}, using majority vote")

            # Run majority vote as fallback/supplement
            popv.annotation.compute_consensus(processed_adata)

            logger.info("PopV annotation complete")
            return processed_adata

        except Exception as e:
            logger.error(f"PopV full pipeline failed: {e}")
            logger.info("Falling back to HubModel approach...")
            return self._run_popv_hubmodels(query_adata, organ, config)

    def _run_popv_hubmodels(
        self,
        query_adata: Any,
        organ: str,
        config: PopVConfig,
    ) -> Any:
        """Fallback: Run popV using HuggingFace HubModels.

        Args:
            query_adata: Query AnnData object
            organ: Organ name
            config: PopV configuration

        Returns:
            Annotated AnnData with predictions
        """
        from popv.hub import HubModel

        # Map organ to HubModel name
        hub_name = f"popV/tabula_sapiens_{organ}"

        logger.info(f"Loading HubModel: {hub_name}")

        try:
            hub_model = HubModel.pull_from_huggingface_hub(hub_name)

            # Check gene format
            sample_genes = list(query_adata.var_names[:5])
            genes_are_ensembl = all(str(g).startswith("ENSG") for g in sample_genes)
            gene_symbols_param = None if genes_are_ensembl else "feature_name"

            # Run prediction (API changed in popv 0.6.0: predict -> annotate_data)
            if hasattr(hub_model, 'annotate_data'):
                # PopV 0.6.0+ API
                result = hub_model.annotate_data(
                    query_adata,
                    gene_symbols=gene_symbols_param,
                )
            elif hasattr(hub_model, 'predict'):
                # Older PopV API
                result = hub_model.predict(
                    query_adata,
                    gene_symbols=gene_symbols_param,
                )
            else:
                raise RuntimeError("HubModel has no prediction method (tried annotate_data and predict)")

            logger.info("HubModel prediction complete")
            return result

        except Exception as e:
            logger.error(f"HubModel prediction failed: {e}")
            raise RuntimeError(f"PopV annotation failed: {e}") from e

    def _map_organ_name(self, organ: str) -> str:
        """Map common organ names to Tabula Sapiens names."""
        mapping = {
            "breast": "Mammary",
            "mammary": "Mammary",
            "lung": "Lung",
            "liver": "Liver",
            "brain": "Brain",
            "kidney": "Kidney",
            "heart": "Heart",
            "blood": "Blood",
            "pbmc": "Blood",
            "skin": "Skin",
            "colon": "Large_Intestine",
            "intestine": "Small_Intestine",
        }
        normalized = organ.lower().replace(" ", "_")
        return mapping.get(normalized, organ)

    def _extract_results(
        self,
        adata: Any,
        config: PopVConfig,
    ) -> AnnotationResult:
        """Extract popV results to DAPIDL AnnotationResult format.

        Args:
            adata: Annotated AnnData from popV
            config: PopV configuration

        Returns:
            AnnotationResult with standardized format
        """
        import numpy as np

        # Get cell IDs
        if "cell_id" in adata.obs.columns:
            cell_ids = adata.obs["cell_id"].values
        else:
            cell_ids = adata.obs_names.values

        # Find prediction columns
        pred_col = None
        score_col = None

        # Priority order for prediction column
        pred_candidates = [
            "popv_prediction",
            "popv_majority_vote_prediction",
            "popv_Mammary_prediction",
            "predicted_labels",
        ]
        for col in pred_candidates:
            if col in adata.obs.columns:
                pred_col = col
                break

        # Find score column
        score_candidates = [
            "popv_majority_vote_score",
            "popv_consensus_score",
            "popv_score",
        ]
        for col in score_candidates:
            if col in adata.obs.columns:
                score_col = col
                break

        if pred_col is None:
            # Look for any popv prediction column
            for col in adata.obs.columns:
                if "popv" in col.lower() and "prediction" in col.lower():
                    pred_col = col
                    break

        if pred_col is None:
            raise ValueError(
                f"No popV prediction column found. "
                f"Available columns: {list(adata.obs.columns)}"
            )

        logger.info(f"Using prediction column: {pred_col}")

        # Build annotations DataFrame
        annotations_data = []

        for i, cid in enumerate(cell_ids):
            predicted_type = adata.obs[pred_col].iloc[i]

            # Get consensus score
            if score_col and score_col in adata.obs.columns:
                raw_score = adata.obs[score_col].iloc[i]
                # Normalize to 0-1 if needed (popV scores are 0-8)
                if isinstance(raw_score, (int, float)):
                    consensus_score = int(raw_score)
                    confidence = raw_score / 8.0 if raw_score <= 8 else raw_score
                else:
                    consensus_score = 0
                    confidence = 0.5
            else:
                consensus_score = 8
                confidence = 1.0

            # Map to broad category
            broad_cat = map_to_broad_category(str(predicted_type))

            row = {
                "cell_id": str(cid),
                "predicted_type": str(predicted_type),
                "broad_category": broad_cat,
                "confidence": float(confidence),
                "consensus_score": consensus_score,
            }

            # Add ontology parent if available
            if config.include_ontology_parent:
                if "popv_parent" in adata.obs.columns:
                    row["ontology_parent"] = str(adata.obs["popv_parent"].iloc[i])

            # Add per-method predictions if requested
            if config.include_method_predictions:
                for col in adata.obs.columns:
                    if col.startswith("popv_") and col.endswith("_prediction"):
                        if col != pred_col:
                            row[col] = str(adata.obs[col].iloc[i])

            annotations_data.append(row)

        annotations_df = pl.DataFrame(annotations_data)

        # Filter out Unknown category
        annotations_df = annotations_df.filter(
            pl.col("broad_category") != "Unknown"
        )

        # Filter by consensus score if threshold set
        min_score = config.min_consensus_score
        n_before = annotations_df.height

        high_confidence_df = annotations_df.filter(
            pl.col("consensus_score") >= min_score
        )
        n_high_confidence = high_confidence_df.height

        # Build class mapping
        if config.fine_grained:
            unique_types = sorted(annotations_df["predicted_type"].unique().to_list())
        else:
            unique_types = sorted(annotations_df["broad_category"].unique().to_list())

        class_mapping = {name: i for i, name in enumerate(unique_types)}
        index_to_class = {i: name for i, name in enumerate(unique_types)}

        # Calculate statistics
        consensus_scores = annotations_df["consensus_score"].to_numpy()
        class_dist = (
            annotations_df.group_by("broad_category")
            .count()
            .to_pandas()
            .set_index("broad_category")["count"]
            .to_dict()
        )

        stats = {
            "n_annotated": annotations_df.height,
            "n_high_confidence": n_high_confidence,
            "high_confidence_rate": n_high_confidence / n_before if n_before > 0 else 0,
            "consensus_score_mean": float(np.mean(consensus_scores)),
            "consensus_score_std": float(np.std(consensus_scores)),
            "class_distribution": class_dist,
            "prediction_column": pred_col,
            "min_consensus_threshold": min_score,
            "popv_mode": config.mode,
        }

        # Add consensus score distribution
        score_dist = {}
        for score in range(9):
            count = int((consensus_scores == score).sum())
            if count > 0:
                score_dist[score] = count
        stats["consensus_distribution"] = score_dist

        return AnnotationResult(
            annotations_df=annotations_df,
            class_mapping=class_mapping,
            index_to_class=index_to_class,
            stats=stats,
        )

    def _load_expression(self, path: Path) -> Any:
        """Load expression data from file.

        Args:
            path: Path to expression file (h5ad, zarr, h5)

        Returns:
            AnnData object
        """
        import anndata as ad

        path = Path(path)

        if path.suffix in [".h5", ".h5ad"]:
            return ad.read_h5ad(path)
        elif path.suffix == ".zarr" or path.is_dir():
            return ad.read_zarr(path)
        else:
            raise ValueError(f"Unsupported expression format: {path}")

    def get_available_references(self) -> list[str]:
        """Get list of available Tabula Sapiens organs."""
        return TABULA_SAPIENS_ORGANS.copy()


# =============================================================================
# Standalone annotation function
# =============================================================================


def annotate_with_popv(
    expression_path: str | Path,
    output_path: str | Path | None = None,
    organ: str = "auto",
    mode: str = "fast",
    min_consensus: int = 6,
    use_wandb: bool = False,
    use_clearml: bool = False,
    project_name: str = "dapidl-popv",
) -> AnnotationResult:
    """Annotate cells using popV (standalone function).

    This is a convenience function for running popV annotation outside
    of the full DAPIDL pipeline.

    Args:
        expression_path: Path to expression matrix (h5ad, zarr)
        output_path: Optional path to save annotations parquet
        organ: Organ name or 'auto' for detection
        mode: PopV mode ('fast', 'inference', 'retrain')
        min_consensus: Minimum consensus score threshold
        use_wandb: Enable Weights & Biases logging
        use_clearml: Enable ClearML logging
        project_name: Project name for logging backends

    Returns:
        AnnotationResult with popV predictions

    Example:
        >>> result = annotate_with_popv(
        ...     "expression.h5ad",
        ...     output_path="annotations.parquet",
        ...     organ="Mammary",
        ...     mode="fast"
        ... )
        >>> print(f"Annotated {result.n_annotated} cells")
    """
    # Initialize logging if requested
    if use_wandb:
        try:
            import wandb

            wandb.init(project=project_name, job_type="popv_annotation")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
            use_wandb = False

    if use_clearml:
        try:
            from clearml import Task

            task = Task.init(project_name=project_name, task_name="popv_annotation")
        except ImportError:
            logger.warning("clearml not installed, skipping ClearML logging")
            use_clearml = False

    # Create config
    config = PopVConfig(
        reference_organ=organ,
        mode=mode,
        min_consensus_score=min_consensus,
    )

    # Run annotation
    annotator = PopVAnnotator(config)
    result = annotator.annotate(expression_path=Path(expression_path))

    # Log metrics
    if use_wandb:
        import wandb

        wandb.log(
            {
                "popv/n_annotated": result.stats["n_annotated"],
                "popv/n_high_confidence": result.stats["n_high_confidence"],
                "popv/consensus_mean": result.stats["consensus_score_mean"],
                "popv/high_confidence_rate": result.stats["high_confidence_rate"],
            }
        )

    if use_clearml:
        task.get_logger().report_scalar(
            "popv", "n_annotated", result.stats["n_annotated"], 0
        )
        task.get_logger().report_scalar(
            "popv", "high_confidence_rate", result.stats["high_confidence_rate"], 0
        )

    # Save output if requested
    if output_path:
        output_path = Path(output_path)
        result.annotations_df.write_parquet(output_path)
        logger.info(f"Saved annotations to: {output_path}")

        if use_wandb:
            import wandb

            wandb.log_artifact(str(output_path), name="popv_annotations", type="dataset")

    return result
