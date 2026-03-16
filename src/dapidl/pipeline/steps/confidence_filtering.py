"""Annotation Confidence Filtering Pipeline Step.

Step 3.5: Filter low-confidence annotations before LMDB creation.

This step wraps the GT-free annotation confidence module to:
1. Compute per-cell confidence using marker enrichment, spatial coherence,
   cross-method consensus, and proportion plausibility
2. Filter low-confidence cells by relabeling them to "Unknown"
3. Output filtered annotations for downstream LMDB creation

Runs between Annotation (step 3) and LMDB Creation (step 4).

Queue: default (CPU — moderate compute for KNN spatial coherence)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from dapidl.pipeline.base import PipelineStep, StepArtifacts, resolve_artifact_path


@dataclass
class ConfidenceFilteringConfig:
    """Configuration for annotation confidence filtering."""

    # Enable/disable
    enabled: bool = True

    # Tissue type for proportion plausibility check
    tissue_type: str = "breast"

    # Confidence thresholds
    min_confidence: float = 0.4
    high_confidence_threshold: float = 0.7
    low_confidence_threshold: float = 0.3

    # Marker enrichment
    use_panglao_markers: bool = True
    min_markers_per_type: int = 3

    # Spatial coherence
    spatial_k: int = 15

    # Per-type marker score floor (drop entire types below this)
    min_marker_score: float | None = None

    # Label for filtered cells
    unknown_label: str = "Unknown"


class ConfidenceFilteringStep(PipelineStep):
    """Step 3.5: Filter annotations by confidence score.

    Uses GT-free annotation confidence estimation to identify and relabel
    low-confidence cell type predictions before they enter training data.

    Signals used:
    - Marker enrichment (PanglaoDB z-scores)
    - Spatial coherence (KNN fraction same-type neighbors)
    - Cross-method consensus (if multiple annotation methods available)
    - Proportion plausibility (tissue-specific expected fractions)

    Queue: default (CPU)
    """

    name = "confidence_filtering"
    description = "Filter low-confidence annotations before training"

    def __init__(self, config: ConfidenceFilteringConfig | None = None):
        self.config = config or ConfidenceFilteringConfig()
        self._task = None

    def get_parameter_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean", "default": True},
                "tissue_type": {"type": "string", "default": "breast"},
                "min_confidence": {"type": "number", "default": 0.4},
                "spatial_k": {"type": "integer", "default": 15},
                "use_panglao_markers": {"type": "boolean", "default": True},
            },
        }

    def validate_inputs(self, artifacts: StepArtifacts) -> bool:
        outputs = artifacts.outputs
        return "annotations_parquet" in outputs and "data_path" in outputs

    def execute(self, artifacts: StepArtifacts, **params: Any) -> StepArtifacts:
        """Execute confidence filtering on annotations.

        Args:
            artifacts: Input artifacts from annotation step, containing:
                - annotations_parquet: Path to annotations with predictions
                - data_path: Path to raw data (for loading expression matrix)
                - annotation_methods: List of methods used (for consensus)

        Returns:
            Output artifacts with filtered annotations and confidence stats.
        """
        cfg = self.config
        inputs = artifacts.outputs

        if not cfg.enabled:
            logger.info("Confidence filtering DISABLED — passing through")
            return StepArtifacts(
                inputs=inputs,
                outputs={**inputs, "confidence_filtering_skipped": True},
            )

        # Load annotations
        annotations_key = (
            "cl_annotations_parquet"
            if "cl_annotations_parquet" in inputs
            else "annotations_parquet"
        )
        annotations_path = resolve_artifact_path(inputs[annotations_key], annotations_key)
        if annotations_path is None:
            raise ValueError(f"{annotations_key} artifact is required")

        annotations_df = pl.read_parquet(annotations_path)
        logger.info(f"Loaded {len(annotations_df)} annotations for confidence filtering")

        # Check if hierarchical confidence_level column exists (from ensemble annotation)
        has_hierarchical = "confidence_level" in annotations_df.columns

        # Determine prediction column
        if "cl_name" in annotations_df.columns:
            pred_col = "cl_name"
        elif "predicted_type" in annotations_df.columns:
            pred_col = "predicted_type"
        else:
            pred_col = "broad_category"
        logger.info(f"Using prediction column: {pred_col}")

        predictions = annotations_df[pred_col].to_numpy().astype(str)

        # Get spatial coordinates
        spatial_coords = self._extract_spatial_coords(annotations_df, inputs)

        # Try to load AnnData for marker enrichment
        adata = self._try_load_adata(inputs)

        # Subset AnnData to match annotations (raw data may have more cells
        # than survived consensus filtering during annotation)
        if adata is not None and adata.n_obs != len(predictions):
            annotation_cell_ids = set(annotations_df["cell_id"].cast(pl.Utf8).to_list())
            mask = [obs_name in annotation_cell_ids for obs_name in adata.obs_names]
            adata = adata[mask].copy()
            logger.info(
                f"Subset AnnData from {sum(1 for _ in mask)} to {adata.n_obs} cells "
                f"to match {len(predictions)} annotations"
            )

        # Build predictions dict for consensus (if multiple methods)
        predictions_dict = self._build_predictions_dict(predictions, pred_col, annotations_df, inputs)

        # Compute confidence
        from dapidl.validation.annotation_confidence import (
            AnnotationConfidenceConfig,
            compute_annotation_confidence,
            filter_predictions,
        )

        ac_config = AnnotationConfidenceConfig(
            tissue_type=cfg.tissue_type,
            use_panglao_markers=cfg.use_panglao_markers,
            min_markers_per_type=cfg.min_markers_per_type,
            spatial_k=cfg.spatial_k,
            high_confidence_threshold=cfg.high_confidence_threshold,
            low_confidence_threshold=cfg.low_confidence_threshold,
        )

        if adata is not None:
            confidence_result = compute_annotation_confidence(
                adata=adata,
                predictions=predictions_dict,
                spatial_coords=spatial_coords,
                config=ac_config,
            )
        else:
            # Without expression data, only spatial coherence + consensus work
            logger.warning(
                "No AnnData available — marker enrichment will be skipped. "
                "Only spatial coherence and consensus will be used."
            )
            confidence_result = compute_annotation_confidence(
                adata=self._create_dummy_adata(len(predictions)),
                predictions=predictions_dict,
                spatial_coords=spatial_coords,
                config=AnnotationConfidenceConfig(
                    use_panglao_markers=False,  # Can't do marker enrichment without expression
                    spatial_k=cfg.spatial_k,
                    tissue_type=cfg.tissue_type,
                    high_confidence_threshold=cfg.high_confidence_threshold,
                    low_confidence_threshold=cfg.low_confidence_threshold,
                ),
            )

        # Log confidence report
        logger.info(f"\n{confidence_result.summary()}")

        if has_hierarchical:
            # Hierarchical mode: use signals to DOWNGRADE confidence_level
            filtered_df = self._apply_hierarchical_filtering(
                annotations_df, confidence_result, pred_col, cfg
            )
        else:
            # Legacy binary mode: keep/remove based on threshold
            filter_result = filter_predictions(
                predictions=predictions,
                confidence_result=confidence_result,
                min_confidence=cfg.min_confidence,
                min_marker_score=cfg.min_marker_score,
                unknown_label=cfg.unknown_label,
            )
            logger.info(f"\n{filter_result.summary()}")

            filtered_df = annotations_df.with_columns(
                pl.Series(name=pred_col, values=filter_result.predictions),
                pl.Series(name="cell_confidence", values=confidence_result.cell_confidence),
            )

        # Write filtered annotations
        filtered_path = Path(str(annotations_path).replace(".parquet", "_filtered.parquet"))
        filtered_df.write_parquet(filtered_path)
        logger.info(f"Wrote filtered annotations to {filtered_path}")

        # Build stats
        confidence_stats = {
            "overall_score": float(confidence_result.overall_score),
            "overall_marker_score": float(confidence_result.overall_marker_score),
            "overall_spatial_coherence": float(confidence_result.overall_spatial_coherence),
            "proportion_plausible": confidence_result.proportion_plausible,
            "min_confidence": cfg.min_confidence,
            "tissue_type": cfg.tissue_type,
        }
        if not np.isnan(confidence_result.overall_consensus_score):
            confidence_stats["overall_consensus_score"] = float(confidence_result.overall_consensus_score)

        if has_hierarchical:
            # Add per-level retention stats
            level_counts = dict(
                filtered_df.group_by("confidence_level")
                .agg(pl.len().alias("count"))
                .iter_rows()
            )
            total = filtered_df.height
            confidence_stats["retention_by_level"] = {
                "coarse": sum(v for k, v in level_counts.items() if k >= 0) / max(total, 1),
                "medium": sum(v for k, v in level_counts.items() if k >= 1) / max(total, 1),
                "fine": level_counts.get(2, 0) / max(total, 1),
            }
            confidence_stats["confidence_distribution"] = {
                "coarse_only": level_counts.get(0, 0),
                "coarse_medium": level_counts.get(1, 0),
                "all_levels": level_counts.get(2, 0),
            }
            confidence_stats["n_kept"] = total
            confidence_stats["n_filtered"] = 0  # All kept, just at different levels
            confidence_stats["retention_rate"] = 1.0
        else:
            confidence_stats["n_kept"] = filter_result.n_kept
            confidence_stats["n_filtered"] = filter_result.n_filtered
            confidence_stats["retention_rate"] = filter_result.n_kept / (filter_result.n_kept + filter_result.n_filtered)
            confidence_stats["per_type_kept"] = filter_result.per_type_kept
            confidence_stats["per_type_filtered"] = filter_result.per_type_filtered

        # Update the annotations parquet key to point to filtered version
        updated_outputs = {**inputs}
        updated_outputs[annotations_key] = str(filtered_path)
        updated_outputs["confidence_filtering_skipped"] = False
        updated_outputs["confidence_stats"] = confidence_stats
        updated_outputs["cell_confidence"] = confidence_result.cell_confidence

        return StepArtifacts(inputs=inputs, outputs=updated_outputs)

    def _apply_hierarchical_filtering(
        self,
        annotations_df: pl.DataFrame,
        confidence_result: Any,
        pred_col: str,
        cfg: "ConfidenceFilteringConfig",
    ) -> pl.DataFrame:
        """Use marker/spatial signals to adjust confidence_level downward.

        Instead of binary keep/remove, downgrades the hierarchical confidence_level
        based on quality signals. Low marker score → downgrade type's cells by 1 level.
        Low spatial coherence → downgrade individual cells by 1 level.
        """
        cell_confidence = confidence_result.cell_confidence
        per_type = confidence_result.per_type  # dict[str, CellTypeConfidence]

        # Start with existing confidence_level from ensemble annotation
        confidence_levels = annotations_df["confidence_level"].to_numpy().copy()

        # Signal 1: Per-type marker score — downgrade all cells of low-marker types
        if per_type:
            predictions = annotations_df[pred_col].to_numpy()
            for cell_type, type_conf in per_type.items():
                if type_conf.marker_score < cfg.low_confidence_threshold:
                    # Low marker enrichment for this type → downgrade by 1 level
                    type_mask = predictions == cell_type
                    confidence_levels[type_mask] = np.maximum(
                        confidence_levels[type_mask] - 1, -1
                    )
                    logger.info(
                        f"Downgraded '{cell_type}' (marker_score={type_conf.marker_score:.3f} "
                        f"< {cfg.low_confidence_threshold}) by 1 level"
                    )

        # Signal 2: Per-cell spatial coherence — downgrade spatially incoherent cells
        if cell_confidence is not None:
            # cell_confidence incorporates spatial coherence; use it as proxy
            low_spatial_mask = cell_confidence < cfg.low_confidence_threshold
            n_downgraded = int(low_spatial_mask.sum())
            if n_downgraded > 0:
                confidence_levels[low_spatial_mask] = np.maximum(
                    confidence_levels[low_spatial_mask] - 1, -1
                )
                logger.info(
                    f"Downgraded {n_downgraded} cells with low cell confidence "
                    f"(< {cfg.low_confidence_threshold})"
                )

        # Cells that drop below 0 get relabeled as Unknown
        excluded_mask = confidence_levels < 0
        n_excluded = excluded_mask.sum()
        if n_excluded > 0:
            logger.info(
                f"Excluded {n_excluded} cells after signal-based downgrade "
                f"(confidence_level dropped below 0)"
            )

        # Update the DataFrame with adjusted confidence levels and cell confidence
        result_df = annotations_df.with_columns(
            pl.Series(name="confidence_level", values=confidence_levels),
            pl.Series(name="cell_confidence", values=cell_confidence),
        )

        # Relabel excluded cells
        if n_excluded > 0:
            excluded_indices = np.where(excluded_mask)[0].tolist()
            # Build mask as polars expression
            result_df = result_df.with_row_index("_idx").with_columns(
                pl.when(pl.col("_idx").is_in(excluded_indices))
                .then(pl.lit(cfg.unknown_label))
                .otherwise(pl.col(pred_col))
                .alias(pred_col)
            ).drop("_idx")

        # Log level distribution after adjustment
        level_counts = dict(
            result_df.group_by("confidence_level")
            .agg(pl.len().alias("count"))
            .iter_rows()
        )
        logger.info(
            f"After signal-based adjustment: "
            f"fine(2)={level_counts.get(2, 0)}, "
            f"medium(1)={level_counts.get(1, 0)}, "
            f"coarse(0)={level_counts.get(0, 0)}, "
            f"excluded(<0)={level_counts.get(-1, 0)}"
        )

        return result_df

    def _extract_spatial_coords(
        self, df: pl.DataFrame, inputs: dict
    ) -> np.ndarray | None:
        """Extract spatial coordinates from annotations or raw data."""
        # Check annotation columns first
        if "x_centroid" in df.columns and "y_centroid" in df.columns:
            return np.column_stack([
                df["x_centroid"].to_numpy(),
                df["y_centroid"].to_numpy(),
            ])
        if "x" in df.columns and "y" in df.columns:
            return np.column_stack([df["x"].to_numpy(), df["y"].to_numpy()])
        if "center_x" in df.columns and "center_y" in df.columns:
            return np.column_stack([
                df["center_x"].to_numpy(),
                df["center_y"].to_numpy(),
            ])

        # Try loading from raw data cells.parquet
        data_path = resolve_artifact_path(inputs.get("data_path", ""), "data_path")
        if data_path is not None:
            cells_path = data_path / "cells.parquet"
            if cells_path.exists():
                cells_df = pl.read_parquet(cells_path, columns=["cell_id", "x_centroid", "y_centroid"])
                # Cast cell_id to match types (annotations use str, cells.parquet uses int)
                cells_df = cells_df.with_columns(pl.col("cell_id").cast(pl.Utf8))
                # Join on cell_id
                if "cell_id" in df.columns:
                    merged = df.with_columns(
                        pl.col("cell_id").cast(pl.Utf8)
                    ).join(cells_df, on="cell_id", how="left")
                    if "x_centroid" in merged.columns:
                        coords = np.column_stack([
                            merged["x_centroid"].to_numpy(),
                            merged["y_centroid"].to_numpy(),
                        ])
                        valid = ~np.isnan(coords).any(axis=1)
                        if valid.sum() > 0:
                            logger.info(f"Loaded spatial coords for {valid.sum()}/{len(df)} cells")
                            return coords

        logger.warning("No spatial coordinates found — spatial coherence will use neutral defaults")
        return None

    def _try_load_adata(self, inputs: dict) -> "Any | None":
        """Try to load AnnData for marker enrichment computation."""
        try:
            import anndata as ad
            import scanpy as sc
        except ImportError:
            logger.info("anndata/scanpy not available — skipping marker enrichment")
            return None

        # Check for cached h5ad from annotation step
        data_path = resolve_artifact_path(inputs.get("data_path", ""), "data_path")
        if data_path is None:
            return None

        # Try annotation step's cached h5ad
        h5ad_candidates = [
            data_path / "pipeline_outputs" / "annotation" / "annotated.h5ad",
            data_path / "pipeline_outputs" / "ensemble_annotation" / "annotated.h5ad",
        ]
        for h5ad_path in h5ad_candidates:
            if h5ad_path.exists():
                logger.info(f"Loading AnnData from {h5ad_path}")
                return ad.read_h5ad(h5ad_path)

        # Try building from Xenium output
        try:
            from scipy.sparse import csr_matrix

            from dapidl.data.xenium import XeniumDataReader
            reader = XeniumDataReader(data_path)
            expr_matrix, gene_names, cell_ids = reader.load_expression_matrix()
            adata = ad.AnnData(X=csr_matrix(expr_matrix))
            adata.var_names = gene_names
            adata.obs_names = [str(c) for c in cell_ids]
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            logger.info(f"Built AnnData from Xenium data: {adata.shape}")
            return adata
        except Exception as e:
            logger.debug(f"Could not build AnnData from raw data: {e}")

        return None

    def _build_predictions_dict(
        self,
        primary_predictions: np.ndarray,
        pred_col: str,
        df: pl.DataFrame,
        inputs: dict,
    ) -> dict[str, np.ndarray]:
        """Build multi-method predictions dict for consensus scoring."""
        predictions_dict: dict[str, np.ndarray] = {"primary": primary_predictions}

        # Check for individual method columns in the annotations
        # Ensemble annotation stores per-method predictions as separate columns
        method_cols = [c for c in df.columns if c.startswith("pred_") and c != pred_col]
        for col in method_cols:
            method_name = col.replace("pred_", "")
            predictions_dict[method_name] = df[col].to_numpy().astype(str)

        if len(predictions_dict) > 1:
            logger.info(f"Found {len(predictions_dict)} prediction methods for consensus")
        else:
            # Check annotation_methods in inputs
            methods = inputs.get("annotation_methods", [])
            if methods:
                logger.info(
                    f"Annotation used {len(methods)} methods but per-method columns not found — "
                    "consensus scoring will use single-method mode"
                )

        return predictions_dict

    def _create_dummy_adata(self, n_cells: int) -> "Any":
        """Create minimal AnnData when no expression data is available."""
        import anndata as ad
        from scipy.sparse import csr_matrix

        return ad.AnnData(
            X=csr_matrix((n_cells, 1)),
            var={"gene": ["dummy"]},
        )

    def get_queue(self) -> str:
        return "default"
