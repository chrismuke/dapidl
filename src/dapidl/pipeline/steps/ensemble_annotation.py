"""Ensemble Annotation Pipeline Step.

Step 2: Cell type annotation using ensemble of multiple methods.

This step:
1. Runs multiple CellTypist models
2. Optionally runs SingleR (R-based reference mapping)
3. Builds consensus across all methods
4. Creates derived annotated dataset with ClearML lineage
5. Uploads to S3 if configured

The ensemble approach improves annotation quality by:
- Reducing model-specific biases
- Providing confidence through agreement
- Leveraging different reference datasets
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from dapidl.pipeline.base import (
    AnnotationConfig,
    AnnotationResult,
    PipelineStep,
    StepArtifacts,
    get_pipeline_output_dir,
    resolve_artifact_path,
)


@dataclass
class MethodSpec:
    """Declarative specification for a single annotation method."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MethodSpec:
        return cls(name=d["name"], params=d.get("params", {}))


@dataclass
class EnsembleAnnotationConfig:
    """Configuration for ensemble annotation step."""

    methods: list[MethodSpec] = field(default_factory=list)

    # Consensus settings
    min_agreement: int = 2
    confidence_threshold: float = 0.5
    use_confidence_weighting: bool = True

    # Output settings
    fine_grained: bool = True
    create_derived_dataset: bool = True
    parent_dataset_id: str | None = None
    skip_if_exists: bool = True
    upload_to_s3: bool = True
    s3_bucket: str = "dapidl"
    s3_endpoint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "methods": [m.to_dict() for m in self.methods],
            "min_agreement": self.min_agreement,
            "confidence_threshold": self.confidence_threshold,
            "use_confidence_weighting": self.use_confidence_weighting,
            "fine_grained": self.fine_grained,
            "create_derived_dataset": self.create_derived_dataset,
            "parent_dataset_id": self.parent_dataset_id,
            "skip_if_exists": self.skip_if_exists,
            "upload_to_s3": self.upload_to_s3,
            "s3_bucket": self.s3_bucket,
            "s3_endpoint": self.s3_endpoint,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnsembleAnnotationConfig:
        methods_raw = d.get("methods", d.get("General/methods", "[]"))
        if isinstance(methods_raw, str):
            methods_raw = json.loads(methods_raw)
        if isinstance(methods_raw, list) and all(isinstance(m, dict) for m in methods_raw):
            methods = [MethodSpec.from_dict(m) for m in methods_raw]
        else:
            methods = []

        def _pb(val: Any, default: bool = True) -> bool:
            """Parse bool from string or bool (ClearML sends 'True'/'False' strings)."""
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ("true", "1", "yes")
            return bool(val) if val is not None else default

        return cls(
            methods=methods,
            min_agreement=int(d.get("min_agreement", d.get("General/min_agreement", 2))),
            confidence_threshold=float(d.get("confidence_threshold", d.get("General/confidence_threshold", 0.5))),
            use_confidence_weighting=_pb(d.get("use_confidence_weighting", d.get("General/use_confidence_weighting", True))),
            fine_grained=_pb(d.get("fine_grained", d.get("General/fine_grained", True))),
            create_derived_dataset=_pb(d.get("create_derived_dataset", True)),
            parent_dataset_id=d.get("parent_dataset_id"),
            skip_if_exists=_pb(d.get("skip_if_exists", True)),
            upload_to_s3=_pb(d.get("upload_to_s3", True)),
            s3_bucket=str(d.get("s3_bucket", "dapidl")),
            s3_endpoint=str(d.get("s3_endpoint", "")),
        )


class EnsembleAnnotationStep(PipelineStep):
    """Step 2: Cell Type Prediction with Ensemble Annotation.

    Runs multiple annotation methods and creates consensus:
    1. CellTypist (multiple models)
    2. SingleR (R-based reference mapping)
    3. scType (marker-based, optional)

    Creates a derived annotated dataset and registers it with ClearML.
    Uses parent_datasets for lineage to avoid re-uploading raw data.

    Queue: default (CPU-bound - CellTypist and R are CPU-only)
    """

    name = "ensemble_annotation"
    description = "Ensemble cell type annotation with multiple methods"

    def __init__(self, config: EnsembleAnnotationConfig | None = None):
        """Initialize ensemble annotation step.

        Args:
            config: Ensemble annotation configuration
        """
        self.config = config or EnsembleAnnotationConfig()
        self._task = None

    def get_parameter_schema(self) -> dict[str, Any]:
        """Return JSON schema for ClearML UI parameters."""
        return {
            "type": "object",
            "properties": {
                "celltypist_models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
                    "description": "CellTypist model names (comma-separated)",
                },
                "include_singler": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include SingleR annotation (requires R)",
                },
                "singler_reference": {
                    "type": "string",
                    "enum": ["blueprint", "hpca", "monaco"],
                    "default": "blueprint",
                    "description": "SingleR reference dataset",
                },
                "min_agreement": {
                    "type": "integer",
                    "default": 2,
                    "minimum": 1,
                    "description": "Minimum annotators that must agree",
                },
                "confidence_threshold": {
                    "type": "number",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum consensus confidence",
                },
                "fine_grained": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use detailed cell types (vs 3 broad categories)",
                },
                "skip_if_exists": {
                    "type": "boolean",
                    "default": True,
                    "description": "Skip if matching annotations exist in ClearML",
                },
            },
        }

    def validate_inputs(self, artifacts: StepArtifacts) -> bool:
        """Validate step inputs.

        Requires:
        - expression_path: Path to expression data (H5, H5AD, CSV)
        - data_path: Path to Xenium/MERSCOPE output directory
        """
        return "expression_path" in artifacts.outputs and "data_path" in artifacts.outputs

    def _validate_methods(self, methods: list[MethodSpec]) -> None:
        """Fail fast if any requested method is unavailable."""
        from dapidl.pipeline.registry import list_annotators

        available = list_annotators()
        for spec in methods:
            if spec.name not in available:
                raise ValueError(
                    f"Annotator '{spec.name}' is not registered. "
                    f"Available: {available}. "
                    f"Check that required dependencies are installed."
                )

    def _build_annotator_config(self, spec: MethodSpec) -> AnnotationConfig:
        """Translate MethodSpec params into annotator's expected config."""
        base = AnnotationConfig(method=spec.name)
        for key, value in spec.params.items():
            if key == "model":
                base.model_names = [value]
            elif key == "reference":
                base.singler_reference = value
            elif hasattr(base, key):
                setattr(base, key, value)
        return base

    @staticmethod
    def _make_source_label(spec: MethodSpec) -> str:
        """Create a human-readable source label for consensus tracking."""
        suffix = spec.params.get("model", spec.params.get("reference", ""))
        if suffix:
            return f"{spec.name}_{suffix}"
        return spec.name

    def _run_methods(
        self,
        methods: list[MethodSpec],
        adata: Any,
        data_path: Path,
    ) -> list[AnnotationResult]:
        """Run all declared annotation methods via the registry."""
        from dapidl.pipeline.registry import get_annotator

        results = []
        for spec in methods:
            annotator = get_annotator(spec.name, config=self._build_annotator_config(spec))
            result = annotator.annotate(
                adata=adata,
                expression_path=data_path,
            )
            result.stats["source"] = self._make_source_label(spec)
            results.append(result)
            logger.info(f"{spec.name}: {len(result.annotations_df)} cells annotated")
        return results

    def _get_config_hash(self, cfg: EnsembleAnnotationConfig, raw_dataset_id: str) -> str:
        """Generate deterministic hash of annotation configuration."""
        methods_normalized = sorted(
            [json.dumps(m.to_dict(), sort_keys=True) for m in cfg.methods]
        )
        config_str = (
            f"methods={'|'.join(methods_normalized)}|"
            f"min_agree={cfg.min_agreement}|"
            f"conf={cfg.confidence_threshold}|"
            f"fine={cfg.fine_grained}|"
            f"raw={raw_dataset_id}"
        )
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]

    def _check_existing_annotations(
        self, cfg: EnsembleAnnotationConfig, raw_dataset_id: str, platform: str
    ) -> str | None:
        """Check if matching annotation dataset already exists in ClearML.

        Returns dataset ID if found, None otherwise.
        """
        try:
            from clearml import Dataset
        except ImportError:
            logger.debug("ClearML not available, skipping cache check")
            return None

        config_hash = self._get_config_hash(cfg, raw_dataset_id)
        expected_name = f"annotated-{platform}-{config_hash}"

        try:
            datasets = Dataset.list_datasets(
                dataset_project="DAPIDL/annotated",
                partial_name=expected_name,
            )

            for ds in datasets:
                meta = ds.get("metadata", {})
                # Verify configuration matches
                if (
                    meta.get("config_hash") == config_hash
                    and meta.get("raw_dataset_id") == raw_dataset_id
                    and meta.get("celltypist_models") == sorted(cfg.celltypist_models)
                ):
                    logger.info(f"Found cached annotations: {ds['id']} ({ds['name']})")
                    return ds["id"]

        except Exception as e:
            logger.debug(f"Error checking existing annotations: {e}")

        return None

    def _get_cached_annotations(self, dataset_id: str) -> tuple[Path, dict] | None:
        """Get cached annotations from ClearML dataset.

        Returns (annotations_path, class_mapping) or None if failed.
        """
        try:
            from clearml import Dataset

            ds = Dataset.get(dataset_id=dataset_id)
            local_path = Path(ds.get_local_copy())

            annotations_path = local_path / "annotations.parquet"
            mapping_path = local_path / "class_mapping.json"

            if annotations_path.exists() and mapping_path.exists():
                with open(mapping_path) as f:
                    mapping_data = json.load(f)
                return annotations_path, mapping_data

        except Exception as e:
            logger.warning(f"Failed to get cached annotations: {e}")

        return None

    def execute(self, artifacts: StepArtifacts) -> StepArtifacts:
        """Execute ensemble annotation.

        Args:
            artifacts: Input artifacts from DataLoaderStep

        Returns:
            Output artifacts containing:
            - annotations_parquet: Path to consensus annotation DataFrame
            - class_mapping: Dict mapping class names to indices
            - annotated_dataset_id: ClearML Dataset ID (if registered)
            - annotation_stats: Dict with annotation statistics
        """
        cfg = self.config
        inputs = artifacts.outputs

        # Resolve artifact URLs to local paths
        data_path = resolve_artifact_path(inputs["data_path"], "data_path")
        expression_path = resolve_artifact_path(inputs.get("expression_path"), "expression_path")

        if data_path is None:
            raise ValueError("data_path artifact is required")
        if expression_path is None:
            raise ValueError("expression_path artifact is required")

        # Get platform
        platform = self._resolve_platform(inputs)

        # Get raw dataset ID for cache key
        raw_dataset_id = inputs.get("raw_dataset_id", cfg.parent_dataset_id or "local")

        # Check for existing LOCAL annotations first (skip logic)
        output_dir = get_pipeline_output_dir("ensemble_annotation", data_path)
        local_annotations = output_dir / "annotations.parquet"
        local_mapping = output_dir / "class_mapping.json"
        config_path = output_dir / "config.json"

        if cfg.skip_if_exists and local_annotations.exists() and local_mapping.exists():
            # Validate config matches (if config file exists)
            config_matches = True
            if config_path.exists():
                with open(config_path) as f:
                    saved_config = json.load(f)
                # Check key annotation parameters
                config_matches = (
                    sorted(saved_config.get("celltypist_models", []))
                    == sorted(cfg.celltypist_models)
                    and saved_config.get("include_singler") == cfg.include_singler
                    and saved_config.get("singler_reference") == cfg.singler_reference
                    and saved_config.get("fine_grained") == cfg.fine_grained
                )
                if not config_matches:
                    logger.info("Config mismatch - re-running annotation")

            if config_matches:
                logger.info(f"Skipping annotation - outputs already exist at {output_dir}")
                with open(local_mapping) as f:
                    mapping_data = json.load(f)
                return StepArtifacts(
                    inputs=inputs,
                    outputs={
                        **inputs,
                        "annotations_parquet": str(local_annotations),
                        "class_mapping": mapping_data.get("class_mapping", {}),
                        "index_to_class": mapping_data.get("index_to_class", {}),
                        "annotated_dataset_id": None,
                        "skipped": True,
                        "annotation_stats": {
                            "skipped": True,
                            "reason": "local_outputs_exist",
                        },
                    },
                )

        # Check for existing ClearML annotations (skip logic)
        if cfg.skip_if_exists:
            existing_id = self._check_existing_annotations(cfg, raw_dataset_id, platform)
            if existing_id:
                cached = self._get_cached_annotations(existing_id)
                if cached:
                    annotations_path, mapping_data = cached
                    logger.info(f"Using cached annotations from dataset: {existing_id}")

                    # Copy cached files to local output directory (reuse paths from above)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Copy or link cached files
                    import shutil

                    if not local_annotations.exists():
                        shutil.copy2(annotations_path, local_annotations)
                    if not local_mapping.exists():
                        with open(local_mapping, "w") as f:
                            json.dump(mapping_data, f, indent=2)

                    return StepArtifacts(
                        inputs=inputs,
                        outputs={
                            **inputs,
                            "annotations_parquet": str(local_annotations),
                            "class_mapping": mapping_data.get("class_mapping", {}),
                            "index_to_class": mapping_data.get("index_to_class", {}),
                            "annotated_dataset_id": existing_id,
                            "skipped": True,
                            "annotation_stats": {
                                "skipped": True,
                                "cached_dataset_id": existing_id,
                            },
                        },
                    )

        # Load expression data
        adata = self._load_expression(expression_path, data_path, platform)
        logger.info(f"Loaded expression data: {adata.n_obs} cells, {adata.n_vars} genes")

        # Run annotation methods
        all_predictions = []

        # 1. Run CellTypist models
        for model_name in cfg.celltypist_models:
            try:
                result = self._run_celltypist(adata, model_name)
                all_predictions.append(result)
                logger.info(f"CellTypist {model_name}: {len(result['predictions'])} cells")
            except Exception as e:
                logger.warning(f"CellTypist {model_name} failed: {e}")

        # 2. Run SingleR if enabled
        if cfg.include_singler:
            try:
                result = self._run_singler(adata, cfg.singler_reference, data_path)
                all_predictions.append(result)
                logger.info(f"SingleR {cfg.singler_reference}: {len(result['predictions'])} cells")
            except Exception as e:
                logger.warning(f"SingleR failed: {e}")

        # 3. Run scType if enabled
        if cfg.include_sctype:
            try:
                result = self._run_sctype(adata)
                all_predictions.append(result)
                logger.info(f"scType: {len(result['predictions'])} cells")
            except Exception as e:
                logger.warning(f"scType failed: {e}")

        if not all_predictions:
            raise ValueError("No annotation methods succeeded")

        # Build consensus
        consensus_df, stats = self._build_consensus(
            all_predictions,
            adata.obs.index.tolist(),
            cfg,
        )
        logger.info(f"Consensus built: {len(consensus_df)} cells with agreement")

        # Compute class mapping
        class_mapping = self._compute_class_mapping(consensus_df, cfg.fine_grained)
        index_to_class = {v: k for k, v in class_mapping.items()}

        # Save outputs
        output_dir = get_pipeline_output_dir("ensemble_annotation", data_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config for skip logic validation
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "celltypist_models": cfg.celltypist_models,
                    "include_singler": cfg.include_singler,
                    "singler_reference": cfg.singler_reference,
                    "include_sctype": cfg.include_sctype,
                    "fine_grained": cfg.fine_grained,
                    "min_agreement": cfg.min_agreement,
                    "confidence_threshold": cfg.confidence_threshold,
                },
                f,
                indent=2,
            )

        annotations_path = output_dir / "annotations.parquet"
        consensus_df.write_parquet(annotations_path)
        logger.info(f"Saved annotations to {annotations_path}")

        # Save class mapping
        mapping_path = output_dir / "class_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(
                {
                    "class_mapping": class_mapping,
                    "index_to_class": index_to_class,
                },
                f,
                indent=2,
            )

        # Create derived dataset with lineage
        dataset_id = None
        if cfg.create_derived_dataset:
            try:
                dataset_id = self._create_annotated_dataset(
                    output_dir, inputs, consensus_df, stats, cfg
                )
                logger.info(f"Created annotated dataset: {dataset_id}")
            except Exception as e:
                logger.warning(f"Failed to create ClearML dataset: {e}")

        return StepArtifacts(
            inputs=inputs,
            outputs={
                **inputs,  # Pass through
                "annotations_parquet": str(annotations_path),
                "class_mapping": class_mapping,
                "index_to_class": index_to_class,
                "annotated_dataset_id": dataset_id,
                "annotation_stats": stats,
                "annotation_methods": [p["source"] for p in all_predictions],
            },
        )

    def _resolve_platform(self, inputs: dict) -> str:
        """Resolve platform from inputs."""
        platform_value = inputs.get("platform", "xenium")
        platform_path = resolve_artifact_path(platform_value, "platform")
        if platform_path and platform_path.exists() and platform_path.is_file():
            return platform_path.read_text().strip()
        return str(platform_value)

    def _load_expression(
        self,
        expression_path: Path,
        data_path: Path,
        platform: str,
    ):
        """Load expression data as AnnData."""
        import anndata as ad
        import scanpy as sc

        if expression_path.suffix == ".h5ad":
            adata = ad.read_h5ad(expression_path)
        elif expression_path.suffix == ".h5":
            adata = sc.read_10x_h5(expression_path)
        elif expression_path.is_dir():
            adata = sc.read_10x_mtx(expression_path)
        elif expression_path.suffix == ".csv":
            adata = self._load_csv_expression(expression_path)
        else:
            raise ValueError(f"Unsupported expression format: {expression_path}")

        # Ensure cell_id in obs
        if "cell_id" not in adata.obs.columns:
            adata.obs["cell_id"] = adata.obs.index.astype(str)

        return adata

    def _load_csv_expression(self, expression_path: Path):
        """Load expression from CSV (MERSCOPE format)."""
        import anndata as ad
        import pandas as pd
        import scipy.sparse as sp

        expr_df = pd.read_csv(expression_path, index_col=0)
        X = sp.csr_matrix(expr_df.values)

        adata = ad.AnnData(X=X)
        adata.obs_names = expr_df.index.astype(str)
        adata.var_names = expr_df.columns
        adata.obs["cell_id"] = adata.obs_names

        return adata

    def _build_consensus(
        self,
        results: list[AnnotationResult],
        cfg: EnsembleAnnotationConfig,
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Build consensus from AnnotationResult objects using polars."""
        all_votes = pl.concat([
            r.annotations_df.select([
                pl.col("cell_id"),
                pl.col("predicted_type"),
                pl.col("broad_category"),
                pl.col("confidence"),
            ]).with_columns(
                pl.lit(r.stats["source"]).alias("source")
            )
            for r in results
        ])

        if cfg.use_confidence_weighting:
            vote_scores = (
                all_votes
                .group_by(["cell_id", "broad_category"])
                .agg([
                    pl.col("confidence").sum().alias("vote_score"),
                    pl.len().alias("vote_count"),
                    pl.col("predicted_type").sort_by("confidence", descending=True).first().alias("best_predicted_type"),
                ])
            )
        else:
            vote_scores = (
                all_votes
                .group_by(["cell_id", "broad_category"])
                .agg([
                    pl.len().alias("vote_score"),
                    pl.len().alias("vote_count"),
                    pl.col("predicted_type").first().alias("best_predicted_type"),
                ])
            )

        winners = (
            vote_scores
            .sort("vote_score", descending=True)
            .group_by("cell_id")
            .first()
        )

        total_votes_per_cell = (
            all_votes
            .group_by("cell_id")
            .agg(pl.len().alias("n_votes"))
        )

        consensus = (
            winners
            .join(total_votes_per_cell, on="cell_id")
            .select([
                pl.col("cell_id"),
                pl.col("best_predicted_type").alias("predicted_type"),
                pl.col("broad_category"),
                (pl.col("vote_score") / pl.col("n_votes")).alias("confidence"),
                pl.col("n_votes"),
                pl.col("vote_count").alias("n_agreement"),
            ])
        )

        consensus = consensus.filter(pl.col("n_agreement") >= cfg.min_agreement)

        n_total = all_votes.select("cell_id").n_unique()
        n_unanimous = consensus.filter(pl.col("n_agreement") == pl.col("n_votes")).height
        stats = {
            "total_cells": n_total,
            "annotated_cells": consensus.height,
            "unanimous_agreement": n_unanimous,
            "majority_agreement": consensus.height - n_unanimous,
            "insufficient_votes": n_total - consensus.height,
            "methods_used": [r.stats["source"] for r in results],
        }

        return consensus, stats

    def _compute_class_mapping(
        self, annotations_df: pl.DataFrame, fine_grained: bool
    ) -> dict[str, int]:
        """Compute class mapping from annotations DataFrame."""
        if fine_grained:
            col = "predicted_type"
        else:
            col = "broad_category"

        cell_types = sorted(annotations_df[col].unique().to_list())
        # Filter out Unknown if present, add at end
        if "Unknown" in cell_types:
            cell_types.remove("Unknown")
            cell_types.append("Unknown")

        return {ct: i for i, ct in enumerate(cell_types)}

    def _create_annotated_dataset(
        self,
        output_dir: Path,
        inputs: dict,
        consensus_df: pl.DataFrame,
        stats: dict,
        cfg: EnsembleAnnotationConfig,
    ) -> str | None:
        """Create ClearML Dataset with lineage to raw data via S3.

        IMPORTANT: Never upload large data directly to ClearML.
        Pattern:
        1. Upload to S3 first
        2. Register with ClearML using add_external_files() (metadata only)

        Uses parent_datasets for lineage tracking.
        """
        try:
            from clearml import Dataset
        except ImportError:
            logger.warning("ClearML not available, skipping dataset creation")
            return None

        # Determine parent dataset for lineage
        parent_ids = []
        if cfg.parent_dataset_id:
            parent_ids.append(cfg.parent_dataset_id)
        elif inputs.get("raw_dataset_id"):
            parent_ids.append(inputs["raw_dataset_id"])

        # Create dataset name with config hash for cache lookup
        platform = inputs.get("platform", "unknown")
        raw_dataset_id = inputs.get("raw_dataset_id", cfg.parent_dataset_id or "local")
        config_hash = self._get_config_hash(cfg, raw_dataset_id)
        dataset_name = f"annotated-{platform}-{config_hash}"

        # S3 path for this dataset
        s3_path = f"datasets/annotated/{dataset_name}"

        try:
            if cfg.upload_to_s3:
                # Step 1: Upload to S3 first (NOT to ClearML)
                from dapidl.utils.s3 import upload_to_s3

                s3_uri = upload_to_s3(output_dir, s3_path)
                logger.info(f"Uploaded annotations to S3: {s3_uri}")

                # Step 2: Register with ClearML as external reference
                dataset = Dataset.create(
                    dataset_project="DAPIDL/annotated",
                    dataset_name=dataset_name,
                    parent_datasets=parent_ids if parent_ids else None,
                    # NOTE: Do NOT set output_uri - we don't upload to ClearML
                )

                # Add external reference to S3 - this does NOT upload!
                dataset.add_external_files(
                    source_url=s3_uri,
                    dataset_path="/",
                )
            else:
                # Local only - register local path reference
                s3_uri = None
                dataset = Dataset.create(
                    dataset_project="DAPIDL/annotated",
                    dataset_name=dataset_name,
                    parent_datasets=parent_ids if parent_ids else None,
                )
                dataset.add_external_files(
                    source_url=f"file://{output_dir}",
                    dataset_path="/",
                )

            # Metadata (this goes to ClearML - small JSON only)
            dataset.set_metadata(
                {
                    "config_hash": config_hash,
                    "raw_dataset_id": raw_dataset_id,
                    "celltypist_models": sorted(cfg.celltypist_models),
                    "include_singler": cfg.include_singler,
                    "singler_reference": cfg.singler_reference,
                    "include_sctype": cfg.include_sctype,
                    "min_agreement": cfg.min_agreement,
                    "confidence_threshold": cfg.confidence_threshold,
                    "fine_grained": cfg.fine_grained,
                    "annotation_methods": stats["methods_used"],
                    "n_cells": len(consensus_df),
                    "n_classes": len(consensus_df["predicted_type"].unique()),
                    "parent_dataset": cfg.parent_dataset_id,
                    "platform": platform,
                    "unanimous_agreement_pct": stats["unanimous_agreement"]
                    / stats["annotated_cells"]
                    if stats["annotated_cells"] > 0
                    else 0,
                    "s3_uri": s3_uri,  # Store S3 location in metadata
                    "local_path": str(output_dir),  # For reference
                    "registration_type": "external_reference",
                    "uploaded_to_clearml": False,
                }
            )

            dataset.finalize()
            # NOTE: Do NOT call dataset.upload() - files are already on S3

            logger.info(f"Registered ClearML dataset: {dataset.id} -> {s3_uri or output_dir}")
            return dataset.id

        except Exception as e:
            logger.warning(f"Failed to register ClearML dataset: {e}")
            return None

    def get_queue(self) -> str:
        """Return queue name for this step."""
        return "default"  # CPU queue for CellTypist/SingleR

    def create_clearml_task(
        self,
        project: str = "DAPIDL/pipeline",
        task_name: str | None = None,
    ):
        """Create ClearML Task for this step."""
        from pathlib import Path

        from clearml import Task

        task_name = task_name or f"step-{self.name}"

        # Use the runner script for remote execution (avoids uv entry point issues)
        runner_script = (
            Path(__file__).parent.parent.parent.parent.parent
            / "scripts"
            / f"clearml_step_runner_{self.name}.py"
        )

        self._task = Task.create(
            project_name=project,
            task_name=task_name,
            task_type=Task.TaskTypes.data_processing,
            script=str(runner_script),
            branch="main",
            argparse_args=[f"--step={self.name}"],
            # Enable auto Task.init() injection - each step has unique script file
            add_task_init_call=False,  # Handle in step runner
            # Explicitly include clearml to ensure it's installed
            # even if editable install has issues with the agent's venv
            packages=["-e .", "clearml>=1.16"],
        )

        # Connect parameters
        params = {
            "step_name": self.name,
            "celltypist_models": ",".join(self.config.celltypist_models),
            "include_singler": self.config.include_singler,
            "singler_reference": self.config.singler_reference,
            "min_agreement": self.config.min_agreement,
            "confidence_threshold": self.config.confidence_threshold,
            "fine_grained": self.config.fine_grained,
        }
        self._task.set_parameters(params, __parameters_prefix="step_config")

        return self._task
