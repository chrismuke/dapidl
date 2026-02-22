"""CL Standardization Pipeline Step.

This step standardizes cell type annotations to Cell Ontology (CL) terms,
enabling cross-dataset, cross-annotator comparison and training.

Features:
    - Maps annotations from any method to CL IDs
    - Rolls up to target hierarchy level (broad, coarse, fine)
    - Filters by mapping confidence
    - Tracks mapping statistics and unmapped labels
    - Produces CL-standardized annotations for downstream training

Usage in ClearML Pipeline:
    ```python
    from dapidl.pipeline.steps import CLStandardizationStep, CLStandardizationConfig

    step = CLStandardizationStep(config=CLStandardizationConfig(
        target_level="coarse",
        min_confidence=0.7,
        include_unmapped=False,
    ))
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from dapidl.ontology import (
    CLMapper,
    MapperConfig,
    MappingMethod,
    get_all_annotator_mappings,
    get_all_gt_mappings,
    get_broad_category,
    get_coarse_category,
)
from dapidl.pipeline.base import (
    PipelineStep,
    StepArtifacts,
    get_pipeline_output_dir,
    resolve_artifact_path,
)


@dataclass
class CLStandardizationConfig:
    """Configuration for CL standardization step."""

    # Hierarchy level for output labels
    target_level: str = "coarse"  # "broad", "coarse", or "fine"

    # Mapping parameters
    min_confidence: float = 0.5  # Minimum confidence to include
    fuzzy_threshold: float = 0.85  # Fuzzy matching threshold

    # Filtering
    include_unmapped: bool = False  # Include cells with UNMAPPED CL ID
    drop_duplicates: bool = True  # Drop duplicate cell_id entries

    # Column names
    input_label_col: str = "predicted_type"  # Column with labels to standardize
    output_cl_id_col: str = "cl_id"  # Output column for CL ID
    output_cl_name_col: str = "cl_name"  # Output column for CL name
    output_category_col: str = "cl_category"  # Output column for target level category

    # Load full OBO
    use_obo_loader: bool = False  # Whether to load full OBO (slower but more complete)


class CLStandardizationStep(PipelineStep):
    """Standardize annotations to Cell Ontology.

    This step:
    1. Loads annotation DataFrame from previous step
    2. Maps all cell type labels to CL IDs
    3. Rolls up to target hierarchy level
    4. Filters by confidence and mapping status
    5. Outputs standardized DataFrame

    Queue: default (CPU - no GPU required)
    """

    name = "cl_standardization"
    description = "Standardize cell type annotations to Cell Ontology"
    queue = "default"

    def __init__(self, config: CLStandardizationConfig | None = None):
        """Initialize the step.

        Args:
            config: Step configuration
        """
        self.config = config or CLStandardizationConfig()
        self._task = None
        self._mapper: CLMapper | None = None

    def get_parameter_schema(self) -> dict[str, Any]:
        """Return JSON schema for ClearML UI parameters."""
        return {
            "type": "object",
            "properties": {
                "target_level": {
                    "type": "string",
                    "enum": ["broad", "coarse", "fine"],
                    "default": "coarse",
                    "description": "Hierarchy level for output categories",
                },
                "min_confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "Minimum mapping confidence to include",
                },
                "fuzzy_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.85,
                    "description": "Threshold for fuzzy string matching",
                },
                "include_unmapped": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include cells with UNMAPPED CL ID",
                },
                "input_label_col": {
                    "type": "string",
                    "default": "predicted_type",
                    "description": "Column containing labels to standardize",
                },
            },
        }

    def validate_inputs(self, artifacts: StepArtifacts) -> bool:
        """Validate required inputs.

        Requires annotations_parquet from annotation step.
        """
        return "annotations_parquet" in artifacts.outputs

    def execute(self, artifacts: StepArtifacts) -> StepArtifacts:
        """Execute CL standardization.

        Args:
            artifacts: Input artifacts from annotation step

        Returns:
            Output artifacts with:
            - cl_annotations_parquet: CL-standardized annotations
            - cl_class_mapping: Category → index mapping
            - cl_stats: Standardization statistics
        """
        cfg = self.config
        inputs = artifacts.outputs

        # Load annotations
        annotations_path = resolve_artifact_path(
            inputs.get("annotations_parquet"), "annotations_parquet"
        )
        if annotations_path is None:
            raise ValueError("annotations_parquet artifact required")

        logger.info(f"Loading annotations from {annotations_path}")
        df = pl.read_parquet(annotations_path)
        logger.info(f"Loaded {len(df)} annotations")

        # Initialize mapper
        self._init_mapper()

        # Determine label column
        label_col = cfg.input_label_col
        if label_col not in df.columns:
            # Try alternatives
            for alt in ["predicted_type_1", "broad_category", "cell_type"]:
                if alt in df.columns:
                    label_col = alt
                    logger.info(f"Using alternative column: {label_col}")
                    break
            else:
                raise ValueError(
                    f"Label column '{cfg.input_label_col}' not found. "
                    f"Available: {df.columns}"
                )

        # Get unique labels
        unique_labels = df[label_col].unique().to_list()
        logger.info(f"Unique labels to map: {len(unique_labels)}")

        # Map all labels
        mappings = {}
        for label in unique_labels:
            if label is None or str(label).lower() in ["none", "null", "nan"]:
                mappings[label] = {
                    "cl_id": "UNMAPPED",
                    "cl_name": "Unknown",
                    "confidence": 0.0,
                    "method": "unmapped",
                    "broad_category": "Unknown",
                    "coarse_category": "Unknown",
                }
            else:
                result = self._mapper.map_with_info(str(label))
                mappings[label] = {
                    "cl_id": result.cl_id,
                    "cl_name": result.cl_name,
                    "confidence": result.confidence,
                    "method": result.method.value,
                    "broad_category": result.broad_category,
                    "coarse_category": result.coarse_category,
                }

        # Log mapping statistics
        stats = self._compute_stats(mappings)
        logger.info(f"Mapping stats: {stats}")

        # Add CL columns to DataFrame
        df = df.with_columns([
            pl.col(label_col).map_elements(
                lambda x: mappings.get(x, {}).get("cl_id", "UNMAPPED"),
                return_dtype=pl.Utf8
            ).alias(cfg.output_cl_id_col),
            pl.col(label_col).map_elements(
                lambda x: mappings.get(x, {}).get("cl_name", "Unknown"),
                return_dtype=pl.Utf8
            ).alias(cfg.output_cl_name_col),
            pl.col(label_col).map_elements(
                lambda x: mappings.get(x, {}).get("confidence", 0.0),
                return_dtype=pl.Float64
            ).alias("cl_confidence"),
            pl.col(label_col).map_elements(
                lambda x: mappings.get(x, {}).get("method", "unmapped"),
                return_dtype=pl.Utf8
            ).alias("cl_method"),
        ])

        # Add target level category
        if cfg.target_level == "broad":
            df = df.with_columns([
                pl.col(label_col).map_elements(
                    lambda x: mappings.get(x, {}).get("broad_category", "Unknown"),
                    return_dtype=pl.Utf8
                ).alias(cfg.output_category_col)
            ])
        elif cfg.target_level == "coarse":
            df = df.with_columns([
                pl.col(label_col).map_elements(
                    lambda x: mappings.get(x, {}).get("coarse_category", "Unknown"),
                    return_dtype=pl.Utf8
                ).alias(cfg.output_category_col)
            ])
        else:  # fine
            df = df.with_columns([
                pl.col(cfg.output_cl_name_col).alias(cfg.output_category_col)
            ])

        # Filter by confidence
        before_count = len(df)
        df = df.filter(pl.col("cl_confidence") >= cfg.min_confidence)
        logger.info(
            f"Filtered by confidence >= {cfg.min_confidence}: "
            f"{before_count} → {len(df)}"
        )

        # Filter unmapped
        if not cfg.include_unmapped:
            before_count = len(df)
            df = df.filter(pl.col(cfg.output_cl_id_col) != "UNMAPPED")
            logger.info(f"Removed unmapped: {before_count} → {len(df)}")

        # Drop duplicates
        if cfg.drop_duplicates and "cell_id" in df.columns:
            before_count = len(df)
            df = df.unique(subset=["cell_id"])
            logger.info(f"Deduplicated: {before_count} → {len(df)}")

        # Compute class mapping for target level
        categories = sorted(df[cfg.output_category_col].unique().to_list())
        if "Unknown" in categories:
            categories.remove("Unknown")
            categories.append("Unknown")
        class_mapping = {cat: i for i, cat in enumerate(categories)}
        index_to_class = {i: cat for cat, i in class_mapping.items()}

        logger.info(f"CL standardization complete: {len(class_mapping)} categories")

        # Save outputs
        data_path = resolve_artifact_path(inputs.get("data_path"), "data_path")
        output_dir = get_pipeline_output_dir("cl_standardization", data_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "cl_annotations.parquet"
        df.write_parquet(output_path)
        logger.info(f"Saved CL annotations to {output_path}")

        # Save class mapping
        import json
        mapping_path = output_dir / "cl_class_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump({
                "class_mapping": class_mapping,
                "index_to_class": index_to_class,
                "target_level": cfg.target_level,
            }, f, indent=2)

        # Save mapping details
        mapping_details_path = output_dir / "cl_mappings.json"
        with open(mapping_details_path, "w") as f:
            json.dump(mappings, f, indent=2, default=str)

        return StepArtifacts(
            inputs=inputs,
            outputs={
                **inputs,
                "cl_annotations_parquet": str(output_path),
                "cl_class_mapping": class_mapping,
                "cl_index_to_class": index_to_class,
                "cl_stats": stats,
                "cl_target_level": cfg.target_level,
            },
        )

    def _init_mapper(self) -> None:
        """Initialize the CL mapper with all annotator mappings."""
        if self._mapper is not None:
            return

        cfg = self.config

        # Get all mappings
        annotator_maps = get_all_annotator_mappings()
        gt_maps = get_all_gt_mappings()

        # Create mapper config
        mapper_config = MapperConfig(
            fuzzy_threshold=cfg.fuzzy_threshold,
            use_obo_loader=cfg.use_obo_loader,
        )

        self._mapper = CLMapper(
            config=mapper_config,
            annotator_mappings=annotator_maps,
            ground_truth_mappings=gt_maps,
        )

        logger.info(f"Initialized CLMapper: {self._mapper.get_mapping_stats()}")

    def _compute_stats(self, mappings: dict) -> dict:
        """Compute mapping statistics."""
        total = len(mappings)
        if total == 0:
            return {"total": 0}

        by_method = {}
        unmapped = 0
        for m in mappings.values():
            method = m.get("method", "unmapped")
            by_method[method] = by_method.get(method, 0) + 1
            if m.get("cl_id") == "UNMAPPED":
                unmapped += 1

        return {
            "total_labels": total,
            "mapped": total - unmapped,
            "unmapped": unmapped,
            "map_rate": (total - unmapped) / total if total > 0 else 0.0,
            "by_method": by_method,
        }

    def create_clearml_task(
        self,
        project: str = "DAPIDL/pipeline",
        task_name: str | None = None,
    ):
        """Create ClearML Task for this step."""
        from clearml import Task

        task_name = task_name or f"step-{self.name}"

        # Get per-step runner script path
        runner_script = (
            Path(__file__).parent.parent.parent.parent.parent
            / "scripts" / f"clearml_step_runner_{self.name}.py"
        )

        self._task = Task.create(
            project_name=project,
            task_name=task_name,
            task_type=Task.TaskTypes.data_processing,
            script=str(runner_script),
            argparse_args=[f"--step={self.name}"],
            add_task_init_call=False,
            packages=["-e ."],
        )

        # Connect parameters
        params = {
            "step_name": self.name,
            "target_level": self.config.target_level,
            "min_confidence": self.config.min_confidence,
            "fuzzy_threshold": self.config.fuzzy_threshold,
            "include_unmapped": self.config.include_unmapped,
            "input_label_col": self.config.input_label_col,
        }
        self._task.set_parameters(params, __parameters_prefix="step_config")

        return self._task


def standardize_annotations(
    annotations_df: pl.DataFrame,
    label_col: str = "predicted_type",
    target_level: str = "coarse",
    min_confidence: float = 0.5,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Convenience function to standardize annotations.

    Args:
        annotations_df: DataFrame with cell annotations
        label_col: Column containing cell type labels
        target_level: Hierarchy level ("broad", "coarse", "fine")
        min_confidence: Minimum mapping confidence

    Returns:
        Tuple of (standardized DataFrame, class_mapping dict)
    """
    config = CLStandardizationConfig(
        target_level=target_level,
        min_confidence=min_confidence,
        input_label_col=label_col,
    )

    step = CLStandardizationStep(config)

    # Create mock artifacts
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        annotations_df.write_parquet(f.name)

        artifacts = StepArtifacts(
            outputs={
                "annotations_parquet": f.name,
                "data_path": str(Path(f.name).parent),
            }
        )

        result = step.execute(artifacts)

        # Load result
        result_df = pl.read_parquet(result.outputs["cl_annotations_parquet"])
        class_mapping = result.outputs["cl_class_mapping"]

        return result_df, class_mapping
