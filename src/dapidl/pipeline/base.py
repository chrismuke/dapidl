"""Base classes and protocols for the DAPIDL pipeline.

This module defines the core abstractions:
- StepArtifacts: Input/output artifacts between steps
- PipelineStep: ABC for pipeline step implementations
- SegmentationConfig/Result: Segmenter interface types
- AnnotationConfig/Result: Annotator interface types

Protocols are used for segmenters and annotators to enable
structural subtyping - components just need to implement
the required methods without inheriting from a base class.

Note: PipelineConfig is defined in controller.py for the full pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl
from loguru import logger


# =============================================================================
# Artifact URL Resolution
# =============================================================================


def resolve_artifact_path(value: str | Path | None, artifact_name: str = "") -> Path | None:
    """Resolve an artifact value to a local path.

    Handles multiple artifact formats:
    1. Local path string/Path - returns directly
    2. Path reference JSON - extracts local_path from {"local_path": ..., "type": "path_reference"}
    3. ClearML URL - downloads from ClearML file server
    4. S3 URI - downloads from S3

    Args:
        value: Path string, URL string, JSON string, or Path object
        artifact_name: Name for logging (optional)

    Returns:
        Local Path object, or None if value is None/empty
    """
    import json as json_module

    if value is None or value == "":
        return None

    value_str = str(value)

    # Check if it's a path reference JSON (our local path format)
    if value_str.startswith("{") and "local_path" in value_str:
        try:
            ref = json_module.loads(value_str)
            if ref.get("type") == "path_reference" and "local_path" in ref:
                local_path = Path(ref["local_path"])
                if local_path.exists():
                    logger.info(f"Using local path reference: {local_path}")
                    return local_path
                else:
                    logger.warning(f"Local path reference does not exist: {local_path}")
                    # Fall through to try other methods
        except json_module.JSONDecodeError:
            pass  # Not valid JSON, try other methods

    # Check if it's already a valid local path
    potential_path = Path(value_str)
    if potential_path.exists():
        logger.info(f"Using existing local path: {potential_path}")
        return potential_path

    # Check if it's an S3 URI — use ClearML StorageManager (boto3-based, no aws CLI needed)
    if value_str.startswith("s3://"):
        logger.info(f"Resolving S3 URI: {artifact_name or value_str[:80]}...")
        from clearml import StorageManager

        local_path = StorageManager.get_local_copy(value_str)
        if local_path is None:
            raise ValueError(f"Failed to download from S3: {value_str}")
        local_path = Path(local_path)
        logger.info(f"Downloaded from S3 to: {local_path}")

        # If it's a text file, check if it contains a path reference JSON
        if local_path.suffix == ".txt":
            try:
                content = local_path.read_text().strip()
                if content.startswith("{") and "local_path" in content:
                    ref = json_module.loads(content)
                    if ref.get("type") == "path_reference" and "local_path" in ref:
                        ref_path = Path(ref["local_path"])
                        if ref_path.exists():
                            logger.info(f"Resolved path reference from S3 artifact: {ref_path}")
                            return ref_path
                        else:
                            logger.warning(f"Path reference does not exist: {ref_path}")
            except (json_module.JSONDecodeError, OSError) as e:
                logger.debug(f"Not a path reference JSON: {e}")

        return local_path

    # Check if it's a ClearML URL
    if value_str.startswith("http://") or value_str.startswith("https://"):
        logger.info(f"Resolving artifact URL: {artifact_name or value_str[:80]}...")

        try:
            from clearml import StorageManager

            # Download to local cache and get local path
            local_path = StorageManager.get_local_copy(value_str)
            if local_path is None:
                raise ValueError(f"Failed to download artifact: {value_str}")

            local_path = Path(local_path)
            logger.info(f"Downloaded artifact to: {local_path}")

            # If it's a text file, check if it contains a path reference JSON
            if local_path.suffix == ".txt":
                try:
                    content = local_path.read_text().strip()
                    if content.startswith("{") and "local_path" in content:
                        ref = json_module.loads(content)
                        if ref.get("type") == "path_reference" and "local_path" in ref:
                            ref_path = Path(ref["local_path"])
                            if ref_path.exists():
                                logger.info(f"Resolved path reference from artifact: {ref_path}")
                                return ref_path
                            else:
                                logger.warning(f"Path reference does not exist: {ref_path}")
                except (json_module.JSONDecodeError, OSError) as e:
                    logger.debug(f"Not a path reference JSON: {e}")

            # If it's a zip file, extract it
            if local_path.suffix == ".zip":
                import zipfile
                import tempfile

                extract_dir = Path(tempfile.mkdtemp(prefix="dapidl_artifact_"))
                logger.info(f"Extracting zip to: {extract_dir}")

                with zipfile.ZipFile(local_path, "r") as zf:
                    zf.extractall(extract_dir)

                # Check if all files are in a single subdirectory
                extracted_items = list(extract_dir.iterdir())
                if len(extracted_items) == 1 and extracted_items[0].is_dir():
                    # Return the single subdirectory
                    local_path = extracted_items[0]
                else:
                    local_path = extract_dir

                logger.info(f"Extracted to: {local_path}")

            return local_path

        except ImportError:
            logger.warning("ClearML not available, treating URL as path")
            return Path(value_str)

    # Treat as a path (might not exist yet)
    return Path(value_str)


# =============================================================================
# Pipeline Output Directory
# =============================================================================


def get_pipeline_output_dir(step_name: str, data_path: Path | None = None) -> Path:
    """Get output directory for a pipeline step.

    Writes to a dedicated pipeline_outputs directory, NOT inside raw datasets.
    Raw datasets are immutable and backed up on S3.

    Priority:
    1. DAPIDL_PIPELINE_OUTPUTS env var (explicit override)
    2. {data_path}/../../pipeline_outputs/{step_name} (sibling to raw data)
    3. /tmp/dapidl/pipeline_outputs/{step_name} (fallback)

    Args:
        step_name: Pipeline step name (e.g., "annotation", "segmentation")
        data_path: Path to the raw dataset (used to derive sibling output dir)

    Returns:
        Path to the output directory (created if needed)
    """
    import os

    env_root = os.environ.get("DAPIDL_PIPELINE_OUTPUTS")
    if env_root:
        output_dir = Path(env_root) / step_name
    elif data_path is not None:
        # Place outputs as sibling to the dataset within a shared pipeline_outputs dir
        # e.g., /mnt/work/datasets/raw/xenium/xenium-lung-2fov → /mnt/work/datasets/pipeline_outputs/xenium-lung-2fov/annotation
        # Strip trailing "outs" or "outs/" (Xenium convention)
        effective_path = data_path
        if effective_path.name == "outs":
            effective_path = effective_path.parent
        dataset_name = effective_path.name
        # Walk up to find the datasets root (parent of raw/)
        datasets_root = effective_path
        for _ in range(5):
            if (datasets_root / "raw").is_dir():
                break
            datasets_root = datasets_root.parent
        else:
            # Fallback: use parent of data_path
            datasets_root = effective_path.parent.parent
        output_dir = datasets_root / "pipeline_outputs" / dataset_name / step_name
    else:
        output_dir = Path("/tmp/dapidl/pipeline_outputs") / step_name

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =============================================================================
# Step Artifacts
# =============================================================================


@dataclass
class StepArtifacts:
    """Container for artifacts passed between pipeline steps.

    Each step receives inputs from parent steps and produces outputs
    for child steps. Artifacts can be:
    - File paths (str or Path)
    - ClearML Dataset IDs
    - DataFrames (serialized to parquet)
    - JSON-serializable dicts
    """

    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)

    def get_input(self, key: str, default: Any = None) -> Any:
        """Get an input artifact by key."""
        return self.inputs.get(key, default)

    def set_output(self, key: str, value: Any) -> None:
        """Set an output artifact."""
        self.outputs[key] = value

    def require_input(self, key: str) -> Any:
        """Get a required input artifact, raising if not present."""
        if key not in self.inputs:
            raise ValueError(f"Required input artifact '{key}' not found")
        return self.inputs[key]


# =============================================================================
# Pipeline Step Base Class
# =============================================================================


class PipelineStep(ABC):
    """Abstract base class for pipeline steps.

    Each step is responsible for:
    1. Defining its parameter schema (for ClearML UI)
    2. Validating required inputs
    3. Executing its logic
    4. Producing output artifacts

    Steps can run locally or remotely via ClearML execute_remotely().
    """

    name: str = "base_step"
    description: str = "Base pipeline step"

    @abstractmethod
    def get_parameter_schema(self) -> dict[str, Any]:
        """Return JSON schema for UI-editable parameters.

        This schema is used by ClearML to render parameter inputs.
        """
        ...

    @abstractmethod
    def validate_inputs(self, artifacts: StepArtifacts) -> bool:
        """Validate that required inputs are present.

        Returns True if valid, raises ValueError if not.
        """
        ...

    @abstractmethod
    def execute(self, artifacts: StepArtifacts, **params: Any) -> StepArtifacts:
        """Execute the step logic.

        Args:
            artifacts: Input artifacts from parent steps
            **params: UI-configured parameters

        Returns:
            Updated artifacts with outputs added
        """
        ...

    def get_queue(self) -> str:
        """Return the queue name for this step (default: CPU)."""
        return "default"


# =============================================================================
# Segmentation Types
# =============================================================================


@dataclass
class SegmentationConfig:
    """Configuration for nucleus segmentation.

    UI-editable parameters for segmentation step.
    """

    method: str = "cellpose"  # cellpose, native
    diameter: int = 40  # Nucleus diameter in pixels (Xenium: 40, MERSCOPE: 70)
    flow_threshold: float = 0.4  # Cellpose flow threshold
    cellprob_threshold: float = 0.0  # Cellpose cell probability threshold
    match_threshold_um: float = 5.0  # Max centroid-nucleus distance for matching
    min_size: int = 15  # Minimum nucleus area in pixels
    platform: str = "xenium"  # xenium or merscope (affects defaults)
    tile_size: int = 4096  # Tile size for large images
    tile_overlap: int = 256  # Overlap between tiles
    pixel_size_um: float = 0.0  # Pixel size in microns (0 = auto-detect)
    gpu: bool = True  # Use GPU for Cellpose
    skip_boundaries: bool = True  # Skip slow boundary extraction (not needed for training)
    parallel_boundaries: bool = True  # Use parallel processing for boundaries (if extracted)
    n_boundary_workers: int = 8  # Number of workers for parallel boundary extraction


@dataclass
class SegmentationResult:
    """Output from nucleus segmentation.

    Contains centroids, boundaries, and optional masks.
    """

    # Centroid positions (cell_id, x, y, matched_nucleus_id, distance_um)
    centroids_df: pl.DataFrame

    # Boundary polygons (cell_id, boundary_vertices)
    boundaries_df: pl.DataFrame | None = None

    # Full mask array (optional, memory-intensive)
    masks: np.ndarray | None = None

    # Statistics about matching
    matching_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        """Number of cells with matched nuclei."""
        return len(self.centroids_df)

    @property
    def match_rate(self) -> float:
        """Proportion of cells with matched nuclei."""
        return self.matching_stats.get("match_rate", 0.0)


@runtime_checkable
class SegmenterProtocol(Protocol):
    """Protocol for segmenter implementations.

    Segmenters detect nuclei from DAPI images and optionally
    match them to existing cell boundaries.
    """

    name: str

    def segment(
        self,
        dapi_image: np.ndarray,
        config: SegmentationConfig,
    ) -> SegmentationResult:
        """Segment nuclei from DAPI image.

        Args:
            dapi_image: 2D DAPI image (H, W), uint16
            config: Segmentation parameters

        Returns:
            Segmentation results with centroids and boundaries
        """
        ...

    def segment_and_match(
        self,
        dapi_image: np.ndarray,
        cells_df: pl.DataFrame,
        config: SegmentationConfig,
    ) -> SegmentationResult:
        """Segment nuclei and match to existing cell data.

        Args:
            dapi_image: 2D DAPI image (H, W), uint16
            cells_df: Cell metadata with x_centroid, y_centroid columns
            config: Segmentation parameters

        Returns:
            Segmentation results with nucleus-cell matching
        """
        ...


# =============================================================================
# Annotation Types
# =============================================================================


@dataclass
class AnnotationConfig:
    """Configuration for cell type annotation.

    UI-editable parameters for annotation step.
    """

    method: str = "celltypist"  # celltypist, ground_truth, popv

    # CellTypist options
    strategy: str = "consensus"  # single, consensus, hierarchical
    model_names: list[str] = field(
        default_factory=lambda: ["Cells_Adult_Breast.pkl"]
    )
    confidence_threshold: float = 0.5
    majority_voting: bool = True
    extended_consensus: bool = False  # Use 6 CellTypist models for better coverage

    # Ground truth options
    ground_truth_file: str | None = None
    ground_truth_sheet: str | None = None  # Excel sheet name
    cell_id_column: str = "cell_id"
    celltype_column: str = "cell_type"

    # Output options
    fine_grained: bool = False  # Use detailed cell types vs broad categories


@dataclass
class AnnotationResult:
    """Output from cell type annotation.

    Contains cell type predictions and class mapping.
    """

    # Annotations (cell_id, predicted_type, broad_category, confidence)
    annotations_df: pl.DataFrame

    # Mapping from class names to indices
    class_mapping: dict[str, int] = field(default_factory=dict)

    # Reverse mapping from indices to class names
    index_to_class: dict[int, str] = field(default_factory=dict)

    # Statistics
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_annotated(self) -> int:
        """Number of annotated cells."""
        return len(self.annotations_df)

    @property
    def n_classes(self) -> int:
        """Number of unique classes."""
        return len(self.class_mapping)

    def get_class_distribution(self) -> dict[str, int]:
        """Return counts per class."""
        if "broad_category" in self.annotations_df.columns:
            col = "broad_category"
        else:
            col = "predicted_type"
        return (
            self.annotations_df.group_by(col)
            .count()
            .to_pandas()
            .set_index(col)["count"]
            .to_dict()
        )


@runtime_checkable
class AnnotatorProtocol(Protocol):
    """Protocol for annotator implementations.

    Annotators assign cell type labels using various methods:
    - CellTypist: Gene expression-based classification
    - Ground Truth: Load from curated Excel/CSV files
    - PopV: Voting-based annotation
    """

    name: str

    def annotate(
        self,
        config: AnnotationConfig,
        adata: Any | None = None,  # AnnData object
        expression_path: Path | None = None,
        cells_df: pl.DataFrame | None = None,
    ) -> AnnotationResult:
        """Annotate cells with type labels.

        At least one of adata, expression_path, or cells_df must be provided.

        Args:
            config: Annotation parameters
            adata: AnnData object with expression data
            expression_path: Path to expression matrix (h5, h5ad)
            cells_df: Cell metadata DataFrame

        Returns:
            Annotation results with cell types and confidence
        """
        ...
