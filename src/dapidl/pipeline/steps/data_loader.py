"""Data Loader Pipeline Step.

Step 1: Load raw spatial transcriptomics data with smart source detection.

This step uses a **local-first fallback** strategy:
1. Check local dataset registry (fastest)
2. Check S3 cache if S3 URI provided
3. Download from ClearML Dataset if dataset_id provided
4. Download from S3 if s3_uri provided

This allows you to specify remote URIs while automatically using
local copies when available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from dapidl.pipeline.base import PipelineStep, StepArtifacts


# ============================================================================
# LOCAL DATA REGISTRY
# ============================================================================
# Maps S3 URIs and dataset names to known local paths.
# Add your local datasets here to avoid re-downloading.
# The loader checks these paths FIRST before downloading.

LOCAL_DATA_REGISTRY: dict[str, str] = {
    # S3 URIs -> local paths
    "s3://dapidl/raw-data/xenium-breast-cancer-rep1/": "/mnt/work/datasets/raw/xenium/breast_tumor_rep1/outs",
    "s3://dapidl/raw-data/xenium-lung-2fov/": "/mnt/work/datasets/raw/xenium/lung_2fov/outs",
    "s3://dapidl/raw-data/xenium-ovarian-cancer/": "/mnt/work/datasets/raw/xenium/ovarian_cancer/outs",
    # Dataset names -> local paths (for ClearML datasets)
    "xenium-breast-cancer-rep1": "/mnt/work/datasets/raw/xenium/breast_tumor_rep1/outs",
    "xenium-lung-2fov": "/mnt/work/datasets/raw/xenium/lung_2fov/outs",
    "xenium-ovarian-cancer": "/mnt/work/datasets/raw/xenium/ovarian_cancer/outs",
    "merscope-breast": "/mnt/work/datasets/raw/merscope/breast",
}


@dataclass
class DataLoaderConfig:
    """Configuration for data loading step."""

    # ClearML Dataset configuration
    dataset_id: str | None = None
    dataset_project: str = "DAPIDL/datasets"
    dataset_name: str | None = None
    dataset_version: str | None = None

    # Platform detection
    platform: str = "auto"  # "auto", "xenium", "merscope"

    # Local path override (for debugging)
    local_path: str | None = None

    # S3 URI (e.g., s3://dapidl/raw-data/xenium-breast/)
    s3_uri: str | None = None

    # Validation options
    validate_files: bool = True
    required_files: list[str] = field(
        default_factory=lambda: ["morphology_focus.ome.tif", "cells.parquet"]
    )


class DataLoaderStep(PipelineStep):
    """Load raw data from ClearML Dataset or local path.

    This is the entry point for the pipeline. It downloads/locates
    the spatial transcriptomics data and validates the required files.

    Supports:
    - Xenium: morphology_focus.ome.tif, cells.parquet, cell_feature_matrix/
    - MERSCOPE: images/mosaic_DAPI_z*.tif, cell_metadata_*.csv, cell_by_gene*.csv
    """

    name = "data_loader"
    queue = "default"  # CPU queue

    def __init__(self, config: DataLoaderConfig | None = None):
        """Initialize data loader step.

        Args:
            config: Data loader configuration
        """
        self.config = config or DataLoaderConfig()
        self._task = None

    def get_parameter_schema(self) -> dict[str, Any]:
        """Return JSON schema for ClearML UI parameters."""
        return {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "ClearML Dataset ID",
                },
                "dataset_project": {
                    "type": "string",
                    "default": "DAPIDL/datasets",
                    "description": "ClearML project containing dataset",
                },
                "dataset_name": {
                    "type": "string",
                    "description": "Dataset name (alternative to ID)",
                },
                "platform": {
                    "type": "string",
                    "enum": ["auto", "xenium", "merscope"],
                    "default": "auto",
                    "description": "Spatial platform type",
                },
                "local_path": {
                    "type": "string",
                    "description": "Local path override (debugging)",
                },
                "s3_uri": {
                    "type": "string",
                    "description": "S3 URI for data (e.g., s3://dapidl/raw-data/xenium-breast/)",
                },
            },
        }

    def validate_inputs(self, artifacts: StepArtifacts) -> bool:
        """Validate step inputs.

        For DataLoaderStep, we need either:
        - dataset_id or dataset_name in config
        - local_path in config
        - s3_uri in config
        """
        cfg = self.config

        if cfg.local_path:
            return Path(cfg.local_path).exists()

        if cfg.s3_uri:
            return cfg.s3_uri.startswith("s3://")

        return bool(cfg.dataset_id or cfg.dataset_name)

    def execute(self, artifacts: StepArtifacts) -> StepArtifacts:
        """Execute data loading step.

        Args:
            artifacts: Input artifacts (typically empty for first step)

        Returns:
            Output artifacts containing:
            - data_path: Path to dataset root
            - platform: Detected or specified platform
            - cells_parquet: Path to cells file
            - expression_path: Path to expression matrix
            - metadata: Dict with sample info
        """
        cfg = self.config

        # Get data path with LOCAL-FIRST fallback strategy
        data_path = self._get_data_path_with_fallback()

        # Resolve effective data path (handle outs/ subdirectory)
        data_path = self._resolve_data_path(data_path)
        logger.info(f"Resolved data path: {data_path}")

        # Detect platform
        platform = self._detect_platform(data_path) if cfg.platform == "auto" else cfg.platform
        logger.info(f"Platform: {platform}")

        # Validate required files
        if cfg.validate_files:
            self._validate_files(data_path, platform)

        # Get file paths
        cells_path, expression_path = self._get_paths(data_path, platform)

        # Extract metadata
        metadata = self._extract_metadata(data_path, platform)

        return StepArtifacts(
            inputs=artifacts.inputs,
            outputs={
                "data_path": str(data_path),
                "platform": platform,
                "cells_parquet": str(cells_path) if cells_path else None,
                "expression_path": str(expression_path) if expression_path else None,
                "metadata": metadata,
            },
        )

    def _get_data_path_with_fallback(self) -> Path:
        """Get data path using local-first fallback strategy.

        Priority order:
        1. Explicit local_path in config
        2. Registry lookup for s3_uri or dataset_name
        3. S3 cache directory
        4. Download from S3 (if s3_uri provided)
        5. Download from ClearML (if dataset_id/name provided)
        """
        cfg = self.config

        # 1. Explicit local path always wins
        if cfg.local_path:
            local = Path(cfg.local_path)
            if local.exists():
                logger.info(f"✓ Using explicit local path: {local}")
                return local
            raise FileNotFoundError(f"Local path not found: {local}")

        # 2. Check registry for S3 URI
        if cfg.s3_uri:
            registry_path = LOCAL_DATA_REGISTRY.get(cfg.s3_uri.rstrip("/") + "/")
            if registry_path and Path(registry_path).exists():
                logger.info(f"✓ Found local copy via registry: {registry_path}")
                return Path(registry_path)

            # 3. Check S3 cache directory
            parts = cfg.s3_uri.rstrip("/").split("/")
            dataset_name = parts[-1]
            cache_dir = Path.home() / ".cache" / "dapidl" / "s3_downloads" / dataset_name
            if cache_dir.exists() and any(cache_dir.iterdir()):
                logger.info(f"✓ Using S3 cache: {cache_dir}")
                return cache_dir

            # 4. Download from S3
            logger.info(f"⬇ Downloading from S3: {cfg.s3_uri}")
            return self._download_from_s3()

        # 2b. Check registry for dataset name
        if cfg.dataset_name:
            registry_path = LOCAL_DATA_REGISTRY.get(cfg.dataset_name)
            if registry_path and Path(registry_path).exists():
                logger.info(f"✓ Found local copy via registry: {registry_path}")
                return Path(registry_path)

        # 5. Download from ClearML
        if cfg.dataset_id or cfg.dataset_name:
            logger.info(f"⬇ Downloading from ClearML...")
            return self._download_dataset()

        raise ValueError(
            "No data source specified. Provide one of: "
            "local_path, s3_uri, dataset_id, or dataset_name"
        )

    def _download_dataset(self) -> Path:
        """Download ClearML Dataset."""
        from clearml import Dataset

        cfg = self.config

        if cfg.dataset_id:
            dataset = Dataset.get(dataset_id=cfg.dataset_id)
        else:
            dataset = Dataset.get(
                dataset_project=cfg.dataset_project,
                dataset_name=cfg.dataset_name,
                dataset_version=cfg.dataset_version,
            )

        # Download to local cache
        data_path = Path(dataset.get_local_copy())
        logger.info(f"Downloaded dataset to: {data_path}")

        return data_path

    def _download_from_s3(self) -> Path:
        """Download data from S3 URI.

        Uses AWS CLI with iDrive e2 configuration from CLAUDE.md.
        Downloads to a local cache directory.
        """
        import subprocess
        import tempfile
        import os

        cfg = self.config
        s3_uri = cfg.s3_uri

        # Parse S3 URI to get dataset name for cache dir
        # s3://dapidl/raw-data/xenium-breast/ -> xenium-breast
        parts = s3_uri.rstrip("/").split("/")
        dataset_name = parts[-1]

        # Use persistent cache directory
        cache_dir = Path.home() / ".cache" / "dapidl" / "s3_downloads" / dataset_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded (simple existence check)
        if any(cache_dir.iterdir()):
            logger.info(f"Using cached S3 download: {cache_dir}")
            return cache_dir

        logger.info(f"Downloading from S3: {s3_uri}")

        # S3 configuration for iDrive e2
        env = os.environ.copy()
        env["AWS_ACCESS_KEY_ID"] = "evkizOGyflbhx5uSi4oV"
        env["AWS_SECRET_ACCESS_KEY"] = "zHoIBfkh2qgKub9c2R5rgmD0ISfSJDDQQ55cZkk9"

        # Download using AWS CLI
        cmd = [
            "aws", "s3", "sync",
            s3_uri, str(cache_dir),
            "--endpoint-url", "https://s3.eu-central-2.idrivee2.com",
            "--region", "eu-central-2",
        ]

        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
            logger.info(f"Downloaded to: {cache_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"S3 download failed: {e.stderr}")
            raise RuntimeError(f"Failed to download from S3: {e.stderr}") from e

        return cache_dir

    def _resolve_data_path(self, data_path: Path) -> Path:
        """Resolve the effective data path, handling outs/ subdirectory.

        Xenium datasets can be structured as:
        - /path/to/dataset/morphology_focus.ome.tif (files at root)
        - /path/to/dataset/outs/morphology_focus.ome.tif (files in outs/)

        This method returns the path containing the actual data files.
        """
        # Check for Xenium markers at root
        if (data_path / "morphology_focus.ome.tif").exists():
            return data_path
        if (data_path / "morphology.ome.tif").exists():
            return data_path

        # Check for MERSCOPE markers at root
        if (data_path / "images").exists():
            return data_path

        # Check outs/ subdirectory (common Xenium structure)
        outs_path = data_path / "outs"
        if outs_path.exists():
            if (outs_path / "morphology_focus.ome.tif").exists():
                logger.info("Found Xenium data in outs/ subdirectory")
                return outs_path
            if (outs_path / "morphology.ome.tif").exists():
                logger.info("Found Xenium data in outs/ subdirectory")
                return outs_path

        # Check output/ subdirectory (alternative structure)
        output_path = data_path / "output"
        if output_path.exists():
            if (output_path / "morphology_focus.ome.tif").exists():
                logger.info("Found Xenium data in output/ subdirectory")
                return output_path

        # Return original path, let _detect_platform raise error if needed
        return data_path

    def _detect_platform(self, data_path: Path) -> str:
        """Auto-detect platform from file structure."""
        # Xenium markers
        if (data_path / "morphology_focus.ome.tif").exists():
            return "xenium"
        if (data_path / "morphology.ome.tif").exists():
            return "xenium"

        # MERSCOPE markers
        images_dir = data_path / "images"
        if images_dir.exists():
            dapi_files = list(images_dir.glob("mosaic_DAPI_z*.tif"))
            if dapi_files:
                return "merscope"

        raise ValueError(
            f"Could not detect platform from {data_path}. "
            "Expected Xenium (morphology*.ome.tif) or MERSCOPE (images/mosaic_DAPI_z*.tif)"
        )

    def _validate_files(self, data_path: Path, platform: str) -> None:
        """Validate required files exist."""
        if platform == "xenium":
            required = [
                ("morphology_focus.ome.tif", "morphology.ome.tif"),  # Either
                ("cells.parquet",),
            ]
        else:  # merscope
            required = [
                ("images",),
                ("cell_metadata.csv", "cell_metadata_fov.csv"),  # Either
            ]

        for alternatives in required:
            found = any((data_path / f).exists() for f in alternatives)
            if not found:
                raise FileNotFoundError(
                    f"Missing required file: {' or '.join(alternatives)} in {data_path}"
                )

        logger.info("All required files validated")

    def _get_paths(self, data_path: Path, platform: str) -> tuple[Path | None, Path | None]:
        """Get paths to cells and expression data."""
        if platform == "xenium":
            cells_path = data_path / "cells.parquet"
            if not cells_path.exists():
                cells_path = None

            # Expression matrix location
            expression_path = data_path / "cell_feature_matrix.h5"
            if not expression_path.exists():
                expression_path = data_path / "cell_feature_matrix"
                if not expression_path.exists():
                    expression_path = None

            return cells_path, expression_path

        else:  # merscope
            # Cell metadata
            cells_path = data_path / "cell_metadata.csv"
            if not cells_path.exists():
                # Try FOV-level metadata
                cells_path = list(data_path.glob("cell_metadata*.csv"))
                cells_path = cells_path[0] if cells_path else None

            # Expression (cell by gene)
            expression_path = data_path / "cell_by_gene.csv"
            if not expression_path.exists():
                expression_path = list(data_path.glob("cell_by_gene*.csv"))
                expression_path = expression_path[0] if expression_path else None

            return cells_path, expression_path

    def _extract_metadata(self, data_path: Path, platform: str) -> dict[str, Any]:
        """Extract sample metadata."""
        metadata = {
            "platform": platform,
            "data_path": str(data_path),
            "sample_name": data_path.name,
        }

        if platform == "xenium":
            # Try to read experiment.xenium for more metadata
            experiment_file = data_path / "experiment.xenium"
            if experiment_file.exists():
                try:
                    import json

                    with open(experiment_file) as f:
                        exp_data = json.load(f)
                    metadata["xenium_version"] = exp_data.get("analysis_sw_version")
                    metadata["panel"] = exp_data.get("panel_name")
                except Exception as e:
                    logger.warning(f"Could not parse experiment.xenium: {e}")

        return metadata

    def create_clearml_task(
        self,
        project: str = "DAPIDL/pipeline",
        task_name: str | None = None,
    ):
        """Create ClearML Task for this step.

        Args:
            project: ClearML project name
            task_name: Task name (default: step name)

        Returns:
            ClearML Task instance
        """
        from pathlib import Path

        from clearml import Task

        task_name = task_name or f"step-{self.name}"

        # Use the runner script for remote execution (avoids uv entry point issues)
        # Path: src/dapidl/pipeline/steps -> 5 parents to reach repo root
        runner_script = Path(__file__).parent.parent.parent.parent.parent / "scripts" / "clearml_step_runner.py"

        self._task = Task.create(
            project_name=project,
            task_name=task_name,
            task_type=Task.TaskTypes.data_processing,
            script=str(runner_script),
            argparse_args=[f"--step={self.name}"],
            add_task_init_call=False,
            # Install dapidl from the cloned repo
            packages=["-e ."],
        )

        # Connect parameters for UI editing
        # step_name is used by clearml_step_runner.py to identify which step to run
        params = {
            "step_name": self.name,
            "dataset_id": self.config.dataset_id or "",
            "dataset_project": self.config.dataset_project,
            "dataset_name": self.config.dataset_name or "",
            "platform": self.config.platform,
            "local_path": self.config.local_path or "",
        }
        self._task.set_parameters(params, __parameters_prefix="step_config")

        return self._task
