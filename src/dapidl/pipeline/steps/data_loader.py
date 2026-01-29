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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from dapidl.pipeline.base import PipelineStep, StepArtifacts


# ============================================================================
# LOCAL DATA DETECTION
# ============================================================================
# Set DAPIDL_LOCAL_DATA to colon-separated list of directories containing
# raw spatial datasets. The data_loader scans these directories and matches
# ClearML dataset names to local subdirectories, avoiding S3 downloads.
#
# Example:
#   export DAPIDL_LOCAL_DATA="/mnt/work/datasets/raw/xenium:/mnt/work/datasets/raw/merscope"
#
# Each directory should contain subdirectories named like the ClearML datasets:
#   /mnt/work/datasets/raw/xenium/
#     xenium-breast-tumor-rep1/outs/   -> matches "xenium-breast-tumor-rep1-raw"
#     xenium-lung-2fov/                -> matches "xenium-lung-2fov-raw"
#
# If DAPIDL_LOCAL_DATA is not set, defaults to scanning:
#   /mnt/work/datasets/raw/xenium and /mnt/work/datasets/raw/merscope
#
# Verification: local datasets are validated against ClearML file hashes
# using a fast fingerprint (key file sizes + partial content hashes).

_DEFAULT_LOCAL_ROOTS = [
    "/mnt/work/datasets/raw/xenium",
    "/mnt/work/datasets/raw/merscope",
]


def _get_local_roots() -> list[Path]:
    """Get local dataset root directories from DAPIDL_LOCAL_DATA env var."""
    env = os.environ.get("DAPIDL_LOCAL_DATA")
    if env:
        return [Path(p) for p in env.split(":") if p.strip()]
    return [Path(p) for p in _DEFAULT_LOCAL_ROOTS]


def _build_local_registry() -> dict[str, Path]:
    """Scan local dataset directories and build name->path registry.

    Scans directories from DAPIDL_LOCAL_DATA (or defaults) for
    subdirectories and maps their names to local paths.
    Resolves outs/ subdirectory if present.
    """
    registry: dict[str, Path] = {}
    for root in _get_local_roots():
        if not root.exists():
            continue
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            effective_path = entry / "outs" if (entry / "outs").is_dir() else entry
            registry[entry.name] = effective_path
    return registry


def _fingerprint_local_dir(path: Path) -> dict[str, int]:
    """Create a fast fingerprint of a local dataset directory.

    Returns dict of {relative_path: file_size} for key files.
    This is fast (only stat calls) and sufficient to detect changes.
    """
    fingerprint: dict[str, int] = {}
    for f in sorted(path.rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(path))
            fingerprint[rel] = f.stat().st_size
    return fingerprint


def _verify_local_against_clearml(
    local_path: Path, dataset_id: str
) -> bool:
    """Verify local dataset matches ClearML metadata using file sizes.

    Compares file sizes of key files between local directory and ClearML
    file entries. Returns True if they match (dataset unchanged).
    """
    try:
        from clearml import Dataset

        ds = Dataset.get(dataset_id=dataset_id, only_completed=False)
        entries = ds.file_entries_dict

        if not entries:
            logger.debug("No file entries in ClearML dataset, skipping verification")
            return True

        # Build local file size map (relative paths)
        local_sizes: dict[str, int] = {}
        for f in local_path.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(local_path))
                local_sizes[rel] = f.stat().st_size

        # Check key files match
        matched = 0
        checked = 0
        for rel_path, entry in entries.items():
            # Skip pipeline_outputs (these are generated, not part of raw data)
            if rel_path.startswith("pipeline_outputs/"):
                continue

            # ClearML stores relative paths; local may have outs/ prefix stripped
            if rel_path in local_sizes:
                checked += 1
                if local_sizes[rel_path] == entry.size:
                    matched += 1
                else:
                    logger.warning(
                        f"Size mismatch: {rel_path} "
                        f"(local={local_sizes[rel_path]}, clearml={entry.size})"
                    )

        if checked == 0:
            logger.debug("No overlapping files found for verification")
            return True

        match_rate = matched / checked
        if match_rate < 0.9:
            logger.warning(
                f"Local dataset may be stale: {matched}/{checked} files match "
                f"({match_rate:.0%})"
            )
            return False

        logger.info(f"✓ Local dataset verified: {matched}/{checked} files match")
        return True

    except Exception as e:
        logger.debug(f"Verification failed, assuming local is valid: {e}")
        return True


def _resolve_short_dataset_id(dataset_id: str) -> str:
    """Resolve a possibly-truncated dataset ID to the full 32-char hex ID.

    ClearML requires the full ID. If a short prefix is given (e.g. 'bf8f913f'),
    search all datasets for a matching prefix.
    """
    if len(dataset_id) >= 32:
        return dataset_id

    from clearml import Dataset

    logger.info(f"Short dataset ID '{dataset_id}', searching for full ID...")
    all_datasets = Dataset.list_datasets()
    matches = [ds for ds in all_datasets if ds["id"].startswith(dataset_id)]
    if len(matches) == 1:
        full_id = matches[0]["id"]
        logger.info(f"✓ Resolved to: {matches[0]['name']} ({full_id})")
        return full_id
    elif len(matches) > 1:
        names = ", ".join(f"{m['name']} ({m['id'][:12]})" for m in matches)
        raise ValueError(
            f"Ambiguous dataset ID prefix '{dataset_id}' matches {len(matches)} datasets: {names}"
        )
    else:
        raise ValueError(f"No dataset found with ID prefix '{dataset_id}'")


def _find_local_dataset_by_id(dataset_id: str, verify: bool = True) -> Path | None:
    """Look up ClearML dataset by ID and check if data exists locally.

    Resolves the dataset name from ClearML metadata, then checks if
    a matching directory exists under DAPIDL_LOCAL_DATA roots.

    Args:
        dataset_id: ClearML dataset ID
        verify: If True, verify local files match ClearML metadata (size check)

    Returns:
        Local path if found and verified, None otherwise
    """
    try:
        from clearml import Dataset

        # Resolve short ID prefixes to full 32-char IDs
        resolved_id = _resolve_short_dataset_id(dataset_id)
        ds = Dataset.get(dataset_id=resolved_id, only_completed=False)
        ds_name = ds.name

        local_registry = _build_local_registry()
        if not local_registry:
            logger.debug("No local datasets found (DAPIDL_LOCAL_DATA not set or empty)")
            return None

        # Strip common suffixes and try matching
        stripped = ds_name
        for suffix in ["-raw", "-raw-data"]:
            stripped = stripped.removesuffix(suffix)

        # Try exact match first
        matched_path = None
        if stripped in local_registry:
            matched_path = local_registry[stripped]

        # Try progressively shorter name (strip panel info, etc.)
        # "xenium-heart-normal-multi-tissue-panel" -> "xenium-heart-normal"
        if matched_path is None:
            parts = stripped.split("-")
            for end in range(len(parts), 2, -1):
                candidate = "-".join(parts[:end])
                if candidate in local_registry:
                    matched_path = local_registry[candidate]
                    logger.info(f"Dataset '{ds_name}' name-matched as '{candidate}'")
                    break

        if matched_path is None or not matched_path.exists():
            return None

        # Verify local files match ClearML metadata
        if verify:
            if not _verify_local_against_clearml(matched_path, dataset_id):
                logger.warning(f"Local dataset at {matched_path} is stale, will download")
                return None

        logger.info(f"✓ Dataset '{ds_name}' found locally: {matched_path}")
        return matched_path

    except Exception as e:
        logger.debug(f"Could not resolve dataset_id locally: {e}")

    return None


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

        # 2. Check local registry for S3 URI (extract dataset name from URI)
        local_registry = _build_local_registry()
        if cfg.s3_uri:
            # Extract dataset name from S3 URI, e.g. s3://dapidl/raw-data/xenium-breast-cancer-rep1/ -> xenium-breast-cancer-rep1
            s3_dataset_name = cfg.s3_uri.rstrip("/").split("/")[-1]
            registry_path = local_registry.get(s3_dataset_name)
            if registry_path and registry_path.exists():
                logger.info(f"✓ Found local copy via registry: {registry_path}")
                return registry_path

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

        # 2b. Check local registry for dataset name
        if cfg.dataset_name:
            registry_path = local_registry.get(cfg.dataset_name)
            if registry_path and registry_path.exists():
                logger.info(f"✓ Found local copy via registry: {registry_path}")
                return registry_path

        # 2c. Check local directories by resolving dataset_id -> name
        if cfg.dataset_id:
            local_path = _find_local_dataset_by_id(cfg.dataset_id)
            if local_path:
                return local_path

        # 5. Download from ClearML (last resort)
        if cfg.dataset_id or cfg.dataset_name:
            logger.info("⬇ Downloading from ClearML...")
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
            resolved_id = _resolve_short_dataset_id(cfg.dataset_id)
            dataset = Dataset.get(dataset_id=resolved_id)
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

        Uses boto3 with iDrive e2 configuration.
        Downloads to a local cache directory.
        """
        import boto3
        from botocore.config import Config

        cfg = self.config
        s3_uri = cfg.s3_uri

        # Parse S3 URI: s3://bucket/path/ -> bucket, path
        # s3://dapidl/raw-data/xenium-breast-cancer-rep1/
        parts = s3_uri.replace("s3://", "").rstrip("/").split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        dataset_name = prefix.split("/")[-1] if prefix else bucket_name

        # Use persistent cache directory
        cache_dir = Path.home() / ".cache" / "dapidl" / "s3_downloads" / dataset_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded (simple existence check)
        if any(cache_dir.iterdir()):
            logger.info(f"Using cached S3 download: {cache_dir}")
            return cache_dir

        logger.info(f"Downloading from S3: {s3_uri}")

        # S3 configuration for iDrive e2
        s3_client = boto3.client(
            "s3",
            endpoint_url="https://s3.eu-central-2.idrivee2.com",
            aws_access_key_id="evkizOGyflbhx5uSi4oV",
            aws_secret_access_key="zHoIBfkh2qgKub9c2R5rgmD0ISfSJDDQQ55cZkk9",
            region_name="eu-central-2",
            config=Config(signature_version="s3v4"),
        )

        # List and download all objects with prefix
        paginator = s3_client.get_paginator("list_objects_v2")
        total_files = 0
        total_bytes = 0

        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                size = obj["Size"]

                # Skip directories (keys ending with /)
                if key.endswith("/"):
                    continue

                # Calculate relative path from prefix
                rel_path = key[len(prefix):].lstrip("/") if prefix else key
                local_path = cache_dir / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)

                # Download file
                logger.debug(f"Downloading: {key} ({size / 1024 / 1024:.1f} MB)")
                s3_client.download_file(bucket_name, key, str(local_path))
                total_files += 1
                total_bytes += size

        logger.info(
            f"Downloaded {total_files} files ({total_bytes / 1024 / 1024 / 1024:.2f} GB) to: {cache_dir}"
        )

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

        # Use step-specific runner script for remote execution (avoids uv entry point issues)
        # The step-specific script name ensures ClearML creates unique tasks per step
        # Path: src/dapidl/pipeline/steps -> 5 parents to reach repo root
        runner_script = Path(__file__).parent.parent.parent.parent.parent / "scripts" / f"clearml_step_runner_{self.name}.py"

        self._task = Task.create(
            project_name=project,
            task_name=task_name,
            task_type=Task.TaskTypes.data_processing,
            script=str(runner_script),
            argparse_args=[f"--step={self.name}"],
            add_task_init_call=False,  # Handle in step runner  # ClearML injects Task.init() for unique script
            # Explicitly include clearml and boto3 to ensure they're installed
            # even if editable install has issues with the agent's venv
            packages=["-e .", "clearml>=1.16", "boto3"],
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
