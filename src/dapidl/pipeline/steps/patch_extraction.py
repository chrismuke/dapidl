"""Patch Extraction Pipeline Step.

Step 4: Extract nucleus-centered LMDB patches for training.

This step:
1. Joins annotations with cell centroids
2. Extracts patches around each annotated nucleus
3. Saves to LMDB format for fast training I/O
4. Creates ClearML Dataset for versioning
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from dapidl.pipeline.base import PipelineStep, StepArtifacts, get_pipeline_output_dir, resolve_artifact_path


@dataclass
class PatchExtractionConfig:
    """Configuration for patch extraction step."""

    # Patch parameters
    patch_size: int = 128  # 32, 64, 128, 256
    output_format: str = "lmdb"  # "lmdb" or "zarr"

    # Physical-space normalization (for cross-platform compatibility)
    # If enabled, patches cover the same physical area regardless of source platform
    normalize_physical_size: bool = False  # Enable physical-space normalization
    target_pixel_size_um: float = 0.2125  # Xenium default (0.2125 µm/px)
    source_pixel_size_um: float = 0.0  # 0 = auto-detect from platform

    # Segmentation source
    use_segmented_boundaries: bool = True
    use_cellpose_centroids: bool = True

    # Normalization
    normalize: bool = True
    normalization_method: str = "adaptive"  # "adaptive", "percentile", "minmax"
    percentile_low: float = 1.0
    percentile_high: float = 99.0

    # Filtering
    min_confidence: float = 0.0
    exclude_edge_cells: bool = True
    edge_margin_px: int = 64

    # ClearML Dataset
    create_dataset: bool = True
    dataset_project: str = "DAPIDL/datasets"
    dataset_name: str | None = None  # Auto-generate if None

    # S3 settings
    upload_to_s3: bool = True
    s3_bucket: str = "dapidl"

    # Caching
    skip_if_exists: bool = True


class PatchExtractionStep(PipelineStep):
    """Extract nucleus-centered patches for training.

    Creates LMDB database with patches and labels, suitable for
    NVIDIA DALI fast loading.

    Queue: default (CPU - I/O bound)
    """

    name = "patch_extraction"
    queue = "default"  # CPU queue (I/O bound)

    def __init__(self, config: PatchExtractionConfig | None = None):
        """Initialize patch extraction step.

        Args:
            config: Patch extraction configuration
        """
        self.config = config or PatchExtractionConfig()
        self._task = None

    def get_parameter_schema(self) -> dict[str, Any]:
        """Return JSON schema for ClearML UI parameters."""
        return {
            "type": "object",
            "properties": {
                "patch_size": {
                    "type": "integer",
                    "enum": [32, 64, 128, 256],
                    "default": 128,
                    "description": "Patch size in pixels",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["lmdb", "zarr"],
                    "default": "lmdb",
                    "description": "Output format (lmdb for DALI)",
                },
                "use_segmented_boundaries": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use Cellpose vs native boundaries",
                },
                "normalization_method": {
                    "type": "string",
                    "enum": ["adaptive", "percentile", "minmax"],
                    "default": "adaptive",
                    "description": "Image normalization method",
                },
                "exclude_edge_cells": {
                    "type": "boolean",
                    "default": True,
                    "description": "Exclude cells near image edge",
                },
                "create_dataset": {
                    "type": "boolean",
                    "default": True,
                    "description": "Create ClearML Dataset",
                },
            },
        }

    def validate_inputs(self, artifacts: StepArtifacts) -> bool:
        """Validate step inputs.

        Requires:
        - data_path (DAPI image)
        - annotations_parquet (cell types)
        - Either cells_parquet or centroids_parquet (coordinates)
        """
        outputs = artifacts.outputs
        required = ["data_path", "annotations_parquet"]

        if not all(key in outputs for key in required):
            return False

        # Need either platform cells or segmentation centroids
        return "cells_parquet" in outputs or "centroids_parquet" in outputs

    def execute(self, artifacts: StepArtifacts) -> StepArtifacts:
        """Execute patch extraction step.

        Args:
            artifacts: Input artifacts from previous steps

        Returns:
            Output artifacts containing:
            - patches_path: Path to LMDB/Zarr dataset
            - dataset_id: ClearML Dataset ID (if created)
            - extraction_stats: Dict with statistics
        """
        cfg = self.config
        inputs = artifacts.outputs

        # Resolve artifact URLs to local paths
        data_path = resolve_artifact_path(inputs["data_path"], "data_path")
        if data_path is None:
            raise ValueError("data_path artifact is required")

        # Platform can be a URL to a text file or a direct value
        platform_value = inputs.get("platform", "xenium")
        platform_path = resolve_artifact_path(platform_value, "platform")
        if platform_path and platform_path.exists() and platform_path.is_file():
            platform = platform_path.read_text().strip()
            logger.info(f"Read platform from artifact: {platform}")
        else:
            platform = str(platform_value)

        # Check for existing outputs (skip if exists)
        output_dir = get_pipeline_output_dir("patches", data_path)
        if cfg.output_format == "lmdb":
            patches_path = output_dir / "patches.lmdb"
        else:
            patches_path = output_dir / "patches.zarr"
        metadata_path = output_dir / "metadata.parquet"
        config_path = output_dir / "config.json"

        if cfg.skip_if_exists and patches_path.exists() and metadata_path.exists():
            # Validate config matches (if config file exists)
            import json

            config_matches = True
            if config_path.exists():
                with open(config_path) as f:
                    saved_config = json.load(f)
                # Check key patch extraction parameters
                config_matches = (
                    saved_config.get("patch_size") == cfg.patch_size
                    and saved_config.get("output_format") == cfg.output_format
                    and saved_config.get("normalize_physical_size") == cfg.normalize_physical_size
                    and saved_config.get("exclude_edge_cells") == cfg.exclude_edge_cells
                )
                if not config_matches:
                    logger.info("Config mismatch - re-running patch extraction")

            if config_matches:
                logger.info(f"Skipping patch extraction - outputs already exist at {output_dir}")
                existing_class_mapping = inputs.get("class_mapping", {})
                # Derive additional outputs for training step
                num_classes = len(existing_class_mapping) if existing_class_mapping else 3
                class_names = [k for k, v in sorted(existing_class_mapping.items(), key=lambda x: x[1])] if existing_class_mapping else ["Epithelial", "Immune", "Stromal"]
                index_to_class = {v: k for k, v in existing_class_mapping.items()} if existing_class_mapping else {0: "Epithelial", 1: "Immune", 2: "Stromal"}
                return StepArtifacts(
                    inputs=inputs,
                    outputs={
                        **inputs,
                        "patches_path": str(patches_path),
                        "metadata_parquet": str(metadata_path),
                        "dataset_id": None,
                        "extraction_stats": {"skipped": True, "reason": "outputs_exist"},
                        "class_mapping": existing_class_mapping,
                        "num_classes": num_classes,
                        "class_names": class_names,
                        "index_to_class": index_to_class,
                    },
                )

        # Load DAPI image
        dapi_image = self._load_dapi_image(data_path, platform)
        logger.info(f"Loaded DAPI: {dapi_image.shape}, dtype={dapi_image.dtype}")

        # Compute normalization parameters (DON'T convert whole image to float32!)
        # For large images, use lazy normalization - normalize patches on-the-fly
        h, w = dapi_image.shape
        use_lazy_norm = h * w > 100_000_000  # > 100M pixels
        norm_params = None
        if cfg.normalize:
            norm_params = self._compute_normalization_params(dapi_image, cfg)
            if use_lazy_norm:
                logger.info(f"Using LAZY normalization for {h * w / 1e9:.2f}B pixel image (saves ~{h * w * 4 / 1e9:.1f}GB)")
            else:
                # Small image - pre-normalize for speed
                p_low, p_high = norm_params
                dapi_image = np.clip(dapi_image, p_low, p_high).astype(np.float32)
                dapi_image = (dapi_image - p_low) / (p_high - p_low + 1e-8)
                norm_params = None  # Already normalized

        # Load annotations (resolve artifact URL)
        annotations_path = resolve_artifact_path(
            inputs["annotations_parquet"], "annotations_parquet"
        )
        if annotations_path is None:
            raise ValueError("annotations_parquet artifact is required")
        annotations_df = pl.read_parquet(annotations_path)
        logger.info(f"Loaded {annotations_df.height} annotations")

        # Load centroids (prefer segmentation output if available and valid)
        use_segmentation_centroids = False
        centroids_df: pl.DataFrame | None = None
        if cfg.use_cellpose_centroids and inputs.get("centroids_parquet"):
            # Check if Cellpose matching succeeded (match_rate > 0)
            matching_stats = inputs.get("matching_stats", {})
            match_rate = matching_stats.get("match_rate", 0.0) if matching_stats else 0.0

            if match_rate > 0:
                # Use segmentation output centroids
                centroids_path = resolve_artifact_path(
                    inputs["centroids_parquet"], "centroids_parquet"
                )
                if centroids_path is not None:
                    centroids_df = self._load_centroids(centroids_path, platform)
                    use_segmentation_centroids = True
                    logger.info(f"Using Cellpose centroids (match_rate={match_rate:.1%})")
            else:
                logger.warning(f"Cellpose matching failed (match_rate={match_rate:.1%}), falling back to native cells")

        if not use_segmentation_centroids:
            # Fall back to native cell coordinates
            cells_path_raw = inputs.get("cells_parquet")
            if cells_path_raw:
                cells_path = resolve_artifact_path(cells_path_raw, "cells_parquet")
                if cells_path is not None:
                    centroids_df = self._load_centroids(cells_path, platform)
                    logger.info("Using native cell coordinates")
                else:
                    raise ValueError("cells_parquet path could not be resolved")
            else:
                raise ValueError("No cell coordinates available")

        if centroids_df is None:
            raise ValueError("Failed to load cell centroids")

        logger.info(f"Loaded {centroids_df.height} cell centroids")

        # Join annotations with centroids
        merged_df = self._merge_data(annotations_df, centroids_df)
        logger.info(f"Merged: {merged_df.height} annotated cells with coordinates")

        # Filter edge cells
        if cfg.exclude_edge_cells:
            merged_df = self._filter_edge_cells(
                merged_df, dapi_image.shape, cfg.edge_margin_px, cfg.patch_size
            )

        # Filter by confidence
        if cfg.min_confidence > 0 and "confidence" in merged_df.columns:
            merged_df = merged_df.filter(pl.col("confidence") >= cfg.min_confidence)

        # Create class label column
        class_mapping = inputs.get("class_mapping", {})
        merged_df = self._add_class_labels(merged_df, class_mapping)

        # Extract patches
        output_dir = get_pipeline_output_dir("patches", data_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine pixel sizes for physical-space normalization
        source_pixel_size = 0.0
        target_pixel_size = 0.0
        if cfg.normalize_physical_size:
            source_pixel_size = (
                cfg.source_pixel_size_um
                if cfg.source_pixel_size_um > 0
                else self._get_platform_pixel_size(platform)
            )
            target_pixel_size = cfg.target_pixel_size_um
            logger.info(
                f"Physical-space normalization enabled: "
                f"source={source_pixel_size} µm/px, target={target_pixel_size} µm/px"
            )

        if cfg.output_format == "lmdb":
            patches_path = output_dir / "patches.lmdb"
            stats = self._extract_to_lmdb(
                dapi_image, merged_df, patches_path, cfg.patch_size,
                source_pixel_size=source_pixel_size,
                target_pixel_size=target_pixel_size,
                norm_params=norm_params,  # Pass for lazy normalization
            )
        else:
            patches_path = output_dir / "patches.zarr"
            stats = self._extract_to_zarr(
                dapi_image, merged_df, patches_path, cfg.patch_size
            )

        # Save metadata
        metadata_path = output_dir / "metadata.parquet"
        merged_df.write_parquet(metadata_path)

        # Create ClearML Dataset
        dataset_id = None
        if cfg.create_dataset:
            dataset_id = self._create_clearml_dataset(
                output_dir, cfg, inputs, stats
            )

        # Save config for cache validation
        import json

        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "patch_size": cfg.patch_size,
                    "output_format": cfg.output_format,
                    "normalize_physical_size": cfg.normalize_physical_size,
                    "exclude_edge_cells": cfg.exclude_edge_cells,
                    "target_pixel_size_um": cfg.target_pixel_size_um,
                },
                f,
                indent=2,
            )

        # Derive additional outputs for training step
        num_classes = len(class_mapping) if class_mapping else 3
        class_names = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])] if class_mapping else ["Epithelial", "Immune", "Stromal"]
        index_to_class = {v: k for k, v in class_mapping.items()} if class_mapping else {0: "Epithelial", 1: "Immune", 2: "Stromal"}

        return StepArtifacts(
            inputs=inputs,
            outputs={
                **inputs,
                "patches_path": str(patches_path),
                "metadata_parquet": str(metadata_path),
                "dataset_id": dataset_id,
                "extraction_stats": stats,
                "class_mapping": class_mapping,
                "num_classes": num_classes,
                "class_names": class_names,
                "index_to_class": index_to_class,
            },
        )

    def _load_dapi_image(self, data_path: Path, platform: str) -> np.ndarray:
        """Load DAPI image.

        For Xenium, handles multiple file formats:
        1. morphology_focus.ome.tif - single best-focus image (preferred)
        2. morphology_focus/ directory - tiled focus images (Xenium Prime/IO)
        3. morphology.ome.tif - raw z-stack (fallback, uses middle z-plane)
        """
        import tifffile

        if platform == "xenium":
            # Option 1: Single focus image (standard Xenium)
            focus_file = data_path / "morphology_focus.ome.tif"
            if focus_file.exists():
                logger.info(f"Loading single focus file: {focus_file.name}")
                image = tifffile.imread(focus_file)
                if image.ndim == 3:
                    image = image[0]
                return image

            # Option 2: Tiled focus images (Xenium Prime/IO)
            focus_dir = data_path / "morphology_focus"
            if focus_dir.exists():
                tiled_files = sorted(focus_dir.glob("morphology_focus_*.ome.tif"))
                if tiled_files:
                    # For single tile, use it directly
                    if len(tiled_files) == 1:
                        logger.info(f"Loading single tiled focus image: {tiled_files[0].name}")
                        image = tifffile.imread(tiled_files[0])
                        if image.ndim == 3:
                            image = image[0]
                        return image
                    # For multiple tiles, fall back to z-stack (stitching is too memory-intensive)
                    logger.info(f"Found {len(tiled_files)} tiled focus images - using z-stack middle plane instead")

            # Option 3: Raw z-stack - use MIDDLE z-plane (not first!)
            zstack_file = data_path / "morphology.ome.tif"
            if zstack_file.exists():
                logger.info(f"Loading z-stack: {zstack_file.name}")
                image = tifffile.imread(zstack_file)
                if image.ndim == 3:
                    # Use middle z-plane where focus is typically best
                    mid_z = image.shape[0] // 2
                    logger.info(f"Using middle z-plane {mid_z} of {image.shape[0]}")
                    image = image[mid_z]
                return image

            raise FileNotFoundError(f"No DAPI image found in {data_path}")

        else:  # merscope
            images_dir = data_path / "images"
            dapi_files = sorted(images_dir.glob("mosaic_DAPI_z*.tif"))
            mid_idx = len(dapi_files) // 2
            return tifffile.imread(dapi_files[mid_idx])

    def _compute_normalization_params(
        self, image: np.ndarray, cfg: PatchExtractionConfig
    ) -> tuple[float, float]:
        """Compute normalization parameters without converting image.

        Returns:
            Tuple of (p_low, p_high) percentile values for normalization.
        """
        h, w = image.shape
        n_pixels = h * w

        # For large images, use sampled percentile estimation (same as Cellpose fix)
        if n_pixels > 100_000_000:  # > 100M pixels
            sample_size = min(1_000_000, n_pixels)
            rng = np.random.default_rng(42)  # Fixed seed for reproducibility
            flat_indices = rng.choice(n_pixels, size=sample_size, replace=False)
            sample = image.flat[flat_indices].astype(np.float32)
            logger.info(f"Computing normalization params from {sample_size:,} samples ({n_pixels:,} total pixels)")
        else:
            sample = image.astype(np.float32)

        if cfg.normalization_method == "adaptive":
            p_low = float(np.percentile(sample, cfg.percentile_low))
            p_high = float(np.percentile(sample, cfg.percentile_high))
        elif cfg.normalization_method == "percentile":
            p_low = float(np.percentile(sample, cfg.percentile_low))
            p_high = float(np.percentile(sample, cfg.percentile_high))
        else:  # minmax
            p_low = float(image.min())
            p_high = float(image.max())

        logger.info(f"Normalization params: p_low={p_low:.1f}, p_high={p_high:.1f}")
        return p_low, p_high

    def _normalize_patch(
        self, patch: np.ndarray, p_low: float, p_high: float
    ) -> np.ndarray:
        """Normalize a single patch to 0-1 range.

        Args:
            patch: Raw uint16 patch
            p_low: Lower percentile value
            p_high: Upper percentile value

        Returns:
            Normalized float32 patch (0-1 range)
        """
        patch_f = patch.astype(np.float32)
        patch_f = np.clip(patch_f, p_low, p_high)
        patch_f = (patch_f - p_low) / (p_high - p_low + 1e-8)
        return patch_f

    def _normalize_image(
        self, image: np.ndarray, cfg: PatchExtractionConfig
    ) -> np.ndarray:
        """Normalize image to 0-1 range.

        WARNING: This method is DEPRECATED for large images. Use
        _compute_normalization_params() + _normalize_patch() instead.
        """
        p_low, p_high = self._compute_normalization_params(image, cfg)

        # Clip and scale - ONLY for small images!
        h, w = image.shape
        if h * w > 100_000_000:
            logger.warning("_normalize_image called on large image - consider using lazy normalization")

        image = np.clip(image, p_low, p_high)
        image = (image - p_low) / (p_high - p_low + 1e-8)

        return image.astype(np.float32)

    def _load_centroids(self, cells_path: Path, platform: str) -> pl.DataFrame:
        """Load cell centroids.

        Handles multiple input formats:
        - Pipeline segmentation output: centroid_x_um, centroid_y_um (in microns)
        - Raw Xenium cells.parquet: x_centroid, y_centroid (in pixels)
        - Raw MERSCOPE: EntityID, center_x, center_y (in microns)
        """
        if cells_path.suffix == ".parquet":
            df = pl.read_parquet(cells_path)
        else:
            df = pl.read_csv(cells_path)

        cols = df.columns

        # Pipeline segmentation output (in microns)
        if "centroid_x_um" in cols and "centroid_y_um" in cols:
            pixel_size = 0.2125 if platform == "xenium" else 0.108
            return df.select([
                pl.col("cell_id").cast(pl.Utf8),
                (pl.col("centroid_x_um") / pixel_size).alias("x"),
                (pl.col("centroid_y_um") / pixel_size).alias("y"),
            ])

        # Raw Xenium cells.parquet (in MICRONS, need conversion to pixels)
        # Note: Xenium coordinates are in microns, not pixels!
        if "x_centroid" in cols and "y_centroid" in cols:
            pixel_size = 0.2125  # Xenium pixel size (µm/px)
            logger.info(f"Converting Xenium coordinates from microns to pixels (÷{pixel_size})")
            return df.select([
                pl.col("cell_id").cast(pl.Utf8),
                (pl.col("x_centroid") / pixel_size).alias("x"),
                (pl.col("y_centroid") / pixel_size).alias("y"),
            ])

        # Raw MERSCOPE (in microns) - handles both EntityID and unnamed first column
        if "center_x" in cols and "center_y" in cols:
            pixel_size = 0.108  # MERSCOPE pixel size

            # Find cell_id column (could be 'EntityID' or '' or first column)
            if "EntityID" in cols:
                cell_id_col = "EntityID"
            elif "" in cols:
                cell_id_col = ""
            else:
                cell_id_col = cols[0]  # First column

            # Shift coordinates to make all positive (MERSCOPE can have negative coords)
            x_min = df["center_x"].min()
            y_min = df["center_y"].min()
            if x_min is None:
                x_min = 0.0
            if y_min is None:
                y_min = 0.0

            logger.info(f"MERSCOPE coordinate offset: x_min={x_min:.2f}, y_min={y_min:.2f} µm")

            return df.select([
                pl.col(cell_id_col).cast(pl.Utf8).alias("cell_id"),
                ((pl.col("center_x") - x_min) / pixel_size).alias("x"),
                ((pl.col("center_y") - y_min) / pixel_size).alias("y"),
            ])

        raise ValueError(f"Unknown centroids format. Columns: {cols}")

    def _merge_data(
        self, annotations_df: pl.DataFrame, centroids_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Merge annotations with centroids."""
        # Ensure cell_id types match
        annotations_df = annotations_df.with_columns(
            pl.col("cell_id").cast(pl.Utf8)
        )
        centroids_df = centroids_df.with_columns(
            pl.col("cell_id").cast(pl.Utf8)
        )

        return annotations_df.join(centroids_df, on="cell_id", how="inner")

    def _filter_edge_cells(
        self,
        df: pl.DataFrame,
        image_shape: tuple,
        margin: int,
        patch_size: int,
    ) -> pl.DataFrame:
        """Filter cells too close to image edge."""
        half_patch = patch_size // 2
        min_coord = margin + half_patch
        max_y = image_shape[0] - margin - half_patch
        max_x = image_shape[1] - margin - half_patch

        before = df.height
        df = df.filter(
            (pl.col("x") >= min_coord)
            & (pl.col("x") < max_x)
            & (pl.col("y") >= min_coord)
            & (pl.col("y") < max_y)
        )
        logger.info(f"Edge filter: {before} -> {df.height}")
        return df

    def _add_class_labels(
        self, df: pl.DataFrame, class_mapping: dict[str, int]
    ) -> pl.DataFrame:
        """Add numeric class labels.

        Automatically detects fine-grained vs coarse mode based on class_mapping keys.
        """
        if not class_mapping:
            # Default coarse mapping
            class_mapping = {"Epithelial": 0, "Immune": 1, "Stromal": 2}

        # Detect if we're in fine-grained mode by checking if class_mapping
        # contains broad categories or fine-grained types
        broad_categories = {"Epithelial", "Immune", "Stromal", "Endothelial", "Unknown"}
        is_fine_grained = not all(k in broad_categories for k in class_mapping.keys())

        # Choose label column based on mode
        if is_fine_grained:
            # Fine-grained: use predicted_type
            if "predicted_type" in df.columns:
                label_col = "predicted_type"
            elif "predicted_type_1" in df.columns:
                label_col = "predicted_type_1"
            else:
                logger.warning("Fine-grained mode but no predicted_type column, falling back to broad_category")
                label_col = "broad_category" if "broad_category" in df.columns else "predicted_type"
        else:
            # Coarse: use broad_category
            label_col = "broad_category" if "broad_category" in df.columns else "predicted_type"

        logger.info(f"Using label column '{label_col}' for class mapping (fine_grained={is_fine_grained})")

        df = df.with_columns(
            pl.col(label_col)
            .map_elements(
                lambda x: class_mapping.get(x, -1),
                return_dtype=pl.Int32,
            )
            .alias("label")
        )

        # Filter out unmapped classes
        before = df.height
        df = df.filter(pl.col("label") >= 0)
        if df.height < before:
            logger.info(f"Filtered unmapped classes: {before} -> {df.height}")

        return df

    def _get_platform_pixel_size(self, platform: str) -> float:
        """Get pixel size in µm for a platform."""
        pixel_sizes = {
            "xenium": 0.2125,   # Xenium: 0.2125 µm/pixel
            "merscope": 0.108,  # MERSCOPE: 0.108 µm/pixel (2x higher resolution)
        }
        return pixel_sizes.get(platform.lower(), 0.2125)

    def _extract_to_lmdb(
        self,
        image: np.ndarray,
        df: pl.DataFrame,
        output_path: Path,
        patch_size: int,
        source_pixel_size: float = 0.0,
        target_pixel_size: float = 0.0,
        norm_params: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """Extract patches to LMDB database.

        Uses numpy's native tobytes()/frombuffer() for safe serialization.

        Args:
            image: DAPI image to extract from (uint16 or float32)
            df: DataFrame with x, y, label columns
            output_path: Path to save LMDB
            patch_size: Output patch size in pixels
            source_pixel_size: Source platform pixel size (µm/px). 0 = no rescaling.
            target_pixel_size: Target pixel size (µm/px). 0 = no rescaling.
            norm_params: If provided, (p_low, p_high) for lazy normalization.
                        None means image is already normalized.
        """
        import cv2
        import lmdb

        # Check if we need lazy normalization
        use_lazy_norm = norm_params is not None
        p_low: float = 0.0
        p_high: float = 1.0
        if use_lazy_norm and norm_params is not None:
            p_low, p_high = norm_params
            logger.info(f"Using lazy patch normalization: p_low={p_low:.1f}, p_high={p_high:.1f}")

        # Calculate extraction size for physical-space normalization
        do_rescale = source_pixel_size > 0 and target_pixel_size > 0
        if do_rescale:
            target_physical_um = patch_size * target_pixel_size
            source_extract_size = int(target_physical_um / source_pixel_size)
            scale_factor = patch_size / source_extract_size
            logger.info(
                f"Physical-space normalization: extracting {source_extract_size}px "
                f"(={target_physical_um:.1f}µm) -> resize to {patch_size}px "
                f"(scale={scale_factor:.2f}x)"
            )
            half = source_extract_size // 2
        else:
            source_extract_size = patch_size
            half = patch_size // 2
        n_cells = df.height

        # Open LMDB
        env = lmdb.open(
            str(output_path),
            map_size=int(1e12),  # 1TB max
            readonly=False,
            meminit=False,
            map_async=True,
        )

        # Extract patches in batches
        batch_size = 1000
        n_extracted = 0
        class_counts = {}

        with env.begin(write=True) as txn:
            for start_idx in range(0, n_cells, batch_size):
                batch_df = df.slice(start_idx, batch_size)

                for row in batch_df.iter_rows(named=True):
                    cx = int(row["x"])
                    cy = int(row["y"])
                    label = row["label"]

                    # Extract patch (may be larger/smaller for rescaling)
                    patch = image[
                        cy - half : cy + half,
                        cx - half : cx + half,
                    ]

                    # Validate extraction size
                    if patch.shape != (source_extract_size, source_extract_size):
                        continue

                    # Lazy normalization: normalize this patch only
                    if use_lazy_norm:
                        patch = patch.astype(np.float32)
                        patch = np.clip(patch, p_low, p_high)
                        patch = (patch - p_low) / (p_high - p_low + 1e-8)

                    # Resize to target size if using physical-space normalization
                    if do_rescale:
                        # Ensure float32 for cv2.resize
                        if patch.dtype != np.float32:
                            patch = patch.astype(np.float32)
                        patch = cv2.resize(
                            patch,
                            (patch_size, patch_size),
                            interpolation=cv2.INTER_LINEAR
                        )

                    # Store in format compatible with training.py:
                    # - Key: big-endian uint64 (sequential index)
                    # - Value: int64 label (8 bytes) + uint16 patch data
                    # Normalized patches (0-1) are scaled back to uint16 for storage
                    key = struct.pack(">Q", n_extracted)

                    # Convert normalized float32 to uint16 for storage
                    if patch.dtype == np.float32:
                        # Normalize to 0-1 range was already done, scale to uint16
                        patch_uint16 = (patch * 65535).clip(0, 65535).astype(np.uint16)
                    else:
                        patch_uint16 = patch.astype(np.uint16)

                    # Pack label as int64 + patch as uint16
                    label_bytes = np.array([label], dtype=np.int64).tobytes()
                    value = label_bytes + patch_uint16.tobytes()

                    txn.put(key, value)

                    n_extracted += 1
                    class_counts[label] = class_counts.get(label, 0) + 1

                logger.info(f"Extracted {n_extracted}/{n_cells} patches")

            # Store metadata as JSON (safe serialization)
            # Note: Use both 'length' and 'n_patches' for backwards compatibility
            metadata = {
                "length": n_extracted,
                "n_patches": n_extracted,
                "patch_size": patch_size,
                "dtype": "uint16",  # Patches stored as normalized uint16 (0-65535)
                "source_dtype": str(image.dtype),  # Original image dtype
                "class_counts": {str(k): v for k, v in class_counts.items()},
                "source_pixel_size_um": source_pixel_size if do_rescale else None,
                "target_pixel_size_um": target_pixel_size if do_rescale else None,
                "physical_normalized": do_rescale,
                "lazy_normalized": use_lazy_norm,
            }
            txn.put(b"__metadata__", json.dumps(metadata).encode())

        env.close()
        msg = f"Created LMDB with {n_extracted} patches at {output_path}"
        if do_rescale:
            msg += f" (physical-normalized to {target_pixel_size} µm/px)"
        logger.info(msg)

        return {
            "n_extracted": n_extracted,
            "n_skipped": n_cells - n_extracted,
            "class_counts": class_counts,
            "patch_size": patch_size,
            "format": "lmdb",
            "physical_normalized": do_rescale,
        }

    def _extract_to_zarr(
        self,
        image: np.ndarray,
        df: pl.DataFrame,
        output_path: Path,
        patch_size: int,
    ) -> dict[str, Any]:
        """Extract patches to Zarr array."""
        import zarr

        half = patch_size // 2
        n_cells = df.height

        # Pre-allocate arrays
        patches = zarr.open(
            str(output_path / "patches.zarr"),
            mode="w",
            shape=(n_cells, patch_size, patch_size),
            chunks=(1000, patch_size, patch_size),
            dtype=np.float32,
        )

        labels = []
        cell_ids = []
        valid_indices = []

        for idx, row in enumerate(df.iter_rows(named=True)):
            cx = int(row["x"])
            cy = int(row["y"])

            # Extract patch
            patch = image[
                cy - half : cy + half,
                cx - half : cx + half,
            ]

            if patch.shape == (patch_size, patch_size):
                patches[len(valid_indices)] = patch
                labels.append(row["label"])
                cell_ids.append(row["cell_id"])
                valid_indices.append(idx)

        # Trim to actual size
        n_extracted = len(valid_indices)
        patches.resize(n_extracted, patch_size, patch_size)

        # Save labels
        np.save(output_path / "labels.npy", np.array(labels))

        # Calculate class counts
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1

        logger.info(f"Created Zarr with {n_extracted} patches at {output_path}")

        return {
            "n_extracted": n_extracted,
            "n_skipped": n_cells - n_extracted,
            "class_counts": class_counts,
            "patch_size": patch_size,
            "format": "zarr",
        }

    def _create_clearml_dataset(
        self,
        output_dir: Path,
        cfg: PatchExtractionConfig,
        inputs: dict,
        stats: dict,
    ) -> str | None:
        """Create ClearML Dataset from patches via S3.

        IMPORTANT: Never upload large data directly to ClearML.
        Pattern:
        1. Upload to S3 first
        2. Register with ClearML using add_external_files() (metadata only)
        """
        try:
            from clearml import Dataset
        except ImportError:
            logger.warning("ClearML not available, skipping dataset creation")
            return None

        # Generate name if not provided
        dataset_name = cfg.dataset_name
        if not dataset_name:
            platform = inputs.get("platform", "xenium")
            annotator = inputs.get("annotator_used", "unknown")
            patch_size = cfg.patch_size
            dataset_name = f"{platform}-{annotator}-p{patch_size}"

        # S3 path for this dataset
        s3_path = f"datasets/patches/{dataset_name}"

        try:
            if cfg.upload_to_s3:
                # Step 1: Upload to S3 first (NOT to ClearML)
                from dapidl.utils.s3 import upload_to_s3

                s3_uri = upload_to_s3(output_dir, s3_path)
                logger.info(f"Uploaded patches to S3: {s3_uri}")

                # Step 2: Register with ClearML as external reference
                dataset = Dataset.create(
                    dataset_project=cfg.dataset_project,
                    dataset_name=dataset_name,
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
                    dataset_project=cfg.dataset_project,
                    dataset_name=dataset_name,
                )
                dataset.add_external_files(
                    source_url=f"file://{output_dir}",
                    dataset_path="/",
                )

            # Metadata (this goes to ClearML - small JSON only)
            dataset.set_metadata({
                "patch_size": cfg.patch_size,
                "normalization": cfg.normalization_method,
                "n_patches": stats["n_extracted"],
                "class_counts": stats["class_counts"],
                "source_platform": inputs.get("platform"),
                "annotator": inputs.get("annotator_used"),
                "s3_uri": s3_uri,  # Store S3 location in metadata
                "local_path": str(output_dir),  # For reference
                "registration_type": "external_reference",
                "uploaded_to_clearml": False,
            })

            dataset.finalize()
            # NOTE: Do NOT call dataset.upload() - files are already on S3

            logger.info(f"Registered ClearML Dataset: {dataset.id} -> {s3_uri or output_dir}")
            return dataset.id

        except Exception as e:
            logger.warning(f"Failed to register ClearML dataset: {e}")
            return None

    def create_clearml_task(
        self,
        project: str = "DAPIDL/pipeline",
        task_name: str | None = None,
    ):
        """Create ClearML Task for this step."""
        from pathlib import Path

        from clearml import Task

        task_name = task_name or f"step-{self.name}"

        # Use the per-step runner script for remote execution
        # Per-step scripts handle task ID discovery via workers API
        # Path: src/dapidl/pipeline/steps -> 5 parents to reach repo root
        runner_script = Path(__file__).parent.parent.parent.parent.parent / "scripts" / f"clearml_step_runner_{self.name}.py"

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

        # step_name is used by clearml_step_runner.py to identify which step to run
        params = {
            "step_name": self.name,
            "patch_size": self.config.patch_size,
            "output_format": self.config.output_format,
            "normalization_method": self.config.normalization_method,
            "exclude_edge_cells": self.config.exclude_edge_cells,
            "create_dataset": self.config.create_dataset,
        }
        self._task.set_parameters(params, __parameters_prefix="step_config")

        return self._task
