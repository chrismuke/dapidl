"""LMDB Dataset Creation Pipeline Step.

Step 3: Create LMDB training dataset from annotated data.

This step:
1. Checks if matching LMDB dataset already exists (skip logic)
2. Extracts nucleus-centered patches
3. Saves to LMDB format for fast DALI loading
4. Registers as ClearML Dataset with lineage to parent
5. Uploads to S3 if configured

Key features:
- Skip logic: Avoids recreating existing datasets
- Dataset lineage: Uses parent_datasets for space efficiency
- Multi-patch-size support: Can be called for different sizes
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

from dapidl.pipeline.base import PipelineStep, StepArtifacts, resolve_artifact_path


@dataclass
class LMDBCreationConfig:
    """Configuration for LMDB dataset creation."""

    # Patch parameters
    patch_size: int = 128
    output_format: str = "lmdb"  # "lmdb" or "zarr"

    # Physical-space normalization (cross-platform compatibility)
    normalize_physical_size: bool = True
    target_pixel_size_um: float = 0.2125  # Xenium default

    # Normalization
    normalization_method: str = "adaptive"  # "adaptive", "percentile", "minmax"

    # Filtering
    min_confidence: float = 0.0
    exclude_edge_cells: bool = True
    edge_margin_px: int = 64

    # Skip logic
    skip_if_exists: bool = True
    existing_lmdb_dataset_id: str | None = None  # Pre-check

    # Dataset registration
    create_clearml_dataset: bool = True
    parent_dataset_id: str | None = None  # For lineage

    # S3 settings
    upload_to_s3: bool = True
    s3_bucket: str = "dapidl"
    s3_endpoint: str = "https://s3.eu-central-2.idrivee2.com"


class LMDBCreationStep(PipelineStep):
    """Step 3: Create LMDB Dataset from Annotated Data.

    Supports skipping if LMDB dataset already exists with matching parameters.
    Uses parent_datasets for lineage to avoid re-uploading raw data.

    Queue: default (CPU - I/O bound)
    """

    name = "lmdb_creation"
    description = "Create LMDB training dataset with skip logic"

    def __init__(self, config: LMDBCreationConfig | None = None):
        """Initialize LMDB creation step.

        Args:
            config: LMDB creation configuration
        """
        self.config = config or LMDBCreationConfig()
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
                "normalization_method": {
                    "type": "string",
                    "enum": ["adaptive", "percentile", "minmax"],
                    "default": "adaptive",
                    "description": "Image normalization method",
                },
                "normalize_physical_size": {
                    "type": "boolean",
                    "default": True,
                    "description": "Normalize to consistent physical size (cross-platform)",
                },
                "skip_if_exists": {
                    "type": "boolean",
                    "default": True,
                    "description": "Skip if matching LMDB dataset exists",
                },
                "create_clearml_dataset": {
                    "type": "boolean",
                    "default": True,
                    "description": "Register as ClearML Dataset",
                },
                "upload_to_s3": {
                    "type": "boolean",
                    "default": True,
                    "description": "Upload to S3 storage",
                },
            },
        }

    def validate_inputs(self, artifacts: StepArtifacts) -> bool:
        """Validate step inputs.

        Requires:
        - data_path (DAPI image)
        - annotations_parquet (cell types + coordinates)
        """
        outputs = artifacts.outputs
        required = ["data_path", "annotations_parquet"]
        return all(key in outputs for key in required)

    def execute(self, artifacts: StepArtifacts) -> StepArtifacts:
        """Execute LMDB dataset creation.

        Args:
            artifacts: Input artifacts from annotation step

        Returns:
            Output artifacts containing:
            - lmdb_path: Path to LMDB dataset
            - lmdb_dataset_id: ClearML Dataset ID (if registered)
            - skipped: Boolean indicating if step was skipped
            - extraction_stats: Dict with statistics
        """
        cfg = self.config
        inputs = artifacts.outputs

        # 1. Check if we should skip
        if cfg.skip_if_exists:
            existing_id = self._check_existing_lmdb(inputs, cfg)
            if existing_id:
                logger.info(f"Skipping LMDB creation - existing dataset: {existing_id}")

                # Get path from existing dataset
                lmdb_path = self._get_dataset_path(existing_id)

                return StepArtifacts(
                    inputs=inputs,
                    outputs={
                        **inputs,
                        "lmdb_path": str(lmdb_path) if lmdb_path else None,
                        "lmdb_dataset_id": existing_id,
                        "patch_size": cfg.patch_size,
                        "skipped": True,
                        "extraction_stats": {"skipped": True, "existing_dataset_id": existing_id},
                    },
                )

        # 2. Resolve input paths
        data_path = resolve_artifact_path(inputs["data_path"], "data_path")

        # Prefer CL-standardized annotations if available, otherwise fall back to raw
        annotations_key = "cl_annotations_parquet" if "cl_annotations_parquet" in inputs else "annotations_parquet"
        annotations_path = resolve_artifact_path(
            inputs[annotations_key], annotations_key
        )
        logger.info(f"Using annotations from: {annotations_key}")

        if data_path is None:
            raise ValueError("data_path artifact is required")
        if annotations_path is None:
            raise ValueError(f"{annotations_key} artifact is required")

        # Load annotations
        annotations_df = pl.read_parquet(annotations_path)
        logger.info(f"Loaded {len(annotations_df)} annotations")

        # Load class mapping - prefer CL mapping for CL annotations
        # Priority: cl_class_mapping > class_mapping > build from annotations
        using_cl_annotations = "cl_annotations_parquet" in inputs
        raw_class_mapping = inputs.get("cl_class_mapping") if using_cl_annotations else None
        if raw_class_mapping is None:
            raw_class_mapping = inputs.get("class_mapping")

        if isinstance(raw_class_mapping, dict):
            # Already a dict - use directly
            class_mapping = raw_class_mapping
            logger.info(f"Using class mapping dict with {len(class_mapping)} classes")
        elif isinstance(raw_class_mapping, (str, Path)):
            # It's a path - load from file
            class_mapping_path = resolve_artifact_path(raw_class_mapping, "class_mapping")
            if class_mapping_path and class_mapping_path.exists():
                with open(class_mapping_path) as f:
                    mapping_data = json.load(f)
                    class_mapping = mapping_data.get("class_mapping", mapping_data)
                logger.info(f"Loaded class mapping from {class_mapping_path}")
            else:
                class_mapping = {}
                logger.warning("Class mapping path not found, building from annotations")
        else:
            class_mapping = {}
            logger.warning("No class mapping provided, building from annotations")

        # If class_mapping is empty or we're using CL annotations, build from unique labels
        if not class_mapping or using_cl_annotations:
            # Determine label column for building mapping
            if "cl_name" in annotations_df.columns:
                label_col_for_mapping = "cl_name"
            elif "cl_category" in annotations_df.columns:
                label_col_for_mapping = "cl_category"
            elif "predicted_type" in annotations_df.columns:
                label_col_for_mapping = "predicted_type"
            else:
                label_col_for_mapping = "broad_category"

            unique_labels = sorted(annotations_df[label_col_for_mapping].unique().to_list())
            # Remove None/Unknown if present
            unique_labels = [l for l in unique_labels if l and l not in ("Unknown", "Unmapped")]
            class_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            logger.info(f"Built class mapping from {label_col_for_mapping}: {len(class_mapping)} classes")

        # Get platform for pixel size detection
        platform = self._resolve_platform(inputs)

        # 3. Create LMDB
        lmdb_path, stats = self._create_lmdb(
            data_path,
            annotations_df,
            class_mapping,
            platform,
            cfg,
        )
        logger.info(f"Created LMDB at {lmdb_path}: {stats['n_patches']} patches")

        # 4. Register dataset with lineage
        dataset_id = None
        if cfg.create_clearml_dataset:
            try:
                dataset_id = self._register_lmdb_dataset(
                    lmdb_path, inputs, stats, cfg
                )
                logger.info(f"Registered LMDB dataset: {dataset_id}")
            except Exception as e:
                logger.warning(f"Failed to register ClearML dataset: {e}")

        return StepArtifacts(
            inputs=inputs,
            outputs={
                **inputs,
                "lmdb_path": str(lmdb_path),
                "lmdb_dataset_id": dataset_id,
                "patch_size": cfg.patch_size,
                "skipped": False,
                "extraction_stats": stats,
                # Class information for training step
                "num_classes": stats["n_classes"],
                "class_names": stats["class_names"],
                "index_to_class": stats["index_to_class"],
            },
        )

    def _resolve_platform(self, inputs: dict) -> str:
        """Resolve platform from inputs."""
        platform_value = inputs.get("platform", "xenium")
        platform_path = resolve_artifact_path(platform_value, "platform")
        if platform_path and platform_path.exists() and platform_path.is_file():
            return platform_path.read_text().strip()
        return str(platform_value)

    def _check_existing_lmdb(
        self, inputs: dict, cfg: LMDBCreationConfig
    ) -> str | None:
        """Check if matching LMDB dataset already exists."""
        try:
            from clearml import Dataset
        except ImportError:
            return None

        # Generate expected dataset name
        platform = inputs.get("platform", "unknown")
        annotated_id = inputs.get("annotated_dataset_id", "")
        id_suffix = annotated_id[:8] if annotated_id else "local"

        expected_name = f"lmdb-{platform}-p{cfg.patch_size}-{id_suffix}"

        try:
            datasets = Dataset.list_datasets(
                dataset_project="DAPIDL/lmdb",
                partial_name=expected_name,
            )

            for ds in datasets:
                # Check metadata matches
                meta = ds.get("metadata", {})
                if (
                    meta.get("patch_size") == cfg.patch_size
                    and meta.get("parent_annotated_id") == annotated_id
                    and meta.get("normalization") == cfg.normalization_method
                ):
                    logger.info(f"Found matching LMDB dataset: {ds['id']}")
                    return ds["id"]
        except Exception as e:
            logger.debug(f"Error checking existing datasets: {e}")

        return None

    def _get_dataset_path(self, dataset_id: str) -> Path | None:
        """Get local path to ClearML Dataset."""
        try:
            from clearml import Dataset

            ds = Dataset.get(dataset_id=dataset_id)
            local_path = ds.get_local_copy()
            return Path(local_path) if local_path else None
        except Exception as e:
            logger.warning(f"Failed to get dataset path: {e}")
            return None

    def _create_lmdb(
        self,
        data_path: Path,
        annotations_df: pl.DataFrame,
        class_mapping: dict,
        platform: str,
        cfg: LMDBCreationConfig,
    ) -> tuple[Path, dict]:
        """Create LMDB dataset from patches.

        Returns:
            Tuple of (lmdb_path, stats_dict)
        """
        import lmdb

        # Load DAPI image
        dapi_image = self._load_dapi_image(data_path, platform)
        logger.info(f"Loaded DAPI image: {dapi_image.shape}, dtype={dapi_image.dtype}")

        # Get cell coordinates
        if "x_centroid" in annotations_df.columns:
            x_col, y_col = "x_centroid", "y_centroid"
        elif "x" in annotations_df.columns:
            x_col, y_col = "x", "y"
        elif "center_x" in annotations_df.columns:
            # MERSCOPE format
            x_col, y_col = "center_x", "center_y"
        else:
            # Need to join with cells data
            cells_path = data_path / "cells.parquet"
            cell_metadata_path = data_path / "cell_metadata.csv"

            if cells_path.exists():
                # Xenium format: cells.parquet with x_centroid, y_centroid
                cells_df = pl.read_parquet(cells_path)
                coord_cols = ["cell_id", "x_centroid", "y_centroid"]
                x_col, y_col = "x_centroid", "y_centroid"
            elif cell_metadata_path.exists():
                # MERSCOPE format: cell_metadata.csv with center_x, center_y
                logger.info(f"Loading MERSCOPE cell coordinates from {cell_metadata_path}")
                cells_df = pl.read_csv(cell_metadata_path)
                # MERSCOPE uses unnamed first column as cell_id (index)
                if "" in cells_df.columns:
                    cells_df = cells_df.rename({"": "cell_id"})
                coord_cols = ["cell_id", "center_x", "center_y"]
                x_col, y_col = "center_x", "center_y"
            else:
                raise ValueError("No cell coordinates available")

            # Ensure cell_id types match for join
            # CellTypist outputs string cell_ids, Xenium uses int32, MERSCOPE uses int64
            if annotations_df.schema["cell_id"] != cells_df.schema["cell_id"]:
                # Cast annotations cell_id to match cells_df type
                target_type = cells_df.schema["cell_id"]
                if target_type == pl.Int32 or target_type == pl.Int64:
                    annotations_df = annotations_df.with_columns(
                        pl.col("cell_id").cast(pl.Int64)
                    )
                    cells_df = cells_df.with_columns(
                        pl.col("cell_id").cast(pl.Int64)
                    )
                else:
                    # Cast cells to string if annotations are strings
                    cells_df = cells_df.with_columns(
                        pl.col("cell_id").cast(pl.Utf8)
                    )
                logger.debug(f"Aligned cell_id types for join")

            annotations_df = annotations_df.join(
                cells_df.select(coord_cols),
                on="cell_id",
                how="left",
            )
            logger.info(f"Joined coordinates: {len(annotations_df)} cells with {x_col}, {y_col}")

        # Apply MERSCOPE coordinate transformation (microns to pixels)
        if x_col == "center_x" or platform == "merscope":
            transform_path = data_path / "images" / "micron_to_mosaic_pixel_transform.csv"
            if not transform_path.exists():
                # Also check parent directory
                transform_path = data_path / "micron_to_mosaic_pixel_transform.csv"
            if transform_path.exists():
                logger.info(f"Applying MERSCOPE coordinate transform from {transform_path}")
                transform = np.loadtxt(transform_path).reshape(3, 3)
                # Affine transform: pixel = transform @ [micron_x, micron_y, 1]
                scale_x, offset_x = transform[0, 0], transform[0, 2]
                scale_y, offset_y = transform[1, 1], transform[1, 2]
                logger.info(
                    f"Transform: scale=({scale_x:.4f}, {scale_y:.4f}), "
                    f"offset=({offset_x:.2f}, {offset_y:.2f})"
                )
                # Convert coordinates
                annotations_df = annotations_df.with_columns(
                    (pl.col(x_col) * scale_x + offset_x).alias("x_pixel"),
                    (pl.col(y_col) * scale_y + offset_y).alias("y_pixel"),
                )
                x_col, y_col = "x_pixel", "y_pixel"
                logger.info(
                    f"Transformed coords: x=[{annotations_df[x_col].min():.1f}, {annotations_df[x_col].max():.1f}], "
                    f"y=[{annotations_df[y_col].min():.1f}, {annotations_df[y_col].max():.1f}]"
                )
            else:
                logger.warning(
                    f"MERSCOPE transform file not found at {transform_path}, "
                    "assuming coordinates are already in pixels"
                )

        # Handle multi-Z images (e.g., morphology.ome.tif with focus stacking)
        if dapi_image.ndim == 3:
            n_z = dapi_image.shape[0]
            logger.info(f"Multi-Z image detected ({n_z} levels), using max intensity projection")
            dapi_image = dapi_image.max(axis=0)

        # Filter edge cells
        img_h, img_w = dapi_image.shape
        half_patch = cfg.patch_size // 2
        margin = cfg.edge_margin_px

        if cfg.exclude_edge_cells:
            annotations_df = annotations_df.filter(
                (pl.col(x_col) >= half_patch + margin)
                & (pl.col(x_col) < img_w - half_patch - margin)
                & (pl.col(y_col) >= half_patch + margin)
                & (pl.col(y_col) < img_h - half_patch - margin)
            )
            logger.info(f"After edge filtering: {len(annotations_df)} cells")

        # Normalize image (memory-efficient for large images like MERSCOPE)
        logger.info(f"Normalizing image ({dapi_image.nbytes / 1024**3:.1f} GB)...")
        dapi_normalized = self._normalize_image(dapi_image, cfg.normalization_method)
        # Note: _normalize_image frees original for large images internally
        logger.info(f"Normalized to float32 ({dapi_normalized.nbytes / 1024**3:.1f} GB)")

        # Determine label column - prefer CL-standardized columns
        if cfg.output_format == "lmdb":
            # Priority: cl_name (standardized) > cl_category > predicted_type > broad_category
            if "cl_name" in annotations_df.columns:
                label_col = "cl_name"
                logger.info("Using CL-standardized labels (cl_name)")
            elif "cl_category" in annotations_df.columns:
                label_col = "cl_category"
                logger.info("Using CL category labels (cl_category)")
            elif "predicted_type" in annotations_df.columns and any(
                k not in ["Epithelial", "Immune", "Stromal", "Endothelial", "Other"]
                for k in class_mapping.keys()
            ):
                label_col = "predicted_type"
            else:
                label_col = "broad_category"

        # Output path - create directory structure compatible with MultiTissueDataset
        # Structure: {output}/lmdb_p{size}/patches.lmdb/, labels.npy, class_mapping.json
        dataset_dir = data_path / "pipeline_outputs" / f"lmdb_p{cfg.patch_size}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        lmdb_path = dataset_dir / "patches.lmdb"

        # Create LMDB - estimate size based on patch count
        # Each 128x128 float32 patch ≈ 64KB, add 50% overhead for LMDB
        estimated_size = len(annotations_df) * cfg.patch_size * cfg.patch_size * 4 * 1.5
        map_size = max(50 * 1024**3, int(estimated_size * 2))  # At least 50GB or 2x estimated
        env = lmdb.open(str(lmdb_path), map_size=map_size)

        n_patches = 0
        class_counts = {}
        all_labels = []

        with env.begin(write=True) as txn:
            for row in annotations_df.iter_rows(named=True):
                x = int(row[x_col])
                y = int(row[y_col])

                # Extract patch
                x1 = x - half_patch
                y1 = y - half_patch
                x2 = x + half_patch
                y2 = y + half_patch

                if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                    continue

                patch = dapi_normalized[y1:y2, x1:x2]

                # Get label
                label_str = row.get(label_col, row.get("broad_category", "Unknown"))
                label_idx = class_mapping.get(label_str, -1)

                if label_idx == -1:
                    continue

                # Serialize using format expected by MultiTissueDataset:
                # - Key: big-endian uint64 (sequential index)
                # - Value: int64 label (8 bytes) + uint16 patch data
                key = struct.pack(">Q", n_patches)

                # Pack as bytes: label (int64) + patch (uint16)
                # Note: MultiTissueDataset expects uint16 raw patches, not normalized
                label_bytes = np.array([label_idx], dtype=np.int64).tobytes()
                # Convert back from normalized to uint16
                # IMPORTANT: Must convert float16 to float32 BEFORE multiplying by 65535
                # because float16 max is 65504 < 65535, causing overflow
                # For simplicity, scale [0,1] to [0, 65535] uint16
                patch_uint16 = (patch.astype(np.float32) * 65535).clip(0, 65535).astype(np.uint16)
                patch_bytes = label_bytes + patch_uint16.tobytes()

                txn.put(key, patch_bytes)
                n_patches += 1
                all_labels.append(label_idx)

                # Count classes
                class_counts[label_str] = class_counts.get(label_str, 0) + 1

            # Store metadata
            metadata = {
                "n_patches": n_patches,
                "patch_size": cfg.patch_size,
                "class_mapping": class_mapping,
                "class_counts": class_counts,
                "normalization": cfg.normalization_method,
                "platform": platform,
            }
            txn.put(b"__metadata__", json.dumps(metadata).encode())

        env.close()

        # Save additional files for MultiTissueDataset compatibility
        # 1. labels.npy - integer label for each patch
        labels_array = np.array(all_labels, dtype=np.int64)
        np.save(dataset_dir / "labels.npy", labels_array)
        logger.info(f"Saved labels.npy: {len(labels_array)} labels")

        # 2. class_mapping.json - flat dict of class_name -> index
        with open(dataset_dir / "class_mapping.json", "w") as f:
            json.dump(class_mapping, f, indent=2)

        # 3. metadata.json
        metadata_out = {
            "n_samples": n_patches,
            "n_classes": len(class_mapping),
            "patch_size": cfg.patch_size,
            "normalization": cfg.normalization_method,
            "platform": platform,
            "class_counts": class_counts,
        }
        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata_out, f, indent=2)

        # Build index_to_class from class_mapping
        index_to_class = {idx: name for name, idx in class_mapping.items()}
        class_names = [index_to_class[i] for i in range(len(index_to_class))]

        stats = {
            "n_patches": n_patches,
            "patch_size": cfg.patch_size,
            "class_counts": class_counts,
            "n_classes": len(class_mapping),
            "class_names": class_names,
            "index_to_class": index_to_class,
            "class_mapping": class_mapping,
            "lmdb_path": str(dataset_dir),  # Return dataset directory, not LMDB subdirectory
        }

        return dataset_dir, stats  # Return dataset_dir for compatibility

    def _load_dapi_image(self, data_path: Path, platform: str) -> np.ndarray:
        """Load DAPI image from Xenium or MERSCOPE output."""
        import tifffile

        # Try various paths
        dapi_paths = [
            data_path / "morphology_focus" / "morphology_focus_0000.ome.tif",
            data_path / "morphology.ome.tif",
            data_path / "morphology_focus.ome.tif",
            data_path / "images" / "mosaic_DAPI_z0.tif",  # MERSCOPE
            data_path / "images" / "DAPI.tif",
        ]

        for dapi_path in dapi_paths:
            if dapi_path.exists():
                logger.info(f"Loading DAPI from {dapi_path}")
                return tifffile.imread(str(dapi_path))

        raise FileNotFoundError(f"No DAPI image found in {data_path}")

    def _normalize_image(self, image: np.ndarray, method: str) -> np.ndarray:
        """Normalize image to 0-1 range.

        For large images, uses memory-efficient approach:
        - Computes percentiles on original dtype (faster, less memory)
        - Converts to float32 AFTER freeing original
        """
        import gc

        # Compute percentiles on original dtype (memory efficient)
        if method == "adaptive":
            p_low, p_high = np.percentile(image, [1, 99.5])
        elif method == "percentile":
            p_low, p_high = np.percentile(image, [1, 99])
        elif method == "minmax":
            p_low, p_high = float(image.min()), float(image.max())
        else:
            p_low, p_high = 0, 65535  # Assume uint16

        # For large images (>1GB), use float16 to halve memory usage
        # Peak memory: original (uint16 = 2 bytes) + normalized (float16 = 2 bytes) = 4 bytes/pixel
        # vs float32 which would be 2 + 4 = 6 bytes/pixel
        if image.nbytes > 1 * 1024**3:
            logger.info(
                f"Large image ({image.nbytes / 1024**3:.1f} GB), "
                f"using float16 to reduce memory (saves {image.nbytes / 1024**3:.1f} GB)"
            )

            # Allocate output as float16 (half the memory of float32)
            img_norm = np.empty(image.shape, dtype=np.float16)

            # Process in row chunks to limit memory spike during conversion
            chunk_rows = max(1, 2 * 1024**3 // (image.shape[1] * 4))  # ~2GB chunks
            for start in range(0, image.shape[0], chunk_rows):
                end = min(start + chunk_rows, image.shape[0])
                # Convert chunk to float32 for precision, normalize, then back to float16
                chunk = image[start:end].astype(np.float32)
                chunk = (chunk - p_low) / (p_high - p_low + 1e-8)
                img_norm[start:end] = np.clip(chunk, 0, 1).astype(np.float16)
                del chunk

            # Free original
            del image
            gc.collect()

            return img_norm
        else:
            # Small image - standard approach
            img_float = image.astype(np.float32)
            img_norm = (img_float - p_low) / (p_high - p_low + 1e-8)
            return np.clip(img_norm, 0, 1)

    def _register_lmdb_dataset(
        self,
        lmdb_path: Path,
        inputs: dict,
        stats: dict,
        cfg: LMDBCreationConfig,
    ) -> str | None:
        """Register LMDB dataset with ClearML via S3.

        IMPORTANT: Never upload large data directly to ClearML.
        Pattern:
        1. Upload to S3 first
        2. Register with ClearML using add_external_files() (metadata only)

        Uses parent_datasets for lineage tracking.
        """
        try:
            from clearml import Dataset
        except ImportError:
            logger.warning("ClearML not available, skipping dataset registration")
            return None

        # Determine parent dataset for lineage
        parent_ids = []
        if cfg.parent_dataset_id:
            parent_ids.append(cfg.parent_dataset_id)
        elif inputs.get("annotated_dataset_id"):
            parent_ids.append(inputs["annotated_dataset_id"])

        # Create dataset name
        platform = inputs.get("platform", "unknown")
        annotated_id = inputs.get("annotated_dataset_id", "")
        id_suffix = annotated_id[:8] if annotated_id else "local"
        dataset_name = f"lmdb-{platform}-p{cfg.patch_size}-{id_suffix}"

        # S3 path for this dataset
        s3_path = f"datasets/lmdb/{dataset_name}"

        try:
            if cfg.upload_to_s3:
                # Step 1: Upload to S3 first (NOT to ClearML)
                from dapidl.utils.s3 import upload_to_s3

                s3_uri = upload_to_s3(lmdb_path, s3_path)
                logger.info(f"Uploaded LMDB to S3: {s3_uri}")

                # Step 2: Register with ClearML as external reference
                dataset = Dataset.create(
                    dataset_project="DAPIDL/lmdb",
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
                # Local only - just register the local path reference
                s3_uri = None
                dataset = Dataset.create(
                    dataset_project="DAPIDL/lmdb",
                    dataset_name=dataset_name,
                    parent_datasets=parent_ids if parent_ids else None,
                )
                # Store local path as external reference for consistency
                # (ClearML will resolve this on the local machine)
                dataset.add_external_files(
                    source_url=f"file://{lmdb_path}",
                    dataset_path="/",
                )

            # Metadata (this goes to ClearML - small JSON only)
            dataset.set_metadata({
                "patch_size": cfg.patch_size,
                "n_patches": stats["n_patches"],
                "n_classes": stats["n_classes"],
                "class_counts": stats["class_counts"],
                "normalization": cfg.normalization_method,
                "parent_annotated_id": inputs.get("annotated_dataset_id"),
                "platform": platform,
                "normalize_physical_size": cfg.normalize_physical_size,
                "s3_uri": s3_uri,  # Store S3 location in metadata
                "local_path": str(lmdb_path),  # For reference
                "registration_type": "external_reference",
                "uploaded_to_clearml": False,
            })

            dataset.finalize()
            # NOTE: Do NOT call dataset.upload() - files are already on S3

            logger.info(f"Registered ClearML dataset: {dataset.id} -> {s3_uri or lmdb_path}")
            return dataset.id

        except Exception as e:
            logger.warning(f"Failed to register ClearML dataset: {e}")
            return None

    def get_queue(self) -> str:
        """Return queue name for this step."""
        return "default"  # CPU queue for I/O

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
        runner_script = Path(__file__).parent.parent.parent.parent.parent / "scripts" / f"clearml_step_runner_{self.name}.py"

        self._task = Task.create(
            project_name=project,
            task_name=task_name,
            task_type=Task.TaskTypes.data_processing,
            script=str(runner_script),
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
            "patch_size": self.config.patch_size,
            "normalization_method": self.config.normalization_method,
            "normalize_physical_size": self.config.normalize_physical_size,
            "skip_if_exists": self.config.skip_if_exists,
            "create_clearml_dataset": self.config.create_clearml_dataset,
            "upload_to_s3": self.config.upload_to_s3,
        }
        self._task.set_parameters(params, __parameters_prefix="step_config")

        return self._task
