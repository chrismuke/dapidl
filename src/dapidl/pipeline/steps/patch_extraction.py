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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from dapidl.pipeline.base import PipelineStep, StepArtifacts


@dataclass
class PatchExtractionConfig:
    """Configuration for patch extraction step."""

    # Patch parameters
    patch_size: int = 128  # 32, 64, 128, 256
    output_format: str = "lmdb"  # "lmdb" or "zarr"

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

        data_path = Path(inputs["data_path"])
        platform = inputs.get("platform", "xenium")

        # Load DAPI image
        dapi_image = self._load_dapi_image(data_path, platform)
        logger.info(f"Loaded DAPI: {dapi_image.shape}, dtype={dapi_image.dtype}")

        # Normalize image
        if cfg.normalize:
            dapi_image = self._normalize_image(dapi_image, cfg)

        # Load annotations
        annotations_df = pl.read_parquet(inputs["annotations_parquet"])
        logger.info(f"Loaded {annotations_df.height} annotations")

        # Load centroids (prefer segmentation output if available)
        if cfg.use_cellpose_centroids and inputs.get("centroids_parquet"):
            # Use segmentation output centroids
            centroids_df = self._load_centroids(
                Path(inputs["centroids_parquet"]), platform
            )
        else:
            cells_path = inputs.get("cells_parquet")
            if cells_path:
                centroids_df = self._load_centroids(Path(cells_path), platform)
            else:
                raise ValueError("No cell coordinates available")

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
        output_dir = data_path / "pipeline_outputs" / "patches"
        output_dir.mkdir(parents=True, exist_ok=True)

        if cfg.output_format == "lmdb":
            patches_path = output_dir / "patches.lmdb"
            stats = self._extract_to_lmdb(
                dapi_image, merged_df, patches_path, cfg.patch_size
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

        return StepArtifacts(
            inputs=inputs,
            outputs={
                **inputs,
                "patches_path": str(patches_path),
                "metadata_parquet": str(metadata_path),
                "dataset_id": dataset_id,
                "extraction_stats": stats,
                "class_mapping": class_mapping,
            },
        )

    def _load_dapi_image(self, data_path: Path, platform: str) -> np.ndarray:
        """Load DAPI image."""
        import tifffile

        if platform == "xenium":
            dapi_path = data_path / "morphology_focus.ome.tif"
            if not dapi_path.exists():
                dapi_path = data_path / "morphology.ome.tif"
            image = tifffile.imread(dapi_path)
            if image.ndim == 3:
                image = image[0]
            return image
        else:  # merscope
            images_dir = data_path / "images"
            dapi_files = sorted(images_dir.glob("mosaic_DAPI_z*.tif"))
            mid_idx = len(dapi_files) // 2
            return tifffile.imread(dapi_files[mid_idx])

    def _normalize_image(
        self, image: np.ndarray, cfg: PatchExtractionConfig
    ) -> np.ndarray:
        """Normalize image to 0-1 range."""
        if cfg.normalization_method == "adaptive":
            # Adaptive percentile based on image statistics
            p_low = np.percentile(image, cfg.percentile_low)
            p_high = np.percentile(image, cfg.percentile_high)
        elif cfg.normalization_method == "percentile":
            p_low = np.percentile(image, cfg.percentile_low)
            p_high = np.percentile(image, cfg.percentile_high)
        else:  # minmax
            p_low = image.min()
            p_high = image.max()

        # Clip and scale
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

        # Raw Xenium cells.parquet (already in pixels)
        if "x_centroid" in cols and "y_centroid" in cols:
            return df.select([
                pl.col("cell_id").cast(pl.Utf8),
                pl.col("x_centroid").alias("x"),
                pl.col("y_centroid").alias("y"),
            ])

        # Raw MERSCOPE (in microns)
        if "EntityID" in cols and "center_x" in cols:
            pixel_size = 0.108  # MERSCOPE pixel size
            return df.select([
                pl.col("EntityID").cast(pl.Utf8).alias("cell_id"),
                (pl.col("center_x") / pixel_size).alias("x"),
                (pl.col("center_y") / pixel_size).alias("y"),
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
        """Add numeric class labels."""
        if not class_mapping:
            # Default coarse mapping
            class_mapping = {"Epithelial": 0, "Immune": 1, "Stromal": 2}

        # Map broad_category to class index
        label_col = "broad_category" if "broad_category" in df.columns else "predicted_type"

        df = df.with_columns(
            pl.col(label_col)
            .map_elements(
                lambda x: class_mapping.get(x, -1),
                return_dtype=pl.Int32,
            )
            .alias("label")
        )

        # Filter out unmapped classes
        return df.filter(pl.col("label") >= 0)

    def _extract_to_lmdb(
        self,
        image: np.ndarray,
        df: pl.DataFrame,
        output_path: Path,
        patch_size: int,
    ) -> dict[str, Any]:
        """Extract patches to LMDB database.

        Uses numpy's native tobytes()/frombuffer() for safe serialization
        instead of pickle.
        """
        import lmdb

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

                    # Extract patch
                    patch = image[
                        cy - half : cy + half,
                        cx - half : cx + half,
                    ]

                    # Validate patch size
                    if patch.shape != (patch_size, patch_size):
                        continue

                    # Store patch as raw bytes (safe serialization)
                    key = f"{n_extracted:08d}".encode()
                    txn.put(key, patch.tobytes())

                    # Store label in separate key
                    label_key = f"label_{n_extracted:08d}".encode()
                    txn.put(label_key, str(label).encode())

                    n_extracted += 1
                    class_counts[label] = class_counts.get(label, 0) + 1

                logger.info(f"Extracted {n_extracted}/{n_cells} patches")

            # Store metadata as JSON (safe serialization)
            metadata = {
                "length": n_extracted,
                "patch_size": patch_size,
                "dtype": str(image.dtype),
                "class_counts": {str(k): v for k, v in class_counts.items()},
            }
            txn.put(b"__metadata__", json.dumps(metadata).encode())

        env.close()
        logger.info(f"Created LMDB with {n_extracted} patches at {output_path}")

        return {
            "n_extracted": n_extracted,
            "n_skipped": n_cells - n_extracted,
            "class_counts": class_counts,
            "patch_size": patch_size,
            "format": "lmdb",
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
        """Create ClearML Dataset from patches."""
        from clearml import Dataset

        # Generate name if not provided
        dataset_name = cfg.dataset_name
        if not dataset_name:
            platform = inputs.get("platform", "xenium")
            annotator = inputs.get("annotator_used", "unknown")
            patch_size = cfg.patch_size
            dataset_name = f"{platform}-{annotator}-p{patch_size}"

        dataset = Dataset.create(
            dataset_project=cfg.dataset_project,
            dataset_name=dataset_name,
        )

        # Add files
        dataset.add_files(str(output_dir))

        # Add metadata
        dataset.set_metadata({
            "patch_size": cfg.patch_size,
            "normalization": cfg.normalization_method,
            "n_patches": stats["n_extracted"],
            "class_counts": stats["class_counts"],
            "source_platform": inputs.get("platform"),
            "annotator": inputs.get("annotator_used"),
        })

        # Finalize
        dataset.finalize()
        dataset.upload()

        logger.info(f"Created ClearML Dataset: {dataset.id}")
        return dataset.id

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
