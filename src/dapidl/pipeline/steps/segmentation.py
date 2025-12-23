"""Segmentation Pipeline Step.

Step 2: Nucleus detection using configurable segmentation method.

This step:
1. Loads DAPI image from data path
2. Runs segmentation (Cellpose or native pass-through)
3. Matches detected nuclei to platform-provided cell centroids
4. Outputs boundary data and matching statistics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from dapidl.pipeline.base import (
    PipelineStep,
    SegmentationConfig,
    StepArtifacts,
)
from dapidl.pipeline.registry import get_segmenter


@dataclass
class SegmentationStepConfig:
    """Configuration for segmentation step."""

    # Method selection
    segmenter: str = "cellpose"  # "cellpose" or "native"

    # Cellpose parameters
    diameter: int = 40
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    gpu: bool = True

    # Matching parameters
    match_threshold_um: float = 5.0

    # Platform info (from data loader)
    platform: str = "xenium"
    pixel_size_um: float = 0.0  # Auto-detect if 0

    # Output options
    save_masks: bool = False
    output_format: str = "parquet"


class SegmentationStep(PipelineStep):
    """Nucleus segmentation step.

    Uses Cellpose for re-segmentation or native platform boundaries.
    Matches detected nuclei to platform cell centroids for label transfer.

    Queue: gpu (Cellpose requires GPU for reasonable performance)
    """

    name = "segmentation"
    queue = "gpu"  # GPU queue for Cellpose

    def __init__(self, config: SegmentationStepConfig | None = None):
        """Initialize segmentation step.

        Args:
            config: Segmentation configuration
        """
        self.config = config or SegmentationStepConfig()
        self._task = None

    def get_parameter_schema(self) -> dict[str, Any]:
        """Return JSON schema for ClearML UI parameters."""
        return {
            "type": "object",
            "properties": {
                "segmenter": {
                    "type": "string",
                    "enum": ["cellpose", "native"],
                    "default": "cellpose",
                    "description": "Segmentation method",
                },
                "diameter": {
                    "type": "integer",
                    "default": 40,
                    "description": "Expected nucleus diameter (pixels)",
                },
                "flow_threshold": {
                    "type": "number",
                    "default": 0.4,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Cellpose flow threshold",
                },
                "match_threshold_um": {
                    "type": "number",
                    "default": 5.0,
                    "description": "Max centroid-nucleus distance (µm)",
                },
                "gpu": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use GPU for Cellpose",
                },
                "save_masks": {
                    "type": "boolean",
                    "default": False,
                    "description": "Save mask arrays (large files)",
                },
            },
        }

    def validate_inputs(self, artifacts: StepArtifacts) -> bool:
        """Validate step inputs.

        Requires outputs from DataLoaderStep:
        - data_path: Path to raw data
        - platform: Platform type
        """
        required = ["data_path", "platform"]
        return all(key in artifacts.outputs for key in required)

    def execute(self, artifacts: StepArtifacts) -> StepArtifacts:
        """Execute segmentation step.

        Args:
            artifacts: Input artifacts from DataLoaderStep

        Returns:
            Output artifacts containing:
            - boundaries_parquet: Path to boundary data
            - centroids_parquet: Path to detected centroids
            - matching_stats: Dict with match statistics
            - masks_path: Path to masks (if save_masks=True)
        """
        cfg = self.config
        inputs = artifacts.outputs  # From data loader

        data_path = Path(inputs["data_path"])
        platform = inputs.get("platform", cfg.platform)

        # Update config with platform info
        if cfg.pixel_size_um == 0:
            cfg.pixel_size_um = self._get_pixel_size(platform)

        # Load DAPI image
        dapi_image = self._load_dapi_image(data_path, platform)
        logger.info(f"Loaded DAPI image: {dapi_image.shape}, dtype={dapi_image.dtype}")

        # Load cell centroids from platform data
        cells_df = self._load_cells(data_path, platform, inputs.get("cells_parquet"))

        # Create segmentation config
        seg_config = SegmentationConfig(
            method=cfg.segmenter,
            diameter=cfg.diameter,
            flow_threshold=cfg.flow_threshold,
            cellprob_threshold=cfg.cellprob_threshold,
            match_threshold_um=cfg.match_threshold_um,
            pixel_size_um=cfg.pixel_size_um,
            platform=platform,
            gpu=cfg.gpu,
        )

        # Get segmenter and run
        segmenter = get_segmenter(cfg.segmenter, seg_config)

        if cfg.segmenter == "native":
            # Native segmenter loads boundaries from platform files
            result = segmenter.load_from_platform(data_path, platform)
        else:
            # Cellpose: segment and match to platform centroids
            result = segmenter.segment_and_match(
                dapi_image=dapi_image,
                cells_df=cells_df,
                config=seg_config,
            )

        logger.info(f"Segmentation complete: {result.matching_stats}")

        # Save outputs
        output_dir = data_path / "pipeline_outputs" / "segmentation"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save boundaries
        boundaries_path = output_dir / "boundaries.parquet"
        if result.boundaries_df is not None:
            result.boundaries_df.write_parquet(boundaries_path)
            logger.info(f"Saved boundaries to {boundaries_path}")

        # Save centroids
        centroids_path = output_dir / "centroids.parquet"
        if result.centroids_df is not None:
            result.centroids_df.write_parquet(centroids_path)

        # Save masks (optional - very large!)
        masks_path = None
        if cfg.save_masks and result.masks is not None:
            import numpy as np

            masks_path = output_dir / "masks.npy"
            np.save(masks_path, result.masks)
            logger.info(f"Saved masks to {masks_path}")

        return StepArtifacts(
            inputs=inputs,
            outputs={
                **inputs,  # Pass through data loader outputs
                "boundaries_parquet": str(boundaries_path),
                "centroids_parquet": str(centroids_path),
                "masks_path": str(masks_path) if masks_path else None,
                "matching_stats": result.matching_stats,
                "segmenter_used": cfg.segmenter,
            },
        )

    def _get_pixel_size(self, platform: str) -> float:
        """Get pixel size for platform."""
        pixel_sizes = {
            "xenium": 0.2125,  # µm/pixel
            "merscope": 0.108,  # µm/pixel
        }
        return pixel_sizes.get(platform, 0.2125)

    def _load_dapi_image(self, data_path: Path, platform: str):
        """Load DAPI image from platform data."""
        import numpy as np

        if platform == "xenium":
            # Xenium: OME-TIFF
            dapi_path = data_path / "morphology_focus.ome.tif"
            if not dapi_path.exists():
                dapi_path = data_path / "morphology.ome.tif"

            import tifffile

            dapi_image = tifffile.imread(dapi_path)

            # Handle multi-channel (use first/DAPI channel)
            if dapi_image.ndim == 3:
                dapi_image = dapi_image[0]

            return dapi_image

        else:  # merscope
            # MERSCOPE: Multiple z-slices, use middle z
            images_dir = data_path / "images"
            dapi_files = sorted(images_dir.glob("mosaic_DAPI_z*.tif"))

            if not dapi_files:
                raise FileNotFoundError(f"No DAPI images found in {images_dir}")

            # Use middle z-slice
            mid_idx = len(dapi_files) // 2
            dapi_path = dapi_files[mid_idx]

            import tifffile

            return tifffile.imread(dapi_path)

    def _load_cells(
        self,
        data_path: Path,
        platform: str,
        cells_path: str | None = None,
    ) -> pl.DataFrame:
        """Load cell centroids from platform data."""
        if cells_path:
            path = Path(cells_path)
        elif platform == "xenium":
            path = data_path / "cells.parquet"
        else:  # merscope
            path = data_path / "cell_metadata.csv"
            if not path.exists():
                path = list(data_path.glob("cell_metadata*.csv"))[0]

        if path.suffix == ".parquet":
            cells_df = pl.read_parquet(path)
        else:
            cells_df = pl.read_csv(path)

        # Standardize column names
        if platform == "xenium":
            # Xenium uses x_centroid, y_centroid
            cells_df = cells_df.select([
                pl.col("cell_id"),
                pl.col("x_centroid").alias("x"),
                pl.col("y_centroid").alias("y"),
            ])
        else:  # merscope
            # MERSCOPE uses center_x, center_y (in µm, need to convert to pixels)
            pixel_size = self._get_pixel_size(platform)
            cells_df = cells_df.select([
                pl.col("EntityID").alias("cell_id"),
                (pl.col("center_x") / pixel_size).alias("x"),
                (pl.col("center_y") / pixel_size).alias("y"),
            ])

        logger.info(f"Loaded {cells_df.height} cells from {path}")
        return cells_df

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
        runner_script = Path(__file__).parent.parent.parent.parent / "scripts" / "clearml_step_runner.py"

        self._task = Task.create(
            project_name=project,
            task_name=task_name,
            task_type=Task.TaskTypes.data_processing,
            script=str(runner_script),
            argparse_args=[f"--step={self.name}"],
            add_task_init_call=False,
        )

        # Connect parameters for UI editing
        params = {
            "segmenter": self.config.segmenter,
            "diameter": self.config.diameter,
            "flow_threshold": self.config.flow_threshold,
            "cellprob_threshold": self.config.cellprob_threshold,
            "match_threshold_um": self.config.match_threshold_um,
            "gpu": self.config.gpu,
            "save_masks": self.config.save_masks,
        }
        self._task.set_parameters(params, __parameters_prefix="step_config")

        return self._task
