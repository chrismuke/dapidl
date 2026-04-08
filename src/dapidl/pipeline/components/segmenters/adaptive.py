"""Adaptive consensus segmentation combining StarDist and Cellpose.

This module implements a density-adaptive consensus segmenter that runs
both StarDist and Cellpose, then selects the best strategy based on
the detection density ratio.

Strategy:
- Run both Cellpose (default) and StarDist
- Compute density ratio = n_stardist / n_cellpose
- If ratio > threshold (dense tissue): use StarDist as base, fill gaps
  with aggressive Cellpose
- Otherwise: use Cellpose default result

Key features:
- Automatic density-based method selection
- Gap-filling with overlap-aware merging
- TF/PyTorch GPU memory management between framework switches
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import cv2
import numpy as np
import polars as pl
from loguru import logger
from scipy import ndimage
from skimage.segmentation import relabel_sequential
from tqdm import tqdm

from dapidl.pipeline.base import SegmentationConfig, SegmentationResult
from dapidl.pipeline.components.segmenters.stardist import (
    configure_tf_gpu,
    reset_gpu_memory,
)
from dapidl.pipeline.registry import register_segmenter


@register_segmenter
class AdaptiveSegmenter:
    """Density-adaptive consensus combining Cellpose and StarDist.

    Runs both methods, computes detection ratio, and selects the
    best strategy per image. In dense regions, StarDist typically
    outperforms Cellpose; aggressive Cellpose fills remaining gaps.
    """

    name = "adaptive"

    def __init__(self, config: SegmentationConfig | None = None):
        """Initialize the adaptive consensus segmenter.

        Args:
            config: Segmentation configuration. If None, uses defaults.
        """
        self.config = config or SegmentationConfig()
        self._cp_model = None
        self._sd_model = None

    @property
    def cellpose_model(self):
        """Lazy-load Cellpose model on first use."""
        if self._cp_model is None:
            from cellpose import models

            use_gpu = self.config.gpu if self.config else True
            logger.info(f"Initializing CellposeModel with gpu={use_gpu}")
            self._cp_model = models.CellposeModel(gpu=use_gpu)
            logger.info(f"CellposeModel initialized - device: {self._cp_model.device}")
        return self._cp_model

    @property
    def stardist_model(self):
        """Lazy-load StarDist model on first use with TF GPU memory capping."""
        if self._sd_model is None:
            configure_tf_gpu(
                self.config.stardist_tf_memory_limit_mb
                if hasattr(self.config, "stardist_tf_memory_limit_mb")
                else 12288
            )
            from stardist.models import StarDist2D

            logger.info("Loading StarDist 2D_versatile_fluo model...")
            self._sd_model = StarDist2D.from_pretrained("2D_versatile_fluo")
            logger.info("StarDist model loaded")
        return self._sd_model

    def segment(
        self,
        dapi_image: np.ndarray,
        config: SegmentationConfig | None = None,
    ) -> SegmentationResult:
        """Segment nuclei from DAPI image using adaptive consensus.

        Args:
            dapi_image: 2D DAPI image (H, W), uint16
            config: Override config for this call

        Returns:
            Segmentation results with centroids and boundaries
        """
        cfg = config or self.config

        # Run adaptive tiled segmentation
        masks, strategy_info = self._segment_tiled_adaptive(dapi_image, cfg)

        # Extract centroids and boundaries
        centroids_df, boundaries_df = self._extract_properties(masks, cfg)

        return SegmentationResult(
            centroids_df=centroids_df,
            boundaries_df=boundaries_df,
            masks=masks,
            matching_stats={
                "n_nuclei": int(masks.max()),
                "image_shape": dapi_image.shape,
                **strategy_info,
            },
        )

    def segment_and_match(
        self,
        dapi_image: np.ndarray,
        cells_df: pl.DataFrame,
        config: SegmentationConfig | None = None,
    ) -> SegmentationResult:
        """Segment nuclei and match to existing cell data.

        Uses fast vectorized centroid-to-mask lookup: for each cell centroid,
        we check if it falls within a detected nucleus mask.
        Only extracts boundaries for matched cells.

        Args:
            dapi_image: 2D DAPI image (H, W), uint16
            cells_df: Cell metadata with x_centroid, y_centroid columns
            config: Override config for this call

        Returns:
            Segmentation results with nucleus-cell matching
        """
        cfg = config or self.config

        # Get pixel size based on platform
        pixel_size = self._get_pixel_size(cfg.platform)

        # Run adaptive tiled segmentation
        masks, strategy_info = self._segment_tiled_adaptive(dapi_image, cfg)

        # Fast centroid-to-mask matching
        matches_df = self._match_centroids_to_masks(cells_df, masks, pixel_size)

        # Extract boundaries only if needed
        boundaries_df = None
        if not cfg.skip_boundaries:
            boundaries_df = self._extract_matched_boundaries(masks, matches_df, pixel_size)

        # Build centroids DataFrame with matching info
        centroids_df = self._build_centroids_df(matches_df, pixel_size)

        # Calculate matching statistics
        n_matched = matches_df.filter(pl.col("matched")).height
        n_total = matches_df.height

        return SegmentationResult(
            centroids_df=centroids_df,
            boundaries_df=boundaries_df,
            masks=masks,
            matching_stats={
                "n_nuclei": int(masks.max()),
                "n_cells": n_total,
                "n_matched": n_matched,
                "match_rate": n_matched / n_total if n_total > 0 else 0.0,
                "image_shape": dapi_image.shape,
                **strategy_info,
            },
        )

    def _get_pixel_size(self, platform: str) -> float:
        """Get pixel size in microns for platform."""
        pixel_sizes = {
            "xenium": 0.2125,
            "merscope": 0.108,
        }
        return pixel_sizes.get(platform.lower(), 0.2125)

    def _run_cellpose(
        self, tile: np.ndarray, cfg: SegmentationConfig, aggressive: bool = False
    ) -> np.ndarray:
        """Run Cellpose inference on a tile."""
        ft = cfg.adaptive_aggressive_flow_threshold if aggressive else cfg.flow_threshold
        cp = cfg.adaptive_aggressive_cellprob_threshold if aggressive else cfg.cellprob_threshold
        # Cellpose's inference API method
        masks, _, _ = self.cellpose_model.eval(
            tile,
            diameter=cfg.diameter,
            channels=[0, 0],
            flow_threshold=ft,
            cellprob_threshold=cp,
            batch_size=8,
        )
        return masks

    def _run_stardist(self, tile: np.ndarray) -> np.ndarray:
        """Run StarDist inference on a tile (expects pre-normalized [0,1] input)."""
        labels, _ = self.stardist_model.predict_instances(tile)
        return labels

    def _merge_fill(
        self,
        base_masks: np.ndarray,
        fill_masks: np.ndarray,
        pixel_size: float,
        cfg: SegmentationConfig,
    ) -> np.ndarray:
        """Merge fill_masks into base_masks for uncovered regions.

        For each fill cell, add to base if overlap < threshold and area > minimum.
        """
        min_area_px = cfg.adaptive_min_fill_area_um2 / (pixel_size**2)
        max_overlap = cfg.adaptive_max_overlap_fraction
        result = base_masks.copy()
        next_label = int(result.max()) + 1

        fill_ids = np.unique(fill_masks)
        fill_ids = fill_ids[fill_ids > 0]

        added = 0
        for fid in fill_ids:
            fill_region = fill_masks == fid
            area = fill_region.sum()
            if area < min_area_px:
                continue

            # Check overlap with existing base masks
            overlap = (result > 0) & fill_region
            overlap_frac = overlap.sum() / area if area > 0 else 1.0
            if overlap_frac <= max_overlap:
                result[fill_region & (result == 0)] = next_label
                next_label += 1
                added += 1

        logger.debug(f"Merge fill: added {added} cells from fill masks")
        return result

    def _segment_tiled_adaptive(
        self,
        dapi_image: np.ndarray,
        cfg: SegmentationConfig,
    ) -> tuple[np.ndarray, dict]:
        """Run adaptive consensus segmentation using tiled processing.

        For each tile, runs both Cellpose and StarDist, checks density ratio,
        and selects strategy accordingly. Then stitches all tiles together.
        """
        h, w = dapi_image.shape
        n_pixels = h * w
        logger.info(
            f"Computing percentiles for image ({h}x{w}, {n_pixels / 1e9:.2f} billion pixels)..."
        )

        # Compute percentiles (sampled for large images)
        if n_pixels > 100_000_000:
            sample_size = min(1_000_000, n_pixels)
            rng = np.random.default_rng(42)
            flat_indices = rng.choice(n_pixels, size=sample_size, replace=False)
            sample = dapi_image.flat[flat_indices].astype(np.float32)
            p_low, p_high = np.percentile(sample, [1, 99.8])
            logger.info(
                f"Used sampled percentiles ({sample_size:,} samples): p_low={p_low:.1f}, p_high={p_high:.1f}"
            )
        else:
            p_low, p_high = np.percentile(dapi_image, [1, 99.8])
            logger.info(f"Percentiles: p_low={p_low:.1f}, p_high={p_high:.1f}")

        if p_high - p_low < 1e-6:
            logger.warning("Image has near-zero dynamic range, returning empty mask")
            return np.zeros((h, w), dtype=np.uint32), {"strategy": "empty"}

        use_lazy_norm = n_pixels > 100_000_000
        if not use_lazy_norm:
            dapi_norm = dapi_image.astype(np.float32)
            dapi_norm = np.clip((dapi_norm - p_low) / (p_high - p_low), 0, 1)

        tile_size = cfg.tile_size
        overlap = cfg.tile_overlap
        density_threshold = cfg.adaptive_density_threshold

        full_masks = np.zeros((h, w), dtype=np.uint32)
        max_label = 0

        y_starts = list(range(0, h, tile_size - overlap))
        x_starts = list(range(0, w, tile_size - overlap))
        total_tiles = len(y_starts) * len(x_starts)

        logger.info(
            f"Starting adaptive tiled segmentation: {total_tiles} tiles ({len(y_starts)}x{len(x_starts)})"
        )

        tile_strategies = {"cellpose_default": 0, "stardist_base_cellpose_fill": 0}
        tile_count = 0
        pixel_size = self._get_pixel_size(cfg.platform)

        with tqdm(total=total_tiles, desc="Segmenting tiles (adaptive)") as pbar:
            for y_start in y_starts:
                for x_start in x_starts:
                    y_end = min(y_start + tile_size, h)
                    x_end = min(x_start + tile_size, w)

                    # Extract and normalize tile
                    if use_lazy_norm:
                        tile_uint16 = dapi_image[y_start:y_end, x_start:x_end]
                        tile = tile_uint16.astype(np.float32)
                        tile = np.clip((tile - p_low) / (p_high - p_low), 0, 1)
                    else:
                        tile = dapi_norm[y_start:y_end, x_start:x_end]

                    # Run both methods
                    cp_masks = self._run_cellpose(tile, cfg, aggressive=False)
                    reset_gpu_memory()
                    sd_masks = self._run_stardist(tile)
                    reset_gpu_memory()

                    n_cp = int(cp_masks.max())
                    n_sd = int(sd_masks.max())
                    ratio = n_sd / max(n_cp, 1)

                    if ratio > density_threshold and n_sd > 0:
                        # Dense region: StarDist base + aggressive Cellpose fill
                        cp_aggressive = self._run_cellpose(tile, cfg, aggressive=True)
                        reset_gpu_memory()
                        masks = self._merge_fill(sd_masks, cp_aggressive, pixel_size, cfg)
                        tile_strategies["stardist_base_cellpose_fill"] += 1
                    else:
                        masks = cp_masks
                        tile_strategies["cellpose_default"] += 1

                    # Calculate the non-overlapping region to keep
                    keep_y_start = overlap // 2 if y_start > 0 else 0
                    keep_x_start = overlap // 2 if x_start > 0 else 0
                    keep_y_end = masks.shape[0] - overlap // 2 if y_end < h else masks.shape[0]
                    keep_x_end = masks.shape[1] - overlap // 2 if x_end < w else masks.shape[1]

                    tile_region = masks[keep_y_start:keep_y_end, keep_x_start:keep_x_end]

                    tile_region_relabeled = np.where(tile_region > 0, tile_region + max_label, 0)

                    if masks.max() > 0:
                        max_label += masks.max()

                    dest_y_start = y_start + keep_y_start
                    dest_y_end = y_start + keep_y_end
                    dest_x_start = x_start + keep_x_start
                    dest_x_end = x_start + keep_x_end

                    full_masks[dest_y_start:dest_y_end, dest_x_start:dest_x_end] = (
                        tile_region_relabeled
                    )

                    pbar.update(1)
                    tile_count += 1

                    if tile_count % 50 == 0:
                        logger.info(
                            f"Segmentation progress: {tile_count}/{total_tiles} tiles "
                            f"({100 * tile_count / total_tiles:.1f}%)"
                        )

        # Final cleanup
        reset_gpu_memory()
        full_masks, _, _ = relabel_sequential(full_masks)

        n_nuclei = full_masks.max()
        logger.info(
            f"Adaptive segmentation complete: {n_nuclei:,} nuclei detected from {total_tiles} tiles"
        )
        logger.info(f"Tile strategies: {tile_strategies}")

        strategy_info = {
            "strategy": "adaptive",
            "tile_strategies": tile_strategies,
            "density_threshold": density_threshold,
        }

        return full_masks, strategy_info

    def _match_centroids_to_masks(
        self,
        cells_df: pl.DataFrame,
        masks: np.ndarray,
        pixel_size: float,
    ) -> pl.DataFrame:
        """Fast vectorized matching: check if cell centroids fall within masks."""
        if "x_centroid" in cells_df.columns:
            x_col, y_col = "x_centroid", "y_centroid"
        elif "centroid_x" in cells_df.columns:
            x_col, y_col = "centroid_x", "centroid_y"
        elif "x" in cells_df.columns:
            x_col, y_col = "x", "y"
        else:
            raise ValueError(f"Could not find centroid columns. Available: {cells_df.columns}")

        cx_px = (cells_df[x_col].to_numpy() / pixel_size).astype(int)
        cy_px = (cells_df[y_col].to_numpy() / pixel_size).astype(int)

        h, w = masks.shape
        cx_px = np.clip(cx_px, 0, w - 1)
        cy_px = np.clip(cy_px, 0, h - 1)

        mask_values = masks[cy_px, cx_px]

        return pl.DataFrame(
            {
                "cell_id": cells_df["cell_id"],
                "adaptive_id": mask_values,
                "centroid_x_px": cx_px,
                "centroid_y_px": cy_px,
                "matched": mask_values > 0,
            }
        )

    def _extract_matched_boundaries(
        self,
        masks: np.ndarray,
        matches_df: pl.DataFrame,
        pixel_size: float,
    ) -> pl.DataFrame:
        """Extract polygon boundaries only for matched nuclei."""
        matched = matches_df.filter(pl.col("matched"))
        adaptive_ids = matched["adaptive_id"].unique().to_numpy()

        id_to_cells = (
            matched.group_by("adaptive_id")
            .agg(pl.col("cell_id"))
            .to_pandas()
            .set_index("adaptive_id")["cell_id"]
            .to_dict()
        )

        slices = ndimage.find_objects(masks)
        polygons_data = []

        for a_id in tqdm(adaptive_ids, desc="Extracting boundaries"):
            if a_id == 0 or slices[a_id - 1] is None:
                continue

            slice_y, slice_x = slices[a_id - 1]
            region = (masks[slice_y, slice_x] == a_id).astype(np.uint8)

            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)

            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) > 20:
                indices = np.linspace(0, len(approx) - 1, 13, dtype=int)
                approx = approx[indices]

            cell_ids = id_to_cells.get(a_id, [])

            for cell_id in cell_ids:
                for point in approx:
                    x_local, y_local = point[0]
                    x_global = x_local + slice_x.start
                    y_global = y_local + slice_y.start
                    polygons_data.append(
                        {
                            "cell_id": cell_id,
                            "vertex_x": x_global * pixel_size,
                            "vertex_y": y_global * pixel_size,
                        }
                    )

                x_local, y_local = approx[0][0]
                x_global = x_local + slice_x.start
                y_global = y_local + slice_y.start
                polygons_data.append(
                    {
                        "cell_id": cell_id,
                        "vertex_x": x_global * pixel_size,
                        "vertex_y": y_global * pixel_size,
                    }
                )

        return pl.DataFrame(polygons_data)

    def _extract_properties(
        self,
        masks: np.ndarray,
        cfg: SegmentationConfig,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Extract centroids and boundaries from masks (without cell matching)."""
        from skimage import measure

        pixel_size = self._get_pixel_size(cfg.platform)
        regions = measure.regionprops(masks)

        centroids_data = []
        polygons_data = []

        for region in tqdm(regions, desc="Extracting properties"):
            label = region.label

            cy_px, cx_px = region.centroid
            cx_um = cx_px * pixel_size
            cy_um = cy_px * pixel_size
            area_um2 = region.area * (pixel_size**2)

            centroids_data.append(
                {
                    "adaptive_id": label,
                    "centroid_x_um": cx_um,
                    "centroid_y_um": cy_um,
                    "centroid_x_px": cx_px,
                    "centroid_y_px": cy_px,
                    "area_um2": area_um2,
                }
            )

            contours = measure.find_contours(masks == label, 0.5)
            if contours:
                contour = max(contours, key=len)

                if len(contour) > 20:
                    indices = np.linspace(0, len(contour) - 1, 13, dtype=int)
                    contour = contour[indices]

                for y_px, x_px in contour:
                    polygons_data.append(
                        {
                            "adaptive_id": label,
                            "vertex_x": x_px * pixel_size,
                            "vertex_y": y_px * pixel_size,
                        }
                    )

                polygons_data.append(
                    {
                        "adaptive_id": label,
                        "vertex_x": contour[0, 1] * pixel_size,
                        "vertex_y": contour[0, 0] * pixel_size,
                    }
                )

        return pl.DataFrame(centroids_data), pl.DataFrame(polygons_data)

    def _build_centroids_df(
        self,
        matches_df: pl.DataFrame,
        pixel_size: float,
    ) -> pl.DataFrame:
        """Build centroids DataFrame from matching results."""
        return matches_df.with_columns(
            [
                (pl.col("centroid_x_px") * pixel_size).alias("centroid_x_um"),
                (pl.col("centroid_y_px") * pixel_size).alias("centroid_y_um"),
            ]
        )
