"""Cellpose-based nucleus segmentation component.

This module implements deep learning-based nucleus segmentation using Cellpose,
with tiled processing for large images and fast vectorized centroid matching.

Key features:
- GPU-accelerated Cellpose segmentation
- Tiled processing to handle large images (avoids OOM)
- Fast centroid-to-mask matching (O(n) vectorized lookup)
- Boundary extraction using OpenCV contour detection
"""

from __future__ import annotations

import cv2
import numpy as np
import polars as pl
from loguru import logger
from scipy import ndimage
from skimage.segmentation import relabel_sequential
from tqdm import tqdm

from dapidl.pipeline.base import SegmentationConfig, SegmentationResult
from dapidl.pipeline.registry import register_segmenter


def _extract_batch_boundaries(batch_regions: list) -> list:
    """Worker function for parallel boundary extraction.

    Runs in a separate process to extract boundaries from mask regions.

    Args:
        batch_regions: List of (region, slice_x_start, slice_y_start, cell_ids, pixel_size)

    Returns:
        List of polygon vertex dictionaries
    """
    polygons_data = []

    for region, x_offset, y_offset, cell_ids, pixel_size in batch_regions:
        # Find contours using OpenCV
        contours, _ = cv2.findContours(
            region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        # Get the largest contour (outer boundary)
        contour = max(contours, key=cv2.contourArea)

        # Simplify contour to reduce vertices
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If still too many points, subsample to ~13 vertices
        if len(approx) > 20:
            indices = np.linspace(0, len(approx) - 1, 13, dtype=int)
            approx = approx[indices]

        # Add vertices for each cell_id
        for cell_id in cell_ids:
            for point in approx:
                x_local, y_local = point[0]
                x_global = x_local + x_offset
                y_global = y_local + y_offset
                polygons_data.append({
                    "cell_id": cell_id,
                    "vertex_x": x_global * pixel_size,
                    "vertex_y": y_global * pixel_size,
                })

            # Close the polygon
            x_local, y_local = approx[0][0]
            x_global = x_local + x_offset
            y_global = y_local + y_offset
            polygons_data.append({
                "cell_id": cell_id,
                "vertex_x": x_global * pixel_size,
                "vertex_y": y_global * pixel_size,
            })

    return polygons_data


@register_segmenter
class CellposeSegmenter:
    """Nucleus segmentation using Cellpose deep learning model.

    Processes large images in tiles to avoid GPU OOM errors.
    Matches detected nuclei to existing cells using fast vectorized
    centroid-to-mask lookup.
    """

    name = "cellpose"

    def __init__(self, config: SegmentationConfig | None = None):
        """Initialize the Cellpose segmenter.

        Args:
            config: Segmentation configuration. If None, uses defaults.
        """
        self.config = config or SegmentationConfig()
        self._model = None

    @property
    def model(self):
        """Lazy-load Cellpose model on first use."""
        if self._model is None:
            from cellpose import models

            use_gpu = self.config.gpu if self.config else True
            logger.info(f"Initializing CellposeModel with gpu={use_gpu}")
            self._model = models.CellposeModel(gpu=use_gpu)
            logger.info(f"CellposeModel initialized - device: {self._model.device}")
        return self._model

    def segment(
        self,
        dapi_image: np.ndarray,
        config: SegmentationConfig | None = None,
    ) -> SegmentationResult:
        """Segment nuclei from DAPI image.

        Args:
            dapi_image: 2D DAPI image (H, W), uint16
            config: Override config for this call

        Returns:
            Segmentation results with centroids and boundaries
        """
        cfg = config or self.config

        # Run tiled segmentation
        masks = self._segment_tiled(dapi_image, cfg)

        # Extract centroids and boundaries
        centroids_df, boundaries_df = self._extract_properties(masks, cfg)

        return SegmentationResult(
            centroids_df=centroids_df,
            boundaries_df=boundaries_df,
            masks=masks,
            matching_stats={
                "n_nuclei": int(masks.max()),
                "image_shape": dapi_image.shape,
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
        we check if it falls within a Cellpose-detected nucleus mask.
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

        # Run tiled segmentation
        masks = self._segment_tiled(dapi_image, cfg)

        # Fast centroid-to-mask matching
        matches_df = self._match_centroids_to_masks(cells_df, masks, pixel_size)

        # Extract boundaries only if needed (skip for training - saves ~5 hours!)
        boundaries_df = None
        if not cfg.skip_boundaries:
            if cfg.parallel_boundaries:
                boundaries_df = self._extract_matched_boundaries_parallel(
                    masks, matches_df, pixel_size, cfg.n_boundary_workers
                )
            else:
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
            },
        )

    def _get_pixel_size(self, platform: str) -> float:
        """Get pixel size in microns for platform."""
        pixel_sizes = {
            "xenium": 0.2125,  # ~4.7 pixels/um
            "merscope": 0.108,  # ~9.3 pixels/um
        }
        return pixel_sizes.get(platform.lower(), 0.2125)

    def _run_cellpose_inference(self, tile: np.ndarray, cfg: SegmentationConfig):
        """Run Cellpose model inference on a single tile."""
        # model.eval is Cellpose's inference method (not Python's eval)
        masks, flows, styles = self.model.eval(
            tile,
            diameter=cfg.diameter,
            channels=[0, 0],
            flow_threshold=cfg.flow_threshold,
            cellprob_threshold=cfg.cellprob_threshold,
            batch_size=8,
        )
        return masks, flows, styles

    def _run_cellpose_inference_batch(
        self, tiles: list[np.ndarray], cfg: SegmentationConfig
    ) -> list[np.ndarray]:
        """Run Cellpose inference on a batch of tiles for better GPU utilization.

        Args:
            tiles: List of normalized tile images
            cfg: Segmentation configuration

        Returns:
            List of mask arrays, one per input tile
        """
        if not tiles:
            return []

        # Cellpose model.eval accepts a list of images for batch processing
        masks_list, _, _ = self.model.eval(
            tiles,
            diameter=cfg.diameter,
            channels=[0, 0],
            flow_threshold=cfg.flow_threshold,
            cellprob_threshold=cfg.cellprob_threshold,
            batch_size=len(tiles),
        )
        return masks_list

    def _segment_tiled(
        self,
        dapi_image: np.ndarray,
        cfg: SegmentationConfig,
    ) -> np.ndarray:
        """Run Cellpose segmentation using tiled processing.

        Process large images in overlapping tiles to avoid GPU OOM.
        Only keeps the non-overlapping center region of each tile
        to avoid duplicate detections at boundaries.
        """
        import torch

        # Calculate percentiles using sampling for large images (lazy normalization)
        # We DON'T convert entire image to float32 - normalize each tile on-demand
        h, w = dapi_image.shape
        n_pixels = h * w
        logger.info(f"Computing percentiles for image ({h}x{w}, {n_pixels/1e9:.2f} billion pixels)...")

        # For large images, use sampled percentile estimation (much faster, ~same accuracy)
        if n_pixels > 100_000_000:  # > 100M pixels
            # Sample ~1M pixels for percentile estimation
            sample_size = min(1_000_000, n_pixels)
            rng = np.random.default_rng(42)  # Fixed seed for reproducibility
            flat_indices = rng.choice(n_pixels, size=sample_size, replace=False)
            sample = dapi_image.flat[flat_indices].astype(np.float32)
            p_low, p_high = np.percentile(sample, [1, 99.5])
            logger.info(f"Used sampled percentiles ({sample_size:,} samples): p_low={p_low:.1f}, p_high={p_high:.1f}")
            logger.info(f"Using LAZY normalization (tiles normalized on-demand to save memory)")
        else:
            p_low, p_high = np.percentile(dapi_image, [1, 99.5])
            logger.info(f"Percentiles: p_low={p_low:.1f}, p_high={p_high:.1f}")

        # For small images, pre-normalize (faster). For large images, use lazy normalization.
        use_lazy_norm = n_pixels > 100_000_000
        if not use_lazy_norm:
            dapi_norm = dapi_image.astype(np.float32)
            dapi_norm = np.clip((dapi_norm - p_low) / (p_high - p_low), 0, 1)
            logger.info(f"Pre-normalized image to float32")

        tile_size = cfg.tile_size
        overlap = cfg.tile_overlap

        # Initialize full mask array
        full_masks = np.zeros((h, w), dtype=np.uint32)
        max_label = 0

        # Calculate tile positions
        y_starts = list(range(0, h, tile_size - overlap))
        x_starts = list(range(0, w, tile_size - overlap))

        total_tiles = len(y_starts) * len(x_starts)

        # Batch size for GPU processing (4 tiles = good balance of speed vs memory)
        gpu_batch_size = cfg.gpu_batch_size if hasattr(cfg, 'gpu_batch_size') else 4
        logger.info(f"Starting tiled segmentation: {total_tiles} tiles ({len(y_starts)}x{len(x_starts)})")
        logger.info(f"Using GPU: {self.model.gpu}, device: {self.model.device}, batch_size: {gpu_batch_size}")

        # Collect all tile coordinates
        tile_coords = [
            (y_start, x_start)
            for y_start in y_starts
            for x_start in x_starts
        ]

        tile_count = 0
        with tqdm(total=total_tiles, desc="Segmenting tiles") as pbar:
            # Process tiles in batches
            for batch_start in range(0, len(tile_coords), gpu_batch_size):
                batch_coords = tile_coords[batch_start:batch_start + gpu_batch_size]

                # Extract and normalize tiles for this batch
                batch_tiles = []
                batch_metadata = []  # Store coords and boundaries for each tile

                for y_start, x_start in batch_coords:
                    y_end = min(y_start + tile_size, h)
                    x_end = min(x_start + tile_size, w)

                    # Extract and normalize tile (lazy normalization for large images)
                    if use_lazy_norm:
                        tile_uint16 = dapi_image[y_start:y_end, x_start:x_end]
                        tile = tile_uint16.astype(np.float32)
                        tile = np.clip((tile - p_low) / (p_high - p_low), 0, 1)
                    else:
                        tile = dapi_norm[y_start:y_end, x_start:x_end]

                    batch_tiles.append(tile)
                    batch_metadata.append((y_start, x_start, y_end, x_end, tile.shape))

                # Run Cellpose inference on entire batch
                masks_list = self._run_cellpose_inference_batch(batch_tiles, cfg)

                # Process results for each tile in batch
                for idx, (masks, (y_start, x_start, y_end, x_end, tile_shape)) in enumerate(
                    zip(masks_list, batch_metadata)
                ):
                    # Calculate the non-overlapping region to keep
                    keep_y_start = overlap // 2 if y_start > 0 else 0
                    keep_x_start = overlap // 2 if x_start > 0 else 0
                    keep_y_end = (
                        tile_shape[0] - overlap // 2
                        if y_end < h
                        else tile_shape[0]
                    )
                    keep_x_end = (
                        tile_shape[1] - overlap // 2
                        if x_end < w
                        else tile_shape[1]
                    )

                    # Get the region to place in full mask
                    tile_region = masks[
                        keep_y_start:keep_y_end, keep_x_start:keep_x_end
                    ]

                    # Relabel to avoid conflicts
                    tile_region_relabeled = np.where(
                        tile_region > 0, tile_region + max_label, 0
                    )

                    # Update max label
                    if masks.max() > 0:
                        max_label += masks.max()

                    # Place in full mask
                    dest_y_start = y_start + keep_y_start
                    dest_y_end = y_start + keep_y_end
                    dest_x_start = x_start + keep_x_start
                    dest_x_end = x_start + keep_x_end

                    full_masks[
                        dest_y_start:dest_y_end, dest_x_start:dest_x_end
                    ] = tile_region_relabeled

                    pbar.update(1)
                    tile_count += 1

                # Clear GPU memory after each batch
                torch.cuda.empty_cache()

                # Log progress every 50 tiles
                if tile_count % 50 == 0:
                    logger.info(f"Segmentation progress: {tile_count}/{total_tiles} tiles ({100*tile_count/total_tiles:.1f}%)")

        # Relabel to ensure consecutive labels
        full_masks, _, _ = relabel_sequential(full_masks)

        n_nuclei = full_masks.max()
        logger.info(f"Tiled segmentation complete: {n_nuclei:,} nuclei detected from {total_tiles} tiles")

        return full_masks

    def _match_centroids_to_masks(
        self,
        cells_df: pl.DataFrame,
        masks: np.ndarray,
        pixel_size: float,
    ) -> pl.DataFrame:
        """Fast vectorized matching: check if cell centroids fall within masks.

        This is O(n) complexity - we just look up mask values at each
        centroid's pixel location. Much faster than KD-tree for this use case.
        """
        # Get centroid columns (handle multiple naming conventions)
        # Priority: x_centroid (Xenium) > centroid_x > x (simple)
        if "x_centroid" in cells_df.columns:
            x_col, y_col = "x_centroid", "y_centroid"
        elif "centroid_x" in cells_df.columns:
            x_col, y_col = "centroid_x", "centroid_y"
        elif "x" in cells_df.columns:
            x_col, y_col = "x", "y"
        else:
            raise ValueError(
                f"Could not find centroid columns. Available: {cells_df.columns}"
            )

        # Convert coordinates from microns to pixels
        cx_px = (cells_df[x_col].to_numpy() / pixel_size).astype(int)
        cy_px = (cells_df[y_col].to_numpy() / pixel_size).astype(int)

        # Clip to image bounds
        h, w = masks.shape
        cx_px = np.clip(cx_px, 0, w - 1)
        cy_px = np.clip(cy_px, 0, h - 1)

        # Vectorized lookup: get mask value at each centroid location
        mask_values = masks[cy_px, cx_px]

        # Build matches DataFrame
        return pl.DataFrame(
            {
                "cell_id": cells_df["cell_id"],
                "cellpose_id": mask_values,
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
        """Extract polygon boundaries only for matched nuclei.

        Uses OpenCV contour detection for efficiency.
        Only processes nuclei that were actually matched to cells.
        """
        # Get unique matched cellpose IDs
        matched = matches_df.filter(pl.col("matched"))
        cellpose_ids = matched["cellpose_id"].unique().to_numpy()

        # Create mapping from cellpose_id to cell_id(s)
        cp_to_cells = (
            matched.group_by("cellpose_id")
            .agg(pl.col("cell_id"))
            .to_pandas()
            .set_index("cellpose_id")["cell_id"]
            .to_dict()
        )

        # Find bounding boxes for each label (faster than iterating over full image)
        slices = ndimage.find_objects(masks)

        polygons_data = []

        for cp_id in tqdm(cellpose_ids, desc="Extracting boundaries"):
            if cp_id == 0 or slices[cp_id - 1] is None:
                continue

            slice_y, slice_x = slices[cp_id - 1]

            # Extract small region containing this nucleus
            region = (masks[slice_y, slice_x] == cp_id).astype(np.uint8)

            # Find contours using OpenCV
            contours, _ = cv2.findContours(
                region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            # Get the largest contour (outer boundary)
            contour = max(contours, key=cv2.contourArea)

            # Simplify contour to reduce vertices
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # If still too many points, subsample to ~13 vertices
            if len(approx) > 20:
                indices = np.linspace(0, len(approx) - 1, 13, dtype=int)
                approx = approx[indices]

            # Get cell_ids for this cellpose nucleus
            cell_ids = cp_to_cells.get(cp_id, [])

            # Add vertices for each cell_id
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

                # Close the polygon
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

    def _extract_matched_boundaries_parallel(
        self,
        masks: np.ndarray,
        matches_df: pl.DataFrame,
        pixel_size: float,
        n_workers: int = 8,
    ) -> pl.DataFrame:
        """Extract polygon boundaries using parallel processing.

        Uses multiprocessing to speed up boundary extraction by ~8x.
        Processes nuclei in batches across multiple workers.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Get unique matched cellpose IDs
        matched = matches_df.filter(pl.col("matched"))
        cellpose_ids = matched["cellpose_id"].unique().to_numpy()

        # Create mapping from cellpose_id to cell_id(s)
        cp_to_cells = (
            matched.group_by("cellpose_id")
            .agg(pl.col("cell_id"))
            .to_pandas()
            .set_index("cellpose_id")["cell_id"]
            .to_dict()
        )

        # Find bounding boxes for each label
        slices = ndimage.find_objects(masks)

        # Prepare work items
        work_items = []
        for cp_id in cellpose_ids:
            if cp_id == 0 or slices[cp_id - 1] is None:
                continue
            slice_y, slice_x = slices[cp_id - 1]
            cell_ids = cp_to_cells.get(cp_id, [])
            if cell_ids:
                work_items.append((cp_id, slice_y, slice_x, cell_ids))

        # Process in parallel batches
        polygons_data = []
        batch_size = max(1, len(work_items) // (n_workers * 4))

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i in range(0, len(work_items), batch_size):
                batch = work_items[i : i + batch_size]
                # Extract mask regions for this batch
                batch_regions = []
                for cp_id, slice_y, slice_x, cell_ids in batch:
                    region = (masks[slice_y, slice_x] == cp_id).astype(np.uint8)
                    batch_regions.append((region, slice_x.start, slice_y.start, cell_ids, pixel_size))
                future = executor.submit(_extract_batch_boundaries, batch_regions)
                futures.append(future)

            # Collect results with progress
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting boundaries (parallel)"):
                batch_polygons = future.result()
                polygons_data.extend(batch_polygons)

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

            # Centroid
            cy_px, cx_px = region.centroid
            cx_um = cx_px * pixel_size
            cy_um = cy_px * pixel_size
            area_um2 = region.area * (pixel_size**2)

            centroids_data.append(
                {
                    "cellpose_id": label,
                    "centroid_x_um": cx_um,
                    "centroid_y_um": cy_um,
                    "centroid_x_px": cx_px,
                    "centroid_y_px": cy_px,
                    "area_um2": area_um2,
                }
            )

            # Boundary polygon
            contours = measure.find_contours(masks == label, 0.5)
            if contours:
                contour = max(contours, key=len)

                # Subsample to ~13 vertices
                if len(contour) > 20:
                    indices = np.linspace(0, len(contour) - 1, 13, dtype=int)
                    contour = contour[indices]

                for y_px, x_px in contour:
                    polygons_data.append(
                        {
                            "cellpose_id": label,
                            "vertex_x": x_px * pixel_size,
                            "vertex_y": y_px * pixel_size,
                        }
                    )

                # Close polygon
                polygons_data.append(
                    {
                        "cellpose_id": label,
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
