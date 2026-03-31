"""BenchmarkRunner: orchestrates FOV selection, segmentation, evaluation,
and reporting for the DAPIDL segmentation benchmark.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from dapidl.benchmark.segmenters.base import SegmentationOutput, SegmenterAdapter
from dapidl.benchmark.fov_selector import (
    FOVTile,
    extract_fov_tile,
    load_dapi_mosaic,
    select_fovs,
)
from dapidl.benchmark.evaluation.morphometric import compute_morphometric_metrics
from dapidl.benchmark.evaluation.biological import compute_biological_metrics
from dapidl.benchmark.evaluation.cross_method import compute_cross_method_metrics
from dapidl.benchmark.consensus.majority_voting import majority_voting_consensus
from dapidl.benchmark.consensus.iou_weighted import iou_weighted_consensus
from dapidl.benchmark.consensus.topological_voting import topological_voting_consensus
from dapidl.benchmark.reporting import generate_report

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Empty output sentinel for failed segmentations
# ---------------------------------------------------------------------------

_EMPTY_MASK = np.zeros((1, 1), dtype=np.int32)
_EMPTY_CENTROIDS = np.empty((0, 2), dtype=np.float64)


def _empty_output(method_name: str) -> SegmentationOutput:
    """Return a zero-cell SegmentationOutput (used when a segmenter errors)."""
    return SegmentationOutput(
        masks=_EMPTY_MASK,
        centroids=_EMPTY_CENTROIDS,
        n_cells=0,
        runtime_seconds=0.0,
        peak_memory_mb=0.0,
        method_name=method_name,
        metadata={"error": True},
    )


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Orchestrates the full segmentation benchmark pipeline.

    Args:
        dapi_path: Path to the DAPI TIFF mosaic image.
        cell_metadata_path: Path to the cell metadata CSV (e.g. cell_metadata.csv
            from MERSCOPE output).
        output_dir: Directory where results, tiles, masks and the report are saved.
        pixel_size_um: Physical pixel size in micrometres (default 0.108 for Xenium).
        scale: Pixels-per-micron scale factor from the affine transform.
        offset_x: Pixel x-offset from the affine transform.
        offset_y: Pixel y-offset from the affine transform.
    """

    def __init__(
        self,
        dapi_path: str | Path,
        cell_metadata_path: str | Path,
        output_dir: str | Path,
        pixel_size_um: float = 0.108,
        scale: float = 9.259259,
        offset_x: float = 357.2,
        offset_y: float = 2007.97,
    ) -> None:
        self.dapi_path = Path(dapi_path)
        self.cell_metadata_path = Path(cell_metadata_path)
        self.output_dir = Path(output_dir)
        self.pixel_size_um = pixel_size_um
        self.scale = scale
        self.offset_x = offset_x
        self.offset_y = offset_y

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_tile(self, tile: np.ndarray, fov: FOVTile) -> Path:
        """Save a FOV tile as a TIFF and return the path."""
        import tifffile

        tiles_dir = self.output_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        path = tiles_dir / f"fov_{fov.fov_id}_{fov.label}.tif"
        tifffile.imwrite(str(path), tile)
        return path

    def _save_mask(
        self,
        masks: np.ndarray,
        fov: FOVTile,
        method_name: str,
    ) -> Path:
        """Save a segmentation mask as a .npy file and return the path."""
        masks_dir = self.output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        path = masks_dir / f"fov_{fov.fov_id}_{fov.label}_{method_name}.npy"
        np.save(str(path), masks)
        return path

    def _native_centroids_for_fov(
        self,
        cell_metadata: pl.DataFrame,
        fov: FOVTile,
    ) -> np.ndarray:
        """Return native centroids for a FOV as (N, 2) pixel-coordinate array.

        Coordinates are converted from microns to pixels and made relative to
        the FOV tile origin so they match the extracted tile's coordinate system.

        The pixel_bbox is (y_min, y_max, x_min, x_max).  The origin of the tile
        is (y_min, x_min) in the full mosaic, so we subtract those offsets.

        The MERSCOPE affine transform is:
            pixel_x = center_x * scale + offset_x
            pixel_y = center_y * scale + offset_y

        After subtracting the tile origin we get coordinates relative to the tile.
        """
        fov_cells = cell_metadata.filter(pl.col("fov") == fov.fov_id)
        if fov_cells.is_empty():
            return np.empty((0, 2), dtype=np.float64)

        # pixel_bbox = (y_min, y_max, x_min, x_max)
        tile_y_origin = fov.pixel_bbox[0]
        tile_x_origin = fov.pixel_bbox[2]

        native_centroids_px = np.column_stack([
            fov_cells["center_y"].to_numpy() * self.scale + self.offset_y - tile_y_origin,
            fov_cells["center_x"].to_numpy() * self.scale + self.offset_x - tile_x_origin,
        ])
        return native_centroids_px.astype(np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        segmenters: list[SegmenterAdapter],
        n_fovs: int = 5,
        run_consensus: bool = True,
    ) -> Path:
        """Run the full benchmark pipeline.

        Steps:
        1. Load cell metadata.
        2. Select representative FOVs.
        3. Extract and save FOV tiles.
        4. For each segmenter x FOV: run segmentation, save masks, collect metrics.
        5. Optionally run three consensus methods per FOV (if >=2 individual methods).
        6. Evaluate all outputs: morphometric + biological + practical.
        7. Compute cross-method metrics per FOV.
        8. Generate and return the report path.

        Args:
            segmenters: List of SegmenterAdapter instances to benchmark.
            n_fovs: Number of representative FOVs to evaluate on.
            run_consensus: Whether to run consensus methods (requires >=2 individual
                methods to produce meaningful results).

        Returns:
            Path to the generated ``report.md``.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Step 1: Load cell metadata
        # ------------------------------------------------------------------
        logger.info("Loading cell metadata from %s", self.cell_metadata_path)
        cell_metadata = pl.read_csv(str(self.cell_metadata_path))

        # ------------------------------------------------------------------
        # Step 2: Select FOVs
        # ------------------------------------------------------------------
        logger.info("Selecting %d representative FOVs", n_fovs)
        fovs: list[FOVTile] = select_fovs(
            cell_metadata,
            n_fovs=n_fovs,
            scale=self.scale,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
        )
        logger.info("Selected FOVs: %s", [f.label for f in fovs])

        # ------------------------------------------------------------------
        # Step 3: Load mosaic once, extract tiles
        # ------------------------------------------------------------------
        dapi_image = load_dapi_mosaic(self.dapi_path)

        tiles: dict[str, np.ndarray] = {}  # fov_label -> tile array
        for fov in fovs:
            logger.info("Extracting tile for FOV %d (%s)", fov.fov_id, fov.label)
            tile = extract_fov_tile(dapi_image, fov)
            tiles[fov.label] = tile
            self._save_tile(tile, fov)
            logger.info(
                "  FOV %s: tile shape %s, %d native cells",
                fov.label,
                tile.shape,
                fov.n_cells,
            )

        # Free the full mosaic — only keep FOV tiles
        del dapi_image

        # ------------------------------------------------------------------
        # Step 4: Segment each FOV with each method
        # all_outputs[fov_label][method_name] = SegmentationOutput
        # ------------------------------------------------------------------
        all_outputs: dict[str, dict[str, SegmentationOutput]] = {
            fov.label: {} for fov in fovs
        }

        for segmenter in segmenters:
            logger.info("Running segmenter: %s", segmenter.name)
            for fov in fovs:
                tile = tiles[fov.label]
                try:
                    output = segmenter.segment(tile, pixel_size_um=self.pixel_size_um)
                    logger.info(
                        "  %s / FOV %s: %d cells in %.1fs",
                        segmenter.name,
                        fov.label,
                        output.n_cells,
                        output.runtime_seconds,
                    )
                except Exception as exc:
                    logger.warning(
                        "  %s / FOV %s FAILED: %s — using empty output",
                        segmenter.name,
                        fov.label,
                        exc,
                    )
                    output = _empty_output(segmenter.name)

                # Use the adapter's canonical name (may differ from segmenter.name if
                # the output has been overridden by an error stub)
                all_outputs[fov.label][segmenter.name] = output
                self._save_mask(output.masks, fov, segmenter.name)

        # ------------------------------------------------------------------
        # Step 5: Consensus methods
        # ------------------------------------------------------------------
        if run_consensus and len(segmenters) >= 2:
            consensus_fns = [
                ("consensus_majority", majority_voting_consensus),
                ("consensus_iou_weighted", iou_weighted_consensus),
                ("consensus_topological", topological_voting_consensus),
            ]
            for fov in fovs:
                fov_results = all_outputs[fov.label]
                # Only include methods that actually produced cells
                valid_results = {
                    name: out
                    for name, out in fov_results.items()
                    if out.n_cells > 0
                }
                if len(valid_results) < 2:
                    logger.info(
                        "FOV %s: fewer than 2 valid results, skipping consensus",
                        fov.label,
                    )
                    continue
                for consensus_name, consensus_fn in consensus_fns:
                    try:
                        consensus_out = consensus_fn(valid_results)
                        all_outputs[fov.label][consensus_name] = consensus_out
                        self._save_mask(consensus_out.masks, fov, consensus_name)
                        logger.info(
                            "  %s / FOV %s: %d cells",
                            consensus_name,
                            fov.label,
                            consensus_out.n_cells,
                        )
                    except Exception as exc:
                        logger.warning(
                            "  %s / FOV %s FAILED: %s",
                            consensus_name,
                            fov.label,
                            exc,
                        )

        # ------------------------------------------------------------------
        # Step 6: Evaluate — morphometric + biological + practical
        # all_metrics[method_name][fov_label] = {morphometric, biological, practical}
        # ------------------------------------------------------------------
        all_metrics: dict[str, dict] = {}

        for fov in fovs:
            native_centroids = self._native_centroids_for_fov(cell_metadata, fov)
            for method_name, output in all_outputs[fov.label].items():
                if method_name not in all_metrics:
                    all_metrics[method_name] = {}

                morph = compute_morphometric_metrics(
                    output.masks,
                    pixel_size_um=self.pixel_size_um,
                )

                # Use an appropriately shaped empty mask for biological eval when
                # the output is the sentinel empty mask (1x1).
                bio_masks = output.masks
                if output.n_cells == 0 and output.masks.shape == (1, 1):
                    # create empty mask matching tile shape
                    tile = tiles[fov.label]
                    bio_masks = np.zeros(tile.shape[:2], dtype=np.int32)

                bio = compute_biological_metrics(
                    bio_masks,
                    native_centroids,
                    pixel_size_um=self.pixel_size_um,
                )

                practical = {
                    "runtime_seconds": output.runtime_seconds,
                    "peak_memory_mb": output.peak_memory_mb,
                }

                all_metrics[method_name][fov.label] = {
                    "morphometric": morph,
                    "biological": bio,
                    "practical": practical,
                }

        # ------------------------------------------------------------------
        # Step 7: Cross-method evaluation per FOV
        # ------------------------------------------------------------------
        cross_method_dir = self.output_dir / "cross_method"
        cross_method_dir.mkdir(parents=True, exist_ok=True)
        import json
        from dapidl.benchmark.reporting import _numpy_safe

        for fov in fovs:
            fov_results = all_outputs[fov.label]
            if len(fov_results) >= 2:
                try:
                    cross = compute_cross_method_metrics(fov_results)
                    cross_path = cross_method_dir / f"fov_{fov.fov_id}_{fov.label}.json"
                    with open(cross_path, "w") as fh:
                        json.dump(cross, fh, indent=2, default=_numpy_safe)
                except Exception as exc:
                    logger.warning("Cross-method eval failed for FOV %s: %s", fov.label, exc)

        # ------------------------------------------------------------------
        # Step 8: Generate report
        # ------------------------------------------------------------------
        logger.info("Generating report in %s", self.output_dir)
        report_path = generate_report(all_metrics, self.output_dir)
        logger.info("Benchmark complete. Report: %s", report_path)
        return report_path
