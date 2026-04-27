"""Starpose segmentation adapter.

Delegates DAPIDL's segmentation step to the standalone `starpose` library.
This adapter is a thin shim around `starpose.segment()` that translates
between dapidl's `SegmentationConfig` / `SegmentationResult` types and
starpose's own types, and that matches detected nuclei to the platform
cell centroids the same way the in-tree segmenters do.

Why an adapter?
- Starpose is the canonical segmentation library and is published / installed
  separately. Keeping a thin adapter here means dapidl can still register
  "starpose" as a `Segmenter` without depending on starpose internals.
- One adapter handles every method starpose ships (cellpose, cellpose_cyto3,
  cellpose_nuclei, stardist, instanseg, cellvit, adaptive, majority,
  topological); the dapidl config simply forwards the method name.

Method choice via `SegmentationConfig.method`:
    "starpose"               -> default starpose method ("adaptive")
    "starpose:<method_name>" -> explicit starpose method
    "starpose:topological"   -> e.g. topological consensus
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger

from dapidl.pipeline.base import SegmentationConfig, SegmentationResult
from dapidl.pipeline.registry import register_segmenter

if TYPE_CHECKING:
    pass


def _parse_method(name: str) -> str:
    """Extract the underlying starpose method from a 'starpose[:method]' name.

    Defaults to "adaptive" (starpose's recommended density-adaptive consensus).
    """
    if ":" in name:
        return name.split(":", 1)[1] or "adaptive"
    return "adaptive"


def _starpose_to_dapidl(
    sp_result, pixel_size_um: float, method_name: str, runtime: float
) -> SegmentationResult:
    """Convert a starpose SegmentationResult into a dapidl SegmentationResult.

    Starpose centroids are (N, 2) in pixel coords as [y, x]; dapidl wants a
    polars DataFrame with `cell_id`, `centroid_x_um`, `centroid_y_um`, etc.
    """
    centroids = sp_result.centroids
    if centroids is None or len(centroids) == 0:
        centroids_df = pl.DataFrame(
            {"cell_id": [], "centroid_x_um": [], "centroid_y_um": [], "matched": []}
        )
    else:
        # starpose: (N, 2) [y_pix, x_pix]; convert to microns
        y_pix = centroids[:, 0]
        x_pix = centroids[:, 1]
        centroids_df = pl.DataFrame(
            {
                "cell_id": [f"sp_{i}" for i in range(len(centroids))],
                "centroid_x_um": x_pix * pixel_size_um,
                "centroid_y_um": y_pix * pixel_size_um,
                "matched": [True] * len(centroids),
            }
        )

    matching_stats = {
        "method": method_name,
        "n_detected": int(sp_result.n_cells),
        "runtime_s": float(runtime),
        "starpose_metadata": dict(sp_result.metadata) if sp_result.metadata else {},
    }

    return SegmentationResult(
        centroids_df=centroids_df,
        boundaries_df=None,
        masks=sp_result.masks,
        matching_stats=matching_stats,
    )


@register_segmenter
class StarposeSegmenter:
    """Segmenter backed by the standalone `starpose` library.

    Supports every method starpose registers — pass ``method="starpose"`` for
    the default adaptive consensus, or ``method="starpose:<name>"`` to pick a
    specific one (e.g. ``"starpose:topological"``, ``"starpose:cellpose"``).

    Pixel size:
    - If ``config.pixel_size_um > 0``, that value is used.
    - Otherwise we fall back to platform defaults: 0.2125 (Xenium) / 0.108
      (MERSCOPE) / 0.2125 (STHELAR + everything else).

    Tiling:
    - Starpose auto-tiles images larger than ``tile_size`` along any axis;
      ``config.tile_size`` and ``config.tile_overlap`` are forwarded.
    """

    name = "starpose"

    def __init__(self, config: SegmentationConfig | None = None):
        self.config = config or SegmentationConfig()

    @staticmethod
    def _platform_pixel_size(platform: str | None) -> float:
        if platform is None:
            return 0.2125
        p = platform.lower()
        if p == "merscope":
            return 0.108
        return 0.2125  # xenium, sthelar, default

    def segment(
        self,
        dapi_image: np.ndarray,
        config: SegmentationConfig | None = None,
    ) -> SegmentationResult:
        """Run starpose on a DAPI image and return a dapidl-shaped result."""
        cfg = config or self.config
        method = _parse_method(cfg.method)
        pixel_size = (
            cfg.pixel_size_um
            if cfg.pixel_size_um and cfg.pixel_size_um > 0
            else self._platform_pixel_size(cfg.platform)
        )

        logger.info(
            f"StarposeSegmenter: method={method} pixel_size={pixel_size:.4f} um/px "
            f"image_shape={dapi_image.shape} dtype={dapi_image.dtype}"
        )

        # Lazy import to avoid pulling starpose into module load time
        try:
            import starpose
        except ImportError as exc:
            raise ImportError(
                "starpose is not installed. Install with: uv add starpose "
                "(or pip install starpose) — and then `pip install starpose[cellpose,stardist]` "
                "for the model deps."
            ) from exc

        t0 = time.time()
        sp_result = starpose.segment(
            dapi_image,
            method=method,
            gpu=cfg.gpu,
            pixel_size=pixel_size,
            tile_size=cfg.tile_size,
            overlap=cfg.tile_overlap,
        )
        runtime = time.time() - t0

        logger.info(
            f"StarposeSegmenter: detected {sp_result.n_cells:,} cells in "
            f"{runtime:.1f}s (method={method})"
        )
        return _starpose_to_dapidl(sp_result, pixel_size, method, runtime)

    def segment_and_match(
        self,
        dapi_image: np.ndarray,
        cells_df: pl.DataFrame,
        config: SegmentationConfig | None = None,
    ) -> SegmentationResult:
        """Run starpose, then match its nuclei to platform-provided centroids.

        Matching uses ``config.match_threshold_um`` like the in-tree segmenters.
        Cells without a matched starpose nucleus are still included (with
        ``matched=False``) so downstream patch extraction stays
        platform-driven.
        """
        cfg = config or self.config
        sp = self.segment(dapi_image, cfg)
        if sp.centroids_df.is_empty() or cells_df.is_empty():
            return sp

        nuc_xy_um = sp.centroids_df.select(["centroid_x_um", "centroid_y_um"]).to_numpy()
        x_col = "x_centroid" if "x_centroid" in cells_df.columns else "centroid_x"
        y_col = "y_centroid" if "y_centroid" in cells_df.columns else "centroid_y"
        cell_xy_um = cells_df.select([pl.col(x_col), pl.col(y_col)]).to_numpy()

        if len(nuc_xy_um) == 0 or len(cell_xy_um) == 0:
            return sp

        # KDTree match cell centroids to nearest starpose nucleus
        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:
            raise ImportError("Matching requires scipy") from exc

        tree = cKDTree(nuc_xy_um)
        dist_um, idx = tree.query(cell_xy_um, k=1)
        matched = dist_um <= cfg.match_threshold_um

        cell_id_col = "cell_id" if "cell_id" in cells_df.columns else cells_df.columns[0]
        matched_df = cells_df.with_columns(
            pl.Series("matched", matched.astype(bool)),
            pl.Series("nuc_distance_um", dist_um),
        ).select(
            pl.col(cell_id_col).alias("cell_id"),
            pl.col(x_col).alias("centroid_x_um"),
            pl.col(y_col).alias("centroid_y_um"),
            pl.col("matched"),
            pl.col("nuc_distance_um"),
        )

        match_rate = float(matched.mean()) if len(matched) else 0.0
        stats = {
            **sp.matching_stats,
            "n_matched": int(matched.sum()),
            "n_unmatched": int((~matched).sum()),
            "match_rate": match_rate,
            "match_threshold_um": cfg.match_threshold_um,
        }

        logger.info(
            f"StarposeSegmenter matching: "
            f"{int(matched.sum()):,}/{len(matched):,} cells matched "
            f"({match_rate:.1%}) at threshold {cfg.match_threshold_um} um"
        )
        return SegmentationResult(
            centroids_df=matched_df,
            boundaries_df=sp.boundaries_df,
            masks=sp.masks,
            matching_stats=stats,
        )

    @classmethod
    def list_methods(cls) -> list[str]:
        """List the underlying starpose methods accessible via this adapter."""
        try:
            import starpose

            return [m.name for m in starpose.list_methods() if m.available]
        except ImportError:
            return []
