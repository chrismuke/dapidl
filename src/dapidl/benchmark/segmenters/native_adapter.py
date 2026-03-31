"""Native (ground-truth) segmentation adapter for the DAPIDL benchmark framework.

Creates circular pseudo-masks from the Xenium platform's own cell centroids
and nucleus volume measurements.  This serves as the reference / upper-bound
segmenter in comparative benchmarks — no ML model is involved.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from dapidl.benchmark.segmenters.base import SegmentationOutput, SegmenterAdapter


def _radius_from_volume_um(volume_um3: float) -> float:
    """Convert a sphere volume in µm³ to a radius in µm.

    Uses the sphere volume formula ``V = (4/3) π r³``, solved for r:
    ``r = (3V / (4π))^(1/3)``.

    Args:
        volume_um3: Nuclear volume in cubic micrometres.

    Returns:
        Estimated nuclear radius in micrometres.
    """
    return (3.0 * volume_um3 / (4.0 * np.pi)) ** (1.0 / 3.0)


def _draw_disk(
    canvas: np.ndarray,
    cy: float,
    cx: float,
    radius_px: float,
    cell_id: int,
) -> None:
    """Paint a filled disk onto *canvas* in-place with value *cell_id*.

    Pixels already assigned a non-zero label are overwritten only when the
    current cell has a higher ID (last-write wins), which avoids systematic
    bias toward any particular cell.

    Args:
        canvas: 2-D integer array to draw onto (modified in-place).
        cy: Disk centre y coordinate in pixels.
        cx: Disk centre x coordinate in pixels.
        radius_px: Disk radius in pixels.
        cell_id: Integer label to paint (must be > 0).
    """
    h, w = canvas.shape
    r = radius_px

    y_lo = max(0, int(cy - r))
    y_hi = min(h, int(cy + r) + 2)
    x_lo = max(0, int(cx - r))
    x_hi = min(w, int(cx + r) + 2)

    ys = np.arange(y_lo, y_hi)
    xs = np.arange(x_lo, x_hi)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    inside = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2
    canvas[y_lo:y_hi, x_lo:x_hi][inside] = cell_id


class NativeAdapter(SegmenterAdapter):
    """Reference segmenter that reconstructs circular pseudo-masks from Xenium metadata.

    Instead of running an ML model, this adapter reads cell centroids and
    nucleus volumes from a polars DataFrame (as provided by
    :class:`~dapidl.data.xenium.XeniumDataReader`) and draws a filled circle
    for every cell whose centroid falls within the requested FOV region.

    The circle radius is derived from the reported nucleus volume using the
    sphere-volume formula, which gives a reasonable approximation of the
    nuclear cross-section visible in a 2-D DAPI image.

    Attributes:
        cell_metadata: Polars DataFrame with columns including at minimum
            ``cell_id``, ``x_centroid``, ``y_centroid``, and
            ``nucleus_size`` (volume in µm³).
        fov_id: Optional FOV identifier used to filter *cell_metadata* when
            the DataFrame covers multiple fields of view.  Set to ``None`` to
            use all rows.
        x_offset: X coordinate of the image origin in the global coordinate
            system (µm), used to convert global centroids to pixel positions.
        y_offset: Y coordinate of the image origin (µm).
        pixel_size_um: Physical pixel size in µm (default 0.108 µm for Xenium).
    """

    def __init__(
        self,
        cell_metadata: pl.DataFrame,
        fov_id: int | str | None = None,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        pixel_size_um: float = 0.108,
    ) -> None:
        """Initialise the native adapter.

        Args:
            cell_metadata: Polars DataFrame containing per-cell metadata.
                Required columns: ``x_centroid``, ``y_centroid``,
                ``nucleus_size`` (volume in µm³).  Optional column: ``fov_id``
                (used to filter when *fov_id* is not None).
            fov_id: If provided, only rows where ``cell_metadata["fov_id"] ==
                fov_id`` are used.  Pass ``None`` to use all rows.
            x_offset: X-axis offset of the image frame in the global Xenium
                coordinate system (µm).  Centroid x positions are shifted by
                ``-x_offset`` before converting to pixels.
            y_offset: Y-axis offset (µm).
            pixel_size_um: Conversion factor from µm to pixels.  Defaults to
                0.108 µm/pixel (Xenium standard).
        """
        self._cell_metadata = cell_metadata
        self._fov_id = fov_id
        self._x_offset = x_offset
        self._y_offset = y_offset
        self._pixel_size_um = pixel_size_um

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "native"

    def segment(
        self,
        image: np.ndarray,
        pixel_size_um: float = 0.108,
    ) -> SegmentationOutput:
        """Build circular pseudo-masks from native Xenium cell metadata.

        The *pixel_size_um* argument on this method takes precedence over the
        value supplied at construction time, so that the adapter can be called
        in the same uniform way as every other SegmenterAdapter.

        Args:
            image: 2-D grayscale image array.  Only the shape is used
                (masks are drawn to match the image dimensions).
            pixel_size_um: Physical pixel size in µm.  Overrides the
                constructor value when supplied.

        Returns:
            :class:`~dapidl.benchmark.segmenters.base.SegmentationOutput`
            with circular pseudo-masks, centroids, and timing stats.
            ``peak_memory_mb`` is always 0.0 (no GPU used).
        """
        pix = pixel_size_um  # shorthand

        # ------------------------------------------------------------------
        # Filter to the relevant FOV (if requested)
        # ------------------------------------------------------------------
        df = self._cell_metadata
        if self._fov_id is not None and "fov_id" in df.columns:
            df = df.filter(pl.col("fov_id") == self._fov_id)

        h, w = image.shape[:2]
        masks = np.zeros((h, w), dtype=np.int32)

        t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # Draw one disk per cell
        # ------------------------------------------------------------------
        # Pull required columns as numpy arrays for fast iteration
        xs_global = df["x_centroid"].to_numpy()
        ys_global = df["y_centroid"].to_numpy()
        volumes = df["nucleus_size"].to_numpy()

        centroids_list: list[tuple[float, float]] = []

        for cell_id, (x_g, y_g, vol) in enumerate(
            zip(xs_global, ys_global, volumes), start=1
        ):
            # Convert global µm coordinates to pixel coordinates
            cx = (x_g - self._x_offset) / pix
            cy = (y_g - self._y_offset) / pix

            # Skip cells whose centroids fall outside the image
            if cx < 0 or cx >= w or cy < 0 or cy >= h:
                continue

            # Estimate radius from sphere volume (in µm), convert to pixels
            r_um = _radius_from_volume_um(float(vol)) if vol > 0 else pix
            r_px = r_um / pix

            _draw_disk(masks, cy, cx, r_px, cell_id)
            centroids_list.append((cy, cx))

        runtime = time.perf_counter() - t0

        centroids = (
            np.array(centroids_list, dtype=np.float64)
            if centroids_list
            else np.empty((0, 2), dtype=np.float64)
        )
        n_cells = len(centroids_list)

        return SegmentationOutput(
            masks=masks,
            centroids=centroids,
            n_cells=n_cells,
            runtime_seconds=runtime,
            peak_memory_mb=0.0,
            method_name=self.name,
            metadata={
                "fov_id": self._fov_id,
                "x_offset": self._x_offset,
                "y_offset": self._y_offset,
                "pixel_size_um": pix,
            },
        )
