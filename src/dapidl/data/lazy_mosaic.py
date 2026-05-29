"""Low-RAM lazy mosaic access for LMDB builds (review 2026-05-29 B8).

The Xenium ``morphology_focus.ome.tif`` is tiled + compressed (verified:
``is_memmappable == False``), so the spec's ``tifffile.memmap`` does not apply.
Instead we read it as a lazy tifffile→zarr store: each per-cell crop decodes only
the tiles it touches, and the normalization percentiles come from a strided
spatial subsample — never the full ~900-Mpx ``float32`` copy that risked OOM on
the 62 GB host. STHELAR's DAPI is already a ``(1, H, W)`` zarr (tiled), so the
same ``LazyMosaic`` wrapper serves both sources.

``normalize_crop`` reproduces the legacy full-image normalize EXACTLY on a single
crop (given the same percentiles), so switching to per-crop normalization changes
peak RAM but not the patch bytes (modulo the subsample percentile estimate).
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np


def normalize_crop(crop_u16: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """Percentile-normalize a raw uint16 crop to a uint16 [0, 65535] patch.

    Identical math to the legacy `normalize_dapi` + `(patch*65535).clip().astype`
    pipeline, applied per crop instead of to the whole mosaic.
    """
    if p_high <= p_low:
        p_high = p_low + 1.0
    c = np.clip(crop_u16.astype(np.float32), p_low, p_high)
    c = (c - p_low) / (p_high - p_low)
    return (c * 65535.0).clip(0, 65535).astype(np.uint16)


class LazyMosaic:
    """Bounds-aware crop reader over a 2D or (1, H, W) array-like (numpy or zarr).

    Slicing a zarr array decodes only the touched chunks, keeping peak RAM at a
    few tiles per crop rather than the whole mosaic.
    """

    def __init__(self, arr) -> None:
        if arr.ndim == 2:
            self._ch = None                       # plain (H, W), e.g. Xenium aszarr
            self.shape = (int(arr.shape[0]), int(arr.shape[1]))
        elif arr.ndim == 3:
            # channels-first (C, H, W); DAPI is channel 0 — matches
            # SthelarDataReader._load_dapi's arr[0] (some STHELAR slides ship a
            # multi-stain morpho, e.g. s6 is (5, H, W)).
            self._ch = 0
            self.shape = (int(arr.shape[1]), int(arr.shape[2]))
        else:
            raise ValueError(f"unsupported mosaic ndim {arr.ndim} (want 2D or (C,H,W))")
        self.arr = arr

    def read(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Decode and return the uint16 crop [y0:y1, x0:x1] (lazy tile reads)."""
        if self._ch is None:
            return np.asarray(self.arr[y0:y1, x0:x1])
        return np.asarray(self.arr[self._ch, y0:y1, x0:x1])

    def subsample_percentiles(self, low: float = 1.0, high: float = 99.5,
                              target_px: int = 15_000_000) -> tuple[float, float]:
        """(p_low, p_high) from a strided spatial subsample (~target_px samples)."""
        h, w = self.shape
        stride = max(1, int(np.ceil((h * w / max(target_px, 1)) ** 0.5)))
        if self._ch is None:
            sub = np.asarray(self.arr[::stride, ::stride])
        else:
            sub = np.asarray(self.arr[self._ch, ::stride, ::stride])
        return float(np.percentile(sub.reshape(-1), low)), float(np.percentile(sub.reshape(-1), high))


@contextmanager
def open_xenium_mosaic(image_path):
    """Yield a LazyMosaic over level 0 of a (tiled/compressed) Xenium OME-TIFF.

    Uses tifffile's aszarr store so crops decode lazily; the store is closed on
    exit. Falls back to a full in-memory load only if the zarr path is unavailable.
    """
    import tifffile
    try:
        import zarr
        store = tifffile.imread(image_path, aszarr=True, level=0)
        try:
            z = zarr.open(store, mode="r")
            yield LazyMosaic(z)
        finally:
            store.close()
    except Exception:  # noqa: BLE001 — last-resort fallback keeps the build working
        with tifffile.TiffFile(image_path) as tif:
            arr = tif.pages[0].asarray() if len(tif.pages) > 1 else tif.asarray()
        yield LazyMosaic(arr)
