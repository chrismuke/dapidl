"""Cellpose segmentation adapters for the DAPIDL benchmark framework.

Wraps Cellpose 4.x models (SAM, cyto3, nuclei) behind the standard
SegmenterAdapter interface.  Also exports helper functions that other
adapters (StarDist, Mesmer, InstanSeg) can reuse:

    _centroids_from_masks  – extract (N,2) centroids from a label image
    _measure_gpu_memory    – peak GPU allocation in MB
    _reset_gpu_memory      – reset peak GPU allocation counter
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from dapidl.benchmark.segmenters.base import SegmentationOutput, SegmenterAdapter


# ---------------------------------------------------------------------------
# Module-level helpers (reused by other adapters)
# ---------------------------------------------------------------------------


def _centroids_from_masks(masks: np.ndarray) -> np.ndarray:
    """Extract cell centroids from an integer label image.

    Args:
        masks: 2-D integer label array where 0 = background and 1..N = cell IDs.

    Returns:
        Float64 array of shape (N, 2) containing [y, x] centroid coordinates.
        Returns an empty (0, 2) array if no cells are found.
    """
    from scipy.ndimage import center_of_mass

    n_cells = int(masks.max())
    if n_cells == 0:
        return np.empty((0, 2), dtype=np.float64)

    labels = list(range(1, n_cells + 1))
    coords = center_of_mass(masks, masks, labels)
    return np.array(coords, dtype=np.float64)


def _measure_gpu_memory() -> float:
    """Return peak GPU memory allocated since last reset, in megabytes.

    Returns 0.0 if CUDA is unavailable.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    except ImportError:
        return 0.0


def _reset_gpu_memory() -> None:
    """Reset the peak GPU memory allocation counter.

    No-op if CUDA is unavailable or torch is not installed.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class _CellposeAdapterBase(SegmenterAdapter):
    """Shared implementation for all Cellpose model variants.

    Subclasses provide :attr:`name` and :attr:`model_type` and may override
    :attr:`supports_cell_boundaries`.
    """

    #: Cellpose model identifier passed to ``CellposeModel(model_type=...)``.
    _model_type: str

    def __init__(
        self,
        model_type: Optional[str] = None,
        gpu: bool = True,
        diameter: float = 0,
    ) -> None:
        """Initialise the adapter.

        Args:
            model_type: Override the default model type for this adapter.
                Normally left as *None* so the class default is used.
            gpu: Whether to use GPU acceleration.
            diameter: Expected cell diameter in pixels.  Pass 0 (default) to
                let Cellpose estimate it automatically.
        """
        if model_type is not None:
            self._model_type = model_type
        self._gpu = gpu
        self._diameter = diameter
        self.__model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _get_model(self):
        """Return the Cellpose model, loading it on first call."""
        if self.__model is None:
            from cellpose import models

            self.__model = models.CellposeModel(
                model_type=self._model_type,
                gpu=self._gpu,
            )
        return self.__model

    # ------------------------------------------------------------------
    # Inference helpers (separated for test mockability)
    # ------------------------------------------------------------------

    def _run_cellpose(
        self,
        image: np.ndarray,
        pixel_size_um: float,
    ) -> np.ndarray:
        """Run Cellpose on *image* and return the integer mask array.

        Args:
            image: 2-D grayscale image of any unsigned integer dtype.
            pixel_size_um: Physical pixel size in µm (unused here but
                available to subclasses that need it).

        Returns:
            Integer label array (int32) of the same spatial shape as *image*.
        """
        model = self._get_model()
        masks, _flows, _styles = model.eval_batch(
            [image],
            diameter=self._diameter,
            channels=[0, 0],
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )
        return masks[0].astype(np.int32)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def segment(
        self,
        image: np.ndarray,
        pixel_size_um: float = 0.108,
    ) -> SegmentationOutput:
        """Segment nuclei/cells in a single-channel DAPI image.

        Args:
            image: 2-D grayscale image array (any unsigned integer dtype).
            pixel_size_um: Physical pixel size in µm (default 0.108 µm for
                Xenium).

        Returns:
            :class:`~dapidl.benchmark.segmenters.base.SegmentationOutput`
            with masks, centroids, timing, and GPU memory stats.
        """
        _reset_gpu_memory()

        t0 = time.perf_counter()
        masks = self._run_cellpose(image, pixel_size_um)
        runtime = time.perf_counter() - t0

        peak_mem = _measure_gpu_memory()
        centroids = _centroids_from_masks(masks)
        n_cells = len(centroids)

        return SegmentationOutput(
            masks=masks,
            centroids=centroids,
            n_cells=n_cells,
            runtime_seconds=runtime,
            peak_memory_mb=peak_mem,
            method_name=self.name,
        )


# ---------------------------------------------------------------------------
# Concrete adapters
# ---------------------------------------------------------------------------


class CellposeSAMAdapter(_CellposeAdapterBase):
    """Cellpose SAM (nuclei) adapter."""

    _model_type = "nuclei"

    @property
    def name(self) -> str:
        return "cellpose_sam"


class CellposeCyto3Adapter(_CellposeAdapterBase):
    """Cellpose cyto3 adapter — returns full cell boundaries."""

    _model_type = "cyto3"

    @property
    def name(self) -> str:
        return "cellpose_cyto3"

    @property
    def supports_cell_boundaries(self) -> bool:
        return True


class CellposeNucleiAdapter(_CellposeAdapterBase):
    """Cellpose nuclei adapter."""

    _model_type = "nuclei"

    @property
    def name(self) -> str:
        return "cellpose_nuclei"
