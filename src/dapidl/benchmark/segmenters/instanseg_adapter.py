"""InstanSeg segmentation adapter for the DAPIDL benchmark framework.

Wraps the InstanSeg ``instanseg_fluorescence_nuclei_cells`` model behind the
standard SegmenterAdapter interface.
"""

from __future__ import annotations

import time

import numpy as np

from dapidl.benchmark.segmenters.base import SegmentationOutput, SegmenterAdapter
from dapidl.benchmark.segmenters.cellpose_adapter import (
    _centroids_from_masks,
    _measure_gpu_memory,
    _reset_gpu_memory,
)


class InstanSegAdapter(SegmenterAdapter):
    """InstanSeg fluorescence nuclei/cell segmentation adapter.

    Uses the ``instanseg_fluorescence_nuclei_cells`` pretrained model.
    ``model.run()`` returns a tuple ``(nuclei_masks, cell_masks)``; only the
    nuclei masks are used here so that the output is consistent with the other
    nuclear-segmentation adapters in this benchmark.

    The input image is cast to float32 before inference.  Physical pixel size
    (``pixel_size``) is forwarded to ``model.run()`` so that InstanSeg can
    apply scale-aware processing.
    """

    _MODEL_NAME = "instanseg_fluorescence_nuclei_cells"

    def __init__(self) -> None:
        self.__model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _get_model(self):
        """Return the InstanSeg model, loading it on first call."""
        if self.__model is None:
            from instanseg import InstanSeg

            self.__model = InstanSeg(self._MODEL_NAME)
        return self.__model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "instanseg"

    @property
    def supports_cell_boundaries(self) -> bool:
        return True

    def segment(
        self,
        image: np.ndarray,
        pixel_size_um: float = 0.108,
    ) -> SegmentationOutput:
        """Segment nuclei/cells in a single-channel DAPI image using InstanSeg.

        Args:
            image: 2-D grayscale image array (any numeric dtype).
            pixel_size_um: Physical pixel size in µm, forwarded to
                ``model.run()`` as ``pixel_size``.

        Returns:
            :class:`~dapidl.benchmark.segmenters.base.SegmentationOutput`
            with masks (nuclei), centroids, timing, and GPU memory stats.
        """
        _reset_gpu_memory()

        model = self._get_model()
        img_float = image.astype(np.float32)

        t0 = time.perf_counter()
        result = model.run(img_float, pixel_size=pixel_size_um)
        runtime = time.perf_counter() - t0

        # result is a tuple (nuclei_masks, cell_masks); use nuclei
        if isinstance(result, tuple):
            nuclei = result[0]
        else:
            nuclei = result

        masks = np.asarray(nuclei).astype(np.int32)

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
            metadata={
                "model_name": self._MODEL_NAME,
                "pixel_size_um": pixel_size_um,
            },
        )
