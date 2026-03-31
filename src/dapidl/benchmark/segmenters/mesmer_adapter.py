"""Mesmer (DeepCell) segmentation adapter for the DAPIDL benchmark framework.

Wraps DeepCell's Mesmer application behind the standard SegmenterAdapter
interface.  Mesmer is a multi-channel model; for DAPI-only input a zeros
membrane channel is synthesised, matching the nuclear-only segmentation
workflow described in the DeepCell documentation.
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


class MesmerAdapter(SegmenterAdapter):
    """DeepCell Mesmer nuclear segmentation adapter.

    Mesmer expects a 4-D tensor of shape ``(batch, H, W, channels)`` where
    channels[0] is the nuclear marker and channels[1] is the membrane/
    cytoplasm marker.  When only a DAPI image is available a zero-filled
    membrane channel is used, which is the standard approach for nuclear-only
    segmentation with Mesmer.

    The ``image_mpp`` (microns per pixel) argument is forwarded to
    ``app.predict()`` so that the model can apply appropriate scale-dependent
    post-processing.
    """

    def __init__(self) -> None:
        self.__app = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _get_app(self):
        """Return the Mesmer application object, loading it on first call."""
        if self.__app is None:
            from deepcell.applications import Mesmer

            self.__app = Mesmer()
        return self.__app

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "mesmer"

    @property
    def supports_cell_boundaries(self) -> bool:
        return True

    def segment(
        self,
        image: np.ndarray,
        pixel_size_um: float = 0.108,
    ) -> SegmentationOutput:
        """Segment nuclei in a single-channel DAPI image using Mesmer.

        Args:
            image: 2-D grayscale image array (any numeric dtype).
            pixel_size_um: Physical pixel size in µm.  Passed to Mesmer as
                ``image_mpp`` for scale-aware post-processing.

        Returns:
            :class:`~dapidl.benchmark.segmenters.base.SegmentationOutput`
            with masks, centroids, timing, and GPU memory stats.
        """
        _reset_gpu_memory()

        app = self._get_app()

        # Build (1, H, W, 2) input: nuclear channel + zeros membrane channel
        h, w = image.shape[:2]
        nuclear = image.astype(np.float32)
        membrane = np.zeros((h, w), dtype=np.float32)
        img_4d = np.stack([nuclear, membrane], axis=-1)[np.newaxis]  # (1, H, W, 2)

        t0 = time.perf_counter()
        result = app.predict(img_4d, image_mpp=pixel_size_um, compartment="nuclear")
        runtime = time.perf_counter() - t0

        # result has shape (1, H, W, 1) for nuclear compartment
        masks = result[0, :, :, 0].astype(np.int32)

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
            metadata={"pixel_size_um": pixel_size_um},
        )
