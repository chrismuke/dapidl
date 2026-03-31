"""StarDist segmentation adapter for the DAPIDL benchmark framework.

Wraps the StarDist 2D_versatile_fluo pretrained model behind the standard
SegmenterAdapter interface.
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


class StarDistAdapter(SegmenterAdapter):
    """StarDist 2D_versatile_fluo adapter for fluorescence nuclei segmentation.

    Uses the pretrained ``2D_versatile_fluo`` model which is optimised for
    fluorescence microscopy images.  The input is normalised to float32 using
    percentile-based normalisation (1st and 99.8th percentile) before
    inference, which is the approach recommended by the StarDist authors.
    """

    def __init__(self, model_name: str = "2D_versatile_fluo") -> None:
        """Initialise the adapter.

        Args:
            model_name: StarDist pretrained model name.  Defaults to
                ``"2D_versatile_fluo"``, the standard fluorescence model.
        """
        self._model_name = model_name
        self.__model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _get_model(self):
        """Return the StarDist model, loading it on first call."""
        if self.__model is None:
            from stardist.models import StarDist2D

            self.__model = StarDist2D.from_pretrained(self._model_name)
        return self.__model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(image: np.ndarray) -> np.ndarray:
        """Percentile-normalise *image* to float32 in [0, 1].

        Args:
            image: Input image of any numeric dtype.

        Returns:
            Float32 array clipped and scaled to [0, 1].
        """
        lo = float(np.percentile(image, 1))
        hi = float(np.percentile(image, 99.8))
        if hi <= lo:
            return np.zeros_like(image, dtype=np.float32)
        normalised = (image.astype(np.float32) - lo) / (hi - lo)
        return np.clip(normalised, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "stardist"

    def segment(
        self,
        image: np.ndarray,
        pixel_size_um: float = 0.108,
    ) -> SegmentationOutput:
        """Segment nuclei in a single-channel DAPI image using StarDist.

        Args:
            image: 2-D grayscale image array (any unsigned integer dtype).
            pixel_size_um: Physical pixel size in µm.  Not used directly by
                the model but recorded in metadata.

        Returns:
            :class:`~dapidl.benchmark.segmenters.base.SegmentationOutput`
            with masks, centroids, timing, and GPU memory stats.
        """
        _reset_gpu_memory()

        model = self._get_model()
        img_norm = self._normalise(image)

        t0 = time.perf_counter()
        labels, _details = model.predict_instances(img_norm)
        runtime = time.perf_counter() - t0

        masks = labels.astype(np.int32)
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
            metadata={"model_name": self._model_name},
        )
