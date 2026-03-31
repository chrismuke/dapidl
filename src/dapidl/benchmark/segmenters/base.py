"""Base classes for segmentation method adapters.

Defines the standard interface all segmenter adapters must implement
and the SegmentationOutput dataclass returned by every adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SegmentationOutput:
    """Standard output from any segmentation method.

    Attributes:
        masks: Integer label image where 0=background, 1..N=cell IDs. Shape (H, W), dtype int32.
        centroids: Cell centroids as (N, 2) array of [y, x] coordinates, dtype float64.
        n_cells: Number of detected cells (must equal len(centroids) and max(masks)).
        runtime_seconds: Wall-clock time taken by the segmentation call.
        peak_memory_mb: Peak RSS memory increase during segmentation, in megabytes.
        method_name: Human-readable name of the segmentation method.
        metadata: Optional extra information (hyperparameters, version strings, etc.).
    """

    masks: np.ndarray
    centroids: np.ndarray
    n_cells: int
    runtime_seconds: float
    peak_memory_mb: float
    method_name: str
    metadata: dict = field(default_factory=dict)


class SegmenterAdapter(ABC):
    """Abstract base class for all segmentation method adapters.

    Subclasses wrap a specific segmentation library (Cellpose, StarDist,
    Mesmer, InstanSeg, etc.) and expose a uniform interface so benchmarking
    code can treat every method identically.
    """

    @abstractmethod
    def segment(
        self,
        image: np.ndarray,
        pixel_size_um: float = 0.108,
    ) -> SegmentationOutput:
        """Segment nuclei/cells in a single-channel DAPI image.

        Args:
            image: 2-D grayscale image array, any unsigned integer dtype.
            pixel_size_um: Physical pixel size in micrometres (default 0.108 µm
                for Xenium).  Adapters may use this to scale diameter estimates.

        Returns:
            SegmentationOutput with masks, centroids, timing, and memory stats.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, unique identifier for this segmentation method."""

    @property
    def supports_cell_boundaries(self) -> bool:
        """True if this adapter can return full polygon boundaries per cell.

        Defaults to False; override in adapters that support it.
        """
        return False
