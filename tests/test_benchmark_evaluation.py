"""Initial tests for the segmentation benchmark framework.

Tests cover:
- SegmentationOutput dataclass creation and field validation
- SegmenterAdapter ABC cannot be instantiated directly
"""

import numpy as np
import pytest

from dapidl.benchmark import SegmentationOutput, SegmenterAdapter


def test_segmentation_output_creation():
    """Create a SegmentationOutput with 2 cells and verify all fields."""
    masks = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32)
    centroids = np.array([[0.5, 1.0], [1.0, 0.5]], dtype=np.float64)

    output = SegmentationOutput(
        masks=masks,
        centroids=centroids,
        n_cells=2,
        runtime_seconds=1.23,
        peak_memory_mb=42.0,
        method_name="test_method",
    )

    assert output.masks.dtype == np.int32
    assert output.masks.shape == (2, 3)
    assert output.centroids.dtype == np.float64
    assert output.centroids.shape == (2, 2)
    assert output.n_cells == 2
    assert output.runtime_seconds == pytest.approx(1.23)
    assert output.peak_memory_mb == pytest.approx(42.0)
    assert output.method_name == "test_method"
    assert output.metadata == {}


def test_segmentation_output_metadata_default_is_independent():
    """Each SegmentationOutput instance gets its own metadata dict."""
    masks = np.zeros((4, 4), dtype=np.int32)
    centroids = np.empty((0, 2), dtype=np.float64)

    a = SegmentationOutput(
        masks=masks, centroids=centroids, n_cells=0,
        runtime_seconds=0.0, peak_memory_mb=0.0, method_name="a",
    )
    b = SegmentationOutput(
        masks=masks, centroids=centroids, n_cells=0,
        runtime_seconds=0.0, peak_memory_mb=0.0, method_name="b",
    )

    a.metadata["key"] = "value"
    assert "key" not in b.metadata


def test_segmenter_adapter_is_abstract():
    """SegmenterAdapter cannot be instantiated directly."""
    with pytest.raises(TypeError):
        SegmenterAdapter()  # type: ignore[abstract]


def test_segmenter_adapter_concrete_subclass():
    """A concrete subclass implementing required methods can be instantiated."""

    class DummySegmenter(SegmenterAdapter):
        @property
        def name(self) -> str:
            return "dummy"

        def segment(self, image, pixel_size_um=0.108):
            masks = np.zeros(image.shape, dtype=np.int32)
            centroids = np.empty((0, 2), dtype=np.float64)
            return SegmentationOutput(
                masks=masks,
                centroids=centroids,
                n_cells=0,
                runtime_seconds=0.0,
                peak_memory_mb=0.0,
                method_name=self.name,
            )

    seg = DummySegmenter()
    assert seg.name == "dummy"
    assert seg.supports_cell_boundaries is False

    image = np.zeros((10, 10), dtype=np.uint16)
    result = seg.segment(image)
    assert isinstance(result, SegmentationOutput)
    assert result.n_cells == 0
