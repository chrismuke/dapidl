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


# ---------------------------------------------------------------------------
# Cellpose adapter tests
# ---------------------------------------------------------------------------

import unittest.mock as mock


def _make_fake_cellpose_result(n_cells=5, h=512, w=512):
    """Create a fake segmentation mask with *n_cells* non-overlapping squares.

    Cell centres are spaced so they fit within (h, w) regardless of n_cells.
    """
    masks = np.zeros((h, w), dtype=np.int32)
    # Space centres evenly across the image with a 15-px half-size guard
    step_r = max(1, (h - 30) // (n_cells + 1))
    step_c = max(1, (w - 30) // (n_cells + 1))
    for i in range(1, n_cells + 1):
        r = 15 + i * step_r
        c = 15 + i * step_c
        r = min(r, h - 16)
        c = min(c, w - 16)
        masks[r - 15 : r + 15, c - 15 : c + 15] = i
    return masks


def test_cellpose_sam_adapter_name():
    from dapidl.benchmark.segmenters.cellpose_adapter import CellposeSAMAdapter
    assert CellposeSAMAdapter().name == "cellpose_sam"


def test_cellpose_cyto3_adapter_name():
    from dapidl.benchmark.segmenters.cellpose_adapter import CellposeCyto3Adapter
    assert CellposeCyto3Adapter().name == "cellpose_cyto3"


def test_cellpose_nuclei_adapter_name():
    from dapidl.benchmark.segmenters.cellpose_adapter import CellposeNucleiAdapter
    assert CellposeNucleiAdapter().name == "cellpose_nuclei"


def test_cellpose_adapter_segment_returns_output():
    from dapidl.benchmark.segmenters.cellpose_adapter import CellposeSAMAdapter
    adapter = CellposeSAMAdapter()
    fake_masks = _make_fake_cellpose_result(3, 256, 256)
    with mock.patch.object(adapter, "_run_cellpose", return_value=fake_masks):
        image = np.random.randint(0, 65535, (256, 256), dtype=np.uint16)
        result = adapter.segment(image, pixel_size_um=0.108)
    assert isinstance(result, SegmentationOutput)
    assert result.n_cells == 3
    assert result.masks.shape == (256, 256)
    assert result.method_name == "cellpose_sam"
    assert result.runtime_seconds >= 0


# ---------------------------------------------------------------------------
# Morphometric evaluation tests
# ---------------------------------------------------------------------------

from dapidl.benchmark.evaluation.morphometric import compute_morphometric_metrics


def _make_two_cell_mask():
    masks = np.zeros((200, 200), dtype=np.int32)
    yy, xx = np.ogrid[:200, :200]
    masks[((yy - 50) ** 2 + (xx - 50) ** 2) <= 20**2] = 1
    masks[((yy - 150) ** 2 + (xx - 150) ** 2) <= 15**2] = 2
    return masks


def test_morphometric_metrics_structure():
    masks = _make_two_cell_mask()
    metrics = compute_morphometric_metrics(masks, pixel_size_um=0.108)
    assert "mean_area_um2" in metrics
    assert "mean_solidity" in metrics
    assert "n_detected" in metrics


def test_morphometric_area_values():
    masks = _make_two_cell_mask()
    metrics = compute_morphometric_metrics(masks, pixel_size_um=0.108)
    assert metrics["n_detected"] == 2
    assert metrics["mean_area_um2"] > 5.0


def test_morphometric_empty_mask():
    masks = np.zeros((100, 100), dtype=np.int32)
    metrics = compute_morphometric_metrics(masks, pixel_size_um=0.108)
    assert metrics["n_detected"] == 0


# ---------------------------------------------------------------------------
# Biological evaluation tests
# ---------------------------------------------------------------------------

from dapidl.benchmark.evaluation.biological import compute_biological_metrics


def test_biological_metrics_perfect_match():
    masks = np.zeros((100, 100), dtype=np.int32)
    masks[10:30, 10:30] = 1
    masks[60:80, 60:80] = 2
    native_centroids = np.array([[20.0, 20.0], [70.0, 70.0]])
    metrics = compute_biological_metrics(masks=masks, native_centroids=native_centroids)
    assert metrics["native_recovery_rate"] == 1.0
    assert metrics["n_recovered"] == 2


def test_biological_metrics_partial_recovery():
    masks = np.zeros((100, 100), dtype=np.int32)
    masks[10:30, 10:30] = 1
    native_centroids = np.array([[20.0, 20.0], [70.0, 70.0]])
    metrics = compute_biological_metrics(masks=masks, native_centroids=native_centroids)
    assert metrics["native_recovery_rate"] == 0.5


def test_biological_metrics_oversegmentation():
    masks = np.zeros((100, 100), dtype=np.int32)
    masks[10:20, 10:30] = 1
    masks[20:30, 10:30] = 2
    native_centroids = np.array([[20.0, 20.0]])
    metrics = compute_biological_metrics(masks=masks, native_centroids=native_centroids)
    assert "split_cell_rate" in metrics
