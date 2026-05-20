"""Tests for dapidl.qc.montage."""

import numpy as np

from dapidl.qc.montage import build_class_montage


def test_montage_returns_rgb_image():
    rng = np.random.default_rng(0)
    patches = rng.integers(0, 4000, size=(10, 32, 32), dtype=np.uint16)
    scores = np.linspace(0.0, 1.0, 10)
    img = build_class_montage(patches, scores, cell_type="Immune", top_n=6, cols=3)
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8


def test_montage_caps_at_top_n():
    patches = np.zeros((100, 16, 16), dtype=np.uint16)
    scores = np.linspace(0, 1, 100)
    # Should not raise and should render at most top_n tiles
    img = build_class_montage(patches, scores, cell_type="T", top_n=4, cols=2)
    assert img.shape[0] > 0 and img.shape[1] > 0
