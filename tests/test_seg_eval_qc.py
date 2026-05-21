"""Tests for QC patch scoring around centroids."""
import numpy as np
from dapidl.seg_eval.qc_compare import score_centroid_patches


def test_score_centroid_patches_shapes():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 4000, size=(512, 512), dtype=np.uint16)
    centroids = np.array([[100, 100], [200, 250], [400, 400]], dtype=float)  # [y,x]
    scores = score_centroid_patches(img, centroids, patch=128)
    assert set(["qc_score", "focus_score", "detection_score"]).issubset(scores.columns)
    assert scores.height == 3


def test_edge_centroids_skipped():
    img = np.zeros((256, 256), dtype=np.uint16)
    centroids = np.array([[5, 5], [128, 128]], dtype=float)  # first too close to edge
    scores = score_centroid_patches(img, centroids, patch=128)
    assert scores.height == 1  # only the in-bounds patch scored
