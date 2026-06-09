# tests/test_graph_registry.py
import numpy as np
import pytest
from dapidl.graph.registry import (
    patch_correlation,
    replay_xenium,
    verify_content,
    verify_counts,
)


def test_replay_xenium_filters_and_bounds():
    cell_ids = ["a", "b", "c", "d"]
    gt = {"a": "Endothelial", "b": "Unlabeled", "c": "CD8+_T_Cells", "d": "Invasive_Tumor"}
    fine_to_coarse = {"Endothelial": "Endothelial", "Unlabeled": None,
                      "CD8+_T_Cells": "Immune", "Invasive_Tumor": "Epithelial"}.get
    coarse_to_idx = {"Endothelial": 0, "Epithelial": 1, "Immune": 2, "Stromal": 3}
    centroids = np.array([[100, 100], [100, 100], [1, 1], [200, 200]], float)  # c is OOB (edge)
    rows = replay_xenium(cell_ids, gt, fine_to_coarse, coarse_to_idx,
                         centroids, h=300, w=300, half=64)
    # b dropped (Unlabeled->None); c dropped (OOB, x0=1-64<0); a and d kept, in order
    assert [r[0] for r in rows] == ["a", "d"]
    assert [r[3] for r in rows] == [0, 1]


def test_verify_counts_raises_on_desync():
    src = np.array(["xenium_rep1", "xenium_rep1", "sthelar_breast_s0"])
    verify_counts(["xenium_rep1", "xenium_rep1", "sthelar_breast_s0"], src)  # ok, no raise
    with pytest.raises(AssertionError):  # length mismatch
        verify_counts(["xenium_rep1", "xenium_rep1"], src)
    with pytest.raises(AssertionError):  # per-source count/order mismatch
        verify_counts(["xenium_rep1", "sthelar_breast_s0", "sthelar_breast_s0"], src)


def test_patch_correlation_high_for_same_low_for_different():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 8))
    assert patch_correlation(a, a) > 0.99                         # identical -> 1
    assert patch_correlation(a, a + 0.01 * rng.normal(size=(8, 8))) > 0.9   # same cell, noise
    assert abs(patch_correlation(a, rng.normal(size=(8, 8)))) < 0.6         # different cell
    assert patch_correlation(np.ones((4, 4)), a[:4, :4]) == 0.0   # constant -> 0
    assert patch_correlation(a, a[:4]) == 0.0                     # size mismatch -> 0


def test_verify_content_raises_below_threshold():
    verify_content({"xenium_rep1": 0.98, "sthelar_breast_s0": 0.95}, threshold=0.9)  # ok
    with pytest.raises(AssertionError):
        verify_content({"xenium_rep1": 0.98, "sthelar_breast_s0": 0.40}, threshold=0.9)
