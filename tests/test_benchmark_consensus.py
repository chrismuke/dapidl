"""Tests for benchmark consensus and instance matching modules."""

import numpy as np
import pytest

from dapidl.benchmark.consensus.instance_matching import match_instances_iou
from dapidl.benchmark.segmenters.base import SegmentationOutput
from dapidl.benchmark.consensus.majority_voting import majority_voting_consensus
from dapidl.benchmark.consensus.iou_weighted import iou_weighted_consensus
from dapidl.benchmark.consensus.topological_voting import topological_voting_consensus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_three_method_results() -> dict[str, SegmentationOutput]:
    """Three methods with two cells each. Methods a and b are identical; c is shifted."""
    m1 = np.zeros((100, 100), dtype=np.int32)
    m1[10:30, 10:30] = 1
    m1[50:70, 50:70] = 2

    m2 = np.zeros((100, 100), dtype=np.int32)
    m2[10:30, 10:30] = 1
    m2[50:70, 50:70] = 2

    m3 = np.zeros((100, 100), dtype=np.int32)
    m3[10:30, 10:30] = 1
    m3[55:75, 55:75] = 2

    results: dict[str, SegmentationOutput] = {}
    for name, masks in [("a", m1), ("b", m2), ("c", m3)]:
        centroids = np.array([[20.0, 20.0], [60.0, 60.0]])
        results[name] = SegmentationOutput(
            masks=masks,
            centroids=centroids,
            n_cells=2,
            runtime_seconds=1.0,
            peak_memory_mb=0.0,
            method_name=name,
        )
    return results


# ---------------------------------------------------------------------------
# Instance matching tests
# ---------------------------------------------------------------------------

def test_match_instances_perfect():
    """Identical masks should yield perfect IoU=1.0 matches for all instances."""
    m = np.zeros((100, 100), dtype=np.int32)
    m[10:30, 10:30] = 1
    m[50:70, 50:70] = 2

    matches, ious = match_instances_iou(m, m)

    assert len(matches) == 2
    assert all(iou == 1.0 for iou in ious)


def test_match_instances_shifted():
    """One cell perfectly overlaps, one is shifted — IoUs should differ."""
    m1 = np.zeros((100, 100), dtype=np.int32)
    m1[10:30, 10:30] = 1
    m1[50:70, 50:70] = 2

    m2 = np.zeros((100, 100), dtype=np.int32)
    m2[15:35, 15:35] = 1  # shifted
    m2[50:70, 50:70] = 2  # identical

    matches, ious = match_instances_iou(m1, m2)

    assert len(matches) == 2
    assert any(iou == 1.0 for iou in ious)
    assert any(0 < iou < 1.0 for iou in ious)


def test_match_instances_no_overlap():
    """Non-overlapping instances should yield no matches above threshold."""
    m1 = np.zeros((100, 100), dtype=np.int32)
    m1[10:20, 10:20] = 1

    m2 = np.zeros((100, 100), dtype=np.int32)
    m2[80:90, 80:90] = 1

    matches, ious = match_instances_iou(m1, m2, iou_threshold=0.1)

    assert len(matches) == 0


def test_match_instances_empty_masks():
    """Empty masks should return no matches."""
    m1 = np.zeros((50, 50), dtype=np.int32)
    m2 = np.zeros((50, 50), dtype=np.int32)
    m2[10:20, 10:20] = 1

    matches, ious = match_instances_iou(m1, m2)
    assert len(matches) == 0
    assert len(ious) == 0


# ---------------------------------------------------------------------------
# Majority voting tests
# ---------------------------------------------------------------------------

def test_majority_voting_returns_output():
    """Majority voting should return a valid SegmentationOutput."""
    out = majority_voting_consensus(_make_three_method_results())

    assert isinstance(out, SegmentationOutput)
    assert out.method_name == "consensus_majority"
    assert out.n_cells >= 1
    assert out.masks.shape == (100, 100)
    assert out.centroids.shape[1] == 2


def test_majority_voting_unanimous_agreement():
    """When all three methods agree exactly, output should reproduce the cells."""
    m = np.zeros((100, 100), dtype=np.int32)
    m[10:30, 10:30] = 1
    m[50:70, 50:70] = 2

    results = {}
    for name in ["x", "y", "z"]:
        results[name] = SegmentationOutput(
            masks=m.copy(),
            centroids=np.array([[20.0, 20.0], [60.0, 60.0]]),
            n_cells=2,
            runtime_seconds=0.0,
            peak_memory_mb=0.0,
            method_name=name,
        )

    out = majority_voting_consensus(results, min_agreement=0.5)
    assert out.n_cells == 2


# ---------------------------------------------------------------------------
# IoU-weighted consensus tests
# ---------------------------------------------------------------------------

def test_iou_weighted_returns_output():
    """IoU-weighted consensus should return a valid SegmentationOutput."""
    out = iou_weighted_consensus(_make_three_method_results())

    assert isinstance(out, SegmentationOutput)
    assert out.method_name == "consensus_iou_weighted"
    assert out.n_cells >= 1
    assert "weights" in out.metadata


def test_iou_weighted_weights_stored():
    """Weights for each method should be stored in metadata."""
    results = _make_three_method_results()
    out = iou_weighted_consensus(results)

    weights = out.metadata["weights"]
    assert set(weights.keys()) == set(results.keys())
    assert all(w >= 0.01 for w in weights.values())


# ---------------------------------------------------------------------------
# Topological voting tests
# ---------------------------------------------------------------------------

def test_topological_voting_returns_output():
    """Topological voting should return a valid SegmentationOutput."""
    out = topological_voting_consensus(_make_three_method_results())

    assert isinstance(out, SegmentationOutput)
    assert out.method_name == "consensus_topological"
    assert out.n_cells >= 1
    assert out.masks.shape == (100, 100)


def test_topological_voting_small_patches():
    """Topological voting with small patches should still complete."""
    out = topological_voting_consensus(
        _make_three_method_results(),
        patch_size=32,
        overlap=16,
    )

    assert isinstance(out, SegmentationOutput)
    assert out.method_name == "consensus_topological"
    assert out.n_cells >= 1
