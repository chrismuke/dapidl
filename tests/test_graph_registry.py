# tests/test_graph_registry.py
import numpy as np
import pytest
from dapidl.graph.registry import replay_xenium, replay_sthelar, verify_alignment


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


def test_verify_alignment_raises_on_desync():
    rows = [("a", 100.0, 100.0, 0), ("d", 200.0, 200.0, 1)]
    sources = np.array(["xenium_rep1", "xenium_rep1"])
    labels = np.array([0, 1])
    verify_alignment(rows, sources_seq=["xenium_rep1", "xenium_rep1"],
                     stored_sources=sources, stored_labels=labels)  # ok, no raise
    with pytest.raises(AssertionError):
        verify_alignment(rows, sources_seq=["xenium_rep1", "xenium_rep1"],
                         stored_sources=sources, stored_labels=np.array([0, 2]))
