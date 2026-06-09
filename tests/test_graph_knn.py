# tests/test_graph_knn.py
import numpy as np
from dapidl.graph.knn_graph import build_within_slide_knn


def test_edges_stay_within_slide():
    # slide A at x~0, slide B at x~1000 -> a global k-NN would wrongly connect them
    coords = np.array([[0, 0], [1, 0], [2, 0], [1000, 0], [1001, 0], [1002, 0]], float)
    slide_ids = np.array(["A", "A", "A", "B", "B", "B"])
    ei = build_within_slide_knn(coords, slide_ids, k=2)
    assert ei.shape[0] == 2
    # every edge connects two cells on the same slide
    assert np.all(slide_ids[ei[0]] == slide_ids[ei[1]])
    # no self loops
    assert np.all(ei[0] != ei[1])


def test_k_capped_to_slide_size():
    # a 2-cell slide can yield at most 1 neighbour per node even if k=8
    coords = np.array([[0, 0], [1, 0]], float)
    slide_ids = np.array(["A", "A"])
    ei = build_within_slide_knn(coords, slide_ids, k=8)
    assert ei.shape == (2, 2)  # each of 2 nodes -> its single neighbour
    assert set(map(tuple, ei.T.tolist())) == {(0, 1), (1, 0)}


def test_singleton_slide_contributes_no_edges():
    coords = np.array([[0, 0], [5, 5], [6, 6]], float)
    slide_ids = np.array(["solo", "pair", "pair"])
    ei = build_within_slide_knn(coords, slide_ids, k=3)
    assert "solo" not in set(slide_ids[ei[0]].tolist())
