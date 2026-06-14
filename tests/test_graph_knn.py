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


def test_nbr_table_shape_and_padding():
    import numpy as np
    from dapidl.graph.knn_graph import build_within_slide_nbr_table
    # slide "a": 3 cells; slide "b": 1 cell -> all -1 padded
    coords = np.array([[0, 0], [0, 1], [0, 2], [5, 5]], dtype=float)
    src = np.array(["a", "a", "a", "b"])
    nbr = build_within_slide_nbr_table(coords, src, k=8)
    assert nbr.shape == (4, 8)
    assert np.all(nbr[3] == -1)                       # singleton slide -> no neighbours
    assert (nbr[0] >= 0).sum() == 2                   # cell 0 has 2 same-slide neighbours


def test_nbr_within_slide_only_and_excludes_self():
    import numpy as np
    from dapidl.graph.knn_graph import build_within_slide_nbr_table
    coords = np.array([[0, 0], [0, 1], [10, 0], [10, 1]], dtype=float)
    src = np.array(["a", "a", "b", "b"])
    nbr = build_within_slide_nbr_table(coords, src, k=8)
    for i in range(4):
        for j in nbr[i]:
            if j >= 0:
                assert src[j] == src[i]               # neighbour on same slide
                assert j != i                          # never self
