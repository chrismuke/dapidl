import numpy as np
from dapidl.graph.smooth import transition_matrix, smooth, correct_and_smooth


def _line_graph():
    # 0-1-2 path + isolated node 3 ; directed edges both ways along the path
    edge = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    return edge, 4


def test_transition_matrix_row_stochastic_incl_isolated():
    edge, n = _line_graph()
    T = transition_matrix(edge, n).toarray()
    assert np.allclose(T.sum(1), 1.0)            # every row (incl isolated node 3) sums to 1
    assert T[3, 3] == 1.0                         # isolated node maps to itself (self-loop)


def test_smooth_alpha_zero_is_identity():
    edge, n = _line_graph()
    T = transition_matrix(edge, n)
    p = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8]])
    assert np.allclose(smooth(p, T, alpha=0.0, iters=10), p)


def test_smooth_preserves_simplex():
    edge, n = _line_graph()
    T = transition_matrix(edge, n)
    p = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.3, 0.7]])
    out = smooth(p, T, alpha=0.8, iters=25)
    assert np.allclose(out.sum(1), 1.0)          # convex combination keeps rows on the simplex


def test_correct_step_inert_when_no_train_labels():
    # held-out-slide case: train_idx empty -> C&S reduces to smoothing-only
    edge, n = _line_graph()
    T = transition_matrix(edge, n)
    p = np.array([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5], [0.9, 0.1]])
    cs = correct_and_smooth(p, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                            T, num_classes=2, iters=20)
    assert np.allclose(cs, smooth(p, T, alpha=0.8, iters=20))
