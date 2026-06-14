# tests/test_graph_banksy.py
import numpy as np
from dapidl.graph.banksy_features import banksy_augment


def _line_graph():
    # node 1 has neighbours 0 and 2 (a directed star centred on 1 for this test)
    coords = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    feats = np.array([[2.0], [0.0], [4.0]])  # D=1
    edge_index = np.array([[1, 1], [0, 2]])  # 1->0, 1->2
    return feats, edge_index, coords


def test_lambda_zero_is_identity_padded_with_zeros():
    feats, ei, coords = _line_graph()
    aug = banksy_augment(feats, ei, coords, lambda_=0.0)
    assert aug.shape == (3, 3)            # N x 3D
    np.testing.assert_allclose(aug[:, 0:1], feats)   # self term == x
    np.testing.assert_allclose(aug[:, 1:], 0.0)      # neighbour terms vanish


def test_neighbour_mean_matches_hand_computation():
    feats, ei, coords = _line_graph()
    aug = banksy_augment(feats, ei, coords, lambda_=0.5)
    # node 1 neighbour mean of feats {2, 4} = 3.0; scaled by sqrt(0.5/2)=0.5
    assert abs(aug[1, 1] - 0.5 * 3.0) < 1e-9
    # node 0 has no out-edges -> neighbour mean 0
    assert abs(aug[0, 1]) < 1e-9


def test_output_width_is_three_times_feature_dim():
    feats = np.random.default_rng(0).normal(size=(10, 7))
    ei = np.array([[0, 1, 2], [1, 2, 0]])
    coords = np.random.default_rng(1).normal(size=(10, 2))
    assert banksy_augment(feats, ei, coords, lambda_=0.8).shape == (10, 21)


def test_chunking_does_not_change_result():
    # edge accumulation is order-preserving, so chunk size must not affect output
    # (this is the guard that lets the function scale to ~18M edges without OOM)
    rng = np.random.default_rng(2)
    n, e = 40, 200
    feats = rng.normal(size=(n, 5))
    coords = rng.normal(size=(n, 2))
    ei = rng.integers(0, n, size=(2, e))
    whole = banksy_augment(feats, ei, coords, lambda_=0.5, chunk=10_000)
    chunked = banksy_augment(feats, ei, coords, lambda_=0.5, chunk=7)
    np.testing.assert_allclose(whole, chunked, rtol=1e-5, atol=1e-6)
