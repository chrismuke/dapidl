import numpy as np
import torch
from dapidl.graph.encoders import FrozenFeatureEncoder
from dapidl.graph.gnn import MeanAggregator, NoGraphAggregator
from dapidl.graph.harness import GraphArmModel, run_ablation


def test_graph_arm_model_forward_shape_and_aggregator_swap():
    feats = np.random.RandomState(0).randn(6, 4).astype(np.float32)
    enc = FrozenFeatureEncoder(feats, device="cpu")
    rows = np.array([0, 1, 2])
    nbr_rows = np.array([[1, 2], [0, 2], [0, 1]])
    valid = torch.ones(3, 2)
    m_graph = GraphArmModel(enc, MeanAggregator(), node_dim=4, num_classes=4)
    m_nograph = GraphArmModel(enc, NoGraphAggregator(), node_dim=4, num_classes=4)
    assert m_graph(rows, nbr_rows, valid).shape == (3, 4)
    assert m_nograph(rows, nbr_rows, valid).shape == (3, 4)


def test_run_ablation_two_arms_on_synthetic_separable_graph():
    # Build a tiny but learnable problem: 2 slides, class == sign of feature[0].
    rng = np.random.RandomState(0)
    n = 200
    src = np.array(["a"] * 100 + ["b"] * 100)
    feats = rng.randn(n, 4).astype(np.float32)
    labels = (feats[:, 0] > 0).astype(np.int64)            # classes 0 and 1 (both present)
    coords = rng.rand(n, 2)
    from dapidl.graph.knn_graph import build_within_slide_nbr_table
    nbr = build_within_slide_nbr_table(coords, src, k=4)

    class _Split:
        def folds(self):
            tr = np.arange(0, 80)
            va = np.arange(80, 100)
            te = np.arange(100, 200)
            yield "b", tr, va, te

    enc_feats = feats
    res = run_ablation(lambda: FrozenFeatureEncoder(enc_feats, "cpu"),
                       {"nograph": NoGraphAggregator(), "graph": MeanAggregator()},
                       _Split(), nbr=nbr, labels=labels, device="cpu",
                       num_classes=2, epochs=15, patience=5, lr=1e-3)
    assert set(res["folds"]["b"].keys()) >= {"nograph", "graph", "delta_macro",
                                             "mcnemar_graph_vs_nograph"}
    assert res["folds"]["b"]["nograph"]["macro_f1"] > 0.6   # learns the separable signal
    assert "pooled" in res and "macro_graph" in res["pooled"]
