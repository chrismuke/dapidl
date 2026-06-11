import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dapidl.graph.encoders import FrozenFeatureEncoder
from dapidl.graph.gnn import MeanAggregator, NoGraphAggregator
from dapidl.graph.harness import GraphArmModel, run_ablation

_OUT = Path("pipeline_output/spatial_gnn_probe_2026_06")


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
                       {"nograph": NoGraphAggregator, "graph": MeanAggregator},
                       _Split(), nbr=nbr, labels=labels, device="cpu",
                       num_classes=2, epochs=15, patience=5, lr=1e-3)
    assert set(res["folds"]["b"].keys()) >= {"nograph", "graph", "delta_macro",
                                             "mcnemar_graph_vs_nograph"}
    assert res["folds"]["b"]["nograph"]["macro_f1"] > 0.6   # learns the separable signal
    assert "pooled" in res and "macro_graph" in res["pooled"]


@pytest.mark.skipif(not (_OUT / "stage2_proper_harness_metrics.json").exists(),
                    reason="run `--phase stage2_proper_harness` first (controller/GPU step)")
def test_characterization_reproduces_committed_stage2_proper():
    h = json.loads((_OUT / "stage2_proper_harness_metrics.json").read_text())["folds"]["xenium_rep2"]
    assert abs(h["nograph"]["macro_f1"] - 0.537) <= 0.02   # refactor faithful (no-graph arm)
    assert abs(h["graph"]["macro_f1"] - 0.628) <= 0.02     # refactor faithful (graph arm)
    assert h["graph"]["macro_f1"] > h["nograph"]["macro_f1"]


def test_run_ablation_compare_pairs_and_edge_attr():
    import numpy as np

    from dapidl.graph.edge_geometry import build_edge_attr
    from dapidl.graph.encoders import FrozenFeatureEncoder
    from dapidl.graph.gnn import EdgeGATv2Aggregator, MeanAggregator, NoGraphAggregator
    from dapidl.graph.harness import run_ablation
    from dapidl.graph.knn_graph import build_within_slide_nbr_table
    rng = np.random.RandomState(0)
    n = 200
    src = np.array(["a"] * 100 + ["b"] * 100)
    feats = rng.randn(n, 16).astype(np.float32)
    labels = (feats[:, 0] > 0).astype(np.int64)
    coords = rng.rand(n, 2) * 30
    nbr = build_within_slide_nbr_table(coords, src, k=4)
    node_geom = np.column_stack([rng.uniform(-np.pi, np.pi, n), rng.uniform(0, 1, n),
                                 rng.uniform(0, 4, n)]).astype(np.float32)
    edge_attr = build_edge_attr(coords, node_geom, nbr)

    class _Split:
        def folds(self):
            yield "b", np.arange(0, 80), np.arange(80, 100), np.arange(100, 200)

    res = run_ablation(lambda: FrozenFeatureEncoder(feats, "cpu"),
                       {"nograph": NoGraphAggregator, "mean": MeanAggregator,
                        "gatv2": lambda: EdgeGATv2Aggregator(node_dim=16, edge_dim=8, heads=4)},
                       _Split(), nbr=nbr, labels=labels, device="cpu",
                       num_classes=2, epochs=8, patience=4, lr=1e-3,
                       edge_attr=edge_attr,
                       compare_pairs=[("nograph", "mean"), ("mean", "gatv2")])
    f = res["folds"]["b"]
    assert {"nograph", "mean", "gatv2"} <= set(f)
    assert "delta_mean_vs_nograph" in f and "delta_gatv2_vs_mean" in f
    assert "mcnemar_mean_vs_nograph" in f and "mcnemar_gatv2_vs_mean" in f
    assert "delta_gatv2_vs_mean" in res["pooled"]
