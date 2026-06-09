# tests/test_graph_probe_eval.py
import numpy as np
from dapidl.graph.probe_eval import mcnemar_test, bootstrap_macro_f1_ci, gate_decision


def test_mcnemar_detects_directional_disagreement():
    truth = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    base = truth.copy(); base[:6] = (base[:6] + 1) % 4   # base wrong on 6
    new = truth.copy()                                    # new always right
    res = mcnemar_test(truth, base, new)
    assert res["n_new_right_base_wrong"] == 6
    assert res["n_base_right_new_wrong"] == 0
    assert 0.0 <= res["p_value"] <= 1.0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 4, size=400)
    pred = truth.copy()
    flip = rng.choice(400, size=80, replace=False)
    pred[flip] = (pred[flip] + 1) % 4
    lo, point, hi = bootstrap_macro_f1_ci(truth, pred, n_boot=200, seed=0)
    assert lo <= point <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_gate_proceeds_unless_flat_dead():
    # clear lift on endothelial -> proceed
    assert gate_decision(macro_delta=0.04, endo_delta=0.06, stromal_delta=0.01)["proceed"]
    # flat-dead: no macro gain AND <0.01 on both context classes -> stop
    d = gate_decision(macro_delta=-0.01, endo_delta=0.0, stromal_delta=0.005)
    assert not d["proceed"]
