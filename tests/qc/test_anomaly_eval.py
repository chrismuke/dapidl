import numpy as np

from dapidl.qc.anomaly_eval import auroc, fairness_pass, per_class_anomaly, select_embedder


def test_auroc_perfect_and_chance():
    broken = np.array([0, 0, 1, 1])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), broken) == 1.0
    assert abs(auroc(np.array([0.5, 0.5, 0.5, 0.5]), broken) - 0.5) < 1e-9


def test_fairness_fails_when_endothelial_is_most_anomalous():
    coarse = np.array([0, 0, 1, 1, 2, 2])   # 0 = Endothelial
    broken = np.zeros(6, dtype=bool)
    score = np.array([0.9, 0.95, 0.1, 0.1, 0.2, 0.2])  # class 0 highest
    table = per_class_anomaly(score, coarse, broken, endo_idx=0)
    assert fairness_pass(table, endo_idx=0) is False


def test_select_embedder_picks_passing_higher_auroc():
    results = {
        "dinov2_vitb14": {"auroc": 0.71, "fairness_pass": True},
        "nuspire": {"auroc": 0.80, "fairness_pass": False},   # disqualified
    }
    assert select_embedder(results) == "dinov2_vitb14"
    assert select_embedder({"a": {"auroc": 0.6, "fairness_pass": False}}) is None
