"""Tests for paired A/B statistics (review Phase 2): McNemar, bootstrap CIs, per-class F1 CIs."""
import numpy as np

from dapidl.evaluation.ab_stats import (
    bootstrap_macro_f1_diff,
    macro_f1_fast,
    mcnemar,
    per_class_f1_ci,
)

K = 4


def test_macro_f1_fast_matches_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    yt = rng.integers(0, K, 500)
    yp = rng.integers(0, K, 500)
    expect = f1_score(yt, yp, labels=list(range(K)), average="macro", zero_division=0)
    assert abs(macro_f1_fast(yt, yp, K) - expect) < 1e-9


def test_mcnemar_identical_predictions_p1():
    yt = np.array([0, 1, 2, 3, 0, 1])
    r = mcnemar(yt, yt.copy(), yt.copy())
    assert r.n01 == 0 and r.n10 == 0 and r.p_value == 1.0


def test_mcnemar_b_strictly_better_chi2_path():
    yt = np.zeros(30, int)
    a = np.ones(30, int)    # all wrong
    b = np.zeros(30, int)   # all right
    r = mcnemar(yt, a, b)
    assert r.n01 == 30 and r.n10 == 0       # 30 cells where b right, a wrong
    assert r.method == "chi2_cc"            # n >= exact_threshold
    assert r.p_value < 1e-6


def test_mcnemar_exact_path_small_discordance():
    yt = np.zeros(10, int)
    a = np.array([1] * 5 + [0] * 5)         # 5 wrong, 5 right
    b = np.zeros(10, int)                   # all right
    r = mcnemar(yt, a, b)
    assert r.n01 == 5 and r.n10 == 0 and r.method == "exact"
    assert 0.0 < r.p_value < 0.1


def test_bootstrap_diff_zero_when_identical():
    rng = np.random.default_rng(1)
    yt = rng.integers(0, K, 400)
    yp = rng.integers(0, K, 400)
    out = bootstrap_macro_f1_diff(yt, yp, yp, K, n_boot=200, seed=0)
    assert out["diff"] == 0.0
    assert out["ci_lo"] <= 0.0 <= out["ci_hi"]
    assert out["p_excludes_zero"] is False


def test_bootstrap_diff_positive_when_b_better():
    yt = np.tile(np.arange(K), 100)         # 400, balanced
    a = (yt + 1) % K                        # all wrong -> macroF1 ~0
    b = yt.copy()                           # perfect -> macroF1 1
    out = bootstrap_macro_f1_diff(yt, a, b, K, n_boot=200, seed=0)
    assert out["diff"] > 0.9
    assert out["ci_lo"] > 0.0 and out["p_excludes_zero"] is True


def test_per_class_f1_ci_brackets_point_estimate():
    yt = np.tile(np.arange(K), 100)
    yp = yt.copy()
    rows = per_class_f1_ci(yt, yp, K, class_names=["a", "b", "c", "d"],
                           n_boot=200, seed=0)
    assert len(rows) == K
    for r in rows:
        assert r["support"] == 100
        assert r["f1"] == 1.0
        assert r["ci_lo"] <= r["f1"] <= r["ci_hi"] + 1e-9
        assert r["class_name"] in {"a", "b", "c", "d"}
