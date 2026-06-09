"""Probe evaluation helpers: McNemar paired test (new vs baseline predictions),
bootstrap CI on macro-F1, and the Stage-1 -> Stage-2 gate decision."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score


def mcnemar_test(truth, base_pred, new_pred) -> dict:
    """Exact McNemar (binomial) on the discordant pairs where exactly one model is
    right. Reports the two discordant counts and the two-sided p-value."""
    truth = np.asarray(truth); base_pred = np.asarray(base_pred); new_pred = np.asarray(new_pred)
    base_ok = base_pred == truth
    new_ok = new_pred == truth
    b = int(np.sum(new_ok & ~base_ok))   # new right, base wrong
    c = int(np.sum(base_ok & ~new_ok))   # base right, new wrong
    n = b + c
    p = binomtest(b, n, 0.5).pvalue if n > 0 else 1.0
    return {"n_new_right_base_wrong": b, "n_base_right_new_wrong": c, "p_value": float(p)}


def bootstrap_macro_f1_ci(truth, pred, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """(lo, point, hi) macro-F1 with a percentile bootstrap over resampled cells."""
    truth = np.asarray(truth); pred = np.asarray(pred)
    point = f1_score(truth, pred, average="macro", zero_division=0)
    rng = np.random.default_rng(seed)
    n = len(truth)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, size=n)
        boots[i] = f1_score(truth[s], pred[s], average="macro", zero_division=0)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(point), float(hi)


def gate_decision(macro_delta: float, endo_delta: float, stromal_delta: float) -> dict:
    """Proceed to Stage 2 UNLESS Stage 1 is flat-dead: no macro gain AND <0.01 on
    BOTH context classes. (Permissive: Stage 2 is the real test.)"""
    flat_dead = (macro_delta <= 0.0) and (endo_delta < 0.01) and (stromal_delta < 0.01)
    return {"proceed": not flat_dead,
            "reason": "flat-dead: no spatial signal" if flat_dead else "spatial signal present"}
