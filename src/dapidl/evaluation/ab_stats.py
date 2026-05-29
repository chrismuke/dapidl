"""Paired A/B statistics for the gnp-v1 readout (review 2026-05-29, Phase 2).

Pure functions over saved per-cell predictions — no training, no GPU. They make a
single-seed A/B comparison defensible:

- ``mcnemar`` — the paired test that is valid when each arm is trained ONCE
  (Dietterich 1998). It looks only at discordant cells (one model right, the other
  wrong), so it answers "is B reliably different from A on the same cells?".
- ``bootstrap_macro_f1_diff`` — a paired bootstrap CI on macroF1(B) − macroF1(A);
  the decision becomes "does the 95% CI of the difference exclude 0?" instead of
  "is the point delta > 2%?" (which is inside single-seed training noise).
- ``per_class_f1_ci`` — per-class F1 with bootstrap CIs + support, so a 2-pt move
  on a rare class (Stromal ~0.2%) is read with its (wide) uncertainty.

All take integer class labels in ``range(K)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, chi2


@dataclass(frozen=True)
class McNemarResult:
    n01: int          # A wrong, B correct (cells where B wins)
    n10: int          # A correct, B wrong (cells where A wins)
    statistic: float
    p_value: float
    method: str       # "exact" (binomial) | "chi2_cc" (continuity-corrected chi-square)


def macro_f1_fast(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Macro-F1 over labels 0..k-1 via a bincount confusion matrix (fast for bootstrap).

    Absent classes contribute F1=0 (matches sklearn ``zero_division=0`` with explicit
    ``labels=range(k)``) so the average is always over all k classes.
    """
    yt = np.asarray(y_true).astype(np.int64)
    yp = np.asarray(y_pred).astype(np.int64)
    conf = np.bincount(yt * k + yp, minlength=k * k).reshape(k, k).astype(np.float64)
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    f1 = np.where(denom > 0, 2.0 * tp / denom, 0.0)
    return float(f1.mean())


def mcnemar(y_true, y_pred_a, y_pred_b, exact_threshold: int = 25) -> McNemarResult:
    """Paired McNemar test on two classifiers' predictions over the SAME cells.

    Exact binomial p-value when the discordant count is small (< exact_threshold),
    else a continuity-corrected chi-square (df=1). Two-sided.
    """
    yt = np.asarray(y_true)
    a = np.asarray(y_pred_a)
    b = np.asarray(y_pred_b)
    if not (len(yt) == len(a) == len(b)):
        raise ValueError("y_true, y_pred_a, y_pred_b must be the same length")
    a_correct = a == yt
    b_correct = b == yt
    n01 = int(np.sum(~a_correct & b_correct))   # B right, A wrong
    n10 = int(np.sum(a_correct & ~b_correct))   # A right, B wrong
    n = n01 + n10
    if n == 0:
        return McNemarResult(n01, n10, 0.0, 1.0, "exact")
    if n < exact_threshold:
        p = float(binomtest(min(n01, n10), n, 0.5).pvalue)
        return McNemarResult(n01, n10, float(min(n01, n10)), p, "exact")
    stat = (abs(n01 - n10) - 1.0) ** 2 / n
    return McNemarResult(n01, n10, float(stat), float(chi2.sf(stat, df=1)), "chi2_cc")


def bootstrap_macro_f1_diff(
    y_true, y_pred_a, y_pred_b, k: int,
    n_boot: int = 2000, seed: int = 0, ci: float = 0.95,
) -> dict:
    """Paired bootstrap CI for macroF1(B) − macroF1(A). Resamples cells (paired,
    so A and B see the same resample each iteration)."""
    yt = np.asarray(y_true)
    a = np.asarray(y_pred_a)
    b = np.asarray(y_pred_b)
    n = len(yt)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = macro_f1_fast(yt[idx], b[idx], k) - macro_f1_fast(yt[idx], a[idx], k)
    alpha = (1.0 - ci) / 2.0
    lo, hi = (float(x) for x in np.quantile(diffs, [alpha, 1.0 - alpha]))
    point = macro_f1_fast(yt, b, k) - macro_f1_fast(yt, a, k)
    return {
        "diff": float(point), "ci_lo": lo, "ci_hi": hi,
        "p_excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "n_boot": int(n_boot), "ci": float(ci),
    }


def per_class_f1_ci(
    y_true, y_pred, k: int, class_names: list[str] | None = None,
    n_boot: int = 2000, seed: int = 0, ci: float = 0.95,
) -> list[dict]:
    """Per-class F1 point estimate + bootstrap CI + support."""
    yt = np.asarray(y_true).astype(np.int64)
    yp = np.asarray(y_pred).astype(np.int64)
    n = len(yt)
    names = class_names or [str(c) for c in range(k)]
    rng = np.random.default_rng(seed)

    def _per_class_f1(t, p):
        conf = np.bincount(t * k + p, minlength=k * k).reshape(k, k).astype(np.float64)
        tp = np.diag(conf)
        fp = conf.sum(axis=0) - tp
        fn = conf.sum(axis=1) - tp
        denom = 2.0 * tp + fp + fn
        return np.where(denom > 0, 2.0 * tp / denom, 0.0)

    point = _per_class_f1(yt, yp)
    boot = np.empty((n_boot, k), dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = _per_class_f1(yt[idx], yp[idx])
    alpha = (1.0 - ci) / 2.0
    lo = np.quantile(boot, alpha, axis=0)
    hi = np.quantile(boot, 1.0 - alpha, axis=0)
    support = np.bincount(yt, minlength=k)
    return [
        {"class_idx": c, "class_name": names[c], "f1": float(point[c]),
         "ci_lo": float(lo[c]), "ci_hi": float(hi[c]), "support": int(support[c])}
        for c in range(k)
    ]
