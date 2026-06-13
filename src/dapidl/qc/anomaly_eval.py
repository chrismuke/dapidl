"""Validation + embedder selection for the QC anomaly score (pure)."""
from __future__ import annotations

import numpy as np


def auroc(scores, labels):
    """ROC AUC of `scores` predicting binary `labels` (1=positive). Rank-based, ties averaged."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def per_class_anomaly(score, coarse_idx, broken, endo_idx=0):
    """Mean+median anomaly per coarse class over NON-broken crops. Returns {cls: (mean, median)}."""
    score = np.asarray(score, dtype=np.float64)
    coarse_idx = np.asarray(coarse_idx)
    nb = ~np.asarray(broken, dtype=bool) & np.isfinite(score)
    out = {}
    for c in np.unique(coarse_idx[nb]):
        v = score[nb & (coarse_idx == c)]
        out[int(c)] = (float(np.mean(v)), float(np.median(v)))
    return out


def fairness_pass(table, endo_idx=0):
    """HARD gate: the rare Endothelial class must NOT have the highest mean anomaly."""
    if endo_idx not in table:
        return True
    means = {c: m for c, (m, _) in table.items()}
    return means[endo_idx] < max(means.values())


def select_embedder(results):
    """Among embedders that pass fairness, return the one with the highest AUROC, else None."""
    passing = {m: r for m, r in results.items() if r.get("fairness_pass")}
    if not passing:
        return None
    return max(passing, key=lambda m: passing[m]["auroc"])
