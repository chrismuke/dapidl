"""Confident-Learning label cleaning for weak transcriptomic labels.

Given noisy (weak) labels and out-of-sample predicted probabilities, flag cells whose
label is systematically wrong (cleanlab), so the DAPI classifier can be trained on a
cleaner subset. Pure numpy + scikit-learn + cleanlab (no torch); see
docs/superpowers/specs/2026-06-06-label-quality-cleanlab-design.md.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def cv_pred_probs(X, y, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    """Out-of-sample class probabilities for (X, y) via k-fold cross-validation.

    Columns are ordered by ``np.unique(y)``. Folds are reduced if the rarest class
    has fewer than ``n_folds`` members so StratifiedKFold stays valid.
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    folds = max(2, min(n_folds, int(counts.min())))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    enc = {c: i for i, c in enumerate(classes)}
    y_enc = np.array([enc[v] for v in y])
    return cross_val_predict(clf, X, y_enc, cv=cv, method="predict_proba")


def find_label_issues(y, pred_probs) -> tuple[np.ndarray, np.ndarray]:
    """Flag label issues and score per-cell label quality via cleanlab.

    Returns ``(is_issue: bool[n], label_quality: float[n] in [0,1])``. ``y`` is
    encoded to 0..k-1 (sorted) to align with ``pred_probs`` columns.
    """
    from cleanlab.filter import find_label_issues as _find
    from cleanlab.rank import get_label_quality_scores as _quality

    y = np.asarray(y)
    classes = np.unique(y)
    enc = {c: i for i, c in enumerate(classes)}
    y_enc = np.array([enc[v] for v in y])
    is_issue = _find(labels=y_enc, pred_probs=pred_probs, return_indices_ranked_by=None)
    quality = _quality(labels=y_enc, pred_probs=pred_probs)
    return np.asarray(is_issue, dtype=bool), np.asarray(quality, dtype=float)
