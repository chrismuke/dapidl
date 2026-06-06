"""Tests for cleanlab-based label cleaning (pure logic, no torch/network)."""
import numpy as np
from sklearn.datasets import make_classification

from dapidl.data import label_cleaning as lc


def _noisy_dataset(seed=0, flip_frac=0.10):
    X, y = make_classification(
        n_samples=900, n_features=20, n_informative=12,
        n_classes=3, n_clusters_per_class=1, random_state=seed,
    )
    rng = np.random.default_rng(seed)
    flip_idx = rng.choice(len(y), size=int(flip_frac * len(y)), replace=False)
    y_noisy = y.copy()
    for i in flip_idx:
        y_noisy[i] = (y[i] + rng.integers(1, 3)) % 3  # force a different label
    return X, y, y_noisy, set(flip_idx.tolist())


def test_cv_pred_probs_shape_and_normalized():
    X, y, _, _ = _noisy_dataset()
    p = lc.cv_pred_probs(X, y, n_folds=5, seed=0)
    assert p.shape == (len(y), 3)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_find_label_issues_recovers_injected_flips():
    X, y_true, y_noisy, flipped = _noisy_dataset()
    probs = lc.cv_pred_probs(X, y_noisy, n_folds=5, seed=0)
    is_issue, quality = lc.find_label_issues(y_noisy, probs)
    flagged = set(np.where(is_issue)[0].tolist())
    recall = len(flagged & flipped) / len(flipped)
    precision = len(flagged & flipped) / max(len(flagged), 1)
    assert recall >= 0.6, f"recall={recall:.2f}"
    assert precision >= 0.6, f"precision={precision:.2f}"
    assert quality.shape == (len(y_noisy),)
    assert quality.min() >= 0.0 and quality.max() <= 1.0
