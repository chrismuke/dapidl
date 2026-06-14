"""Post-hoc label smoothing over the spatial graph: a row-stochastic random-walk
transition matrix, PPR smoothing, and Correct-and-Smooth (Huang et al. 2020). Pure
numpy/scipy. Smoothing of a probability matrix preserves the simplex (convex update)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def transition_matrix(edge_index: np.ndarray, n: int) -> sp.csr_matrix:
    """Row-stochastic random-walk transition D^-1 (A_sym + I) with self-loops, built
    from a directed edge_index (2, E) of global node indices. Self-loops make every
    node (including isolated ones) a defined, simplex-preserving update."""
    src, dst = edge_index
    A = sp.coo_matrix((np.ones(src.shape[0]), (src, dst)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64)        # symmetrize, binary
    A = (A + sp.identity(n, format="coo")).tocsr()  # self-loops
    deg = np.asarray(A.sum(1)).ravel()
    Dinv = sp.diags(1.0 / deg)
    return (Dinv @ A).tocsr()


def smooth(probs: np.ndarray, transition: sp.csr_matrix, alpha: float, iters: int) -> np.ndarray:
    """PPR diffusion p <- (1-alpha) p0 + alpha (T @ p), `iters` times. alpha=0 is the
    identity; with row-stochastic T and probs rows on the simplex, rows stay on it."""
    p0 = np.asarray(probs, dtype=np.float64)
    p = p0.copy()
    for _ in range(iters):
        p = (1.0 - alpha) * p0 + alpha * (transition @ p)
    return p


def correct_and_smooth(probs, train_idx, train_labels, transition, num_classes: int = 4,
                       alpha_correct: float = 0.8, alpha_smooth: float = 0.8,
                       iters: int = 30) -> np.ndarray:
    """Correct-and-Smooth. Correct: diffuse the train residual (one-hot truth - prob)
    and add it back. Smooth: diffuse the corrected probs. When `train_idx` is empty
    (held-out slide shares no labels with its within-slide graph) the Correct step is a
    no-op and this reduces to smoothing-only."""
    probs = np.asarray(probs, dtype=np.float64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    resid = np.zeros_like(probs)
    if len(train_idx) > 0:
        onehot = np.zeros((len(train_idx), num_classes))
        onehot[np.arange(len(train_idx)), np.asarray(train_labels, dtype=np.int64)] = 1.0
        resid[train_idx] = onehot - probs[train_idx]
    corrected = probs + smooth(resid, transition, alpha_correct, iters)
    corrected = np.clip(corrected, 0.0, None)
    rs = corrected.sum(1, keepdims=True)
    corrected = np.divide(corrected, rs, out=np.zeros_like(corrected), where=rs > 0)
    return smooth(corrected, transition, alpha_smooth, iters)
