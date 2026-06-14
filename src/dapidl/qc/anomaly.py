"""Memory-bank kNN anomaly scoring for DAPI QC crops (pure numpy/sklearn — no torch)."""
from __future__ import annotations

import numpy as np


def select_bank_indices(rows, slides, coarse_idx, broken, grades, *,
                        exclude_slide, per_class_cap, min_grades=None, rng):
    """Indices into the arrays for the 'normal' memory bank.

    Non-broken crops NOT from `exclude_slide`, optionally restricted to `grades in min_grades`,
    capped at `per_class_cap` per coarse class. Deterministic given `rng`.
    """
    slides = np.asarray(slides)
    coarse_idx = np.asarray(coarse_idx)
    broken = np.asarray(broken, dtype=bool)
    grades = np.asarray(grades)

    keep = (~broken) & (slides != exclude_slide)
    if min_grades is not None:
        keep &= np.isin(grades, list(min_grades))
    cand = np.flatnonzero(keep)

    out = []
    for c in np.unique(coarse_idx[cand]):
        members = cand[coarse_idx[cand] == c]
        if len(members) > per_class_cap:
            members = rng.choice(members, size=per_class_cap, replace=False)
        out.append(members)
    return np.sort(np.concatenate(out)) if out else np.empty(0, dtype=int)


def _l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def knn_anomaly_score(query, bank, k):
    """Mean cosine distance of each query row to its k nearest bank rows. Higher = more anomalous."""
    q = _l2norm(np.asarray(query, dtype=np.float64))
    b = _l2norm(np.asarray(bank, dtype=np.float64))
    k = min(k, b.shape[0])
    sims = q @ b.T                      # cosine similarity [Q, B]
    part = np.partition(sims, -k, axis=1)[:, -k:]
    return (1.0 - part).mean(axis=1)


def coreset_subsample(emb, frac, rng):
    """Greedy k-center coreset subsample of rows of `emb`. frac>=1 returns all indices."""
    n = emb.shape[0]
    m = n if frac >= 1.0 else max(1, int(round(n * frac)))
    if m >= n:
        return np.arange(n)
    x = _l2norm(np.asarray(emb, dtype=np.float64))
    start = int(rng.integers(n))
    chosen = [start]
    mind = 1.0 - (x @ x[start])
    for _ in range(m - 1):
        nxt = int(np.argmax(mind))
        chosen.append(nxt)
        mind = np.minimum(mind, 1.0 - (x @ x[nxt]))
    return np.sort(np.array(chosen))


def score_all_slides_loso(emb, rows, slides, coarse_idx, broken, grades, *,
                          k, per_class_cap, rng, coreset_frac=1.0):
    """For each slide, build the bank from all OTHER slides and score this slide's crops.

    Leave-one-slide-out orchestration: scores all rows with per-slide exclusion from the memory bank.
    """
    slides = np.asarray(slides)
    out = np.full(len(rows), np.nan, dtype=np.float64)
    for s in np.unique(slides):
        bank_idx = select_bank_indices(rows, slides, coarse_idx, broken, grades,
                                        exclude_slide=s, per_class_cap=per_class_cap, rng=rng)
        if len(bank_idx) == 0:
            continue
        bank_emb = emb[bank_idx]
        if coreset_frac < 1.0:
            bank_emb = bank_emb[coreset_subsample(bank_emb, coreset_frac, rng)]
        q = np.flatnonzero(slides == s)
        out[q] = knn_anomaly_score(emb[q], bank_emb, k=k)
    return out
