"""Pure orchestration helpers for the subnuclear-triangulation scripts (LMDB index
selection, balanced subsetting, feature-column filters). No I/O, no GPU."""

from __future__ import annotations

import numpy as np


def select_pass_indices(sources, labels, keep_sources, max_per_source=None,
                        seed=0, limit=None, drop_unlabeled=False) -> np.ndarray:
    """Global LMDB indices for the chosen sources, optionally capped per source
    (seeded), label -1 optionally dropped, optionally truncated to ``limit``.
    Returns a sorted int array."""
    sources = np.asarray(sources)
    labels = np.asarray(labels)
    keep = np.isin(sources, list(keep_sources))
    if drop_unlabeled:
        keep &= labels != -1
    idx = np.where(keep)[0]
    if max_per_source is not None:
        rng = np.random.default_rng(seed)
        picked = []
        for s in keep_sources:
            si = idx[sources[idx] == s]
            if len(si) > max_per_source:
                si = rng.choice(si, size=max_per_source, replace=False)
            picked.append(si)
        # picked is empty only when keep_sources is empty (then idx is already
        # empty); return an explicit empty array rather than the uncapped idx.
        idx = np.concatenate(picked) if picked else np.array([], dtype=int)
    idx = np.sort(idx)
    if limit is not None:
        idx = idx[:limit]
    return idx


def balanced_subset(labels, per_class, seed=0) -> np.ndarray:
    """Up to ``per_class`` indices per non-negative class label (seeded). Sorted."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    picked = []
    for c in np.unique(labels):
        if c == -1:
            continue
        ci = np.where(labels == c)[0]
        if len(ci) > per_class:
            ci = rng.choice(ci, size=per_class, replace=False)
        picked.append(ci)
    return np.sort(np.concatenate(picked)) if picked else np.array([], dtype=int)


def nuc_feature_columns(columns) -> list[str]:
    return [c for c in columns if c.startswith("nuc_")]


def ctx_feature_columns(columns) -> list[str]:
    return [c for c in columns if c.startswith("ctx_")]
