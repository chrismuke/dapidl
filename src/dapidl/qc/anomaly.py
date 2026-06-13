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
