"""BANKSY-style neighbour augmentation: concatenate each cell's own feature with
its neighbour mean and azimuthal-gradient magnitude (AGF). lambda_ weights the
neighbourhood vs the self term (lambda_=0 -> self only)."""
from __future__ import annotations

import numpy as np


def banksy_augment(feats, edge_index, coords, lambda_: float = 0.5) -> np.ndarray:
    feats = np.asarray(feats, dtype=float)
    coords = np.asarray(coords, dtype=float)
    n, d = feats.shape
    src, dst = np.asarray(edge_index)
    deg = np.zeros(n)
    nbr_mean = np.zeros((n, d))
    agf = np.zeros((n, d), dtype=complex)
    if src.size:
        theta = np.arctan2(coords[dst, 1] - coords[src, 1],
                           coords[dst, 0] - coords[src, 0])
        phase = np.exp(1j * theta)[:, None]            # (E, 1)
        diff = feats[dst] - feats[src]                  # (E, D)
        np.add.at(nbr_mean, src, feats[dst])
        np.add.at(agf, src, phase * diff)
        np.add.at(deg, src, 1.0)
    deg_safe = np.maximum(deg, 1.0)[:, None]
    nbr_mean = nbr_mean / deg_safe
    agf_mag = np.abs(agf / deg_safe)
    return np.concatenate([
        np.sqrt(1.0 - lambda_) * feats,
        np.sqrt(lambda_ / 2.0) * nbr_mean,
        np.sqrt(lambda_ / 2.0) * agf_mag,
    ], axis=1)
