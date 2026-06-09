"""BANKSY-style neighbour augmentation: concatenate each cell's own feature with
its neighbour mean and azimuthal-gradient magnitude (AGF). lambda_ weights the
neighbourhood vs the self term (lambda_=0 -> self only).

Edges are accumulated in chunks and features kept in float32 so this scales to the
probe's ~18M edges (2.3M cells x k=8) without materialising E x D temporaries that
would OOM the host. Chunking is exact -- np.add.at accumulation is associative, so
the result is independent of ``chunk``."""
from __future__ import annotations

import numpy as np


def banksy_augment(feats, edge_index, coords, lambda_: float = 0.5,
                   chunk: int = 2_000_000) -> np.ndarray:
    feats = np.asarray(feats, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float64)
    n, d = feats.shape
    src, dst = np.asarray(edge_index)
    deg = np.zeros(n, dtype=np.float32)
    nbr_mean = np.zeros((n, d), dtype=np.float32)
    agf = np.zeros((n, d), dtype=np.complex64)
    e = int(src.size)
    for lo in range(0, e, chunk):
        si = src[lo:lo + chunk]
        di = dst[lo:lo + chunk]
        theta = np.arctan2(coords[di, 1] - coords[si, 1],
                           coords[di, 0] - coords[si, 0])
        phase = np.exp(1j * theta).astype(np.complex64)[:, None]   # (chunk, 1)
        fdst = feats[di]                                            # (chunk, D)
        np.add.at(nbr_mean, si, fdst)
        np.add.at(agf, si, phase * (fdst - feats[si]))
        np.add.at(deg, si, np.float32(1.0))
    deg_safe = np.maximum(deg, 1.0)[:, None]
    nbr_mean = nbr_mean / deg_safe
    agf_mag = np.abs(agf / deg_safe).astype(np.float32)
    return np.concatenate([
        np.float32(np.sqrt(1.0 - lambda_)) * feats,
        np.float32(np.sqrt(lambda_ / 2.0)) * nbr_mean,
        np.float32(np.sqrt(lambda_ / 2.0)) * agf_mag,
    ], axis=1)
