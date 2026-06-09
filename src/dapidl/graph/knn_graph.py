"""Within-slide k-nearest-neighbour graph over cell centroids. Edges never cross
slides (each tissue section is its own connected component); no self-loops."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def build_within_slide_knn(coords, slide_ids, k: int = 8) -> np.ndarray:
    """Directed edge_index (2, E) of GLOBAL node indices. Edge (src, dst) means
    ``dst`` is one of ``src``'s k nearest neighbours on the same slide. Slides with
    a single cell contribute no edges; k is capped at (slide_size - 1)."""
    coords = np.asarray(coords, dtype=float)
    slide_ids = np.asarray(slide_ids)
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    for s in np.unique(slide_ids):
        idx = np.where(slide_ids == s)[0]
        if len(idx) < 2:
            continue
        kk = min(k, len(idx) - 1)
        tree = cKDTree(coords[idx])
        _, nn = tree.query(coords[idx], k=kk + 1)  # col 0 is the node itself
        nn = np.atleast_2d(nn)
        for col in range(1, kk + 1):
            src_parts.append(idx)
            dst_parts.append(idx[nn[:, col]])
    if not src_parts:
        return np.empty((2, 0), dtype=np.int64)
    return np.stack([np.concatenate(src_parts), np.concatenate(dst_parts)]).astype(np.int64)
