"""Rotation-invariant edge attributes for the GATv2 arm. (N, k, 8) per neighbour slot:
RBF(dist) x3, cos/sin(2*(edge_angle - axis_i)), |cos(axis_i - axis_j)|, ecc_j,
|log_area_i - log_area_j|. Pure numpy. node_geom is (N,3)=[angle, ecc, log_area];
angle may be nan (no orientation) -> the 3 directional terms are zeroed for that edge."""

from __future__ import annotations

import numpy as np


def rbf(dist: np.ndarray, centers, gamma: float) -> np.ndarray:
    """Gaussian RBF expansion: exp(-gamma (dist - c)^2) for each center c. -> (..., len(centers))."""
    d = np.asarray(dist)[..., None]
    c = np.asarray(centers, dtype=float)
    return np.exp(-gamma * (d - c) ** 2)


def build_edge_attr(
    coords: np.ndarray,
    node_geom: np.ndarray,
    nbr: np.ndarray,
    rbf_centers=(4.0, 8.0, 16.0),
    rbf_gamma: float = 0.05,
) -> np.ndarray:
    """coords (N,2); node_geom (N,3)=[angle,ecc,log_area]; nbr (N,k) global indices (-1 pad).
    Returns edge_attr (N, k, 8), all rotation-invariant. -1 slots -> zero rows."""
    coords = np.asarray(coords, dtype=float)
    ng = np.asarray(node_geom, dtype=float)
    nbr = np.asarray(nbr)
    n, k = nbr.shape
    valid = nbr >= 0
    j = np.where(valid, nbr, 0)  # safe gather index
    src = np.broadcast_to(np.arange(n)[:, None], (n, k))

    d_xy = coords[j] - coords[src]  # (N,k,2) from i to j
    dist = np.linalg.norm(d_xy, axis=2)  # (N,k)
    edge_angle = np.arctan2(d_xy[..., 1], d_xy[..., 0])  # (N,k)
    axis_i = ng[src, 0]
    axis_j = ng[j, 0]  # (N,k)
    ecc_j = ng[j, 1]
    dlog_area = np.abs(ng[src, 2] - ng[j, 2])

    rbf_feats = rbf(dist, rbf_centers, rbf_gamma)  # (N,k,3)
    rel = edge_angle - axis_i  # nan where axis_i is nan
    cos2 = np.cos(2.0 * rel)
    sin2 = np.sin(2.0 * rel)
    align = np.abs(np.cos(axis_i - axis_j))
    directional_nan = ~np.isfinite(cos2) | ~np.isfinite(align)  # either endpoint axis nan
    cos2 = np.where(directional_nan, 0.0, cos2)
    sin2 = np.where(directional_nan, 0.0, sin2)
    align = np.where(directional_nan, 0.0, align)

    ea = np.concatenate(
        [
            rbf_feats,
            cos2[..., None],
            sin2[..., None],
            align[..., None],
            ecc_j[..., None],
            dlog_area[..., None],
        ],
        axis=2,
    ).astype(np.float32)
    ea[~valid] = 0.0  # zero padded slots
    return ea
