"""Nuclear shape descriptors from a 2-D point set (polygon vertices or mask pixels).
Pure numpy. angle is the major-axis direction in the same (x, y) frame as the input
points (so it composes with edge directions atan2(dy, dx))."""

from __future__ import annotations

import numpy as np


def ellipse_from_points(points_xy: np.ndarray) -> tuple[float, float]:
    """PCA of a 2-D point cloud. Returns (angle_rad, eccentricity). angle = atan2 of the
    principal eigenvector (major-axis direction, defined mod pi); eccentricity =
    sqrt(1 - lam_min/lam_max) in [0, 1). Degenerate input (< 3 distinct points or zero
    variance) -> (nan, 0.0)."""
    p = np.asarray(points_xy, dtype=float)
    if p.shape[0] < 3:
        return float("nan"), 0.0
    p = p - p.mean(0)
    cov = (p.T @ p) / p.shape[0]
    evals, evecs = np.linalg.eigh(cov)  # ascending
    lam_max, lam_min = float(evals[1]), float(evals[0])
    if lam_max <= 1e-12:
        return float("nan"), 0.0
    major = evecs[:, 1]  # eigenvector of the larger eigenvalue
    angle = float(np.arctan2(major[1], major[0]))
    ecc = float(np.sqrt(max(0.0, 1.0 - lam_min / lam_max)))
    return angle, ecc
