import numpy as np
from dapidl.graph.geometry import ellipse_from_points


def _ellipse_pts(a, b, theta, n=200):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.stack([a * np.cos(t), b * np.sin(t)], 1)  # axis-aligned, semi-axes a>b
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return xy @ R.T  # rotated by theta


def test_axis_aligned_angle_and_eccentricity():
    ang, ecc = ellipse_from_points(_ellipse_pts(10.0, 2.0, 0.0))
    assert (
        abs(((ang + np.pi / 2) % np.pi) - np.pi / 2) < 0.05
    )  # major axis ~ x-axis (angle ~0 mod pi)
    assert 0.9 < ecc < 1.0  # elongated -> high eccentricity


def test_rotation_equivariant_angle_invariant_ecc():
    phi = 0.7
    a0, e0 = ellipse_from_points(_ellipse_pts(10.0, 3.0, 0.0))
    a1, e1 = ellipse_from_points(_ellipse_pts(10.0, 3.0, phi))
    # angle rotates by phi (mod pi); eccentricity unchanged
    assert abs(((a1 - a0 - phi) % np.pi)) < 0.02 or abs(((a1 - a0 - phi) % np.pi) - np.pi) < 0.02
    assert abs(e1 - e0) < 0.01


def test_degenerate_returns_nan_angle():
    ang, ecc = ellipse_from_points(np.zeros((2, 2)))  # < 3 points
    assert np.isnan(ang) and ecc == 0.0
    ang2, ecc2 = ellipse_from_points(np.zeros((5, 2)))  # zero variance
    assert np.isnan(ang2) and ecc2 == 0.0
