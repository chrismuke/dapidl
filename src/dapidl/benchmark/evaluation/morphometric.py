"""Morphometric evaluation metrics for segmentation masks.

Computes statistics about detected cell sizes, shapes, and segmentation quality
using scikit-image regionprops.
"""

from __future__ import annotations

import numpy as np
from skimage.measure import regionprops


def compute_morphometric_metrics(
    masks: np.ndarray,
    pixel_size_um: float = 0.108,
) -> dict:
    """Compute morphometric quality metrics from a segmentation mask.

    Parameters
    ----------
    masks:
        2D integer array where 0 = background and each positive integer labels
        one segmented cell.
    pixel_size_um:
        Physical size of one pixel in micrometres.

    Returns
    -------
    dict
        Keys:
        - n_detected: total number of detected cells
        - mean_area_um2, median_area_um2, std_area_um2: area statistics
        - mean_eccentricity: mean eccentricity across all cells (0=circle, 1=line)
        - mean_solidity: mean solidity (area / convex hull area)
        - size_outlier_rate: fraction of cells that are debris (<20 um2) or merged (>500 um2)
        - detection_density_per_1000um2: cells per 1000 um2 of image area
        - small_debris_count: cells with area < 20 um2
        - large_merged_count: cells with area > 500 um2
        - area_p10_um2, area_p90_um2: 10th and 90th percentile of cell area
        - eccentricity_p90: 90th percentile of eccentricity
        - solidity_p10: 10th percentile of solidity
    """
    pixel_area_um2 = pixel_size_um ** 2
    image_area_um2 = masks.size * pixel_area_um2

    props = regionprops(masks)
    n_detected = len(props)

    _zero: dict = {
        "n_detected": 0,
        "mean_area_um2": 0.0,
        "median_area_um2": 0.0,
        "std_area_um2": 0.0,
        "mean_eccentricity": 0.0,
        "mean_solidity": 0.0,
        "size_outlier_rate": 0.0,
        "detection_density_per_1000um2": 0.0,
        "small_debris_count": 0,
        "large_merged_count": 0,
        "area_p10_um2": 0.0,
        "area_p90_um2": 0.0,
        "eccentricity_p90": 0.0,
        "solidity_p10": 0.0,
    }

    if n_detected == 0:
        return _zero

    areas_px = np.array([p.area for p in props], dtype=float)
    areas_um2 = areas_px * pixel_area_um2

    eccentricities = np.array([p.eccentricity for p in props], dtype=float)
    solidities = np.array([p.solidity for p in props], dtype=float)

    small_debris_count = int(np.sum(areas_um2 < 20.0))
    large_merged_count = int(np.sum(areas_um2 > 500.0))
    size_outlier_rate = (small_debris_count + large_merged_count) / n_detected

    density = n_detected / image_area_um2 * 1000.0 if image_area_um2 > 0 else 0.0

    return {
        "n_detected": n_detected,
        "mean_area_um2": float(np.mean(areas_um2)),
        "median_area_um2": float(np.median(areas_um2)),
        "std_area_um2": float(np.std(areas_um2)),
        "mean_eccentricity": float(np.mean(eccentricities)),
        "mean_solidity": float(np.mean(solidities)),
        "size_outlier_rate": float(size_outlier_rate),
        "detection_density_per_1000um2": float(density),
        "small_debris_count": small_debris_count,
        "large_merged_count": large_merged_count,
        "area_p10_um2": float(np.percentile(areas_um2, 10)),
        "area_p90_um2": float(np.percentile(areas_um2, 90)),
        "eccentricity_p90": float(np.percentile(eccentricities, 90)),
        "solidity_p10": float(np.percentile(solidities, 10)),
    }
