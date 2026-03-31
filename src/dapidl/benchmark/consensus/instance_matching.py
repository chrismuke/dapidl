"""Instance-level matching between segmentation outputs using IoU.

Provides tools for matching detected cell instances across two segmentation
masks using the Hungarian algorithm on an IoU cost matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _compute_iou_matrix(masks_a: np.ndarray, masks_b: np.ndarray) -> np.ndarray:
    """Compute an (Na, Nb) IoU matrix between all instance pairs.

    Parameters
    ----------
    masks_a:
        2D integer label image, 0 = background.
    masks_b:
        2D integer label image, same shape as masks_a.

    Returns
    -------
    np.ndarray
        Shape (Na, Nb) where Na = number of unique positive labels in masks_a
        and Nb = number of unique positive labels in masks_b.
        iou_matrix[i, j] = IoU between instance (i+1) in a and instance (j+1) in b,
        using label indices sorted in ascending order.
    """
    labels_a = np.unique(masks_a)
    labels_a = labels_a[labels_a > 0]
    labels_b = np.unique(masks_b)
    labels_b = labels_b[labels_b > 0]

    na = len(labels_a)
    nb = len(labels_b)

    if na == 0 or nb == 0:
        return np.zeros((na, nb), dtype=np.float64)

    iou_matrix = np.zeros((na, nb), dtype=np.float64)

    for i, la in enumerate(labels_a):
        mask_a = masks_a == la
        area_a = mask_a.sum()
        for j, lb in enumerate(labels_b):
            mask_b = masks_b == lb
            intersection = np.logical_and(mask_a, mask_b).sum()
            if intersection == 0:
                continue
            area_b = mask_b.sum()
            union = area_a + area_b - intersection
            iou_matrix[i, j] = intersection / union if union > 0 else 0.0

    return iou_matrix


def match_instances_iou(
    masks_a: np.ndarray,
    masks_b: np.ndarray,
    iou_threshold: float = 0.3,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Match instances between two label masks via the Hungarian algorithm.

    Builds a cost matrix from (1 - IoU), runs linear_sum_assignment, and
    returns only those matches where the IoU exceeds the threshold.

    Parameters
    ----------
    masks_a:
        2D integer label image, 0 = background.
    masks_b:
        2D integer label image, same shape as masks_a.
    iou_threshold:
        Minimum IoU for a match to be accepted. Default 0.3.

    Returns
    -------
    matches:
        List of (label_a, label_b) integer pairs that were matched above threshold.
    ious:
        Corresponding IoU values for each matched pair.
    """
    labels_a = np.unique(masks_a)
    labels_a = labels_a[labels_a > 0]
    labels_b = np.unique(masks_b)
    labels_b = labels_b[labels_b > 0]

    iou_matrix = _compute_iou_matrix(masks_a, masks_b)

    if iou_matrix.size == 0:
        return [], []

    cost_matrix = 1.0 - iou_matrix
    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    matches: list[tuple[int, int]] = []
    ious: list[float] = []

    for r, c in zip(row_idx, col_idx):
        iou_val = iou_matrix[r, c]
        if iou_val >= iou_threshold:
            matches.append((int(labels_a[r]), int(labels_b[c])))
            ious.append(float(iou_val))

    return matches, ious
