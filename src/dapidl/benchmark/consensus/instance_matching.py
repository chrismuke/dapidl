"""Instance-level matching between segmentation outputs using IoU.

Provides tools for matching detected cell instances across two segmentation
masks using the Hungarian algorithm on an IoU cost matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _compute_iou_matrix(masks_a: np.ndarray, masks_b: np.ndarray) -> np.ndarray:
    """Compute an (Na, Nb) IoU matrix between all instance pairs.

    Uses a fast sparse overlap approach: flattens both masks, computes
    a confusion matrix via np.bincount, then derives IoU from the counts.
    Complexity is O(H*W + Na*Nb_overlap) instead of O(Na*Nb*H*W).

    Parameters
    ----------
    masks_a:
        2D integer label image, 0 = background.
    masks_b:
        2D integer label image, same shape as masks_a.

    Returns
    -------
    np.ndarray
        Shape (Na, Nb) IoU matrix.
    """
    labels_a = np.unique(masks_a)
    labels_a = labels_a[labels_a > 0]
    labels_b = np.unique(masks_b)
    labels_b = labels_b[labels_b > 0]

    na = len(labels_a)
    nb = len(labels_b)

    if na == 0 or nb == 0:
        return np.zeros((na, nb), dtype=np.float64)

    # Remap labels to contiguous 0..Na-1, 0..Nb-1
    remap_a = np.zeros(int(masks_a.max()) + 1, dtype=np.int32)
    for idx, la in enumerate(labels_a):
        remap_a[la] = idx + 1  # 1-indexed so 0 stays background
    remap_b = np.zeros(int(masks_b.max()) + 1, dtype=np.int32)
    for idx, lb in enumerate(labels_b):
        remap_b[lb] = idx + 1

    ra = remap_a[masks_a.ravel()]
    rb = remap_b[masks_b.ravel()]

    # Compute overlap via 2D histogram (confusion matrix)
    # Only count pixels where both are foreground
    fg_mask = (ra > 0) & (rb > 0)
    if not fg_mask.any():
        return np.zeros((na, nb), dtype=np.float64)

    ra_fg = ra[fg_mask]
    rb_fg = rb[fg_mask]

    # Confusion matrix: overlap[i, j] = # pixels where remapped_a==i and remapped_b==j
    overlap = np.zeros((na + 1, nb + 1), dtype=np.int64)
    np.add.at(overlap, (ra_fg, rb_fg), 1)

    # Slice off background row/col
    overlap = overlap[1:, 1:]  # (Na, Nb)

    # Area of each instance
    area_a = np.bincount(ra, minlength=na + 1)[1:]  # (Na,)
    area_b = np.bincount(rb, minlength=nb + 1)[1:]  # (Nb,)

    # IoU = intersection / union = overlap / (area_a + area_b - overlap)
    union = area_a[:, None] + area_b[None, :] - overlap
    iou_matrix = np.where(union > 0, overlap / union, 0.0).astype(np.float64)

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
