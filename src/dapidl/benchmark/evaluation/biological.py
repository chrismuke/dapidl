"""Biological evaluation metrics for segmentation masks.

Compares segmentation results against native (ground-truth) cell centroids from
spatial transcriptomics platforms such as MERSCOPE/Xenium.
"""

from __future__ import annotations

import numpy as np


def compute_biological_metrics(
    masks: np.ndarray,
    native_centroids: np.ndarray,
    pixel_size_um: float = 0.108,
    search_radius_px: int = 20,
) -> dict:
    """Compute biological quality metrics by matching segmentation to native centroids.

    For each native centroid [y, x] in pixel coordinates:
      1. Look up masks[iy, ix]. If > 0, the cell is recovered.
      2. If 0, search nearby pixels (step=2, within search_radius_px) for any
         non-zero mask label.
      3. If still 0, the cell is not recovered.

    Parameters
    ----------
    masks:
        2D integer array where 0 = background and positive integers label cells.
    native_centroids:
        Array of shape (N, 2) containing [y, x] centroid coordinates in pixels.
    pixel_size_um:
        Physical size of one pixel in micrometres (reserved for future use).
    search_radius_px:
        Radius in pixels for the nearby search when the direct lookup fails.

    Returns
    -------
    dict
        Keys:
        - n_native: total number of native centroids
        - n_recovered: number of native centroids that map to a non-zero mask label
        - native_recovery_rate: n_recovered / n_native  (0.0 when n_native == 0)
        - underseg_rate: fraction of unique recovered mask labels that contain
          more than one native centroid (multiple natives in one segment)
        - split_cell_rate: max(0, (n_segments - n_native) / n_native)
        - n_segments: total number of unique non-zero labels in masks
        - segments_per_native: n_segments / n_native  (0.0 when n_native == 0)
    """
    h, w = masks.shape
    n_native = len(native_centroids)

    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]
    n_segments = int(len(unique_labels))

    _zero: dict = {
        "n_native": n_native,
        "n_recovered": 0,
        "native_recovery_rate": 0.0,
        "underseg_rate": 0.0,
        "split_cell_rate": 0.0,
        "n_segments": n_segments,
        "segments_per_native": 0.0,
    }

    if n_native == 0:
        return _zero

    # Build search offsets once (step=2, within radius)
    offsets: list[tuple[int, int]] = []
    for dy in range(-search_radius_px, search_radius_px + 1, 2):
        for dx in range(-search_radius_px, search_radius_px + 1, 2):
            if dy == 0 and dx == 0:
                continue
            if dy * dy + dx * dx <= search_radius_px * search_radius_px:
                offsets.append((dy, dx))

    recovered_labels: list[int] = []  # label found for each native centroid (0 = miss)

    for centroid in native_centroids:
        iy = int(round(float(centroid[0])))
        ix = int(round(float(centroid[1])))
        iy = max(0, min(iy, h - 1))
        ix = max(0, min(ix, w - 1))

        label = int(masks[iy, ix])

        if label == 0:
            # Nearby search
            for dy, dx in offsets:
                ny = iy + dy
                nx = ix + dx
                if 0 <= ny < h and 0 <= nx < w:
                    nb_label = int(masks[ny, nx])
                    if nb_label > 0:
                        label = nb_label
                        break

        recovered_labels.append(label)

    recovered_arr = np.array(recovered_labels, dtype=np.int32)
    n_recovered = int(np.sum(recovered_arr > 0))
    native_recovery_rate = n_recovered / n_native

    # Undersegmentation: how many unique recovered labels contain >1 native centroid
    recovered_nonzero = recovered_arr[recovered_arr > 0]
    if len(recovered_nonzero) > 0:
        unique_recovered, counts = np.unique(recovered_nonzero, return_counts=True)
        n_underseg = int(np.sum(counts > 1))
        underseg_rate = n_underseg / len(unique_recovered)
    else:
        underseg_rate = 0.0

    split_cell_rate = max(0.0, (n_segments - n_native) / n_native)
    segments_per_native = n_segments / n_native

    return {
        "n_native": n_native,
        "n_recovered": n_recovered,
        "native_recovery_rate": float(native_recovery_rate),
        "underseg_rate": float(underseg_rate),
        "split_cell_rate": float(split_cell_rate),
        "n_segments": n_segments,
        "segments_per_native": float(segments_per_native),
    }
