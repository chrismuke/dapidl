"""Majority voting consensus for combining segmentation outputs.

Each pixel votes for foreground if the majority (min_agreement fraction)
of methods detect it as part of a cell. Connected components are then
labelled to produce the final instance mask.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label as ndimage_label

from dapidl.benchmark.segmenters.base import SegmentationOutput


def majority_voting_consensus(
    results: dict[str, SegmentationOutput],
    min_agreement: float = 0.5,
) -> SegmentationOutput:
    """Pixel-level foreground majority voting across segmentation methods.

    Each method contributes a binary foreground vote per pixel. Pixels
    with a fraction of votes >= min_agreement are kept as foreground.
    Connected components of the foreground are labelled to form instances.

    Parameters
    ----------
    results:
        Mapping from method name to its SegmentationOutput.
    min_agreement:
        Minimum fraction of methods that must agree on a pixel being
        foreground for it to be included. Default 0.5.

    Returns
    -------
    SegmentationOutput
        Consensus mask with method_name="consensus_majority".
        n_cells is the number of connected components found.
        centroids are the centroids of those components.
        runtime_seconds and peak_memory_mb are set to 0.0.
    """
    if not results:
        raise ValueError("results dict must not be empty")

    method_list = list(results.values())
    shape = method_list[0].masks.shape
    n_methods = len(method_list)

    vote_sum = np.zeros(shape, dtype=np.float32)
    for output in method_list:
        vote_sum += (output.masks > 0).astype(np.float32)

    threshold = min_agreement * n_methods
    foreground = vote_sum >= threshold

    labelled, n_labels = ndimage_label(foreground)
    labelled = labelled.astype(np.int32)

    centroids: list[list[float]] = []
    if n_labels > 0:
        from skimage.measure import regionprops
        for prop in regionprops(labelled):
            centroids.append([prop.centroid[0], prop.centroid[1]])

    centroids_arr = np.array(centroids, dtype=np.float64) if centroids else np.empty((0, 2), dtype=np.float64)

    return SegmentationOutput(
        masks=labelled,
        centroids=centroids_arr,
        n_cells=n_labels,
        runtime_seconds=0.0,
        peak_memory_mb=0.0,
        method_name="consensus_majority",
        metadata={"n_methods": n_methods, "min_agreement": min_agreement},
    )
