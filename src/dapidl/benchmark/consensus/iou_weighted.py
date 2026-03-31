"""IoU-weighted consensus for combining segmentation outputs.

Weights each method by its morphometric quality score and performs
weighted foreground voting across methods.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label as ndimage_label

from dapidl.benchmark.segmenters.base import SegmentationOutput
from dapidl.benchmark.evaluation.morphometric import compute_morphometric_metrics


def iou_weighted_consensus(
    results: dict[str, SegmentationOutput],
    pixel_size_um: float = 0.108,
) -> SegmentationOutput:
    """Morphometric quality-weighted foreground voting consensus.

    Each method is weighted by its morphometric quality score:
        weight = solidity * (1 - outlier_rate), clamped to a minimum of 0.01.

    The weighted foreground sum is normalised by the total weight, and pixels
    with a normalised value >= 0.5 are classified as foreground. Connected
    components are labelled to produce the final instance mask.

    Parameters
    ----------
    results:
        Mapping from method name to its SegmentationOutput.
    pixel_size_um:
        Physical pixel size in micrometres. Used when computing morphometric
        quality weights. Default 0.108 (Xenium).

    Returns
    -------
    SegmentationOutput
        Consensus mask with method_name="consensus_iou_weighted".
        metadata includes per-method weights.
        runtime_seconds and peak_memory_mb are set to 0.0.
    """
    if not results:
        raise ValueError("results dict must not be empty")

    method_items = list(results.items())
    shape = method_items[0][1].masks.shape

    weights: dict[str, float] = {}
    for name, output in method_items:
        metrics = compute_morphometric_metrics(output.masks, pixel_size_um=pixel_size_um)
        solidity = metrics.get("mean_solidity", 0.0)
        outlier_rate = metrics.get("size_outlier_rate", 0.0)
        raw_weight = solidity * (1.0 - outlier_rate)
        weights[name] = max(raw_weight, 0.01)

    total_weight = sum(weights.values())

    weighted_fg = np.zeros(shape, dtype=np.float64)
    for name, output in method_items:
        w = weights[name] / total_weight
        weighted_fg += w * (output.masks > 0).astype(np.float64)

    foreground = weighted_fg >= 0.5

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
        method_name="consensus_iou_weighted",
        metadata={"weights": weights, "pixel_size_um": pixel_size_um},
    )
