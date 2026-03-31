"""Topological (local best-method) voting consensus.

Divides the image into overlapping patches, selects the best segmentation
method per patch based on morphometric quality, and assembles a consensus
mask using first-writer-wins in overlap regions.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label as ndimage_label

from dapidl.benchmark.segmenters.base import SegmentationOutput
from dapidl.benchmark.evaluation.morphometric import compute_morphometric_metrics


def _patch_quality_score(masks_patch: np.ndarray, pixel_size_um: float) -> float:
    """Compute a scalar quality score for a mask patch.

    Score = mean_solidity * (1 - size_outlier_rate).
    Returns 0.0 if the patch is empty.
    """
    if not np.any(masks_patch > 0):
        return 0.0
    metrics = compute_morphometric_metrics(masks_patch, pixel_size_um=pixel_size_um)
    solidity = metrics.get("mean_solidity", 0.0)
    outlier_rate = metrics.get("size_outlier_rate", 0.0)
    return solidity * (1.0 - outlier_rate)


def topological_voting_consensus(
    results: dict[str, SegmentationOutput],
    patch_size: int = 128,
    overlap: int = 64,
    pixel_size_um: float = 0.108,
) -> SegmentationOutput:
    """Local best-method selection consensus.

    The image is divided into overlapping patches with the given patch_size
    and stride = patch_size - overlap. For each patch, every method is scored
    by morphometric quality; the best-scoring method's labels are copied into
    the output mask. In overlap regions, the first writer wins (patches are
    processed in raster order).

    After assembly, connected components are re-labelled to produce a clean
    instance mask.

    Parameters
    ----------
    results:
        Mapping from method name to its SegmentationOutput.
    patch_size:
        Side length of each square patch in pixels. Default 128.
    overlap:
        Overlap between adjacent patches in pixels. Default 64.
    pixel_size_um:
        Physical pixel size in micrometres. Default 0.108.

    Returns
    -------
    SegmentationOutput
        Consensus mask with method_name="consensus_topological".
        runtime_seconds and peak_memory_mb are set to 0.0.
    """
    if not results:
        raise ValueError("results dict must not be empty")

    method_names = list(results.keys())
    method_masks = [results[name].masks for name in method_names]
    shape = method_masks[0].shape
    H, W = shape

    stride = max(1, patch_size - overlap)

    output_fg = np.zeros(shape, dtype=bool)
    written = np.zeros(shape, dtype=bool)

    y_starts = list(range(0, H, stride))
    x_starts = list(range(0, W, stride))

    for y0 in y_starts:
        y1 = min(y0 + patch_size, H)
        for x0 in x_starts:
            x1 = min(x0 + patch_size, W)

            best_score = -1.0
            best_fg: np.ndarray | None = None

            for masks in method_masks:
                patch = masks[y0:y1, x0:x1]
                score = _patch_quality_score(patch, pixel_size_um)
                if score > best_score:
                    best_score = score
                    best_fg = patch > 0

            if best_fg is None:
                best_fg = method_masks[0][y0:y1, x0:x1] > 0

            unwritten_region = ~written[y0:y1, x0:x1]
            output_fg[y0:y1, x0:x1] = np.where(unwritten_region, best_fg, output_fg[y0:y1, x0:x1])
            written[y0:y1, x0:x1] = True

    labelled, n_labels = ndimage_label(output_fg)
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
        method_name="consensus_topological",
        metadata={
            "patch_size": patch_size,
            "overlap": overlap,
            "pixel_size_um": pixel_size_um,
        },
    )
