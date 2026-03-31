"""Cross-method evaluation metrics for comparing segmentation outputs.

Computes pairwise IoU statistics and per-method consensus scores across
a collection of segmentation results.
"""

from __future__ import annotations

import numpy as np

from dapidl.benchmark.segmenters.base import SegmentationOutput
from dapidl.benchmark.consensus.instance_matching import match_instances_iou


def compute_cross_method_metrics(
    results: dict[str, SegmentationOutput],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute pairwise IoU and match-rate matrices across segmentation methods.

    For every pair of methods, matches instances using the Hungarian algorithm
    and computes the mean IoU of matched pairs and the fraction of instances
    in method A that were matched in method B (match rate).

    A per-method consensus score is the mean of all pairwise match rates
    involving that method.

    Parameters
    ----------
    results:
        Mapping from method name to its SegmentationOutput.
    iou_threshold:
        IoU threshold used by match_instances_iou. Default 0.5.

    Returns
    -------
    dict
        Keys:
        - method_names: list[str] — ordered list of method names
        - pairwise_mean_iou: list[list[float]] — (N, N) matrix; diagonal = 1.0
        - pairwise_match_rate: list[list[float]] — (N, N) matrix; diagonal = 1.0
          entry [i][j] = fraction of method_i instances matched in method_j
        - method_consensus_score: list[float] — per-method mean pairwise match rate
    """
    method_names = list(results.keys())
    n = len(method_names)

    pairwise_mean_iou = [[0.0] * n for _ in range(n)]
    pairwise_match_rate = [[0.0] * n for _ in range(n)]

    for i in range(n):
        pairwise_mean_iou[i][i] = 1.0
        pairwise_match_rate[i][i] = 1.0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            name_i = method_names[i]
            name_j = method_names[j]
            masks_i = results[name_i].masks
            masks_j = results[name_j].masks
            n_i = results[name_i].n_cells

            matches, ious = match_instances_iou(masks_i, masks_j, iou_threshold=iou_threshold)

            if ious:
                pairwise_mean_iou[i][j] = float(np.mean(ious))
            else:
                pairwise_mean_iou[i][j] = 0.0

            if n_i > 0:
                pairwise_match_rate[i][j] = len(matches) / n_i
            else:
                pairwise_match_rate[i][j] = 0.0

    method_consensus_score: list[float] = []
    for i in range(n):
        off_diag = [pairwise_match_rate[i][j] for j in range(n) if j != i]
        score = float(np.mean(off_diag)) if off_diag else 1.0
        method_consensus_score.append(score)

    return {
        "method_names": method_names,
        "pairwise_mean_iou": pairwise_mean_iou,
        "pairwise_match_rate": pairwise_match_rate,
        "method_consensus_score": method_consensus_score,
    }
