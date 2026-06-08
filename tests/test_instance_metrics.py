"""Tests for dapidl.benchmark.instance_metrics."""

import numpy as np
import pytest

from dapidl.benchmark.instance_metrics import (
    aji_plus,
    average_precision,
    confusion_matrix,
    match_instances_iou,
    panoptic_quality,
    per_class_f1,
    segmentation_pq,
)


def _two_cell_masks():
    """Build identical 16×16 masks with two non-touching cells."""
    m = np.zeros((16, 16), dtype=np.uint16)
    m[2:6, 2:6] = 1  # 4×4 cell, area 16
    m[10:14, 10:14] = 2  # 4×4 cell, area 16
    return m, m.copy()


def _shifted_masks():
    """Pred shifted 1px down — IoU drops."""
    pred, _ = _two_cell_masks()
    gt = np.roll(pred, shift=1, axis=0)
    return pred, gt


# ---------------------------------------------------------------------------
def test_match_instances_iou_perfect():
    p, g = _two_cell_masks()
    res = match_instances_iou(p, g, iou_threshold=0.5)
    assert len(res["matches"]) == 2
    assert all(m[2] == 1.0 for m in res["matches"])
    assert res["unmatched_pred"] == []
    assert res["unmatched_gt"] == []


def test_segmentation_pq_perfect_and_shifted():
    p, g = _two_cell_masks()
    res = segmentation_pq(p, g)
    assert res["pq"] == pytest.approx(1.0)
    assert res["tp"] == 2 and res["fp"] == 0 and res["fn"] == 0

    p2, g2 = _shifted_masks()
    res2 = segmentation_pq(p2, g2, iou_threshold=0.3)
    assert 0.3 < res2["pq"] < 1.0
    assert res2["tp"] == 2


def test_panoptic_quality_class_aware():
    p, g = _two_cell_masks()
    pred_cls = {1: 0, 2: 1}
    gt_cls = {1: 0, 2: 1}
    res = panoptic_quality(p, g, pred_cls, gt_cls, n_classes=2)
    assert res["pq_mean"] == pytest.approx(1.0)
    assert res["n_tp_per_class"] == [1, 1]

    # Class mismatch → fp+fn for class 1
    pred_cls_mismatch = {1: 0, 2: 0}
    res2 = panoptic_quality(p, g, pred_cls_mismatch, gt_cls, n_classes=2)
    assert res2["n_tp_per_class"][1] == 0
    assert res2["n_fp_per_class"][0] == 1
    assert res2["n_fn_per_class"][1] == 1


def test_aji_plus_perfect():
    p, g = _two_cell_masks()
    assert aji_plus(p, g) == pytest.approx(1.0)

    p_miss = p.copy()
    p_miss[10:14, 10:14] = 0  # miss cell 2 entirely
    assert aji_plus(p_miss, g) < 0.6  # significant drop


def test_average_precision_perfect():
    p, g = _two_cell_masks()
    pred_cls = {1: 0, 2: 1}
    gt_cls = {1: 0, 2: 1}
    pred_score = {1: 1.0, 2: 1.0}
    res = average_precision(p, g, pred_cls, gt_cls, pred_score, n_classes=2)
    assert res["AP@0.5"] == pytest.approx(1.0)


def test_per_class_f1_recall_drop():
    p, g = _two_cell_masks()
    p_miss = p.copy()
    p_miss[10:14, 10:14] = 0  # drop cell 2 (class 1)
    pred_cls = {1: 0}
    gt_cls = {1: 0, 2: 1}
    res = per_class_f1(p_miss, g, pred_cls, gt_cls, n_classes=2)
    assert res["recall_per_class"][1] == 0.0
    assert res["recall_per_class"][0] == 1.0


def test_confusion_matrix():
    p, g = _two_cell_masks()
    pred_cls = {1: 0, 2: 0}  # both predicted as class 0
    gt_cls = {1: 0, 2: 1}
    cm = confusion_matrix(p, g, pred_cls, gt_cls, n_classes=2)
    assert cm[0, 0] == 1  # gt=0, pred=0
    assert cm[1, 0] == 1  # gt=1, pred=0 (class confusion)
    assert cm[0, 1] == 0
    assert cm[1, 1] == 0
