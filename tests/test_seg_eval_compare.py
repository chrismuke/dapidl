"""Tests for detection metrics between two label masks."""
import numpy as np
from dapidl.seg_eval.compare import detection_metrics


def _two_squares(shape=(40, 40)):
    m = np.zeros(shape, dtype=np.int32)
    m[2:10, 2:10] = 1
    m[20:28, 20:28] = 2
    return m


def test_identical_masks_perfect():
    m = _two_squares()
    r = detection_metrics(m, m, iou_thr=0.5)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0
    assert r["n_pred"] == 2 and r["n_true"] == 2
    assert r["count_ratio"] == 1.0


def test_one_missing_instance():
    pred = _two_squares()
    true = _two_squares()
    pred[20:28, 20:28] = 0
    r = detection_metrics(pred, true, iou_thr=0.5)
    assert r["n_pred"] == 1 and r["n_true"] == 2
    assert r["recall"] == 0.5 and r["precision"] == 1.0
