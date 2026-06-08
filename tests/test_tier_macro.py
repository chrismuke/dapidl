"""Present-class macro-F1: average only over tier classes present in the GT,
marking absent classes N/A (None) instead of forcing F1=0. Pure logic — no
sklearn/scanpy import (fast)."""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "tier_macro", Path(__file__).resolve().parent.parent / "scripts" / "tier_macro.py")
tm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tm)

COARSE4 = ["Endothelial", "Epithelial", "Immune", "Stromal"]


def test_absent_class_is_na_and_excluded_from_macro():
    # GT has NO Endothelial (the breast_s1 case)
    gt = np.array(["Epithelial"] * 5 + ["Immune"] * 3 + ["Stromal"] * 2, dtype=object)
    per_class = {"Endothelial": 0.0, "Epithelial": 0.9, "Immune": 0.7, "Stromal": 0.74}
    macro, per_na, present = tm.present_class_macro(gt, per_class, COARSE4)
    assert present == ["Epithelial", "Immune", "Stromal"]
    assert per_na["Endothelial"] is None          # absent -> N/A, not 0
    assert per_na["Epithelial"] == 0.9
    assert abs(macro - (0.9 + 0.7 + 0.74) / 3) < 1e-9   # 0.78, NOT 0.585


def test_all_present_matches_full_mean():
    gt = np.array(["Endothelial", "Epithelial", "Immune", "Stromal"], dtype=object)
    per_class = {"Endothelial": 0.5, "Epithelial": 0.9, "Immune": 0.7, "Stromal": 0.74}
    macro, per_na, present = tm.present_class_macro(gt, per_class, COARSE4)
    assert present == COARSE4
    assert all(per_na[c] is not None for c in COARSE4)
    assert abs(macro - float(np.mean([0.5, 0.9, 0.7, 0.74]))) < 1e-9


def test_empty_gt_is_zero_all_na():
    gt = np.array([], dtype=object)
    per_class = dict.fromkeys(COARSE4, 0.0)
    macro, per_na, present = tm.present_class_macro(gt, per_class, COARSE4)
    assert present == []
    assert macro == 0.0
    assert all(per_na[c] is None for c in COARSE4)
