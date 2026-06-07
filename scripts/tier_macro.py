"""Present-class macro-F1 for tiered cell-type evaluation.

Averages F1 only over tier classes that have >=1 sample in the ground truth,
marking absent classes N/A (None) rather than forcing F1=0. This avoids
penalizing a method for a class the GT cannot measure — e.g. STHELAR's Tangram
GT has no Endothelial label on breast_s1/s3/s6, so scoring Endothelial as 0
drags the macro down for a *data* gap, not a method failure.

Consistent with annotation_run_2026_05.macro_f1 (sklearn average="macro"),
which equals the mean of per_class_f1 over the label set; here we just restrict
that mean to the GT-present subset.
"""
from __future__ import annotations

import numpy as np


def present_class_macro(gt, per_class, all_classes):
    """Macro-F1 over GT-present classes only.

    Args:
        gt: array-like of ground-truth tier labels.
        per_class: dict {class: f1} for every class in ``all_classes``
            (e.g. from per_class_f1 over the full label set).
        all_classes: ordered list of all tier classes.

    Returns:
        (macro, per_class_na, present):
          macro        — mean F1 over classes present in ``gt`` (0.0 if none).
          per_class_na  — {class: float for present classes, None for absent}.
          present       — list of classes (in ``all_classes`` order) with >=1 GT cell.
    """
    gt_set = set(np.asarray(gt).tolist())
    present = [c for c in all_classes if c in gt_set]
    per_class_na = {c: (float(per_class[c]) if c in gt_set else None) for c in all_classes}
    macro = float(np.mean([float(per_class[c]) for c in present])) if present else 0.0
    return macro, per_class_na, present
