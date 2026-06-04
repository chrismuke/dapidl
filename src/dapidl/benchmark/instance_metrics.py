"""Instance-segmentation evaluation metrics for STHELAR breast slides.

Computes panoptic-PQ + segmentation-PQ + AJI+ + AP@0.5 + AP@[0.5:0.95] +
per-class F1 + confusion matrix for instance segmentation with optional
per-instance class predictions.

All metrics operate on 2-D integer label images (`uint16`, 0 = background)
plus dicts mapping instance IDs to class indices and confidence scores.

References:
- Kirillov et al. (2019) "Panoptic Segmentation" — PQ definition.
- Kumar et al. (2017) "A Dataset and a Technique for Generalized Nuclear
  Segmentation" — AJI+.
- COCO 101-point AP interpolation.
"""

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Internal: vectorized IoU matrix via sparse contingency table.
# ---------------------------------------------------------------------------
def _iou_matrix(
    pred_mask: np.ndarray, gt_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (iou_matrix [N_pred, N_gt], pred_labels, gt_labels).

    Cost is O(H*W + N_pred·N_gt) via numpy.add.at on a contingency table.
    """
    pred_labels = np.unique(pred_mask)
    pred_labels = pred_labels[pred_labels > 0]
    gt_labels = np.unique(gt_mask)
    gt_labels = gt_labels[gt_labels > 0]
    n_p = len(pred_labels)
    n_g = len(gt_labels)
    if n_p == 0 or n_g == 0:
        return np.zeros((n_p, n_g), dtype=np.float64), pred_labels, gt_labels

    remap_p = np.zeros(int(pred_mask.max()) + 1, dtype=np.int32)
    remap_p[pred_labels] = np.arange(1, n_p + 1)
    remap_g = np.zeros(int(gt_mask.max()) + 1, dtype=np.int32)
    remap_g[gt_labels] = np.arange(1, n_g + 1)

    rp = remap_p[pred_mask.ravel()]
    rg = remap_g[gt_mask.ravel()]
    fg = (rp > 0) & (rg > 0)
    if not fg.any():
        return np.zeros((n_p, n_g), dtype=np.float64), pred_labels, gt_labels

    overlap = np.zeros((n_p + 1, n_g + 1), dtype=np.int64)
    np.add.at(overlap, (rp[fg], rg[fg]), 1)
    overlap = overlap[1:, 1:]

    area_p = np.bincount(rp, minlength=n_p + 1)[1:]
    area_g = np.bincount(rg, minlength=n_g + 1)[1:]
    union = area_p[:, None] + area_g[None, :] - overlap
    iou = np.where(union > 0, overlap / union, 0.0).astype(np.float64)
    return iou, pred_labels, gt_labels


def _greedy_match(
    iou: np.ndarray,
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Greedy descending-IoU matching (COCO/panoptic standard)."""
    if iou.size == 0:
        return []
    rows, cols = np.where(iou >= iou_threshold)
    if len(rows) == 0:
        return []
    vals = iou[rows, cols]
    order = np.argsort(-vals)
    rows, cols, vals = rows[order], cols[order], vals[order]
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for r, c, v in zip(rows, cols, vals):
        if r in used_p or c in used_g:
            continue
        used_p.add(int(r))
        used_g.add(int(c))
        matches.append((int(pred_labels[r]), int(gt_labels[c]), float(v)))
    return matches


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------
def match_instances_iou(
    pred_mask: np.ndarray, gt_mask: np.ndarray, iou_threshold: float = 0.5
) -> dict:
    """Greedy IoU matching between pred and GT instances."""
    iou, pl, gl = _iou_matrix(pred_mask, gt_mask)
    matches = _greedy_match(iou, pl, gl, iou_threshold)
    matched_p = {m[0] for m in matches}
    matched_g = {m[1] for m in matches}
    return {
        "matches": matches,
        "unmatched_pred": [int(p) for p in pl if int(p) not in matched_p],
        "unmatched_gt": [int(g) for g in gl if int(g) not in matched_g],
        "iou_matrix": iou,
        "pred_labels": pl,
        "gt_labels": gl,
    }


def panoptic_quality(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    pred_class_per_instance: dict[int, int],
    gt_class_per_instance: dict[int, int],
    n_classes: int,
    iou_threshold: float = 0.5,
) -> dict:
    """Class-aware Panoptic Quality.

    A pred-GT match counts as TP for class c only if both pred and GT are class
    c. Class-mismatched matches → both an FP for pred class and FN for GT class.

    Returns per-class PQ/SQ/RQ + macro means + tp/fp/fn counts.
    """
    matched = match_instances_iou(pred_mask, gt_mask, iou_threshold)
    iou_sum = np.zeros(n_classes, dtype=np.float64)
    tp = np.zeros(n_classes, dtype=np.int64)
    fp = np.zeros(n_classes, dtype=np.int64)
    fn = np.zeros(n_classes, dtype=np.int64)

    for p_id, g_id, iou in matched["matches"]:
        pc = pred_class_per_instance.get(p_id)
        gc = gt_class_per_instance.get(g_id)
        if pc is not None and gc is not None and pc == gc:
            tp[pc] += 1
            iou_sum[pc] += iou
        else:
            if pc is not None:
                fp[pc] += 1
            if gc is not None:
                fn[gc] += 1
    for p in matched["unmatched_pred"]:
        pc = pred_class_per_instance.get(p)
        if pc is not None:
            fp[pc] += 1
    for g in matched["unmatched_gt"]:
        gc = gt_class_per_instance.get(g)
        if gc is not None:
            fn[gc] += 1

    sq = np.where(tp > 0, iou_sum / np.maximum(tp, 1), 0.0)
    denom = tp + 0.5 * fp + 0.5 * fn
    rq = np.where(denom > 0, tp / np.maximum(denom, 1e-9), 0.0)
    pq = sq * rq

    return {
        "pq_per_class": pq.tolist(),
        "sq_per_class": sq.tolist(),
        "rq_per_class": rq.tolist(),
        "pq_mean": float(pq.mean()) if n_classes else 0.0,
        "sq_mean": float(sq.mean()) if n_classes else 0.0,
        "rq_mean": float(rq.mean()) if n_classes else 0.0,
        "n_tp_per_class": tp.tolist(),
        "n_fp_per_class": fp.tolist(),
        "n_fn_per_class": fn.tolist(),
    }


def segmentation_pq(
    pred_mask: np.ndarray, gt_mask: np.ndarray, iou_threshold: float = 0.5
) -> dict:
    """Class-agnostic Panoptic Quality (segmentation only)."""
    matched = match_instances_iou(pred_mask, gt_mask, iou_threshold)
    tp = len(matched["matches"])
    fp = len(matched["unmatched_pred"])
    fn = len(matched["unmatched_gt"])
    sq = (
        sum(m[2] for m in matched["matches"]) / tp if tp > 0 else 0.0
    )
    denom = tp + 0.5 * fp + 0.5 * fn
    rq = tp / denom if denom > 0 else 0.0
    return {
        "pq": float(sq * rq),
        "sq": float(sq),
        "rq": float(rq),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def aji_plus(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Aggregated Jaccard Index, plus version (Kumar et al. 2017).

    For each GT, find the pred with max IoU, accumulate intersection and union.
    Unmatched preds contribute their full area to the union (the "plus").
    """
    iou, pl, gl = _iou_matrix(pred_mask, gt_mask)
    if len(pl) == 0 and len(gl) == 0:
        return 1.0
    if len(pl) == 0 or len(gl) == 0:
        return 0.0

    area_p = np.bincount(pred_mask.ravel())
    area_g = np.bincount(gt_mask.ravel())
    used_p: set[int] = set()
    cap = 0.0
    cup = 0.0

    for g_idx, _ in enumerate(gl):
        best_p = int(np.argmax(iou[:, g_idx]))
        best_iou = iou[best_p, g_idx]
        if best_iou > 0:
            ap = float(area_p[int(pl[best_p])])
            ag = float(area_g[int(gl[g_idx])])
            inter = best_iou * (ap + ag) / (1.0 + best_iou)
            cap += inter
            cup += ap + ag - inter
            used_p.add(best_p)
        else:
            cup += float(area_g[int(gl[g_idx])])

    for p_idx in range(len(pl)):
        if p_idx not in used_p:
            cup += float(area_p[int(pl[p_idx])])

    return float(cap / cup) if cup > 0 else 0.0


def average_precision(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    pred_class_per_instance: dict[int, int],
    gt_class_per_instance: dict[int, int],
    pred_score_per_instance: dict[int, float],
    n_classes: int,
    iou_thresholds: tuple[float, ...] = (0.5,),
) -> dict:
    """Class-aware mean Average Precision via 101-point COCO interpolation."""
    iou, pl, gl = _iou_matrix(pred_mask, gt_mask)

    out: dict = {}
    for thr in iou_thresholds:
        ap_per_class: list[float] = []
        for c in range(n_classes):
            class_pred_idx = [
                i
                for i, p in enumerate(pl)
                if pred_class_per_instance.get(int(p)) == c
            ]
            class_gt_idx = [
                i
                for i, g in enumerate(gl)
                if gt_class_per_instance.get(int(g)) == c
            ]
            if not class_gt_idx:
                ap_per_class.append(float("nan"))
                continue
            if not class_pred_idx:
                ap_per_class.append(0.0)
                continue

            scores = np.array(
                [pred_score_per_instance.get(int(pl[i]), 0.0) for i in class_pred_idx]
            )
            order = np.argsort(-scores)
            class_pred_sorted = [class_pred_idx[i] for i in order]

            n_gt = len(class_gt_idx)
            tp = np.zeros(len(class_pred_sorted), dtype=np.int64)
            fp = np.zeros(len(class_pred_sorted), dtype=np.int64)
            matched_gt: set[int] = set()
            for k, p_idx in enumerate(class_pred_sorted):
                ious_to_gt = iou[p_idx, class_gt_idx]
                best_g = int(np.argmax(ious_to_gt)) if len(ious_to_gt) else -1
                best_iou = ious_to_gt[best_g] if best_g >= 0 else 0.0
                if best_iou >= thr and class_gt_idx[best_g] not in matched_gt:
                    tp[k] = 1
                    matched_gt.add(class_gt_idx[best_g])
                else:
                    fp[k] = 1

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / max(n_gt, 1)
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)

            recall_thresholds = np.linspace(0, 1, 101)
            interp_p = np.zeros(101)
            for t_i, rt in enumerate(recall_thresholds):
                mask = recall >= rt
                interp_p[t_i] = float(precision[mask].max()) if mask.any() else 0.0
            ap_per_class.append(float(interp_p.mean()))

        valid = [v for v in ap_per_class if not np.isnan(v)]
        ap_macro = float(np.mean(valid)) if valid else 0.0
        if len(iou_thresholds) == 1:
            out[f"AP@{thr}"] = ap_macro
            out[f"AP@{thr}_per_class"] = ap_per_class
        else:
            out.setdefault("AP_per_thr", {})[f"AP@{thr}"] = ap_macro

    if len(iou_thresholds) > 1:
        all_per_class = []
        for thr in iou_thresholds:
            ap_per_class_t = []
            for c in range(n_classes):
                class_pred_idx = [
                    i
                    for i, p in enumerate(pl)
                    if pred_class_per_instance.get(int(p)) == c
                ]
                class_gt_idx = [
                    i
                    for i, g in enumerate(gl)
                    if gt_class_per_instance.get(int(g)) == c
                ]
                if not class_gt_idx:
                    ap_per_class_t.append(float("nan"))
                    continue
                if not class_pred_idx:
                    ap_per_class_t.append(0.0)
                    continue
                scores = np.array(
                    [
                        pred_score_per_instance.get(int(pl[i]), 0.0)
                        for i in class_pred_idx
                    ]
                )
                order = np.argsort(-scores)
                class_pred_sorted = [class_pred_idx[i] for i in order]
                n_gt = len(class_gt_idx)
                tp = np.zeros(len(class_pred_sorted), dtype=np.int64)
                fp = np.zeros(len(class_pred_sorted), dtype=np.int64)
                matched_gt = set()
                for k, p_idx in enumerate(class_pred_sorted):
                    ious_to_gt = iou[p_idx, class_gt_idx]
                    best_g = (
                        int(np.argmax(ious_to_gt)) if len(ious_to_gt) else -1
                    )
                    best_iou = ious_to_gt[best_g] if best_g >= 0 else 0.0
                    if (
                        best_iou >= thr
                        and class_gt_idx[best_g] not in matched_gt
                    ):
                        tp[k] = 1
                        matched_gt.add(class_gt_idx[best_g])
                    else:
                        fp[k] = 1
                tp_cum = np.cumsum(tp)
                fp_cum = np.cumsum(fp)
                recall = tp_cum / max(n_gt, 1)
                precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
                recall_thresholds = np.linspace(0, 1, 101)
                interp_p = np.zeros(101)
                for t_i, rt in enumerate(recall_thresholds):
                    mask = recall >= rt
                    interp_p[t_i] = (
                        float(precision[mask].max()) if mask.any() else 0.0
                    )
                ap_per_class_t.append(float(interp_p.mean()))
            all_per_class.append(ap_per_class_t)
        all_per_class = np.array(all_per_class)
        # Macro across (class × threshold), ignore NaN
        valid_means = []
        per_class_combined: list[float] = []
        for c in range(n_classes):
            col = all_per_class[:, c]
            valid_col = col[~np.isnan(col)]
            if len(valid_col) > 0:
                per_class_combined.append(float(np.mean(valid_col)))
                valid_means.append(float(np.mean(valid_col)))
            else:
                per_class_combined.append(float("nan"))
        out["AP@0.5:0.95"] = float(np.mean(valid_means)) if valid_means else 0.0
        out["AP@0.5:0.95_per_class"] = per_class_combined

    return out


def per_class_f1(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    pred_class_per_instance: dict[int, int],
    gt_class_per_instance: dict[int, int],
    n_classes: int,
    iou_threshold: float = 0.5,
) -> dict:
    """Class-aware F1, precision, recall per class."""
    matched = match_instances_iou(pred_mask, gt_mask, iou_threshold)
    tp = np.zeros(n_classes, dtype=np.int64)
    fp = np.zeros(n_classes, dtype=np.int64)
    fn = np.zeros(n_classes, dtype=np.int64)
    for p_id, g_id, _ in matched["matches"]:
        pc = pred_class_per_instance.get(p_id)
        gc = gt_class_per_instance.get(g_id)
        if pc is not None and gc is not None and pc == gc:
            tp[pc] += 1
        else:
            if pc is not None:
                fp[pc] += 1
            if gc is not None:
                fn[gc] += 1
    for p in matched["unmatched_pred"]:
        pc = pred_class_per_instance.get(p)
        if pc is not None:
            fp[pc] += 1
    for g in matched["unmatched_gt"]:
        gc = gt_class_per_instance.get(g)
        if gc is not None:
            fn[gc] += 1
    precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    f1 = np.where(
        precision + recall > 0,
        2 * precision * recall / np.maximum(precision + recall, 1e-9),
        0.0,
    )
    return {
        "f1_per_class": f1.tolist(),
        "f1_macro": float(f1.mean()) if n_classes else 0.0,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
    }


def confusion_matrix(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    pred_class_per_instance: dict[int, int],
    gt_class_per_instance: dict[int, int],
    n_classes: int,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """(n_classes × n_classes) — rows=GT class, cols=pred class. Counts TP-by-class."""
    matched = match_instances_iou(pred_mask, gt_mask, iou_threshold)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for p_id, g_id, _ in matched["matches"]:
        pc = pred_class_per_instance.get(p_id)
        gc = gt_class_per_instance.get(g_id)
        if pc is not None and gc is not None:
            cm[gc, pc] += 1
    return cm
