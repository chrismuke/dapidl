"""Compare two instance label masks: detection P/R/F1, IoU, count ratio."""
import numpy as np
from starpose.consensus.matching import compute_iou_matrix
from starpose.evaluate.morphometric import compute_morphometric


def detection_metrics(pred: np.ndarray, true: np.ndarray, iou_thr: float = 0.5) -> dict:
    """pred/true are (H,W) int label masks. 'true' = source segmentation."""
    n_pred = int(pred.max())
    n_true = int(true.max())
    if n_pred == 0 or n_true == 0:
        return {"n_pred": n_pred, "n_true": n_true, "precision": 0.0, "recall": 0.0,
                "f1": 0.0, "median_iou": 0.0,
                "count_ratio": (n_pred / n_true) if n_true else float("nan")}
    iou = compute_iou_matrix(pred, true)
    best_per_pred = iou.max(axis=1)
    best_per_true = iou.max(axis=0)
    matched_pred = int((best_per_pred >= iou_thr).sum())
    matched_true = int((best_per_true >= iou_thr).sum())
    precision = matched_pred / n_pred
    recall = matched_true / n_true
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n_pred": n_pred, "n_true": n_true,
        "precision": precision, "recall": recall, "f1": f1,
        "median_iou": float(np.median(best_per_pred)),
        "count_ratio": n_pred / n_true,
    }


def morphometrics(masks: np.ndarray, pixel_size: float) -> dict:
    """Thin wrapper over starpose morphometric stats for one mask."""
    return compute_morphometric(masks, pixel_size)
