"""Evaluation metrics for DAPIDL."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from loguru import logger


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive evaluation metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names

    Returns:
        Dictionary of metrics
    """
    metrics = {
        # Overall metrics
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        # Per-class metrics
        "per_class_f1": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "per_class_precision": precision_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist(),
        "per_class_recall": recall_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist(),
        # Confusion matrix
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # Add class names if provided
    if class_names is not None:
        metrics["class_names"] = class_names
        metrics["per_class_metrics"] = {
            name: {
                "f1": metrics["per_class_f1"][i],
                "precision": metrics["per_class_precision"][i],
                "recall": metrics["per_class_recall"][i],
            }
            for i, name in enumerate(class_names)
        }

    return metrics


def print_metrics(metrics: dict[str, Any]) -> None:
    """Print metrics in a formatted way.

    Args:
        metrics: Dictionary from compute_metrics
    """
    logger.info("=" * 60)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 60)

    # Overall metrics
    logger.info("\nOverall Metrics:")
    logger.info(f"  Accuracy:         {metrics['accuracy']:.4f}")
    logger.info(f"  Macro F1:         {metrics['macro_f1']:.4f}")
    logger.info(f"  Weighted F1:      {metrics['weighted_f1']:.4f}")
    logger.info(f"  Macro Precision:  {metrics['macro_precision']:.4f}")
    logger.info(f"  Macro Recall:     {metrics['macro_recall']:.4f}")
    logger.info(f"  MCC:              {metrics['mcc']:.4f}")

    # Per-class metrics
    if "class_names" in metrics:
        logger.info("\nPer-Class Metrics:")
        logger.info(f"{'Class':<20} {'F1':>8} {'Precision':>10} {'Recall':>8}")
        logger.info("-" * 50)
        for name, class_metrics in metrics["per_class_metrics"].items():
            logger.info(
                f"{name:<20} {class_metrics['f1']:>8.4f} "
                f"{class_metrics['precision']:>10.4f} {class_metrics['recall']:>8.4f}"
            )

    logger.info("=" * 60)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str | Path | None = None,
    normalize: bool = True,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """Plot confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save figure
        normalize: Whether to normalize by true labels
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Handle division by zero
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved confusion matrix to {save_path}")

    plt.close()


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a model on a dataloader.

    Args:
        model: PyTorch model
        dataloader: DataLoader to evaluate on
        device: Device to use
        class_names: Optional class names

    Returns:
        Dictionary of metrics
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return compute_metrics(all_labels, all_preds, class_names)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> str:
    """Get sklearn classification report.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional class names

    Returns:
        Classification report string
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
