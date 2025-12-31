"""Visualization utilities for HPO experiments.

Provides functions to generate sample prediction images, class grids,
and confusion matrices for ClearML logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from clearml import Task
    from torch.utils.data import DataLoader

    from dapidl.models.classifier import CellTypeClassifier

logger = logging.getLogger(__name__)


def generate_sample_predictions(
    model: CellTypeClassifier,
    dataloader: DataLoader,
    class_names: list[str],
    n_samples_per_class: int = 5,
    device: str = "cuda",
) -> dict[str, list[tuple[np.ndarray, str, bool]]]:
    """Generate sample predictions for each class.

    Args:
        model: Trained classifier model
        dataloader: DataLoader with test data
        class_names: List of class names
        n_samples_per_class: Number of samples per class
        device: Device to run inference on

    Returns:
        Dict mapping class name to list of (image, predicted_class, is_correct)
    """
    model.to(device)
    model.eval()

    # Collect samples per class
    samples: dict[str, list[tuple[np.ndarray, str, bool]]] = {
        name: [] for name in class_names
    }

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            for i in range(len(images)):
                true_class = class_names[labels[i].item()]
                pred_class = class_names[predicted[i].item()]
                is_correct = labels[i].item() == predicted[i].item()

                # Get image as numpy array (denormalize if needed)
                img = images[i].cpu().numpy()
                if img.shape[0] == 1:
                    img = img[0]  # Remove channel dim for grayscale
                elif img.shape[0] == 3:
                    img = img[0]  # Take first channel if RGB

                # Normalize to 0-255
                img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(
                    np.uint8
                )

                if len(samples[true_class]) < n_samples_per_class:
                    samples[true_class].append((img, pred_class, is_correct))

            # Check if we have enough samples
            if all(len(s) >= n_samples_per_class for s in samples.values()):
                break

    return samples


def create_class_prediction_grid(
    samples: dict[str, list[tuple[np.ndarray, str, bool]]],
    class_names: list[str],
    patch_size: int = 128,
    border_width: int = 3,
) -> Image.Image:
    """Create a grid image showing predictions per class.

    Each row is a class, each column is a sample.
    Green border = correct prediction, Red border = incorrect.

    Args:
        samples: Output from generate_sample_predictions
        class_names: List of class names (determines row order)
        patch_size: Size of each patch in grid
        border_width: Width of colored border

    Returns:
        PIL Image with prediction grid
    """
    n_classes = len(class_names)
    n_samples = max(len(s) for s in samples.values()) if samples else 5

    # Calculate grid dimensions
    cell_size = patch_size + 2 * border_width
    label_width = 150  # Width for class labels
    grid_width = label_width + n_samples * cell_size
    grid_height = n_classes * cell_size

    # Create grid image
    grid = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)

    for row_idx, class_name in enumerate(class_names):
        y_offset = row_idx * cell_size

        # Draw class label
        draw.text((5, y_offset + cell_size // 2 - 10), class_name[:15], fill=(0, 0, 0))

        # Draw samples
        class_samples = samples.get(class_name, [])
        for col_idx, (img, pred_class, is_correct) in enumerate(class_samples):
            x_offset = label_width + col_idx * cell_size

            # Border color: green for correct, red for incorrect
            border_color = (0, 200, 0) if is_correct else (200, 0, 0)

            # Draw border
            draw.rectangle(
                [x_offset, y_offset, x_offset + cell_size - 1, y_offset + cell_size - 1],
                fill=border_color,
            )

            # Resize image if needed
            pil_img = Image.fromarray(img)
            if pil_img.size != (patch_size, patch_size):
                pil_img = pil_img.resize((patch_size, patch_size), Image.BILINEAR)

            # Convert to RGB if grayscale
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            # Paste image inside border
            grid.paste(
                pil_img, (x_offset + border_width, y_offset + border_width)
            )

    return grid


def log_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    task: Task,
    title: str = "Confusion Matrix",
    series: str = "test",
    iteration: int = 0,
) -> None:
    """Log confusion matrix to ClearML.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        task: ClearML Task object
        title: Title for the plot
        series: Series name for logging
        iteration: Iteration number
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    # Log as ClearML confusion matrix
    task.get_logger().report_confusion_matrix(
        title=title,
        series=series,
        matrix=cm,
        xaxis="Predicted",
        yaxis="True",
        xlabels=class_names,
        ylabels=class_names,
        iteration=iteration,
    )


def log_metrics_to_clearml(
    metrics: dict[str, float],
    task: Task,
    title: str = "metrics",
    iteration: int = 0,
) -> None:
    """Log metrics dictionary to ClearML.

    Args:
        metrics: Dict of metric name -> value
        task: ClearML Task object
        title: Title/series for logging
        iteration: Iteration number
    """
    logger_obj = task.get_logger()

    for name, value in metrics.items():
        logger_obj.report_scalar(
            title=title,
            series=name,
            value=value,
            iteration=iteration,
        )


def save_prediction_grid(
    grid: Image.Image,
    output_path: Path,
    task: Task | None = None,
) -> Path:
    """Save prediction grid and optionally upload to ClearML.

    Args:
        grid: PIL Image with prediction grid
        output_path: Path to save image
        task: Optional ClearML task for artifact upload

    Returns:
        Path to saved image
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)

    if task is not None:
        task.upload_artifact(
            name="prediction_grid",
            artifact_object=str(output_path),
        )

    return output_path


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, float]]:
    """Compute per-class precision, recall, and F1.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        Dict mapping class name to metrics dict
    """
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )

    results = {}
    for i, name in enumerate(class_names):
        results[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return results
