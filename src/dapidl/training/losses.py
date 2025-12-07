"""Loss functions for DAPIDL training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Focal loss down-weights easy examples and focuses on hard ones,
    which is useful for imbalanced datasets.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", 2017
    """

    def __init__(
        self,
        alpha: float | torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        """Initialize Focal Loss.

        Args:
            alpha: Class weights (scalar or tensor of shape num_classes)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'none', 'mean', or 'sum'
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Predicted logits of shape (B, C)
            targets: Ground truth labels of shape (B,)

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_weight = (1 - pt) ** self.gamma

        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(inputs.device)[targets]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean",
        weight: torch.Tensor | None = None,
    ) -> None:
        """Initialize.

        Args:
            smoothing: Label smoothing factor
            reduction: 'none', 'mean', or 'sum'
            weight: Optional class weights
        """
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss.

        Args:
            inputs: Predicted logits of shape (B, C)
            targets: Ground truth labels of shape (B,)

        Returns:
            Loss value
        """
        return F.cross_entropy(
            inputs,
            targets,
            weight=self.weight.to(inputs.device) if self.weight is not None else None,
            label_smoothing=self.smoothing,
            reduction=self.reduction,
        )


def get_class_weights(
    labels: np.ndarray | torch.Tensor,
    num_classes: int,
    method: str = "inverse",
) -> torch.Tensor:
    """Compute class weights for imbalanced data.

    Args:
        labels: Array of class labels
        num_classes: Number of classes
        method: Weighting method:
            - 'inverse': 1 / count
            - 'inverse_sqrt': 1 / sqrt(count)
            - 'effective': Effective number of samples (Class-Balanced Loss)

    Returns:
        Tensor of class weights
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1)  # Avoid division by zero

    if method == "inverse":
        weights = 1.0 / counts
    elif method == "inverse_sqrt":
        weights = 1.0 / np.sqrt(counts)
    elif method == "effective":
        # Effective number of samples (Class-Balanced Loss)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize weights
    weights = weights / weights.sum() * num_classes

    return torch.FloatTensor(weights)
