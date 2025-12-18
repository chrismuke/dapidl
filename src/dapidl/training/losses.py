"""Loss functions for DAPIDL training.

Includes:
- Classification losses: FocalLoss, LabelSmoothingCrossEntropy
- Segmentation losses: DiceLoss, BinarySegmentationLoss
- Multi-task losses: MultiTaskLoss (combines classification + segmentation)
"""

from typing import Literal

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
    max_weight_ratio: float = 10.0,
) -> torch.Tensor:
    """Compute class weights for imbalanced data.

    Args:
        labels: Array of class labels
        num_classes: Number of classes
        method: Weighting method:
            - 'inverse': 1 / count
            - 'inverse_sqrt': 1 / sqrt(count)
            - 'effective': Effective number of samples (Class-Balanced Loss)
        max_weight_ratio: Maximum ratio between largest and smallest weight.
            Prevents extreme over-weighting of rare classes which causes mode
            collapse. Default 10.0 means rare classes get at most 10x the
            weight of the most common class. Set to 0 or None to disable.

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

    # Cap the weight ratio to prevent mode collapse
    if max_weight_ratio is not None and max_weight_ratio > 0:
        min_weight = weights.min()
        max_allowed = min_weight * max_weight_ratio
        weights = np.minimum(weights, max_allowed)

    # Normalize weights
    weights = weights / weights.sum() * num_classes

    return torch.FloatTensor(weights)


# =============================================================================
# Segmentation Losses (for multi-task learning)
# =============================================================================


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation.

    Dice loss measures the overlap between predicted and target masks.
    It's particularly good for imbalanced segmentation tasks.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Dice Loss = 1 - Dice
    """

    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize Dice loss.

        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            inputs: Predicted logits of shape (B, 1, H, W) or (B, H, W)
            targets: Ground truth masks of shape (B, 1, H, W) or (B, H, W)
                     Values should be 0 or 1

        Returns:
            Dice loss value
        """
        # Apply sigmoid to get probabilities
        inputs = torch.sigmoid(inputs)

        # Flatten spatial dimensions
        inputs = inputs.view(inputs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1).float()

        # Compute Dice coefficient
        intersection = (inputs * targets).sum(dim=1)
        union = inputs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice

        # Apply reduction
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class BinarySegmentationLoss(nn.Module):
    """Combined loss for binary segmentation.

    Combines BCE (pixel-wise) and Dice (region-wise) losses for
    robust segmentation training.

    Total = bce_weight * BCE + dice_weight * Dice
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: float | None = None,
    ) -> None:
        """Initialize combined segmentation loss.

        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
            smooth: Smoothing factor for Dice loss
            pos_weight: Positive class weight for BCE (for imbalanced masks)
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        # BCE with optional positive weight
        pos_weight_tensor = torch.tensor([pos_weight]) if pos_weight else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.dice = DiceLoss(smooth=smooth)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined segmentation loss.

        Args:
            inputs: Predicted logits of shape (B, 1, H, W)
            targets: Ground truth masks of shape (B, 1, H, W)

        Returns:
            Combined loss value
        """
        bce_loss = self.bce(inputs, targets.float())
        dice_loss = self.dice(inputs, targets)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# =============================================================================
# Multi-Task Loss (for joint classification + segmentation)
# =============================================================================


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning (classification + segmentation).

    This loss combines:
    1. Classification loss (main task): CrossEntropy, Focal, etc.
    2. Segmentation loss (auxiliary task): Dice + BCE

    The segmentation task is auxiliary - it helps the backbone learn better
    features but we ultimately care about classification performance.

    Usage:
        loss_fn = MultiTaskLoss(
            num_classes=3,
            class_weights=get_class_weights(labels),
            seg_weight=0.5,
        )

        class_logits, seg_mask = model(images, return_seg=True)
        loss, loss_dict = loss_fn(class_logits, labels, seg_mask, mask_targets)
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
        classification_loss: Literal["ce", "focal", "label_smooth"] = "ce",
        seg_weight: float = 0.5,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ) -> None:
        """Initialize multi-task loss.

        Args:
            num_classes: Number of classification classes
            class_weights: Optional class weights for classification loss
            classification_loss: Type of classification loss ('ce', 'focal', 'label_smooth')
            seg_weight: Weight for segmentation loss (classification weight is 1.0)
            focal_gamma: Gamma parameter for focal loss
            label_smoothing: Smoothing factor for label smoothing
        """
        super().__init__()
        self.seg_weight = seg_weight

        # Classification loss
        if classification_loss == "focal":
            self.class_loss = FocalLoss(
                alpha=class_weights,
                gamma=focal_gamma,
            )
        elif classification_loss == "label_smooth":
            self.class_loss = LabelSmoothingCrossEntropy(
                smoothing=label_smoothing,
                weight=class_weights,
            )
        else:  # ce
            self.class_loss = nn.CrossEntropyLoss(
                weight=class_weights,
            )

        # Segmentation loss
        self.seg_loss = BinarySegmentationLoss(
            bce_weight=0.5,
            dice_weight=0.5,
        )

    def forward(
        self,
        class_logits: torch.Tensor,
        class_targets: torch.Tensor,
        seg_logits: torch.Tensor | None = None,
        seg_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute multi-task loss.

        Args:
            class_logits: Classification predictions (B, num_classes)
            class_targets: Classification labels (B,)
            seg_logits: Segmentation predictions (B, 1, H, W), optional
            seg_targets: Segmentation masks (B, 1, H, W), optional

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains
            individual loss components for logging
        """
        # Classification loss (always computed)
        loss_class = self.class_loss(class_logits, class_targets)

        loss_dict = {
            "loss_class": loss_class.item(),
        }

        total_loss = loss_class

        # Segmentation loss (only if masks provided)
        if seg_logits is not None and seg_targets is not None:
            # Ensure seg_targets has channel dimension
            if seg_targets.dim() == 3:
                seg_targets = seg_targets.unsqueeze(1)

            # Resize seg_logits to match seg_targets if needed
            if seg_logits.shape[-2:] != seg_targets.shape[-2:]:
                seg_logits = F.interpolate(
                    seg_logits,
                    size=seg_targets.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss_seg = self.seg_loss(seg_logits, seg_targets)
            loss_dict["loss_seg"] = loss_seg.item()

            total_loss = total_loss + self.seg_weight * loss_seg

        loss_dict["loss_total"] = total_loss.item()

        return total_loss, loss_dict


def compute_segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute segmentation metrics.

    Args:
        pred: Predicted logits or probabilities (B, 1, H, W)
        target: Ground truth masks (B, 1, H, W)
        threshold: Threshold for binarizing predictions

    Returns:
        Dictionary with IoU, Dice, precision, recall
    """
    # Binarize predictions
    if pred.requires_grad:
        pred = pred.detach()
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target = target.float()

    # Flatten (use reshape instead of view for non-contiguous DALI tensors)
    pred_flat = pred_binary.reshape(-1)
    target_flat = target.reshape(-1)

    # True positives, false positives, false negatives
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()

    # Metrics
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "seg_iou": iou.item(),
        "seg_dice": dice.item(),
        "seg_precision": precision.item(),
        "seg_recall": recall.item(),
    }
