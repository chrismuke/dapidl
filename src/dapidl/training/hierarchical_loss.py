"""Hierarchical loss functions for multi-head classification.

Loss Components:
    - L_coarse: Classification loss at coarse level (weight 1.0)
    - L_medium: Classification loss at medium level (weight 0.5)
    - L_fine: Classification loss at fine level (weight 0.3)
    - L_consistency: Penalty for hierarchical inconsistency (weight 0.1)

Total: L_total = L_coarse + 0.5*L_medium + 0.3*L_fine + 0.1*L_consistency

Consistency Loss:
    Penalizes when fine predictions disagree with their expected coarse parent.
    Uses soft cross-entropy between predicted fine-level probabilities
    marginalized to coarse level vs actual coarse predictions.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from dapidl.models.hierarchical import HierarchyConfig, HierarchicalOutput


class HierarchicalLoss(nn.Module):
    """Combined loss for hierarchical classification.

    Computes weighted sum of classification losses at each hierarchy level
    plus a consistency penalty for hierarchical disagreement.

    Attributes:
        hierarchy_config: Configuration with class mappings
        coarse_weight: Weight for coarse-level loss
        medium_weight: Weight for medium-level loss
        fine_weight: Weight for fine-level loss
        consistency_weight: Weight for consistency penalty
    """

    def __init__(
        self,
        hierarchy_config: HierarchyConfig,
        coarse_weight: float = 1.0,
        medium_weight: float = 0.5,
        fine_weight: float = 0.3,
        consistency_weight: float = 0.1,
        coarse_class_weights: torch.Tensor | None = None,
        medium_class_weights: torch.Tensor | None = None,
        fine_class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
    ) -> None:
        """Initialize hierarchical loss.

        Args:
            hierarchy_config: Configuration with hierarchy mappings
            coarse_weight: Loss weight for coarse level (default 1.0)
            medium_weight: Loss weight for medium level (default 0.5)
            fine_weight: Loss weight for fine level (default 0.3)
            consistency_weight: Weight for consistency penalty (default 0.1)
            coarse_class_weights: Class weights for coarse loss
            medium_class_weights: Class weights for medium loss
            fine_class_weights: Class weights for fine loss
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
            use_focal: Use focal loss instead of cross-entropy
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()

        self.hierarchy_config = hierarchy_config
        self.coarse_weight = coarse_weight
        self.medium_weight = medium_weight
        self.fine_weight = fine_weight
        self.consistency_weight = consistency_weight
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma

        # Store class weights (will be moved to device in forward)
        self.register_buffer(
            "coarse_class_weights",
            coarse_class_weights if coarse_class_weights is not None else None,
        )
        self.register_buffer(
            "medium_class_weights",
            medium_class_weights if medium_class_weights is not None else None,
        )
        self.register_buffer(
            "fine_class_weights",
            fine_class_weights if fine_class_weights is not None else None,
        )

        # Build fine-to-coarse mapping tensor for vectorized consistency computation
        # Shape: (num_fine,) where value[i] = coarse index for fine class i
        fine_to_coarse_tensor = torch.zeros(
            hierarchy_config.num_fine, dtype=torch.long
        )
        for fine_idx, coarse_idx in hierarchy_config.fine_to_coarse.items():
            fine_to_coarse_tensor[fine_idx] = coarse_idx
        self.register_buffer("fine_to_coarse_map", fine_to_coarse_tensor)

        # Build medium-to-coarse mapping tensor
        medium_to_coarse_tensor = torch.zeros(
            hierarchy_config.num_medium, dtype=torch.long
        )
        for medium_idx, coarse_idx in hierarchy_config.medium_to_coarse.items():
            medium_to_coarse_tensor[medium_idx] = coarse_idx
        self.register_buffer("medium_to_coarse_map", medium_to_coarse_tensor)

        logger.info(
            f"HierarchicalLoss: weights=(coarse={coarse_weight}, "
            f"medium={medium_weight}, fine={fine_weight}, "
            f"consistency={consistency_weight}), "
            f"label_smoothing={label_smoothing}, use_focal={use_focal}"
        )

    def _compute_classification_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute classification loss (CE or Focal).

        Args:
            logits: Predictions of shape (B, num_classes)
            targets: Ground truth labels of shape (B,)
            class_weights: Optional class weights

        Returns:
            Scalar loss value
        """
        if self.use_focal:
            # Focal loss
            ce_loss = F.cross_entropy(
                logits,
                targets,
                weight=class_weights,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.focal_gamma

            if class_weights is not None:
                alpha_t = class_weights[targets]
                focal_loss = alpha_t * focal_weight * ce_loss
            else:
                focal_loss = focal_weight * ce_loss

            return focal_loss.mean()
        else:
            # Standard cross-entropy
            return F.cross_entropy(
                logits,
                targets,
                weight=class_weights,
                label_smoothing=self.label_smoothing,
            )

    def _compute_consistency_loss(
        self,
        output: HierarchicalOutput,
        coarse_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute consistency loss between hierarchy levels.

        Penalizes when fine/medium predictions marginalized to coarse
        disagree with actual coarse predictions.

        The consistency loss measures KL divergence between:
        1. Predicted coarse distribution from fine probs (marginalized)
        2. Actual coarse prediction distribution

        Args:
            output: Model output with logits at each level
            coarse_targets: Ground truth coarse labels

        Returns:
            Scalar consistency loss
        """
        batch_size = coarse_targets.shape[0]
        device = coarse_targets.device
        consistency_loss = torch.tensor(0.0, device=device)

        # Fine → Coarse consistency
        if output.fine_logits is not None:
            fine_probs = output.fine_probs  # (B, num_fine)

            # Marginalize fine probs to coarse level
            # For each coarse class, sum probabilities of all fine classes that map to it
            num_coarse = self.hierarchy_config.num_coarse
            marginalized_coarse = torch.zeros(batch_size, num_coarse, device=device)

            for coarse_idx in range(num_coarse):
                # Find all fine classes that map to this coarse class
                fine_mask = (self.fine_to_coarse_map == coarse_idx)
                marginalized_coarse[:, coarse_idx] = fine_probs[:, fine_mask].sum(dim=1)

            # KL divergence: D_KL(coarse_pred || marginalized_fine)
            coarse_log_probs = F.log_softmax(output.coarse_logits, dim=-1)
            marginalized_log_probs = torch.log(marginalized_coarse + 1e-8)

            # KL divergence per sample
            kl_div = F.kl_div(
                marginalized_log_probs,
                coarse_log_probs.exp(),
                reduction="batchmean",
            )
            consistency_loss = consistency_loss + kl_div

        # Medium → Coarse consistency
        if output.medium_logits is not None:
            medium_probs = output.medium_probs  # (B, num_medium)

            num_coarse = self.hierarchy_config.num_coarse
            marginalized_coarse = torch.zeros(batch_size, num_coarse, device=device)

            for coarse_idx in range(num_coarse):
                medium_mask = (self.medium_to_coarse_map == coarse_idx)
                marginalized_coarse[:, coarse_idx] = medium_probs[:, medium_mask].sum(dim=1)

            coarse_log_probs = F.log_softmax(output.coarse_logits, dim=-1)
            marginalized_log_probs = torch.log(marginalized_coarse + 1e-8)

            kl_div = F.kl_div(
                marginalized_log_probs,
                coarse_log_probs.exp(),
                reduction="batchmean",
            )
            consistency_loss = consistency_loss + 0.5 * kl_div  # Lower weight for medium

        return consistency_loss

    def forward(
        self,
        output: HierarchicalOutput,
        coarse_targets: torch.Tensor,
        medium_targets: torch.Tensor | None = None,
        fine_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total hierarchical loss.

        Args:
            output: Model output with logits at each level
            coarse_targets: Ground truth coarse labels (B,)
            medium_targets: Ground truth medium labels (B,) or None
            fine_targets: Ground truth fine labels (B,) or None

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains
            individual loss components for logging
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=output.coarse_logits.device)

        # Coarse loss (always computed)
        loss_coarse = self._compute_classification_loss(
            output.coarse_logits,
            coarse_targets,
            self.coarse_class_weights,
        )
        total_loss = total_loss + self.coarse_weight * loss_coarse
        loss_dict["loss_coarse"] = loss_coarse.item()

        # Medium loss (if targets and logits available)
        if medium_targets is not None and output.medium_logits is not None:
            loss_medium = self._compute_classification_loss(
                output.medium_logits,
                medium_targets,
                self.medium_class_weights,
            )
            total_loss = total_loss + self.medium_weight * loss_medium
            loss_dict["loss_medium"] = loss_medium.item()

        # Fine loss (if targets and logits available)
        if fine_targets is not None and output.fine_logits is not None:
            loss_fine = self._compute_classification_loss(
                output.fine_logits,
                fine_targets,
                self.fine_class_weights,
            )
            total_loss = total_loss + self.fine_weight * loss_fine
            loss_dict["loss_fine"] = loss_fine.item()

        # Consistency loss
        if self.consistency_weight > 0 and (
            output.medium_logits is not None or output.fine_logits is not None
        ):
            loss_consistency = self._compute_consistency_loss(output, coarse_targets)
            total_loss = total_loss + self.consistency_weight * loss_consistency
            loss_dict["loss_consistency"] = loss_consistency.item()

        loss_dict["loss_total"] = total_loss.item()

        return total_loss, loss_dict


def get_hierarchical_class_weights(
    labels: torch.Tensor | list,
    num_classes: int,
    method: Literal["inverse", "inverse_sqrt", "effective"] = "inverse",
    max_weight_ratio: float = 10.0,
) -> torch.Tensor:
    """Compute class weights for hierarchical level.

    Same logic as get_class_weights in losses.py but as standalone function.

    Args:
        labels: Array/tensor of class labels
        num_classes: Total number of classes
        method: Weighting method (inverse, inverse_sqrt, effective)
        max_weight_ratio: Maximum ratio between largest and smallest weight

    Returns:
        Tensor of class weights
    """
    import numpy as np

    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)

    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1)  # Avoid division by zero

    if method == "inverse":
        weights = 1.0 / counts
    elif method == "inverse_sqrt":
        weights = 1.0 / np.sqrt(counts)
    elif method == "effective":
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown method: {method}")

    # Cap weight ratio
    if max_weight_ratio is not None and max_weight_ratio > 0:
        min_weight = weights.min()
        max_allowed = min_weight * max_weight_ratio
        weights = np.minimum(weights, max_allowed)

    # Normalize
    weights = weights / weights.sum() * num_classes

    return torch.FloatTensor(weights)


class CurriculumScheduler:
    """Scheduler for curriculum learning in hierarchical classification.

    Progressively activates finer classification heads during training:
    - Phase 1 (epochs 1-20): Coarse only
    - Phase 2 (epochs 21-50): Coarse + Medium
    - Phase 3 (epochs 51+): All heads (Coarse + Medium + Fine)

    Also adjusts loss weights during transitions for smoother learning.

    Phase transition fixes (v2):
    - Quadratic warmup curve for smoother gradient introduction
    - LR reduction during warmup to protect backbone features
    - Optional backbone freezing during initial warmup epochs
    """

    def __init__(
        self,
        coarse_only_epochs: int = 20,
        coarse_medium_epochs: int = 50,
        warmup_epochs: int = 5,
        transition_lr_factor: float = 0.1,
        freeze_backbone_epochs: int = 2,
    ) -> None:
        """Initialize curriculum scheduler.

        Args:
            coarse_only_epochs: Train coarse only until this epoch
            coarse_medium_epochs: Train coarse+medium until this epoch
            warmup_epochs: Warmup period when activating new heads
            transition_lr_factor: LR multiplier during warmup (0.1 = 10% of base LR)
            freeze_backbone_epochs: Freeze backbone for this many epochs at phase start
        """
        self.coarse_only_epochs = coarse_only_epochs
        self.coarse_medium_epochs = coarse_medium_epochs
        self.warmup_epochs = warmup_epochs
        self.transition_lr_factor = transition_lr_factor
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self._prev_phase = None

    def get_active_heads(self, epoch: int) -> set[str]:
        """Get set of active heads for current epoch.

        Args:
            epoch: Current epoch (1-indexed)

        Returns:
            Set of active head names
        """
        if epoch <= self.coarse_only_epochs:
            return {"coarse"}
        elif epoch <= self.coarse_medium_epochs:
            return {"coarse", "medium"}
        else:
            return {"coarse", "medium", "fine"}

    def _quadratic_warmup(self, progress: float) -> float:
        """Quadratic warmup curve for smoother gradient introduction.

        Uses x^2 curve which starts slow and accelerates, preventing
        the initial shock of random gradients from new heads.
        """
        return progress ** 2

    def get_epochs_since_phase_start(self, epoch: int) -> int:
        """Get number of epochs since the current phase started."""
        if epoch <= self.coarse_only_epochs:
            return epoch
        elif epoch <= self.coarse_medium_epochs:
            return epoch - self.coarse_only_epochs
        else:
            return epoch - self.coarse_medium_epochs

    def should_freeze_backbone(self, epoch: int) -> bool:
        """Check if backbone should be frozen for this epoch.

        Freezes backbone for the first few epochs of Phase 2 and 3
        to allow new heads to warm up without corrupting backbone.
        """
        epochs_in_phase = self.get_epochs_since_phase_start(epoch)
        phase = self.get_phase_number(epoch)

        # Only freeze during Phase 2 and 3 transitions
        if phase > 1 and epochs_in_phase <= self.freeze_backbone_epochs:
            return True
        return False

    def get_phase_number(self, epoch: int) -> int:
        """Get current phase number (1, 2, or 3)."""
        if epoch <= self.coarse_only_epochs:
            return 1
        elif epoch <= self.coarse_medium_epochs:
            return 2
        else:
            return 3

    def get_lr_multiplier(self, epoch: int) -> float:
        """Get learning rate multiplier for current epoch.

        Reduces LR during warmup period to protect backbone features
        from noisy gradients of randomly-initialized new heads.
        """
        epochs_in_phase = self.get_epochs_since_phase_start(epoch)
        phase = self.get_phase_number(epoch)

        # Phase 1: full LR
        if phase == 1:
            return 1.0

        # During warmup: use reduced LR, ramping up to full
        if epochs_in_phase <= self.warmup_epochs:
            # Ramp from transition_lr_factor to 1.0
            progress = epochs_in_phase / self.warmup_epochs
            return self.transition_lr_factor + (1.0 - self.transition_lr_factor) * progress

        return 1.0

    def get_loss_weights(
        self,
        epoch: int,
        base_weights: tuple[float, float, float, float] = (1.0, 0.5, 0.3, 0.1),
    ) -> dict[str, float]:
        """Get loss weights for current epoch with quadratic warmup.

        During warmup after activating a new head, gradually increases
        its weight from 0 to the base value using a quadratic curve
        for smoother gradient introduction.

        Args:
            epoch: Current epoch (1-indexed)
            base_weights: (coarse, medium, fine, consistency) base weights

        Returns:
            Dict with coarse_weight, medium_weight, fine_weight, consistency_weight
        """
        coarse_w, medium_w, fine_w, consistency_w = base_weights

        # Phase 1: Coarse only
        if epoch <= self.coarse_only_epochs:
            return {
                "coarse_weight": coarse_w,
                "medium_weight": 0.0,
                "fine_weight": 0.0,
                "consistency_weight": 0.0,
            }

        # Phase 2: Coarse + Medium (with quadratic warmup)
        elif epoch <= self.coarse_medium_epochs:
            linear_progress = min(1.0, (epoch - self.coarse_only_epochs) / self.warmup_epochs)
            warmup_progress = self._quadratic_warmup(linear_progress)
            return {
                "coarse_weight": coarse_w,
                "medium_weight": medium_w * warmup_progress,
                "fine_weight": 0.0,
                "consistency_weight": consistency_w * warmup_progress * 0.5,
            }

        # Phase 3: All heads (with quadratic warmup for fine)
        else:
            linear_progress = min(1.0, (epoch - self.coarse_medium_epochs) / self.warmup_epochs)
            warmup_progress = self._quadratic_warmup(linear_progress)
            return {
                "coarse_weight": coarse_w,
                "medium_weight": medium_w,
                "fine_weight": fine_w * warmup_progress,
                "consistency_weight": consistency_w,
            }

    def get_phase_name(self, epoch: int) -> str:
        """Get human-readable phase name.

        Args:
            epoch: Current epoch

        Returns:
            Phase description string
        """
        if epoch <= self.coarse_only_epochs:
            return "Phase 1: Coarse Only"
        elif epoch <= self.coarse_medium_epochs:
            return "Phase 2: Coarse + Medium"
        else:
            return "Phase 3: All Heads"
