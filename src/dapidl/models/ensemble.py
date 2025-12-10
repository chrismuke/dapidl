"""Multi-scale ensemble model for cell type classification."""

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from dapidl.models.classifier import CellTypeClassifier


class MultiScaleEnsemble(nn.Module):
    """Ensemble of models trained at different patch sizes.

    Combines predictions from multiple models trained on different patch sizes
    by averaging their softmax probabilities (soft voting) or taking the
    majority class (hard voting).
    """

    def __init__(
        self,
        models: list[CellTypeClassifier],
        patch_sizes: list[int],
        fusion_method: Literal["soft", "hard", "weighted"] = "soft",
        weights: list[float] | None = None,
    ) -> None:
        """Initialize multi-scale ensemble.

        Args:
            models: List of pre-trained CellTypeClassifier models
            patch_sizes: List of patch sizes corresponding to each model
            fusion_method: How to combine predictions:
                - 'soft': average softmax probabilities (default)
                - 'hard': majority voting on predicted classes
                - 'weighted': weighted average of softmax probabilities
            weights: Weights for each model (only used with 'weighted' fusion)
        """
        super().__init__()

        if len(models) != len(patch_sizes):
            raise ValueError(
                f"Number of models ({len(models)}) must match "
                f"number of patch sizes ({len(patch_sizes)})"
            )

        if len(models) < 2:
            raise ValueError("Ensemble requires at least 2 models")

        self.models = nn.ModuleList(models)
        self.patch_sizes = patch_sizes
        self.fusion_method = fusion_method
        self.num_classes = models[0].num_classes

        # Validate all models have same number of classes
        for i, model in enumerate(models):
            if model.num_classes != self.num_classes:
                raise ValueError(
                    f"Model {i} has {model.num_classes} classes, "
                    f"expected {self.num_classes}"
                )

        # Handle weights
        if fusion_method == "weighted":
            if weights is None:
                # Default to equal weights
                weights = [1.0 / len(models)] * len(models)
            elif len(weights) != len(models):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of models ({len(models)})"
                )
            # Normalize weights
            total = sum(weights)
            self.register_buffer(
                "weights", torch.tensor([w / total for w in weights])
            )
        else:
            self.weights = None

        logger.info(
            f"MultiScaleEnsemble: {len(models)} models at patch sizes "
            f"{patch_sizes}, fusion={fusion_method}"
        )

    def forward(
        self, patches_dict: dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass with multi-scale patches.

        Args:
            patches_dict: Dictionary mapping patch_size -> tensor of patches
                Each tensor should have shape (B, 1, H, W) where H=W=patch_size

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # Collect predictions from all models
        all_probs = []

        for model, patch_size in zip(self.models, self.patch_sizes):
            if patch_size not in patches_dict:
                raise ValueError(
                    f"Missing patches for size {patch_size}. "
                    f"Available sizes: {list(patches_dict.keys())}"
                )

            patches = patches_dict[patch_size]
            logits = model(patches)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)

        # Stack probabilities: (num_models, B, num_classes)
        stacked_probs = torch.stack(all_probs, dim=0)

        if self.fusion_method == "soft":
            # Average probabilities
            fused_probs = stacked_probs.mean(dim=0)

        elif self.fusion_method == "weighted":
            # Weighted average of probabilities
            # weights: (num_models,) -> (num_models, 1, 1) for broadcasting
            weights = self.weights.view(-1, 1, 1)
            fused_probs = (stacked_probs * weights).sum(dim=0)

        elif self.fusion_method == "hard":
            # Hard voting: count votes for each class
            # Get predicted class for each model
            predictions = stacked_probs.argmax(dim=2)  # (num_models, B)

            # Count votes per class
            batch_size = predictions.shape[1]
            votes = torch.zeros(
                batch_size, self.num_classes,
                device=predictions.device, dtype=torch.float
            )
            for i in range(len(self.models)):
                votes.scatter_add_(
                    1,
                    predictions[i].unsqueeze(1),
                    torch.ones_like(predictions[i].unsqueeze(1), dtype=torch.float)
                )
            # Convert votes to probabilities
            fused_probs = votes / len(self.models)

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Return logits (log probabilities for consistency with single models)
        return torch.log(fused_probs + 1e-8)

    def forward_single_scale(
        self, patches: torch.Tensor, target_size: int = 256
    ) -> torch.Tensor:
        """Forward pass resizing a single patch to all scales.

        This method takes patches at one size and resizes them to each
        model's expected input size. Useful when you only have one patch
        per cell but want to use multi-scale ensemble.

        Args:
            patches: Input patches of shape (B, 1, H, W)
            target_size: The size of the input patches (used for validation)

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # Create patches dict by resizing
        patches_dict = {}
        for patch_size in self.patch_sizes:
            if patch_size == target_size:
                patches_dict[patch_size] = patches
            else:
                # Resize using bilinear interpolation
                resized = F.interpolate(
                    patches.float(),
                    size=(patch_size, patch_size),
                    mode="bilinear",
                    align_corners=False,
                )
                patches_dict[patch_size] = resized

        return self.forward(patches_dict)

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: list[str | Path],
        patch_sizes: list[int],
        fusion_method: Literal["soft", "hard", "weighted"] = "soft",
        weights: list[float] | None = None,
        device: str = "cuda",
    ) -> "MultiScaleEnsemble":
        """Load ensemble from checkpoint files.

        Args:
            checkpoint_paths: Paths to model checkpoint files
            patch_sizes: Patch sizes corresponding to each checkpoint
            fusion_method: How to combine predictions
            weights: Optional weights for weighted fusion
            device: Device to load models to

        Returns:
            Loaded MultiScaleEnsemble model
        """
        models = []
        for path in checkpoint_paths:
            model = CellTypeClassifier.from_checkpoint(str(path))
            model = model.to(device)
            model.eval()
            models.append(model)

        ensemble = cls(
            models=models,
            patch_sizes=patch_sizes,
            fusion_method=fusion_method,
            weights=weights,
        )
        ensemble = ensemble.to(device)
        ensemble.eval()

        return ensemble

    def eval_mode(self) -> "MultiScaleEnsemble":
        """Set all models to eval mode."""
        self.eval()
        for model in self.models:
            model.eval()
        return self

    def get_individual_predictions(
        self, patches_dict: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Get predictions from each individual model.

        Args:
            patches_dict: Dictionary mapping patch_size -> tensor of patches

        Returns:
            Dictionary mapping patch_size -> probability tensor (B, num_classes)
        """
        predictions = {}
        for model, patch_size in zip(self.models, self.patch_sizes):
            patches = patches_dict[patch_size]
            logits = model(patches)
            predictions[patch_size] = F.softmax(logits, dim=1)
        return predictions
