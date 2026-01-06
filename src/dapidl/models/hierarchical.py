"""Hierarchical multi-head classifier for Universal DAPI Classifier.

Architecture:
    INPUT (B, 1, 128, 128)
            │
            ▼
    ┌─────────────────────────────────────┐
    │         SHARED BACKBONE              │
    │     (EfficientNetV2-S / ConvNeXt)    │
    │            → 1792 features           │
    └─────────────────────────────────────┘
            │
            ├─────────────┬─────────────┐
            ▼             ▼             ▼
      COARSE HEAD   MEDIUM HEAD   FINE HEAD
       (3 classes)  (15 classes)  (50 classes)

Features:
    - Shared backbone extracts morphological features
    - Three classification heads for different granularities
    - Confidence-based inference fallback (fine → medium → coarse)
    - Curriculum learning support (progressively activate heads)
    - Hierarchical consistency between predictions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from dapidl.models.backbone import (
    create_backbone,
    SingleChannelAdapter,
    BACKBONE_PRESETS,
)


@dataclass
class HierarchyConfig:
    """Configuration for hierarchical classification levels.

    Defines the class structure at each hierarchy level and mappings
    between fine-grained classes and their parent categories.
    """

    # Number of classes at each level
    num_coarse: int = 3  # Epithelial, Immune, Stromal
    num_medium: int = 15  # T_Cell, B_Cell, Macrophage, Fibroblast, etc.
    num_fine: int = 50  # CD4+ T cell, CD8+ T cell, Plasma cell, etc.

    # Class names at each level
    coarse_names: list[str] = field(default_factory=list)
    medium_names: list[str] = field(default_factory=list)
    fine_names: list[str] = field(default_factory=list)

    # Mapping from fine → medium → coarse (indices)
    fine_to_medium: dict[int, int] = field(default_factory=dict)
    medium_to_coarse: dict[int, int] = field(default_factory=dict)

    # Convenience: direct fine → coarse mapping
    fine_to_coarse: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_cl_mappings(
        cls,
        class_mapping: dict[str, int],
        target_level: str = "fine",
    ) -> "HierarchyConfig":
        """Create config from CL-standardized class mapping.

        Uses the ontology module to map classes to hierarchy levels.

        Args:
            class_mapping: Fine-grained class name → index mapping
            target_level: Which level the class_mapping represents

        Returns:
            HierarchyConfig with all mappings populated
        """
        from dapidl.ontology import get_broad_category, get_coarse_category, get_term_by_name

        # Collect unique categories at each level
        fine_names = list(class_mapping.keys())
        medium_set = set()
        coarse_set = set()

        fine_to_medium_name = {}
        fine_to_coarse_name = {}

        for fine_name in fine_names:
            # Try to find CL term
            term = get_term_by_name(fine_name)
            if term:
                medium = get_coarse_category(term.cl_id) or "Unknown"
                coarse = get_broad_category(term.cl_id) or "Unknown"
            else:
                # Fallback: use the name as-is for medium, guess coarse
                medium = fine_name
                coarse = "Unknown"

            medium_set.add(medium)
            coarse_set.add(coarse)
            fine_to_medium_name[fine_name] = medium
            fine_to_coarse_name[fine_name] = coarse

        # Create index mappings
        medium_names = sorted(medium_set)
        coarse_names = sorted(coarse_set)

        medium_to_idx = {name: i for i, name in enumerate(medium_names)}
        coarse_to_idx = {name: i for i, name in enumerate(coarse_names)}

        fine_to_medium = {}
        fine_to_coarse = {}
        medium_to_coarse = {}

        for fine_name, fine_idx in class_mapping.items():
            medium_name = fine_to_medium_name[fine_name]
            coarse_name = fine_to_coarse_name[fine_name]

            fine_to_medium[fine_idx] = medium_to_idx[medium_name]
            fine_to_coarse[fine_idx] = coarse_to_idx[coarse_name]

            # Build medium → coarse mapping
            if medium_to_idx[medium_name] not in medium_to_coarse:
                medium_to_coarse[medium_to_idx[medium_name]] = coarse_to_idx[coarse_name]

        return cls(
            num_coarse=len(coarse_names),
            num_medium=len(medium_names),
            num_fine=len(fine_names),
            coarse_names=coarse_names,
            medium_names=medium_names,
            fine_names=fine_names,
            fine_to_medium=fine_to_medium,
            medium_to_coarse=medium_to_coarse,
            fine_to_coarse=fine_to_coarse,
        )


@dataclass
class HierarchicalOutput:
    """Output from hierarchical classifier.

    Contains logits and predictions at each hierarchy level,
    plus confidence scores for fallback inference.
    """

    # Logits at each level
    coarse_logits: torch.Tensor  # (B, num_coarse)
    medium_logits: torch.Tensor | None  # (B, num_medium)
    fine_logits: torch.Tensor | None  # (B, num_fine)

    # Features from backbone (for downstream use)
    features: torch.Tensor | None = None  # (B, num_features)

    @property
    def coarse_probs(self) -> torch.Tensor:
        """Softmax probabilities for coarse predictions."""
        return F.softmax(self.coarse_logits, dim=-1)

    @property
    def medium_probs(self) -> torch.Tensor | None:
        """Softmax probabilities for medium predictions."""
        if self.medium_logits is None:
            return None
        return F.softmax(self.medium_logits, dim=-1)

    @property
    def fine_probs(self) -> torch.Tensor | None:
        """Softmax probabilities for fine predictions."""
        if self.fine_logits is None:
            return None
        return F.softmax(self.fine_logits, dim=-1)

    def get_predictions(
        self,
        hierarchy_config: HierarchyConfig | None = None,
        fine_threshold: float = 0.7,
        medium_threshold: float = 0.6,
        require_consistency: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Get predictions with confidence-based fallback.

        Uses the finest granularity that meets confidence threshold,
        optionally requiring hierarchical consistency.

        Args:
            hierarchy_config: Config with parent mappings (needed for consistency)
            fine_threshold: Minimum confidence for fine predictions
            medium_threshold: Minimum confidence for medium predictions
            require_consistency: Whether to check hierarchical consistency

        Returns:
            Tuple of (predictions, confidences, level_used)
            where level_used is 'fine', 'medium', or 'coarse'
        """
        batch_size = self.coarse_logits.shape[0]
        device = self.coarse_logits.device

        # Start with coarse as default
        predictions = self.coarse_logits.argmax(dim=-1)
        confidences = self.coarse_probs.max(dim=-1).values
        levels = ["coarse"] * batch_size

        # Try medium if available
        if self.medium_logits is not None:
            medium_preds = self.medium_logits.argmax(dim=-1)
            medium_conf = self.medium_probs.max(dim=-1).values

            for i in range(batch_size):
                if medium_conf[i] >= medium_threshold:
                    # Check consistency if required
                    if require_consistency and hierarchy_config is not None:
                        expected_coarse = hierarchy_config.medium_to_coarse.get(
                            medium_preds[i].item(), -1
                        )
                        actual_coarse = predictions[i].item()
                        if expected_coarse != actual_coarse:
                            continue  # Inconsistent, keep coarse

                    predictions[i] = medium_preds[i]
                    confidences[i] = medium_conf[i]
                    levels[i] = "medium"

        # Try fine if available
        if self.fine_logits is not None:
            fine_preds = self.fine_logits.argmax(dim=-1)
            fine_conf = self.fine_probs.max(dim=-1).values

            for i in range(batch_size):
                if fine_conf[i] >= fine_threshold:
                    # Check consistency if required
                    if require_consistency and hierarchy_config is not None:
                        expected_coarse = hierarchy_config.fine_to_coarse.get(
                            fine_preds[i].item(), -1
                        )
                        actual_coarse = self.coarse_logits[i].argmax().item()
                        if expected_coarse != actual_coarse:
                            continue  # Inconsistent, keep current level

                    predictions[i] = fine_preds[i]
                    confidences[i] = fine_conf[i]
                    levels[i] = "fine"

        # Return most common level as the summary
        from collections import Counter

        level_counts = Counter(levels)
        dominant_level = level_counts.most_common(1)[0][0]

        return predictions, confidences, dominant_level


class HierarchicalClassifier(nn.Module):
    """Hierarchical multi-head classifier for Universal DAPI Classifier.

    Uses a shared backbone with three classification heads for different
    granularities (coarse, medium, fine). Supports curriculum learning
    where heads are progressively activated during training.

    Attributes:
        backbone_name: Name of the backbone model
        hierarchy_config: Configuration with class counts and mappings
        num_features: Number of features from backbone
    """

    def __init__(
        self,
        hierarchy_config: HierarchyConfig,
        backbone_name: str = "efficientnetv2_rw_s",
        pretrained: bool = True,
        dropout: float = 0.3,
        input_adapter: str = "auto",
        use_shared_projection: bool = True,
        projection_dim: int = 512,
    ) -> None:
        """Initialize hierarchical classifier.

        Args:
            hierarchy_config: Configuration with class counts and mappings
            backbone_name: Name of backbone model (from BACKBONE_PRESETS or timm)
            pretrained: Use pretrained weights for backbone
            dropout: Dropout probability for classification heads
            input_adapter: How to handle single-channel input (auto/replicate/learned/none)
            use_shared_projection: Add shared projection layer before heads
            projection_dim: Dimension of shared projection (if used)
        """
        super().__init__()

        self.hierarchy_config = hierarchy_config
        self.backbone_name = backbone_name
        self.use_shared_projection = use_shared_projection

        # Determine input adapter based on backbone
        if input_adapter == "auto":
            if backbone_name in BACKBONE_PRESETS:
                native_channels = BACKBONE_PRESETS[backbone_name]["native_channels"]
                input_adapter = "none" if native_channels == 1 else "replicate"
            else:
                input_adapter = "replicate"

        self.input_adapter_name = input_adapter

        # Input adapter for single-channel DAPI images
        if input_adapter in ("replicate", "learned"):
            self.input_adapter = SingleChannelAdapter(method=input_adapter)
            in_channels = 3
        else:
            self.input_adapter = None
            in_channels = 1

        # Shared backbone
        self.backbone, num_features = create_backbone(
            model_name=backbone_name,
            pretrained=pretrained,
            in_channels=in_channels,
        )
        self.num_features = num_features

        # Optional shared projection layer
        if use_shared_projection:
            self.projection = nn.Sequential(
                nn.Linear(num_features, projection_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            head_input_dim = projection_dim
        else:
            self.projection = None
            head_input_dim = num_features

        # Classification heads
        self.coarse_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_input_dim, hierarchy_config.num_coarse),
        )

        self.medium_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_input_dim, hierarchy_config.num_medium),
        )

        self.fine_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_input_dim, hierarchy_config.num_fine),
        )

        # Track which heads are active (for curriculum learning)
        self._active_heads = {"coarse", "medium", "fine"}

        logger.info(
            f"HierarchicalClassifier: {backbone_name}, "
            f"coarse={hierarchy_config.num_coarse}, "
            f"medium={hierarchy_config.num_medium}, "
            f"fine={hierarchy_config.num_fine}, "
            f"projection={use_shared_projection}"
        )

    def set_active_heads(self, heads: set[str]) -> None:
        """Set which heads are active during forward pass.

        Used for curriculum learning to progressively activate heads.

        Args:
            heads: Set of active heads ('coarse', 'medium', 'fine')
        """
        valid_heads = {"coarse", "medium", "fine"}
        if not heads.issubset(valid_heads):
            raise ValueError(f"Invalid heads: {heads - valid_heads}")

        if "coarse" not in heads:
            raise ValueError("Coarse head must always be active")

        self._active_heads = heads
        logger.info(f"Active heads: {sorted(self._active_heads)}")

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> HierarchicalOutput:
        """Forward pass through all active heads.

        Args:
            x: Input tensor of shape (B, 1, H, W)
            return_features: Whether to include features in output

        Returns:
            HierarchicalOutput with logits at each active level
        """
        # Adapt input channels if needed
        if self.input_adapter is not None:
            x = self.input_adapter(x)

        # Extract features from backbone
        features = self.backbone(x)

        # Apply shared projection if present
        if self.projection is not None:
            projected = self.projection(features)
        else:
            projected = features

        # Coarse classification (always active)
        coarse_logits = self.coarse_head(projected)

        # Medium classification (may be inactive during curriculum)
        medium_logits = None
        if "medium" in self._active_heads:
            medium_logits = self.medium_head(projected)

        # Fine classification (may be inactive during curriculum)
        fine_logits = None
        if "fine" in self._active_heads:
            fine_logits = self.fine_head(projected)

        return HierarchicalOutput(
            coarse_logits=coarse_logits,
            medium_logits=medium_logits,
            fine_logits=fine_logits,
            features=features if return_features else None,
        )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification heads.

        Args:
            x: Input tensor of shape (B, 1, H, W)

        Returns:
            Feature tensor of shape (B, num_features)
        """
        if self.input_adapter is not None:
            x = self.input_adapter(x)
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning heads only."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")

    def freeze_heads(self, heads: set[str]) -> None:
        """Freeze specific classification heads.

        Args:
            heads: Set of heads to freeze ('coarse', 'medium', 'fine')
        """
        if "coarse" in heads:
            for param in self.coarse_head.parameters():
                param.requires_grad = False
        if "medium" in heads:
            for param in self.medium_head.parameters():
                param.requires_grad = False
        if "fine" in heads:
            for param in self.fine_head.parameters():
                param.requires_grad = False
        logger.info(f"Froze heads: {heads}")

    def unfreeze_heads(self, heads: set[str]) -> None:
        """Unfreeze specific classification heads.

        Args:
            heads: Set of heads to unfreeze ('coarse', 'medium', 'fine')
        """
        if "coarse" in heads:
            for param in self.coarse_head.parameters():
                param.requires_grad = True
        if "medium" in heads:
            for param in self.medium_head.parameters():
                param.requires_grad = True
        if "fine" in heads:
            for param in self.fine_head.parameters():
                param.requires_grad = True
        logger.info(f"Unfroze heads: {heads}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "HierarchicalClassifier":
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        hparams = checkpoint.get("hparams", {})
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Reconstruct hierarchy config
        hierarchy_config = HierarchyConfig(
            num_coarse=hparams.get("num_coarse", 3),
            num_medium=hparams.get("num_medium", 15),
            num_fine=hparams.get("num_fine", 50),
            coarse_names=hparams.get("coarse_names", []),
            medium_names=hparams.get("medium_names", []),
            fine_names=hparams.get("fine_names", []),
            fine_to_medium=hparams.get("fine_to_medium", {}),
            medium_to_coarse=hparams.get("medium_to_coarse", {}),
            fine_to_coarse=hparams.get("fine_to_coarse", {}),
        )

        model = cls(
            hierarchy_config=hierarchy_config,
            backbone_name=hparams.get("backbone_name", "efficientnetv2_rw_s"),
            pretrained=False,
            dropout=hparams.get("dropout", 0.3),
            input_adapter=hparams.get("input_adapter", "replicate"),
            use_shared_projection=hparams.get("use_shared_projection", True),
            projection_dim=hparams.get("projection_dim", 512),
        )

        model.load_state_dict(state_dict)
        logger.info(f"Loaded hierarchical model from {checkpoint_path}")

        return model

    def save_checkpoint(
        self,
        path: str,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int = 0,
        metrics: dict | None = None,
    ) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer to save
            epoch: Current epoch
            metrics: Optional metrics dict
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "hparams": {
                "num_coarse": self.hierarchy_config.num_coarse,
                "num_medium": self.hierarchy_config.num_medium,
                "num_fine": self.hierarchy_config.num_fine,
                "coarse_names": self.hierarchy_config.coarse_names,
                "medium_names": self.hierarchy_config.medium_names,
                "fine_names": self.hierarchy_config.fine_names,
                "fine_to_medium": self.hierarchy_config.fine_to_medium,
                "medium_to_coarse": self.hierarchy_config.medium_to_coarse,
                "fine_to_coarse": self.hierarchy_config.fine_to_coarse,
                "backbone_name": self.backbone_name,
                "input_adapter": self.input_adapter_name,
                "use_shared_projection": self.use_shared_projection,
                "projection_dim": self.projection[0].out_features if self.projection else 0,
            },
            "epoch": epoch,
            "metrics": metrics or {},
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved hierarchical checkpoint to {path}")
