"""Multi-task cell type classifier with auxiliary segmentation.

This module implements a multi-task learning approach inspired by CelloType
(Nature Methods 2024). The key insight is that jointly training for
classification AND segmentation improves both tasks by forcing the backbone
to learn better morphological features.

Architecture Overview:
---------------------
Phase 1 (Current): Simple multi-task
    - Shared backbone (EfficientNetV2-S, etc.)
    - Classification head (main task)
    - Simple segmentation decoder (auxiliary task)

Phase 2 (Future): Multi-scale features
    - Feature Pyramid Network (FPN) for multi-scale
    - Enhanced segmentation decoder

Phase 3 (Future): Full CelloType-style
    - Swin Transformer backbone
    - MaskDINO-style decoder
    - Instance segmentation + classification

Reference:
    Pang et al., "CelloType: a unified model for segmentation and
    classification of tissue images", Nature Methods, 2024.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from dapidl.models.backbone import (
    BACKBONE_PRESETS,
    SingleChannelAdapter,
    create_backbone,
)


class SegmentationDecoder(nn.Module):
    """Simple segmentation decoder for auxiliary task.

    Phase 1: Simple bilinear upsampling with conv layers
    Phase 2: Will be extended with FPN and skip connections

    The decoder takes global features and produces a binary nucleus mask.
    This is an auxiliary task - the mask prediction helps the backbone
    learn better spatial features for classification.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        output_size: tuple[int, int] = (128, 128),
        decoder_type: Literal["simple", "unet_light"] = "simple",
    ) -> None:
        """Initialize segmentation decoder.

        Args:
            in_features: Number of input features from backbone
            hidden_dim: Hidden dimension for decoder layers
            output_size: Target output size (H, W) for segmentation mask
            decoder_type: Type of decoder architecture:
                - 'simple': Bilinear upsampling + conv (Phase 1)
                - 'unet_light': Light U-Net style with skip connections (Phase 2)
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.decoder_type = decoder_type

        if decoder_type == "simple":
            # Phase 1: Simple decoder
            # Reshape global features to spatial, then upsample
            self.project = nn.Sequential(
                nn.Linear(in_features, hidden_dim * 4 * 4),
                nn.ReLU(inplace=True),
            )

            self.decoder = nn.Sequential(
                # 4x4 -> 8x8
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),

                # 8x8 -> 16x16
                nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.ReLU(inplace=True),

                # 16x16 -> 32x32
                nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim // 4),
                nn.ReLU(inplace=True),

                # 32x32 -> 64x64
                nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, 4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim // 8),
                nn.ReLU(inplace=True),

                # 64x64 -> 128x128
                nn.ConvTranspose2d(hidden_dim // 8, 1, 4, stride=2, padding=1),
            )

        elif decoder_type == "unet_light":
            # Phase 2: Light U-Net style (placeholder for future)
            # Will use skip connections from backbone feature maps
            raise NotImplementedError(
                "unet_light decoder is planned for Phase 2. "
                "Use 'simple' for now."
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def forward(
        self,
        features: torch.Tensor,
        target_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Generate segmentation mask from features.

        Args:
            features: Global features from backbone, shape (B, in_features)
            target_size: Optional target output size (H, W). If None, uses self.output_size

        Returns:
            Segmentation logits of shape (B, 1, H, W)
        """
        target_size = target_size or self.output_size
        batch_size = features.shape[0]

        if self.decoder_type == "simple":
            # Project to spatial features
            x = self.project(features)  # (B, hidden_dim * 4 * 4)
            x = x.view(batch_size, self.hidden_dim, 4, 4)  # (B, hidden_dim, 4, 4)

            # Decode to mask
            x = self.decoder(x)  # (B, 1, 128, 128)

            # Resize if needed
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

            return x

        raise NotImplementedError(f"Decoder type {self.decoder_type} not implemented")


class MultiTaskClassifier(nn.Module):
    """Multi-task cell type classifier with auxiliary segmentation.

    This model extends CellTypeClassifier with an auxiliary segmentation task.
    The segmentation task forces the backbone to learn better spatial features
    about nuclear morphology, which improves classification performance.

    Key insight from CelloType paper:
        "DAPI-only classification achieves ~90% of DAPI+transcripts performance"

    The multi-task learning further improves this by joint optimization.

    Usage:
        # Training mode: get both outputs for multi-task loss
        class_logits, seg_mask = model(x, return_seg=True)
        loss = loss_class + lambda_seg * loss_seg

        # Inference mode: only classification (faster)
        class_logits = model(x, return_seg=False)
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "efficientnetv2_rw_s",
        pretrained: bool = True,
        dropout: float = 0.3,
        input_adapter: str = "auto",
        seg_hidden_dim: int = 256,
        seg_decoder_type: str = "simple",
        seg_weight: float = 0.5,
    ) -> None:
        """Initialize multi-task classifier.

        Args:
            num_classes: Number of cell type classes
            backbone_name: Name of backbone model
            pretrained: Use pretrained weights
            dropout: Dropout probability for classification head
            input_adapter: How to handle single-channel input
            seg_hidden_dim: Hidden dimension for segmentation decoder
            seg_decoder_type: Type of segmentation decoder ('simple', 'unet_light')
            seg_weight: Weight for segmentation loss (stored for reference)
        """
        super().__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.seg_weight = seg_weight

        # Determine input adapter based on backbone
        if input_adapter == "auto":
            if backbone_name in BACKBONE_PRESETS:
                native_channels = BACKBONE_PRESETS[backbone_name]["native_channels"]
                if native_channels == 1:
                    input_adapter = "none"
                else:
                    input_adapter = "replicate"
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

        # Classification head (main task)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes),
        )

        # Segmentation decoder (auxiliary task)
        self.seg_decoder = SegmentationDecoder(
            in_features=num_features,
            hidden_dim=seg_hidden_dim,
            output_size=(128, 128),
            decoder_type=seg_decoder_type,
        )

        logger.info(
            f"MultiTaskClassifier: {backbone_name}, "
            f"num_classes={num_classes}, "
            f"seg_decoder={seg_decoder_type}, "
            f"seg_weight={seg_weight}"
        )

    def forward(
        self,
        x: torch.Tensor,
        return_seg: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 1, H, W)
            return_seg: If True, return both classification and segmentation outputs.
                       If False, only return classification (faster inference).

        Returns:
            If return_seg=True: (class_logits, seg_mask)
                - class_logits: shape (B, num_classes)
                - seg_mask: shape (B, 1, H, W) - segmentation logits (not sigmoid)
            If return_seg=False: class_logits only
        """
        # Get input spatial size for segmentation output
        input_size = x.shape[-2:]

        # Adapt input channels if needed
        if self.input_adapter is not None:
            x = self.input_adapter(x)

        # Extract features from backbone
        features = self.backbone(x)  # (B, num_features)

        # Classification (main task)
        class_logits = self.classifier(features)  # (B, num_classes)

        if return_seg:
            # Segmentation (auxiliary task)
            seg_mask = self.seg_decoder(features, target_size=input_size)  # (B, 1, H, W)
            return class_logits, seg_mask

        return class_logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without task heads.

        Args:
            x: Input tensor of shape (B, 1, H, W)

        Returns:
            Feature tensor of shape (B, num_features)
        """
        if self.input_adapter is not None:
            x = self.input_adapter(x)
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")

    def classification_only_mode(self) -> None:
        """Switch to classification-only mode (disable segmentation gradients)."""
        for param in self.seg_decoder.parameters():
            param.requires_grad = False
        logger.info("Segmentation decoder frozen (classification-only mode)")

    def multi_task_mode(self) -> None:
        """Switch to multi-task mode (enable segmentation gradients)."""
        for param in self.seg_decoder.parameters():
            param.requires_grad = True
        logger.info("Segmentation decoder unfrozen (multi-task mode)")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "MultiTaskClassifier":
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        hparams = checkpoint.get("hparams", {})
        model = cls(
            num_classes=hparams.get("num_classes", 3),
            backbone_name=hparams.get("backbone_name", "efficientnetv2_rw_s"),
            pretrained=False,
            dropout=hparams.get("dropout", 0.3),
            input_adapter=hparams.get("input_adapter", "replicate"),
            seg_hidden_dim=hparams.get("seg_hidden_dim", 256),
            seg_decoder_type=hparams.get("seg_decoder_type", "simple"),
            seg_weight=hparams.get("seg_weight", 0.5),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded MultiTaskClassifier from {checkpoint_path}")

        return model

    @classmethod
    def from_classifier(
        cls,
        classifier_checkpoint: str,
        seg_hidden_dim: int = 256,
        seg_decoder_type: str = "simple",
        seg_weight: float = 0.5,
    ) -> "MultiTaskClassifier":
        """Initialize from a pretrained CellTypeClassifier checkpoint.

        This allows upgrading an existing trained classifier to multi-task
        by copying the backbone and classification head weights.

        Args:
            classifier_checkpoint: Path to CellTypeClassifier checkpoint
            seg_hidden_dim: Hidden dimension for segmentation decoder
            seg_decoder_type: Type of segmentation decoder
            seg_weight: Weight for segmentation loss

        Returns:
            MultiTaskClassifier with pretrained classification weights
        """
        checkpoint = torch.load(classifier_checkpoint, map_location="cpu", weights_only=False)

        hparams = checkpoint.get("hparams", {})
        model = cls(
            num_classes=hparams.get("num_classes", 3),
            backbone_name=hparams.get("backbone_name", "efficientnetv2_rw_s"),
            pretrained=False,
            dropout=hparams.get("dropout", 0.3),
            input_adapter=hparams.get("input_adapter", "replicate"),
            seg_hidden_dim=seg_hidden_dim,
            seg_decoder_type=seg_decoder_type,
            seg_weight=seg_weight,
        )

        # Load compatible weights (backbone and classifier)
        old_state = checkpoint["model_state_dict"]
        new_state = model.state_dict()

        # Map old keys to new keys
        key_mapping = {
            "head.0.": "classifier.0.",  # Dropout
            "head.1.": "classifier.1.",  # Linear
        }

        loaded_keys = []
        for old_key, value in old_state.items():
            new_key = old_key
            for old_prefix, new_prefix in key_mapping.items():
                if old_key.startswith(old_prefix):
                    new_key = old_key.replace(old_prefix, new_prefix, 1)
                    break

            if new_key in new_state:
                new_state[new_key] = value
                loaded_keys.append(new_key)

        model.load_state_dict(new_state)
        logger.info(
            f"Loaded {len(loaded_keys)} weights from CellTypeClassifier. "
            f"Segmentation decoder initialized randomly."
        )

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
                "num_classes": self.num_classes,
                "backbone_name": self.backbone_name,
                "input_adapter": self.input_adapter_name,
                "seg_hidden_dim": self.seg_decoder.hidden_dim,
                "seg_decoder_type": self.seg_decoder.decoder_type,
                "seg_weight": self.seg_weight,
            },
            "epoch": epoch,
            "metrics": metrics or {},
            "model_type": "MultiTaskClassifier",
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved MultiTaskClassifier checkpoint to {path}")


# Alias for convenience
MultiTaskCellTypeClassifier = MultiTaskClassifier
