"""Cell type classifier model."""

import torch
import torch.nn as nn
from loguru import logger

from dapidl.models.backbone import create_backbone, SingleChannelAdapter


class CellTypeClassifier(nn.Module):
    """Cell type classifier using pretrained backbone.

    Combines a pretrained backbone (EfficientNet, ResNet, etc.) with
    a classification head for cell type prediction from DAPI patches.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "efficientnetv2_rw_s",
        pretrained: bool = True,
        dropout: float = 0.3,
        input_adapter: str = "replicate",
    ) -> None:
        """Initialize classifier.

        Args:
            num_classes: Number of cell type classes
            backbone_name: Name of timm backbone model
            pretrained: Use pretrained weights
            dropout: Dropout probability for classification head
            input_adapter: How to handle single-channel input ('replicate' or 'learned')
        """
        super().__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.input_adapter_name = input_adapter

        # Input adapter for single-channel DAPI images
        if input_adapter in ("replicate", "learned"):
            self.input_adapter = SingleChannelAdapter(method=input_adapter)
            in_channels = 3
        else:
            self.input_adapter = None
            in_channels = 1

        # Backbone
        self.backbone, num_features = create_backbone(
            model_name=backbone_name,
            pretrained=pretrained,
            in_channels=in_channels,
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes),
        )

        logger.info(
            f"CellTypeClassifier: {backbone_name}, "
            f"num_classes={num_classes}, dropout={dropout}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 1, H, W)

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # Adapt input channels if needed
        if self.input_adapter is not None:
            x = self.input_adapter(x)

        # Extract features
        features = self.backbone(x)

        # Classify
        logits = self.head(features)

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head.

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

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "CellTypeClassifier":
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract hyperparameters
        hparams = checkpoint.get("hparams", {})
        model = cls(
            num_classes=hparams.get("num_classes", 5),
            backbone_name=hparams.get("backbone_name", "efficientnetv2_rw_s"),
            pretrained=False,
            dropout=hparams.get("dropout", 0.3),
            input_adapter=hparams.get("input_adapter", "replicate"),
        )

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model from {checkpoint_path}")

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
            },
            "epoch": epoch,
            "metrics": metrics or {},
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
