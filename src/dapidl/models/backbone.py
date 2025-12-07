"""Backbone architectures for DAPIDL."""

import timm
import torch
import torch.nn as nn
from loguru import logger


def create_backbone(
    model_name: str = "efficientnetv2_rw_s",
    pretrained: bool = True,
    in_channels: int = 1,
) -> tuple[nn.Module, int]:
    """Create a backbone model from timm.

    Args:
        model_name: Name of the timm model
        pretrained: Whether to use pretrained weights
        in_channels: Number of input channels (1 for DAPI)

    Returns:
        Tuple of (backbone model, number of output features)
    """
    # Create model without classification head
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,  # Remove classification head
        in_chans=in_channels,
    )

    # Get feature dimension
    with torch.no_grad():
        dummy_input = torch.zeros(1, in_channels, 128, 128)
        features = backbone(dummy_input)
        num_features = features.shape[1]

    logger.info(
        f"Created backbone: {model_name}, pretrained={pretrained}, "
        f"in_channels={in_channels}, num_features={num_features}"
    )

    return backbone, num_features


class SingleChannelAdapter(nn.Module):
    """Adapts single-channel input to 3-channel for pretrained models.

    Replicates the single channel to 3 channels to match pretrained
    ImageNet models that expect RGB input.
    """

    def __init__(self, method: str = "replicate") -> None:
        """Initialize adapter.

        Args:
            method: Adaptation method ('replicate' or 'learned')
        """
        super().__init__()
        self.method = method

        if method == "learned":
            # Learned 1x1 conv to expand channels
            self.expand = nn.Conv2d(1, 3, kernel_size=1, bias=False)
            nn.init.ones_(self.expand.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand single channel to 3 channels.

        Args:
            x: Input tensor of shape (B, 1, H, W)

        Returns:
            Tensor of shape (B, 3, H, W)
        """
        if self.method == "replicate":
            return x.repeat(1, 3, 1, 1)
        elif self.method == "learned":
            return self.expand(x)
        else:
            raise ValueError(f"Unknown method: {self.method}")
