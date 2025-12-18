"""Backbone architectures for DAPIDL."""

from typing import Literal

import timm
import torch
import torch.nn as nn
from loguru import logger


# Available backbone presets for easy selection
BACKBONE_PRESETS = {
    # ImageNet pretrained (requires channel adaptation for single-channel input)
    "efficientnetv2_rw_s": {
        "description": "EfficientNetV2-S (default, ~20M params)",
        "pretrained": True,
        "native_channels": 3,
    },
    "resnet18": {
        "description": "ResNet-18 (lighter, ~11M params)",
        "pretrained": True,
        "native_channels": 3,
    },
    "resnet34": {
        "description": "ResNet-34 (medium, ~21M params)",
        "pretrained": True,
        "native_channels": 3,
    },
    "convnext_tiny": {
        "description": "ConvNeXt-Tiny (modern, ~28M params)",
        "pretrained": True,
        "native_channels": 3,
    },
    "convnext_small": {
        "description": "ConvNeXt-Small (larger modern, ~50M params)",
        "pretrained": True,
        "native_channels": 3,
    },
    # Custom microscopy-optimized (native single-channel)
    "microscopy_cnn": {
        "description": "Custom lightweight CNN for microscopy (~1M params)",
        "pretrained": False,
        "native_channels": 1,
    },
    "microscopy_cnn_deep": {
        "description": "Deeper custom CNN for microscopy (~21M params)",
        "pretrained": False,
        "native_channels": 1,
    },
    # Pathology Foundation Models
    "phikon_v2": {
        "description": "Phikon-v2 ViT-L/16 pathology foundation model (~0.3B params)",
        "pretrained": True,
        "native_channels": 3,
        "foundation_model": True,
        "hf_repo": "owkin/phikon-v2",
    },
    "uni": {
        "description": "UNI ViT-L/16 pathology foundation model (~0.3B params, requires access)",
        "pretrained": True,
        "native_channels": 3,
        "foundation_model": True,
        "hf_repo": "MahmoodLab/UNI",
    },
    # Vision Transformers (via timm)
    "vit_base_patch16_224": {
        "description": "ViT-Base/16 ImageNet pretrained (~86M params)",
        "pretrained": True,
        "native_channels": 3,
    },
    "vit_large_patch16_224": {
        "description": "ViT-Large/16 ImageNet pretrained (~304M params)",
        "pretrained": True,
        "native_channels": 3,
    },
    "vit_base_patch16_224.dino": {
        "description": "ViT-Base/16 DINO self-supervised (~86M params)",
        "pretrained": True,
        "native_channels": 3,
    },
}


def list_backbones() -> dict[str, str]:
    """List available backbone presets with descriptions.

    Returns:
        Dict mapping backbone name to description
    """
    return {name: info["description"] for name, info in BACKBONE_PRESETS.items()}


def create_backbone(
    model_name: str = "efficientnetv2_rw_s",
    pretrained: bool = True,
    in_channels: int = 1,
) -> tuple[nn.Module, int]:
    """Create a backbone model.

    Args:
        model_name: Name of the backbone (see BACKBONE_PRESETS or any timm model)
        pretrained: Whether to use pretrained weights (ignored for custom models)
        in_channels: Number of input channels (1 for DAPI)

    Returns:
        Tuple of (backbone model, number of output features)
    """
    # Check if it's a custom microscopy model
    if model_name == "microscopy_cnn":
        backbone = MicroscopyCNN(in_channels=in_channels, deep=False)
        num_features = backbone.num_features
        logger.info(
            f"Created backbone: {model_name}, "
            f"in_channels={in_channels}, num_features={num_features}"
        )
        return backbone, num_features

    elif model_name == "microscopy_cnn_deep":
        backbone = MicroscopyCNN(in_channels=in_channels, deep=True)
        num_features = backbone.num_features
        logger.info(
            f"Created backbone: {model_name}, "
            f"in_channels={in_channels}, num_features={num_features}"
        )
        return backbone, num_features

    # Check if it's a pathology foundation model
    elif model_name in BACKBONE_PRESETS and BACKBONE_PRESETS[model_name].get(
        "foundation_model", False
    ):
        backbone = PathologyFoundationModel(
            model_name=model_name,
            pretrained=pretrained,
        )
        num_features = backbone.num_features
        logger.info(
            f"Created pathology foundation model: {model_name}, "
            f"num_features={num_features}"
        )
        return backbone, num_features

    # Otherwise use timm
    # Check if it's a ViT model that needs dynamic image size
    is_vit = model_name.startswith("vit_") or ".dino" in model_name

    # Build kwargs - only add dynamic_img_size for ViT models
    timm_kwargs: dict = {
        "pretrained": pretrained,
        "num_classes": 0,  # Remove classification head
        "in_chans": in_channels,
    }
    if is_vit:
        timm_kwargs["dynamic_img_size"] = True

    backbone = timm.create_model(model_name, **timm_kwargs)

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


class MicroscopyCNN(nn.Module):
    """Custom lightweight CNN optimized for single-channel microscopy images.

    Designed specifically for DAPI nuclear staining classification:
    - Native single-channel input (no channel adaptation needed)
    - Optimized for 128x128 patches
    - Smaller, faster than pretrained ImageNet models
    - Uses batch normalization and residual connections
    """

    def __init__(self, in_channels: int = 1, deep: bool = False) -> None:
        """Initialize MicroscopyCNN.

        Args:
            in_channels: Number of input channels (typically 1 for DAPI)
            deep: If True, use deeper architecture with more capacity
        """
        super().__init__()
        self.deep = deep

        if deep:
            # Deeper version (~4M params)
            self.features = nn.Sequential(
                # Block 1: 128 -> 64
                ConvBlock(in_channels, 64, stride=2),
                ResidualBlock(64, 64),

                # Block 2: 64 -> 32
                ConvBlock(64, 128, stride=2),
                ResidualBlock(128, 128),
                ResidualBlock(128, 128),

                # Block 3: 32 -> 16
                ConvBlock(128, 256, stride=2),
                ResidualBlock(256, 256),
                ResidualBlock(256, 256),

                # Block 4: 16 -> 8
                ConvBlock(256, 512, stride=2),
                ResidualBlock(512, 512),
                ResidualBlock(512, 512),

                # Block 5: 8 -> 4
                ConvBlock(512, 512, stride=2),
                ResidualBlock(512, 512),

                # Global pooling
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.num_features = 512
        else:
            # Lightweight version (~1M params)
            self.features = nn.Sequential(
                # Block 1: 128 -> 64
                ConvBlock(in_channels, 32, stride=2),

                # Block 2: 64 -> 32
                ConvBlock(32, 64, stride=2),
                ResidualBlock(64, 64),

                # Block 3: 32 -> 16
                ConvBlock(64, 128, stride=2),
                ResidualBlock(128, 128),

                # Block 4: 16 -> 8
                ConvBlock(128, 256, stride=2),
                ResidualBlock(256, 256),

                # Block 5: 8 -> 4
                ConvBlock(256, 256, stride=2),

                # Global pooling
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.num_features = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Feature tensor of shape (B, num_features)
        """
        return self.features(x)


class ConvBlock(nn.Module):
    """Basic convolution block with BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with two convolutions."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Skip connection with 1x1 conv if channels change
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


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


class PathologyFoundationModel(nn.Module):
    """Wrapper for pathology foundation models (Phikon-v2, UNI, etc.).

    These models are pretrained on large histopathology datasets and can provide
    better feature extraction for cell type classification than ImageNet models.

    Note: Foundation models expect 3-channel RGB input. For DAPI (single-channel),
    the classifier's SingleChannelAdapter handles the channel expansion.
    """

    def __init__(
        self,
        model_name: str = "phikon_v2",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize pathology foundation model.

        Args:
            model_name: Name of the foundation model ('phikon_v2', 'uni')
            pretrained: Whether to load pretrained weights
            freeze_backbone: Whether to freeze backbone weights (for linear probing)
        """
        super().__init__()
        self.model_name = model_name

        if model_name not in BACKBONE_PRESETS:
            raise ValueError(f"Unknown model: {model_name}")

        preset = BACKBONE_PRESETS[model_name]
        if not preset.get("foundation_model", False):
            raise ValueError(f"{model_name} is not a foundation model")

        hf_repo = preset["hf_repo"]

        if model_name == "phikon_v2":
            self._load_phikon_v2(hf_repo, pretrained)
        elif model_name == "uni":
            self._load_uni(hf_repo, pretrained)
        else:
            raise ValueError(f"Unknown foundation model: {model_name}")

        if freeze_backbone:
            self.freeze()

    def _load_phikon_v2(self, hf_repo: str, pretrained: bool) -> None:
        """Load Phikon-v2 model using transformers."""
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "transformers library required for Phikon-v2. "
                "Install with: pip install transformers"
            )

        if pretrained:
            logger.info(f"Loading Phikon-v2 from HuggingFace: {hf_repo}")
            self.model = AutoModel.from_pretrained(hf_repo)
        else:
            # Load architecture without pretrained weights
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(hf_repo)
            self.model = AutoModel.from_config(config)

        self.num_features = 1024  # ViT-L output dimension

    def _load_uni(self, hf_repo: str, pretrained: bool) -> None:
        """Load UNI model using timm.

        Note: UNI requires HuggingFace access approval.
        Visit https://huggingface.co/MahmoodLab/UNI to request access.
        """
        try:
            from huggingface_hub import login
        except ImportError:
            raise ImportError(
                "huggingface_hub library required for UNI. "
                "Install with: pip install huggingface-hub"
            )

        if pretrained:
            logger.info(f"Loading UNI from HuggingFace: {hf_repo}")
            logger.info("Note: UNI requires HuggingFace access approval")

            try:
                # UNI uses timm with HuggingFace hub
                self.model = timm.create_model(
                    f"hf-hub:{hf_repo}",
                    pretrained=True,
                    init_values=1e-5,
                    dynamic_img_size=True,
                    num_classes=0,  # Remove classification head
                )
            except Exception as e:
                logger.error(
                    f"Failed to load UNI. Make sure you have access approved at "
                    f"https://huggingface.co/MahmoodLab/UNI and are logged in "
                    f"with `huggingface-cli login`. Error: {e}"
                )
                raise
        else:
            # Load architecture without pretrained weights
            self.model = timm.create_model(
                "vit_large_patch16_224",
                pretrained=False,
                init_values=1e-5,
                dynamic_img_size=True,
                num_classes=0,
            )

        self.num_features = 1024  # ViT-L output dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input.

        Args:
            x: Input tensor of shape (B, 3, H, W)
               Note: Foundation models expect 3-channel input.
               Use SingleChannelAdapter for DAPI images.

        Returns:
            Feature tensor of shape (B, num_features)
        """
        if self.model_name == "phikon_v2":
            # Phikon-v2 returns last_hidden_state, take CLS token
            outputs = self.model(pixel_values=x)
            # CLS token is first position
            features = outputs.last_hidden_state[:, 0, :]
        else:
            # UNI (timm model) returns features directly
            features = self.model(x)

        return features

    def freeze(self) -> None:
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info(f"Froze {self.model_name} backbone")

    def unfreeze(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info(f"Unfroze {self.model_name} backbone")
