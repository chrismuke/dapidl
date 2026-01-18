"""Domain adaptation utilities for cross-platform inference.

This module provides Adaptive Batch Normalization (AdaBN) and related techniques
for adapting models trained on one platform (e.g., Xenium) to work well on
another platform (e.g., MERSCOPE).

Key technique: AdaBN (Adaptive Batch Normalization)
    - Updates BatchNorm running statistics on target domain data
    - No gradient computation or weight updates required
    - Achieves 5-15% accuracy improvement with minimal compute cost
    - Reference: "Test-Time Domain Adaptation by Learning Domain-Aware
                 Batch Normalization" (AAAI 2024)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def adapt_batch_norm(
    model: nn.Module,
    target_loader: DataLoader | Iterator[torch.Tensor],
    num_batches: int = 10,
    device: str | torch.device | None = None,
) -> nn.Module:
    """Adapt BatchNorm statistics to target domain (AdaBN).

    This technique updates the running mean and variance of all BatchNorm
    layers in the model using target domain data, without modifying any
    learned parameters. This helps bridge the domain gap between different
    imaging platforms (e.g., Xenium vs MERSCOPE).

    Args:
        model: PyTorch model with BatchNorm layers
        target_loader: DataLoader or iterator yielding target domain batches.
                       Each batch should be a tensor of shape (B, C, H, W) or
                       a tuple (images, labels) where images is (B, C, H, W).
        num_batches: Number of batches to use for adaptation (default: 10).
                     More batches = more stable statistics but slower.
                     Typically 10-50 batches is sufficient.
        device: Device to run adaptation on. If None, uses model's device.

    Returns:
        The same model with updated BatchNorm statistics.

    Example:
        model = CellTypeClassifier.from_checkpoint("model.pt")
        target_loader = create_adaptation_loader("merscope_data/", batch_size=64)
        model = adapt_batch_norm(model, target_loader, num_batches=20)
        # Now use model for inference on MERSCOPE data
    """
    if device is None:
        # Infer device from model parameters
        device = next(model.parameters()).device

    # Count BN layers for logging
    bn_layers = _get_bn_layers(model)
    if not bn_layers:
        logger.warning("No BatchNorm layers found in model. AdaBN has no effect.")
        return model

    logger.info(
        f"Adapting {len(bn_layers)} BatchNorm layers using {num_batches} batches"
    )

    # Reset running statistics to prepare for fresh computation
    _reset_bn_statistics(model)

    # Enable training mode to update BN running stats
    # but disable gradient computation since we're not training
    model.train()

    batches_processed = 0
    with torch.no_grad():
        for batch in target_loader:
            if batches_processed >= num_batches:
                break

            # Handle both (images,) and (images, labels) formats
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            # Move to device and run forward pass
            images = images.to(device)
            _ = model(images)

            batches_processed += 1

    # Return to mode for inference
    model.train(False)

    logger.info(
        f"AdaBN complete: processed {batches_processed} batches, "
        f"adapted {len(bn_layers)} BatchNorm layers"
    )

    return model


def _get_bn_layers(model: nn.Module) -> list[nn.BatchNorm2d | nn.BatchNorm1d]:
    """Get all BatchNorm layers in a model."""
    bn_layers = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_layers.append(module)
    return bn_layers


def _reset_bn_statistics(model: nn.Module) -> None:
    """Reset BatchNorm running statistics to initial state.

    This ensures fresh statistics are computed during adaptation,
    rather than mixing with source domain statistics.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.running_mean.zero_()
            module.running_var.fill_(1.0)
            module.num_batches_tracked.zero_()


def create_adaptation_loader(
    data_path: str | Path,
    batch_size: int = 64,
    num_samples: int | None = None,
    patch_size: int = 128,
    normalize: bool = True,
    stats: dict[str, float] | None = None,
) -> DataLoader:
    """Create a DataLoader for AdaBN adaptation from LMDB or Zarr dataset.

    This is a lightweight loader specifically for domain adaptation,
    loading only the image patches without labels.

    Args:
        data_path: Path to dataset directory containing patches.lmdb or patches.zarr
        batch_size: Batch size for adaptation
        num_samples: Maximum number of samples to load. If None, uses all samples
                     up to batch_size * 50 (enough for stable statistics).
        patch_size: Expected patch size (for LMDB parsing)
        normalize: Whether to apply normalization transforms
        stats: Normalization statistics. If None, computes from data.

    Returns:
        DataLoader yielding batches of shape (B, 1, H, W)

    Example:
        loader = create_adaptation_loader(
            "datasets/merscope-breast-p128/",
            batch_size=64,
            num_samples=1000,
        )
        model = adapt_batch_norm(model, loader)
    """
    from torch.utils.data import DataLoader

    from dapidl.data.transforms import compute_dataset_stats

    data_path = Path(data_path)

    # Compute stats if not provided
    if normalize and stats is None:
        stats = compute_dataset_stats(data_path)

    # Create simple dataset
    dataset = _AdaptationDataset(
        data_path=data_path,
        patch_size=patch_size,
        num_samples=num_samples,
        normalize=normalize,
        stats=stats,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single-threaded for simplicity
        pin_memory=True,
    )


class _AdaptationDataset:
    """Lightweight dataset for AdaBN adaptation.

    Only loads image patches without labels, optimized for quick iteration.
    """

    def __init__(
        self,
        data_path: Path,
        patch_size: int = 128,
        num_samples: int | None = None,
        normalize: bool = True,
        stats: dict[str, float] | None = None,
    ):
        self.data_path = Path(data_path)
        self.patch_size = patch_size
        self.normalize = normalize
        self.stats = stats

        # Load patches from LMDB or Zarr
        lmdb_path = self.data_path / "patches.lmdb"
        zarr_path = self.data_path / "patches.zarr"

        if lmdb_path.exists():
            self._load_lmdb(lmdb_path, num_samples)
        elif zarr_path.exists():
            self._load_zarr(zarr_path, num_samples)
        else:
            raise FileNotFoundError(
                f"No patches.lmdb or patches.zarr found in {data_path}"
            )

        # Setup transform
        if normalize and stats:
            from dapidl.data.transforms import get_val_transforms
            self.transform = get_val_transforms(patch_size, stats=stats)
        else:
            self.transform = None

        logger.info(f"AdaptationDataset: loaded {len(self.patches)} patches")

    def _load_lmdb(self, lmdb_path: Path, num_samples: int | None) -> None:
        """Load patches from LMDB.

        Handles two formats:
        1. Standard: 8-byte int64 label + uint16 patch data
        2. HQ format: Pure uint16 patch (labels in separate labels.npy)
        """
        import lmdb as lmdb_lib

        env = lmdb_lib.open(str(lmdb_path), readonly=True, lock=False)

        patches = []
        expected_patch_bytes = self.patch_size * self.patch_size * 2  # uint16

        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                if key == b"__metadata__":
                    continue
                # Skip label keys from HQ format (label_00000000)
                if key.startswith(b"label_"):
                    continue

                # Detect format based on value length
                if len(value) == expected_patch_bytes:
                    # HQ format: pure uint16 patch (no label prefix)
                    patch_bytes = value
                elif len(value) == expected_patch_bytes + 8:
                    # Standard format: 8-byte int64 label + uint16 patch
                    patch_bytes = value[8:]
                else:
                    # Unknown format, skip
                    logger.warning(
                        f"Unexpected value length {len(value)}, expected "
                        f"{expected_patch_bytes} or {expected_patch_bytes + 8}"
                    )
                    continue

                patch = np.frombuffer(patch_bytes, dtype=np.uint16)
                patch = patch.reshape(self.patch_size, self.patch_size)
                patches.append(patch)

                if num_samples and len(patches) >= num_samples:
                    break

        env.close()
        self.patches = patches

    def _load_zarr(self, zarr_path: Path, num_samples: int | None) -> None:
        """Load patches from Zarr."""
        import zarr

        z = zarr.open(zarr_path, mode="r")
        n_total = z.shape[0]
        n_load = min(num_samples or n_total, n_total)

        # Random sample if limiting
        if n_load < n_total:
            rng = np.random.default_rng(42)
            indices = rng.choice(n_total, size=n_load, replace=False)
            self.patches = [z[int(i)] for i in indices]
        else:
            self.patches = [z[i] for i in range(n_load)]

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> torch.Tensor:
        patch = self.patches[idx]

        # Add channel dimension if needed (H, W) -> (H, W, 1)
        if patch.ndim == 2:
            patch = patch[:, :, np.newaxis]

        if self.transform:
            transformed = self.transform(image=patch)
            patch = transformed["image"]
        else:
            # Manual conversion to tensor
            patch = torch.from_numpy(patch.astype(np.float32))
            if patch.ndim == 2:
                patch = patch.unsqueeze(0)
            elif patch.ndim == 3 and patch.shape[-1] == 1:
                patch = patch.permute(2, 0, 1)

        return patch


def compute_domain_shift_metrics(
    model: nn.Module,
    target_loader: DataLoader,
    source_loader: DataLoader | None = None,
    num_batches: int = 10,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Compute metrics quantifying domain shift.

    When only target_loader is provided, computes basic feature statistics.
    When both loaders are provided, computes comparative metrics including MMD.

    Args:
        model: Model with BatchNorm layers
        target_loader: DataLoader for target domain (required)
        source_loader: DataLoader for source domain (optional, for comparison)
        num_batches: Number of batches to sample from each domain
        device: Device for computation

    Returns:
        Dictionary with domain shift metrics:
        - feature_mean: Mean of target feature activations
        - feature_std: Std of target feature activations
        - feature_mean_diff: (if source provided) Difference in means
        - feature_var_diff: (if source provided) Difference in variances
        - feature_mmd: (if source provided) Maximum Mean Discrepancy
    """
    if device is None:
        device = next(model.parameters()).device

    model.train(False)

    # Collect features from target domain
    target_features = []

    with torch.no_grad():
        for i, batch in enumerate(target_loader):
            if i >= num_batches:
                break
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device)
            # Get features before classification head
            if hasattr(model, "get_features"):
                features = model.get_features(images)
            else:
                features = model.backbone(images)
            target_features.append(features.cpu())

    target_features = torch.cat(target_features, dim=0)

    # Basic target statistics
    metrics = {
        "feature_mean": target_features.mean().item(),
        "feature_std": target_features.std().item(),
    }

    # If source loader provided, compute comparative metrics
    if source_loader is not None:
        source_features = []
        with torch.no_grad():
            for i, batch in enumerate(source_loader):
                if i >= num_batches:
                    break
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(device)
                if hasattr(model, "get_features"):
                    features = model.get_features(images)
                else:
                    features = model.backbone(images)
                source_features.append(features.cpu())

        source_features = torch.cat(source_features, dim=0)

        # Compute MMD (Maximum Mean Discrepancy)
        mmd = _compute_mmd(source_features, target_features)

        # Compute mean/variance differences in feature space
        source_mean = source_features.mean(dim=0)
        target_mean = target_features.mean(dim=0)
        mean_diff = (source_mean - target_mean).abs().mean().item()

        source_var = source_features.var(dim=0)
        target_var = target_features.var(dim=0)
        var_diff = (source_var - target_var).abs().mean().item()

        metrics.update({
            "feature_mean_diff": mean_diff,
            "feature_var_diff": var_diff,
            "feature_mmd": mmd,
        })

    return metrics


def _compute_mmd(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Maximum Mean Discrepancy between two distributions.

    MMD is a kernel-based distance metric between distributions.
    Higher values indicate larger domain shift.
    """
    # Simple RBF kernel MMD
    n_x = x.shape[0]
    n_y = y.shape[0]

    # Subsample if too large
    max_samples = 500
    if n_x > max_samples:
        idx = torch.randperm(n_x)[:max_samples]
        x = x[idx]
        n_x = max_samples
    if n_y > max_samples:
        idx = torch.randperm(n_y)[:max_samples]
        y = y[idx]
        n_y = max_samples

    # Compute pairwise distances
    xx = torch.cdist(x, x, p=2)
    yy = torch.cdist(y, y, p=2)
    xy = torch.cdist(x, y, p=2)

    # RBF kernel with median heuristic for bandwidth
    sigma = torch.median(xy).item()
    if sigma == 0:
        sigma = 1.0

    k_xx = torch.exp(-xx ** 2 / (2 * sigma ** 2))
    k_yy = torch.exp(-yy ** 2 / (2 * sigma ** 2))
    k_xy = torch.exp(-xy ** 2 / (2 * sigma ** 2))

    # MMD estimate
    mmd = (
        k_xx.sum() / (n_x * n_x)
        - 2 * k_xy.sum() / (n_x * n_y)
        + k_yy.sum() / (n_y * n_y)
    )

    return mmd.item()


class AdaptiveInference:
    """High-level interface for adaptive cross-platform inference.

    Wraps model loading, AdaBN adaptation, and inference in a clean API.

    Example:
        inference = AdaptiveInference("model.pt")
        inference.adapt_to_platform("datasets/merscope-breast-p128/")
        predictions = inference.predict(images)
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ):
        """Initialize adaptive inference.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device for inference. If None, uses CUDA if available.
        """
        from dapidl.models.classifier import CellTypeClassifier

        self.checkpoint_path = Path(checkpoint_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load model and move to device
        self.model = CellTypeClassifier.from_checkpoint(str(checkpoint_path))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.adapted = False
        self.target_stats = None

        logger.info(f"Loaded model from {checkpoint_path} on {device}")

    def adapt_to_platform(
        self,
        target_data_path: str | Path,
        num_batches: int = 20,
        batch_size: int = 64,
    ) -> "AdaptiveInference":
        """Adapt model to target platform using AdaBN.

        Args:
            target_data_path: Path to target domain dataset
            num_batches: Number of batches for adaptation
            batch_size: Batch size for adaptation

        Returns:
            self for method chaining
        """
        from dapidl.data.transforms import compute_dataset_stats

        target_data_path = Path(target_data_path)

        # Compute normalization stats for target domain
        self.target_stats = compute_dataset_stats(target_data_path)
        logger.info(f"Target domain stats: {self.target_stats}")

        # Create adaptation loader
        loader = create_adaptation_loader(
            target_data_path,
            batch_size=batch_size,
            num_samples=num_batches * batch_size,
            stats=self.target_stats,
        )

        # Adapt BN statistics
        self.model = adapt_batch_norm(
            self.model,
            loader,
            num_batches=num_batches,
            device=self.device,
        )

        self.adapted = True
        logger.info(f"Model adapted to {target_data_path}")

        return self

    def predict(
        self,
        images: torch.Tensor | np.ndarray,
        return_probabilities: bool = False,
    ) -> np.ndarray:
        """Run inference on images.

        Args:
            images: Input images of shape (B, 1, H, W) or (B, H, W)
            return_probabilities: If True, return softmax probabilities
                                  instead of class indices

        Returns:
            Predicted class indices (B,) or probabilities (B, num_classes)
        """
        # Convert numpy to tensor if needed
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images.astype(np.float32))

        # Add batch dimension if needed
        if images.ndim == 2:
            images = images.unsqueeze(0).unsqueeze(0)
        elif images.ndim == 3:
            if images.shape[0] != 1:  # (B, H, W)
                images = images.unsqueeze(1)  # -> (B, 1, H, W)
            else:  # (1, H, W)
                images = images.unsqueeze(0)  # -> (1, 1, H, W)

        images = images.to(self.device)

        self.model.train(False)
        with torch.no_grad():
            logits = self.model(images)

            if return_probabilities:
                probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy()
            else:
                preds = logits.argmax(dim=1)
                return preds.cpu().numpy()

    def predict_with_confidence(
        self,
        images: torch.Tensor | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference and return predictions with confidence scores.

        Args:
            images: Input images of shape (B, 1, H, W) or (B, H, W)

        Returns:
            Tuple of (predictions, confidences) where:
            - predictions: Class indices (B,)
            - confidences: Confidence scores (B,) from softmax
        """
        probs = self.predict(images, return_probabilities=True)
        predictions = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        return predictions, confidences
