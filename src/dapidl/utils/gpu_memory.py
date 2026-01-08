"""GPU Memory Prediction and Monitoring Utilities.

Predicts GPU VRAM requirements and checks resource availability
before starting memory-intensive operations.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class GPUStatus:
    """Current GPU memory status."""

    total_mb: int
    used_mb: int
    free_mb: int
    utilization: int = 0
    temperature: int = 0

    @property
    def free_gb(self) -> float:
        return self.free_mb / 1024

    @property
    def used_gb(self) -> float:
        return self.used_mb / 1024

    @property
    def usage_percent(self) -> float:
        return (self.used_mb / self.total_mb) * 100 if self.total_mb > 0 else 0


@dataclass
class MemoryEstimate:
    """Estimated memory requirements for a task."""

    vram_mb: int
    ram_mb: int
    description: str
    safe_margin_mb: int = 2048  # 2GB safety buffer

    @property
    def vram_gb(self) -> float:
        return self.vram_mb / 1024

    @property
    def ram_gb(self) -> float:
        return self.ram_mb / 1024


def get_gpu_status() -> GPUStatus | None:
    """Get current GPU memory status using nvidia-smi.

    Returns:
        GPUStatus or None if nvidia-smi fails
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        parts = result.stdout.strip().split(",")
        return GPUStatus(
            used_mb=int(parts[0].strip()),
            total_mb=int(parts[1].strip()),
            free_mb=int(parts[2].strip()),
            utilization=int(parts[3].strip()),
            temperature=int(parts[4].strip()),
        )
    except Exception as e:
        logger.warning(f"Failed to get GPU status: {e}")
        return None


# Memory estimates based on empirical measurements (24GB RTX 4090)
# Values are in MB and based on ClearML monitoring data
BACKBONE_BASE_VRAM = {
    "efficientnetv2_rw_s": 1200,  # ~1.2 GB base
    "resnet50": 800,  # ~0.8 GB base
    "convnext_tiny": 1000,  # ~1 GB base
    "efficientnet_b3": 600,  # ~0.6 GB base
}

# VRAM per batch item (varies with patch size)
VRAM_PER_BATCH_ITEM = {
    32: 2,  # 32x32 patches: ~2 MB per batch item
    64: 8,  # 64x64 patches: ~8 MB per batch item
    128: 32,  # 128x128 patches: ~32 MB per batch item
    256: 128,  # 256x256 patches: ~128 MB per batch item
}

# Cellpose inference VRAM (per tile)
CELLPOSE_VRAM = 2500  # ~2.5 GB for Cellpose inference

# RAM estimates (based on image size and number of cells)
RAM_PER_MEGAPIXEL = 100  # ~100 MB RAM per megapixel of DAPI image
RAM_PER_10K_CELLS = 50  # ~50 MB RAM per 10K cells


def estimate_training_memory(
    backbone: str = "efficientnetv2_rw_s",
    batch_size: int = 64,
    patch_size: int = 128,
    n_classes: int = 3,
) -> MemoryEstimate:
    """Estimate GPU VRAM needed for training.

    Args:
        backbone: Model backbone name
        batch_size: Training batch size
        patch_size: Patch size in pixels
        n_classes: Number of output classes

    Returns:
        MemoryEstimate with VRAM and RAM requirements
    """
    # Base backbone memory
    base_vram = BACKBONE_BASE_VRAM.get(backbone, 1000)

    # Batch memory (forward + backward = 2x)
    vram_per_item = VRAM_PER_BATCH_ITEM.get(patch_size, 32)
    batch_vram = batch_size * vram_per_item * 2  # 2x for gradients

    # Optimizer states (Adam uses 2x model size)
    optimizer_vram = base_vram * 2

    # Total estimate
    total_vram = base_vram + batch_vram + optimizer_vram

    # RAM estimate (data loading + transforms)
    # patch_size^2 * 4 bytes (float32) * batch_size * 3 workers / 1024^2 to MB
    ram_mb = (batch_size * patch_size * patch_size * 4 * 3) // (1024 * 1024) + 500  # +500 MB baseline

    return MemoryEstimate(
        vram_mb=total_vram,
        ram_mb=ram_mb,
        description=f"Training {backbone} with batch_size={batch_size}, patch={patch_size}px",
    )


def estimate_segmentation_memory(
    image_shape: tuple[int, int],
    segmenter: str = "cellpose",
    tile_size: int = 4096,
) -> MemoryEstimate:
    """Estimate GPU VRAM needed for segmentation.

    Args:
        image_shape: (height, width) of DAPI image
        segmenter: "cellpose" or "native"
        tile_size: Tile size for tiled processing

    Returns:
        MemoryEstimate with VRAM and RAM requirements
    """
    if segmenter == "native":
        # Native just loads boundaries, minimal GPU
        return MemoryEstimate(
            vram_mb=256,
            ram_mb=500,
            description="Native segmentation (no GPU)",
        )

    # Cellpose VRAM per tile + model
    vram_mb = CELLPOSE_VRAM

    # RAM: full DAPI image + mask array
    h, w = image_shape
    image_mb = (h * w * 2) // (1024 * 1024)  # uint16
    mask_mb = (h * w * 4) // (1024 * 1024)  # uint32
    ram_mb = (image_mb + mask_mb) * 2 + 4000  # 2x for processing + buffer

    return MemoryEstimate(
        vram_mb=vram_mb,
        ram_mb=ram_mb,
        description=f"Cellpose segmentation of {h}x{w} image ({h*w//1e6:.1f} megapixels)",
    )


def check_resources_available(estimate: MemoryEstimate) -> tuple[bool, str]:
    """Check if sufficient GPU and RAM resources are available.

    Args:
        estimate: Memory requirements estimate

    Returns:
        (is_available, message) tuple
    """
    gpu_status = get_gpu_status()

    if gpu_status is None:
        return False, "Could not query GPU status"

    required_vram = estimate.vram_mb + estimate.safe_margin_mb

    if gpu_status.free_mb < required_vram:
        return False, (
            f"Insufficient GPU memory: need {required_vram} MB, "
            f"only {gpu_status.free_mb} MB free"
        )

    # Check system RAM using /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available_kb = int(line.split()[1])
                    available_mb = available_kb // 1024

                    required_ram = estimate.ram_mb + 2048  # 2GB buffer

                    if available_mb < required_ram:
                        return False, (
                            f"Insufficient RAM: need {required_ram} MB, "
                            f"only {available_mb} MB free"
                        )
                    break
    except Exception as e:
        logger.warning(f"Could not check RAM: {e}")

    return True, (
        f"Resources OK: GPU {gpu_status.free_mb} MB free "
        f"(need {estimate.vram_mb} MB + {estimate.safe_margin_mb} MB buffer)"
    )


def wait_for_gpu_resources(
    required_mb: int = 4000,
    check_interval_sec: int = 30,
    max_wait_sec: int = 3600,
) -> bool:
    """Wait until GPU has sufficient free memory.

    Args:
        required_mb: Required free GPU memory in MB
        check_interval_sec: Seconds between checks
        max_wait_sec: Maximum seconds to wait

    Returns:
        True if resources became available, False if timeout
    """
    import time

    start_time = time.time()

    while time.time() - start_time < max_wait_sec:
        gpu_status = get_gpu_status()
        if gpu_status and gpu_status.free_mb >= required_mb:
            logger.info(f"GPU resources available: {gpu_status.free_mb} MB free")
            return True

        logger.info(
            f"Waiting for GPU memory ({gpu_status.free_mb if gpu_status else 'unknown'} MB free, "
            f"need {required_mb} MB)..."
        )
        time.sleep(check_interval_sec)

    logger.warning(f"Timeout waiting for GPU resources after {max_wait_sec}s")
    return False


def log_gpu_status() -> None:
    """Log current GPU status."""
    gpu_status = get_gpu_status()
    if gpu_status:
        logger.info(
            f"GPU Status: {gpu_status.used_mb} MB used / {gpu_status.total_mb} MB total "
            f"({gpu_status.free_mb} MB free, {gpu_status.utilization}% util, {gpu_status.temperature}°C)"
        )
    else:
        logger.warning("Could not query GPU status")
