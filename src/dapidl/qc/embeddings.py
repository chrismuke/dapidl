"""Frozen vision-FM embedders for DAPI QC crops (DINOv2 + NuSPIRe), with per-model preprocessing."""
from __future__ import annotations

import numpy as np

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _stretch01(patch):
    p = np.asarray(patch, dtype=np.float32)
    lo, hi = np.percentile(p, [1.0, 99.0])
    hi = hi if hi > lo else lo + 1.0
    return np.clip((p - lo) / (hi - lo), 0.0, 1.0)


def _resize(img2d, size):
    from PIL import Image
    return np.asarray(Image.fromarray((img2d * 255).astype(np.uint8)).resize((size, size),
                                                                             Image.BILINEAR),
                      dtype=np.float32) / 255.0


def preprocess_dinov2(patch, size=224):
    g = _resize(_stretch01(patch), size)             # [size,size] in [0,1]
    rgb = np.repeat(g[None, :, :], 3, axis=0)        # [3,size,size]
    return ((rgb - _IMAGENET_MEAN[:, None, None]) / _IMAGENET_STD[:, None, None]).astype(np.float32)


def preprocess_nuspire(patch, size=112):
    g = _resize(_stretch01(patch), size)
    # NuSPIRe trained on single-channel DAPI; 0.5/0.5 placeholder normalization,
    # to be replaced with the loader's documented stats when the real model is wired in.
    return (((g - 0.5) / 0.5)[None, :, :]).astype(np.float32)
