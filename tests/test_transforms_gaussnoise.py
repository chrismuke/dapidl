"""GaussNoise augmentation magnitude — regression guard for albumentations 2.x API drift.

albumentations 2.0 removed ``GaussNoise(var_limit=...)`` in favour of ``std_range``.
Passing the old ``var_limit`` is silently ignored, so the transform falls back to
the library default ``std_range=(0.2, 0.44)`` — 3-8x stronger than intended on the
normalized float patches DAPIDL trains on. These tests pin the *intended* noise std
(derived from the documented variance via ``std = sqrt(var)``) so the silent drop
cannot recur.
"""
from __future__ import annotations

import math

import albumentations as A
import pytest

from dapidl.data.transforms import (
    get_heavy_augmentation_transforms,
    get_train_transforms,
)

# What GaussNoise reverts to when its std argument is silently dropped.
ALBUMENTATIONS_DEFAULT_STD_RANGE = (0.2, 0.44)

# Representative adaptive-normalization stats (values are arbitrary but valid).
STATS = {"p_low": 200.0, "p_high": 5000.0, "mean": 0.5, "std": 0.25}


def _gaussnoise_std_ranges(compose: A.Compose) -> list[tuple[float, float]]:
    """Collect ``std_range`` from every GaussNoise in a (possibly nested) Compose."""
    found: list[tuple[float, float]] = []
    stack = list(compose.transforms)
    while stack:
        t = stack.pop()
        if isinstance(t, A.GaussNoise):
            found.append((float(t.std_range[0]), float(t.std_range[1])))
        inner = getattr(t, "transforms", None)
        if inner:
            stack.extend(inner)
    return found


@pytest.mark.parametrize("stats", [STATS, None], ids=["adaptive", "legacy"])
def test_train_gaussnoise_uses_intended_std_not_library_default(stats):
    # Documented intent: variance (0.001, 0.01) on [0,1] float -> std = sqrt(var).
    expected = (math.sqrt(0.001), math.sqrt(0.01))
    ranges = _gaussnoise_std_ranges(get_train_transforms(stats=stats))

    assert ranges, "no GaussNoise found in train transforms"
    for lo, hi in ranges:
        assert (lo, hi) != ALBUMENTATIONS_DEFAULT_STD_RANGE, (
            "GaussNoise fell back to the albumentations default -> "
            "the noise magnitude was silently dropped"
        )
        assert lo == pytest.approx(expected[0], abs=1e-4)
        assert hi == pytest.approx(expected[1], abs=1e-4)


@pytest.mark.parametrize("stats", [STATS, None], ids=["adaptive", "legacy"])
def test_heavy_gaussnoise_uses_intended_std_not_library_default(stats):
    # Documented intent: variance (0.005, 0.02) -> std = sqrt(var).
    expected = (math.sqrt(0.005), math.sqrt(0.02))
    ranges = _gaussnoise_std_ranges(get_heavy_augmentation_transforms(stats=stats))

    assert ranges, "no GaussNoise found in heavy transforms"
    for lo, hi in ranges:
        assert (lo, hi) != ALBUMENTATIONS_DEFAULT_STD_RANGE, (
            "GaussNoise fell back to the albumentations default -> "
            "the noise magnitude was silently dropped"
        )
        assert lo == pytest.approx(expected[0], abs=1e-4)
        assert hi == pytest.approx(expected[1], abs=1e-4)
