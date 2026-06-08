"""albumentations 2.x deprecation migration — always_apply removal + ShiftScaleRotate->Affine.

albumentations 2.0 deprecated the ``always_apply`` kwarg (governed by ``p`` now)
and ``ShiftScaleRotate`` (use ``Affine``). Both still work but emit UserWarnings,
and ShiftScaleRotate->Affine changes the API surface. These tests (a) prove no
deprecation warning is emitted when the pipelines are built and (b) pin the Affine
params to the original ShiftScaleRotate intent, so the geometry is not silently
altered by the migration (the GaussNoise-var_limit lesson applied to geometry).
"""
from __future__ import annotations

import warnings

import albumentations as A
import pytest

from dapidl.data.transforms import (
    get_heavy_augmentation_transforms,
    get_reinhard_inference_transforms,
    get_train_transforms,
)

STATS = {"p_low": 200.0, "p_high": 5000.0, "mean": 0.5, "std": 0.25}


def _flatten(compose: A.Compose) -> list:
    """All transforms in a (possibly nested) Compose."""
    out: list = []
    stack = list(compose.transforms)
    while stack:
        t = stack.pop()
        out.append(t)
        inner = getattr(t, "transforms", None)
        if inner:
            stack.extend(inner)
    return out


def _all_pipelines() -> list[A.Compose]:
    return [
        get_train_transforms(stats=STATS),
        get_train_transforms(stats=None),
        get_heavy_augmentation_transforms(stats=STATS),
        get_heavy_augmentation_transforms(stats=None),
        # Reinhard pipeline constructs the 2nd custom transform (ReinhardNormalize),
        # which also passed the deprecated always_apply kwarg.
        get_reinhard_inference_transforms(source_stats=STATS, target_stats=STATS),
    ]


def test_pipelines_emit_no_albumentations_deprecation_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _all_pipelines()
    offenders = [
        str(w.message)
        for w in caught
        if "always_apply" in str(w.message) or "ShiftScaleRotate" in str(w.message)
    ]
    assert not offenders, f"deprecation warnings still emitted: {offenders}"


@pytest.mark.parametrize("stats", [STATS, None], ids=["adaptive", "legacy"])
@pytest.mark.parametrize(
    "builder, shift, scale_lim, rot",
    [
        (get_train_transforms, 0.1, 0.1, 15.0),
        (get_heavy_augmentation_transforms, 0.15, 0.2, 30.0),
    ],
    ids=["train", "heavy"],
)
def test_shiftscalerotate_migrated_to_affine_faithfully(builder, shift, scale_lim, rot, stats):
    flat = _flatten(builder(stats=stats))

    assert not any(isinstance(t, A.ShiftScaleRotate) for t in flat), (
        "ShiftScaleRotate is deprecated -> should be migrated to Affine"
    )
    affines = [t for t in flat if isinstance(t, A.Affine)]
    assert len(affines) == 1, f"expected exactly one Affine, got {len(affines)}"

    aff = affines[0]
    # ShiftScaleRotate is the special case: shift->translate_percent(-s, s),
    # scale->(1-sc, 1+sc), rotate->(-r, r). Pin those so geometry is preserved.
    assert aff.rotate == (-rot, rot)
    assert aff.scale == {
        "x": (1 - scale_lim, 1 + scale_lim),
        "y": (1 - scale_lim, 1 + scale_lim),
    }
    assert aff.translate_percent == {"x": (-shift, shift), "y": (-shift, shift)}
