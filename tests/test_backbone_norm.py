"""F1: NuSPIRe must receive its pretraining normalization, not the default DAPI
stats. Feeding the encoder (img-0.485)/0.229 instead of its native
(img-0.219)/0.181 handicaps it and would bias an EfficientNet-vs-NuSPIRe A/B.
"""
import sys
from pathlib import Path

# scripts/ is not on pytest's pythonpath (which is ["src"]); add the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.breast_pooled_train import (  # noqa: E402
    DAPI_NORM_MEAN,
    DAPI_NORM_STD,
    backbone_norm,
)

from dapidl.models.nuspire import NUSPIRE_NORM_MEAN, NUSPIRE_NORM_STD  # noqa: E402


def test_nuspire_uses_its_pretraining_norm():
    mean, std = backbone_norm("nuspire")
    assert mean == NUSPIRE_NORM_MEAN
    assert std == NUSPIRE_NORM_STD
    assert mean != DAPI_NORM_MEAN  # the F1 bug: would have been the default


def test_default_backbone_keeps_dapi_norm():
    assert backbone_norm("efficientnetv2_rw_s") == (DAPI_NORM_MEAN, DAPI_NORM_STD)


def test_unknown_backbone_falls_back_to_dapi_norm():
    assert backbone_norm("resnet18") == (DAPI_NORM_MEAN, DAPI_NORM_STD)
