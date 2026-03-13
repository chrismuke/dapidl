"""Tests for declarative annotation registry refactor."""

from dapidl.pipeline.base import AnnotationConfig


def test_annotation_config_has_singler_reference():
    """AnnotationConfig must have singler_reference for SingleR param passing."""
    cfg = AnnotationConfig()
    assert hasattr(cfg, "singler_reference")
    assert cfg.singler_reference == "blueprint"


def test_annotation_config_singler_reference_custom():
    cfg = AnnotationConfig(singler_reference="hpca")
    assert cfg.singler_reference == "hpca"
