"""Component registry for swappable pipeline parts.

This module provides a registry pattern for segmenters and annotators,
allowing runtime selection of implementations via configuration.

Usage:
    # Register a component (typically in the component's __init__.py)
    @register_segmenter
    class CellposeSegmenter:
        name = "cellpose"
        ...

    # Get a component by name
    segmenter = get_segmenter("cellpose", config)

    # List available components
    names = list_segmenters()  # ["cellpose", "native"]
"""

from typing import Any, TypeVar

from dapidl.pipeline.base import (
    AnnotationConfig,
    AnnotatorProtocol,
    SegmentationConfig,
    SegmenterProtocol,
)

# Type variables for generic registry
S = TypeVar("S", bound=SegmenterProtocol)
A = TypeVar("A", bound=AnnotatorProtocol)

# Global registries
_SEGMENTER_REGISTRY: dict[str, type[SegmenterProtocol]] = {}
_ANNOTATOR_REGISTRY: dict[str, type[AnnotatorProtocol]] = {}


# =============================================================================
# Segmenter Registry
# =============================================================================


def register_segmenter(cls: type[S]) -> type[S]:
    """Decorator to register a segmenter class.

    The class must have a 'name' attribute used as the registry key.

    Example:
        @register_segmenter
        class CellposeSegmenter:
            name = "cellpose"
            ...
    """
    if not hasattr(cls, "name"):
        raise ValueError(f"Segmenter class {cls.__name__} must have a 'name' attribute")

    name = cls.name
    if name in _SEGMENTER_REGISTRY:
        raise ValueError(f"Segmenter '{name}' is already registered")

    _SEGMENTER_REGISTRY[name] = cls
    return cls


def get_segmenter(
    name: str,
    config: SegmentationConfig | None = None,
    **kwargs: Any,
) -> SegmenterProtocol:
    """Get a segmenter instance by name.

    Args:
        name: Registered segmenter name (e.g., "cellpose", "native")
        config: Optional configuration to pass to constructor
        **kwargs: Additional constructor arguments

    Returns:
        Instantiated segmenter

    Raises:
        ValueError: If segmenter name is not registered
    """
    if name not in _SEGMENTER_REGISTRY:
        available = ", ".join(_SEGMENTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown segmenter '{name}'. Available: {available}"
        )

    cls = _SEGMENTER_REGISTRY[name]

    # Pass config if the constructor accepts it
    if config is not None:
        return cls(config=config, **kwargs)
    return cls(**kwargs)


def list_segmenters() -> list[str]:
    """List all registered segmenter names."""
    return list(_SEGMENTER_REGISTRY.keys())


def get_segmenter_class(name: str) -> type[SegmenterProtocol]:
    """Get a segmenter class by name (without instantiation)."""
    if name not in _SEGMENTER_REGISTRY:
        available = ", ".join(_SEGMENTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown segmenter '{name}'. Available: {available}"
        )
    return _SEGMENTER_REGISTRY[name]


# =============================================================================
# Annotator Registry
# =============================================================================


def register_annotator(cls: type[A]) -> type[A]:
    """Decorator to register an annotator class.

    The class must have a 'name' attribute used as the registry key.

    Example:
        @register_annotator
        class CellTypistAnnotator:
            name = "celltypist"
            ...
    """
    if not hasattr(cls, "name"):
        raise ValueError(f"Annotator class {cls.__name__} must have a 'name' attribute")

    name = cls.name
    if name in _ANNOTATOR_REGISTRY:
        raise ValueError(f"Annotator '{name}' is already registered")

    _ANNOTATOR_REGISTRY[name] = cls
    return cls


def get_annotator(
    name: str,
    config: AnnotationConfig | None = None,
    **kwargs: Any,
) -> AnnotatorProtocol:
    """Get an annotator instance by name.

    Args:
        name: Registered annotator name (e.g., "celltypist", "ground_truth")
        config: Optional configuration to pass to constructor
        **kwargs: Additional constructor arguments

    Returns:
        Instantiated annotator

    Raises:
        ValueError: If annotator name is not registered
    """
    if name not in _ANNOTATOR_REGISTRY:
        available = ", ".join(_ANNOTATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown annotator '{name}'. Available: {available}"
        )

    cls = _ANNOTATOR_REGISTRY[name]

    # Pass config if the constructor accepts it
    if config is not None:
        return cls(config=config, **kwargs)
    return cls(**kwargs)


def list_annotators() -> list[str]:
    """List all registered annotator names."""
    return list(_ANNOTATOR_REGISTRY.keys())


def get_annotator_class(name: str) -> type[AnnotatorProtocol]:
    """Get an annotator class by name (without instantiation)."""
    if name not in _ANNOTATOR_REGISTRY:
        available = ", ".join(_ANNOTATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown annotator '{name}'. Available: {available}"
        )
    return _ANNOTATOR_REGISTRY[name]


# =============================================================================
# Registry Utilities
# =============================================================================


def clear_registries() -> None:
    """Clear all registries (useful for testing)."""
    _SEGMENTER_REGISTRY.clear()
    _ANNOTATOR_REGISTRY.clear()


def get_component_info() -> dict[str, dict[str, list[str]]]:
    """Get information about all registered components.

    Returns:
        Dict with 'segmenters' and 'annotators' keys,
        each containing list of registered names.
    """
    return {
        "segmenters": list_segmenters(),
        "annotators": list_annotators(),
    }
