"""Segmentation method adapters for the DAPIDL benchmark framework."""

from dapidl.benchmark.segmenters.base import SegmentationOutput, SegmenterAdapter
from dapidl.benchmark.segmenters.cellpose_adapter import (
    CellposeCyto3Adapter,
    CellposeNucleiAdapter,
    CellposeSAMAdapter,
)
from dapidl.benchmark.segmenters.native_adapter import NativeAdapter

try:
    from dapidl.benchmark.segmenters.stardist_adapter import StarDistAdapter
except ImportError:
    StarDistAdapter = None  # type: ignore[assignment,misc]

try:
    from dapidl.benchmark.segmenters.mesmer_adapter import MesmerAdapter
except ImportError:
    MesmerAdapter = None  # type: ignore[assignment,misc]

try:
    from dapidl.benchmark.segmenters.instanseg_adapter import InstanSegAdapter
except ImportError:
    InstanSegAdapter = None  # type: ignore[assignment,misc]

__all__ = [
    "SegmentationOutput",
    "SegmenterAdapter",
    "CellposeCyto3Adapter",
    "CellposeNucleiAdapter",
    "CellposeSAMAdapter",
    "NativeAdapter",
    "StarDistAdapter",
    "MesmerAdapter",
    "InstanSegAdapter",
]
