"""DAPIDL segmentation benchmark framework.

Provides a uniform adapter interface for comparing cell/nucleus segmentation
methods on MERSCOPE and Xenium spatial transcriptomics data.
"""

from dapidl.benchmark.segmenters.base import SegmentationOutput, SegmenterAdapter

__all__ = ["SegmentationOutput", "SegmenterAdapter"]
