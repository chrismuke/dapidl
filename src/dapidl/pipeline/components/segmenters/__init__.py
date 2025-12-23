"""Nucleus segmentation components.

Available segmenters:
- CellposeSegmenter: Deep learning segmentation with Cellpose
- NativeSegmenter: Pass-through using platform boundaries

Usage:
    from dapidl.pipeline import get_segmenter

    segmenter = get_segmenter("cellpose", config)
    result = segmenter.segment_and_match(dapi_image, cells_df, config)
"""

from dapidl.pipeline.components.segmenters.cellpose import CellposeSegmenter
from dapidl.pipeline.components.segmenters.native import NativeSegmenter

__all__ = ["CellposeSegmenter", "NativeSegmenter"]
