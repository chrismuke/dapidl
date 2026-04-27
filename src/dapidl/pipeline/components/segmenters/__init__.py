"""Nucleus segmentation components.

Available segmenters:
- CellposeSegmenter: Deep learning segmentation with Cellpose
- NativeSegmenter: Pass-through using platform boundaries
- StarDistSegmenter: Deep learning segmentation with StarDist (TensorFlow)
- AdaptiveSegmenter: Density-adaptive consensus (StarDist + Cellpose)
- StarposeSegmenter: Adapter to the standalone `starpose` library
  (delegates to all starpose methods: cellpose, stardist, instanseg,
  cellvit, adaptive, majority, topological)

Usage:
    from dapidl.pipeline import get_segmenter

    segmenter = get_segmenter("cellpose", config)
    result = segmenter.segment_and_match(dapi_image, cells_df, config)

    # Starpose with its default density-adaptive consensus
    config.method = "starpose"
    # Starpose with a specific underlying method
    config.method = "starpose:topological"
"""

from dapidl.pipeline.components.segmenters.adaptive import AdaptiveSegmenter
from dapidl.pipeline.components.segmenters.cellpose import CellposeSegmenter
from dapidl.pipeline.components.segmenters.native import NativeSegmenter
from dapidl.pipeline.components.segmenters.stardist import StarDistSegmenter
from dapidl.pipeline.components.segmenters.starpose import StarposeSegmenter

__all__ = [
    "CellposeSegmenter",
    "NativeSegmenter",
    "StarDistSegmenter",
    "AdaptiveSegmenter",
    "StarposeSegmenter",
]
