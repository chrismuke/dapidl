"""Cell type annotation components.

Available annotators:
- CellTypistAnnotator: Gene expression-based classification using CellTypist
- GroundTruthAnnotator: Load from curated Excel/CSV/Parquet files
- PopVAnnotator: Voting-based annotation ensemble

Usage:
    from dapidl.pipeline import get_annotator

    annotator = get_annotator("celltypist", config)
    result = annotator.annotate(config, adata=adata)
"""

from dapidl.pipeline.components.annotators.celltypist import CellTypistAnnotator
from dapidl.pipeline.components.annotators.ground_truth import GroundTruthAnnotator
from dapidl.pipeline.components.annotators.mapping import (
    BROAD_CATEGORY_MAPPING,
    CELL_TYPE_HIERARCHY,
    COARSE_CLASS_NAMES,
    map_to_broad_category,
)

# PopV annotator is optional (heavy dependencies)
try:
    from dapidl.pipeline.components.annotators.popv import PopVAnnotator
    __all__ = [
        "CellTypistAnnotator",
        "GroundTruthAnnotator",
        "PopVAnnotator",
        "CELL_TYPE_HIERARCHY",
        "BROAD_CATEGORY_MAPPING",
        "COARSE_CLASS_NAMES",
        "map_to_broad_category",
    ]
except ImportError:
    __all__ = [
        "CellTypistAnnotator",
        "GroundTruthAnnotator",
        "CELL_TYPE_HIERARCHY",
        "BROAD_CATEGORY_MAPPING",
        "COARSE_CLASS_NAMES",
        "map_to_broad_category",
    ]
