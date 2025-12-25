"""Cell type annotation components.

Available annotators:
- CellTypistAnnotator: Gene expression-based classification using CellTypist
- GroundTruthAnnotator: Load from curated Excel/CSV/Parquet files
- PopVAnnotator: Voting-based annotation ensemble
- ConsensusAnnotator: Multi-model consensus with automatic model selection

Usage:
    from dapidl.pipeline import get_annotator

    # Standard CellTypist
    annotator = get_annotator("celltypist", config)
    result = annotator.annotate(config, adata=adata)

    # Auto-consensus (best for training data)
    annotator = get_annotator("consensus", config)
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

# Consensus annotator (auto model selection)
from dapidl.pipeline.components.annotators.consensus import (
    ConsensusAnnotator,
    ConsensusConfig,
    annotate_with_auto_consensus,
    get_high_confidence_cells,
)

# Auto selector module
from dapidl.pipeline.components.annotators.auto_selector import (
    AutoModelSelector,
    ModelScore,
    TISSUE_MODELS,
    CANONICAL_MARKERS,
)

# PopV annotator is optional (heavy dependencies)
try:
    from dapidl.pipeline.components.annotators.popv import PopVAnnotator
    __all__ = [
        "CellTypistAnnotator",
        "GroundTruthAnnotator",
        "PopVAnnotator",
        "ConsensusAnnotator",
        "ConsensusConfig",
        "AutoModelSelector",
        "ModelScore",
        "annotate_with_auto_consensus",
        "get_high_confidence_cells",
        "TISSUE_MODELS",
        "CANONICAL_MARKERS",
        "CELL_TYPE_HIERARCHY",
        "BROAD_CATEGORY_MAPPING",
        "COARSE_CLASS_NAMES",
        "map_to_broad_category",
    ]
except ImportError:
    __all__ = [
        "CellTypistAnnotator",
        "GroundTruthAnnotator",
        "ConsensusAnnotator",
        "ConsensusConfig",
        "AutoModelSelector",
        "ModelScore",
        "annotate_with_auto_consensus",
        "get_high_confidence_cells",
        "TISSUE_MODELS",
        "CANONICAL_MARKERS",
        "CELL_TYPE_HIERARCHY",
        "BROAD_CATEGORY_MAPPING",
        "COARSE_CLASS_NAMES",
        "map_to_broad_category",
    ]
