"""Cell type annotation components.

Available annotators:
- CellTypistAnnotator: Gene expression-based classification using CellTypist
- GroundTruthAnnotator: Load from curated Excel/CSV/Parquet files
- PopVAnnotator: Voting-based annotation ensemble
- ConsensusAnnotator: Multi-model consensus with automatic model selection
- SingleRAnnotator: R-based reference annotation (requires R with SingleR package)
- ScTypeAnnotator: Marker-based annotation using scType method
- SCINAAnnotator: EM-based marker annotation using SCINA algorithm
- ScANVIAnnotator: Deep learning semi-supervised annotation
- ScArchesAnnotator: Reference mapping via surgical training
- AzimuthAnnotator: R-based Seurat/Azimuth reference mapping

Usage:
    from dapidl.pipeline import get_annotator

    # Standard CellTypist
    annotator = get_annotator("celltypist", config)
    result = annotator.annotate(config, adata=adata)

    # Auto-consensus (best for training data)
    annotator = get_annotator("consensus", config)
    result = annotator.annotate(config, adata=adata)

    # SingleR (best accuracy on Xenium - 92%)
    annotator = get_annotator("singler", config)
    result = annotator.annotate(config, adata=adata)

    # Marker-based (no reference needed)
    annotator = get_annotator("sctype", config)
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
    _has_popv = True
except ImportError:
    _has_popv = False

# SingleR annotator is optional (requires R with SingleR package)
try:
    from dapidl.pipeline.components.annotators.singler import SingleRAnnotator
    _has_singler = True
except ImportError:
    _has_singler = False

# Marker-based annotators
from dapidl.pipeline.components.annotators.sctype import ScTypeAnnotator
from dapidl.pipeline.components.annotators.scina import SCINAAnnotator

# Deep learning annotators (optional - require scvi-tools/scarches)
try:
    from dapidl.pipeline.components.annotators.scanvi import ScANVIAnnotator
    _has_scanvi = True
except ImportError:
    _has_scanvi = False

try:
    from dapidl.pipeline.components.annotators.scarches import ScArchesAnnotator
    _has_scarches = True
except ImportError:
    _has_scarches = False

# Azimuth annotator is optional (requires R with Seurat/Azimuth)
try:
    from dapidl.pipeline.components.annotators.azimuth import AzimuthAnnotator
    _has_azimuth = True
except ImportError:
    _has_azimuth = False

# Build __all__ based on available optional annotators
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
    "ScTypeAnnotator",
    "SCINAAnnotator",
]

if _has_popv:
    __all__.append("PopVAnnotator")

if _has_singler:
    __all__.append("SingleRAnnotator")

if _has_scanvi:
    __all__.append("ScANVIAnnotator")

if _has_scarches:
    __all__.append("ScArchesAnnotator")

if _has_azimuth:
    __all__.append("AzimuthAnnotator")
