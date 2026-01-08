"""Cell Ontology Module for DAPIDL.

This module provides Cell Ontology (CL) integration for standardizing cell type
labels across different annotation methods, datasets, and tissues.

Core Components:
    - CLMapper: Main interface for mapping labels to CL IDs
    - CLLoader: OBO file loader for full ontology access
    - cl_database: Curated ~75 CL terms for human tissues

Usage:
    ```python
    from dapidl.ontology import CLMapper, map_label

    # Simple mapping
    cl_id = map_label("CD4+ T-cells")  # Returns "CL:0000624"

    # Full mapper with custom mappings
    mapper = CLMapper()
    result = mapper.map_with_info("helper T cell")
    print(f"{result.cl_id}: {result.cl_name}")
    print(f"Category: {result.broad_category} > {result.coarse_category}")
    ```

Hierarchy Levels:
    - Super-coarse (5): Epithelial, Immune, Stromal, Endothelial, Neural
    - Coarse (15): Training target for DAPIDL classifier
    - Medium (30): Common cell types
    - Fine (75): Maximum DAPI discrimination limit

ClearML Pipeline Integration:
    The CLStandardizationStep can be used as a pipeline component to
    standardize annotations before training:

    ```python
    from dapidl.pipeline.steps import CLStandardizationStep

    step = CLStandardizationStep(config=CLStandardizationConfig(
        target_level="coarse",
        min_confidence=0.7,
    ))
    ```
"""

from dapidl.ontology.cl_database import (
    # Data classes
    CLTerm,
    HierarchyLevel,
    # Term collections
    SUPER_COARSE_TERMS,
    COARSE_TERMS,
    MEDIUM_TERMS,
    FINE_TERMS,
    # Lookup functions
    get_all_terms,
    get_term,
    get_term_by_name,
    get_terms_by_level,
    # DAPIDL mappings
    DAPIDL_BROAD_CATEGORIES,
    DAPIDL_COARSE_CATEGORIES,
    CL_TO_BROAD_CATEGORY,
    CL_TO_COARSE_CATEGORY,
    get_broad_category,
    get_coarse_category,
)
from dapidl.ontology.cl_loader import (
    CLLoader,
    get_loader,
    ensure_ontology_loaded,
)
from dapidl.ontology.cl_mapper import (
    CLMapper,
    MapperConfig,
    MappingResult,
    MappingMethod,
    get_mapper,
    map_label,
    map_labels,
)
from dapidl.ontology.annotator_mappings import (
    # Per-annotator mappings
    SINGLER_TO_CL,
    CELLTYPIST_TO_CL,
    SCTYPE_TO_CL,
    SCINA_TO_CL,
    AZIMUTH_TO_CL,
    # Ground truth mappings
    XENIUM_BREAST_GT_TO_CL,
    MERSCOPE_BREAST_GT_TO_CL,
    PATHOLOGY_TO_CL,
    # Factory functions
    get_annotator_mappings,
    get_gt_mappings,
    get_all_annotator_mappings,
    get_all_gt_mappings,
)

__all__ = [
    # Core classes
    "CLTerm",
    "HierarchyLevel",
    "CLMapper",
    "MapperConfig",
    "MappingResult",
    "MappingMethod",
    "CLLoader",
    # Term collections
    "SUPER_COARSE_TERMS",
    "COARSE_TERMS",
    "MEDIUM_TERMS",
    "FINE_TERMS",
    # Database functions
    "get_all_terms",
    "get_term",
    "get_term_by_name",
    "get_terms_by_level",
    # DAPIDL mappings
    "DAPIDL_BROAD_CATEGORIES",
    "DAPIDL_COARSE_CATEGORIES",
    "CL_TO_BROAD_CATEGORY",
    "CL_TO_COARSE_CATEGORY",
    "get_broad_category",
    "get_coarse_category",
    # Loader functions
    "get_loader",
    "ensure_ontology_loaded",
    # Mapper functions
    "get_mapper",
    "map_label",
    "map_labels",
    # Annotator mappings
    "SINGLER_TO_CL",
    "CELLTYPIST_TO_CL",
    "SCTYPE_TO_CL",
    "SCINA_TO_CL",
    "AZIMUTH_TO_CL",
    "XENIUM_BREAST_GT_TO_CL",
    "MERSCOPE_BREAST_GT_TO_CL",
    "PATHOLOGY_TO_CL",
    "get_annotator_mappings",
    "get_gt_mappings",
    "get_all_annotator_mappings",
    "get_all_gt_mappings",
]
