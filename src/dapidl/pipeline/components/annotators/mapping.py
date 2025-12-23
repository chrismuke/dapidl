"""Cell type mappings and hierarchies for annotation.

This module defines:
- CELL_TYPE_HIERARCHY: Map detailed cell types to broad categories
- GROUND_TRUTH_MAPPING: Map Xenium ground truth labels to broad categories
- COARSE_CLASS_NAMES: Standard 3-class naming
- map_to_broad_category(): Function to resolve cell types

The hierarchies support prefix matching, so "CD4+" matches "CD4-positive, alpha-beta T cell".
More specific terms should come first in each category's list.
"""

from __future__ import annotations

# Standard coarse class names (alphabetical order for consistent indexing)
COARSE_CLASS_NAMES = ["Epithelial", "Immune", "Stromal"]

# Fine-grained classes from Xenium ground truth (17 types)
FINEGRAINED_CLASS_NAMES = [
    "B_Cells",
    "CD4+_T_Cells",
    "CD8+_T_Cells",
    "DCIS_1",
    "DCIS_2",
    "Endothelial",
    "IRF7+_DCs",
    "Invasive_Tumor",
    "LAMP3+_DCs",
    "Macrophages_1",
    "Macrophages_2",
    "Mast_Cells",
    "Myoepi_ACTA2+",
    "Myoepi_KRT15+",
    "Perivascular-Like",
    "Prolif_Invasive_Tumor",
    "Stromal",
]


# Mapping from CellTypist and popV cell types to broad categories
# Includes both CellTypist (Cells_Adult_Breast.pkl) and popV (Tabula Sapiens) cell types
# Note: Uses prefix/substring matching - order matters (more specific first)
CELL_TYPE_HIERARCHY: dict[str, list[str]] = {
    "Epithelial": [
        # CellTypist breast model
        "LummHR",
        "LummHR-SCGB",
        "LummHR-active",
        "LummHR-major",
        "Lumsec",
        "Lumsec-HLA",
        "Lumsec-KIT",
        "Lumsec-basal",
        "Lumsec-lac",
        "Lumsec-major",
        "Lumsec-myo",
        "Lumsec-prol",
        "basal",
        # popV Mammary/Epithelium models (Cell Ontology names)
        "luminal epithelial cell of mammary gland",
        "progenitor cell of mammary luminal epithelium",
        "basal cell",
        "myoepithelial cell",
        "epithelial cell",
        "epithelial fate stem cell",
        "glandular epithelial cell",
        # Organ-specific epithelial
        "hepatocyte",
        "enterocyte",
        "enterochromaffin",
        "enteroendocrine",
        "intestinal crypt stem cell",
        "intestinal tuft cell",
        "cholangiocyte",
        "kidney epithelial cell",
        "goblet cell",
        "club cell",
        "ciliated",
        "alveolar",
        "pulmonary",
        "acinar cell",
        "pancreatic",
        "paneth cell",
        "mucus secreting cell",
        "ductal cell",
        "duct epithelial cell",
        "urothelial cell",
        "keratinocyte",
        "keratocyte",
        "corneal epithelial cell",
        "conjunctival epithelial cell",
        "retinal pigment epithelial cell",
        "thymic epithelial cell",
        "salivary gland cell",
        "serous cell",
        "ionocyte",
        "ovarian surface epithelial cell",
        "mesothelial cell",
        "luminal cell of prostate",
        "sebum secreting cell",
    ],
    "Immune": [
        # CellTypist breast model
        "CD4",
        "CD8",
        "T_prol",
        "GD",  # gamma-delta T cells
        "NKT",
        "Treg",
        "b_naive",
        "bmem_switched",
        "bmem_unswitched",
        "plasma",
        "plasma_IgA",
        "plasma_IgG",
        "Macro",
        "Mono",
        "cDC",
        "mDC",
        "pDC",
        "mye-prol",
        "NK",
        "NK-ILCs",
        "Mast",
        "Neutrophil",
        # CellTypist Immune_All_High model
        "B cell",
        "B cells",
        "T cell",
        "T cells",
        "Macrophage",
        "Macrophages",
        "Dendritic",
        "DC",
        "Monocyte",
        "Monocytes",
        # popV Immune model (Cell Ontology names)
        "CD4-positive",
        "CD8-positive",
        "alpha-beta T cell",
        "alpha-beta thymocyte",
        "gamma-delta T cell",
        "regulatory T cell",
        "T follicular helper cell",
        "thymocyte",
        "natural killer cell",
        "mature NK T cell",
        "innate lymphoid cell",
        "plasma cell",
        "macrophage",
        "tissue-resident macrophage",
        "colon macrophage",
        "microglial cell",
        "monocyte",
        "classical monocyte",
        "intermediate monocyte",
        "non-classical monocyte",
        "mononuclear phagocyte",
        "neutrophil",
        "basophil",
        "mast cell",
        "granulocyte",
        "dendritic cell",
        "myeloid dendritic cell",
        "plasmacytoid dendritic cell",
        "Langerhans cell",
        "erythrocyte",
        "erythroid",
        "platelet",
        "hematopoietic",
        "leukocyte",
        "myeloid cell",
        "myeloid leukocyte",
    ],
    "Stromal": [
        # CellTypist breast model
        "Fibro",
        "Fibro-SFRP4",
        "Fibro-major",
        "Fibro-matrix",
        "Fibro-prematrix",
        "Fibroblast",
        "Fibroblasts",
        "pericytes",
        "Pericyte",
        "Pericytes",
        "vsmc",
        "Smooth muscle",
        # popV Stromal model (Cell Ontology names)
        "fibroblast",
        "fibroblast of breast",
        "myofibroblast",
        "vascular associated smooth muscle cell",
        "blood vessel smooth muscle cell",
        "smooth muscle cell",
        "pericyte",
        "adventitial cell",
        "mesenchymal stem cell",
        "mesenchymal cell",
        "stromal cell",
        "interstitial cell",
        "peritubular myoid cell",
        "stellate cell",
        "adipocyte",
        "fat cell",
        # Endothelial (grouped with Stromal for 3-class)
        "Vas",
        "Vas-arterial",
        "Vas-capillary",
        "Vas-venous",
        "Endothelial",
        "Lymph",
        "Lymph-immune",
        "Lymph-major",
        "Lymph-valve1",
        "Lymph-valve2",
        "endothelial cell",
        "vascular endothelial cell",
        "lymphatic endothelial cell",
        "capillary endothelial cell",
        "arterial endothelial cell",
        "venous endothelial cell",
    ],
}


# Ground truth cell type to broad category mapping (from Cell_Barcode_Type_Matrices.xlsx)
GROUND_TRUTH_MAPPING: dict[str, str] = {
    # Epithelial/Tumor cells
    "DCIS_1": "Epithelial",
    "DCIS_2": "Epithelial",
    "Invasive_Tumor": "Epithelial",
    "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_ACTA2+": "Epithelial",
    "Myoepi_KRT15+": "Epithelial",
    # Immune cells
    "B_Cells": "Immune",
    "CD4+_T_Cells": "Immune",
    "CD8+_T_Cells": "Immune",
    "Macrophages_1": "Immune",
    "Macrophages_2": "Immune",
    "IRF7+_DCs": "Immune",
    "LAMP3+_DCs": "Immune",
    "Mast_Cells": "Immune",
    # Stromal cells
    "Stromal": "Stromal",
    "Endothelial": "Stromal",
    "Perivascular-Like": "Stromal",
    # Hybrid/Unlabeled - excluded from training by default
    "Stromal_&_T_Cell_Hybrid": "Hybrid",
    "T_Cell_&_Tumor_Hybrid": "Hybrid",
    "Unlabeled": "Unlabeled",
    # Identity mappings for marker-based annotations
    "Epithelial": "Epithelial",
    "Immune": "Immune",
}


# Mapping from broad category to class index
BROAD_CATEGORY_MAPPING: dict[str, int] = {
    "Epithelial": 0,
    "Immune": 1,
    "Stromal": 2,
}


def map_to_broad_category(cell_type: str) -> str:
    """Map a detailed cell type to a broad category.

    Uses prefix matching - checks if the cell type starts with any keyword
    in the hierarchy. More specific keywords are checked first.

    Args:
        cell_type: Detailed cell type string from CellTypist/popV/ground truth

    Returns:
        Broad category name ("Epithelial", "Immune", "Stromal") or "Unknown"
    """
    # First try exact match in ground truth mapping
    if cell_type in GROUND_TRUTH_MAPPING:
        return GROUND_TRUTH_MAPPING[cell_type]

    # Then try prefix/substring matching in hierarchy
    cell_type_lower = cell_type.lower()
    for broad_cat, keywords in CELL_TYPE_HIERARCHY.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if (
                cell_type_lower == keyword_lower
                or cell_type_lower.startswith(keyword_lower)
            ):
                return broad_cat

    return "Unknown"


def get_class_index(cell_type: str, fine_grained: bool = False) -> int | None:
    """Get the class index for a cell type.

    Args:
        cell_type: Cell type string
        fine_grained: If True, return fine-grained index (0-16),
                     else return coarse index (0-2)

    Returns:
        Class index or None if cell type is excluded (Hybrid, Unlabeled, Unknown)
    """
    if fine_grained:
        # Fine-grained: use exact match
        if cell_type in FINEGRAINED_CLASS_NAMES:
            return FINEGRAINED_CLASS_NAMES.index(cell_type)
        return None
    else:
        # Coarse: map to broad category
        broad_cat = map_to_broad_category(cell_type)
        if broad_cat in BROAD_CATEGORY_MAPPING:
            return BROAD_CATEGORY_MAPPING[broad_cat]
        return None


def get_class_names(fine_grained: bool = False) -> list[str]:
    """Get the list of class names for the classification task.

    Args:
        fine_grained: If True, return 17 fine-grained classes,
                     else return 3 coarse classes

    Returns:
        List of class names
    """
    return FINEGRAINED_CLASS_NAMES if fine_grained else COARSE_CLASS_NAMES
