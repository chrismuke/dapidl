"""Cell Ontology-based standardization for cell type annotations.

This module provides standardized mappings from various annotation method outputs
to Cell Ontology (CL) terms, and from CL terms to ground truth labels.

References:
    - Cell Ontology: https://obofoundry.org/ontology/cl.html
    - CellOntologyMapper: https://github.com/Starlitnightly/CellOntologyMapper
    - SingleR CL integration: https://bioconductor.org/books/release/SingleRBook/exploiting-the-cell-ontology.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CellOntologyTerm:
    """A Cell Ontology term with ID, name, and synonyms."""
    cl_id: str  # e.g., "CL:0000624"
    name: str  # e.g., "CD4-positive, alpha-beta T cell"
    synonyms: list[str]  # Alternative names
    parent_id: Optional[str] = None  # Parent term for hierarchy


# =============================================================================
# Cell Ontology Terms (subset relevant for breast tissue)
# =============================================================================

CELL_ONTOLOGY = {
    # T cells
    "CL:0000084": CellOntologyTerm("CL:0000084", "T cell", ["T-cell", "T lymphocyte"]),
    "CL:0000624": CellOntologyTerm("CL:0000624", "CD4-positive, alpha-beta T cell",
                                    ["CD4+ T cell", "helper T cell", "CD4 T cell"], parent_id="CL:0000084"),
    "CL:0000625": CellOntologyTerm("CL:0000625", "CD8-positive, alpha-beta T cell",
                                    ["CD8+ T cell", "cytotoxic T cell", "CD8 T cell"], parent_id="CL:0000084"),
    "CL:0000815": CellOntologyTerm("CL:0000815", "regulatory T cell",
                                    ["Treg", "Tregs", "suppressor T cell"], parent_id="CL:0000624"),

    # B cells
    "CL:0000236": CellOntologyTerm("CL:0000236", "B cell", ["B-cell", "B lymphocyte"]),
    "CL:0000786": CellOntologyTerm("CL:0000786", "plasma cell",
                                    ["plasmacyte", "plasma B cell"], parent_id="CL:0000236"),

    # Myeloid cells
    "CL:0000235": CellOntologyTerm("CL:0000235", "macrophage", ["Mφ", "histiocyte"]),
    "CL:0000576": CellOntologyTerm("CL:0000576", "monocyte", []),
    "CL:0000451": CellOntologyTerm("CL:0000451", "dendritic cell", ["DC"]),
    "CL:0000097": CellOntologyTerm("CL:0000097", "mast cell", ["mastocyte"]),
    "CL:0000623": CellOntologyTerm("CL:0000623", "natural killer cell", ["NK cell", "NK"]),

    # Epithelial
    "CL:0000066": CellOntologyTerm("CL:0000066", "epithelial cell", ["epithelium"]),
    "CL:0002327": CellOntologyTerm("CL:0002327", "mammary gland epithelial cell",
                                    ["breast epithelial cell"], parent_id="CL:0000066"),
    "CL:0002325": CellOntologyTerm("CL:0002325", "mammary luminal epithelial cell",
                                    ["luminal cell"], parent_id="CL:0002327"),
    "CL:0000646": CellOntologyTerm("CL:0000646", "basal cell",
                                    ["basal epithelial cell"], parent_id="CL:0000066"),
    "CL:0000185": CellOntologyTerm("CL:0000185", "myoepithelial cell",
                                    ["myoepithelium"], parent_id="CL:0000066"),

    # Stromal
    "CL:0000057": CellOntologyTerm("CL:0000057", "fibroblast", []),
    "CL:0000192": CellOntologyTerm("CL:0000192", "smooth muscle cell", ["SMC"]),
    "CL:0000669": CellOntologyTerm("CL:0000669", "pericyte", ["mural cell"]),
    "CL:0000136": CellOntologyTerm("CL:0000136", "adipocyte", ["fat cell", "adipose cell"]),
    "CL:0000186": CellOntologyTerm("CL:0000186", "myofibroblast", []),

    # Endothelial
    "CL:0000115": CellOntologyTerm("CL:0000115", "endothelial cell", ["endothelium"]),
    "CL:0002138": CellOntologyTerm("CL:0002138", "lymphatic endothelial cell",
                                    ["LEC"], parent_id="CL:0000115"),
}


# =============================================================================
# Mapping: Annotator Output → Cell Ontology ID
# =============================================================================

ANNOTATOR_TO_CL = {
    # === SingleR (BlueprintEncode reference) ===
    "Epithelial cells": "CL:0000066",
    "Keratinocytes": "CL:0000646",
    "CD4+ T-cells": "CL:0000624",
    "CD8+ T-cells": "CL:0000625",
    "T-cells": "CL:0000084",
    "B-cells": "CL:0000236",
    "Macrophages": "CL:0000235",
    "Monocytes": "CL:0000576",
    "DC": "CL:0000451",
    "NK cells": "CL:0000623",
    "Fibroblasts": "CL:0000057",
    "Smooth muscle": "CL:0000192",
    "Endothelial cells": "CL:0000115",
    "Adipocytes": "CL:0000136",
    "Neutrophils": "CL:0000576",  # Group with monocytes
    "Eosinophils": "CL:0000576",

    # === CellTypist outputs ===
    "Luminal epithelial cell of mammary gland": "CL:0002325",
    "Basal cell": "CL:0000646",
    "T cell": "CL:0000084",
    "CD4-positive, alpha-beta T cell": "CL:0000624",
    "CD8-positive, alpha-beta T cell": "CL:0000625",
    "B cell": "CL:0000236",
    "Macrophage": "CL:0000235",
    "Classical monocyte": "CL:0000576",
    "Non-classical monocyte": "CL:0000576",
    "Dendritic cell": "CL:0000451",
    "Mast cell": "CL:0000097",
    "Fibroblast": "CL:0000057",
    "Endothelial cell": "CL:0000115",
    "Smooth muscle cell": "CL:0000192",
    "Pericyte": "CL:0000669",
    "Adipocyte": "CL:0000136",
    "NK cell": "CL:0000623",
    "Plasma cell": "CL:0000786",

    # === scType outputs ===
    "Epithelial": "CL:0000066",
    "T cells": "CL:0000084",
    "CD4+ T cells": "CL:0000624",
    "CD8+ T cells": "CL:0000625",
    "Regulatory T cells": "CL:0000815",
    "B cells": "CL:0000236",
    "Plasma cells": "CL:0000786",
    "Macrophages": "CL:0000235",
    "Monocytes": "CL:0000576",
    "Dendritic cells": "CL:0000451",
    "Mast cells": "CL:0000097",
    "NK cells": "CL:0000623",
    "Fibroblasts": "CL:0000057",
    "Myofibroblasts": "CL:0000186",
    "Pericytes": "CL:0000669",
    "Adipocytes": "CL:0000136",
    "Endothelial cells": "CL:0000115",
    "Lymphatic endothelial": "CL:0002138",

    # === SCINA outputs (underscores) ===
    "T_cells": "CL:0000084",
    "CD4_T_cells": "CL:0000624",
    "CD8_T_cells": "CL:0000625",
    "Tregs": "CL:0000815",
    "B_cells": "CL:0000236",
    "Plasma_cells": "CL:0000786",
    "Dendritic_cells": "CL:0000451",
    "Mast_cells": "CL:0000097",
    "NK_cells": "CL:0000623",
    "Endothelial": "CL:0000115",
    "Lymphatic_endothelial": "CL:0002138",
}


# =============================================================================
# Mapping: Cell Ontology ID → Ground Truth Label (Breast Tumor Dataset)
# =============================================================================

# This mapping is dataset-specific (Janesick et al. breast tumor ground truth)
CL_TO_GROUND_TRUTH = {
    # Epithelial → Tumor labels (breast cancer context)
    "CL:0000066": "Invasive_Tumor",      # Generic epithelial → tumor
    "CL:0002325": "Invasive_Tumor",      # Luminal epithelial → tumor
    "CL:0002327": "Invasive_Tumor",      # Mammary epithelial → tumor
    "CL:0000646": "Myoepi_KRT15+",       # Basal cell → KRT15+ myoepithelial
    "CL:0000185": "Myoepi_ACTA2+",       # Myoepithelial → ACTA2+ myoepithelial

    # T cells
    "CL:0000084": "CD4+_T_Cells",        # Generic T cell → CD4 (most common)
    "CL:0000624": "CD4+_T_Cells",        # CD4+ T cell
    "CL:0000625": "CD8+_T_Cells",        # CD8+ T cell
    "CL:0000815": "CD4+_T_Cells",        # Treg → CD4

    # B cells
    "CL:0000236": "B_Cells",
    "CL:0000786": "B_Cells",             # Plasma cell → B cells

    # Myeloid cells
    "CL:0000235": "Macrophages_1",       # Macrophage
    "CL:0000576": "Macrophages_1",       # Monocyte → macrophage
    "CL:0000451": "IRF7+_DCs",           # Dendritic cell
    "CL:0000097": "Mast_Cells",
    "CL:0000623": "CD8+_T_Cells",        # NK cell → CD8 (functional similarity)

    # Stromal
    "CL:0000057": "Stromal",             # Fibroblast
    "CL:0000192": "Stromal",             # Smooth muscle
    "CL:0000136": "Stromal",             # Adipocyte
    "CL:0000186": "Myoepi_ACTA2+",       # Myofibroblast → ACTA2+
    "CL:0000669": "Perivascular-Like",   # Pericyte

    # Endothelial
    "CL:0000115": "Endothelial",
    "CL:0002138": "Endothelial",         # Lymphatic endothelial
}


def annotator_to_ground_truth(label: str, annotator: str | None = None) -> str:
    """Map an annotator output label to ground truth label via Cell Ontology.

    Args:
        label: The cell type label from an annotator
        annotator: Optional annotator name for method-specific mappings

    Returns:
        The ground truth label, or "Unlabeled" if no mapping exists
    """
    # Direct ground truth match
    gt_labels = {
        "Stromal", "Invasive_Tumor", "DCIS_1", "DCIS_2", "Macrophages_1",
        "Endothelial", "CD4+_T_Cells", "Myoepi_ACTA2+", "CD8+_T_Cells", "B_Cells",
        "Prolif_Invasive_Tumor", "Myoepi_KRT15+", "Macrophages_2", "Perivascular-Like",
        "Stromal_&_T_Cell_Hybrid", "T_Cell_&_Tumor_Hybrid", "IRF7+_DCs", "LAMP3+_DCs",
        "Mast_Cells", "Unlabeled"
    }
    if label in gt_labels:
        return label

    # Try to map through Cell Ontology
    cl_id = ANNOTATOR_TO_CL.get(label)
    if cl_id and cl_id in CL_TO_GROUND_TRUTH:
        return CL_TO_GROUND_TRUTH[cl_id]

    # Handle Unknown/Unlabeled explicitly
    if label.lower() in ["unknown", "unlabeled", "unassigned", "na", "nan"]:
        return "Unlabeled"

    # Fuzzy matching fallback
    label_lower = label.lower()

    # Epithelial subtypes
    if any(x in label_lower for x in ["epithelial", "luminal", "tumor"]):
        return "Invasive_Tumor"
    if any(x in label_lower for x in ["myoepithelial", "basal"]):
        return "Myoepi_KRT15+"

    # T cells
    if "cd4" in label_lower:
        return "CD4+_T_Cells"
    if "cd8" in label_lower:
        return "CD8+_T_Cells"
    if "t cell" in label_lower or "t-cell" in label_lower:
        return "CD4+_T_Cells"  # Default T cells to CD4

    # B cells
    if "b cell" in label_lower or "b-cell" in label_lower or "plasma" in label_lower:
        return "B_Cells"

    # Myeloid
    if "macrophage" in label_lower or "monocyte" in label_lower:
        return "Macrophages_1"
    if "dendritic" in label_lower or label_lower == "dc":
        return "IRF7+_DCs"
    if "mast" in label_lower:
        return "Mast_Cells"
    if "nk" in label_lower or "natural killer" in label_lower:
        return "CD8+_T_Cells"  # Group with CD8

    # Stromal
    if any(x in label_lower for x in ["fibroblast", "stromal", "smooth muscle", "adipocyte"]):
        return "Stromal"
    if "pericyte" in label_lower or "perivascular" in label_lower:
        return "Perivascular-Like"
    if "myofibroblast" in label_lower:
        return "Myoepi_ACTA2+"

    # Endothelial
    if "endothelial" in label_lower:
        return "Endothelial"

    return "Unlabeled"


def get_cl_term(label: str) -> CellOntologyTerm | None:
    """Get the Cell Ontology term for a label."""
    cl_id = ANNOTATOR_TO_CL.get(label)
    if cl_id:
        return CELL_ONTOLOGY.get(cl_id)
    return None


def standardize_label(label: str) -> str:
    """Standardize a label to its Cell Ontology canonical name.

    Args:
        label: Any cell type label

    Returns:
        The canonical CL name, or the original label if no mapping exists
    """
    cl_id = ANNOTATOR_TO_CL.get(label)
    if cl_id and cl_id in CELL_ONTOLOGY:
        return CELL_ONTOLOGY[cl_id].name
    return label


# =============================================================================
# Summary Statistics
# =============================================================================

def print_mapping_coverage():
    """Print coverage statistics for the mappings."""
    print("=" * 60)
    print("Cell Ontology Mapping Coverage")
    print("=" * 60)

    print(f"\nCell Ontology terms defined: {len(CELL_ONTOLOGY)}")
    print(f"Annotator → CL mappings: {len(ANNOTATOR_TO_CL)}")
    print(f"CL → Ground Truth mappings: {len(CL_TO_GROUND_TRUTH)}")

    # Check coverage
    cl_ids_used = set(ANNOTATOR_TO_CL.values())
    cl_ids_mapped = set(CL_TO_GROUND_TRUTH.keys())

    unmapped = cl_ids_used - cl_ids_mapped
    if unmapped:
        print(f"\n⚠️ CL IDs without ground truth mapping: {unmapped}")
    else:
        print("\n✓ All CL IDs have ground truth mappings")


if __name__ == "__main__":
    print_mapping_coverage()

    # Test some mappings
    test_labels = [
        ("CD4+ T-cells", "singler"),
        ("CD4-positive, alpha-beta T cell", "celltypist"),
        ("CD4+ T cells", "sctype"),
        ("CD4_T_cells", "scina"),
        ("Epithelial cells", "singler"),
        ("Fibroblasts", "sctype"),
        ("Unknown", None),
    ]

    print("\n" + "=" * 60)
    print("Test Mappings")
    print("=" * 60)
    for label, annotator in test_labels:
        gt = annotator_to_ground_truth(label, annotator)
        cl_term = get_cl_term(label)
        cl_name = cl_term.name if cl_term else "N/A"
        print(f"  {label:40} → {gt:20} (CL: {cl_name})")
