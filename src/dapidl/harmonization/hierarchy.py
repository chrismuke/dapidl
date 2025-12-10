"""Cell type hierarchy definitions.

Defines multi-level hierarchies for cell type classification.
Each hierarchy has three levels:
- Broad: Top-level categories (Epithelial, Immune, Stromal)
- Mid: Intermediate groupings (T_Cell, Myeloid, Fibroblast)
- Fine: Original fine-grained labels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

HierarchyLevel = Literal["broad", "mid", "fine"]


@dataclass
class CellTypeNode:
    """A node in the cell type hierarchy."""

    name: str
    level: HierarchyLevel
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    is_cancer: bool = False


@dataclass
class CellTypeHierarchy:
    """A hierarchical organization of cell types.

    Supports mapping between three levels:
    - broad: Epithelial, Immune, Stromal
    - mid: T_Cell, B_Cell, Myeloid, NK, Tumor, etc.
    - fine: Original labels like CD4-Tem, Invasive_Tumor
    """

    name: str
    description: str
    nodes: dict[str, CellTypeNode] = field(default_factory=dict)
    _alias_cache: dict[str, str] = field(default_factory=dict, repr=False)

    def add_node(
        self,
        name: str,
        level: HierarchyLevel,
        parent: str | None = None,
        aliases: list[str] | None = None,
        is_cancer: bool = False,
    ) -> None:
        """Add a node to the hierarchy."""
        node = CellTypeNode(
            name=name,
            level=level,
            parent=parent,
            aliases=aliases or [],
            is_cancer=is_cancer,
        )
        self.nodes[name] = node

        # Update alias cache for O(1) lookups
        self._alias_cache[name.lower()] = name
        for alias in node.aliases:
            self._alias_cache[alias.lower()] = name

        # Update parent's children
        if parent and parent in self.nodes:
            self.nodes[parent].children.append(name)

    def get_parent(self, name: str, target_level: HierarchyLevel) -> str | None:
        """Get ancestor at target level."""
        if name not in self.nodes:
            return None

        node = self.nodes[name]

        # Already at target level
        if node.level == target_level:
            return name

        # Walk up the tree
        current = name
        while current:
            node = self.nodes.get(current)
            if not node:
                return None
            if node.level == target_level:
                return current
            current = node.parent

        return None

    def get_broad(self, name: str) -> str | None:
        """Get broad category for a cell type."""
        return self.get_parent(name, "broad")

    def get_mid(self, name: str) -> str | None:
        """Get mid-level category for a cell type."""
        return self.get_parent(name, "mid")

    def get_all_at_level(self, level: HierarchyLevel) -> list[str]:
        """Get all node names at a given level."""
        return [name for name, node in self.nodes.items() if node.level == level]

    def find_by_alias(self, alias: str) -> str | None:
        """Find node name by alias (case-insensitive).

        Uses cached lookups for O(1) performance.
        """
        return self._alias_cache.get(alias.lower())


def _build_breast_hierarchy() -> CellTypeHierarchy:
    """Build the breast tissue cell type hierarchy."""
    h = CellTypeHierarchy(
        name="breast",
        description="Breast tissue cell type hierarchy (healthy + cancer)",
    )

    # =====================================================================
    # BROAD LEVEL (Level 0)
    # =====================================================================
    h.add_node("Epithelial", level="broad")
    h.add_node("Immune", level="broad")
    h.add_node("Stromal", level="broad")

    # =====================================================================
    # MID LEVEL (Level 1)
    # =====================================================================
    # Epithelial subtypes
    h.add_node("Tumor", level="mid", parent="Epithelial", is_cancer=True)
    h.add_node("Luminal", level="mid", parent="Epithelial")
    h.add_node("Basal_Myoepithelial", level="mid", parent="Epithelial")

    # Immune subtypes
    h.add_node("T_Cell", level="mid", parent="Immune")
    h.add_node("B_Cell", level="mid", parent="Immune")
    h.add_node("Myeloid", level="mid", parent="Immune")
    h.add_node("NK_Cell", level="mid", parent="Immune")
    h.add_node("Dendritic", level="mid", parent="Immune")
    h.add_node("Mast_Cell", level="mid", parent="Immune")

    # Stromal subtypes
    h.add_node("Fibroblast", level="mid", parent="Stromal")
    h.add_node("Endothelial", level="mid", parent="Stromal")
    h.add_node("Pericyte", level="mid", parent="Stromal")
    h.add_node("Smooth_Muscle", level="mid", parent="Stromal")
    h.add_node("Lymphatic", level="mid", parent="Stromal")

    # =====================================================================
    # FINE LEVEL - Ground Truth (Xenium Breast Cancer)
    # =====================================================================
    # Tumor types (from ground truth)
    h.add_node(
        "Invasive_Tumor",
        level="fine",
        parent="Tumor",
        aliases=["Invasive Tumor", "invasive_tumor"],
        is_cancer=True,
    )
    h.add_node(
        "Prolif_Invasive_Tumor",
        level="fine",
        parent="Tumor",
        aliases=["Proliferating Invasive Tumor"],
        is_cancer=True,
    )
    h.add_node(
        "DCIS_1",
        level="fine",
        parent="Tumor",
        aliases=["DCIS 1", "DCIS1"],
        is_cancer=True,
    )
    h.add_node(
        "DCIS_2",
        level="fine",
        parent="Tumor",
        aliases=["DCIS 2", "DCIS2"],
        is_cancer=True,
    )

    # Myoepithelial types (from ground truth)
    h.add_node(
        "Myoepi_ACTA2+",
        level="fine",
        parent="Basal_Myoepithelial",
        aliases=["Myoepi ACTA2+", "Myoepithelial ACTA2"],
    )
    h.add_node(
        "Myoepi_KRT15+",
        level="fine",
        parent="Basal_Myoepithelial",
        aliases=["Myoepi KRT15+", "Myoepithelial KRT15"],
    )

    # T cell types (from ground truth)
    h.add_node(
        "CD4+_T_Cells",
        level="fine",
        parent="T_Cell",
        aliases=["CD4+ T Cells", "CD4_T_Cells", "CD4 T cells"],
    )
    h.add_node(
        "CD8+_T_Cells",
        level="fine",
        parent="T_Cell",
        aliases=["CD8+ T Cells", "CD8_T_Cells", "CD8 T cells"],
    )

    # B cell types
    h.add_node(
        "B_Cells",
        level="fine",
        parent="B_Cell",
        aliases=["B Cells", "B cells", "B_cells"],
    )

    # Myeloid types (from ground truth)
    h.add_node(
        "Macrophages_1",
        level="fine",
        parent="Myeloid",
        aliases=["Macrophages 1", "Macrophage_1"],
    )
    h.add_node(
        "Macrophages_2",
        level="fine",
        parent="Myeloid",
        aliases=["Macrophages 2", "Macrophage_2"],
    )

    # Dendritic types (from ground truth)
    h.add_node(
        "IRF7+_DCs",
        level="fine",
        parent="Dendritic",
        aliases=["IRF7+ DCs", "IRF7_DCs", "pDC"],
    )
    h.add_node(
        "LAMP3+_DCs",
        level="fine",
        parent="Dendritic",
        aliases=["LAMP3+ DCs", "LAMP3_DCs", "mature DC"],
    )

    # Mast cells
    h.add_node(
        "Mast_Cells",
        level="fine",
        parent="Mast_Cell",
        aliases=["Mast Cells", "Mast cells", "Mast"],
    )

    # Stromal types (from ground truth)
    # Note: Named "Stromal_Cells" to avoid collision with broad-level "Stromal" category
    h.add_node(
        "Stromal_Cells",
        level="fine",
        parent="Fibroblast",
        aliases=["Stromal", "Stroma", "Fibroblasts", "CAF"],
    )
    # Note: Named "Endothelial_Cells" to avoid collision with mid-level "Endothelial" category
    h.add_node(
        "Endothelial_Cells",
        level="fine",
        parent="Endothelial",
        aliases=["Endothelial", "Endothelial cells", "Vascular endothelial"],
    )
    h.add_node(
        "Perivascular-Like",
        level="fine",
        parent="Pericyte",
        aliases=["Perivascular-Like", "Perivascular", "Pericytes"],
    )

    # =====================================================================
    # FINE LEVEL - CellTypist Breast Model (58 types)
    # =====================================================================
    # T cells - CD4
    for ct in ["CD4-Tem", "CD4-Th", "CD4-Th-like", "CD4-Treg", "CD4-activated", "CD4-naive"]:
        h.add_node(ct, level="fine", parent="T_Cell")
    # T cells - CD8
    for ct in ["CD8-Tem", "CD8-Trm", "CD8-activated"]:
        h.add_node(ct, level="fine", parent="T_Cell")
    h.add_node("T_prol", level="fine", parent="T_Cell", aliases=["Proliferating T cell"])
    h.add_node("GD", level="fine", parent="T_Cell", aliases=["Gamma-delta T cell"])
    h.add_node("NKT", level="fine", parent="T_Cell", aliases=["NKT cell"])

    # NK cells
    h.add_node("NK", level="fine", parent="NK_Cell", aliases=["NK cell"])
    h.add_node("NK-ILCs", level="fine", parent="NK_Cell", aliases=["ILCs"])

    # B cells
    h.add_node("b_naive", level="fine", parent="B_Cell", aliases=["Naive B cell"])
    h.add_node("bmem_switched", level="fine", parent="B_Cell", aliases=["Memory B cell switched"])
    h.add_node("bmem_unswitched", level="fine", parent="B_Cell", aliases=["Memory B cell unswitched"])
    h.add_node("plasma_IgA", level="fine", parent="B_Cell", aliases=["Plasma cell IgA"])
    h.add_node("plasma_IgG", level="fine", parent="B_Cell", aliases=["Plasma cell IgG"])

    # Myeloid - Macrophages
    for ct in ["Macro-IFN", "Macro-lipo", "Macro-m1", "Macro-m1-CCL", "Macro-m2", "Macro-m2-CXCL"]:
        h.add_node(ct, level="fine", parent="Myeloid")

    # Myeloid - Monocytes
    h.add_node("Mono-classical", level="fine", parent="Myeloid")
    h.add_node("Mono-non-classical", level="fine", parent="Myeloid")
    h.add_node("Neutrophil", level="fine", parent="Myeloid")
    h.add_node("mye-prol", level="fine", parent="Myeloid", aliases=["Proliferating myeloid"])

    # Dendritic cells
    for ct in ["cDC1", "cDC2", "mDC", "pDC"]:
        h.add_node(ct, level="fine", parent="Dendritic")

    # Mast cells
    h.add_node("Mast", level="fine", parent="Mast_Cell")

    # Luminal epithelial (CellTypist breast model)
    for ct in ["LummHR-SCGB", "LummHR-active", "LummHR-major"]:
        h.add_node(ct, level="fine", parent="Luminal", aliases=[ct.replace("-", " ")])
    for ct in [
        "Lumsec-HLA",
        "Lumsec-KIT",
        "Lumsec-basal",
        "Lumsec-lac",
        "Lumsec-major",
        "Lumsec-myo",
        "Lumsec-prol",
    ]:
        h.add_node(ct, level="fine", parent="Luminal")

    # Basal/Myoepithelial
    h.add_node("basal", level="fine", parent="Basal_Myoepithelial", aliases=["Basal cell"])

    # Fibroblasts
    for ct in ["Fibro-SFRP4", "Fibro-major", "Fibro-matrix", "Fibro-prematrix"]:
        h.add_node(ct, level="fine", parent="Fibroblast")

    # Vascular endothelial
    for ct in ["Vas-arterial", "Vas-capillary", "Vas-venous"]:
        h.add_node(ct, level="fine", parent="Endothelial")

    # Pericytes and smooth muscle
    h.add_node("pericytes", level="fine", parent="Pericyte")
    h.add_node("vsmc", level="fine", parent="Smooth_Muscle", aliases=["Vascular smooth muscle"])

    # Lymphatic
    for ct in ["Lymph-immune", "Lymph-major", "Lymph-valve1", "Lymph-valve2"]:
        h.add_node(ct, level="fine", parent="Lymphatic")

    return h


# Pre-built hierarchy for breast tissue
BREAST_HIERARCHY = _build_breast_hierarchy()
