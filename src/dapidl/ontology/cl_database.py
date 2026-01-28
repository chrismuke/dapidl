"""Cell Ontology Database - Core CL terms for human tissue classification.

This module defines the ~75 CL terms relevant for DAPIDL's universal cell type
classification across human tissues. These represent the "backbone vocabulary"
that all annotator outputs and ground truth labels map to.

References:
    - Cell Ontology: https://obofoundry.org/ontology/cl.html
    - CL Browser: https://www.ebi.ac.uk/ols/ontologies/cl

Hierarchy Structure:
    Level 0 (Root): cell (CL:0000000)
    Level 1 (Super-coarse): ~5 broad lineages
    Level 2 (Coarse): ~15 major categories - DAPIDL training target
    Level 3 (Medium): ~30 common types
    Level 4 (Fine): ~75 specific subtypes - DAPIDL inference limit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class HierarchyLevel(Enum):
    """Hierarchy levels for classification."""

    ROOT = 0
    SUPER_COARSE = 1  # Epithelial, Immune, Stromal, Endothelial, Neural
    COARSE = 2  # ~15 major categories (training target)
    MEDIUM = 3  # ~30 common types
    FINE = 4  # ~75 specific subtypes


@dataclass
class CLTerm:
    """A Cell Ontology term with full metadata."""

    cl_id: str  # e.g., "CL:0000624"
    name: str  # Canonical name, e.g., "CD4-positive, alpha-beta T cell"
    synonyms: list[str] = field(default_factory=list)
    parent_id: Optional[str] = None  # Direct parent CL ID
    level: HierarchyLevel = HierarchyLevel.FINE  # Classification level
    definition: str = ""  # OBO definition
    markers: list[str] = field(default_factory=list)  # Key marker genes

    def __hash__(self):
        return hash(self.cl_id)

    def __eq__(self, other):
        if isinstance(other, CLTerm):
            return self.cl_id == other.cl_id
        return False


# =============================================================================
# Level 1: Super-Coarse Categories (5 classes)
# These are the broadest meaningful divisions visible in DAPI
# =============================================================================

SUPER_COARSE_TERMS = {
    "CL:0000066": CLTerm(
        "CL:0000066",
        "epithelial cell",
        ["epithelium", "epithelial"],
        parent_id="CL:0000000",
        level=HierarchyLevel.SUPER_COARSE,
        definition="A cell that is part of the epithelium.",
        markers=["EPCAM", "CDH1", "KRT8", "KRT18"],
    ),
    "CL:0000738": CLTerm(
        "CL:0000738",
        "leukocyte",
        ["white blood cell", "WBC", "immune cell"],
        parent_id="CL:0000000",
        level=HierarchyLevel.SUPER_COARSE,
        definition="An achromatic cell of the myeloid or lymphoid lineages.",
        markers=["PTPRC", "CD45"],
    ),
    "CL:0000499": CLTerm(
        "CL:0000499",
        "stromal cell",
        ["stroma", "stromal"],
        parent_id="CL:0000000",
        level=HierarchyLevel.SUPER_COARSE,
        definition="A connective tissue cell that forms the supportive framework.",
        markers=["VIM", "COL1A1", "COL1A2"],
    ),
    "CL:0000115": CLTerm(
        "CL:0000115",
        "endothelial cell",
        ["endothelium", "vascular endothelial"],
        parent_id="CL:0000000",
        level=HierarchyLevel.SUPER_COARSE,
        definition="A cell that lines blood vessels.",
        markers=["PECAM1", "CD31", "VWF", "CDH5"],
    ),
    "CL:0000540": CLTerm(
        "CL:0000540",
        "neuron",
        ["nerve cell", "neural cell"],
        parent_id="CL:0000000",
        level=HierarchyLevel.SUPER_COARSE,
        definition="An electrically excitable cell that processes information.",
        markers=["RBFOX3", "MAP2", "TUBB3"],
    ),
}


# =============================================================================
# Level 2: Coarse Categories (~15 classes) - Primary training target
# =============================================================================

COARSE_TERMS = {
    # === Immune - Lymphoid ===
    "CL:0000542": CLTerm(
        "CL:0000542",
        "lymphocyte",
        ["lymphoid cell"],
        parent_id="CL:0000738",
        level=HierarchyLevel.COARSE,
        markers=["CD3D", "CD3E", "CD19", "CD79A"],
    ),
    # === Immune - Myeloid ===
    "CL:0000766": CLTerm(
        "CL:0000766",
        "myeloid cell",
        ["myeloid leukocyte"],
        parent_id="CL:0000738",
        level=HierarchyLevel.COARSE,
        markers=["CD14", "CD68", "ITGAM"],
    ),
    # === Epithelial subtypes ===
    "CL:0002327": CLTerm(
        "CL:0002327",
        "mammary gland epithelial cell",
        ["breast epithelial cell", "mammary epithelial"],
        parent_id="CL:0000066",
        level=HierarchyLevel.COARSE,
        markers=["EPCAM", "KRT8", "KRT18", "KRT19"],
    ),
    "CL:0000182": CLTerm(
        "CL:0000182",
        "hepatocyte",
        ["liver cell", "hepatic cell"],
        parent_id="CL:0000066",
        level=HierarchyLevel.COARSE,
        markers=["ALB", "HNF4A", "APOB"],
    ),
    "CL:0002062": CLTerm(
        "CL:0002062",
        "type I pneumocyte",
        ["alveolar epithelial type I", "AT1"],
        parent_id="CL:0000066",
        level=HierarchyLevel.COARSE,
        markers=["AGER", "PDPN", "AQP5"],
    ),
    "CL:0002063": CLTerm(
        "CL:0002063",
        "type II pneumocyte",
        ["alveolar epithelial type II", "AT2"],
        parent_id="CL:0000066",
        level=HierarchyLevel.COARSE,
        markers=["SFTPC", "SFTPB", "ABCA3"],
    ),
    "CL:0002071": CLTerm(
        "CL:0002071",
        "enterocyte",
        ["intestinal epithelial cell", "gut epithelial"],
        parent_id="CL:0000066",
        level=HierarchyLevel.COARSE,
        markers=["VIL1", "FABP2", "SI"],
    ),
    "CL:0000312": CLTerm(
        "CL:0000312",
        "keratinocyte",
        ["skin epithelial cell"],
        parent_id="CL:0000066",
        level=HierarchyLevel.COARSE,
        markers=["KRT14", "KRT5", "TP63"],
    ),
    # === Stromal subtypes ===
    "CL:0000057": CLTerm(
        "CL:0000057",
        "fibroblast",
        ["fibrocyte"],
        parent_id="CL:0000499",
        level=HierarchyLevel.COARSE,
        markers=["VIM", "COL1A1", "DCN", "LUM"],
    ),
    "CL:0000192": CLTerm(
        "CL:0000192",
        "smooth muscle cell",
        ["SMC", "vascular smooth muscle"],
        parent_id="CL:0000499",
        level=HierarchyLevel.COARSE,
        markers=["ACTA2", "MYH11", "TAGLN"],
    ),
    "CL:0000669": CLTerm(
        "CL:0000669",
        "pericyte",
        ["mural cell", "perivascular cell"],
        parent_id="CL:0000499",
        level=HierarchyLevel.COARSE,
        markers=["PDGFRB", "RGS5", "NOTCH3"],
    ),
    "CL:0000136": CLTerm(
        "CL:0000136",
        "adipocyte",
        ["fat cell", "adipose cell"],
        parent_id="CL:0000499",
        level=HierarchyLevel.COARSE,
        markers=["ADIPOQ", "LEP", "PPARG"],
    ),
    # === Endothelial subtypes ===
    "CL:0002138": CLTerm(
        "CL:0002138",
        "lymphatic endothelial cell",
        ["LEC", "lymphatic endothelium"],
        parent_id="CL:0000115",
        level=HierarchyLevel.COARSE,
        markers=["LYVE1", "PROX1", "PDPN"],
    ),
    # === Neural subtypes ===
    "CL:0000679": CLTerm(
        "CL:0000679",
        "glutamatergic neuron",
        ["excitatory neuron", "Glut neuron"],
        parent_id="CL:0000540",
        level=HierarchyLevel.COARSE,
        markers=["SLC17A7", "SLC17A6", "GRIN1"],
    ),
    "CL:0000617": CLTerm(
        "CL:0000617",
        "GABAergic neuron",
        ["inhibitory neuron", "GABA neuron", "interneuron"],
        parent_id="CL:0000540",
        level=HierarchyLevel.COARSE,
        markers=["GAD1", "GAD2", "SLC32A1"],
    ),
    "CL:0000128": CLTerm(
        "CL:0000128",
        "oligodendrocyte",
        ["Oligo", "myelinating glia"],
        parent_id="CL:0000540",  # Technically parent is glial cell
        level=HierarchyLevel.COARSE,
        markers=["MBP", "MOG", "OLIG2"],
    ),
    "CL:0000127": CLTerm(
        "CL:0000127",
        "astrocyte",
        ["Astro", "astroglial cell"],
        parent_id="CL:0000540",  # Technically parent is glial cell
        level=HierarchyLevel.COARSE,
        markers=["GFAP", "AQP4", "SLC1A3"],
    ),
    "CL:0000129": CLTerm(
        "CL:0000129",
        "microglial cell",
        ["microglia", "brain macrophage"],
        parent_id="CL:0000540",  # Technically parent is glial cell
        level=HierarchyLevel.COARSE,
        markers=["CX3CR1", "P2RY12", "TMEM119"],
    ),
    "CL:0002608": CLTerm(
        "CL:0002608",
        "hippocampal pyramidal neuron",
        ["CA1 neuron", "CA3 neuron", "hippocampal neuron"],
        parent_id="CL:0000679",  # Child of glutamatergic
        level=HierarchyLevel.COARSE,
        markers=["PROX1", "NEUROD1"],
    ),
    "CL:0002453": CLTerm(
        "CL:0002453",
        "oligodendrocyte precursor cell",
        ["OPC", "NG2 glia"],
        parent_id="CL:0000128",  # Child of oligodendrocyte
        level=HierarchyLevel.COARSE,
        markers=["PDGFRA", "CSPG4"],
    ),
}


# =============================================================================
# Level 3: Medium Categories (~30 classes)
# =============================================================================

MEDIUM_TERMS = {
    # === T cells ===
    "CL:0000084": CLTerm(
        "CL:0000084",
        "T cell",
        ["T-cell", "T lymphocyte"],
        parent_id="CL:0000542",
        level=HierarchyLevel.MEDIUM,
        markers=["CD3D", "CD3E", "CD3G"],
    ),
    # === B cells ===
    "CL:0000236": CLTerm(
        "CL:0000236",
        "B cell",
        ["B-cell", "B lymphocyte"],
        parent_id="CL:0000542",
        level=HierarchyLevel.MEDIUM,
        markers=["CD19", "CD79A", "MS4A1"],
    ),
    # === NK cells ===
    "CL:0000623": CLTerm(
        "CL:0000623",
        "natural killer cell",
        ["NK cell", "NK", "large granular lymphocyte"],
        parent_id="CL:0000542",
        level=HierarchyLevel.MEDIUM,
        markers=["NCAM1", "NKG7", "GNLY", "KLRD1"],
    ),
    # === Myeloid subtypes ===
    "CL:0000235": CLTerm(
        "CL:0000235",
        "macrophage",
        ["histiocyte", "Mφ"],
        parent_id="CL:0000766",
        level=HierarchyLevel.MEDIUM,
        markers=["CD68", "CD163", "MRC1"],
    ),
    "CL:0000576": CLTerm(
        "CL:0000576",
        "monocyte",
        [],
        parent_id="CL:0000766",
        level=HierarchyLevel.MEDIUM,
        markers=["CD14", "FCGR3A", "CSF1R"],
    ),
    "CL:0000451": CLTerm(
        "CL:0000451",
        "dendritic cell",
        ["DC", "antigen-presenting cell"],
        parent_id="CL:0000766",
        level=HierarchyLevel.MEDIUM,
        markers=["ITGAX", "CD1C", "CLEC9A", "HLA-DRA"],
    ),
    "CL:0000097": CLTerm(
        "CL:0000097",
        "mast cell",
        ["mastocyte"],
        parent_id="CL:0000766",
        level=HierarchyLevel.MEDIUM,
        markers=["TPSAB1", "KIT", "CPA3"],
    ),
    "CL:0000775": CLTerm(
        "CL:0000775",
        "neutrophil",
        ["PMN", "polymorphonuclear leukocyte"],
        parent_id="CL:0000766",
        level=HierarchyLevel.MEDIUM,
        markers=["FCGR3B", "CSF3R", "S100A8"],
    ),
    # === Epithelial specialized ===
    "CL:0002325": CLTerm(
        "CL:0002325",
        "mammary luminal epithelial cell",
        ["luminal cell", "luminal epithelial"],
        parent_id="CL:0002327",
        level=HierarchyLevel.MEDIUM,
        markers=["KRT8", "KRT18", "GATA3"],
    ),
    "CL:0000646": CLTerm(
        "CL:0000646",
        "basal cell",
        ["basal epithelial cell"],
        parent_id="CL:0000066",
        level=HierarchyLevel.MEDIUM,
        markers=["KRT5", "KRT14", "TP63"],
    ),
    "CL:0000185": CLTerm(
        "CL:0000185",
        "myoepithelial cell",
        ["myoepithelium"],
        parent_id="CL:0000066",
        level=HierarchyLevel.MEDIUM,
        markers=["ACTA2", "KRT14", "MYLK"],
    ),
    # === Stromal specialized ===
    "CL:0000186": CLTerm(
        "CL:0000186",
        "myofibroblast",
        ["activated fibroblast"],
        parent_id="CL:0000057",
        level=HierarchyLevel.MEDIUM,
        markers=["ACTA2", "FAP", "POSTN"],
    ),
}


# =============================================================================
# Level 4: Fine Categories (~75 classes) - Maximum DAPI resolution
# =============================================================================

FINE_TERMS = {
    # === T cell subtypes ===
    "CL:0000624": CLTerm(
        "CL:0000624",
        "CD4-positive, alpha-beta T cell",
        ["CD4+ T cell", "helper T cell", "CD4 T cell", "Th cell"],
        parent_id="CL:0000084",
        level=HierarchyLevel.FINE,
        markers=["CD4", "CD3D", "IL7R"],
    ),
    "CL:0000625": CLTerm(
        "CL:0000625",
        "CD8-positive, alpha-beta T cell",
        ["CD8+ T cell", "cytotoxic T cell", "CD8 T cell", "CTL"],
        parent_id="CL:0000084",
        level=HierarchyLevel.FINE,
        markers=["CD8A", "CD8B", "CD3D"],
    ),
    "CL:0000815": CLTerm(
        "CL:0000815",
        "regulatory T cell",
        ["Treg", "Tregs", "suppressor T cell", "regulatory T-cell"],
        parent_id="CL:0000624",
        level=HierarchyLevel.FINE,
        markers=["FOXP3", "IL2RA", "CTLA4"],
    ),
    "CL:0000897": CLTerm(
        "CL:0000897",
        "CD4-positive, alpha-beta memory T cell",
        ["CD4 memory T cell", "memory CD4 T cell"],
        parent_id="CL:0000624",
        level=HierarchyLevel.FINE,
        markers=["CD4", "CD44", "CCR7"],
    ),
    "CL:0000909": CLTerm(
        "CL:0000909",
        "CD8-positive, alpha-beta memory T cell",
        ["CD8 memory T cell", "memory CD8 T cell"],
        parent_id="CL:0000625",
        level=HierarchyLevel.FINE,
        markers=["CD8A", "CD44", "CCR7"],
    ),
    "CL:0002420": CLTerm(
        "CL:0002420",
        "CD4-positive, alpha-beta cytotoxic T cell",
        ["CD4 CTL", "cytotoxic CD4 T cell"],
        parent_id="CL:0000624",
        level=HierarchyLevel.FINE,
        markers=["CD4", "GZMB", "PRF1"],
    ),
    "CL:0000798": CLTerm(
        "CL:0000798",
        "gamma-delta T cell",
        ["γδ T cell", "gamma delta T cell"],
        parent_id="CL:0000084",
        level=HierarchyLevel.FINE,
        markers=["TRDC", "TRGC1"],
    ),
    # === B cell subtypes ===
    "CL:0000786": CLTerm(
        "CL:0000786",
        "plasma cell",
        ["plasmacyte", "plasma B cell", "antibody-secreting cell"],
        parent_id="CL:0000236",
        level=HierarchyLevel.FINE,
        markers=["SDC1", "IGHG1", "XBP1"],
    ),
    "CL:0000787": CLTerm(
        "CL:0000787",
        "memory B cell",
        ["memory B lymphocyte"],
        parent_id="CL:0000236",
        level=HierarchyLevel.FINE,
        markers=["CD27", "CD38"],
    ),
    "CL:0000816": CLTerm(
        "CL:0000816",
        "immature B cell",
        ["pre-B cell", "transitional B cell"],
        parent_id="CL:0000236",
        level=HierarchyLevel.FINE,
        markers=["CD10", "CD24"],
    ),
    "CL:0001201": CLTerm(
        "CL:0001201",
        "naive B cell",
        [],
        parent_id="CL:0000236",
        level=HierarchyLevel.FINE,
        markers=["IGHD", "CD38"],
    ),
    # === Monocyte subtypes ===
    "CL:0000860": CLTerm(
        "CL:0000860",
        "classical monocyte",
        ["CD14+ monocyte", "inflammatory monocyte"],
        parent_id="CL:0000576",
        level=HierarchyLevel.FINE,
        markers=["CD14", "LYZ", "S100A9"],
    ),
    "CL:0000875": CLTerm(
        "CL:0000875",
        "non-classical monocyte",
        ["CD16+ monocyte", "patrolling monocyte"],
        parent_id="CL:0000576",
        level=HierarchyLevel.FINE,
        markers=["FCGR3A", "CX3CR1"],
    ),
    # === Macrophage subtypes ===
    "CL:0000863": CLTerm(
        "CL:0000863",
        "inflammatory macrophage",
        ["M1 macrophage", "classically activated macrophage"],
        parent_id="CL:0000235",
        level=HierarchyLevel.FINE,
        markers=["NOS2", "IL1B", "TNF"],
    ),
    "CL:0000890": CLTerm(
        "CL:0000890",
        "alternatively activated macrophage",
        ["M2 macrophage", "tissue-resident macrophage"],
        parent_id="CL:0000235",
        level=HierarchyLevel.FINE,
        markers=["MRC1", "CD163", "ARG1"],
    ),
    # === Dendritic cell subtypes ===
    "CL:0000990": CLTerm(
        "CL:0000990",
        "conventional dendritic cell",
        ["cDC", "myeloid DC"],
        parent_id="CL:0000451",
        level=HierarchyLevel.FINE,
        markers=["ITGAX", "FLT3"],
    ),
    "CL:0000784": CLTerm(
        "CL:0000784",
        "plasmacytoid dendritic cell",
        ["pDC"],
        parent_id="CL:0000451",
        level=HierarchyLevel.FINE,
        markers=["CLEC4C", "IL3RA", "IRF7"],
    ),
    "CL:0002399": CLTerm(
        "CL:0002399",
        "CD1c-positive myeloid dendritic cell",
        ["cDC2", "type 2 conventional DC"],
        parent_id="CL:0000990",
        level=HierarchyLevel.FINE,
        markers=["CD1C", "FCER1A"],
    ),
    "CL:0002394": CLTerm(
        "CL:0002394",
        "CLEC9A-positive myeloid dendritic cell",
        ["cDC1", "type 1 conventional DC"],
        parent_id="CL:0000990",
        level=HierarchyLevel.FINE,
        markers=["CLEC9A", "XCR1"],
    ),
    "CL:0001056": CLTerm(
        "CL:0001056",
        "dendritic cell of lymphoid tissue",
        ["lymphoid DC"],
        parent_id="CL:0000451",
        level=HierarchyLevel.FINE,
        markers=["LAMP3", "CCR7"],
    ),
    # === NK cell subtypes ===
    "CL:0000938": CLTerm(
        "CL:0000938",
        "CD56-bright natural killer cell",
        ["CD56bright NK", "immunoregulatory NK"],
        parent_id="CL:0000623",
        level=HierarchyLevel.FINE,
        markers=["NCAM1", "SELL"],
    ),
    "CL:0000939": CLTerm(
        "CL:0000939",
        "CD56-dim natural killer cell",
        ["CD56dim NK", "cytotoxic NK"],
        parent_id="CL:0000623",
        level=HierarchyLevel.FINE,
        markers=["FCGR3A", "PRF1", "GZMB"],
    ),
    # === Fibroblast subtypes ===
    "CL:0002553": CLTerm(
        "CL:0002553",
        "fibroblast of breast",
        ["breast fibroblast", "mammary fibroblast"],
        parent_id="CL:0000057",
        level=HierarchyLevel.FINE,
        markers=["VIM", "COL1A1"],
    ),
    "CL:0002557": CLTerm(
        "CL:0002557",
        "fibroblast of lung",
        ["lung fibroblast", "pulmonary fibroblast"],
        parent_id="CL:0000057",
        level=HierarchyLevel.FINE,
        markers=["VIM", "COL1A1"],
    ),
    # === Specialized epithelial cells (CellTypist) ===
    "CL:0000064": CLTerm(
        "CL:0000064",
        "ciliated cell",
        ["ciliated epithelial cell", "Ciliated"],
        parent_id="CL:0000066",  # epithelial cell
        level=HierarchyLevel.FINE,
        markers=["FOXJ1", "CFAP299"],
    ),
    "CL:0000160": CLTerm(
        "CL:0000160",
        "goblet cell",
        ["mucus-secreting cell", "Goblet"],
        parent_id="CL:0000066",  # epithelial cell
        level=HierarchyLevel.FINE,
        markers=["MUC5AC", "MUC5B"],
    ),
    "CL:0000158": CLTerm(
        "CL:0000158",
        "club cell",
        ["Clara cell", "bronchiolar secretory cell", "Secretory_Club"],
        parent_id="CL:0000066",  # epithelial cell
        level=HierarchyLevel.FINE,
        markers=["SCGB1A1", "SCGB3A1"],
    ),
    "CL:0002204": CLTerm(
        "CL:0002204",
        "tuft cell",
        ["brush cell", "Tuft"],
        parent_id="CL:0000066",  # epithelial cell
        level=HierarchyLevel.FINE,
        markers=["DCLK1", "TRPM5"],
    ),
    # === NKT cells ===
    "CL:0000814": CLTerm(
        "CL:0000814",
        "mature NK T cell",
        ["NKT cell", "NKT", "natural killer T cell"],
        parent_id="CL:0000084",  # T cell (→ lymphocyte → leukocyte → Immune)
        level=HierarchyLevel.FINE,
        markers=["CD3D", "NCAM1", "KLRB1"],
    ),
}


# =============================================================================
# Combined Database
# =============================================================================


def get_all_terms() -> dict[str, CLTerm]:
    """Get all CL terms from all hierarchy levels."""
    all_terms = {}
    all_terms.update(SUPER_COARSE_TERMS)
    all_terms.update(COARSE_TERMS)
    all_terms.update(MEDIUM_TERMS)
    all_terms.update(FINE_TERMS)
    return all_terms


def get_terms_by_level(level: HierarchyLevel) -> dict[str, CLTerm]:
    """Get all CL terms at a specific hierarchy level."""
    level_maps = {
        HierarchyLevel.SUPER_COARSE: SUPER_COARSE_TERMS,
        HierarchyLevel.COARSE: COARSE_TERMS,
        HierarchyLevel.MEDIUM: MEDIUM_TERMS,
        HierarchyLevel.FINE: FINE_TERMS,
    }
    return level_maps.get(level, {})


def get_term(cl_id: str) -> CLTerm | None:
    """Look up a CL term by ID."""
    return get_all_terms().get(cl_id)


def get_term_by_name(name: str) -> CLTerm | None:
    """Look up a CL term by canonical name."""
    for term in get_all_terms().values():
        if term.name.lower() == name.lower():
            return term
    return None


# =============================================================================
# DAPIDL-Specific Mappings
# =============================================================================

# Map super-coarse to 5 DAPIDL broad categories
DAPIDL_BROAD_CATEGORIES = {
    "Epithelial": "CL:0000066",
    "Immune": "CL:0000738",
    "Stromal": "CL:0000499",
    "Endothelial": "CL:0000115",
    "Neural": "CL:0000540",
}

# Reverse mapping
CL_TO_BROAD_CATEGORY = {v: k for k, v in DAPIDL_BROAD_CATEGORIES.items()}


def get_broad_category(cl_id: str) -> str:
    """Get the DAPIDL broad category for any CL ID by traversing ancestry."""
    # Check direct mapping first
    if cl_id in CL_TO_BROAD_CATEGORY:
        return CL_TO_BROAD_CATEGORY[cl_id]

    # Traverse parent chain
    term = get_term(cl_id)
    if term and term.parent_id:
        return get_broad_category(term.parent_id)

    return "Unknown"


# Coarse categories for DAPIDL training (~20 classes)
DAPIDL_COARSE_CATEGORIES = [
    # Immune - Lymphoid
    "T_Cell",
    "B_Cell",
    "NK_Cell",
    # Immune - Myeloid
    "Macrophage",
    "Monocyte",
    "Dendritic_Cell",
    "Mast_Cell",
    "Neutrophil",
    # Epithelial
    "Epithelial_Luminal",
    "Epithelial_Basal",
    "Epithelial_Secretory",
    # Stromal
    "Fibroblast",
    "Smooth_Muscle",
    "Pericyte",
    "Adipocyte",
    # Endothelial
    "Vascular_Endothelial",
    "Lymphatic_Endothelial",
    # Neural
    "Glutamatergic_Neuron",
    "GABAergic_Neuron",
    "Oligodendrocyte",
    "Astrocyte",
    "Microglia",
]

# Mapping from CL IDs to DAPIDL coarse categories
CL_TO_COARSE_CATEGORY = {
    # T cells
    "CL:0000084": "T_Cell",
    "CL:0000624": "T_Cell",
    "CL:0000625": "T_Cell",
    "CL:0000815": "T_Cell",
    "CL:0000897": "T_Cell",
    "CL:0000909": "T_Cell",
    "CL:0002420": "T_Cell",
    "CL:0000798": "T_Cell",
    # B cells
    "CL:0000236": "B_Cell",
    "CL:0000786": "B_Cell",
    "CL:0000787": "B_Cell",
    "CL:0000816": "B_Cell",
    "CL:0001201": "B_Cell",
    # NK cells
    "CL:0000623": "NK_Cell",
    "CL:0000938": "NK_Cell",
    "CL:0000939": "NK_Cell",
    # Macrophages
    "CL:0000235": "Macrophage",
    "CL:0000863": "Macrophage",
    "CL:0000890": "Macrophage",
    # Monocytes
    "CL:0000576": "Monocyte",
    "CL:0000860": "Monocyte",
    "CL:0000875": "Monocyte",
    # Dendritic cells
    "CL:0000451": "Dendritic_Cell",
    "CL:0000990": "Dendritic_Cell",
    "CL:0000784": "Dendritic_Cell",
    "CL:0002399": "Dendritic_Cell",
    "CL:0002394": "Dendritic_Cell",
    "CL:0001056": "Dendritic_Cell",
    # Other myeloid
    "CL:0000097": "Mast_Cell",
    "CL:0000775": "Neutrophil",
    # Epithelial
    "CL:0000066": "Epithelial_Luminal",  # Default
    "CL:0002325": "Epithelial_Luminal",
    "CL:0002327": "Epithelial_Luminal",
    "CL:0000182": "Epithelial_Secretory",  # Hepatocyte
    "CL:0002062": "Epithelial_Luminal",  # AT1
    "CL:0002063": "Epithelial_Secretory",  # AT2
    "CL:0002071": "Epithelial_Luminal",  # Enterocyte
    "CL:0000312": "Epithelial_Basal",  # Keratinocyte
    "CL:0000646": "Epithelial_Basal",
    "CL:0000185": "Epithelial_Basal",  # Myoepithelial
    # Stromal
    "CL:0000499": "Fibroblast",  # Default stromal
    "CL:0000057": "Fibroblast",
    "CL:0000186": "Fibroblast",  # Myofibroblast
    "CL:0002553": "Fibroblast",
    "CL:0002557": "Fibroblast",
    "CL:0000192": "Smooth_Muscle",
    "CL:0000669": "Pericyte",
    "CL:0000136": "Adipocyte",
    # Endothelial
    "CL:0000115": "Vascular_Endothelial",
    "CL:0002138": "Lymphatic_Endothelial",
    # Neural
    "CL:0000540": "Glutamatergic_Neuron",  # Default neuron → excitatory
    "CL:0000679": "Glutamatergic_Neuron",
    "CL:0000617": "GABAergic_Neuron",
    "CL:0002608": "Glutamatergic_Neuron",  # Hippocampal pyramidal → excitatory
    "CL:0000128": "Oligodendrocyte",
    "CL:0002453": "Oligodendrocyte",  # OPC → Oligodendrocyte
    "CL:0000127": "Astrocyte",
    "CL:0000129": "Microglia",
}


def get_coarse_category(cl_id: str) -> str:
    """Get the DAPIDL coarse category for a CL ID."""
    if cl_id in CL_TO_COARSE_CATEGORY:
        return CL_TO_COARSE_CATEGORY[cl_id]

    # Try parent chain
    term = get_term(cl_id)
    if term and term.parent_id:
        return get_coarse_category(term.parent_id)

    return "Unknown"


if __name__ == "__main__":
    # Print database statistics
    all_terms = get_all_terms()
    print(f"Total CL terms defined: {len(all_terms)}")
    print(f"  Super-coarse (Level 1): {len(SUPER_COARSE_TERMS)}")
    print(f"  Coarse (Level 2): {len(COARSE_TERMS)}")
    print(f"  Medium (Level 3): {len(MEDIUM_TERMS)}")
    print(f"  Fine (Level 4): {len(FINE_TERMS)}")
    print(f"\nDAPIDF Broad Categories: {len(DAPIDL_BROAD_CATEGORIES)}")
    print(f"DAPIDL Coarse Categories: {len(DAPIDL_COARSE_CATEGORIES)}")
