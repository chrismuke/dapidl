"""Canonical 3-tier CL-based training ontology.

Defines the hierarchy used for both annotation evaluation and DAPI training.
Single source of truth for "coarse / medium / fine" semantics.

Tiers:
    COARSE (5)  CL Super-Coarse                   compartment level
    MEDIUM (12) CL Coarse + L3, data-pruned       body-wide cell types
    FINE   (18) CL Medium L3 + L4 + 2 pathology   subtype + cancer stage

For breast specifically, drop Neural from COARSE → 4. For non-breast, drop the
2 pathology IDs from FINE → 16. The class lists below are the universal union.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TierClass:
    name: str          # short DAPIDL-style display name
    cl_id: str         # CL:xxxxxxx or DAPIDL:xxx (for pathology extensions)
    cl_name: str       # canonical CL name
    parent_cl: str | None = None  # for rollup


# ---------------------------------------------------------------------------
# COARSE — 5 classes (CL Super-Coarse, Level 1)
# ---------------------------------------------------------------------------
COARSE = [
    TierClass("Epithelial",   "CL:0000066", "epithelial cell"),
    TierClass("Immune",       "CL:0000738", "leukocyte"),
    TierClass("Stromal",      "CL:0000499", "stromal cell"),
    TierClass("Endothelial",  "CL:0000115", "endothelial cell"),
    TierClass("Neural",       "CL:0000540", "neuron"),
]

# ---------------------------------------------------------------------------
# MEDIUM — 12 classes (CL Coarse L2 ∪ Medium L3, body-wide)
# ---------------------------------------------------------------------------
MEDIUM = [
    TierClass("Epithelial_Luminal", "CL:0002325", "mammary luminal epithelial cell", "CL:0000066"),
    TierClass("Epithelial_Basal",   "CL:0000646", "basal cell",                      "CL:0000066"),
    TierClass("T_Cell",             "CL:0000084", "T cell",                          "CL:0000738"),
    TierClass("B_Cell",             "CL:0000236", "B cell",                          "CL:0000738"),
    TierClass("Macrophage",         "CL:0000235", "macrophage",                      "CL:0000738"),
    TierClass("Dendritic_Cell",     "CL:0000451", "dendritic cell",                  "CL:0000738"),
    TierClass("Mast_Cell",          "CL:0000097", "mast cell",                       "CL:0000738"),
    TierClass("Fibroblast",         "CL:0000057", "fibroblast",                      "CL:0000499"),
    TierClass("Pericyte",           "CL:0000669", "pericyte",                        "CL:0000499"),
    TierClass("Adipocyte",          "CL:0000136", "adipocyte",                       "CL:0000499"),
    TierClass("Endothelial",        "CL:0000115", "endothelial cell",                "CL:0000115"),
    TierClass("Neural",             "CL:0000540", "neuron",                          "CL:0000540"),
]

# ---------------------------------------------------------------------------
# FINE — 18 classes (15 CL + 2 pathology + 1 universal additions)
# ---------------------------------------------------------------------------
FINE = [
    # T-cell subtypes
    TierClass("CD4_T_Cell",   "CL:0000624", "CD4-positive, alpha-beta T cell", "CL:0000084"),
    TierClass("CD8_T_Cell",   "CL:0000625", "CD8-positive, alpha-beta T cell", "CL:0000084"),
    TierClass("Treg",         "CL:0000815", "regulatory T cell",               "CL:0000624"),
    # B lineage
    TierClass("B_Cell",       "CL:0000236", "B cell",                          "CL:0000738"),
    TierClass("Plasma_Cell",  "CL:0000786", "plasma cell",                     "CL:0000236"),
    # NK
    TierClass("NK_Cell",      "CL:0000623", "natural killer cell",             "CL:0000738"),
    # Myeloid
    TierClass("Macrophage",   "CL:0000235", "macrophage",                      "CL:0000738"),
    TierClass("pDC",          "CL:0000784", "plasmacytoid dendritic cell",     "CL:0000451"),
    TierClass("cDC",          "CL:0000990", "conventional dendritic cell",     "CL:0000451"),
    TierClass("Mast_Cell",    "CL:0000097", "mast cell",                       "CL:0000738"),
    # Epithelial (breast-specific + universal)
    TierClass("Mammary_Luminal", "CL:0002325", "mammary luminal epithelial cell", "CL:0000066"),
    TierClass("Myoepithelial",   "CL:0000185", "myoepithelial cell",              "CL:0000646"),
    # Stromal
    TierClass("Fibroblast",   "CL:0000057", "fibroblast",                      "CL:0000499"),
    TierClass("Pericyte",     "CL:0000669", "pericyte",                        "CL:0000499"),
    TierClass("Adipocyte",    "CL:0000136", "adipocyte",                       "CL:0000499"),
    # Endothelial (rolled up — split if data sufficient)
    TierClass("Endothelial",  "CL:0000115", "endothelial cell",                "CL:0000115"),
    # Pathology extensions (no CL equivalent — cancer states, breast only)
    TierClass("DCIS",         "DAPIDL:DCIS", "ductal carcinoma in situ",       "CL:0002325"),
    TierClass("Invasive",     "DAPIDL:INV",  "invasive carcinoma",             "CL:0002325"),
]

# Indexes for quick lookup
COARSE_NAMES = [t.name for t in COARSE]
MEDIUM_NAMES = [t.name for t in MEDIUM]
FINE_NAMES   = [t.name for t in FINE]

CL_TO_COARSE_NAME = {t.cl_id: t.name for t in COARSE}
CL_TO_MEDIUM_NAME = {t.cl_id: t.name for t in MEDIUM}
CL_TO_FINE_NAME   = {t.cl_id: t.name for t in FINE}

# Pragmatic CL→MEDIUM anchors for types that exist in the ontology but whose
# ancestry walk doesn't naturally land on a MEDIUM-12 entry.
# setdefault: never overrides a real MEDIUM entry.
CL_TO_MEDIUM_NAME.setdefault("CL:0000185", "Epithelial_Basal")   # myoepithelial cell
CL_TO_MEDIUM_NAME.setdefault("CL:0002325", "Epithelial_Luminal")  # mammary luminal epithelial
CL_TO_MEDIUM_NAME.setdefault("CL:0000786", "B_Cell")              # plasma cell → B_Cell tier


# Janesick pathology label → DAPIDL pseudo-CL ID (no real CL match)
JANESICK_PATHOLOGY = {
    "DCIS_1":               "DAPIDL:DCIS",
    "DCIS_2":               "DAPIDL:DCIS",
    "Invasive_Tumor":       "DAPIDL:INV",
    "Prolif_Invasive_Tumor": "DAPIDL:INV",
}


def derive_tier_label(raw_label: str, tier: str, mapper=None) -> str:
    """Map a raw cell-type label to one of the 3 tier vocabularies.

    Args:
        raw_label: source label (Janesick GT, STHELAR ct_tangram, CellTypist, etc.)
        tier: 'coarse' | 'medium' | 'fine'
        mapper: CLMapper instance (singleton via get_mapper() if None)

    Returns:
        One of the tier's class names, or 'Unknown' if not mappable.
    """
    if mapper is None:
        from dapidl.ontology.cl_mapper import get_mapper
        mapper = get_mapper()

    # Janesick pathology labels: route through DAPIDL: pseudo-CL IDs
    if raw_label in JANESICK_PATHOLOGY:
        pseudo_cl = JANESICK_PATHOLOGY[raw_label]
        if tier == "fine":
            return CL_TO_FINE_NAME.get(pseudo_cl, "Unknown")
        # Pathology rolls up to mammary luminal at MEDIUM, epithelial at COARSE
        if tier == "medium":
            return "Epithelial_Luminal"
        if tier == "coarse":
            return "Epithelial"

    # Map to CL
    cl_id = mapper.map(raw_label)
    if cl_id == "UNMAPPED":
        return "Unknown"

    # Roll up to target tier
    if tier == "coarse":
        # Use CLMapper's own rollup (uses CL ancestry chain)
        return mapper.get_hierarchy_level(cl_id, target_level="broad")  # broad ≡ super-coarse

    if tier == "medium":
        return _walk_to_target(cl_id, CL_TO_MEDIUM_NAME)

    if tier == "fine":
        return _walk_to_target(cl_id, CL_TO_FINE_NAME)

    raise ValueError(f"unknown tier: {tier!r}")


def derive_labels(raw_name: str, source: str) -> tuple[str, str]:
    """Single CL-grounded (coarse, medium) derivation. source in {xenium_rep1, xenium_rep2,
    sthelar_breast_s0/s1/s3/s6}; STHELAR raw = ct_tangram, Xenium raw = Janesick-17.
    Unmappable -> ('Unknown', 'Unknown'), never silently mis-binned."""
    from dapidl.ontology.cl_mapper import get_mapper
    mapper = get_mapper()
    return (derive_tier_label(raw_name, "coarse", mapper),
            derive_tier_label(raw_name, "medium", mapper))


def _walk_to_target(cl_id: str, target_set: dict[str, str]) -> str:
    """Walk CL parent chain until we hit a target_set member.

    Fallback: if no specific match found but ancestor is one of the 5 super-coarse
    compartments, return that compartment name (so non-mammary epithelia, etc.
    still get a useful label rather than 'Unknown').
    """
    from dapidl.ontology.cl_database import get_term
    coarse_names = CL_TO_COARSE_NAME  # super-coarse fallback set
    visited = set()
    cur = cl_id
    while cur and cur not in visited:
        visited.add(cur)
        if cur in target_set:
            return target_set[cur]
        term = get_term(cur)
        cur = term.parent_id if term else None
    # Specific class not found — try super-coarse fallback
    visited = set()
    cur = cl_id
    while cur and cur not in visited:
        visited.add(cur)
        if cur in coarse_names:
            return coarse_names[cur]  # generic compartment label
        term = get_term(cur)
        cur = term.parent_id if term else None
    return "Unknown"
