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
        # Human_Lung_Atlas.pkl - alveolar types
        "AT1",  # alveolar type 1
        "AT2",  # alveolar type 2
        "AT2 proliferating",
        "Club (nasal)",
        "Club (non-nasal)",
        "Goblet (nasal)",
        "Goblet (bronchial)",
        "Multiciliated (nasal)",
        "Multiciliated (non-nasal)",
        "SMG duct",  # submandibular gland duct
        "SMG serous",
        "SMG mucous",
        "Deuterosomal",
        "Suprabasal",
        "Surface epithelium",
        "Hillock-like",  # transitional epithelium
        # Developing_Human_Organs.pkl
        "Distal lung epithelium",
        "Gastrointestinal epithelium",
        "Intestinal epithelium",
        # Cells_Intestinal_Tract.pkl
        "BEST4+ epithelial",
        "Colonocyte",
        "Crypt",  # intestinal crypt
        "FDCSP epithelium",
        # Cells_Human_Tonsil.pkl
        "Squamous epithelium",
        # Adult_Human_Skin.pkl - keratinocyte subtypes
        "Undifferentiated_KC",  # keratinocyte
        "Differentiated_KC",
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
        # Immune_All_High.pkl - additional immune types
        "Cycling cells",  # proliferating immune cells
        "Double-positive thymocytes",
        "Double-negative thymocytes",
        "ILC",  # innate lymphoid cells
        "MNP",  # mononuclear phagocytes
        "Myelocytes",
        # Immune_All_Low.pkl - detailed immune subtypes
        "Age-associated B cells",
        "CD16+ NK cells",
        "CD16- NK cells",
        "CD16+CD56- NK",
        "CD16-CD56+ NK",
        "CD16-CD56dim NK",
        "CRTAM+ gamma-delta T cells",
        "Cycling B cells",
        "Cycling NK cells",
        "Cycling T cells",
        "Cycling B cell",
        "Cycling plasma cell",
        "Early erythroid",
        "Erythroblast",
        "Follicular B cells",
        "GC B cell",  # germinal center B cell
        "IgA plasma cell",
        "IgG plasma cell",
        "Large pre-B cells",
        "Small pre-B cells",
        "MAIT cells",  # mucosal-associated invariant T cells
        "Memory B cells",
        "MBC FCRL5+",  # memory B cell
        "Naive B cells",
        "NBC IFN-activated",  # naive B cell
        "Naive CD8 T",
        "SCM CD8 T",  # stem cell memory
        "CM CD8 T",  # central memory
        "RM CD8 T",  # resident memory
        "RM CD8 activated T",
        "ZNF683+ CD8 T",
        "SELL+ CD4 T",
        "SELL+ CD8 T",
        "Activated CD4 T",
        "Activated CD8 T",
        "Activated T",
        "Tcm/Naive cytotoxic T cells",
        "Tcm/Naive helper T cells",
        "Tem/Effector helper T cells",
        "Tem/Temra cytotoxic T cells",
        "Tem/Trm cytotoxic T cells",
        "Trm cytotoxic T cells",
        "Type 1 helper T cells",
        "Type 17 helper T cells",
        "Resident NK",
        # Cells_Intestinal_Tract.pkl - additional immune
        "CLC+ Mast cell",
        "BEST2+ Goblet cell",  # sometimes classified as immune-associated
        "Erythrophagocytic macrophages",
        "Intermediate macrophages",
        "LYVE1+ Macrophage",
        "MMP9+ Inflammatory macrophage",
        # Cells_Human_Tonsil.pkl - tonsil-specific immune
        "C1Q Slan-like",  # dendritic-like
        "CM Pre-non-Tfh",
        "CM PreTfh",
        "Cycling",  # cycling immune cells in tonsil
        # Adult_Human_Skin.pkl - skin immune
        "ILC1_3",
        "ILC1_NK",
        "ILC2",
        "Inf_mac",  # inflammatory macrophage
        "LC",  # Langerhans cell
        "Tc",  # cytotoxic T cell
        # Cells_Human_Tonsil.pkl - B cell lineage (germinal center)
        "DZ early Sphase",  # dark zone B cells
        "DZ late Sphase",
        "DZ late G2Mphase",
        "DZ non proliferative",
        "DZ migratory PC precursor",
        "DZ_LZ transition",
        "LZ non proliferative",
        "LZ Tbet",
        "MBC derived PC precursor",
        "IgD PC precursor",
        "Precursor MBCs",  # memory B cell precursors
        "ncsMBC",  # non-class-switched memory B cells
        "csMBC FCRL4/5+",  # class-switched memory B cells
        "Early MBC",
        "Early GC-commited NBC",
        "Proliferative NBC",
        "Naive B",
        "Immature B",
        # Cells_Human_Tonsil.pkl - T cell lineage
        "Th",  # T helper cells
        "Tfr",  # T follicular regulatory
        "T-Eff-Mem",  # effector memory T
        "T-Trans-Mem",  # transitional memory T
        "Eff-Tregs",  # effector Tregs
        "Eff-Tregs-IL32",
        "DN",  # double negative T cells
        "Tpex",  # progenitor exhausted T cells
        "T(agonist)",
        "MAIT/CD161+TRDV2+ gd T-cells",
        # Additional dendritic cell types
        "aDC3",  # activated DC
        "MigDC",  # migratory DC
        "Migratory DCs",
        "migLC",  # migratory Langerhans cells
        "moDC",  # monocyte-derived DC
        # Liver-specific immune
        "Hofbauer cells",  # fetal macrophages
        # ILC subtypes
        "LTi-like NCR+ ILC3",
        "LTi-like NCR- ILC3",
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
        # Human_Lung_Atlas.pkl - lung stromal
        "Adventitial fibroblasts",
        "Peribronchial fibroblasts",
        "Subpleural fibroblasts",
        "Immature pericyte",
        "EC aerocyte capillary",
        "EC arterial",
        "EC general capillary",
        "EC venous pulmonary",
        "EC venous systemic",
        "Interstitial Mph perivascular",
        # Developing_Human_Organs.pkl - developing stromal
        "Mesenchyme subtype 1",
        "Mesenchyme subtype 2",
        "Mesenchyme subtype 3",
        "Mesenchyme subtype 4",
        "Mesenchyme subtype 5",
        "Proliferative mesenchyme",
        "Mesoderm 1 (HAND1+)",  # mesoderm subtype
        "PNS glia",  # peripheral nervous system glia
        "PNS neuron",
        "Schwann_1",  # Schwann cell subtype
        "cycling ENCC/glia",  # cycling enteric neural crest/glia
        "ENCC/glia Progenitor",
        "Glia 2 (ELN+)",
        "Glia 3 (BCAN+)",
        # Cells_Intestinal_Tract.pkl - intestinal stromal
        "Stromal 1 (CCL11+)",
        "Stromal 1 (ADAMDEC1+)",
        "Stromal 2 (CH25H+)",
        "Stromal 2 (NPY+)",
        "Stromal 3 (C7+)",
        "Stromal 3 (KCNN3+)",
        "Stromal 4 (MMP1+)",
        "CLDN10+ cells",  # often stromal-associated
        "cycling stromal",
        "Distal progenitor",  # intestinal progenitor (stromal-like)
        "Fetal arterial EC",  # fetal endothelial
        "Fetal venous EC",
        "LEC6 (ADAMTS4+)",  # lymphatic endothelial
        # Cells_Human_Tonsil.pkl - tonsil stromal
        "COL27A1+ FDC",  # follicular dendritic cell (stromal-like)
        "FDCSP+ FDC",
        # Adult_Human_Skin.pkl - skin fibroblasts
        "F1",  # fibroblast subtype 1
        "F2",  # fibroblast subtype 2
        "F3",  # fibroblast subtype 3
        "LE1",  # lymphatic endothelial
        "LE2",
        "VE1",  # vascular endothelial
        "VE2",
        "VE3",
        # Adult_Human_Vascular.pkl - vascular cell types
        "art_ec_1",  # arterial endothelial
        "art_ec_2",
        "art_smc",  # arterial smooth muscle
        "brain_art_ec",
        "brain_art_smc",  # brain arterial smooth muscle
        "aorta_coronary_smc",
        "cap_ec",  # capillary endothelial
        "cap_pc",  # capillary pericyte
        "cap_lec",  # capillary lymphatic endothelial
        "adip_cap_ec",  # adipose capillary endothelial
        "kidney_cap_ec",  # kidney capillary endothelial
        "liver_pc",  # liver pericyte
        "liver_hsec",  # liver hepatic sinusoidal endothelial
        # Additional Adult_Human_Vascular.pkl types
        "kidney_art_ec",  # kidney arterial endothelial
        "uterine_pc",  # uterine pericyte
        "uterine_smc",  # uterine smooth muscle
        "endometrium_cap_ec",  # endometrium capillary endothelial
        "ven_ec_1",  # venous endothelial subtypes
        "ven_ec_2",
        "pul_cap_ec",  # pulmonary capillary endothelial
        "pul_pc",  # pulmonary pericyte
        "glomeruli_ec",  # glomerular endothelial
        "myo_cap_ec",  # myocardial capillary endothelial
        "pericentral_cap_ec",  # pericentral capillary
        "periportal_cap_ec",  # periportal capillary
        "ceiling_lec",  # ceiling lymphatic endothelial
        "perifollicular_sinus_lec",  # perifollicular sinus lymphatic
        "SMC (PART1/CAPN3+)",  # smooth muscle cell subtype
        "Mature venous EC",
        "heart_pc",  # heart pericyte
        "floor_lec",  # floor lymphatic endothelial
        "littoral_EC",  # littoral cell (splenic endothelial)
        "bridging_lec",  # bridging lymphatic endothelial
        "vein_ec",  # venous endothelial
        "vein_smc",  # venous smooth muscle
        "collecting_lec",  # collecting lymphatic
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
