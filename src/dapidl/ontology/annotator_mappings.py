"""Annotator-Specific Mappings to Cell Ontology.

This module provides curated mappings from each annotation method's output
vocabulary to Cell Ontology IDs. These mappings are the primary path for
standardizing labels from different sources.

Supported Annotators:
    - SingleR (BlueprintEncode reference)
    - CellTypist (all models)
    - scType (marker-based)
    - SCINA (marker-based)
    - Azimuth (Satija lab references)
    - popV/scArches

Ground Truth Mappings:
    - Xenium Breast (Janesick et al.)
    - MERSCOPE Breast
    - Common pathology terms

Usage:
    from dapidl.ontology.annotator_mappings import (
        get_annotator_mappings,
        get_gt_mappings,
    )

    # Get all SingleR mappings
    singler_maps = get_annotator_mappings("singler")

    # Create mapper with all annotator mappings
    all_maps = get_all_annotator_mappings()
    mapper = CLMapper(annotator_mappings=all_maps)
"""

from __future__ import annotations


# =============================================================================
# SingleR Mappings (BlueprintEncode reference)
# =============================================================================

SINGLER_TO_CL = {
    # Epithelial
    "Epithelial cells": "CL:0000066",
    "Keratinocytes": "CL:0000312",
    # T cells
    "CD4+ T-cells": "CL:0000624",
    "CD8+ T-cells": "CL:0000625",
    "T-cells": "CL:0000084",
    "Tregs": "CL:0000815",
    # B cells
    "B-cells": "CL:0000236",
    "Plasma cells": "CL:0000786",
    # Myeloid
    "Macrophages": "CL:0000235",
    "Monocytes": "CL:0000576",
    "DC": "CL:0000451",
    "Mast cells": "CL:0000097",
    "Neutrophils": "CL:0000775",
    "Eosinophils": "CL:0000771",  # Eosinophil CL ID
    "Basophils": "CL:0000767",
    # NK
    "NK cells": "CL:0000623",
    # Stromal
    "Fibroblasts": "CL:0000057",
    "Smooth muscle": "CL:0000192",
    "Pericytes": "CL:0000669",
    "Adipocytes": "CL:0000136",
    # Endothelial
    "Endothelial cells": "CL:0000115",
    # Other
    "Erythrocytes": "CL:0000232",
    "Megakaryocytes": "CL:0000556",
    "Platelets": "CL:0000233",
    "HSC": "CL:0000037",  # Hematopoietic stem cell
    "CMP": "CL:0000049",  # Common myeloid progenitor
    "GMP": "CL:0000557",  # Granulocyte-monocyte progenitor
    "MEP": "CL:0000050",  # Megakaryocyte-erythroid progenitor
}


# =============================================================================
# CellTypist Mappings (multiple models)
# =============================================================================

CELLTYPIST_TO_CL = {
    # === Common across models ===
    # T cells
    "T cell": "CL:0000084",
    "CD4-positive, alpha-beta T cell": "CL:0000624",
    "CD8-positive, alpha-beta T cell": "CL:0000625",
    "Regulatory T cell": "CL:0000815",
    "Memory T cell": "CL:0000897",
    "Effector T cell": "CL:0000911",
    "Naive T cell": "CL:0000898",
    "gamma-delta T cell": "CL:0000798",
    "Proliferating T cell": "CL:0000084",  # Map to T cell

    # B cells
    "B cell": "CL:0000236",
    "Plasma cell": "CL:0000786",
    "Memory B cell": "CL:0000787",
    "Naive B cell": "CL:0001201",
    "Pre-B cell": "CL:0000816",
    "Germinal center B cell": "CL:0000844",

    # Myeloid
    "Macrophage": "CL:0000235",
    "Monocyte": "CL:0000576",
    "Classical monocyte": "CL:0000860",
    "Non-classical monocyte": "CL:0000875",
    "Dendritic cell": "CL:0000451",
    "Conventional dendritic cell": "CL:0000990",
    "Plasmacytoid dendritic cell": "CL:0000784",
    "cDC1": "CL:0002394",
    "cDC2": "CL:0002399",
    "Mast cell": "CL:0000097",
    "Neutrophil": "CL:0000775",
    "Eosinophil": "CL:0000771",
    "Basophil": "CL:0000767",

    # NK/ILC
    "NK cell": "CL:0000623",
    "Natural killer cell": "CL:0000623",
    "ILC1": "CL:0001068",
    "ILC2": "CL:0001069",
    "ILC3": "CL:0001070",

    # Epithelial
    "Epithelial cell": "CL:0000066",
    "Luminal epithelial cell": "CL:0002325",
    "Luminal epithelial cell of mammary gland": "CL:0002325",
    "Basal cell": "CL:0000646",
    "Myoepithelial cell": "CL:0000185",
    "Secretory cell": "CL:0000151",
    "Ciliated cell": "CL:0000064",
    "Goblet cell": "CL:0000160",
    "Keratinocyte": "CL:0000312",
    "Hepatocyte": "CL:0000182",
    "Cholangiocyte": "CL:0000227",
    "Enterocyte": "CL:0002071",
    "Paneth cell": "CL:0000510",
    "Club cell": "CL:0000158",

    # Stromal
    "Fibroblast": "CL:0000057",
    "Myofibroblast": "CL:0000186",
    "Smooth muscle cell": "CL:0000192",
    "Pericyte": "CL:0000669",
    "Adipocyte": "CL:0000136",
    "Stellate cell": "CL:0000632",

    # Endothelial
    "Endothelial cell": "CL:0000115",
    "Vascular endothelial cell": "CL:0000115",
    "Lymphatic endothelial cell": "CL:0002138",
    "Arterial endothelial cell": "CL:1000413",
    "Venous endothelial cell": "CL:0002543",

    # Immune_All models specific
    "Age-associated B cell": "CL:0000236",
    "Alveolar macrophage": "CL:0000583",
    "CD16+ NK cell": "CL:0000939",
    "CD56bright NK cell": "CL:0000938",
    "Kupffer cell": "CL:0000091",
    "Langerhans cell": "CL:0000453",
    "Megakaryocyte": "CL:0000556",
    "Platelet": "CL:0000233",

    # Breast model specific
    "Breast glandular cell": "CL:0002327",
}


# =============================================================================
# scType Mappings (marker-based)
# =============================================================================

SCTYPE_TO_CL = {
    # T cells
    "T cells": "CL:0000084",
    "CD4+ T cells": "CL:0000624",
    "CD8+ T cells": "CL:0000625",
    "Regulatory T cells": "CL:0000815",
    "Naive T cells": "CL:0000898",
    "Memory T cells": "CL:0000897",

    # B cells
    "B cells": "CL:0000236",
    "Plasma cells": "CL:0000786",
    "Memory B cells": "CL:0000787",

    # Myeloid
    "Macrophages": "CL:0000235",
    "M1 Macrophages": "CL:0000863",
    "M2 Macrophages": "CL:0000890",
    "Monocytes": "CL:0000576",
    "Dendritic cells": "CL:0000451",
    "cDC1": "CL:0002394",
    "cDC2": "CL:0002399",
    "pDC": "CL:0000784",
    "Mast cells": "CL:0000097",
    "Neutrophils": "CL:0000775",

    # NK
    "NK cells": "CL:0000623",

    # Epithelial
    "Epithelial": "CL:0000066",
    "Epithelial cells": "CL:0000066",
    "Luminal cells": "CL:0002325",
    "Basal cells": "CL:0000646",
    "Myoepithelial cells": "CL:0000185",

    # Stromal
    "Fibroblasts": "CL:0000057",
    "Myofibroblasts": "CL:0000186",
    "Smooth muscle cells": "CL:0000192",
    "Pericytes": "CL:0000669",
    "Adipocytes": "CL:0000136",

    # Endothelial
    "Endothelial cells": "CL:0000115",
    "Lymphatic endothelial": "CL:0002138",
    "Blood endothelial": "CL:0000115",
}


# =============================================================================
# SCINA Mappings (underscore convention)
# =============================================================================

SCINA_TO_CL = {
    # T cells
    "T_cells": "CL:0000084",
    "CD4_T_cells": "CL:0000624",
    "CD8_T_cells": "CL:0000625",
    "Tregs": "CL:0000815",
    "Regulatory_T_cells": "CL:0000815",

    # B cells
    "B_cells": "CL:0000236",
    "Plasma_cells": "CL:0000786",

    # Myeloid
    "Macrophages": "CL:0000235",
    "Monocytes": "CL:0000576",
    "Dendritic_cells": "CL:0000451",
    "Mast_cells": "CL:0000097",
    "Neutrophils": "CL:0000775",

    # NK
    "NK_cells": "CL:0000623",

    # Epithelial
    "Epithelial_cells": "CL:0000066",
    "Luminal_cells": "CL:0002325",
    "Basal_cells": "CL:0000646",

    # Stromal
    "Fibroblasts": "CL:0000057",
    "Smooth_muscle_cells": "CL:0000192",

    # Endothelial
    "Endothelial": "CL:0000115",
    "Endothelial_cells": "CL:0000115",
    "Lymphatic_endothelial": "CL:0002138",
}


# =============================================================================
# Azimuth Mappings (Satija lab)
# =============================================================================

AZIMUTH_TO_CL = {
    # PBMC reference
    "CD4 Naive": "CL:0000624",
    "CD4 TCM": "CL:0000897",
    "CD4 TEM": "CL:0000897",
    "CD4 CTL": "CL:0002420",
    "CD8 Naive": "CL:0000625",
    "CD8 TCM": "CL:0000909",
    "CD8 TEM": "CL:0000909",
    "MAIT": "CL:0000940",
    "Treg": "CL:0000815",
    "gdT": "CL:0000798",
    "dnT": "CL:0000084",
    "NK": "CL:0000623",
    "NK_CD56bright": "CL:0000938",
    "NK Proliferating": "CL:0000623",
    "B naive": "CL:0001201",
    "B intermediate": "CL:0000236",
    "B memory": "CL:0000787",
    "Plasmablast": "CL:0000980",
    "CD14 Mono": "CL:0000860",
    "CD16 Mono": "CL:0000875",
    "cDC1": "CL:0002394",
    "cDC2": "CL:0002399",
    "pDC": "CL:0000784",
    "ASDC": "CL:0000451",
    "HSPC": "CL:0000037",
    "Platelet": "CL:0000233",
    "Eryth": "CL:0000232",
    "ILC": "CL:0001065",

    # Lung reference
    "AT1": "CL:0002062",
    "AT2": "CL:0002063",
    "Basal": "CL:0000646",
    "Ciliated": "CL:0000064",
    "Club": "CL:0000158",
    "Goblet": "CL:0000160",
    "Mucous": "CL:0000319",
    "Alveolar Mph": "CL:0000583",
    "Interstitial Mph": "CL:0000235",
    "Mast": "CL:0000097",
    "Cap": "CL:0000115",
    "Arterial": "CL:1000413",
    "Venous": "CL:0002543",
    "Lymphatic": "CL:0002138",
    "Pericyte": "CL:0000669",
    "SMC": "CL:0000192",
    "Fibroblast": "CL:0000057",
    "Myofibroblast": "CL:0000186",
}


# =============================================================================
# Ground Truth Mappings (Dataset-Specific)
# =============================================================================

# Xenium Breast Cancer (Janesick et al.)
XENIUM_BREAST_GT_TO_CL = {
    # Epithelial/Tumor
    "Invasive_Tumor": "CL:0000066",
    "DCIS_1": "CL:0000066",
    "DCIS_2": "CL:0000066",
    "Prolif_Invasive_Tumor": "CL:0000066",
    "Myoepi_ACTA2+": "CL:0000185",
    "Myoepi_KRT15+": "CL:0000185",

    # Immune
    "CD4+_T_Cells": "CL:0000624",
    "CD8+_T_Cells": "CL:0000625",
    "B_Cells": "CL:0000236",
    "Macrophages_1": "CL:0000235",
    "Macrophages_2": "CL:0000235",
    "IRF7+_DCs": "CL:0000784",  # IRF7+ suggests pDC
    "LAMP3+_DCs": "CL:0001056",  # LAMP3+ mature DC
    "Mast_Cells": "CL:0000097",

    # Stromal
    "Stromal": "CL:0000057",
    "Perivascular-Like": "CL:0000669",

    # Endothelial
    "Endothelial": "CL:0000115",

    # Hybrid/Artifact
    "Stromal_&_T_Cell_Hybrid": "UNMAPPED",
    "T_Cell_&_Tumor_Hybrid": "UNMAPPED",
    "Unlabeled": "UNMAPPED",
}

# MERSCOPE Breast (Vizgen)
MERSCOPE_BREAST_GT_TO_CL = {
    # Similar structure to Xenium
    "Tumor": "CL:0000066",
    "Epithelial": "CL:0000066",
    "Luminal": "CL:0002325",
    "Basal": "CL:0000646",
    "basal": "CL:0000646",  # Lowercase variant
    "T_Cell": "CL:0000084",
    "CD4_T_Cell": "CL:0000624",
    "CD8_T_Cell": "CL:0000625",
    "B_Cell": "CL:0000236",
    "Macrophage": "CL:0000235",
    "Dendritic_Cell": "CL:0000451",
    "NK_Cell": "CL:0000623",
    "Mast_Cell": "CL:0000097",
    "Fibroblast": "CL:0000057",
    "Pericyte": "CL:0000669",
    "Endothelial": "CL:0000115",
    # MERSCOPE consensus annotation labels (Jan 2025)
    "Fibro-major": "CL:0000057",  # Fibroblast
    "Fibro-matrix": "CL:0000057",  # Fibroblast (matrix-producing)
    "Lumsec-HLA": "CL:0002325",  # Luminal secretory epithelial
    "Lumsec-basal": "CL:0002325",  # Luminal-basal transitional
    "Lumsec-myo": "CL:0000185",  # Myoepithelial (luminal-myo)
    "Lymph-major": "CL:0000542",  # Lymphocyte (general)
    "Macro-lipo": "CL:0000235",  # Lipid-associated macrophage
    "Mono-classical": "CL:0000860",  # Classical monocyte
    "Mono-NonClassical": "CL:0000875",  # Non-classical monocyte
    "Vas-arterial": "CL:1000413",  # Arterial endothelial
    "Vas-capillary": "CL:0000115",  # Capillary endothelial (use generic endothelial)
    "bmem_unswitched": "CL:0000787",  # Memory B cell (unswitched isotype)
    "mDC": "CL:0000990",  # Conventional/myeloid dendritic cell
}

# Common pathology terms
PATHOLOGY_TO_CL = {
    "lymphocytes": "CL:0000542",
    "lymphocyte": "CL:0000542",
    "TILs": "CL:0000542",
    "tumor infiltrating lymphocytes": "CL:0000542",
    "histiocytes": "CL:0000235",
    "histiocyte": "CL:0000235",
    "carcinoma": "CL:0000066",
    "carcinoma cells": "CL:0000066",
    "tumor cells": "CL:0000066",
    "cancer cells": "CL:0000066",
    "stroma": "CL:0000499",
    "stromal cells": "CL:0000499",
    "spindle cells": "CL:0000057",
    "plasma cells": "CL:0000786",
    "mast cells": "CL:0000097",
}

# Mouse Brain (Xenium, CellTypist Mouse_Isocortex_Hippocampus)
# Allen Brain Atlas nomenclature
MOUSE_BRAIN_GT_TO_CL = {
    # ===================================================================
    # Mouse_Isocortex_Hippocampus.pkl labels (simple format)
    # ===================================================================

    # Cortical Glutamatergic Neurons (IT = Intratelencephalic, PT = Pyramidal tract)
    "L4/5 IT CTX": "CL:0000679",  # Glutamatergic neuron
    "L2/3 IT CTX": "CL:0000679",  # Glutamatergic neuron
    "L2/3 IT ENTl": "CL:0000679",  # Entorhinal cortex IT neuron
    "L5 PT CTX": "CL:0000679",  # Layer 5 pyramidal tract neuron
    "L5/6 IT TPE-ENT": "CL:0000679",  # Layer 5/6 IT neuron
    "L6 IT CTX": "CL:0000679",  # Layer 6 IT neuron
    "L6 CT CTX": "CL:0000679",  # Layer 6 corticothalamic neuron
    "L6b CTX": "CL:0000679",  # Layer 6b neuron
    "L6 IT ENTl": "CL:0000679",  # Layer 6 IT entorhinal
    "L2/3 IT PPP": "CL:0000679",  # Layer 2/3 IT perirhinal
    "NP PPP": "CL:0000679",  # Near-projecting pyramidal
    "CT SUB": "CL:0000679",  # Corticothalamic subiculum

    # GABAergic Interneurons
    "Vip": "CL:0000617",  # VIP+ GABAergic interneuron
    "Lamp5": "CL:0000617",  # Lamp5+ GABAergic interneuron
    "Sst": "CL:0000617",  # Somatostatin+ interneuron
    "Pvalb": "CL:0000617",  # Parvalbumin+ interneuron

    # Hippocampal neurons
    "DG": "CL:0000679",  # Dentate gyrus granule cell
    "CA1-ProS": "CL:0002608",  # Hippocampal pyramidal neuron
    "CA2-IG-FC": "CL:0002608",  # Hippocampal pyramidal neuron
    "SUB-ProS": "CL:0002608",  # Subiculum pyramidal neuron

    # Glial cells
    "Oligo": "CL:0000128",  # Oligodendrocyte
    "OPC": "CL:0002453",  # Oligodendrocyte precursor cell
    "Astro": "CL:0000127",  # Astrocyte
    "Micro-PVM": "CL:0000129",  # Microglia

    # Vascular
    "Endo": "CL:0000115",  # Endothelial cell
    "VLMC": "CL:0000669",  # Vascular leptomeningeal cell (pericyte-like)

    # Generic neuron mapping for unspecified types
    "Neuron": "CL:0000540",  # Neuron

    # ===================================================================
    # Mouse_Whole_Brain.pkl labels (cluster ID prefix format)
    # These are the same cell types with Allen Brain Atlas cluster IDs
    # ===================================================================

    # Glutamatergic neurons (suffix "Glut")
    "006 L4/5 IT CTX Glut": "CL:0000679",
    "007 L2/3 IT CTX Glut": "CL:0000679",
    "003 L5/6 IT TPE-ENT Glut": "CL:0000679",
    "005 L5 IT CTX Glut": "CL:0000679",
    "022 L5 ET CTX Glut": "CL:0000679",
    "016 CA1-ProS Glut": "CL:0002608",
    "012 MEA Slc17a7 Glut": "CL:0000679",
    "134 PH-ant-LHA Otp Bsx Glut": "CL:0000679",
    "222 PB Evx2 Glut": "CL:0000679",
    "170 PAG-MRN Tfap2b Glut": "CL:0000679",
    "167 PRC-PAG Tcf7l2 Irx2 Glut": "CL:0000679",
    "168 SPA-SPFm-SPFp-POL-PIL-PoT Sp9 Glut": "CL:0000679",
    "118 ADP-MPO Trp73 Glut": "CL:0000679",
    "245 SPVI-SPVC Tlx3 Ebf3 Glut": "CL:0000679",

    # GABAergic neurons (suffix "Gaba")
    "049 Lamp5 Gaba": "CL:0000617",
    "046 Vip Gaba": "CL:0000617",
    "090 BST-MPN Six3 Nrgn Gaba": "CL:0000617",
    "063 STR D1 Sema5a Gaba": "CL:0000617",
    "210 PRT Mecom Gaba": "CL:0000617",

    # Glycinergic/GABAergic neurons
    "285 MY Lhx1 Gly-Gaba": "CL:0000617",
}


# =============================================================================
# Factory Functions
# =============================================================================


def get_annotator_mappings(annotator: str) -> dict[str, str]:
    """Get CL mappings for a specific annotator.

    Args:
        annotator: Annotator name (case-insensitive)

    Returns:
        Dict mapping annotator outputs to CL IDs
    """
    annotator_maps = {
        "singler": SINGLER_TO_CL,
        "celltypist": CELLTYPIST_TO_CL,
        "sctype": SCTYPE_TO_CL,
        "scina": SCINA_TO_CL,
        "azimuth": AZIMUTH_TO_CL,
    }
    return annotator_maps.get(annotator.lower(), {})


def get_gt_mappings(dataset: str) -> dict[str, str]:
    """Get CL mappings for a specific ground truth dataset.

    Args:
        dataset: Dataset name

    Returns:
        Dict mapping GT labels to CL IDs
    """
    gt_maps = {
        "xenium_breast": XENIUM_BREAST_GT_TO_CL,
        "merscope_breast": MERSCOPE_BREAST_GT_TO_CL,
        "pathology": PATHOLOGY_TO_CL,
        "mouse_brain": MOUSE_BRAIN_GT_TO_CL,
    }
    return gt_maps.get(dataset.lower(), {})


def get_all_annotator_mappings() -> dict[str, str]:
    """Get combined mappings from all annotators.

    Returns:
        Dict with all annotator output → CL ID mappings.
        Later annotators override earlier ones if there are conflicts.
    """
    combined = {}
    for name in ["singler", "celltypist", "sctype", "scina", "azimuth"]:
        combined.update(get_annotator_mappings(name))
    return combined


def get_all_gt_mappings() -> dict[str, str]:
    """Get combined mappings from all ground truth datasets.

    Returns:
        Dict with all GT label → CL ID mappings.
    """
    combined = {}
    for name in ["xenium_breast", "merscope_breast", "pathology", "mouse_brain"]:
        combined.update(get_gt_mappings(name))
    return combined


def print_mapping_stats() -> None:
    """Print statistics about available mappings."""
    print("=" * 60)
    print("Annotator Mapping Statistics")
    print("=" * 60)

    for name in ["singler", "celltypist", "sctype", "scina", "azimuth"]:
        maps = get_annotator_mappings(name)
        print(f"{name:15} {len(maps):4} mappings")

    print("\nGround Truth Mapping Statistics")
    print("-" * 60)

    for name in ["xenium_breast", "merscope_breast", "pathology", "mouse_brain"]:
        maps = get_gt_mappings(name)
        print(f"{name:15} {len(maps):4} mappings")

    print("\nTotal unique CL IDs covered:")
    all_cl_ids = set()
    for maps in [get_all_annotator_mappings(), get_all_gt_mappings()]:
        all_cl_ids.update(v for v in maps.values() if v != "UNMAPPED")
    print(f"  {len(all_cl_ids)} unique CL terms")


if __name__ == "__main__":
    print_mapping_stats()
