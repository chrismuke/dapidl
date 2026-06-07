"""DEFAULT_MARKERS copied verbatim from
dapidl.pipeline.components.annotators.sctype (lines 29-109).

Extracted as pure data so the cloud worker never imports the dapidl.pipeline
package (which pulls pydantic + the full annotation/registry stack). Identical
to get_default_markers() used for the other 5 slides.
"""
DEFAULT_MARKERS = {
    # Epithelial
    "Epithelial": {
        "positive": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "KRT7", "MUC1"],
        "negative": ["PTPRC", "VIM", "PECAM1"],
    },
    # Immune - T cells
    "T cells": {
        "positive": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC"],
        "negative": ["CD19", "CD14", "NCAM1"],
    },
    "CD4+ T cells": {
        "positive": ["CD3D", "CD3E", "CD4", "IL7R"],
        "negative": ["CD8A", "CD8B"],
    },
    "CD8+ T cells": {
        "positive": ["CD3D", "CD3E", "CD8A", "CD8B", "GZMB"],
        "negative": ["CD4"],
    },
    "Regulatory T cells": {
        "positive": ["CD3D", "CD4", "FOXP3", "IL2RA", "CTLA4"],
        "negative": ["CD8A"],
    },
    # Immune - B cells
    "B cells": {
        "positive": ["CD19", "CD79A", "CD79B", "MS4A1", "PAX5"],
        "negative": ["CD3D", "CD14"],
    },
    "Plasma cells": {
        "positive": ["SDC1", "MZB1", "JCHAIN", "IGHG1", "XBP1"],
        "negative": ["MS4A1", "CD19"],
    },
    # Immune - Myeloid
    "Macrophages": {
        "positive": ["CD68", "CD163", "CSF1R", "MARCO", "MSR1"],
        "negative": ["CD3D", "CD19"],
    },
    "Monocytes": {
        "positive": ["CD14", "FCGR3A", "CSF1R", "LYZ", "S100A8"],
        "negative": ["CD3D", "CD19", "CD68"],
    },
    "Dendritic cells": {
        "positive": ["ITGAX", "CD1C", "CLEC9A", "FLT3", "HLA-DRA"],
        "negative": ["CD3D", "CD14", "CD19"],
    },
    "Mast cells": {
        "positive": ["KIT", "TPSAB1", "CPA3", "MS4A2", "FCER1A"],
        "negative": ["CD3D", "CD14"],
    },
    # Immune - NK
    "NK cells": {
        "positive": ["NCAM1", "NKG7", "GNLY", "KLRD1", "KLRF1"],
        "negative": ["CD3D", "CD19"],
    },
    # Stromal
    "Fibroblasts": {
        "positive": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA", "FAP"],
        "negative": ["PTPRC", "EPCAM", "PECAM1"],
    },
    "Myofibroblasts": {
        "positive": ["ACTA2", "TAGLN", "MYL9", "COL1A1"],
        "negative": ["PTPRC", "EPCAM"],
    },
    "Pericytes": {
        "positive": ["PDGFRB", "RGS5", "CSPG4", "ACTA2"],
        "negative": ["PECAM1", "PTPRC"],
    },
    "Adipocytes": {
        "positive": ["ADIPOQ", "LEP", "PLIN1", "FABP4"],
        "negative": ["PTPRC", "EPCAM"],
    },
    # Endothelial
    "Endothelial cells": {
        "positive": ["PECAM1", "VWF", "CDH5", "KDR", "CLDN5"],
        "negative": ["PTPRC", "EPCAM", "ACTA2"],
    },
    "Lymphatic endothelial": {
        "positive": ["PROX1", "LYVE1", "PDPN", "FLT4"],
        "negative": ["PTPRC", "EPCAM"],
    },
}
