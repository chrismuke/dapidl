"""Automatic CellTypist model selection without ground truth.

This module implements automatic model selection using:
1. Confidence score distribution
2. Marker gene enrichment (AUCell-inspired)
3. Spatial coherence validation
4. Proportion plausibility checks

The AutoModelSelector can:
- Score individual models
- Select the best model(s) for a tissue
- Build weighted consensus from top models

Usage:
    from dapidl.pipeline.components.annotators.auto_selector import AutoModelSelector

    selector = AutoModelSelector(tissue_type="breast_tumor")
    best_models = selector.select_models(adata, n_models=3)
    consensus = selector.build_consensus(adata, models=best_models)
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

# Tissue → CellTypist model candidates for auto-selection.
# Ordered by relevance: tissue-specific first, then immune, then general.
# Validated against gene panel overlap (March 2026 analysis).
TISSUE_MODELS = {
    "breast": [
        "Cells_Adult_Breast.pkl",       # 77% overlap w/ hBreast_320g panel, 58 types
        "Immune_All_High.pkl",          # 78% overlap, 32 broad immune types
    ],
    "breast_tumor": [
        "Cells_Adult_Breast.pkl",       # Tissue-specific, 58 types
        "Immune_All_High.pkl",          # Broad immune, 32 types
    ],
    "lung": [
        "Human_Lung_Atlas.pkl",         # 81-88% overlap, 61 types (HLCA)
        "Cells_Lung_Airway.pkl",        # 80-91% overlap, 78 types (more granular)
        "Immune_All_High.pkl",          # Immune detail
    ],
    "liver": [
        "Healthy_Human_Liver.pkl",      # 48-50% overlap but correct types (hepatocytes!)
        "Immune_All_High.pkl",          # Immune detail
    ],
    "kidney": [
        "Immune_All_High.pkl",          # No kidney-specific CT model exists
        "Pan_Fetal_Human.pkl",          # 82% overlap, broad coverage
    ],
    "heart": [
        "Healthy_Adult_Heart.pkl",      # 51% overlap, 75 types (cardiomyocytes)
        "Immune_All_High.pkl",          # Immune detail
    ],
    "colon": [
        "Cells_Intestinal_Tract.pkl",   # 70% overlap, 134 types (most granular)
        "Human_Colorectal_Cancer.pkl",  # 84% overlap, 36 types (CRC-specific)
        "Immune_All_High.pkl",          # Immune detail
    ],
    "colorectal": [
        "Human_Colorectal_Cancer.pkl",  # 69% overlap, 36 types (CRC-specific)
        "Cells_Intestinal_Tract.pkl",   # 65% overlap, 134 types
        "Immune_All_High.pkl",          # Immune detail
    ],
    "skin": [
        "Adult_Human_Skin.pkl",         # 57% overlap, 34 types (KC, melanocyte, etc.)
        "Immune_All_High.pkl",          # Immune detail
    ],
    "tonsil": [
        "Cells_Human_Tonsil.pkl",       # 51% overlap, 117 types (lymphoid-heavy)
        "Immune_All_High.pkl",          # Immune detail
    ],
    "lymph_node": [
        "Cells_Human_Tonsil.pkl",       # Closest lymphoid model
        "Immune_All_High.pkl",          # Immune detail
        "Immune_All_Low.pkl",           # Fine-grained immune
    ],
    "pancreas": [
        "Immune_All_High.pkl",          # PancreaticIslet model is useless (18% overlap)
        "Pan_Fetal_Human.pkl",          # Broad coverage
    ],
    "brain": [
        "Immune_All_High.pkl",          # No adult human brain CT model
        "Pan_Fetal_Human.pkl",          # Broad coverage
    ],
    "ovary": [
        "Immune_All_High.pkl",          # No ovary-specific CT model
        "Pan_Fetal_Human.pkl",          # Broad coverage
    ],
    "cervix": [
        "Immune_All_High.pkl",          # No cervix-specific CT model
        "Pan_Fetal_Human.pkl",          # Broad coverage
    ],
    "mouse_brain": [
        "Mouse_Isocortex_Hippocampus.pkl",  # 97% overlap, 42 types
        "Mouse_Whole_Brain.pkl",            # 96% overlap, 334 types
    ],
    "generic": [
        "Immune_All_High.pkl",          # Works everywhere with immune markers
        "Pan_Fetal_Human.pkl",          # Broadest coverage (138 types)
    ],
}

# Per-dataset model mapping: dataset directory name → best CellTypist models.
# Primary model is first, immune model second, optional extras after.
# Validated against gene panel overlap analysis (March 2026).
DATASET_MODELS = {
    # ── Breast ──────────────────────────────────────────────────────
    "xenium-breast-tumor-rep1": {
        "tissue": "breast",
        "models": ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
        "panel_genes": 313,
        "notes": "hBreast_320g panel, GT available",
    },
    "xenium-breast-tumor-rep2": {
        "tissue": "breast",
        "models": ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
        "panel_genes": 313,
        "notes": "hBreast_320g panel, GT available",
    },
    "xenium-breast-cancer-prime": {
        "tissue": "breast",
        "models": ["Cells_Adult_Breast.pkl", "Immune_All_Low.pkl"],
        "panel_genes": 5101,
        "notes": "5K panel → fine-grained immune possible",
    },
    "merscope-breast": {
        "tissue": "breast",
        "models": ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
        "panel_genes": 500,
        "notes": "MERSCOPE 500g, weak epithelial markers (EPCAM/CDH1 only)",
    },
    # ── Lung ────────────────────────────────────────────────────────
    "xenium-lung-2fov": {
        "tissue": "lung",
        "models": ["Cells_Lung_Airway.pkl", "Immune_All_High.pkl"],
        "panel_genes": 289,
        "notes": "Lung-specific panel, 91% overlap w/ Lung_Airway — best fit!",
    },
    "xenium-lung-cancer": {
        "tissue": "lung",
        "models": ["Human_Lung_Atlas.pkl", "Cells_Lung_Airway.pkl", "Immune_All_High.pkl"],
        "panel_genes": 377,
        "notes": "Multi-tissue panel, 81% HLCA overlap",
    },
    # ── Liver ───────────────────────────────────────────────────────
    "xenium-liver-normal": {
        "tissue": "liver",
        "models": ["Healthy_Human_Liver.pkl", "Immune_All_High.pkl"],
        "panel_genes": 377,
        "notes": "48% overlap but correct liver types (hepatocytes, Kupffer)",
    },
    "xenium-liver-cancer": {
        "tissue": "liver",
        "models": ["Healthy_Human_Liver.pkl", "Immune_All_High.pkl"],
        "panel_genes": 474,
        "notes": "Multi-tissue+addon panel, 50% overlap",
    },
    # ── Kidney ──────────────────────────────────────────────────────
    "xenium-kidney-normal": {
        "tissue": "kidney",
        "models": ["Immune_All_High.pkl"],
        "panel_genes": 377,
        "notes": "No kidney CT model; use immune + BANKSY/scType for broad types",
    },
    "xenium-kidney-cancer": {
        "tissue": "kidney",
        "models": ["Immune_All_High.pkl"],
        "panel_genes": 377,
        "notes": "No kidney CT model",
    },
    # ── Heart ───────────────────────────────────────────────────────
    "xenium-heart-normal": {
        "tissue": "heart",
        "models": ["Healthy_Adult_Heart.pkl", "Immune_All_High.pkl"],
        "panel_genes": 377,
        "notes": "51% overlap, 75 types incl. cardiomyocyte subtypes",
    },
    # ── Colon ───────────────────────────────────────────────────────
    "xenium-colon-cancer": {
        "tissue": "colon",
        "models": ["Cells_Intestinal_Tract.pkl", "Human_Colorectal_Cancer.pkl", "Immune_All_High.pkl"],
        "panel_genes": 325,
        "notes": "Colon-specific panel, 70% overlap w/ IntestinalTract",
    },
    "xenium-colon-normal": {
        "tissue": "colon",
        "models": ["Cells_Intestinal_Tract.pkl", "Immune_All_High.pkl"],
        "panel_genes": 325,
        "notes": "Same panel as colon-cancer",
    },
    "xenium-colorectal-cancer": {
        "tissue": "colorectal",
        "models": ["Human_Colorectal_Cancer.pkl", "Cells_Intestinal_Tract.pkl", "Immune_All_High.pkl"],
        "panel_genes": 480,
        "notes": "Immuno-Oncology panel A, 69% CRC model overlap",
    },
    # ── Skin ────────────────────────────────────────────────────────
    "xenium-skin-normal-sample1": {
        "tissue": "skin",
        "models": ["Adult_Human_Skin.pkl", "Immune_All_High.pkl"],
        "panel_genes": 377,
        "notes": "57% overlap, multi-tissue panel",
    },
    "xenium-skin-normal-sample2": {
        "tissue": "skin",
        "models": ["Adult_Human_Skin.pkl", "Immune_All_High.pkl"],
        "panel_genes": 377,
        "notes": "Same panel as sample1",
    },
    "xenium-skin-prime-ffpe": {
        "tissue": "skin",
        "models": ["Adult_Human_Skin.pkl", "Immune_All_Low.pkl"],
        "panel_genes": 5006,
        "notes": "5K panel → fine-grained immune possible",
    },
    # ── Tonsil/Lymphoid ─────────────────────────────────────────────
    "xenium-tonsil-lymphoid": {
        "tissue": "tonsil",
        "models": ["Cells_Human_Tonsil.pkl", "Immune_All_High.pkl"],
        "panel_genes": 377,
        "notes": "51% overlap, heavily lymphoid tissue",
    },
    "xenium-tonsil-reactive": {
        "tissue": "tonsil",
        "models": ["Cells_Human_Tonsil.pkl", "Immune_All_High.pkl"],
        "panel_genes": 377,
        "notes": "Same panel, 1.3M cells (largest dataset)",
    },
    "xenium-lymph-node-normal": {
        "tissue": "lymph_node",
        "models": ["Cells_Human_Tonsil.pkl", "Immune_All_High.pkl"],
        "panel_genes": 377,
        "notes": "Tonsil model is closest lymphoid reference",
    },
    # ── Pancreas ────────────────────────────────────────────────────
    "xenium-pancreas-cancer": {
        "tissue": "pancreas",
        "models": ["Immune_All_High.pkl"],
        "panel_genes": 474,
        "notes": "PancreaticIslet model useless (18% overlap, islet-only). Use immune+BANKSY",
    },
    # ── Brain ───────────────────────────────────────────────────────
    "xenium-brain-gbm": {
        "tissue": "brain",
        "models": ["Immune_All_High.pkl"],
        "panel_genes": 480,
        "notes": "No adult human brain CT model; immuno-onc panel B",
    },
    "xenium-mouse-brain": {
        "tissue": "mouse_brain",
        "models": ["Mouse_Isocortex_Hippocampus.pkl", "Mouse_Whole_Brain.pkl"],
        "panel_genes": 248,
        "notes": "Mouse genes, 97% overlap w/ Isocortex model",
    },
    # ── Ovarian ─────────────────────────────────────────────────────
    "xenium-ovarian-cancer": {
        "tissue": "ovary",
        "models": ["Immune_All_Low.pkl", "Immune_All_High.pkl"],
        "panel_genes": 5101,
        "notes": "5K panel, no ovary CT model; fine-grained immune possible",
    },
    "xenium-ovary-cancer-ff": {
        "tissue": "ovary",
        "models": ["Immune_All_High.pkl"],
        "panel_genes": 477,
        "notes": "No ovary CT model",
    },
    # ── Cervical ────────────────────────────────────────────────────
    "xenium-cervical-cancer-prime": {
        "tissue": "cervix",
        "models": ["Immune_All_Low.pkl", "Immune_All_High.pkl"],
        "panel_genes": 5101,
        "notes": "5K panel, no cervix CT model; fine-grained immune possible",
    },
}


def _extract_dataset_name(dataset_name_or_path: str) -> str:
    """Extract a canonical dataset name from a path or name string.

    Handles:
    - Full paths: /mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1/outs/
    - Relative paths: xenium-breast-tumor-rep1/outs
    - Plain names: xenium-breast-tumor-rep1
    - Underscored names: breast_tumor_rep1

    Returns the directory name most likely to match DATASET_MODELS keys.
    """
    from pathlib import Path

    path = Path(dataset_name_or_path)

    # Walk up from the deepest component, skipping generic dirs like 'outs'
    skip_dirs = {"outs", "output", "data", "raw", "processed", "derived"}
    for part in reversed(path.parts):
        if part in skip_dirs or part == "/":
            continue
        # Return the first non-generic directory component
        return part

    return str(path.name or path)


# Tissue keywords to search for in dataset names and paths.
# Ordered longest-first so "lymph_node" matches before "lymph".
_TISSUE_KEYWORDS = [
    ("colorectal", "colorectal"),
    ("lymph_node", "lymph_node"),
    ("lymph-node", "lymph_node"),
    ("mouse_brain", "mouse_brain"),
    ("mouse-brain", "mouse_brain"),
    ("breast", "breast"),
    ("lung", "lung"),
    ("liver", "liver"),
    ("kidney", "kidney"),
    ("heart", "heart"),
    ("colon", "colon"),
    ("skin", "skin"),
    ("tonsil", "tonsil"),
    ("pancreas", "pancreas"),
    ("brain", "brain"),
    ("ovary", "ovary"),
    ("ovarian", "ovary"),
    ("cervix", "cervix"),
    ("cervical", "cervix"),
]


def get_models_for_dataset(dataset_name_or_path: str) -> list[str]:
    """Get the recommended CellTypist models for a dataset.

    Accepts a dataset name, directory name, or full filesystem path.
    Falls back: DATASET_MODELS → tissue keyword match → generic.
    """
    name = _extract_dataset_name(dataset_name_or_path)

    # 1. Direct lookup by canonical name
    if name in DATASET_MODELS:
        return DATASET_MODELS[name]["models"]

    # 2. Try underscore variant (breast_tumor_rep1 → xenium-breast-tumor-rep1)
    dashed = name.replace("_", "-")
    if dashed in DATASET_MODELS:
        return DATASET_MODELS[dashed]["models"]

    # 3. Fuzzy: check if any DATASET_MODELS key is contained in the name
    name_lower = name.lower()
    for key, cfg in DATASET_MODELS.items():
        if key in name_lower or name_lower in key:
            return cfg["models"]

    # 4. Tissue keyword matching from name or full path
    tissue = _detect_tissue_from_string(dataset_name_or_path)
    if tissue in TISSUE_MODELS:
        return TISSUE_MODELS[tissue]

    return TISSUE_MODELS["generic"]


def get_tissue_for_dataset(dataset_name_or_path: str) -> str:
    """Get the tissue type for a dataset.

    Accepts a dataset name, directory name, or full filesystem path.
    Falls back: DATASET_MODELS → tissue keyword match → generic.
    """
    name = _extract_dataset_name(dataset_name_or_path)

    # 1. Direct lookup
    if name in DATASET_MODELS:
        return DATASET_MODELS[name]["tissue"]

    # 2. Underscore variant
    dashed = name.replace("_", "-")
    if dashed in DATASET_MODELS:
        return DATASET_MODELS[dashed]["tissue"]

    # 3. Fuzzy match
    name_lower = name.lower()
    for key, cfg in DATASET_MODELS.items():
        if key in name_lower or name_lower in key:
            return cfg["tissue"]

    # 4. Tissue keyword matching
    return _detect_tissue_from_string(dataset_name_or_path)


def _detect_tissue_from_string(s: str) -> str:
    """Detect tissue type from any string (path, name, description).

    Scans the full string for tissue keywords. Longest match wins.
    """
    s_lower = s.lower().replace("_", "-")
    for keyword, tissue in _TISSUE_KEYWORDS:
        if keyword in s_lower:
            return tissue
    return "generic"

# Expected proportions by tissue (for plausibility checks)
# Expected cell type proportions by tissue for plausibility checks.
# (min, max) ranges allow ~50% tolerance for biological variation.
TISSUE_PROPORTIONS = {
    "breast": {
        "Epithelial": (0.30, 0.70),
        "Immune": (0.10, 0.40),
        "Stromal": (0.10, 0.30),
    },
    "breast_tumor": {
        "Epithelial": (0.30, 0.70),
        "Immune": (0.10, 0.40),
        "Stromal": (0.10, 0.30),
    },
    "lung": {
        "Epithelial": (0.30, 0.60),
        "Immune": (0.15, 0.35),
        "Stromal": (0.10, 0.30),
    },
    "liver": {
        "Epithelial": (0.50, 0.80),  # hepatocytes dominate
        "Immune": (0.05, 0.25),
        "Stromal": (0.05, 0.20),
    },
    "kidney": {
        "Epithelial": (0.50, 0.80),  # tubular epithelium
        "Immune": (0.05, 0.20),
        "Stromal": (0.10, 0.30),
    },
    "heart": {
        "Epithelial": (0.00, 0.10),  # minimal epithelial in heart
        "Immune": (0.05, 0.20),
        "Stromal": (0.40, 0.80),  # cardiomyocytes + fibroblasts + endothelial
    },
    "colon": {
        "Epithelial": (0.40, 0.70),
        "Immune": (0.15, 0.35),
        "Stromal": (0.10, 0.25),
    },
    "colorectal": {
        "Epithelial": (0.30, 0.60),  # may have more immune in tumor
        "Immune": (0.15, 0.40),
        "Stromal": (0.10, 0.30),
    },
    "skin": {
        "Epithelial": (0.40, 0.80),  # keratinocytes dominate
        "Immune": (0.05, 0.25),
        "Stromal": (0.10, 0.30),
    },
    "tonsil": {
        "Epithelial": (0.00, 0.15),
        "Immune": (0.60, 0.90),  # lymphoid tissue
        "Stromal": (0.05, 0.20),
    },
    "lymph_node": {
        "Epithelial": (0.00, 0.10),
        "Immune": (0.70, 0.95),
        "Stromal": (0.05, 0.20),
    },
    "pancreas": {
        "Epithelial": (0.50, 0.80),  # acinar + ductal + islet
        "Immune": (0.05, 0.25),
        "Stromal": (0.10, 0.25),
    },
    "brain": {
        "Epithelial": (0.00, 0.05),  # essentially none
        "Immune": (0.05, 0.30),  # microglia + infiltrating
        "Stromal": (0.20, 0.50),  # endothelial + pericytes
    },
    "ovary": {
        "Epithelial": (0.30, 0.60),
        "Immune": (0.10, 0.35),
        "Stromal": (0.15, 0.40),
    },
    "cervix": {
        "Epithelial": (0.40, 0.70),
        "Immune": (0.10, 0.30),
        "Stromal": (0.10, 0.30),
    },
    "generic": {
        "Epithelial": (0.20, 0.70),
        "Immune": (0.05, 0.40),
        "Stromal": (0.10, 0.40),
    },
}

# Canonical marker genes for validation
CANONICAL_MARKERS = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "MUC1"],
    "Immune": ["PTPRC", "CD3D", "CD3E", "CD19", "MS4A1", "CD68", "CD163", "NKG7"],
    "Stromal": ["COL1A1", "COL1A2", "DCN", "LUM", "PECAM1", "VWF", "ACTA2"],
    "T_cell": ["CD3D", "CD3E", "CD4", "CD8A", "CD8B"],
    "B_cell": ["CD19", "MS4A1", "CD79A", "CD79B"],
    "Macrophage": ["CD68", "CD163", "CSF1R", "MARCO"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "FAP"],
}

# Pattern-based mapping to broad categories
BROAD_CATEGORY_PATTERNS = {
    "Epithelial": [
        "epithelial", "keratinocyte", "hepatocyte", "ductal", "luminal",
        "basal", "alveolar", "secretory", "goblet", "enterocyte", "colonocyte",
        "cholangiocyte", "acinar", "ciliated", "club", "tuft", "paneth",
        "absorptive", "transit", "stem", "progenitor", "myoepithelial",
        "lumm", "lums",  # CellTypist breast
    ],
    "Immune": [
        "t cell", "b cell", "nk cell", "macrophage", "monocyte", "dendritic",
        "mast", "neutrophil", "eosinophil", "plasma", "lymphocyte", "immune",
        "myeloid", "cd4", "cd8", "treg", "th1", "th2", "memory", "naive",
        "effector", "regulatory", "helper", "cytotoxic", "innate", "adaptive",
        "granulocyte", "basophil", "langerhans", "microglia", "kupffer",
        "follicular", "germinal", "marginal zone", "plasmablast",
    ],
    "Stromal": [
        "fibroblast", "endothelial", "pericyte", "smooth muscle", "stromal",
        "vascular", "mesenchymal", "adipocyte", "stellate", "myofibroblast",
        "caf", "cancer-associated", "lymphatic", "blood vessel", "capillary",
        "venous", "arterial", "tip cell", "stalk cell",
    ],
}


def map_to_broad(cell_type: str) -> str:
    """Map fine-grained cell type to broad category."""
    cell_type_lower = cell_type.lower()
    for broad, patterns in BROAD_CATEGORY_PATTERNS.items():
        if any(p in cell_type_lower for p in patterns):
            return broad
    return "Other"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ModelScore:
    """Score for a single CellTypist model."""

    model_name: str
    # Confidence metrics
    mean_confidence: float = 0.0
    median_confidence: float = 0.0
    pct_high_confidence: float = 0.0  # % above 0.5
    # Marker validation
    marker_enrichment_score: float = 0.0
    marker_details: dict[str, float] = field(default_factory=dict)
    # Spatial coherence
    spatial_coherence_score: float = 0.0
    # Proportion plausibility
    proportion_score: float = 0.0
    predicted_proportions: dict[str, float] = field(default_factory=dict)
    # Composite
    composite_score: float = 0.0
    # Metadata
    n_cell_types: int = 0
    error: str | None = None


@dataclass
class ConsensusResult:
    """Result from consensus annotation."""

    annotations_df: pl.DataFrame
    model_scores: list[ModelScore]
    best_model: str
    stats: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Marker Gene Validation (AUCell-inspired)
# =============================================================================


def compute_marker_enrichment(
    adata: Any,
    predictions: np.ndarray,
    available_genes: set[str],
) -> tuple[float, dict[str, float]]:
    """Compute marker gene enrichment score (AUCell-inspired).

    For each predicted cell type, check if marker genes are enriched.
    Uses ranking-based approach for robustness to normalization.

    Returns:
        (enrichment_score, details_dict)
    """
    import numpy as np

    # Map predictions to broad categories
    broad_preds = np.array([map_to_broad(p) for p in predictions])

    enrichments = []
    details = {}

    for broad_cat in ["Epithelial", "Immune", "Stromal"]:
        markers = CANONICAL_MARKERS.get(broad_cat, [])
        present_markers = [m for m in markers if m in available_genes]

        if not present_markers:
            continue

        # Get cells predicted as this type
        mask = broad_preds == broad_cat
        if mask.sum() < 10:
            continue

        for marker in present_markers:
            try:
                # Get expression
                if hasattr(adata.X, "toarray"):
                    expr = adata[:, marker].X.toarray().flatten()
                else:
                    expr = np.array(adata[:, marker].X).flatten()

                expr_in_type = expr[mask]
                expr_in_others = expr[~mask]

                # Fold change with pseudocount
                fc = (np.mean(expr_in_type) + 0.1) / (np.mean(expr_in_others) + 0.1)
                enrichments.append(fc)
                details[f"{broad_cat}_{marker}"] = float(fc)

            except Exception:
                continue

    if not enrichments:
        return 50.0, details  # Neutral score if no markers

    # Score: % of markers with >1.5x enrichment
    good_enrichment = np.mean([e > 1.5 for e in enrichments]) * 100
    mean_fc = np.mean(enrichments)

    # Combine
    score = 0.7 * good_enrichment + 0.3 * min(mean_fc * 10, 100)

    return float(score), details


def compute_spatial_coherence(
    adata: Any,
    predictions: np.ndarray,
    n_neighbors: int = 10,
    n_sample: int = 5000,
) -> float:
    """Compute spatial coherence score.

    Cells of the same type should cluster spatially.
    """
    from scipy.spatial import cKDTree

    # Find spatial coordinates
    x_col = y_col = None
    for xc, yc in [("x_centroid", "y_centroid"), ("X", "Y"), ("centroid_x", "centroid_y")]:
        if xc in adata.obs.columns:
            x_col, y_col = xc, yc
            break

    if x_col is None:
        return 50.0  # Neutral if no spatial data

    try:
        coords = adata.obs[[x_col, y_col]].values
        tree = cKDTree(coords)

        # Map to broad categories
        broad_preds = np.array([map_to_broad(p) for p in predictions])

        # Sample for speed
        n_sample = min(n_sample, len(adata))
        sample_idx = np.random.choice(len(adata), n_sample, replace=False)

        same_type_fractions = []
        for idx in sample_idx:
            _, neighbor_idx = tree.query(coords[idx], k=n_neighbors + 1)
            neighbor_idx = neighbor_idx[1:]  # Exclude self
            same_type = np.mean(broad_preds[neighbor_idx] == broad_preds[idx])
            same_type_fractions.append(same_type)

        # Compare to random
        n_categories = len(np.unique(broad_preds))
        random_baseline = 1.0 / max(n_categories, 1)
        spatial_enrichment = np.mean(same_type_fractions) / max(random_baseline, 0.01)

        # Convert to 0-100 score
        score = min(spatial_enrichment * 25, 100)
        return float(score)

    except Exception as e:
        logger.warning(f"Spatial coherence failed: {e}")
        return 50.0


def compute_proportion_score(
    predictions: np.ndarray,
    expected_proportions: dict[str, tuple[float, float]],
) -> float:
    """Score based on biological plausibility of proportions."""
    broad_preds = np.array([map_to_broad(p) for p in predictions])

    # Compute observed proportions
    unique, counts = np.unique(broad_preds, return_counts=True)
    observed = dict(zip(unique, counts / len(broad_preds)))

    scores = []
    for cell_type, (low, high) in expected_proportions.items():
        obs_prop = observed.get(cell_type, 0)

        if low <= obs_prop <= high:
            scores.append(100)
        elif obs_prop < low:
            distance = low - obs_prop
            scores.append(max(0, 100 - distance * 200))
        else:
            distance = obs_prop - high
            scores.append(max(0, 100 - distance * 200))

    return float(np.mean(scores)) if scores else 50.0


# =============================================================================
# Model Scoring
# =============================================================================


def score_model(
    adata: Any,
    model_name: str,
    expected_proportions: dict[str, tuple[float, float]],
) -> ModelScore:
    """Score a single CellTypist model on quality metrics."""
    import celltypist
    from celltypist import models

    try:
        import scanpy as sc

        # Load model
        model = models.Model.load(model_name)

        # Normalize for CellTypist
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)

        # Run prediction
        result = celltypist.annotate(
            adata_norm,
            model=model,
            majority_voting=True,
        )

        predictions = result.predicted_labels["majority_voting"].values
        prob_matrix = result.probability_matrix
        max_conf = prob_matrix.max(axis=1).values

        # Confidence scores
        mean_conf = float(np.mean(max_conf))
        median_conf = float(np.median(max_conf))
        pct_high = float(np.mean(max_conf > 0.5) * 100)

        # Marker enrichment
        available_genes = set(adata.var_names)
        marker_score, marker_details = compute_marker_enrichment(
            adata, predictions, available_genes
        )

        # Spatial coherence
        spatial_score = compute_spatial_coherence(adata, predictions)

        # Proportion plausibility
        proportion_score = compute_proportion_score(predictions, expected_proportions)

        # Predicted proportions
        broad_preds = np.array([map_to_broad(p) for p in predictions])
        unique, counts = np.unique(broad_preds, return_counts=True)
        pred_proportions = {k: float(v / len(broad_preds)) for k, v in zip(unique, counts)}

        # Composite score (weighted by empirical importance)
        composite = (
            0.40 * pct_high +  # Confidence (strongest predictor)
            0.30 * marker_score +  # Marker enrichment
            0.15 * spatial_score +  # Spatial coherence
            0.15 * proportion_score  # Proportion plausibility
        )

        return ModelScore(
            model_name=model_name,
            mean_confidence=mean_conf,
            median_confidence=median_conf,
            pct_high_confidence=pct_high,
            marker_enrichment_score=marker_score,
            marker_details=marker_details,
            spatial_coherence_score=spatial_score,
            proportion_score=proportion_score,
            predicted_proportions=pred_proportions,
            composite_score=composite,
            n_cell_types=len(np.unique(predictions)),
        )

    except Exception as e:
        logger.warning(f"Failed to score {model_name}: {e}")
        return ModelScore(model_name=model_name, error=str(e))


# =============================================================================
# AutoModelSelector Class
# =============================================================================


class AutoModelSelector:
    """Automatic CellTypist model selection without ground truth.

    Uses confidence scores, marker enrichment, and spatial coherence
    to select the best model(s) for a given tissue type.

    Example:
        selector = AutoModelSelector(tissue_type="breast_tumor")

        # Select best models
        models = selector.select_models(adata, n_models=3)

        # Build consensus
        result = selector.build_consensus(adata, models=models)
    """

    def __init__(
        self,
        tissue_type: str = "generic",
        candidate_models: list[str] | None = None,
        sample_size: int = 20000,
        n_workers: int = 4,
    ):
        """Initialize the model selector.

        Args:
            tissue_type: One of the predefined tissue types
            candidate_models: Override default candidates
            sample_size: Sample size for scoring (for speed)
            n_workers: Number of parallel workers
        """
        self.tissue_type = tissue_type
        self.sample_size = sample_size
        self.n_workers = n_workers

        if candidate_models:
            self.candidate_models = candidate_models
        else:
            self.candidate_models = TISSUE_MODELS.get(
                tissue_type, TISSUE_MODELS["generic"]
            )

        self.expected_proportions = TISSUE_PROPORTIONS.get(
            tissue_type, TISSUE_PROPORTIONS["generic"]
        )

    def select_models(
        self,
        adata: Any,
        n_models: int = 3,
    ) -> list[ModelScore]:
        """Select the best models for the given data.

        Args:
            adata: AnnData object with expression data
            n_models: Number of models to select

        Returns:
            List of ModelScore objects, sorted by composite_score
        """
        import scanpy as sc

        logger.info(f"AutoModelSelector: Scoring {len(self.candidate_models)} models")
        logger.info(f"Tissue type: {self.tissue_type}")

        # Sample if needed
        if len(adata) > self.sample_size:
            logger.info(f"Sampling {self.sample_size:,} cells from {len(adata):,}")
            sample_idx = np.random.choice(len(adata), self.sample_size, replace=False)
            adata_sample = adata[sample_idx].copy()
        else:
            adata_sample = adata

        # Score all models
        scores = []
        for i, model_name in enumerate(self.candidate_models):
            logger.info(f"[{i+1}/{len(self.candidate_models)}] Scoring {model_name}...")
            score = score_model(adata_sample, model_name, self.expected_proportions)

            if score.error is None:
                scores.append(score)
                logger.info(
                    f"  → Composite: {score.composite_score:.1f} | "
                    f"Conf: {score.pct_high_confidence:.1f}% | "
                    f"Markers: {score.marker_enrichment_score:.1f}"
                )
            else:
                logger.warning(f"  → Failed: {score.error}")

        # Sort by composite score
        scores.sort(key=lambda x: x.composite_score, reverse=True)

        # Log results
        logger.info("\n=== Model Selection Results ===")
        for i, s in enumerate(scores[:n_models], 1):
            logger.info(f"{i}. {s.model_name}: {s.composite_score:.1f}")

        return scores[:n_models]

    def build_consensus(
        self,
        adata: Any,
        models: list[ModelScore] | list[str] | None = None,
        min_agreement: float = 0.5,
        confidence_weight: bool = True,
    ) -> ConsensusResult:
        """Build consensus annotation from multiple models.

        Args:
            adata: AnnData object with expression data
            models: Models to use (from select_models or model names)
            min_agreement: Minimum agreement threshold
            confidence_weight: Weight votes by model confidence

        Returns:
            ConsensusResult with annotations and statistics
        """
        import celltypist
        from celltypist import models as ct_models
        import scanpy as sc

        # Determine models to use
        if models is None:
            model_names = self.candidate_models[:5]
        elif isinstance(models[0], ModelScore):
            model_names = [m.model_name for m in models]
        else:
            model_names = models

        logger.info(f"Building consensus from {len(model_names)} models: {model_names}")

        # Normalize once
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)

        # Run all models
        results = []
        for model_name in model_names:
            try:
                model = ct_models.Model.load(model_name)
                pred = celltypist.annotate(
                    adata_norm, model=model, majority_voting=True
                )
                predictions = pred.predicted_labels["majority_voting"].values
                confidence = pred.probability_matrix.max(axis=1).values
                broad_preds = np.array([map_to_broad(p) for p in predictions])

                results.append({
                    "model": model_name,
                    "predictions": predictions,
                    "broad": broad_preds,
                    "confidence": confidence,
                })
                logger.info(f"  ✓ {model_name}")
            except Exception as e:
                logger.warning(f"  ✗ {model_name}: {e}")

        if len(results) < 2:
            raise ValueError("Need at least 2 successful models for consensus")

        # Build voting matrix
        n_cells = len(adata)
        broad_categories = ["Epithelial", "Immune", "Stromal", "Other"]
        votes = np.zeros((n_cells, len(broad_categories)))
        confidence_sum = np.zeros((n_cells, len(broad_categories)))

        for r in results:
            for i, cat in enumerate(broad_categories):
                mask = r["broad"] == cat
                if confidence_weight:
                    votes[mask, i] += r["confidence"][mask]
                    confidence_sum[mask, i] += r["confidence"][mask]
                else:
                    votes[mask, i] += 1
                    confidence_sum[mask, i] += r["confidence"][mask]

        # Normalize
        if confidence_weight:
            total_conf = sum(r["confidence"] for r in results)
            votes_normalized = votes / (total_conf[:, np.newaxis] + 1e-10)
        else:
            votes_normalized = votes / len(results)

        # Get consensus
        consensus_idx = np.argmax(votes_normalized, axis=1)
        consensus_broad = np.array([broad_categories[i] for i in consensus_idx])
        consensus_score = np.max(votes_normalized, axis=1)

        # Count raw agreement
        raw_votes = np.zeros((n_cells, len(broad_categories)))
        for r in results:
            for i, cat in enumerate(broad_categories):
                raw_votes[r["broad"] == cat, i] += 1
        n_models_agree = np.max(raw_votes, axis=1).astype(int)

        # Average confidence of agreeing models
        consensus_confidence = np.zeros(n_cells)
        for i in range(n_cells):
            cat = consensus_broad[i]
            cat_idx = broad_categories.index(cat)
            if raw_votes[i, cat_idx] > 0:
                consensus_confidence[i] = confidence_sum[i, cat_idx] / raw_votes[i, cat_idx]

        # Fine-grained consensus
        fine_predictions = []
        for i in range(n_cells):
            cat = consensus_broad[i]
            fine_votes = {}
            for r in results:
                if r["broad"][i] == cat:
                    fine = r["predictions"][i]
                    weight = r["confidence"][i] if confidence_weight else 1
                    fine_votes[fine] = fine_votes.get(fine, 0) + weight
            if fine_votes:
                fine_predictions.append(max(fine_votes, key=fine_votes.get))
            else:
                fine_predictions.append("Unknown")

        # Build result DataFrame
        annotations_df = pl.DataFrame({
            "cell_id": range(len(adata)),
            "consensus_broad": consensus_broad,
            "consensus_fine": fine_predictions,
            "consensus_score": consensus_score,
            "consensus_confidence": consensus_confidence,
            "n_models_agree": n_models_agree,
            "is_high_confidence": (consensus_score >= min_agreement) &
                                   (consensus_confidence >= 0.3),
        })

        # Add per-model predictions
        for r in results:
            model_short = r["model"].replace(".pkl", "")
            # Convert to list/array to handle Categorical types from CellTypist
            preds = list(r["predictions"]) if hasattr(r["predictions"], "__iter__") else r["predictions"]
            broad = list(r["broad"]) if hasattr(r["broad"], "__iter__") else r["broad"]
            annotations_df = annotations_df.with_columns([
                pl.Series(f"{model_short}_pred", preds),
                pl.Series(f"{model_short}_broad", broad),
                pl.Series(f"{model_short}_conf", r["confidence"]),
            ])

        # Statistics
        n_high_conf = (annotations_df["is_high_confidence"]).sum()
        stats = {
            "n_cells": n_cells,
            "n_models": len(results),
            "n_high_confidence": n_high_conf,
            "pct_high_confidence": n_high_conf / n_cells * 100,
            "broad_distribution": dict(zip(
                *np.unique(consensus_broad, return_counts=True)
            )),
        }

        # Get model scores if we have them
        model_scores = []
        if isinstance(models, list) and len(models) > 0 and isinstance(models[0], ModelScore):
            model_scores = models

        return ConsensusResult(
            annotations_df=annotations_df,
            model_scores=model_scores,
            best_model=model_names[0] if model_names else "",
            stats=stats,
        )


    def build_tiered_consensus(
        self,
        adata: Any,
        models: list[str] | None = None,
        min_agreement: float = 0.5,
    ) -> ConsensusResult:
        """Build tiered consensus: tissue model primary, immune model for refinement.

        Strategy:
        1. Use tissue model (first) as primary: its fine-grained labels already
           contain stromal/endothelial types that map correctly to broad categories.
        2. For cells the tissue model calls "Immune", let the immune model refine
           the subtype if it has higher confidence.
        3. For low-confidence primary predictions, allow secondary model to
           override if its confidence is significantly higher.
        4. Boost confidence when models agree on broad, penalize when they disagree.

        Args:
            adata: AnnData object with expression data
            models: Model names (first = tissue-specific, rest = immune/refinement)
            min_agreement: Minimum agreement for high-confidence flag

        Returns:
            ConsensusResult with tiered annotations
        """
        import celltypist
        from celltypist import models as ct_models
        import scanpy as sc

        model_names = models or self.candidate_models[:5]
        if len(model_names) < 2:
            return self.build_consensus(adata, models=model_names)

        logger.info(f"Tiered consensus: primary={model_names[0]}, "
                     f"refinement={model_names[1:]}")

        # Normalize once
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)

        # Run all models
        results = []
        for model_name in model_names:
            try:
                model = ct_models.Model.load(model_name)
                pred = celltypist.annotate(
                    adata_norm, model=model, majority_voting=True
                )
                predictions = pred.predicted_labels["majority_voting"].values
                confidence = pred.probability_matrix.max(axis=1).values
                broad_preds = np.array([map_to_broad(p) for p in predictions])

                results.append({
                    "model": model_name,
                    "predictions": predictions,
                    "broad": broad_preds,
                    "confidence": confidence,
                })
                logger.info(f"  ✓ {model_name}")
            except Exception as e:
                logger.warning(f"  ✗ {model_name}: {e}")

        if not results:
            raise ValueError("No models succeeded")
        if len(results) == 1:
            return self.build_consensus(adata, models=[results[0]["model"]])

        primary = results[0]
        secondary = results[1:]
        n_cells = len(adata)

        # Start with primary model's predictions
        consensus_broad = primary["broad"].copy()
        consensus_fine = list(primary["predictions"])
        consensus_confidence = primary["confidence"].copy()
        n_models_agree = np.ones(n_cells, dtype=int)

        for sec in secondary:
            agrees = primary["broad"] == sec["broad"]
            n_models_agree[agrees] += 1

        # ── Override rules ──
        for sec in secondary:
            for i in range(n_cells):
                pri_broad = primary["broad"][i]
                sec_broad = sec["broad"][i]
                pri_conf = primary["confidence"][i]
                sec_conf = sec["confidence"][i]

                # Rule 1: Both agree on immune → use higher-confidence fine label
                if pri_broad == "Immune" and sec_broad == "Immune":
                    if sec_conf > pri_conf:
                        consensus_fine[i] = sec["predictions"][i]
                        consensus_confidence[i] = (pri_conf + sec_conf) / 2

                # Rule 2: Primary low-confidence, secondary high → override
                elif pri_conf < 0.3 and sec_conf > 0.5:
                    consensus_broad[i] = sec_broad
                    consensus_fine[i] = sec["predictions"][i]
                    consensus_confidence[i] = sec_conf * 0.8  # Slight penalty

                # Rule 3: Agreement on broad → boost confidence
                elif pri_broad == sec_broad:
                    consensus_confidence[i] = (pri_conf + sec_conf) / 2

                # Rule 4: Disagreement, both low-conf → flag low confidence
                elif pri_conf < 0.4 and sec_conf < 0.4:
                    consensus_confidence[i] = min(pri_conf, sec_conf) * 0.5

        # Re-derive broad from fine (in case overrides changed things)
        consensus_broad = np.array([map_to_broad(f) for f in consensus_fine])

        # Consensus score
        total_models = len(results)
        consensus_score = n_models_agree / total_models

        # Medium-grained
        medium_predictions = [_map_to_medium(f, b) for f, b in zip(consensus_fine, consensus_broad)]

        # Build DataFrame
        annotations_df = pl.DataFrame({
            "cell_id": range(n_cells),
            "consensus_broad": consensus_broad.tolist(),
            "consensus_medium": medium_predictions,
            "consensus_fine": consensus_fine,
            "consensus_score": consensus_score,
            "consensus_confidence": consensus_confidence,
            "n_models_agree": n_models_agree,
            "is_high_confidence": (consensus_score >= min_agreement) &
                                   (consensus_confidence >= 0.3),
        })

        for r in results:
            model_short = r["model"].replace(".pkl", "")
            preds = list(r["predictions"]) if hasattr(r["predictions"], "__iter__") else r["predictions"]
            broad = list(r["broad"]) if hasattr(r["broad"], "__iter__") else r["broad"]
            annotations_df = annotations_df.with_columns([
                pl.Series(f"{model_short}_pred", preds),
                pl.Series(f"{model_short}_broad", broad),
                pl.Series(f"{model_short}_conf", r["confidence"]),
            ])

        n_high_conf = int(annotations_df["is_high_confidence"].sum())
        stats = {
            "n_cells": n_cells,
            "n_models": total_models,
            "n_high_confidence": n_high_conf,
            "pct_high_confidence": n_high_conf / n_cells * 100,
            "mean_agreement": float(n_models_agree.mean()),
            "strategy": "tiered",
            "broad_distribution": dict(zip(
                *np.unique(consensus_broad, return_counts=True)
            )),
        }

        return ConsensusResult(
            annotations_df=annotations_df,
            model_scores=[],
            best_model=model_names[0],
            stats=stats,
        )


# =============================================================================
# Medium-Grained Category Mapping
# =============================================================================

# Maps fine-grained CellTypist labels → ~10-15 medium categories
_MEDIUM_PATTERNS = {
    # Epithelial subtypes
    "Epithelial_Luminal": [
        "lumm", "lums", "lumsec", "luminal", "secretory", "club",
        "ductal", "duct", "goblet", "mucus", "serous",
        "at2", "alveolar type 2", "at2 proliferating",
        "colonocyte", "enterocyte", "absorptive",
        "hepatocyte",
    ],
    "Epithelial_Basal": [
        "basal", "myoepi", "myoepithelial", "suprabasal",
        "undifferentiated_kc", "keratinocyte", "krt15",
    ],
    "Epithelial_Ciliated": [
        "ciliated", "multiciliated", "deuterosomal",
    ],
    "Epithelial_Other": [
        "at1", "alveolar type 1",
        "ionocyte", "tuft", "paneth", "enteroendocrine",
        "enterochromaffin", "hillock", "crypt",
        "best4", "fdcsp", "squamous",
    ],
    # Immune subtypes
    "T_Cell": [
        "cd4", "cd8", "t cell", "t_cell", "treg", "t_prol",
        "th1", "th2", "th17", "tfh", "tfr", "mait", "nkt",
        "gamma-delta", "gd ", "cytotoxic t", "helper t",
        "memory t", "naive t", "effector t", "t-eff", "t-trans",
        "sell+", "tcm", "tem", "trm", "tpex", "t(agonist)",
        "dn ", "activated t", "activated cd",
    ],
    "B_Cell": [
        "b cell", "b_cell", "b_naive", "bmem", "plasma",
        "naive b", "memory b", "follicular b", "gc b",
        "germinal", "dz ", "lz ", "mbc", "nbc",
        "iga plasma", "igg plasma", "igm plasma", "igd",
        "cycling b", "immature b", "age-associated b",
        "pre-b", "precursor",
    ],
    "NK_Cell": [
        "nk cell", "nk ", "nk-", "natural killer",
        "cd16+ nk", "cd16- nk", "cd16-cd56",
        "ilc1_nk", "ilc", "innate lymphoid",
    ],
    "Macrophage": [
        "macro", "macrophage", "monocyte", "mono ",
        "kupffer", "hofbauer", "microglia",
        "myeloid", "mnp", "myelocyte",
        "inf_mac", "lyve1+", "mmp9+", "erythrophagocytic",
    ],
    "Dendritic_Cell": [
        "dendritic", "dc", "cdc", "pdc", "mdc",
        "langerhans", "lc ", "migdc", "miglc", "modc",
        "adc3", "c1q slan", "irf7",
    ],
    "Mast_Cell": [
        "mast", "basophil", "granulocyte", "neutrophil",
        "eosinophil", "clc+ mast",
    ],
    # Stromal subtypes
    "Fibroblast": [
        "fibro", "fibroblast", "myofibroblast", "caf",
        "stromal 1", "stromal 2", "stromal 3", "stromal 4",
        "mesenchymal", "mesenchyme", "stellate", "adventitial",
        "f1", "f2", "f3",
    ],
    "Endothelial": [
        "endothelial", "vas-", "vas ", "vascular",
        "lymph-", "lymphatic", "capillary", "arterial", "venous",
        "ec ", "lec", "vec", "art_ec", "cap_ec", "ven_ec",
        "pul_cap", "liver_hsec",
    ],
    "Pericyte_SMC": [
        "pericyte", "pericytes", "vsmc", "smooth muscle", "smc",
        "cap_pc", "liver_pc", "heart_pc",
    ],
}


def _map_to_medium(fine_label: str, broad_cat: str) -> str:
    """Map a fine-grained CellTypist label to medium-grained category."""
    fine_lower = fine_label.lower()
    for medium_cat, patterns in _MEDIUM_PATTERNS.items():
        for p in patterns:
            if p in fine_lower:
                return medium_cat
    # Fallback: use broad category as medium
    return broad_cat


# =============================================================================
# Pipeline Integration
# =============================================================================


def get_auto_selector_config() -> dict[str, Any]:
    """Return configuration schema for auto model selection."""
    return {
        "tissue_type": {
            "type": "string",
            "enum": list(TISSUE_MODELS.keys()),
            "default": "generic",
            "description": "Tissue type for model selection",
        },
        "n_models": {
            "type": "integer",
            "default": 3,
            "minimum": 1,
            "maximum": 10,
            "description": "Number of models for consensus",
        },
        "min_agreement": {
            "type": "number",
            "default": 0.5,
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Minimum agreement for high-confidence",
        },
        "sample_size": {
            "type": "integer",
            "default": 20000,
            "description": "Sample size for model scoring",
        },
    }
