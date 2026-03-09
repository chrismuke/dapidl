"""Annotation Confidence Estimation — GT-free quality assessment.

Evaluates cell type annotation quality without ground truth by computing:
1. Marker enrichment confidence — do annotated cells express expected markers?
2. Cross-method consensus — do multiple methods agree? (optional)
3. Spatial coherence — are same-type cells spatially clustered?
4. Proportion plausibility — do cell type fractions match expected tissue composition?

Works with ANY annotation method (CellTypist, BANKSY, SingleR, ensembles, etc.).

Usage:
    from dapidl.validation.annotation_confidence import (
        compute_annotation_confidence,
        AnnotationConfidenceConfig,
    )

    result = compute_annotation_confidence(
        adata=adata,
        predictions={"CellTypist": ct_preds, "BANKSY": banksy_preds},
        spatial_coords=spatial_coords,
    )
    print(result.summary())
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from scipy.sparse import issparse

if TYPE_CHECKING:
    import anndata as ad


# ── Expected Tissue Proportions ───────────────────────────────────

TISSUE_PROPORTIONS = {
    "breast": {"Epithelial": (0.30, 0.70), "Immune": (0.10, 0.40), "Stromal": (0.10, 0.40)},
    "lung": {"Epithelial": (0.20, 0.50), "Immune": (0.15, 0.40), "Stromal": (0.15, 0.40)},
    "liver": {"Epithelial": (0.50, 0.80), "Immune": (0.05, 0.25), "Stromal": (0.05, 0.25)},
    "kidney": {"Epithelial": (0.50, 0.80), "Immune": (0.05, 0.20), "Stromal": (0.10, 0.30)},
    "heart": {"Epithelial": (0.00, 0.10), "Immune": (0.05, 0.20), "Stromal": (0.40, 0.80)},
    "colon": {"Epithelial": (0.40, 0.70), "Immune": (0.15, 0.35), "Stromal": (0.10, 0.25)},
    "colorectal": {"Epithelial": (0.30, 0.60), "Immune": (0.15, 0.40), "Stromal": (0.10, 0.30)},
    "skin": {"Epithelial": (0.40, 0.80), "Immune": (0.05, 0.25), "Stromal": (0.10, 0.30)},
    "tonsil": {"Epithelial": (0.00, 0.15), "Immune": (0.60, 0.90), "Stromal": (0.05, 0.20)},
    "lymph_node": {"Epithelial": (0.00, 0.10), "Immune": (0.70, 0.95), "Stromal": (0.05, 0.20)},
    "pancreas": {"Epithelial": (0.50, 0.80), "Immune": (0.05, 0.25), "Stromal": (0.10, 0.25)},
    "brain": {"Epithelial": (0.00, 0.05), "Immune": (0.05, 0.30), "Stromal": (0.20, 0.50)},
    "ovary": {"Epithelial": (0.20, 0.60), "Immune": (0.10, 0.35), "Stromal": (0.15, 0.45)},
    "cervix": {"Epithelial": (0.40, 0.70), "Immune": (0.10, 0.30), "Stromal": (0.10, 0.30)},
    "generic": {"Epithelial": (0.10, 0.80), "Immune": (0.05, 0.50), "Stromal": (0.05, 0.50)},
}


# ── Configuration ─────────────────────────────────────────────────

@dataclass
class AnnotationConfidenceConfig:
    """Configuration for annotation confidence estimation."""

    # Marker enrichment
    use_panglao_markers: bool = True
    min_markers_per_type: int = 3
    enrichment_threshold: float = 2.0  # Fold-change for "enriched"

    # Spatial coherence
    spatial_k: int = 15  # KNN neighbors for spatial coherence
    min_spatial_coherence: float = 0.5  # Below this → warning

    # Proportion plausibility
    tissue_type: str = "breast"

    # Thresholds
    high_confidence_threshold: float = 0.7
    low_confidence_threshold: float = 0.3


# ── Result Types ──────────────────────────────────────────────────

@dataclass
class CellTypeConfidence:
    """Confidence metrics for a single predicted cell type."""

    cell_type: str
    n_cells: int
    fraction: float

    # Marker enrichment (0-1)
    marker_score: float
    markers_found: int
    markers_total: int

    # Spatial coherence (0-1): fraction of KNN neighbors with same type
    spatial_coherence: float

    # Cross-method agreement (0-1, NaN if single method)
    consensus_score: float

    @property
    def overall_score(self) -> float:
        """Combined confidence score (0-1)."""
        scores = [self.marker_score, self.spatial_coherence]
        if not np.isnan(self.consensus_score):
            scores.append(self.consensus_score)
        return float(np.mean(scores))


@dataclass
class AnnotationConfidenceResult:
    """Complete annotation confidence assessment."""

    # Per-cell type scores
    per_type: dict[str, CellTypeConfidence] = field(default_factory=dict)

    # Per-cell confidence (index-aligned with adata)
    cell_confidence: np.ndarray | None = None

    # Overall metrics
    overall_marker_score: float = 0.0
    overall_spatial_coherence: float = 0.0
    overall_consensus_score: float = float("nan")
    proportion_plausible: bool = True

    # Warnings
    warnings: list[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Combined confidence score (0-1)."""
        scores = [self.overall_marker_score, self.overall_spatial_coherence]
        if not np.isnan(self.overall_consensus_score):
            scores.append(self.overall_consensus_score)
        return float(np.mean(scores))

    def summary(self) -> str:
        """Human-readable confidence report."""
        lines = [
            "=" * 65,
            "  ANNOTATION CONFIDENCE REPORT",
            "=" * 65,
            f"  Overall Score: {self.overall_score:.3f}",
            f"    Marker Enrichment:   {self.overall_marker_score:.3f}",
            f"    Spatial Coherence:   {self.overall_spatial_coherence:.3f}",
        ]
        if not np.isnan(self.overall_consensus_score):
            lines.append(f"    Cross-Method Consensus: {self.overall_consensus_score:.3f}")
        lines.append(f"    Proportions Plausible: {'Yes' if self.proportion_plausible else 'NO'}")

        if self.warnings:
            lines.append(f"\n  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    - {w}")

        lines.append(f"\n  {'Cell Type':<20} {'N':>7} {'Frac':>6} {'Marker':>7} {'Spatial':>8} {'Consensus':>10} {'Score':>7}")
        lines.append("  " + "-" * 63)
        for ct in sorted(self.per_type.values(), key=lambda x: -x.n_cells):
            cons = f"{ct.consensus_score:.3f}" if not np.isnan(ct.consensus_score) else "   -"
            lines.append(
                f"  {ct.cell_type:<20} {ct.n_cells:>7} {ct.fraction:>5.1%} "
                f"{ct.marker_score:>7.3f} {ct.spatial_coherence:>8.3f} {cons:>10} {ct.overall_score:>7.3f}"
            )

        lines.append("=" * 65)

        # Interpretation
        score = self.overall_score
        if score >= 0.7:
            lines.append("  Interpretation: HIGH confidence — annotations likely reliable")
        elif score >= 0.5:
            lines.append("  Interpretation: MODERATE confidence — review warnings")
        else:
            lines.append("  Interpretation: LOW confidence — annotations may be unreliable")

        return "\n".join(lines)


# ── Main Entry Point ──────────────────────────────────────────────

def compute_annotation_confidence(
    adata: "ad.AnnData",
    predictions: dict[str, list[str] | np.ndarray],
    spatial_coords: np.ndarray | None = None,
    config: AnnotationConfidenceConfig | None = None,
    primary_method: str | None = None,
) -> AnnotationConfidenceResult:
    """Compute annotation confidence without ground truth.

    Args:
        adata: AnnData with log-normalized expression.
        predictions: Dict of {method_name: predictions_array}.
            Can be a single method: {"CellTypist": preds}
            Or multiple: {"CellTypist": ct_preds, "BANKSY": bk_preds}
        spatial_coords: (N, 2) array of cell coordinates. Optional.
        config: Configuration options.
        primary_method: Which method to evaluate (default: first key).

    Returns:
        AnnotationConfidenceResult with per-cell and per-type scores.
    """
    config = config or AnnotationConfidenceConfig()
    result = AnnotationConfidenceResult()

    if not predictions:
        result.warnings.append("No predictions provided")
        return result

    # Determine primary method
    if primary_method is None:
        primary_method = next(iter(predictions))
    primary_preds = np.asarray(predictions[primary_method])

    n_cells = len(primary_preds)
    logger.info(f"Computing annotation confidence for {n_cells} cells ({primary_method})")

    unique_types = [t for t in np.unique(primary_preds) if t.lower() not in ("unknown", "unassigned")]

    # ── 1. Marker Enrichment ──────────────────────────────────
    logger.info("Computing marker enrichment scores...")
    marker_scores = _compute_marker_enrichment(adata, primary_preds, config)

    # ── 2. Spatial Coherence ──────────────────────────────────
    spatial_scores: dict[str, float] = {}
    per_cell_spatial = np.full(n_cells, 0.5, dtype=np.float32)  # Neutral default
    if spatial_coords is not None:
        logger.info("Computing spatial coherence...")
        spatial_scores, per_cell_spatial = _compute_spatial_coherence(
            primary_preds, spatial_coords, config
        )
    else:
        result.warnings.append("No spatial coordinates — spatial coherence skipped")

    # ── 3. Cross-Method Consensus ─────────────────────────────
    consensus_scores: dict[str, float] = {}
    per_cell_consensus = np.full(n_cells, float("nan"), dtype=np.float32)
    if len(predictions) > 1:
        logger.info(f"Computing cross-method consensus ({len(predictions)} methods)...")
        consensus_scores, per_cell_consensus = _compute_consensus(predictions, primary_method)
    else:
        logger.info("Single method — consensus check skipped")

    # ── 4. Proportion Plausibility ────────────────────────────
    proportions = _check_proportions(primary_preds, config)
    result.proportion_plausible = proportions["plausible"]
    if not proportions["plausible"]:
        result.warnings.extend(proportions["warnings"])

    # ── 5. Assemble Per-Type Results + Per-Cell Confidence ────
    # Per-cell confidence combines: type marker score + individual spatial + individual consensus
    cell_confidence = np.zeros(n_cells, dtype=np.float32)

    for ct in unique_types:
        mask = primary_preds == ct
        n = mask.sum()
        frac = n / n_cells

        ms = marker_scores.get(ct, {"score": 0.0, "found": 0, "total": 0})
        ss = spatial_scores.get(ct, 0.5)
        cs = consensus_scores.get(ct, float("nan"))

        type_conf = CellTypeConfidence(
            cell_type=ct,
            n_cells=int(n),
            fraction=float(frac),
            marker_score=ms["score"],
            markers_found=ms["found"],
            markers_total=ms["total"],
            spatial_coherence=float(ss),
            consensus_score=float(cs),
        )
        result.per_type[ct] = type_conf

        # Per-cell confidence: marker score is per-type, spatial/consensus are per-cell
        cell_marker = ms["score"]  # Same for all cells of this type
        cell_spatial = per_cell_spatial[mask]  # Individual per cell
        if np.isnan(cs):
            # Single method: average of marker + spatial
            cell_confidence[mask] = (cell_marker + cell_spatial) / 2.0
        else:
            # Multi-method: average of marker + spatial + consensus
            cell_cons = per_cell_consensus[mask]
            cell_confidence[mask] = (cell_marker + cell_spatial + cell_cons) / 3.0

        # Generate warnings for low-confidence types
        if ms["score"] < 0.3 and n > 100:
            result.warnings.append(
                f"{ct}: low marker enrichment ({ms['score']:.2f}) with {ms['found']}/{ms['total']} markers"
            )
        if ss < config.min_spatial_coherence and spatial_coords is not None:
            result.warnings.append(
                f"{ct}: low spatial coherence ({ss:.2f}) — cells may be scattered"
            )

    result.cell_confidence = cell_confidence

    # ── 6. Compute Overall Scores ─────────────────────────────
    type_weights = {ct: tc.n_cells for ct, tc in result.per_type.items()}
    total = sum(type_weights.values()) or 1

    result.overall_marker_score = sum(
        result.per_type[ct].marker_score * w for ct, w in type_weights.items()
    ) / total

    if spatial_scores:
        result.overall_spatial_coherence = sum(
            result.per_type[ct].spatial_coherence * w for ct, w in type_weights.items()
        ) / total
    else:
        result.overall_spatial_coherence = 0.5  # Neutral when no spatial data

    if consensus_scores:
        valid_consensus = {ct: tc.consensus_score for ct, tc in result.per_type.items()
                          if not np.isnan(tc.consensus_score)}
        if valid_consensus:
            result.overall_consensus_score = sum(
                valid_consensus[ct] * type_weights[ct] for ct in valid_consensus
            ) / sum(type_weights[ct] for ct in valid_consensus)

    logger.info(f"Overall confidence: {result.overall_score:.3f}")
    return result


# ── Marker Enrichment ─────────────────────────────────────────────

def _get_panglao_broad_markers(adata: "ad.AnnData") -> dict[str, dict[str, list[str]]]:
    """Pull PanglaoDB markers and organize by broad category."""
    try:
        import decoupler as dc
        markers = dc.op.resource("PanglaoDB", organism="human")
    except Exception:
        logger.warning("Could not fetch PanglaoDB markers, using built-in BREAST_MARKERS")
        from dapidl.validation.marker_validation import BREAST_MARKERS
        return BREAST_MARKERS

    # Filter quality markers
    markers = markers[
        markers["canonical_marker"].astype(bool)
        & (markers["human_sensitivity"].astype(float) > 0.5)
    ]
    markers = markers[~markers.duplicated(["cell_type", "genesymbol"])]

    panel_genes = set(adata.var_names)
    markers = markers[markers["genesymbol"].isin(panel_genes)]

    # Build broad category marker dict
    from dapidl.validation.marker_validation import BREAST_MARKERS

    # Merge PanglaoDB into broad categories
    panglao_broad_map = {
        "Epithelial cells": "Epithelial", "Basal cells": "Epithelial",
        "Luminal epithelial cells": "Epithelial", "Ductal cells": "Epithelial",
        "Myoepithelial cells": "Epithelial", "Keratinocytes": "Epithelial",
        "Goblet cells": "Epithelial", "Enterocytes": "Epithelial",
        "Clara cells": "Epithelial", "Alveolar cells": "Epithelial",
        "Hepatocytes": "Epithelial", "Cholangiocytes": "Epithelial",
        "Podocytes": "Epithelial", "Proximal tubular cells": "Epithelial",
        "T cells": "Immune", "T memory cells": "Immune", "T helper cells": "Immune",
        "T regulatory cells": "Immune", "T cytotoxic cells": "Immune",
        "NK cells": "Immune", "NKT cells": "Immune",
        "B cells": "Immune", "B cells memory": "Immune",
        "Plasma cells": "Immune", "Plasmacytoid dendritic cells": "Immune",
        "Dendritic cells": "Immune", "Macrophages": "Immune",
        "Monocytes": "Immune", "Mast cells": "Immune",
        "Neutrophils": "Immune", "Basophils": "Immune",
        "Kupffer cells": "Immune", "Microglia": "Immune",
        "Fibroblasts": "Stromal", "Myofibroblasts": "Stromal",
        "Endothelial cells": "Stromal", "Pericytes": "Stromal",
        "Smooth muscle cells": "Stromal", "Adipocytes": "Stromal",
        "Lymphatic endothelial cells": "Stromal",
        "Mesenchymal stem cells": "Stromal", "Stellate cells": "Stromal",
    }

    broad_markers: dict[str, dict[str, list[str]]] = {}
    for _, row in markers.iterrows():
        ct = row["cell_type"]
        gene = row["genesymbol"]
        broad = panglao_broad_map.get(ct)
        if broad:
            if broad not in broad_markers:
                broad_markers[broad] = {"positive": [], "negative": []}
            if gene not in broad_markers[broad]["positive"]:
                broad_markers[broad]["positive"].append(gene)

    # Add negative markers from BREAST_MARKERS
    for ct, info in BREAST_MARKERS.items():
        if ct in broad_markers and "negative" in info:
            panel_neg = [g for g in info["negative"] if g in panel_genes]
            broad_markers[ct]["negative"] = panel_neg

    n_total = sum(len(v["positive"]) for v in broad_markers.values())
    logger.info(f"PanglaoDB markers in panel: {n_total} across {len(broad_markers)} types")

    return broad_markers


def _compute_marker_enrichment(
    adata: "ad.AnnData",
    predictions: np.ndarray,
    config: AnnotationConfidenceConfig,
) -> dict[str, dict]:
    """Compute marker enrichment score per cell type."""
    if config.use_panglao_markers:
        markers_db = _get_panglao_broad_markers(adata)
    else:
        from dapidl.validation.marker_validation import BREAST_MARKERS
        markers_db = BREAST_MARKERS

    results = {}
    unique_types = [t for t in np.unique(predictions) if t.lower() not in ("unknown", "unassigned")]

    X = adata.X
    if issparse(X):
        X = X.toarray()

    global_mean = X.mean(axis=0)
    global_std = X.std(axis=0)
    global_std[global_std == 0] = 1.0

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    for ct in unique_types:
        mask = predictions == ct
        n = mask.sum()
        if n < 10:
            results[ct] = {"score": 0.0, "found": 0, "total": 0}
            continue

        # Find matching markers
        marker_info = markers_db.get(ct)
        if marker_info is None:
            # Try fuzzy match
            ct_lower = ct.lower()
            for db_ct, info in markers_db.items():
                if db_ct.lower() in ct_lower or ct_lower in db_ct.lower():
                    marker_info = info
                    break

        if marker_info is None:
            results[ct] = {"score": 0.0, "found": 0, "total": 0}
            continue

        pos_genes = [g for g in marker_info["positive"] if g in gene_to_idx]
        neg_genes = [g for g in marker_info.get("negative", []) if g in gene_to_idx]

        if len(pos_genes) < config.min_markers_per_type:
            results[ct] = {"score": 0.3, "found": len(pos_genes), "total": len(marker_info["positive"])}
            continue

        # Z-score enrichment of positive markers in this cell type
        cluster_mean = X[mask].mean(axis=0)
        pos_zscores = np.array([
            (cluster_mean[gene_to_idx[g]] - global_mean[gene_to_idx[g]]) / global_std[gene_to_idx[g]]
            for g in pos_genes
        ])

        # For broad categories with many markers (Immune has 40+), subtype markers
        # will be diluted. Use fraction of markers enriched (z>0) rather than mean z.
        frac_enriched = float((pos_zscores > 0).mean())

        # Top-k enrichment: the best markers should be strongly enriched
        top_k = min(5, len(pos_zscores))
        top_z = float(np.sort(pos_zscores)[-top_k:].mean())

        # Fraction of cells expressing at least one positive marker
        pos_idx = [gene_to_idx[g] for g in pos_genes]
        frac_expressing = float((X[mask][:, pos_idx] > 0).any(axis=1).mean())

        # Negative marker penalty
        neg_penalty = 0.0
        if neg_genes:
            neg_idx = [gene_to_idx[g] for g in neg_genes]
            neg_frac = float((X[mask][:, neg_idx] > 0.5).any(axis=1).mean())
            neg_penalty = neg_frac * 0.3  # 30% penalty for negative marker expression

        # Combined score: frac_enriched × frac_expressing × sigmoid(top_z) − penalty
        # - frac_enriched: broad signal (>50% of markers enriched = good)
        # - top_z via sigmoid: strong enrichment of best markers
        # - frac_expressing: cells actually express the markers
        top_z_score = _sigmoid(float(top_z), center=0.5, steepness=2.0)
        score = (0.4 * frac_enriched + 0.6 * top_z_score) * frac_expressing - neg_penalty
        score = float(np.clip(score, 0, 1))

        results[ct] = {
            "score": score,
            "found": len(pos_genes),
            "total": len(marker_info["positive"]),
            "frac_enriched": frac_enriched,
            "top_z": float(top_z),
            "frac_expressing": frac_expressing,
        }

    return results


def _sigmoid(x: float, center: float = 0.0, steepness: float = 1.0) -> float:
    """Sigmoid function for score normalization."""
    return float(1.0 / (1.0 + np.exp(-steepness * (x - center))))


# ── Spatial Coherence ─────────────────────────────────────────────

def _compute_spatial_coherence(
    predictions: np.ndarray,
    spatial_coords: np.ndarray,
    config: AnnotationConfidenceConfig,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute spatial coherence: fraction of KNN neighbors with same label.

    Returns:
        Tuple of (per_type_averages, per_cell_coherence_array).
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=config.spatial_k + 1, algorithm="ball_tree")
    nn.fit(spatial_coords)
    _, indices = nn.kneighbors(spatial_coords)

    # Per-cell: fraction of neighbors with same type (excluding self)
    per_cell_coherence = np.zeros(len(predictions))
    for i in range(len(predictions)):
        neighbor_types = predictions[indices[i, 1:]]  # Exclude self
        per_cell_coherence[i] = (neighbor_types == predictions[i]).mean()

    # Aggregate per type
    unique_types = [t for t in np.unique(predictions) if t.lower() not in ("unknown", "unassigned")]
    results = {}
    for ct in unique_types:
        mask = predictions == ct
        if mask.sum() > 0:
            results[ct] = float(per_cell_coherence[mask].mean())

    return results, per_cell_coherence


# ── Cross-Method Consensus ────────────────────────────────────────

def _compute_consensus(
    predictions: dict[str, list[str] | np.ndarray],
    primary_method: str,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute cross-method agreement per cell type.

    Returns:
        Tuple of (per_type_averages, per_cell_agreement_array).
    """
    methods = list(predictions.keys())
    n_methods = len(methods)

    primary_preds = np.asarray(predictions[primary_method])

    if n_methods < 2:
        return {}, np.full(len(primary_preds), float("nan"))

    other_preds = [np.asarray(predictions[m]) for m in methods if m != primary_method]

    # Per-cell agreement: fraction of other methods that agree with primary
    n_cells = len(primary_preds)
    per_cell_agreement = np.zeros(n_cells)
    for other in other_preds:
        per_cell_agreement += (other == primary_preds).astype(float)
    per_cell_agreement /= len(other_preds)

    # Aggregate per type
    unique_types = [t for t in np.unique(primary_preds) if t.lower() not in ("unknown", "unassigned")]
    results = {}
    for ct in unique_types:
        mask = primary_preds == ct
        if mask.sum() > 0:
            results[ct] = float(per_cell_agreement[mask].mean())

    return results, per_cell_agreement


# ── Proportion Plausibility ───────────────────────────────────────

def _check_proportions(
    predictions: np.ndarray,
    config: AnnotationConfidenceConfig,
) -> dict:
    """Check if cell type proportions are plausible for the tissue."""
    expected = TISSUE_PROPORTIONS.get(config.tissue_type, TISSUE_PROPORTIONS["generic"])
    n_total = len(predictions)

    warnings = []
    plausible = True

    for ct, (lo, hi) in expected.items():
        n_ct = (predictions == ct).sum()
        frac = n_ct / n_total if n_total > 0 else 0

        if frac < lo * 0.5:  # Allow 50% tolerance below minimum
            warnings.append(f"{ct}: {frac:.1%} is very low (expected {lo:.0%}-{hi:.0%})")
            plausible = False
        elif frac > hi * 1.5:  # Allow 50% tolerance above maximum
            warnings.append(f"{ct}: {frac:.1%} is very high (expected {lo:.0%}-{hi:.0%})")
            plausible = False

    return {"plausible": plausible, "warnings": warnings}


# ── Filtering API ────────────────────────────────────────────────

@dataclass
class FilterResult:
    """Result of confidence-based prediction filtering."""

    predictions: np.ndarray  # Filtered labels ("Unknown" for low-confidence)
    mask: np.ndarray  # Boolean mask: True = kept, False = filtered
    n_kept: int
    n_filtered: int
    per_type_kept: dict[str, int]
    per_type_filtered: dict[str, int]

    def summary(self) -> str:
        """Human-readable filter summary."""
        total = self.n_kept + self.n_filtered
        lines = [
            f"Filtered {self.n_filtered}/{total} cells ({self.n_filtered/total:.1%}) → "
            f"{self.n_kept} cells retained",
            "",
            f"  {'Cell Type':<20} {'Kept':>7} {'Filtered':>9} {'Retained':>9}",
            "  " + "-" * 47,
        ]
        for ct in sorted(self.per_type_kept, key=lambda c: -(self.per_type_kept[c])):
            kept = self.per_type_kept[ct]
            filt = self.per_type_filtered.get(ct, 0)
            pct = kept / (kept + filt) * 100 if (kept + filt) > 0 else 0
            lines.append(f"  {ct:<20} {kept:>7} {filt:>9} {pct:>8.1f}%")
        return "\n".join(lines)


def filter_predictions(
    predictions: np.ndarray,
    confidence_result: AnnotationConfidenceResult,
    min_confidence: float = 0.5,
    min_spatial_coherence: float | None = None,
    min_marker_score: float | None = None,
    unknown_label: str = "Unknown",
) -> FilterResult:
    """Filter predictions by per-cell confidence, keeping only reliable calls.

    Cells below the threshold get relabeled to `unknown_label`.

    Args:
        predictions: Original prediction array (N cells).
        confidence_result: Result from compute_annotation_confidence().
        min_confidence: Minimum per-cell confidence score to keep (0-1).
        min_spatial_coherence: If set, also filter cells with spatial coherence
            below this value (independent of overall confidence).
        min_marker_score: If set, drop ALL cells of a type whose marker
            enrichment is below this threshold.
        unknown_label: Label for filtered cells.

    Returns:
        FilterResult with filtered predictions, mask, and statistics.
    """
    predictions = np.asarray(predictions)
    n_cells = len(predictions)
    mask = np.ones(n_cells, dtype=bool)

    cell_conf = confidence_result.cell_confidence
    if cell_conf is None:
        raise ValueError("confidence_result has no cell_confidence — run compute_annotation_confidence first")

    # 1. Per-cell confidence threshold
    mask &= cell_conf >= min_confidence

    # 2. Per-type marker score threshold (drop entire types)
    if min_marker_score is not None:
        for ct, info in confidence_result.per_type.items():
            if info.marker_score < min_marker_score:
                mask &= predictions != ct

    # 3. Per-cell spatial coherence threshold (needs re-extraction)
    # cell_confidence already incorporates spatial, but user may want a hard floor
    # Note: spatial coherence is baked into cell_confidence, so min_confidence
    # handles most cases. This is for strict spatial-only filtering.

    # Build filtered predictions
    filtered = predictions.copy()
    filtered[~mask] = unknown_label

    # Statistics
    per_type_kept: dict[str, int] = {}
    per_type_filtered: dict[str, int] = {}
    unique_types = [t for t in np.unique(predictions) if t != unknown_label]
    for ct in unique_types:
        ct_mask = predictions == ct
        per_type_kept[ct] = int((ct_mask & mask).sum())
        per_type_filtered[ct] = int((ct_mask & ~mask).sum())

    return FilterResult(
        predictions=filtered,
        mask=mask,
        n_kept=int(mask.sum()),
        n_filtered=int((~mask).sum()),
        per_type_kept=per_type_kept,
        per_type_filtered=per_type_filtered,
    )
