"""Unified Marker Gene Database — replaces hardcoded BREAST_MARKERS.

Loads marker genes from two curated databases:
1. Cell Marker Accordion (Nature Comms 2025) — 141K entries, 728 CL cell types,
   21 source databases aggregated with Evidence Counts (ECs) quality scores
2. CellMarker 2.0 (Nucleic Acids Research 2023) — 60K human entries,
   1,715 cell names with breast-specific tissue coverage

Provides a unified lookup by CL ID, DAPIDL category name, or free-text label,
with optional tissue-aware filtering and quality thresholds.

Usage:
    from dapidl.validation.marker_database import get_marker_db

    db = get_marker_db()

    # Get markers for a CL ID
    markers = db.get_markers("CL:0000084")  # T cell
    # {"positive": ["CD3D", "CD3E", ...], "negative": ["CD19", ...]}

    # Get markers for a DAPIDL category name
    markers = db.get_markers_for_label("T_Cell")

    # Build markers_db dict compatible with annotation_confidence.py
    markers_db = db.build_markers_db(predictions, panel_genes=set(adata.var_names))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
from loguru import logger

# ── Database paths ──────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "marker_databases"
_ACCORDION_PATH = _DATA_DIR / "CellMarkerAccordion_v1.0.0.xlsx"
_CELLMARKER2_PATH = _DATA_DIR / "Cell_marker_Human.xlsx"

# Tissue name aliases — user-facing names to Uberon/CellMarker tissue strings
_TISSUE_ALIASES: dict[str, list[str]] = {
    "breast": ["breast", "mammary gland", "mammary", "mammary epithelium"],
    "lung": ["lung", "bronchus", "trachea", "respiratory"],
    "liver": ["liver", "hepatic"],
    "kidney": ["kidney", "renal"],
    "heart": ["heart", "cardiac", "myocardium"],
    "brain": ["brain", "cerebral", "hippocampus", "cortex", "cerebellum"],
    "colon": ["colon", "large intestine", "colorectal", "intestine"],
    "skin": ["skin", "dermis", "epidermis", "cutaneous"],
    "ovary": ["ovary", "ovarian"],
    "pancreas": ["pancreas", "pancreatic", "islet"],
    "tonsil": ["tonsil", "palatine tonsil"],
    "lymph_node": ["lymph node", "lymphoid"],
    "cervix": ["cervix", "cervical", "uterine cervix"],
}


# ── Marker entry ────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class MarkerEntry:
    """A single marker gene with provenance."""

    gene: str
    cl_id: str
    marker_type: str  # "positive" or "negative"
    evidence_count: int  # ECs_global from Accordion or 1 for CellMarker 2.0
    source: str  # "accordion" or "cellmarker2"
    tissue: str | None = None  # Uberon tissue or CellMarker tissue_type


# ── Unified Marker Database ────────────────────────────────────────

class UnifiedMarkerDB:
    """Unified marker gene database backed by Accordion + CellMarker 2.0."""

    def __init__(self) -> None:
        self._markers: dict[str, list[MarkerEntry]] = {}  # cl_id → entries
        self._cl_name_to_id: dict[str, str] = {}  # lowercase name → cl_id
        self._label_to_cl_ids: dict[str, list[str]] = {}  # DAPIDL label → cl_ids
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load_all()
            self._loaded = True

    @staticmethod
    def _tissue_matches(entry_tissue: str | None, query_tissue: str) -> bool:
        """Check if an entry's tissue matches the query, using aliases."""
        if entry_tissue is None:
            return True  # No tissue annotation = universal marker
        entry_lower = entry_tissue.lower()
        query_lower = query_tissue.lower()

        # Direct substring match
        if query_lower in entry_lower or entry_lower in query_lower:
            return True

        # Alias-based matching
        aliases = _TISSUE_ALIASES.get(query_lower, [query_lower])
        return any(alias in entry_lower for alias in aliases)

    def _load_all(self) -> None:
        """Load and merge both databases."""
        # Build DAPIDL label → CL ID index from our ontology
        self._build_label_index()

        n_accordion = self._load_accordion()
        n_cellmarker2 = self._load_cellmarker2()

        total_cl_ids = len(self._markers)
        total_entries = sum(len(v) for v in self._markers.values())
        logger.info(
            f"Unified marker DB loaded: {total_entries} entries across "
            f"{total_cl_ids} CL types (Accordion: {n_accordion}, CellMarker2: {n_cellmarker2})"
        )

    def _build_label_index(self) -> None:
        """Build index from DAPIDL category labels to CL IDs."""
        from dapidl.ontology.cl_database import (
            CL_TO_COARSE_CATEGORY,
            get_all_terms,
        )

        # Reverse: coarse category name → list of CL IDs
        coarse_to_cl: dict[str, list[str]] = {}
        for cl_id, cat in CL_TO_COARSE_CATEGORY.items():
            coarse_to_cl.setdefault(cat, []).append(cl_id)
        self._label_to_cl_ids = coarse_to_cl

        # Build name → cl_id index from all known terms
        for cl_id, term in get_all_terms().items():
            self._cl_name_to_id[term.name.lower()] = cl_id
            for syn in term.synonyms:
                self._cl_name_to_id[syn.lower()] = cl_id

    def _load_accordion(self) -> int:
        """Load Cell Marker Accordion database."""
        if not _ACCORDION_PATH.exists():
            logger.warning(f"Accordion DB not found at {_ACCORDION_PATH}")
            return 0

        logger.debug(f"Loading Accordion from {_ACCORDION_PATH}")
        df = pl.read_excel(_ACCORDION_PATH)

        # Filter: human, has CL_ID, has marker gene
        df = df.filter(
            (pl.col("species") == "Human")
            & pl.col("CL_ID").is_not_null()
            & (pl.col("CL_ID") != "")
            & pl.col("marker").is_not_null()
            & (pl.col("marker") != "")
        )

        count = 0
        for row in df.iter_rows(named=True):
            cl_id = str(row["CL_ID"]).strip()
            gene = str(row["marker"]).strip()
            marker_type = str(row.get("marker_type", "positive") or "positive").strip().lower()
            if marker_type not in ("positive", "negative"):
                marker_type = "positive"

            # Parse evidence count — may be int, float, or string
            ecs_raw = row.get("ECs_global", 1)
            try:
                ecs = int(float(ecs_raw)) if ecs_raw is not None else 1
            except (ValueError, TypeError):
                ecs = 1

            tissue = str(row.get("Uberon_tissue", "") or "").strip() or None

            # Also index by CL name
            cl_name = str(row.get("CL_celltype", "") or "").strip().lower()
            if cl_name and cl_id and cl_name not in self._cl_name_to_id:
                self._cl_name_to_id[cl_name] = cl_id

            entry = MarkerEntry(
                gene=gene,
                cl_id=cl_id,
                marker_type=marker_type,
                evidence_count=ecs,
                source="accordion",
                tissue=tissue,
            )
            self._markers.setdefault(cl_id, []).append(entry)
            count += 1

        logger.debug(f"Accordion: {count} entries, {len({r['CL_ID'] for r in df.iter_rows(named=True)})} CL types")
        return count

    def _load_cellmarker2(self) -> int:
        """Load CellMarker 2.0 database."""
        if not _CELLMARKER2_PATH.exists():
            logger.warning(f"CellMarker 2.0 DB not found at {_CELLMARKER2_PATH}")
            return 0

        logger.debug(f"Loading CellMarker 2.0 from {_CELLMARKER2_PATH}")
        df = pl.read_excel(_CELLMARKER2_PATH)

        # Filter: normal cells (not cancer), has gene symbol
        df = df.filter(
            (pl.col("cell_type") == "Normal cell")
            & pl.col("Symbol").is_not_null()
            & (pl.col("Symbol") != "")
        )

        count = 0
        for row in df.iter_rows(named=True):
            gene = str(row["Symbol"]).strip()
            if not gene:
                continue

            # CellMarker 2.0 uses cellontology_id (may be absent)
            cl_id = str(row.get("cellontology_id", "") or "").strip()
            cell_name = str(row.get("cell_name", "") or "").strip()
            tissue = str(row.get("tissue_type", "") or "").strip() or None

            # If no CL ID, try to resolve via our ontology
            if not cl_id or cl_id == "None":
                cl_id = self._cl_name_to_id.get(cell_name.lower(), "")

            if not cl_id:
                continue  # Can't map without CL ID

            # Index the cell name
            if cell_name and cl_id:
                name_lower = cell_name.lower()
                if name_lower not in self._cl_name_to_id:
                    self._cl_name_to_id[name_lower] = cl_id

            entry = MarkerEntry(
                gene=gene,
                cl_id=cl_id,
                marker_type="positive",  # CellMarker 2.0 doesn't distinguish
                evidence_count=1,  # Single source, no aggregation score
                source="cellmarker2",
                tissue=tissue,
            )
            self._markers.setdefault(cl_id, []).append(entry)
            count += 1

        return count

    # ── Public API ──────────────────────────────────────────────────

    def get_markers(
        self,
        cl_id: str,
        *,
        tissue: str | None = None,
        min_evidence: int = 1,
        max_markers: int = 20,
        include_parents: bool = True,
    ) -> dict[str, list[str]]:
        """Get positive and negative markers for a CL ID.

        Args:
            cl_id: Cell Ontology ID (e.g., "CL:0000084")
            tissue: Optional tissue filter (Uberon name or CellMarker tissue_type)
            min_evidence: Minimum ECs_global to include a marker (Accordion only)
            max_markers: Maximum markers to return per type (sorted by evidence)
            include_parents: If True, also collect markers from parent CL terms

        Returns:
            {"positive": ["CD3D", ...], "negative": ["CD19", ...]}
        """
        self._ensure_loaded()

        # Collect entries from this CL ID and optionally parents
        entries = list(self._markers.get(cl_id, []))

        if include_parents:
            from dapidl.ontology.cl_database import get_term

            term = get_term(cl_id)
            visited = {cl_id}
            while term and term.parent_id and term.parent_id not in visited:
                parent_entries = self._markers.get(term.parent_id, [])
                entries.extend(parent_entries)
                visited.add(term.parent_id)
                term = get_term(term.parent_id)

        # Filter by tissue if specified (prefer tissue-specific, fall back to all)
        if tissue:
            tissue_filtered = [e for e in entries if self._tissue_matches(e.tissue, tissue)]
            if tissue_filtered:
                entries = tissue_filtered

        # Filter by minimum evidence
        entries = [e for e in entries if e.evidence_count >= min_evidence]

        # Separate positive/negative and rank by evidence
        pos_genes: dict[str, int] = {}
        neg_genes: dict[str, int] = {}

        for e in entries:
            target = pos_genes if e.marker_type == "positive" else neg_genes
            # Keep highest evidence count per gene
            if e.gene not in target or e.evidence_count > target[e.gene]:
                target[e.gene] = e.evidence_count

        # Sort by evidence count descending, take top N
        pos_sorted = sorted(pos_genes.items(), key=lambda x: -x[1])[:max_markers]
        neg_sorted = sorted(neg_genes.items(), key=lambda x: -x[1])[:max_markers]

        return {
            "positive": [g for g, _ in pos_sorted],
            "negative": [g for g, _ in neg_sorted],
        }

    def get_markers_for_label(
        self,
        label: str,
        *,
        tissue: str | None = None,
        min_evidence: int = 1,
        max_markers: int = 20,
    ) -> dict[str, list[str]]:
        """Get markers for a DAPIDL category name or free-text label.

        Resolves the label to CL IDs via:
        1. DAPIDL coarse category mapping (e.g., "T_Cell" → CL:0000084, CL:0000624, ...)
        2. Direct name/synonym match in CL ontology
        3. CLMapper fuzzy matching
        4. Fallback: curated CLTerm markers from cl_database.py
        """
        self._ensure_loaded()

        result: dict[str, list[str]] = {"positive": [], "negative": []}

        # 1. Try DAPIDL coarse category
        cl_ids = self._label_to_cl_ids.get(label, [])
        if cl_ids:
            result = self._merge_markers_from_cl_ids(
                cl_ids, tissue=tissue, min_evidence=min_evidence, max_markers=max_markers
            )
            if result["positive"]:
                return result

        # 2. Try name/synonym match
        label_lower = label.lower().replace("_", " ")
        cl_id = self._cl_name_to_id.get(label_lower)
        if cl_id:
            result = self.get_markers(cl_id, tissue=tissue, min_evidence=min_evidence, max_markers=max_markers)
            if result["positive"]:
                return result

        # 3. Try CLMapper
        try:
            from dapidl.ontology import CLMapper
            mapper = CLMapper()
            mapping = mapper.map_with_info(label)
            if mapping.cl_id != "UNMAPPED":
                result = self.get_markers(
                    mapping.cl_id, tissue=tissue, min_evidence=min_evidence, max_markers=max_markers
                )
                if result["positive"]:
                    return result
        except Exception:
            pass

        # 4. Fallback: curated markers from cl_database.py CLTerm definitions
        result = self._get_curated_markers(label, cl_ids or ([] if not cl_id else [cl_id]))
        if result["positive"]:
            return result

        logger.debug(f"No markers found for label '{label}'")
        return {"positive": [], "negative": []}

    def _get_curated_markers(self, label: str, cl_ids: list[str]) -> dict[str, list[str]]:
        """Fall back to curated CLTerm markers from cl_database.py."""
        from dapidl.ontology.cl_database import get_all_terms

        all_terms = get_all_terms()
        markers: list[str] = []

        # Collect markers from the CL IDs we know about
        for cl_id in cl_ids:
            term = all_terms.get(cl_id)
            if term and term.markers:
                markers.extend(m for m in term.markers if m not in markers)

        # Also try matching by name
        if not markers:
            label_lower = label.lower().replace("_", " ")
            for term in all_terms.values():
                if term.name.lower() == label_lower or any(s.lower() == label_lower for s in term.synonyms):
                    markers.extend(m for m in term.markers if m not in markers)

        return {"positive": markers, "negative": []}

    def _merge_markers_from_cl_ids(
        self,
        cl_ids: list[str],
        *,
        tissue: str | None = None,
        min_evidence: int = 1,
        max_markers: int = 20,
    ) -> dict[str, list[str]]:
        """Merge markers from multiple CL IDs (for coarse categories)."""
        all_pos: dict[str, int] = {}
        all_neg: dict[str, int] = {}

        for cl_id in cl_ids:
            entries = self._markers.get(cl_id, [])
            # Apply tissue filter with fallback (prefer tissue-specific, fall back to all)
            if tissue:
                tissue_entries = [e for e in entries if self._tissue_matches(e.tissue, tissue)]
                if tissue_entries:
                    entries = tissue_entries
            for e in entries:
                if e.evidence_count < min_evidence:
                    continue
                target = all_pos if e.marker_type == "positive" else all_neg
                if e.gene not in target or e.evidence_count > target[e.gene]:
                    target[e.gene] = e.evidence_count

        pos_sorted = sorted(all_pos.items(), key=lambda x: -x[1])[:max_markers]
        neg_sorted = sorted(all_neg.items(), key=lambda x: -x[1])[:max_markers]

        return {
            "positive": [g for g, _ in pos_sorted],
            "negative": [g for g, _ in neg_sorted],
        }

    def build_markers_db(
        self,
        cell_type_labels: list[str] | set[str],
        *,
        panel_genes: set[str] | None = None,
        tissue: str | None = None,
        min_evidence: int = 2,
        max_markers: int = 20,
    ) -> dict[str, dict[str, list[str]]]:
        """Build a markers_db dict compatible with annotation_confidence.py.

        This is the primary integration point — produces the same format as
        BREAST_MARKERS: {cell_type: {"positive": [...], "negative": [...]}}.

        Args:
            cell_type_labels: Set of predicted cell type labels to look up
            panel_genes: If provided, filter markers to genes in the panel
            tissue: Optional tissue for tissue-aware filtering
            min_evidence: Minimum evidence count threshold
            max_markers: Max markers per cell type per polarity

        Returns:
            Dict matching the BREAST_MARKERS format.
        """
        self._ensure_loaded()

        markers_db: dict[str, dict[str, list[str]]] = {}

        for label in cell_type_labels:
            if label.lower() in ("unknown", "unassigned", ""):
                continue

            result = self.get_markers_for_label(
                label, tissue=tissue, min_evidence=min_evidence, max_markers=max_markers,
            )

            # Filter to panel genes if specified
            if panel_genes:
                result = {
                    "positive": [g for g in result["positive"] if g in panel_genes],
                    "negative": [g for g in result["negative"] if g in panel_genes],
                }

            if result["positive"]:  # Only include if we found positive markers
                markers_db[label] = result

        logger.info(
            f"Built markers_db: {len(markers_db)}/{len(cell_type_labels)} types with markers, "
            f"tissue={tissue}, min_evidence={min_evidence}"
        )
        return markers_db

    def stats(self) -> dict:
        """Return database statistics."""
        self._ensure_loaded()

        n_accordion = sum(
            1 for entries in self._markers.values()
            for e in entries if e.source == "accordion"
        )
        n_cellmarker2 = sum(
            1 for entries in self._markers.values()
            for e in entries if e.source == "cellmarker2"
        )

        return {
            "total_cl_ids": len(self._markers),
            "total_entries": sum(len(v) for v in self._markers.values()),
            "accordion_entries": n_accordion,
            "cellmarker2_entries": n_cellmarker2,
            "name_index_size": len(self._cl_name_to_id),
            "label_index_size": len(self._label_to_cl_ids),
        }


# ── Module-level singleton ──────────────────────────────────────────

_db_instance: UnifiedMarkerDB | None = None


def get_marker_db() -> UnifiedMarkerDB:
    """Get the singleton unified marker database."""
    global _db_instance
    if _db_instance is None:
        _db_instance = UnifiedMarkerDB()
    return _db_instance
