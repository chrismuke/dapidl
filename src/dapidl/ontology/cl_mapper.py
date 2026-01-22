"""Cell Ontology Mapper - Main mapping interface with multi-level fallback.

This module provides the primary interface for mapping cell type labels from
any source (annotators, ground truth, external datasets) to standardized
Cell Ontology IDs.

Fallback Strategy:
    1. Exact Match → Check curated database
    2. Synonym Match → Check OBO synonyms
    3. Fuzzy Match → String similarity > threshold
    4. Ancestor Rollup → Map to closest known ancestor
    5. UNMAPPED → Return special ID

Usage:
    from dapidl.ontology import CLMapper

    mapper = CLMapper()

    # Map a single label
    cl_id = mapper.map("CD4+ T-cells")  # Returns "CL:0000624"

    # Map with confidence
    cl_id, confidence, method = mapper.map_with_info("helper T cell")

    # Batch mapping
    results = mapper.map_batch(["CD4+ T-cells", "Macrophages", "Unknown cell"])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Optional

from loguru import logger

from dapidl.ontology.cl_database import (
    DAPIDL_BROAD_CATEGORIES,
    CL_TO_BROAD_CATEGORY,
    CL_TO_COARSE_CATEGORY,
    get_all_terms,
    get_broad_category,
    get_coarse_category,
    get_term,
)
from dapidl.ontology.cl_loader import CLLoader, get_loader


class MappingMethod(Enum):
    """Method used to obtain mapping."""

    EXACT = "exact"  # Direct match in curated database
    SYNONYM = "synonym"  # Matched via OBO synonym
    FUZZY = "fuzzy"  # String similarity match
    ANCESTOR = "ancestor"  # Rolled up to ancestor
    GROUND_TRUTH = "ground_truth"  # Mapped from GT-specific label
    UNMAPPED = "unmapped"  # Could not map


@dataclass
class MappingResult:
    """Result of mapping a label to Cell Ontology."""

    original_label: str
    cl_id: str  # Mapped CL ID or "UNMAPPED"
    cl_name: str  # Canonical CL name
    confidence: float  # 0.0-1.0
    method: MappingMethod
    broad_category: str  # DAPIDL broad category
    coarse_category: str  # DAPIDL coarse category
    notes: str = ""  # Additional info (e.g., similarity score)


@dataclass
class MapperConfig:
    """Configuration for the CL mapper."""

    # Fuzzy matching
    fuzzy_threshold: float = 0.85  # Minimum similarity for fuzzy match
    fuzzy_method: str = "ratio"  # rapidfuzz method: ratio, partial_ratio, etc.

    # Ancestor rollup
    max_ancestor_depth: int = 5  # Max depth for ancestor rollup
    use_obo_loader: bool = True  # Load full OBO for advanced features

    # Caching
    cache_size: int = 10000  # LRU cache size for mappings


class CLMapper:
    """Maps cell type labels to Cell Ontology IDs.

    Implements a multi-level fallback strategy for robust mapping:
    1. Exact Match - Curated annotator-specific mappings
    2. Synonym Match - OBO file synonyms
    3. Fuzzy Match - String similarity matching
    4. Ancestor Rollup - Map to closest ancestor in curated set
    5. UNMAPPED - Return special marker
    """

    def __init__(
        self,
        config: MapperConfig | None = None,
        annotator_mappings: dict[str, str] | None = None,
        ground_truth_mappings: dict[str, str] | None = None,
    ):
        """Initialize the mapper.

        Args:
            config: Mapper configuration
            annotator_mappings: Additional annotator output → CL ID mappings
            ground_truth_mappings: GT label → CL ID mappings (dataset-specific)
        """
        self.config = config or MapperConfig()

        # Load curated database
        self._db_terms = get_all_terms()
        self._db_name_to_id = {t.name.lower(): t.cl_id for t in self._db_terms.values()}
        self._db_synonym_to_id: dict[str, str] = {}
        for term in self._db_terms.values():
            for syn in term.synonyms:
                self._db_synonym_to_id[syn.lower()] = term.cl_id

        # Custom mappings
        self._annotator_mappings = annotator_mappings or {}
        self._gt_mappings = ground_truth_mappings or {}

        # OBO loader for advanced features
        self._loader: Optional[CLLoader] = None
        if self.config.use_obo_loader:
            self._loader = get_loader()

        # Fuzzy matcher (lazy load)
        self._fuzzy_available = False
        self._check_fuzzy()

        logger.debug(
            f"CLMapper initialized with {len(self._db_terms)} curated terms, "
            f"{len(self._annotator_mappings)} custom mappings"
        )

    def _check_fuzzy(self) -> None:
        """Check if rapidfuzz is available."""
        try:
            import rapidfuzz  # noqa: F401

            self._fuzzy_available = True
        except ImportError:
            logger.warning(
                "rapidfuzz not installed - fuzzy matching disabled. "
                "Install with: pip install rapidfuzz"
            )
            self._fuzzy_available = False

    @lru_cache(maxsize=10000)
    def map(self, label: str) -> str:
        """Map a label to a CL ID.

        Args:
            label: Cell type label from any source

        Returns:
            CL ID (e.g., "CL:0000624") or "UNMAPPED"
        """
        result = self.map_with_info(label)
        return result.cl_id

    def map_with_info(self, label: str) -> MappingResult:
        """Map a label with full metadata.

        Args:
            label: Cell type label from any source

        Returns:
            MappingResult with CL ID, confidence, and method
        """
        original = label
        label_clean = self._normalize(label)

        # 1. Check custom annotator mappings (exact)
        if label in self._annotator_mappings:
            cl_id = self._annotator_mappings[label]
            return self._make_result(original, cl_id, 1.0, MappingMethod.EXACT)

        # 2. Check ground truth mappings
        if label in self._gt_mappings:
            cl_id = self._gt_mappings[label]
            return self._make_result(original, cl_id, 1.0, MappingMethod.GROUND_TRUTH)

        # 3. Exact match in curated database (by name)
        if label_clean in self._db_name_to_id:
            cl_id = self._db_name_to_id[label_clean]
            return self._make_result(original, cl_id, 1.0, MappingMethod.EXACT)

        # 4. Synonym match in curated database
        if label_clean in self._db_synonym_to_id:
            cl_id = self._db_synonym_to_id[label_clean]
            return self._make_result(original, cl_id, 0.95, MappingMethod.SYNONYM)

        # 5. Check OBO loader for synonyms
        if self._loader and self._loader.is_loaded:
            obo_cl_id = self._loader.find_by_name(label_clean)
            if obo_cl_id:
                return self._make_result(original, obo_cl_id, 0.9, MappingMethod.EXACT)

            obo_synonyms = self._loader.find_by_synonym(label_clean)
            if obo_synonyms:
                # Take first match
                return self._make_result(
                    original, obo_synonyms[0], 0.85, MappingMethod.SYNONYM
                )

        # 6. Pattern-based matching (handles variations)
        # Use label.lower() instead of label_clean to preserve _ and - separators
        # which are important for patterns like "DC_1", "Macro-IFN", "Fibro_adventitial"
        pattern_result = self._pattern_match(label.lower())
        if pattern_result:
            return self._make_result(
                original, pattern_result, 0.8, MappingMethod.FUZZY,
                notes="pattern match"
            )

        # 7. Fuzzy matching
        if self._fuzzy_available:
            fuzzy_result = self._fuzzy_match(label_clean)
            if fuzzy_result:
                cl_id, score = fuzzy_result
                return self._make_result(
                    original, cl_id, score, MappingMethod.FUZZY,
                    notes=f"similarity={score:.2f}"
                )

        # 8. Keyword-based fallback
        keyword_result = self._keyword_match(label)
        if keyword_result:
            return self._make_result(
                original, keyword_result, 0.6, MappingMethod.FUZZY,
                notes="keyword match"
            )

        # 9. Unmapped
        return MappingResult(
            original_label=original,
            cl_id="UNMAPPED",
            cl_name="Unknown",
            confidence=0.0,
            method=MappingMethod.UNMAPPED,
            broad_category="Unknown",
            coarse_category="Unknown",
        )

    def _normalize(self, label: str) -> str:
        """Normalize a label for matching."""
        # Lowercase
        label = label.lower()
        # Replace common separators
        label = label.replace("_", " ").replace("-", " ")
        # Remove extra whitespace
        label = " ".join(label.split())
        return label

    def _make_result(
        self,
        original: str,
        cl_id: str,
        confidence: float,
        method: MappingMethod,
        notes: str = "",
    ) -> MappingResult:
        """Create a MappingResult from a CL ID."""
        term = get_term(cl_id)

        if term:
            cl_name = term.name
        elif self._loader and self._loader.is_loaded:
            cl_name = self._loader.get_name(cl_id) or "Unknown"
        else:
            cl_name = "Unknown"

        return MappingResult(
            original_label=original,
            cl_id=cl_id,
            cl_name=cl_name,
            confidence=confidence,
            method=method,
            broad_category=get_broad_category(cl_id),
            coarse_category=get_coarse_category(cl_id),
            notes=notes,
        )

    def _pattern_match(self, label: str) -> str | None:
        """Match common patterns to CL IDs.

        Enhanced patterns to catch CellTypist-specific naming conventions
        like CD4-Tem (effector memory), CD8-Trm (tissue-resident memory),
        Macro-IFN (interferon-stimulated), etc.
        """
        patterns = [
            # T cells - enhanced for memory/effector subtypes
            # CD4+ T cells: CD4-Tem, CD4-Th, CD4-naive, CD4_naive/CM
            (r"cd4.*t|cd4.*(tem|th|naive|cm|em|trm)|t.*cd4|helper.*t|th\d*\s*cell", "CL:0000624"),
            # CD8+ T cells: CD8-Tem, CD8-Trm, CD8_EM, CD8_TRM/EM, Activated CD8 T
            (r"cd8.*t|cd8.*(tem|tmem|trm|em|activated)|t.*cd8|cytotoxic.*t|ctl\b|activated.*cd8", "CL:0000625"),
            (r"\btreg\b|regulatory.*t", "CL:0000815"),
            (r"\bnkt\b|nk.*t\s*cell", "CL:0000814"),  # NKT cells
            (r"thymocyte|double.*positive.*thymocyte", "CL:0000084"),  # Thymocytes
            (r"^t\s*cell$|^t-cell$|t.*lymphocyte", "CL:0000084"),
            # B cells - enhanced for memory subtypes
            (r"plasma.*cell|plasmacyte|plasma_", "CL:0000786"),
            (r"bmem|memory.*b", "CL:0000787"),  # Memory B cells
            (r"^b\s*cell$|^b-cell$|b.*lymphocyte", "CL:0000236"),
            # Myeloid - enhanced for macrophage/monocyte subtypes
            # Macrophages: Macro-IFN, Macro_interstitial, Macro_intravascular
            (r"macrophage|macro[-_]|histiocyte|\bm[12φ]\b", "CL:0000235"),
            # Monocytes: Mono-non-classical, Mono-classical
            (r"monocyte|mono[-_]", "CL:0000576"),
            # Myeloid proliferating
            (r"mye[-_]prol|myeloid.*prol", "CL:0000766"),
            # Dendritic cells: DC_1, DC_2, DC_activated
            (r"dendritic|^dc$|^dc_\d|dc_activated|\bcdc\b|\bpdc\b", "CL:0000451"),
            (r"mast.*cell|mastocyte", "CL:0000097"),
            (r"neutrophil|\bpmn\b", "CL:0000775"),
            # NK/ILC - enhanced, careful to avoid "unknown"
            (r"\bnk\s*cell|\bnatural\s*killer|\bnk[-_]ilc|\bilc\b", "CL:0000623"),
            # Lymphocyte general
            (r"lymphocyte", "CL:0000542"),
            # Epithelial - enhanced for specialized types
            (r"epithelial|epithelium", "CL:0000066"),
            # Luminal: LummHR-SCGB, Lumsec-prol
            (r"luminal|lumm|lumsec", "CL:0002325"),
            (r"basal.*cell|basal.*epithelial|^basal$", "CL:0000646"),
            (r"myoepithelial", "CL:0000185"),
            (r"hepatocyte|liver.*cell|cholangiocyte", "CL:0000182"),
            (r"keratinocyte", "CL:0000312"),
            # Specialized epithelial: colonocyte, ciliated, goblet, club, tuft
            (r"colonocyte|enterocyte|intestin.*epithelial", "CL:0002071"),
            (r"ciliated", "CL:0000064"),
            (r"goblet", "CL:0000160"),
            (r"club|secretory.*club", "CL:0000158"),
            (r"tuft", "CL:0002204"),
            # Stromal - enhanced for subtypes
            (r"fibroblast|fibro[-_]", "CL:0000057"),
            (r"myofibroblast", "CL:0000186"),
            (r"smooth.*muscle|\bsmc\b", "CL:0000192"),
            (r"pericyte|mural.*cell", "CL:0000669"),
            (r"adipocyte|fat.*cell", "CL:0000136"),
            (r"stromal|stroma", "CL:0000499"),
            # Endothelial - enhanced for EC abbreviations
            # Mature venous EC, cycling EC, Endothelia_Lymphatic
            (r"endotheli|endothelium|venous.*ec|cycling.*ec|\bec\b", "CL:0000115"),
            (r"lymphatic.*endotheli|\blec\b", "CL:0002138"),
        ]

        for pattern, cl_id in patterns:
            if re.search(pattern, label, re.IGNORECASE):
                return cl_id

        return None

    def _fuzzy_match(self, label: str) -> tuple[str, float] | None:
        """Find best fuzzy match in curated database."""
        if not self._fuzzy_available:
            return None

        from rapidfuzz import fuzz, process

        # Collect all searchable strings
        choices = list(self._db_name_to_id.keys()) + list(self._db_synonym_to_id.keys())

        if not choices:
            return None

        # Find best match
        result = process.extractOne(
            label, choices, scorer=fuzz.ratio, score_cutoff=self.config.fuzzy_threshold * 100
        )

        if result:
            match_str, score, _ = result
            score = score / 100.0  # Convert to 0-1

            # Get CL ID
            if match_str in self._db_name_to_id:
                return self._db_name_to_id[match_str], score
            elif match_str in self._db_synonym_to_id:
                return self._db_synonym_to_id[match_str], score

        return None

    def _keyword_match(self, label: str) -> str | None:
        """Simple keyword-based matching as last resort."""
        label_lower = label.lower()

        # Epithelial keywords
        if any(kw in label_lower for kw in ["epithelial", "tumor", "carcinoma", "luminal"]):
            return "CL:0000066"

        # Immune keywords
        if any(kw in label_lower for kw in ["immune", "leukocyte", "lymphocyte"]):
            return "CL:0000738"

        # Stromal keywords
        if any(kw in label_lower for kw in ["stromal", "stroma", "connective"]):
            return "CL:0000499"

        return None

    def map_batch(
        self,
        labels: list[str],
        return_df: bool = False,
    ) -> list[MappingResult] | "pl.DataFrame":
        """Map multiple labels.

        Args:
            labels: List of labels to map
            return_df: If True, return polars DataFrame

        Returns:
            List of MappingResults or DataFrame
        """
        results = [self.map_with_info(label) for label in labels]

        if return_df:
            import polars as pl

            return pl.DataFrame([
                {
                    "original_label": r.original_label,
                    "cl_id": r.cl_id,
                    "cl_name": r.cl_name,
                    "confidence": r.confidence,
                    "method": r.method.value,
                    "broad_category": r.broad_category,
                    "coarse_category": r.coarse_category,
                }
                for r in results
            ])

        return results

    def get_hierarchy_level(self, cl_id: str, target_level: str = "coarse") -> str:
        """Roll up a CL ID to a specific hierarchy level.

        Args:
            cl_id: CL ID to roll up
            target_level: "broad", "coarse", or "fine"

        Returns:
            Category name at the specified level
        """
        if target_level == "broad":
            return get_broad_category(cl_id)
        elif target_level == "coarse":
            return get_coarse_category(cl_id)
        else:
            # Fine level - return CL name directly
            term = get_term(cl_id)
            return term.name if term else "Unknown"

    def add_annotator_mapping(self, label: str, cl_id: str) -> None:
        """Add a custom annotator mapping.

        Args:
            label: Annotator output label
            cl_id: CL ID to map to
        """
        self._annotator_mappings[label] = cl_id
        # Clear cache
        self.map.cache_clear()

    def add_ground_truth_mapping(self, label: str, cl_id: str) -> None:
        """Add a ground truth mapping.

        Args:
            label: GT label
            cl_id: CL ID to map to
        """
        self._gt_mappings[label] = cl_id
        # Clear cache
        self.map.cache_clear()

    def get_mapping_stats(self) -> dict:
        """Get statistics about the mapper."""
        return {
            "curated_terms": len(self._db_terms),
            "curated_synonyms": len(self._db_synonym_to_id),
            "annotator_mappings": len(self._annotator_mappings),
            "gt_mappings": len(self._gt_mappings),
            "obo_loaded": self._loader.is_loaded if self._loader else False,
            "fuzzy_available": self._fuzzy_available,
        }


# Singleton instance
_default_mapper: Optional[CLMapper] = None


def get_mapper() -> CLMapper:
    """Get the default CLMapper instance with all curated mappings loaded."""
    global _default_mapper
    if _default_mapper is None:
        # Import here to avoid circular imports
        from dapidl.ontology.annotator_mappings import (
            get_all_annotator_mappings,
            get_all_gt_mappings,
        )
        _default_mapper = CLMapper(
            annotator_mappings=get_all_annotator_mappings(),
            ground_truth_mappings=get_all_gt_mappings(),
        )
    return _default_mapper


def map_label(label: str) -> str:
    """Convenience function to map a single label."""
    return get_mapper().map(label)


def map_labels(labels: list[str]) -> list[str]:
    """Convenience function to map multiple labels."""
    mapper = get_mapper()
    return [mapper.map(label) for label in labels]


if __name__ == "__main__":
    # Test the mapper
    mapper = CLMapper()

    test_labels = [
        "CD4+ T-cells",
        "CD4-positive, alpha-beta T cell",
        "helper T cell",
        "CD4 T cells",
        "Macrophages",
        "Epithelial cells",
        "Fibroblasts",
        "Some Unknown Type",
        "T_cells",
        "NK cells",
        "pDC",
        "M1 macrophage",
    ]

    print("=" * 80)
    print("Cell Ontology Mapper Test")
    print("=" * 80)

    for label in test_labels:
        result = mapper.map_with_info(label)
        print(f"\n{label}")
        print(f"  → {result.cl_id} ({result.cl_name})")
        print(f"    Method: {result.method.value}, Confidence: {result.confidence:.2f}")
        print(f"    Broad: {result.broad_category}, Coarse: {result.coarse_category}")

    print("\n" + "=" * 80)
    print("Mapper Stats:")
    print(mapper.get_mapping_stats())
