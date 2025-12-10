"""Label harmonization and mapping between annotation sources.

Maps cell type labels from different sources (CellTypist, popV, ground truth)
to a common hierarchy for meaningful comparison.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from dapidl.harmonization.hierarchy import (
    BREAST_HIERARCHY,
    CellTypeHierarchy,
    HierarchyLevel,
)

logger = logging.getLogger(__name__)

# Directory containing mapping JSON files
MAPPINGS_DIR = Path(__file__).parent / "mappings"


@dataclass
class MappingEntry:
    """A single label mapping entry."""

    source_label: str
    hierarchy_label: str
    confidence: float = 1.0
    notes: str = ""


@dataclass
class LabelMapping:
    """Complete mapping from a source to the hierarchy."""

    source_name: str
    description: str
    entries: dict[str, MappingEntry] = field(default_factory=dict)
    unmapped_labels: list[str] = field(default_factory=list)

    def add(
        self,
        source_label: str,
        hierarchy_label: str,
        confidence: float = 1.0,
        notes: str = "",
    ) -> None:
        """Add a mapping entry."""
        self.entries[source_label.lower()] = MappingEntry(
            source_label=source_label,
            hierarchy_label=hierarchy_label,
            confidence=confidence,
            notes=notes,
        )

    def get(self, source_label: str) -> str | None:
        """Get hierarchy label for a source label."""
        entry = self.entries.get(source_label.lower())
        return entry.hierarchy_label if entry else None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_name": self.source_name,
            "description": self.description,
            "mappings": {
                k: {
                    "hierarchy_label": v.hierarchy_label,
                    "confidence": v.confidence,
                    "notes": v.notes,
                }
                for k, v in self.entries.items()
            },
            "unmapped_labels": self.unmapped_labels,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LabelMapping:
        """Create from dictionary."""
        mapping = cls(
            source_name=data["source_name"],
            description=data.get("description", ""),
            unmapped_labels=data.get("unmapped_labels", []),
        )
        for source_label, entry_data in data.get("mappings", {}).items():
            mapping.add(
                source_label=source_label,
                hierarchy_label=entry_data["hierarchy_label"],
                confidence=entry_data.get("confidence", 1.0),
                notes=entry_data.get("notes", ""),
            )
        return mapping


@dataclass
class HarmonizationResult:
    """Result of label harmonization with metrics at multiple levels."""

    source_labels: list[str]
    target_labels: list[str]
    harmonized_source: dict[str, list[str]]  # level -> labels
    harmonized_target: dict[str, list[str]]  # level -> labels
    unmapped_source: list[str]
    unmapped_target: list[str]
    metrics: dict[str, dict]  # level -> metrics dict

    def summary(self) -> str:
        """Generate summary string."""
        lines = ["Label Harmonization Summary", "=" * 40]

        for level in ["broad", "mid", "fine"]:
            if level in self.metrics:
                m = self.metrics[level]
                lines.append(f"\n{level.upper()} Level:")
                lines.append(f"  Accuracy: {m.get('accuracy', 0):.3f}")
                lines.append(f"  Macro F1: {m.get('f1_macro', 0):.3f}")
                lines.append(f"  Weighted F1: {m.get('f1_weighted', 0):.3f}")

        if self.unmapped_source:
            lines.append(f"\nUnmapped source labels: {len(self.unmapped_source)}")
            lines.append(f"  {self.unmapped_source[:5]}...")

        if self.unmapped_target:
            lines.append(f"\nUnmapped target labels: {len(self.unmapped_target)}")
            lines.append(f"  {self.unmapped_target[:5]}...")

        return "\n".join(lines)


def load_mapping(source_name: str) -> LabelMapping | None:
    """Load a mapping from JSON file."""
    mapping_file = MAPPINGS_DIR / f"{source_name}.json"
    if not mapping_file.exists():
        return None

    with open(mapping_file) as f:
        data = json.load(f)

    return LabelMapping.from_dict(data)


def save_mapping(mapping: LabelMapping) -> None:
    """Save a mapping to JSON file."""
    MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
    mapping_file = MAPPINGS_DIR / f"{mapping.source_name}.json"

    with open(mapping_file, "w") as f:
        json.dump(mapping.to_dict(), f, indent=2)


def get_available_mappings() -> list[str]:
    """Get list of available mapping names."""
    if not MAPPINGS_DIR.exists():
        return []
    return [f.stem for f in MAPPINGS_DIR.glob("*.json")]


class LabelHarmonizer:
    """Harmonizes cell type labels across different annotation sources.

    Handles the challenge of comparing labels from different sources
    (CellTypist, popV, ground truth) by mapping to a common hierarchy.

    Key features:
    - Multi-level comparison (broad, mid, fine)
    - Handles unmapped/cancer-specific labels gracefully
    - Computes metrics at each granularity level
    - Supports automatic and manual mappings

    Example:
        harmonizer = LabelHarmonizer()

        # Compare predictions to ground truth
        result = harmonizer.compare(
            predictions,
            ground_truth,
            source="celltypist_breast",
            target="xenium_breast",
        )

        print(result.summary())
        print(result.metrics["broad"]["f1_macro"])
    """

    def __init__(
        self,
        hierarchy: CellTypeHierarchy | None = None,
        mappings: dict[str, LabelMapping] | None = None,
    ):
        """Initialize harmonizer.

        Args:
            hierarchy: Cell type hierarchy to use (defaults to BREAST_HIERARCHY)
            mappings: Pre-loaded mappings (source_name -> LabelMapping)
        """
        self.hierarchy = hierarchy or BREAST_HIERARCHY
        self.mappings: dict[str, LabelMapping] = mappings or {}
        self._label_cache: dict[tuple[str, str | None, str], tuple[str | None, bool]] = {}
        self._load_default_mappings()

    def _load_default_mappings(self) -> None:
        """Load default mappings from JSON files."""
        for name in get_available_mappings():
            if name not in self.mappings:
                mapping = load_mapping(name)
                if mapping:
                    self.mappings[name] = mapping

    def register_mapping(self, mapping: LabelMapping) -> None:
        """Register a label mapping."""
        self.mappings[mapping.source_name] = mapping

    def map_label(
        self,
        label: str,
        source: str | None = None,
        level: HierarchyLevel = "fine",
    ) -> tuple[str | None, bool]:
        """Map a single label to the hierarchy.

        Args:
            label: Original label
            source: Source name for explicit mapping lookup
            level: Target hierarchy level

        Returns:
            Tuple of (mapped_label, was_mapped)
        """
        # Check cache first
        cache_key = (label, source, level)
        if cache_key in self._label_cache:
            return self._label_cache[cache_key]

        result = self._map_label_uncached(label, source, level)
        self._label_cache[cache_key] = result
        return result

    def _map_label_uncached(
        self,
        label: str,
        source: str | None,
        level: HierarchyLevel,
    ) -> tuple[str | None, bool]:
        """Map a single label without caching."""
        # Try explicit mapping first
        if source and source in self.mappings:
            hierarchy_label = self.mappings[source].get(label)
            if hierarchy_label:
                return self._to_level(hierarchy_label, level), True

        # Try direct hierarchy lookup
        if label in self.hierarchy.nodes:
            return self._to_level(label, level), True

        # Try alias lookup
        found = self.hierarchy.find_by_alias(label)
        if found:
            return self._to_level(found, level), True

        # Try normalized lookup
        normalized = self._normalize_label(label)
        if normalized in self.hierarchy.nodes:
            return self._to_level(normalized, level), True

        found = self.hierarchy.find_by_alias(normalized)
        if found:
            return self._to_level(found, level), True

        return None, False

    def _to_level(self, label: str, level: HierarchyLevel) -> str | None:
        """Convert label to specified hierarchy level."""
        if level == "fine":
            return label
        elif level == "mid":
            return self.hierarchy.get_mid(label)
        elif level == "broad":
            return self.hierarchy.get_broad(label)
        return None

    def _normalize_label(self, label: str) -> str:
        """Normalize label for matching."""
        # Replace common separators
        normalized = label.replace(" ", "_").replace("-", "_").replace("+", "+")
        return normalized

    def map_labels(
        self,
        labels: Sequence[str],
        source: str | None = None,
        level: HierarchyLevel = "fine",
        unknown_label: str = "Unknown",
    ) -> tuple[list[str], list[str]]:
        """Map a sequence of labels.

        Args:
            labels: Original labels
            source: Source name for mapping lookup
            level: Target hierarchy level
            unknown_label: Label to use for unmapped types

        Returns:
            Tuple of (mapped_labels, unmapped_original_labels)
        """
        mapped = []
        unmapped_set: set[str] = set()

        for label in labels:
            result, was_mapped = self.map_label(label, source, level)
            if was_mapped and result:
                mapped.append(result)
            else:
                mapped.append(unknown_label)
                unmapped_set.add(label)

        return mapped, list(unmapped_set)

    def compare(
        self,
        source_labels: Sequence[str],
        target_labels: Sequence[str],
        source_name: str | None = None,
        target_name: str | None = None,
        levels: list[HierarchyLevel] | None = None,
    ) -> HarmonizationResult:
        """Compare two sets of labels at multiple hierarchy levels.

        Args:
            source_labels: Predicted/source labels
            target_labels: Ground truth/target labels
            source_name: Source annotation system name
            target_name: Target annotation system name
            levels: Hierarchy levels to compare at

        Returns:
            HarmonizationResult with metrics at each level
        """
        if len(source_labels) != len(target_labels):
            raise ValueError(
                f"Label arrays must have same length: "
                f"{len(source_labels)} vs {len(target_labels)}"
            )

        levels = levels or ["broad", "mid", "fine"]

        harmonized_source: dict[str, list[str]] = {}
        harmonized_target: dict[str, list[str]] = {}
        all_unmapped_source: set[str] = set()
        all_unmapped_target: set[str] = set()
        metrics: dict[str, dict] = {}

        for level in levels:
            # Map both sets of labels
            src_mapped, src_unmapped = self.map_labels(
                source_labels, source_name, level
            )
            tgt_mapped, tgt_unmapped = self.map_labels(
                target_labels, target_name, level
            )

            harmonized_source[level] = src_mapped
            harmonized_target[level] = tgt_mapped
            all_unmapped_source.update(src_unmapped)
            all_unmapped_target.update(tgt_unmapped)

            # Compute metrics (excluding Unknown)
            mask = np.array(
                [s != "Unknown" and t != "Unknown" for s, t in zip(src_mapped, tgt_mapped)]
            )
            if mask.sum() > 0:
                src_valid = np.array(src_mapped)[mask]
                tgt_valid = np.array(tgt_mapped)[mask]

                metrics[level] = {
                    "accuracy": float(accuracy_score(tgt_valid, src_valid)),
                    "f1_macro": float(
                        f1_score(tgt_valid, src_valid, average="macro", zero_division=0)
                    ),
                    "f1_weighted": float(
                        f1_score(tgt_valid, src_valid, average="weighted", zero_division=0)
                    ),
                    "n_samples": int(mask.sum()),
                    "n_excluded": int((~mask).sum()),
                }
            else:
                metrics[level] = {
                    "accuracy": 0.0,
                    "f1_macro": 0.0,
                    "f1_weighted": 0.0,
                    "n_samples": 0,
                    "n_excluded": len(source_labels),
                }

        return HarmonizationResult(
            source_labels=list(source_labels),
            target_labels=list(target_labels),
            harmonized_source=harmonized_source,
            harmonized_target=harmonized_target,
            unmapped_source=list(all_unmapped_source),
            unmapped_target=list(all_unmapped_target),
            metrics=metrics,
        )

    def get_confusion_matrix(
        self,
        source_labels: Sequence[str],
        target_labels: Sequence[str],
        source_name: str | None = None,
        target_name: str | None = None,
        level: HierarchyLevel = "broad",
    ) -> tuple[np.ndarray, list[str]]:
        """Get confusion matrix at specified level.

        Returns:
            Tuple of (confusion_matrix, class_names)
        """
        src_mapped, _ = self.map_labels(source_labels, source_name, level)
        tgt_mapped, _ = self.map_labels(target_labels, target_name, level)

        # Get unique labels (excluding Unknown)
        all_labels = sorted(
            set(src_mapped) | set(tgt_mapped) - {"Unknown"}
        )

        cm = confusion_matrix(tgt_mapped, src_mapped, labels=all_labels)
        return cm, all_labels

    def get_classification_report(
        self,
        source_labels: Sequence[str],
        target_labels: Sequence[str],
        source_name: str | None = None,
        target_name: str | None = None,
        level: HierarchyLevel = "broad",
    ) -> str:
        """Get sklearn classification report at specified level."""
        src_mapped, _ = self.map_labels(source_labels, source_name, level)
        tgt_mapped, _ = self.map_labels(target_labels, target_name, level)

        # Filter out Unknown
        mask = [s != "Unknown" and t != "Unknown" for s, t in zip(src_mapped, tgt_mapped)]
        src_valid = [s for s, m in zip(src_mapped, mask) if m]
        tgt_valid = [t for t, m in zip(tgt_mapped, mask) if m]

        return classification_report(tgt_valid, src_valid, zero_division=0)

    def auto_map_celltypist(self, model_name: str) -> LabelMapping:
        """Automatically create mapping for a CellTypist model.

        Uses the hierarchy to map CellTypist cell types based on
        naming patterns and aliases.
        """
        try:
            from celltypist import models

            model = models.Model.load(model=model_name)
            cell_types = model.cell_types
        except Exception as e:
            logger.warning(f"Failed to load CellTypist model {model_name}: {e}")
            return LabelMapping(
                source_name=f"celltypist_{model_name}",
                description=f"Auto-mapping for CellTypist {model_name}",
            )

        mapping = LabelMapping(
            source_name=f"celltypist_{model_name.replace('.pkl', '')}",
            description=f"Auto-generated mapping for CellTypist {model_name}",
        )

        for ct in cell_types:
            # Try direct hierarchy lookup
            hierarchy_label, found = self.map_label(ct, level="fine")
            if found and hierarchy_label:
                mapping.add(ct, hierarchy_label)
            else:
                mapping.unmapped_labels.append(ct)

        return mapping

    def report_mapping_coverage(
        self,
        labels: Sequence[str],
        source: str | None = None,
    ) -> dict:
        """Report mapping coverage for a set of labels.

        Returns:
            Dictionary with coverage statistics
        """
        mapped, unmapped = self.map_labels(labels, source, level="fine")

        unique_labels = set(labels)
        mapped_unique = {
            label for label in unique_labels
            if self.map_label(label, source, "fine")[1]
        }

        return {
            "total_labels": len(labels),
            "unique_labels": len(unique_labels),
            "mapped_unique": len(mapped_unique),
            "unmapped_unique": len(unique_labels - mapped_unique),
            "coverage": len(mapped_unique) / len(unique_labels) if unique_labels else 0,
            "unmapped_list": list(unique_labels - mapped_unique),
        }
