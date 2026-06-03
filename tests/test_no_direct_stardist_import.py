"""The seam: dapidl consumes segmentation via starpose, never StarDist directly.

Phase 1 routed the StarDist-grounded QC scorer through starpose. Three legacy
direct-StarDist call sites remain in dapidl's own segmenter components and
benchmark adapter; they are tracked below as known-pending migrations (Phase 3
consolidates the benchmark framework; the pipeline segmenter components are
superseded by StarposeSegmenter). The ratchet forbids any NEW direct import and
guarantees the qc/ subtree stays clean.
"""
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "dapidl"
PATTERN = re.compile(r"^\s*(from\s+stardist|import\s+stardist)\b", re.MULTILINE)

# Pre-existing direct-StarDist call sites pending migration to starpose.
# Do NOT add to this list — route new code through starpose.qc / starpose.create.
KNOWN_PENDING = {
    "pipeline/components/segmenters/adaptive.py",
    "pipeline/components/segmenters/stardist.py",
    "benchmark/segmenters/stardist_adapter.py",
}


def _offenders() -> set[str]:
    out = set()
    for py in SRC.rglob("*.py"):
        if PATTERN.search(py.read_text(encoding="utf-8")):
            out.add(py.relative_to(SRC).as_posix())
    return out


def test_no_new_direct_stardist_imports_in_dapidl_src():
    """Ratchet: no direct StarDist imports beyond the known-pending allowlist."""
    new = _offenders() - KNOWN_PENDING
    assert not new, f"new direct StarDist import(s) — route via starpose: {sorted(new)}"


def test_qc_subtree_has_no_direct_stardist_import():
    """The QC seam Phase 1 cleaned must stay clean."""
    qc = SRC / "qc"
    offenders = [p.relative_to(SRC).as_posix() for p in qc.rglob("*.py")
                 if PATTERN.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"qc/ must import StarDist via starpose: {offenders}"


def test_known_pending_list_is_not_stale():
    """If a pending file was migrated, drop it from KNOWN_PENDING (keeps the ratchet honest)."""
    stale = KNOWN_PENDING - _offenders()
    assert not stale, f"no longer import StarDist directly — remove from KNOWN_PENDING: {sorted(stale)}"
