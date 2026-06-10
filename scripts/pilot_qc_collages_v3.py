"""Native-p64 QC collage renderer: DAPI grayscale + 1px cyan central-nucleus outline +
1px magenta resolved-cell outline.

render_tile is pure (unit-tested).
main() is the grouping driver: joins seg_scores + registry, groups by
(granularity, slide, class, qc_group), renders montages, and writes counts.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Optional

import lmdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

from dapidl.ontology.training_tiers import derive_labels  # noqa: E402

CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)

LMDB_DIR = Path("/mnt/work/datasets/derived/breast-pilot-6source-dapi-p64-nuc-v1")
OUT = Path("pipeline_output/pilot_qc_collages_v3")

# Geometric broken reasons (mis-centering / clipping) vs signal-quality reasons
GEOM_REASONS = {"off_center", "cut_at_edge", "edge_cut", "merged_blob"}

TAU_HI = 0.60
TAU_LO = 0.30

# Natural sort order for qc groups within a (class, slide) panel
GROUP_ORDER = [
    "Excellent",
    "Good",
    "Weak-passing",
    "Broken-geom",
    "Broken-quality",
]


# ---------------------------------------------------------------------------
# Core tile renderer (pure, unit-tested)
# ---------------------------------------------------------------------------

def _stretch_to_uint8(patch: np.ndarray) -> np.ndarray:
    p = patch.astype(np.float64)
    lo, hi = np.percentile(p, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((p - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def render_tile(patch, nucleus_mask, cell_mask) -> np.ndarray:
    """uint16 DAPI patch + bool nucleus/cell masks -> (H,W,3) uint8 RGB with 1px inner
    outlines: cyan nucleus, magenta cell. Empty masks paint nothing."""
    g = _stretch_to_uint8(np.asarray(patch))
    rgb = np.repeat(g[:, :, None], 3, axis=2)
    cell = np.asarray(cell_mask, dtype=bool)
    nuc = np.asarray(nucleus_mask, dtype=bool)
    if cell.any():
        rgb[find_boundaries(cell, mode="inner")] = MAGENTA
    if nuc.any():
        rgb[find_boundaries(nuc, mode="inner")] = CYAN   # nucleus drawn last (on top)
    return rgb


# ---------------------------------------------------------------------------
# Mask index builder
# ---------------------------------------------------------------------------

def build_mask_index(masks_dir: Path, prefix: str) -> dict[int, tuple[Path, int]]:
    """Scan all <prefix>_chunk_*.npz files under masks_dir.

    Returns dict mapping row_idx -> (npz_path, position_in_chunk).

    Each npz has:
        row_idx: int64 array (n,)  — global row indices for this chunk
        packed:  uint8 array (n, ceil(64*64/8))  — bit-packed masks

    The chunks are NOT written in global row order (they are grouped by slide),
    so we must scan every chunk's row_idx array to build the index.
    """
    index: dict[int, tuple[Path, int]] = {}
    for npz_path in sorted(masks_dir.glob(f"{prefix}_chunk_*.npz")):
        npz = np.load(npz_path)
        row_idxs = npz["row_idx"]  # shape (n,)
        for pos, ridx in enumerate(row_idxs):
            index[int(ridx)] = (npz_path, pos)
    return index


def load_mask_from_index(
    index: dict[int, tuple[Path, int]],
    row_idx: int,
    patch_size: int = 64,
) -> np.ndarray:
    """Load a single bool mask (H,W) for the given row_idx.

    Falls back to all-False mask when the row_idx is absent from the index
    (e.g. masks were never written for broken patches on earlier runs).
    """
    if row_idx not in index:
        return np.zeros((patch_size, patch_size), dtype=bool)
    npz_path, pos = index[row_idx]
    npz = np.load(npz_path)
    packed = npz["packed"]  # (n, ceil(H*W/8))
    flat = np.unpackbits(packed[pos : pos + 1], axis=1)[0, : patch_size * patch_size]
    return flat.reshape(patch_size, patch_size).astype(bool)


# ---------------------------------------------------------------------------
# Patch reader
# ---------------------------------------------------------------------------

def load_patch(env: lmdb.Environment, row_idx: int, patch_size: int = 64) -> np.ndarray:
    """Read a uint16 patch from an open (readonly) LMDB environment."""
    key = struct.pack(">Q", row_idx)
    with env.begin() as txn:
        value = txn.get(key)
    if value is None:
        return np.zeros((patch_size, patch_size), dtype=np.uint16)
    return np.frombuffer(value[8:], dtype=np.uint16).reshape(patch_size, patch_size)


# ---------------------------------------------------------------------------
# Grid montage builder
# ---------------------------------------------------------------------------

def build_montage(tiles: list[np.ndarray], patch_size: int = 64) -> np.ndarray:
    """Arrange a list of (H,W,3) uint8 tiles into a near-square grid.

    Pads with black tiles to fill the last row.  Returns (R*H, C*W, 3) uint8.
    """
    n = len(tiles)
    if n == 0:
        return np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    # Pad to fill grid
    black = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    padded = list(tiles) + [black] * (rows * cols - n)
    row_strips = []
    for r in range(rows):
        row_strips.append(
            np.concatenate(padded[r * cols : (r + 1) * cols], axis=1)
        )
    return np.concatenate(row_strips, axis=0)


# ---------------------------------------------------------------------------
# Group assignment
# ---------------------------------------------------------------------------

def assign_group(broken: bool, broken_reason: str, grade: str) -> str:
    """Map (broken, broken_reason, grade) -> one of the 5 canonical QC groups."""
    if broken:
        if broken_reason in GEOM_REASONS:
            return "Broken-geom"
        return "Broken-quality"
    return grade  # Excellent | Good | Weak-passing


# ---------------------------------------------------------------------------
# Sort key helper
# ---------------------------------------------------------------------------

_SORT_ASCENDING = {"grade"}  # natural group order, not a numeric metric


def sort_key_for(df: pl.DataFrame, sort_by: str) -> pl.DataFrame:
    """Return df sorted so the best tiles come first for collage sampling."""
    if sort_by == "grade":
        # Sort by the numeric quality_min descending (NaN = broken → end)
        return df.sort("quality_min", descending=True, nulls_last=True)
    return df.sort(sort_by, descending=True, nulls_last=True)


# ---------------------------------------------------------------------------
# README / summary writers
# ---------------------------------------------------------------------------

def write_readme(out_dir: Path) -> None:
    text = f"""\
# QC Groups — p64 Collage README

## Four quality axes (absolute calibration)

| Axis       | Signal used                                              |
|------------|----------------------------------------------------------|
| detection  | StarDist confidence × centeredness × dominant_central    |
| focus      | Brenner gradient (calibrated to percentile range)        |
| texture    | Mean(GLCM entropy, interior CoV, 1−GLCM ASM)             |
| brightness | Intensity ratio (calibrated) × (1 − saturation penalty) |

Calibration: for each axis the [1 %, 99 %] range of non-broken patches is
mapped to [0, 1].  No per-slide rescaling — values are dataset-absolute.

## Grade rule

    quality_min = min(detection, focus, texture, brightness)
    grade = "Excellent"    if quality_min >= {TAU_HI}
            "Good"         if quality_min >= {TAU_LO}
            "Weak-passing" otherwise (but NOT broken)

## Broken taxonomy

| Group          | broken_reason values                             |
|----------------|--------------------------------------------------|
| Broken-geom    | off_center, cut_at_edge, edge_cut, merged_blob   |
| Broken-quality | false_detection, no_nucleus, flat_interior, …   |

## Overlay legend

Each tile is a 64×64 DAPI patch with:
- **Cyan (1-px inner boundary)** — central nucleus from StarDist
- **Magenta (1-px inner boundary)** — resolved cell boundary
  (native Xenium polygon ▶ Voronoi expansion ▶ all-False if unavailable)

## Output path pattern

    collages/p64/<granularity>/<slide>/<class>/<group>.png
"""
    (out_dir / "qc_groups_README.md").write_text(text)


def write_counts_summary(counts: pl.DataFrame, out_dir: Path) -> None:
    lines = ["# QC-group counts\n"]
    for gran in counts["granularity"].unique().sort().to_list():
        lines.append(f"\n## {gran}\n")
        sub = counts.filter(pl.col("granularity") == gran)
        lines.append(sub.to_pandas().to_string(index=False))
        lines.append("")
    (out_dir / "counts_summary.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build per-(granularity, slide, class, qc_group) collages from p64 QC scores."
    )
    ap.add_argument(
        "--granularity", choices=["coarse", "medium", "both"], default="both",
        help="Which label granularity to use for grouping (default: both).",
    )
    ap.add_argument(
        "--sort-by",
        choices=["focus", "texture", "brightness", "detection", "grade"],
        default="grade",
        help="Metric to sort tiles by within each bucket (default: grade → quality_min desc).",
    )
    ap.add_argument(
        "--max-tiles", type=int, default=64,
        help="Maximum tiles per collage (default: 64).",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Process at most this many rows total (0 = all). For fast smoke-tests.",
    )
    ap.add_argument(
        "--lmdb-dir", type=Path, default=LMDB_DIR,
        help=f"LMDB dataset directory (default: {LMDB_DIR}).",
    )
    ap.add_argument(
        "--out", type=Path, default=OUT,
        help=f"Output root directory (default: {OUT}).",
    )
    args = ap.parse_args()

    lmdb_dir: Path = args.lmdb_dir
    out_root: Path = args.out
    patch_size = 64

    # -----------------------------------------------------------------------
    # 1. Load inputs
    # -----------------------------------------------------------------------
    qc_dir = lmdb_dir / "qc"
    scores_path = qc_dir / "seg_scores.parquet"
    registry_path = lmdb_dir / "patch_registry.parquet"
    class_mapping_path = lmdb_dir / "class_mapping.json"

    if not scores_path.exists():
        raise FileNotFoundError(
            f"seg_scores.parquet not found at {scores_path}. "
            "The rescore pass must complete before running this script."
        )

    scores = pl.read_parquet(scores_path)
    registry = pl.read_parquet(registry_path)

    with open(class_mapping_path) as fh:
        class_mapping: dict[str, int] = json.load(fh)
    idx_to_coarse: dict[int, str] = {v: k for k, v in class_mapping.items()}

    print(f"[collage] scores: {scores.shape}  registry: {registry.shape}", flush=True)

    # -----------------------------------------------------------------------
    # 2. Join scores ↔ registry on (row_idx, slide)
    # -----------------------------------------------------------------------
    # registry brings: raw_label (needed for medium label derivation)
    joined = scores.join(
        registry.select(["row_idx", "slide", "raw_label"]),
        on=["row_idx", "slide"],
        how="left",
    )

    if args.limit > 0:
        joined = joined.head(args.limit)
        print(f"[collage] --limit {args.limit}: using {len(joined)} rows", flush=True)

    # -----------------------------------------------------------------------
    # 3. Derive label columns
    # -----------------------------------------------------------------------
    coarse_labels: list[str] = []
    medium_labels: list[str] = []
    groups: list[str] = []

    for row in joined.iter_rows(named=True):
        # coarse
        coarse_labels.append(idx_to_coarse.get(int(row["coarse_idx"]), "Unknown"))

        # medium: strip _nuc suffix from slide name
        slide = str(row["slide"])
        source = slide.replace("_nuc", "")
        raw = str(row["raw_label"] or "")
        _, med = derive_labels(raw, source)
        medium_labels.append(med)

        # group
        groups.append(
            assign_group(
                bool(row["broken"]),
                str(row["broken_reason"] or ""),
                str(row["grade"] or "Weak-passing"),
            )
        )

    joined = joined.with_columns([
        pl.Series("coarse_label", coarse_labels),
        pl.Series("medium_label", medium_labels),
        pl.Series("qc_group", groups),
    ])

    # -----------------------------------------------------------------------
    # 4. Build mask indices (scan all chunk files once)
    # -----------------------------------------------------------------------
    masks_dir = qc_dir / "masks"
    print(f"[collage] building mask indices from {masks_dir} …", flush=True)
    nuc_index = build_mask_index(masks_dir, "nuc")
    cell_index = build_mask_index(masks_dir, "cell")
    print(
        f"[collage] nuc index: {len(nuc_index)} entries  "
        f"cell index: {len(cell_index)} entries",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # 5. Open LMDB
    # -----------------------------------------------------------------------
    env = lmdb.open(
        str(lmdb_dir / "patches.lmdb"),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    # -----------------------------------------------------------------------
    # 6. Determine which granularities to run
    # -----------------------------------------------------------------------
    gran_map: dict[str, str] = {}  # granularity name -> column name in joined
    if args.granularity in ("coarse", "both"):
        gran_map["coarse"] = "coarse_label"
    if args.granularity in ("medium", "both"):
        gran_map["medium"] = "medium_label"

    # -----------------------------------------------------------------------
    # 7. Build collages, accumulate counts
    # -----------------------------------------------------------------------
    count_rows: list[dict] = []

    for gran_name, label_col in gran_map.items():
        print(f"[collage] === granularity: {gran_name} ===", flush=True)

        for slide in sorted(joined["slide"].unique().to_list()):
            slide_df = joined.filter(pl.col("slide") == slide)

            for cls in sorted(slide_df[label_col].unique().to_list()):
                cls_df = slide_df.filter(pl.col(label_col) == cls)

                for group in GROUP_ORDER:
                    grp_df = cls_df.filter(pl.col("qc_group") == group)
                    n_total = len(grp_df)
                    if n_total == 0:
                        continue

                    # Sort and cap
                    grp_df = sort_key_for(grp_df, args.sort_by)
                    grp_df = grp_df.head(args.max_tiles)
                    n_shown = len(grp_df)

                    # Render tiles
                    tiles: list[np.ndarray] = []
                    for row in grp_df.iter_rows(named=True):
                        ridx = int(row["row_idx"])
                        patch = load_patch(env, ridx, patch_size)
                        nuc_mask = load_mask_from_index(nuc_index, ridx, patch_size)
                        cell_mask = load_mask_from_index(cell_index, ridx, patch_size)
                        tiles.append(render_tile(patch, nuc_mask, cell_mask))

                    # Build montage
                    grid = build_montage(tiles, patch_size)

                    # Save PNG
                    # Sanitise class name for filesystem (e.g. "CD4+_T_Cells" → safe)
                    cls_safe = cls.replace("/", "_").replace(" ", "_")
                    out_dir = (
                        out_root
                        / "collages"
                        / "p64"
                        / gran_name
                        / slide
                        / cls_safe
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)
                    png_path = out_dir / f"{group}.png"

                    fig, ax = plt.subplots(
                        figsize=(grid.shape[1] / patch_size * 2,
                                 grid.shape[0] / patch_size * 2),
                        dpi=72,
                    )
                    ax.imshow(grid, interpolation="nearest")
                    ax.set_title(
                        f"{slide} | {cls} | {group}  ({n_shown}/{n_total})",
                        fontsize=7,
                    )
                    ax.axis("off")
                    fig.tight_layout(pad=0.2)
                    fig.savefig(png_path, dpi=72, bbox_inches="tight")
                    plt.close(fig)

                    count_rows.append(
                        {
                            "granularity": gran_name,
                            "slide": slide,
                            "class": cls,
                            "group": group,
                            "n": n_total,
                            "n_shown": n_shown,
                        }
                    )
                    print(
                        f"[collage]  {gran_name}/{slide}/{cls_safe}/{group}"
                        f"  n={n_total}  shown={n_shown}  → {png_path}",
                        flush=True,
                    )

    env.close()

    # -----------------------------------------------------------------------
    # 8. Write counts
    # -----------------------------------------------------------------------
    out_root.mkdir(parents=True, exist_ok=True)
    counts_df = pl.DataFrame(count_rows)
    counts_path = out_root / "counts.parquet"
    counts_df.write_parquet(counts_path)
    print(f"[collage] counts → {counts_path}", flush=True)

    write_counts_summary(counts_df, out_root)
    write_readme(out_root)
    print(f"[collage] README + counts_summary.md → {out_root}", flush=True)
    print("[collage] DONE", flush=True)


if __name__ == "__main__":
    main()
