"""Reconstruct a per-row spatial registry (row_idx, source, cell_id, x_px, y_px)
for a pre-registry LMDB by replaying scripts/breast_dapi_lmdb.py's DETERMINISTIC
build order, then proving alignment against the stored sources.npy / labels.npy.

Pure replay cores mirror extract_xenium_breast / extract_sthelar_breast exactly:
candidate label filter, then the int(round(centroid)) +/- half OOB-bounds check.
No patches are read; only reader metadata + the mosaic shape are needed."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def _in_bounds(cx, cy, h, w, half) -> bool:
    cx, cy = int(round(cx)), int(round(cy))
    return not (cy - half < 0 or cx - half < 0 or cy + half > h or cx + half > w)


def replay_xenium(cell_ids, gt_lookup, fine_to_coarse, coarse_to_idx,
                  centroids, h, w, half) -> list[tuple]:
    rows = []
    for i, cid in enumerate(cell_ids):
        fine = gt_lookup.get(str(cid))
        if fine is None:
            continue
        coarse = fine_to_coarse(fine)
        if coarse is None:
            continue
        if not _in_bounds(centroids[i, 0], centroids[i, 1], h, w, half):
            continue
        rows.append((str(cid), float(centroids[i, 0]), float(centroids[i, 1]),
                     coarse_to_idx[coarse]))
    return rows


def replay_sthelar(ordered_cids, coarse_by_cid, centroid_by_cid, coarse_to_idx,
                   h, w, half) -> list[tuple]:
    """``ordered_cids`` is reader.nucleus_df cell_id order AFTER the coarse-not-null
    filter (mirrors extract_sthelar_breast's nuc_df.iter_rows)."""
    rows = []
    for cid in ordered_cids:
        if cid not in centroid_by_cid:
            continue
        cx, cy = centroid_by_cid[cid]
        if not _in_bounds(cx, cy, h, w, half):
            continue
        rows.append((str(cid), float(cx), float(cy), coarse_to_idx[coarse_by_cid[cid]]))
    return rows


def verify_alignment(rows, sources_seq, stored_sources, stored_labels) -> None:
    assert len(rows) == len(stored_sources) == len(stored_labels), (
        f"length desync: rows={len(rows)} sources={len(stored_sources)} labels={len(stored_labels)}")
    replay_labels = np.array([r[3] for r in rows], dtype=np.int64)
    assert np.array_equal(np.asarray(sources_seq), np.asarray(stored_sources)), "source-sequence desync"
    assert np.array_equal(replay_labels, np.asarray(stored_labels)), "label-sequence desync"


def build_spatial_registry(lmdb_dir: Path) -> pl.DataFrame:
    """Wire the real readers in build order, replay, verify, return the registry.
    Mirrors scripts/breast_dapi_lmdb.py main() source order: rep1, rep2, then
    sorted STHELAR breast zarrs. Imports the build script's constants so the label
    maps never drift."""
    import sys
    sys.path.insert(0, "scripts")
    import breast_dapi_lmdb as B  # noqa: E402  (constants + readers, no side effects)
    from dapidl.data.lazy_mosaic import open_xenium_mosaic, LazyMosaic
    from dapidl.data.sthelar import SthelarDataReader
    from dapidl.data.xenium import XeniumDataReader

    lmdb_dir = Path(lmdb_dir)
    # SAFE: sources.npy is written exclusively by scripts/breast_dapi_lmdb.py and
    # contains only a 1-D object array of plain Python strings (source-name labels
    # such as "xenium_rep1").  It is never user-supplied, never downloaded from an
    # external network, and lives inside the controlled LMDB build directory.
    # allow_pickle=True is required because numpy persists object-dtype arrays with
    # pickle; there is no viable alternative without rewriting the build script.
    stored_sources = np.load(lmdb_dir / "sources.npy", allow_pickle=True)
    stored_labels = np.load(lmdb_dir / "labels.npy")
    half = 64  # patch_size 128
    all_rows: list[tuple] = []
    all_src: list[str] = []

    for rep in ["rep1", "rep2"]:
        raw = B.XENIUM_BASE / f"xenium-breast-tumor-{rep}"
        reader = XeniumDataReader(raw / "outs")
        gt = B._load_xenium_supervised_gt(rep)
        cents = reader.get_centroids_pixels()
        cids = reader.get_cell_ids()
        with open_xenium_mosaic(reader.image_path) as m:
            h, w = m.shape
        rows = replay_xenium(cids, gt, B._xenium_fine_to_coarse, B.COARSE_TO_IDX, cents, h, w, half)
        all_rows += rows
        all_src += [f"xenium_{rep}"] * len(rows)

    for z in sorted(B.STHELAR_BASE.glob("sdata_breast_s*.zarr")):
        if not z.is_dir():
            continue
        name = z.name.replace("sdata_", "").replace(".zarr", "")
        reader = SthelarDataReader(z)
        ndf = reader.nucleus_df.with_columns(
            pl.col("label1").replace_strict(B.STHELAR_LABEL1_TO_COARSE, default=None).alias("coarse")
        ).filter(pl.col("coarse").is_not_null())
        ordered = ndf["cell_id"].to_list()
        coarse_by = dict(zip(ndf["cell_id"].to_list(), ndf["coarse"].to_list()))
        cents = reader.get_centroids_pixels()
        rcids = reader.get_cell_ids()
        cent_by = {cid: (cents[i, 0], cents[i, 1]) for i, cid in enumerate(rcids)}
        h, w = LazyMosaic(reader.dapi_lazy).shape
        rows = replay_sthelar(ordered, coarse_by, cent_by, B.COARSE_TO_IDX, h, w, half)
        all_rows += rows
        all_src += [f"sthelar_{name}"] * len(rows)

    verify_alignment(all_rows, all_src, stored_sources, stored_labels)
    return pl.DataFrame({
        "row_idx": np.arange(len(all_rows), dtype=np.int64),
        "source": all_src,
        "cell_id": [r[0] for r in all_rows],
        "x_px": [r[1] for r in all_rows],
        "y_px": [r[2] for r in all_rows],
        "coarse_idx": [r[3] for r in all_rows],
    })
