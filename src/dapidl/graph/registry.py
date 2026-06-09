"""Reconstruct a per-row spatial registry (row_idx, source, cell_id, x_px, y_px,
coarse_idx) for the pre-registry ``breast-6source-dapi-p128`` LMDB.

Data reality (discovered 2026-06-09, see the design spec's "Data reality" note):
this LMDB was NOT built by the current scripts/breast_dapi_lmdb.py. Its STHELAR
labels came from a FINER annotation (stored in raw_labels.npy: CAF,
Mammary_luminal_cell, Endothelial_Pericyte_Smooth_muscle, ...) and the STHELAR
source annotations have since drifted, so no current column reproduces the stored
labels. BUT the per-source cell SET and ORDER are stable (the ``label1 != less10``
filter + deterministic nucleus_df order), so replaying the iteration recovers the
correct (cell_id, centroid) for each LMDB row. We therefore:
  1. replay the iteration to get per-row (cell_id, x_px, y_px),
  2. take the coarse label from the authoritative labels.npy (incl. -1 = unlabeled),
  3. PROVE alignment with a content check -- crop the source DAPI at each replayed
     centroid and correlate against the stored LMDB patch (drift-proof; no labels).

Pure cores (replay_xenium / replay_sthelar / patch_correlation / verify_counts /
verify_content) are unit-tested; build_spatial_registry is the GPU-free I/O wrapper
run by the controller."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import polars as pl


def _in_bounds(cx, cy, h, w, half) -> bool:
    cx, cy = int(round(cx)), int(round(cy))
    return not (cy - half < 0 or cx - half < 0 or cy + half > h or cx + half > w)


def replay_xenium(cell_ids, gt_lookup, fine_to_coarse, coarse_to_idx,
                  centroids, h, w, half) -> list[tuple]:
    """Mirror extract_xenium_breast's candidate filter + OOB bounds. The 4th tuple
    field (coarse) is used only for the inclusion filter; the registry's authoritative
    label comes from labels.npy, not this value."""
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
    """``ordered_cids`` is reader.nucleus_df cell_id order AFTER the label1 (coarse
    not-null) filter -- mirrors the build's nuc_df.iter_rows order."""
    rows = []
    for cid in ordered_cids:
        if cid not in centroid_by_cid:
            continue
        cx, cy = centroid_by_cid[cid]
        if not _in_bounds(cx, cy, h, w, half):
            continue
        rows.append((str(cid), float(cx), float(cy), coarse_to_idx[coarse_by_cid[cid]]))
    return rows


def patch_correlation(a, b) -> float:
    """Pearson correlation of two flattened patches (0.0 if either is constant or the
    sizes differ). ~1 = same nucleus at this centroid (aligned); ~0 = a different
    cell (misaligned)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def verify_counts(sources_seq, stored_sources) -> None:
    """Exact per-source count + order alignment against the stored sources.npy."""
    assert len(sources_seq) == len(stored_sources), (
        f"length desync: replay={len(sources_seq)} stored={len(stored_sources)}")
    assert np.array_equal(np.asarray(sources_seq), np.asarray(stored_sources)), \
        "source/count-sequence desync (per-source counts or order differ from the LMDB)"


def verify_content(spot_corr: dict, threshold: float = 0.9) -> None:
    """Assert every source's mean patch correlation clears ``threshold`` -- proves the
    replayed centroids index the same cells the LMDB stored, independent of labels."""
    bad = {s: round(c, 4) for s, c in spot_corr.items() if c < threshold}
    assert not bad, f"content desync (mean patch corr < {threshold}): {bad}"


def _read_stored_patch(txn, row_idx, patch_size=128) -> np.ndarray:
    """Decode a Format-B LMDB record (int64 label + uint16 square) -> patch float64."""
    value = txn.get(struct.pack(">Q", int(row_idx)))
    return np.frombuffer(value[8:], dtype=np.uint16).reshape(patch_size, patch_size).astype(np.float64)


def _spot_check_source(mosaic, txn, source_rows, global_start, half,
                       k=48, seed=0) -> float:
    """Mean patch correlation over k sampled rows: crop the source DAPI at the
    replayed centroid (clipped to [1, 99.5]% like the build's normalize) and correlate
    against the stored LMDB patch at the corresponding global row index."""
    n = len(source_rows)
    if n == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    sample = rng.choice(n, size=min(k, n), replace=False)
    corrs = []
    for j in sample:
        _, x, y, _ = source_rows[int(j)]
        cx, cy = int(round(x)), int(round(y))
        crop = mosaic.read(cy - half, cy + half, cx - half, cx + half)
        if crop.shape != (2 * half, 2 * half):
            continue
        lo, hi = np.percentile(crop, [1.0, 99.5])
        crop_n = np.clip(crop, lo, hi)
        corrs.append(patch_correlation(crop_n, _read_stored_patch(txn, global_start + int(j))))
    return float(np.mean(corrs)) if corrs else 0.0


def build_spatial_registry(lmdb_dir: Path, spot_k: int = 48) -> pl.DataFrame:
    """Wire the real readers in build order (rep1, rep2, sorted STHELAR), replay to
    recover (cell_id, centroid) per row, verify alignment by exact per-source counts
    AND per-source patch-correlation, then attach the authoritative coarse label from
    labels.npy (which includes -1 = unlabeled). Returns the full row-aligned registry."""
    import sys
    import lmdb
    sys.path.insert(0, "scripts")
    import breast_dapi_lmdb as B  # noqa: E402  (constants + readers, no side effects)
    from dapidl.data.lazy_mosaic import LazyMosaic, open_xenium_mosaic
    from dapidl.data.sthelar import SthelarDataReader
    from dapidl.data.xenium import XeniumDataReader

    lmdb_dir = Path(lmdb_dir)
    # SAFE: sources.npy / labels.npy are our own LMDB build artifacts (plain arrays,
    # never user-supplied or network-sourced). allow_pickle is required for the
    # object-dtype sources array.
    stored_sources = np.load(lmdb_dir / "sources.npy", allow_pickle=True)
    stored_labels = np.load(lmdb_dir / "labels.npy")
    half = 64  # patch_size 128
    all_rows: list[tuple] = []
    all_src: list[str] = []
    spot: dict[str, float] = {}

    env = lmdb.open(str(lmdb_dir / "patches.lmdb"), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            for rep in ["rep1", "rep2"]:
                reader = XeniumDataReader(B.XENIUM_BASE / f"xenium-breast-tumor-{rep}" / "outs")
                gt = B._load_xenium_supervised_gt(rep)
                cents = reader.get_centroids_pixels()
                cids = reader.get_cell_ids()
                with open_xenium_mosaic(reader.image_path) as m:
                    h, w = m.shape
                    rows = replay_xenium(cids, gt, B._xenium_fine_to_coarse,
                                         B.COARSE_TO_IDX, cents, h, w, half)
                    spot[f"xenium_{rep}"] = _spot_check_source(m, txn, rows, len(all_rows), half, spot_k)
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
                m = LazyMosaic(reader.dapi_lazy)
                h, w = m.shape
                rows = replay_sthelar(ordered, coarse_by, cent_by, B.COARSE_TO_IDX, h, w, half)
                spot[f"sthelar_{name}"] = _spot_check_source(m, txn, rows, len(all_rows), half, spot_k)
                all_rows += rows
                all_src += [f"sthelar_{name}"] * len(rows)

            verify_counts(all_src, stored_sources)   # exact cell-level alignment
            verify_content(spot, threshold=0.9)       # drift-proof content alignment
    finally:
        env.close()

    return pl.DataFrame({
        "row_idx": np.arange(len(all_rows), dtype=np.int64),
        "source": all_src,
        "cell_id": [r[0] for r in all_rows],
        "x_px": [r[1] for r in all_rows],
        "y_px": [r[2] for r in all_rows],
        "coarse_idx": np.asarray(stored_labels),   # authoritative (incl. -1)
    })
