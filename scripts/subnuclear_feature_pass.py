"""Stream the breast-6source LMDB through StarDist + nucleus_feature_vector,
writing seg_features.parquet in chunks (+ optional packed-bit center-mask cache).

A --limit smoke prints patches/sec and a projected full-pass ETA before any long
run is launched. Pass the RAW uint16 patch to StarDist/features (they normalize
internally); do NOT /65535 here.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import polars as pl
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from starpose.qc.segmentation_grounded import (  # noqa: E402
    SegmentationGroundedScorer,
    SegQCConfig,
    select_center_nucleus,
)
from subnuclear_common import select_pass_indices  # noqa: E402

from dapidl.qc.patch_features import nucleus_feature_vector  # noqa: E402

LMDB_DIR = Path("/mnt/work/datasets/derived/breast-6source-dapi-p128")
TRAIN = ["xenium_rep1", "sthelar_breast_s0", "sthelar_breast_s1",
         "sthelar_breast_s3", "sthelar_breast_s6"]
TEST = ["xenium_rep2"]


def _read_patch(txn, idx: int) -> np.ndarray:
    value = txn.get(struct.pack(">Q", int(idx)))
    return np.frombuffer(value[8:], dtype=np.uint16).reshape(128, 128)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="all", help="all|train|test|comma list")
    ap.add_argument("--max-per-source", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke: process N then stop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--save-masks", choices=["none", "all"], default="none")
    ap.add_argument("--out-dir", default="pipeline_output/subnuclear_2026_06")
    ap.add_argument("--cpu", action="store_true", help="force CPU StarDist")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = np.load(LMDB_DIR / "labels.npy")
    # allow_pickle=True is safe here: sources.npy is a self-built LMDB artifact stored on local
    # disk under /mnt/work/datasets/ — it contains only Python str objects (source names) created
    # by our own pipeline steps. No external/untrusted data is loaded.
    sources = np.load(LMDB_DIR / "sources.npy", allow_pickle=True)
    keep = {"all": TRAIN + TEST, "train": TRAIN, "test": TEST}.get(
        args.sources, args.sources.split(","))
    idx = select_pass_indices(sources, labels, keep, max_per_source=args.max_per_source,
                              seed=args.seed, limit=args.limit, drop_unlabeled=False)
    print(f"[feature-pass] {len(idx)} patches from sources={keep} "
          f"max_per_source={args.max_per_source} limit={args.limit}")

    scorer = SegmentationGroundedScorer(SegQCConfig(), gpu=not args.cpu, pixel_size=0.2125)
    cfg = scorer.cfg

    env = lmdb.open(str(LMDB_DIR / "patches.lmdb"), readonly=True, lock=False,
                    readahead=False, meminit=False)
    writer = None
    rows: list[dict] = []
    mask_buf: list[np.ndarray] = []
    mask_idx: list[int] = []
    t0 = time.time()
    n_done = n_nuc = 0
    suffix = f"{args.sources}" + (f"_cap{args.max_per_source}" if args.max_per_source else "")
    pq_path = out / f"seg_features_{suffix}.parquet"

    def flush_rows():
        nonlocal writer, rows
        if not rows:
            return
        table = pl.DataFrame(rows).to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(pq_path, table.schema)
        writer.write_table(table)
        rows = []

    def flush_masks():
        nonlocal mask_buf, mask_idx
        if args.save_masks == "none" or not mask_buf:
            mask_buf, mask_idx = [], []
            return
        packed = np.packbits(np.stack(mask_buf).reshape(len(mask_buf), -1), axis=1)
        (out / "center_masks").mkdir(exist_ok=True)
        np.savez_compressed(out / "center_masks" / f"chunk_{mask_idx[0]:09d}.npz",
                            idx=np.array(mask_idx), masks=packed, shape=np.array([128, 128]))
        mask_buf, mask_idx = [], []

    try:
        with env.begin() as txn:
            for gi in idx:
                patch = _read_patch(txn, gi)
                masks, probs = scorer._segment(patch)
                cn = select_center_nucleus(masks, probs, cfg)
                mask = cn.mask if cn is not None else None
                prob = cn.prob if cn is not None else 0.0
                feats = nucleus_feature_vector(patch, mask, prob, cfg, scorer.pixel_size)
                row = {"global_idx": int(gi), "source": str(sources[gi]),
                       "label": int(labels[gi]), **feats}
                rows.append(row)
                if cn is not None:
                    n_nuc += 1
                    if args.save_masks == "all":
                        mask_buf.append(cn.mask.astype(bool))
                        mask_idx.append(int(gi))
                n_done += 1
                if n_done % args.chunk == 0:
                    flush_rows()
                    flush_masks()
                    rate = n_done / (time.time() - t0)
                    print(f"[feature-pass] {n_done}/{len(idx)} ({rate:.1f}/s) "
                          f"nucleus_coverage={n_nuc/n_done:.3f}")
        flush_rows()
        flush_masks()
    finally:
        if writer is not None:
            writer.close()

    dt = time.time() - t0
    rate = n_done / dt if dt > 0 else 0.0
    print(f"[feature-pass] DONE {n_done} patches in {dt:.1f}s ({rate:.1f}/s); "
          f"nucleus_coverage={n_nuc/max(n_done,1):.3f}; wrote {pq_path}")
    if args.limit is not None:
        full = len(select_pass_indices(sources, labels, TRAIN + TEST))
        rep2 = len(select_pass_indices(sources, labels, TEST))
        eta_h = full / rate / 3600 if rate > 0 else float("inf")
        rep2_h = rep2 / rate / 3600 if rate > 0 else float("inf")
        print(f"[ETA] full {full} patches @ {rate:.1f}/s ≈ {eta_h:.1f} h "
              f"(rep2 only: {rep2} ≈ {rep2_h:.2f} h)")
        print("[smoke] re-run without --limit for the full pass")


if __name__ == "__main__":
    main()
