"""Bridge for the CelloType fine-tune smoke test.

Reads N tiles from the STHELAR cache (zarr v3 — only readable in dapidl venv,
py3.12) and exports them as a self-contained .npz the cellotype-env (py3.10)
training smoke test can load without zarr.

Output schema:
    images: (N, 1024, 1024) uint16  — DAPI tiles
    instance_maps: (N, 1024, 1024) uint16  — per-tile instance IDs (0=bg)
    tile_idx: (N,) int64
    fine_class_per_tile: list-of-list of (instance_id, fine_idx) — saved as
        two parallel ragged arrays via np.savez (`per_tile_offsets`, `per_tile_iid`,
        `per_tile_fine_idx`).

Class indexing matches dapidl's `TANGRAM_TO_BROAD` /  `TANGRAM_TO_MEDIUM` —
we use `MEDIUM_CLASS_NAMES` (10 classes) as the supervision target for the
smoke test (medium-tier is the metric we care about).
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import zarr
from loguru import logger

MEDIUM_CLASSES = [
    "Epithelial_Luminal",
    "Epithelial_Basal",
    "Epithelial_Tumor",
    "T_Cell",
    "B_Cell",
    "Myeloid",
    "NK_Cell",
    "Stromal_Fibroblast",
    "Stromal_Pericyte",
    "Endothelial",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--slide", default="breast_s3")
    ap.add_argument(
        "--split", default="train", choices=["train", "val", "test", "all"]
    )
    ap.add_argument("--n-tiles", type=int, default=30)
    ap.add_argument("--non-empty-only", action="store_true", default=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sdir = args.cache_root / args.slide
    manifest = pl.read_parquet(sdir / "manifest.parquet")
    if args.split != "all":
        manifest = manifest.filter(pl.col("split") == args.split)
    if args.non_empty_only:
        manifest = manifest.filter(pl.col("n_cells") > 0)
    manifest = manifest.head(args.n_tiles)
    n = len(manifest)
    if n == 0:
        raise SystemExit(f"no tiles after filter")
    logger.info(f"exporting {n} tiles from {args.slide} split={args.split}")

    img_z = zarr.open(str(sdir / "images.zarr"), mode="r")
    inst_z = zarr.open(str(sdir / "instances.zarr"), mode="r")
    labels = pl.read_parquet(sdir / "labels.parquet")

    medium_to_idx = {c: i for i, c in enumerate(MEDIUM_CLASSES)}

    images = np.zeros((n, img_z.shape[1], img_z.shape[2]), dtype=np.uint16)
    instance_maps = np.zeros_like(images, dtype=np.uint16)
    tile_idx_arr = np.zeros(n, dtype=np.int64)

    per_tile_iid: list[np.ndarray] = []
    per_tile_fine_idx: list[np.ndarray] = []

    for k, row in enumerate(manifest.iter_rows(named=True)):
        tile_idx = int(row["tile_idx"])
        tile_idx_arr[k] = tile_idx
        images[k] = np.asarray(img_z[tile_idx])
        instance_maps[k] = np.asarray(inst_z[tile_idx])

        tlabels = labels.filter(pl.col("tile_idx") == tile_idx).filter(
            ~pl.col("is_border")
        )
        iids = tlabels["instance_id"].to_numpy().astype(np.int64)
        meds = tlabels["medium"].to_list()
        fine_idx = np.array(
            [medium_to_idx.get(m, -1) for m in meds], dtype=np.int64
        )
        # Drop instances that don't map to a known medium class (-1)
        keep = fine_idx >= 0
        per_tile_iid.append(iids[keep])
        per_tile_fine_idx.append(fine_idx[keep])

    # Pack ragged arrays
    offsets = np.zeros(n + 1, dtype=np.int64)
    for k in range(n):
        offsets[k + 1] = offsets[k] + len(per_tile_iid[k])
    flat_iid = np.concatenate(per_tile_iid) if per_tile_iid else np.zeros(0, dtype=np.int64)
    flat_fine = (
        np.concatenate(per_tile_fine_idx) if per_tile_fine_idx else np.zeros(0, dtype=np.int64)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        images=images,
        instance_maps=instance_maps,
        tile_idx=tile_idx_arr,
        per_tile_offsets=offsets,
        per_tile_iid=flat_iid,
        per_tile_fine_idx=flat_fine,
    )
    import json as _json
    (args.out.with_suffix(".classes.json")).write_text(
        _json.dumps({"medium_classes": MEDIUM_CLASSES}, indent=2)
    )
    logger.success(
        f"wrote {n} tiles, {len(flat_iid)} instances → {args.out} "
        f"({args.out.stat().st_size / 1024**2:.1f} MiB)"
    )


if __name__ == "__main__":
    main()
