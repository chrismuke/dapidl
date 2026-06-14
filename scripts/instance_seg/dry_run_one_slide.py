"""Phase 0 dry-run: materialize one slide (default s3) and measure
compression + throughput. Used to commit the cache layout before the full build.

What it measures:
  - Rasterization throughput (tiles/sec, after centroid precompute).
  - Compressed zarr size per tile (instance map and DAPI image).
  - Total zarr size on disk vs the raw uint16 estimate.
  - Loader throughput (zarr → numpy round-trip; loose proxy for dataloader).

Output:
  pipeline_output/instance_seg/dry_run_<slide>.json
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import zarr
from loguru import logger
from zarr.codecs import BloscCodec

from dapidl.data.instance_rasterizer import compute_centroids_px, rasterize_tile
from dapidl.data.sthelar import (
    load_nucleus_geometry_with_labels,
    select_dapi_channel,
)


def _compressor():
    return BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")

DEFAULT_ROOT = Path("/mnt/work/datasets/STHELAR/sdata_slides")


def slide_path(root: Path, slide: str) -> Path:
    return root / f"sdata_{slide}.zarr" / f"sdata_{slide}.zarr"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slide", default="breast_s3")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/mnt/work/datasets/derived/sthelar_breast_tiles_dryrun"),
    )
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--pad", type=int, default=64)
    ap.add_argument("--max-tiles", type=int, default=200, help="cap for dry run")
    args = ap.parse_args()

    sroot = slide_path(args.root, args.slide)
    if not sroot.exists():
        raise FileNotFoundError(f"{sroot} missing")

    if args.out.exists():
        logger.warning(f"removing previous dry-run output {args.out}")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Load slide
    logger.info(f"loading {args.slide}")
    t0 = time.time()
    gdf = load_nucleus_geometry_with_labels(sroot, ["ct_tangram"])
    geom_load_s = time.time() - t0
    logger.info(
        f"  {len(gdf)} polys, dropped {gdf.attrs['n_invalid_dropped']} invalid "
        f"({geom_load_s:.1f}s)"
    )

    gdf["instance_id"] = np.arange(1, len(gdf) + 1, dtype=np.uint32)

    # Image + channel
    dapi_ch = select_dapi_channel(sroot)
    grp = zarr.open(str(sroot), mode="r")
    morpho = grp["images/morpho/0"]
    img_shape = morpho.shape
    if len(img_shape) == 3:
        c, h, w = img_shape
    else:
        c, h, w = 1, *img_shape
    logger.info(f"  morpho shape={img_shape}, dapi_channel={dapi_ch}")

    # 2. Precompute centroids once
    t0 = time.time()
    cents = compute_centroids_px(gdf)
    logger.info(f"  centroids precomputed in {time.time()-t0:.1f}s")

    # 3. Build tile origin grid
    ys = list(range(0, h - args.tile_size + 1, args.stride))
    xs = list(range(0, w - args.tile_size + 1, args.stride))
    grid = [(y, x) for y in ys for x in xs]
    logger.info(f"  grid: {len(grid)} tiles")

    # 4. Open zarr writers (zarr v3 API)
    n_w = min(args.max_tiles, len(grid))
    images_z = zarr.create_array(
        store=str(args.out / "images.zarr"),
        shape=(n_w, args.tile_size, args.tile_size),
        chunks=(1, args.tile_size, args.tile_size),
        dtype="uint16",
        compressors=_compressor(),
        overwrite=True,
    )
    inst_z = zarr.create_array(
        store=str(args.out / "instances.zarr"),
        shape=(n_w, args.tile_size, args.tile_size),
        chunks=(1, args.tile_size, args.tile_size),
        dtype="uint16",
        compressors=_compressor(),
        overwrite=True,
    )

    # 5. Materialize tiles and time it
    t_raster = 0.0
    t_image = 0.0
    t_write = 0.0
    n_nonempty = 0
    n_total_instances = 0
    for tile_idx, (y0, x0) in enumerate(grid[: args.max_tiles]):
        # Rasterize
        t0 = time.time()
        inst_map, is_border, in_tile_idx = rasterize_tile(
            gdf,
            "instance_id",
            x0=x0,
            y0=y0,
            tile_size=args.tile_size,
            pad=args.pad,
            centroids_px=cents,
        )
        t_raster += time.time() - t0

        # DAPI image read
        t0 = time.time()
        if c == 1:
            img_tile = morpho[0, y0 : y0 + args.tile_size, x0 : x0 + args.tile_size]
        else:
            img_tile = morpho[
                dapi_ch, y0 : y0 + args.tile_size, x0 : x0 + args.tile_size
            ]
        # Pad if at slide edge (shouldn't happen in our grid but be defensive)
        if img_tile.shape != (args.tile_size, args.tile_size):
            pad_h = args.tile_size - img_tile.shape[0]
            pad_w = args.tile_size - img_tile.shape[1]
            img_tile = np.pad(img_tile, ((0, pad_h), (0, pad_w)))
        t_image += time.time() - t0

        # Write
        t0 = time.time()
        images_z[tile_idx] = img_tile.astype(np.uint16)
        inst_z[tile_idx] = inst_map
        t_write += time.time() - t0

        if len(in_tile_idx) > 0:
            n_nonempty += 1
            n_total_instances += int(len(in_tile_idx))

    n_tiles_done = min(args.max_tiles, len(grid))

    # 6. Measure on-disk size
    def dir_size(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    img_bytes = dir_size(args.out / "images.zarr")
    inst_bytes = dir_size(args.out / "instances.zarr")
    raw_per_tile = args.tile_size * args.tile_size * 2  # uint16
    total_raw = raw_per_tile * n_tiles_done * 2  # image + instance

    # 7. Loader speed test (read-back round trip)
    t0 = time.time()
    n_read = min(50, n_tiles_done)
    for i in range(n_read):
        _ = images_z[i]
        _ = inst_z[i]
    t_load = time.time() - t0

    summary = {
        "slide": args.slide,
        "tile_size": args.tile_size,
        "stride": args.stride,
        "pad": args.pad,
        "n_polygons_loaded": len(gdf),
        "n_invalid_dropped_upstream": gdf.attrs["n_invalid_dropped"],
        "tiles_in_grid": len(grid),
        "tiles_materialized": n_tiles_done,
        "non_empty_tiles": n_nonempty,
        "total_instances_in_tiles": n_total_instances,
        "geom_load_s": round(geom_load_s, 2),
        "raster_s": round(t_raster, 2),
        "raster_tiles_per_sec": round(n_tiles_done / max(t_raster, 1e-3), 2),
        "image_read_s": round(t_image, 2),
        "image_read_tiles_per_sec": round(n_tiles_done / max(t_image, 1e-3), 2),
        "zarr_write_s": round(t_write, 2),
        "loader_s": round(t_load, 2),
        "loader_tiles_per_sec": round(n_read / max(t_load, 1e-3), 2),
        "img_bytes_on_disk": img_bytes,
        "inst_bytes_on_disk": inst_bytes,
        "total_bytes_on_disk": img_bytes + inst_bytes,
        "raw_uint16_total_bytes": total_raw,
        "compression_ratio": round(total_raw / max(img_bytes + inst_bytes, 1), 2),
        "img_compression_ratio": round(
            (raw_per_tile * n_tiles_done) / max(img_bytes, 1), 2
        ),
        "inst_compression_ratio": round(
            (raw_per_tile * n_tiles_done) / max(inst_bytes, 1), 2
        ),
    }

    out_report = Path(f"pipeline_output/instance_seg/dry_run_{args.slide}.json")
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(summary, indent=2))

    logger.info("=" * 60)
    logger.info(f"DRY RUN ({args.slide}, stride={args.stride})")
    logger.info("=" * 60)
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"report → {out_report}")


if __name__ == "__main__":
    main()
