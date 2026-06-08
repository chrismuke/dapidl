"""STHELAR tile cache builder for instance-segmentation training.

Phase A of the joint instance-seg + classification design (see
`docs/superpowers/specs/2026-05-02-instance-seg-breast-design.md`).

Key design choices:
- **Stride 1024 + uint16 instance maps** by default (codex BLOCKER 2 fix).
- **Spatial-block parent split** — 4096²-px blocks assigned to train/val/test;
  child tiles are generated *inside* each parent block, never across the
  train/val boundary. Hard assertion: `train_cell_ids ∩ val_cell_ids = ∅`
  (codex HIGH 5 fix).
- Polygon → mask via `instance_rasterizer.rasterize_tile`, with `make_valid`
  repair already applied upstream by `load_nucleus_geometry_with_labels`.
- Class labels come from `ct_tangram` mapped through `TANGRAM_TO_BROAD` /
  `TANGRAM_TO_MEDIUM` (codex BLOCKER 1 fix).

Cache layout (matches design doc §2.5):

    cache_root/
        manifest.parquet            # all-slide tile manifest
        s0/
            images.zarr             # (n_tiles, 1024, 1024) uint16
            instances.zarr          # (n_tiles, 1024, 1024) uint16
            labels.parquet          # (tile_idx, instance_id, cell_id, fine, medium, broad, ...)
        s1/ ...
        s3/ ...
        s6/ ...
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import zarr
from loguru import logger
from zarr.codecs import BloscCodec

from dapidl.data.instance_rasterizer import compute_centroids_px, rasterize_tile
from dapidl.data.sthelar import (
    TANGRAM_TO_BROAD,
    TANGRAM_TO_MEDIUM,
    load_nucleus_geometry_with_labels,
    select_dapi_channel,
)


@dataclass
class TileCacheConfig:
    """Configuration for the tile-cache build."""

    cache_root: Path
    sthelar_root: Path = Path("/mnt/work/datasets/STHELAR/sdata_slides")
    slides: list[str] = field(
        default_factory=lambda: ["breast_s0", "breast_s1", "breast_s3", "breast_s6"]
    )
    tile_size: int = 1024
    stride: int = 1024
    pad: int = 64
    parent_block_size: int = 4096  # spatial-block split granularity (px)
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1  # remainder
    seed: int = 42
    blosc_clevel: int = 3


def _slide_path(root: Path, slide: str) -> Path:
    return root / f"sdata_{slide}.zarr" / f"sdata_{slide}.zarr"


def _compressor(clevel: int = 3) -> BloscCodec:
    return BloscCodec(cname="zstd", clevel=clevel, shuffle="bitshuffle")


def _spatial_block_split(
    centroids_px: np.ndarray,
    img_h: int,
    img_w: int,
    block_size: int,
    rng: np.random.Generator,
    train_frac: float,
    val_frac: float,
) -> tuple[np.ndarray, dict[int, str]]:
    """Assign each cell to a parent block, then assign blocks to splits.

    Returns:
        cell_split: (N,) array of "train"/"val"/"test" strings
        block_to_split: dict block_idx → split label

    Block indexing: `block_idx = block_y * n_block_x + block_x` where
    `block_x = floor(x / block_size)`, `block_y = floor(y / block_size)`.
    """
    n_block_x = (img_w + block_size - 1) // block_size
    bx = (centroids_px[:, 0] // block_size).astype(np.int64)
    by = (centroids_px[:, 1] // block_size).astype(np.int64)
    block_idx = by * n_block_x + bx
    unique_blocks = np.unique(block_idx)
    rng.shuffle(unique_blocks)

    n_blocks = len(unique_blocks)
    n_train = int(train_frac * n_blocks)
    n_val = int(val_frac * n_blocks)
    train_blocks = set(unique_blocks[:n_train].tolist())
    val_blocks = set(unique_blocks[n_train : n_train + n_val].tolist())
    # Remaining → test

    block_to_split: dict[int, str] = {}
    for b in unique_blocks:
        if b in train_blocks:
            block_to_split[int(b)] = "train"
        elif b in val_blocks:
            block_to_split[int(b)] = "val"
        else:
            block_to_split[int(b)] = "test"

    cell_split = np.array(
        [block_to_split[int(b)] for b in block_idx], dtype="<U5"
    )
    return cell_split, block_to_split


def _tile_grid_in_block(
    block_x: int,
    block_y: int,
    block_size: int,
    img_h: int,
    img_w: int,
    tile_size: int,
    stride: int,
) -> list[tuple[int, int, int]]:
    """Yield (x0, y0, parent_block_idx) tile origins inside a parent block.

    Tiles are not allowed to cross the parent-block boundary on either axis,
    to preserve the spatial-block split (codex HIGH 5).
    Each tile must fit fully inside the image (no edge-padding).
    """
    n_block_x = (img_w + block_size - 1) // block_size
    parent_idx = block_y * n_block_x + block_x
    x_start = block_x * block_size
    y_start = block_y * block_size
    x_end = min((block_x + 1) * block_size, img_w)
    y_end = min((block_y + 1) * block_size, img_h)

    out: list[tuple[int, int, int]] = []
    for y0 in range(y_start, y_end - tile_size + 1, stride):
        for x0 in range(x_start, x_end - tile_size + 1, stride):
            out.append((x0, y0, parent_idx))
    return out


def build_one_slide(slide: str, cfg: TileCacheConfig) -> dict:
    """Materialize tile cache for a single slide.

    Returns a per-slide manifest dict with metrics (timing, n_tiles, etc.).
    """
    sroot = _slide_path(cfg.sthelar_root, slide)
    out_dir = cfg.cache_root / slide
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{slide}] loading geometry + labels")
    t0 = time.time()
    gdf = load_nucleus_geometry_with_labels(
        sroot, ["ct_tangram", "label1", "label2"]
    )
    geom_load_s = time.time() - t0
    n_invalid_dropped = int(gdf.attrs.get("n_invalid_dropped", 0))
    logger.info(
        f"[{slide}]   {len(gdf)} polys, dropped {n_invalid_dropped} invalid "
        f"({geom_load_s:.1f}s)"
    )

    # Drop cells whose ct_tangram has no fine→broad mapping (e.g. nan, empty).
    n_pre = len(gdf)
    gdf = gdf[gdf["ct_tangram"].isin(TANGRAM_TO_BROAD.keys())].copy()
    n_unmapped_dropped = n_pre - len(gdf)
    if n_unmapped_dropped > 0:
        logger.warning(
            f"[{slide}]   dropped {n_unmapped_dropped} cells with "
            "unmapped ct_tangram values"
        )

    # Image + DAPI channel
    grp = zarr.open(str(sroot), mode="r")
    morpho = grp["images/morpho/0"]
    img_shape = morpho.shape
    if len(img_shape) == 3:
        n_ch, img_h, img_w = img_shape
    else:
        n_ch, img_h, img_w = 1, img_shape[0], img_shape[1]
    dapi_ch = select_dapi_channel(sroot)
    logger.info(
        f"[{slide}]   morpho {img_shape} (h={img_h}, w={img_w}, "
        f"channels={n_ch}, dapi={dapi_ch})"
    )

    # Precompute centroids
    cents = compute_centroids_px(gdf)

    # Spatial-block split
    rng = np.random.default_rng(cfg.seed + abs(hash(slide)) % (2**32))
    cell_split, block_to_split = _spatial_block_split(
        cents,
        img_h,
        img_w,
        cfg.parent_block_size,
        rng,
        cfg.train_frac,
        cfg.val_frac,
    )
    gdf["split"] = cell_split
    n_train = int((cell_split == "train").sum())
    n_val = int((cell_split == "val").sum())
    n_test = int((cell_split == "test").sum())
    logger.info(
        f"[{slide}]   spatial-block split: train={n_train}, val={n_val}, test={n_test}"
    )

    # Hard assert: cells are partitioned, not duplicated
    assert n_train + n_val + n_test == len(gdf), "cell split sums don't match"

    # Per-cell instance ID (uint16 per slide)
    if len(gdf) > 65535:
        # IDs span multiple tiles; uint16 is per-tile. We assign a global
        # uint32 ID for labels.parquet, but per-tile rasterization uses a
        # local 1..n_in_tile ID (set inside the loop).
        gdf["global_instance_id"] = np.arange(1, len(gdf) + 1, dtype=np.uint32)
    else:
        gdf["global_instance_id"] = np.arange(1, len(gdf) + 1, dtype=np.uint16)

    # Build tile grid: per parent block, tiles inside that block only
    n_block_x = (img_w + cfg.parent_block_size - 1) // cfg.parent_block_size
    n_block_y = (img_h + cfg.parent_block_size - 1) // cfg.parent_block_size
    grid: list[tuple[int, int, int, str]] = []
    for by in range(n_block_y):
        for bx in range(n_block_x):
            block_idx = by * n_block_x + bx
            split = block_to_split.get(block_idx, "train")
            for x0, y0, parent_idx in _tile_grid_in_block(
                bx,
                by,
                cfg.parent_block_size,
                img_h,
                img_w,
                cfg.tile_size,
                cfg.stride,
            ):
                grid.append((x0, y0, parent_idx, split))
    n_tiles = len(grid)
    logger.info(f"[{slide}]   {n_tiles} tile origins generated")

    # Allocate zarrs
    images_z = zarr.create_array(
        store=str(out_dir / "images.zarr"),
        shape=(n_tiles, cfg.tile_size, cfg.tile_size),
        chunks=(1, cfg.tile_size, cfg.tile_size),
        dtype="uint16",
        compressors=_compressor(cfg.blosc_clevel),
        overwrite=True,
    )
    inst_z = zarr.create_array(
        store=str(out_dir / "instances.zarr"),
        shape=(n_tiles, cfg.tile_size, cfg.tile_size),
        chunks=(1, cfg.tile_size, cfg.tile_size),
        dtype="uint16",
        compressors=_compressor(cfg.blosc_clevel),
        overwrite=True,
    )

    # Iterate tiles
    label_rows: list[dict] = []
    manifest_rows: list[dict] = []
    t_raster = 0.0
    t_image = 0.0
    t_write = 0.0
    t0 = time.time()

    cells_in_train_tiles: set[str] = set()
    cells_in_val_tiles: set[str] = set()
    cells_in_test_tiles: set[str] = set()

    fine_arr = gdf["ct_tangram"].to_numpy()
    cell_id_arr = np.asarray(gdf.index.tolist())

    for tile_idx, (x0, y0, parent_idx, split) in enumerate(grid):
        # Rasterize: assign per-tile local instance ID = row order within tile
        ts = time.time()
        # We need a temp ID column equal to position-in-this-tile, so build a
        # filtered sub-gdf and assign IDs 1..n.
        cents_in_tile_mask = (
            (cents[:, 0] >= x0)
            & (cents[:, 0] < x0 + cfg.tile_size)
            & (cents[:, 1] >= y0)
            & (cents[:, 1] < y0 + cfg.tile_size)
        )
        in_tile_global_idx = np.where(cents_in_tile_mask)[0]
        # Filter cells whose split assignment matches the tile's parent-block
        # split. Cells outside split (e.g. small overlap due to centroid near
        # block boundary) get dropped — preserves the hard partition.
        cell_split_arr = gdf["split"].to_numpy()
        in_split_mask = cell_split_arr[in_tile_global_idx] == split
        in_tile_global_idx = in_tile_global_idx[in_split_mask]
        n_in_tile = len(in_tile_global_idx)

        if n_in_tile > 65535:
            # Should never happen with reasonable tile sizes; fail loudly.
            raise RuntimeError(
                f"Tile ({x0},{y0}) has {n_in_tile} > 65535 cells; uint16 cap exceeded"
            )

        if n_in_tile == 0:
            inst_map = np.zeros(
                (cfg.tile_size, cfg.tile_size), dtype=np.uint16
            )
            border_arr = np.zeros(0, dtype=bool)
        else:
            sub = gdf.iloc[in_tile_global_idx].copy()
            sub["_tile_id"] = np.arange(1, n_in_tile + 1, dtype=np.uint16)
            inst_map, border_arr, _ = rasterize_tile(
                sub,
                "_tile_id",
                x0=x0,
                y0=y0,
                tile_size=cfg.tile_size,
                pad=cfg.pad,
                centroids_px=cents[in_tile_global_idx],
            )
        t_raster += time.time() - ts

        # Image read
        ts = time.time()
        if n_ch == 1:
            img_tile = morpho[
                0, y0 : y0 + cfg.tile_size, x0 : x0 + cfg.tile_size
            ]
        else:
            img_tile = morpho[
                dapi_ch,
                y0 : y0 + cfg.tile_size,
                x0 : x0 + cfg.tile_size,
            ]
        t_image += time.time() - ts

        # Write
        ts = time.time()
        images_z[tile_idx] = img_tile.astype(np.uint16)
        inst_z[tile_idx] = inst_map
        t_write += time.time() - ts

        # Build label rows
        n_border = 0
        if n_in_tile > 0:
            for local_i, gi in enumerate(in_tile_global_idx):
                fine = fine_arr[gi]
                medium = TANGRAM_TO_MEDIUM.get(fine, "Unknown")
                broad = TANGRAM_TO_BROAD.get(fine, "Unknown")
                cy = float(cents[gi, 1] - y0)
                cx = float(cents[gi, 0] - x0)
                is_b = bool(border_arr[local_i])
                if is_b:
                    n_border += 1
                label_rows.append(
                    {
                        "tile_idx": tile_idx,
                        "instance_id": local_i + 1,  # uint16, 1-indexed
                        "cell_id": str(cell_id_arr[gi]),
                        "global_instance_id": int(
                            gdf["global_instance_id"].to_numpy()[gi]
                        ),
                        "fine": fine,
                        "medium": medium,
                        "broad": broad,
                        "cy_px": cy,
                        "cx_px": cx,
                        "is_border": is_b,
                    }
                )
                target = (
                    cells_in_train_tiles
                    if split == "train"
                    else cells_in_val_tiles
                    if split == "val"
                    else cells_in_test_tiles
                )
                target.add(str(cell_id_arr[gi]))

        manifest_rows.append(
            {
                "slide": slide,
                "tile_idx": tile_idx,
                "x0_px": x0,
                "y0_px": y0,
                "parent_block_idx": parent_idx,
                "split": split,
                "n_cells": n_in_tile,
                "n_border_cells": n_border,
                "dapi_channel": dapi_ch,
            }
        )

    elapsed = time.time() - t0
    logger.info(
        f"[{slide}]   built {n_tiles} tiles in {elapsed:.1f}s "
        f"(raster={t_raster:.1f}s, image={t_image:.1f}s, write={t_write:.1f}s)"
    )

    # Hard assert: no cell appears in tiles from more than one split
    leak_train_val = cells_in_train_tiles & cells_in_val_tiles
    leak_train_test = cells_in_train_tiles & cells_in_test_tiles
    leak_val_test = cells_in_val_tiles & cells_in_test_tiles
    if leak_train_val or leak_train_test or leak_val_test:
        raise AssertionError(
            f"[{slide}] split leakage detected: "
            f"train∩val={len(leak_train_val)}, "
            f"train∩test={len(leak_train_test)}, "
            f"val∩test={len(leak_val_test)}"
        )
    logger.success(f"[{slide}]   ✓ no cell-id leakage between splits")

    # Write parquets
    if label_rows:
        pl.DataFrame(label_rows).write_parquet(out_dir / "labels.parquet")
    else:
        # Empty schema
        pl.DataFrame(
            schema={
                "tile_idx": pl.Int64,
                "instance_id": pl.Int64,
                "cell_id": pl.Utf8,
                "global_instance_id": pl.Int64,
                "fine": pl.Utf8,
                "medium": pl.Utf8,
                "broad": pl.Utf8,
                "cy_px": pl.Float64,
                "cx_px": pl.Float64,
                "is_border": pl.Boolean,
            }
        ).write_parquet(out_dir / "labels.parquet")
    pl.DataFrame(manifest_rows).write_parquet(out_dir / "manifest.parquet")

    return {
        "slide": slide,
        "n_tiles": n_tiles,
        "n_cells_in_train_tiles": len(cells_in_train_tiles),
        "n_cells_in_val_tiles": len(cells_in_val_tiles),
        "n_cells_in_test_tiles": len(cells_in_test_tiles),
        "n_invalid_dropped": n_invalid_dropped,
        "n_unmapped_dropped": n_unmapped_dropped,
        "geom_load_s": geom_load_s,
        "raster_s": t_raster,
        "image_s": t_image,
        "write_s": t_write,
        "total_s": elapsed,
    }


def build_cache(cfg: TileCacheConfig) -> dict:
    """Build cache for all slides in cfg.slides. Writes top-level manifest.parquet."""
    cfg.cache_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for slide in cfg.slides:
        s = build_one_slide(slide, cfg)
        summaries.append(s)

    # Aggregate top-level manifest
    all_manifests = []
    for slide in cfg.slides:
        m = pl.read_parquet(cfg.cache_root / slide / "manifest.parquet")
        all_manifests.append(m)
    pl.concat(all_manifests).write_parquet(cfg.cache_root / "manifest.parquet")

    # Save build summary
    (cfg.cache_root / "build_summary.json").write_text(
        json.dumps(
            {
                "config": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in cfg.__dict__.items()
                },
                "summaries": summaries,
            },
            indent=2,
        )
    )
    return {"summaries": summaries}
