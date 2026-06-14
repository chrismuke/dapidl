"""Build the STHELAR breast tile cache for instance-segmentation training.

CLI wrapper over `dapidl.data.sthelar_tile_cache.build_cache`.

Usage:
    uv run python scripts/instance_seg/build_tile_cache.py --slides breast_s0 breast_s1 breast_s3 breast_s6
"""

import argparse
from pathlib import Path

from loguru import logger

from dapidl.data.sthelar_tile_cache import TileCacheConfig, build_cache


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--slides",
        nargs="+",
        default=["breast_s0", "breast_s1", "breast_s3", "breast_s6"],
    )
    ap.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/mnt/work/datasets/derived/sthelar_breast_tiles"),
    )
    ap.add_argument(
        "--sthelar-root",
        type=Path,
        default=Path("/mnt/work/datasets/STHELAR/sdata_slides"),
    )
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--pad", type=int, default=64)
    ap.add_argument("--parent-block-size", type=int, default=4096)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blosc-clevel", type=int, default=3)
    args = ap.parse_args()

    cfg = TileCacheConfig(
        cache_root=args.cache_root,
        sthelar_root=args.sthelar_root,
        slides=args.slides,
        tile_size=args.tile_size,
        stride=args.stride,
        pad=args.pad,
        parent_block_size=args.parent_block_size,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        blosc_clevel=args.blosc_clevel,
    )
    logger.info(f"Cache root: {cfg.cache_root}")
    logger.info(f"Slides: {cfg.slides}")
    logger.info(
        f"Tile {cfg.tile_size}² @ stride {cfg.stride}, parent block {cfg.parent_block_size}²"
    )

    result = build_cache(cfg)

    print()
    print("=" * 80)
    print(f"{'slide':<14} {'n_tiles':>8} {'train':>10} {'val':>10} {'test':>10}")
    print("-" * 80)
    for s in result["summaries"]:
        print(
            f"{s['slide']:<14} {s['n_tiles']:>8} "
            f"{s['n_cells_in_train_tiles']:>10} "
            f"{s['n_cells_in_val_tiles']:>10} "
            f"{s['n_cells_in_test_tiles']:>10}"
        )
    print("=" * 80)
    logger.success(f"build summary → {cfg.cache_root}/build_summary.json")


if __name__ == "__main__":
    main()
