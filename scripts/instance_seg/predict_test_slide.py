"""Walk test-split tiles through the CelloType subprocess and write the
tile-prediction npz format that `eval_pilot.py` / `stitch_predictions.py`
expect.

Output schema (one npz per slide, matches stitch_predictions.load_tile_predictions_npz):
    tile_idx: (N,) int64
    x0_px: (N,) int64
    y0_px: (N,) int64
    instance_maps_<i>: (1024, 1024) uint16
    class_keys_<i>: (n_inst,) int64
    class_values_<i>: (n_inst,) int64
    score_keys_<i>: (n_inst,) int64
    score_values_<i>: (n_inst,) float32

Usage:
    uv run python scripts/instance_seg/predict_test_slide.py \\
        --cache-root /mnt/work/datasets/derived/sthelar_breast_tiles_test \\
        --slide breast_s3 --split test \\
        --out pipeline_output/instance_seg/preds/breast_s3_cellotype.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import zarr
from loguru import logger

from dapidl.training.instance.cellotype_subprocess import CellotypeSubprocessRunner

DEFAULT_WEIGHTS = Path(
    "/home/chrism/git/CelloType/models/tissuenet_model_0019999.pth"
)
DEFAULT_CONFIG = Path(
    "/home/chrism/git/CelloType/configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--slide", required=True)
    ap.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "all"],
        help="Filter manifest by split (default test)",
    )
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, only predict the first N tiles (for smoke testing)")
    ap.add_argument("--non-empty-only", action="store_true",
                    help="Skip tiles with n_cells=0 in manifest")
    ap.add_argument("--batch-size", type=int, default=200,
                    help="Tiles per subprocess invocation (amortizes 3.5s startup)")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sdir = args.cache_root / args.slide
    if not sdir.exists():
        raise SystemExit(f"slide cache missing: {sdir}")

    manifest = pl.read_parquet(sdir / "manifest.parquet")
    if args.split != "all":
        manifest = manifest.filter(pl.col("split") == args.split)
    if args.non_empty_only:
        manifest = manifest.filter(pl.col("n_cells") > 0)
    if args.limit:
        manifest = manifest.head(args.limit)
    n = len(manifest)
    if n == 0:
        raise SystemExit(f"no tiles after filter: split={args.split}, slide={args.slide}")
    logger.info(f"predicting {n} tiles from {args.slide} split={args.split}")

    img_z = zarr.open(str(sdir / "images.zarr"), mode="r")

    runner = CellotypeSubprocessRunner(
        weights=args.weights,
        config=args.config,
        device=args.device,
    )

    # Iterate in batches
    all_payload: dict = {
        "tile_idx": np.empty(n, dtype=np.int64),
        "x0_px": np.empty(n, dtype=np.int64),
        "y0_px": np.empty(n, dtype=np.int64),
    }
    failures = 0
    t_start = time.time()

    rows = list(manifest.iter_rows(named=True))
    for batch_start in range(0, n, args.batch_size):
        batch_end = min(batch_start + args.batch_size, n)
        batch_rows = rows[batch_start:batch_end]
        batch_imgs = [
            np.asarray(img_z[int(r["tile_idx"])]) for r in batch_rows
        ]
        logger.info(
            f"  batch {batch_start}-{batch_end}: {len(batch_imgs)} tiles"
        )
        t0 = time.time()
        preds = runner.predict_tiles(batch_imgs)
        logger.info(f"  → {time.time()-t0:.1f}s")

        for offset, (row, pred) in enumerate(zip(batch_rows, preds)):
            i = batch_start + offset
            all_payload["tile_idx"][i] = int(row["tile_idx"])
            all_payload["x0_px"][i] = int(row["x0_px"])
            all_payload["y0_px"][i] = int(row["y0_px"])
            if pred.get("instance_map") is None:
                logger.warning(
                    f"  tile {row['tile_idx']}: ERROR ({pred.get('error', '?')[:120]})"
                )
                failures += 1
                # Write empty mask + empty dicts so eval_pilot can still load
                all_payload[f"instance_maps_{i}"] = np.zeros((1024, 1024), dtype=np.uint16)
                all_payload[f"class_keys_{i}"] = np.zeros(0, dtype=np.int64)
                all_payload[f"class_values_{i}"] = np.zeros(0, dtype=np.int64)
                all_payload[f"score_keys_{i}"] = np.zeros(0, dtype=np.int64)
                all_payload[f"score_values_{i}"] = np.zeros(0, dtype=np.float32)
                continue
            inst_map = pred["instance_map"]
            n_inst = int(pred["n_instances"])
            class_keys = np.arange(1, n_inst + 1, dtype=np.int64)
            class_values = pred["class_per_instance_id"].astype(np.int64)
            score_keys = class_keys
            score_values = pred["score_per_instance_id"].astype(np.float32)
            all_payload[f"instance_maps_{i}"] = inst_map.astype(np.uint16)
            all_payload[f"class_keys_{i}"] = class_keys
            all_payload[f"class_values_{i}"] = class_values
            all_payload[f"score_keys_{i}"] = score_keys
            all_payload[f"score_values_{i}"] = score_values

    elapsed = time.time() - t_start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **all_payload)
    logger.success(
        f"wrote {n} tile predictions in {elapsed:.0f}s ({failures} failures) → {args.out}"
    )


if __name__ == "__main__":
    main()
