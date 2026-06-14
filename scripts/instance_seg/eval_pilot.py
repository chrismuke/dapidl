"""Phase D pilot evaluation: aggregate metrics for the held-out test slide.

Combines:
- `dapidl.benchmark.instance_metrics` (panopticPQ, segPQ, AJI+, AP, F1, CM)
- `scripts.instance_seg.stitch_predictions` (slide-level stitching)

Inputs:
- Tile-level predictions written during inference (one npz per slide / method).
- Tile-level GT from the cache.

Outputs:
- `pipeline_output/instance_seg/{run}/comparison.parquet`
- `pipeline_output/instance_seg/{run}/confusion_{method}.npy`
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import zarr
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from stitch_predictions import (  # noqa: E402
    load_tile_predictions_npz,
    stitch_with_nms,
)

from dapidl.benchmark.instance_metrics import (
    aji_plus,
    average_precision,
    confusion_matrix,
    panoptic_quality,
    per_class_f1,
    segmentation_pq,
)


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


def load_gt_for_slide(
    cache_root: Path,
    slide: str,
    tile_size: int = 1024,
    gt_split: str | None = None,
    pred_tile_idx: list[int] | None = None,
) -> tuple[np.ndarray, dict[int, int]]:
    """Build a slide-level GT mask + class lookup from the cache.

    Args:
        cache_root: tile-cache root.
        slide: slide name (e.g. "breast_s3").
        tile_size: tile spatial dim.
        gt_split: if set ("train"|"val"|"test"), only that split's tiles
            contribute to GT. Use this to scope GT to the same tiles the
            predictions cover; otherwise PR/PQ are dominated by tiles that
            were never predicted.
        pred_tile_idx: if set, further restrict GT to only these tile indices.
            Use for sub-split smoke tests.
    """
    sdir = cache_root / slide
    manifest = pl.read_parquet(sdir / "manifest.parquet")
    if gt_split is not None:
        manifest = manifest.filter(pl.col("split") == gt_split)
    if pred_tile_idx is not None:
        manifest = manifest.filter(pl.col("tile_idx").is_in(pred_tile_idx))
    labels = pl.read_parquet(sdir / "labels.parquet")
    inst_z = zarr.open(str(sdir / "instances.zarr"), mode="r")

    h_max = int((manifest["y0_px"].max() or 0) + tile_size)
    w_max = int((manifest["x0_px"].max() or 0) + tile_size)
    gt_mask = np.zeros((h_max, w_max), dtype=np.uint32)
    class_lookup: dict[int, int] = {}
    medium_to_idx = {n: i for i, n in enumerate(MEDIUM_CLASSES)}

    for row in manifest.iter_rows(named=True):
        tile_idx = int(row["tile_idx"])
        x0, y0 = int(row["x0_px"]), int(row["y0_px"])
        local_inst = np.asarray(inst_z[tile_idx], dtype=np.uint32)
        tile_labels = labels.filter(pl.col("tile_idx") == tile_idx)
        for lr in tile_labels.iter_rows(named=True):
            local_id = int(lr["instance_id"])
            global_id = int(lr["global_instance_id"])
            class_lookup[global_id] = medium_to_idx.get(lr["medium"], -1)
            mask = local_inst == local_id
            sub = gt_mask[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]]
            sub[mask] = global_id
    return gt_mask, class_lookup


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--test-slide", default="breast_s6")
    ap.add_argument("--predictions", type=Path, action="append", default=[])
    ap.add_argument("--method", action="append", default=[])
    ap.add_argument("--n-classes", type=int, default=10)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument(
        "--gt-split",
        default=None,
        choices=[None, "train", "val", "test"],
        help="If set, scope GT to only this split (matches predict_test_slide.py --split)",
    )
    ap.add_argument(
        "--scope-gt-to-preds",
        action="store_true",
        help="Restrict GT to only tile_idx values present in the predictions",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("pipeline_output/instance_seg/eval_pilot")
    )
    args = ap.parse_args()

    if len(args.predictions) != len(args.method):
        raise SystemExit("--predictions and --method counts must match")
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for method, pred_path in zip(args.method, args.predictions):
        logger.info(f"--- {method} ---")
        tile_preds = load_tile_predictions_npz(pred_path)
        pred_tile_idx = (
            [tp["tile_idx"] for tp in tile_preds]
            if args.scope_gt_to_preds
            else None
        )
        logger.info(f"loading GT for {args.test_slide}")
        gt_mask, gt_classes = load_gt_for_slide(
            args.cache_root,
            args.test_slide,
            gt_split=args.gt_split,
            pred_tile_idx=pred_tile_idx,
        )
        logger.info(
            f"  GT mask shape={gt_mask.shape}, {len(gt_classes)} instances"
            f"{' (scoped to predictions)' if args.scope_gt_to_preds else ''}"
        )
        stitched = stitch_with_nms(
            tile_preds, gt_mask.shape[0], gt_mask.shape[1], args.iou_threshold
        )
        pred_mask = stitched["mask"]
        pred_classes = stitched["class_per_instance"]
        pred_scores = stitched["score_per_instance"]

        seg = segmentation_pq(pred_mask, gt_mask, args.iou_threshold)
        pan = panoptic_quality(
            pred_mask, gt_mask, pred_classes, gt_classes, args.n_classes,
            args.iou_threshold,
        )
        aji = aji_plus(pred_mask, gt_mask)
        ap50 = average_precision(
            pred_mask, gt_mask, pred_classes, gt_classes, pred_scores,
            args.n_classes, iou_thresholds=(args.iou_threshold,),
        )
        f1 = per_class_f1(
            pred_mask, gt_mask, pred_classes, gt_classes, args.n_classes,
            args.iou_threshold,
        )
        cm = confusion_matrix(
            pred_mask, gt_mask, pred_classes, gt_classes, args.n_classes,
            args.iou_threshold,
        )
        np.save(args.out / f"confusion_{method}.npy", cm)

        rows.append(
            {
                "method": method,
                "panoptic_pq": pan["pq_mean"],
                "segmentation_pq": seg["pq"],
                "aji_plus": aji,
                "AP@0.5": ap50.get("AP@0.5", 0.0),
                "f1_macro_medium": f1["f1_macro"],
                "n_pred_kept": stitched["n_predictions_kept"],
                "n_pred_total": stitched["n_predictions_total"],
            }
        )
        logger.info(
            f"  panoptic_pq={pan['pq_mean']:.4f}, "
            f"seg_pq={seg['pq']:.4f}, aji+={aji:.4f}, "
            f"AP@0.5={ap50.get('AP@0.5', 0.0):.4f}, "
            f"f1_macro={f1['f1_macro']:.4f}"
        )

    df = pl.DataFrame(rows)
    df.write_parquet(args.out / "comparison.parquet")
    print(df)
    logger.success(f"eval pilot complete → {args.out}")


if __name__ == "__main__":
    main()
