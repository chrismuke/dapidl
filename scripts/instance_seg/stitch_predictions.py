"""Slide-level stitching + NMS for instance-segmentation predictions.

Phase D evaluation primary tool. Predicted instance maps from overlapping
tiles are stitched onto a full-slide canvas with IoU-based NMS to resolve
duplicate detections at tile boundaries.

Tile-prediction format (npz, one per slide):
- `tile_idx: (N_tiles,) int64`
- `x0_px: (N_tiles,) int64`, `y0_px: (N_tiles,) int64`
- `instance_maps: object array of (H, W) uint16 arrays per tile`
- `class_per_local_id: list[dict[int, int]]` per tile (saved as object array of dicts)
- `score_per_local_id: list[dict[int, float]]` per tile

Output:
- `stitched_mask (slide_h, slide_w) uint32` global IDs
- `class_per_global: dict[int, int]`
- `score_per_global: dict[int, float]`
"""

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger


def stitch_with_nms(
    tile_predictions: list[dict],
    slide_h: int,
    slide_w: int,
    iou_threshold: float = 0.5,
) -> dict:
    """Stitch overlapping tile predictions into a slide-level instance map.

    Greedy NMS over instances ranked by score (descending). Earlier (higher
    score) instances claim their pixels first; later instances that overlap
    a claimed instance by IoU ≥ threshold get dropped.

    Args:
        tile_predictions: list of dicts with `tile_idx`, `x0_px`, `y0_px`,
            `instance_map (H, W) uint16`, `class_per_instance: dict[int,int]`,
            `score_per_instance: dict[int,float]`.
        slide_h, slide_w: slide canvas size.
        iou_threshold: NMS overlap threshold.

    Returns:
        dict with `mask`, `class_per_instance`, `score_per_instance`,
        `n_predictions_total`, `n_predictions_kept`.
    """
    canvas = np.zeros((slide_h, slide_w), dtype=np.uint32)
    class_map: dict[int, int] = {}
    score_map: dict[int, float] = {}
    next_global_id = 1

    flat_preds: list[tuple[float, int, dict, np.ndarray]] = []
    for tp in tile_predictions:
        x0, y0 = int(tp["x0_px"]), int(tp["y0_px"])
        inst_map = np.asarray(tp["instance_map"], dtype=np.uint32)
        for local_id in np.unique(inst_map):
            if local_id == 0:
                continue
            score = float(tp["score_per_instance"].get(int(local_id), 0.0))
            mask = inst_map == local_id
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            flat_preds.append(
                (
                    score,
                    int(local_id),
                    {
                        "tile_idx": int(tp["tile_idx"]),
                        "x0_px": x0,
                        "y0_px": y0,
                        "class": int(tp["class_per_instance"].get(int(local_id), -1)),
                    },
                    mask,
                )
            )
    flat_preds.sort(key=lambda x: -x[0])

    n_total = len(flat_preds)
    n_kept = 0
    for score, local_id, meta, mask in flat_preds:
        x0, y0 = meta["x0_px"], meta["y0_px"]
        ys, xs = np.where(mask)
        gys = ys + y0
        gxs = xs + x0
        claimed = canvas[gys, gxs]
        already = claimed[claimed > 0]
        if len(already) > 0:
            unique, counts = np.unique(already, return_counts=True)
            best_count = counts.max()
            iou = best_count / max(len(gxs), 1)
            if iou >= iou_threshold:
                continue
        canvas[gys, gxs] = next_global_id
        class_map[next_global_id] = meta["class"]
        score_map[next_global_id] = score
        next_global_id += 1
        n_kept += 1

    return {
        "mask": canvas,
        "class_per_instance": class_map,
        "score_per_instance": score_map,
        "n_predictions_total": n_total,
        "n_predictions_kept": n_kept,
    }


def load_tile_predictions_npz(path: Path) -> list[dict]:
    """Load tile predictions from a single npz file produced by inference.

    Expected keys:
      - tile_idx (N,) int
      - x0_px (N,) int, y0_px (N,) int
      - instance_maps_<i>: (H, W) uint16 — saved per-tile
      - class_keys_<i>, class_values_<i>: int arrays for class lookup
      - score_keys_<i>, score_values_<i>: same for scores
    """
    data = np.load(path, allow_pickle=False)
    n = len(data["tile_idx"])
    out = []
    for i in range(n):
        ck = data[f"class_keys_{i}"]
        cv = data[f"class_values_{i}"]
        sk = data[f"score_keys_{i}"]
        sv = data[f"score_values_{i}"]
        out.append(
            {
                "tile_idx": int(data["tile_idx"][i]),
                "x0_px": int(data["x0_px"][i]),
                "y0_px": int(data["y0_px"][i]),
                "instance_map": data[f"instance_maps_{i}"],
                "class_per_instance": dict(zip(ck.tolist(), cv.tolist())),
                "score_per_instance": dict(zip(sk.tolist(), sv.tolist())),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions-npz", type=Path, required=True)
    ap.add_argument("--slide-h", type=int, required=True)
    ap.add_argument("--slide-w", type=int, required=True)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument(
        "--out", type=Path, default=Path("pipeline_output/instance_seg/stitched.npz")
    )
    args = ap.parse_args()

    tile_preds = load_tile_predictions_npz(args.predictions_npz)
    logger.info(f"loaded {len(tile_preds)} tile predictions")
    result = stitch_with_nms(
        tile_preds, args.slide_h, args.slide_w, args.iou_threshold
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        mask=result["mask"],
        class_keys=np.array(list(result["class_per_instance"].keys()), dtype=np.int64),
        class_values=np.array(
            list(result["class_per_instance"].values()), dtype=np.int64
        ),
        score_keys=np.array(list(result["score_per_instance"].keys()), dtype=np.int64),
        score_values=np.array(
            list(result["score_per_instance"].values()), dtype=np.float32
        ),
    )
    summary = {
        "n_predictions_total": result["n_predictions_total"],
        "n_predictions_kept": result["n_predictions_kept"],
        "iou_threshold": args.iou_threshold,
    }
    (args.out.with_suffix(".json")).write_text(json.dumps(summary, indent=2))
    logger.success(
        f"stitched {result['n_predictions_kept']}/{result['n_predictions_total']} → {args.out}"
    )


if __name__ == "__main__":
    main()
