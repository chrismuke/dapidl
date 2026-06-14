"""CelloType inference worker — runs inside /home/chrism/cellotype-env.

Invoked by `dapidl.training.instance.cellotype_subprocess.CellotypeSubprocessRunner`
via the launcher `/home/chrism/cellotype-env/bin/run_cellotype.sh`.

Why this lives in dapidl scripts/ but runs in a different venv:
    The two venvs cannot import each other (cellotype is py3.10 + torch 2.4.1,
    dapidl is py3.12 + torch 2.5+). This script intentionally has zero dapidl
    imports — it speaks pure numpy / PIL / cellotype. The I/O contract is
    npz files on disk.

I/O contract:
    --in-dir DIR    DIR contains `tile_<i>.npz` with key `image: (H, W) uint16` DAPI
                    or `image: (H, W, 3) uint8` RGB. uint16 DAPI is normalized via
                    percentile (1, 99) and replicated to 3 channels.
    --out-dir DIR   DIR will receive `pred_<i>.npz` with keys:
                        instance_map: (H, W) uint16  (0=bg, 1..N=instance)
                        class_per_instance_id: (N,) int32  (argmax class per instance)
                        score_per_instance_id: (N,) float32  (confidence)
                        n_instances: int
                    On error, writes `error_<i>.txt` with the traceback.

Usage (typically invoked via run_cellotype.sh):
    bash /home/chrism/cellotype-env/bin/run_cellotype.sh \\
        scripts/instance_seg/cellotype_worker.py \\
        --weights /path/tissuenet_model_0019999.pth \\
        --config /path/configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \\
        --in-dir /tmp/tile_in \\
        --out-dir /tmp/tile_out
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def _percentile_to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """uint16 DAPI (H, W) → uint8 RGB (H, W, 3) via p1/p99 normalization."""
    if image.ndim == 3 and image.shape[-1] == 3 and image.dtype == np.uint8:
        return image
    if image.ndim != 2:
        raise ValueError(f"unexpected image shape {image.shape}")
    p1, p99 = np.percentile(image, [1.0, 99.0])
    if p99 <= p1:
        return np.zeros((*image.shape, 3), dtype=np.uint8)
    norm = ((image.astype(np.float32) - p1) / (p99 - p1)).clip(0.0, 1.0)
    rgb = (norm * 255.0).astype(np.uint8)
    return np.repeat(rgb[..., None], 3, axis=-1)


def _build_predictor(weights: Path, config: Path, device: str = "cuda"):
    """Lazy import — heavy. Called once per worker invocation."""
    from cellotype.predict import CelloTypePredictor  # type: ignore

    model = CelloTypePredictor(
        model_path=str(weights),
        confidence_thresh=0.3,
        max_det=1000,
        device=device,
        config_path=str(config),
    )
    return model


def _extract_per_instance(model, instance_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pull class predictions and scores from the predictor's last call.

    The CelloType `predict()` returns the instance map only; class info is
    held on the underlying detectron2 instance. We re-run the predictor to
    capture full Instances by overriding the call. For now (pretrained
    tissuenet) the head is 5-class (Pannuke); after fine-tune it'll be 17.

    Returns:
        class_per_id: (n_instances,) int32, position i is the class for ID i+1.
        score_per_id: (n_instances,) float32.
    """
    n = int(instance_map.max())
    if n == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

    # detectron2 stores last predictions on model.predictor — peek via the
    # Visualizer-friendly API. This is best-effort; on failure we return
    # zeros (segmentation works regardless).
    try:
        last_outputs = getattr(model, "_last_outputs", None)
        if last_outputs is None:
            return np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.float32)
        instances = last_outputs.get("instances")
        if instances is None:
            return np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.float32)
        instances = instances.to("cpu")
        classes = instances.pred_classes.numpy().astype(np.int32)
        scores = instances.scores.numpy().astype(np.float32)
        # CelloType labels instances starting from 1 in the same order as
        # `instances`. If the array length disagrees with `n`, pad with zeros.
        if len(classes) >= n:
            return classes[:n], scores[:n]
        pad = n - len(classes)
        return (
            np.concatenate([classes, np.zeros(pad, dtype=np.int32)]),
            np.concatenate([scores, np.zeros(pad, dtype=np.float32)]),
        )
    except Exception:
        return np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.float32)


def _patch_predictor_to_capture(model) -> None:
    """Wrap `model.predictor.__call__` so we keep the last detectron2 outputs."""
    orig_predict = model.predict

    def new_predict(self, img):  # type: ignore[no-redef]
        rgb = img if (img.ndim == 3 and img.shape[-1] == 3) else _percentile_to_rgb_uint8(img)
        outputs = self.predictor(rgb)
        instances = outputs["instances"].to("cpu")
        confident = instances[instances.scores > self.confidence_thresh]
        self._last_outputs = {"instances": confident}
        # Reconstruct the instance_map in the same way upstream's predict does.
        masks = confident.pred_masks.numpy().copy()  # (N, H, W) bool
        instance_map = np.zeros(rgb.shape[:2], dtype=np.uint16)
        for i in range(masks.shape[0]):
            instance_map[masks[i, :, :] == True] = i + 1  # noqa: E712
        return instance_map

    import types

    model.predict = types.MethodType(new_predict, model)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tile_files = sorted(args.in_dir.glob("tile_*.npz"))
    if not tile_files:
        print(f"no tile_*.npz in {args.in_dir}", file=sys.stderr)
        return 2

    t0 = time.time()
    model = _build_predictor(args.weights, args.config, args.device)
    _patch_predictor_to_capture(model)
    load_s = time.time() - t0
    print(f"model loaded in {load_s:.1f}s", file=sys.stderr)

    n_ok = 0
    n_err = 0
    for tile_path in tile_files:
        idx = tile_path.stem.split("_", 1)[-1]
        out_path = args.out_dir / f"pred_{idx}.npz"
        try:
            data = np.load(tile_path)
            image = data["image"]
            t0 = time.time()
            instance_map = model.predict(image)  # patched: captures outputs
            classes, scores = _extract_per_instance(model, instance_map)
            np.savez_compressed(
                out_path,
                instance_map=instance_map.astype(np.uint16),
                class_per_instance_id=classes,
                score_per_instance_id=scores,
                n_instances=int(instance_map.max()),
            )
            elapsed = time.time() - t0
            print(
                f"  {tile_path.name} → {out_path.name} "
                f"({int(instance_map.max())} instances, {elapsed:.2f}s)",
                file=sys.stderr,
            )
            n_ok += 1
        except Exception:
            err_path = args.out_dir / f"error_{idx}.txt"
            err_path.write_text(traceback.format_exc())
            print(f"  {tile_path.name}: ERROR → {err_path.name}", file=sys.stderr)
            n_err += 1

    print(f"DONE: {n_ok} ok, {n_err} errors", file=sys.stderr)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
