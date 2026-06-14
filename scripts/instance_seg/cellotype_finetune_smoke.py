"""CelloType fine-tune smoke test — runs in /home/chrism/cellotype-env (py3.10).

Loads pre-exported smoke tiles + ground-truth (export_smoke_tiles.py output),
builds a 10-class STHELAR MaskDINO head on top of the TissueNet backbone, and
runs N training iterations to measure:

    - Per-iter wall-clock time
    - Peak GPU memory
    - Loss trajectory (sanity: should decrease)

Output: pipeline_output/instance_seg/finetune/smoke_<run>.json with metrics.

Invocation (always via the launcher to set LD_LIBRARY_PATH/CUDA_HOME):
    bash /home/chrism/cellotype-env/bin/run_cellotype.sh \\
        scripts/instance_seg/cellotype_finetune_smoke.py \\
        --tiles pipeline_output/instance_seg/finetune/smoke_tiles_s3.npz \\
        --weights /home/chrism/git/CelloType/models/tissuenet_model_0019999.pth \\
        --config /home/chrism/git/CelloType/configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \\
        --n-iters 20 --batch-size 1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


N_CLASSES = 10  # MEDIUM tier


def _build_model(config_path: Path, weights_path: Path, n_classes: int, device: str = "cuda"):
    """Build MaskDINO with a fresh n_classes head, warm-start backbone from weights."""
    from cellotype.maskdino.config import add_maskdino_config
    from cellotype.trainer import Trainer
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(str(config_path))

    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = n_classes
    cfg.MODEL.IN_CHANS = 3
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.SOLVER.AMP.ENABLED = True
    cfg.MODEL.DEVICE = device
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.MAX_ITER = 1
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.OUTPUT_DIR = "pipeline_output/instance_seg/finetune/_smoke_tmp"
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    cfg.freeze()

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    return model, cfg


def _percentile_to_rgb_uint8(image_u16: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(image_u16, [1.0, 99.0])
    if p99 <= p1:
        return np.zeros((*image_u16.shape, 3), dtype=np.uint8)
    n = ((image_u16.astype(np.float32) - p1) / (p99 - p1)).clip(0.0, 1.0)
    rgb = (n * 255.0).astype(np.uint8)
    return np.repeat(rgb[..., None], 3, axis=-1)


def _build_batched_input(
    image_u16: np.ndarray,
    instance_map: np.ndarray,
    iids: np.ndarray,
    fine_idx: np.ndarray,
    device: torch.device,
) -> dict:
    """Convert one tile + GT into Detectron2's batched-input dict."""
    from detectron2.structures import Boxes, Instances

    rgb = _percentile_to_rgb_uint8(image_u16)
    image_chw = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float()

    h, w = instance_map.shape
    if len(iids) == 0:
        instances = Instances((h, w))
        instances.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
        instances.gt_classes = torch.zeros(0, dtype=torch.int64)
        instances.gt_masks = torch.zeros((0, h, w), dtype=torch.bool)
        return {"image": image_chw, "instances": instances, "height": h, "width": w}

    masks_bool = np.zeros((len(iids), h, w), dtype=bool)
    boxes = np.zeros((len(iids), 4), dtype=np.float32)
    valid = []
    for i, iid in enumerate(iids):
        m = instance_map == int(iid)
        if not m.any():
            continue
        masks_bool[i] = m
        ys, xs = np.where(m)
        boxes[i] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        valid.append(i)
    if not valid:
        instances = Instances((h, w))
        instances.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
        instances.gt_classes = torch.zeros(0, dtype=torch.int64)
        instances.gt_masks = torch.zeros((0, h, w), dtype=torch.bool)
        return {"image": image_chw, "instances": instances, "height": h, "width": w}

    valid_arr = np.array(valid)
    masks_bool = masks_bool[valid_arr]
    boxes = boxes[valid_arr]
    classes = fine_idx[valid_arr].astype(np.int64)

    instances = Instances((h, w))
    instances.gt_boxes = Boxes(torch.from_numpy(boxes))
    instances.gt_classes = torch.from_numpy(classes)
    # CelloType MaskDINO expects gt_masks as a raw tensor (N, H, W), not BitMasks.
    instances.gt_masks = torch.from_numpy(masks_bool)
    return {"image": image_chw, "instances": instances, "height": h, "width": w}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--n-iters", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("pipeline_output/instance_seg/finetune/smoke.json"))
    args = ap.parse_args()

    print("=" * 60, flush=True)
    print(f"CelloType STHELAR fine-tune SMOKE", flush=True)
    print(f"  tiles file: {args.tiles}", flush=True)
    print(f"  iterations: {args.n_iters}, batch_size: {args.batch_size}", flush=True)
    print("=" * 60, flush=True)

    data = np.load(args.tiles)
    images = data["images"]
    instance_maps = data["instance_maps"]
    offsets = data["per_tile_offsets"]
    flat_iid = data["per_tile_iid"]
    flat_fine = data["per_tile_fine_idx"]
    n_tiles = len(images)
    print(f"loaded {n_tiles} tiles, {flat_iid.size} GT instances total", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, GPU: {torch.cuda.get_device_name(0)}", flush=True)
    torch.cuda.reset_peak_memory_stats()

    print(f"building model with {N_CLASSES}-class head (warm-start from TissueNet)...", flush=True)
    t0 = time.time()
    model, cfg = _build_model(args.config, args.weights, N_CLASSES, device=str(device))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=0.05)
    print(f"  model + optim built in {time.time()-t0:.1f}s", flush=True)

    inputs_all: list[dict] = []
    t0 = time.time()
    for k in range(n_tiles):
        inputs_all.append(
            _build_batched_input(
                images[k],
                instance_maps[k],
                flat_iid[offsets[k] : offsets[k + 1]],
                flat_fine[offsets[k] : offsets[k + 1]],
                device,
            )
        )
    print(f"  built {n_tiles} input dicts in {time.time()-t0:.1f}s", flush=True)

    scaler = torch.amp.GradScaler("cuda") if cfg.SOLVER.AMP.ENABLED else None

    iter_times: list[float] = []
    losses: list[float] = []
    rss_peaks: list[int] = []

    rng = np.random.default_rng(0)
    for it in range(args.n_iters):
        idxs = rng.choice(n_tiles, size=args.batch_size, replace=False).tolist()
        batch = [inputs_all[i] for i in idxs]

        torch.cuda.reset_peak_memory_stats()
        t_iter = time.time()
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss_dict = model(batch)
                losses_t = sum(v for k, v in loss_dict.items() if "loss" in k.lower())
            scaler.scale(losses_t).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(batch)
            losses_t = sum(v for k, v in loss_dict.items() if "loss" in k.lower())
            losses_t.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.time() - t_iter

        peak_mb = torch.cuda.max_memory_allocated() // (1024**2)
        loss_val = float(losses_t.detach().item())
        iter_times.append(elapsed)
        losses.append(loss_val)
        rss_peaks.append(peak_mb)
        print(
            f"  iter {it:3d}: loss={loss_val:.3f}  time={elapsed*1000:.0f}ms  "
            f"GPU peak={peak_mb} MiB  batch_idxs={idxs}",
            flush=True,
        )

    if not iter_times:
        print("no iterations completed", flush=True)
        return 1

    summary = {
        "n_iters": args.n_iters,
        "batch_size": args.batch_size,
        "n_tiles_in_smoke": int(n_tiles),
        "n_classes": N_CLASSES,
        "amp_enabled": bool(cfg.SOLVER.AMP.ENABLED),
        "iter_time_ms_mean": float(np.mean(iter_times) * 1000),
        "iter_time_ms_median": float(np.median(iter_times) * 1000),
        "iter_time_ms_min": float(np.min(iter_times) * 1000),
        "iter_time_ms_max": float(np.max(iter_times) * 1000),
        "gpu_peak_mib_max": int(np.max(rss_peaks)),
        "gpu_peak_mib_mean": int(np.mean(rss_peaks)),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": float(min(losses)),
        "iter_times_ms": [float(t * 1000) for t in iter_times],
        "losses": [float(l) for l in losses],
        "gpu_peaks_mib": rss_peaks,
        "weights": str(args.weights),
        "config": str(args.config),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print("=" * 60, flush=True)
    print(f"SUMMARY", flush=True)
    print(f"  iter_time_ms median: {summary['iter_time_ms_median']:.0f}", flush=True)
    print(f"  GPU peak MiB max: {summary['gpu_peak_mib_max']}", flush=True)
    print(f"  loss first → last: {summary['loss_first']:.3f} → {summary['loss_last']:.3f}", flush=True)
    print(f"  → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
