"""Attribution readout (D): Integrated-Gradients concentration inside the nucleus
on a balanced rep2 subset. Writes saliency_summary.json + a few overlay PNGs.

IG uses the EXACT training transform (/65535 -> standardize); segmentation/area
use the raw patch via the QC scorer.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from breast_pooled_train import DapiClassifier  # noqa: E402  (import-safe)
from starpose.qc.segmentation_grounded import (  # noqa: E402
    SegmentationGroundedScorer,
    SegQCConfig,
    select_center_nucleus,
)
from subnuclear_common import balanced_subset  # noqa: E402

from dapidl.qc.attribution import (  # noqa: E402
    attribution_concentration,
    fraction_in_mask,
    integrated_gradients,
)

LMDB_DIR = Path("/mnt/work/datasets/derived/breast-6source-dapi-p128")
CKPT = Path("pipeline_output/h2h_2026_05_30/efficientnetv2_rw_s/best_model.pt")
CLASS_NAMES = ["Endothelial", "Epithelial", "Immune", "Stromal"]
NORM_MEAN, NORM_STD = 0.485, 0.229


def _read_patch(txn, idx: int) -> np.ndarray:
    return np.frombuffer(txn.get(struct.pack(">Q", int(idx)))[8:],
                         dtype=np.uint16).reshape(128, 128)


def _to_input(patch: np.ndarray, device) -> torch.Tensor:
    img = (patch.astype(np.float32) / 65535.0 - NORM_MEAN) / NORM_STD
    return torch.from_numpy(img)[None, None].to(device)


def _save_overlay(path, patch, mask, attr, true, pred, conc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(patch, cmap="gray")
    ax[0].set_title("DAPI")
    ax[0].axis("off")
    ax[0].contour(find_boundaries(mask), colors="lime", linewidths=0.6)
    ax[1].imshow(np.abs(attr), cmap="magma")
    ax[1].axis("off")
    ax[1].contour(find_boundaries(mask), colors="lime", linewidths=0.6)
    ax[1].set_title(f"|IG|  conc={conc:.2f}")
    fig.suptitle(f"true={CLASS_NAMES[true]} pred={CLASS_NAMES[pred]}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _iqr(a):
    a = np.asarray(a, float)
    return [float(np.percentile(a, 25)), float(np.percentile(a, 75))] if a.size else [float("nan")] * 2


def _summarize(records, names, path, args):
    conc = np.array([r["concentration"] for r in records], float)
    summary = {
        "n": len(records), "per_class": args.per_class, "ig_steps": args.steps,
        "checkpoint": str(CKPT),
        "overall": {"mean_concentration": float(conc.mean()) if conc.size else float("nan"),
                    "iqr": _iqr(conc),
                    "mean_fraction_in_mask": float(np.mean([r["fraction_in_mask"] for r in records])) if records else float("nan")},
        "by_class": {}, "by_correct": {},
    }
    for c, n in enumerate(names):
        cc = np.array([r["concentration"] for r in records if r["label"] == c], float)
        summary["by_class"][n] = {"n": int(cc.size),
                                  "mean_concentration": float(cc.mean()) if cc.size else float("nan"),
                                  "iqr": _iqr(cc)}
    for ok in (True, False):
        cc = np.array([r["concentration"] for r in records if r["correct"] == ok], float)
        summary["by_correct"]["correct" if ok else "incorrect"] = {
            "n": int(cc.size), "mean_concentration": float(cc.mean()) if cc.size else float("nan")}
    Path(path).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["overall"], indent=2))
    print(f"[saliency] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=750)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-overlays", type=int, default=12)
    ap.add_argument("--out-dir", default="pipeline_output/subnuclear_2026_06")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    labels = np.load(LMDB_DIR / "labels.npy")
    # allow_pickle=True: sources.npy contains string object-dtype arrays written by our own
    # pipeline (breast_pooled_train / build_lmdb). It is read from a controlled local path
    # (/mnt/work/datasets/derived/…) and never sourced from untrusted input.
    sources = np.load(LMDB_DIR / "sources.npy", allow_pickle=True)
    rep2 = np.where((sources == "xenium_rep2") & (labels != -1))[0]
    sub_local = balanced_subset(labels[rep2], args.per_class, seed=args.seed)
    subset = rep2[sub_local]

    model = DapiClassifier(4, backbone="efficientnetv2_rw_s").to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()

    scorer = SegmentationGroundedScorer(SegQCConfig(), gpu=not args.cpu, pixel_size=0.2125)
    cfg = scorer.cfg
    env = lmdb.open(str(LMDB_DIR / "patches.lmdb"), readonly=True, lock=False,
                    readahead=False, meminit=False)

    records = []
    overlays_left = args.n_overlays
    with env.begin() as txn:
        for gi in subset:
            patch = _read_patch(txn, int(gi))
            masks, probs = scorer._segment(patch)
            cn = select_center_nucleus(masks, probs, cfg)
            if cn is None:
                continue
            x = _to_input(patch, device)
            with torch.no_grad():
                pred = int(model(x).argmax(1).item())
            attr = integrated_gradients(model, x, pred, steps=args.steps)
            attr_hw = attr.squeeze().detach().cpu().numpy()
            area_frac = float(cn.mask.sum()) / cn.mask.size
            frac = fraction_in_mask(attr_hw, cn.mask)
            conc = attribution_concentration(attr_hw, cn.mask, area_frac)
            true = int(labels[gi])
            records.append({"label": true, "pred": pred, "correct": pred == true,
                            "area_fraction": area_frac, "fraction_in_mask": frac,
                            "concentration": conc})
            if overlays_left > 0:
                _save_overlay(out / "overlays" / f"ex_{int(gi)}.png",
                              patch, cn.mask, attr_hw, true, pred, conc)
                overlays_left -= 1

    _summarize(records, CLASS_NAMES, out / "saliency_summary.json", args)


if __name__ == "__main__":
    main()
