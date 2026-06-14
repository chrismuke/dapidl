#!/usr/bin/env python3
"""Cross-source DAPI evaluation: train-on-A → test-on-B for breast 4-class.

Loads a checkpoint trained on ONE LMDB (e.g., STHELAR-only or Xenium-only) and
evaluates it on a DIFFERENT LMDB at the SAME patch size. Computes macro F1 +
per-class metrics + confusion matrix on the entire test LMDB.

Used for the breast cross-source matrix:
  Direction A: STHELAR-trained → tested on Xenium rep1+rep2
  Direction B: Xenium-trained  → tested on STHELAR breast

Output: pipeline_output/breast_cross_source/<train>_to_<test>_p<size>.json

Usage:
    uv run python scripts/breast_cross_source_eval.py \\
        --train-source sthelar --test-source xenium --patch-size 128 \\
        --ckpt pipeline_output/breast_dapi_sthelar_p128/best_model.pt \\
        --eval-lmdb breast-multisource-dapi-p128
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset

from dapidl.models.backbone import create_backbone


DERIVED = Path("/mnt/work/datasets/derived")
OUT_BASE = Path("/mnt/work/git/dapidl/pipeline_output/breast_cross_source")
OUT_BASE.mkdir(parents=True, exist_ok=True)

DAPI_NORM_MEAN = 0.485
DAPI_NORM_STD = 0.229


class BreastPatchDataset(Dataset):
    def __init__(self, lmdb_dir: Path, indices: np.ndarray, patch_size: int):
        self.lmdb_path = lmdb_dir / "patches.lmdb"
        self.indices = indices
        self.patch_size = patch_size
        self.env: lmdb.Environment | None = None

    def _open(self):
        self.env = lmdb.open(str(self.lmdb_path), readonly=True,
                             lock=False, readahead=False, meminit=False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self.env is None:
            self._open()
        assert self.env is not None
        key = struct.pack(">Q", int(self.indices[i]))
        with self.env.begin() as txn:
            value = txn.get(key)
        label = np.frombuffer(value[:8], dtype=np.int64)[0]
        if label < 0 or label > 1000:
            label = struct.unpack(">q", value[:8])[0]
        patch = np.frombuffer(value[8:], dtype=np.uint16).reshape(
            self.patch_size, self.patch_size
        )
        img = patch.astype(np.float32) / 65535.0
        img = (img - DAPI_NORM_MEAN) / DAPI_NORM_STD
        return torch.from_numpy(img).unsqueeze(0), int(label)


class BreastClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone, feat_dim = create_backbone(
            "efficientnetv2_rw_s", pretrained=False, in_channels=3
        )
        self.backbone = backbone
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        x = x.expand(-1, 3, -1, -1)
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.head(feat)


def load_ckpt(model: nn.Module, ckpt_path: Path) -> nn.Module:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys: {missing}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-source", required=True,
                    choices=["sthelar", "xenium", "both"])
    ap.add_argument("--test-source", required=True,
                    choices=["sthelar", "xenium", "both"])
    ap.add_argument("--patch-size", type=int, required=True)
    ap.add_argument("--ckpt", required=True, help="Path to best_model.pt")
    ap.add_argument("--eval-lmdb", required=True,
                    help="LMDB dir name under /mnt/work/datasets/derived/")
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=6)
    args = ap.parse_args()

    eval_lmdb_dir = DERIVED / args.eval_lmdb
    meta = json.loads((eval_lmdb_dir / "metadata.json").read_text())
    class_names = meta["class_names"]
    print(f"Eval LMDB: {eval_lmdb_dir}  ({meta['n_samples']:,} samples, {len(class_names)} classes)")

    labels = np.load(eval_lmdb_dir / "labels.npy")
    indices = np.arange(len(labels))

    ds = BreastPatchDataset(eval_lmdb_dir, indices, args.patch_size)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BreastClassifier(num_classes=args.n_classes).to(device)
    model = load_ckpt(model, Path(args.ckpt))
    model.train(False)

    t0 = time.time()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for img, lab in loader:
            img = img.to(device, non_blocking=True)
            logits = model(img)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(lab.numpy())
    elapsed = time.time() - t0

    yt = np.concatenate(all_labels)
    yp = np.concatenate(all_preds)

    acc = float((yt == yp).mean())
    macro = float(f1_score(yt, yp, labels=list(range(args.n_classes)),
                            average="macro", zero_division=0))
    weighted = float(f1_score(yt, yp, labels=list(range(args.n_classes)),
                               average="weighted", zero_division=0))
    p, r, f1, support = precision_recall_fscore_support(
        yt, yp, labels=list(range(args.n_classes)), zero_division=0
    )
    cm = confusion_matrix(yt, yp, labels=list(range(args.n_classes)))

    per_class = {
        class_names[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(args.n_classes)
    }

    result = {
        "train_source": args.train_source,
        "test_source": args.test_source,
        "patch_size": args.patch_size,
        "ckpt": str(args.ckpt),
        "eval_lmdb": str(eval_lmdb_dir),
        "n_eval": int(len(yt)),
        "accuracy": acc,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "elapsed_s": round(elapsed, 1),
    }

    out_path = OUT_BASE / f"{args.train_source}_to_{args.test_source}_p{args.patch_size}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[{args.train_source} → {args.test_source} | p{args.patch_size}]")
    print(f"  macro F1: {macro:.4f}  acc: {acc:.4f}  weighted F1: {weighted:.4f}")
    print(f"  per-class F1: " + ", ".join(
        f"{c}={per_class[c]['f1']:.3f}" for c in class_names
    ))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
