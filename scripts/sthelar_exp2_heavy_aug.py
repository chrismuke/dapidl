#!/usr/bin/env python3
"""Exp 2: Heavy augmentation training (LMDB-compatible).

The stock --heavy-aug CLI flag only works with zarr-backed datasets; our
multi-tissue dataset is LMDB-only. This script reuses the LmdbPatchDataset
but applies heavy albumentations augmentation (CLAHE, GaussNoise, elastic,
grid distortion, etc.) to the training patches only.

Usage:
    uv run python scripts/sthelar_exp2_heavy_aug.py \\
        --output pipeline_output/sthelar_exp2_heavy_aug --epochs 10
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import albumentations as A
import lmdb
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dapidl.models.classifier import CellTypeClassifier
from dapidl.training.losses import get_class_weights


def build_train_aug() -> A.Compose:
    """Heavy augmentation pipeline (matching get_heavy_augmentation_transforms)."""
    return A.Compose([
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2,
                           rotate_limit=30, border_mode=0, p=0.8),
        A.ElasticTransform(alpha=80, sigma=12, border_mode=0, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.GaussNoise(var_limit=(0.005, 0.02), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.2),
    ])


class LmdbPatchDatasetAug(Dataset):
    """LMDB patch dataset with optional heavy training-time augmentation."""

    def __init__(self, lmdb_path: Path, indices: np.ndarray, labels_all: np.ndarray,
                 train: bool = False):
        self.lmdb_path = str(lmdb_path)
        self.indices = indices
        self.labels = labels_all[indices]
        self.train = train
        self.aug = build_train_aug() if train else None
        self.env = None

    def _open(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False,
                             readahead=False, meminit=False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self.env is None:
            self._open()
        patch_idx = int(self.indices[i])
        key = struct.pack(">Q", patch_idx)
        with self.env.begin() as txn:
            value = txn.get(key)
        label = np.frombuffer(value[:8], dtype=np.int64)[0]
        if label < 0 or label > 10000:
            label = struct.unpack(">q", value[:8])[0]
        patch = np.frombuffer(value[8:], dtype=np.uint16).reshape(128, 128)
        img = patch.astype(np.float32) / 65535.0  # [0, 1]
        if self.train:
            # albumentations expects HWC uint8 or float; we use float in [0,1]
            img = img[..., np.newaxis]  # (H, W, 1)
            img = self.aug(image=img)["image"].squeeze(-1)  # back to (H, W)
        # ImageNet-style normalization (same as baseline)
        img = (img - 0.485) / 0.229
        return torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0), int(label)


def build_splits(data_dir: Path):
    labels = np.load(data_dir / "labels.npy")
    idx = np.arange(len(labels))
    train, temp = train_test_split(idx, train_size=0.7, random_state=42, stratify=labels)
    val, test = train_test_split(temp, train_size=0.5, random_state=42, stratify=labels[temp])
    return train, val, test, labels


def reconstruct_tissue_idx(data_dir: Path):
    with open(data_dir / "slide_stats.json") as f:
        slide_stats = json.load(f)
    tissues = []
    for _, info in slide_stats.items():
        tissues.extend([info["tissue"]] * info["patches_written"])
    tissue_names = sorted(set(tissues))
    tissue_idx = np.array([tissue_names.index(t) for t in tissues], dtype=np.int32)
    return tissue_idx, tissue_names


def evaluate(model, loader, device):
    model.train(False)
    y_true, y_pred = [], []
    loss_total, n = 0.0, 0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y_dev = y.to(device, non_blocking=True)
            logits = model(x)
            loss = ce(logits, y_dev)
            loss_total += float(loss) * x.size(0)
            n += x.size(0)
            y_true.append(y.numpy())
            y_pred.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return {
        "loss": loss_total / n,
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path("/mnt/work/datasets/derived/sthelar-multitissue-p128"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=6)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}, output={args.output}")

    with open(args.data_dir / "class_mapping.json") as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    class_names = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])]

    train_idx, val_idx, test_idx, labels = build_splits(args.data_dir)
    tissue_idx, tissue_names = reconstruct_tissue_idx(args.data_dir)
    logger.info(f"splits: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_ds = LmdbPatchDatasetAug(args.data_dir / "patches.lmdb", train_idx, labels, train=True)
    val_ds = LmdbPatchDatasetAug(args.data_dir / "patches.lmdb", val_idx, labels, train=False)
    test_ds = LmdbPatchDatasetAug(args.data_dir / "patches.lmdb", test_idx, labels, train=False)

    train_labels = labels[train_idx]
    class_w_capped = get_class_weights(
        train_labels, num_classes, method="inverse", max_weight_ratio=10.0,
    ).numpy()
    sample_w = class_w_capped[train_labels]
    sampler = WeightedRandomSampler(sample_w.tolist(), num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = CellTypeClassifier(
        num_classes=num_classes, backbone_name="efficientnetv2_rw_s",
        input_adapter="replicate", pretrained=True, dropout=0.3,
    ).to(device)

    class_weights = torch.from_numpy(class_w_capped).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    history = []
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train(True)
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        t0 = time.time()
        for bi, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss) * x.size(0)
            train_correct += int((logits.argmax(1) == y).sum())
            train_total += x.size(0)
            if bi % 500 == 0:
                logger.info(f"  epoch {epoch+1} [{bi}/{len(train_loader)}] "
                            f"loss={float(loss):.4f} acc={train_correct/max(1,train_total):.4f}")
        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        val_metrics = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0
        logger.info(f"epoch {epoch+1}/{args.epochs} "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_acc={val_metrics['accuracy']:.4f} "
                    f"val_f1={val_metrics['macro_f1']:.4f} "
                    f"time={epoch_time:.0f}s")
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"], "val_weighted_f1": val_metrics["weighted_f1"],
            "lr": optimizer.param_groups[0]["lr"], "time_s": epoch_time,
        })
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "hparams": {"num_classes": num_classes, "backbone_name": "efficientnetv2_rw_s",
                            "input_adapter": "replicate"},
                "epoch": epoch + 1,
                "metrics": {
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"], "val_weighted_f1": val_metrics["weighted_f1"],
                    "val_precision": 0.0, "val_recall": 0.0,
                    "epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
                },
            }, args.output / "best_model.pt")
            logger.info(f"  saved best model (val_f1={best_val_f1:.4f})")

    logger.info(f"training done in {(time.time()-t_start)/60:.1f} min, best val f1={best_val_f1:.4f}")
    with open(args.output / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load(args.output / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    tissue_test = tissue_idx[test_idx]

    p, r, f1s, support = precision_recall_fscore_support(
        test_metrics["y_true"], test_metrics["y_pred"],
        labels=list(range(num_classes)), zero_division=0,
    )
    per_class = {
        class_names[i]: {"precision": float(p[i]), "recall": float(r[i]),
                         "f1": float(f1s[i]), "support": int(support[i])}
        for i in range(num_classes)
    }
    per_tissue = {}
    for ti, tname in enumerate(tissue_names):
        mask = tissue_test == ti
        if mask.sum() < 10:
            continue
        per_tissue[tname] = {
            "n": int(mask.sum()),
            "accuracy": float((test_metrics["y_true"][mask] == test_metrics["y_pred"][mask]).mean()),
            "macro_f1": float(f1_score(test_metrics["y_true"][mask], test_metrics["y_pred"][mask],
                                       average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(test_metrics["y_true"][mask], test_metrics["y_pred"][mask],
                                          average="weighted", zero_division=0)),
        }

    (args.output / "analysis").mkdir(exist_ok=True)
    with open(args.output / "analysis" / "summary.json", "w") as f:
        json.dump({
            "experiment": "heavy-augmentation-lmdb",
            "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "best_epoch": int(ckpt["epoch"]), "best_val_macro_f1": float(best_val_f1),
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "class_names": class_names,
        }, f, indent=2)
    with open(args.output / "analysis" / "per_class_metrics.json", "w") as f:
        json.dump({"accuracy": test_metrics["accuracy"], "macro_f1": test_metrics["macro_f1"],
                   "weighted_f1": test_metrics["weighted_f1"], "per_class": per_class}, f, indent=2)
    with open(args.output / "analysis" / "per_tissue_metrics.json", "w") as f:
        json.dump(per_tissue, f, indent=2)
    np.savez_compressed(
        args.output / "analysis" / "predictions.npz",
        y_true=test_metrics["y_true"], y_pred=test_metrics["y_pred"],
        tissue_idx=tissue_test,
    )

    logger.info(f"TEST: acc={test_metrics['accuracy']:.4f} "
                f"macro_f1={test_metrics['macro_f1']:.4f} "
                f"weighted_f1={test_metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
