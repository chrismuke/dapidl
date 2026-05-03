#!/usr/bin/env python3
"""Data-scale learning curve on STHELAR breast s0 (DAPI, p128).

Fixed val + test (15% + 15%, seed 42 stratified). Training pool = remaining 70%.
Subsample the training pool to {5, 10, 25, 50, 100}% (stratified) to trace the
test-F1 vs N_train curve. All other hyperparameters identical to the
sthelar_modality_train baseline.

Usage:
    uv run python scripts/sthelar_data_scale.py --fraction 0.05 \\
        --output pipeline_output/data_scale_2026_05/frac_005

Run all five fractions with scripts/run_data_scale.sh.
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
from loguru import logger
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dapidl.models.backbone import create_backbone
from dapidl.training.losses import get_class_weights


DATA_DIR = Path("/mnt/work/datasets/derived/sthelar-breast_s0-finegrained-p128")
DAPI_NORM_MEAN = 0.485
DAPI_NORM_STD = 0.229


class DapiPatchDataset(Dataset):
    def __init__(self, indices: np.ndarray):
        self.indices = indices
        self.env = None

    def _open(self):
        self.env = lmdb.open(str(DATA_DIR / "patches.lmdb"), readonly=True,
                             lock=False, readahead=False, meminit=False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self.env is None:
            self._open()
        key = struct.pack(">Q", int(self.indices[i]))
        with self.env.begin() as txn:
            value = txn.get(key)
        label = np.frombuffer(value[:8], dtype=np.int64)[0]
        if label < 0 or label > 10000:
            label = struct.unpack(">q", value[:8])[0]
        patch = np.frombuffer(value[8:], dtype=np.uint16).reshape(128, 128)
        img = patch.astype(np.float32) / 65535.0
        img = (img - DAPI_NORM_MEAN) / DAPI_NORM_STD
        return torch.from_numpy(img).unsqueeze(0), int(label)


class DapiClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone, feat_dim = create_backbone("efficientnetv2_rw_s", pretrained=True, in_channels=3)
        self.backbone = backbone
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        x = x.expand(-1, 3, -1, -1)
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.head(feat)


def build_fixed_splits(num_samples: int, labels: np.ndarray):
    """70/15/15 stratified, seed 42 — identical to baseline pipeline."""
    idx = np.arange(num_samples)
    train, temp = train_test_split(idx, train_size=0.7, random_state=42, stratify=labels)
    temp_labels = labels[temp]
    val, test = train_test_split(temp, train_size=0.5, random_state=42, stratify=temp_labels)
    return train, val, test


def stratified_subsample(train_idx: np.ndarray, labels_all: np.ndarray,
                         fraction: float, seed: int = 42) -> np.ndarray:
    """Stratified subsample preserving class proportions."""
    if fraction >= 1.0:
        return train_idx
    train_labels = labels_all[train_idx]
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(train_labels):
        cls_idx = train_idx[train_labels == cls]
        n_keep = max(1, int(round(len(cls_idx) * fraction)))
        chosen = rng.choice(cls_idx, size=n_keep, replace=False)
        keep.append(chosen)
    return np.sort(np.concatenate(keep))


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
    ap.add_argument("--fraction", type=float, required=True,
                    help="Fraction of the train pool to use (0 < f <= 1)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(DATA_DIR / "class_mapping.json") as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    class_names = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])]

    labels_all = np.load(DATA_DIR / "labels.npy")
    n_total = len(labels_all)
    train_pool, val_idx, test_idx = build_fixed_splits(n_total, labels_all)
    train_idx = stratified_subsample(train_pool, labels_all, args.fraction, seed=args.seed)

    logger.info(f"fraction={args.fraction} → train={len(train_idx):,} "
                f"(pool {len(train_pool):,}) val={len(val_idx):,} test={len(test_idx):,}")

    train_class_counts = {
        class_names[c]: int((labels_all[train_idx] == c).sum())
        for c in range(num_classes)
    }
    logger.info(f"per-class train counts: {train_class_counts}")

    train_ds = DapiPatchDataset(train_idx)
    val_ds = DapiPatchDataset(val_idx)
    test_ds = DapiPatchDataset(test_idx)

    train_labels = labels_all[train_idx]
    class_w = get_class_weights(train_labels, num_classes,
                                method="inverse", max_weight_ratio=10.0).numpy()
    sample_w = class_w[train_labels]
    sampler = WeightedRandomSampler(sample_w.tolist(),
                                    num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = DapiClassifier(num_classes=num_classes).to(device)
    class_weights = torch.from_numpy(class_w).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history = []
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train(True)
        train_loss, train_correct, train_total = 0.0, 0, 0
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
        scheduler.step()
        train_loss /= max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        val_metrics = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0
        logger.info(
            f"epoch {epoch+1}/{args.epochs} frac={args.fraction} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} time={epoch_time:.0f}s"
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "elapsed_s": epoch_time,
        })

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.output / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info(f"early stop after {args.patience} stagnant epochs")
                break

    model.load_state_dict(torch.load(args.output / "best_model.pt"))
    test_metrics = evaluate(model, test_loader, device)
    p, r, f, sup = precision_recall_fscore_support(
        test_metrics["y_true"], test_metrics["y_pred"],
        labels=list(range(num_classes)), zero_division=0
    )
    per_class = [
        {
            "class_name": class_names[c],
            "precision": float(p[c]),
            "recall": float(r[c]),
            "f1": float(f[c]),
            "support": int(sup[c]),
        }
        for c in range(num_classes)
    ]

    summary = {
        "fraction": args.fraction,
        "n_train": int(len(train_idx)),
        "n_train_pool": int(len(train_pool)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "num_classes": num_classes,
        "class_names": class_names,
        "train_class_counts": train_class_counts,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "per_class": per_class,
        "wall_clock_s": time.time() - t_start,
    }
    with open(args.output / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    with open(args.output / "history.json", "w") as fp:
        json.dump(history, fp, indent=2)

    logger.success(
        f"frac={args.fraction} test_macro_f1={test_metrics['macro_f1']:.4f} "
        f"acc={test_metrics['accuracy']:.4f} best_epoch={best_epoch} "
        f"wall={summary['wall_clock_s']:.0f}s"
    )


if __name__ == "__main__":
    main()
