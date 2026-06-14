#!/usr/bin/env python3
"""Focused DAPI training for the breast patch-size sweep.

Single-modality, single-LMDB, 4-class coarse training. Mirrors the architecture
and hyperparameters of `sthelar_modality_train.py` but stripped of multi-tissue
infrastructure (no tissue index, no HE intersection, no LOTO).

Usage:
    uv run python scripts/breast_dapi_train.py \\
        --lmdb breast-multisource-dapi-p32 \\
        --patch-size 32 \\
        --output pipeline_output/breast_dapi_p32 \\
        --epochs 21
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
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dapidl.models.backbone import create_backbone
from dapidl.training.losses import get_class_weights


DERIVED = Path("/mnt/work/datasets/derived")

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
            "efficientnetv2_rw_s", pretrained=True, in_channels=3
        )
        self.backbone = backbone
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        x = x.expand(-1, 3, -1, -1)
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.head(feat)


def stratified_split(labels: np.ndarray, seed: int = 42):
    """70/15/15 stratified split, returns (train_idx, val_idx, test_idx)."""
    n = len(labels)
    idx = np.arange(n)
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, stratify=labels, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=labels[temp_idx], random_state=seed
    )
    return train_idx, val_idx, test_idx


def evaluate_split(model, loader, device):
    """Forward-pass over a loader; return (preds, labels)."""
    model.train(False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for img, lab in loader:
            img = img.to(device, non_blocking=True)
            logits = model(img)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(lab.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", required=True, help="LMDB dir name under /mnt/work/datasets/derived/")
    ap.add_argument("--patch-size", type=int, required=True)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--output", required=True, help="Output dir for analysis/best_model.pt")
    ap.add_argument("--epochs", type=int, default=21)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lmdb_dir = DERIVED / args.lmdb
    out_dir = Path(args.output)
    (out_dir / "analysis").mkdir(parents=True, exist_ok=True)

    meta = json.loads((lmdb_dir / "metadata.json").read_text())
    class_names = meta["class_names"]
    n_classes = len(class_names)
    if n_classes != args.classes:
        print(f"  WARN: --classes={args.classes} but metadata has {n_classes}, using {n_classes}")

    labels = np.load(lmdb_dir / "labels.npy")
    print(f"Total samples: {len(labels):,}, classes: {n_classes}")

    train_idx, val_idx, test_idx = stratified_split(labels, seed=args.seed)
    print(f"Split: train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    train_ds = BreastPatchDataset(lmdb_dir, train_idx, args.patch_size)
    val_ds = BreastPatchDataset(lmdb_dir, val_idx, args.patch_size)
    test_ds = BreastPatchDataset(lmdb_dir, test_idx, args.patch_size)

    # Class weights with cap
    train_labels = labels[train_idx]
    class_weights = get_class_weights(train_labels, n_classes, max_weight_ratio=10.0)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # Sampler for balanced training
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_idx), replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = BreastClassifier(num_classes=n_classes).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    weighted_loss = nn.CrossEntropyLoss(
        weight=weights_tensor.to(device), label_smoothing=0.1
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    patience_left = args.patience

    for epoch in range(args.epochs):
        model.train(True)
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for img, lab in train_loader:
            img = img.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(img)
            loss = weighted_loss(logits, lab)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        val_preds, val_labels = evaluate_split(model, val_loader, device)
        val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        val_acc = float((val_preds == val_labels).mean())
        elapsed = time.time() - t0
        print(
            f"ep{epoch:02d} loss={avg_loss:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} val_acc={val_acc:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} ({elapsed:.0f}s)"
        )
        history.append({
            "epoch": epoch, "loss": avg_loss,
            "val_macro_f1": float(val_macro_f1), "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"], "elapsed_s": round(elapsed, 1),
        })

        if val_macro_f1 > best_val_f1:
            best_val_f1 = float(val_macro_f1)
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}")
                break

    # Reload best model for test
    model.load_state_dict(
        torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    test_preds, test_labels = evaluate_split(model, test_loader, device)
    test_acc = float((test_preds == test_labels).mean())
    test_macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(test_labels, test_preds, average="weighted", zero_division=0)
    p, r, f1, support = precision_recall_fscore_support(
        test_labels, test_preds, labels=list(range(n_classes)), zero_division=0
    )
    per_class = {
        class_names[i]: {
            "precision": float(p[i]), "recall": float(r[i]),
            "f1": float(f1[i]), "support": int(support[i]),
        }
        for i in range(n_classes)
    }

    summary = {
        "experiment": f"breast_dapi_p{args.patch_size}",
        "lmdb": str(lmdb_dir),
        "patch_size": args.patch_size,
        "n_classes": n_classes,
        "class_names": class_names,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_f1),
        "test_accuracy": test_acc,
        "test_macro_f1": float(test_macro_f1),
        "test_weighted_f1": float(test_weighted_f1),
        "per_class": per_class,
    }
    (out_dir / "analysis" / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "analysis" / "per_class_metrics.json").write_text(
        json.dumps({"per_class": per_class}, indent=2)
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nFinal: test_macro_f1={test_macro_f1:.4f} acc={test_acc:.4f}")
    print(f"Wrote {out_dir / 'analysis' / 'summary.json'}")


if __name__ == "__main__":
    main()
