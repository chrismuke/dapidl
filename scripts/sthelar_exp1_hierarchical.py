#!/usr/bin/env python3
"""Exp 1: Hierarchical-lite training with auxiliary coarse-group head.

Standard 9-class fine CE loss plus an auxiliary 3-class coarse CE (immune /
epithelial / stromal). The coarse label is derived from the fine label at
batch time. Total loss = fine_CE + 0.3 * coarse_CE.

Motivation: force representations that respect the broad cell-type hierarchy.
Minority classes within a super-group (mast inside immune, pericyte inside
stromal) get an extra signal from the auxiliary head without competing
directly with the dominant 9-class CE.

Usage:
    uv run python scripts/sthelar_exp1_hierarchical.py \\
        --output pipeline_output/sthelar_exp1_hierarchical --epochs 10
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dapidl.models.backbone import create_backbone
from dapidl.models.classifier import SingleChannelAdapter
from dapidl.training.losses import get_class_weights


# 9 class indices per sthelar-multitissue-p128/class_mapping.json
# 0 macrophage, 1 fibroblast, 2 B cell, 3 T cell, 4 endothelial cell,
# 5 adipocyte, 6 mast cell, 7 epithelial cell, 8 pericyte
COARSE_MAP = {
    0: 0, 2: 0, 3: 0, 6: 0,              # immune: macrophage, B, T, mast
    7: 1,                                 # epithelial: epithelial cell
    1: 2, 4: 2, 5: 2, 8: 2,              # stromal: fibroblast, endothelial, adipocyte, pericyte
}
COARSE_NAMES = ["immune", "epithelial", "stromal"]
NUM_COARSE = 3


class LmdbPatchDataset(Dataset):
    def __init__(self, lmdb_path: Path, indices: np.ndarray, labels_all: np.ndarray):
        self.lmdb_path = str(lmdb_path)
        self.indices = indices
        self.labels = labels_all[indices]
        self.env = None

    def _open(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

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
        img = patch.astype(np.float32) / 65535.0
        img = (img - 0.485) / 0.229
        return torch.from_numpy(img).unsqueeze(0), int(label)


class HierarchicalLiteModel(nn.Module):
    def __init__(self, num_classes: int = 9, num_coarse: int = 3):
        super().__init__()
        self.input_adapter = SingleChannelAdapter(method="replicate")
        backbone, feat_dim = create_backbone("efficientnetv2_rw_s", pretrained=True, in_channels=3)
        self.backbone = backbone
        self.dropout = nn.Dropout(0.3)
        self.fine_head = nn.Linear(feat_dim, num_classes)
        self.coarse_head = nn.Linear(feat_dim, num_coarse)

    def forward(self, x):
        x = self.input_adapter(x)
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.fine_head(feat), self.coarse_head(feat)


def fine_to_coarse(labels: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(labels)
    for fine, coarse in COARSE_MAP.items():
        out[labels == fine] = coarse
    return out


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


def evaluate(model, loader, device, num_classes):
    model.train(False)
    y_true, y_pred = [], []
    loss_total, n = 0.0, 0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y_dev = y.to(device, non_blocking=True)
            fine_logits, _ = model(x)
            loss = ce(fine_logits, y_dev)
            loss_total += float(loss) * x.size(0)
            n += x.size(0)
            y_true.append(y.numpy())
            y_pred.append(fine_logits.argmax(1).cpu().numpy())
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
    ap.add_argument("--coarse-weight", type=float, default=0.3)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}, output={args.output}, coarse_weight={args.coarse_weight}")

    with open(args.data_dir / "class_mapping.json") as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    class_names = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])]

    train_idx, val_idx, test_idx, labels = build_splits(args.data_dir)
    tissue_idx, tissue_names = reconstruct_tissue_idx(args.data_dir)
    logger.info(f"splits: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_ds = LmdbPatchDataset(args.data_dir / "patches.lmdb", train_idx, labels)
    val_ds = LmdbPatchDataset(args.data_dir / "patches.lmdb", val_idx, labels)
    test_ds = LmdbPatchDataset(args.data_dir / "patches.lmdb", test_idx, labels)

    train_labels = labels[train_idx]
    # Sampler weights: use the same 10x-capped class weights as the loss
    # (uncapped inverse-frequency on rare classes reproduced the merscope_v2 mode collapse).
    fine_w_capped = get_class_weights(
        train_labels, num_classes, method="inverse", max_weight_ratio=10.0,
    ).numpy()
    sample_w = fine_w_capped[train_labels]
    sampler = WeightedRandomSampler(sample_w.tolist(), num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = HierarchicalLiteModel(num_classes=num_classes, num_coarse=NUM_COARSE).to(device)

    fine_w = get_class_weights(train_labels, num_classes, method="inverse",
                               max_weight_ratio=10.0).to(device)
    coarse_labels = np.vectorize(lambda x: COARSE_MAP[x])(train_labels)
    coarse_counts = np.bincount(coarse_labels, minlength=NUM_COARSE)
    coarse_weights = 1.0 / np.maximum(coarse_counts, 1)
    coarse_weights = np.minimum(coarse_weights, coarse_weights.min() * 10.0)
    coarse_weights = coarse_weights / coarse_weights.sum() * NUM_COARSE
    coarse_w = torch.FloatTensor(coarse_weights).to(device)

    fine_ce = nn.CrossEntropyLoss(weight=fine_w, label_smoothing=0.1)
    coarse_ce = nn.CrossEntropyLoss(weight=coarse_w, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    history = []
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train(True)
        running_fine, running_coarse, train_correct, train_total = 0.0, 0.0, 0, 0
        t0 = time.time()
        for bi, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_coarse = fine_to_coarse(y)
            fine_logits, coarse_logits = model(x)
            lf = fine_ce(fine_logits, y)
            lc = coarse_ce(coarse_logits, y_coarse)
            loss = lf + args.coarse_weight * lc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_fine += float(lf) * x.size(0)
            running_coarse += float(lc) * x.size(0)
            train_correct += int((fine_logits.argmax(1) == y).sum())
            train_total += x.size(0)
            if bi % 200 == 0:
                logger.info(f"  epoch {epoch+1} [{bi}/{len(train_loader)}] "
                            f"fine={float(lf):.4f} coarse={float(lc):.4f} "
                            f"acc={train_correct/max(1,train_total):.4f}")
        scheduler.step()
        train_fine = running_fine / train_total
        train_coarse = running_coarse / train_total
        train_acc = train_correct / train_total

        val_metrics = evaluate(model, val_loader, device, num_classes)
        epoch_time = time.time() - t0
        logger.info(f"epoch {epoch+1}/{args.epochs} "
                    f"train_fine={train_fine:.4f} train_coarse={train_coarse:.4f} "
                    f"train_acc={train_acc:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_acc={val_metrics['accuracy']:.4f} "
                    f"val_f1={val_metrics['macro_f1']:.4f} "
                    f"time={epoch_time:.0f}s")
        history.append({
            "epoch": epoch + 1, "train_fine_loss": train_fine, "train_coarse_loss": train_coarse,
            "train_acc": train_acc, "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "lr": optimizer.param_groups[0]["lr"], "time_s": epoch_time,
        })
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "hparams": {"num_classes": num_classes, "num_coarse": NUM_COARSE,
                            "backbone_name": "efficientnetv2_rw_s",
                            "input_adapter": "replicate",
                            "coarse_weight": args.coarse_weight},
                "epoch": epoch + 1,
                "metrics": {
                    "train_loss": train_fine + args.coarse_weight * train_coarse,
                    "train_acc": train_acc, "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_weighted_f1": val_metrics["weighted_f1"],
                    "val_precision": 0.0, "val_recall": 0.0,
                    "epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
                },
            }, args.output / "best_model.pt")
            logger.info(f"  saved best model (val_f1={best_val_f1:.4f})")

    logger.info(f"training done in {(time.time()-t_start)/60:.1f} min, best val f1={best_val_f1:.4f}")

    with open(args.output / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final test eval
    ckpt = torch.load(args.output / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, num_classes)
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
            "experiment": "hierarchical-lite",
            "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "best_epoch": int(ckpt["epoch"]), "best_val_macro_f1": float(best_val_f1),
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "class_names": class_names,
            "coarse_weight": args.coarse_weight,
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
