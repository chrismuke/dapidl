#!/usr/bin/env python3
"""Generic test-set evaluation for any STHELAR multi-tissue experiment.

Reconstructs the trainer's test split, loads the checkpoint, computes
per-class and per-tissue metrics, writes predictions and JSON artifacts.

Usage:
    uv run python scripts/sthelar_exp_analysis.py --model-dir PATH [--data-dir PATH] [--min-samples N]

The data dir defaults to the multi-tissue LMDB. Pass --min-samples to
reproduce the 7-class-filtered split.
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
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class LmdbPatchDataset(Dataset):
    def __init__(self, lmdb_path: Path, indices: np.ndarray, patch_size: int = 128):
        self.lmdb_path = str(lmdb_path)
        self.indices = indices
        self.patch_size = patch_size
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
        if value is None:
            raise KeyError(f"missing key {patch_idx}")
        label = np.frombuffer(value[:8], dtype=np.int64)[0]
        if label < 0 or label > 10000:
            label = struct.unpack(">q", value[:8])[0]
        patch = np.frombuffer(value[8:], dtype=np.uint16).reshape(self.patch_size, self.patch_size)
        img = (patch.astype(np.float32) / 65535.0)
        img = (img - 0.485) / 0.229
        return torch.from_numpy(img).unsqueeze(0), int(label)


def reconstruct_splits(data_dir: Path, min_samples: int | None = None):
    """Reproduce trainer 70/15/15 stratified split with seed 42."""
    labels_all = np.load(data_dir / "labels.npy")
    n_total = len(labels_all)
    indices = np.arange(n_total)

    if min_samples is not None:
        unique, counts = np.unique(labels_all, return_counts=True)
        valid = unique[counts >= min_samples]
        mask = np.isin(labels_all, valid)
        indices = indices[mask]
        labels_kept = labels_all[mask]
    else:
        labels_kept = labels_all

    train_idx, temp_idx = train_test_split(
        indices, train_size=0.7, random_state=42, stratify=labels_kept
    )
    temp_labels = labels_all[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=0.5, random_state=42, stratify=temp_labels
    )
    return train_idx, val_idx, test_idx, labels_all


def reconstruct_tissue_idx(data_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Reconstruct per-patch tissue name from slide_stats cumsum order."""
    with open(data_dir / "slide_stats.json") as f:
        slide_stats = json.load(f)
    tissues = []
    for _slide_name, info in slide_stats.items():
        tissues.extend([info["tissue"]] * info["patches_written"])
    tissue_names = sorted(set(tissues))
    tissue_to_idx = {t: i for i, t in enumerate(tissue_names)}
    return np.array([tissue_to_idx[t] for t in tissues], dtype=np.int32), tissue_names


def load_model(model_dir: Path, device):
    from dapidl.models.classifier import CellTypeClassifier

    ckpt = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    model = CellTypeClassifier(
        num_classes=ckpt["hparams"]["num_classes"],
        backbone_name=ckpt["hparams"]["backbone_name"],
        input_adapter=ckpt["hparams"].get("input_adapter", "replicate"),
        pretrained=False,
        dropout=0.3,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.train(False)
    return model, ckpt


def evaluate(model, loader, device):
    y_true, y_pred, y_prob = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for b, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            pred = prob.argmax(axis=1)
            y_true.append(y.numpy())
            y_pred.append(pred)
            y_prob.append(prob)
            if b % 50 == 0:
                print(f"  batch {b}/{len(loader)}  elapsed={time.time()-t0:.1f}s")
    return (
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(y_prob),
        time.time() - t0,
    )


def compute_metrics_with_labels(y_true, y_pred, class_names, kept_ids):
    """Compute metrics restricted to `kept_ids` original class indices."""
    acc = float((y_true == y_pred).mean())
    macro_f1 = f1_score(y_true, y_pred, labels=kept_ids, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=kept_ids, average="weighted", zero_division=0)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=kept_ids, zero_division=0,
    )
    per_class = {
        class_names[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(kept_ids))
    }
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


def compute_metrics(y_true, y_pred, class_names, num_classes):  # legacy
    return compute_metrics_with_labels(y_true, y_pred, class_names, list(range(num_classes)))


def compute_per_tissue(y_true, y_pred, tissue_test, tissue_names):
    per_tissue = {}
    for ti, tname in enumerate(tissue_names):
        mask = tissue_test == ti
        if mask.sum() < 10:
            continue
        per_tissue[tname] = {
            "n": int(mask.sum()),
            "accuracy": float((y_true[mask] == y_pred[mask]).mean()),
            "macro_f1": float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true[mask], y_pred[mask], average="weighted", zero_division=0)),
        }
    return per_tissue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/mnt/work/datasets/derived/sthelar-multitissue-p128"),
    )
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument(
        "--holdout-tissue", type=str, default=None,
        help="If set, test set is just this tissue (for LOTO eval)",
    )
    args = ap.parse_args()

    analysis_dir = args.model_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, model_dir={args.model_dir}, data_dir={args.data_dir}")

    with open(args.data_dir / "class_mapping.json") as f:
        class_mapping = json.load(f)

    # Always work in the ORIGINAL 9-class index space. When min_samples is used,
    # the model still has 9 output heads but 2 classes had no training data.
    # We evaluate with scikit-learn `labels=kept_ids` to restrict metrics to the
    # classes we care about while still allowing the model to predict any of 9.
    idx_to_name_full = {v: k for k, v in class_mapping.items()}
    all_num_classes = len(class_mapping)

    if args.min_samples is not None:
        labels_all = np.load(args.data_dir / "labels.npy")
        unique, counts = np.unique(labels_all, return_counts=True)
        kept_ids = sorted(unique[counts >= args.min_samples].tolist())
        class_names = [idx_to_name_full[i] for i in kept_ids]
        print(f"min_samples={args.min_samples} -> evaluating on {len(kept_ids)} kept classes: {class_names}")
    else:
        kept_ids = list(range(all_num_classes))
        class_names = [idx_to_name_full[i] for i in kept_ids]
    num_classes = all_num_classes

    train_idx, val_idx, test_idx, labels_all = reconstruct_splits(args.data_dir, args.min_samples)
    tissue_idx, tissue_names = reconstruct_tissue_idx(args.data_dir)

    if args.holdout_tissue is not None:
        mask = tissue_idx == tissue_names.index(args.holdout_tissue)
        test_idx = np.where(mask)[0]
        print(f"holdout_tissue={args.holdout_tissue}: test_idx has {len(test_idx)} patches")

    print(f"splits: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    model, ckpt = load_model(args.model_dir, device)
    print(f"loaded best checkpoint (epoch {ckpt['epoch']}, val_f1={ckpt['metrics'].get('val_macro_f1', 'n/a')})")

    ds = LmdbPatchDataset(args.data_dir / "patches.lmdb", test_idx)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=6, pin_memory=True)

    y_true, y_pred, y_prob, elapsed = evaluate(model, loader, device)
    tissue_test = tissue_idx[test_idx]
    print(f"inference done in {elapsed:.1f}s, n_test={len(y_true)}")

    # Filter test set to samples whose true label is one of the kept classes
    # (for min_samples case; otherwise all samples are kept).
    keep_mask = np.isin(y_true, kept_ids)
    y_true = y_true[keep_mask]
    y_pred = y_pred[keep_mask]
    y_prob = y_prob[keep_mask]
    tissue_test = tissue_test[keep_mask]

    np.savez_compressed(
        analysis_dir / "predictions.npz",
        y_true=y_true, y_pred=y_pred, y_prob=y_prob.astype(np.float16), tissue_idx=tissue_test,
    )

    metrics = compute_metrics_with_labels(y_true, y_pred, class_names, kept_ids)
    per_tissue = compute_per_tissue(y_true, y_pred, tissue_test, tissue_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    with open(analysis_dir / "per_class_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(analysis_dir / "per_tissue_metrics.json", "w") as f:
        json.dump(per_tissue, f, indent=2)
    with open(analysis_dir / "confusion_matrix.json", "w") as f:
        json.dump({"counts": cm.tolist(), "normalized": cm_norm.tolist(), "classes": class_names}, f, indent=2)
    with open(analysis_dir / "classification_report.txt", "w") as fr:
        fr.write(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

    summary = {
        "model_dir": str(args.model_dir),
        "data_dir": str(args.data_dir),
        "min_samples": args.min_samples,
        "holdout_tissue": args.holdout_tissue,
        "n_test": int(len(y_true)),
        "n_classes": num_classes,
        "class_names": class_names,
        "best_epoch": int(ckpt["epoch"]),
        "val_macro_f1": float(ckpt["metrics"].get("val_macro_f1", float("nan"))),
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_weighted_f1": metrics["weighted_f1"],
    }
    with open(analysis_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}")
    print(f"outputs in {analysis_dir}")


if __name__ == "__main__":
    main()
