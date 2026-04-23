#!/usr/bin/env python3
"""Evaluate best_model.pt on the STHELAR multi-tissue test split and produce figures.

Runs stratified 70/15/15 split (seed=42) to exactly reproduce the training split,
loads the best checkpoint, predicts on the held-out test set, and writes:

- predictions.npz      (y_true, y_pred, y_prob, tissue_idx)
- per_class_metrics.json
- per_tissue_metrics.json
- confusion_matrix.json
- figures/*.png        (all plots for the Obsidian report)
"""
from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import lmdb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path("/mnt/work/datasets/derived/sthelar-multitissue-p128")
MODEL_DIR = Path("/mnt/work/git/dapidl/pipeline_output/sthelar_multitissue_9class")
ANALYSIS_DIR = MODEL_DIR / "analysis"
FIG_DIR = Path("/home/chrism/obsidian/llmbrain/DAPIDL/multitissue_report/figures")

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


class LmdbPatchDataset(Dataset):
    def __init__(self, lmdb_path: Path, indices: np.ndarray, labels: np.ndarray, patch_size: int = 128):
        self.lmdb_path = str(lmdb_path)
        self.indices = indices
        self.labels = labels
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    with open(DATA_DIR / "class_mapping.json") as f:
        class_mapping = json.load(f)
    with open(DATA_DIR / "slide_stats.json") as f:
        slide_stats = json.load(f)

    num_classes = meta["n_classes"]
    idx_to_name = {v: k for k, v in class_mapping.items()}
    class_names = [idx_to_name[i] for i in range(num_classes)]
    labels_all = np.load(DATA_DIR / "labels.npy")
    n_total = len(labels_all)
    print(f"n_total={n_total}, n_classes={num_classes}")
    print(f"class_names: {class_names}")

    tissues = []
    for slide_name, info in slide_stats.items():
        tissues.extend([info["tissue"]] * info["patches_written"])
    assert len(tissues) == n_total, f"tissue count mismatch {len(tissues)} vs {n_total}"
    tissue_names = sorted(set(tissues))
    tissue_to_idx = {t: i for i, t in enumerate(tissue_names)}
    tissue_idx = np.array([tissue_to_idx[t] for t in tissues], dtype=np.int32)

    indices = np.arange(n_total)
    train_idx, temp_idx = train_test_split(
        indices, train_size=0.7, random_state=42, stratify=labels_all
    )
    temp_labels = labels_all[temp_idx]
    val_size = 0.15 / (0.15 + 0.15)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_size, random_state=42, stratify=temp_labels
    )
    print(f"splits: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    ckpt = torch.load(MODEL_DIR / "best_model.pt", map_location=device, weights_only=False)
    from dapidl.models.classifier import CellTypeClassifier

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
    print(f"loaded best checkpoint (epoch {ckpt['epoch']}, val_f1={ckpt['metrics']['val_macro_f1']:.4f})")

    test_ds = LmdbPatchDataset(DATA_DIR / "patches.lmdb", test_idx, labels_all)
    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=6, pin_memory=True)

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
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    tissue_test = tissue_idx[test_idx]
    print(f"inference done in {time.time()-t0:.1f}s, n_test={len(y_true)}")

    np.savez_compressed(
        ANALYSIS_DIR / "predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob.astype(np.float16),
        tissue_idx=tissue_test,
    )

    acc = float((y_true == y_pred).mean())
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    per_class = {
        class_names[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(num_classes)
    }
    with open(ANALYSIS_DIR / "per_class_metrics.json", "w") as f:
        json.dump({"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "per_class": per_class}, f, indent=2)
    print(f"accuracy={acc:.4f} macro_f1={macro_f1:.4f} weighted_f1={weighted_f1:.4f}")

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
    with open(ANALYSIS_DIR / "per_tissue_metrics.json", "w") as f:
        json.dump(per_tissue, f, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    with open(ANALYSIS_DIR / "confusion_matrix.json", "w") as f:
        json.dump({"counts": cm.tolist(), "normalized": cm_norm.tolist(), "classes": class_names}, f, indent=2)

    with open(ANALYSIS_DIR / "classification_report.txt", "w") as fr:
        fr.write(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

    sns.set_style("whitegrid")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 10})

    counts = np.array([meta["class_counts"][c] for c in class_names])
    order = np.argsort(-counts)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette("viridis", n_colors=num_classes)
    bars = ax.bar(range(num_classes), counts[order], color=[colors[i] for i in order])
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels([class_names[i] for i in order], rotation=30, ha="right")
    ax.set_ylabel("patches")
    ax.set_title(f"Class distribution — {n_total:,} total patches (9 cell types)")
    ax.set_yscale("log")
    for bar, c in zip(bars, counts[order]):
        ax.text(bar.get_x() + bar.get_width() / 2, c * 1.05, f"{c:,}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_class_distribution.png", bbox_inches="tight")
    plt.close(fig)

    tissues_sorted = sorted(meta["tissue_counts"].items(), key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("mako", n_colors=len(tissues_sorted))
    tnames = [t for t, _ in tissues_sorted]
    tcounts = [c for _, c in tissues_sorted]
    ax.bar(tnames, tcounts, color=colors)
    ax.set_xticklabels(tnames, rotation=45, ha="right")
    ax.set_ylabel("patches")
    ax.set_title(f"Tissue distribution — {meta['n_tissues']} tissues across {meta['n_slides']} slides")
    for i, c in enumerate(tcounts):
        ax.text(i, c * 1.01, f"{c:,}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_tissue_distribution.png", bbox_inches="tight")
    plt.close(fig)

    tissue_class = np.zeros((len(tissue_names), num_classes), dtype=np.int64)
    for i in range(n_total):
        tissue_class[tissue_idx[i], labels_all[i]] += 1
    tissue_class_norm = tissue_class / np.maximum(tissue_class.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        tissue_class_norm, annot=tissue_class_norm > 0.02, fmt=".2f",
        xticklabels=class_names, yticklabels=tissue_names,
        cmap="rocket_r", cbar_kws={"label": "fraction of tissue"}, ax=ax,
    )
    ax.set_title("Cell-type composition by tissue (row-normalized)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_tissue_class_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "recall"}, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Test-set confusion matrix (n={len(y_true):,})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    class_order = np.argsort(-np.array([per_class[c]["f1"] for c in class_names]))
    fig, ax = plt.subplots(figsize=(9, 5))
    f1_vals = [per_class[class_names[i]]["f1"] for i in class_order]
    sup_vals = [per_class[class_names[i]]["support"] for i in class_order]
    bars = ax.bar(range(num_classes), f1_vals,
                  color=[plt.cm.RdYlGn(v) for v in f1_vals])
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels([class_names[i] for i in class_order], rotation=30, ha="right")
    ax.set_ylabel("test F1")
    ax.set_ylim(0, 1)
    ax.axhline(macro_f1, color="k", linestyle="--", alpha=0.6, label=f"macro F1 = {macro_f1:.3f}")
    ax.set_title("Per-class F1 on test set")
    for bar, v, s in zip(bars, f1_vals, sup_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}\nn={s:,}",
                ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_per_class_f1.png", bbox_inches="tight")
    plt.close(fig)

    t_items = sorted(per_tissue.items(), key=lambda x: -x[1]["macro_f1"])
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [t for t, _ in t_items]
    f1s = [d["macro_f1"] for _, d in t_items]
    accs = [d["accuracy"] for _, d in t_items]
    x = np.arange(len(names))
    w = 0.4
    ax.bar(x - w / 2, f1s, w, label="macro F1", color="#1f77b4")
    ax.bar(x + w / 2, accs, w, label="accuracy", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Per-tissue test performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_per_tissue_performance.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    rows = [
        ("best epoch", str(ckpt["epoch"])),
        ("train loss", f"{ckpt['metrics']['train_loss']:.4f}"),
        ("train acc", f"{ckpt['metrics']['train_acc']:.4f}"),
        ("val loss", f"{ckpt['metrics']['val_loss']:.4f}"),
        ("val acc", f"{ckpt['metrics']['val_acc']:.4f}"),
        ("val macro F1", f"{ckpt['metrics']['val_macro_f1']:.4f}"),
        ("val weighted F1", f"{ckpt['metrics']['val_weighted_f1']:.4f}"),
        ("test acc", f"{acc:.4f}"),
        ("test macro F1", f"{macro_f1:.4f}"),
        ("test weighted F1", f"{weighted_f1:.4f}"),
        ("lr @ best", f"{ckpt['metrics']['lr']:.2e}"),
    ]
    tbl = ax.table(cellText=rows, colLabels=["metric", "value"], loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.6)
    ax.set_title("Training summary (best checkpoint, epoch 6)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_training_summary.png", bbox_inches="tight")
    plt.close(fig)

    print("figures written to", FIG_DIR)
    print("metrics written to", ANALYSIS_DIR)


if __name__ == "__main__":
    main()
