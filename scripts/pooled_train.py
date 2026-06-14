"""Cross-source breast training — train on selected sources, score on others.

Reads the combined breast-6source LMDB (rep1, rep2, s0, s1, s3, s6) at 4-class
CL Super-Coarse {Endothelial, Epithelial, Immune, Stromal}, filters cells by
source tag, trains a EffNetV2-S head, then scores on each held-out source
separately.

Usage:
    uv run python scripts/breast_pooled_train.py \
        --train-sources xenium_rep1,xenium_rep2 \
        --test-sources sthelar_breast_s0,sthelar_breast_s1,sthelar_breast_s3,sthelar_breast_s6 \
        --output pipeline_output/breast_pooled_2026_05/A_janesick_to_sthelar \
        --epochs 25
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

DERIVED = Path("/mnt/work/datasets/derived")
LMDB_DIR = DERIVED / "breast-6source-dapi-p128"  # default; overridden by --lmdb-dir

DAPI_NORM_MEAN = 0.485
DAPI_NORM_STD = 0.229


def load_class_names(tier: str) -> list[str]:
    """Load class names for the chosen tier from class_mapping*.json."""
    map_path = LMDB_DIR / ("class_mapping.json" if tier == "coarse"
                           else f"class_mapping_{tier}.json")
    if not map_path.exists():
        raise SystemExit(f"missing {map_path}")
    name_to_idx = json.loads(map_path.read_text())
    return [n for n, _ in sorted(name_to_idx.items(), key=lambda kv: kv[1])]


def load_labels(tier: str):
    """Load label array for the chosen tier."""
    label_file = "labels.npy" if tier == "coarse" else f"labels_{tier}.npy"
    return np.load(LMDB_DIR / label_file)


def load_indices_for_sources(sources_to_keep):
    sources = np.load(LMDB_DIR / "sources.npy", allow_pickle=True)
    mask = np.isin(sources, sources_to_keep)
    return np.where(mask)[0]


class DapiPatchDataset(Dataset):
    """LMDB-backed DAPI patch dataset.

    Labels come from an external array (one entry per global LMDB index), not
    the LMDB header. The first 8 bytes of each LMDB value (a stale coarse-class
    label written at LMDB build time) are skipped.
    """

    def __init__(self, indices, labels, augment=False):
        self.indices = indices
        self.labels = labels  # external label per global LMDB index
        self.augment = augment
        self.env = None

    def _open(self):
        self.env = lmdb.open(str(LMDB_DIR / "patches.lmdb"), readonly=True,
                             lock=False, readahead=False, meminit=False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self.env is None:
            self._open()
        global_idx = int(self.indices[i])
        key = struct.pack(">Q", global_idx)
        with self.env.begin() as txn:
            value = txn.get(key)
        label = int(self.labels[global_idx])
        patch = np.frombuffer(value[8:], dtype=np.uint16).reshape(128, 128)
        img = patch.astype(np.float32) / 65535.0
        if self.augment:
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
            if np.random.rand() < 0.5:
                img = np.flipud(img).copy()
            k = np.random.randint(0, 4)
            if k:
                img = np.rot90(img, k=k).copy()
        img = (img - DAPI_NORM_MEAN) / DAPI_NORM_STD
        return torch.from_numpy(img).unsqueeze(0), int(label)


class DapiClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone, feat_dim = create_backbone("efficientnetv2_rw_s", pretrained=True, in_channels=3)
        self.backbone = backbone
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        x = x.expand(-1, 3, -1, -1)
        feat = self.backbone(x)
        return self.head(self.dropout(feat))


def class_weights(labels, n_classes, max_ratio=10.0):
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    inv = inv / inv.min()
    inv = np.minimum(inv, max_ratio)
    return torch.from_numpy((inv / inv.mean()).astype(np.float32))


def score_loader(model, loader, device, class_names):
    model.train(False)
    all_p, all_t = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            all_p.append(logits.argmax(1).cpu().numpy())
            all_t.append(y.numpy())
    p = np.concatenate(all_p)
    t = np.concatenate(all_t)
    macro_f1 = f1_score(t, p, average="macro", zero_division=0)
    weighted_f1 = f1_score(t, p, average="weighted", zero_division=0)
    accuracy = (p == t).mean()
    pre, rec, f1, sup = precision_recall_fscore_support(
        t, p, labels=list(range(len(class_names))), zero_division=0)
    per_class = {class_names[k]: dict(precision=float(pre[k]), recall=float(rec[k]),
                                        f1=float(f1[k]), support=int(sup[k]))
                 for k in range(len(class_names))}
    return dict(macro_f1=float(macro_f1), weighted_f1=float(weighted_f1),
                accuracy=float(accuracy), per_class=per_class, n_eval=int(len(t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb-dir", required=True, type=Path,
                    help="Path to LMDB directory (must contain patches.lmdb, labels.npy, sources.npy, class_mapping.json)")
    ap.add_argument("--train-sources", required=True)
    ap.add_argument("--test-sources", required=True,
                    help="Comma-separated test sources, OR 'INTERNAL' for held-out internal split.")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15,
                    help="Held-out internal test fraction when --test-sources=INTERNAL.")
    ap.add_argument("--tier", choices=["coarse", "medium", "fine"], default="coarse")
    args = ap.parse_args()

    global LMDB_DIR
    LMDB_DIR = args.lmdb_dir.resolve()
    logger.info(f"LMDB_DIR = {LMDB_DIR}")

    args.output.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_sources = args.train_sources.split(",")
    internal_split = args.test_sources == "INTERNAL"
    test_sources = [] if internal_split else args.test_sources.split(",")

    class_names = load_class_names(args.tier)
    labels_full = load_labels(args.tier)
    logger.info(f"Tier: {args.tier}  ({len(class_names)} classes)")

    train_pool = load_indices_for_sources(train_sources)
    if len(train_pool) == 0:
        raise SystemExit(f"No cells for train sources: {train_sources}")
    # Drop cells whose label is -1 (less10/Unknown/unmapped)
    n_before = len(train_pool)
    train_pool = train_pool[labels_full[train_pool] != -1]
    n_dropped = n_before - len(train_pool)
    if n_dropped:
        logger.info(f"Dropped {n_dropped:,} cells ({100*n_dropped/n_before:.1f}%) with label=-1 from train pool")

    pool_labels = labels_full[train_pool]
    if internal_split:
        # Held-out internal test split BEFORE val split, stratified by label
        trainval_idx, internal_test_idx = train_test_split(
            train_pool, test_size=args.test_frac, stratify=pool_labels, random_state=args.seed)
        trainval_labels = labels_full[trainval_idx]
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=args.val_frac, stratify=trainval_labels, random_state=args.seed)
        logger.info(f"INTERNAL split: {len(train_idx):,} train + {len(val_idx):,} val + {len(internal_test_idx):,} test")
    else:
        train_idx, val_idx = train_test_split(
            train_pool, test_size=args.val_frac, stratify=pool_labels, random_state=args.seed)
        internal_test_idx = None
    train_labels = labels_full[train_idx]

    logger.info(f"Train sources: {train_sources}  ->  {len(train_idx):,} train + {len(val_idx):,} val")
    logger.info(f"Train class dist: {dict(zip(*np.unique(train_labels, return_counts=True)))}")

    weights_per_cell = class_weights(train_labels, len(class_names))[train_labels]
    sampler = WeightedRandomSampler(weights_per_cell, num_samples=len(train_idx), replacement=True)
    train_loader = DataLoader(DapiPatchDataset(train_idx, labels_full, augment=True),
                              batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(DapiPatchDataset(val_idx, labels_full, augment=False),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DapiClassifier(len(class_names)).to(device)
    cw = class_weights(train_labels, len(class_names)).to(device)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2)

    best_val_f1 = -1.0
    best_epoch = -1
    history = []
    epochs_since_improve = 0
    ckpt_path = args.output / "best_model.pt"
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train(True)
        running = 0.0; n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0); n += x.size(0)
        sched.step()
        train_loss = running / n
        val = score_loader(model, val_loader, device, class_names)
        history.append(dict(epoch=epoch, train_loss=train_loss, **val))
        logger.info(f"epoch {epoch:>2d}  train_loss={train_loss:.3f}  val_F1={val['macro_f1']:.3f}  val_acc={val['accuracy']:.3f}")
        if val["macro_f1"] > best_val_f1:
            best_val_f1 = val["macro_f1"]; best_epoch = epoch; epochs_since_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                logger.info(f"early stop at epoch {epoch} (no improve in {args.patience})")
                break
    train_time = time.time() - t0

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    per_test_results = {}
    if internal_split and internal_test_idx is not None:
        loader = DataLoader(DapiPatchDataset(internal_test_idx, labels_full, augment=False),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        m = score_loader(model, loader, device, class_names)
        per_test_results["INTERNAL_HELDOUT"] = m
        logger.info(f"TEST {'INTERNAL_HELDOUT':35s} n={m['n_eval']:>7d}  macro_F1={m['macro_f1']:.3f}  acc={m['accuracy']:.3f}")
    for ts in test_sources:
        idx = load_indices_for_sources([ts])
        idx = idx[labels_full[idx] != -1]  # drop -1 cells from test too
        if len(idx) == 0:
            logger.warning(f"  {ts}: no cells")
            continue
        loader = DataLoader(DapiPatchDataset(idx, labels_full, augment=False),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        m = score_loader(model, loader, device, class_names)
        per_test_results[ts] = m
        logger.info(f"TEST {ts:35s} n={m['n_eval']:>7d}  macro_F1={m['macro_f1']:.3f}  acc={m['accuracy']:.3f}")

    summary = dict(
        tier=args.tier,
        train_sources=train_sources, test_sources=test_sources,
        n_train=int(len(train_idx)), n_val=int(len(val_idx)),
        best_epoch=int(best_epoch), best_val_macro_f1=float(best_val_f1),
        train_time_s=float(train_time),
        train_class_dist={class_names[int(k)]: int(v)
                          for k, v in zip(*np.unique(train_labels, return_counts=True))},
        per_test=per_test_results, class_names=class_names, epochs_run=len(history),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "history.json").write_text(json.dumps(history, indent=2))
    logger.info(f"wrote {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
