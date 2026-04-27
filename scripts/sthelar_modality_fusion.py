#!/usr/bin/env python3
"""Proper multimodal fusion: separate DAPI/H&E backbones + bidirectional cross-attention.

Replaces the naive 4→3 1×1 conv adapter (sthelar_modality_train.py mode=both)
with a proper two-stem architecture:

  DAPI (1ch → expanded to 3) → EfficientNetV2-S backbone → (B, 1792, 4, 4)
  H&E   (3ch)                → EfficientNetV2-S backbone → (B, 1792, 4, 4)
                                                                ↓
                              Each spatial map → 16 tokens of 1792 dim
                                                                ↓
                Bidirectional cross-attention (8-head) + FFN block
                                                                ↓
                          Mean-pool tokens → concat → linear head

Same 70/15/15 stratified HE-intersection split (seed 42) as the modality
3-way comparison so results are directly comparable to:
- DAPI baseline    (sthelar_modality_dapi)
- H&E only         (sthelar_modality_he)
- Naive DAPI+H&E   (sthelar_modality_both, 4→3 adapter)
- Fusion DAPI+H&E  (this script, sthelar_modality_fusion)

Usage:
    uv run python scripts/sthelar_modality_fusion.py \\
        --output pipeline_output/sthelar_modality_fusion \\
        --epochs 21 --batch-size 48
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


DAPI_DIR = Path("/mnt/work/datasets/derived/sthelar-multitissue-p128")
HE_DIR = Path("/mnt/work/datasets/derived/sthelar-multitissue-p128-he")

DAPI_NORM_MEAN = 0.485
DAPI_NORM_STD = 0.229
HE_NORM_MEAN = (0.485, 0.456, 0.406)
HE_NORM_STD = (0.229, 0.224, 0.225)


# --------------------------- dataset (4-ch) -------------------------------

class DapiHePatchDataset(Dataset):
    """Returns (4, 128, 128) — channel 0 = DAPI, channels 1-3 = RGB H&E.

    The fusion model splits this into a 1ch DAPI tensor and a 3ch H&E tensor
    inside its forward pass. We could pass two tensors, but a single 4-ch
    tensor works seamlessly with WeightedRandomSampler / DataLoader / pin_memory.
    """
    def __init__(self, indices: np.ndarray):
        self.indices = indices
        self.dapi_env = None
        self.he_env = None

    def _open(self):
        self.dapi_env = lmdb.open(str(DAPI_DIR / "patches.lmdb"), readonly=True,
                                  lock=False, readahead=False, meminit=False)
        self.he_env = lmdb.open(str(HE_DIR / "patches.lmdb"), readonly=True,
                                lock=False, readahead=False, meminit=False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self.dapi_env is None:
            self._open()
        key = struct.pack(">Q", int(self.indices[i]))

        with self.dapi_env.begin() as txn:
            dvalue = txn.get(key)
        with self.he_env.begin() as txn:
            hvalue = txn.get(key)

        d_label = np.frombuffer(dvalue[:8], dtype=np.int64)[0]
        if d_label < 0 or d_label > 10000:
            d_label = struct.unpack(">q", dvalue[:8])[0]

        dapi = np.frombuffer(dvalue[8:], dtype=np.uint16).reshape(128, 128).astype(np.float32) / 65535.0
        dapi = (dapi - DAPI_NORM_MEAN) / DAPI_NORM_STD

        he = np.frombuffer(hvalue[8:], dtype=np.uint8).reshape(3, 128, 128).astype(np.float32) / 255.0
        for c in range(3):
            he[c] = (he[c] - HE_NORM_MEAN[c]) / HE_NORM_STD[c]

        merged = np.empty((4, 128, 128), dtype=np.float32)
        merged[0] = dapi
        merged[1:] = he
        return torch.from_numpy(merged), int(d_label)


# --------------------------- fusion model ---------------------------------

class CrossAttentionBlock(nn.Module):
    """One bidirectional cross-attention layer + per-stream FFN.

    Pre-norm style: norm before attention/FFN, residual after. This stabilises
    training when stacking on top of pretrained backbones.
    """
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm_a1 = nn.LayerNorm(dim)
        self.norm_b1 = nn.LayerNorm(dim)
        self.attn_a = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_b = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.norm_a2 = nn.LayerNorm(dim)
        self.norm_b2 = nn.LayerNorm(dim)
        self.ffn_a = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.ffn_b = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        # a, b: (B, N, dim) — independent token sequences from the two modalities
        an, bn = self.norm_a1(a), self.norm_b1(b)
        a_attn, _ = self.attn_a(an, bn, bn)  # DAPI tokens query H&E tokens
        b_attn, _ = self.attn_b(bn, an, an)  # H&E  tokens query DAPI tokens
        a = a + a_attn
        b = b + b_attn
        a = a + self.ffn_a(self.norm_a2(a))
        b = b + self.ffn_b(self.norm_b2(b))
        return a, b


class FusionModel(nn.Module):
    def __init__(self, num_classes: int, num_fusion_layers: int = 2,
                 num_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.dapi_backbone, dim = create_backbone("efficientnetv2_rw_s", pretrained=True, in_channels=3)
        self.he_backbone, dim_he = create_backbone("efficientnetv2_rw_s", pretrained=True, in_channels=3)
        assert dim == dim_he, f"backbone feat-dim mismatch {dim} vs {dim_he}"

        # Lightweight learnable positional embeddings for the 4×4 spatial grid (16 tokens).
        # Pretrained backbones lose absolute position when we flatten to tokens; this lets
        # cross-attention preserve spatial correspondence between modalities.
        self.pos_a = nn.Parameter(torch.zeros(1, 16, dim))
        self.pos_b = nn.Parameter(torch.zeros(1, 16, dim))
        nn.init.trunc_normal_(self.pos_a, std=0.02)
        nn.init.trunc_normal_(self.pos_b, std=0.02)

        self.fusion_blocks = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads=num_heads, dropout=0.1)
            for _ in range(num_fusion_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, 128, 128)
        dapi = x[:, 0:1]                     # (B, 1, 128, 128)
        he = x[:, 1:4]                       # (B, 3, 128, 128)

        d_feat = self.dapi_backbone.forward_features(dapi.expand(-1, 3, -1, -1))
        h_feat = self.he_backbone.forward_features(he)
        # both: (B, 1792, 4, 4)
        d_tok = d_feat.flatten(2).transpose(1, 2) + self.pos_a   # (B, 16, 1792)
        h_tok = h_feat.flatten(2).transpose(1, 2) + self.pos_b

        for block in self.fusion_blocks:
            d_tok, h_tok = block(d_tok, h_tok)

        d_pooled = d_tok.mean(dim=1)
        h_pooled = h_tok.mean(dim=1)
        fused = torch.cat([d_pooled, h_pooled], dim=1)
        return self.head(self.dropout(fused))


# --------------------------- helpers --------------------------------------

def get_he_valid_mask(num_samples: int) -> np.ndarray:
    cache_path = HE_DIR / "he_valid_mask.npy"
    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached) == num_samples:
            return cached.astype(bool)
    env = lmdb.open(str(HE_DIR / "patches.lmdb"), readonly=True,
                    lock=False, readahead=False, meminit=False)
    valid = np.zeros(num_samples, dtype=bool)
    with env.begin() as txn:
        cur = txn.cursor()
        for k in cur.iternext(keys=True, values=False):
            kb = bytes(k)
            if len(kb) == 8:
                idx = struct.unpack(">Q", kb)[0]
                if idx < num_samples:
                    valid[idx] = True
    env.close()
    np.save(cache_path, valid)
    return valid


def build_splits(num_samples: int, labels: np.ndarray, valid_mask: np.ndarray):
    idx = np.where(valid_mask)[0]
    keep_labels = labels[idx]
    train, temp = train_test_split(idx, train_size=0.7, random_state=42, stratify=keep_labels)
    temp_labels = labels[temp]
    val, test = train_test_split(temp, train_size=0.5, random_state=42, stratify=temp_labels)
    return train, val, test


def reconstruct_tissue_idx(slide_stats_path: Path):
    with open(slide_stats_path) as f:
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


# --------------------------- main -----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=21)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--num-fusion-layers", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=8)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"FUSION model: output={args.output} epochs={args.epochs} bs={args.batch_size} "
                f"layers={args.num_fusion_layers} heads={args.num_heads}")

    with open(DAPI_DIR / "class_mapping.json") as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    class_names = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])]

    labels_all = np.load(DAPI_DIR / "labels.npy")
    n_total = len(labels_all)

    logger.info("scanning HE LMDB for valid indices...")
    valid_mask = get_he_valid_mask(n_total)
    logger.info(f"HE intersection: {valid_mask.sum():,} / {n_total:,} indices")

    train_idx, val_idx, test_idx = build_splits(n_total, labels_all, valid_mask=valid_mask)
    tissue_idx, tissue_names = reconstruct_tissue_idx(DAPI_DIR / "slide_stats.json")
    logger.info(f"splits: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_ds = DapiHePatchDataset(train_idx)
    val_ds = DapiHePatchDataset(val_idx)
    test_ds = DapiHePatchDataset(test_idx)

    train_labels = labels_all[train_idx]
    class_w = get_class_weights(train_labels, num_classes,
                                method="inverse", max_weight_ratio=10.0).numpy()
    sample_w = class_w[train_labels]
    sampler = WeightedRandomSampler(sample_w.tolist(),
                                    num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = FusionModel(num_classes=num_classes,
                        num_fusion_layers=args.num_fusion_layers,
                        num_heads=args.num_heads).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model params: {n_params/1e6:.1f}M")

    class_weights = torch.from_numpy(class_w).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_f1 = -1.0
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
            if bi % 500 == 0:
                logger.info(f"  epoch {epoch+1} [{bi}/{len(train_loader)}] "
                            f"loss={float(loss):.4f} acc={train_correct/max(1,train_total):.4f}")
        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        val_metrics = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0
        logger.info(f"epoch {epoch+1}/{args.epochs} (FUSION) "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                    f"val_f1={val_metrics['macro_f1']:.4f} time={epoch_time:.0f}s")
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"], "val_weighted_f1": val_metrics["weighted_f1"],
            "lr": optimizer.param_groups[0]["lr"], "time_s": epoch_time,
        })
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "hparams": {"mode": "fusion", "num_classes": num_classes,
                            "backbone_name": "efficientnetv2_rw_s",
                            "num_fusion_layers": args.num_fusion_layers,
                            "num_heads": args.num_heads},
                "epoch": epoch + 1,
                "metrics": {
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_weighted_f1": val_metrics["weighted_f1"],
                    "epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
                },
            }, args.output / "best_model.pt")
            logger.info(f"  saved best (val_f1={best_val_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info(f"early stopping (patience {args.patience} reached)")
                break

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
            "experiment": "modality-fusion",
            "mode": "fusion",
            "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "best_epoch": int(ckpt["epoch"]), "best_val_macro_f1": float(best_val_f1),
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "class_names": class_names,
            "n_params": int(n_params),
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
    logger.info(f"TEST FUSION: acc={test_metrics['accuracy']:.4f} "
                f"macro_f1={test_metrics['macro_f1']:.4f} "
                f"weighted_f1={test_metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
