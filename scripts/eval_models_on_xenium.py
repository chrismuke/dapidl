#!/usr/bin/env python3
"""Cross-platform inference: STHELAR-trained models -> Xenium rep1/rep2.

Loads each STHELAR-trained DAPI checkpoint, runs inference on the Xenium
rep1/rep2 LMDBs (built at p128, finegrained 57-class), maps both predictions
and ground-truth into a unified 4-class coarse scheme (Endothelial, Epithelial,
Immune, Stromal), and computes macro F1 + per-class metrics + confusion matrix.

This is the FIRST cross-platform measurement.

Output:
    pipeline_output/model_eval_2026_05/cross_platform_{rep1,rep2}.json
    pipeline_output/model_eval_2026_05/cross_platform_summary.md

Usage:
    uv run python scripts/eval_models_on_xenium.py
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


PIPELINE = Path("/mnt/work/git/dapidl/pipeline_output")
OUT = PIPELINE / "model_eval_2026_05"
OUT.mkdir(parents=True, exist_ok=True)

DERIVED = Path("/mnt/work/datasets/derived")
REP_LMDBS = {
    "rep1": DERIVED / "xenium-breast-tumor-rep1-local-finegrained-p128",
    "rep2": DERIVED / "xenium-breast-tumor-rep2-local-finegrained-p128",
}

DAPI_NORM_MEAN = 0.485
DAPI_NORM_STD = 0.229
PATCH_SIZE = 128

COARSE_CLASSES = ["Endothelial", "Epithelial", "Immune", "Stromal"]
COARSE_IDX = {name: i for i, name in enumerate(COARSE_CLASSES)}

STHELAR_9_TO_COARSE = {
    "endothelial cell": "Endothelial",
    "epithelial cell": "Epithelial",
    "T cell": "Immune",
    "B cell": "Immune",
    "macrophage": "Immune",
    "mast cell": "Immune",
    "fibroblast": "Stromal",
    "pericyte": "Stromal",
    "adipocyte": "Stromal",
}

STHELAR_7_TO_COARSE = {
    "endothelial cell": "Endothelial",
    "epithelial cell": "Epithelial",
    "T cell": "Immune",
    "B cell": "Immune",
    "macrophage": "Immune",
    "fibroblast": "Stromal",
    "pericyte": "Stromal",
}


def _xenium57_to_coarse(name: str) -> str | None:
    n = name.lower()
    if n.startswith("vas-") or n.startswith("lymph-"):
        return "Endothelial"
    if n.startswith(("lummhr", "lumsec")) or n == "basal":
        return "Epithelial"
    if n.startswith(("cd4-", "cd8-", "macro-", "mono-", "mast", "neutro",
                     "nk", "gd", "nkt", "t_prol", "mye-", "cdc", "mdc", "pdc",
                     "b_", "bmem", "plasma")):
        return "Immune"
    if n.startswith(("fibro-",)) or n in ("pericytes", "vsmc"):
        return "Stromal"
    return None


class XeniumDapiPatchDataset(Dataset):
    def __init__(self, lmdb_dir: Path, indices: np.ndarray):
        self.lmdb_path = lmdb_dir / "patches.lmdb"
        self.indices = indices
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
        if label < 0 or label > 10000:
            label = struct.unpack(">q", value[:8])[0]
        patch = np.frombuffer(value[8:], dtype=np.uint16).reshape(
            PATCH_SIZE, PATCH_SIZE
        )
        img = patch.astype(np.float32) / 65535.0
        img = (img - DAPI_NORM_MEAN) / DAPI_NORM_STD
        return torch.from_numpy(img).unsqueeze(0), int(label)


class ModalityClassifierFlatHead(nn.Module):
    """Head is a single nn.Linear (modality_dapi style)."""

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


class ModalityClassifierSequentialHead(nn.Module):
    """Head is nn.Sequential(Dropout, Linear) (multitissue/exp5 style)."""

    def __init__(self, num_classes: int):
        super().__init__()
        backbone, feat_dim = create_backbone(
            "efficientnetv2_rw_s", pretrained=False, in_channels=3
        )
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        x = x.expand(-1, 3, -1, -1)
        feat = self.backbone(x)
        return self.head(feat)


def _detect_head_style(state: dict) -> str:
    """Inspect a state dict and return 'flat' or 'sequential'."""
    if any(k.startswith("head.1.") for k in state.keys()):
        return "sequential"
    return "flat"


MODEL_REGISTRY: dict[str, dict] = {
    "sthelar_modality_dapi": {
        "ckpt": PIPELINE / "sthelar_modality_dapi" / "best_model.pt",
        "n_classes": 9,
        "class_names": [
            "macrophage", "fibroblast", "B cell", "T cell", "endothelial cell",
            "adipocyte", "mast cell", "epithelial cell", "pericyte",
        ],
        "to_coarse": STHELAR_9_TO_COARSE,
    },
    "sthelar_exp5_7class": {
        # Checkpoint actually has 9-output head; trained with adipocyte+mast masked from loss.
        "ckpt": PIPELINE / "sthelar_exp5_7class" / "best_model.pt",
        "n_classes": 9,
        "class_names": [
            "macrophage", "fibroblast", "B cell", "T cell", "endothelial cell",
            "adipocyte", "mast cell", "epithelial cell", "pericyte",
        ],
        "to_coarse": STHELAR_9_TO_COARSE,
    },
    "sthelar_multitissue_9class": {
        "ckpt": PIPELINE / "sthelar_multitissue_9class" / "best_model.pt",
        "n_classes": 9,
        "class_names": [
            "macrophage", "fibroblast", "B cell", "T cell", "endothelial cell",
            "adipocyte", "mast cell", "epithelial cell", "pericyte",
        ],
        "to_coarse": STHELAR_9_TO_COARSE,
    },
}


def load_model(model_key: str, device: torch.device) -> tuple[nn.Module, dict]:
    spec = MODEL_REGISTRY[model_key]
    state = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}

    head_style = _detect_head_style(state)
    if head_style == "sequential":
        model: nn.Module = ModalityClassifierSequentialHead(num_classes=spec["n_classes"])
    else:
        model = ModalityClassifierFlatHead(num_classes=spec["n_classes"])

    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.startswith(("aux_", "coarse_head", "fine_head"))]
    if real_missing:
        raise RuntimeError(f"Missing keys when loading {model_key} ({head_style}): {real_missing}")
    if unexpected:
        # Allow extra heads (hierarchical model has fine_head + coarse_head)
        pass
    model.train(False)
    model.to(device)
    return model, spec


def build_xenium_label_map(lmdb_dir: Path) -> tuple[list[str], np.ndarray]:
    meta = json.loads((lmdb_dir / "metadata.json").read_text())
    fine_names: list[str] = meta["class_names"]
    coarse_lookup = np.full(len(fine_names), -1, dtype=np.int64)
    for i, n in enumerate(fine_names):
        coarse = _xenium57_to_coarse(n)
        if coarse is not None:
            coarse_lookup[i] = COARSE_IDX[coarse]
    return fine_names, coarse_lookup


def run_inference(
    model_key: str,
    dataset_key: str,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 6,
) -> dict:
    lmdb_dir = REP_LMDBS[dataset_key]
    fine_names, coarse_lookup = build_xenium_label_map(lmdb_dir)
    n_fine = len(fine_names)
    valid_idx = np.where(coarse_lookup >= 0)[0]
    print(f"[{dataset_key}] {len(valid_idx)}/{n_fine} fine classes mapped to coarse")

    labels = np.load(lmdb_dir / "labels.npy")
    eligible = np.where(np.isin(labels, valid_idx))[0]
    print(f"[{dataset_key}] {len(eligible):,} of {len(labels):,} cells map to coarse classes")

    ds = XeniumDapiPatchDataset(lmdb_dir, eligible)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )

    model, spec = load_model(model_key, device)
    sthelar_to_coarse = spec["to_coarse"]
    pred_to_coarse = np.array([
        COARSE_IDX[sthelar_to_coarse[c]] for c in spec["class_names"]
    ], dtype=np.int64)

    y_true_coarse: list[int] = []
    y_pred_coarse: list[int] = []

    t0 = time.time()
    with torch.no_grad():
        for img, lab in loader:
            img = img.to(device, non_blocking=True)
            logits = model(img)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred_coarse.extend(pred_to_coarse[preds].tolist())
            y_true_coarse.extend(coarse_lookup[lab.numpy()].tolist())
    elapsed = time.time() - t0

    yt = np.array(y_true_coarse)
    yp = np.array(y_pred_coarse)

    acc = float((yt == yp).mean())
    macro = float(f1_score(yt, yp, labels=list(range(4)), average="macro", zero_division=0))
    weighted = float(f1_score(yt, yp, labels=list(range(4)), average="weighted", zero_division=0))
    p, r, f1, support = precision_recall_fscore_support(
        yt, yp, labels=list(range(4)), zero_division=0
    )
    cm = confusion_matrix(yt, yp, labels=list(range(4)))

    per_class = {
        COARSE_CLASSES[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(4)
    }

    print(f"[{model_key} on {dataset_key}] coarse macro F1={macro:.4f} acc={acc:.4f} "
          f"weighted={weighted:.4f}  ({elapsed:.0f}s)")
    return {
        "model": model_key,
        "dataset": dataset_key,
        "n_eval": int(len(yt)),
        "accuracy": acc,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "coarse_classes": COARSE_CLASSES,
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODEL_REGISTRY.keys()),
                    help="Comma-separated model keys")
    ap.add_argument("--datasets", default="rep1,rep2",
                    help="Comma-separated dataset keys")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    selected_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    all_results = []
    for ds in selected_datasets:
        per_dataset = []
        for mk in selected_models:
            ckpt = MODEL_REGISTRY[mk]["ckpt"]
            if not ckpt.exists():
                print(f"  SKIP {mk}: checkpoint missing ({ckpt})")
                continue
            res = run_inference(mk, ds, device, batch_size=args.batch_size)
            per_dataset.append(res)
            all_results.append(res)
        out = OUT / f"cross_platform_{ds}.json"
        out.write_text(json.dumps(per_dataset, indent=2))
        print(f"Wrote {out}")

    summary = OUT / "cross_platform_summary.md"
    lines = [
        "# Cross-Platform Inference: STHELAR -> Xenium",
        "",
        "STHELAR-trained DAPI models tested on Xenium breast rep1 and rep2 at",
        "coarse 4-class level (Endothelial, Epithelial, Immune, Stromal).",
        "",
        "| Model | Dataset | n_eval | Accuracy | Macro F1 | Weighted F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in all_results:
        lines.append(
            f"| `{r['model']}` | {r['dataset']} | {r['n_eval']:,} | "
            f"{r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} |"
        )
    lines.append("")
    lines.append("## Per-class F1 by model+dataset")
    lines.append("")
    lines.append("| Model | Dataset | " + " | ".join(COARSE_CLASSES) + " |")
    lines.append("|---|---|" + "|".join(["---"] * len(COARSE_CLASSES)) + "|")
    for r in all_results:
        per = r["per_class"]
        lines.append(
            f"| `{r['model']}` | {r['dataset']} | "
            + " | ".join(f"{per[c]['f1']:.3f}" for c in COARSE_CLASSES)
            + " |"
        )
    summary.write_text("\n".join(lines))
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
