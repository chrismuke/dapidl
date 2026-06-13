"""Frozen vision-FM embedders for DAPI QC crops (DINOv2 + NuSPIRe), with per-model preprocessing."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _stretch01(patch):
    p = np.asarray(patch, dtype=np.float32)
    lo, hi = np.percentile(p, [1.0, 99.0])
    hi = hi if hi > lo else lo + 1.0
    return np.clip((p - lo) / (hi - lo), 0.0, 1.0)


def _resize(img2d, size):
    from PIL import Image
    return np.asarray(Image.fromarray((img2d * 255).astype(np.uint8)).resize((size, size),
                                                                             Image.BILINEAR),
                      dtype=np.float32) / 255.0


def preprocess_dinov2(patch, size=224):
    g = _resize(_stretch01(patch), size)             # [size,size] in [0,1]
    rgb = np.repeat(g[None, :, :], 3, axis=0)        # [3,size,size]
    return ((rgb - _IMAGENET_MEAN[:, None, None]) / _IMAGENET_STD[:, None, None]).astype(np.float32)


# NuSPIRe normalization stats (from dapidl.models.nuspire; trained on [0,1] DAPI grayscale)
_NUSPIRE_MEAN = 0.21869252622127533
_NUSPIRE_STD = 0.1809280514717102


def preprocess_nuspire(patch, size=112):
    g = _resize(_stretch01(patch), size)
    return (((g - _NUSPIRE_MEAN) / _NUSPIRE_STD)[None, :, :]).astype(np.float32)


# ---------------------------------------------------------------------------
# Frozen model loaders (torch/timm imported lazily so the preprocessing above
# stays torch-free for the pure unit tests) + the embedding pass.
# ---------------------------------------------------------------------------

def _load_dinov2(device):
    import timm
    m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True,
                          num_classes=0, dynamic_img_size=True)
    return m.eval().to(device)


def _load_nuspire(device):
    from dapidl.models.nuspire import NuSPIReBackbone
    return NuSPIReBackbone(pretrained=True, pool="mean").eval().to(device)


EMBEDDERS = {
    "dinov2_vitb14": {"load": _load_dinov2, "preprocess": preprocess_dinov2, "size": 224, "dim": 768},
    "nuspire": {"load": _load_nuspire, "preprocess": preprocess_nuspire, "size": 112, "dim": 768},
}


def _read_patch(txn, idx, ps=64):
    v = txn.get(struct.pack(">Q", int(idx)))
    if v is None:
        return np.zeros((ps, ps), dtype=np.uint16)
    return np.frombuffer(v[8:], dtype=np.uint16).reshape(ps, ps)


def compute_embeddings(lmdb_dir, model="dinov2_vitb14", device="cuda", batch_size=256,
                       recompute=False):
    """Frozen-model embedding of every p64 crop. Returns (rows int64[N], emb float16[N,D]);
    caches to qc/embeddings_<model>.npy (+ _rows.npy)."""
    import lmdb
    import polars as pl
    import torch

    spec = EMBEDDERS[model]
    lmdb_dir = Path(lmdb_dir)
    qc = lmdb_dir / "qc"
    emb_path = qc / f"embeddings_{model}.npy"
    rows_path = qc / f"embeddings_{model}_rows.npy"
    rows = pl.read_parquet(qc / "seg_scores.parquet")["row_idx"].to_numpy()
    if emb_path.exists() and rows_path.exists() and not recompute:
        cached = np.load(rows_path)
        if len(cached) == len(rows):
            return cached, np.load(emb_path)

    net = spec["load"](device)
    pre = spec["preprocess"]
    env = lmdb.open(str(lmdb_dir / "patches.lmdb"), readonly=True, lock=False, readahead=False)
    embs = []
    with env.begin() as txn, torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            x = np.stack([pre(_read_patch(txn, r)) for r in chunk])
            out = net(torch.from_numpy(x).float().to(device))
            embs.append(out.detach().cpu().numpy().astype(np.float16))
    env.close()
    emb = np.concatenate(embs)
    qc.mkdir(exist_ok=True)
    np.save(emb_path, emb)
    np.save(rows_path, rows)
    return rows, emb
