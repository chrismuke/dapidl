"""Frozen-EffNet embedding extraction over the LMDB + PCA reduction. The pure
helpers (decode_record, pca_fit_transform) are unit-tested; extract_embeddings is
GPU and run by the controller."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def decode_record(value: bytes, patch_size: int = 128):
    """Format B record -> (int label, uint16 patch). value = int64 label + uint16 square."""
    label = int(np.frombuffer(value[:8], dtype=np.int64)[0])
    patch = np.frombuffer(value[8:], dtype=np.uint16).reshape(patch_size, patch_size)
    return label, patch


def pca_fit_transform(emb, n_components: int = 128, fit_sample: int = 200_000, seed: int = 0):
    """Fit PCA on a random row-sample (RAM), transform all rows. Returns (reduced, model)."""
    emb = np.asarray(emb)
    rng = np.random.default_rng(seed)
    n = len(emb)
    sample = emb if n <= fit_sample else emb[rng.choice(n, size=fit_sample, replace=False)]
    model = PCA(n_components=min(n_components, emb.shape[1]), random_state=seed)
    model.fit(sample.astype(np.float32))
    return model.transform(emb.astype(np.float32)).astype(np.float32), model


def extract_embeddings(lmdb_dir: Path, ckpt: Path, out_path: Path,
                       n: int, batch_size: int = 256, patch_size: int = 128) -> None:
    """[GPU] Stream the LMDB in row order through the frozen DapiClassifier backbone
    (penultimate 1792-d features) -> float16 memmap (n, 1792)."""
    import sys
    import lmdb
    import torch
    sys.path.insert(0, "scripts")
    from breast_pooled_train import DapiClassifier  # same class the checkpoint was saved from

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DapiClassifier(num_classes=4, backbone="efficientnetv2_rw_s")
    # weights_only=True: our own trusted checkpoint, but keep the secure default
    # (a state_dict is tensors + basic types, so this loads fine).
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = state.get("model_state_dict") or state.get("model") or state
    model.load_state_dict(sd)
    model.eval().to(device)

    feat_dim = model.head.in_features
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, feat_dim))
    env = lmdb.open(str(lmdb_dir / "patches.lmdb"), readonly=True, lock=False)
    buf_imgs: list[np.ndarray] = []
    buf_rows: list[int] = []

    def flush():
        if not buf_rows:
            return
        x = np.stack(buf_imgs).astype(np.float32) / 65535.0
        x = (x - 0.485) / 0.229
        t = torch.from_numpy(x)[:, None, :, :].to(device)
        with torch.no_grad():
            feat = model.backbone(t.expand(-1, 3, -1, -1))
        out[buf_rows] = feat.cpu().numpy().astype(np.float16)
        buf_imgs.clear(); buf_rows.clear()

    with env.begin() as txn:
        for idx in range(n):
            _, patch = decode_record(txn.get(struct.pack(">Q", idx)), patch_size)
            buf_imgs.append(patch); buf_rows.append(idx)
            if len(buf_rows) == batch_size:
                flush()
        flush()
    out.flush()
    env.close()
