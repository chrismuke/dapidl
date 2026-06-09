# scripts/spatial_gnn_probe.py
"""Spatial-GNN probe driver: registry -> stage1 -> gate -> stage2 -> readout.
Run phases individually; GPU phases print nvidia-smi guidance first."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

LMDB_DIR = Path("/mnt/work/datasets/derived/breast-6source-dapi-p128")
OUT = Path("pipeline_output/spatial_gnn_probe_2026_06")


def phase_registry() -> None:
    from dapidl.graph.registry import build_spatial_registry
    OUT.mkdir(parents=True, exist_ok=True)
    reg = build_spatial_registry(LMDB_DIR)
    reg.write_parquet(OUT / "spatial_registry.parquet")
    logger.info(f"registry: {len(reg)} rows verified-aligned -> {OUT/'spatial_registry.parquet'}")


def phase_embed() -> None:
    import numpy as np
    from dapidl.graph.embed import extract_embeddings, pca_fit_transform
    n = int(np.load(LMDB_DIR / "labels.npy").shape[0])
    ckpt = Path("pipeline_output/h2h_2026_05_30/efficientnetv2_rw_s/best_model.pt")
    emb_path = OUT / "embeddings_f16.npy"
    extract_embeddings(LMDB_DIR, ckpt, emb_path, n=n, batch_size=256)
    emb = np.load(emb_path, mmap_mode="r")
    red, _ = pca_fit_transform(emb, n_components=128)
    np.save(OUT / "embeddings_pca128.npy", red)
    logger.info(f"embeddings: {emb.shape} -> pca {red.shape}")


CLASSES = ["Endothelial", "Epithelial", "Immune", "Stromal"]
TEST_SLIDE = "xenium_rep2"
VAL_SLIDE = "sthelar_breast_s3"


def _per_class(truth, pred):
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    _, _, f1, sup = precision_recall_fscore_support(truth, pred, labels=[0, 1, 2, 3], zero_division=0)
    return {"macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "per_class": {CLASSES[k]: {"f1": float(f1[k]), "support": int(sup[k])} for k in range(4)}}


def phase_stage1() -> None:
    """BANKSY-augment frozen-EffNet PCA-128 embeddings over the within-slide graph,
    classify with LightGBM. lambda=0 is the no-graph ablation. -1 cells are graph
    context only (masked from train/val/test)."""
    import lightgbm as lgb
    import numpy as np
    import polars as pl
    from dapidl.graph.banksy_features import banksy_augment
    from dapidl.graph.knn_graph import build_within_slide_knn

    reg = pl.read_parquet(OUT / "spatial_registry.parquet")
    pca = np.load(OUT / "embeddings_pca128.npy")          # (N,128) row-aligned
    src = reg["source"].to_numpy()
    coords = reg.select(["x_px", "y_px"]).to_numpy()
    labels = reg["coarse_idx"].to_numpy()

    edge_index = build_within_slide_knn(coords, src, k=8)  # over ALL cells (incl -1 context)
    labeled = labels != -1
    test_mask = (src == TEST_SLIDE) & labeled
    val_mask = (src == VAL_SLIDE) & labeled
    train_mask = ~(src == TEST_SLIDE) & ~(src == VAL_SLIDE) & labeled
    truth = labels[test_mask]
    np.save(OUT / "stage1_truth.npy", truth)

    results = {}
    for lam in [0.0, 0.2, 0.5, 0.8]:
        feats = banksy_augment(pca, edge_index, coords, lambda_=lam)
        booster = lgb.train(
            {"objective": "multiclass", "num_class": 4, "learning_rate": 0.05,
             "num_leaves": 63, "metric": "multi_logloss", "is_unbalance": True,
             "verbose": -1, "seed": 0},
            lgb.Dataset(feats[train_mask], labels[train_mask]),
            num_boost_round=600,
            valid_sets=[lgb.Dataset(feats[val_mask], labels[val_mask])],
            callbacks=[lgb.early_stopping(40, verbose=False)])
        pred = booster.predict(feats[test_mask]).argmax(1)
        np.save(OUT / f"stage1_pred_lambda{lam}.npy", pred)
        results[f"lambda_{lam}"] = {**_per_class(truth, pred),
                                    "best_iteration": int(booster.best_iteration)}
        logger.info(f"stage1 lambda={lam}: macro_f1={results[f'lambda_{lam}']['macro_f1']:.4f}")
        del feats
    (OUT / "stage1_metrics.json").write_text(json.dumps(
        {"baseline_effnet_macro_f1": 0.619, "results": results}, indent=2))
    logger.info("stage1 done -> stage1_metrics.json")


def phase_gate() -> None:
    """Best non-zero lambda vs the lambda=0 ablation: gate decision, McNemar, CI."""
    import numpy as np
    from dapidl.graph.probe_eval import bootstrap_macro_f1_ci, gate_decision, mcnemar_test
    m = json.loads((OUT / "stage1_metrics.json").read_text())["results"]
    best_lam = max([0.2, 0.5, 0.8], key=lambda L: m[f"lambda_{L}"]["macro_f1"])
    base, best = m["lambda_0.0"], m[f"lambda_{best_lam}"]
    truth = np.load(OUT / "stage1_truth.npy")
    pred0 = np.load(OUT / "stage1_pred_lambda0.0.npy")
    predb = np.load(OUT / f"stage1_pred_lambda{best_lam}.npy")
    gate = gate_decision(
        macro_delta=best["macro_f1"] - base["macro_f1"],
        endo_delta=best["per_class"]["Endothelial"]["f1"] - base["per_class"]["Endothelial"]["f1"],
        stromal_delta=best["per_class"]["Stromal"]["f1"] - base["per_class"]["Stromal"]["f1"])
    lo, point, hi = bootstrap_macro_f1_ci(truth, predb, n_boot=1000, seed=0)
    out = {"best_lambda": best_lam, "gate": gate,
           "mcnemar_graph_vs_nograph": mcnemar_test(truth, pred0, predb),
           "macro_f1_ci": {"lo": lo, "point": point, "hi": hi},
           "effnet_baseline": 0.619}
    (OUT / "stage1_gate.json").write_text(json.dumps(out, indent=2))
    logger.info(f"gate: proceed={gate['proceed']} ({gate['reason']}); best_lambda={best_lam}; "
                f"macro_f1={point:.4f} CI[{lo:.4f},{hi:.4f}] vs EffNet 0.619")


def phase_stage2() -> None:
    """Clean learned-BANKSY: a nucleus-local CNN + 1-hop neighbour-mean SAGE, trained
    with neighbour-sampled mini-batches (GPU-feasible). Graph over ALL cells (context);
    loss/metrics on labelled nodes only."""
    import struct
    import sys

    import lmdb
    import numpy as np
    import polars as pl
    import torch
    from scipy.spatial import cKDTree
    from sklearn.metrics import f1_score
    from torch import nn

    from dapidl.graph.embed import decode_record
    from dapidl.graph.gnn import NucleusNodeCNN, scatter_mean
    sys.path.insert(0, "scripts")
    from breast_pooled_train import class_weights

    reg = pl.read_parquet(OUT / "spatial_registry.parquet")
    src = reg["source"].to_numpy()
    coords = reg.select(["x_px", "y_px"]).to_numpy()
    labels = reg["coarse_idx"].to_numpy()
    n = len(reg)
    k = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # (N, k) within-slide neighbour table (pad -1); built per slide so no cross-slide edges.
    nbr = np.full((n, k), -1, dtype=np.int64)
    for s in np.unique(src):
        idx = np.where(src == s)[0]
        if len(idx) < 2:
            continue
        kk = min(k, len(idx) - 1)
        _, nn_ = cKDTree(coords[idx]).query(coords[idx], k=kk + 1)
        nn_ = np.atleast_2d(nn_)
        for c in range(kk):
            nbr[idx, c] = idx[nn_[:, c + 1]]

    # nucleus-local 40x40 crops (centre of each 128 patch) cached in RAM (~7.3 GB).
    crop, off = 40, (128 - 40) // 2
    crops = np.empty((n, crop, crop), dtype=np.uint16)
    env = lmdb.open(str(LMDB_DIR / "patches.lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        for i in range(n):
            _, p = decode_record(txn.get(struct.pack(">Q", i)), 128)
            crops[i] = p[off:off + crop, off:off + crop]
    env.close()

    def encode(rows):  # (B,) global indices -> (B,128) node embeddings
        x = crops[rows].astype(np.float32) / 65535.0
        x = (x - 0.485) / 0.229
        return cnn(torch.from_numpy(x)[:, None].to(device))

    class Sage1(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(128 * 2, 64)
            self.head = nn.Linear(64, 4)

        def forward(self, self_emb, nbr_emb, nbr_valid):
            # nbr_emb (B,k,128), nbr_valid (B,k) -> masked neighbour mean
            cnt = nbr_valid.sum(1, keepdim=True).clamp_min(1.0)
            agg = (nbr_emb * nbr_valid[:, :, None]).sum(1) / cnt
            return self.head(torch.relu(self.lin(torch.cat([self_emb, agg], 1))))

    cnn = NucleusNodeCNN(out_dim=128).to(device)
    sage = Sage1().to(device)
    w = class_weights(labels[labels != -1], 4, max_ratio=10.0).to(device)
    opt = torch.optim.Adam(list(cnn.parameters()) + list(sage.parameters()), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    lossf = nn.CrossEntropyLoss(weight=w)
    rng = np.random.default_rng(0)

    train_idx = np.where((src != TEST_SLIDE) & (src != VAL_SLIDE) & (labels != -1))[0]
    val_idx = np.where((src == VAL_SLIDE) & (labels != -1))[0]
    test_idx = np.where((src == TEST_SLIDE) & (labels != -1))[0]
    batch = 256

    def step(targets, train):
        nb = nbr[targets]                                  # (B,k) global, -1 pad
        valid = torch.from_numpy((nb >= 0).astype(np.float32)).to(device)
        flat = np.where(nb >= 0, nb, targets[:, None]).reshape(-1)  # pad with self (masked out)
        self_emb = encode(targets)
        nbr_emb = encode(flat).reshape(len(targets), k, 128)
        logits = sage(self_emb, nbr_emb, valid)
        if train:
            y = torch.from_numpy(labels[targets]).long().to(device)
            loss = lossf(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        return logits.argmax(1).detach().cpu().numpy()

    @torch.no_grad()
    def evaluate(idx):
        cnn.eval(); sage.eval()
        preds = np.concatenate([step(idx[i:i + batch], train=False)
                                for i in range(0, len(idx), batch)])
        cnn.train(); sage.train()
        return f1_score(labels[idx], preds, average="macro", zero_division=0), preds

    best_val, patience = -1.0, 0
    for epoch in range(40):
        order = rng.permutation(train_idx)
        for i in range(0, len(order), batch):
            step(order[i:i + batch], train=True)
        sched.step()
        vf1, _ = evaluate(val_idx)
        logger.info(f"stage2 epoch {epoch}: val_macro_f1={vf1:.4f}")
        if vf1 > best_val:
            best_val, patience = vf1, 0
            torch.save({"cnn": cnn.state_dict(), "sage": sage.state_dict()}, OUT / "stage2_best.pt")
        else:
            patience += 1
            if patience >= 5:
                break
    ckpt = torch.load(OUT / "stage2_best.pt", weights_only=True)
    cnn.load_state_dict(ckpt["cnn"]); sage.load_state_dict(ckpt["sage"])
    _, tpred = evaluate(test_idx)
    truth = labels[test_idx]
    np.save(OUT / "stage2_pred.npy", tpred); np.save(OUT / "stage2_truth.npy", truth)
    (OUT / "stage2_metrics.json").write_text(json.dumps(
        {**_per_class(truth, tpred), "val_macro_f1": float(best_val),
         "baseline_effnet_macro_f1": 0.619}, indent=2))
    logger.info(f"stage2 done -> stage2_metrics.json (best val {best_val:.4f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["registry", "embed", "stage1", "gate", "stage2"])
    args = ap.parse_args()
    phases = {"registry": phase_registry, "embed": phase_embed, "stage1": phase_stage1,
              "gate": phase_gate, "stage2": phase_stage2}
    phases.get(args.phase, lambda: logger.error("phase not yet implemented"))()


if __name__ == "__main__":
    main()
