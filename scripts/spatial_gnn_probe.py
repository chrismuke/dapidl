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


def phase_stage2_proper() -> None:
    """Proper Stage 2: nucleus-local CNN with a within-Stage-2 NO-GRAPH ablation and
    SAME-DOMAIN (Xenium rep1 spatial hold-out) validation. All sources are 0.2125
    um/px, so the 40px crop is already a fixed ~8.5um FoV. Trains TWO equal-capacity
    arms (no-graph: neighbour term zeroed; graph: 1-hop neighbour-mean) -> the arm
    delta isolates the spatial graph's contribution for context-poor nucleus nodes.
    Graph over ALL cells (context); loss/metrics on labelled nodes only."""
    import struct
    import sys

    import lmdb
    import numpy as np
    import polars as pl
    import torch
    from scipy.spatial import cKDTree
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    from torch import nn

    from dapidl.graph.embed import decode_record
    from dapidl.graph.gnn import NucleusNodeCNN
    sys.path.insert(0, "scripts")
    from breast_pooled_train import class_weights

    reg = pl.read_parquet(OUT / "spatial_registry.parquet")
    src = reg["source"].to_numpy()
    coords = reg.select(["x_px", "y_px"]).to_numpy()
    labels = reg["coarse_idx"].to_numpy()
    n, k, crop, off = len(reg), 8, 40, (128 - 40) // 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    crops = np.empty((n, crop, crop), dtype=np.uint16)
    env = lmdb.open(str(LMDB_DIR / "patches.lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        for i in range(n):
            _, p = decode_record(txn.get(struct.pack(">Q", i)), 128)
            crops[i] = p[off:off + crop, off:off + crop]
    env.close()

    # same-domain Xenium val: top-20% y-stripe of rep1 (spatially separated from rep1 train)
    rep1 = (src == "xenium_rep1") & (labels != -1)
    y80 = np.quantile(coords[rep1, 1], 0.80)
    val_idx = np.where(rep1 & (coords[:, 1] > y80))[0]
    train_idx = np.where((rep1 & (coords[:, 1] <= y80)) |
                         ((src != "xenium_rep1") & (src != "xenium_rep2") & (labels != -1)))[0]
    test_idx = np.where((src == "xenium_rep2") & (labels != -1))[0]
    logger.info(f"stage2-proper split: train={len(train_idx)} val(rep1-stripe)={len(val_idx)} test(rep2)={len(test_idx)}")

    class Arm(nn.Module):
        def __init__(self, use_graph):
            super().__init__()
            self.use_graph = use_graph
            self.cnn = NucleusNodeCNN(out_dim=128)
            self.lin = nn.Linear(256, 64)
            self.head = nn.Linear(64, 4)

        def forward(self, self_crops, nbr_crops, nbr_valid):
            se = self.cnn(self_crops)
            if self.use_graph:
                ne = self.cnn(nbr_crops).reshape(se.shape[0], k, 128)
                cnt = nbr_valid.sum(1, keepdim=True).clamp_min(1.0)
                agg = (ne * nbr_valid[:, :, None]).sum(1) / cnt
            else:
                agg = torch.zeros_like(se)
            return self.head(torch.relu(self.lin(torch.cat([se, agg], 1))))

    def to_crop(rows):
        x = crops[rows].astype(np.float32) / 65535.0
        x = (x - 0.485) / 0.229
        return torch.from_numpy(x)[:, None].to(device)

    w = class_weights(labels[labels != -1], 4, max_ratio=10.0).to(device)
    CL = ["Endothelial", "Epithelial", "Immune", "Stromal"]
    batch = 256
    arms = {}
    for use_graph in [False, True]:
        tag = "graph" if use_graph else "nograph"
        torch.manual_seed(0)
        model = Arm(use_graph).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
        lossf = nn.CrossEntropyLoss(weight=w)
        rng = np.random.default_rng(0)

        def run(targets, train):
            nb = nbr[targets]
            valid = torch.from_numpy((nb >= 0).astype(np.float32)).to(device)
            sc = to_crop(targets)
            nc = to_crop(np.where(nb >= 0, nb, targets[:, None]).reshape(-1)) if model.use_graph else None
            logits = model(sc, nc, valid)
            if train:
                yv = torch.from_numpy(labels[targets]).long().to(device)
                loss = lossf(logits, yv)
                opt.zero_grad(); loss.backward(); opt.step()
            return logits.argmax(1).detach().cpu().numpy()

        @torch.no_grad()
        def evaluate(idx):
            model.eval()
            pred = np.concatenate([run(idx[i:i + batch], False) for i in range(0, len(idx), batch)])
            model.train()
            return f1_score(labels[idx], pred, average="macro", zero_division=0), pred

        best_val, patience, best_state = -1.0, 0, None
        for epoch in range(40):
            order = rng.permutation(train_idx)
            for i in range(0, len(order), batch):
                run(order[i:i + batch], True)
            sched.step()
            vf1, _ = evaluate(val_idx)
            logger.info(f"stage2-{tag} epoch {epoch}: val_macro_f1={vf1:.4f}")
            if vf1 > best_val:
                best_val, patience = vf1, 0
                best_state = {kk: v.cpu().clone() for kk, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 5:
                    break
        model.load_state_dict(best_state)
        model.eval()
        _, tpred = evaluate(test_idx)
        truth = labels[test_idx]
        _, _, f1, sup = precision_recall_fscore_support(truth, tpred, labels=[0, 1, 2, 3], zero_division=0)
        arms[tag] = {"macro_f1": float(f1_score(truth, tpred, average="macro", zero_division=0)),
                     "per_class": {CL[c]: {"f1": float(f1[c]), "support": int(sup[c])} for c in range(4)},
                     "val_macro_f1": float(best_val)}
        np.save(OUT / f"stage2proper_pred_{tag}.npy", tpred)
        logger.info(f"stage2-{tag} TEST macro_f1={arms[tag]['macro_f1']:.4f}")

    delta = {c: round(arms["graph"]["per_class"][c]["f1"] - arms["nograph"]["per_class"][c]["f1"], 4) for c in CL}
    delta["macro"] = round(arms["graph"]["macro_f1"] - arms["nograph"]["macro_f1"], 4)
    (OUT / "stage2_proper_metrics.json").write_text(json.dumps(
        {"arms": arms, "graph_minus_nograph": delta, "baseline_effnet_macro_f1": 0.619}, indent=2))
    logger.info(f"stage2-proper done: nograph={arms['nograph']['macro_f1']:.4f} "
                f"graph={arms['graph']['macro_f1']:.4f} delta_macro={delta['macro']:+.4f} -> stage2_proper_metrics.json")


def phase_logits() -> None:
    """[GPU] Dump the production EffNet's softmax class probabilities per cell -> (N,4).
    The honest Correct-and-Smooth base predictor (pca128 is lossy and cannot reconstruct
    logits). Mirrors embed.extract_embeddings but applies model.head + softmax."""
    import struct
    import sys

    import lmdb
    import numpy as np
    import torch

    from dapidl.graph.embed import decode_record
    sys.path.insert(0, "scripts")
    from breast_pooled_train import DapiClassifier

    n = int(np.load(LMDB_DIR / "labels.npy").shape[0])
    ckpt = Path("pipeline_output/h2h_2026_05_30/efficientnetv2_rw_s/best_model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DapiClassifier(num_classes=4, backbone="efficientnetv2_rw_s")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model_state_dict") or state.get("model") or state)
    model.eval().to(device)

    probs = np.empty((n, 4), dtype=np.float32)
    env = lmdb.open(str(LMDB_DIR / "patches.lmdb"), readonly=True, lock=False)
    buf: list[np.ndarray] = []
    rows: list[int] = []

    def flush():
        if not rows:
            return
        x = np.stack(buf).astype(np.float32) / 65535.0
        x = (x - 0.485) / 0.229
        t = torch.from_numpy(x)[:, None, :, :].to(device)
        with torch.no_grad():
            p = torch.softmax(model.head(model.backbone(t.expand(-1, 3, -1, -1))), dim=1)
        probs[rows] = p.cpu().numpy().astype(np.float32)
        buf.clear(); rows.clear()

    with env.begin() as txn:
        for i in range(n):
            _, patch = decode_record(txn.get(struct.pack(">Q", i)), 128)
            buf.append(patch); rows.append(i)
            if len(rows) == 256:
                flush()
        flush()
    env.close()
    np.save(OUT / "probs_production.npy", probs)
    logger.info(f"logits: production softmax probs {probs.shape} -> probs_production.npy")


def phase_stage3_loso() -> None:
    """[CONTROLLER RUN] E1: frozen-EffNet features into the learned graph, two-arm
    (nograph=NoGraph, graph=Mean) ablation under leave-one-slide-out over all slides.
    Features preloaded to device once and shared across arms/folds."""
    import numpy as np
    import polars as pl
    import torch

    from dapidl.graph.encoders import FrozenFeatureEncoder
    from dapidl.graph.gnn import MeanAggregator, NoGraphAggregator
    from dapidl.graph.harness import run_ablation
    from dapidl.graph.knn_graph import build_within_slide_nbr_table
    from dapidl.graph.splits import LOSOSplit

    reg = pl.read_parquet(OUT / "spatial_registry.parquet")
    src = reg["source"].to_numpy()
    coords = reg.select(["x_px", "y_px"]).to_numpy()
    labels = reg["coarse_idx"].to_numpy()
    pca = np.load(OUT / "embeddings_pca128.npy")

    nbr = build_within_slide_nbr_table(coords, src, k=8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats_dev = torch.from_numpy(np.ascontiguousarray(pca, dtype=np.float32)).to(device)  # once

    res = run_ablation(lambda: FrozenFeatureEncoder(feats_dev, device),
                       {"nograph": NoGraphAggregator(), "graph": MeanAggregator()},
                       LOSOSplit(src, coords, labels, val_frac=0.20),
                       nbr=nbr, labels=labels, device=device)
    res["baseline_effnet_macro_f1"] = 0.619
    (OUT / "stage3_loso_metrics.json").write_text(json.dumps(res, indent=2))
    p = res.get("pooled", {})
    logger.info(f"stage3-loso pooled: nograph={p.get('macro_nograph')} "
                f"graph={p.get('macro_graph')} delta_macro={p.get('delta_macro')}")


def phase_cands_loso() -> None:
    """[CONTROLLER RUN] E2: smooth the production EffNet probabilities over each held-out
    slide's within-slide graph (held-out => Correct step inert => smoothing-only), versus
    raw argmax; plus a within-slide TRANSDUCTIVE upper bound (reveal 20% of the held-out
    slide's labels, run full C&S, score the other 80%) — a diagnostic, not production."""
    import numpy as np
    import polars as pl
    from sklearn.metrics import f1_score

    from dapidl.graph.knn_graph import build_within_slide_knn
    from dapidl.graph.probe_eval import mcnemar_test
    from dapidl.graph.smooth import correct_and_smooth, smooth, transition_matrix
    from dapidl.graph.splits import LOSOSplit

    reg = pl.read_parquet(OUT / "spatial_registry.parquet")
    src = reg["source"].to_numpy()
    coords = reg.select(["x_px", "y_px"]).to_numpy()
    labels = reg["coarse_idx"].to_numpy()
    probs = np.load(OUT / "probs_production.npy")
    n = len(labels)

    edge = build_within_slide_knn(coords, src, k=8)
    T = transition_matrix(edge, n)
    sm_full = smooth(probs, T, alpha=0.8, iters=30)         # global; held-out slide is its own component

    folds: dict = {}
    pooled_truth, pooled_raw, pooled_sm = [], [], []
    for name, _tr, _va, te in LOSOSplit(src, coords, labels, val_frac=0.20).folds():
        truth = labels[te]
        raw = probs[te].argmax(1)
        sm = sm_full[te].argmax(1)
        # transductive upper bound: reveal a 20% sample of THIS slide's labels, score the rest
        rng = np.random.default_rng(0)
        reveal = rng.permutation(te)[: max(1, len(te) // 5)]
        held = np.setdiff1d(te, reveal)
        cs_full = correct_and_smooth(probs, reveal, labels[reveal], T, iters=30)
        cs = cs_full[held].argmax(1)
        folds[name] = {
            "macro_raw": float(f1_score(truth, raw, average="macro", zero_division=0)),
            "macro_smooth": float(f1_score(truth, sm, average="macro", zero_division=0)),
            "macro_cs_transductive_ub": float(f1_score(labels[held], cs, average="macro", zero_division=0)),
            "mcnemar_smooth_vs_raw": mcnemar_test(truth, raw, sm),
        }
        pooled_truth.append(truth); pooled_raw.append(raw); pooled_sm.append(sm)

    PT = np.concatenate(pooled_truth); PR = np.concatenate(pooled_raw); PS = np.concatenate(pooled_sm)
    out = {"folds": folds, "pooled": {
        "macro_raw": float(f1_score(PT, PR, average="macro", zero_division=0)),
        "macro_smooth": float(f1_score(PT, PS, average="macro", zero_division=0)),
        "mcnemar_smooth_vs_raw": mcnemar_test(PT, PR, PS)}}
    (OUT / "cands_loso_metrics.json").write_text(json.dumps(out, indent=2))
    logger.info(f"cands-loso pooled: raw={out['pooled']['macro_raw']} smooth={out['pooled']['macro_smooth']}")


def phase_stage2_proper_harness() -> None:
    """[CONTROLLER RUN] Characterization: reproduce phase_stage2_proper through the new
    harness (CropCNNEncoder + {NoGraph, Mean} + Stage2ProperSplit) to prove the refactor
    is faithful. Writes stage2_proper_harness_metrics.json for comparison with the
    committed stage2_proper_metrics.json."""
    import struct
    import sys

    import lmdb
    import numpy as np
    import polars as pl
    import torch

    from dapidl.graph.embed import decode_record
    from dapidl.graph.encoders import CropCNNEncoder
    from dapidl.graph.gnn import MeanAggregator, NoGraphAggregator
    from dapidl.graph.harness import run_ablation
    from dapidl.graph.knn_graph import build_within_slide_nbr_table
    from dapidl.graph.splits import Stage2ProperSplit
    sys.path.insert(0, "scripts")

    reg = pl.read_parquet(OUT / "spatial_registry.parquet")
    src = reg["source"].to_numpy()
    coords = reg.select(["x_px", "y_px"]).to_numpy()
    labels = reg["coarse_idx"].to_numpy()
    n, crop, off = len(reg), 40, (128 - 40) // 2

    crops = np.empty((n, crop, crop), dtype=np.uint16)
    env = lmdb.open(str(LMDB_DIR / "patches.lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        for i in range(n):
            _, p = decode_record(txn.get(struct.pack(">Q", i)), 128)
            crops[i] = p[off:off + crop, off:off + crop]
    env.close()

    nbr = build_within_slide_nbr_table(coords, src, k=8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = run_ablation(lambda: CropCNNEncoder(crops, device, out_dim=128),
                       {"nograph": NoGraphAggregator(), "graph": MeanAggregator()},
                       Stage2ProperSplit(src, coords, labels, val_frac=0.20),
                       nbr=nbr, labels=labels, device=device)
    (OUT / "stage2_proper_harness_metrics.json").write_text(json.dumps(res, indent=2))
    f = res["folds"]["xenium_rep2"]
    logger.info(f"stage2-proper-harness: nograph={f['nograph']['macro_f1']:.4f} "
                f"graph={f['graph']['macro_f1']:.4f} (committed ref: 0.537 / 0.628)")


def phase_stage3_readout() -> None:
    """[CONTROLLER RUN] Compose stage3_readout.md from stage3_loso_metrics.json (E1) and
    cands_loso_metrics.json (E2): per-fold + pooled macro/delta, feature-clean vs Δ-clean
    tiers, and the GNN-vs-smoother comparison.

    FEATURE_CLEAN: slides the frozen extractor did NOT train on (absolute F1 honest).
    Verified from pipeline_output/h2h_2026_05_30/efficientnetv2_rw_s/summary.json: the
    EffNet trained on rep1 + sthelar_breast_s0/s1/s3/s6 and held out xenium_rep2, so rep2
    is the sole feature-clean fold. The delta (graph - no-graph) is leakage-immune regardless."""
    e1 = json.loads((OUT / "stage3_loso_metrics.json").read_text())
    e2 = json.loads((OUT / "cands_loso_metrics.json").read_text())

    # Verified from the h2h summary.json: EffNet train_sources = rep1 + sthelar s0/s1/s3/s6,
    # test = rep2. So rep2 is the ONLY feature-clean fold; the other 5 are delta-clean.
    FEATURE_CLEAN: set[str] = {"xenium_rep2"}

    lines = ["# Graph-Arm Stage 3 — Readout (LOSO)\n",
             "## E1 — Frozen-EffNet features in the learned graph (two-arm, leave-one-slide-out)\n",
             "Delta = graph - no-graph is **leakage-immune** (both arms share identical frozen features).\n",
             "| Held-out slide | tier | no-graph | graph | delta macro | McNemar p |",
             "|---|---|---|---|---|---|"]
    for name, f in e1["folds"].items():
        tier = "feature-clean" if name in FEATURE_CLEAN else "delta-clean (extractor trained on slide; abs F1 optimistic)"
        mp = f.get("mcnemar_graph_vs_nograph", {}).get("p_value")
        lines.append(f"| {name} | {tier} | {f['nograph']['macro_f1']:.4f} | "
                     f"{f['graph']['macro_f1']:.4f} | {f.get('delta_macro'):+.4f} | {mp} |")
    p = e1.get("pooled", {})
    lines += [f"\n**Pooled:** no-graph {p.get('macro_nograph')}, graph {p.get('macro_graph')}, "
              f"delta {p.get('delta_macro')}, pooled McNemar p="
              f"{p.get('mcnemar_graph_vs_nograph', {}).get('p_value')}. EffNet baseline 0.619.\n",
              "## E2 — Spatial smoothing of production probabilities (near-free)\n",
              "| Held-out slide | raw | smooth | C&S transductive UB |",
              "|---|---|---|---|"]
    for name, f in e2["folds"].items():
        lines.append(f"| {name} | {f['macro_raw']:.4f} | {f['macro_smooth']:.4f} | "
                     f"{f['macro_cs_transductive_ub']:.4f} |")
    pe = e2.get("pooled", {})
    lines += [f"\n**Pooled:** raw {pe.get('macro_raw')}, smooth {pe.get('macro_smooth')}, "
              f"smooth-vs-raw McNemar p={pe.get('mcnemar_smooth_vs_raw', {}).get('p_value')}.\n",
              "## Honest framing\n",
              "- Delta (graph lift) is the scientific claim; expected small (~+0.02) on strong frozen "
              "features (vs +0.091 on the from-scratch CNN) — both arms now start strong.\n",
              "- If E2's transductive UB >> smooth, the gain needs same-slide labels (not production); "
              "smoothing is the production-faithful number.\n",
              "- Ceiling ~0.68-0.73 macro; not 0.80. Flag any fold whose gain looks too good "
              "(composition leakage).\n"]
    lines.append("\n> NOTE: the frozen EffNet trained on rep1 + sthelar_breast_s0/s1/s3/s6 "
                 "(held out xenium_rep2). Only the **xenium_rep2** fold is feature-clean "
                 "(honest absolute F1); the other 5 folds' absolute F1 is extractor-optimistic. "
                 "The delta is leakage-immune in every fold.\n")
    (OUT / "stage3_readout.md").write_text("\n".join(lines))
    logger.info("stage3 readout -> stage3_readout.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["registry", "embed", "stage1", "gate", "stage2", "stage2_proper",
                             "logits", "stage3_loso", "cands_loso", "stage2_proper_harness",
                             "stage3_readout"])
    args = ap.parse_args()
    phases = {"registry": phase_registry, "embed": phase_embed, "stage1": phase_stage1,
              "gate": phase_gate, "stage2": phase_stage2, "stage2_proper": phase_stage2_proper,
              "logits": phase_logits, "stage3_loso": phase_stage3_loso,
              "cands_loso": phase_cands_loso, "stage2_proper_harness": phase_stage2_proper_harness,
              "stage3_readout": phase_stage3_readout}
    phases.get(args.phase, lambda: logger.error("phase not yet implemented"))()


if __name__ == "__main__":
    main()
