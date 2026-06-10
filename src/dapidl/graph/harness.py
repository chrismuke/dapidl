"""Pluggable probe harness: GraphArmModel (encoder + aggregator + head) and the
train/eval/ablation loop extracted from phase_stage2_proper. The two arms of an
ablation differ ONLY in the aggregator. `k` is implicit in the (n, k) nbr table."""
from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import nn

CLASSES = ["Endothelial", "Epithelial", "Immune", "Stromal"]


class GraphArmModel(nn.Module):
    """encoder + aggregator + head. The two arms differ ONLY in `aggregator`. Neighbour
    features are encoded only when the aggregator needs them (no-graph skips the work)."""

    def __init__(self, encoder, aggregator, node_dim: int, hidden: int = 64, num_classes: int = 4):
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.lin = nn.Linear(node_dim * 2, hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, self_rows, nbr_rows, valid):
        se = self.encoder.encode(self_rows)                              # [B, d]
        if getattr(self.aggregator, "needs_neighbours", True):
            b, k = nbr_rows.shape
            ne = self.encoder.encode(nbr_rows.reshape(-1)).reshape(b, k, -1)
        else:
            ne = se[:, None, :]                                          # placeholder, unused
        agg = self.aggregator(se, ne, valid)                            # [B, d]
        return self.head(torch.relu(self.lin(torch.cat([se, agg], 1))))


@dataclass
class ArmResult:
    macro_f1: float
    per_class: dict
    val_macro_f1: float
    pred: np.ndarray


def train_arm(encoder_factory: Callable[[], nn.Module], aggregator, *, nbr, labels,
              train_idx, val_idx, test_idx, num_classes: int = 4, device: str = "cpu",
              epochs: int = 40, patience: int = 5, seed: int = 0, batch: int = 256,
              lr: float = 3e-4) -> ArmResult:
    """Train one arm with early stopping on val macro-F1, evaluate on test. Reproduces
    phase_stage2_proper's loop (Adam, cosine T_max=epochs, weighted CE with
    class_weights max_ratio=10.0)."""
    sys.path.insert(0, "scripts")
    from breast_pooled_train import class_weights

    torch.manual_seed(seed)
    encoder = encoder_factory()
    model = GraphArmModel(encoder, aggregator, node_dim=encoder.out_dim, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = class_weights(labels[train_idx], num_classes, max_ratio=10.0).to(device)
    lossf = nn.CrossEntropyLoss(weight=w)
    rng = np.random.default_rng(seed)

    def step(rows, train):
        nb = nbr[rows]
        valid = torch.from_numpy((nb >= 0).astype(np.float32)).to(device)
        safe = np.where(nb >= 0, nb, rows[:, None])
        logits = model(rows, safe, valid)
        if train:
            y = torch.from_numpy(labels[rows]).long().to(device)
            loss = lossf(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return logits.argmax(1).detach().cpu().numpy()

    @torch.no_grad()
    def evaluate(idx):
        if len(idx) == 0:
            return 0.0, np.empty(0, dtype=np.int64)
        model.eval()
        pred = np.concatenate([step(idx[i:i + batch], False) for i in range(0, len(idx), batch)])
        model.train()
        return f1_score(labels[idx], pred, average="macro", zero_division=0), pred

    best_val, wait, best_state = -1.0, 0, None
    for _ in range(epochs):
        order = rng.permutation(train_idx)
        for i in range(0, len(order), batch):
            step(order[i:i + batch], True)
        sched.step()
        vf1, _ = evaluate(val_idx)
        if vf1 > best_val:
            best_val, wait = vf1, 0
            best_state = {kk: v.cpu().clone() for kk, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    _, tpred = evaluate(test_idx)
    truth = labels[test_idx]
    _, _, f1, sup = precision_recall_fscore_support(
        truth, tpred, labels=list(range(num_classes)), zero_division=0)
    per_class = {CLASSES[c]: {"f1": float(f1[c]), "support": int(sup[c])} for c in range(num_classes)}
    return ArmResult(float(f1_score(truth, tpred, average="macro", zero_division=0)),
                     per_class, float(best_val), tpred)


def run_ablation(encoder_factory, aggregators: dict, splitter, *, nbr, labels,
                 num_classes: int = 4, device: str = "cpu", **train_kw) -> dict:
    """Loop folds x arms; per fold record each arm's metrics, the graph-nograph delta,
    and per-fold McNemar; then pool predictions across folds for a pooled macro, pooled
    per-class, and pooled McNemar. Requires arm tags 'nograph' and 'graph' for the delta."""
    from dapidl.graph.probe_eval import mcnemar_test

    out: dict = {"folds": {}, "pooled": {}}
    pooled_pred = {tag: [] for tag in aggregators}
    pooled_truth: list[np.ndarray] = []
    for name, tr, va, te in splitter.folds():
        fold: dict = {}
        preds: dict = {}
        for tag, agg in aggregators.items():
            res = train_arm(encoder_factory, agg, nbr=nbr, labels=labels,
                            train_idx=tr, val_idx=va, test_idx=te,
                            num_classes=num_classes, device=device, **train_kw)
            fold[tag] = {"macro_f1": res.macro_f1, "per_class": res.per_class,
                         "val_macro_f1": res.val_macro_f1}
            preds[tag] = res.pred
            pooled_pred[tag].append(res.pred)
        truth = labels[te]
        pooled_truth.append(truth)
        if "graph" in preds and "nograph" in preds:
            fold["delta_macro"] = round(fold["graph"]["macro_f1"] - fold["nograph"]["macro_f1"], 4)
            fold["mcnemar_graph_vs_nograph"] = mcnemar_test(truth, preds["nograph"], preds["graph"])
        out["folds"][name] = fold

    if "graph" in aggregators and "nograph" in aggregators and pooled_truth:
        T = np.concatenate(pooled_truth)
        G = np.concatenate(pooled_pred["graph"])
        N = np.concatenate(pooled_pred["nograph"])
        _, _, f1g, _ = precision_recall_fscore_support(T, G, labels=list(range(num_classes)), zero_division=0)
        out["pooled"] = {
            "macro_graph": float(f1_score(T, G, average="macro", zero_division=0)),
            "macro_nograph": float(f1_score(T, N, average="macro", zero_division=0)),
            "per_class_graph": {CLASSES[c]: float(f1g[c]) for c in range(num_classes)},
            "mcnemar_graph_vs_nograph": mcnemar_test(T, N, G),
        }
        out["pooled"]["delta_macro"] = round(out["pooled"]["macro_graph"] - out["pooled"]["macro_nograph"], 4)
    return out
