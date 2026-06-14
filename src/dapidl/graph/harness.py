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

    def forward(self, self_rows, nbr_rows, valid, edge_attr=None):
        se = self.encoder.encode(self_rows)                              # [B, d]
        if getattr(self.aggregator, "needs_neighbours", True):
            b, k = nbr_rows.shape
            ne = self.encoder.encode(nbr_rows.reshape(-1)).reshape(b, k, -1)
        else:
            ne = se[:, None, :]                                          # placeholder, unused
        if getattr(self.aggregator, "needs_edge_attr", False):
            agg = self.aggregator(se, ne, valid, edge_attr)
        else:
            agg = self.aggregator(se, ne, valid)
        return self.head(torch.relu(self.lin(torch.cat([se, agg], 1))))


@dataclass
class ArmResult:
    macro_f1: float
    per_class: dict
    val_macro_f1: float
    pred: np.ndarray


def train_arm(encoder_factory: Callable[[], nn.Module], aggregator_factory: Callable[[], nn.Module], *,
              nbr, labels, train_idx, val_idx, test_idx, num_classes: int = 4, device: str = "cpu",
              epochs: int = 40, patience: int = 5, seed: int = 0, batch: int = 256,
              lr: float = 3e-4, edge_attr=None) -> ArmResult:
    """Train one arm with early stopping on val macro-F1, evaluate on test. Reproduces
    phase_stage2_proper's loop (Adam, cosine T_max=epochs, weighted CE with
    class_weights max_ratio=10.0). aggregator_factory is called fresh per invocation."""
    sys.path.insert(0, "scripts")
    from breast_pooled_train import class_weights

    torch.manual_seed(seed)
    encoder = encoder_factory()
    aggregator = aggregator_factory()
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
        ea = torch.from_numpy(edge_attr[rows]).float().to(device) if edge_attr is not None else None
        logits = model(rows, safe, valid, ea)
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


def run_ablation(encoder_factory, aggregator_factories: dict, splitter, *, nbr, labels,
                 num_classes: int = 4, device: str = "cpu", compare_pairs=None, **train_kw) -> dict:
    """Loop folds x arms (each arm an aggregator FACTORY -> fresh module per fold). For each
    (baseline, candidate) in compare_pairs (default [("nograph","graph")]) record per-fold and
    pooled delta_macro + McNemar. train_kw (incl. edge_attr=...) is forwarded to train_arm."""
    from dapidl.graph.probe_eval import mcnemar_test
    pairs = compare_pairs if compare_pairs is not None else [("nograph", "graph")]

    out: dict = {"folds": {}, "pooled": {}}
    pooled_pred = {tag: [] for tag in aggregator_factories}
    pooled_truth: list[np.ndarray] = []
    for name, tr, va, te in splitter.folds():
        fold: dict = {}
        preds: dict = {}
        for tag, fac in aggregator_factories.items():
            res = train_arm(encoder_factory, fac, nbr=nbr, labels=labels,
                            train_idx=tr, val_idx=va, test_idx=te,
                            num_classes=num_classes, device=device, **train_kw)
            fold[tag] = {"macro_f1": res.macro_f1, "per_class": res.per_class,
                         "val_macro_f1": res.val_macro_f1}
            preds[tag] = res.pred
            pooled_pred[tag].append(res.pred)
        truth = labels[te]
        pooled_truth.append(truth)
        for base, cand in pairs:
            if base in preds and cand in preds:
                fold[f"delta_{cand}_vs_{base}"] = round(fold[cand]["macro_f1"] - fold[base]["macro_f1"], 4)
                fold[f"mcnemar_{cand}_vs_{base}"] = mcnemar_test(truth, preds[base], preds[cand])
                if (base, cand) == ("nograph", "graph"):            # Stage-3 backward-compat alias
                    fold["delta_macro"] = fold[f"delta_{cand}_vs_{base}"]
        out["folds"][name] = fold

    if pooled_truth:
        T = np.concatenate(pooled_truth)
        for tag in aggregator_factories:
            P = np.concatenate(pooled_pred[tag])
            _, _, f1c, _ = precision_recall_fscore_support(T, P, labels=list(range(num_classes)), zero_division=0)
            out["pooled"][f"macro_{tag}"] = float(f1_score(T, P, average="macro", zero_division=0))
            out["pooled"][f"per_class_{tag}"] = {CLASSES[c]: float(f1c[c]) for c in range(num_classes)}
        for base, cand in pairs:
            if base in aggregator_factories and cand in aggregator_factories:
                B = np.concatenate(pooled_pred[base])
                C = np.concatenate(pooled_pred[cand])
                out["pooled"][f"delta_{cand}_vs_{base}"] = round(
                    out["pooled"][f"macro_{cand}"] - out["pooled"][f"macro_{base}"], 4)
                out["pooled"][f"mcnemar_{cand}_vs_{base}"] = mcnemar_test(T, B, C)
                if (base, cand) == ("nograph", "graph"):            # Stage-3 backward-compat alias
                    out["pooled"]["delta_macro"] = out["pooled"][f"delta_{cand}_vs_{base}"]
    return out
