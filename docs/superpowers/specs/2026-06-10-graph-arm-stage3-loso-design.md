# Graph-Arm Stage 3 — Pluggable Probe Harness + Run-First Pair under LOSO

**Date:** 2026-06-10
**Branch:** `feat/spatial-gnn-probe` (Stage 3 of the spatial-GNN probe)
**Status:** Design approved, ready for implementation plan

## Context

The spatial-GNN probe (`pipeline_output/spatial_gnn_probe_2026_06/readout.md`) confirmed that a
within-slide spatial-neighbour graph on DAPI features helps the context-defined classes
(Endothelial/Stromal). Stage-2-proper (`scripts/spatial_gnn_probe.py::phase_stage2_proper`)
gave a clean two-arm ablation on context-poor 40px nucleus nodes: no-graph **0.537** →
graph **0.628** (+0.091 macro), graph arm ≈ production EfficientNet (0.619).

A 4-way fan-out brainstorm (codex / gemini / literature scout / codebase architecture critic)
produced a ranked improvement roadmap (`memory: project_graph_arm_improvement_roadmap`).
The two "run-first" experiments — cheap, low-risk, high-ROl — are:

- **E1 — Frozen-EffNet features into the learned graph.** Replace the from-scratch
  `NucleusNodeCNN` node encoder with the cached production-EffNet embeddings
  (`embeddings_pca128.npy`). All four sources ranked this the #1 move: the readout itself
  blames the from-scratch CNN as Stage-2's weak link.
- **E2 — Correct-and-Smooth / spatial smoothing post-hoc** on the production model's own
  class probabilities, over the same within-slide k-NN graph. Near-zero training cost; a
  cheap diagnostic of how much of the graph's benefit is "just" label diffusion.

**Decisions taken during brainstorming:**
- **Structural approach C** — refactor the Stage-2-proper harness into a pluggable evaluator
  `(encoder × aggregator × splitter × post_hoc)`; the run-first pair are configurations of it,
  and the Tier-2 swings (edge-geometry/GATv2, expression-teacher distillation) slot in later
  as new plugins. The refactor is **bounded**: only the four seams already implicit in
  `phase_stage2_proper`, justified by concrete near-term consumers — no speculative framework.
- **Evaluation = full leave-one-slide-out (LOSO)** across all 6 sources, for the strongest
  defense against confounded-split / composition-leakage concerns.

## Goal

Refactor the spatial-GNN harness into a pluggable `(encoder × aggregator × splitter × post_hoc)`
evaluator, and use it to answer — under LOSO — (E1) does feeding frozen production-EffNet
features into the learned graph beat a matched no-graph MLP, and (E2) how much of any gain a
near-free spatial smoother captures. Report the **leakage-immune graph-lift Δ** with honest
per-tier accounting.

## Key design principle — the LOSO leakage seam, turned into a strength

The production EffNet that generates the node features was itself trained on some of the 6 slides
(at minimum Xenium rep1). On folds whose held-out slide the extractor has seen, the **absolute**
test F1 carries mild optimism. But the scientific claim is the **graph-lift Δ = (graph arm −
no-graph arm)**, and that Δ is **leakage-immune**: both arms consume the *identical* frozen
features, so any extractor optimism cancels. The readout therefore reports two tiers:

- **Feature-clean folds** — held-out slide was NOT in the frozen extractor's training set
  (determined from the `breast_pooled_train` run config that produced the checkpoint). Absolute
  F1 AND Δ are both honest. **If that training set cannot be recovered, the readout conservatively
  reports ALL folds as Δ-clean only** and labels every absolute F1 "extractor-membership
  unverified" — the Δ claim never depends on knowing the manifest.
- **Δ-clean folds** — held-out slide WAS (or may have been) in the extractor's training set. Only
  the Δ is honest; the absolute F1 is flagged as optimistic.

This makes LOSO a genuine rigor upgrade rather than a confound generator.

## Architecture — the four seams

`phase_stage2_proper` already contains the four seams inline; Stage 3 extracts each into a
named, swappable unit. The two-arm ablation becomes "swap the `Aggregator`, hold encoder+head
identical" — a stronger equal-capacity guarantee than today's `if self.use_graph` branch.

| Seam | Today (inline in `phase_stage2_proper`) | New home | Plugins |
|---|---|---|---|
| **Encoder** | `NucleusNodeCNN` + `to_crop` (40px crop) | `src/dapidl/graph/encoders.py` (new) | `CropCNNEncoder` (trainable conv) · `FrozenFeatureEncoder` (cached pca128 lookup + optional `LayerNorm→Linear`) |
| **Aggregator** | `if use_graph: masked-mean else zeros` | `src/dapidl/graph/gnn.py` (extend) | `NoGraphAggregator` · `MeanAggregator` · *(future: `GATv2Aggregator`)* |
| **Splitter** | inline rep1-stripe val / rep2 test | `src/dapidl/graph/splits.py` (new, pure) | `Stage2ProperSplit` · `LOSOSplit` |
| **Post-hoc** | *(none)* | `src/dapidl/graph/smooth.py` (new, pure) | `IdentityPostHoc` · `SmoothCorrectSmooth` |

### Module layout & responsibilities

- **`src/dapidl/graph/harness.py` (new)** — the protocols, the `GraphArmModel`, and the
  train/eval/ablation core. Depends on: encoders, aggregators (via protocols), `probe_eval`
  (gates), torch, numpy. Orchestration only — no plugin internals.
- **`src/dapidl/graph/encoders.py` (new)** — `NodeEncoder` implementations. Each holds its own
  data reference (the `crops` array, or the `pca128` array) and exposes `encode(rows)→Tensor`,
  hiding whether features come from a conv or a table lookup.
- **`src/dapidl/graph/gnn.py` (extend)** — keep `NucleusNodeCNN`, `SageLayer`, `scatter_mean`;
  add `NoGraphAggregator`, `MeanAggregator` (thin `nn.Module`s over the existing masked-mean).
- **`src/dapidl/graph/splits.py` (new, pure)** — `Stage2ProperSplit`, `LOSOSplit`. Pure numpy
  over `(source, coords, labels)`; no torch, no IO. Fully unit-testable.
- **`src/dapidl/graph/smooth.py` (new, pure)** — `normalized_adjacency`, `smooth`,
  `correct_and_smooth`. Pure numpy/scipy; no torch, no IO. Fully unit-testable.
- **`scripts/spatial_gnn_probe.py` (extend)** — thin phases that *configure* the harness
  (`phase_logits`, `phase_stage3_loso`, `phase_cands_loso`, `phase_stage3_readout`), plus a
  re-expression of `phase_stage2_proper` through the harness for the characterization test.

### Interfaces (`harness.py`)

```python
class NodeEncoder(Protocol):
    out_dim: int
    def encode(self, rows: np.ndarray) -> Tensor: ...   # -> [len(rows), out_dim] on device
    def parameters(self) -> Iterable: ...               # trainable params (possibly empty)

class Aggregator(Protocol):
    # se:[B,d]  ne:[B,k,d]  valid:[B,k] (1.0 where neighbour exists) -> [B,d]
    def __call__(self, se: Tensor, ne: Tensor, valid: Tensor) -> Tensor: ...

class Splitter(Protocol):
    # yields (fold_name, train_idx, val_idx, test_idx); all exclude label==-1
    def folds(self) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray]]: ...

class PostHoc(Protocol):
    def apply(self, probs: np.ndarray, nbr: np.ndarray,
              train_idx: np.ndarray, train_labels: np.ndarray) -> np.ndarray: ...

class GraphArmModel(nn.Module):
    """encoder + aggregator + head. The two arms differ ONLY in `aggregator`."""
    def __init__(self, encoder: NodeEncoder, aggregator: Aggregator,
                 node_dim: int, hidden: int = 64, num_classes: int = 4): ...
    def forward(self, self_rows, nbr_rows, valid):
        se = self.encoder.encode(self_rows)                       # [B,d]
        ne = self.encoder.encode(nbr_rows.reshape(-1)).reshape(B, k, d)
        agg = self.aggregator(se, ne, valid)                      # NoGraph->zeros; Mean->masked mean
        return self.head(torch.relu(self.lin(torch.cat([se, agg], 1))))

@dataclass
class ArmResult:
    macro_f1: float
    per_class: dict[str, dict]      # {class: {"f1": float, "support": int}}
    val_macro_f1: float
    pred: np.ndarray                # argmax predictions over the fold's test_idx

def train_arm(encoder_factory: Callable[[], NodeEncoder], aggregator: Aggregator, *,
              nbr: np.ndarray, labels: np.ndarray,
              train_idx, val_idx, test_idx, k: int, num_classes: int,
              device: str, epochs: int = 40, patience: int = 5, seed: int = 0) -> ArmResult: ...

def run_ablation(encoder_factory, aggregators: dict[str, Aggregator],
                 splitter: Splitter, *, nbr, labels, post_hoc: PostHoc = IdentityPostHoc(),
                 **kw) -> dict: ...   # loops folds × arms; deltas + per-fold/pooled McNemar via probe_eval
```

`encoder_factory` (a callable, not an instance) so each arm/fold re-initialises weights cleanly.
The epoch loop, early-stopping (`patience=5`), `evaluate`, weighted `CrossEntropyLoss`
(`class_weights(..., max_ratio=10.0)`), per-class `precision_recall_fscore_support`, and the
`graph−nograph` delta move verbatim from `phase_stage2_proper` into `train_arm`/`run_ablation`.

## Components in detail

### `splits.py` (pure)
- `Stage2ProperSplit(source, coords, labels, val_frac=0.20)` — reproduces the current split:
  val = top-`val_frac` y-stripe of `xenium_rep1`; train = rest of rep1 + all STHELAR; test = rep2.
- `LOSOSplit(source, coords, labels, val_frac=0.20)` — `folds()` yields one fold per source
  (6 total). For held-out slide `S`: `test_idx` = all labeled cells of `S`; `val_idx` = the top
  `val_frac` y-stripe of **each training slide** pooled (spatially separates val from train
  within every training domain); `train_idx` = remaining labeled training-slide cells.
  `label == -1` cells are excluded from all three sets (they remain in the graph as context only).

### `encoders.py`
- `CropCNNEncoder(crops, off, crop_size, device, out_dim=128)` — wraps `NucleusNodeCNN`;
  `encode(rows)` normalises the 40px crop (`/65535`, `(x-0.485)/0.229`) and runs the conv.
  Trainable. Reproduces today's behaviour exactly (characterization target).
- `FrozenFeatureEncoder(pca128, device, proj_dim=None)` — `encode(rows)` returns
  `torch.from_numpy(pca128[rows]).to(device)`; if `proj_dim` set, applies a trainable
  `LayerNorm→Linear(128→proj_dim)`. The frozen features are not back-propagated.

### `gnn.py` aggregators
- `NoGraphAggregator()` — `__call__` returns `torch.zeros_like(se)` (the no-graph arm).
- `MeanAggregator()` — masked neighbour mean: `(ne * valid[...,None]).sum(1) / valid.sum(1).clamp_min(1)`
  (the existing line 316–317 logic).

### `smooth.py` (pure)
- `normalized_adjacency(nbr, n)` — build symmetric sparse `D^-1/2 A D^-1/2` from the `(n,k)`
  neighbour table (scipy CSR).
- `smooth(probs, adj, alpha, iters)` — PPR diffusion `p ← (1-alpha)·p0 + alpha·(adj @ p)`.
- `correct_and_smooth(probs, train_idx, train_labels, adj, alpha_c, alpha_s, iters)` — general
  Huang-et-al. C&S (residual correct + smooth). On a held-out-slide within-slide graph,
  `train_idx ∩ slide = ∅`, so the Correct step is naturally inert → smoothing-only (documented,
  not special-cased).

### `SmoothCorrectSmooth` post-hoc
Wraps `smooth.py` to the `PostHoc` protocol. Used by `phase_cands_loso`.

## Run-first pair as configs (`scripts/spatial_gnn_probe.py`)

- **`phase_logits` [GPU, once]** — mirror `embed.extract_embeddings` but apply
  `model.head` + softmax → `probs_production.npy` (N, 4). The honest C&S base predictor
  (pca128 is lossy → cannot reconstruct logits). Uses the same production checkpoint
  `phase_embed`/`extract_embeddings` already loads.
- **`phase_stage3_loso`** —
  `run_ablation(FrozenFeatureEncoder, {"nograph": NoGraphAggregator(), "graph": MeanAggregator()}, LOSOSplit())`.
  2 arms × 6 folds on cached `embeddings_pca128.npy`. → `stage3_loso_metrics.json`.
- **`phase_cands_loso` [CPU, ~free]** — per fold, `SmoothCorrectSmooth` over the held-out
  slide's within-slide graph vs raw `probs_production` argmax; plus a within-slide **transductive
  upper-bound diagnostic** (label a fraction of the held-out slide, run full C&S) to answer
  "is the GNN just diffusion?". → `cands_loso_metrics.json`.
- **`phase_stage3_readout`** — write `stage3_readout.md`: per-fold + pooled macro/per-class,
  **feature-clean vs Δ-clean tiers**, per-fold + pooled McNemar (via `probe_eval`), and the
  GNN-vs-smoother comparison.
- **`phase_stage2_proper` re-expressed through the harness** —
  `CropCNNEncoder + {NoGraph, Mean} + Stage2ProperSplit`. The characterization test asserts it
  reproduces the committed 0.537 / 0.628.

## Data flow

1. Reuse `phase_registry` (`spatial_registry.parquet`: source, x_px, y_px, coarse_idx) and
   `phase_embed` (`embeddings_pca128.npy`).
2. One-time `phase_logits` → `probs_production.npy`.
3. `LOSOSplit.folds()` yields 6 × (train, val, test). Per-slide `edge_index`/neighbour table
   cached to disk (shared by E1 and E2 — computed once per slide, not per phase).
4. E1: `run_ablation` trains 2 arms × 6 folds on cached features. E2: smoothing per fold.
5. `phase_stage3_readout` aggregates → `stage3_readout.md`.

## Eval & success criteria

- **E1:** graph-arm mean macro-F1 across folds > no-graph arm, with the **Δ** pooled-McNemar
  significant and concentrated on Endothelial/Stromal. Reported honestly: Δ is expected to shrink
  to ~+0.02 on strong frozen features (vs +0.091 on the weak from-scratch CNN) — both arms now
  start from a strong baseline (no-graph ≈ Stage-1's 0.699 ballpark).
- **E2:** does smoothing beat raw production argmax on held-out slides, and what fraction of E1's
  Δ does the cheap smoother capture? If smoothing ≈ GNN, the GNN is mostly label diffusion.
- **Tiers:** feature-clean (extractor never saw the held-out slide) vs Δ-clean (it did).
- **Honest ceiling:** ~0.68–0.73 macro realistic; NOT 0.80. Flag any fold whose gain looks
  too good (composition leakage).

## Error handling & edge cases

- **Tiny slide graphs:** a slide with < 2 cells (won't occur for these 6, but guard) → skip kNN;
  neighbour rows fall back to self (existing `np.where(nb>=0, nb, targets)` pattern).
- **All-test, no-train-labels in a held-out slide's graph:** expected; C&S Correct step inert →
  smoothing-only. Documented, asserted in a `smooth.py` test.
- **Class absent from a held-out slide:** `f1_score(..., zero_division=0)`; support recorded so
  the readout can weight/annotate.
- **OOM:** node features are cached (~1.2 GB for ~2.28M × 128 float32). The neighbour-sampled
  minibatch loop (`batch=256`) is preserved by the harness — `encode(rows)` is per-batch, never a
  dense `(E, d)` materialisation. Per-slide kNN reuses the chunked `cKDTree` builder.

## Testing strategy (TDD, pure cores first)

- `splits.py`: sets disjoint; `-1` excluded from all three; `LOSOSplit` yields 6 folds; each
  fold's test = exactly one slide; val drawn only from training slides.
- `smooth.py`: `alpha=0` is identity; output rows stay on the simplex (row-stochastic);
  iteration converges; Correct step inert when `train_idx` disjoint from the graph's slide.
- `encoders.py` / aggregators: `encode(rows)` output shape `[len(rows), out_dim]`;
  `NoGraphAggregator` returns zeros; `MeanAggregator` equals a hand-computed masked mean on a
  toy graph.
- `harness.py`: `GraphArmModel.forward` output shape `[B, num_classes]`; swapping the aggregator
  changes only the graph term.
- **Characterization test (required gate):** the harness configured as Stage-2-proper reproduces
  the committed `stage2_proper_metrics.json` (0.537 / 0.628) within tolerance. Pins the refactor
  before any new experiment runs. (Controller-run; marked slow/GPU, not a unit test.)

## Compute / RAM budget

- 1 GPU logit pass (`phase_logits`) over the p128 LMDB (~minutes).
- 12 GNN runs on cached features (2 arms × 6 folds), ~1–3 h total.
- 6 free CPU smooths.
- ~1.2 GB resident for cached features; comfortably within the 62 GB host.

## Out of scope (Tier-2 swings — plug in later via the same seams)

- Edge-geometry + GATv2 / multi-scale physical graph (new `Aggregator` + edge_attr from the kNN
  builder, which currently discards distances).
- Expression-teacher → DAPI-student LUPI distillation (new training objective / `post_hoc` or a
  soft-target loss in `train_arm`).
- Self-supervised pretraining, spatial-FM teachers, pure directional GNN — deprioritised per the
  roadmap.

These are deliberately deferred; the harness is designed so each is an additive plugin, not a
rewrite.
