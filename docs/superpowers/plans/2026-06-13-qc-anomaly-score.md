# QC Embedding-Anomaly Score (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an unsupervised, embedding-based "broken-crop" anomaly score to the p64 DAPI QC pipeline — benchmarking DINOv2 vs NuSPIRe embedders, fair to rare classes (hard fairness stop), LOSO-scored, surfaced in a new Label Studio comparison project.

**Architecture:** Frozen vision-FM embeddings → per-slide stratified non-broken memory bank (same-slide masked) → kNN anomaly score → LOSO driver → validation + embedder selection → new Label Studio project (anomaly-disagreement selection; project 5 stays as stratified baseline).

**Tech Stack:** Python 3.11 (`uv run`), polars, numpy, scikit-learn, torch (DINOv2 via torch.hub/timm; NuSPIRe in-repo), lmdb, Pillow, matplotlib, requests. Design: `docs/superpowers/specs/2026-06-13-qc-anomaly-score-design.md`.

**Worktree:** execute in `/mnt/work/git/dapidl-p64qc` on `feat/p64-qc-collages` (already isolated). New code: `src/dapidl/qc/embeddings.py`, `src/dapidl/qc/anomaly.py`. New/extended scripts: `scripts/qc_anomaly_score.py`, `scripts/qc_review_ls_build.py`, `scripts/qc_review_ls_push.py`. Tests: `tests/qc/`.

**Dataset constants:** `DSET = /mnt/work/datasets/derived/breast-pilot-6source-dapi-p64-nuc-v1`; crops in `DSET/patches.lmdb` (key `struct.pack(">Q", row_idx)`, value `8-byte header + uint16 bytes`, 64×64); `DSET/qc/seg_scores.parquet` has `row_idx, slide, coarse_idx, broken, broken_reason, grade` + axes; `DSET/class_mapping.json = {Endothelial:0, Epithelial:1, Immune:2, Stromal:3}`.

---

## File Structure

- `src/dapidl/qc/anomaly.py` — PURE: `select_bank_indices`, `coreset_subsample`, `knn_anomaly_score`, `score_all_slides_loso`. No torch.
- `src/dapidl/qc/anomaly_eval.py` — PURE: `auroc`, `per_class_anomaly`, `fairness_pass`, `select_embedder`.
- `src/dapidl/qc/embeddings.py` — torch: `preprocess_dinov2`, `preprocess_nuspire`, `EMBEDDERS`, `compute_embeddings`.
- `scripts/qc_anomaly_score.py` — driver (score + `--eval` + missed-break montage). [CONTROLLER RUN for GPU]
- `scripts/qc_review_ls_build.py` — extend: carry `anomaly_score`/`anomaly_pct`, add `--select`.
- `scripts/qc_review_ls_push.py` — reuse `--create` to push the anomaly-ranked project. [CONTROLLER RUN]
- `tests/qc/test_anomaly.py`, `tests/qc/test_anomaly_eval.py`, `tests/qc/test_embeddings_preprocess.py`, `tests/qc/test_review_select.py`.

---

## Task 0: Commit existing QC review scripts (clean baseline)

**Files:** Modify (commit): `scripts/qc_review_ls_build.py`, `qc_review_ls_push.py`, `qc_review_ls_pull.py`, `docs/qc-review-tooling-test.md`.

- [ ] **Step 1: Commit the untracked Track-0 review scripts so the branch is clean before Track-1 code.**

```bash
cd /mnt/work/git/dapidl-p64qc
git add scripts/qc_review_ls_build.py scripts/qc_review_ls_push.py scripts/qc_review_ls_pull.py scripts/qc_review_export.py docs/qc-review-tooling-test.md 2>/dev/null
git commit -q -m "feat(qc): Label Studio p64 review loop (build/push/pull) + tooling guide"
git log --oneline -1
```
Expected: a new commit; `git status` no longer lists those scripts as untracked.

---

## Task 1: `anomaly.py` — `select_bank_indices` (memory-bank selection)

**Files:** Create `src/dapidl/qc/anomaly.py`; Test `tests/qc/test_anomaly.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/qc/test_anomaly.py
import numpy as np
from dapidl.qc.anomaly import select_bank_indices


def test_select_bank_excludes_slide_and_broken_caps_per_class():
    rows = np.arange(12)
    slides = np.array(["A"] * 6 + ["B"] * 6)
    coarse = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 2, 2])  # classes 0,1,2
    broken = np.array([False, False, True, False, False, False,
                       False, False, False, False, False, False])
    grades = np.array(["Good"] * 12)
    rng = np.random.default_rng(0)

    idx = select_bank_indices(rows, slides, coarse, broken, grades,
                              exclude_slide="B", per_class_cap=2, rng=rng)

    chosen = set(idx.tolist())
    assert chosen.issubset(set(range(6)))          # only slide A
    assert 2 not in chosen                          # broken excluded
    # per-class cap: class 0 in slide A has rows {0,1} (2 valid) -> at most 2
    cls0 = [i for i in idx if coarse[i] == 0]
    assert len(cls0) <= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/qc/test_anomaly.py::test_select_bank_excludes_slide_and_broken_caps_per_class -v`
Expected: FAIL with `ImportError`/`ModuleNotFoundError` (anomaly.py missing).

- [ ] **Step 3: Write minimal implementation**

```python
# src/dapidl/qc/anomaly.py
"""Memory-bank kNN anomaly scoring for DAPI QC crops (pure numpy/sklearn — no torch)."""
from __future__ import annotations

import numpy as np


def select_bank_indices(rows, slides, coarse_idx, broken, grades, *,
                        exclude_slide, per_class_cap, min_grades=None, rng):
    """Indices into the arrays for the 'normal' memory bank.

    Non-broken crops NOT from `exclude_slide`, optionally restricted to `grades in min_grades`,
    capped at `per_class_cap` per coarse class. Deterministic given `rng`.
    """
    slides = np.asarray(slides)
    coarse_idx = np.asarray(coarse_idx)
    broken = np.asarray(broken, dtype=bool)
    grades = np.asarray(grades)

    keep = (~broken) & (slides != exclude_slide)
    if min_grades is not None:
        keep &= np.isin(grades, list(min_grades))
    cand = np.flatnonzero(keep)

    out = []
    for c in np.unique(coarse_idx[cand]):
        members = cand[coarse_idx[cand] == c]
        if len(members) > per_class_cap:
            members = rng.choice(members, size=per_class_cap, replace=False)
        out.append(members)
    return np.sort(np.concatenate(out)) if out else np.empty(0, dtype=int)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/qc/test_anomaly.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/qc/anomaly.py tests/qc/test_anomaly.py
git commit -q -m "feat(qc): select_bank_indices — stratified non-broken memory-bank selection"
```

---

## Task 2: `anomaly.py` — `knn_anomaly_score` + `coreset_subsample`

**Files:** Modify `src/dapidl/qc/anomaly.py`; Modify `tests/qc/test_anomaly.py`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/qc/test_anomaly.py
from dapidl.qc.anomaly import knn_anomaly_score, coreset_subsample


def test_knn_anomaly_score_planted_outlier_scores_highest():
    rng = np.random.default_rng(0)
    bank = rng.normal(0, 0.01, size=(50, 8)) + np.array([1.0] + [0] * 7)
    query = np.vstack([bank[:3], np.array([[-1.0] + [0] * 7])])  # last row is far
    s = knn_anomaly_score(query, bank, k=5)
    assert s.shape == (4,)
    assert s[-1] == s.max()                 # planted outlier is most anomalous
    assert s[0] < s[-1]                      # an in-bank-like point is less anomalous


def test_knn_anomaly_score_guards_k_gt_bank():
    bank = np.eye(3)
    s = knn_anomaly_score(bank, bank, k=10)  # k > bank size
    assert np.allclose(s, 0.0, atol=1e-6)    # each query == a bank member -> 0


def test_coreset_subsample_is_deterministic():
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    emb = np.random.default_rng(1).normal(size=(20, 4))
    a = coreset_subsample(emb, frac=0.5, rng=rng1)
    b = coreset_subsample(emb, frac=0.5, rng=rng2)
    assert np.array_equal(a, b)
    assert len(a) == 10
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/qc/test_anomaly.py -k "knn or coreset" -v`
Expected: FAIL (functions not defined).

- [ ] **Step 3: Implement**

```python
# append to src/dapidl/qc/anomaly.py
def _l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def knn_anomaly_score(query, bank, k):
    """Mean cosine distance of each query row to its k nearest bank rows. Higher = more anomalous."""
    q = _l2norm(np.asarray(query, dtype=np.float64))
    b = _l2norm(np.asarray(bank, dtype=np.float64))
    k = min(k, b.shape[0])
    sims = q @ b.T                      # cosine similarity [Q, B]
    # top-k similarities -> distances = 1 - sim
    part = np.partition(sims, -k, axis=1)[:, -k:]
    return (1.0 - part).mean(axis=1)


def coreset_subsample(emb, frac, rng):
    """Greedy k-center coreset subsample of rows of `emb`. frac>=1 returns all indices."""
    n = emb.shape[0]
    m = n if frac >= 1.0 else max(1, int(round(n * frac)))
    if m >= n:
        return np.arange(n)
    x = _l2norm(np.asarray(emb, dtype=np.float64))
    start = int(rng.integers(n))
    chosen = [start]
    mind = 1.0 - (x @ x[start])
    for _ in range(m - 1):
        nxt = int(np.argmax(mind))
        chosen.append(nxt)
        mind = np.minimum(mind, 1.0 - (x @ x[nxt]))
    return np.sort(np.array(chosen))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/qc/test_anomaly.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/qc/anomaly.py tests/qc/test_anomaly.py
git commit -q -m "feat(qc): knn_anomaly_score (cosine kNN) + greedy coreset subsample"
```

---

## Task 3: `anomaly.py` — `score_all_slides_loso` (LOSO orchestration, pure)

**Files:** Modify `src/dapidl/qc/anomaly.py`; Modify `tests/qc/test_anomaly.py`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/qc/test_anomaly.py
from dapidl.qc.anomaly import score_all_slides_loso


def test_score_all_slides_loso_masks_same_slide_and_scores_all_rows():
    rng = np.random.default_rng(0)
    # two slides, class 0 tight cluster; one planted outlier in slide A
    embA = rng.normal(0, 0.01, size=(10, 6))
    embB = rng.normal(0, 0.01, size=(10, 6))
    emb = np.vstack([embA, embB]).astype(np.float32)
    emb[3] += 5.0                                   # planted outlier in slide A
    rows = np.arange(20)
    slides = np.array(["A"] * 10 + ["B"] * 10)
    coarse = np.zeros(20, dtype=int)
    broken = np.zeros(20, dtype=bool)
    grades = np.array(["Good"] * 20)

    score = score_all_slides_loso(emb, rows, slides, coarse, broken, grades,
                                  k=3, per_class_cap=100, rng=rng)
    assert score.shape == (20,)
    assert np.isfinite(score).all()
    assert int(np.argmax(score)) == 3               # planted outlier most anomalous
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/qc/test_anomaly.py -k loso -v` → FAIL.

- [ ] **Step 3: Implement**

```python
# append to src/dapidl/qc/anomaly.py
def score_all_slides_loso(emb, rows, slides, coarse_idx, broken, grades, *,
                          k, per_class_cap, rng, coreset_frac=1.0):
    """For each slide, build the bank from all OTHER slides and score this slide's crops."""
    slides = np.asarray(slides)
    out = np.full(len(rows), np.nan, dtype=np.float64)
    for s in np.unique(slides):
        bank_idx = select_bank_indices(rows, slides, coarse_idx, broken, grades,
                                        exclude_slide=s, per_class_cap=per_class_cap, rng=rng)
        if len(bank_idx) == 0:
            continue
        bank_emb = emb[bank_idx]
        if coreset_frac < 1.0:
            bank_emb = bank_emb[coreset_subsample(bank_emb, coreset_frac, rng)]
        q = np.flatnonzero(slides == s)
        out[q] = knn_anomaly_score(emb[q], bank_emb, k=k)
    return out
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/qc/test_anomaly.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/qc/anomaly.py tests/qc/test_anomaly.py
git commit -q -m "feat(qc): score_all_slides_loso — leave-one-slide-out anomaly scoring"
```

---

## Task 4: `anomaly_eval.py` — AUROC, fairness, embedder selection (pure)

**Files:** Create `src/dapidl/qc/anomaly_eval.py`; Test `tests/qc/test_anomaly_eval.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/qc/test_anomaly_eval.py
import numpy as np
from dapidl.qc.anomaly_eval import auroc, per_class_anomaly, fairness_pass, select_embedder


def test_auroc_perfect_and_chance():
    broken = np.array([0, 0, 1, 1])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), broken) == 1.0
    assert abs(auroc(np.array([0.5, 0.5, 0.5, 0.5]), broken) - 0.5) < 1e-9


def test_fairness_fails_when_endothelial_is_most_anomalous():
    coarse = np.array([0, 0, 1, 1, 2, 2])   # 0 = Endothelial
    broken = np.zeros(6, dtype=bool)
    score = np.array([0.9, 0.95, 0.1, 0.1, 0.2, 0.2])  # class 0 highest
    table = per_class_anomaly(score, coarse, broken, endo_idx=0)
    assert fairness_pass(table, endo_idx=0) is False


def test_select_embedder_picks_passing_higher_auroc():
    results = {
        "dinov2_vitb14": {"auroc": 0.71, "fairness_pass": True},
        "nuspire": {"auroc": 0.80, "fairness_pass": False},   # disqualified
    }
    assert select_embedder(results) == "dinov2_vitb14"
    assert select_embedder({"a": {"auroc": 0.6, "fairness_pass": False}}) is None
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/qc/test_anomaly_eval.py -v` → FAIL.

- [ ] **Step 3: Implement**

```python
# src/dapidl/qc/anomaly_eval.py
"""Validation + embedder selection for the QC anomaly score (pure)."""
from __future__ import annotations

import numpy as np


def auroc(scores, labels):
    """ROC AUC of `scores` predicting binary `labels` (1=positive). Rank-based, ties averaged."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def per_class_anomaly(score, coarse_idx, broken, endo_idx=0):
    """Mean+median anomaly per coarse class over NON-broken crops. Returns {cls: (mean, median)}."""
    score = np.asarray(score, dtype=np.float64)
    coarse_idx = np.asarray(coarse_idx)
    nb = ~np.asarray(broken, dtype=bool) & np.isfinite(score)
    out = {}
    for c in np.unique(coarse_idx[nb]):
        v = score[nb & (coarse_idx == c)]
        out[int(c)] = (float(np.mean(v)), float(np.median(v)))
    return out


def fairness_pass(table, endo_idx=0):
    """HARD gate: the rare Endothelial class must NOT have the highest mean anomaly."""
    if endo_idx not in table:
        return True
    means = {c: m for c, (m, _) in table.items()}
    return means[endo_idx] < max(means.values())


def select_embedder(results):
    """Among embedders that pass fairness, return the one with the highest AUROC, else None."""
    passing = {m: r for m, r in results.items() if r.get("fairness_pass")}
    if not passing:
        return None
    return max(passing, key=lambda m: passing[m]["auroc"])
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/qc/test_anomaly_eval.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/qc/anomaly_eval.py tests/qc/test_anomaly_eval.py
git commit -q -m "feat(qc): anomaly_eval — AUROC, hard per-class fairness gate, embedder selection"
```

---

## Task 5: `embeddings.py` — per-model preprocessing (pure, TDD)

**Files:** Create `src/dapidl/qc/embeddings.py`; Test `tests/qc/test_embeddings_preprocess.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/qc/test_embeddings_preprocess.py
import numpy as np
from dapidl.qc.embeddings import preprocess_dinov2, preprocess_nuspire


def test_preprocess_dinov2_shape_range_and_stretch():
    patch = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64))
    out = preprocess_dinov2(patch, size=224)
    assert out.shape == (3, 224, 224)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_preprocess_dinov2_constant_patch_no_nan():
    patch = np.full((64, 64), 7, dtype=np.uint16)
    out = preprocess_dinov2(patch)
    assert np.isfinite(out).all()


def test_preprocess_nuspire_shape():
    patch = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    out = preprocess_nuspire(patch, size=112)
    assert out.shape == (1, 112, 112)
    assert np.isfinite(out).all()
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/qc/test_embeddings_preprocess.py -v` → FAIL.

- [ ] **Step 3: Implement (preprocess only; model loaders added in Task 6)**

```python
# src/dapidl/qc/embeddings.py
"""Frozen vision-FM embedders for DAPI QC crops (DINOv2 + NuSPIRe), with per-model preprocessing."""
from __future__ import annotations

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


def preprocess_nuspire(patch, size=112):
    g = _resize(_stretch01(patch), size)
    # NuSPIRe trained on single-channel DAPI; mean/std ~0.5 placeholder normalization,
    # REPLACE with the loader's documented stats when wiring the real model in Task 6.
    return (((g - 0.5) / 0.5)[None, :, :]).astype(np.float32)
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/qc/test_embeddings_preprocess.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/qc/embeddings.py tests/qc/test_embeddings_preprocess.py
git commit -q -m "feat(qc): DAPI crop preprocessing for DINOv2 + NuSPIRe embedders"
```

---

## Task 6: `embeddings.py` — model loaders + `compute_embeddings` [CONTROLLER RUN for GPU]

**Files:** Modify `src/dapidl/qc/embeddings.py`.

- [ ] **Step 1: Locate the in-repo NuSPIRe loader**

```bash
grep -rniE "nuspire|nu_spire|NuSPIRe" /mnt/work/git/dapidl/src /mnt/work/git/dapidl/scripts | head
```
Record the loader entrypoint + its expected input size/normalization; update `preprocess_nuspire` stats accordingly. If no usable in-repo loader, fall back to the published weights (TongjiZhanglab/NuSPIRe, MIT) and note it.

- [ ] **Step 2: Implement loaders + `compute_embeddings`**

```python
# append to src/dapidl/qc/embeddings.py — torch is imported lazily so the pure tests stay torch-free
import struct
from pathlib import Path


def _load_dinov2(device):
    import torch
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    return m.eval().to(device)


def _load_nuspire(device):
    raise NotImplementedError("Wire to the in-repo NuSPIRe loader found in Task 6 Step 1.")


EMBEDDERS = {
    "dinov2_vitb14": {"load": _load_dinov2, "preprocess": preprocess_dinov2, "size": 224, "dim": 768},
    "nuspire": {"load": _load_nuspire, "preprocess": preprocess_nuspire, "size": 112, "dim": None},
}


def _read_patch(txn, idx, ps=64):
    v = txn.get(struct.pack(">Q", int(idx)))
    if v is None:
        return np.zeros((ps, ps), dtype=np.uint16)
    return np.frombuffer(v[8:], dtype=np.uint16).reshape(ps, ps)


def compute_embeddings(lmdb_dir, model="dinov2_vitb14", device="cuda", batch_size=256,
                       recompute=False):
    import lmdb, polars as pl, torch
    spec = EMBEDDERS[model]
    qc = Path(lmdb_dir) / "qc"
    emb_path, rows_path = qc / f"embeddings_{model}.npy", qc / f"embeddings_{model}_rows.npy"
    reg = pl.read_parquet(Path(lmdb_dir) / "qc" / "seg_scores.parquet").select("row_idx")
    rows = reg["row_idx"].to_numpy()
    if emb_path.exists() and rows_path.exists() and not recompute:
        cached = np.load(rows_path)
        if len(cached) == len(rows):
            return cached, np.load(emb_path)
    net, pre = spec["load"](device), spec["preprocess"]
    env = lmdb.open(str(Path(lmdb_dir) / "patches.lmdb"), readonly=True, lock=False, readahead=False)
    embs = []
    with env.begin() as txn, torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            x = np.stack([pre(_read_patch(txn, r)) for r in chunk])
            t = torch.from_numpy(x).float().to(device)
            out = net(t)                              # [B, dim] (CLS / pooled)
            embs.append(out.cpu().numpy().astype(np.float16))
    env.close()
    emb = np.concatenate(embs)
    qc.mkdir(exist_ok=True)
    np.save(emb_path, emb); np.save(rows_path, rows)
    return rows, emb
```

- [ ] **Step 3: Smoke-run on the real LMDB (GPU)** — check `nvidia-smi` first (CLAUDE.md, 2-4 GB buffer):

```bash
cd /mnt/work/git/dapidl-p64qc
nvidia-smi --query-gpu=memory.free --format=csv
uv run python -c "from dapidl.qc.embeddings import compute_embeddings as c; import numpy as np; r,e=c('/mnt/work/datasets/derived/breast-pilot-6source-dapi-p64-nuc-v1','dinov2_vitb14'); print(r.shape, e.shape, np.isfinite(e.astype('f4')).all())"
```
Expected: `(17990,) (17990, 768) True`. Repeat for `nuspire` once its loader is wired.

- [ ] **Step 4: Commit**

```bash
git add src/dapidl/qc/embeddings.py
git commit -q -m "feat(qc): compute_embeddings — frozen DINOv2/NuSPIRe forward + cache"
```

---

## Task 7: `qc_anomaly_score.py` driver — score + `--eval` + montage [CONTROLLER RUN]

**Files:** Create `scripts/qc_anomaly_score.py`.

- [ ] **Step 1: Implement the driver** (uses the pure modules + `compute_embeddings`):
  - Load `seg_scores.parquet` (rows, slide, coarse_idx, broken, grade).
  - For each `--models`: `compute_embeddings(...)`, then `score_all_slides_loso(...)` → `anomaly_score_<model>`; add `anomaly_pct_<model>` (global rank).
  - `--eval`: per model compute pooled `auroc(score, broken)` and `per_class_anomaly`/`fairness_pass` (endo_idx from `class_mapping.json` Endothelial=0). Build `results = {model: {auroc, fairness_pass}}`; `winner = select_embedder(results)`. If `winner is None`: print FAIL + do not write canonical columns; else copy `anomaly_score_<winner>`→`anomaly_score`, `anomaly_pct_<winner>`→`anomaly_pct`.
  - Missed-break montage: top-K rows where `grade in {Good, Weak-passing}` and highest `anomaly_pct`, render via `src/dapidl/qc/montage.py` → `qc/anomaly_missed_breaks.png` + CSV.
  - Write `qc/seg_scores_anom.parquet`. Print the per-class fairness table + AUROC per model + the chosen winner.
  - Args: `--lmdb`, `--models dinov2_vitb14 nuspire`, `--k 20`, `--per-class-cap 1500`, `--coreset-frac 1.0`, `--eval`, `--out`.

- [ ] **Step 2: Run end-to-end** (GPU):

```bash
cd /mnt/work/git/dapidl-p64qc && nvidia-smi --query-gpu=memory.free --format=csv
uv run python scripts/qc_anomaly_score.py --models dinov2_vitb14 nuspire --eval 2>&1 | tail -40
```
Expected: per-model AUROC printed; the fairness table; a declared winner (or a loud FAIL if both fail); `qc/seg_scores_anom.parquet` + `qc/anomaly_missed_breaks.png` written.

- [ ] **Step 3: Inspect the missed-break montage** and confirm the fairness gate behaved (Endothelial not top). Send the montage to the user for a sanity look.

- [ ] **Step 4: Commit**

```bash
git add scripts/qc_anomaly_score.py
git commit -q -m "feat(qc): anomaly-score driver — LOSO scoring, embedder benchmark, missed-break montage"
```

---

## Task 8: Review selection — `qc_review_ls_build.py --select anomaly_disagree` (TDD core)

**Files:** Modify `scripts/qc_review_ls_build.py`; Create `tests/qc/test_review_select.py` + extract a pure `select_review_rows` into `src/dapidl/qc/review_select.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/qc/test_review_select.py
import polars as pl
from dapidl.qc.review_select import select_review_rows


def test_anomaly_disagree_prefers_high_anomaly_good_crops():
    df = pl.DataFrame({
        "row_idx": [1, 2, 3, 4],
        "slide": ["A"] * 4,
        "cell_class": ["Immune"] * 4,
        "grade": ["Good", "Good", "broken", "Weak-passing"],
        "anomaly_pct": [99.0, 5.0, 99.0, 80.0],
    })
    out = select_review_rows(df, mode="anomaly_disagree", n=2)
    picked = out["row_idx"].to_list()
    assert 1 in picked            # high anomaly + Good = top disagreement
    assert 3 not in picked        # already broken (classical agrees) -> not a disagreement
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/qc/test_review_select.py -v` → FAIL.

- [ ] **Step 3: Implement `select_review_rows`**

```python
# src/dapidl/qc/review_select.py
"""Review-set selection strategies for the Label Studio QC loop (pure polars)."""
from __future__ import annotations

import polars as pl

_GOOD = {"Good", "Weak-passing", "Excellent"}


def select_review_rows(df: pl.DataFrame, *, mode: str, n: int, seed: int = 0) -> pl.DataFrame:
    if mode == "anomaly_disagree":
        cand = df.filter(pl.col("grade").is_in(list(_GOOD)))
        return cand.sort("anomaly_pct", descending=True, nulls_last=True).head(n)
    # stratified default: shuffle then head per (slide, cell_class, grade) is handled in the build;
    # here just return a deterministic sample for the flat case.
    return df.sample(n=min(n, df.height), shuffle=True, seed=seed)
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/qc/test_review_select.py -v` → PASS.

- [ ] **Step 5: Wire into `qc_review_ls_build.py`** — read `qc/seg_scores_anom.parquet` if present (else `seg_scores.parquet`); carry `anomaly_score`/`anomaly_pct` into the manifest + LS task `data`; add `--select {stratified, anomaly_disagree}` using `select_review_rows`. Default `stratified` (unchanged for project 5).

- [ ] **Step 6: Commit**

```bash
git add src/dapidl/qc/review_select.py tests/qc/test_review_select.py scripts/qc_review_ls_build.py
git commit -q -m "feat(qc): anomaly-disagreement review selection + carry anomaly score to Label Studio"
```

---

## Task 9: Push the anomaly-ranked comparison project [CONTROLLER RUN]

**Files:** none new (reuse `qc_review_ls_build.py` + `qc_review_ls_push.py`).

- [ ] **Step 1: Build the anomaly-disagreement sample**

```bash
cd /mnt/work/git/dapidl-p64qc && source ~/.zshrc.local
uv run --with pillow python scripts/qc_review_ls_build.py --select anomaly_disagree
```

- [ ] **Step 2: Push as a NEW DAPIDL project** (leaves project 5 untouched). Set the project title to "p64 QC — anomaly-ranked (vs project 5)" in `qc_review_ls_push.py` (or via a `--title` arg), then:

```bash
uv run --with requests python scripts/qc_review_ls_push.py --create
```
Expected: a new project id (e.g. 6) reported; verify task count + that project 5 is unchanged (581 tasks).

- [ ] **Step 3: Report** both project URLs to the user for the side-by-side comparison (stratified project 5 vs anomaly-ranked new project).

---

## Final review

- [ ] Run the full suite: `uv run pytest tests/qc/ -v` (all green).
- [ ] `uv run ruff check src/dapidl/qc/ scripts/qc_anomaly_score.py && uv run ruff format --check src/dapidl/qc/`.
- [ ] Dispatch a final code-review subagent over the whole diff; then use superpowers:finishing-a-development-branch.
