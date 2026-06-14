# QC Embedding-Anomaly Score (Phase 1) — Design

**Author:** Christian Meß <chrism@sonora.ai> · Sonora Technologies GmbH
**Date:** 2026-06-13
**Status:** Approved design — Phase 1 of the QC anomaly + active-learning track

---

## Overview

Add an **unsupervised, embedding-based "broken-crop" anomaly score** to the p64 DAPI QC
pipeline. It **augments** the existing StarDist-grounded classical grade (never replaces it),
is **fair to rare cell classes**, is validated **leave-one-slide-out**, and feeds the Label
Studio review loop. It requires **no human labels**, so it is buildable and testable today.

This is **Phase 1** of "Track 1" from the 2026-06-13 embeddings/RAG research (the
highest-confidence quick win). **Phase 2** — active-learning selection + label propagation to
cut the reviewer's workload — is deferred until enough human labels exist to calibrate it, and
is out of scope here.

### Decisions folded in (2026-06-13)

- **Benchmark two embedders.** Compute anomaly scores with **both** DINOv2 (generic, Apache-2.0)
  and **NuSPIRe** (in-house, DAPI-native, MIT); the validation step picks the winner.
- **New comparison Label Studio project.** Push a **new** project whose review set uses
  anomaly-disagreement selection. **Project 5 (stratified)** is kept untouched as the baseline so
  the two selection strategies can be compared side by side.
- **Hard fairness stop.** If the per-class fairness check fails (Endothelial most-anomalous), that
  embedder is **disqualified** and the run stops loudly rather than producing biased scores.

### Why embeddings here

Frozen-feature anomaly detection (PatchCore / AnomalyDINO) gives **broad** broken-crop coverage:
one distance score catches defocus, debris, multi-nucleus, empty, and segmentation failures that
the hand-crafted axes handle separately. The only nuclear-stain head-to-head in the literature
(Yang et al. 2018, Hoechst focus QC) shows a learned detector edges the classical focus metric
(F 0.89 vs 0.84). The value here is **breadth + a review re-ranker + a "missed-break" catcher**,
not replacing the interpretable classical axes — so we keep both and AND-gate.

---

## Context (current state)

- **Dataset:** `/mnt/work/datasets/derived/breast-pilot-6source-dapi-p64-nuc-v1/` — 17,990
  native-64px DAPI crops across 6 sources (`xenium_rep1_nuc`, `xenium_rep2_nuc`,
  `sthelar_breast_s0/s1/s3/s6`), 4 coarse classes.
- **`qc/seg_scores.parquet`** (31 cols) already carries `row_idx, slide, coarse_idx, broken,
  broken_reason, grade` (Excellent/Good/Weak-passing/broken) + the 4 classical axes.
- **`qc/masks/`** holds bit-packed nucleus/cell masks (not needed for Phase 1).
- **Label Studio** review project 5 (DAPIDL org) is live; built/pushed by
  `scripts/qc_review_ls_build.py` / `qc_review_ls_push.py`.
- **NuSPIRe** in-house weights/loader exist from the backbone head-to-head work (locate and reuse
  the existing loading path rather than re-implementing).
- **GPU** is free (~23 GB).

---

## Architecture

Three units + one driver + a Label Studio integration. Two embedders are run and benchmarked:

```
p64 LMDB crops ──[1] DINOv2 + NuSPIRe (frozen)──> qc/embeddings_<model>.npy  [N, D] f16
                                              │
qc/seg_scores.parquet (grade, broken, coarse_idx) ─┤
                                              ▼
   [2] per-slide stratified NON-BROKEN memory bank (SAME-SLIDE MASKED)   — per model
                                              ▼
   [2] kNN anomaly score per crop ──[3] LOSO driver──> qc/seg_scores_anom.parquet
                                              ▼          (anomaly_score_<model> + pct per model)
   [4] validation: AUROC + HARD per-class fairness gate, per model
        → pick the embedder that PASSES fairness with the higher AUROC → canonical anomaly_score
                                              ▼
   [5] NEW Label Studio project: anomaly-disagreement-ranked review set
        (project 5 stratified baseline kept for side-by-side comparison)
```

---

## Components

### 1. `src/dapidl/qc/embeddings.py` — frozen vision-FM embedders (two)

- Each entry in the `EMBEDDERS` registry bundles its **own** loader, preprocessing, and output
  dim, because the two models differ:
  - `"dinov2_vitb14"` — torch.hub `facebookresearch/dinov2`, **Apache-2.0**, D=768. Preprocess:
    uint16 crop → 1–99% percentile stretch to [0,1] → replicate 1→3 channels → resize to 224
    (multiple of 14) → ImageNet mean/std. CLS token.
  - `"nuspire"` — in-house DAPI-native ViT (MIM), **MIT**. Preprocess: 1→ native 112px, the
    model's own training normalization, single channel. Pooled/CLS embedding. Reuse the existing
    in-repo NuSPIRe loader.
- `preprocess_dinov2(patch, size=224) -> np.ndarray` and `preprocess_nuspire(patch, size=112) ->
  np.ndarray` — **PURE**, unit-tested per model.
- `compute_embeddings(lmdb_dir: Path, model: str, device="cuda", batch_size=256,
  recompute=False) -> tuple[np.ndarray, np.ndarray]` — iterate the LMDB, batch-forward the frozen
  model, return `(rows int64[N], emb float16[N,D])`; writes `qc/embeddings_<model>.npy` +
  `qc/embeddings_<model>_rows.npy`. Skips work if the cache matches the row count unless
  `recompute`.

### 2. `src/dapidl/qc/anomaly.py` — memory-bank kNN scoring (PURE)

- `select_bank_indices(rows, slides, coarse_idx, broken, grades, *, exclude_slide: str,
  per_class_cap: int, min_grades=None, rng) -> np.ndarray` — indices of **classically non-broken**
  crops (`broken == False`; optionally further restricted to `grades ∈ min_grades`) that are NOT
  from `exclude_slide`, capped at `per_class_cap` per coarse class (so rare Endothelial is
  represented but common classes do not swamp the bank). The bank defines "normal" = anything the
  classical scorer accepted, so an anomaly is a crop unlike any accepted crop (a candidate
  missed-break). Deterministic given `rng`.
- `coreset_subsample(emb, frac, rng) -> np.ndarray` — optional greedy k-center (PatchCore-style)
  subsample; `frac >= 1.0` returns all. Deterministic.
- `knn_anomaly_score(query, bank, k) -> np.ndarray` — L2-normalize both, return the mean cosine
  distance of each query row to its `k` nearest bank rows. Higher = more anomalous. Guards
  `k > len(bank)`.

### 3. `scripts/qc_anomaly_score.py` — LOSO driver (per model)

- For each model in `--models dinov2_vitb14 nuspire`, for each slide `s`:
  `bank = select_bank_indices(exclude_slide=s, ...)` over all OTHER slides;
  `score = knn_anomaly_score(query=emb[slide==s], bank=emb[bank])`. Assemble per `row_idx`.
- Writes `qc/seg_scores_anom.parquet` = `seg_scores` + `anomaly_score_<model>` +
  `anomaly_pct_<model>` for each model, plus a canonical `anomaly_score` / `anomaly_pct` copied
  from the **winning** embedder (set by the validation step).
- Args: `--lmdb`, `--models dinov2_vitb14 nuspire`, `--k 20`, `--per-class-cap 1500`,
  `--coreset-frac 1.0`, `--out`, `--recompute`.

### 4. Validation + embedder selection — `scripts/qc_anomaly_score.py --eval`

For **each** embedder, on held-out slides:
- **Sanity AUROC:** AUROC(`anomaly_score_<model>` vs classical `broken`) — expect > 0.5
  (target > 0.7). Reported per held-out slide + pooled.
- **Fairness guardrail (HARD):** mean + median anomaly per coarse class over non-broken crops.
  If **Endothelial is the most-anomalous class**, that embedder **FAILS** — it is disqualified and
  the run aborts loudly for it (the stratified bank did not protect the rare class).
- **Embedder selection:** among embedders that **pass** fairness, the one with the higher pooled
  held-out AUROC becomes the canonical `anomaly_score`. **If both fail fairness → no canonical
  score is written; fall back to classical-only and report why.**
- **Missed-break surfacing:** for the winning embedder, top-K crops whose classical
  `grade ∈ {Good, Weak-passing}` (not broken) but `anomaly_pct` is highest → render a montage
  (reuse `src/dapidl/qc/montage.py`) to `qc/anomaly_missed_breaks.png` + a CSV, for a human look
  at whether it surfaces real breaks the classical axes missed.

### 5. Label Studio integration — extend `scripts/qc_review_ls_build.py` + push a NEW project

- `qc_review_ls_build.py` reads `qc/seg_scores_anom.parquet` when present (else `seg_scores.parquet`),
  carries the canonical `anomaly_score` + `anomaly_pct` into the manifest and LS task `data`
  (filterable/sortable, shown in the task header), and adds `--select {stratified, anomaly_disagree}`
  (default `stratified`). `anomaly_disagree` prioritizes crops where `anomaly_pct` is high AND
  classical `grade` is good/weak (the most informative disagreements).
- `qc_review_ls_push.py --create` pushes the `anomaly_disagree` set as a **new** DAPIDL project
  (e.g. "p64 QC — anomaly-ranked"). **Project 5 (stratified) is left untouched** so the reviewer
  can compare the two selection strategies directly.

---

## Error handling / constraints

- **GPU:** check free memory before each embedding pass; batch 256 (DINOv2-B < 2 GB, NuSPIRe
  smaller). Embedding arrays are ~27 MB (DINOv2) — trivial RAM.
- **Determinism:** fixed `rng` seed (0) for bank selection + coreset; cached embeddings make the
  whole pipeline reproducible.
- **License:** DINOv2 Apache-2.0, NuSPIRe MIT — both commercial-OK; no gated/NC weights.
- **No human labels** are consumed anywhere in Phase 1.

---

## Testing (TDD)

- **Unit `embeddings.py`:** `preprocess_dinov2` returns `[3,224,224]` float32 in range and maps
  p1→0/p99→1; `preprocess_nuspire` returns the NuSPIRe-expected `[1,112,112]` shape; constant
  patch does not divide-by-zero; 1→3 channel replication (DINOv2).
- **Unit `anomaly.py`:** `select_bank_indices` respects `per_class_cap`, excludes `exclude_slide`,
  excludes broken, includes all present classes; `knn_anomaly_score` matches hand-computed
  distances on a toy set, returns 0 for query==bank-member, guards `k>len(bank)`; cosine on
  normalized vectors.
- **Integration:** a small slice (1 slide as query, 1 as bank, a few hundred crops) runs
  end-to-end for one model → `seg_scores_anom.parquet` gains finite `anomaly_score_*`.
- **Validation gates** (run on real data, reported): per-embedder AUROC > 0.5; HARD fairness stop;
  embedder selection picks the passing higher-AUROC model.

---

## Scope

**In:** embeddings for **both** DINOv2 and NuSPIRe (benchmarked), the memory-bank anomaly score,
LOSO scoring, validation gates incl. **embedder selection + hard fairness stop**, and a **new**
Label Studio comparison project using anomaly-disagreement selection (project 5 kept as the
stratified baseline).

**Out (Phase 2):** active-learning selection (ProbCover/TypiClust), label propagation
(LabelSpreading/Poisson), threshold calibration on the reviewer's broken/good labels, and any
replacement of the classical grade.

---

## Risks

1. **Rare-class false positives** — a valid-but-rare Endothelial nucleus is an outlier → flagged
   "broken". Mitigated by the **class-stratified bank** + the **HARD per-class fairness gate**
   that **disqualifies an embedder** if Endothelial is the most-anomalous class. If both embedders
   fail, fall back to classical-only.
2. **Slide/batch leakage** — the score matches a crop's own-slide signature instead of biology.
   Mitigated by **same-slide masking** + **LOSO scoring** + per-slide reporting.
3. **Neither embedder separates breaks from valid morphology on single-channel DAPI** — surfaced
   early by the per-embedder AUROC + missed-break montage; benchmarking two models hedges this, and
   the fallback is classical-only. Sunk cost is low (frozen, no training).
4. **Modest payoff** — research predicts breadth, not a large accuracy jump; acceptable because it
   is a cheap re-ranker + missed-break catcher, gated by validation before adoption.
