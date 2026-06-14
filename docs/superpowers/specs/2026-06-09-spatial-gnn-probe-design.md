---
title: Spatial-GNN Probe ("Learned BANKSY") — Design
date: 2026-06-09
branch: feat/spatial-gnn-probe
status: design (awaiting review)
---

# Spatial-GNN Probe ("Learned BANKSY") — Design

**The bet, in one sentence.** Determine whether injecting a **spatial-neighbour graph** into DAPI features beats the production EfficientNet-V2-S (test macro-F1 **0.619** on held-out Xenium rep2) — specifically whether it lifts the **context-defined** classes **Endothelial (0.31)** and **Stromal (0.65)** — run **staged cheap→clean** so the answer is trustworthy whichever way it falls.

This is a **scientific probe**, not a production model. Success = a clean, defensible A/B number against the existing 0.619 baseline on the identical rep2 test slide.

---

## Why this, why now (evidence base)

Three independent internal results say the predictive signal is **spatial context, not the individual nucleus**: (1) the patch-size sweep (bigger patch → higher F1); (2) the nucleus-only backbone NuSPIRe losing to the context-aware EfficientNet (McNemar p≈5e-254); (3) the subnuclear-triangulation integrated-gradients ablation (attribution is concentrated inside the nucleus only for immune cells; below area-proportional for epithelial/endothelial). The worst class, **endothelial**, is recoverable only with spatial-neighbour information — in the annotation benchmark it scores ~0 without BANKSY, and BANKSY-style methods reach ~0.80.

Literature scan (2026-06-09, OpenAlex + fetched sources): the two halves are independently proven in top venues but **the combination is open**:
- Spatial-graph cell annotation works — **STELLAR** (Brbić et al., *Nat Methods* 2022, doi:10.1038/s41592-022-01651-8); **SPACE-GM** — but on **protein/expression** node features.
- Cell-graph GNNs over nuclear morphology work — a mature histopathology sub-field (*Graph-Transformer for WSI*, IEEE TMI 2022; cell-graph survival GNNs; *GNNs in histopathology* review, Med Image Anal 2025) — but on **slide-level** tasks (grading/survival), on H&E.
- **Nobody feeds DAPI-morphology embeddings as graph nodes for per-cell type inference.** That is this probe's niche, and the BANKSY/STELLAR precedent says the spatial graph is exactly what rescues the context-defined classes.

---

## Locked decisions

- **D1 — Goal = probe.** A/B vs EfficientNet 0.619 on rep2. Not a production model.
- **D2 — Staged.** Stage 1 = cheap BANKSY-augmentation (no GNN). Gate. Stage 2 = clean learned-BANKSY GNN. Stage 1 alone risks a false-negative (context-rich nodes); Stage 2 alone is more upfront work; staged is the cheapest path to a trustworthy answer.
- **D3 — Reuse `breast-6source-dapi-p128`.** Train = rep1 + STHELAR s0/s1/s3/s6; **test = rep2** (identical split to the baseline → true A/B).
- **D4 — Within-slide k-NN graph** on centroids, **k = 8** (sweep 6–10). **Pixel** coordinates (k-NN is scale-invariant; no µm conversion needed unless we later switch to radius graphs).
- **D5 — Node features:** Stage 1 = frozen production-EffNet embedding, **PCA-reduced to 128-d** (RAM, see §Stage 1); Stage 2 = a small CNN on a **tight 40px nucleus crop** (context-poor, so the graph is the sole context source).
- **D6 — Integrity controls are part of the deliverable:** McNemar vs EffNet preds, batch-identity probe, the 0.497 handcrafted floor as the morphology baseline, bootstrap CIs.

---

## Data substrate (verified)

`/mnt/work/datasets/derived/breast-6source-dapi-p128/` — 2,277,877 patches, 128px, uint16, per-slide adaptive normalization, coarse 4-class. Row-aligned arrays present: `labels.npy`, `sources.npy`, `raw_labels.npy`, `confidence.npy`. **No `patch_registry.parquet`** (this LMDB predates the registry refactor) → Phase 0 reconstructs it.

| Slide | Role | Patches |
|---|---|---|
| xenium_rep1 | train | 157,231 |
| sthelar_breast_s0 | train | 542,818 |
| sthelar_breast_s1 | train | 757,374 |
| sthelar_breast_s3 | train | 345,805 |
| sthelar_breast_s6 | train | 360,923 |
| **xenium_rep2** | **test** | **113,726** |

Class counts (authoritative `labels.npy` bincount; `metadata.json` is stale): **Endothelial 48,086 / Epithelial 993,342 / Immune 521,543 / Stromal 381,544 / unlabeled (-1) 333,362**. The 333,362 `-1` cells are real nuclei whose fine annotation was unmapped — they are **graph nodes (spatial context) but masked out of train/val/test/eval** (the probe classifies the 4 labelled classes only).

Baseline to beat: EfficientNet-V2-S, **macro-F1 0.619** on rep2 (per-class Endo 0.31 / Epi 0.86 / Imm 0.66 / Str 0.65). Checkpoint: `pipeline_output/h2h_2026_05_30/efficientnetv2_rw_s/best_model.pt`.

LMDB read format (Format B): key `struct.pack(">Q", row_idx)`; value = `int64 label (8 bytes)` + `uint16 patch (128×128)`. So `value[8:]` → `np.frombuffer(..., uint16).reshape(128,128)`.

---

## Phase 0 — Spatial registry (replay + verify; cheap, no GPU)

`src/dapidl/graph/registry.py :: build_spatial_registry(lmdb_dir) -> pl.DataFrame`

**Data reality correction (discovered 2026-06-09 via systematic debugging).** This LMDB was **not** built by the current `scripts/breast_dapi_lmdb.py`. Its STHELAR labels came from a *finer* annotation (preserved in `raw_labels.npy`: `CAF`, `Mammary_luminal_cell`, `Endothelial_Pericyte_Smooth_muscle`, …), and the STHELAR source annotations have since **drifted** — so replaying the current script recomputes ~509k STHELAR coarse labels *wrongly* (the stable `label1` column disagrees with the build's fine column, e.g. `Endothelial_Pericyte_Smooth_muscle`→Endothelial vs `Perivascular`→Stromal). Two layers of drift (script + source data) mean **no label-based replay can reproduce the stored labels.** However, the per-source cell **set and order are stable** (the `label1 != "less10"` filter + deterministic `nucleus_df` order), so replaying the iteration still recovers the **correct `(cell_id, centroid)` for each LMDB row**.

So the registry: (1) replays rep1 → rep2 → `sorted(STHELAR glob)` to get per-row `(cell_id, x_px, y_px)`; (2) attaches the coarse label from the **authoritative `labels.npy`** (what the 0.619 baseline trained on, incl. `-1`); (3) **proves alignment by content, not labels** — `verify_counts` asserts exact per-source counts/order, and `verify_content` crops the source DAPI at each replayed centroid and asserts the **mean Pearson correlation against the stored LMDB patch > 0.9** per source (a sampled spot-check; drift-proof because it compares pixels, not annotations). On real data all 6 sources clear 0.9.

Output: `pipeline_output/spatial_gnn_probe_2026_06/spatial_registry.parquet` (`row_idx, source, cell_id, x_px, y_px, coarse_idx`). Pixel coords suffice (k-NN scale-invariant).

---

## Stage 1 — BANKSY-augmentation probe (no GNN; ~1–2 h)

1. **Embed** (`embed.py`). Load `DapiClassifier(num_classes=4, backbone="efficientnetv2_rw_s")` from the h2h checkpoint, `eval()`. Stream the LMDB in row order; eval transform = `patch/65535` then `(x − 0.485)/0.229`; extract the **penultimate pooled features** (EffNetV2-S `num_features = 1792`) via `backbone.forward_features` + global pool (or a forward hook). Write `embeddings_f16.npy` (2.28M × 1792, float16 ≈ **8.1 GB** on disk — streamed, never all-resident). GPU ~30–40 min (check `nvidia-smi` first; batch to fit a 2–4 GB buffer).
2. **PCA-reduce** to **128-d** (`embed.py`). Fit PCA on a 200k random-row sample (float32), transform all rows → `embeddings_pca128.npy` (2.28M × 128 float32 ≈ **1.16 GB**). *This reduction is both standard (BANKSY operates on PCA features) and the RAM fix — see §RAM.*
3. **Graph** (`knn_graph.py`). Per slide, build k-NN (k=8) on `(x_px, y_px)` via `scipy.spatial.cKDTree`; emit `edge_index` with **within-slide isolation** (no cross-slide edges).
4. **BANKSY-augment** (`banksy_features.py`). For each cell with PCA feature `x` and neighbours `{x_j}`: `neighbour_mean = mean_j x_j`; azimuthal gradient `AGF = | (1/n) Σ_j e^{iθ_j}(x_j − x) |` (per-dim magnitude, θ_j = angle of neighbour j about the cell). Augmented vector = `concat[ √(1−λ)·x, √(λ/2)·neighbour_mean, √(λ/2)·AGF ]` (3 × 128 = **384-d**). Sweep **λ ∈ {0.2, 0.5, 0.8}**.
5. **Classify.** Class-weighted **LightGBM** (multiclass): **train on rep1 + STHELAR s0/s1/s6, validate on STHELAR s3** (early-stopping), **test on rep2**. Compare to: (a) the same LightGBM on **un-augmented** PCA features (λ=0 ablation — isolates the graph's contribution), (b) the 0.497 handcrafted floor, (c) the 0.619 CNN.

RAM for the LightGBM matrix: 2.16M × 384 × 4 B ≈ **3.3 GB**. Fine on a 62 GB host.

---

## Decision gate

Proceed to Stage 2 **unless** Stage 1 shows essentially no spatial signal — defined as: Δmacro-F1 ≤ 0 vs the λ=0 ablation **and** ΔF1 < 0.01 on **both** Endothelial and Stromal. In that case, pause and reconsider (the graph may not help, or node features are too context-rich). Otherwise Stage 2 runs as the clean confirmation. If Stage 1 already beats 0.619, we likely have the headline answer; Stage 2 still confirms it with context-poor nodes.

---

## Stage 2 — Clean learned-BANKSY GNN (gated; ~few h GPU)

- **Node** (`gnn.py`). Small CNN on the **centre 40×40** crop of each 128px patch (nucleus-local → context-poor): 3 conv blocks (stride-2, BN, ReLU) → global-avg-pool → 128-d node embedding.
- **Message passing.** 2–3 layer **GraphSAGE (mean aggregation)** over the within-slide graph. Hand-rolled mean-aggregation (scatter-mean) to avoid a new heavyweight dependency; PyTorch-Geometric is the fallback if we need GAT/attention. Big slides (s1 = 757k) use neighbour-sampled mini-batches.
- **Head.** Linear → 4 classes. Class-weighted cross-entropy with `max_weight_ratio=10` (project standard, avoids rare-class mode collapse).
- **Training.** Adam, warmup-cosine schedule (the NuSPIRe scheduler lesson: avoids the oscillation that cost ~0.03 F1), early stopping patience 5, grouped by slide (no cell leakage). Seeds fixed; 3-seed repeat for the headline number.
- **Eval.** Identical harness to Stage 1 (rep2; macro-F1 + per-class + McNemar + bootstrap CIs).

---

## Integrity controls (what makes the result survive a viva)

- **Grouped split** by slide — inherent (train slides vs the rep2 test slide); no cell or patch leakage.
- **McNemar** test on rep2 predictions vs the EfficientNet 0.619 predictions (regenerate rep2 preds from the checkpoint if `preds.parquet` is absent — cheap inference).
- **Batch-identity probe** — train a small classifier to predict *slide id* from the node embeddings; high accuracy ⇒ the "spatial biology" is partly batch confound. Report it.
- **Morphology baseline** — the existing **0.497** handcrafted-nucleus LightGBM floor; the graph must beat it to justify itself.
- **λ=0 ablation** (Stage 1) — same pipeline minus the neighbour terms — isolates the graph's marginal contribution from the node features alone.
- **Bootstrap 95% CIs** on macro-F1 (resample rep2 cells).

---

## File structure

```
src/dapidl/graph/
  __init__.py
  registry.py        # build_spatial_registry() — replay + verify vs sources/labels
  knn_graph.py       # build_within_slide_knn(coords, slide_ids, k) -> edge_index
  banksy_features.py # banksy_augment(feats, edge_index, lambda_) -> (N, 3*D)
  embed.py           # extract_embeddings(lmdb, ckpt) ; pca_reduce(emb, d=128)
  gnn.py             # NucleusNodeCNN ; SageCellTyper (Stage 2)
scripts/
  spatial_gnn_probe.py   # driver: Phase 0 -> Stage 1 -> gate -> Stage 2 -> readout
tests/
  test_graph_registry.py     test_knn_graph.py
  test_banksy_features.py     test_graph_embed.py
  test_gnn.py
```

Each module has one responsibility and a small, testable surface. `registry.py` and `knn_graph.py` are pure/CPU; `embed.py` and `gnn.py` are the only GPU-touching units.

---

## Testing (TDD — write the failing test first)

- **registry**: a 3-row synthetic LMDB + fake readers → replayed `(source,label)` matches the stored arrays; a deliberately desynced fixture → raises. Centroid join correctness.
- **knn_graph**: two slides of known points → every edge stays within its slide; each node has exactly k neighbours (or n−1 if the slide is small); symmetry/self-loop policy explicit.
- **banksy_features**: a hand-computed 3-node example → neighbour-mean and AGF match by hand; λ=0 ⇒ augmented == `concat[x, 0, 0]` (neighbour terms vanish); column count == 3·D; AGF rotation-equivariant magnitude is finite for a degenerate (single-neighbour) cell.
- **embed**: PCA round-trip preserves variance ordering; output shape (N,128); deterministic given a fixed PCA fit.
- **gnn**: NucleusNodeCNN forward shape (B,128) from (B,1,40,40); SageCellTyper forward shape (N,4) on a tiny graph; scatter-mean aggregation equals a hand-computed neighbour mean.

---

## Outputs

`pipeline_output/spatial_gnn_probe_2026_06/`
- `spatial_registry.parquet`
- `stage1_metrics.json` (per-λ macro-F1 + per-class, λ=0 ablation, vs 0.497 floor / 0.619 CNN)
- `stage2_metrics.json` (3-seed macro-F1 + per-class + CIs) — if gated in
- `mcnemar.json`, `batch_probe.json`
- `readout.md` — one-page verdict: did the spatial graph beat 0.619, and did it specifically lift Endo/Str?

---

## RAM / compute budget (estimate before running)

- EffNet embeddings: 2.28M × 1792 float16 ≈ **8.1 GB** (streamed to disk, not resident). GPU inference ~30–40 min — check `nvidia-smi` for a 2–4 GB free buffer first.
- PCA-128 features: 2.28M × 128 float32 ≈ **1.16 GB** (resident, fine).
- Stage 1 LightGBM matrix: 2.16M × 384 × 4 B ≈ **3.3 GB** (fine on 62 GB host).
- k-NN: cKDTree on ≤757k 2D points per slide — seconds, <1 GB.
- Stage 2: small CNN + GNN; neighbour-sampled mini-batches keep GPU memory bounded.

The PCA-to-128 step is **load-bearing for RAM**: skipping it would make the BANKSY matrix 2.16M × 5376 × 4 B ≈ 46 GB and OOM the host.

---

## Risks & mitigations

- **Stage 1 false-negative (context-rich nodes).** Mitigated by the staged design — Stage 2's context-poor nodes give the clean test.
- **Registry replay mismatch.** The element-wise assert against `sources.npy`/`labels.npy` catches it deterministically; fall back to a registry-emitting rebuild.
- **Cross-source µm/px differences.** Within-slide k-NN is immune (only relative distances inside one slide matter).
- **Batch confound.** The batch-identity probe quantifies it; report rather than hide.
- **Big-slide graph memory.** Neighbour sampling for s1 (757k); per-slide processing.

---

## Out of scope (YAGNI)

Production pipeline integration; leave-one-slide-out across all 6; medium/fine label tiers; non-breast tissue; radius (vs k-NN) graphs; GAT/attention (GraphSAGE-mean first). All deferred until the probe shows the lift.

---

## Provenance

Internal: backbone H2H (`project_backbone_h2h_2026_05`), patch-size sweep, subnuclear triangulation (`pipeline_output/subnuclear_2026_06/`, floor 0.497/0.563 vs 0.619), annotation benchmark (BANKSY 0.802). Literature (2026-06-09 scan): STELLAR (*Nat Methods* 2022, 10.1038/s41592-022-01651-8); SPACE-GM; cell-graph GNNs (IEEE TMI 2022; Med Image Anal 2025 review); BANKSY (neighbour-augmented cell typing). Baseline checkpoint `pipeline_output/h2h_2026_05_30/efficientnetv2_rw_s/best_model.pt`.
