# "Good Nucleus Patches v1" — Multi-Agent Review + Literature Research

**Date:** 2026-05-29
**Branch:** `feat/nucleus-qc-scorer`
**Reviews the:** `docs/superpowers/specs/2026-05-25-good-nucleus-patches-design.md` (master spec) and all current QC-scorer + nucleus-re-centering code.
**Method:** 8 parallel review/research subagents + a gemini second-opinion (codex hung on stdin in v0.135, ignored) + literature MCPs (paper-search, openalex, consensus, bioRxiv) + web. Every code finding below was re-verified against the live code at `file:line`.

---

## 0. Executive summary

1. **The experiment design is genuinely good** — the build→audit→score→filter→A/B/C structure, the STHELAR byte-identical placebo arm, and the cohort-drift-aware filtering comparison are all sound ideas. The pairing audit gate already **PASSED** (all 6 sources `frac_paired=1.000`, `frac_centroid_in_cell=1.000`).
2. **But there are P0 correctness bugs that would silently invalidate the results if the experiment ran today.** Two independent agents (training-side and data-side) converged on the same root cause: **patch↔label↔QC alignment is positional, not `cell_id`-keyed**, so the centering A/B is not actually paired across LMDBs; and the filtering arm in `breast_pooled_train.py` reads a **non-existent `qc_score` column** from the seg-QC output. A QC crash on empty masks and a **class-censoring risk** in the broken-decision gate compound this.
3. **Literature validates the core choices but caps expectations.** Keep **StarDist** (only OOTB DAPI segmenter with a clean per-object probability — which the QC objectness signal depends on). The DAPI-morphology premise is real but the **ceiling is modest**: coarse 4-class macro-F1 ≈ 0.62–0.72 (DAPIDL is already near it), medium 12-class ≈ 0.40–0.50, with immune/stromal subtypes fundamentally capped <0.4. The biggest available upgrade is a **DAPI-native foundation encoder (NuSPIRe)** and a **two-stream nucleus+context** design (NuClass).
4. **QC `structure_score` (LoG-in-mask) is defensible but weak alone** — LoG is the *most noise-sensitive* focus operator and conflates "flat Z-cap" with "dim/noisy". Add a **GLCM entropy/ASM** texture term (the literature's actual Z-cap discriminator) + Brenner; optionally cross-check with the **Yang-2018** Hoechst-trained focus CNN.
5. **The eval protocol needs cheap rigor upgrades.** A single seed + a flat "2% macro-F1" threshold is **not defensible** (run-to-run noise is 1–5%). Add — all nearly free — **McNemar on paired predictions + paired-bootstrap CIs on the difference + ≥3 seeds on the winner**, plus **MCC/balanced-accuracy** alongside macro-F1.

**Status of this review's own fixes:** **Phases 0, 1, 2, and the code-only part of Phase 3 are landed** on `feat/nucleus-qc-scorer` (commits 26327a1, c6af2b4, 0508e03, 1db8b28, 83fb76b) — all P0s, the P1 experiment-correctness items, the Phase-2 readout-rigor harness, and the Phase-3 QC texture metrics + noise-robust GCE loss are done with tests (**48 QC/centroid/ab-stats/loss tests pass**; the §5 re-centering gate ran green on real data; the A/B readout was smoke-tested end-to-end). Remaining: the ≥3-seed *runtime* runs (code ready) and the last heavy Phase-3 item needing a GPU (test-time BN — recipe in §3.2/§5). **Two-stream NuClass is DONE (52c28f0)** — `--backbone nuclass`, dual-scale v1 (NuSPIRe nucleus crop + microscopy-CNN context + learnable gated fusion), GPU-smoke-verified (dim 512, Δ=0, loss 1.178→0.001). The **NuSPIRe encoder is DONE (174f331)** — `--backbone nuspire`, loaded as `ViTMAEModel(mask_ratio=0)` + mean-pooled patch tokens (the card's `ViTModel`/`pooler_output` path was proven to leave the pooler randomly initialized), bit-exact deterministic, GPU-smoke-verified (dim 768, fine-tune loss 1.249→0.001, 5 GB VRAM). **B8 (the full-mosaic OOM risk) is FIXED (dfc9cda)** — the LMDB rebuild is now low-RAM — and **B10 (LMDB format tag) is FIXED (03b7247)**: `read_patches` now prefers an explicit self-describing tag over the first-key byte-heuristic, closing the P2 ledger. **Phase 4 hygiene:** the safe win is done (`harmonization` deprecated → `ontology`); the controller/heist/script cleanups are de-risked into an evidence-backed recipe in §5 but deferred (they need a ClearML pipeline run to verify, which isn't possible in this environment) — the measured 1.4 s import cost did not justify risking the registration. The experiment is correctness- and readout-ready, still gated on the user's pilot visual-validation review before the rebuild + A/B/C runs.

---

## 1. Current work state (verified)

| Component | State |
|---|---|
| QC scorer (`qc/segmentation_grounded.py`, `quality_control_seg.py`, `montage.py`) | Built + committed, 20 unit tests pass (GPU-free, StarDist monkeypatched). |
| `XeniumDataReader.get_nucleus_centroids_pixels()` | Committed (`data/xenium.py`). |
| Pilot nuc-centered LMDB (`build_pilot_lmdb_v3.py`) | Built — 3 sources, cell_id tracked via `patch_registry.parquet`. |
| Pairing audit (`pairing_audit.py`) | **PASS** all 6 sources. |
| Pilot QC collages + QA spreadsheets (`pilot_qc_collages_v2`) | Delivered for user visual review. |
| Full 6-source nuc LMDB / QC scoring / filter / A/B/C retrain | **Not started** (the real next phase). |
| `recentering_audit.py` (spec §5 gate) | **Does not exist yet.** |

---

## 2. Bug ledger (verified at file:line)

Severity: **P0** = would invalidate the experiment / crash; **P1** = biases results; **P2** = correctness/robustness.

| # | Sev | Location | Issue | Fix | Status |
|---|-----|----------|-------|-----|--------|
| B1 | P0 | `scripts/breast_pooled_train.py:208,232` | Filtering reads `["qc_score"]` from `qc_scores.parquet`; the seg-QC writes `seg_scores.parquet` with `structure_score`/`objectness_score`/**`broken`** — no `qc_score`. Filtering arm crashes on the new QC output. `.sort("cell_id")` is a footgun (cell_id is positional). | Drop `.sort`; add `--filter-broken` reading the `broken` column (positional within one LMDB = correct for the filtering arm); guard `--qc-threshold` with a clear error if its column is absent. | **FIXED** |
| B2 | P0 | `scripts/breast_dapi_lmdb.py:191-209,263-298` + `pipeline/steps/quality_control.py:_load_patch_labels` | Production builder persists **no `cell_id`**; `cell_id` is synthesized as the positional global index everywhere downstream. The nuc rebuild re-extracts Xenium through a different path; a nucleus-centered window can be OOB where the cell-centered one wasn't → the OOB-skip set differs → global-index→cell_id map diverges → **M_cell vs M_nuc_full is not paired by cell_id** (spec §4.5 claim fails silently). | Back-port `patch_registry.parquet` + `cell_ids.npy` from `build_pilot_lmdb_v3.py` into the production builder; assert cell_id sets identical (or take the intersection); derive splits from `hash(cell_id)`, not positional `train_test_split`. | **FIXED** — registry+`cell_ids.npy`+`--nucleus-centered` (c6af2b4); `--split-by-cell-id` stratified hash split (0508e03). Rebuild both arms with the registry builder; cross-LMDB cell_id-set assertion lives in the eval (Phase 2). |
| B3 | P0 | `qc/segmentation_grounded.py:172` | `regionprops(mask.astype(int))[0]` → `IndexError` on an all-False mask. Latent today (select_center guards) but detonates the whole 2.28M-patch batch if any future selector/relabel yields an empty mask. | Early-return a neutral metrics dict when `not mask.any()`. | **FIXED** |
| B4 | P1 | `qc/segmentation_grounded.py:243` | `morph_ok<1.0` (solidity≥0.50 & ecc≤0.98) and `intensity_ok<1.0` (ratio≥1.10) are **hard `false_detection` drop reasons** — but the function docstring says morphology must be "lenient (only extreme outliers count)". 0.50/1.10 are *moderate*, not extreme. This is the spec's stated **#1 risk (class-correlated censoring)**: small/dim immune & pyknotic nuclei get dropped. gemini + 2 agents independently flagged it. | Demote morph/intensity to **soft** objectness-score multipliers only (they already are, lines 182-183); hard gate = StarDist `prob` + area bounds. Adds `solidity_hard_min`/`eccentricity_hard_max` for genuinely degenerate shapes. | **FIXED** |
| B5 | P1 | `scripts/breast_pooled_train.py:184-185` | Incomplete determinism: no cuDNN-deterministic, no DataLoader `generator`/`worker_init_fn`, Python `random` unseeded. Same seed ≠ same result → a 2% delta is inside noise. | Full `set_seed()` + deterministic DataLoader; then ≥3 seeds on the winner. | **FIXED** (0508e03) — full `set_seed()` + cuDNN-deterministic + seeded generator/`worker_init_fn`. The ≥3-seed runs are Phase 2. |
| B6 | P1 | `scripts/breast_pooled_train.py:228-244` | `class_weights` + sampler recomputed from each arm's *filtered* counts → M_nuc_filt changes both data quality AND the imbalance prior, confounding the filtering effect. | Compute class weights once on the unfiltered nuc pool; inject the same vector into all arms. | **FIXED** (0508e03) — `--fixed-class-weights`: per-class vector from the pre-filter pool, reused for sampler + loss. |
| B7 | P1 | `data/xenium.py:212-219` | `group_by("cell_id").agg(...)` then left-join with **no `cell_id` dtype cast**: a dtype mismatch → all-null `xc/yc` → 100% silent fallback to cell centroid → the "nuc-centered" LMDB is byte-identical to cell-centered and the whole experiment is a no-op that *looks* like it ran. | Cast cell_id to `cells_df` dtype before join; warn if fallback >10%; write `fallback_count`/`centroid_source` to `metadata.json` (spec §4.1). | **FIXED** (c6af2b4) — pure `_nucleus_centroids_from_boundaries()` casts the join key + logs fallback; +2 unit tests. Real-data audit: rep1/rep2 fallback **0.0%**, median Δ≈2µm (proves the join resolves). |
| B8 | P2 | `data/xenium.py:124-155`, `data/sthelar.py:414-431`, builders | Full-mosaic `tif.asarray()` + full float32 normalize copy (~8–10 GB/Xenium rep; ~5 GB STHELAR) — spec §4.1 says use `tifffile.memmap` but it isn't. OOM risk on the 62 GB host. | memmap Xenium; `SthelarDataReader.load_dapi_region` for STHELAR; percentile on a subsample; normalize per-crop; chunk LMDB write txns. | **FIXED** (dfc9cda) — the OME-TIFF is tiled+compressed (NOT memmappable), so use a lazy `tifffile→zarr` store + new `LazyMosaic` (per-crop tile reads, channel-0 for multi-stain STHELAR) + per-crop `normalize_crop` (byte-identical to legacy) + subsample percentiles + 5k-patch txn chunks. Verified byte-equal to full load on real rep1; end-to-end smoke on all 6 sources. |
| B9 | P2 | `qc/segmentation_grounded.py` | Spec sub-scores `completeness_score`, `saturated_frac`, reserved `consensus_iou` are missing/dead; `stratified_audit` lacks the **per-`broken_reason`** breakdown that the censoring guardrail needs. | Emit `saturated_frac`; add per-reason audit; make `completeness` a real score or delete. | **PARTIAL** (0508e03) — per-`broken_reason`×class `seg_reason_audit.parquet` + per-class drop-rate canary done; `saturated_frac`/`completeness` cleanup deferred to P2. |
| B10 | P2 | `qc/io.py:48` | LMDB format A/B detected by "is first key ASCII-digit?" — both formats prefix 8 bytes; a format-A LMDB with `>Q` keys is misread. | Stamp an explicit `FORMAT_KEY` inside the LMDB; reader prefers it, unknown tag errors, untagged legacy falls back to the heuristic. | **DONE (03b7247)** — +4 tests; builders stamp the tag |

---

## 3. Research findings & recommendations

All citations cross-checked across ≥2 sources by the research agents; confidence noted.

### 3.1 Nucleus segmentation — **keep StarDist** (confidence: HIGH)

On DAPI/IF nuclear segmentation the top models are within 2–4 F1 of each other (Sankaranarayanan et al. 2025, *Comms Bio*: **Mesmer 0.67 / Cellpose-nuclei 0.65 / StarDist 0.63** @ IoU 0.5). For DAPIDL's actual needs — accurate **centroid** (re-centering) + full mask + **per-object confidence** (QC objectness) — StarDist wins on the deciding axes:
- **Centroid** error is far less sensitive than boundary IoU; StarDist's star-convex prior gives stable centroids on round, sparse Xenium nuclei, at the best speed/VRAM for 2.3M nuclei on a 3090.
- **Per-object probability:** StarDist emits a clean `polys["prob"]` — which `segmentation_grounded.py` already keys on (`prob_min=0.40`). **InstanSeg and Mesmer expose no usable per-object score**, so switching would silently break the QC objectness signal.
- **Watch-item:** StarDist degrades in *dense* nuclei (F1 0.69→0.47). If pilot QC shows merge errors in crowded tissue, do a targeted Cellpose-`nuclei`/InstanSeg second pass on dense tiles only — don't switch wholesale. Both adapters already exist in `benchmark/segmenters/`.
- **MERSCOPE (future spec C):** use **Cellpose-`nuclei`/Cellpose-SAM** (matches Vizgen tooling + the ~16× brighter intensity regime), not StarDist.

Refs: Schmidt et al. 2018 (StarDist, MICCAI); Goldsborough et al. 2024 (InstanSeg, arXiv:2408.15954); Pachitariu et al. 2025 (Cellpose-SAM, bioRxiv 2025.04.28.651001); Greenwald et al. 2022 (Mesmer, *Nat Biotech*); Sankaranarayanan et al. 2025 (*Comms Bio* 10.1038/s42003-025-08184-8).

### 3.2 Classifier — EfficientNetV2-S is a sound baseline; **NuSPIRe + two-stream is the upgrade** (confidence: HIGH on direction)

Two papers do *exactly* DAPIDL's task (Xenium DAPI + transcriptomic labels):
- **NuClass** (Xu et al. 2025, arXiv:2511.13586): 2.05M cells, 8 organs, 16 classes → **overall macro-F1 0.6235, acc 0.76**. Two-stream **224² nucleus crop + 1024² context**, gated — context adds +10 F1 on context-sensitive classes (lung T-cell 13.2→43.9→54.1) but *hurts* morphology-dominant ones (pancreas endocrine 87.4→68.9). This is the single most relevant benchmark.
- arXiv:2604.23481 (2026): breast Xenium DAPI, CellViT, 5+Unknown classes → macro-F1 0.3492.

**Per-class ceiling (from NuClass):** epithelial/tumor 0.94–0.96 (easy), endothelial 0.49–0.84, fibroblast 0.35–0.72, macrophage 0.35, **T-cell 0.22–0.54 (hardest)** — matches DAPIDL's own findings and confirms immune/stromal subtypes are fundamentally capped.

**Backbone:** EfficientNetV2-S remains a defensible low-data CNN baseline, but **NuSPIRe** (Hua et al. 2026, *Genome Biology* 10.1186/s13059-026-03987-2) — a ViT MIM-pretrained on **15.52M DAPI nuclei** (incl. Xenium), MIT-licensed, HF weights, 112² input — **beats EfficientNet/ConvNeXt/ResNet head-to-head** on cell-type ID and widens its lead in the few-shot/imbalanced regime DAPIDL lives in. H&E foundation models (UNI2/Virchow2/H-optimus) carry a DAPI domain gap; NuSPIRe is the only DAPI-native one.

**Label noise (DAPIDL's QC-filter idea is well-founded):** abstain on low-confidence transcriptomic labels (add an "Unknown"/ignore class — NuClass uses marker-vote τ=5) + a noise-robust loss (Generalized/symmetric CE, label smoothing) over plain weighted CE. Confidence-weighted CE is the floor.

**Realistic DAPIDL targets:** coarse 4-class **0.62–0.72** (already near — gains will come from the stromal class via context + cleaner labels, not the backbone); medium 12-class **0.40–0.50**. *Report per-class F1, not just macro.*

### 3.3 QC structure metric — keep LoG, **add GLCM entropy + Brenner** (confidence: HIGH)

Pertuz et al. 2013 (*Pattern Recognition*, the canonical 36-operator comparison, full text verified): Laplacian operators are **best at clean conditions but the single MOST noise-sensitive family**; statistics operators (normalized variance) are most noise-robust. So LoG works on bright nuclei and fails on the dim/noisy hard cases — exactly DAPIDL's prior visual-failure observation.

The **Z-cap problem is a texture problem, not a focus problem**: an in-focus apical/basal cap has *no chromatin texture* but normal LoG/intensity. LoG cannot separate "flat because tangential" from "low-energy because dim". The literature discriminator is **GLCM ASM/entropy** inside the nucleus (Lee et al. 2021 *Cytometry A* on breast nuclei; Haralick 1973): high ASM = homogeneous (cap), high entropy = textured chromatin (equatorial). **Add one in-mask GLCM term** — highest-value single change. Also add **Brenner gradient** (robust nucleus-autofocus workhorse) and a **normalized-variance** term (LoG's noise-robust complement).

**Learned cross-check (optional, near-free):** Yang et al. 2018 (*BMC Bioinformatics* 19:77; Apache-2.0, `github.com/google/microscopeimagequality`, CellProfiler/Fiji plugins) is a **Hoechst-trained** (≈DAPI) defocus CNN, 84² input, binary in/out-of-focus F=0.89. Use as an independent audit channel + an "uncertain" tier, not the sole gate.

**Critical guardrail (gemini #3 + 2 agents):** a texture/intensity gate **preferentially deletes the rare Immune/Stromal nuclei** (small, low-texture, dim) — the exact classes DAPIDL is already weak on (87/13/0.2%). Elevate **per-class drop-rate** from logged-metadata to a **hard verification gate**: if any class drops >25% above the dataset mean, re-tune. Keep per-slide normalization. Keep a soft "borderline" tier so QC strictness is itself ablatable.

### 3.4 Eval methodology — single-seed + 2% threshold is **not defensible** (confidence: HIGH)

Run-to-run variance (init + data order + cuDNN nondeterminism) is empirically 1–5% (Picard 2021; Pham et al. 2021 DEVIATE: up to 5.1% spread with *fixed* seeds; Bouthillier et al. 2021 MLSys). A 2% single-seed delta is inside that band. Three nearly-free upgrades, in priority order:

1. **McNemar's test on paired A-vs-B predictions** (Dietterich 1998: the *only* valid test when you can train once) — **zero extra training**, just dump per-cell `(cell_id, y_true, y_pred)` per arm. Highest ROI.
2. **Paired-bootstrap 95% CI on the macro-F1 *difference*** + per-class CIs (esp. Stromal: ~300 test cells is borderline — Welinder et al. 2021). Decision rule becomes "CI of the difference excludes 0", not "delta>2%".
3. **≥3 seeds on the winner** (full determinism first) so the STHELAR placebo becomes a *distribution*, not one draw.

**STHELAR placebo:** the A/A-test *concept* is sound and is the design's strength, **but** (gemini #4 + agent): M_cell and M_nuc_full are *different trained networks*, so the STHELAR delta measures cohort-swap perturbation, not the seed noise floor — it's biased away from zero. Reframe it as a **canary** (high STHELAR delta ⇒ run instability), and get the real noise floor from multi-seed reruns.

**Metrics:** add **MCC** (Chicco & Jurman 2020) and **balanced accuracy** — macro-F1 alone misleads under 0.2% imbalance + pseudo-label noise.

**Cross-platform (free win):** **test-time BatchNorm re-estimation** on the target platform's unlabeled patches directly attacks the 16× intensity shift (Vianna 2025: AUC 0.78→0.97 cross-institution). Prefer **Deep CORAL** over DANN (Sicilia 2023: DANN can *hurt* under the label shift DAPIDL has).

---

## 4. Repo architecture (consolidation debt)

Critical path for gnp-v1 is **tiny** (~8 src modules + 4 scripts) and bypasses the large `pipeline/` package. The mass is in dormant empires. Top consolidations (value/risk):

1. **6 pipeline controllers → 1.** `controller/enhanced/sota/unified/universal/orchestrator` (~4.8k LOC) all self-describe as "legacy"; CLI drives only `unified_controller`+`orchestrator`. `pipeline/__init__.py` **eagerly imports all six** at package import. Keep 2, gate the rest, drop from eager imports. (high value / low risk)
2. **Dual hierarchy systems → keep `ontology`, retire `harmonization`.** `harmonization.hierarchy` (broad/mid/fine, no CL IDs, 4 importers, last touched 2025-12-10) is superseded by `ontology` (coarse/medium/fine, CL-anchored, 26 importers, 2026-05). Port `cli compare-labels`, delete `harmonization/`.
3. **Quarantine `models/heist/`** (graph-NN experiment, only CLI importer, 2026-01).
4. Prune annotator zoo (18 → ~5 exercised); delete `breast_pooled_train.py.bak`, `data/instance_augment.py` (0 importers); gitignore committed `__pycache__`.

**Risk for current work:** importing `dapidl.pipeline.*` pulls all controllers + heavy optional deps — import the QC step by full module path. Hard dep on external editable `starpose` (`qc/segmentation_grounded.py` → `starpose.qc.base`) — pin the commit before the QC run.

---

## 5. Prioritized action plan

**Phase 0 — DONE (26327a1):** B1, B3, B4 — P0/P1 surgical fixes, tested.

**Phase 1 — DONE (c6af2b4, 0508e03): experiment correctness before any A/B/C run.**
- B7 (c6af2b4): pure dtype-guarded `_nucleus_centroids_from_boundaries()` + 2 tests.
- B2 (c6af2b4 + 0508e03): `breast_dapi_lmdb.py --nucleus-centered` persists `patch_registry.parquet` + `cell_ids.npy`; `breast_pooled_train.py --split-by-cell-id` does the stratified hash split.
- `recentering_audit.py` (c6af2b4): the §5 gate — **ran green on real data** (rep1 Δ=1.97µm, rep2 2.09µm, 0% fallback, STHELAR Δ=0).
- B5 (0508e03): full `set_seed()` + deterministic DataLoader.
- B6 (0508e03): `--fixed-class-weights` (pre-filter vector for sampler + loss).
- B9 (0508e03): `seg_reason_audit.parquet` + per-class drop-rate canary.

**Phase 1 runtime steps still owed by the operator (no code left):** (a) rebuild BOTH `breast-6source-dapi-p128` (cell) and `…-p128-nuc` with the registry builder so cell_ids align; (b) run `pairing_audit.py` + `recentering_audit.py` as gates; (c) gated on the user's pilot visual-validation review.

**Phase 2 — DONE (1db8b28): readout rigor.**
- `src/dapidl/evaluation/ab_stats.py` (+7 tests): `mcnemar` (exact/chi2), `bootstrap_macro_f1_diff` (paired CI), `per_class_f1_ci`, `macro_f1_fast`.
- `breast_pooled_train.py`: per-cell `preds.parquet` (source, cell_id, y_true, y_pred) + MCC/balanced-accuracy.
- `scripts/gnp_ab_readout.py`: pairs arms on shared (source,cell_id) with a y_true-consistency check; McNemar + bootstrap-CI per source & pooled-Xenium; per-class F1 CIs; multi-seed mean±SD. Smoke-tested end-to-end.
- **Still owed at runtime:** ≥3-seed runs per arm (just run `breast_pooled_train.py` 3× with different `--seed`, then pass all three `preds.parquet` to the readout's `--preds-*`).

**Phase 3 — method upgrades (research-backed).**
- **DONE (83fb76b):** QC texture metrics — `glcm_texture` (entropy+ASM, patch-range quantized → brightness-invariant Z-cap discriminator), `interior_cov`, `brenner` — written to `seg_scores.parquet` for laddering (diagnostic, not yet a gate; +5 tests). Noise-robust **GCE loss** (`GeneralizedCrossEntropy`, Zhang & Sabuncu 2018) in `training/losses.py` + `breast_pooled_train.py --loss gce` (+6 tests). Abstain-on-low-confidence already handled (label=-1 drop + `--filter-broken`).
- **NuSPIRe encoder — DONE 2026-05-30 (174f331), GPU-smoke-verified.** Empirically corrected the load: it's a **ViT-MAE**, not a plain ViT, so use `ViTMAEModel.from_pretrained("TongjiZhanglab/NuSPIRe", mask_ratio=0.0)` (0 missing encoder keys) — the recipe's original `ViTModel`+`pooler_output` leaves `pooler.dense` *randomly initialized* (ViT-MAE has no pooler) and HF warns the conversion is unsupported. Readout = **mean of patch tokens** with an explicit identity `noise` so features are bit-exact deterministic despite MAE's mask_ratio=0 patch shuffle (a paired A/B needs that). Native 1-channel (no `SingleChannelAdapter`), 128→112 resize inside the wrapper. `src/dapidl/models/nuspire.py` (`NuSPIReBackbone`, lazy `transformers` import) + `models/backbone.py` preset/route + `breast_pooled_train.py --backbone nuspire` (default unchanged → existing runs byte-identical) + `scripts/nuspire_smoke.py`. 9 unit tests (tiny injected encoder, no download). Smoke: dim 768, Δ=0, loss 1.249→0.001, 5 GB VRAM, 22 ms/fwd@B8.
- **Two-stream (NuClass) — DONE 2026-05-30 (52c28f0), GPU-smoke-verified.** `--backbone nuclass`: NuSPIRe on a tight 64px nucleus crop + a native-1-channel microscopy-CNN context encoder on the full patch, learnable **gated fusion** (`g·nucleus + (1−g)·context`; `gate_values` is an inspectable per-class nucleus-vs-context readout). **v1 caveat:** dual-scale of the *existing* 128px patch → context is only ~27µm, NOT the ~1024px wide-tissue FOV NuClass uses; the architecture is unchanged for the faithful version — only the context stream's input widens (a source-slide extraction pass). `src/dapidl/models/nuclass.py` + `backbone.py` preset/route + `breast_pooled_train.py --backbone nuclass` (NuSPIRe norm; default path byte-identical) + `scripts/nuclass_smoke.py`. 13 tests (injected stub streams, no download). Smoke: dim 512, Δ=0, loss 1.178→0.001, 5.7 GB VRAM, 25 ms/fwd@B8, 107M params.
- **TODO (needs GPU, separate spike):**
  - **Test-time BN** for cross-platform transfer — keep it OUT of the gnp-v1 A/B (would change two things at once); belongs to the separate transfer goal.
- **Manage expectations:** coarse ceiling ~0.62–0.72 (near-current), medium ~0.40–0.50; report per-class F1, not just macro.

**Phase 4 — repo hygiene (verification-first; mostly de-risked, not executed blind).**
Investigation found the import-cost fear overstated and the big deletions un-verifiable here (they touch the ClearML pipeline / CLI), so only the safe, unambiguous win was executed:
- **DONE:** `dapidl.harmonization` marked **deprecated** in favour of the CL-anchored `dapidl.ontology` (docstring + non-fatal `DeprecationWarning`; verified imports + exports intact; `compare-labels` unaffected). This stops the dual-hierarchy bug spreading without deleting anything.
- **Measured, NOT refactored:** `import dapidl.pipeline.steps.quality_control_seg` = **1.4 s**, and the catastrophic deps (scvi / torch_geometric / celltypist) are **import-guarded** (not loaded). A lazy-`__init__` refactor would risk the ClearML component-registration (unverifiable here) for ~0 benefit → **don't**.
- **Deferred (need ClearML verification or are user-owned files), with evidence:**
  - *6 controllers → keep `unified_controller` + `orchestrator`.* `pipeline/__init__.py` eagerly imports all 5 controller modules + `unified_config` + registry + components. External users: only `cli.py` (lazy, in-function: `EnhancedDAPIDLPipelineController`/`GUIPipelineConfig`, `SOTAPipelineController`, `list_*`). Recipe: move the controller imports behind PEP-562 `__getattr__`, keep the components-registration eager; verify `list_segmenters()`/`list_annotators()` non-empty AND a local ClearML pipeline run succeeds before deleting any controller.
  - *`models/heist` quarantine.* Only importer is `cli.py` (`heist-prepare`/`heist-train`, lazy). Safe to remove the 2 CLI commands + the 3 heist modules once you confirm the experiment is abandoned.
  - *Superseded scripts* `scripts/build_pilot_lmdb.py` (v1, byte-copy that lost cell_id) and `scripts/pilot_qc_collages.py` (v1) are unreferenced (superseded by `_v3`/`_v2`); safe to delete — left in place as user-owned files (surface, don't unilaterally delete).
  - *`data/instance_augment.py` + `data/sthelar_instance_dataset.py`* are the parked instance-seg WIP — **do not touch**.

---

## 6. Provenance

8 parallel subagents (QC-scorer correctness, data/re-centering pipeline, training/eval/A-B-C statistics, repo architecture, + 4 literature scouts: nucleus-seg, DAPI-morphology classification, focus/texture QC, cross-platform DA & eval methodology); gemini-3 second opinion (4 corroborating critiques); literature via paper-search / openalex / consensus / bioRxiv MCPs + web. codex (v0.135 `exec`) hung on stdin and was not used. Every code claim re-verified at `file:line` before inclusion.
