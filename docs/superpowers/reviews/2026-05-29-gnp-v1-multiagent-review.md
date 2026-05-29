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

**Status of this review's own fixes:** **Phase 0 + Phase 1 are landed** on `feat/nucleus-qc-scorer` (commits 26327a1, c6af2b4, 0508e03) — all P0s and the P1 experiment-correctness items are fixed with tests (35 QC/centroid tests pass; the §5 re-centering gate ran green on real data). Remaining: Phase 2 readout rigor (McNemar/bootstrap/MCC/3-seed), Phase 3 method upgrades, Phase 4 repo hygiene, and the two P2 robustness items (B8 memmap, B10 format tag). The experiment is now correctness-ready but still gated on the user's pilot visual-validation review before the rebuild + A/B/C runs.

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
| B8 | P2 | `data/xenium.py:124-155`, `data/sthelar.py:414-431`, builders | Full-mosaic `tif.asarray()` + full float32 normalize copy (~8–10 GB/Xenium rep; ~5 GB STHELAR) — spec §4.1 says use `tifffile.memmap` but it isn't. OOM risk on the 62 GB host. | memmap Xenium; `SthelarDataReader.load_dapi_region` for STHELAR; percentile on a subsample; normalize per-crop; chunk LMDB write txns. | PLANNED (§5 P2) |
| B9 | P2 | `qc/segmentation_grounded.py` | Spec sub-scores `completeness_score`, `saturated_frac`, reserved `consensus_iou` are missing/dead; `stratified_audit` lacks the **per-`broken_reason`** breakdown that the censoring guardrail needs. | Emit `saturated_frac`; add per-reason audit; make `completeness` a real score or delete. | **PARTIAL** (0508e03) — per-`broken_reason`×class `seg_reason_audit.parquet` + per-class drop-rate canary done; `saturated_frac`/`completeness` cleanup deferred to P2. |
| B10 | P2 | `qc/io.py:48` | LMDB format A/B detected by "is first key ASCII-digit?" — both formats prefix 8 bytes; a format-A LMDB with `>Q` keys is misread. | Tag format in `metadata.json` instead of guessing. | PLANNED (§5 P2) |

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

**Phase 2 — readout rigor (cheap, high-trust):** McNemar + paired-bootstrap CIs + MCC/balanced-acc in the A/B/C readout; ≥3 seeds on the winner; reframe STHELAR as canary; dump per-cell predictions + assert cell_id-set equality across the two LMDBs at eval time.

**Phase 3 — method upgrades (research-backed, after v1 baseline lands):**
- QC: add in-mask **GLCM entropy/ASM** + Brenner to `structure`; optional Yang-2018 audit channel.
- Classifier: prototype **NuSPIRe** encoder + **two-stream nucleus+context**; abstain-on-low-confidence + symmetric/GCE loss; test-time BN for cross-platform.
- Manage expectations: coarse ceiling ~0.62–0.72 (near-current), medium ~0.40–0.50.

**Phase 4 — repo hygiene (independent):** controller consolidation, retire `harmonization`, quarantine `heist`, dead-script cleanup.

---

## 6. Provenance

8 parallel subagents (QC-scorer correctness, data/re-centering pipeline, training/eval/A-B-C statistics, repo architecture, + 4 literature scouts: nucleus-seg, DAPI-morphology classification, focus/texture QC, cross-platform DA & eval methodology); gemini-3 second opinion (4 corroborating critiques); literature via paper-search / openalex / consensus / bioRxiv MCPs + web. codex (v0.135 `exec`) hung on stdin and was not used. Every code claim re-verified at `file:line` before inclusion.
