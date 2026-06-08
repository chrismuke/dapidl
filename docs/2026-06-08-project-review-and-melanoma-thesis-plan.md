# DAPIDL — Project Status Review & Melanoma Thesis Plan

**Date:** 2026-06-08
**Author:** Status review synthesised from internal benchmarks + three independent expert reviews (gemini, MCP literature scout, codex/gpt-5.5).
**Scope:** (A) where the universal DAPI→cell-type pipeline stands, (B) the visual-QA system, (C) next steps for the pipeline, (D) what spatial-transcriptomics data to use next, (E) validating against in-house manual immune counts, (F) honest appraisal + 12-month plan for the melanoma DAPI doctoral thesis.

---

## The bet, in one sentence

DAPIDL predicts cell types from the **DAPI nuclear stain alone**, trained with **free labels** harvested automatically from spatial transcriptomics (Xenium/MERSCOPE → CellTypist/popV/SingleR annotate each cell by gene expression → a CNN learns to predict that label from a 128 px DAPI patch). The wager: nuclear morphology + chromatin texture + local context carry enough signal to recover cell identity, and millions of free ST labels make it generalise.

---

## A. Where the universal pipeline actually stands (verified)

### A.1 What works — production-grade

| Component | Status | Evidence |
|---|---|---|
| **Classification backbone** | ✅ Production | EfficientNet-V2-S; coarse 4-class (Endo/Epi/Imm/Str) **test macro-F1 0.619** on held-out Xenium rep2 (per-class Endo 0.31 / Epi 0.86 / **Imm 0.66** / Str 0.65). Beats DAPI-native NuSPIRe (0.591) decisively (McNemar p≈5e-254). NuClass two-stream rejected. Book closed 2026-05-31. |
| **Multimodal DAPI+H&E fusion** | ✅ Best model you have | Cross-attention fusion on STHELAR (16 tissues, 9 classes): **test F1 0.5577**, beats DAPI-only (0.478) on *all 9 classes and 16 tissues*. Checkpoint: `pipeline_output/sthelar_modality_fusion/best_model.pt`. |
| **Automatic labelling stack** | ✅ Production | CellTypist + popV ensemble + SingleR + scType + SCINA + ground-truth; default `consensus` strategy, tissue-specific model selection. Unweighted popV + Blueprint SingleR ≈ 0.84 F1 coarse breast. |
| **Visual QA (broken-patch detector)** | ✅ Production, gated | StarDist-grounded `SegmentationGroundedScorer` on main; 48 tests; validated on 100k+ patches; anti-censoring stratified audit. (See §B.) |
| **Segmentation** | ⚠️ Consolidating | Moving onto the `starpose` library (Phase 1 plumbing done; backends/benchmark Phases 2–3 pending). StarDist remains the DAPI workhorse; CellViT/Mesmer avoided (non-commercial → Sonora licensing risk). |
| **Data scale** | ✅ Substantial, breast-heavy | ~300 GB local LMDB. Breast (Xenium, 6 sources) ≈ 70%; STHELAR pan-tissue (16 tissues, +H&E) ≈ 30%; skin (4 sources); mouse = brain pilot only. |

### A.2 What is hard / open — honest

- **Coarse ceiling ≈ 0.62–0.72.** Immune (T/B/macrophage) and luminal-epithelial classify well; **fibroblast, basal/myoepithelial, pericyte are the wall** (fine-grained stromal).
- **Inner chromatin signal is probably modest.** A 128 px patch carries four cue families — (1) inner subnuclear texture, (2) nuclear silhouette, (3) integrated intensity (DNA content), (4) tissue context. The patch-size sweep (bigger→better) and NuSPIRe (nucleus-only) losing to EfficientNet both point to **context dominating**. The experiment that would *quantify* the inner-chromatin contribution (`docs/superpowers/specs/2026-06-02-subnuclear-triangulation-design.md`: LightGBM information-floor + integrated-gradients attribution) is **designed but unrun**.
- **Two clean negative results** (don't re-litigate): re-centering patches on the StarDist nucleus (vs cell centroid) → **null**; QC-filtering "broken" patches before training → **slightly hurt**.
- **All pre-2026-06-08 absolute F1 is provisional.** A latent albumentations bug ran GaussNoise ~3–8× too strong on every training run before the fix. Relative rankings hold (same aug across arms); absolute numbers must be re-established on retrain.

---

## B. Visual-QA system — status verdict

**Bottom line: production-ready code, deployed on `main`, currently in a "gated-but-ready" state awaiting your final visual go/no-go.** Not a prototype, not abandoned.

- **Architecture:** `SegmentationGroundedScorer` (lives in `starpose.qc.segmentation_grounded`; dapidl re-exports via a shim + `src/dapidl/pipeline/steps/quality_control_seg.py`). Per patch it (1) StarDist-segments the 128² DAPI, (2) selects the centre nucleus, (3) scores **interior chromatin structure** (eroded-mask LoG energy, per-slide normalised), (4) **objectness** (StarDist `prob` + lenient morphology), (5) **centeredness/completeness/edge-cut**, and emits a high-specificity `broken_reason ∈ {no_nucleus, cut_at_edge, off_center, false_detection, ok}`.
- **The classical scorer is deprecated** — whole-patch Laplacian+Otsu did not visually separate good from bad (design review 2026-05-22).
- **Validated:** 48 tests; pilot collages v1/v2 over 100k+ patches show clear 5-group separation; **stratified per-(source×class×size) audit** confirms *no class-correlated censoring* (the key risk: don't silently delete small pyknotic lymphocytes — that's real biology).
- **Design intent achieved:** "a high-specificity rejector of *obviously* broken patches, grounded in segmentation, with a reason — when in doubt, keep." Filtering by structure score alone was tested (`pipeline_output/qc_threshold_2026_05/`) and **rejected** because it widened the cross-platform gap.
- **Open / deferred:** not a standalone CLI (runs inside pipeline steps); GLCM-entropy texture upgrade landed but not in defaults; starpose Phase 2/3 (InstanSeg fix, license metadata, PQ/F1 benchmark refresh) pending.

**This QA system is exactly the right substrate for the candidate's segmentation-validation milestone (§F).**

---

## C. Next steps for the universal pipeline (ranked)

Synthesised from internal state + codex's independent priorities. They agree.

1. **Re-establish the benchmark after the GaussNoise fix.** Retrain every surviving baseline (EfficientNet, NuSPIRe, fusion) with identical seeds and **grouped splits**; treat pre-fix absolute scores as obsolete. Without this, every downstream claim rests on a known-buggy baseline. *(Cheap, unblocks everything.)*
2. **Fix label/target quality before any new architecture.** The ceiling is a *label* problem, not a backbone problem. Use hierarchical labels, ensemble probabilities, annotation-disagreement signals, and **abstention**; stop forcing unreliable stromal subclasses; build a small **orthogonal IF/IHC-validated** label subset. (The cleanlab label-cleaning work already started is the right seed: rep1 precision 0.65/recall 0.33, single-CellTypist error ~38% → use the ensemble.)
3. **Run the subnuclear-triangulation experiment (designed, unrun).** It answers "how much does inner chromatin actually contribute vs context?" with *no retraining*. This directly de-risks the thesis's whole mechanistic premise and tells you whether architecture effort on isolated nuclei can ever help.
4. **Train for domain diversity, evaluate leave-one-domain-out.** Self-supervised pretraining on **physically normalised** DAPI (fixed µm/px) pooled across tissues/platforms/scanners/labs; report leave-one-domain-out. This is the only route to a defensible "universal" claim and the only thing that makes melanoma/mouse transfer plausible.
5. **Make uncertainty a first-class output + add multiscale context.** Calibrated hierarchical predictions, open-set detection, abstention, and composition-level validation. Fine stromal identity won't be solved by backbone churn on single nuclei — combine nucleus morphology with local neighbourhood + tissue architecture.

---

## D. What spatial-transcriptomics data to use next

You have **saturated breast**. Three strategic directions, in priority order:

### D.1 Melanoma / skin — opens the thesis *and* a new tissue (DO THIS FIRST)

All have a raw DAPI image and immuno-oncology markers. Verify the exact panel (`gene_panel.json`) before committing — flagged below.

| Dataset | Platform | Organism | DAPI image? | Exhaustion/activation markers | Access |
|---|---|---|---|---|---|
| **Xenium Prime 5K — Primary Dermal Melanoma** | Xenium Prime 5K (5001 genes) | Human | ✅ `morphology_focus/` OME-TIFF (+ post-run H&E) | 5K panel covers IO: CD3E, CD8A, GZMB, CTLA4, PDCD1, LAG3, TIGIT, TOX, HAVCR2 *(verify in panel)* | 10x portal, CC BY 4.0 |
| **Xenium Human Skin melanoma + 100 add-on** | Xenium v1 (282 + add-on) | Human (87,499 cells) | ✅ `morphology_mip.ome.tif` | CD3E/CD8A/GZMB likely; full checkpoint set UNVERIFIED | 10x preview datasets |
| **MERSCOPE FFPE Human IO — Melanoma (×2)** | MERSCOPE (500-gene IO) | Human | ✅ DAPI + boundary stains | PD-1, CTLA-4, LAG-3, TIGIT, GZMB, CD3/CD8; TOX/HAVCR2 UNVERIFIED | Vizgen FFPE showcase (registration) |
| **CosMx Human Melanoma** | CosMx (1000-plex IO + protein) | Human | ✅ DAPI | CD8A, CD3E, PDCD1, **HAVCR2, LAG3, TIGIT**, CTLA4, GZMB | Bruker/NanoString portal |
| **HTAN melanoma t-CyCIF (Gaglia 2022, *Cancer Cell*)** | t-CyCIF (protein) | Human | ✅ DAPI cycle 0 | PD-1, TIM-3, LAG-3, GzmB, CD8, CD3 (protein → no mRNA dropout) | Synapse **syn22781389** |
| GSE273530 — B16 melanoma | CosMx 1000-plex | **Mouse** | ⚠️ **images withheld in GEO** (matrix/RDS only) | Pdcd1 present; Havcr2/Lag3/Tox plausible | GEO GSE273530 (request images from authors) |

**The image-availability gap is itself a finding:** no public **mouse** melanoma ST ships downloadable DAPI with the full exhaustion panel. That makes **the candidate's own ~18,000 mouse DAPI images + manual counts uniquely valuable** — but it also means mouse exhaustion labels *cannot* be bootstrapped from public ST the way breast labels were.

**Recommended move:** lead with **Xenium Prime 5K human dermal melanoma** — it drops straight into the existing Xenium loader, lets you run the *same* automatic-labelling pipeline to build a melanoma immune-composition model, and gives a human anchor before tackling mouse transfer. Use t-CyCIF (syn22781389) as a protein-level ground-truth check.

### D.2 Cross-platform robustness — substantiate the "universal" claim

**CosMx is the untested commercial platform with native DAPI** and the richest exhaustion panel. Adding CosMx (and more MERSCOPE tissues) and reporting leave-one-platform-out is what turns "works on Xenium" into "platform-agnostic."

### D.3 Not useful for DAPIDL

Visium / Visium HD — **H&E only, no DAPI.** GeoMx — ROI-level, not cell-level. Skip for the DAPI pipeline.

---

## E. Validating against your in-house manual immune-cell counts

This is the **highest-value, most-defensible** use of the whole system: an external, orthogonal, hand-curated test set. Design it before touching the data.

**Step 0 — the question that decides everything: are the manual counts cell-registered or field/ROI-level?**
- **Field/ROI-level** (most likely for "manually analysed & quantified"): single-cell metrics are impossible. Primary endpoint = **composition/density agreement** per ROI.
- **Cell-registered** (manual marks on the same DAPI you score): single-cell metrics are valid.

**If field/ROI-level (assume this):**
- Compare predicted vs manual fraction/density per ROI with **Lin's Concordance Correlation Coefficient + Bland–Altman + MAE** — *not Pearson/Spearman alone* (a model that finds exactly 50% of cells scores Pearson = 1.0 but is 2× biased; CCC and Bland–Altman expose that).
- Coarse grid bins (e.g. 250 µm) to absorb serial-section registration drift.

**If cell-registered:**
- **Macro-F1 + Cohen's/weighted kappa**, matching predicted↔manual within 2–3 µm.

**Mandatory controls (this is what survives a viva):**
- Split by **mouse/tumour and acquisition batch**, never by cell or patch.
- A **batch-prediction probe**: can the embedding predict slide/scanner/batch? If yes, your "biology" is confounded.
- A **handcrafted-morphology baseline** (nuclear area/intensity/Haralick → simple classifier). The CNN must beat it to justify itself.

**Known pitfalls:** registration drift (serial sections never match 1:1 → ROI density only); membrane-marker "halo" bleed (PD-1/TIM-3 are membranous, DAPI is nuclear — IHC signal of one cell bleeds onto a neighbour's nucleus); necrotic nuclear fragmentation inflating CD8-like small-dense predictions (exclude high-DAPI-variance necrotic ROIs).

---

## F. The melanoma DAPI doctoral thesis — appraisal & 12-month plan

### F.1 Three independent reviews converged

gemini, an MCP literature scout, and codex/gpt-5.5 — different models, different evidence paths — independently reached the **same** conclusions. That agreement is worth taking seriously.

### F.2 Three scientific corrections the proposal MUST make

1. **Delete "DAPI reveals subtle perinuclear mRNA signals." It is a category error.** DAPI binds the AT-rich minor groove of **double-stranded DNA**; its weak RNA binding is spectrally red-shifted and rejected by standard DAPI filters (Kapuściński 1979/1995, *Nucleic Acids Res*; PMID 1372825). Any perinuclear DAPI signal is mitochondrial/cytoplasmic dsDNA, **not** mRNA. mRNA needs RNA-FISH/probes. Keeping this sentence will sink the methods section.
2. **The activated-vs-exhausted *binary* is biologically unsound as written — reframe to marker-defined phenotypes.** The proposed labels are defective:
   - **PD-1 is induced by ordinary T-cell activation** (PMID 18802087) — "CD8⁺PD-1⁺ = exhausted" is wrong.
   - **TIM-3 is neither necessary nor sufficient for exhaustion** and marks effector activation too (PMID 29463725).
   - GzmB⁺TIM-3⁺ coexist in proliferating *transitional* states (PMID 31810882).
   - Melanoma CD8 states are a **continuum, not two clean classes** (Sade-Feldman 2018, PMID 30595452).
   - **Fix:** predict each marker (CD137, GzmB, PD-1, TIM-3) as a separate **multilabel** endpoint; treat "activated"/"exhausted" composites as *secondary, exploratory* analyses, never the primary claim.
3. **There is a real morphological signal — but exhaustion-from-DAPI is unproven.** FOR: DAPI nuclear morphology predicts senescence at 95% (Heckenbach 2022, *Nat Aging*, 10.1038/s43587-022-00263-3) and kidney cell type at 80% (Woloshuk 2021, *Cytometry A*, 10.1002/cyto.a.24274); activation drives imageable chromatin de-condensation (Xu 2024, *Commun Biol*, 10.1038/s42003-024-06479-w; Bediaga 2021) and immune-cell morphology maps to transcriptional state (Severin 2021, *Sci Adv*, 10.1126/sciadv.abf6692); exhaustion involves genome-wide chromatin remodelling (Sen 2016; Pauken 2016, *Science*). AGAINST: **no published work shows DAPI distinguishes exhausted from activated CD8**, and that evidence comes from high-NA confocal/super-resolution — at Xenium's ~0.2 µm a 5–7 µm lymphocyte nucleus is only 10–25 px wide. So: *cell identity* from DAPI is literature-backed; *functional state* is a **hypothesis to be tested, with a modest ceiling.**

### F.3 Realistic ceilings

- Coarse immune composition (CD8 vs non-CD8, immune/tumour/stroma): **strong, defensible.**
- Per-marker positivity from DAPI under honest mouse-level, batch-independent testing: **AUROC ≈ 0.65–0.70, macro-F1 ≈ 0.55–0.62.** Treat any AUROC > 0.75 as **confounding until independently replicated**.

### F.4 Domain shift (breast→melanoma, human→mouse) — de-risking

Risks: **species** (mouse vs human nuclei, immune programs, genome AT-content → DAPI texture); **tissue** (melanin, necrosis, tumour atypia, different immune niches); **protocol** (fixation, antigen retrieval, DAPI concentration, multiplex cycles, mounting); **scanner/PSF** (NA, focus, bit depth, compression, spectral bleed); **magnification** (128 px is meaningless until converted to a fixed µm field of view); **task shift** (coarse identity → CD8 functional state is *not* ordinary transfer).

De-risk: resample everything to a **fixed µm/pixel**; manually validate segmentation on **1,000–2,000 stratified nuclei per batch**; **split by mouse/tumour + batch**; reserve a whole staining/scanning batch as an **untouched external test**; compare **frozen encoder / linear probe / fine-tune / from-scratch**; require **cell-registered mIF** for any cell-level claim; include morphology baselines and a batch-identity probe.

### F.5 Rebaselined 12-month plan (the 04/2025–03/2026 schedule has expired → 07/2026–06/2027)

| Months | Milestone | Key outputs / gates |
|---|---|---|
| **1–2** | **Data audit + preregistration** | Count *independent mice, tumours, slides, batches, fields* — "18,000 images" is **not** the sample size. Determine cell- vs field-level labels. Freeze endpoints, splits, exclusions, and **go/no-go thresholds** in writing. |
| **2–3** | **Segmentation validation** | Manually contour a stratified nuclear set. Report detection-F1, **PQ/AJI**, Dice, centroid error, merge/split rates. Lock StarPose/StarDist before any classification. *(Uses the §B QA system.)* |
| **4–6** | **Classification validation** | CD8-vs-non-CD8 + coarse immune/tumour/stroma first. Compare zero-shot transfer / fine-tune / from-scratch / handcrafted morphology. Report mouse-bootstrap CIs, calibration, AUROC, AUPRC, macro-F1. |
| **7–9** | **T-cell-state experiment (gated)** | Predict CD137, GzmB, PD-1, TIM-3 as separate multilabel endpoints. **Go forward only if external AUROC ≥ 0.65 consistently across mice; else publish the negative result and pivot to composition.** |
| **10–12** | **Composition + writing** | Predicted vs immunostain fractions via MAE / concordance / calibration / Bland–Altman (not correlation alone). Write up. |

### F.6 Single biggest viva risk

**Confounded validation masquerading as biology.** A patch-level split lets the network identify the *mouse, slide, scanner, staining batch, treatment arm, or tumour region* instead of T-cell state — producing a beautiful, meaningless AUROC. Pre-empt with: mouse-level splits, an external batch held out untouched, cell-registered labels, a **batch-prediction control**, **simple-morphology baselines**, and preregistered endpoints. *A rigorous null result survives a viva; a leaked AUROC of 0.90 does not.*

### F.7 What is publishable vs aspirational vs not defensible

- **Publishable:** validated DAPI nucleus segmentation on melanoma + a rigorous domain-transfer analysis + coarse immune composition validated against manual counts + a principled upper bound on state inference.
- **Aspirational (bonus):** modest, reproducible per-marker positivity prediction.
- **Not defensible:** definitive DAPI-only diagnosis of activated vs exhausted CD8 cells, or any "perinuclear mRNA" mechanism.

### F.8 Honest working title

> *"Assessment of DAPI-derived nuclear and spatial morphology as a surrogate for marker-defined CD8 phenotypes and immune composition in mouse melanoma."*

---

## G. Concrete actions (next 2 weeks)

1. **Pull Xenium Prime 5K human dermal melanoma**, confirm its panel contains the IO markers, and run the existing automatic-labelling pipeline on it — first non-breast melanoma anchor.
2. **Audit the candidate's data**: independent mice/tumours/slides/batches; are the manual counts cell-registered or field-level? This single answer rewrites the stats plan.
3. **Run the subnuclear-triangulation experiment** (designed, unrun) — settles how much inner chromatin contributes, cheaply, with no retrain.
4. **Kick off the post-GaussNoise benchmark rebuild** so every future comparison rests on a clean baseline.
5. **Share the three-corrections report (§F.2) with the candidate** before another word of methods is written.

---

*Evidence base: internal benchmarks (`project_backbone_h2h_2026_05`, `project_fusion_resume_state`, `2026-06-02-subnuclear-triangulation-design`, `2026-05-22-nucleus-qc-scorer-design`, `2026-05-29-gnp-v1-multiagent-review`) + three independent 2026-06-08 expert reviews. Key citations: Kapuściński 1979/1995; PMID 18802087, 29463725, 31810882, 30595452; Heckenbach 2022 (10.1038/s43587-022-00263-3); Woloshuk 2021 (10.1002/cyto.a.24274); Severin 2021 (10.1126/sciadv.abf6692); Xu 2024 (10.1038/s42003-024-06479-w).*
