# Three corrections to strengthen the melanoma-DAPI thesis proposal

**Re:** Melanoma DAPI thesis proposal — *"Charakterisierung von Melanomen durch Deep-Learning-basierte Zellkern-Segmentierung und Klassifikation"*
**Date:** 2026-06-08
**Status:** constructive pre-implementation review

---

## Why this document exists

The core of the project is sound and worth doing: DAPI nucleus segmentation + DAPI-based cell typing on melanoma, validated against manual immunostain quantification, is a defensible and publishable thesis. The deep-learning model already works for coarse cell identity, your in-house mouse melanoma DAPI images are a genuinely rare and valuable resource (no public mouse-melanoma spatial dataset ships downloadable DAPI images), and the validation design is realistic.

Three claims in the current proposal, however, are scientifically vulnerable and would be challenged in a viva. All three are **fixable without weakening the project** — in fact the fixes make it stronger and easier to defend. Each correction below gives the problem, the evidence, and the concrete remedy.

> A note on the citations: these PMIDs/DOIs come from an independent literature review. **Verify each one yourself before citing it in the thesis** — that is good scholarly practice regardless.

| # | The claim as written | The problem | The fix |
|---|---|---|---|
| 1 | DAPI reveals "subtle perinuclear mRNA signals" | Category error — DAPI stains dsDNA, not mRNA | Remove the mRNA claim; ground the mechanism in **chromatin/nuclear morphology** |
| 2 | Distinguish "activated" (CD8⁺CD137⁺, GzmB) vs "exhausted" (CD8⁺PD‑1⁺, TIM‑3⁺) as two classes | The marker definitions are biologically unsound; the states are a continuum | Predict **each marker separately (multilabel)**; treat activated/exhausted as exploratory, not primary |
| 3 | The model will classify these states from DAPI with clinical precision | State-from-DAPI is unproven; identity-from-DAPI is what's established | Set an **honest ceiling**; make immune **composition** the primary endpoint |

---

## Correction 1 — "Perinuclear mRNA signals from DAPI" is a category error

**What the proposal says.** That DAPI staining visualises "kaum interpretierbare perinukleäre mRNA-Signale," and that the model exploits these.

**Why it is a problem.** DAPI (4′,6‑diamidino‑2‑phenylindole) binds the **AT-rich minor groove of double-stranded DNA**, where it fluoresces strongly (~460 nm). It binds RNA only weakly and non-specifically, with a red-shifted, much dimmer emission that standard DAPI filter sets actively reject — and it cannot distinguish mRNA from rRNA or identify any transcript (Kapuściński & Szer 1979; Kapuściński 1995). Detecting **mRNA** in situ requires sequence-specific RNA-FISH / probe-based methods (e.g. single-molecule FISH), with DAPI used only as a separate nuclear counterstain. Any genuine perinuclear DAPI signal would be **mitochondrial or cytoplasmic dsDNA**, not mRNA.

A reviewer or examiner will catch this immediately, and it undermines the credibility of the methods section.

**The fix.** Delete the mRNA claim entirely. The defensible — and still novel — mechanism is **nuclear morphology and chromatin organisation**: nuclear size and shape, heterochromatin condensation/clumping, nucleolar prominence, and nuclear-envelope integrity. That is a real, imageable signal (see Correction 3) and is what the model actually learns.

---

## Correction 2 — The activated/exhausted binary is biologically unsound as written

**What the proposal says.** Two cleanly separated CD8 T-cell classes: **activated** = CD8⁺CD137⁺ / Granzyme B; **exhausted** = CD8⁺PD‑1⁺ / TIM‑3⁺.

**Why it is a problem.** The marker logic does not hold:

- **PD‑1 is induced by ordinary T-cell activation**, not just exhaustion (PMID 18802087). "CD8⁺PD‑1⁺ = exhausted" therefore mislabels recently activated effector cells as exhausted.
- **TIM‑3 is neither necessary nor sufficient for exhaustion** and also marks effector activation (PMID 29463725).
- **Granzyme B and TIM‑3 co-occur in proliferating, transitional populations** that sit *within* the exhaustion trajectory — so the "activated" and "exhausted" marker sets overlap on the same cells (PMID 31810882).
- In melanoma specifically, **CD8 T-cell states form a continuum**, not two discrete bins (PMID 30595452).

Training a classifier on labels that are themselves biologically incorrect produces a model that learns the wrong target, and the error is not recoverable downstream.

**The fix.** Do not predict a two-class "activated vs exhausted" label. Instead:

1. Treat each marker — **CD137, Granzyme B, PD‑1, TIM‑3** — as a **separate binary (multilabel) endpoint**. This makes no assumption about how markers combine into states.
2. Report per-marker performance, and only *then*, as a **secondary, exploratory** analysis, examine marker co-occurrence patterns (e.g. PD‑1⁺TIM‑3⁺ vs CD137⁺GzmB⁺) as candidate state signatures.
3. Frame "activation" and "exhaustion" as **interpretive hypotheses about marker combinations**, never as the ground-truth classes the model is trained on.

This is more honest, more flexible, and — importantly — still lets you tell the activation/exhaustion story, just without staking the thesis on a definition examiners will reject.

---

## Correction 3 — Cell *identity* from DAPI is established; functional *state* from DAPI is unproven

**What the proposal says.** That the established model will resolve these immune states from DAPI "mit klinisch relevanter Präzision und Robustheit."

**Why it is a problem — and where the proposal is actually right.** There is strong precedent that a **DNA stain alone carries deep biological information**:

- DAPI nuclear morphology predicts **cellular senescence** at up to ~95% accuracy across cell types and species (Heckenbach et al. 2022, *Nature Aging*, 10.1038/s43587-022-00263-3).
- A 3D DAPI-only CNN classifies kidney **cell types** at ~80% balanced accuracy (Woloshuk et al. 2021, *Cytometry A*, 10.1002/cyto.a.24274).
- T-cell **activation** drives imageable chromatin de-condensation and nuclear-envelope changes (Xu et al. 2024, *Commun Biol*, 10.1038/s42003-024-06479-w; Bediaga et al. 2021, *Sci Rep*), and immune-cell morphology maps to transcriptional state (Severin et al. 2021, *Sci Adv*, 10.1126/sciadv.abf6692). Exhaustion involves genome-wide chromatin remodelling (Sen et al. 2016; Pauken et al. 2016, *Science*).

So **cell identity from nuclei is literature-backed**, and there *is* a real morphological basis for activation state. **But:**

- **No published work shows DAPI distinguishes *exhausted* from *activated* CD8 T cells.** That specific claim is a hypothesis, not an established result.
- The supporting evidence comes from high-NA confocal / super-resolution microscopy. At a spatial-transcriptomics resolution (~0.2 µm/pixel), a 5–7 µm lymphocyte nucleus is only ~10–25 pixels wide — subtle internal chromatin topology is largely unresolvable, and apparent "texture" in dense tissue is often physical crowding, not chromatin state.

**The fix — set an honest ceiling and endpoint hierarchy.**

- **Primary endpoint:** immune **composition** (e.g. CD8 vs non-CD8; immune/tumour/stroma fractions per field/ROI). This is robust and defensible.
- **Secondary endpoint:** per-marker positivity (Correction 2).
- **Exploratory:** activation/exhaustion signatures.
- **Quantitative expectation:** under honest, batch-independent testing, plan for per-marker **AUROC ≈ 0.65–0.70** (macro-F1 ≈ 0.55–0.62). **Treat any AUROC above ~0.75 as a red flag for confounding** (e.g. the model recognising the mouse, slide, or staining batch rather than T-cell biology) until independently replicated on a held-out batch.

---

## The reframed working title

> **"Assessment of DAPI-derived nuclear and spatial morphology as a surrogate for marker-defined CD8 phenotypes and immune composition in mouse melanoma."**

This says exactly what the study can defend: it measures *DAPI nuclear/spatial morphology* against *marker-defined* phenotypes and *composition* — with no overclaim about reading mRNA or diagnosing discrete exhaustion states.

---

## What stays completely sound (the defensible thesis)

- **DAPI nucleus segmentation** on melanoma, validated against manual contours (detection-F1, PQ/AJI, Dice). The required QC/segmentation tooling already exists and is production-ready.
- **Coarse DAPI-based cell typing** (immune/tumour/stroma, CD8 vs non-CD8), validated against your manual immunostain counts.
- A rigorous **domain-transfer analysis** (human→mouse, breast→melanoma) — itself a publishable contribution.
- A principled **upper bound** on what DAPI can and cannot say about T-cell functional state — a *rigorous negative or modest-positive result that survives a viva*, which a leaked, confounded AUROC of 0.90 does not.

None of the three corrections shrinks the thesis. They move the strong claims from *indefensible* to *exactly defensible*, which is what gets a doctorate through its examination.

---

### References to verify and cite

- Kapuściński J. & Szer W. (1979) *Nucleic Acids Res* — DAPI fluoresces with dsDNA, not RNA.
- Kapuściński J. (1995) *Biotech Histochem* 70:220–233 — DAPI as a DNA-specific probe.
- PD‑1 induced by T-cell activation — PMID 18802087.
- TIM‑3 not exhaustion-specific — PMID 29463725.
- Granzyme B⁺TIM‑3⁺ transitional populations — PMID 31810882.
- Melanoma CD8 states form a continuum — PMID 30595452.
- Heckenbach et al. (2022) *Nature Aging* — 10.1038/s43587-022-00263-3.
- Woloshuk et al. (2021) *Cytometry A* — 10.1002/cyto.a.24274.
- Severin et al. (2021) *Science Advances* — 10.1126/sciadv.abf6692.
- Xu et al. (2024) *Communications Biology* — 10.1038/s42003-024-06479-w.
- Bediaga et al. (2021) *Scientific Reports* — 10.1038/s41598-021-93180-1.
- Sen et al. (2016) / Pauken et al. (2016) *Science* — chromatin remodelling in T-cell exhaustion.
