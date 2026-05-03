# DAPIDL Slide Rewrite — v2 Proposal

Goal: rebuild the deck around a **clear, data-driven story** so reviewers see
the science, not just the bullet points. Compresses 24 slides → ~16 (10
content + 4 result + 2 framing). Each result slide carries one figure already
generated under `pipeline_output/figures_v2/`.

---

## Story arc (the line a reviewer should remember)

> **Spatial transcriptomics gives us cheap labels. We turn DAPI nuclei into
> a cell-type classifier that reaches a ~0.6 F1 ceiling at coarse
> granularity, generalises to other tissues *if* their class diversity is
> reasonable, and improves another 5 F1 points when H&E is fused in. We
> quantified every step of that pipeline.**

Three claims, in order:
1. **Pipeline produces near-pathologist labels** (F1 ≈ 0.84 vs Janesick GT).
2. **There is a hard DAPI ceiling** (~0.6 F1 macro). Stromal/endothelial are the bottleneck; rare classes need 10× more data.
3. **Cross-tissue + cross-platform generalisation are bounded by data, not by architecture.**

---

## Slide-by-slide proposal

| #  | Title | Content | Figure / Source |
|----|-------|---------|-----------------|
| 1  | Title | Same image — but show real DAPI tile + your model's overlay, not the green montage | replace cover image |
| 2  | Goal & Vision | One slide: nuclear morphology → cell types via deep learning + spatial transcriptomics for labels. Drop the gold-standard table. | trim slides 3+4+5 → 1 |
| 3  | Spatial Transcriptomics in 30s | Keep current slide 6, but cut slide 7 (the colorful 10x marketing image isn't load-bearing). | keep slide 6 |
| 4  | **Headline numbers** (NEW) | 3 big numbers: Annotation F1 = 0.84 vs pathologist · DAPI ceiling F1 = 0.60 · Multimodal +0.05 F1. One sentence each. | NEW slide |
| 5  | The Pipeline | Keep current slide 8 (DAPIDL Pipeline funnel). Strong as-is. | keep slide 8 |
| 6  | Annotation pipeline benchmark | Replace slides 9 + 10 (algo consensus + ontology) with **Figure 4** repurposed: pop-V ensemble vs Janesick GT. | use existing Fig 4 |
| 7  | Cell Ontology hierarchy | Compact slide 11. Drop slide 12 (confidence tiers — fine but redundant with slide 13). | keep slide 11 |
| 8  | Ensemble voting | Compact slide 13 (combine with the dropped slide 12). | keep slide 13 |
| 9  | Architecture | One slide showing **only what's specific** to DAPIDL: SingleChannelAdapter, EffNetV2-S backbone, 3 heads (broad/medium/fine). Drop generic ViT diagram (slide 16) and generic transfer-learning diagram (slide 17). | redo slide 14 |
| 10 | **Patch size sweep**     | Bar chart: F1 vs {32, 64, 128, 256} for both Xenium and STHELAR | **fig02_patch_size_sweep.png** |
| 11 | **Granularity ladder**   | F1 vs n_classes; per-class F1 of best 4-class model | **fig01_granularity.png** |
| 12 | **Backbone & foundation models** | Keep current Fig 5 (CNNs vs frozen vs LoRA-tuned ViTs) | keep Fig 5 |
| 13 | **Cross-tissue (LOTO) + brain paradox** | Per-tissue LOTO + diversity scatter | **fig03_loto_diversity.png** |
| 14 | **Modality: DAPI vs H&E vs fusion** | Bar chart of 4 modality variants | **fig04_modality.png** |
| 15 | **Cross-source: STHELAR ↔ Xenium**| Bar chart with in-domain and out-of-domain bars | **fig05_cross_source.png** |
| 16 | **Training-data scale curve** | F1 vs train cells (5 fractions) | **fig06_data_scale.png** *(running)* |
| 17 | Limitations | Honest list: DAPI F1 ceiling ≈ 0.6, stromal/rare = bottleneck, single-tissue homogeneous data is a trap (cf. brain), domain shift costs 10–20 pp | NEW slide |
| 18 | Impact & Outlook | Same as current slide 24, slightly tightened | keep slide 24 |

**Total: 18 slides** (down from 24). 7 of them are figure-driven.

---

## Slides to delete or compress

| Old slide | Action | Why |
|-----------|--------|-----|
| Slide 4 — Gold Standard | merge into slide 2 | the icons-comparison adds little vs. one bullet |
| Slide 5 — DAPI examples (Wu & Shroff) | drop or move to backup | reviewers know what DAPI looks like |
| Slide 7 — Spatial Transcriptomics image #2 | drop | redundant with slide 6 |
| Slide 12 — Confidence Tiers | merge into slide 13 | overlapping content |
| Slide 14 — Generic CNN architecture | rewrite | currently illustrative-only |
| Slide 15 — Generic ViT architecture | drop | nothing DAPIDL-specific |
| Slide 16 — Generic transfer-learning diagram | drop | pop-sci visual, takes a whole slide |
| Slide 18 — Patch Size Comparison (text) | replace | use **fig02** instead |
| Slide 19 — Training Loop curriculum | drop or move to backup | the synthetic curve isn't from real data |

---

## Cross-cutting fixes

### Color contract
Pin to the same palette across **every** figure:

```
Epithelial    #ED553B   warm red
Immune        #3D5A80   navy
Stromal       #BC8D5A   warm brown
Endothelial   #4A9D9C   teal
Neoplastic    #9C2A2A   deep red
Adipocyte     #9DA17B   olive
Mast          #C58CD3   muted purple
DAPI          #2E86AB   blue
H&E           #B03A48   brick red
DAPI+H&E      #6F4C9C   purple
```

(All v2 figures use `scripts/figures_v2/_style.py` for these.)

### Title style
- Keep the bold blue title bar — it works.
- Add a **subtitle** under each figure-slide title, written as a complete
  sentence (the takeaway), not bullets. Example: "*Larger patches add
  tissue context — and consistently raise F1.*"

### Figure caption convention
Every figure-slide has, at the bottom in 9pt grey:
- The data source (which LMDB / which dataset)
- The number of test cells
- The model used (EffNetV2-S etc.)

This makes the figures defensible standalone if a reviewer screenshots them.

### Title image
The current cover (green-tinted breast slide) is striking but doesn't show
DAPI patches. Replace with a 4-panel: raw DAPI tile · segmentation overlay ·
predicted classes (false-color) · attention map.

---

## Headline-numbers slide (suggested copy)

> ### Three numbers to remember
>
> **0.84** — pop-V ensemble F1 vs expert pathologist (Janesick et al.)
> *"Our automatic labels match a pathologist on coarse breast-cancer types."*
>
> **0.60** — DAPI-only test F1 ceiling, coarse classification
> *"Nuclear morphology alone gets you most of the way."*
>
> **+0.05** — improvement when H&E is fused with DAPI (cross-attention)
> *"Adding a second stain helps but is not a silver bullet."*

Each number is a hook into a later slide.

---

## Where to draw the line for v2

What this rewrite does NOT add:
- A pathologist evaluation (would need new wet-lab work)
- A multi-tissue universal model (already shown to be weaker than per-tissue)
- A new architecture beyond EffNetV2-S (you've already tested 8)
- A `patch-size × backbone` cross (would require ~12 new training runs)

The story is complete with what you have, plus the data-scale curve.

---

## Where the v2 figures live

```
pipeline_output/figures_v2/
├── fig01_granularity.png        ← slide 11
├── fig02_patch_size_sweep.png   ← slide 10
├── fig03_loto_diversity.png     ← slide 13
├── fig04_modality.png           ← slide 14
├── fig05_cross_source.png       ← slide 15
└── fig06_data_scale.png         ← slide 16  (rendered after data-scale runs finish)
```

Source scripts: `scripts/figures_v2/fig_*.py`. To re-render after metric
updates, just rerun each script (no GPU needed except for the data-scale
training itself).

---

## Figure provenance & defensibility

For your supervisor / committee, here is exactly what each figure is built from:

| Figure | Source data | n cells | Reproducibility |
|--------|-------------|---------|-----------------|
| 01 granularity | breast_dapi_p128 (4-class), breast_dapi_sthelar_p128 (6-class), sthelar_exp5_7class, sthelar_modality_dapi (9-class) | 40k–190k per run | summaries in master_metrics.parquet |
| 02 patch size  | breast_dapi_p{32,64,128,256}, breast_dapi_sthelar_p{32,64,128,256} | ~40k test per p | analysis/summary.json each |
| 03 LOTO + div  | master_metrics.parquet (groups loto_dapi, loto_he, exp `brain`) + computed entropy from sthelar-multitissue-p128/labels.npy | 16 LOTO rows × 2 modalities | _tissue_class_dist.parquet cached |
| 04 modality    | master_metrics.parquet (group=modality) | 189,668 (same test split for all 4) | sthelar_modality_*/best_model.pt |
| 05 cross-source| breast_cross_source/*.json | 270k–280k each | direct .json |
| 06 data-scale  | data_scale_2026_05/frac_*/summary.json | 71,228 test (fixed across all 5) | this rewrite's new run |

---

## Suggested supervisor-ready chart of "what we measured"

(Could go on a methods page, optional.)

```
                        TRAINING                  EVALUATION
Annotation methods    n/a (vote)              vs Janesick GT (Xenium)         ─→ Fig 4 (existing)
Annotation methods    n/a (vote)              vs STHELAR ct_tangram (breast)  ─→ Fig 6 (existing)
Patch size sweep      Xenium / STHELAR        held-out test split             ─→ Fig 02 (new)
Granularity           4 / 6 / 7 / 9 classes   held-out test                   ─→ Fig 01 (new)
Backbone choice       8 architectures         coarse breast                   ─→ Fig 5 (existing)
Cross-tissue (LOTO)   15 STHELAR tissues      held-out tissue                 ─→ Fig 03 (new)
Modality fusion       DAPI / HE / both        same 9-class test               ─→ Fig 04 (new)
Cross-platform        Xenium ↔ STHELAR        opposite platform               ─→ Fig 05 (new)
Training-data scale   5 fractions of breast   fixed val/test                  ─→ Fig 06 (new)
```

Nine measurements, all on consistent eval splits. That's a lot of empirical
support for one deck.
