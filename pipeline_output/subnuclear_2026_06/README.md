# Subnuclear-Structure Triangulation — outputs

Quantifies how much **inner subnuclear DAPI structure** contributes to the production
EfficientNet-V2-S coarse classifier (test macro-F1 **0.619** on `xenium_rep2`), via two
**no-retrain** readouts over the `breast-6source-dapi-p128` LMDB. One shared StarDist pass
feeds both. Spec: `docs/superpowers/specs/2026-06-02-subnuclear-triangulation-design.md`;
plan: `docs/superpowers/plans/2026-06-08-subnuclear-triangulation.md`.

## Readouts

- **(C) Information floor** — `subnuclear_floor.py` → `floor_metrics.json`. LightGBM on
  per-nucleus features (`nuc_*`-only, then `nuc_*`+`ctx_*`), trained on the 5 train sources,
  tested on `xenium_rep2`. How close can a tree on simple features get to 0.619?
- **(D) Attribution concentration** — `subnuclear_saliency.py` → `saliency_summary.json` +
  `overlays/*.png`. Integrated Gradients on the EfficientNet checkpoint over a balanced rep2
  subset; headline metric = `fraction_of_|IG|_inside_nucleus ÷ nucleus_area_fraction`.

## Decision rubric

| Readout | Reads | Interpretation |
|---|---|---|
| (C) nuc-only floor vs 0.619 | macro-F1 gap | small gap → CNN's spatial subnuclear modelling adds little; large gap → it has value |
| (C) +context vs nuc-only | macro-F1 lift | how much context adds on top of nucleus summary stats |
| (D) concentration ratio | ÷ area-fraction | ≈1 nucleus ignored; ≫1 subnuclear-driven; <1 context-driven |

Combined → greenlight or skip the deferred phase-A retrain ablation matrix, per class
(e.g. immune may be subnuclear-driven while stromal is context-driven).

## Run

```bash
# Stage 1 only (smoke + ETA, stops for go/no-go):
bash scripts/run_subnuclear_triangulation.sh
# Full experiment (overnight, ~13 h at ~3.5 patch/s): full rep2 + 50k train sample
RUN_FULL=1 nohup bash scripts/run_subnuclear_triangulation.sh > sn.log 2>&1 &
# Faster ~6 h first-read (rep2 capped to 20k):
RUN_FULL=1 REP2_CAP=20000 nohup bash scripts/run_subnuclear_triangulation.sh > sn.log 2>&1 &
```

## Outputs (this directory)

- `seg_features_<src>[_cap<n>].parquet` — per-source feature tables (raw); merged →
- `seg_features.parquet` — the combined table (reusable: a seg-scores table over the LMDB).
- `center_masks/chunk_*.npz` — packed-bit center-nucleus masks (reused by the deferred phase A).
- `floor_metrics.json`, `saliency_summary.json`, `overlays/*.png`.

## Caveats

- **Throughput**: StarDist runs at ~3.5 patch/s here, so the full 2.28 M LMDB (~182 h) is
  infeasible. The run uses **full rep2** (for a fair (C) comparison to 0.619) + a **stratified
  train sample**. `REP2_CAP` subsamples rep2 for speed at a small comparability cost (noted in
  the readout via `n_test`).
- **`area_um2`** uses Xenium `0.2125 µm/px` for *all* sources (it's a feature, not a label).
- **No retraining** in this phase — both readouts run on the existing checkpoint.
