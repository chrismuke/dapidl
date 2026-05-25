# Nucleus Re-centering of Xenium Patches — Design Spec

**Date:** 2026-05-25
**Branch (parent for new work):** `main` (the `feat/nucleus-qc-scorer` branch is parked at `3174ae1` — independent)
**Status:** Approved by user (sections 1–3 of brainstorming), spec for review.

---

## 1. Problem

DAPIDL training patches for Xenium sources (`xenium-breast-tumor-rep1`, `xenium-breast-tumor-rep2`) in `breast-6source-dapi-p128` are extracted **centered on Xenium's `cells.parquet` cell centroid** (whole-cell centroid, includes cytoplasm), not on the nucleus. This produces systematically off-center nuclei in 128 px patches.

**Measured magnitude (rep1 + rep2, StarDist re-segmentation in the QC smoke):**

| metric | value |
|---|---|
| nucleus→patch-center offset median | ~1.3 µm |
| frac > 2.7 µm | ~25 % |
| frac > 4.8 µm | ~11 % |

**STHELAR sources are not affected.** STHELAR's `SthelarDataReader.get_centroids_pixels()` returns the centroid from `table_nuclei` (DAPI-based nucleus annotation). Empirical check on `breast_s0` (n = 576 582): STHELAR's stored centroid vs its own `nucleus_boundaries` polygon centroid → median = 0.000 µm, max = 0.046 µm, frac > 2.7 µm = 0.000. STHELAR patches are already on the nucleus polygon centroid.

**Why it matters:**
1. Confounds the nucleus-centeredness signal in the QC scorer (4/113 off-center flags in the smoke were genuine; the rest were a cell-centering artifact).
2. The CNN trains on slightly off-center nuclei — premise of the model ("predict cell type from a centered nucleus crop") is violated.
3. Retro-explains why p32/p64 underperform p128: smaller patches clip systematically offset nuclei (see auto-memory `project_patch_size_sweep_finding`).

## 2. Goal

Produce a parallel nucleus-centered training LMDB for the 6-source breast pool, verify the shift actually happened (and the QC metric responds), then A/B re-train the cross-source classifier to measure the classifier-level effect.

## 3. Non-goals

- Not running StarDist per patch (Xenium ships its own nucleus segmentation; using StarDist would *disagree* with the segmentation that the labels are derived from).
- Not running full-slide StarDist (overkill).
- Not re-extracting STHELAR (already nucleus-centered; patches would be byte-identical).
- Not changing patch size (`128 px` stays — sweep showed bigger is better; this addresses framing, not size).
- Not changing labels, splits, or model — only the LMDB pixels change for Xenium rows.

## 4. Approach

### 4.1 Centroid source: native Xenium `nucleus_boundaries.parquet`

For each Xenium source we compute the nucleus polygon centroid from `outs/nucleus_boundaries.parquet` (columns: `cell_id`, `vertex_x`, `vertex_y` in µm).

```python
# In src/dapidl/data/xenium.py
def get_nucleus_centroids_pixels(self) -> np.ndarray:
    """Per-cell nucleus polygon centroid (mean of vertices), in pixels.

    Returns (N, 2) in the same cell_id order as get_centroids_pixels(),
    so existing patch indices and labels carry over unchanged.
    """
    nb = pl.read_parquet(self._get_outs_path() / "nucleus_boundaries.parquet")
    centroids = (
        nb.group_by("cell_id")
          .agg(pl.col("vertex_x").mean().alias("xc"),
               pl.col("vertex_y").mean().alias("yc"))
    )
    # Left-join onto cells_df to preserve cell_id order. Cells missing a
    # nucleus polygon (rare — DAPI-negative cells) fall back to the cell
    # centroid so the index alignment is preserved.
    df = self.cells_df.join(centroids, on="cell_id", how="left").with_columns(
        pl.col("xc").fill_null(pl.col("x_centroid")),
        pl.col("yc").fill_null(pl.col("y_centroid")),
    )
    return np.column_stack([
        (df["xc"].to_numpy() / self.PIXEL_SIZE),
        (df["yc"].to_numpy() / self.PIXEL_SIZE),
    ])
```

`PIXEL_SIZE = 0.2125` µm/px (existing constant in `XeniumDataReader`).

### 4.2 LMDB build: re-extract Xenium, copy STHELAR

A new build path in `scripts/breast_dapi_lmdb.py` (flag `--nucleus-centered`) does:

1. **For each Xenium source** (rep1, rep2): call the existing `extract_xenium_breast(...)` codepath but swap `reader.get_centroids_pixels()` → `reader.get_nucleus_centroids_pixels()`. Reads `morphology.ome.tif` via `tifffile` memmap (already how it works), tile-resident only.
2. **For each STHELAR source** (s0, s1, s3, s6): **copy LMDB rows by `cell_id` from the existing `breast-6source-dapi-p128`** instead of re-decoding the source `sdata.zarr`. Same patch bytes, same `cell_id`, same label, same source tag — just a byte-copy under a new key range. Saves ~2/3 of the work and removes the only RAM-risky step (STHELAR zarr ingestion).
3. Write `metadata.parquet` and `metadata.json` with the same schema as the existing LMDB so downstream loaders need no change. Add `centroid_source: "nucleus_polygon"` (Xenium) / `"sthelar_nucleus"` (STHELAR) to `metadata.json` for provenance.

**Output:** `~/datasets/derived/breast-6source-dapi-p128-nuccentered/` (same on-disk layout as existing LMDB, ~30 GB).

### 4.3 Cells missing a nucleus polygon

Some Xenium cells don't have an entry in `nucleus_boundaries.parquet` (DAPI-negative or seg failure). Two options:
- **(a) Skip them** — drops the cell from the LMDB. Cleaner, but the LMDB has a different `cell_id` set than the cell-centered one and the A/B is not strictly paired.
- **(b) Fall back to cell centroid** — keeps the cell with its original framing. The A/B is on the same `cell_id` set; the affected cells are then no-change rows in both LMDBs.

**Decision: (b)**, fall back to cell centroid. The A/B is paired by `cell_id`. We log the fallback count in `metadata.json` and audit per source. Expected fallback rate: < 5 % (rough — to confirm empirically).

### 4.4 RAM / IO discipline

- Xenium `morphology.ome.tif`: `tifffile.memmap` → tile-resident reads, no full-slide load. Peak <2 GB per source.
- STHELAR copy path: `lmdb.cursor.put()` from source to dest cursor — no decode, no re-encode. Peak <500 MB.
- Total peak: comfortably <8 GB. Well under the 62 GB OOM ceiling.

## 5. Verification

Two artifacts, both cheap, gating the training run.

### 5.1 Recentering audit (`scripts/recentering_audit.py`)

For Xenium rep1/rep2 only, join the old (cell-centered) and new (nucleus-centered) `cells.parquet`-derived patch indices on `cell_id`, compute Δ µm per cell, report:

- median / p75 / p90 / max shift
- frac > 2.7 µm, frac > 4.8 µm
- fallback rate (cells with no nucleus polygon)

**Pass criteria:**
- median shift ∈ [0.5, 5] µm (matches expected ~1.3)
- fallback rate < 10 %

**Fail → abort before training.**

### 5.2 Centeredness recheck via existing seg QC

Re-run `quality_control_seg` (the `SegmentationGroundedScorer` built in `feat/nucleus-qc-scorer` T1–8) on the new LMDB. Compare per-source `centeredness` distributions side-by-side with the old LMDB.

**Pass criteria:**
- STHELAR sources: centeredness unchanged ± 0.02 (byte-identical patches, sanity)
- Xenium sources: centeredness rises from ~0.44–0.49 into the STHELAR range (~0.59–0.78)

**Output:** a one-page `nucleus_recentering_qc.md` summary with the two tables.

This gate also validates the QC metric itself: if patches actually re-center but `centeredness` doesn't respond, the QC scorer has a bug to fix before we trust it on future data.

## 6. A/B re-train

### 6.1 Two runs, only LMDB differs

| run | LMDB | model / hparams / splits / epochs |
|---|---|---|
| **A (control)** | `breast-6source-dapi-p128` | unchanged |
| **B (treatment)** | `breast-6source-dapi-p128-nuccentered` | unchanged |

Driver: existing `scripts/breast_pooled_train.py`, two invocations with `--dataset` flipped. `--patience 5` (per auto-memory `feedback_pooled_training_patience`). **Same seed across both runs** (paired splits, paired model init — so any A/B delta is from the LMDB, not from stochastic training variance). Logged to ClearML with tags `recentering-ab/control` and `recentering-ab/treatment`.

### 6.2 Readout

Held-out test split macro F1, reported three ways:

1. **Pooled** (all 6 sources)
2. **Per source** (rep1, rep2, s0, s1, s3, s6) — the headline view. STHELAR rows are unchanged → form a noise floor; Xenium rows carry the treatment.
3. **Per class** (Epithelial, Immune, Stromal, Endothelial) — surfaces whether the gain concentrates where nucleus morphology matters most.

### 6.3 Decision criteria

"Xenium-only macro F1" below = macro-averaged F1 across classes, computed on test rows whose `source ∈ {rep1, rep2}` (Xenium rows pooled). "Noise floor" = absolute |B−A| on the analogous STHELAR-only macro F1, which should be near zero since STHELAR patches are byte-identical.

| outcome | interpretation | action |
|---|---|---|
| **B > A by > 2 % on Xenium-only macro F1, and B−A on Xenium > noise floor on STHELAR** | recentering is a real win | nucleus-centered becomes default Xenium build; retire cell-centered path |
| **B ≈ A (within noise floor measured on STHELAR)** | recentering is geometrically correct but classifier doesn't need it | ship anyway (principled extraction); no urgency |
| **B < A by > 2 %** | unexpected regression | investigate (boundary cells now clipped at slide edge? higher fallback rate than expected? seg-vs-cell label mismatch?) before drawing conclusions |

### 6.4 Budget

~12–18 h per pooled run on the 3090 at current settings → ~1.5 days for both runs sequential. No new infrastructure.

## 7. Components & file map

| file | change |
|---|---|
| `src/dapidl/data/xenium.py` | **+ method** `get_nucleus_centroids_pixels()` |
| `scripts/breast_dapi_lmdb.py` | **+ flag** `--nucleus-centered`; new branch that swaps centroid source for Xenium and byte-copies STHELAR rows from an existing source LMDB |
| `scripts/recentering_audit.py` | **new** — old↔new cell-id-joined shift histogram + pass/fail gates |
| `tests/data/test_xenium_nucleus_centroids.py` | **new** — unit tests on a tiny synthetic `nucleus_boundaries.parquet` (correct polygon centroid for square / triangle / off-center polygon; ID-order preservation via left-join; fallback for missing-polygon cell) |
| docs / memory | update auto-memory `project_dapidl_patches_cell_centered` after audit with the actual measured shift |

**No changes** to: `data/patches.py`, `data/dataset.py`, the model, the training loop, the QC scorer, or any STHELAR code path.

## 8. Risks & mitigations

| risk | mitigation |
|---|---|
| Pixel-size / unit mistake (vertices in pixels not µm, or vice versa) — silent wrong centroid | TDD: unit test asserts known polygon → known pixel centroid using `PIXEL_SIZE = 0.2125`. Audit gate (5.1) catches median-shift outside [0.5, 5] µm. |
| Polygon centroid (mean-of-vertices) vs true area centroid disagree on irregular polygons | Mean-of-vertices is the same convention STHELAR uses (verified to match polygon's stored centroid to 0.000 µm). Keep consistent. |
| Nucleus extends beyond cell-centered crop → re-extraction needed from source slide, not crop-of-crop | Already the design — re-extract from `morphology.ome.tif`, not from the existing LMDB tiles. |
| Memory blow-up on full-slide DAPI | `tifffile.memmap` tile reads; already the existing extraction codepath. |
| Cells with no nucleus polygon get nonsense centroid | Left-join + fill-null with cell centroid (4.3); fallback rate logged. |
| QC metric (`centeredness`) doesn't move even though patches re-center | Caught by 5.2; means QC scorer needs a fix before we trust its outputs (worth knowing). |
| A/B noise > Xenium-only effect — can't tell signal from variance | Per-source breakdown uses STHELAR as built-in placebo: if STHELAR moves > 0.5 % between A and B, that's the noise floor and Xenium must beat it. |

## 9. Out of scope (explicit)

- StarDist-on-patch or full-slide re-segmentation
- Re-extracting STHELAR
- Changing patch size, model, or pipeline
- Generalizing to other Xenium datasets beyond rep1/rep2 (the build script is generic, but only these two are re-extracted in this work)
- Updating the existing `breast-6source-dapi-p128` in place (the new LMDB is parallel; the old one stays for A/B and for QC-scorer comparison)
