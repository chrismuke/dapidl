# StarPose Segmentation Consolidation — Design Spec

**Date:** 2026-06-02
**Primary repo:** `starpose` (`/home/chrism/git/starpose`, MIT, v0.2.0 → **v0.3.0**)
**Consuming repo:** `dapidl` (`/mnt/work/git/dapidl`, branch `feat/nucleus-qc-scorer`)
**Status:** approved design, pre-implementation

## 1. Goal

Make **`starpose` the single, authoritative home for all cell/nucleus segmentation, segmentation-grounded QC, and the segmentation benchmark.** `dapidl` (and later SONORA) consume it purely as a **library + CLI** — no direct segmenter imports, no duplicated consensus/benchmark code. Then **validate the dispatcher's default** (`topological`) with a proper PQ/F1 benchmark, and harden the weak backends.

### The seam (the one architectural idea)

> **starpose owns `image → masks → segmentation-grounded features / QC / benchmark`.
> dapidl owns `patches → labels → classifier → attribution`.**

The interface across the seam is `starpose.types.SegmentationResult` + the QC feature table. Everything in this spec is moving code to the correct side of that seam and hardening what's there.

## 2. Motivation

1. **A real fork exists.** dapidl's pipeline routes segmentation through starpose, **but** the newer QC scorer `dapidl/src/dapidl/qc/segmentation_grounded.py` calls `from stardist.models import StarDist2D` **directly**, and the consensus code is **duplicated** (`dapidl/src/dapidl/benchmark/consensus/topological_voting.py` ≡ `starpose/src/starpose/consensus/topological.py`). The original QC spec (`2026-05-22-nucleus-qc-scorer`) even placed the scorer in `starpose/qc/` — it drifted into dapidl. This spec restores that intent.
2. **The default is under-validated.** `dispatch.py` hardcodes `return "topological"` citing "60.4% recovery" — but that MERSCOPE benchmark used a **"recovery" proxy, not PQ/F1**, the **InstanSeg arm was broken** (0.260, 2 cells in a dense FOV; dismissed as "designed for H&E" — false), and **Xenium showed no ensemble benefit**. See `project_starpose_seg_benchmark`.
3. **License visibility (not a blocker — academic use).** The H&E SOTA (CellViT/CellViT++), Mesmer/DeepCell, and CelloType's PanNuke weights carry **non-commercial** terms (Commons Clause / CC BY-NC-SA / academic-only). Under **academic use these are all usable**, so the hub records license as *informational metadata* (future-proofing if usage ever becomes commercial) — it does **not** gate or skip backends. This also unblocks **CellViT++** as the H&E SOTA default.
4. **2025–2026 evidence** (see research synthesis): nothing cleanly beats StarDist for round DAPI nuclei; its only real weakness is crowding; **InstanSeg** (Apache-2.0, TorchScript) is the permissive modern generalist worth fixing/promoting.

## 3. Scope

**In scope** — one spec, three sequenced phases:
- **Phase 1 — Plumbing:** starpose becomes the single entry point; move QC + de-dup consensus; dapidl consumes via thin shims.
- **Phase 2 — Backend hygiene:** fix InstanSeg adapter; license metadata + commercial gating; density-aware dispatch.
- **Phase 3 — Benchmark refresh:** proper PQ/F1/Jaccard/AP vs STHELAR masks + a hand-annotated Xenium spot-check; confirm or revise the default.

**Out of scope:** SONORA's own adoption (separate); the subnuclear-triangulation *build* (parked — **resumes on `starpose.qc` after Phase 1**); any change to the cellotype/joint instance-seg WIP in either repo.

## 4. Current state (what's where today)

**starpose** (`src/starpose/`): `core.py`, `dispatch.py` (`AdaptiveDispatcher`, default `topological`), `types.py` (`ModalityBundle`, `SegmentationResult`, `SegmentationLevel`), `cli.py` (Typer: `segment|benchmark|evaluate|methods|export`), `tiling.py`; `methods/` (`base.py` = `Segmenter`/`CellExpander` ABCs; `__init__.py` registry `register/create/list_methods`; backends `stardist`, `cellpose`, `instanseg` ⚠️53-line stub, `cellvit`, `cellotype`, `joint`); `consensus/topological.py` (`TopologicalVoting`); `qc/` (`QualityScorer` ABC + `ClassicalQualityScorer`). Optional-deps per backend; exporters (SpatialData/Xenium/MERSCOPE).

**dapidl** consuming today: editable dep on starpose; `pipeline/components/segmenters/starpose.py` (`StarposeSegmenter` adapter, the pipeline seg path); `pipeline/steps/quality_control.py` imports `starpose.qc.classical`; `data/xenium.py` imports `starpose.io.transcripts`. **Forked pieces to fix:** `qc/segmentation_grounded.py` (direct StarDist), `benchmark/consensus/topological_voting.py` (dup), `benchmark/runner.py` (segmentation benchmark).

## 5. Target architecture

### 5.1 Public library API (what dapidl imports)
```python
import starpose
res = starpose.segment(image, method="adaptive", pixel_size=0.2125, gpu=True)  # -> SegmentationResult
from starpose.qc import (QualityScorer, ClassicalQualityScorer,
                         SegmentationGroundedScorer, SegQCConfig,
                         nucleus_feature_vector)        # moved from dapidl
from starpose.methods import create, list_methods       # registry (now with license metadata)
```
Stability: `starpose.segment`, `starpose.qc.*`, `starpose.methods.create/list_methods`, `starpose.types.*` are the **stable public surface** for v0.3.0. CLI must reach everything the library does.

### 5.2 Module moves (starpose ← dapidl), Phase 1
| From (dapidl) | To (starpose) | Note |
|---|---|---|
| `qc/segmentation_grounded.py` (SegmentationGroundedScorer, SegQCConfig, scorers) | `starpose/qc/segmentation_grounded.py` | `_segment` swapped to `starpose.create("stardist").segment` — one StarDist path |
| planned `qc/patch_features.py` (subnuclear features) | `starpose/qc/patch_features.py` | segmentation-grounded ⇒ lives here |
| `benchmark/consensus/topological_voting.py` (+ majority/iou dup) | *delete* | `starpose/consensus/*` is canonical |
| `benchmark/runner.py` (segmentation parts) | folded into `starpose benchmark` | port dapidl-only FOV/metric logic |

dapidl keeps: classifier, LMDB, training, the IG **attribution** (model-level, not segmentation), and `quality_control*.py` *steps* (now thin wrappers over `starpose.qc`).

### 5.3 Thin back-compat shims (chosen cut strategy)
Each moved dapidl module becomes a re-export with a deprecation note, e.g. `dapidl/qc/segmentation_grounded.py`:
```python
"""DEPRECATED shim — moved to starpose.qc.segmentation_grounded (v0.3.0)."""
from starpose.qc.segmentation_grounded import *  # noqa: F401,F403
```
Primary call sites updated to import starpose directly; shims cover the long tail of `pipeline_output` scripts. Shims removed in a later cleanup.

## 6. Phase 2 — Backend hygiene

- **InstanSeg fix.** Replace the 53-line stub: select the **fluorescence/DAPI** InstanSeg model variant, correct input normalization, TorchScript inference. Acceptance: a smoke test on a known DAPI tile yields a sane cell count (the "2 cells in dense" pathology must fail the test).
- **License metadata (informational, not gating — academic use).** Add a `license` string to the registry: `register(name, cls, description, install_hint="", license=None)`. `list_methods()` and `starpose methods` surface it. **No skipping/gating** — the project is academic, so non-commercial backends (CellViT, Mesmer, CelloType's PanNuke weights) are usable. The field exists purely for visibility and to make a future commercial-gating switch trivial if ever needed. Recorded values: stardist/cellpose/instanseg/topological = BSD-3/Apache-2.0/MIT; cellvit = Apache-2.0 + Commons-Clause (NC); cellotype = Apache-2.0 code, PanNuke weights NC; mesmer/deepcell (if added) = academic-only.
- **Density-aware dispatch.** Replace hardcoded `return "topological"` with a rule: estimate density (e.g. StarDist/Cellpose count ratio or nuclei/area); **low density (Xenium-like, ratio<2×) → single fast method (`stardist`)**; **high density → `cellpose_nuclei`/`instanseg`/`topological`**. Default becomes evidence-driven; `_force_nucleus` override preserved.

## 7. Phase 3 — Benchmark refresh

- **Ground truth:** STHELAR provided nucleus masks (held-out slice, multi-tissue) as bulk GT **+ ~10–15 hand-annotated Xenium DAPI FOVs** (QuPath-assisted) as a gold spot-check. **Circularity guard:** do not score CellViT against STHELAR's own (CellViT-derived) masks.
- **Metrics:** **PQ (panoptic quality), F1@IoU{0.5,0.75}, mean Jaccard/IoU, AP** — per method × tissue × density-bin. (Replaces the "recovery"/"solidity" proxies.)
- **Methods:** stardist, cellpose(_nuclei/_sam), **instanseg (fixed)**, topological — on Xenium DAPI + STHELAR.
- **Outputs:** `starpose benchmark` writes a report (parquet + md) under starpose; `dispatch.py`'s default rationale updated to cite refreshed numbers (confirm or revise `topological`). Expectation per current evidence: single-method (StarDist or Cellpose-nuclei) likely wins/ties on Xenium; topological earns its keep only where density warrants.
- **Compute:** 3090 GPU, RAM-aware STHELAR tile streaming (62 GB host), via `uv run`.

## 8. Testing

TDD throughout (starpose has `tests/`). Moved QC code brings its tests (synthetic-patch unit tests for `SegmentationGroundedScorer`/`patch_features`). New: InstanSeg smoke test (sane counts), license-gating unit test (non-commercial skipped unless opted in), density-dispatch unit test (Xenium-like → stardist; dense → ensemble), benchmark metric unit tests (PQ/F1 on toy masks). dapidl side: a test asserting the shims re-export and that no dapidl module imports `stardist` directly anymore.

## 9. Constraints

- starpose stays **MIT**. License recorded as informational metadata only (academic use — non-commercial backends usable); no commercial gating.
- `uv run`; polars-first (pandas only at external-lib boundaries); per-backend optional-deps preserved.
- **Do NOT touch** the cellotype/joint instance-seg WIP (`starpose/methods/{cellotype,joint}.py`, dapidl `training/instance/*`).
- Version bump starpose v0.2.0 → **v0.3.0**; update dapidl's editable dep; keep `dapidl` test suite green via shims.

## 10. Risks & mitigations

- **Cross-repo move breaks dapidl call sites** → thin shims + update-and-test; a test forbids direct `stardist` imports in dapidl.
- **STHELAR GT is model-derived** → hand-annotated Xenium spot-check anchors it; circularity flag for CellViT.
- **InstanSeg fix needs a specific model/version** → pin it in the `instanseg` optional-dep + smoke test.
- **Scope is large** → three phases, each independently testable and shippable; Phase 1 unblocks the parked triangulation.

## 11. Sequencing

P1 (plumbing) → unblocks the parked **subnuclear-triangulation** (rebuilds on `starpose.qc`) → P2 (backend hygiene) → P3 (benchmark refresh, which may revise the P2 dispatch default). Each phase is a self-contained, testable unit for writing-plans.
