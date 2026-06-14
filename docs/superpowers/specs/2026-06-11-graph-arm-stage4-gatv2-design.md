# Graph-Arm Stage 4 — Edge-Geometry GATv2 Arm

**Date:** 2026-06-11
**Branch:** `feat/spatial-gnn-probe` (Stage 4 of the spatial-GNN probe)
**Status:** Design approved, ready for implementation plan

## Context

Stage 3 (`docs/superpowers/specs/2026-06-10-graph-arm-stage3-loso-design.md`,
`pipeline_output/spatial_gnn_probe_2026_06/stage3_readout.md`) refactored the harness into a
pluggable `(encoder x aggregator x splitter)` evaluator and ran the run-first pair under LOSO.
Result: on production-strength frozen-EffNet node features the plain **mean** graph adds ~nothing
pooled (delta -0.0075); on the feature-clean rep2 fold it adds **+0.0161 macro**, and near-free
Correct-and-Smooth adds **+0.0159** — i.e. the mean graph's contribution collapses to label
diffusion. The roadmap (`memory: project_graph_arm_improvement_roadmap`) identified the next lever:
give the graph **structure that mean-pool cannot capture** — edge-conditioned attention over
**edge geometry** (distance + rotation-invariant nuclear orientation). This is that experiment.

The harness was built for it: the Stage-3 spec listed "edge-geometry + GATv2 (new Aggregator +
edge_attr from the kNN builder)" as the planned extension.

## Goal

Add an edge-geometry **GATv2** arm to the LOSO harness and test whether edge-conditioned attention
extracts spatial signal that plain mean-pool (and free Correct-and-Smooth) cannot — specifically on
the feature-clean rep2 fold and pooled across slides. Keep the same k=8 graph and the same
frozen-EffNet node features as Stage-3 E1 so the `mean -> gatv2` delta isolates the aggregator.

## Decisions taken during brainstorming

- **Single-scale (approach A).** Same k=8 within-slide graph and same frozen-EffNet PCA-128 node
  features as Stage-3 E1. Only the aggregator and the (new) edge attributes change. Multi-scale
  radius-banded graphs are deferred (they would change the graph and confound "attention vs scale").
- **True nuclear orientation on both platforms.** STHELAR from its native nucleus polygons (CPU);
  Xenium via a targeted StarDist pass over the ~286k rep1+rep2 DAPI patches (GPU). (The registry has
  centroids only; Xenium rep1/rep2 ship `cell_boundaries.parquet` + `cells.parquet` but NOT
  `nucleus_boundaries.parquet`, so Xenium nuclear orientation requires segmentation.)
- **GATv2** (Brody et al. 2022), not GAT or scalar edge-gating.
- **Additive edge_attr threading** so the existing 2 arms and the green characterization gate are
  untouched (edge geometry is opt-in).

## Data reality (verified)

- Registry `pipeline_output/spatial_gnn_probe_2026_06/spatial_registry.parquet`: columns
  `row_idx, source, cell_id, x_px, y_px, coarse_idx` (centroids only). N rows row-aligned to
  `embeddings_pca128.npy` and `labels.npy`.
- `cell_id` formats match the sources for joining: Xenium = integer-as-string (`"1"`, `"2"`);
  STHELAR = `"aaaaaaaa-1"` style.
- STHELAR nucleus polygons: `dapidl.data.sthelar.load_nucleus_geometry_with_labels(slide_root,
  label_cols) -> GeoDataFrame[geometry, ...]` indexed by `cell_id`, geometry in **pixel** coords.
- Xenium DAPI patches: the existing LMDB `/mnt/work/datasets/derived/breast-6source-dapi-p128`
  (`patches.lmdb`, 128px uint16; `decode_record(value, 128)`), row-aligned to the registry.
- StarDist center-nucleus extraction already exists and is used by `scripts/pilot_qc_rescore.py`
  via the `starpose.qc` scorer (`_segment(patch) -> (masks, probs)`, `select_center_nucleus`).
- Sources (6): `xenium_rep1`, `xenium_rep2`, `sthelar_breast_s0/s1/s3/s6`. All 0.2125 um/px.

## Architecture

All node features stay **frozen-EffNet PCA-128** (identical across nograph/mean/gatv2 — apples to
apples). Geometry feeds **edge attributes only**, so the three arms differ solely in how they
aggregate neighbours.

### New units

1. **`src/dapidl/graph/geometry.py` (new, pure)** — `ellipse_from_points(points_xy: np.ndarray) ->
   (angle_rad: float, eccentricity: float)`. PCA on a 2-D point set: `angle` = atan2 of the principal
   eigenvector (the major-axis direction, defined mod pi); `eccentricity` = `sqrt(1 - lambda_min/lambda_max)`
   in `[0, 1)`. Works identically for STHELAR polygon vertices and Xenium StarDist mask pixel coords.
   Degenerate input (< 3 points, or zero variance) -> `(nan, 0.0)`. Pure numpy, unit-tested.

2. **`phase_node_geometry` (new driver phase, `scripts/spatial_gnn_probe.py`)** — produces
   `pipeline_output/spatial_gnn_probe_2026_06/node_geom.npy`, shape `(N, 3)` =
   `[orient_angle_rad, eccentricity, log_area]`, row-aligned to the registry:
   - **STHELAR rows [CPU]:** per slide, `load_nucleus_geometry_with_labels` -> polygon per `cell_id`;
     `ellipse_from_points(polygon.exterior.coords)` for `(angle, ecc)`; `log_area = log1p(polygon.area)`
     (pixel area). Match registry rows by `(source, cell_id)`. Processed per-slide to bound RAM.
   - **Xenium rows [GPU]:** StarDist the cell's 128px DAPI patch (from `patches.lmdb`) via the
     `starpose.qc` scorer; `select_center_nucleus` -> the central mask; `ellipse_from_points(mask
     pixel coords)` for `(angle, ecc)`; `log_area = log1p(mask pixel count)`. ~286k cells, batched.
   - **Invalid / no-nucleus rows:** `angle = nan`, `ecc = 0.0`, `log_area = median(log_area)`. Logged
     count. (`build_edge_attr` zeros the directional terms wherever `angle` is nan — see below.)

3. **`src/dapidl/graph/edge_geometry.py` (new, pure)** — `build_edge_attr(coords, node_geom, nbr,
   rbf_centers=(4.0, 8.0, 16.0), rbf_gamma=0.05) -> edge_attr: np.ndarray (N, k, 8)`. Per neighbour
   slot `(i, j)` (where `nbr[i, slot] = j`, or `-1` padding -> all-zero row), all **rotation-invariant**:
   - `RBF(dist_ij)` x3 — Gaussian RBF of centroid distance (px) at the 3 centers. (dist is invariant.)
   - `cos(2*(theta_ij - axis_i))`, `sin(2*(theta_ij - axis_i))` — edge direction `theta_ij =
     atan2(dy, dx)` in node i's nuclear frame; the factor 2 makes it invariant to the mod-pi axis
     ambiguity. (Slide rotation rotates `theta_ij` and `axis_i` equally -> difference invariant.)
   - `|cos(axis_i - axis_j)|` — alignment of the two nuclei's major axes (the abs handles mod-pi).
   - `ecc_j` — neighbour eccentricity.
   - `|log_area_i - log_area_j|` — relative nuclear size.
   **NaN handling:** if `axis_i` or `axis_j` is nan (no orientation), the three directional terms
   (`cos2`, `sin2`, `|cos dAxis|`) are set to `0.0` for that edge; RBF/ecc/area terms still apply.
   This preserves rotation-invariance for fallback nodes. Pure numpy, unit-tested.

4. **`src/dapidl/graph/gnn.py` — `EdgeGATv2Aggregator(nn.Module)`** (new): GATv2 attention with edge
   features. `__init__(node_dim, edge_dim, heads=4)` with `head_dim = node_dim // heads` so the
   concatenated multi-head output is `node_dim` (matching `Mean`/`NoGraph` output width, so
   `GraphArmModel`'s `lin = Linear(node_dim*2, hidden)` is unchanged). Class attrs
   `needs_neighbours = True`, `needs_edge_attr = True`. `forward(se, ne, valid, edge_attr)`:
   - GATv2 score `e_ij = a^T LeakyReLU(W [se || ne_j || edge_ij])` per head, over the k neighbour
     slots; mask `valid == 0` slots with `-inf`; `softmax` over k.
   - value `= W_v ne_j`; output `= sum_j alpha_ij * value_j`, heads concatenated -> `[B, node_dim]`.
   - A node with no valid neighbours (all `-inf`) -> uniform-zero attention -> zero output (guarded).
   Has learnable parameters (unlike Mean/NoGraph) — see the harness factory change below.

5. **Harness edge_attr threading (additive; `src/dapidl/graph/harness.py`)**:
   - `GraphArmModel.forward(self_rows, nbr_rows, valid, edge_attr=None)`: gather `ne` as today; then
     `agg = self.aggregator(se, ne, valid, edge_attr) if getattr(self.aggregator, "needs_edge_attr",
     False) else self.aggregator(se, ne, valid)`. **`NoGraphAggregator`/`MeanAggregator` are NOT
     edited** (their 3-arg signature is only ever called the old way).
   - `train_arm(..., edge_attr=None, ...)`: in `step(rows, train)`, when `edge_attr is not None`,
     gather `ea = torch.from_numpy(edge_attr[rows]).float().to(device)` (`[B, k, F]`) and pass to
     `model(rows, safe, valid, ea)`; else pass no edge_attr.
   - `run_ablation(..., edge_attr=None, compare_pairs=None, ...)`: forward `edge_attr` to every
     `train_arm` call. **Generalize the comparison logic** (Stage 3 hard-coded `graph` vs `nograph`):
     `compare_pairs: list[tuple[str, str]]` of `(baseline_tag, candidate_tag)`; default
     `[("nograph", "graph")]` (Stage-3 backward-compat). For each pair present in the arm set, record
     per-fold and pooled `delta_macro = candidate - baseline` and `mcnemar = mcnemar_test(truth,
     base_pred, cand_pred)`. Stage 4 passes `compare_pairs=[("nograph","mean"), ("mean","gatv2"),
     ("nograph","gatv2")]`. (The characterization test only reads per-arm `macro_f1`, so this change
     does not affect the gate.)

6. **Aggregator factories (required interface change; `harness.py`)** — because
   `EdgeGATv2Aggregator` is **parametric**, a single shared instance reused across folds/arms would
   leak trained weights. Change the aggregator argument from instances to **factories**, mirroring
   the existing `encoder_factory`:
   - `train_arm(encoder_factory, aggregator_factory: Callable[[], nn.Module], ...)`: call
     `aggregator = aggregator_factory()` (fresh per fold/arm).
   - `run_ablation(encoder_factory, aggregator_factories: dict[str, Callable[[], nn.Module]], ...)`.
   - Update the two existing Stage-3 call sites (`phase_stage3_loso`, `phase_stage2_proper_harness`)
     to wrap aggregators in lambdas: `{"nograph": NoGraphAggregator, "graph": MeanAggregator}` (the
     classes themselves are zero-arg factories) or `lambda: NoGraphAggregator()`.
   - **The characterization gate is the safety net:** re-run `phase_stage2_proper_harness` after the
     refactor; it must still reproduce 0.537/0.628 (+/-0.02). Stateless Mean/NoGraph give identical
     results under factories.

7. **`phase_stage4_gatv2` (new driver phase) + `phase_stage4_readout`**:
   - `run_ablation(lambda: FrozenFeatureEncoder(feats_dev, device), {"nograph": NoGraphAggregator,
     "mean": MeanAggregator, "gatv2": lambda: EdgeGATv2Aggregator(node_dim=128, edge_dim=8,
     heads=4)}, LOSOSplit(...), nbr=nbr, labels=labels, edge_attr=edge_geom, device=device,
     compare_pairs=[("nograph","mean"), ("mean","gatv2"), ("nograph","gatv2")])`.
     The nograph/mean arms ignore `edge_attr` (apples-to-apples with Stage-3 E1); only gatv2 uses it.
     -> `stage4_gatv2_metrics.json` (per-arm metrics + the three pairwise deltas/McNemar).
   - Readout `stage4_readout.md`: per-fold + pooled `mean` vs `gatv2` delta (the isolation), the
     feature-clean rep2 row, McNemar (gatv2 vs mean), and the comparison vs E1 (mean graph) and E2
     (free C&S): does gatv2 clear the ~+0.016 bar both stalled at? Same feature-clean tiering
     (`FEATURE_CLEAN = {"xenium_rep2"}`, verified from the h2h summary).

## Data flow

Reuse: registry, `embeddings_pca128.npy`, the k=8 `build_within_slide_nbr_table`. New one-time
`node_geom.npy` (phase_node_geometry) -> `edge_attr (N,k,8)` (build_edge_attr, cached in-memory in
the phase) -> 3-arm LOSO via the harness -> readout. The Xenium StarDist pass is the only new GPU
step; everything else reuses Stage-3 artifacts.

## Success criterion

The decision question: on the feature-clean **rep2** fold, does the `gatv2 - mean` delta clear the
~+0.016 macro bar that plain mean (E1) and free C&S (E2) both stalled at — and is the lift
concentrated on Endothelial/Stromal? If yes, edge-geometry attention is the real lever and the
multi-scale follow-on is justified. If gatv2 ~ mean, the graph genuinely caps at diffusion on these
features and we stop. Honest LOSO + feature-clean tiering + per-fold/pooled McNemar throughout.

## Error handling & edge cases

- **No central nucleus (Xenium) / no polygon match (STHELAR):** record `angle=nan, ecc=0,
  log_area=median`; `build_edge_attr` zeros directional terms for those edges. Log the count.
- **Padded neighbour slots (`nbr == -1`):** `edge_attr` row all-zero; already masked by `valid` in
  the aggregator.
- **Node with zero valid neighbours:** GATv2 softmax over all `-inf` -> guarded to zero output (the
  `lin(cat([se, 0]))` still uses the self features).
- **STHELAR polygon in microns vs pixels:** `load_nucleus_geometry_with_labels` already converts to
  pixel coords (its docstring) — orientation is scale-invariant anyway; area uses the converted
  pixel polygon for consistency with Xenium pixel masks.
- **OOM:** `node_geom` is `(N,3)` (~27 MB); `edge_attr` is `(N,8,8)` float32 (~580 MB) — built once,
  fits. STHELAR polygons loaded per-slide (not all at once). StarDist runs the existing batched QC
  path. GNN runs on cached features.

## Testing strategy (TDD, pure cores first)

- `geometry.py`: a known axis-aligned ellipse of points -> expected angle (0 or pi/2) and ecc;
  rotating the points by phi rotates `angle` by phi (equivariance) and leaves `ecc` unchanged
  (invariance); degenerate (<3 pts) -> `(nan, 0)`.
- `edge_geometry.py`: **rotation-invariance** — rotate all coords + all `node_geom` angles by phi,
  assert `edge_attr` unchanged; NaN axis -> directional terms zeroed; `-1` neighbour -> zero row;
  output shape `(N, k, 8)`.
- `gnn.py` `EdgeGATv2Aggregator`: forward output shape `[B, node_dim]`; attention weights over valid
  neighbours sum to 1; a node with all-invalid neighbours -> finite (zero) output; changing
  `edge_attr` changes the output (edge features actually influence attention).
- `harness.py`: `GraphArmModel.forward` with an edge-attr aggregator returns `[B, num_classes]` and
  passes `edge_attr`; with a non-edge aggregator the old 3-arg path is used (regression);
  `run_ablation`/`train_arm` accept factories and an `edge_attr` array.
- **Characterization gate (required):** re-run `phase_stage2_proper_harness` after the factory
  refactor; assert it still reproduces 0.537/0.628 (+/-0.02).
- GPU/LOSO phases are controller-run.

## Compute / RAM budget

- StarDist over ~286k Xenium patches: ~30-90 min GPU (dominant new cost).
- STHELAR polygon geometry: minutes CPU, per-slide.
- `edge_attr` ~580 MB; `node_geom` ~27 MB.
- 18 GNN runs (3 arms x 6 folds) on cached features: ~1-2 h.
- Comfortably within the 62 GB host / 24 GB GPU.

## Out of scope (follow-ons)

- Multi-scale / radius-banded graph (only if single-scale gatv2 beats mean).
- Augmenting **node** features with shape (we keep nodes = frozen EffNet for a clean ablation).
- Expression-teacher LUPI distillation (the other Tier-2 swing; separate spec).
