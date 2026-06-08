# cleanlab Label-Cleaning — Design Spec (2026-06-06)

Sub-project 1 of the label-quality effort (BANKSY = sub-project 2, deferred).

## Goal
Detect systematically-mislabeled cells in **weak transcriptomic labels** (Confident
Learning) so a DAPI classifier trains on cleaner labels. Ship a small, tested module;
validate it honestly against ground truth.

## Approach (approved)
Probs source = a **cross-validated expression classifier**. Run cleanlab on
`(weak_label, out-of-sample pred_probs)` → per-cell `is_issue` + `label_quality`.
v1 = **flag + exclude** (no relabeling).

## Module (the TDD deliverable) — `src/dapidl/data/label_cleaning.py`
Pure, dependency-light (numpy, scikit-learn, cleanlab; NO torch):
- `cv_pred_probs(X, y, n_folds=5, seed=42) -> np.ndarray` — out-of-sample class
  probabilities via `sklearn.model_selection.cross_val_predict(LogisticRegression(max_iter=1000,
  class_weight="balanced"), X, y, method="predict_proba", cv=StratifiedKFold(n_folds))`.
  Columns ordered by `np.unique(y)`. Guards: classes with < n_folds members → reduce folds.
- `find_label_issues(y, pred_probs) -> tuple[np.ndarray, np.ndarray]` — returns
  `(is_issue: bool[n], label_quality: float[n])` via `cleanlab.filter.find_label_issues(
  labels=y_encoded, pred_probs=pred_probs, return_indices_ranked_by=None)` and
  `cleanlab.rank.get_label_quality_scores(...)`. `y` mapped to 0..k-1 internally.
- `write_label_issues(out_path, row_idx, cell_ids, sources, labels, is_issue, label_quality)`
  → parquet with columns `row_idx, cell_id, source, label, label_quality, broken`
  where `broken == is_issue`. **The `broken` column means `breast_pooled_train.py
  --filter-broken --qc-scores <this>` consumes it UNCHANGED** (index-aligned, no trainer edit).

## Validation (operator run, after the module — reuses gnp-v1 infra)
Breast Xenium rep1 has both a weak-label source *and* Janesick GT, so we can grade cleanlab:
1. **Generate weak labels** for rep1 cells (CellTypist via `data/annotation.py`), mapped to
   coarse 4-class — these are the noisy labels to clean.
2. **cleanlab** on (weak label, `cv_pred_probs` from the rep1 expression matrix) → `is_issue`.
3. **Grade vs GT** (the cells where weak ≠ Janesick coarse GT are the *true* errors):
   report precision / recall / F1 of `is_issue` at detecting true errors. This is the
   honest test of whether cleanlab finds real label noise.
4. **Downstream A/B (deferred follow-on):** measuring a training-F1 lift needs an LMDB
   built on the *weak* labels — cell-r20 uses GT, so it can't host this cleanly. When
   pursued: rebuild rep1 patches with weak labels, train M_dirty vs M_clean
   (`--filter-broken`), test rep2 via `gnp_ab_readout.py`. The **core validation is
   steps 1-3** (detection precision/recall vs GT) — sufficient to decide whether cleanlab
   earns its place before spending on a weak-labeled rebuild.

Validation outputs: `pipeline_output/label_cleaning/rep1_cleanlab_report.md`.

## Error handling / edges
- Single-class input → return all-not-issue (cleanlab needs ≥2 classes).
- A class smaller than `n_folds` → `cv_pred_probs` lowers folds to `min(n_folds, min class count)`.
- `pred_probs` rows renormalized defensively; assert `len(y) == len(pred_probs)`.

## Testing (TDD)
- **Synthetic recovery (core guarantee):** `make_classification`/`make_blobs` 3-class,
  flip a known 10% of labels → `cv_pred_probs` + `find_label_issues` recover the flips at
  **recall ≥ 0.6 and precision ≥ 0.6** on the injected set.
- `cv_pred_probs`: returns `(n, k)`, rows sum ≈ 1, deterministic under seed.
- `find_label_issues`: `label_quality ∈ [0,1]`; clean data → few/no issues.
- `write_label_issues`: parquet round-trips; `broken == is_issue`; `row_idx` contiguous 0..n-1.

## Files
- Create `src/dapidl/data/label_cleaning.py`
- Create `tests/test_label_cleaning.py`
- Create `scripts/run_label_cleaning.py` (the validation run: weak labels → cleanlab → GT grade)
- Modify `pyproject.toml` (add `cleanlab`)

## Scope (YAGNI)
v1 = flag+exclude, breast Xenium rep1, no relabeling, **no trainer change** (reuse
`--filter-broken`). Defer: confident *relabeling*, per-class issue thresholds, multi-tissue,
STHELAR, and BANKSY (sub-project 2).
