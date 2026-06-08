# cleanlab Label-Cleaning Implementation Plan

> **For agentic workers:** Inline TDD execution (superpowers:executing-plans). DRY · YAGNI · TDD · frequent commits.

**Goal:** A small tested module that flags systematically-mislabeled cells in weak labels (Confident Learning), plus a validation runner that grades it against breast-Xenium GT.

**Architecture:** `src/dapidl/data/label_cleaning.py` — pure (numpy + sklearn + cleanlab, no torch): `cv_pred_probs` (out-of-sample probs) → `find_label_issues` (cleanlab) → `write_label_issues` (parquet with a `broken` column so `breast_pooled_train --filter-broken` consumes it unchanged). A runner script wires it to real rep1 weak-vs-GT labels.

**Tech Stack:** numpy, scikit-learn, cleanlab, polars, pytest.

---

## Task 1: cleanlab core — `cv_pred_probs` + `find_label_issues`

**Files:** Create `src/dapidl/data/label_cleaning.py`; Test `tests/test_label_cleaning.py`; Modify `pyproject.toml` (add cleanlab).

- [ ] **Step 1: add dep** — `uv add cleanlab` (expect pyproject + uv.lock updated; `uv run python -c "import cleanlab"` OK).
- [ ] **Step 2: failing test** (`tests/test_label_cleaning.py`):
```python
import numpy as np
from sklearn.datasets import make_classification
from dapidl.data import label_cleaning as lc


def _noisy_dataset(seed=0, flip_frac=0.10):
    X, y = make_classification(n_samples=900, n_features=20, n_informative=12,
                               n_classes=3, n_clusters_per_class=1, random_state=seed)
    rng = np.random.default_rng(seed)
    flip_idx = rng.choice(len(y), size=int(flip_frac * len(y)), replace=False)
    y_noisy = y.copy()
    for i in flip_idx:
        y_noisy[i] = (y[i] + rng.integers(1, 3)) % 3  # force a different label
    return X, y, y_noisy, set(flip_idx.tolist())


def test_cv_pred_probs_shape_and_normalized():
    X, y, _, _ = _noisy_dataset()
    p = lc.cv_pred_probs(X, y, n_folds=5, seed=0)
    assert p.shape == (len(y), 3)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_find_label_issues_recovers_injected_flips():
    X, y_true, y_noisy, flipped = _noisy_dataset()
    probs = lc.cv_pred_probs(X, y_noisy, n_folds=5, seed=0)
    is_issue, quality = lc.find_label_issues(y_noisy, probs)
    flagged = set(np.where(is_issue)[0].tolist())
    recall = len(flagged & flipped) / len(flipped)
    precision = len(flagged & flipped) / max(len(flagged), 1)
    assert recall >= 0.6 and precision >= 0.6
    assert quality.shape == (len(y_noisy),) and quality.min() >= 0 and quality.max() <= 1
```
- [ ] **Step 3: run, expect FAIL** (`uv run pytest tests/test_label_cleaning.py -v`) — ModuleNotFoundError / AttributeError.
- [ ] **Step 4: implement** `label_cleaning.py`:
```python
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def cv_pred_probs(X, y, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    folds = max(2, min(n_folds, int(counts.min())))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    # encode labels to 0..k-1 so prob columns align to sorted classes
    enc = {c: i for i, c in enumerate(classes)}
    y_enc = np.array([enc[v] for v in y])
    probs = cross_val_predict(clf, X, y_enc, cv=cv, method="predict_proba")
    return probs


def find_label_issues(y, pred_probs):
    from cleanlab.filter import find_label_issues as _fli
    from cleanlab.rank import get_label_quality_scores as _q
    y = np.asarray(y)
    classes = np.unique(y)
    enc = {c: i for i, c in enumerate(classes)}
    y_enc = np.array([enc[v] for v in y])
    is_issue = _fli(labels=y_enc, pred_probs=pred_probs, return_indices_ranked_by=None)
    quality = _q(labels=y_enc, pred_probs=pred_probs)
    return np.asarray(is_issue, dtype=bool), np.asarray(quality, dtype=float)
```
- [ ] **Step 5: run, expect PASS.**
- [ ] **Step 6: commit** `git add src/dapidl/data/label_cleaning.py tests/test_label_cleaning.py pyproject.toml uv.lock && git commit -m "feat(label-quality): cleanlab CV-probs + label-issue detection"`

## Task 2: `write_label_issues` (parquet, `broken`-compatible)

**Files:** Modify `src/dapidl/data/label_cleaning.py`; `tests/test_label_cleaning.py`.

- [ ] **Step 1: failing test:**
```python
def test_write_label_issues_roundtrip(tmp_path):
    import polars as pl
    out = tmp_path / "label_issues.parquet"
    n = 5
    lc.write_label_issues(out, row_idx=list(range(n)), cell_ids=[f"c{i}" for i in range(n)],
                          sources=["xenium_rep1"] * n, labels=[0, 1, 2, 0, 1],
                          is_issue=np.array([True, False, False, True, False]),
                          label_quality=np.array([0.1, 0.9, 0.8, 0.2, 0.7]))
    df = pl.read_parquet(out)
    assert df.columns == ["row_idx", "cell_id", "source", "label", "label_quality", "broken"]
    assert df["broken"].to_list() == [True, False, False, True, False]
    assert df["row_idx"].to_list() == list(range(n))
```
- [ ] **Step 2: run, expect FAIL** (AttributeError: write_label_issues).
- [ ] **Step 3: implement** (append to label_cleaning.py):
```python
def write_label_issues(out_path, row_idx, cell_ids, sources, labels, is_issue, label_quality):
    import polars as pl
    df = pl.DataFrame({
        "row_idx": list(row_idx), "cell_id": [str(c) for c in cell_ids],
        "source": list(sources), "label": list(labels),
        "label_quality": [float(q) for q in label_quality],
        "broken": [bool(b) for b in is_issue],
    })
    df.write_parquet(out_path)
    return out_path
```
- [ ] **Step 4: run, expect PASS.**
- [ ] **Step 5: commit** `git add src/dapidl/data/label_cleaning.py tests/test_label_cleaning.py && git commit -m "feat(label-quality): write_label_issues parquet (broken-compatible)"`

## Task 3: validation runner `scripts/run_label_cleaning.py`

**Files:** Create `scripts/run_label_cleaning.py`.

Operator script (no unit test — manual run). Loads breast Xenium rep1 expression + weak labels (CellTypist via `data/annotation.py`) + Janesick GT (via the supervised-GT loader), runs `cv_pred_probs` + `find_label_issues`, and prints/writes a report: detection precision/recall/F1 of `is_issue` vs the true `weak != GT` errors → `pipeline_output/label_cleaning/rep1_cleanlab_report.md`. Writes `label_issues.parquet` alongside.

- [ ] **Step 1: implement** the runner (CLI: `--rep rep1 --max-cells N --out <dir>`), reusing `XeniumDataReader` for expression + `_load_xenium_supervised_gt` for GT + `CellTypeAnnotator` for weak labels.
- [ ] **Step 2: smoke** `uv run python scripts/run_label_cleaning.py --rep rep1 --max-cells 3000 --out pipeline_output/label_cleaning` — confirm it prints precision/recall vs GT and writes the report.
- [ ] **Step 3: commit** `git add scripts/run_label_cleaning.py && git commit -m "feat(label-quality): rep1 cleanlab validation runner (GT-graded)"`

---

## Self-Review
- **Spec coverage:** module (T1+T2), validation/GT-grade (T3), `broken`-compatible parquet for `--filter-broken` reuse (T2), test thresholds (T1). Downstream training A/B is spec-deferred (no task — correct). ✓
- **Type consistency:** `cv_pred_probs(X,y,n_folds,seed)→ndarray`, `find_label_issues(y,probs)→(bool[],float[])`, `write_label_issues(out,row_idx,cell_ids,sources,labels,is_issue,label_quality)` — consistent T1↔T2↔T3.
- **Risk:** cleanlab API names (`cleanlab.filter.find_label_issues`, `cleanlab.rank.get_label_quality_scores`) — verified at GREEN; adjust import path to the installed cleanlab version if needed.
