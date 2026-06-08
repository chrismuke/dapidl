import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from subnuclear_common import (  # noqa: E402
    balanced_subset,
    ctx_feature_columns,
    nuc_feature_columns,
    select_pass_indices,
)


def test_select_pass_indices_stratified_cap():
    sources = np.array(["a"] * 10 + ["b"] * 5)
    labels = np.zeros(15, int)
    idx = select_pass_indices(sources, labels, ["a", "b"], max_per_source=3, seed=1)
    assert (sources[idx] == "a").sum() == 3
    assert (sources[idx] == "b").sum() == 3


def test_select_pass_indices_deterministic():
    sources = np.array(["a"] * 100)
    labels = np.zeros(100, int)
    a = select_pass_indices(sources, labels, ["a"], max_per_source=10, seed=7)
    b = select_pass_indices(sources, labels, ["a"], max_per_source=10, seed=7)
    assert np.array_equal(a, b)


def test_select_pass_indices_drop_unlabeled_and_limit():
    sources = np.array(["a"] * 6)
    labels = np.array([-1, 0, 1, 2, 3, -1])
    idx = select_pass_indices(sources, labels, ["a"], drop_unlabeled=True, limit=3)
    assert len(idx) == 3
    assert (labels[idx] != -1).all()


def test_balanced_subset_caps_per_class_drops_unlabeled():
    labels = np.array([0] * 100 + [1] * 5 + [2] * 50 + [-1] * 9)
    s = balanced_subset(labels, 10, seed=0)
    import collections
    c = collections.Counter(labels[s].tolist())
    assert c[0] == 10 and c[1] == 5 and c[2] == 10 and c[-1] == 0


def test_column_filters():
    cols = ["global_idx", "source", "label", "has_nucleus",
            "nuc_area_um2", "nuc_contrast", "ctx_int_mean", "ctx_contrast"]
    assert nuc_feature_columns(cols) == ["nuc_area_um2", "nuc_contrast"]
    assert ctx_feature_columns(cols) == ["ctx_int_mean", "ctx_contrast"]
