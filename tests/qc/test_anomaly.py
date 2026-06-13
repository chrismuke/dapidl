import numpy as np

from dapidl.qc.anomaly import select_bank_indices


def test_select_bank_excludes_slide_and_broken_caps_per_class():
    rows = np.arange(12)
    slides = np.array(["A"] * 6 + ["B"] * 6)
    coarse = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 2, 2])  # classes 0,1,2
    broken = np.array([False, False, True, False, False, False,
                       False, False, False, False, False, False])
    grades = np.array(["Good"] * 12)
    rng = np.random.default_rng(0)

    idx = select_bank_indices(rows, slides, coarse, broken, grades,
                              exclude_slide="B", per_class_cap=2, rng=rng)

    chosen = set(idx.tolist())
    assert chosen.issubset(set(range(6)))          # only slide A
    assert 2 not in chosen                          # broken excluded
    cls0 = [i for i in idx if coarse[i] == 0]
    assert len(cls0) <= 2
