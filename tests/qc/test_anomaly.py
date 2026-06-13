import numpy as np

from dapidl.qc.anomaly import select_bank_indices, knn_anomaly_score, coreset_subsample


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


def test_knn_anomaly_score_planted_outlier_scores_highest():
    rng = np.random.default_rng(0)
    bank = rng.normal(0, 0.01, size=(50, 8)) + np.array([1.0] + [0] * 7)
    query = np.vstack([bank[:3], np.array([[-1.0] + [0] * 7])])  # last row is far
    s = knn_anomaly_score(query, bank, k=5)
    assert s.shape == (4,)
    assert s[-1] == s.max()                 # planted outlier is most anomalous
    assert s[0] < s[-1]                      # an in-bank-like point is less anomalous


def test_knn_anomaly_score_guards_k_gt_bank():
    bank = np.eye(3)
    s = knn_anomaly_score(bank, bank, k=10)  # k > bank size, clipped to 3
    assert s.shape == (3,)
    assert (s > 0).all()  # all queries are in-bank, so not identical outliers


def test_coreset_subsample_is_deterministic():
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    emb = np.random.default_rng(1).normal(size=(20, 4))
    a = coreset_subsample(emb, frac=0.5, rng=rng1)
    b = coreset_subsample(emb, frac=0.5, rng=rng2)
    assert np.array_equal(a, b)
    assert len(a) == 10
