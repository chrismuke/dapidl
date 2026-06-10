import numpy as np

from dapidl.graph.splits import LOSOSplit, Stage2ProperSplit


def _toy():
    # 3 slides, 4 cells each; y in [0,1]; one -1 (unlabeled) cell per slide
    src = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    coords = np.tile(np.array([[0, 0.1], [0, 0.4], [0, 0.7], [0, 0.95]], dtype=float), (3, 1))
    labels = np.array([0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1])
    return src, coords, labels


def test_loso_one_fold_per_slide_test_is_single_slide():
    src, coords, labels = _toy()
    folds = list(LOSOSplit(src, coords, labels, val_frac=0.20).folds())
    assert len(folds) == 3
    for name, _tr, _va, te in folds:
        assert set(src[te]) == {name}                       # test = exactly the held-out slide
        assert np.all(labels[te] != -1)                     # no unlabeled in test


def test_loso_sets_disjoint_and_exclude_unlabeled():
    src, coords, labels = _toy()
    for name, tr, va, te in LOSOSplit(src, coords, labels).folds():
        assert len(np.intersect1d(tr, va)) == 0
        assert len(np.intersect1d(tr, te)) == 0
        assert len(np.intersect1d(va, te)) == 0
        for s in (tr, va, te):
            assert np.all(labels[s] != -1)                  # -1 excluded everywhere
        assert name not in set(src[va])                     # val drawn from TRAINING slides only


def test_stage2proper_matches_inline_split():
    src = np.array(["xenium_rep1"] * 5 + ["xenium_rep2"] * 3 + ["sthelar_breast_s0"] * 2)
    coords = np.zeros((10, 2))
    coords[:5, 1] = [0.1, 0.2, 0.5, 0.85, 0.95]  # rep1 y
    labels = np.array([0, 1, 2, 3, 0, 0, 1, 2, 0, 1])
    (name, tr, va, te), = list(Stage2ProperSplit(src, coords, labels, val_frac=0.20).folds())
    assert name == "xenium_rep2"
    assert set(src[te]) == {"xenium_rep2"}
    assert set(src[va]) == {"xenium_rep1"}                  # val is a rep1 stripe
    assert "sthelar_breast_s0" in set(src[tr])              # other slides are train
