"""Train/val/test index splitters over the spatial registry. Pure numpy: no torch,
no IO. label == -1 (unlabeled context cells) are excluded from train/val/test
(they remain in the graph as context only)."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def _ystripe_val(train_pool: np.ndarray, source: np.ndarray, coords: np.ndarray,
                 val_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Within each slide present in `train_pool`, move its top-`val_frac` y-stripe to
    val. Returns (train_idx, val_idx), spatially separating val from train per slide."""
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    pool_src = source[train_pool]
    for s in np.unique(pool_src):
        sl = train_pool[pool_src == s]
        ythr = np.quantile(coords[sl, 1], 1.0 - val_frac)
        is_val = coords[sl, 1] > ythr
        val_parts.append(sl[is_val])
        train_parts.append(sl[~is_val])
    train = np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int64)
    val = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.int64)
    return np.sort(train), np.sort(val)


class LOSOSplit:
    """Leave-one-slide-out. `folds()` yields (held_out_slide, train_idx, val_idx,
    test_idx) once per unique slide: test = all labeled cells of the held-out slide;
    val = top `val_frac` y-stripe of each TRAINING slide; train = the rest. -1 excluded."""

    def __init__(self, source: np.ndarray | list, coords: np.ndarray | list,
                 labels: np.ndarray | list, val_frac: float = 0.20) -> None:
        self.source = np.asarray(source)
        self.coords = np.asarray(coords, dtype=float)
        self.labels = np.asarray(labels)
        self.val_frac = val_frac

    def folds(self) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        labeled = np.where(self.labels != -1)[0]
        lab_src = self.source[labeled]
        for held in np.unique(lab_src):
            test_idx = np.sort(labeled[lab_src == held])
            train_pool = labeled[lab_src != held]
            train_idx, val_idx = _ystripe_val(train_pool, self.source, self.coords, self.val_frac)
            yield str(held), train_idx, val_idx, test_idx


class Stage2ProperSplit:
    """The original Stage-2-proper split: val = top `val_frac` y-stripe of `val_slide`;
    test = `test_slide`; train = rest of `val_slide` + all other non-test slides.
    -1 excluded. Single fold (one yield)."""

    def __init__(self, source: np.ndarray | list, coords: np.ndarray | list,
                 labels: np.ndarray | list, val_frac: float = 0.20,
                 val_slide: str = "xenium_rep1", test_slide: str = "xenium_rep2") -> None:
        self.source = np.asarray(source)
        self.coords = np.asarray(coords, dtype=float)
        self.labels = np.asarray(labels)
        self.val_frac = val_frac
        self.val_slide = val_slide
        self.test_slide = test_slide

    def folds(self) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        labeled = self.labels != -1
        vslide = (self.source == self.val_slide) & labeled
        ythr = np.quantile(self.coords[vslide, 1], 1.0 - self.val_frac)
        val_idx = np.where(vslide & (self.coords[:, 1] > ythr))[0]
        train_idx = np.where(
            (vslide & (self.coords[:, 1] <= ythr))
            | ((self.source != self.val_slide) & (self.source != self.test_slide) & labeled)
        )[0]
        test_idx = np.where((self.source == self.test_slide) & labeled)[0]
        yield self.test_slide, np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)
