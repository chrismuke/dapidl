"""Tests for the Xenium source loader's polygon rasterization."""
import numpy as np
import polars as pl
from dapidl.seg_eval.source_masks import rasterize_polygons


def test_rasterize_two_squares():
    df = pl.DataFrame({
        "cell_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "px": [2, 6, 6, 2, 10, 14, 14, 10],
        "py": [2, 2, 6, 6, 10, 10, 14, 14],
    })
    mask = rasterize_polygons(df, id_col="cell_id", x_col="px", y_col="py",
                              bbox=(0, 0, 20, 20))
    assert mask.shape == (20, 20)
    labels = set(np.unique(mask)) - {0}
    assert len(labels) == 2
    assert mask[4, 4] != 0 and mask[12, 12] != 0
    assert mask[4, 4] != mask[12, 12]
