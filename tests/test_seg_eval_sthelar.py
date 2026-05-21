"""Smoke test for the STHELAR sdata loader against real data."""
import numpy as np
from dapidl.seg_eval.source_masks import load_sthelar

ZARR = "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s0.zarr/sdata_breast_s0.zarr"


def test_sthelar_loader_smoke():
    s = load_sthelar(ZARR)
    dapi = s["dapi"]()
    assert dapi.ndim == 2 and dapi.size > 0
    assert s["centroids"].shape[1] == 2
    nuc = s["nucleus_polys"]()
    assert set(["px", "py", "cell_id"]).issubset(nuc.columns)
