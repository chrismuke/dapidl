"""Smoke test for the per-FOV diagnostic step using synthetic inputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np


class _Res:
    def __init__(self, nuc, cell, ncen):
        self.nucleus_masks = nuc; self.cell_masks = cell
        self.nucleus_centroids = ncen


def test_diagnose_fov_rows(monkeypatch):
    import seg_diagnostic as sd
    dapi = (np.random.default_rng(0).integers(0, 4000, (256, 256))).astype(np.uint16)
    src_nuc = np.zeros((256, 256), np.int32); src_nuc[10:30, 10:30] = 1
    src_cell = src_nuc.copy()
    src_cen = np.array([[20, 20]], float)
    sp = _Res(src_nuc.copy(), src_cell.copy(), src_cen.copy())
    monkeypatch.setattr(sd, "_segment_fov", lambda *a, **k: sp)
    row = sd.diagnose_fov("xenium_rep1", "dense", dapi, src_nuc, src_cell,
                          src_cen, transcripts=None, pixel_size=0.2125)
    assert row["source"] == "xenium_rep1" and row["fov"] == "dense"
    assert "nuc_f1" in row and "qc_own_mean" in row and "qc_src_mean" in row
