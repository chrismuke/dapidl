"""Pure per-nucleus + per-patch DAPI feature vectors for the subnuclear-structure
triangulation (no I/O, no GPU). Reuses starpose.qc scorers; adds Haralick texture."""
from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops

_SMOOTH = {"contrast": 0.0, "homogeneity": 1.0, "energy": 1.0,
           "correlation": 1.0, "asm": 1.0, "entropy": 0.0}


def haralick_features(patch: np.ndarray, interior_mask: np.ndarray,
                      levels: int = 16) -> dict[str, float]:
    """Grey-level co-occurrence texture inside ``interior_mask``.

    Quantizes over the patch's [1, 99] percentile range (brightness-robust),
    maps masked-out pixels to level 0 and drops that row/col of the GLCM — the
    starpose ``glcm_texture`` technique — then averages props over {0, 90 deg}.
    Returns contrast, homogeneity, energy, correlation, ASM, entropy. A
    degenerate region (<8 px or flat dynamic range) -> texture-less defaults.
    """
    interior_mask = np.asarray(interior_mask, dtype=bool)
    if int(interior_mask.sum()) < 8:
        return dict(_SMOOTH)
    p = patch.astype(np.float64)
    lo, hi = np.percentile(p, [1.0, 99.0])
    if hi <= lo:
        return dict(_SMOOTH)
    q = np.clip((p - lo) / (hi - lo), 0.0, 1.0)
    qg = (q * (levels - 1)).astype(np.uint8) + 1          # interior -> levels 1..levels
    qg[~interior_mask] = 0                                 # masked-out -> level 0
    glcm = graycomatrix(qg, distances=[1], angles=[0.0, np.pi / 2.0],
                        levels=levels + 1, symmetric=True)
    glcm = glcm[1:, 1:, :, :].astype(np.float64)           # drop the masked level-0 row/col
    tot = glcm.sum(axis=(0, 1), keepdims=True)
    tot[tot == 0] = 1.0
    pr = glcm / tot
    return {
        "contrast": float(graycoprops(glcm, "contrast").mean()),
        "homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "energy": float(graycoprops(glcm, "energy").mean()),
        "correlation": float(np.nan_to_num(graycoprops(glcm, "correlation")).mean()),
        "asm": float(graycoprops(glcm, "ASM").mean()),
        "entropy": float((-pr * np.log2(pr + 1e-12)).sum(axis=(0, 1)).mean()),
    }
