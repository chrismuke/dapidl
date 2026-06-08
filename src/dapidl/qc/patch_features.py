"""Pure per-nucleus + per-patch DAPI feature vectors for the subnuclear-structure
triangulation (no I/O, no GPU). Reuses starpose.qc scorers; adds Haralick texture."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
from starpose.qc.segmentation_grounded import (
    SegQCConfig,
    area_um2,
    brenner_focus,
    interior_cov,
    structure_raw,
)

_SMOOTH = {"contrast": 0.0, "homogeneity": 1.0, "energy": 1.0,
           "correlation": 1.0, "asm": 1.0, "entropy": 0.0}

_GEOM = ("area_um2", "eccentricity", "solidity", "extent", "major_axis", "minor_axis")
_INT = ("int_mean", "int_std", "int_p10", "int_p50", "int_p90", "int_above_bg")
_TEX = ("structure_raw", "brenner", "interior_cov",
        "contrast", "homogeneity", "energy", "correlation", "asm", "entropy")

NUC_COLUMNS = [f"nuc_{k}" for k in (_GEOM + _INT + _TEX)] + ["nuc_stardist_prob", "nuc_area_fraction"]
CTX_COLUMNS = [f"ctx_{k}" for k in (_INT + _TEX)]


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


def _intensity(patch: np.ndarray, region: np.ndarray, bg: np.ndarray) -> dict[str, float]:
    v = patch[region].astype(np.float64)
    bgv = patch[bg].astype(np.float64)
    bg_med = float(np.median(bgv)) if bgv.size else 0.0
    if v.size == 0:
        return {k: np.nan for k in _INT}  # noqa: C420
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    return {"int_mean": float(v.mean()), "int_std": float(v.std()),
            "int_p10": float(p10), "int_p50": float(p50), "int_p90": float(p90),
            "int_above_bg": float(v.mean() - bg_med)}


def _texture(patch: np.ndarray, region: np.ndarray, interior: np.ndarray,
             cfg: SegQCConfig) -> dict[str, float]:
    out = {"structure_raw": structure_raw(patch, region, cfg),
           "brenner": brenner_focus(patch, interior),
           "interior_cov": interior_cov(patch, interior)}
    out.update(haralick_features(patch, interior))
    return out


def _scope(patch: np.ndarray, region: np.ndarray, cfg: SegQCConfig,
           pixel_size: float, geom: bool) -> dict[str, float]:
    region = np.asarray(region, dtype=bool)
    out: dict[str, float] = {}
    if geom:
        if region.any():
            pr = regionprops(region.astype(np.int32))[0]
            out["area_um2"] = area_um2(region, pixel_size)
            out["eccentricity"] = float(pr.eccentricity)
            out["solidity"] = float(pr.solidity)
            out["extent"] = float(pr.extent)
            out["major_axis"] = float(pr.axis_major_length)
            out["minor_axis"] = float(pr.axis_minor_length)
        else:
            out.update({k: np.nan for k in _GEOM})  # noqa: C420
        interior = ndimage.binary_erosion(region, iterations=cfg.erode_px)
        bg = ~region
    else:
        interior = region
        bg = region  # ctx "background" = whole patch -> above_bg becomes mean - global median
    out.update(_intensity(patch, region, bg))
    out.update(_texture(patch, region, interior, cfg))
    return out


def nucleus_feature_vector(patch: np.ndarray, mask, prob: float,
                           cfg: SegQCConfig, pixel_size: float) -> dict[str, float]:
    """Two-scope feature row. ``nuc_*`` from the center-nucleus mask (or NaN when
    ``mask is None``); ``ctx_*`` over the whole patch (always computed). Always
    returns the same key set so the parquet schema is stable."""
    patch = np.asarray(patch)
    feats: dict[str, float] = {"has_nucleus": 1.0 if mask is not None else 0.0}
    if mask is None:
        feats.update({c: np.nan for c in NUC_COLUMNS})  # noqa: C420
    else:
        mask = np.asarray(mask, dtype=bool)
        feats.update({f"nuc_{k}": v for k, v in
                      _scope(patch, mask, cfg, pixel_size, geom=True).items()})
        feats["nuc_stardist_prob"] = float(prob)
        feats["nuc_area_fraction"] = float(mask.sum()) / float(mask.size)
    whole = np.ones(patch.shape, dtype=bool)
    feats.update({f"ctx_{k}": v for k, v in
                  _scope(patch, whole, cfg, pixel_size, geom=False).items()})
    return feats
