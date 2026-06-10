"""Multi-axis ABSOLUTE quality model for the p64 QC re-score. Pure on a per-patch
metric dict (the re-score precomputes sat_penalty); no image I/O, so Pass 2 re-runs
without re-segmenting. Four axes in [0,1]: detection, focus, texture, brightness."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

SAT_LEVEL = 0.98  # fraction-of-uint16-max threshold for the saturation penalty (computed in re-score)


@dataclass(frozen=True)
class Calibration:
    brenner_lo: float; brenner_hi: float
    ent_lo: float; ent_hi: float
    cov_lo: float; cov_hi: float
    ir_lo: float; ir_hi: float


def _calib(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def calibrate(raw_table: pl.DataFrame, lo_q: float = 0.05, hi_q: float = 0.95) -> Calibration:
    """Robust p5/p95 of the soft-axis inputs pooled over ALL non-broken patches of ALL
    sources -> a single dataset-wide (absolute) Calibration."""
    nb = raw_table.filter(~pl.col("broken")) if "broken" in raw_table.columns else raw_table
    def lohi(col):
        v = nb[col].drop_nulls().to_numpy()
        if v.size == 0:
            return 0.0, 1.0
        return float(np.quantile(v, lo_q)), float(np.quantile(v, hi_q))
    bl, bh = lohi("brenner"); el, eh = lohi("glcm_entropy")
    cl, ch = lohi("interior_cov"); il, ih = lohi("intensity_ratio")
    return Calibration(bl, bh, el, eh, cl, ch, il, ih)


def axes(m: dict, cal: Calibration) -> dict:
    detection = float(np.clip(m["stardist_prob"], 0, 1)) * float(m["centeredness"]) * float(m["dominant_central"])
    focus = _calib(m["brenner"], cal.brenner_lo, cal.brenner_hi)
    texture = float(np.mean([_calib(m["glcm_entropy"], cal.ent_lo, cal.ent_hi),
                             _calib(m["interior_cov"], cal.cov_lo, cal.cov_hi),
                             1.0 - float(np.clip(m["glcm_asm"], 0, 1))]))
    brightness = _calib(m["intensity_ratio"], cal.ir_lo, cal.ir_hi) * (1.0 - float(m["sat_penalty"]))
    return {"detection": float(np.clip(detection, 0, 1)), "focus": focus,
            "texture": texture, "brightness": float(np.clip(brightness, 0, 1))}


def grade(quality_min: float, tau_hi: float = 0.60, tau_lo: float = 0.30) -> str:
    if quality_min >= tau_hi:
        return "Excellent"
    if quality_min >= tau_lo:
        return "Good"
    return "Weak-passing"
