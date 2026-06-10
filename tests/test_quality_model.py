# tests/test_quality_model.py
import numpy as np
import polars as pl
from dapidl.qc.quality_model import calibrate, axes, grade, Calibration


def _metric(brenner=1.0, glcm_entropy=1.0, interior_cov=1.0, glcm_asm=0.1,
            intensity_ratio=1.0, stardist_prob=0.9, centeredness=1.0,
            dominant_central=1.0, sat_penalty=0.0):
    return dict(brenner=brenner, glcm_entropy=glcm_entropy, interior_cov=interior_cov,
                glcm_asm=glcm_asm, intensity_ratio=intensity_ratio, stardist_prob=stardist_prob,
                centeredness=centeredness, dominant_central=dominant_central, sat_penalty=sat_penalty)


CAL = Calibration(brenner_lo=0.0, brenner_hi=2.0, ent_lo=0.0, ent_hi=2.0,
                  cov_lo=0.0, cov_hi=2.0, ir_lo=0.0, ir_hi=2.0)


def test_axes_in_unit_interval():
    a = axes(_metric(), CAL)
    for v in (a["detection"], a["focus"], a["texture"], a["brightness"]):
        assert 0.0 <= v <= 1.0


def test_focus_monotonic_in_brenner():
    assert axes(_metric(brenner=1.8), CAL)["focus"] > axes(_metric(brenner=0.2), CAL)["focus"]


def test_brightness_drops_with_saturation():
    hi = axes(_metric(intensity_ratio=1.5, sat_penalty=0.0), CAL)["brightness"]
    lo = axes(_metric(intensity_ratio=1.5, sat_penalty=0.5), CAL)["brightness"]
    assert lo < hi


def test_texture_rises_with_entropy_and_cov_falls_with_asm():
    base = axes(_metric(glcm_entropy=0.5, interior_cov=0.5, glcm_asm=0.5), CAL)["texture"]
    more = axes(_metric(glcm_entropy=1.8, interior_cov=1.8, glcm_asm=0.05), CAL)["texture"]
    assert more > base


def test_grade_thresholds():
    assert grade(0.7, tau_hi=0.6, tau_lo=0.3) == "Excellent"
    assert grade(0.4, tau_hi=0.6, tau_lo=0.3) == "Good"
    assert grade(0.1, tau_hi=0.6, tau_lo=0.3) == "Weak-passing"


def test_calibration_is_absolute_slide_independent():
    raw = pl.DataFrame({"broken": [False] * 6,
                        "brenner": [0, 1, 2, 3, 4, 5], "glcm_entropy": [0, 1, 2, 3, 4, 5],
                        "interior_cov": [0, 1, 2, 3, 4, 5], "intensity_ratio": [0, 1, 2, 3, 4, 5]})
    cal = calibrate(raw)
    m = _metric(brenner=2.0)
    assert axes(m, cal) == axes(m, cal)
    assert cal.brenner_hi > cal.brenner_lo
