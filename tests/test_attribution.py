import numpy as np
import torch

from dapidl.qc.attribution import attribution_concentration, fraction_in_mask, integrated_gradients


def test_ig_completeness_on_linear_model():
    torch.manual_seed(0)
    model = torch.nn.Linear(10, 3)
    x = torch.randn(1, 10)
    baseline = torch.zeros(1, 10)
    attr = integrated_gradients(model, x, target=1, baseline=baseline, steps=64)
    delta = (model(x)[0, 1] - model(baseline)[0, 1]).item()
    assert abs(float(attr.sum()) - delta) < 1e-4
    assert attr.shape == x.shape


def test_ig_zero_when_x_equals_baseline():
    torch.manual_seed(1)
    model = torch.nn.Linear(6, 2)
    x = torch.randn(1, 6)
    attr = integrated_gradients(model, x, target=0, baseline=x.clone(), steps=16)
    assert abs(float(attr.sum())) < 1e-6


def test_fraction_in_mask_uses_absolute_value():
    attr = np.array([[1.0, -3.0], [0.0, 0.0]])
    mask = np.array([[True, False], [False, False]])
    assert abs(fraction_in_mask(attr, mask) - 0.25) < 1e-9  # |1| / (|1|+|-3|)


def test_fraction_in_mask_zero_attribution_is_zero():
    attr = np.zeros((2, 2))
    mask = np.ones((2, 2), bool)
    assert fraction_in_mask(attr, mask) == 0.0


def test_concentration_divides_fraction_by_area():
    attr = np.array([[2.0, 0.0], [0.0, 2.0]])
    mask = np.array([[True, False], [False, False]])  # fraction = 0.5
    assert abs(attribution_concentration(attr, mask, 0.25) - 2.0) < 1e-9


def test_concentration_eps_guard_is_finite():
    attr = np.array([[1.0, 0.0]])
    mask = np.array([[True, False]])
    assert np.isfinite(attribution_concentration(attr, mask, 0.0))
