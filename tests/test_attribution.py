import torch

from dapidl.qc.attribution import integrated_gradients


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
