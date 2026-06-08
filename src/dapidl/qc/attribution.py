"""Custom Integrated Gradients + nucleus-mask attribution concentration (no captum)."""

from __future__ import annotations

import numpy as np
import torch


def integrated_gradients(
    model,
    x: torch.Tensor,
    target: int,
    baseline: torch.Tensor | None = None,
    steps: int = 50,
) -> torch.Tensor:
    """Riemann Integrated Gradients of ``model(x)[:, target]`` w.r.t. ``x``.

    Shape-agnostic for single-sample (batch=1) input — (1,D) linear and
    (1,1,H,W) image tensors alike; returns a tensor shaped like ``x``. The
    forward runs under ``model.eval()`` (restored afterwards) so dropout/
    batchnorm cannot make the attribution stochastic. Completeness holds for a linear
    model: ``attr.sum() == f(x) - f(baseline)``. The model may expand channels
    internally (DapiClassifier 1->3) — gradients flow back to the single input
    channel, so no channel reduction is needed.
    """
    if baseline is None:
        baseline = torch.zeros_like(x)
    view = [steps] + [1] * (x.dim() - 1)
    alphas = torch.linspace(0.0, 1.0, steps, device=x.device, dtype=x.dtype).view(*view)
    path = (baseline + alphas * (x - baseline)).detach().requires_grad_(True)
    was_training = model.training
    model.eval()  # deterministic attribution: no dropout / batchnorm drift
    try:
        out = model(path)
        score = out[:, target].sum()
        (grads,) = torch.autograd.grad(score, path)
    finally:
        model.train(was_training)
    avg_grad = grads.mean(dim=0, keepdim=True)
    return ((x - baseline) * avg_grad).detach()


def fraction_in_mask(attr, mask) -> float:
    """Share of total |attribution| that falls inside ``mask`` (∈ [0, 1])."""
    a = np.abs(np.asarray(attr, dtype=np.float64))
    m = np.asarray(mask, dtype=bool)
    total = a.sum()
    if total <= 0:
        return 0.0
    return float(a[m].sum() / total)


def attribution_concentration(
    attr, mask, area_fraction: float, eps: float = 1e-6
) -> float:
    """Headline (D) metric: ``fraction_in_mask / max(area_fraction, eps)``.

    ≈1 nucleus ignored (attribution spread evenly by area); ≫1 subnuclear-driven
    (concentrated inside the nucleus beyond its size); <1 context-driven.
    """
    return fraction_in_mask(attr, mask) / max(area_fraction, eps)
