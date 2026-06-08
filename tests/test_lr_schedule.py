"""Well-scheduled fine-tuning for ViT backbones (NuSPIRe).

The h2h showed NuSPIRe's val F1 oscillating — traced to CosineAnnealingWarmRestarts
re-spiking the LR to peak at epoch 5. A ViT-MAE fine-tune wants instead: a linear
warmup (so the random head doesn't shock the pretrained encoder) followed by a SINGLE
monotonic cosine decay to ~0, plus an optional frozen-backbone head-warmup.

These tests pin the pure schedule math and the freeze toggle.
"""
import math
import sys
from pathlib import Path

import torch.nn as nn

# scripts/ is not on pytest's pythonpath (which is ["src"]); add the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.breast_pooled_train import (  # noqa: E402
    set_backbone_trainable,
    warmup_cosine_lr,
)

PEAK = 1e-4


def test_warmup_ramps_linearly_to_peak():
    # warmup_steps=10, total=100: lr should climb (step+1)/warmup * peak.
    assert math.isclose(warmup_cosine_lr(0, 10, 100, PEAK), PEAK * 0.1, rel_tol=1e-9)
    assert math.isclose(warmup_cosine_lr(4, 10, 100, PEAK), PEAK * 0.5, rel_tol=1e-9)
    # last warmup step reaches peak: (9+1)/10 == 1.0
    assert math.isclose(warmup_cosine_lr(9, 10, 100, PEAK), PEAK, rel_tol=1e-9)


def test_cosine_continuous_at_warmup_boundary_and_zero_at_end():
    # step == warmup_steps -> progress 0 -> peak (continuous with the warmup ramp).
    assert math.isclose(warmup_cosine_lr(10, 10, 100, PEAK), PEAK, rel_tol=1e-9)
    # halfway through the cosine leg -> half of peak.
    assert math.isclose(warmup_cosine_lr(55, 10, 100, PEAK), PEAK * 0.5, rel_tol=1e-9)
    # final step -> min_lr (0).
    assert math.isclose(warmup_cosine_lr(100, 10, 100, PEAK), 0.0, abs_tol=1e-12)


def test_cosine_leg_is_monotonic_no_restart():
    # The whole point: no warm restart. LR must never increase after warmup.
    lrs = [warmup_cosine_lr(s, 10, 100, PEAK) for s in range(10, 101)]
    assert all(lrs[i] >= lrs[i + 1] - 1e-18 for i in range(len(lrs) - 1))


def test_min_lr_floor_is_respected():
    mn = 1e-6
    assert math.isclose(warmup_cosine_lr(100, 10, 100, PEAK, mn), mn, abs_tol=1e-12)
    assert all(warmup_cosine_lr(s, 10, 100, PEAK, mn) >= mn - 1e-18 for s in range(0, 130))


def test_no_warmup_starts_at_peak():
    assert math.isclose(warmup_cosine_lr(0, 0, 100, PEAK), PEAK, rel_tol=1e-9)


def test_clamps_beyond_total_steps():
    # Guards against off-by-one over-running the schedule.
    assert math.isclose(warmup_cosine_lr(150, 10, 100, PEAK), 0.0, abs_tol=1e-12)


class _TwoPartNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)


def test_set_backbone_trainable_toggles_only_the_backbone():
    m = _TwoPartNet()
    set_backbone_trainable(m, False)
    assert all(not p.requires_grad for p in m.backbone.parameters())
    assert all(p.requires_grad for p in m.head.parameters())  # head untouched
    set_backbone_trainable(m, True)
    assert all(p.requires_grad for p in m.backbone.parameters())
