"""Tests for GeneralizedCrossEntropy (review Phase 3): noise-robust loss for weak labels."""
import torch

from dapidl.training.losses import GeneralizedCrossEntropy


def test_gce_lower_loss_for_confident_correct_than_confident_wrong():
    gce = GeneralizedCrossEntropy(q=0.7)
    correct = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    wrong = torch.tensor([[0.0, 0.0, 0.0, 10.0]])
    y = torch.tensor([0])
    assert gce(correct, y).item() < gce(wrong, y).item()


def test_gce_bounded_for_confident_wrong_label():
    """The robustness property: a confidently-WRONG label has loss <= 1/q
    (bounded), whereas plain cross-entropy -> infinity. This is why GCE tolerates
    the noisy transcriptomic pseudo-labels."""
    q = 0.7
    gce = GeneralizedCrossEntropy(q=q)
    logits = torch.tensor([[0.0, 0.0, 0.0, 30.0]])   # p(true class 0) ~ 0
    y = torch.tensor([0])
    assert gce(logits, y).item() <= 1.0 / q + 1e-4


def test_gce_weight_scales_per_class():
    w = torch.tensor([1.0, 5.0, 1.0, 1.0])
    gce_w = GeneralizedCrossEntropy(q=0.7, weight=w)
    gce = GeneralizedCrossEntropy(q=0.7)
    logits = torch.tensor([[5.0, 0.0, 0.0, 0.0]])    # predicts 0
    y = torch.tensor([1])                            # true class 1 (the up-weighted one)
    assert abs(gce_w(logits, y).item() - 5.0 * gce(logits, y).item()) < 1e-4


def test_gce_ignore_index_excludes_samples():
    gce = GeneralizedCrossEntropy(q=0.7, ignore_index=-1)
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 10.0]])
    y = torch.tensor([0, -1])                        # second sample ignored
    gce_one = GeneralizedCrossEntropy(q=0.7)
    assert abs(gce(logits, y).item() - gce_one(logits[:1], y[:1]).item()) < 1e-5


def test_gce_all_ignored_returns_zero_finite():
    gce = GeneralizedCrossEntropy(q=0.7, ignore_index=-1)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y = torch.tensor([-1])
    out = gce(logits, y)
    assert torch.isfinite(out) and out.item() == 0.0


def test_gce_rejects_bad_q():
    import pytest
    with pytest.raises(ValueError):
        GeneralizedCrossEntropy(q=0.0)
    with pytest.raises(ValueError):
        GeneralizedCrossEntropy(q=1.5)
