"""GPU smoke for the NuClass two-stream backbone — proves real integration.

Builds the production DapiClassifier(backbone="nuclass") (downloads NuSPIRe for
the nucleus stream, builds the microscopy CNN for context), and verifies on GPU:
  1. forward [B,1,128,128] -> logits [B,C] via the 512-d gated-fusion feature,
  2. features deterministic across forwards (eval),
  3. the gate is functional (in (0,1), varies across samples) -- the interpretable
     nucleus-vs-context routing signal,
  4. a few optimizer steps on an intensity-separable task drive the loss down
     (both streams + fusion + head train end-to-end),
  5. VRAM + per-forward latency.

Run: uv run python scripts/nuclass_smoke.py
Out of the unit suite (needs network + GPU). Dual-scale v1; NOT in the gnp-v1 A/B.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812  (repo-wide convention)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dapidl.models.nuclass import NUCLASS_FUSION_DIM  # noqa: E402
from scripts.breast_pooled_train import DapiClassifier  # noqa: E402


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} | building DapiClassifier(backbone='nuclass') ...")
    t0 = time.time()
    model = DapiClassifier(num_classes=3, backbone="nuclass").to(dev)
    print(f"  built in {time.time()-t0:.1f}s | feature dim = {model.head.in_features} "
          f"(expected {NUCLASS_FUSION_DIM})")
    assert model.head.in_features == NUCLASS_FUSION_DIM
    assert model._expand3 is False, "nuclass must NOT 3-channel-expand"
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_train/1e6:.1f}M | nucleus_crop={model.backbone.nucleus_crop}")

    x = torch.randn(8, 1, 128, 128, device=dev)
    logits = model(x)
    print(f"forward: {tuple(x.shape)} -> logits {tuple(logits.shape)}  finite={torch.isfinite(logits).all().item()}")
    assert logits.shape == (8, 3)

    model.eval()
    with torch.no_grad():
        f1 = model.backbone(x)
        f2 = model.backbone(x)
        g = model.backbone.gate_values(x)
    dmax = (f1 - f2).abs().max().item()
    print(f"feature determinism: max|Δ|={dmax:.2e}  fused shape={tuple(f1.shape)}")
    assert dmax < 1e-4
    print(f"gate: shape={tuple(g.shape)} mean={g.mean().item():.3f} "
          f"per-sample-mean std={g.mean(1).std().item():.3f} "
          f"range=[{g.min().item():.3f},{g.max().item():.3f}]")
    assert (g >= 0).all() and (g <= 1).all()

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.manual_seed(0)

    def batch(n=24):
        y = torch.randint(0, 3, (n,), device=dev)
        base = torch.tensor([-1.0, 0.0, 1.0], device=dev)[y].view(n, 1, 1, 1)
        xb = base + 0.3 * torch.randn(n, 1, 128, 128, device=dev)
        return xb, y

    losses = []
    for _ in range(30):
        xb, yb = batch()
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"fine-tune loss: {losses[0]:.3f} -> {losses[-1]:.3f}")
    assert losses[-1] < losses[0]

    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            model(x)
    if dev == "cuda":
        torch.cuda.synchronize()
        print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB | "
              f"latency: {(time.time()-t0)/10*1000:.0f} ms/forward (B=8)")
    print("\nNuClass smoke: PASS")


if __name__ == "__main__":
    main()
