"""GPU smoke for the NuSPIRe backbone — proves the real checkpoint integrates.

Downloads TongjiZhanglab/NuSPIRe, builds the production DapiClassifier with
backbone="nuspire", and verifies on GPU:
  1. forward [B,1,128,128] -> logits [B,C] with the expected 768-d feature path,
  2. features are deterministic across forwards (eval),
  3. a few optimizer steps on a trivially intensity-separable synthetic task drive
     the loss down (encoder + head fine-tune end-to-end),
  4. VRAM + per-forward latency.

Run: uv run python scripts/nuspire_smoke.py
Kept OUT of the unit suite (needs network + GPU). Not part of the gnp-v1 A/B.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812  (repo-wide convention)

# Allow `from scripts.<module>` when run as `python scripts/nuspire_smoke.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dapidl.models.nuspire import NUSPIRE_HIDDEN  # noqa: E402
from scripts.breast_pooled_train import DapiClassifier  # noqa: E402


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} | building DapiClassifier(backbone='nuspire') ...")
    t0 = time.time()
    model = DapiClassifier(num_classes=3, backbone="nuspire").to(dev)
    print(f"  built in {time.time()-t0:.1f}s | feature dim wired = {model.head.in_features} "
          f"(expected {NUSPIRE_HIDDEN})")
    assert model.head.in_features == NUSPIRE_HIDDEN
    assert model._expand3 is False, "nuspire must NOT 3-channel-expand"
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_train/1e6:.1f}M")

    # 1. forward shape
    x = torch.randn(8, 1, 128, 128, device=dev)
    logits = model(x)
    print(f"forward: {tuple(x.shape)} -> logits {tuple(logits.shape)}  finite={torch.isfinite(logits).all().item()}")
    assert logits.shape == (8, 3)

    # 2. feature determinism (eval)
    model.eval()
    with torch.no_grad():
        f1 = model.backbone(x)
        f2 = model.backbone(x)
    dmax = (f1 - f2).abs().max().item()
    print(f"feature determinism: max|Δ|={dmax:.2e}  shape={tuple(f1.shape)}")
    assert dmax < 1e-4, f"features not deterministic ({dmax})"

    # 3. can it learn a trivially intensity-separable task?
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.manual_seed(0)

    def batch(n=24):
        y = torch.randint(0, 3, (n,), device=dev)
        # class -> distinct mean intensity; the encoder+head should separate these
        base = torch.tensor([-1.0, 0.0, 1.0], device=dev)[y].view(n, 1, 1, 1)
        xb = base + 0.3 * torch.randn(n, 1, 128, 128, device=dev)
        return xb, y

    losses = []
    for step in range(30):
        xb, yb = batch()
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"fine-tune loss: {losses[0]:.3f} -> {losses[-1]:.3f} (ln3={torch.log(torch.tensor(3.0)).item():.3f})")
    assert losses[-1] < losses[0], "loss did not decrease"

    # 4. resources
    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            model(x)
    if dev == "cuda":
        torch.cuda.synchronize()
        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"VRAM peak: {vram:.2f} GB | latency: {(time.time()-t0)/10*1000:.0f} ms/forward (B=8)")
    print("\nNuSPIRe smoke: PASS")


if __name__ == "__main__":
    main()
