"""
Stage-2 smoke test: FactorScoreNet + FactorGaussianDiffusion wiring.

Checks, on a real batch from the Stage-1 dataset:
  1. forward(r, i) returns a finite scalar loss
  2. backward populates grads on A, the Network trunk, and the period embedding
  3. U/Q buffers stay frozen (no grads)
  4. model_predictions is self-consistent: q_sample(pred_x_start) with the
     predicted noise reconstructs x_t (the pred_x0 → noise conversion)
  5. the untouched image path (Unet + GaussianDiffusion) still runs after the
     Trainer batch-unpack edit, including one Trainer-style `model(*data)` call
"""

import numpy as np
import torch

from diffusion_factor_model import (
    FactorScoreNet, FactorGaussianDiffusion, Unet, GaussianDiffusion,
)

DATA = "simulation_experiment_data/factor_period_data.npz"


def check(name, cond):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}")
    return bool(cond)


def main():
    torch.manual_seed(0)
    z = np.load(DATA)
    R = torch.from_numpy(z["R"])                      # (N, d) float32
    idx = torch.from_numpy(z["period_idx"])           # (N,)
    U, Q = torch.from_numpy(z["U"]), torch.from_numpy(z["Q"])
    k, d = int(z["k"]), int(z["d"])

    net = FactorScoreNet(U, Q, k=k)
    diffusion = FactorGaussianDiffusion(net, d=d, timesteps=200, beta_schedule="cosine")

    sel = torch.randperm(len(R))[:32]
    r, i = R[sel], idx[sel]

    ok = True
    print("FactorScoreNet / FactorGaussianDiffusion:")
    loss = diffusion(r, i)
    ok &= check(f"loss finite scalar (loss={loss.item():.4f})", loss.ndim == 0 and torch.isfinite(loss))
    loss.backward()
    ok &= check("A.grad nonzero", net.A.grad is not None and net.A.grad.abs().max() > 0)
    ok &= check("trunk grads flow", net.in_proj.weight.grad.abs().max() > 0
                and net.out_proj.weight.grad.abs().max() > 0)
    if net.period_emb is not None:
        ok &= check("period embedding grads flow", net.period_emb.weight.grad.abs().max() > 0)
    else:
        ok &= check("no period embedding (default)", net.period_emb is None)
    ok &= check("U/Q are buffers (not trained)", not net.U.requires_grad and not net.Q.requires_grad)

    x0_hat = net(r, torch.full((32,), 100), i)
    ok &= check(f"x̂₀ shape {tuple(x0_hat.shape)} == (32, {d})", x0_hat.shape == (32, d))

    # pred_x0 → noise conversion consistency at a mid-trajectory t
    t = torch.full((32,), 100, dtype=torch.long)
    x_t = diffusion.q_sample(r, t, torch.randn_like(r))
    pred = diffusion.model_predictions(x_t, t, i)
    recon = diffusion.q_sample(pred.pred_x_start, t, pred.pred_noise)
    err = (recon - x_t).abs().max().item()
    ok &= check(f"q_sample(x̂₀, t, ε̂) reconstructs x_t (max err {err:.2e})", err < 1e-4)

    print("Image-path regression (Unet + GaussianDiffusion):")
    unet = Unet(dim=16, channels=1, dim_mults=(1, 2))
    img_diff = GaussianDiffusion(unet, image_size=(8, 8), timesteps=10,
                                 objective="pred_noise", auto_normalize=False)
    data = [torch.randn(2, 1, 8, 8)]          # Trainer now calls model(*data)
    img_loss = img_diff(*data)
    ok &= check(f"image loss finite (loss={img_loss.item():.4f})", torch.isfinite(img_loss))

    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "*** CHECK FAILED ***")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
