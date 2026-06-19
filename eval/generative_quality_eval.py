"""
Generative-quality eval for the trained FactorGaussianDiffusion model:
does the reverse process generate returns whose distribution matches the
Stage-1 training data? Separate from the subspace-recovery eval.

Sampling is the full ancestral reverse process (FactorGaussianDiffusion.sample,
all `timesteps` steps — reported in the output). The period index i is supplied
per the empirical period distribution of the training set (balanced: equal
counts per period).

Reported, in order:
  1. Parity metrics (friend-comparable): ||mean(R_real)||2, ||mean(R_gen)||2,
     ||mean diff||2, ||Sigma_real||_F, ||Sigma_gen||_F.
  2. Eigenvalue spectra of pooled Sigma_real vs Sigma_gen (top 10). NOTE: the
     POOLED covariance mixes P different rank-k period planes, so its rank is
     min(P*k, d) = 36 here, not k — the rank-k claim holds per period.
  3. Per-period: principal angles between top-k eigenspaces of Sigma_gen and
     Sigma_real; top-k eigenvalue ratios gen/real; share of total variance in
     the top k vs the d-k tail (real is 1.0 by construction — leakage check).
No pass/fail threshold.
"""

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root, for diffusion_factor_model
sys.path.insert(0, _HERE)                    # sibling import w/o eval/__init__ (needs numba)

from factor_recovery_eval import load_trained, subspace_metrics  # noqa: E402


def cov(X):
    """(N, d) float64 -> (d, d) covariance."""
    Xc = X - X.mean(dim=0, keepdim=True)
    return Xc.T @ Xc / (len(X) - 1)


def top_eigs(S, n):
    """Largest-n eigenvalues (desc) and their eigenvectors of symmetric S."""
    w, v = torch.linalg.eigh(S)
    return w.flip(0)[:n], v.flip(1)[:, :n]


def main():
    parser = argparse.ArgumentParser(description="Generative-quality eval (distribution match)")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="checkpoint .pt file, or a model dir (latest epoch used)")
    parser.add_argument("--data_path", type=str,
                        default="simulation_experiment_data/factor_period_data.npz")
    parser.add_argument("--n_per_period", type=int, default=2000,
                        help="generated samples per period (2000 matches the training set)")
    parser.add_argument("--sample_batch", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ckpt_path = args.ckpt
    if os.path.isdir(ckpt_path):
        import glob
        ckpt_path = max(glob.glob(os.path.join(ckpt_path, "model-epoch-*.pt")),
                        key=lambda p: int(p.rsplit("-", 1)[1].split(".")[0]))
    print(f"Checkpoint: {ckpt_path}")

    z = np.load(args.data_path)
    P, k, d = int(z["periods"]), int(z["k"]), int(z["d"])
    R_real = torch.from_numpy(z["R"]).double()
    idx_real = torch.from_numpy(z["period_idx"])

    diffusion, net = load_trained(ckpt_path, z)

    # ---- generate via the full reverse process ----
    torch.manual_seed(args.seed)
    i_gen = torch.arange(P).repeat_interleave(args.n_per_period)
    print(f"Sampling: ancestral reverse diffusion, {diffusion.num_timesteps} steps "
          f"(full schedule), N_gen = {P} periods x {args.n_per_period} "
          f"(empirical period distribution)")
    chunks = []
    for s in range(0, len(i_gen), args.sample_batch):
        chunks.append(diffusion.sample(i_gen[s:s + args.sample_batch]))
        print(f"  generated {sum(len(c) for c in chunks)}/{len(i_gen)}")
    R_gen = torch.cat(chunks).double()

    # ---- 1. parity metrics ----
    S_real, S_gen = cov(R_real), cov(R_gen)
    mu_real, mu_gen = R_real.mean(0), R_gen.mean(0)
    print("\n" + "=" * 72)
    print("1. PARITY METRICS (full sample sets)")
    print("=" * 72)
    print(f"  ||mean(R_real)||_2        = {mu_real.norm().item():.6f}")
    print(f"  ||mean(R_gen)||_2         = {mu_gen.norm().item():.6f}")
    print(f"  ||mean diff||_2           = {(mu_real - mu_gen).norm().item():.6f}")
    print(f"  ||Sigma_real||_F          = {S_real.norm().item():.6f}")
    print(f"  ||Sigma_gen||_F           = {S_gen.norm().item():.6f}")
    print(f"  ||Sigma_real-Sigma_gen||_F = {(S_real - S_gen).norm().item():.6f}")

    # ---- 2. pooled spectra ----
    n_top = min(10, d)
    w_real, _ = top_eigs(S_real, d)
    w_gen, _ = top_eigs(S_gen, d)
    print("\n" + "=" * 72)
    print(f"2. EIGENVALUE SPECTRUM of pooled covariance (top {n_top})")
    print(f"   [pooled rank is min(P*k, d) = {min(P * k, d)}, NOT k={k};")
    print(f"    the rank-{k} structure holds per period — see section 3]")
    print("=" * 72)
    print(f"  {'j':>3} {'real':>10} {'gen':>10} {'gen/real':>9}")
    for j in range(n_top):
        print(f"  {j + 1:>3} {w_real[j].item():>10.4f} {w_gen[j].item():>10.4f} "
              f"{(w_gen[j] / w_real[j]).item():>9.4f}")
    pk = P * k
    print(f"  pooled variance share in top {pk}: "
          f"real = {(w_real[:pk].sum() / w_real.sum()).item():.6f}, "
          f"gen = {(w_gen[:pk].sum() / w_gen.sum()).item():.6f}")

    # ---- 3. per-period spectrum / subspace / rank ----
    print("\n" + "=" * 72)
    print(f"3. PER-PERIOD diagnostics (top-{k} eigenspace, rank-{k} ground truth)")
    print("=" * 72)
    print(f"  {'i':>3} {'rho':>9} {'d_proj':>8} {'lam_g/lam_r (j=1..3)':>22} "
          f"{'topk share real':>16} {'topk share gen':>15}")
    rhos, ratios, shares_gen = [], [], []
    for p in range(P):
        Sr, Sg = cov(R_real[idx_real == p]), cov(R_gen[i_gen == p])
        wr, vr = top_eigs(Sr, k)
        wg, vg = top_eigs(Sg, k)
        rho, d_proj, _ = subspace_metrics(vg, vr)
        ratio = (wg / wr)
        share_r = (wr.sum() / torch.linalg.eigvalsh(Sr).sum()).item()
        share_g = (wg.sum() / torch.linalg.eigvalsh(Sg).sum()).item()
        rhos.append(rho); ratios.append(ratio); shares_gen.append(share_g)
        print(f"  {p:>3} {rho:>9.6f} {d_proj:>8.4f} "
              f"{'  '.join(f'{x:.3f}' for x in ratio.tolist()):>22} "
              f"{share_r:>16.6f} {share_g:>15.6f}")

    mean_ratio = torch.stack(ratios).mean().item()
    mean_share = float(np.mean(shares_gen))
    print("\n" + "=" * 72)
    print("READ")
    print("=" * 72)
    print(f"  subspace: mean rho(top-{k} eigenspace gen vs real) = {np.mean(rhos):.6f}")
    print(f"  dispersion: generated top-{k} eigenvalues average "
          f"{100 * mean_ratio:.1f}% of real ({'under' if mean_ratio < 1 else 'over'}-dispersed)")
    print(f"  rank: generated variance share in top {k} directions = {mean_share:.4f} "
          f"(real = 1.0; tail leakage = {100 * (1 - mean_share):.2f}%)")


if __name__ == "__main__":
    main()
