"""
Stage 1 data generator for the NOTES §3.2 factor-recovery experiment.

Noiseless period-indexed returns R_i = U_i T F_i:
  - one frozen T in R^{m x k}, shared across all periods
  - per-period descriptor matrix U_i in R^{d x m} (full column rank, d >= m)
  - factors F ~ N(0, I_k) per observation (p_f fixed for this run)

No residual, no diffusion noise (OU noising happens at train time), and NO
standardization. Thin QR Q_i V_i = U_i is cached alongside the ground truth.
"""

import argparse
import os

import numpy as np
import torch

import config.config as config


def generate(d, m, k, periods, obs_per_period, seed):
    assert k <= m <= d, f"need k <= m <= d, got k={k}, m={m}, d={d}"
    seed = config.set_seed(seed)

    # float64 throughout generation so the sanity checks measure model
    # structure, not float32 rounding; training casts to float32 at load.
    dtype = torch.float64

    T = torch.randn(m, k, dtype=dtype)

    # Unit data scale by construction (standardization is forbidden):
    # Var(r_j | U, T) = sum_c (U_i T)_{jc}^2, so rescale U globally so the
    # mean of that quantity over assets and periods is exactly 1.
    U = torch.randn(periods, d, m, dtype=dtype)
    beta_raw = torch.einsum("pdm,mk->pdk", U, T)
    U = U / beta_raw.pow(2).sum(-1).mean().sqrt()
    Q, V = torch.linalg.qr(U, mode="reduced")  # Q: (P,d,m), V: (P,m,m)

    N = periods * obs_per_period
    F = torch.randn(N, k, dtype=dtype)
    period_idx = torch.arange(periods).repeat_interleave(obs_per_period)

    beta = torch.einsum("pdm,mk->pdk", U, T)  # beta_i = U_i T, (P,d,k)
    R = torch.einsum("ndk,nk->nd", beta[period_idx], F)

    return {
        "R": R, "period_idx": period_idx, "U": U, "Q": Q, "V": V,
        "T": T, "F": F, "beta": beta,
        "d": d, "m": m, "k": k, "periods": periods,
        "obs_per_period": obs_per_period, "seed": seed,
    }


def orth(M):
    """Orthonormal basis of col span of M (d x j), via reduced SVD."""
    Uo, _, _ = torch.linalg.svd(M, full_matrices=False)
    return Uo


def principal_angle_cosines(A, B):
    """Singular values of orth(A)^T orth(B) = cos of principal angles."""
    return torch.linalg.svdvals(orth(A).T @ orth(B))


def sanity_check(data):
    d, m, k, P = data["d"], data["m"], data["k"], data["periods"]
    n = data["obs_per_period"]
    R, U, Q, V, T = data["R"], data["U"], data["Q"], data["V"], data["T"]
    ok = True

    print("=" * 72)
    print(f"Stage-1 sanity check  (d={d}, m={m}, k={k}, P={P}, "
          f"obs/period={n}, seed={data['seed']})")
    print("=" * 72)

    # 1. shapes
    expected = {
        "R": (P * n, d), "period_idx": (P * n,), "U": (P, d, m),
        "Q": (P, d, m), "V": (P, m, m), "T": (m, k), "F": (P * n, k),
    }
    for name, shape in expected.items():
        actual = tuple(data[name].shape)
        status = "ok" if actual == shape else "MISMATCH"
        ok &= actual == shape
        print(f"  shape {name:<10} {str(actual):<16} expected {shape}  [{status}]")

    # QR consistency of the cached side tensors
    qr_err = (torch.einsum("pdm,pmn->pdn", Q, V) - U).abs().max().item()
    qtq_err = (torch.einsum("pdm,pdn->pmn", Q, Q)
               - torch.eye(m, dtype=Q.dtype)).abs().max().item()
    print(f"\n  max |Q V - U|       = {qr_err:.2e}")
    print(f"  max |Q^T Q - I_m|   = {qtq_err:.2e}")
    ok &= qr_err < 1e-10 and qtq_err < 1e-10

    # 2. rank(U_i) = m for every period
    ranks = torch.linalg.matrix_rank(U)
    print(f"\n  rank(U_i): {ranks.tolist()}  (need all = m = {m})")
    ok &= bool((ranks == m).all())

    # 3. factor structure stable across periods: span(R_i) = span(U_i T),
    #    exactly k-dimensional, for every period i
    print(f"\n  {'i':>3} {'s_k/s_1':>10} {'s_k+1/s_1':>10} "
          f"{'max angle vs span(U_i T) [deg]':>32}")
    for i in range(P):
        Ri = R[data["period_idx"] == i]                    # (n, d)
        s = torch.linalg.svdvals(Ri)
        rank_in, rank_out = (s[k - 1] / s[0]).item(), (s[k] / s[0]).item()
        _, _, Vh = torch.linalg.svd(Ri, full_matrices=False)
        cos = principal_angle_cosines(Vh[:k].T, data["beta"][i])
        max_angle = torch.rad2deg(torch.acos(cos.clamp(max=1.0)).max()).item()
        print(f"  {i:>3} {rank_in:>10.3e} {rank_out:>10.3e} {max_angle:>32.2e}")
        # 1e-4 deg ~ 2e-6 rad: float64 roundoff territory, far below anything structural
        ok &= rank_in > 1e-6 and rank_out < 1e-12 and max_angle < 1e-4

    # 4. per-asset variance ~ 1 (unit scale by construction, no standardizing)
    var = R.var(dim=0)
    print(f"\n  per-asset variance: mean={var.mean():.3f}  "
          f"min={var.min():.3f}  max={var.max():.3f}  (target ~1)")
    print(f"  E||r||^2 / d        = {(R.pow(2).sum(1).mean() / d).item():.3f}")
    ok &= 0.8 < var.mean().item() < 1.25

    print("\n  RESULT:", "ALL CHECKS PASSED" if ok else "*** CHECK FAILED ***")
    print("=" * 72)
    return ok


def save(data, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        R=data["R"].numpy().astype(np.float32),
        period_idx=data["period_idx"].numpy().astype(np.int64),
        U=data["U"].numpy(), Q=data["Q"].numpy(), V=data["V"].numpy(),
        T=data["T"].numpy(), F=data["F"].numpy(),
        d=data["d"], m=data["m"], k=data["k"], periods=data["periods"],
        obs_per_period=data["obs_per_period"], seed=data["seed"],
    )
    print(f"Saved dataset to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate noiseless period-indexed factor data (NOTES §3.2)")
    parser.add_argument("--d", type=int, default=64, help="number of assets")
    parser.add_argument("--m", type=int, default=8, help="number of descriptors")
    parser.add_argument("--k", type=int, default=3, help="number of factors")
    parser.add_argument("--periods", type=int, default=12, help="number of periods P")
    parser.add_argument("--obs_per_period", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str,
                        default="simulation_experiment_data/factor_period_data.npz")
    args = parser.parse_args()

    data = generate(args.d, args.m, args.k, args.periods,
                    args.obs_per_period, args.seed)
    passed = sanity_check(data)
    save(data, args.out)
    if not passed:
        raise SystemExit(1)
