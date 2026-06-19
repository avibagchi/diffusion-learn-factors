"""
Consolidated real-data evaluation table for the §3.2 factor model.

The synthetic table (diffusion_table_eval.py) scores against ground-truth T and
C_i = (U_i T)(U_i T)^T, which real data lacks. This is the no-ground-truth
analog: same three-block layout, with the T-based blocks replaced by their
empirical counterparts.

  1. Moment Matching          pooled R_real vs R_gen (means, covariance norms).
  2. Loading Recovery         learned span(U_i A) vs each period's top-k return
                              PCs, bracketed by a descriptor-span ceiling
                              rho(col(U_i),PC) and a random-loading floor.
                              [VALID without a residual: measures the factor
                               subspace, exactly §3.2's question.]
  3. Generative Cov Match     per-period top-k eigenspace of R_gen vs R_real,
                              eigenvalue dispersion, and top-k variance share.
                              [DIAGNOSTIC only: the rank-k generator cannot match
                               real returns' idiosyncratic tail, so a low score
                               partly reflects the noiseless §3.2 simplification,
                               not a training failure.]

Generation uses the proven period-conditioned reverse process (sample_periods).
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from factor_recovery_eval import load_trained, orth          # noqa: E402
from diffusion_table_eval import sample_periods, _attach_q_posterior  # noqa: E402


def cov(X):
    Xc = X - X.mean(0, keepdim=True)
    return Xc.T @ Xc / (len(X) - 1)


def top_eigs(S, n):
    w, V = torch.linalg.eigh(S)
    return w.flip(0)[:n], V.flip(1)[:, :n]


def alignment(B, PC):
    """rho = mean cos^2(principal angle) between col spans, normalized by k=PC cols."""
    k = PC.shape[1]
    s = torch.linalg.svdvals(orth(B).T @ orth(PC))
    return (s.pow(2).sum() / k).item()


def subspace_rho(A_, B_):
    k = A_.shape[1]
    s = torch.linalg.svdvals(orth(A_).T @ orth(B_))
    return (s.pow(2).sum() / k).item()


def fmt(label, value, width=44):
    return f"  {label:<{width}}{value:>14.6f}"


def main():
    p = argparse.ArgumentParser(description="Consolidated real-data §3.2 evaluation table")
    p.add_argument("--ckpt", type=str, required=True,
                   help="checkpoint .pt or model dir (latest epoch used)")
    p.add_argument("--data_path", type=str,
                   default="empirical_analysis_data/real_factor_period_data.npz")
    p.add_argument("--n_per_period", type=int, default=250,
                   help="generated samples per period (~matches daily obs/year)")
    p.add_argument("--sample_batch", type=int, default=2500)
    p.add_argument("--n_rand", type=int, default=200, help="random-A draws for the floor")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ckpt_path = args.ckpt
    if os.path.isdir(ckpt_path):
        ckpt_path = max(glob.glob(os.path.join(ckpt_path, "model-epoch-*.pt")),
                        key=lambda q: int(q.rsplit("-", 1)[1].split(".")[0]))
    print(f"Checkpoint: {ckpt_path}")

    z = np.load(args.data_path)
    P, k, d, m = int(z["periods"]), int(z["k"]), int(z["d"]), int(z["m"])
    R_real = torch.from_numpy(z["R"]).double()
    idx_real = torch.from_numpy(z["period_idx"])
    U = torch.from_numpy(z["U"]).double()

    diffusion, net = load_trained(ckpt_path, z)
    _attach_q_posterior(diffusion)
    A = net.A.detach().double()

    tickers = list(z["tickers"]) if "tickers" in z.files else None
    print(f"Universe: P={P} periods, d={d} firms, m={m} descriptors, k={k} factors")
    if tickers is not None:
        print("Firms:", " ".join(str(t) for t in tickers))

    # ---- generate ----
    i_gen = torch.arange(P).repeat_interleave(args.n_per_period)
    print(f"Sampling: period-conditioned ancestral reverse diffusion, "
          f"{diffusion.num_timesteps} steps, N_gen={len(i_gen)} "
          f"({P}x{args.n_per_period})")
    chunks = []
    for s in range(0, len(i_gen), args.sample_batch):
        chunks.append(sample_periods(diffusion, net, i_gen[s:s + args.sample_batch],
                                     seed=args.seed + s))
    R_gen = torch.cat(chunks).double()

    bar = "=" * 60

    # ====================== 1. moment matching ====================== #
    mu_r, mu_g = R_real.mean(0), R_gen.mean(0)
    C_r, C_g = cov(R_real), cov(R_gen)
    cov_diff = (C_r - C_g).norm().item()
    nr, ng = C_r.norm().item(), C_g.norm().item()
    std_r = torch.sqrt(torch.diag(C_r).clamp(min=0))
    std_g = torch.sqrt(torch.diag(C_g).clamp(min=0))
    rows1 = [
        ("Mean Difference ||mu_r - mu_g||_2", (mu_r - mu_g).norm().item()),
        ("Real Mean Norm ||mu_r||_2", mu_r.norm().item()),
        ("Generated Mean Norm ||mu_g||_2", mu_g.norm().item()),
        ("Real Covariance Norm ||C_r||_F", nr),
        ("Generated Covariance Norm ||C_g||_F", ng),
        ("Covariance Difference ||C_r - C_g||_F", cov_diff),
        ("Covariance Ratio kappa_cov", ng / nr),
        ("Relative Covariance Error eps_cov", cov_diff / nr),
        ("Std Deviation Ratio kappa_std", (std_g.norm() / std_r.norm()).item()),
    ]

    # ====================== 2. loading recovery ===================== #
    g = torch.Generator().manual_seed(args.seed)
    A_rand = [torch.linalg.qr(torch.randn(m, k, generator=g, dtype=torch.double))[0]
              for _ in range(args.n_rand)]
    learned_l, ceil_l, floor_l, frac_l = [], [], [], []
    for pi in range(P):
        PC = top_eigs(cov(R_real[idx_real == pi]), k)[1]
        Up = U[pi]
        learned = alignment(Up @ A, PC)
        ceiling = alignment(Up, PC)
        floor = float(np.mean([alignment(Up @ Ar, PC) for Ar in A_rand]))
        denom = ceiling - floor
        learned_l.append(learned); ceil_l.append(ceiling); floor_l.append(floor)
        frac_l.append((learned - floor) / denom if denom > 1e-6 else float("nan"))
    rows2 = [
        ("Learned   rho(span(U_i A), PC_k)", float(np.mean(learned_l))),
        ("Ceiling   rho(col(U_i),   PC_k)", float(np.mean(ceil_l))),
        ("Floor     rho(span(U_i A_rand))", float(np.mean(floor_l))),
        ("Recovery Fraction (learned-floor)/(ceil-floor)", float(np.nanmean(frac_l))),
    ]

    # ==================== 3. generative cov match =================== #
    sub, disp, share_r_l, share_g_l = [], [], [], []
    for pi in range(P):
        Sr, Sg = cov(R_real[idx_real == pi]), cov(R_gen[i_gen == pi])
        wr, vr = top_eigs(Sr, k)
        wg, vg = top_eigs(Sg, k)
        sub.append(subspace_rho(vg, vr))
        disp.append((wg.sum() / wr.sum()).item())
        share_r_l.append((wr.sum() / torch.linalg.eigvalsh(Sr).sum()).item())
        share_g_l.append((wg.sum() / torch.linalg.eigvalsh(Sg).sum()).item())
    rows3 = [
        ("Subspace rho (gen top-k vs real top-k)", float(np.mean(sub))),
        ("Eigenvalue Dispersion (gen/real top-k)", float(np.mean(disp))),
        ("Top-k Variance Share (real)", float(np.mean(share_r_l))),
        ("Top-k Variance Share (gen)", float(np.mean(share_g_l))),
    ]

    # -------------------------------- print -------------------------------- #
    print("\n" + bar + "\n1. Moment Matching (pooled R_real vs R_gen)\n" + bar)
    for lab, v in rows1:
        print(fmt(lab, v))
    print("\n" + bar + "\n2. Loading Recovery  [valid w/o residual: factor subspace]\n" + bar)
    print("   rho = mean cos^2(principal angle) vs top-k return PCs, in [0,1]")
    for lab, v in rows2:
        print(fmt(lab, v))
    print("\n" + bar + "\n3. Generative Cov Match  [diagnostic: rank-k cannot match tail]\n" + bar)
    for lab, v in rows3:
        print(fmt(lab, v))
    print("\n   Note: gen top-k share = 1.0 by construction (rank-k generator);")
    print("   real < 1.0 is the idiosyncratic variance the noiseless model omits.")


if __name__ == "__main__":
    main()
