"""
Real-data loading-recovery eval for the §3.2 factor model (no ground truth).

On synthetic data we score the learned loading A against the true T
(factor_recovery_eval.py). Real data has no T, so the question becomes:

    does the learned per-period loading span(U_i A) align with the directions
    along which returns actually co-move -- the top-k principal components of
    that period's sample covariance?

We report three numbers per period so the learned value is interpretable:

  learned   rho( span(U_i A_hat) , PC_k )   -- what the trained model achieves
  ceiling   rho( col(U_i)        , PC_k )   -- best any k-plane *inside the
                                               descriptor span* could do; the
                                               descriptors' intrinsic ceiling
  floor     rho( span(U_i A_rand), PC_k )   -- random loadings, averaged over
                                               draws; the chance baseline

rho = (1/k) sum cos^2(principal angle) in [0, 1]; 1 = perfect alignment.
A learned value near the ceiling means A extracted essentially all the
return-relevant signal the 5 descriptors contain; near the floor means the
descriptors (or the training) carry no covariance information.

The gap (ceiling - floor) is how much signal is *available* to recover; where
the learned value sits in that gap is the recovery fraction.
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

from factor_recovery_eval import load_trained, orth  # noqa: E402


def cov(X):
    Xc = X - X.mean(0, keepdim=True)
    return Xc.T @ Xc / (len(X) - 1)


def top_pcs(S, k):
    """Top-k eigenvectors (desc) of symmetric S -> (d, k)."""
    w, V = torch.linalg.eigh(S)
    return V.flip(1)[:, :k]


def alignment(B, PC):
    """rho = mean cos^2(principal angle) between col spans of B and PC.

    Handles unequal column counts (e.g. col(U_i) has m cols, PC_k has k):
    the min(cols) principal cosines are used, normalized by k = PC.shape[1]."""
    k = PC.shape[1]
    s = torch.linalg.svdvals(orth(B).T @ orth(PC))   # min(cols_B, k) cosines
    return (s.pow(2).sum() / k).item()


def main():
    p = argparse.ArgumentParser(description="Real-data loading recovery: span(U_i A) vs return PCs")
    p.add_argument("--ckpt", type=str, required=True,
                   help="checkpoint .pt or model dir (latest epoch used)")
    p.add_argument("--data_path", type=str,
                   default="empirical_analysis_data/real_factor_period_data.npz")
    p.add_argument("--n_rand", type=int, default=200,
                   help="random-A draws for the chance floor")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ckpt_path = args.ckpt
    if os.path.isdir(ckpt_path):
        ckpt_path = max(glob.glob(os.path.join(ckpt_path, "model-epoch-*.pt")),
                        key=lambda q: int(q.rsplit("-", 1)[1].split(".")[0]))
    print(f"Checkpoint: {ckpt_path}")

    z = np.load(args.data_path)
    P, k, d, m = int(z["periods"]), int(z["k"]), int(z["d"]), int(z["m"])
    U = torch.from_numpy(z["U"]).double()                 # (P, d, m)
    R = torch.from_numpy(z["R"]).double()
    idx = torch.from_numpy(z["period_idx"])

    _, net = load_trained(ckpt_path, z)
    A = net.A.detach().double()                           # (m, k)

    g = torch.Generator().manual_seed(args.seed)
    A_rand = [torch.linalg.qr(torch.randn(m, k, generator=g, dtype=torch.double))[0]
              for _ in range(args.n_rand)]

    print(f"\nUniverse: P={P} periods, d={d} firms, m={m} descriptors, k={k} factors")
    print("rho = mean cos^2(principal angle) between span and top-k return PCs  (1 = aligned)\n")
    print(f"  {'period':>6} {'learned':>9} {'ceiling':>9} {'floor':>9} {'recovery%':>10}")

    learned_l, ceil_l, floor_l, frac_l = [], [], [], []
    for pi in range(P):
        Rp = R[idx == pi]
        PC = top_pcs(cov(Rp), k)                           # (d, k) return factor dirs
        Up = U[pi]                                         # (d, m) descriptor span

        learned = alignment(Up @ A, PC)                    # span(U_i A_hat)
        ceiling = alignment(Up, PC)                        # col(U_i) -- best possible
        floor = float(np.mean([alignment(Up @ Ar, PC) for Ar in A_rand]))

        denom = ceiling - floor
        frac = (learned - floor) / denom if denom > 1e-6 else float("nan")
        learned_l.append(learned); ceil_l.append(ceiling)
        floor_l.append(floor); frac_l.append(frac)
        print(f"  {pi:>6} {learned:>9.4f} {ceiling:>9.4f} {floor:>9.4f} "
              f"{100 * frac:>9.1f}%")

    mlearned, mceil, mfloor = np.mean(learned_l), np.mean(ceil_l), np.mean(floor_l)
    mfrac = np.nanmean(frac_l)
    print("\n" + "=" * 60)
    print("SUMMARY (averaged over periods)")
    print("=" * 60)
    print(f"  learned  rho(span(U_i A_hat), PC_k) = {mlearned:.4f}")
    print(f"  ceiling  rho(col(U_i),        PC_k) = {mceil:.4f}   "
          f"(descriptors' intrinsic limit)")
    print(f"  floor    rho(span(U_i A_rand),PC_k) = {mfloor:.4f}   "
          f"(random-loading chance)")
    print(f"  recovery fraction (learned-floor)/(ceiling-floor) = {100 * mfrac:.1f}%")
    print()
    print("  Read: learned near ceiling => A extracted the covariance signal the")
    print("  descriptors carry; near floor => no signal recovered (or untrained).")


if __name__ == "__main__":
    main()
