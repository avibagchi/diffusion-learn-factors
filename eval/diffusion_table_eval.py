"""
Descriptor diffusion model evaluation table (NOTES §3.2 factor experiment).

Reproduces the three-section evaluation table:

  1. Moment Matching          — generated vs. real (training) pooled moments.
  2. Subspace Recovery        — learned A vs. ground-truth T, in R^m.
  3. Ground-Truth Covariance  — per-period sample covariance (training and
                                generated) vs. the analytic true covariance
                                C_i = (U_i T)(U_i T)^T  (factors ~ N(0, I_k)).
                                A ratio < 1 means the generated samples are
                                closer to the true covariance structure than
                                the finite training sample.

The generated samples come from a period-conditioned ancestral reverse
process (implemented here — FactorGaussianDiffusion ships no period-aware
sampler; the inherited image-shaped sample() ignores i).

Metric definitions (documented so the table is reproducible):
  kappa_cov = ||C_g||_F / ||C_r||_F                 (covariance scale ratio)
  eps_cov   = ||C_r - C_g||_F / ||C_r||_F           (relative covariance error)
  kappa_std = ||std_g||_2 / ||std_r||_2             (per-asset std-vector ratio)
  Principal angles  theta_j = arccos(sigma_j), sigma_j = svals(orth(A)^T orth(B))
  Grassmann distance = ||theta||_2  (radians)
  Projector error    = ||P_A - P_B||_F,  P = Q Q^T over an orthonormal basis Q
  Normalized proj.   = ||P_A - P_B||_F / sqrt(2k)   = RMS sin(theta) in [0,1]
  Procrustes residual= min_{R orthogonal} ||orth(A) R - orth(B)||_F
  Subspace Error     = normalized projector error between the top-k sample
                       eigenspace and span(U_i T)        (averaged over periods)
  Eigenvalue Error   = ||lambda_top_k - lambda_true||_2 / ||lambda_true||_2
                                                         (averaged over periods)
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root
sys.path.insert(0, _HERE)

from factor_recovery_eval import load_trained  # noqa: E402


# ----------------------------- linear algebra ------------------------------ #

def orth(M):
    """Orthonormal basis for the column space of M (m x k -> m x k)."""
    Uo, _, _ = torch.linalg.svd(M, full_matrices=False)
    return Uo


def principal_cosines(A, B):
    """sigma_j = cos(theta_j) between col spans of A, B (same ambient dim)."""
    sig = torch.linalg.svdvals(orth(A).T @ orth(B))
    return sig.clamp(0.0, 1.0)


def projector_error(A, B):
    QA, QB = orth(A), orth(B)
    return (QA @ QA.T - QB @ QB.T).norm().item()


def procrustes_residual(A, B):
    """min over orthogonal R of ||orth(A) R - orth(B)||_F."""
    QA, QB = orth(A), orth(B)
    W, S, Vh = torch.linalg.svd(QA.T @ QB)
    R = W @ Vh
    return (QA @ R - QB).norm().item()


def subspace_error(A, B):
    """Normalized projector distance = RMS sin(theta) in [0, 1]."""
    k = A.shape[1]
    sig = principal_cosines(A, B)
    return float(torch.sqrt((1.0 - sig.pow(2).mean()).clamp(min=0)))


def cov(X):
    Xc = X - X.mean(dim=0, keepdim=True)
    return Xc.T @ Xc / (len(X) - 1)


def top_eigs(S, n):
    w, v = torch.linalg.eigh(S)
    return w.flip(0)[:n], v.flip(1)[:, :n]


# --------------------- period-conditioned reverse process ------------------- #

@torch.inference_mode()
def sample_periods(diffusion, net, i, seed=0):
    """Ancestral reverse diffusion conditioned on period index i: (b,) -> (b, d)."""
    g = torch.Generator().manual_seed(seed)
    b = len(i)
    x = torch.randn(b, net.d, generator=g)
    for t in reversed(range(diffusion.num_timesteps)):
        bt = torch.full((b,), t, dtype=torch.long)
        x_start = net(x, bt, i)                                    # pred_x0
        mean, _, log_var, _ = diffusion.q_posterior_helper(x_start, x, bt)
        noise = torch.randn(x.shape, generator=g) if t > 0 else 0.0
        x = mean + (0.5 * log_var).exp() * noise
    return x


def _attach_q_posterior(diffusion):
    """Expose q_posterior with explicit tensor t (parent signature is fine as-is)."""
    if not hasattr(diffusion, "q_posterior_helper"):
        def helper(x_start, x_t, t):
            mean, var, logvar = diffusion.q_posterior(x_start=x_start, x_t=x_t, t=t)
            return mean, var, logvar, x_start
        diffusion.q_posterior_helper = helper


# --------------------------------- table ----------------------------------- #

def fmt(label, value, width=42):
    return f"  {label:<{width}}{value:>14.6f}"


def main():
    p = argparse.ArgumentParser(description="Descriptor diffusion evaluation table")
    p.add_argument("--ckpt", type=str,
                   default="model_results/factor_factor_period_data_ts1781059267_seed42",
                   help="checkpoint .pt or model dir (latest epoch used)")
    p.add_argument("--data_path", type=str,
                   default="simulation_experiment_data/factor_period_data.npz")
    p.add_argument("--n_per_period", type=int, default=2000,
                   help="generated samples per period (2000 matches the training set)")
    p.add_argument("--sample_batch", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latex", action="store_true", help="also emit a LaTeX table")
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
    T = torch.from_numpy(z["T"]).double()
    U = torch.from_numpy(z["U"]).double()

    diffusion, net = load_trained(ckpt_path, z)
    _attach_q_posterior(diffusion)
    A = net.A.detach().double()

    # ---- generate via the period-conditioned reverse process ----
    i_gen = torch.arange(P).repeat_interleave(args.n_per_period)
    print(f"Sampling: period-conditioned ancestral reverse diffusion, "
          f"{diffusion.num_timesteps} steps, N_gen = {len(i_gen)} "
          f"({P} periods x {args.n_per_period})")
    chunks = []
    for s in range(0, len(i_gen), args.sample_batch):
        chunks.append(sample_periods(diffusion, net, i_gen[s:s + args.sample_batch],
                                     seed=args.seed + s))
        print(f"  generated {sum(len(c) for c in chunks)}/{len(i_gen)}")
    R_gen = torch.cat(chunks).double()

    # ============================ 1. moments ============================ #
    mu_r, mu_g = R_real.mean(0), R_gen.mean(0)
    C_r, C_g = cov(R_real), cov(R_gen)
    cov_diff = (C_r - C_g).norm().item()
    real_cov_norm, gen_cov_norm = C_r.norm().item(), C_g.norm().item()
    std_r = torch.sqrt(torch.diag(C_r).clamp(min=0))
    std_g = torch.sqrt(torch.diag(C_g).clamp(min=0))

    rows1 = [
        ("Mean Difference ||mu_r - mu_g||_2", (mu_r - mu_g).norm().item()),
        ("Real Mean Norm ||mu_r||_2", mu_r.norm().item()),
        ("Generated Mean Norm ||mu_g||_2", mu_g.norm().item()),
        ("Real Covariance Norm ||C_r||_F", real_cov_norm),
        ("Generated Covariance Norm ||C_g||_F", gen_cov_norm),
        ("Covariance Difference ||C_r - C_g||_F", cov_diff),
        ("Covariance Ratio kappa_cov", gen_cov_norm / real_cov_norm),
        ("Relative Covariance Error eps_cov", cov_diff / real_cov_norm),
        ("Standard Deviation Ratio kappa_std", (std_g.norm() / std_r.norm()).item()),
    ]

    # ======================= 2. subspace recovery ======================= #
    sig = principal_cosines(A, T)
    theta = torch.arccos(sig)
    rows2_int = [
        ("Rank of A", int(torch.linalg.matrix_rank(A))),
        ("Rank of T", int(torch.linalg.matrix_rank(T))),
    ]
    rows2 = [
        ("Mean Principal Angle (deg)", torch.rad2deg(theta).mean().item()),
        ("Max Principal Angle (deg)", torch.rad2deg(theta).max().item()),
        ("Grassmann Distance", theta.norm().item()),
        ("Projector Error", projector_error(A, T)),
        ("Normalized Projector Error", projector_error(A, T) / np.sqrt(2 * k)),
        ("Procrustes Residual", procrustes_residual(A, T)),
    ]

    # =================== 3. ground-truth covariance ===================== #
    # true per-period covariance C_i = (U_i T)(U_i T)^T  (Sigma_F = I_k)
    sub_tr, sub_gen, eig_tr, eig_gen = [], [], [], []
    for pi in range(P):
        B = U[pi] @ T                                  # (d, k)
        true_basis = orth(B)
        lam_true = torch.linalg.svdvals(B).pow(2)      # nonzero eigenvalues of C_i

        Sr = cov(R_real[idx_real == pi])
        Sg = cov(R_gen[i_gen == pi])
        wr, vr = top_eigs(Sr, k)
        wg, vg = top_eigs(Sg, k)

        sub_tr.append(subspace_error(vr, true_basis))
        sub_gen.append(subspace_error(vg, true_basis))
        eig_tr.append((wr - lam_true).norm().item() / lam_true.norm().item())
        eig_gen.append((wg - lam_true).norm().item() / lam_true.norm().item())

    sub_tr_m, sub_gen_m = float(np.mean(sub_tr)), float(np.mean(sub_gen))
    eig_tr_m, eig_gen_m = float(np.mean(eig_tr)), float(np.mean(eig_gen))
    rows3 = [
        ("Subspace Error (Training)", sub_tr_m),
        ("Subspace Error (Generated)", sub_gen_m),
        ("Subspace Error Ratio", sub_gen_m / sub_tr_m),
        ("Eigenvalue Error (Training)", eig_tr_m),
        ("Eigenvalue Error (Generated)", eig_gen_m),
        ("Eigenvalue Error Ratio", eig_gen_m / eig_tr_m),
    ]

    # -------------------------------- print -------------------------------- #
    bar = "=" * 58
    print("\n" + bar + "\nMoment Matching\n" + bar)
    for lab, v in rows1:
        print(fmt(lab, v))
    print("\n" + bar + "\nSubspace Recovery (Learned A vs Ground Truth T)\n" + bar)
    for lab, v in rows2_int:
        print(f"  {lab:<42}{v:>14d}")
    for lab, v in rows2:
        print(fmt(lab, v))
    print("\n" + bar + "\nGround-Truth Covariance Evaluation\n" + bar)
    for lab, v in rows3:
        print(fmt(lab, v))

    if args.latex:
        emit_latex(rows1, rows2_int, rows2, rows3)


def emit_latex(rows1, rows2_int, rows2, rows3):
    def block(title, items, intitems=None):
        out = [f"        \\textit{{{title}}} & \\\\"]
        for lab, v in (intitems or []):
            out.append(f"        \\quad {lab} & {v} \\\\")
        for lab, v in items:
            out.append(f"        \\quad {lab} & {v:.6f} \\\\")
        return "\n".join(out)

    print("\n% ---- LaTeX (booktabs) ----")
    print("\\begin{table}[t]\n\\centering\n\\begin{tabular}{lr}\n\\toprule")
    print("\\textbf{Metric} & \\textbf{Value} \\\\\n\\midrule")
    print(block("Moment Matching", rows1))
    print("\\midrule")
    print(block("Subspace Recovery (Learned $A$ vs.\\ Ground Truth $T$)", rows2, rows2_int))
    print("\\midrule")
    print(block("Ground-Truth Covariance Evaluation", rows3))
    print("\\bottomrule\n\\end{tabular}\n\\end{table}")


if __name__ == "__main__":
    main()
