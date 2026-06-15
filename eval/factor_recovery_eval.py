"""
Stage-4 validation for the NOTES §3.2 factor-recovery experiment.

1. Subspace recovery (SPEC §5): principal angles between span(A) and span(T)
   in R^m — alignment rho = (1/k) sum sigma_j^2 (-> 1) and projection metric
   d_proj = sqrt(2) * sqrt(k - sum sigma_j^2) (-> 0); same metric per period
   for span(U_i A) vs span(U_i T) in R^d.
2. Analytic posterior check: with p_f = N(0, I_k) the posterior mean is the
   closed-form linear map E[f | Q_i^T r_t] = (M^T M + sigma_t^2 I)^{-1} M^T Q_i^T r_t,
   M = alpha_t V_i T. The network output is compared through the identifiable
   images A·Network(...) vs T·E[f|·] in R^m (factor coordinates alone are not
   identifiable). A sign bug in the score wiring shows up here as cosine ~ -1.
3. PCA benchmark: per-period top-k right singular vectors of the returns vs
   span(U_i T), same principal-angle metric — the pass/fail reference
   (no invented threshold).
"""

import argparse
import glob
import os

import numpy as np
import torch

from diffusion_factor_model import FactorScoreNet, FactorGaussianDiffusion


def orth(M):
    Uo, _, _ = torch.linalg.svd(M, full_matrices=False)
    return Uo


def subspace_metrics(A, B):
    """Principal-angle metrics between col spans of A and B (same ambient dim, k cols)."""
    k = A.shape[1]
    sigma = torch.linalg.svdvals(orth(A).T @ orth(B))
    sum_sq = sigma.pow(2).sum()
    rho = (sum_sq / k).item()
    d_proj = (np.sqrt(2.0) * torch.sqrt((k - sum_sq).clamp(min=0))).item()
    return rho, d_proj, sigma


def load_trained(ckpt_path, z):
    """Rebuild FactorScoreNet + FactorGaussianDiffusion and load the online weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    pe = ckpt["model"].get("model.period_emb.weight")
    U, Q = torch.from_numpy(z["U"]), torch.from_numpy(z["Q"])
    net = FactorScoreNet(U, Q, k=int(z["k"]),
                         period_emb_dim=pe.shape[1] if pe is not None else 0)
    diffusion = FactorGaussianDiffusion(net, d=int(z["d"]), timesteps=200, beta_schedule="cosine")
    diffusion.load_state_dict(ckpt["model"])
    diffusion.eval()
    return diffusion, net


def analytic_posterior_mean(y, alpha_t, sigma2_t, V_i, T):
    """E[f | y = Q_i^T r_t] for f ~ N(0, I_k), r_t = alpha_t U_i T f + sigma_t eps."""
    M = alpha_t * V_i @ T                                        # (m, k)
    k = T.shape[1]
    G = M.T @ M + sigma2_t * torch.eye(k, dtype=M.dtype)         # (k, k)
    return torch.linalg.solve(G, M.T @ y.T).T                    # (b, k)


def posterior_check(diffusion, net, z, t_grid, n_samples=512, seed=0):
    """Compare A·Network(Q^T r_t, t, i) against T·E[f|Q^T r_t] in R^m, per t."""
    g = torch.Generator().manual_seed(seed)
    R = torch.from_numpy(z["R"])
    idx = torch.from_numpy(z["period_idx"])
    V64 = torch.from_numpy(z["V"])
    T64 = torch.from_numpy(z["T"])
    A64 = net.A.detach().double()

    sel = torch.randperm(len(R), generator=g)[:n_samples]
    r0, i = R[sel], idx[sel]

    rows = []
    for t_int in t_grid:
        t = torch.full((n_samples,), t_int, dtype=torch.long)
        alpha_t = diffusion.sqrt_alphas_cumprod[t_int].double()
        sigma2_t = (1.0 - diffusion.alphas_cumprod[t_int]).double()

        eps = torch.randn(r0.shape, generator=g)
        with torch.no_grad():
            x_t = diffusion.q_sample(r0, t, eps)
            y = torch.einsum("bdm,bd->bm", net.Q[i], x_t)        # Q_i^T r_t
            f_net = net.network(y, t, i)                          # (b, k)

        learned = (A64 @ f_net.double().T).T                      # A·Network in R^m
        truth = torch.empty(n_samples, net.m, dtype=torch.float64)
        for p in range(int(z["periods"])):
            mask = i == p
            if mask.any():
                ef = analytic_posterior_mean(y[mask].double(), alpha_t, sigma2_t, V64[p], T64)
                truth[mask] = (T64 @ ef.T).T

        cos = torch.nn.functional.cosine_similarity(learned, truth, dim=1)
        rel = (learned - truth).norm(dim=1) / truth.norm(dim=1).clamp(min=1e-12)
        rows.append((t_int, alpha_t.item(), sigma2_t.item(),
                     cos.mean().item(), cos.min().item(), rel.median().item()))
    return rows


def main():
    parser = argparse.ArgumentParser(description="§3.2 factor-recovery validation")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="checkpoint .pt file, or a model dir (latest epoch used)")
    parser.add_argument("--data_path", type=str,
                        default="simulation_experiment_data/factor_period_data.npz")
    parser.add_argument("--t_grid", type=int, nargs="+",
                        default=[0, 10, 25, 50, 100, 150, 199])
    args = parser.parse_args()

    ckpt_path = args.ckpt
    if os.path.isdir(ckpt_path):
        ckpt_path = max(glob.glob(os.path.join(ckpt_path, "model-epoch-*.pt")),
                        key=lambda p: int(p.rsplit("-", 1)[1].split(".")[0]))
    print(f"Checkpoint: {ckpt_path}")

    z = np.load(args.data_path)
    P, k = int(z["periods"]), int(z["k"])
    T = torch.from_numpy(z["T"])
    U = torch.from_numpy(z["U"])

    diffusion, net = load_trained(ckpt_path, z)
    A = net.A.detach().double()

    print("\n" + "=" * 72)
    print("1. Subspace recovery: span(A) vs span(T)  [m-space, SPEC §5]")
    print("=" * 72)
    rho, d_proj, sigma = subspace_metrics(A, T)
    print(f"  cos(principal angles) = {[f'{s:.6f}' for s in sigma.tolist()]}")
    print(f"  rho    = {rho:.6f}   (-> 1)")
    print(f"  d_proj = {d_proj:.6f}   (-> 0; max possible {np.sqrt(2 * k):.3f})")

    print("\n  Per-period span(U_i A) vs span(U_i T)  [d-space]:")
    rho_d = []
    for p in range(P):
        r_p, dp_p, _ = subspace_metrics(U[p] @ A, U[p] @ T)
        rho_d.append(r_p)
        print(f"    period {p:>2}: rho = {r_p:.6f}   d_proj = {dp_p:.6f}")
    print(f"  mean rho over periods = {np.mean(rho_d):.6f}")

    print("\n" + "=" * 72)
    print("2. Analytic posterior check: A*Network(...) vs T*E[f|Q^T r_t] in R^m")
    print("=" * 72)
    rows = posterior_check(diffusion, net, z, args.t_grid)
    print(f"  {'t':>4} {'alpha_t':>8} {'sigma_t^2':>9} {'mean cos':>9} {'min cos':>9} {'med rel err':>12}")
    for t_int, a, s2, c_mean, c_min, rel in rows:
        print(f"  {t_int:>4} {a:>8.4f} {s2:>9.4f} {c_mean:>9.4f} {c_min:>9.4f} {rel:>12.4f}")

    print("\n" + "=" * 72)
    print("3. PCA benchmark: per-period top-k PCs of returns vs span(U_i T)")
    print("=" * 72)
    R, idx = torch.from_numpy(z["R"]).double(), torch.from_numpy(z["period_idx"])
    rho_pca = []
    for p in range(P):
        _, _, Vh = torch.linalg.svd(R[idx == p], full_matrices=False)
        r_p, dp_p, _ = subspace_metrics(Vh[:k].T, U[p] @ T)
        rho_pca.append(r_p)
        print(f"    period {p:>2}: rho = {r_p:.6f}   d_proj = {dp_p:.6f}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  learned span(A) vs span(T):       rho = {rho:.6f}, d_proj = {d_proj:.6f}")
    print(f"  learned span(U_iA) vs span(U_iT): mean rho = {np.mean(rho_d):.6f}")
    print(f"  PCA benchmark (same returns):     mean rho = {np.mean(rho_pca):.6f}")


if __name__ == "__main__":
    main()
