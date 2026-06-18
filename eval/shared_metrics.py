"""
Shared factor-recovery metrics: fixed-U (one shared loading) vs per-period-U.

One metric implementation both experiments run, so the fixed-U and per-period-U
runs are directly comparable. BOTH metric families are reported side by side so
no information from either side is lost; each is named for what it measures.

Two notions of "subspace error":

  subspace_angle_error    RMS sin(principal angle) between the top-k sample
                          eigenvector span and span(beta_i). Orientation ONLY --
                          eigenvalue/scale-invariant, in [0, 1].
                          (the `experiment` branch's metric)

  covariance_recon_error  ||trunc_k(Sigma_hat) - trunc_k(C_GT)||_F
                            / ||trunc_k(C_GT)||_F
                          Full rank-k covariance reconstruction: orientation AND
                          eigenvalue magnitude AND scale.
                          (the `subspace-recovery-metric` branch's metric)

Two eigenvalue errors, same idea:

  eigenvalue_rel_l2       ||lambda - lambda_true||_2 / ||lambda_true||_2   (experiment)
  eigenvalue_mape         mean| lambda / lambda_true - 1 |                 (subspace-recovery)

Aggregation: per period, then averaged. This is the GENERAL form -- fixed-U is the
special case where every beta_i is identical -- so the two experiments line up on
identical definitions. The pooled covariance-recon error (his original pooled form)
is also reported under a clearly-labelled key so nothing is dropped.

Ground-truth covariance per period is C_i = beta_i beta_i^T with beta_i = U_i T
(noiseless rank-k, as in the descriptor experiments). Pass beta as (P, d, k);
fixed-U is simply beta with all P slices equal (or P = 1).

Pure numpy so both branches can import it (his pipeline is numpy; the torch
`experiment` branch just passes `.cpu().numpy()` arrays).
"""

import numpy as np


# --------------------------- linear-algebra helpers ------------------------- #

def _cov(X):
    """(N, d) -> (d, d) sample covariance, ddof=1 (matches np.cov and your cov())."""
    Xc = X - X.mean(axis=0, keepdims=True)
    return Xc.T @ Xc / (len(X) - 1)


def _orth(M):
    """Orthonormal basis of the column span of M (d, r) via reduced SVD."""
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    return U


def _top_eig(S, k):
    """Top-k eigenpairs of symmetric S, descending: (vals (k,), vecs (d, k))."""
    w, V = np.linalg.eigh(S)
    order = np.argsort(w)[::-1][:k]
    return w[order], V[:, order]


def _trunc_k(S, k):
    """Best rank-k reconstruction of a symmetric PSD matrix (Eckart-Young)."""
    w, V = _top_eig(S, k)
    w = np.clip(w, 0.0, None)
    return (V * w) @ V.T


def _principal_cosines(A, B):
    """cos(principal angles) between the column spans of A and B."""
    s = np.linalg.svd(_orth(A).T @ _orth(B), compute_uv=False)
    return np.clip(s, 0.0, 1.0)


# ------------------------------ metric blocks ------------------------------- #

def moment_matching(R_real, R_gen):
    """Pooled first/second-moment match between real and generated returns.

    Both std-ratio aggregations are kept: `std_ratio_l2` (L2 norm of the per-asset
    std vector, the `experiment` form) and `std_ratio_mean` (mean of per-asset std,
    the `subspace-recovery` form)."""
    mu_r, mu_g = R_real.mean(0), R_gen.mean(0)
    C_r, C_g = _cov(R_real), _cov(R_gen)
    fro_r, fro_g = float(np.linalg.norm(C_r)), float(np.linalg.norm(C_g))
    std_r, std_g = R_real.std(0, ddof=1), R_gen.std(0, ddof=1)
    return {
        "mean_diff_l2": float(np.linalg.norm(mu_r - mu_g)),
        "real_mean_l2": float(np.linalg.norm(mu_r)),
        "gen_mean_l2": float(np.linalg.norm(mu_g)),
        "cov_fro_real": fro_r,
        "cov_fro_gen": fro_g,
        "cov_diff_fro": float(np.linalg.norm(C_r - C_g)),
        "cov_ratio_kappa": fro_g / fro_r,
        "cov_rel_error_eps": float(np.linalg.norm(C_r - C_g) / fro_r),
        "std_ratio_l2": float(np.linalg.norm(std_g) / np.linalg.norm(std_r)),
        "std_ratio_mean": float(std_g.mean() / std_r.mean()),
    }


def loading_recovery(A, T):
    """Learned loadings A vs ground-truth T, compared as subspaces in R^m.

    Parameter-level check from the `experiment` branch; the `subspace-recovery`
    branch has no analog. Pass A=None to skip."""
    cos = _principal_cosines(A, T)
    theta = np.arccos(cos)
    QA, QT = _orth(A), _orth(T)
    proj = float(np.linalg.norm(QA @ QA.T - QT @ QT.T))
    W, _, Vh = np.linalg.svd(QA.T @ QT)
    R = W @ Vh
    k = A.shape[1]
    return {
        "rank_A": int(np.linalg.matrix_rank(A)),
        "rank_T": int(np.linalg.matrix_rank(T)),
        "principal_angle_mean_deg": float(np.degrees(theta).mean()),
        "principal_angle_max_deg": float(np.degrees(theta).max()),
        "grassmann_distance": float(np.linalg.norm(theta)),
        "projector_error": proj,
        "projector_error_normalized": proj / np.sqrt(2 * k),
        "procrustes_residual": float(np.linalg.norm(QA @ R - QT)),
    }


def factor_subspace_recovery(R, idx, beta):
    """Per-period top-k recovery of span(beta_i) from the sample covariance.

    Reports BOTH the orientation-only angle error and the full rank-k covariance
    reconstruction error, plus both eigenvalue errors, each averaged over periods.
    Per-period arrays are returned under `_per_period` for diagnostics."""
    P, d, k = beta.shape
    angle, recon, eig_l2, eig_mape = [], [], [], []
    for p in range(P):
        Rp = R[idx == p]
        Sp = _cov(Rp)
        wp, Vp = _top_eig(Sp, k)
        wp = np.clip(wp, 0.0, None)

        Bp = beta[p]                                   # (d, k) ground-truth loading
        Cp = Bp @ Bp.T                                 # rank-k ground-truth covariance
        lam_true = np.linalg.svd(Bp, compute_uv=False)[:k] ** 2

        cos = _principal_cosines(Vp, Bp)               # orientation only
        angle.append(float(np.sqrt(max(0.0, 1.0 - (cos ** 2).mean()))))

        recon.append(float(np.linalg.norm(_trunc_k(Sp, k) - Cp)
                           / np.linalg.norm(Cp)))       # orientation + magnitude + scale

        eig_l2.append(float(np.linalg.norm(wp - lam_true) / np.linalg.norm(lam_true)))
        eig_mape.append(float(np.abs(wp / lam_true - 1).mean()))

    return {
        "subspace_angle_error": float(np.mean(angle)),
        "covariance_recon_error": float(np.mean(recon)),
        "eigenvalue_rel_l2": float(np.mean(eig_l2)),
        "eigenvalue_mape": float(np.mean(eig_mape)),
        "_per_period": {"angle": angle, "recon": recon,
                        "eig_l2": eig_l2, "eig_mape": eig_mape},
    }


def pooled_covariance_recon(R, idx, beta):
    """His ORIGINAL pooled form, kept for fidelity (no information lost).

    trunc_k of the pooled sample covariance vs trunc_k of the pooled ground-truth
    covariance (1/P) sum_i beta_i beta_i^T. NOTE: for per-period-U the pooled
    ground truth has rank up to min(P*k, d), so truncating to k mixes/loses planes
    -- which is exactly why the per-period form above is the comparable one."""
    P, d, k = beta.shape
    C_gt = np.zeros((d, d))
    for p in range(P):
        C_gt += beta[p] @ beta[p].T
    C_gt /= P
    gt_k = _trunc_k(C_gt, k)
    return float(np.linalg.norm(_trunc_k(_cov(R), k) - gt_k) / np.linalg.norm(gt_k))


# ------------------------------- top level ---------------------------------- #

def compute_metrics(R_real, idx_real, beta, R_gen=None, idx_gen=None, A=None, T=None):
    """Full shared metric set. Generated inputs and (A, T) are optional.

    R_real/R_gen: (N, d) float arrays. idx_*: (N,) int period indices in [0, P).
    beta: (P, d, k) per-period loadings U_i T (fixed-U => all slices equal / P=1).
    A, T: (m, k) learned and ground-truth loadings for the R^m subspace check."""
    R_real = np.asarray(R_real, dtype=np.float64)
    idx_real = np.asarray(idx_real)
    beta = np.asarray(beta, dtype=np.float64)

    out = {"factor_subspace_recovery_train": factor_subspace_recovery(R_real, idx_real, beta),
           "pooled_covariance_recon_train": pooled_covariance_recon(R_real, idx_real, beta)}

    if R_gen is not None:
        R_gen = np.asarray(R_gen, dtype=np.float64)
        idx_gen = np.asarray(idx_gen)
        out["factor_subspace_recovery_gen"] = factor_subspace_recovery(R_gen, idx_gen, beta)
        out["pooled_covariance_recon_gen"] = pooled_covariance_recon(R_gen, idx_gen, beta)
        out["moment_matching"] = moment_matching(R_real, R_gen)

    if A is not None and T is not None:
        out["loading_recovery"] = loading_recovery(np.asarray(A, dtype=np.float64),
                                                   np.asarray(T, dtype=np.float64))
    return out


def _ratio(gen, train):
    # ratio is meaningless (and explosive) when the denominator is ~0,
    # e.g. the noiseless training subspace_angle_error (~1e-8)
    return gen / train if train > 1e-6 else float("nan")


def format_table(m):
    """Render compute_metrics() output as a readable table with gen/train ratios."""
    lines, bar = [], "=" * 64

    def row(label, val, w=44):
        return f"  {label:<{w}}{val:>16.6f}"

    if "moment_matching" in m:
        mm = m["moment_matching"]
        lines += [bar, "Moment Matching (pooled)", bar]
        lines += [
            row("Mean Difference ||mu_r - mu_g||_2", mm["mean_diff_l2"]),
            row("Real / Gen Mean Norm",
                mm["real_mean_l2"]) + f" / {mm['gen_mean_l2']:.6f}",
            row("Cov Norm  ||C_r||_F / ||C_g||_F",
                mm["cov_fro_real"]) + f" / {mm['cov_fro_gen']:.6f}",
            row("Cov Difference ||C_r - C_g||_F", mm["cov_diff_fro"]),
            row("Cov Ratio  kappa_cov", mm["cov_ratio_kappa"]),
            row("Relative Cov Error  eps_cov", mm["cov_rel_error_eps"]),
            row("Std Ratio (L2, experiment)", mm["std_ratio_l2"]),
            row("Std Ratio (mean, subspace-recovery)", mm["std_ratio_mean"]),
        ]

    if "loading_recovery" in m:
        lr = m["loading_recovery"]
        lines += ["", bar, "Loading Recovery: learned A vs ground-truth T (R^m)", bar]
        lines += [f"  {'Rank of A / T':<44}{lr['rank_A']:>7d} / {lr['rank_T']:d}"]
        lines += [
            row("Mean Principal Angle (deg)", lr["principal_angle_mean_deg"]),
            row("Max Principal Angle (deg)", lr["principal_angle_max_deg"]),
            row("Grassmann Distance", lr["grassmann_distance"]),
            row("Projector Error", lr["projector_error"]),
            row("Normalized Projector Error", lr["projector_error_normalized"]),
            row("Procrustes Residual", lr["procrustes_residual"]),
        ]

    tr = m["factor_subspace_recovery_train"]
    gn = m.get("factor_subspace_recovery_gen")
    lines += ["", bar, "Factor-Subspace Recovery (per-period, averaged)", bar]
    lines += [f"  {'metric':<32}{'train':>13}{'generated':>13}{'ratio':>11}"]

    def _num(v):
        return "      -      " if v is None else f"{v:>13.6f}"

    def _rat(v):
        return "     n/a   " if v != v else f"{v:>11.4f}"   # v!=v => NaN

    def pair(label, t, g):
        r = None if g is None else _ratio(g, t)
        return f"  {label:<32}{_num(t)}{_num(g)}{'     -     ' if g is None else _rat(r)}"

    for key in ("subspace_angle_error", "covariance_recon_error",
                "eigenvalue_rel_l2", "eigenvalue_mape"):
        pair_g = gn[key] if gn else None
        lines.append(pair(key, tr[key], pair_g))

    pt = m["pooled_covariance_recon_train"]
    pg = m.get("pooled_covariance_recon_gen")
    lines.append(pair("covariance_recon_error (pooled)", pt, pg))

    return "\n".join(lines)


# --------------------------------- demo ------------------------------------- #

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run shared metrics on a period dataset")
    ap.add_argument("--data_path", type=str,
                    default="simulation_experiment_data/factor_period_data.npz")
    args = ap.parse_args()

    z = np.load(args.data_path)
    R, idx, T = z["R"], z["period_idx"], z["T"]
    beta = z["beta"] if "beta" in z.files else np.einsum("pdm,mk->pdk", z["U"], T)
    print(f"Loaded {args.data_path}: R={R.shape}, P={beta.shape[0]}, "
          f"d={beta.shape[1]}, k={beta.shape[2]}")
    print("(training side only -- pass generated samples / learned A to fill the rest)\n")

    metrics = compute_metrics(R, idx, beta)
    print(format_table(metrics))
