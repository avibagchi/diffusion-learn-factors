"""
Side-by-side comparison of two shared-metric JSON dumps (fixed-U vs per-period-U).

    python eval/compare_runs.py --per_u metrics_experiment.json --fixed_u metrics_fixedU.json
"""

import argparse
import json


def _get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_u", required=True, help="experiment (per-period U) metrics json")
    ap.add_argument("--fixed_u", required=True, help="fixed-U metrics json")
    args = ap.parse_args()

    A = json.load(open(args.per_u))
    B = json.load(open(args.fixed_u))

    # (label, *json path)
    rows = [
        ("MOMENT MATCHING", None),
        ("relative cov error  eps_cov", "moment_matching", "cov_rel_error_eps"),
        ("cov ratio  kappa_cov", "moment_matching", "cov_ratio_kappa"),
        ("mean difference", "moment_matching", "mean_diff_l2"),
        ("LOADING RECOVERY (A vs T)", None),
        ("mean principal angle (deg)", "loading_recovery", "principal_angle_mean_deg"),
        ("normalized projector error", "loading_recovery", "projector_error_normalized"),
        ("FACTOR-SUBSPACE (train)", None),
        ("subspace_angle_error", "factor_subspace_recovery_train", "subspace_angle_error"),
        ("covariance_recon_error", "factor_subspace_recovery_train", "covariance_recon_error"),
        ("eigenvalue_rel_l2", "factor_subspace_recovery_train", "eigenvalue_rel_l2"),
        ("FACTOR-SUBSPACE (generated)", None),
        ("subspace_angle_error", "factor_subspace_recovery_gen", "subspace_angle_error"),
        ("covariance_recon_error", "factor_subspace_recovery_gen", "covariance_recon_error"),
        ("eigenvalue_rel_l2", "factor_subspace_recovery_gen", "eigenvalue_rel_l2"),
    ]

    w = 36
    print(f"\n{'metric':<{w}}{'per-period U':>16}{'fixed U':>16}")
    print("=" * (w + 32))
    for label, *path in rows:
        if path[0] is None:
            print(f"\n{label}")
            continue
        a, b = _get(A, *path), _get(B, *path)
        fa = f"{a:>16.6f}" if isinstance(a, (int, float)) else f"{'-':>16}"
        fb = f"{b:>16.6f}" if isinstance(b, (int, float)) else f"{'-':>16}"
        print(f"  {label:<{w-2}}{fa}{fb}")
    print()


if __name__ == "__main__":
    main()
