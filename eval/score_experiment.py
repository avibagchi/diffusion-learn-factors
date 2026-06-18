"""
Score the per-period-U (experiment branch) model with the shared metric set.

Loads a FactorScoreNet checkpoint, samples returns via the period-conditioned
ancestral reverse process, and runs eval/shared_metrics on (R_real, R_gen) with
ground-truth loadings beta_i = U_i T and learned A vs T.
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

from factor_recovery_eval import load_trained                       # noqa: E402
from diffusion_table_eval import sample_periods, _attach_q_posterior  # noqa: E402
from shared_metrics import compute_metrics, format_table             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt or model dir")
    ap.add_argument("--data_path", default="simulation_experiment_data/factor_period_data.npz")
    ap.add_argument("--n_per_period", type=int, default=2000)
    ap.add_argument("--sample_batch", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    ckpt = args.ckpt
    if os.path.isdir(ckpt):
        ckpt = max(glob.glob(os.path.join(ckpt, "model-epoch-*.pt")),
                   key=lambda q: int(q.rsplit("-", 1)[1].split(".")[0]))
    print(f"Checkpoint: {ckpt}")

    z = np.load(args.data_path)
    P, k, d = int(z["periods"]), int(z["k"]), int(z["d"])
    R_real = z["R"].astype(np.float64)
    idx_real = z["period_idx"]
    T = z["T"].astype(np.float64)
    beta = np.einsum("pdm,mk->pdk", z["U"], z["T"]).astype(np.float64)  # (P, d, k)

    diffusion, net = load_trained(ckpt, z)
    _attach_q_posterior(diffusion)
    A = net.A.detach().double().numpy()

    # period-conditioned ancestral sampling
    i_gen = torch.arange(P).repeat_interleave(args.n_per_period)
    print(f"Sampling {len(i_gen)} returns ({P} periods x {args.n_per_period}) ...")
    chunks = []
    for s in range(0, len(i_gen), args.sample_batch):
        chunks.append(sample_periods(diffusion, net, i_gen[s:s + args.sample_batch],
                                     seed=args.seed + s))
    R_gen = torch.cat(chunks).double().numpy()
    idx_gen = i_gen.numpy()

    metrics = compute_metrics(R_real, idx_real, beta,
                              R_gen=R_gen, idx_gen=idx_gen, A=A, T=T)
    print("\n>>> EXPERIMENT (per-period U)\n")
    print(format_table(metrics))

    if args.json_out:
        import json
        with open(args.json_out, "w") as f:
            json.dump({kk: vv for kk, vv in metrics.items()}, f, indent=2, default=float)
        print(f"\nSaved metrics -> {args.json_out}")


if __name__ == "__main__":
    main()
