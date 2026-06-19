#!/usr/bin/env python3
"""
Train the descriptor score model (Section 3.2) on real period-indexed data.

Loads empirical_analysis_data/real_factor_period_data.npz, which stores
  R (N, d), period_idx (N,), U (P, d, m)
and expands U to per-observation descriptor matrices for DescriptorReturnDataset.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import config as cfg
from diffusion_factor_model.descriptor_score_model import (
    DescriptorFactorDiffusion,
    DescriptorReturnDataset,
    DescriptorTrainer,
)


def load_real_factor_period_data(path: str | Path) -> dict:
    """Load the real-data .npz into the descriptor training format."""
    z = np.load(path)
    returns = z["R"].astype(np.float32)
    period_idx = z["period_idx"].astype(np.int64)
    U_period = z["U"].astype(np.float32)

    if U_period.ndim != 3:
        raise ValueError(f"U must have shape (P, d, m), got {U_period.shape}")
    P, d, m = U_period.shape
    k = int(z["k"])
    if returns.shape != (len(period_idx), d):
        raise ValueError(
            f"returns shape {returns.shape} incompatible with period_idx len "
            f"{len(period_idx)} and d={d}"
        )
    if period_idx.min() < 0 or period_idx.max() >= P:
        raise ValueError(
            f"period_idx out of range [0, {P - 1}]: "
            f"min={period_idx.min()}, max={period_idx.max()}"
        )

    descriptors = U_period[period_idx]
    metadata = {
        "num_samples": int(len(returns)),
        "num_assets": d,
        "num_descriptors": m,
        "num_factors": k,
        "num_periods": P,
        "descriptor_mode": "real_period",
        "source": str(path),
    }
    if "tickers" in z.files:
        metadata["tickers"] = [str(t) for t in z["tickers"]]
    if "years" in z.files:
        metadata["years"] = [int(y) for y in z["years"]]
    if "char_names" in z.files:
        metadata["char_names"] = [str(c) for c in z["char_names"]]

    return {
        "returns": returns,
        "descriptors": descriptors,
        "period_idx": period_idx,
        "metadata": metadata,
    }


def generate_samples(
    model: DescriptorFactorDiffusion,
    data: dict,
    device: torch.device,
    num_gen: int,
    gen_batch_size: int = cfg.SAMPLES_PER_BATCH,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate returns conditioned on real descriptor matrices and period indices."""
    model.eval()
    n = min(num_gen, data["returns"].shape[0])
    rng = np.random.default_rng(seed)
    indices = rng.choice(data["returns"].shape[0], size=n, replace=False)

    chunks = []
    with torch.no_grad():
        for start in range(0, n, gen_batch_size):
            batch_idx = indices[start : start + gen_batch_size]
            U = torch.from_numpy(data["descriptors"][batch_idx]).to(device)
            period_idx = torch.from_numpy(data["period_idx"][batch_idx]).to(device)
            chunks.append(model.sample(U, period_idx).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def evaluate_samples(
    model: DescriptorFactorDiffusion,
    data: dict,
    device: torch.device,
    num_gen: int = cfg.SAMPLE_BATCHES * cfg.SAMPLES_PER_BATCH,
    gen_batch_size: int = cfg.SAMPLES_PER_BATCH,
    seed: Optional[int] = None,
) -> tuple[dict, np.ndarray]:
    generated = generate_samples(
        model, data, device, num_gen, gen_batch_size, seed=seed
    )
    n = generated.shape[0]
    real = data["returns"][:n]
    metrics = {
        "real_mean_l2": float(np.linalg.norm(real.mean(0))),
        "gen_mean_l2": float(np.linalg.norm(generated.mean(0))),
        "mean_diff_l2": float(np.linalg.norm(real.mean(0) - generated.mean(0))),
        "real_cov_fro": float(np.linalg.norm(np.cov(real.T))),
        "gen_cov_fro": float(np.linalg.norm(np.cov(generated.T))),
        "cov_diff_fro": float(np.linalg.norm(np.cov(real.T) - np.cov(generated.T))),
    }

    A = model.score_net.A.detach().cpu().numpy()
    _, s_a, _ = np.linalg.svd(A, full_matrices=False)
    metrics["A_singular_values"] = s_a.tolist()
    return metrics, generated


def save_generated_samples(
    generated: np.ndarray,
    data: dict,
    output_dir: Path,
    metrics: Optional[dict] = None,
) -> Path:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.save(sample_dir / "generated.npy", generated)
    np.save(sample_dir / "real.npy", data["returns"][: generated.shape[0]])
    if metrics is not None:
        with open(sample_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    return sample_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train descriptor score model on real_factor_period_data.npz"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="empirical_analysis_data/real_factor_period_data.npz",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_results/descriptor_real",
    )
    parser.add_argument("--timesteps", type=int, default=cfg.TIMESTEPS)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--gpu", type=int, default=0, help="GPU id; use -1 for CPU")
    parser.add_argument(
        "--save_every",
        type=int,
        default=20,
        help="Save a numbered checkpoint every N epochs (always saves the last epoch too)",
    )
    parser.add_argument(
        "--num_gen",
        type=int,
        default=cfg.SAMPLE_BATCHES * cfg.SAMPLES_PER_BATCH,
        help="Generated samples for post-training metrics",
    )
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        default=cfg.SAMPLES_PER_BATCH,
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        if args.gpu >= 0:
            print("CUDA not available, using CPU")

    data = load_real_factor_period_data(args.data_path)
    meta = data["metadata"]
    print(
        f"Loaded {args.data_path}: "
        f"N={meta['num_samples']} d={meta['num_assets']} "
        f"m={meta['num_descriptors']} k={meta['num_factors']} "
        f"P={meta['num_periods']}"
    )

    dataset = DescriptorReturnDataset(
        data["returns"],
        data["descriptors"],
        data["period_idx"],
    )

    model = DescriptorFactorDiffusion(
        num_assets=meta["num_assets"],
        num_descriptors=meta["num_descriptors"],
        num_factors=meta["num_factors"],
        timesteps=args.timesteps,
        beta_schedule="cosine",
        hidden_size=args.hidden_size,
        max_periods=meta["num_periods"],
    )

    trainer = DescriptorTrainer(
        model,
        dataset,
        train_lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        results_folder=args.output_dir,
        save_every=args.save_every,
        device=str(device),
    )

    print(
        f"Training on {device}: timesteps={args.timesteps}, "
        f"epochs={args.epochs}, save_every={args.save_every}"
    )
    trainer.train()

    model.to(device)
    metrics, generated = evaluate_samples(
        model,
        data,
        device,
        num_gen=args.num_gen,
        gen_batch_size=args.gen_batch_size,
        seed=args.seed,
    )
    print("Post-training metrics:")
    for key, value in metrics.items():
        if key.endswith("singular_values"):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.6f}")

    sample_dir = save_generated_samples(generated, data, Path(args.output_dir), metrics)
    print(f"Saved {generated.shape[0]} samples to {sample_dir}")


if __name__ == "__main__":
    main()
