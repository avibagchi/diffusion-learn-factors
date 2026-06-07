#!/usr/bin/env python3
"""
Train descriptor-based score decomposition model (Section 3.2).

Score: s_θ(r,t,i) = -r/h_t + (α_t/h_t) U_i A · Network(Q_i^T r, t, i)

Designed to run on CPU or a single GPU (e.g. one A100).
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
    generate_descriptor_dataset,
    load_descriptor_dataset,
    save_descriptor_dataset,
)


def generate_samples(
    model: DescriptorFactorDiffusion,
    data: dict,
    device: torch.device,
    num_gen: int,
    gen_batch_size: int = cfg.SAMPLES_PER_BATCH,
) -> np.ndarray:
    """Generate returns in batches to avoid GPU OOM."""
    model.eval()
    n = min(num_gen, data["returns"].shape[0])
    meta = data.get("metadata", {})
    use_zero_period = meta.get("descriptor_mode") == "fixed"

    chunks = []
    with torch.no_grad():
        for start in range(0, n, gen_batch_size):
            end = min(start + gen_batch_size, n)
            U = torch.from_numpy(data["descriptors"][start:end]).to(device)
            if use_zero_period:
                period_idx = torch.zeros(end - start, device=device, dtype=torch.long)
            else:
                period_idx = torch.arange(start, end, device=device, dtype=torch.long)
            chunks.append(model.sample(U, period_idx).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def evaluate_samples(
    model: DescriptorFactorDiffusion,
    data: dict,
    device: torch.device,
    num_gen: int = cfg.SAMPLE_BATCHES * cfg.SAMPLES_PER_BATCH,
    gen_batch_size: int = cfg.SAMPLES_PER_BATCH,
) -> tuple:
    """Quick sanity metrics: mean/cov distance and learned A vs ground-truth T."""
    generated = generate_samples(model, data, device, num_gen, gen_batch_size)
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
    T = data["T"]
    # Compare subspaces (A and T may differ by rotation)
    _, s_a, _ = np.linalg.svd(A, full_matrices=False)
    _, s_t, _ = np.linalg.svd(T, full_matrices=False)
    metrics["A_singular_values"] = s_a.tolist()
    metrics["T_singular_values"] = s_t.tolist()

    return metrics, generated


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    timesteps: int,
    hidden_size: int,
) -> DescriptorFactorDiffusion:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DescriptorFactorDiffusion(
        num_assets=ckpt["num_assets"],
        num_descriptors=ckpt["num_descriptors"],
        num_factors=ckpt["num_factors"],
        timesteps=ckpt.get("num_timesteps", timesteps),
        hidden_size=ckpt.get("hidden_size", hidden_size),
        max_periods=ckpt.get("max_periods", 4096),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model


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
    parser = argparse.ArgumentParser(description="Train descriptor score diffusion model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory with returns.npy and descriptors.npy (generated if missing)",
    )
    parser.add_argument("--output_dir", type=str, default="model_results/descriptor_demo")
    parser.add_argument("--num_samples", type=int, default=cfg.TRAIN_SAMPLES)
    parser.add_argument("--num_assets", type=int, default=32)
    parser.add_argument("--num_descriptors", type=int, default=8)
    parser.add_argument("--num_factors", type=int, default=4)
    parser.add_argument(
        "--descriptor_mode",
        type=str,
        default="fixed",
        choices=["fixed", "perturbed", "random"],
    )
    parser.add_argument("--timesteps", type=int, default=cfg.TIMESTEPS)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--gpu", type=int, default=0, help="GPU id; use -1 for CPU")
    parser.add_argument("--save_every", type=int, default=cfg.SAVE_INTERVAL)
    parser.add_argument(
        "--num_gen",
        type=int,
        default=cfg.SAMPLE_BATCHES * cfg.SAMPLES_PER_BATCH,
        help="Number of samples to generate after training (or with --sample_only)",
    )
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        default=cfg.SAMPLES_PER_BATCH,
        help="Batch size for reverse diffusion sampling",
    )
    parser.add_argument(
        "--sample_only",
        action="store_true",
        help="Skip training; load checkpoint and generate samples only",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for --sample_only (default: output_dir/descriptor_model_final.pt)",
    )
    parser.add_argument(
        "--force_regenerate",
        action="store_true",
        help="Regenerate dataset even if data_dir already exists",
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

    data_dir = args.data_dir or str(
        Path(__file__).resolve().parent / "data" / "descriptor_demo"
    )
    data_path = Path(data_dir)
    needs_regenerate = args.force_regenerate or not (data_path / "returns.npy").exists()
    if not needs_regenerate and (data_path / "metadata.json").exists():
        existing = load_descriptor_dataset(data_path)
        if existing["metadata"].get("num_samples", 0) != args.num_samples:
            print(
                f"Existing dataset has {existing['metadata'].get('num_samples')} samples; "
                f"regenerating with {args.num_samples} ..."
            )
            needs_regenerate = True
        if existing["metadata"].get("descriptor_mode", "random") != args.descriptor_mode:
            print(
                f"Existing dataset mode={existing['metadata'].get('descriptor_mode')}; "
                f"regenerating with mode={args.descriptor_mode} ..."
            )
            needs_regenerate = True

    if needs_regenerate:
        print(
            f"Generating dataset at {data_path} "
            f"({args.num_samples} samples, mode={args.descriptor_mode}) ..."
        )
        data = generate_descriptor_dataset(
            num_samples=args.num_samples,
            num_assets=args.num_assets,
            num_descriptors=args.num_descriptors,
            num_factors=args.num_factors,
            descriptor_mode=args.descriptor_mode,
            seed=args.seed,
        )
        save_descriptor_dataset(data, data_path)
    else:
        data = load_descriptor_dataset(data_path)
        print(f"Loaded dataset from {data_path} ({data['metadata']['num_samples']} samples)")

    meta = data["metadata"]

    if args.sample_only:
        checkpoint_path = Path(
            args.checkpoint or Path(args.output_dir) / "descriptor_model_final.pt"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model = load_model_from_checkpoint(
            checkpoint_path, device, args.timesteps, args.hidden_size
        )
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Generating {args.num_gen} samples (batch size {args.gen_batch_size}) ...")
        metrics, generated = evaluate_samples(
            model,
            data,
            device,
            num_gen=args.num_gen,
            gen_batch_size=args.gen_batch_size,
        )
        print("Sample metrics:")
        for k, v in metrics.items():
            if k.endswith("singular_values"):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v:.6f}")
        sample_dir = save_generated_samples(generated, data, Path(args.output_dir), metrics)
        print(f"Saved {generated.shape[0]} samples to {sample_dir}")
        return

    period_indices = None
    max_periods = max(meta["num_samples"], 4096)
    if meta.get("descriptor_mode") == "fixed":
        period_indices = np.zeros(meta["num_samples"], dtype=np.int64)
        max_periods = 1

    dataset = DescriptorReturnDataset(data["returns"], data["descriptors"], period_indices)

    model = DescriptorFactorDiffusion(
        num_assets=meta["num_assets"],
        num_descriptors=meta["num_descriptors"],
        num_factors=meta["num_factors"],
        timesteps=args.timesteps,
        beta_schedule="cosine",
        hidden_size=args.hidden_size,
        max_periods=max_periods,
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
        f"Training on {device}: "
        f"d={meta['num_assets']} m={meta['num_descriptors']} k={meta['num_factors']} "
        f"n={meta['num_samples']} mode={meta.get('descriptor_mode', '?')} timesteps={args.timesteps}"
    )
    trainer.train()

    model.to(device)
    metrics, generated = evaluate_samples(
        model,
        data,
        device,
        num_gen=args.num_gen,
        gen_batch_size=args.gen_batch_size,
    )
    print("Post-training metrics:")
    for k, v in metrics.items():
        if k.endswith("singular_values"):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.6f}")

    sample_dir = save_generated_samples(generated, data, Path(args.output_dir), metrics)
    print(f"Saved {generated.shape[0]} samples to {sample_dir}")


if __name__ == "__main__":
    main()
