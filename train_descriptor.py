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
    compute_return_norm_stats,
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
    deterministic: bool = False,
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
            chunks.append(
                model.sample(U, period_idx, deterministic=deterministic).cpu().numpy()
            )
    return np.concatenate(chunks, axis=0)


def _orthonormal_basis(M: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Orthonormal basis (m x r) of the column space of M (m x k), via SVD."""
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    rank = int((s > tol * max(s[0], 1.0)).sum())
    return U[:, :rank]


def subspace_recovery_metrics(A: np.ndarray, T: np.ndarray) -> dict:
    """
    Rotation/scaling-invariant comparison of the learned descriptor-to-factor map
    A against the ground-truth T (Section 3.2 identifiability question).

    The identifiable object is the column space (loading space), so A and T are
    only defined up to a k x k change of basis. We therefore compare span(A) and
    span(T) via principal angles, the projector distance, and an orthogonal
    Procrustes alignment of their orthonormal bases. Singular-value comparison
    alone is blind to the column space and is kept only for reference.
    """
    Qa = _orthonormal_basis(A)
    Qt = _orthonormal_basis(T)
    ra, rt = Qa.shape[1], Qt.shape[1]
    k = min(ra, rt)

    # Principal angles: singular values of Qa^T Qt are the cosines.
    cos_angles = np.linalg.svd(Qa.T @ Qt, compute_uv=False)
    cos_angles = np.clip(cos_angles[:k], -1.0, 1.0)
    angles = np.arccos(cos_angles)  # radians, ascending in cos -> descending order

    # Projector (Frobenius) distance between the two subspaces.
    # For equal-dim subspaces, ||P_A - P_T||_F = sqrt(2) * ||sin(theta)||_2.
    Pa = Qa @ Qa.T
    Pt = Qt @ Qt.T
    projector_error = float(np.linalg.norm(Pa - Pt))
    # Max possible projector distance between two k-dim subspaces is sqrt(2k).
    projector_error_normalized = projector_error / float(np.sqrt(2.0 * k)) if k else 0.0

    # Orthogonal Procrustes: best rotation aligning the bases; residual in [0, 1].
    W, _, Zt = np.linalg.svd(Qa.T @ Qt, full_matrices=False)
    R = W @ Zt
    procrustes_residual = float(np.linalg.norm(Qa @ R - Qt) / max(np.linalg.norm(Qt), 1e-12))

    return {
        "A_rank": ra,
        "T_rank": rt,
        "principal_angles_deg": np.degrees(angles).tolist(),
        "mean_principal_angle_deg": float(np.degrees(angles).mean()),
        "max_principal_angle_deg": float(np.degrees(angles).max()),
        "grassmann_distance": float(np.linalg.norm(angles)),
        "subspace_projector_error": projector_error,
        "subspace_projector_error_normalized": projector_error_normalized,
        "procrustes_residual": procrustes_residual,
        "A_singular_values": np.linalg.svd(A, compute_uv=False).tolist(),
        "T_singular_values": np.linalg.svd(T, compute_uv=False).tolist(),
    }


def evaluate_samples(
    model: DescriptorFactorDiffusion,
    data: dict,
    device: torch.device,
    num_gen: int = cfg.SAMPLE_BATCHES * cfg.SAMPLES_PER_BATCH,
    gen_batch_size: int = cfg.SAMPLES_PER_BATCH,
    deterministic: bool = False,
) -> tuple:
    """Quick sanity metrics: mean/cov distance and learned A vs ground-truth T."""
    generated = generate_samples(
        model, data, device, num_gen, gen_batch_size, deterministic=deterministic
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
    T = data["T"]
    # Identifiability: compare span(A) vs span(T) up to rotation/scaling.
    metrics["subspace_recovery"] = subspace_recovery_metrics(A, T)

    return metrics, generated


def print_metrics(metrics: dict, indent: int = 2) -> None:
    pad = " " * indent
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            print_metrics(v, indent + 2)
        elif isinstance(v, list):
            print(f"{pad}{k}: {v}")
        elif isinstance(v, float):
            print(f"{pad}{k}: {v:.6f}")
        else:
            print(f"{pad}{k}: {v}")


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    timesteps: int,
    hidden_size: int,
    clip_denoised: float = 3.0,
    mean_loss_weight: float = 0.0,
) -> DescriptorFactorDiffusion:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DescriptorFactorDiffusion(
        num_assets=ckpt["num_assets"],
        num_descriptors=ckpt["num_descriptors"],
        num_factors=ckpt["num_factors"],
        timesteps=ckpt.get("num_timesteps", timesteps),
        hidden_size=ckpt.get("hidden_size", hidden_size),
        max_periods=ckpt.get("max_periods", 4096),
        clip_denoised=ckpt.get("clip_denoised", clip_denoised),
        mean_loss_weight=ckpt.get("mean_loss_weight", mean_loss_weight),
    )
    model.load_state_dict(ckpt["model"])
    if "return_mean" in ckpt and "return_scale" in ckpt:
        model.set_return_normalization(ckpt["return_mean"], float(ckpt["return_scale"].item()))
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
    parser.add_argument(
        "--normalize_returns",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Standardize returns (per-asset mean, global std) before diffusion",
    )
    parser.add_argument(
        "--mean_loss_weight",
        type=float,
        default=0,
        help="Weight for auxiliary zero-mean x0 loss (0 disables)",
    )
    parser.add_argument(
        "--clip_denoised",
        type=float,
        default=0,
        help="Clamp predicted x0 to [-clip, clip] in normalized space (0 disables)",
    )
    parser.add_argument(
        "--deterministic_sample",
        action="store_true",
        help="Use posterior mean only when sampling (no reverse-step noise)",
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
            checkpoint_path,
            device,
            args.timesteps,
            args.hidden_size,
            clip_denoised=args.clip_denoised,
            mean_loss_weight=args.mean_loss_weight,
        )
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Generating {args.num_gen} samples (batch size {args.gen_batch_size}) ...")
        metrics, generated = evaluate_samples(
            model,
            data,
            device,
            num_gen=args.num_gen,
            gen_batch_size=args.gen_batch_size,
            deterministic=args.deterministic_sample,
        )
        print("Sample metrics:")
        print_metrics(metrics)
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
        clip_denoised=args.clip_denoised,
        mean_loss_weight=args.mean_loss_weight,
    )

    if args.normalize_returns:
        ret_mean, ret_scale = compute_return_norm_stats(data["returns"])
        model.set_return_normalization(ret_mean, ret_scale)
        print(
            f"Return normalization: mean_l2={np.linalg.norm(ret_mean):.6f}, scale={ret_scale:.6f}"
        )
    else:
        model.set_return_normalization(np.zeros(meta["num_assets"], dtype=np.float32), 1.0)

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
        f"n={meta['num_samples']} mode={meta.get('descriptor_mode', '?')} timesteps={args.timesteps} "
        f"normalize={args.normalize_returns} mean_loss={args.mean_loss_weight} "
        f"clip={args.clip_denoised}"
    )
    trainer.train()

    model.to(device)
    metrics, generated = evaluate_samples(
        model,
        data,
        device,
        num_gen=args.num_gen,
        gen_batch_size=args.gen_batch_size,
        deterministic=args.deterministic_sample,
    )
    print("Post-training metrics:")
    print_metrics(metrics)

    sample_dir = save_generated_samples(generated, data, Path(args.output_dir), metrics)
    print(f"Saved {generated.shape[0]} samples to {sample_dir}")


if __name__ == "__main__":
    main()
