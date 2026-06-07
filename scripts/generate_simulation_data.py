#!/usr/bin/env python3
"""
Generate simulated returns for the original 2D U-Net diffusion factor model.

Uses GaussianLatentSampler2D_Finance from the paper setup:
  x = F @ A  (+ optional idiosyncratic noise)
  saved as (num_samples, height, width) for train.py
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

PRESETS = {
    "paper": {
        "num_samples": 2048,
        "num_factors": 16,
        "height": 32,
        "width": 64,
        "noise_scale": 0.1,
    },
    "small": {
        "num_samples": 512,
        "num_factors": 4,
        "height": 8,
        "width": 8,
        "noise_scale": 0.1,
    },
}


class GaussianLatentSampler2D_Finance:
    """Numpy-only copy of the paper's simulated return generator."""

    def __init__(self, d_inner: int, image_size: tuple, seed: int = 42):
        self.image_size = image_size
        self.d_inner = d_inner
        self.d_outer = image_size[0] * image_size[1]
        rng = np.random.default_rng(seed)
        self.A = rng.standard_normal((self.d_inner, self.d_outer))

    def generate_data(
        self,
        N: int,
        latent_mean: np.ndarray,
        latent_cov: np.ndarray,
        noise_mean: Optional[np.ndarray] = None,
        noise_cov: Optional[np.ndarray] = None,
        sort_var: bool = True,
    ):
        latent_diag = np.diag(latent_cov)
        factor_var = (self.A ** 2).T @ latent_diag
        if noise_cov is not None:
            noise_diag = np.diag(noise_cov)
            diagonal_elements = factor_var + noise_diag
        else:
            diagonal_elements = factor_var

        if sort_var:
            sorted_indices = np.argsort(diagonal_elements)[::-1]

        z = np.random.standard_normal((N, self.d_inner))
        factor = latent_mean @ self.A + z * np.sqrt(latent_diag) @ self.A

        if noise_cov is not None:
            noise_diag = np.diag(noise_cov)
            noise = np.random.standard_normal((N, self.d_outer)) * np.sqrt(noise_diag)
            if noise_mean is not None:
                noise = noise + noise_mean
            x = factor + noise
        else:
            x = factor

        if sort_var:
            factor = factor[:, sorted_indices]
            x = x[:, sorted_indices]

        h, w = self.image_size
        return factor, x.reshape((N, h, w))


def generate_simulation_dataset(
    num_samples: int = 2048,
    num_factors: int = 16,
    height: int = 32,
    width: int = 64,
    noise_scale: float = 0.1,
    factor_scale: float = 1.0,
    seed: int = 42,
) -> dict:
    """Simulate factor-model returns reshaped to (N, height, width)."""
    np.random.seed(seed)

    d = height * width
    sampler = GaussianLatentSampler2D_Finance(
        d_inner=num_factors, image_size=(height, width), seed=seed
    )

    latent_mean = np.zeros(num_factors)
    latent_cov = factor_scale * np.eye(num_factors)

    if noise_scale > 0:
        noise_mean = np.zeros(d)
        noise_cov = noise_scale * np.eye(d)
    else:
        noise_mean = None
        noise_cov = None

    factors, returns = sampler.generate_data(
        num_samples,
        latent_mean,
        latent_cov,
        noise_mean=noise_mean,
        noise_cov=noise_cov,
        sort_var=True,
    )

    return {
        "returns": returns.astype(np.float32),
        "factors": factors.astype(np.float32),
        "loading_matrix": sampler.A.astype(np.float32),
        "metadata": {
            "num_samples": num_samples,
            "num_factors": num_factors,
            "height": height,
            "width": width,
            "num_assets": d,
            "noise_scale": noise_scale,
            "factor_scale": factor_scale,
            "seed": seed,
            "model": "GaussianLatentSampler2D_Finance",
            "returns_shape": "(num_samples, height, width)",
        },
    }


def save_simulation_dataset(data: dict, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "training_data.npy", data["returns"])
    np.save(output_dir / "factors.npy", data["factors"])
    np.save(output_dir / "loading_matrix.npy", data["loading_matrix"])

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(data["metadata"], f, indent=2)

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate U-Net simulation training data")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="simulation_experiment_data/demo",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        default=None,
        help="paper=(2048,32,64,k=16) or small=(512,8,8,k=4) for quick tests",
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_factors", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=None,
        help="Idiosyncratic noise std (0 = residual-free)",
    )
    parser.add_argument("--factor_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = PRESETS["paper"].copy()
    if args.preset:
        cfg.update(PRESETS[args.preset])

    for key in ("num_samples", "num_factors", "height", "width", "noise_scale"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    data = generate_simulation_dataset(
        num_samples=cfg["num_samples"],
        num_factors=cfg["num_factors"],
        height=cfg["height"],
        width=cfg["width"],
        noise_scale=cfg["noise_scale"],
        factor_scale=args.factor_scale,
        seed=args.seed,
    )
    out = save_simulation_dataset(data, args.output_dir)
    meta = data["metadata"]

    print(f"Saved simulation data to {out}")
    print(f"  training_data.npy: {data['returns'].shape}  (samples, height, width)")
    print(
        f"  factors={meta['num_factors']} assets={meta['num_assets']} "
        f"noise_scale={meta['noise_scale']} seed={meta['seed']}"
    )
    print(f"\nTrain with:")
    print(f"  python train.py --data_path {out / 'training_data.npy'} --gpu 0")


if __name__ == "__main__":
    main()
