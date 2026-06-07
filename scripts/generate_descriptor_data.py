#!/usr/bin/env python3
"""Generate a synthetic returns + descriptors dataset for descriptor score training."""

import argparse
from pathlib import Path

from config import config as cfg
from diffusion_factor_model.descriptor_score_model import generate_descriptor_dataset, save_descriptor_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate descriptor factor dataset")
    parser.add_argument("--output_dir", type=str, default="data/descriptor_demo")
    parser.add_argument("--num_samples", type=int, default=cfg.TRAIN_SAMPLES)
    parser.add_argument("--num_assets", type=int, default=32)
    parser.add_argument("--num_descriptors", type=int, default=8)
    parser.add_argument("--num_factors", type=int, default=4)
    parser.add_argument(
        "--descriptor_mode",
        type=str,
        default="fixed",
        choices=["fixed", "perturbed", "random"],
        help="fixed = same U every period (recommended); random = hardest",
    )
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    args = parser.parse_args()

    data = generate_descriptor_dataset(
        num_samples=args.num_samples,
        num_assets=args.num_assets,
        num_descriptors=args.num_descriptors,
        num_factors=args.num_factors,
        descriptor_mode=args.descriptor_mode,
        seed=args.seed,
    )
    out = save_descriptor_dataset(data, args.output_dir)
    meta = data["metadata"]
    print(f"Saved dataset to {out}")
    print(
        f"  samples={meta['num_samples']} assets={meta['num_assets']} "
        f"descriptors={meta['num_descriptors']} factors={meta['num_factors']} "
        f"mode={meta['descriptor_mode']}"
    )
    print(f"  returns shape: {data['returns'].shape}")
    print(f"  descriptors shape: {data['descriptors'].shape}")


if __name__ == "__main__":
    main()
