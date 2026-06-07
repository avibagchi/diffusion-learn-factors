#!/usr/bin/env python3
"""
Evaluate generated samples against training data (U-Net simulation or descriptor model).

Supports:
  - U-Net: training_data.npy + sample_batch*.npy + loading_matrix.npy
  - Descriptor: returns.npy + generated.npy + T.npy (or beta.npy)
  - Quick mean/cov metrics (like train_descriptor.py)
  - Paper-style latent subspace and eigenvalue errors (when ground truth is available)
  - Optional marginal histogram plot

Examples:
  export PYTHONPATH=.

  # U-Net simulation
  python scripts/eval_simulation.py \\
    --data_dir simulation_experiment_data/small \\
    --samples_dir samples/dfm_training_data_ts1780873060_seed3407

  # Descriptor model
  python scripts/eval_simulation.py \\
    --data_dir data/descriptor_demo \\
    --samples_dir model_results/descriptor_demo/samples/generated.npy
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def svd(A: np.ndarray, k: int):
    """Top-k SVD components (matches eval.simulation_eval.svd)."""
    U, S, VT = np.linalg.svd(A)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    A_k = U_k @ S_k @ VT[:k, :]
    return np.diag(S_k), A_k


def load_returns_2d(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], -1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 2D or 3D array in {path}, got shape {arr.shape}")


def load_generated_samples(samples_path: Path) -> np.ndarray:
    if samples_path.is_dir():
        batches = sorted(samples_path.glob("sample_batch*.npy"))
        if batches:
            return np.concatenate([load_returns_2d(p) for p in batches], axis=0)
        generated = samples_path / "generated.npy"
        if generated.exists():
            return load_returns_2d(generated)
        raise FileNotFoundError(
            f"No sample_batch*.npy or generated.npy found in {samples_path}"
        )
    return load_returns_2d(samples_path)


def _read_metadata(data_dir: Path) -> dict:
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def resolve_paths(
    data_dir: Optional[Path],
    train_path: Optional[Path],
    samples_path: Path,
) -> Tuple[Path, Optional[Path], str]:
    if train_path is None:
        if data_dir is None:
            raise ValueError("Provide --train_path or --data_dir")
        if (data_dir / "training_data.npy").exists():
            train_path = data_dir / "training_data.npy"
            data_type = "simulation"
        elif (data_dir / "returns.npy").exists():
            train_path = data_dir / "returns.npy"
            data_type = "descriptor"
        else:
            raise FileNotFoundError(
                f"No training_data.npy or returns.npy found in {data_dir}"
            )
    else:
        data_type = "descriptor" if train_path.name == "returns.npy" else "simulation"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples path not found: {samples_path}")

    gt_dir = data_dir if data_dir is not None else train_path.parent
    has_gt = (gt_dir / "loading_matrix.npy").exists() or (gt_dir / "T.npy").exists()
    return train_path, gt_dir if has_gt else None, data_type


def load_simulation_ground_truth(gt_dir: Path, num_factors: Optional[int]) -> Tuple[np.ndarray, np.ndarray, int]:
    A = np.load(gt_dir / "loading_matrix.npy")
    meta = _read_metadata(gt_dir)

    noise_scale = float(meta.get("noise_scale", 0.1))
    k = num_factors or int(meta.get("num_factors", A.shape[0]))
    gt_mean = np.zeros(A.shape[1])
    gt_cov = A.T @ A + noise_scale * np.eye(A.shape[1])
    return gt_mean, gt_cov, k


def load_descriptor_ground_truth(gt_dir: Path, num_factors: Optional[int]) -> Tuple[np.ndarray, np.ndarray, int]:
    meta = _read_metadata(gt_dir)
    T = np.load(gt_dir / "T.npy")
    k = num_factors or int(meta.get("num_factors", T.shape[1]))
    mode = meta.get("descriptor_mode", "fixed")

    if mode == "fixed" and (gt_dir / "beta.npy").exists():
        beta = np.load(gt_dir / "beta.npy")
        gt_mean = np.zeros(beta.shape[0])
        gt_cov = beta @ beta.T
        return gt_mean, gt_cov, k

    if not (gt_dir / "descriptors.npy").exists():
        raise FileNotFoundError(
            f"Descriptor ground truth needs beta.npy (fixed mode) or descriptors.npy in {gt_dir}"
        )

    U = np.load(gt_dir / "descriptors.npy")
    d = U.shape[1]
    gt_cov = np.zeros((d, d), dtype=np.float64)
    for i in range(U.shape[0]):
        beta_i = U[i] @ T
        gt_cov += beta_i @ beta_i.T
    gt_cov /= U.shape[0]
    return np.zeros(d), gt_cov, k


def load_ground_truth(
    gt_dir: Path, num_factors: Optional[int], data_type: str
) -> Tuple[np.ndarray, np.ndarray, int]:
    if data_type == "descriptor":
        return load_descriptor_ground_truth(gt_dir, num_factors)
    return load_simulation_ground_truth(gt_dir, num_factors)


def basic_metrics(real: np.ndarray, gen: np.ndarray) -> dict:
    real_mean = real.mean(0)
    gen_mean = gen.mean(0)
    real_cov = np.cov(real.T)
    gen_cov = np.cov(gen.T)
    real_cov_fro = float(np.linalg.norm(real_cov))
    gen_cov_fro = float(np.linalg.norm(gen_cov))

    return {
        "num_train": int(real.shape[0]),
        "num_generated": int(gen.shape[0]),
        "num_assets": int(real.shape[1]),
        "real_mean_l2": float(np.linalg.norm(real_mean)),
        "gen_mean_l2": float(np.linalg.norm(gen_mean)),
        "mean_diff_l2": float(np.linalg.norm(real_mean - gen_mean)),
        "real_cov_fro": real_cov_fro,
        "gen_cov_fro": gen_cov_fro,
        "cov_diff_fro": float(np.linalg.norm(real_cov - gen_cov)),
        "cov_ratio_gen_over_real": gen_cov_fro / real_cov_fro if real_cov_fro > 0 else float("nan"),
        "cov_diff_relative": float(np.linalg.norm(real_cov - gen_cov) / real_cov_fro)
        if real_cov_fro > 0
        else float("nan"),
        "std_ratio_gen_over_real": float(gen.std(0).mean() / real.std(0).mean())
        if real.std(0).mean() > 0
        else float("nan"),
    }


def top_singular_values(matrix: np.ndarray, k: int) -> np.ndarray:
    return np.linalg.svd(matrix, compute_uv=False)[:k]


def subspace_metrics(real: np.ndarray, gen: np.ndarray, gt_cov: np.ndarray, k: int) -> dict:
    _, real_sub = svd(np.cov(real.T), k)
    _, gen_sub = svd(np.cov(gen.T), k)
    _, gt_sub = svd(gt_cov, k)

    real_evals = top_singular_values(np.cov(real.T), k)
    gen_evals = top_singular_values(np.cov(gen.T), k)
    gt_evals = top_singular_values(gt_cov, k)

    gt_norm = np.linalg.norm(gt_sub)
    real_sub_err = float(np.linalg.norm(real_sub - gt_sub) / gt_norm)
    gen_sub_err = float(np.linalg.norm(gen_sub - gt_sub) / gt_norm)
    subspace_ratio = gen_sub_err / real_sub_err if real_sub_err > 0 else float("nan")

    real_ev_err = float(np.abs(real_evals / gt_evals - 1).mean())
    gen_ev_err = float(np.abs(gen_evals / gt_evals - 1).mean())
    eigenvalue_ratio = gen_ev_err / real_ev_err if real_ev_err > 0 else float("nan")

    return {
        "num_factors": k,
        "subspace_error_training": real_sub_err,
        "subspace_error_generated": gen_sub_err,
        "subspace_error_ratio": subspace_ratio,
        "eigenvalue_error_training": real_ev_err,
        "eigenvalue_error_generated": gen_ev_err,
        "eigenvalue_error_ratio": eigenvalue_ratio,
        "ground_truth_eigenvalues": gt_evals.tolist(),
        "training_top_eigenvalues": real_evals.tolist(),
        "generated_top_eigenvalues": gen_evals.tolist(),
    }


def _load_histplot_fn():
    """Import plot helper without pulling in eval/__init__.py."""
    module_path = ROOT / "eval" / "simulation_eval.py"
    spec = importlib.util.spec_from_file_location("simulation_eval", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.comparision_histplot_simulation


def save_histogram(
    stock_i: int,
    train_path: Path,
    generated: np.ndarray,
    gt_mean: np.ndarray,
    gt_cov: np.ndarray,
    output_path: Path,
    show: bool,
) -> None:
    comparision_histplot_simulation = _load_histplot_fn()

    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tmp_gen = output_path.parent / "_eval_generated_tmp.npy"
    np.save(tmp_gen, generated)
    try:
        comparision_histplot_simulation(
            stock_i=stock_i,
            training_data_path=str(train_path),
            generated_data_path=str(tmp_gen),
            ground_truth_mean=gt_mean,
            ground_truth_cov=gt_cov,
        )
        if not show:
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved histogram to {output_path}")
    finally:
        tmp_gen.unlink(missing_ok=True)


def print_metrics(metrics: dict) -> None:
    print(f"\n=== Basic metrics ({metrics.get('data_type', 'unknown')}) ===")
    for key in (
        "num_train",
        "num_generated",
        "num_assets",
        "std_ratio_gen_over_real",
        "cov_ratio_gen_over_real",
        "cov_diff_relative",
        "mean_diff_l2",
        "real_mean_l2",
        "gen_mean_l2",
        "real_cov_fro",
        "gen_cov_fro",
        "cov_diff_fro",
    ):
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                print(f"  {key}: {val:.6f}")
            else:
                print(f"  {key}: {val}")

    if "subspace_error_ratio" in metrics:
        print("\n=== Ground-truth subspace / eigenvalue metrics ===")
        print("  (ratio < 1 means generated beats training sample)")
        for key in (
            "num_factors",
            "subspace_error_training",
            "subspace_error_generated",
            "subspace_error_ratio",
            "eigenvalue_error_training",
            "eigenvalue_error_generated",
            "eigenvalue_error_ratio",
        ):
            val = metrics[key]
            if isinstance(val, float):
                print(f"  {key}: {val:.6f}")
            else:
                print(f"  {key}: {val}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated samples (U-Net simulation or descriptor model)"
    )
    parser.add_argument(
        "--samples_dir",
        type=str,
        required=True,
        help="Directory with sample_batch*.npy, generated.npy, or a single .npy file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Data dir: simulation (training_data.npy) or descriptor (returns.npy)",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default=None,
        help="Path to training_data.npy (overrides --data_dir/training_data.npy)",
    )
    parser.add_argument("--num_factors", type=int, default=None, help="Latent dim k for subspace eval")
    parser.add_argument("--stock", type=int, default=0, help="Asset index for histogram plot")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save marginal histogram (requires ground truth from data_dir)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot interactively instead of saving",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save metrics.json and optional histogram",
    )
    args = parser.parse_args()

    samples_path = Path(args.samples_dir)
    data_dir = Path(args.data_dir) if args.data_dir else None
    train_path = Path(args.train_path) if args.train_path else None

    train_path, gt_dir, data_type = resolve_paths(data_dir, train_path, samples_path)
    real = load_returns_2d(train_path)
    gen = load_generated_samples(samples_path)

    if real.shape[1] != gen.shape[1]:
        raise ValueError(
            f"Asset dimension mismatch: training {real.shape[1]} vs generated {gen.shape[1]}"
        )

    metrics = basic_metrics(real, gen)
    metrics["data_type"] = data_type

    gt_mean = gt_cov = None
    if gt_dir is not None:
        gt_mean, gt_cov, k = load_ground_truth(gt_dir, args.num_factors, data_type)
        metrics.update(subspace_metrics(real, gen, gt_cov, k))
    elif args.plot:
        raise ValueError("--plot requires --data_dir with loading_matrix.npy or T.npy")

    print_metrics(metrics)

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics to {metrics_path}")

    if args.plot:
        if out_dir is None:
            out_dir = Path("eval_results")
            out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / f"marginal_asset_{args.stock}.png"
        save_histogram(
            args.stock,
            train_path,
            gen,
            gt_mean,
            gt_cov,
            plot_path,
            show=args.show,
        )


if __name__ == "__main__":
    main()
