"""
Descriptor-based score decomposition (Section 3.2, residual-free case).

Score estimator (SS):
    s_θ(r, t, i) = -1/h_t * r + (α_t/h_t) * U_i A · Network(Q_i^T r, t, i)

where U_i = Q_i V_i (thin QR) and A ∈ R^{m×k} estimates the time-invariant T.
The model outputs predicted noise ε for DDPM training: pred_eps = -s_θ * sqrt(h_t).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def thin_qr(U: torch.Tensor) -> torch.Tensor:
    """Return Q from thin QR of U with orthonormal columns. U: (..., d, m), d >= m."""
    Q, _ = torch.linalg.qr(U, mode="reduced")
    return Q


class RMSNorm1d(nn.Module):
    """RMSNorm for vector features (batch, dim)."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.g * self.scale


def init_descriptor_A(num_descriptors: int, num_factors: int) -> torch.Tensor:
    """Orthonormal columns, unit scale — matches typical QR ground-truth T."""
    A = torch.empty(num_descriptors, num_factors)
    nn.init.orthogonal_(A)
    return A


class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


class DescriptorSubspaceNetwork(nn.Module):
    """Network(Q_i^T r, t, i) -> R^k."""

    def __init__(
        self,
        num_descriptors: int,
        num_factors: int,
        hidden_size: int = 128,
        max_periods: int = 4096,
        period_embed_dim: int = 16,
    ):
        super().__init__()
        self.period_embed = nn.Embedding(max_periods, period_embed_dim)
        self.max_periods = max_periods
        self.time_embed = SinusoidalTimeEmbed(hidden_size)
        in_dim = num_descriptors + hidden_size + period_embed_dim
        self.fc1 = nn.Linear(in_dim, hidden_size)
        self.norm1 = RMSNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = RMSNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_factors)

    def forward(self, z: torch.Tensor, t: torch.Tensor, period_idx: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        p_emb = self.period_embed(period_idx.clamp(min=0, max=self.max_periods - 1))
        x = torch.cat([z, t_emb, p_emb], dim=-1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        return self.fc3(x)


class DescriptorScoreNetwork(nn.Module):
    """
    Implements s_θ(r, t, i) = -r/h_t + (α_t/h_t) U_i A · g_ζ(Q_i^T r, t, i).
    """

    def __init__(
        self,
        num_assets: int,
        num_descriptors: int,
        num_factors: int,
        hidden_size: int = 128,
        max_periods: int = 4096,
    ):
        super().__init__()
        self.num_assets = num_assets
        self.num_descriptors = num_descriptors
        self.num_factors = num_factors

        self.A = nn.Parameter(init_descriptor_A(num_descriptors, num_factors))
        self.subspace_net = DescriptorSubspaceNetwork(
            num_descriptors,
            num_factors,
            hidden_size=hidden_size,
            max_periods=max_periods,
        )

    def score(
        self,
        r: torch.Tensor,
        t: torch.Tensor,
        U: torch.Tensor,
        period_idx: torch.Tensor,
        h_t: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        Q = thin_qr(U)
        z = torch.bmm(Q.transpose(1, 2), r.unsqueeze(-1)).squeeze(-1)

        h = h_t.unsqueeze(-1).clamp(min=1e-8)
        alpha = alpha_t.unsqueeze(-1)

        linear_score = -r / h

        g_out = self.subspace_net(z, t, period_idx)
        UA = torch.matmul(U, self.A)
        factor_term = torch.bmm(UA, g_out.unsqueeze(-1)).squeeze(-1)
        subspace_score = (alpha / h) * factor_term

        return linear_score + subspace_score

    def forward(
        self,
        r: torch.Tensor,
        t: torch.Tensor,
        U: torch.Tensor,
        period_idx: torch.Tensor,
        h_t: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        score = self.score(r, t, U, period_idx, h_t, alpha_t)
        sqrt_h = torch.sqrt(h_t.clamp(min=1e-8)).unsqueeze(-1)
        return -score * sqrt_h


class DescriptorFactorDiffusion(nn.Module):
    """DDPM wrapper for descriptor-conditioned vector returns."""

    def __init__(
        self,
        num_assets: int,
        num_descriptors: int,
        num_factors: int,
        timesteps: int = 200,
        beta_schedule: str = "cosine",
        hidden_size: int = 128,
        max_periods: int = 4096,
        clip_denoised: float = 3.0,
        mean_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.num_assets = num_assets
        self.num_descriptors = num_descriptors
        self.num_factors = num_factors
        self.num_timesteps = timesteps
        self.hidden_size = hidden_size
        self.max_periods = max_periods
        self.clip_denoised = clip_denoised
        self.mean_loss_weight = mean_loss_weight

        self.register_buffer("return_mean", torch.zeros(num_assets))
        self.register_buffer("return_scale", torch.tensor(1.0))

        self.score_net = DescriptorScoreNetwork(
            num_assets,
            num_descriptors,
            num_factors,
            hidden_size=hidden_size,
            max_periods=max_periods,
        )
        self._setup_schedule(timesteps, beta_schedule)

    def set_return_normalization(
        self,
        mean: np.ndarray | torch.Tensor,
        scale: float,
    ) -> None:
        mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        if mean_t.shape[0] != self.num_assets:
            raise ValueError(
                f"return mean length {mean_t.shape[0]} != num_assets {self.num_assets}"
            )
        self.return_mean.copy_(mean_t)
        self.return_scale.fill_(max(float(scale), 1e-6))

    def normalize_returns(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.return_mean) / self.return_scale

    def denormalize_returns(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.return_scale + self.return_mean

    def _clip_x_start(self, x_start: torch.Tensor) -> torch.Tensor:
        if self.clip_denoised and self.clip_denoised > 0:
            return x_start.clamp(-self.clip_denoised, self.clip_denoised)
        return x_start

    def _setup_schedule(self, timesteps: int, beta_schedule: str):
        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, timesteps)
        else:
            steps = timesteps + 1
            t = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos((t / timesteps + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        h_t = 1.0 - alphas_cumprod
        alpha_t = torch.sqrt(alphas_cumprod)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("h_t", h_t.float())
        self.register_buffer("alpha_t", alpha_t.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).float()
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod).float()
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1).float()
        )

        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.float())
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)).float(),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)).float(),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)).float(),
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        noise = torch.randn_like(x0) if noise is None else noise
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    def p_losses(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        U: torch.Tensor,
        period_idx: torch.Tensor,
    ) -> torch.Tensor:
        x_t, noise = self.q_sample(x0, t)
        h = self.h_t[t]
        alpha = self.alpha_t[t]
        pred_noise = self.score_net(x_t, t, U, period_idx, h, alpha)
        loss = F.mse_loss(pred_noise, noise)
        if self.mean_loss_weight > 0:
            x0_pred = self.predict_start_from_noise(x_t, t, pred_noise)
            x0_pred = self._clip_x_start(x0_pred)
            loss = loss + self.mean_loss_weight * x0_pred.pow(2).mean()
        return loss

    def forward(
        self,
        x0: torch.Tensor,
        U: torch.Tensor,
        period_idx: torch.Tensor,
    ) -> torch.Tensor:
        x0 = self.normalize_returns(x0)
        b = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x0.device, dtype=torch.long)
        return self.p_losses(x0, t, U, period_idx)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
        b = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, U, period_idx, deterministic: bool = False):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        h = self.h_t[batched_times]
        alpha = self.alpha_t[batched_times]
        pred_noise = self.score_net(x, batched_times, U, period_idx, h, alpha)
        x_start = self.predict_start_from_noise(x, batched_times, pred_noise)
        x_start = self._clip_x_start(x_start)
        model_mean, _, model_log_variance = self.q_posterior(x_start, x, batched_times)
        if deterministic or t == 0:
            return model_mean
        noise = torch.randn_like(x)
        return model_mean + (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def sample(
        self,
        U: torch.Tensor,
        period_idx: torch.Tensor,
        num_steps: Optional[int] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Reverse diffusion conditioned on descriptor matrices U."""
        device = U.device
        b, d, _ = U.shape
        num_steps = num_steps or self.num_timesteps

        x = torch.randn(b, d, device=device)
        if num_steps == self.num_timesteps:
            times = range(self.num_timesteps - 1, -1, -1)
        else:
            times = torch.linspace(self.num_timesteps - 1, 0, num_steps, device=device).long().tolist()

        for t in times:
            x = self.p_sample(x, int(t), U, period_idx, deterministic=deterministic)

        return self.denormalize_returns(x)


class DescriptorReturnDataset(Dataset):
    """Dataset of (returns, descriptors, period_index) tuples."""

    def __init__(
        self,
        returns: np.ndarray | torch.Tensor,
        descriptors: np.ndarray | torch.Tensor,
        period_indices: np.ndarray | torch.Tensor | None = None,
    ):
        self.returns = torch.as_tensor(returns, dtype=torch.float32)
        self.descriptors = torch.as_tensor(descriptors, dtype=torch.float32)
        if self.returns.ndim != 2:
            raise ValueError("returns must have shape (n_samples, num_assets)")
        if self.descriptors.ndim != 3:
            raise ValueError("descriptors must have shape (n_samples, num_assets, num_descriptors)")

        n = self.returns.shape[0]
        if self.descriptors.shape[0] != n:
            raise ValueError("returns and descriptors must have the same number of samples")

        if period_indices is None:
            period_indices = np.arange(n, dtype=np.int64)
        self.period_indices = torch.as_tensor(period_indices, dtype=torch.long)

    def __len__(self) -> int:
        return self.returns.shape[0]

    def __getitem__(self, idx: int):
        return self.returns[idx], self.descriptors[idx], self.period_indices[idx]


def _normalize_descriptor_columns(U: np.ndarray) -> np.ndarray:
    col_norms = np.linalg.norm(U, axis=0, keepdims=True).clip(min=1e-6)
    return (U / col_norms).astype(np.float32)


def generate_descriptor_dataset(
    num_samples: int = 2048,
    num_assets: int = 32,
    num_descriptors: int = 8,
    num_factors: int = 4,
    descriptor_mode: str = "fixed",
    descriptor_noise: float = 0.05,
    descriptor_perturbation: float = 0.02,
    factor_scale: float = 1.0,
    seed: int = 3407,
) -> dict:
    """
    Simulate residual-free returns R_i = U_i T F_i.

    descriptor_mode:
      - "fixed": same U for every period (recommended; time-invariant loadings beta = U T)
      - "perturbed": U_base plus small column-normalized noise each period
      - "random": fresh random U_i each period (harder; stress test only)
    """
    if descriptor_mode not in {"fixed", "perturbed", "random"}:
        raise ValueError(f"Unknown descriptor_mode: {descriptor_mode}")

    rng = np.random.default_rng(seed)

    T, _ = np.linalg.qr(rng.standard_normal((num_descriptors, num_descriptors)))
    T = (T[:, :num_factors] * factor_scale).astype(np.float32)

    U_base = _normalize_descriptor_columns(rng.standard_normal((num_assets, num_descriptors)))
    beta = (U_base @ T).astype(np.float32)

    factors = rng.standard_normal((num_samples, num_factors)).astype(np.float32)
    returns = np.zeros((num_samples, num_assets), dtype=np.float32)
    descriptors = np.zeros((num_samples, num_assets, num_descriptors), dtype=np.float32)

    if descriptor_mode == "fixed":
        returns = factors @ beta.T
        descriptors[:] = U_base

    elif descriptor_mode == "perturbed":
        for i in range(num_samples):
            noise = descriptor_perturbation * rng.standard_normal((num_assets, num_descriptors))
            U_i = _normalize_descriptor_columns(U_base + noise)
            descriptors[i] = U_i
            returns[i] = U_i @ T @ factors[i]

    else:  # random
        for i in range(num_samples):
            U_i = rng.standard_normal((num_assets, num_descriptors))
            U_i += descriptor_noise * rng.standard_normal((num_assets, num_descriptors))
            U_i = _normalize_descriptor_columns(U_i)
            descriptors[i] = U_i
            returns[i] = U_i @ T @ factors[i]

    return {
        "returns": returns,
        "descriptors": descriptors,
        "factors": factors,
        "T": T,
        "U_base": U_base,
        "beta": beta,
        "metadata": {
            "num_samples": num_samples,
            "num_assets": num_assets,
            "num_descriptors": num_descriptors,
            "num_factors": num_factors,
            "descriptor_mode": descriptor_mode,
            "descriptor_noise": descriptor_noise,
            "descriptor_perturbation": descriptor_perturbation,
            "factor_scale": factor_scale,
            "seed": seed,
            "model": "residual_free_R_equals_U_T_F",
        },
    }


def save_descriptor_dataset(data: dict, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "returns.npy", data["returns"])
    np.save(output_dir / "descriptors.npy", data["descriptors"])
    np.save(output_dir / "factors.npy", data["factors"])
    np.save(output_dir / "T.npy", data["T"])
    if "U_base" in data:
        np.save(output_dir / "U_base.npy", data["U_base"])
    if "beta" in data:
        np.save(output_dir / "beta.npy", data["beta"])

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(data["metadata"], f, indent=2)

    return output_dir


def load_descriptor_dataset(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    with open(data_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    result = {
        "returns": np.load(data_dir / "returns.npy"),
        "descriptors": np.load(data_dir / "descriptors.npy"),
        "factors": np.load(data_dir / "factors.npy"),
        "T": np.load(data_dir / "T.npy"),
        "metadata": metadata,
    }
    if (data_dir / "U_base.npy").exists():
        result["U_base"] = np.load(data_dir / "U_base.npy")
    if (data_dir / "beta.npy").exists():
        result["beta"] = np.load(data_dir / "beta.npy")
    return result


def compute_return_norm_stats(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-asset mean and global std for return standardization."""
    mean = returns.mean(axis=0).astype(np.float32)
    scale = float(max(returns.std(), 1e-6))
    return mean, scale


class DescriptorTrainer:
    """Lightweight trainer for CPU or single-GPU runs."""

    def __init__(
        self,
        model: DescriptorFactorDiffusion,
        dataset: DescriptorReturnDataset,
        *,
        train_lr: float = 1e-4,
        weight_decay: float = 0.01,
        batch_size: int = 32,
        epochs: int = 600,
        results_folder: str | Path = "./results_descriptor",
        save_every: int = 1000,
        device: str | None = None,
        num_workers: int = 0,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.epochs = epochs
        self.save_every = save_every
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.dataloader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=True,
            drop_last=len(dataset) >= batch_size,
            num_workers=num_workers,
        )
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=train_lr, weight_decay=weight_decay)

    def train(self) -> List[float]:
        history = []
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for returns, U, period_idx in self.dataloader:
                returns = returns.to(self.device)
                U = U.to(self.device)
                period_idx = period_idx.to(self.device)

                self.optimizer.zero_grad()
                loss = self.model(returns, U, period_idx)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history.append(avg_loss)
            print(f"Epoch {epoch + 1}/{self.epochs}  loss={avg_loss:.6f}")

            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch + 1)

        self._save_checkpoint("final")
        return history

    def _save_checkpoint(self, tag):
        path = self.results_folder / f"descriptor_model_{tag}.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "num_assets": self.model.num_assets,
                "num_descriptors": self.model.num_descriptors,
                "num_factors": self.model.num_factors,
                "num_timesteps": self.model.num_timesteps,
                "hidden_size": self.model.hidden_size,
                "max_periods": self.model.max_periods,
                "clip_denoised": self.model.clip_denoised,
                "mean_loss_weight": self.model.mean_loss_weight,
                "return_mean": self.model.return_mean.cpu(),
                "return_scale": self.model.return_scale.cpu(),
            },
            path,
        )
        print(f"Saved checkpoint to {path}")
