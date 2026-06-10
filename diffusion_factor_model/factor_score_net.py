"""
Factor score network for NOTES §3.2 "Learning Factors Through Score Networks".

FactorScoreNet predicts the clean return
    x̂₀ = Uᵢ A · Network(Qᵢᵀ r, t, i) ∈ ℝ^d
with a single global trainable A ∈ ℝ^{m×k} standing in for the true T.
It is wired as a `pred_x0` model inside FactorGaussianDiffusion, which lets
the existing GaussianDiffusion machinery convert x̂₀ → noise/score, so the
Tweedie sign of the score's factor term is structural rather than hand-coded.
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce

from diffusion_factor_model.diffusion_factor_model import (
    GaussianDiffusion,
    ModelPrediction,
    SinusoidalPosEmb,
    extract,
)


class FactorScoreNet(nn.Module):
    """
    Network(Qᵢᵀr, t, i) → ℝ^k assembled into x̂₀ = Uᵢ A · Network(...) ∈ ℝ^d.

    Per-period side tensors U (P,d,m) and Q (P,d,m) are registered buffers
    gathered by the batch's period index i; A is the only time-invariant
    trainable m×k object multiplying the learned factor coordinate.
    """

    def __init__(
        self,
        U,                  # (P, d, m) descriptor matrices
        Q,                  # (P, d, m) thin-QR orthonormal factors of U
        k,                  # number of factors
        hidden_dim=256,
        depth=3,
        time_emb_dim=64,
        period_emb_dim=32,
    ):
        super().__init__()
        assert U.shape == Q.shape and U.ndim == 3
        P, d, m = U.shape
        self.num_periods, self.d, self.m, self.k = P, d, m, k

        # read by GaussianDiffusion.__init__ (:544–545)
        self.channels = 1
        self.self_condition = False

        self.register_buffer('U', U.to(torch.float32))
        self.register_buffer('Q', Q.to(torch.float32))

        # trainable stand-in for the true T — random init, never set to T
        A = torch.empty(m, k)
        nn.init.orthogonal_(A)
        self.A = nn.Parameter(A)

        time_dim = time_emb_dim * 2
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.period_emb = nn.Embedding(P, period_emb_dim)
        cond_dim = time_dim + period_emb_dim

        self.in_proj = nn.Linear(m, hidden_dim)
        self.blocks = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(depth))
        # FiLM scale/shift per block, as ResnetBlock does (:171–186)
        self.films = nn.ModuleList(
            nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, hidden_dim * 2))
            for _ in range(depth)
        )
        self.out_proj = nn.Linear(hidden_dim, k)

    def network(self, x_m, t, i):
        """The deep network: (b, m) projected returns → (b, k) factor estimate."""
        cond = torch.cat([self.time_mlp(t), self.period_emb(i)], dim=-1)
        h = self.in_proj(x_m)
        for block, film in zip(self.blocks, self.films):
            scale, shift = film(cond).chunk(2, dim=-1)
            h = h + F.silu(block(h) * (scale + 1) + shift)
        return self.out_proj(h)

    def forward(self, r, t, i):
        """
        r: (b, d) noised returns, t: (b,) diffusion steps, i: (b,) period idx
        returns x̂₀: (b, d)
        """
        U_i, Q_i = self.U[i], self.Q[i]                      # (b, d, m)
        x_m = torch.einsum('bdm,bd->bm', Q_i, r)             # Qᵢᵀ r
        f_hat = self.network(x_m, t, i)                      # (b, k)
        return torch.einsum('bdm,mk,bk->bd', U_i, self.A, f_hat)


class FactorGaussianDiffusion(GaussianDiffusion):
    """
    GaussianDiffusion over (b, d) return vectors with a period-indexed
    pred_x0 model. Overrides only the image-shaped entry points; the noising
    schedule, x̂₀→noise conversion, and loss weighting are inherited.
    """

    def __init__(self, model, *, d, auto_normalize=False, **kwargs):
        kwargs.setdefault('objective', 'pred_x0')
        kwargs.setdefault('min_snr_loss_weight', True)
        kwargs.setdefault('min_snr_gamma', 5)
        assert kwargs['objective'] == 'pred_x0', 'the factor model is wired as pred_x0'
        # image_size only satisfies the parent's tuple check; vector paths below never use it
        super().__init__(model, image_size=(1, d), auto_normalize=auto_normalize, **kwargs)
        self.dim = d

    def model_predictions(self, x, t, i, clip_x_start=False, rederive_pred_noise=False):
        x_start = self.model(x, t, i)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start)

    def p_losses(self, x_start, t, i, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x, t, i)

        loss = F.mse_loss(model_out, x_start, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, r, i):
        b, d = r.shape
        assert d == self.dim, f'return dimension must be {self.dim}'
        t = torch.randint(0, self.num_timesteps, (b,), device=r.device).long()
        return self.p_losses(r, t, i)
