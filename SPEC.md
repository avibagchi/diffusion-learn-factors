# SPEC — Learning Factors Through the Score Network (Notes §3.2)

Build-ready implementation spec for the second extension in the modification notes,
**`2026_factor_model_diffusion.pdf` §3.2 "Learning Factors Through Score Networks."**

**Source-of-truth rule:** `2026_factor_model_diffusion.pdf` (hereafter **NOTES**) governs *what*
to build. The DFM paper and other PDFs are cited only for *why* / precise definitions. Where they
conflict, NOTES win and the conflict is flagged.

**PDF-extraction caveat:** every equation below was transcribed from a PDF text layer, which mangles
indices, transposes, sub/superscripts, and norms. Each equation that drives code carries a
`⚠ VERIFY` note listing exactly what to re-check against the rendered PDF before trusting it.

---

## 1. Objective

Keep the DFM baseline unchanged (factor model, Gaussian residual, OU diffusion) but make the **loading
map trainable inside the score network** and ask how much factor structure the trained network
recovers (NOTES §3.2, p.7). Concretely: replace the score network's fixed descriptor→factor map with a
trainable matrix **A** sitting in the same position as the true weight matrix **T**, train by score
matching on period-indexed returns, then read **A** (equivalently the column space of `Uᵢ A`) back out
and compare it to the truth.

**What changes vs. the repo:** the current model is a generic 2-D image U-Net (`Unet`,
`diffusion_factor_model/diffusion_factor_model.py:277`) that predicts noise and has *no* factor
structure, *no* period index, and *no* descriptor matrix. §3.2 requires a new, structurally-constrained
vector score network `s_θ(r, t, i)`, a period-indexed data pipeline, and a recovery metric on **A**.

**Success criterion (one line):** the learned `A` recovers the column space of the true `T` (equivalently
`span(Uᵢ A) ≈ span(Uᵢ T) = span(βᵢ)`) up to the unavoidable `k×k` rotation/scaling ambiguity — measured
by a subspace-distance metric (§5). NOTES give **no numeric threshold** (see §6).

---

## 2. Score network — exact signature of `s_θ(r, t, i)`

### 2.1 The equation NOTES prescribe (NOTES p.8, top)

```
s_θ(r, t, i) = -(1/σ_t²) · r  -  (α_t / σ_t²) · Uᵢ A · Network(Qᵢᵀ r, t, i)
```

- `Network(Qᵢᵀ r, t, i) ∈ ℝ^k` — deep network output (the estimated posterior factor mean; see 2.4).
- `A ∈ ℝ^{m×k}` — trainable, estimates the true `T` (this is §3 below, the crux).
- `Uᵢ A ∈ ℝ^{d×k}`, so `Uᵢ A · Network(...) ∈ ℝ^d`, and `s_θ(r,t,i) ∈ ℝ^d` — matches the score dimension.

⚠ VERIFY (NOTES p.8): the two scalar coefficients are `-1/σ_t²` on `r` and `-α_t/σ_t²` on the
`Uᵢ A·Network` term. Confirm both are **negative**, the second carries `α_t` in the numerator, and both
denominators are `σ_t²` (not `σ_t` or `2σ_t²`).

This mirrors the *true* score NOTES derive for the simplified model (NOTES p.7, bottom):

```
∇ log p_{i,t}(r) = -(1/σ_t²) r
                   - (α_t/σ_t²) Uᵢ T · [ ∫ fᵢ exp(-‖Qᵢᵀr - α_t Vᵢ T fᵢ‖²₂ / (2σ_t²)) p_f(fᵢ) dfᵢ ]
                                        / [ ∫    exp(-‖Qᵢᵀr - α_t Vᵢ T fᵢ‖²₂ / (2σ_t²)) p_f(fᵢ) dfᵢ ]
```

i.e. `Network(Qᵢᵀr, t, i)` is trained to approximate the bracketed integral ratio
`∈ ℝ^k`, which is the posterior mean `E[fᵢ | Qᵢᵀr]` of the factor under Gaussian noising, and the
trainable `A` stands in for `T`.

⚠ VERIFY (NOTES p.7): inside the exponent it is `Qᵢᵀr - α_t Vᵢ T fᵢ` (both terms live in `ℝ^m`,
NOTES confirm this dimensionally on p.8: "`Qᵢᵀ r ∈ ℝ^m`, `Vᵢ T fᵢ ∈ ℝ^m`, integral ratio ∈ ℝ^k").
Re-check the transpose on `Qᵢᵀ` and that the projection inside the kernel is `Qᵢᵀr`, **not** `Qᵢ r`.

### 2.2 Argument shapes & dtypes (per batch of size `b`)

| arg | meaning | shape | dtype |
|---|---|---|---|
| `r` | noised return `Rₜ` for period `i` | `(b, d)` | float32 |
| `t` | diffusion time / step index | `(b,)` | long (step) or float (continuous) — see §6 |
| `i` | period index | `(b,)` | long |
| **out** `s_θ` | score estimate in `ℝ^d` | `(b, d)` | float32 |

Per-period side tensors (not learned, supplied by the data pipeline, §4):
`Uᵢ ∈ ℝ^{d×m}`, `Qᵢ ∈ ℝ^{d×m}` (orthonormal columns), `Vᵢ ∈ ℝ^{m×m}` (upper-triangular), from the
**thin QR** `Uᵢ = Qᵢ Vᵢ` (NOTES p.7). Requires `d ≥ m` for thin QR to have `Qᵢ` with `m` orthonormal
columns. `Vᵢ` is needed only to *generate* data / define the true score; the network forward pass uses
only `Uᵢ` and `Qᵢ` plus the trainable `A`.

Dimensions: `d` = #assets, `m` = #descriptors, `k` = #factors, with `k ≤ m ≤ d` and `k ≪ d`
(NOTES §2.1, §3.2).

### 2.3 Architecture of `Network(·, t, i)`

NOTES specify only: input `Qᵢᵀr ∈ ℝ^m`, output `∈ ℝ^k`, "a deep neural network," conditioned on `t`
and `i`. NOTES do **not** fix depth, width, conditioning mechanism, or how `i` enters (§6). Concrete
build-ready default (matches repo conventions — `SinusoidalPosEmb` at
`diffusion_factor_model.py:115`, SiLU MLP time-embedding at `:171`):

- **Input:** `x = Qᵢᵀ r` → shape `(b, m)`.
- **Time conditioning:** `t` → `SinusoidalPosEmb(dim)` → 2-layer MLP (`Linear→GELU→Linear`) →
  `t_emb ∈ (b, time_dim)`; inject into each hidden block via FiLM scale/shift (as `ResnetBlock` does at
  `:180–186`).
- **Period conditioning `i`:** `nn.Embedding(num_periods, emb_dim)` → concatenated with `t_emb` (or
  added). *Open:* whether `i` should condition `Network` *at all* given that `Uᵢ, Qᵢ` already inject all
  period-specific geometry — see §6.
- **Trunk:** MLP, e.g. `Linear(m → h) → [SiLU → Linear(h→h) + FiLM(t,i)] × L → Linear(h → k)`. Default
  `h = 256`, `L = 3` (placeholders; not from NOTES).
- **Output:** `(b, k)`.

Then assemble per the §2.1 equation: `s_θ = -(1/σ_t²) r - (α_t/σ_t²) · einsum('bdm,mk,bk->bd', U_i, A, net_out)`
where `U_i` is gathered per-sample from `i`.

### 2.4 Where this lands in the repo

- **Replaces** `Unet` (`diffusion_factor_model.py:277`) as the `model` passed to `GaussianDiffusion`
  (`:519`, constructed in `train.py:144`). The U-Net is image-shaped `(b, 1, H, W)` and predicts
  noise; the new module is vector-shaped `(b, d)` and is structurally a **score / denoiser**. This is
  a new module (suggested `FactorScoreNet`), not an edit to `Unet`.
- **Touches** `GaussianDiffusion.model_predictions` (`:681`) and `p_losses` (`:965`): those assume the
  model output is `noise`/`v`/`x0` with image shape and call `self.model(x, t, x_self_cond)`. The new
  model's signature adds `i` and drops `x_self_cond`, and its output is a **score**, so the forward
  call sites and the loss target must change (see §6 — score-matching vs. the repo's MSE-on-noise).
- **Integration tactic (recommended, not from NOTES):** because the simplified model is *noiseless*
  (§3) the true score equals `-(1/σ_t²)(r - α_t x̂₀)` with predicted clean return
  `x̂₀ = Uᵢ A · Network(...)`. So the module can be wired as a **`pred_x0`** model — output
  `x̂₀ ∈ ℝ^d`, let `GaussianDiffusion` convert to noise/score — reusing `model_predictions`/`p_losses`
  with minimal change. Flagged as a design choice in §6, not mandated by NOTES.

---

## 3. The trainable matrix `A` (the crux)

**Definition (NOTES p.8):** `A ∈ ℝ^{m×k}` is "a trainable weight matrix estimating the time-invariant
matrix `T`." `T ∈ ℝ^{m×k}` is the fixed, time-invariant descriptor→factor weight matrix in the
practical loading model `βᵢ = Uᵢ T` (NOTES p.7), where `Uᵢ ∈ ℝ^{d×m}` is the precomputed descriptor
matrix.

- **Shape:** `(m, k)`. ⚠ VERIFY (NOTES p.8): `A ∈ ℝ^{m×k}`, `Uᵢ A ∈ ℝ^{d×k}`. Confirm it is `m×k`
  (not `k×m`); the product `Uᵢ A` `(d×m)(m×k)=(d×k)` only type-checks if `A` is `m×k`.
- **Position in forward pass:** exactly where `T` sits in the true score — left-multiplied by `Uᵢ`,
  right-multiplied by the network's `ℝ^k` output: `Uᵢ A · Network(Qᵢᵀr, t, i)` (§2.1). It is a single
  global `nn.Parameter`, **shared across all periods `i`** and all `t` (time-invariant, like `T`).
- **Initialization:** NOTES do **not** specify (§6). Sensible default: small random
  (`A ~ N(0, 1/m)` or orthonormal-column init via `torch.nn.init.orthogonal_`). Do **not** initialize
  to the true `T` (that would defeat the recovery experiment).
- **How it recovers loading structure:** the architecture forces all period-specific geometry through
  the known `Uᵢ` (and `Qᵢ` in the kernel/encoder), leaving `A` as the *only* time-invariant `m×k`
  object multiplying the learned factor coordinate — the same role `T` plays in the true score. If
  training drives `s_θ → ∇log p_{i,t}` across periods, then `Uᵢ A` must match `Uᵢ T = βᵢ` up to the
  factor-model rotation/scaling ambiguity, so `span(A) → span(T)` (identifiable only as a subspace;
  NOTES §2.1 "rotation and rescaling ambiguity," §3.2 "individual factor coordinates are not unique,
  while factor spaces … are meaningful up to rotation and scaling").
- **Regularization / constraints:** NOTES impose **none explicitly** on `A`. NOTES §2.1 normalize the
  *loadings* `β` to orthonormal columns (`βᵀβ = I_k`) for identifiability, and §3.2 asks "under what
  normalization can `T` or the loading space of `Uᵢ T` be recovered" — i.e. the normalization question
  is posed as *open*, not answered (§6). Do not add a constraint on `A` unless validating a specific
  normalization choice.

⚠ VERIFY (NOTES p.7–8): the chain `βᵢ = Uᵢ T`, thin QR `Uᵢ = Qᵢ Vᵢ` (`Qᵢ ∈ ℝ^{d×m}` orthonormal cols,
`Vᵢ ∈ ℝ^{m×m}` upper-triangular). Re-check `Vᵢ` is `m×m` (square) and that it is `Qᵢ Vᵢ` not `Vᵢ Qᵢ`.

---

## 4. Data pipeline — what "period-indexed" requires

### 4.1 Current pipeline (to be changed)

`train.py:74–134`: loads one `.npy`, reshapes returns into a 2-D "image" `(samples, 1, H, W)`
(`train.py:84–114`), standardizes (`:131–133`), wraps in a bare `TensorDataset(data)` (`:134`). The
training loop pulls `data = data[0]` (`diffusion_factor_model.py:1225`) — **a single returns tensor,
no index, no descriptors.** Confirmed shapes of the shipped examples (npy headers):
- `empirical_analysis_data/training_data_example.npy`: `(1194, 512)` float64 → 1194 obs, `d=512`.
- `simulation_experiment_data/training_data_example.npy`: `(512, 32, 64)` float32 → 512 obs, `d=2048`.

Neither file carries a period index or descriptor matrix `Uᵢ`.

### 4.2 What §3.2 requires concretely

The model `R_i = βᵢ Fᵢ = Uᵢ T Fᵢ` (NOTES p.7) is **period-specific**: each economic period `i` has its
own descriptor matrix `Uᵢ`. So a dataset item must carry both a return and its period index:

- **Item:** `(r, i)` where `r ∈ ℝ^d` is one (noiseless) return vector and `i ∈ {0,…,P-1}` its period.
- **Batch:** `r: (b, d)` float32, `i: (b,)` long.
- **Period side-table (not per-item, loaded once):** `{Uᵢ}` stored as `U: (P, d, m)`; precompute and
  cache `Q: (P, d, m)` and `V: (P, m, m)` via thin QR (`torch.linalg.qr(U, mode='reduced')`). The
  forward pass gathers `U[i]`, `Q[i]` by the batch's `i`.
- **Assignment of `i`:** must come from the data generator / metadata — which period each observation
  belongs to. For the **simulation experiment** (the natural first target, since ground-truth `T`,
  `Uᵢ`, `Fᵢ` are needed for §5), write a new generator: sample a fixed true `T ∈ ℝ^{m×k}`, sample
  per-period `Uᵢ ∈ ℝ^{d×m}`, sample factors `Fᵢ` per observation from `p_f`, form `R = Uᵢ T Fᵢ`, and
  emit `(R, i, U, T, F)`. The existing `GaussianLatentSampler2D_Finance`
  (`diffusion_factor_model.py:482`) is **not** period-indexed and has no descriptor matrix — it draws
  `factor·A` with a single fixed `A`; it must be replaced/extended for this experiment.

### 4.3 Pipeline changes (file-level)

- New `Dataset` returning `(r, i)`; `TensorDataset(data, period_idx)` suffices if `Uᵢ` lives in the
  model/module as a registered buffer indexed by `i`.
- Training loop (`diffusion_factor_model.py:1224–1232`) currently unpacks `data[0]`; change to unpack
  `(r, i)` and pass `i` into the model.
- `train.py` data-loading branch (`:84–114`) reshapes to images — bypass for the vector model; keep
  the standardization step but note it interacts with the factor structure (standardizing per-asset
  rescales `Uᵢ`; decide whether to standardize, and if so fold it into `Uᵢ`). Flagged in §6.

---

## 5. Subspace-alignment validation

**Two subspaces compared:** the **learned** factor-loading space vs. the **true** one. Per NOTES §3.2,
only spaces (not coordinates) are identifiable, so compare column spaces:

- Learned: `span(A)` in `ℝ^m` (or `span(Uᵢ A) = span(β̂ᵢ)` in `ℝ^d`).
- True: `span(T)` in `ℝ^m` (or `span(Uᵢ T) = span(βᵢ)` in `ℝ^d`).

**Recommended metric — principal angles / Grassmann subspace distance** (standard for this question;
NOTES name *no* specific metric — see §6). Given orthonormal bases `Q_A = orth(A)`, `Q_T = orth(T)`
(`m×k`):

```
σ_j = singular values of Q_Aᵀ Q_T,  j = 1..k     (each σ_j = cos θ_j, θ_j the j-th principal angle)
projection metric:   d_proj = ‖Q_A Q_Aᵀ − Q_T Q_Tᵀ‖_F  =  sqrt( k − Σ_j σ_j² ) · √2
alignment score:     ρ = (1/k) Σ_j σ_j²   ∈ [0,1],   ρ = 1 ⇔ identical subspaces
```

Inputs: learned `A` (read from the trained `nn.Parameter`), true `T` (from the simulation generator),
both `m×k`. Optionally evaluate in `ℝ^d` per period using `Uᵢ A` vs `Uᵢ T`.

**Existing repo metric (alternative / for parity with DFM):** `eval/simulation_eval.py` already does a
subspace comparison via top-`k` SVD of a covariance matrix (`svd`, `:106`;
`calculate_latent_subspace`, `:125`) and a **relative Frobenius error**
`‖Ŝ − S_true‖_F / ‖S_true‖_F` (`calculate_frobenius_norm_errors`, `:140–155`). This compares
*reconstructed subspace matrices*, not principal angles, and is sensitive to the rotation ambiguity
unless bases are aligned first — so prefer the principal-angle metric for the `A`-recovery question,
and report the Frobenius metric only for continuity with DFM's reporting.

**Success value:** NOTES give **no threshold** (§6). Report `ρ` (→1) and `d_proj` (→0); for a pass/fail
gate, benchmark against the subspace recovered by PCA/POET on the same simulated returns (the baselines
already in `eval/ft_portfolio_eval.py`) rather than inventing a constant.

---

## 6. Open questions / ambiguities

1. **Discrete DDPM vs. continuous OU score (biggest gap).** The repo is discrete DDPM: `timesteps=200`
   (`config.py:40`), cosine `betas`, objective `pred_noise` (`config.py:41`), MSE on noise
   (`p_losses`, `diffusion_factor_model.py:965–1008`). NOTES §2.2/§3.2 use a **continuous OU** score
   with `α_t = e^{-t/2}`, `h_t = 1−e^{-t}`, and a denoising-score-matching loss. **Conflict:** the new
   `s_θ` outputs a *score*, but the repo trains against *noise/v/x0*. Resolution needed: either (a)
   add a score-matching loss path, or (b) wire `s_θ` as a `pred_x0` model (§2.4) and let the existing
   machinery convert. NOTES win on *what* the model computes; the discretization is an implementation
   choice NOTES don't pin down.

2. **`σ_t` definition / notation mapping.** NOTES §3.2 write `σ_t²` = "scalar Gaussian noising variance
   at diffusion time `t`." In NOTES §2.2 the OU transition is `Rₜ|R₀ ~ N(α_t r₀, h_t I_d)` with
   `α_t=e^{-t/2}`, `h_t=1−e^{-t}`, so **`σ_t² = h_t = 1−e^{-t}`** and the leading `α_t`. The repo's
   equivalents are `sqrt_alphas_cumprod[t]` (↔ `α_t`) and `1 − alphas_cumprod[t]` (↔ `σ_t²`)
   (`diffusion_factor_model.py:593–594`). ⚠ VERIFY this mapping against the rendered PDF; a wrong
   `σ_t` vs `σ_t²` will silently miscalibrate the score's scale.

3. **No residual in §3.2's simplified model.** NOTES §3.2 use `Rᵢ = βᵢ Fᵢ` ("noiseless return"),
   dropping the `ε` of the DFM baseline (§1, §2.1, `R = βF + ε`). **Flag:** all `σ_t²` here is pure
   diffusion noise, no `Λ_t = h_t I + α_t² D` residual term. Conflicts with §1's general model; NOTES
   §3.2 win for this experiment.

4. **`A` initialization & normalization.** Not specified (§3). The identifiability/normalization
   question ("under what normalization can `T` … be recovered," NOTES p.2/§3.2) is posed as *open*.
   Decide: random init; whether to constrain `AᵀA` or normalize columns of `Uᵢ A` to mimic `βᵀβ=I_k`.

5. **How `i` enters `Network`.** NOTES write `Network(Qᵢᵀr, t, i)` but `Uᵢ, Qᵢ` already carry all
   period geometry. Is the explicit `i` an embedding, redundant, or meant only to select `Uᵢ/Qᵢ`?
   Unspecified. Default: include a learned period embedding but ablate it.

6. **`Network` architecture details.** Depth, width, conditioning mechanism — none given by NOTES
   (§2.3). Values above are placeholders.

7. **Validation metric & threshold.** NOTES name no metric and no success threshold (§5). Principal
   angles are the standard choice and are recommended here, but this is *our* decision, not NOTES'.

8. **Data / descriptors.** NOTES assume precomputed `Uᵢ` per period; the shipped data files carry
   neither period index nor descriptors (§4.1). A new period-indexed simulation generator (with known
   `T`, `Uᵢ`, `Fᵢ`) is required before §5 can be evaluated — the existing
   `GaussianLatentSampler2D_Finance` does not provide it.

9. **Standardization vs. factor structure.** `train.py:131–133` standardizes returns per asset, which
   rescales the effective `Uᵢ`. Decide whether to standardize, skip it, or absorb it into `Uᵢ`, so the
   recovered `span(Uᵢ A)` is compared on the same footing as the true `span(βᵢ)` (§4.3).

10. **`d ≥ m` requirement.** Thin QR `Uᵢ = Qᵢ Vᵢ` with `Qᵢ ∈ ℝ^{d×m}` orthonormal columns requires
    `d ≥ m`. The simulation must enforce `k ≤ m ≤ d`. Not stated in NOTES but implied by the QR shapes.
```
