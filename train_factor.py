"""
Stage-3 training entry point for the NOTES §3.2 factor-recovery experiment.

Trains FactorScoreNet (pred_x0, period-indexed) on the Stage-1 generated
dataset via the existing Trainer. Mirrors train.py's Trainer invocation, with
the settings a short CPU run needs: amp off (fp16 raises without GPU),
num_fid_samples=0, a checkpoint cadence that actually fires, and no LR warmup.
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset

from diffusion_factor_model import FactorScoreNet, FactorGaussianDiffusion, Trainer
import config.config as config


def train_factor_model(data_path, seed=None, epochs=80, batch_size=None, lr=None,
                       period_emb_dim=0):
    seed = config.set_seed(seed)

    z = np.load(data_path)
    R = torch.from_numpy(z["R"]).float()                  # (N, d)
    period_idx = torch.from_numpy(z["period_idx"]).long() # (N,)
    U = torch.from_numpy(z["U"])
    Q = torch.from_numpy(z["Q"])
    d, k = int(z["d"]), int(z["k"])
    print(f"Loaded {data_path}: N={len(R)}, d={d}, m={int(z['m'])}, "
          f"k={k}, P={int(z['periods'])}")

    dataset = TensorDataset(R, period_idx)

    net = FactorScoreNet(U, Q, k=k, period_emb_dim=period_emb_dim)
    diffusion = FactorGaussianDiffusion(
        net,
        d=d,
        timesteps=config.TIMESTEPS,
        beta_schedule=config.BETA_SCHEDULE,
    )
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"FactorScoreNet: {n_params} trainable params (A: {tuple(net.A.shape)})")

    tag = "" if period_emb_dim > 0 else "_nope"
    exp_id = f"factor_{os.path.splitext(os.path.basename(data_path))[0]}{tag}_ts{int(time.time())}_seed{seed}"
    model_dir = os.path.join(config.MODELS_DIR, exp_id)

    save_every = max(1, epochs // 4)
    trainer = Trainer(
        diffusion,
        dataset,
        train_batch_size=batch_size or config.BATCH_SIZE,
        train_lr=lr or config.LEARNING_RATE,
        train_epochs=epochs,
        adamw_weight_decay=config.WEIGHT_DECAY,
        cosine_scheduler=True,
        warm_up=False,                 # warmup_fn(0)=0 would train epoch 1 at LR 0
        T_0=epochs,
        eta_min=config.COSINE_LR_MIN,
        gradient_accumulate_every=config.GRADIENT_ACCUMULATION,
        ema_decay=config.EMA_DECAY,
        split_batches=config.SPLIT_BATCHES,
        save_and_sample_every=save_every,   # config.SAVE_INTERVAL=1000 would never fire
        results_folder=model_dir,
        param_path="",
        amp=False,                     # fp16 raises without CUDA
        num_fid_samples=0,             # sampling path unused (and must stay 0)
    )

    print(f"Training {epochs} epochs, checkpoint every {save_every} epochs -> {model_dir}")
    trainer.train()
    last_saved = (epochs // save_every) * save_every
    print(f"Done. Final checkpoint: {model_dir}/model-epoch-{last_saved}.pt")
    return model_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the §3.2 factor score network")
    parser.add_argument("--data_path", type=str,
                        default="simulation_experiment_data/factor_period_data.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--period_emb_dim", type=int, default=0,
                        help="learned period embedding dim (0 = none, the default)")
    args = parser.parse_args()

    train_factor_model(args.data_path, args.seed, args.epochs, args.batch_size, args.lr,
                       args.period_emb_dim)
