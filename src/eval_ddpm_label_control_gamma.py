import argparse
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from train_ddpm_cond_v2_gamma import CondUNet1D


# -----------------------------------------------------------
# Diffusion utilities
# -----------------------------------------------------------

def make_beta_schedule(schedule: str, n_timestep: int) -> torch.Tensor:
    if schedule == "linear":
        beta_start = 1e-4
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, n_timestep, dtype=torch.float32)
    elif schedule == "cosine":
        steps = n_timestep + 1
        x = torch.linspace(0, n_timestep, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / n_timestep) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 1e-8, 0.999)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")


class GaussianDiffusion1D(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int = 400, beta_schedule: str = "linear", device: str = "cpu"):
        super().__init__()
        self.model = model.to(device)
        self.device = torch.device(device)
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule

        betas = make_beta_schedule(beta_schedule, timesteps).to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=self.device), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "sqrt_recipm1_alphas", torch.sqrt(1.0 / alphas - 1.0)
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas
            * torch.sqrt(alphas_cumprod_prev)
            / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod),
        )

    @torch.no_grad()
    def p_mean_variance(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps_theta = self.model(x_t, t, cond)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1)
        sqrt_recipm1_alphas_t = self.sqrt_recipm1_alphas[t].view(-1, 1, 1)
        x0_pred = sqrt_recip_alphas_t * x_t - sqrt_recipm1_alphas_t * eps_theta
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

        posterior_mean = (
            self.posterior_mean_coef1[t].view(-1, 1, 1) * x0_pred
            + self.posterior_mean_coef2[t].view(-1, 1, 1) * x_t
        )
        posterior_var = self.posterior_variance[t].view(-1, 1, 1)
        posterior_log_var = self.posterior_log_variance_clipped[t].view(-1, 1, 1)
        return posterior_mean, posterior_var, posterior_log_var

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        b = x_t.shape[0]
        posterior_mean, _, posterior_log_var = self.p_mean_variance(x_t, t, cond)
        nonzero_mask = (t != 0).float().view(b, 1, 1)
        if noise is None:
            noise = torch.randn_like(x_t)
        return posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_var) * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: Tuple[int, int, int],
        cond: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        b, C, T = shape
        device = self.device
        if noise is None:
            x_t = torch.randn((b, C, T), device=device)
        else:
            x_t = noise.to(device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, t, cond)
        return x_t


def build_ddpm_model(model_dir: str, ckpt_name: str, device: str):
    device_t = torch.device(device)
    ckpt_path = os.path.join(model_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device_t)

    if "ema_state_dict" in ckpt:
        state_dict = ckpt["ema_state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        raise KeyError("No 'ema_state_dict' or 'model_state_dict' in checkpoint.")

    cond_dim = 2
    for k, v in state_dict.items():
        if "time_cond_mlp.cond_mlp.0.weight" in k and v.dim() == 2:
            cond_dim = v.shape[1]
            break

    in_channels = 64
    base_channels = 64
    for k, v in state_dict.items():
        if "in_conv.weight" in k or "init_conv.weight" in k:
            base_channels = v.shape[0]
            in_channels = v.shape[1]
            break

    model = CondUNet1D(
        in_channels=in_channels,
        base_channels=base_channels,
        cond_dim=cond_dim,
    ).to(device_t)

    model.load_state_dict(state_dict)
    model.eval()

    timesteps = int(ckpt.get("timesteps", 400))
    beta_schedule = ckpt.get("beta_schedule", "linear")

    diffusion = GaussianDiffusion1D(
        model=model,
        timesteps=timesteps,
        beta_schedule=beta_schedule,
        device=device,
    )
    return diffusion, timesteps, beta_schedule, cond_dim


# -----------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------

def compute_features_batched(x: np.ndarray, sfreq: float, batch_size: int = 512) -> np.ndarray:
    N, C, T = x.shape
    freqs = np.fft.rfftfreq(T, 1.0 / sfreq)

    def band_mask(lo, hi):
        return (freqs >= lo) & (freqs < hi)

    masks = [
        band_mask(1, 4),   # delta
        band_mask(4, 8),   # theta
        band_mask(8, 14),  # alpha
        band_mask(15, 30), # beta
        band_mask(30, 80), # gamma
    ]

    feats_all = []
    for start in range(0, N, batch_size):
        xb = x[start:start + batch_size]
        psd = np.abs(np.fft.rfft(xb, axis=-1)) ** 2
        band_feats = []
        for m in masks:
            bp = psd[..., m].mean(axis=-1)
            bp_avg = bp.mean(axis=-1)
            band_feats.append(bp_avg)
        std_feat = xb.std(axis=(1, 2), ddof=0)
        feats_b = np.stack(band_feats + [std_feat], axis=1)
        feats_all.append(feats_b.astype(np.float32))

    feats = np.concatenate(feats_all, axis=0)
    return feats


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------
# Sampling: group flip & pain sweep
# -----------------------------------------------------------

def sample_group_flip(
    diffusion: GaussianDiffusion1D,
    cond_dim: int,
    n_pairs: int,
    C: int,
    T: int,
    pain_scale_max: float,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    device_t = torch.device(device)
    diffusion.to(device_t)

    xs = []
    groups = []

    remaining = n_pairs
    while remaining > 0:
        cur_pairs = min(batch_size, remaining)
        noise = torch.randn((cur_pairs, C, T), device=device_t)

        pain_level = 0.7 * pain_scale_max
        cond_h = torch.tensor(
            [[0.0, pain_level / pain_scale_max]] * cur_pairs,
            dtype=torch.float32,
            device=device_t,
        )
        x_h = diffusion.p_sample_loop(
            shape=(cur_pairs, C, T),
            cond=cond_h,
            noise=noise,
        )

        cond_c = torch.tensor(
            [[1.0, pain_level / pain_scale_max]] * cur_pairs,
            dtype=torch.float32,
            device=device_t,
        )
        x_c = diffusion.p_sample_loop(
            shape=(cur_pairs, C, T),
            cond=cond_c,
            noise=noise,
        )

        xs.append(x_h.cpu().numpy())
        xs.append(x_c.cpu().numpy())
        groups.append(np.zeros(cur_pairs, dtype=np.int64))
        groups.append(np.ones(cur_pairs, dtype=np.int64))

        remaining -= cur_pairs

    X = np.concatenate(xs, axis=0)
    y_group = np.concatenate(groups, axis=0)
    return X, y_group


def sample_pain_sweep(
    diffusion: GaussianDiffusion1D,
    cond_dim: int,
    n_per_pain: int,
    n_pain_steps: int,
    C: int,
    T: int,
    pain_scale_max: float,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    device_t = torch.device(device)
    diffusion.to(device_t)

    xs = []
    pain_labels = []

    pain_levels = np.linspace(0.0, pain_scale_max, n_pain_steps)
    for p in pain_levels:
        remaining = n_per_pain
        while remaining > 0:
            cur_bs = min(batch_size, remaining)
            cond = torch.tensor(
                [[1.0, p / pain_scale_max]] * cur_bs,
                dtype=torch.float32,
                device=device_t,
            )
            x_synth = diffusion.p_sample_loop(
                shape=(cur_bs, C, T),
                cond=cond,
                noise=None,
            )
            xs.append(x_synth.cpu().numpy())
            pain_labels.append(np.full(cur_bs, p, dtype=np.float32))
            remaining -= cur_bs

    X = np.concatenate(xs, axis=0)
    pain_arr = np.concatenate(pain_labels, axis=0)
    return X, pain_arr


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Label controllability evaluation for v2.1 (gamma).")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--pain_scale_max", type=float, default=10.0)
    parser.add_argument("--mode", type=str, choices=["group_flip", "pain_sweep"], required=True)
    parser.add_argument("--n_pairs", type=int, default=256, help="Number of noise pairs for group_flip.")
    parser.add_argument("--n_per_pain", type=int, default=128, help="Number of samples per pain level for pain_sweep.")
    parser.add_argument("--n_pain_steps", type=int, default=6, help="Number of pain levels for pain_sweep.")
    parser.add_argument("--sample_batch_size", type=int, default=16)
    parser.add_argument("--match_amp", action="store_true")
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)

    device = args.device

    print("============================================================")
    print("[INFO] Label controllability evaluation")
    print("============================================================")
    print(f"[INFO] DATA      = {args.data_path}")
    print(f"[INFO] META      = {args.meta_path}")
    print(f"[INFO] MODEL_DIR = {args.model_dir}")
    print(f"[INFO] CKPT      = {args.ckpt_name}")
    print(f"[INFO] MODE      = {args.mode}")
    print(f"[INFO] match_amp = {args.match_amp}")
    print("============================================================")

    data = np.load(args.data_path)
    N, C, T = data.shape
    real_std = float(data.std())
    print(f"[INFO] Data shape: {data.shape} (N, C, T)")
    print(f"[INFO] Global real std: {real_std:.4f}")

    diffusion, timesteps, beta_schedule, cond_dim = build_ddpm_model(
        args.model_dir, args.ckpt_name, device=device
    )
    print(f"[INFO] Loaded DDPM with timesteps={timesteps} schedule={beta_schedule} cond_dim={cond_dim}")

    if args.mode == "group_flip":
        print("[INFO] Running group_flip mode...")
        X_synth, group_labels = sample_group_flip(
            diffusion=diffusion,
            cond_dim=cond_dim,
            n_pairs=args.n_pairs,
            C=C,
            T=T,
            pain_scale_max=args.pain_scale_max,
            batch_size=args.sample_batch_size,
            device=device,
        )
        meta_labels = {"group_labels": group_labels}
    else:
        print("[INFO] Running pain_sweep mode...")
        X_synth, pain_labels = sample_pain_sweep(
            diffusion=diffusion,
            cond_dim=cond_dim,
            n_per_pain=args.n_per_pain,
            n_pain_steps=args.n_pain_steps,
            C=C,
            T=T,
            pain_scale_max=args.pain_scale_max,
            batch_size=args.sample_batch_size,
            device=device,
        )
        meta_labels = {"pain_labels": pain_labels}

    synth_std = float(X_synth.std())
    print(f"[INFO] Synthetic shape: {X_synth.shape} (n_synth_total, C, T)")
    print(f"[INFO] Synthetic global std (before match): {synth_std:.4f}")

    if args.match_amp and synth_std > 0:
        amp_scale = real_std / synth_std
        print(f"[INFO] Global amp scale (real_std / synth_std) = {amp_scale:.4f}")
        X_synth = X_synth * amp_scale
    else:
        print("[INFO] Global amp scale disabled or synth_std=0; skipping amplitude match.")

    print("[INFO] Computing features for synthetic data...")
    X_features = compute_features_batched(X_synth, sfreq=args.sfreq, batch_size=512)

    out_path = os.path.join(args.model_dir, args.out_name)
    np.savez(
        out_path,
        X_synth=X_synth.astype(np.float32),
        features=X_features.astype(np.float32),
        real_std=real_std,
        synth_std=synth_std,
        match_amp=args.match_amp,
        mode=args.mode,
        **meta_labels,
    )
    print(f"[INFO] Saved label-control results to {out_path}")
    print("[DONE] Label controllability evaluation finished.")


if __name__ == "__main__":
    main()

