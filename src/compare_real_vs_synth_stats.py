# compare_real_vs_synth_stats.py

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

from train_ddpm_cond_v2_gamma import CondUNet1D


# ---------------------------------------------------------------------
# Beta schedule / diffusion helpers
# ---------------------------------------------------------------------


def make_beta_schedule(timesteps: int, schedule: str, device: torch.device) -> torch.Tensor:
    """Create a beta schedule compatible with your training."""
    schedule = schedule.lower()
    if schedule == "linear":
        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
    else:
        raise ValueError(f"Unsupported beta_schedule: {schedule}")
    return betas


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------


def load_ddpm_model(
    model_dir: str,
    ckpt_name: str,
    device: torch.device,
    in_channels: int,
    use_ema: bool = True,
):
    """
    Load CondUNet1D from a checkpoint saved by train_ddpm_cond_v2_gamma.py.

    Returns:
        model        : CondUNet1D on device, in eval() mode
        timesteps    : int
        beta_schedule: str ("linear", etc.)
        cond_dim     : int
    """
    ckpt_path = os.path.join(model_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # Figure out which state dict to use
    state = None
    if isinstance(ckpt, dict):
        if use_ema and "ema_state_dict" in ckpt:
            state = ckpt["ema_state_dict"]
            print("[INFO] Loaded weights from key: ema_state_dict")
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
            print("[INFO] Loaded weights from key: model_state_dict")

    if state is None:
        # Assume checkpoint *is* the state dict
        state = ckpt
        print("[WARN] Using raw checkpoint as state_dict (no ema_state_dict/model_state_dict keys).")

    # Infer cond_dim from cond_mlp weight
    cond_dim = 2
    cond_w_key = None
    for k in state.keys():
        if "cond_mlp.0.weight" in k:
            cond_w_key = k
            break
    if cond_w_key is not None:
        cond_dim = state[cond_w_key].shape[1]
        print(f"[INFO] Inferred cond_dim={cond_dim} from key {cond_w_key}")
    else:
        print("[WARN] Could not infer cond_dim from state dict. Defaulting to cond_dim=2.")

    # Defaults; override if model_config is present
    timesteps = 400
    beta_schedule = "linear"
    if isinstance(ckpt, dict) and "model_config" in ckpt:
        cfg = ckpt["model_config"]
        timesteps = int(cfg.get("timesteps", timesteps))
        beta_schedule = str(cfg.get("beta_schedule", beta_schedule))
        print("[INFO] Model config from ckpt (for logging only):")
        for k, v in cfg.items():
            print(f"  {k:<14} = {v}")
    else:
        print("[INFO] No model_config found; using timesteps=400, beta_schedule='linear' by default.")

    # Instantiate model with the SAME class used in training
    model = CondUNet1D(
        in_channels=in_channels,
        cond_dim=cond_dim,
        base_channels=64,
        time_emb_dim=256,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()

    return model, timesteps, beta_schedule, cond_dim


# ---------------------------------------------------------------------
# Conditioning from metadata
# ---------------------------------------------------------------------


def build_cond_tensor(meta: dict, indices: np.ndarray, cond_dim: int,
                      pain_scale_max: float, device: torch.device) -> torch.Tensor:
    """
    Build conditioning vectors from metadata.

    Assumes:
        meta["group_num"]      : 0 = control, 1 = patient
        meta["pain_intensity"] : 0..pain_scale_max

    If cond_dim == 2:
        cond = [group_num (0/1), pain_norm]

    If cond_dim == 3:
        cond = [one_hot_control, one_hot_patient, pain_norm]
    """
    group = meta["group_num"][indices].astype(np.int64)          # (N,)
    pain = meta["pain_intensity"][indices].astype(np.float32)    # (N,)

    pain_norm = np.clip(pain / pain_scale_max, 0.0, 1.0)

    if cond_dim == 2:
        cond_np = np.stack(
            [group.astype(np.float32), pain_norm],
            axis=-1
        )
    elif cond_dim == 3:
        g0 = (group == 0).astype(np.float32)
        g1 = (group == 1).astype(np.float32)
        cond_np = np.stack([g0, g1, pain_norm], axis=-1)
    else:
        raise ValueError(f"Unsupported cond_dim={cond_dim}; expected 2 or 3.")

    cond = torch.from_numpy(cond_np).to(device)
    return cond


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------


@torch.no_grad()
def sample_synthetic(
    model: torch.nn.Module,
    cond_all: torch.Tensor,
    timesteps: int,
    beta_schedule: str,
    sample_batch_size: int,
    n_channels: int,
    n_time: int,
    device: torch.device,
) -> torch.Tensor:
    """
    DDPM sampling loop.

    Args:
        cond_all: (N_synth, cond_dim) tensor on device
    Returns:
        X_synth: (N_synth, C, T) on CPU
    """
    model.eval()

    total = cond_all.shape[0]
    betas = make_beta_schedule(timesteps, beta_schedule, device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    out_chunks = []
    start = 0
    while start < total:
        bsz = min(sample_batch_size, total - start)
        c = cond_all[start:start + bsz]  # already on device

        x = torch.randn(bsz, n_channels, n_time, device=device)

        for t in reversed(range(timesteps)):
            t_batch = torch.full((bsz,), t, device=device, dtype=torch.long)
            eps_theta = model(x, t_batch.float(), c)

            # Safety: align time dimension if slight mismatch
            if eps_theta.shape[-1] > x.shape[-1]:
                eps_theta = eps_theta[..., : x.shape[-1]]
            elif eps_theta.shape[-1] < x.shape[-1]:
                pad = x.shape[-1] - eps_theta.shape[-1]
                eps_theta = F.pad(eps_theta, (0, pad))

            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_bar_t = alpha_bars[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1.0 / torch.sqrt(alpha_t)) * (
                x - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t) * eps_theta
            ) + torch.sqrt(beta_t) * noise

        out_chunks.append(x.cpu())
        start += bsz

    X_synth = torch.cat(out_chunks, dim=0)
    return X_synth


# ---------------------------------------------------------------------
# Metrics: amplitude std + PSD / bandpower
# ---------------------------------------------------------------------


def compute_amp_std_metrics(X_real: np.ndarray, X_synth: np.ndarray):
    """
    X_real, X_synth: (N, C, T) arrays
    """
    real_std = X_real.std(axis=2)   # (N, C)
    synth_std = X_synth.std(axis=2)

    real_flat = real_std.reshape(-1)
    synth_flat = synth_std.reshape(-1)

    ratio = synth_flat / (real_flat + 1e-8)

    metrics = {
        "amp_std_ratio_max": float(ratio.max()),
        "amp_std_ratio_mean": float(ratio.mean()),
        "amp_std_ratio_median": float(np.median(ratio)),
        "amp_std_ratio_min": float(ratio.min()),
        "amp_std_real_mean": float(real_flat.mean()),
        "amp_std_synth_mean": float(synth_flat.mean()),
    }
    return metrics


def _compute_bandpowers(X: np.ndarray, sfreq: float, bands: dict):
    """
    Compute relative bandpowers per epoch*channel.

    Returns:
        dict[name] -> (N*C,) array of relative power
    """
    N, C, T = X.shape
    X_flat = X.reshape(N * C, T)

    freqs = np.fft.rfftfreq(T, d=1.0 / sfreq)
    fft = np.fft.rfft(X_flat, axis=1)
    psd = (np.abs(fft) ** 2) / float(T)

    total_power = psd.sum(axis=1, keepdims=True) + 1e-12

    out = {}
    for name, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(idx):
            out[name] = np.zeros((N * C,), dtype=np.float32)
        else:
            bp = psd[:, idx].sum(axis=1) / total_power.squeeze(1)
            out[name] = bp.astype(np.float32)
    return out


def compute_psd_metrics(X_real: np.ndarray, X_synth: np.ndarray, sfreq: float):
    """
    Compute MSE and relative error on relative bandpowers for
    delta / theta / alpha / beta / gamma.
    """
    bands = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }

    bp_real = _compute_bandpowers(X_real, sfreq, bands)
    bp_synth = _compute_bandpowers(X_synth, sfreq, bands)

    metrics = {}
    for name in bands.keys():
        r = bp_real[name]
        s = bp_synth[name]
        mse = float(((s - r) ** 2).mean())
        rel = float(np.abs(s.mean() - r.mean()) / (np.abs(r.mean()) + 1e-8))
        metrics[f"band_{name}_mse"] = mse
        metrics[f"band_{name}_rel_err"] = rel

    return metrics


# ---------------------------------------------------------------------
# CLI main: REAL vs SYNTH stats for a single model
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare REAL vs SYNTH EEG stats (amp std + PSD) for one DDPM model."
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_ema", action="store_true", help="Prefer ema_state_dict if present.")
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--pain_scale_max", type=float, default=10.0)
    parser.add_argument("--n_synth", type=int, default=512)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--metrics_name", type=str, default="metrics_amp_psd_compare.npz")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print("============================================================")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Data path   : {args.data_path}")
    print(f"[INFO] Meta path   : {args.meta_path}")
    print(f"[INFO] Model dir   : {args.model_dir}")
    print(f"[INFO] Checkpoint  : {args.ckpt_name}")
    print(f"[INFO] n_synth     : {args.n_synth}")
    print("============================================================")

    # Load data & metadata
    X = np.load(args.data_path)  # (N, C, T)
    meta_npz = np.load(args.meta_path, allow_pickle=True)
    meta = {k: meta_npz[k] for k in meta_npz.files}

    N, C, T = X.shape
    print(f"[INFO] Data shape: {X.shape} (N, C, T)")

    # Load model
    model, timesteps, beta_schedule, cond_dim = load_ddpm_model(
        args.model_dir, args.ckpt_name, device, in_channels=C, use_ema=args.use_ema
    )
    print("[INFO] timesteps     :", timesteps)
    print("[INFO] beta_schedule :", beta_schedule)
    print("[INFO] cond_dim      :", cond_dim)

    # Choose real subset + build cond for synth
    rng = np.random.default_rng(args.seed)
    n_eval = min(args.n_synth, N)
    indices = rng.choice(N, size=n_eval, replace=False)

    X_real = X[indices].astype(np.float32)
    cond_all = build_cond_tensor(meta, indices, cond_dim, args.pain_scale_max, device)

    print("[INFO] Running REAL vs SYNTH stats (amp std + PSD)")
    print("[INFO] Sampling synthetic epochs...")
    X_synth_t = sample_synthetic(
        model=model,
        cond_all=cond_all,
        timesteps=timesteps,
        beta_schedule=beta_schedule,
        sample_batch_size=args.sample_batch_size,
        n_channels=C,
        n_time=T,
        device=device,
    )
    X_synth = X_synth_t.cpu().numpy()

    print(f"[INFO] Synthetic shape: {X_synth.shape} (n_synth, C, T)")

    # Compute metrics
    print("[INFO] Computing amplitude std statistics...")
    amp_metrics = compute_amp_std_metrics(X_real, X_synth)

    print("[INFO] Computing bandpower / PSD statistics...")
    psd_metrics = compute_psd_metrics(X_real, X_synth, sfreq=args.sfreq)

    metrics = {}
    metrics.update(amp_metrics)
    metrics.update(psd_metrics)
    metrics["n_eval"] = float(n_eval)
    metrics["sfreq"] = float(args.sfreq)
    metrics["timesteps"] = float(timesteps)

    # Pretty print
    print("==== Amplitude STD summary ====")
    for k in sorted(amp_metrics.keys()):
        print(f"{k:<30}: {amp_metrics[k]}")
    print("================================")

    print("==== Bandpower summary (MSE / rel_err) ====")
    for band in ["delta", "theta", "alpha", "beta", "gamma"]:
        mse_key = f"band_{band}_mse"
        rel_key = f"band_{band}_rel_err"
        print(f"{mse_key:<30}: {metrics[mse_key]}")
        print(f"{rel_key:<30}: {metrics[rel_key]}")
    print("===========================================")

    # Save
    out_path = os.path.join(args.model_dir, args.metrics_name)
    np.savez(out_path, **metrics)
    print(f"[INFO] Saved metrics to {out_path}")


if __name__ == "__main__":
    main()

