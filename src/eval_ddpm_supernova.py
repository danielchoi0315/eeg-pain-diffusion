# eval_ddpm_supernova.py

import argparse
import os
import numpy as np
import torch

from compare_real_vs_synth_stats import (
    load_ddpm_model,
    build_cond_tensor,
    sample_synthetic,
    compute_amp_std_metrics,
    compute_psd_metrics,
)


def main():
    parser = argparse.ArgumentParser(
        description="Supernova-style REAL vs SYNTH EEG metrics (amp std + bandpower)."
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--pain_scale_max", type=float, default=10.0)
    parser.add_argument("--n_synth", type=int, default=512)
    parser.add_argument("--batch_size_val", type=int, default=64, help="(Unused; kept for compatibility.)")
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--metrics_name", type=str, default="metrics_supernova.npz")
    parser.add_argument("--seed", type=int, default=123)

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

    # Load model (same as compare_real_vs_synth_stats)
    model, timesteps, beta_schedule, cond_dim = load_ddpm_model(
        args.model_dir, args.ckpt_name, device, in_channels=C, use_ema=args.use_ema
    )
    print("[INFO] timesteps     :", timesteps)
    print("[INFO] beta_schedule :", beta_schedule)
    print("[INFO] cond_dim      :", cond_dim)

    rng = np.random.default_rng(args.seed)
    n_eval = min(args.n_synth, N)
    indices = rng.choice(N, size=n_eval, replace=False)

    X_real = X[indices].astype(np.float32)
    cond_all = build_cond_tensor(meta, indices, cond_dim, args.pain_scale_max, device)

    print("[INFO] Sampling synthetic epochs for Supernova metrics...")
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
    amp_metrics = compute_amp_std_metrics(X_real, X_synth)
    psd_metrics = compute_psd_metrics(X_real, X_synth, sfreq=args.sfreq)

    metrics = {}
    metrics.update(amp_metrics)
    metrics.update(psd_metrics)
    metrics["n_eval"] = float(n_eval)
    metrics["sfreq"] = float(args.sfreq)
    metrics["timesteps"] = float(timesteps)

    # Pretty print in the "Supernova" style
    print("==== Supernova-style metrics (summary) ====")
    for k in [
        "amp_std_ratio_max",
        "amp_std_ratio_mean",
        "amp_std_ratio_median",
        "amp_std_ratio_min",
        "amp_std_real_mean",
        "amp_std_synth_mean",
        "band_alpha_mse",
        "band_alpha_rel_err",
        "band_beta_mse",
        "band_beta_rel_err",
        "band_delta_mse",
        "band_delta_rel_err",
        "band_gamma_mse",
        "band_gamma_rel_err",
        "band_theta_mse",
        "band_theta_rel_err",
        "n_eval",
        "sfreq",
        "timesteps",
    ]:
        if k in metrics:
            print(f"{k:<30}: {metrics[k]}")
    print("===========================================")

    out_path = os.path.join(args.model_dir, args.metrics_name)
    np.savez(out_path, **metrics)
    print(f"[INFO] Saved Supernova metrics to {out_path}")


if __name__ == "__main__":
    main()

