import argparse
import os
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    # Infer cond_dim from state dict
    cond_dim = 2
    for k, v in state_dict.items():
        if "time_cond_mlp.cond_mlp.0.weight" in k and v.dim() == 2:
            cond_dim = v.shape[1]
            break

    # Infer in_channels, base_channels from first conv
    in_channels = 64
    base_channels = 64
    for k, v in state_dict.items():
        if "in_conv.weight" in k or "init_conv.weight" in k:
            base_channels = v.shape[0]
            in_channels = v.shape[1]
            break

    # IMPORTANT: ignore any stored model_config to avoid mismatched kwargs
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
# Meta + features
# -----------------------------------------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_meta_fields(meta: np.lib.npyio.NpzFile, n_epochs: int) -> Dict[str, np.ndarray]:
    keys = list(meta.keys())
    print(f"[DEBUG] meta keys: {keys}")

    # ---------- Subject IDs ----------
    subj_key = None
    candidate_subj_names = ["subject", "subj", "sub_id", "subject_id", "participant", "pid"]
    for k in keys:
        lk = k.lower()
        if any(name in lk for name in candidate_subj_names) and meta[k].shape[0] == n_epochs:
            subj_key = k
            break

    if subj_key is None:
        # Fallback: each epoch = unique "subject" (still valid for subject-wise split, just conservative)
        print("[WARN] Could not infer subject id key from meta; using unique subject id per epoch.")
        subject_ids = np.arange(n_epochs, dtype=np.int64)
    else:
        subject_ids = meta[subj_key]

    # ---------- Binary label ----------
    label_key = None
    label_array = None
    candidate_label_names = [
        "group", "label", "is_patient", "patient", "class",
        "pain", "condition", "status"
    ]

    # Priority: names we like
    for k in keys:
        arr = meta[k]
        if arr.shape[0] != n_epochs:
            continue
        if not np.issubdtype(arr.dtype, np.number):
            continue
        name = k.lower()
        if any(s in name for s in candidate_label_names):
            uniq = np.unique(arr)
            if 2 <= uniq.size <= 4:
                label_key = k
                label_array = arr
                break

    # Fallback: any numeric field with 2–4 unique values
    if label_key is None:
        for k in keys:
            arr = meta[k]
            if arr.shape[0] != n_epochs:
                continue
            if not np.issubdtype(arr.dtype, np.number):
                continue
            uniq = np.unique(arr)
            if 2 <= uniq.size <= 4:
                label_key = k
                label_array = arr
                break

    # Last resort: numeric field binarized by median
    if label_array is None:
        for k in keys:
            arr = meta[k]
            if arr.shape[0] != n_epochs:
                continue
            if np.issubdtype(arr.dtype, np.number):
                label_array = arr
                break

    if label_array is None:
        raise RuntimeError("Could not infer binary label field from meta npz.")

    uniq = np.unique(label_array)
    if uniq.size == 2:
        label_bin = (label_array == uniq.max()).astype(np.int64)
    else:
        med = np.median(label_array)
        label_bin = (label_array > med).astype(np.int64)

    return {
        "subject_ids": subject_ids,
        "labels": label_bin,
    }


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
        xb = x[start:start + batch_size]  # (B, C, T)
        psd = np.abs(np.fft.rfft(xb, axis=-1)) ** 2  # (B, C, F)
        band_feats = []
        for m in masks:
            bp = psd[..., m].mean(axis=-1)  # (B, C)
            bp_avg = bp.mean(axis=-1)       # (B,)
            band_feats.append(bp_avg)
        std_feat = xb.std(axis=(1, 2), ddof=0)  # (B,)
        feats_b = np.stack(band_feats + [std_feat], axis=1)  # (B, 6)
        feats_all.append(feats_b.astype(np.float32))

    feats = np.concatenate(feats_all, axis=0)
    return feats  # (N, 6)


# -----------------------------------------------------------
# Classifier
# -----------------------------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    epochs: int,
    device: str,
) -> SimpleMLP:
    device_t = torch.device(device)
    X_train_t = torch.from_numpy(X_train).float().to(device_t)
    y_train_t = torch.from_numpy(y_train).float().to(device_t)
    X_val_t = torch.from_numpy(X_val).float().to(device_t)
    y_val_t = torch.from_numpy(y_val).float().to(device_t)

    model = SimpleMLP(in_dim=X_train.shape[1]).to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    n_train = X_train.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device_t)
        X_train_t = X_train_t[perm]
        y_train_t = y_train_t[perm]

        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            xb = X_train_t[start:start + batch_size]
            yb = y_train_t[start:start + batch_size]
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t)
            val_loss = criterion(logits_val, y_val_t).item()

        print(f"[CLF] Epoch {epoch+1:03d} train_loss={total_loss / max(n_batches,1):.4f} val_loss={val_loss:.4f}")

    return model


def eval_classifier(
    model: SimpleMLP,
    X: np.ndarray,
    y: np.ndarray,
    device: str,
) -> Dict[str, float]:
    device_t = torch.device(device)
    X_t = torch.from_numpy(X).float().to(device_t)
    y_t = torch.from_numpy(y).float().to(device_t)

    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    y_np = y_t.cpu().numpy()
    probs_np = probs.cpu().numpy()
    preds_np = preds.cpu().numpy()

    eps = 1e-7
    acc = float((preds_np == y_np).mean())
    tp = float(((preds_np == 1) & (y_np == 1)).sum())
    fp = float(((preds_np == 1) & (y_np == 0)).sum())
    fn = float(((preds_np == 0) & (y_np == 1)).sum())
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # very simple AUROC approximation (not perfect but fine for ranking)
    order = np.argsort(probs_np)
    y_sorted = y_np[order]
    cum_pos = np.cumsum(y_sorted[::-1])
    cum_neg = np.cumsum(1 - y_sorted[::-1])
    auc = float((cum_pos * (1 - y_sorted[::-1])).sum() / (cum_pos[-1] * cum_neg[-1] + eps))

    return {"acc": acc, "auc": auc, "f1": f1}


# -----------------------------------------------------------
# Synthetic sampling
# -----------------------------------------------------------

def sample_synthetic_balanced(
    diffusion: GaussianDiffusion1D,
    cond_dim: int,
    n_per_class: int,
    C: int,
    T: int,
    pain_scale_max: float,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    device_t = torch.device(device)
    diffusion.to(device_t)

    xs = []
    ys = []

    for label in [0, 1]:
        remaining = n_per_class
        while remaining > 0:
            cur_bs = min(batch_size, remaining)
            if label == 0:
                pain_level = 0.0
            else:
                pain_level = 0.7 * pain_scale_max
            cond = torch.tensor(
                [[float(label), pain_level / pain_scale_max]] * cur_bs,
                dtype=torch.float32,
                device=device_t,
            )
            x_synth = diffusion.p_sample_loop(
                shape=(cur_bs, C, T),
                cond=cond,
                noise=None,
            )
            xs.append(x_synth.cpu().numpy())
            ys.append(np.full(cur_bs, label, dtype=np.int64))
            remaining -= cur_bs

    X_synth = np.concatenate(xs, axis=0)
    y_synth = np.concatenate(ys, axis=0)
    return X_synth, y_synth


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Subject-wise downstream pain classifier with synthetic augmentation.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--pain_scale_max", type=float, default=10.0)
    parser.add_argument("--n_synth_per_class", type=int, default=2000)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--clf_batch_size", type=int, default=128)
    parser.add_argument("--clf_epochs", type=int, default=20)
    parser.add_argument("--metrics_name", type=str, default="metrics_downstream_pain_clf_subjectwise_gamma.npz")
    parser.add_argument("--match_amp", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)

    device = args.device

    print("============================================================")
    print("[INFO] Subject-wise downstream pain classifier (real / synth / real+synth)")
    print("============================================================")
    print(f"[INFO] DATA      = {args.data_path}")
    print(f"[INFO] META      = {args.meta_path}")
    print(f"[INFO] MODEL_DIR = {args.model_dir}")
    print(f"[INFO] CKPT      = {args.ckpt_name}")
    print(f"[INFO] DEVICE    = {device}")
    print(f"[INFO] match_amp = {args.match_amp}")
    print("============================================================")

    # ----- Load data & meta -----
    data = np.load(args.data_path)
    N, C, T = data.shape
    meta = np.load(args.meta_path)
    fields = infer_meta_fields(meta, N)
    subj_ids = fields["subject_ids"]
    labels = fields["labels"]

    real_std = float(data.std())
    print(f"[INFO] Data shape: {data.shape} (N, C, T)")
    print(f"[INFO] Global real std: {real_std:.4f}")

    # Subject-wise split
    uniq_subj = np.unique(subj_ids)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(uniq_subj)
    n_subj = len(uniq_subj)
    n_train_subj = int(0.7 * n_subj)
    n_val_subj = int(0.15 * n_subj)
    train_subj = uniq_subj[:n_train_subj]
    val_subj = uniq_subj[n_train_subj:n_train_subj + n_val_subj]
    test_subj = uniq_subj[n_train_subj + n_val_subj:]

    idx_train = np.where(np.isin(subj_ids, train_subj))[0]
    idx_val = np.where(np.isin(subj_ids, val_subj))[0]
    idx_test = np.where(np.isin(subj_ids, test_subj))[0]

    print(f"[INFO] n_subj_train={len(train_subj)} n_subj_val={len(val_subj)} n_subj_test={len(test_subj)}")
    print(f"[INFO] n_train={len(idx_train)} n_val={len(idx_val)} n_test={len(idx_test)}")

    diffusion, timesteps, beta_schedule, cond_dim = build_ddpm_model(
        args.model_dir, args.ckpt_name, device=device
    )
    print(f"[INFO] Loaded DDPM with timesteps={timesteps} schedule={beta_schedule} cond_dim={cond_dim}")

    # ----- Features -----
    print("[INFO] Computing features for real data...")
    X_features = compute_features_batched(data, sfreq=args.sfreq, batch_size=512)

    X_train_real = X_features[idx_train]
    y_train_real = labels[idx_train]
    X_val_real = X_features[idx_val]
    y_val_real = labels[idx_val]
    X_test_real = X_features[idx_test]
    y_test_real = labels[idx_test]

    # ----- Regime 1: real only -----
    print("\n[REGIME 1] REAL ONLY (train real -> test real)")
    clf_real = train_classifier(
        X_train_real,
        y_train_real,
        X_val_real,
        y_val_real,
        batch_size=args.clf_batch_size,
        epochs=args.clf_epochs,
        device=device,
    )
    metrics_real = eval_classifier(clf_real, X_test_real, y_test_real, device=device)
    print(f"[RESULT] REAL ONLY: acc={metrics_real['acc']:.4f} auc={metrics_real['auc']:.4f} f1={metrics_real['f1']:.4f}")

    # ----- Sample synthetic -----
    print("\n[INFO] Sampling synthetic data for subject-wise classifier...")
    X_synth, y_synth = sample_synthetic_balanced(
        diffusion=diffusion,
        cond_dim=cond_dim,
        n_per_class=args.n_synth_per_class,
        C=C,
        T=T,
        pain_scale_max=args.pain_scale_max,
        batch_size=args.sample_batch_size,
        device=device,
    )

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
    X_synth_feat = compute_features_batched(X_synth, sfreq=args.sfreq, batch_size=512)

    # ----- Regime 2: synth only -----
    print("\n[REGIME 2] SYNTH ONLY (train synth -> test real)")
    idx_synth = np.arange(X_synth_feat.shape[0])
    rng.shuffle(idx_synth)
    n_synth_train = int(0.8 * len(idx_synth))
    idx_synth_train = idx_synth[:n_synth_train]
    idx_synth_val = idx_synth[n_synth_train:]

    X_train_synth = X_synth_feat[idx_synth_train]
    y_train_synth = y_synth[idx_synth_train]
    X_val_synth = X_synth_feat[idx_synth_val]
    y_val_synth = y_synth[idx_synth_val]

    clf_synth = train_classifier(
        X_train_synth,
        y_train_synth,
        X_val_synth,
        y_val_synth,
        batch_size=args.clf_batch_size,
        epochs=args.clf_epochs,
        device=device,
    )
    metrics_synth = eval_classifier(clf_synth, X_test_real, y_test_real, device=device)
    print(f"[RESULT] SYNTH ONLY: acc={metrics_synth['acc']:.4f} auc={metrics_synth['auc']:.4f} f1={metrics_synth['f1']:.4f}")

    # ----- Regime 3: real + synth -----
    print("\n[REGIME 3] REAL + SYNTH (train real+synth -> test real)")
    X_train_mix = np.concatenate([X_train_real, X_train_synth], axis=0)
    y_train_mix = np.concatenate([y_train_real, y_train_synth], axis=0)
    X_val_mix = np.concatenate([X_val_real, X_val_synth], axis=0)
    y_val_mix = np.concatenate([y_val_real, y_val_synth], axis=0)

    clf_mix = train_classifier(
        X_train_mix,
        y_train_mix,
        X_val_mix,
        y_val_mix,
        batch_size=args.clf_batch_size,
        epochs=args.clf_epochs,
        device=device,
    )
    metrics_mix = eval_classifier(clf_mix, X_test_real, y_test_real, device=device)
    print(f"[RESULT] REAL+SYNTH: acc={metrics_mix['acc']:.4f} auc={metrics_mix['auc']:.4f} f1={metrics_mix['f1']:.4f}")

    # ----- Save -----
    out_path = os.path.join(args.model_dir, args.metrics_name)
    np.savez(
        out_path,
        acc_real=metrics_real["acc"],
        auc_real=metrics_real["auc"],
        f1_real=metrics_real["f1"],
        acc_synth=metrics_synth["acc"],
        auc_synth=metrics_synth["auc"],
        f1_synth=metrics_synth["f1"],
        acc_mix=metrics_mix["acc"],
        auc_mix=metrics_mix["auc"],
        f1_mix=metrics_mix["f1"],
        n_train_real=len(idx_train),
        n_test_real=len(idx_test),
        n_synth_total=X_synth.shape[0],
        match_amp=args.match_amp,
    )
    print(f"[INFO] Saved subject-wise downstream metrics to {out_path}")
    print("[DONE] Subject-wise downstream pain-vs-control evaluation finished.")


if __name__ == "__main__":
    main()

