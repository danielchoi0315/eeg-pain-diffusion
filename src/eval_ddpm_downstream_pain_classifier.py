# eval_ddpm_downstream_pain_classifier.py

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from compare_real_vs_synth_stats import (
    load_ddpm_model,
    build_cond_tensor,
    sample_synthetic,
)


# ---------------------------------------------------------------------
# Simple 1D CNN classifier for chronic vs healthy
# ---------------------------------------------------------------------


class EEGPainClassifier(nn.Module):
    def __init__(self, in_channels: int = 64, n_time: int = 750):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, C, T)
        h = self.net(x)          # (B, 128, 1)
        h = h.squeeze(-1)        # (B, 128)
        logit = self.fc(h)       # (B, 1)
        return logit.squeeze(-1) # (B,)


class NumpyEEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_split_indices(n_samples: int, train_frac: float = 0.8, seed: int = 1234):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_train = int(train_frac * n_samples)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return train_idx, test_idx


def train_and_eval_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    epochs: int,
    device: torch.device,
    regime_name: str = "",
):
    """
    Train classifier and evaluate on test set.

    Returns:
        metrics dict with test_acc, test_auc, test_f1, best_* versions.
    """
    train_ds = NumpyEEGDataset(X_train, y_train)
    test_ds = NumpyEEGDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    C = X_train.shape[1]
    T = X_train.shape[2]
    model = EEGPainClassifier(in_channels=C, n_time=T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    best_auc = 0.0
    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Evaluate
        model.eval()
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_targets.append(yb.numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        preds = (all_probs >= 0.5).astype(np.int64)

        try:
            auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            # Edge case: only one class present
            auc = 0.5

        acc = accuracy_score(all_targets, preds)
        f1 = f1_score(all_targets, preds)

        if acc > best_acc:
            best_acc = acc
        if auc > best_auc:
            best_auc = auc
        if f1 > best_f1:
            best_f1 = f1

        print(
            f"[{regime_name} EPOCH {epoch:03d}] "
            f"train_loss={np.mean(train_losses):.4f}  "
            f"test_acc={acc:.4f}  test_auc={auc:.4f}  test_f1={f1:.4f}"
        )

    return {
        "test_acc": float(acc),
        "test_auc": float(auc),
        "test_f1": float(f1),
        "best_test_acc": float(best_acc),
        "best_test_auc": float(best_auc),
        "best_test_f1": float(best_f1),
    }


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Downstream chronic vs healthy classifier using real & synthetic data."
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, required=True)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--pain_scale_max", type=float, default=10.0)

    parser.add_argument("--n_synth_per_class", type=int, default=2000)
    parser.add_argument("--sample_batch_size", type=int, default=4)

    # Classifier hyperparams
    parser.add_argument("--clf_batch_size", type=int, default=128)
    parser.add_argument("--clf_epochs", type=int, default=20)

    # Aliases for backward compatibility (so --batch_size / --epochs still work)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Alias for --clf_batch_size")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Alias for --clf_epochs")

    # NEW: amplitude matching flag
    parser.add_argument(
        "--match_amp",
        action="store_true",
        help="Globally rescale synthetic amplitude to match real train std.",
    )

    parser.add_argument("--metrics_name", type=str, default="metrics_downstream_pain_clf.npz")
    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()

    # Handle alias args
    if args.batch_size is not None:
        args.clf_batch_size = args.batch_size
    if args.epochs is not None:
        args.clf_epochs = args.epochs

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("============================================================")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Data path   : {args.data_path}")
    print(f"[INFO] Meta path   : {args.meta_path}")
    print(f"[INFO] Model dir   : {args.model_dir}")
    print(f"[INFO] Checkpoint  : {args.ckpt_name}")
    print(f"[INFO] n_synth_per_class: {args.n_synth_per_class}")
    print(f"[INFO] clf_batch_size   : {args.clf_batch_size}")
    print(f"[INFO] clf_epochs       : {args.clf_epochs}")
    print(f"[INFO] match_amp        : {args.match_amp}")
    print("============================================================")

    # Load data & metadata
    X = np.load(args.data_path)  # (N, C, T)
    meta_npz = np.load(args.meta_path, allow_pickle=True)
    meta = {k: meta_npz[k] for k in meta_npz.files}

    N, C, T = X.shape
    print(f"[INFO] Data shape: {X.shape} (N, C, T)")

    if "group_num" not in meta:
        raise KeyError(f"'group_num' not found in metadata. Available keys: {list(meta.keys())}")

    y_group = meta["group_num"].astype(np.int64)  # 0=control, 1=patient

    # Train/test split (epoch-wise, not subject-wise)
    train_idx, test_idx = make_split_indices(N, train_frac=0.8, seed=args.seed)
    print(f"[INFO] n_train_real = {len(train_idx)}  n_test_real = {len(test_idx)}")

    X_train_real = X[train_idx].astype(np.float32)
    y_train_real = y_group[train_idx].astype(np.float32)
    X_test_real = X[test_idx].astype(np.float32)
    y_test_real = y_group[test_idx].astype(np.float32)

    # Load generative model
    model, timesteps, beta_schedule, cond_dim = load_ddpm_model(
        args.model_dir, args.ckpt_name, device, in_channels=C, use_ema=args.use_ema
    )
    print("[INFO] timesteps     :", timesteps)
    print("[INFO] beta_schedule :", beta_schedule)
    print("[INFO] cond_dim      :", cond_dim)

    rng = np.random.default_rng(args.seed)

    # ----------------------------------------------------------
    # REGIME 1: REAL ONLY (train real -> test real)
    # ----------------------------------------------------------
    print("\n[REGIME 1] REAL ONLY (train real -> test real)")
    metrics_real = train_and_eval_classifier(
        X_train_real,
        y_train_real,
        X_test_real,
        y_test_real,
        batch_size=args.clf_batch_size,
        epochs=args.clf_epochs,
        device=device,
        regime_name="REAL ONLY",
    )
    print(f"[RESULT] REAL ONLY: {metrics_real}")

    # ----------------------------------------------------------
    # REGIME 2: SYNTH ONLY (train synth -> test real)
    # ----------------------------------------------------------
    print("\n[REGIME 2] SYNTH ONLY (train synth -> test real)")

    # Build balanced synthetic conditioning
    n_per_class = args.n_synth_per_class
    cond_list = []
    labels_synth = []

    for g in [0, 1]:
        idx_g = np.where(y_group == g)[0]
        if len(idx_g) == 0:
            raise RuntimeError(f"No epochs found for group {g} in metadata.")

        chosen = rng.choice(idx_g, size=n_per_class, replace=True)
        cond_g = build_cond_tensor(meta, chosen, cond_dim, args.pain_scale_max, device)
        cond_list.append(cond_g)
        labels_synth.append(np.full((n_per_class,), g, dtype=np.float32))

    cond_all = torch.cat(cond_list, dim=0)  # (2*n_per_class, cond_dim)
    y_synth = np.concatenate(labels_synth, axis=0)  # (2*n_per_class,)

    print(f"[INFO] Sampling {cond_all.shape[0]} synthetic epochs (balanced classes)...")
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
    print(f"[INFO] Synthetic shape: {X_synth.shape} (n_synth_total, C, T)")

    # === NEW: amplitude matching ===
    if args.match_amp:
        real_std = float(X_train_real.std())
        synth_std = float(X_synth.std())
        scale = real_std / (synth_std + 1e-8)
        print(f"[INFO] Global amp scale (real_std / synth_std) = {scale:.4f}")
        X_synth *= scale

    metrics_synth = train_and_eval_classifier(
        X_synth,
        y_synth,
        X_test_real,
        y_test_real,
        batch_size=args.clf_batch_size,
        epochs=args.clf_epochs,
        device=device,
        regime_name="SYNTH ONLY",
    )
    print(f"[RESULT] SYNTH ONLY: {metrics_synth}")

    # ----------------------------------------------------------
    # REGIME 3: REAL + SYNTH (train real+synth -> test real)
    # ----------------------------------------------------------
    print("\n[REGIME 3] REAL + SYNTH (train real+synth -> test real)")

    X_train_realsynth = np.concatenate([X_train_real, X_synth], axis=0)
    y_train_realsynth = np.concatenate([y_train_real, y_synth], axis=0)

    metrics_realsynth = train_and_eval_classifier(
        X_train_realsynth,
        y_train_realsynth,
        X_test_real,
        y_test_real,
        batch_size=args.clf_batch_size,
        epochs=args.clf_epochs,
        device=device,
        regime_name="REAL+SYNTH",
    )
    print(f"[RESULT] REAL + SYNTH: {metrics_realsynth}")

    # ----------------------------------------------------------
    # Save metrics
    # ----------------------------------------------------------
    out = {
        "n_train_real": float(len(train_idx)),
        "n_test_real": float(len(test_idx)),
        "n_synth_total": float(X_synth.shape[0]),
        "sfreq": float(args.sfreq),
        "timesteps": float(timesteps),
        "beta_schedule": beta_schedule,
        # REAL ONLY
        "real_train_test_test_acc": metrics_real["test_acc"],
        "real_train_test_test_auc": metrics_real["test_auc"],
        "real_train_test_test_f1": metrics_real["test_f1"],
        "real_train_test_best_test_acc": metrics_real["best_test_acc"],
        "real_train_test_best_test_auc": metrics_real["best_test_auc"],
        "real_train_test_best_test_f1": metrics_real["best_test_f1"],
        # SYNTH ONLY
        "synth_train_test_test_acc": metrics_synth["test_acc"],
        "synth_train_test_test_auc": metrics_synth["test_auc"],
        "synth_train_test_test_f1": metrics_synth["test_f1"],
        "synth_train_test_best_test_acc": metrics_synth["best_test_acc"],
        "synth_train_test_best_test_auc": metrics_synth["best_test_auc"],
        "synth_train_test_best_test_f1": metrics_synth["best_test_f1"],
        # REAL + SYNTH
        "realsynth_train_test_test_acc": metrics_realsynth["test_acc"],
        "realsynth_train_test_test_auc": metrics_realsynth["test_auc"],
        "realsynth_train_test_test_f1": metrics_realsynth["test_f1"],
        "realsynth_train_test_best_test_acc": metrics_realsynth["best_test_acc"],
        "realsynth_train_test_best_test_auc": metrics_realsynth["best_test_auc"],
        "realsynth_train_test_best_test_f1": metrics_realsynth["best_test_f1"],
    }

    out_path = os.path.join(args.model_dir, args.metrics_name)
    np.savez(out_path, **out)
    print(f"[INFO] Saved downstream metrics to {out_path}")
    print("[DONE] Downstream pain-vs-control evaluation finished.")


if __name__ == "__main__":
    main()

