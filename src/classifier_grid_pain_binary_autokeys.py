#!/usr/bin/env python

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# Utility: metrics (acc, F1, AUC)
# ============================================================

def compute_accuracy(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    return float((y_true == y_pred).mean())


def compute_f1(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    if precision + recall == 0:
        return 0.0
    return float(2 * (precision * recall) / (precision + recall))


def compute_auc_binary(y_true, y_score):
    """
    Simple AUC implementation without scikit-learn.
    y_true ∈ {0,1}, y_score ∈ R.
    """
    y_true = y_true.astype(int)
    y_score = y_score.astype(float)

    # Sort by score descending
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    P = np.sum(y_true_sorted == 1)
    N = np.sum(y_true_sorted == 0)
    if P == 0 or N == 0:
        return float("nan")

    tp = 0.0
    fp = 0.0
    prev_tpr = 0.0
    prev_fpr = 0.0
    auc = 0.0
    prev_score = None

    for i in range(len(y_true_sorted)):
        score = y_score[order[i]]
        if prev_score is None or score != prev_score:
            # update AUC with trapezoid from previous point
            tpr = tp / P
            fpr = fp / N
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
            prev_tpr = tpr
            prev_fpr = fpr
            prev_score = score

        if y_true_sorted[i] == 1:
            tp += 1.0
        else:
            fp += 1.0

    # last segment to (1,1)
    tpr = tp / P
    fpr = fp / N
    auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0

    return float(auc)


# ============================================================
# Utility: infer label key from metadata
# ============================================================

def infer_label_key(meta, n_samples):
    """
    Try to automatically find a binary label vector in meta:
      - 1D array of length n_samples
      - integer-like dtype
      - 2 unique values
    """
    candidates = []
    for k in meta.files:
        arr = meta[k]
        if arr.shape == (n_samples,):
            if arr.dtype.kind in ("i", "u", "b", "f"):
                uniq = np.unique(arr)
                if 2 <= len(uniq) <= 5:
                    candidates.append((k, len(uniq), uniq))

    if not candidates:
        raise RuntimeError(
            f"Could not find a suitable binary label key in meta. "
            f"Available keys: {meta.files}"
        )

    # Prefer exactly 2 unique values
    for k, nuniq, uniq in candidates:
        if nuniq == 2:
            print(f"[INFO] Inferred label_key='{k}' with uniques={uniq}")
            return k

    # Otherwise just take the first candidate
    k, nuniq, uniq = candidates[0]
    print(f"[INFO] Inferred label_key='{k}' (non-binary but using anyway), uniques={uniq}")
    return k


# ============================================================
# Utility: load synth from v2.1 NPZ
# ============================================================

def load_synth_v21(path):
    """
    Load synthetic epochs and binary labels from the v2.1 label_control NPZ.
    Expected keys:
      - 'X_synth'   : (Ns, C, T)
      - 'pain_labels' or 'group_labels' or 'label_pain_binary'
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Synth NPZ not found: {path}")

    npz = np.load(path)
    X_synth = npz["X_synth"]  # (Ns, C, T)

    label_key = None
    if "label_pain_binary" in npz.files:
        label_key = "label_pain_binary"
    elif "group_labels" in npz.files:
        label_key = "group_labels"
    elif "pain_labels" in npz.files:
        label_key = "pain_labels"
    else:
        raise RuntimeError(
            f"Could not find label vector in synth NPZ {path}. "
            f"Keys={npz.files}"
        )

    y_raw = npz[label_key]
    # Convert to binary {0,1} if needed
    if label_key == "pain_labels":
        # assume numeric pain scores; treat zeros as 0, >0 as 1
        y_synth = (y_raw > 0).astype(int)
    else:
        y_synth = y_raw.astype(int)

    print(f"[INFO] Loaded synth from {path}")
    print(f"[INFO] X_synth shape: {X_synth.shape}, label_key={label_key}")
    uniq = np.unique(y_synth)
    print(f"[INFO] Synth label uniques: {uniq}, counts={[(int(u), int(np.sum(y_synth==u))) for u in uniq]}")

    return X_synth, y_synth


# ============================================================
# Simple logistic classifier in PyTorch
# ============================================================

class LogisticClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x: (B, D)
        logits = self.linear(x).squeeze(-1)
        return logits


def train_classifier(X_train, y_train, X_val, y_val, n_epochs=20, batch_size=128, lr=1e-3, device=None):
    """
    Train a simple logistic classifier on flattened features.
    X_*: numpy arrays (N, D)
    y_*: numpy arrays (N,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Standardize features using train stats
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train_std = (X_train - mu) / sigma
    X_val_std = (X_val - mu) / sigma

    X_train_t = torch.from_numpy(X_train_std.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.float32))
    X_val_t = torch.from_numpy(X_val_std.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.astype(np.float32))

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LogisticClassifier(in_dim=X_train_t.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        if n_batches > 0:
            print(f"[INFO] Epoch {epoch+1:03d}/{n_epochs:03d} loss={epoch_loss/n_batches:.4f}")

    # Evaluate on val
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    y_score = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    y_pred = (y_score >= 0).astype(int)
    acc = compute_accuracy(y_true, y_pred)
    f1 = compute_f1(y_true, y_pred)
    auc = compute_auc_binary(y_true, y_score)

    return {
        "acc": acc,
        "f1": f1,
        "auc": auc,
    }


# ============================================================
# Config sets & job builder
# ============================================================

CONFIG_SETS = {
    # v2.1 gamma – low real-data fractions using v2.1 synth only
    "v21_lowfrac": {
        "real_fracs": [0.05, 0.10, 0.15, 0.20],
        "regimes": ["real_only", "real_classical", "real_plus_v2_1"],
        "n_seeds": 1,
    },

    # you can add other config sets if needed
    # "v21_only": {...}
}


def build_job_list(config_name: str):
    if config_name not in CONFIG_SETS:
        raise ValueError(f"Unknown config_set: {config_name}")

    cfg = CONFIG_SETS[config_name]
    real_fracs = cfg["real_fracs"]
    regimes = cfg["regimes"]
    n_seeds = cfg["n_seeds"]

    jobs = []
    for seed in range(n_seeds):
        for rf in real_fracs:
            for reg in regimes:
                jobs.append(
                    {
                        "real_frac": float(rf),
                        "regime": str(reg),
                        "seed": int(seed),
                    }
                )
    return jobs


# ============================================================
# Main experiment logic
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)

    parser.add_argument(
        "--synth_v2_1",
        type=str,
        required=True,
        help="Path to v2.1 synthetic NPZ (e.g., label_control_pain_sweep_gamma_notch60.npz)",
    )

    parser.add_argument(
        "--config_set",
        type=str,
        default="v21_lowfrac",
        help="Which config set to use (e.g., v21_lowfrac)",
    )

    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Where to append JSON results.",
    )

    parser.add_argument(
        "--seed_array_index",
        type=int,
        default=0,
        help="Index into the job list (used with SLURM_ARRAY_TASK_ID).",
    )

    args = parser.parse_args()

    print("[DEBUG] classifier_grid_pain_binary_autokeys.py executed")
    print(f"[DEBUG] args = {args}")

    # Build job list
    jobs = build_job_list(args.config_set)
    print(f"[DEBUG] Built {len(jobs)} jobs for config_set={args.config_set}")

    if args.seed_array_index < 0 or args.seed_array_index >= len(jobs):
        raise ValueError(
            f"seed_array_index {args.seed_array_index} is out of range 0..{len(jobs)-1}"
        )

    job = jobs[args.seed_array_index]
    real_frac = job["real_frac"]
    regime = job["regime"]
    seed = job["seed"]

    print("============================================================")
    print("[INFO] Classifier grid job config")
    print("============================================================")
    print(f"[INFO] config_set   = {args.config_set}")
    print(f"[INFO] job_index    = {args.seed_array_index} / {len(jobs)-1}")
    print(f"[INFO] real_frac    = {real_frac}")
    print(f"[INFO] regime       = {regime}")
    print(f"[INFO] seed         = {seed}")
    print("============================================================")

    rng = np.random.RandomState(seed)

    # --------------------------------------------------------
    # Load real data + labels
    # --------------------------------------------------------
    X = np.load(args.data_path)  # (N, C, T)
    N, C, T = X.shape
    meta = np.load(args.meta_path, allow_pickle=True)

    label_key = infer_label_key(meta, N)
    y = meta[label_key].astype(int)

    print(f"[INFO] Real data shape: {X.shape}, labels from '{label_key}', uniques={np.unique(y)}")

    # Flatten features
    X_flat = X.reshape(N, C * T)

    # Train/val split at epoch-level (simple, not subject-wise)
    idx_all = np.arange(N)
    rng.shuffle(idx_all)
    n_train = int(0.8 * N)
    train_idx = idx_all[:n_train]
    val_idx = idx_all[n_train:]

    # Subsample train for real_frac
    n_keep = max(1, int(real_frac * len(train_idx)))
    keep_idx = rng.choice(train_idx, size=n_keep, replace=False)

    X_train_real = X_flat[keep_idx]
    y_train_real = y[keep_idx]

    X_val = X_flat[val_idx]
    y_val = y[val_idx]

    print(f"[INFO] Using {len(X_train_real)} real train samples (real_frac={real_frac})")
    print(f"[INFO] Using {len(X_val)} real val samples")

    # --------------------------------------------------------
    # Load synthetic v2.1 data
    # --------------------------------------------------------
    X_synth, y_synth = load_synth_v21(args.synth_v2_1)
    Ns, Cs, Ts = X_synth.shape
    X_synth_flat = X_synth.reshape(Ns, Cs * Ts)

    # Optionally balance synth set with real size
    # We'll just use all synth; you can downsample if needed

    # --------------------------------------------------------
    # Build training set depending on regime
    # --------------------------------------------------------
    if regime == "real_only":
        X_train = X_train_real
        y_train = y_train_real

    elif regime == "real_classical":
        # Simple "classical" augmentation: add noisy copies
        noise_scale = 0.05
        X_aug = X_train_real + noise_scale * rng.randn(*X_train_real.shape)
        y_aug = y_train_real.copy()

        X_train = np.concatenate([X_train_real, X_aug], axis=0)
        y_train = np.concatenate([y_train_real, y_aug], axis=0)
        print(f"[INFO] Classical aug: {len(X_train_real)} -> {len(X_train)} samples")

    elif regime == "real_plus_v2_1":
        # Concatenate all synth to real
        X_train = np.concatenate([X_train_real, X_synth_flat], axis=0)
        y_train = np.concatenate([y_train_real, y_synth], axis=0)
        print(f"[INFO] real_plus_v2_1: real={len(X_train_real)}, synth={len(X_synth_flat)}, total={len(X_train)}")

    else:
        raise ValueError(f"Unknown regime: {regime}")

    # --------------------------------------------------------
    # Train classifier and compute metrics
    # --------------------------------------------------------
    metrics = train_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_epochs=20,
        batch_size=128,
        lr=1e-3,
    )

    print("============================================================")
    print("[INFO] Finished training")
    print("============================================================")
    print(f"[RESULT] ACC = {metrics['acc']:.4f}")
    print(f"[RESULT] AUC = {metrics['auc']:.4f}")
    print(f"[RESULT] F1  = {metrics['f1']:.4f}")

    # --------------------------------------------------------
    # Append to JSON
    # --------------------------------------------------------
    result = {
        "config_set": args.config_set,
        "job_index": int(args.seed_array_index),
        "real_frac": float(real_frac),
        "regime": regime,
        "seed": int(seed),
        "acc": float(metrics["acc"]),
        "auc": float(metrics["auc"]),
        "f1": float(metrics["f1"]),
    }

    out_path = args.output_json
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(result)

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[INFO] Appended result to {out_path}")


if __name__ == "__main__":
    main()

