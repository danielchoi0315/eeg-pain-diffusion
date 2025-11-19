#!/usr/bin/env python
import argparse
import os
import numpy as np


def infer_subject_key(meta, n_epochs):
    """
    Try to guess the subject key from meta.
    We only need subjects to avoid leakage in the train/val split.
    """
    candidates = []
    for k in meta.files:
        arr = meta[k]
        if arr.shape == (n_epochs,) and arr.dtype.kind in ("i", "u", "S", "U"):
            uniq = len(np.unique(arr))
            if uniq > 1:
                candidates.append((k, uniq))
    if not candidates:
        raise RuntimeError("Could not infer subject_key from metadata.")
    # pick the one with the most unique subjects
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def make_subject_split(subjects, train_frac=0.8, seed=0):
    rng = np.random.RandomState(seed)
    unique_subj = np.array(sorted(np.unique(subjects)))
    rng.shuffle(unique_subj)
    n_train = int(len(unique_subj) * train_frac)
    train_subj = unique_subj[:n_train]
    val_subj = unique_subj[n_train:]
    train_mask = np.isin(subjects, train_subj)
    val_mask = np.isin(subjects, val_subj)
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    return train_idx, val_idx, train_subj, val_subj


def compute_band_indices(T, sfreq, line_freq, band_width):
    freqs = np.fft.rfftfreq(T, d=1.0 / sfreq)
    bands = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }
    band_idx = {}
    for name, (fmin, fmax) in bands.items():
        band_idx[name] = (freqs >= fmin) & (freqs <= fmax)
    # line-noise band (for QC only)
    ln_min = line_freq - band_width
    ln_max = line_freq + band_width
    line_noise_idx = (freqs >= ln_min) & (freqs <= ln_max)
    return freqs, band_idx, line_noise_idx


def compute_epoch_features(epoch, sfreq, freqs, band_idx):
    """
    epoch: (C, T)
    Returns a small feature vector:
    [global_std, log10(P_delta), ..., log10(P_gamma)]
    where P_band is mean PSD over channels and freqs in that band.
    """
    C, T = epoch.shape
    # global std
    gstd = float(epoch.std())

    # FFT per epoch
    F = np.fft.rfft(epoch, axis=-1)  # (C, F)
    PSD = (np.abs(F) ** 2) / T       # simple periodogram

    feats = [gstd]
    for name in ["delta", "theta", "alpha", "beta", "gamma"]:
        idx = band_idx[name]
        if not np.any(idx):
            feats.append(0.0)
            continue
        # average over channels and band freqs
        p = PSD[:, idx].mean()
        # log-compress, avoid log(0)
        feats.append(float(np.log10(p + 1e-12)))
    return np.array(feats, dtype=np.float32)


def compute_features_all(X, sfreq, freqs, band_idx):
    """
    X: (N, C, T)
    Returns: (N, D) feature matrix.
    """
    N = X.shape[0]
    feats = np.zeros((N, 6), dtype=np.float32)  # [std + 5 bands]
    for i in range(N):
        feats[i] = compute_epoch_features(X[i], sfreq, freqs, band_idx)
    return feats


def compute_line_noise_all(X, sfreq, freqs, line_noise_idx):
    """
    X: (N, C, T)
    Returns per-epoch line-noise bandpower (float32, shape (N,)).
    """
    N, C, T = X.shape
    out = np.zeros(N, dtype=np.float32)
    for i in range(N):
        epoch = X[i]
        F = np.fft.rfft(epoch, axis=-1)    # (C, F)
        PSD = (np.abs(F) ** 2) / T
        if np.any(line_noise_idx):
            p = PSD[:, line_noise_idx].mean()
        else:
            p = 0.0
        out[i] = float(np.log10(p + 1e-12))
    return out


def nn_min_dists(query, ref, batch=64):
    """
    Compute nearest-neighbor Euclidean distance from each row in `query`
    to rows in `ref`, using chunked matmul to keep memory small.

    query: (Nq, D)
    ref:   (Nr, D)
    Returns: (Nq,) distances.
    """
    query = np.asarray(query, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    Nq, D = query.shape
    Nr, _ = ref.shape

    # precompute squared norms
    ref_norm2 = np.sum(ref ** 2, axis=1)  # (Nr,)

    dists = np.zeros(Nq, dtype=np.float32)
    for start in range(0, Nq, batch):
        end = min(start + batch, Nq)
        chunk = query[start:end]          # (B, D)
        chunk_norm2 = np.sum(chunk ** 2, axis=1)  # (B,)

        # (B, Nr) = (B, D) @ (D, Nr)
        G = chunk @ ref.T
        # dist^2 = ||x||^2 + ||y||^2 - 2 x·y
        dist2 = chunk_norm2[:, None] + ref_norm2[None, :] - 2.0 * G
        # numerical safety
        dist2 = np.maximum(dist2, 0.0)
        d_min = np.sqrt(np.min(dist2, axis=1))
        dists[start:end] = d_min.astype(np.float32)

    return dists


def load_synth_npz(path):
    d = np.load(path)
    # Prefer 'X_synth' if present
    if "X_synth" in d.files:
        X = d["X_synth"]
    else:
        # fallback: first 3D array
        X = None
        for k in d.files:
            arr = d[k]
            if arr.ndim == 3:
                X = arr
                break
        if X is None:
            raise ValueError(
                f"Could not find a 3D synthetic array in {path}. "
                f"Keys: {list(d.files)}"
            )
    return np.asarray(X, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--splits_path", type=str, required=False)
    parser.add_argument("--synth_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--line_freq", type=float, default=60.0)
    parser.add_argument("--band_width", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("============================================================")
    print("[INFO] Privacy & artifact analysis (memory-safe)")
    print("============================================================")
    print(f"[INFO] DATA        = {args.data_path}")
    print(f"[INFO] META        = {args.meta_path}")
    print(f"[INFO] SYNTH       = {args.synth_path}")
    print(f"[INFO] OUTPUT      = {args.output_path}")
    print(f"[INFO] sfreq       = {args.sfreq}")
    print(f"[INFO] line_freq   = {args.line_freq}")
    print(f"[INFO] band_width  = {args.band_width}")
    print("============================================================")

    X = np.load(args.data_path)  # (N, C, T)
    X = np.asarray(X, dtype=np.float32)
    N, C, T = X.shape
    print(f"[INFO] Real data shape: {X.shape} (N, C, T)")

    meta = np.load(args.meta_path, allow_pickle=True)
    subject_key = infer_subject_key(meta, N)
    subjects = meta[subject_key]
    print(f"[INFO] Inferred subject_key: {subject_key}")
    print(f"[INFO] # unique subjects: {len(np.unique(subjects))}")

    train_idx, val_idx, train_subj, val_subj = make_subject_split(
        subjects, train_frac=0.8, seed=args.seed
    )
    print(f"[INFO] train_idx: {len(train_idx)}, val_idx: {len(val_idx)}")

    X_train = X[train_idx]
    X_val = X[val_idx]

    # Load synthetic data
    X_synth = load_synth_npz(args.synth_path)
    print(f"[INFO] Synth data shape: {X_synth.shape} (Ns, C, T)")

    # Band indices
    freqs, band_idx, line_noise_idx = compute_band_indices(
        T, args.sfreq, args.line_freq, args.band_width
    )

    # Per-epoch amplitude + std
    max_amp_real = np.max(np.abs(X_train), axis=(1, 2)).astype(np.float32)
    max_amp_synth = np.max(np.abs(X_synth), axis=(1, 2)).astype(np.float32)
    std_real = X_train.std(axis=(1, 2)).astype(np.float32)
    std_synth = X_synth.std(axis=(1, 2)).astype(np.float32)

    print("[INFO] Computing line-noise bandpower (real)...")
    ln_real = compute_line_noise_all(X_train, args.sfreq, freqs, line_noise_idx)
    print("[INFO] Computing line-noise bandpower (synth)...")
    ln_synth = compute_line_noise_all(X_synth, args.sfreq, freqs, line_noise_idx)

    # Features for privacy distances (small 6D vectors)
    print("[INFO] Computing features for real (train)...")
    feats_train = compute_features_all(X_train, args.sfreq, freqs, band_idx)
    print("[INFO] Computing features for val...")
    feats_val = compute_features_all(X_val, args.sfreq, freqs, band_idx)
    print("[INFO] Computing features for synth...")
    feats_synth = compute_features_all(X_synth, args.sfreq, freqs, band_idx)

    print("[INFO] Computing NN distances: synth -> train...")
    d_synth_to_train = nn_min_dists(feats_synth, feats_train, batch=64)
    print("[INFO] Computing NN distances: val -> train...")
    d_val_to_train = nn_min_dists(feats_val, feats_train, batch=64)

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        args.output_path,
        d_synth_to_train=d_synth_to_train,
        d_val_to_train=d_val_to_train,
        max_amp_real=max_amp_real,
        max_amp_synth=max_amp_synth,
        std_real=std_real,
        std_synth=std_synth,
        ln_real=ln_real,
        ln_synth=ln_synth,
        subject_key=subject_key,
        train_idx=train_idx,
        val_idx=val_idx,
    )
    print(f"[INFO] Saved privacy & artifact metrics to {args.output_path}")
    print("[DONE] Privacy & artifact analysis finished.")
    

if __name__ == "__main__":
    main()

