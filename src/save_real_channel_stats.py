#!/usr/bin/env python
"""
Compute per-channel std from real EEG and save to real_channel_stats.npz.

Usage:
  cd ~/eeg_diffusion
  python save_real_channel_stats.py \
      --data_path preprocessed_eeg_minimal.npy \
      --out_path real_channel_stats.npz
"""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="preprocessed_eeg_minimal.npy",
        help="Path to real EEG array (N, C, T).",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="real_channel_stats.npz",
        help="Where to save per-channel stds.",
    )
    args = parser.parse_args()

    print("[INFO] Loading real EEG from:", args.data_path)
    X = np.load(args.data_path)  # shape (N, C, T)
    if X.ndim != 3:
        raise RuntimeError("Expected real EEG with shape (N, C, T), got {}".format(X.shape))

    # Per-channel std across epochs and time
    channel_std = X.std(axis=(0, 2))
    print("[INFO] Real channel_std shape:", channel_std.shape)
    print("[INFO] channel_std (first 10):", channel_std[:10])

    np.savez(args.out_path, channel_std=channel_std)
    print("[INFO] Saved to", args.out_path)


if __name__ == "__main__":
    main()

