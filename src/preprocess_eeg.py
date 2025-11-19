#!/usr/bin/env python

import argparse
import os

import numpy as np
import pandas as pd
import mne


def load_participant_meta(participants_tsv: str) -> pd.DataFrame:
    """
    Load participants.tsv and return a DataFrame indexed by participant_id.

    REQUIRED columns:
      - participant_id  (e.g., 'sub-FMpa01')
      - group_num       (0 = control, 1 = patient)
      - pain_intensity  (float; 0 for controls or missing)
    OPTIONAL but nice to have (used if present):
      - group, study, diagnosis, sex, age
    """
    df = pd.read_csv(participants_tsv, sep="\t")

    if "participant_id" not in df.columns:
        raise ValueError("participants.tsv must have a 'participant_id' column")
    if "group_num" not in df.columns:
        raise ValueError("participants.tsv must have a 'group_num' column")
    if "pain_intensity" not in df.columns:
        raise ValueError("participants.tsv must have a 'pain_intensity' column")

    df = df.set_index("participant_id")
    return df


def preprocess_all_eeg(
    root_dir: str,
    out_data: str = "preprocessed_eeg_minimal.npy",
    out_meta: str = "preprocessed_eeg_metadata.npz",
    sfreq: float = 250.0,
    epoch_len_sec: float = 3.0,
    l_freq: float = 0.5,
    h_freq: float = 80.0,
    z_reject_thresh: float = 8.0,
):
    """
    Preprocess OSF chronic-pain EEG for diffusion modelling:

    - Walk root_dir, find all .vhdr BrainVision files
    - For each file:
        * Load with MNE
        * Keep only EEG channels
        * Resample to `sfreq` (default 250 Hz)
        * Bandpass [l_freq, h_freq] Hz (default 0.5–80 Hz)
        * Cut into non-overlapping epochs of `epoch_len_sec` (default 3 s)
        * Z-score per epoch per channel
        * Drop epochs whose max |z| > z_reject_thresh (default 8)
    - Attach per-epoch labels:
        * participant_id
        * group_num (0/1)
        * pain_intensity (float)

    Saves:
      - out_data: (N, C, T) float32 array
      - out_meta: npz with epoch-level metadata
    """

    participants_tsv = os.path.join(root_dir, "participants.tsv")
    if not os.path.exists(participants_tsv):
        raise FileNotFoundError(f"participants.tsv not found at {participants_tsv}")

    meta_df = load_participant_meta(participants_tsv)

    all_epochs = []
    all_participant_ids = []
    all_group_nums = []
    all_pain_intensities = []

    # Optional extra metadata if present
    extra_cols = []
    for col in ["group", "study", "diagnosis", "sex", "age"]:
        if col in meta_df.columns:
            extra_cols.append(col)

    extra_meta_values = {col: [] for col in extra_cols}

    n_bad_files = 0
    n_total_files = 0

    epoch_len_samples = int(epoch_len_sec * sfreq)

    print("==========================================")
    print(f"[INFO] Root dir          : {root_dir}")
    print(f"[INFO] participants.tsv : {participants_tsv}")
    print(f"[INFO] Target sfreq     : {sfreq} Hz")
    print(f"[INFO] Epoch length     : {epoch_len_sec} s ({epoch_len_samples} samples)")
    print(f"[INFO] Bandpass         : {l_freq}–{h_freq} Hz")
    print(f"[INFO] Z-reject thresh  : max |z| < {z_reject_thresh}")
    print("==========================================")

    for root, _, files in os.walk(root_dir):
        for fname in files:
            if not fname.endswith(".vhdr"):
                continue

            n_total_files += 1
            vhdr_path = os.path.join(root, fname)

            # Expect filenames like sub-FMpa01_task-rest_eeg.vhdr -> subj_id 'sub-FMpa01'
            subj_id = fname.split("_")[0]

            if subj_id not in meta_df.index:
                print(f"[WARN] {subj_id} not found in participants.tsv, skipping {fname}")
                continue

            print(f"[INFO] Processing {vhdr_path} (subject {subj_id})")

            try:
                raw = mne.io.read_raw_brainvision(
                    vhdr_path, preload=True, verbose="ERROR"
                )
            except Exception as e:
                print(f"[ERROR] Failed to read {vhdr_path}: {e}")
                n_bad_files += 1
                continue

            # Keep only EEG channels
            try:
                picks = mne.pick_types(
                    raw.info, eeg=True, eog=False, stim=False, misc=False
                )
                if len(picks) == 0:
                    print(f"[WARN] No EEG channels found in {vhdr_path}, skipping.")
                    n_bad_files += 1
                    continue
                raw.pick(picks)
            except Exception as e:
                print(f"[ERROR] Picking EEG channels failed for {vhdr_path}: {e}")
                n_bad_files += 1
                continue

            # Resample
            try:
                raw.resample(sfreq, npad="auto")
            except Exception as e:
                print(f"[ERROR] Resampling failed for {vhdr_path}: {e}")
                n_bad_files += 1
                continue

            # Bandpass filter
            try:
                raw.filter(l_freq, h_freq, fir_design="firwin", verbose="ERROR")
            except Exception as e:
                print(f"[ERROR] Filtering failed for {vhdr_path}: {e}")
                n_bad_files += 1
                continue

            data = raw.get_data()  # (C, T)
            n_channels, n_samples = data.shape

            if n_samples < epoch_len_samples:
                print(
                    f"[WARN] {vhdr_path} has only {n_samples} samples "
                    f"(< {epoch_len_samples}), skipping."
                )
                continue

            # Trim to a multiple of epoch_len_samples
            n_epochs = n_samples // epoch_len_samples
            usable_samples = n_epochs * epoch_len_samples
            data = data[:, :usable_samples]  # (C, usable_samples)

            # Reshape to (n_epochs, C, T)
            data = data.reshape(
                n_channels, n_epochs, epoch_len_samples
            )  # (C, n_epochs, T)
            data = np.transpose(data, (1, 0, 2))  # (n_epochs, C, T)

            # Z-score per epoch per channel
            mean = data.mean(axis=-1, keepdims=True)
            std = data.std(axis=-1, keepdims=True) + 1e-8
            data_z = (data - mean) / std

            # Simple artifact rejection: drop epochs with huge z-values
            max_abs = np.max(np.abs(data_z), axis=(1, 2))  # (n_epochs,)
            keep_mask = max_abs < z_reject_thresh

            n_kept = int(keep_mask.sum())
            n_dropped = int((~keep_mask).sum())
            if n_kept == 0:
                print(
                    f"[WARN] All {n_epochs} epochs rejected for {vhdr_path} "
                    f"(z_reject_thresh={z_reject_thresh}), skipping."
                )
                continue

            if n_dropped > 0:
                print(
                    f"[INFO] Rejected {n_dropped}/{n_epochs} epochs for {vhdr_path} "
                    f"(too large |z|)."
                )

            data_z = data_z[keep_mask]  # (n_kept, C, T)

            # Subject-level labels from participants.tsv
            row = meta_df.loc[subj_id]
            group_num = int(row["group_num"])
            pain_intensity = float(row["pain_intensity"])

            all_epochs.append(data_z)
            all_participant_ids.extend([subj_id] * n_kept)
            all_group_nums.extend([group_num] * n_kept)
            all_pain_intensities.extend([pain_intensity] * n_kept)

            for col in extra_cols:
                val = row[col]
                all_vals = extra_meta_values[col]
                all_vals.extend([val] * n_kept)

    if not all_epochs:
        raise RuntimeError("No valid EEG epochs were found. Check paths and file names.")

    X = np.concatenate(all_epochs, axis=0).astype(np.float32)  # (N_total_epochs, C, T)
    participant_ids = np.array(all_participant_ids)
    group_nums = np.array(all_group_nums, dtype=np.int64)
    pain_intensities = np.array(all_pain_intensities, dtype=np.float32)

    print("==========================================")
    print(f"[INFO] Total .vhdr files seen : {n_total_files}")
    print(f"[INFO] Bad/failed files       : {n_bad_files}")
    print(f"[INFO] Final epochs shape     : {X.shape}")
    print(
        f"[INFO] #patient epochs        : {(group_nums == 1).sum()} "
        f"| #control epochs: {(group_nums == 0).sum()}"
    )
    print(f"[INFO] Saving data to         : {out_data}")
    print(f"[INFO] Saving metadata to     : {out_meta}")
    print("==========================================")

    # Build metadata dict for npz
    meta_to_save = {
        "participant_id": participant_ids,
        "group_num": group_nums,
        "pain_intensity": pain_intensities,
    }

    for col in extra_cols:
        meta_to_save[col] = np.array(extra_meta_values[col], dtype=object)

    np.save(out_data, X)
    np.savez(out_meta, **meta_to_save)

    print("[DONE] Preprocessing finished successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess OSF pain EEG (BIDS-like BrainVision) for diffusion model"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="data/osf_raw/data",
        help="Path to BIDS root (contains participants.tsv and sub-*/)",
    )
    parser.add_argument(
        "--out_data",
        type=str,
        default="preprocessed_eeg_minimal.npy",
        help="Output .npy for EEG epochs (N, C, T)",
    )
    parser.add_argument(
        "--out_meta",
        type=str,
        default="preprocessed_eeg_metadata.npz",
        help="Output .npz for metadata per epoch",
    )
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--epoch_len_sec", type=float, default=3.0)
    parser.add_argument("--l_freq", type=float, default=0.5)
    parser.add_argument("--h_freq", type=float, default=80.0)
    parser.add_argument("--z_reject_thresh", type=float, default=8.0)

    args = parser.parse_args()

    preprocess_all_eeg(
        root_dir=args.root_dir,
        out_data=args.out_data,
        out_meta=args.out_meta,
        sfreq=args.sfreq,
        epoch_len_sec=args.epoch_len_sec,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        z_reject_thresh=args.z_reject_thresh,
    )


if __name__ == "__main__":
    main()

