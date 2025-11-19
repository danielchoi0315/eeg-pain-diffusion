#!/usr/bin/env python

import os
import json
from collections import defaultdict

import numpy as np


# ============================================================
# Config: model + result paths
# ============================================================

MODEL_DIR = "eeg_model_cond_v2_gamma/T400_lr2e-4_linear_psd0.1_gamma0.1"

AMP_PSD_PATH = os.path.join(MODEL_DIR, "metrics_amp_psd_compare_gamma.npz")
SUPERNOVA_PATH = os.path.join(MODEL_DIR, "metrics_supernova_gamma.npz")
DOWNSTREAM_GLOBAL_PATH = os.path.join(MODEL_DIR, "metrics_downstream_pain_clf_gamma.npz")
DOWNSTREAM_SUBJECTWISE_PATH = os.path.join(MODEL_DIR, "metrics_downstream_pain_clf_subjectwise_gamma.npz")

LABEL_GROUPFLIP_PATH = os.path.join(MODEL_DIR, "label_control_group_flip_gamma.npz")
LABEL_PAINSWEEP_PATH = os.path.join(MODEL_DIR, "label_control_pain_sweep_gamma.npz")

PRIVACY_PATH = "results/privacy_artifacts_v2_1_auto.npz"

CLF_GRID_MAIN_JSON = "results/classifier_grid_pain_v21_only.json"
CLF_GRID_LOW_JSON = "results/classifier_grid_pain_v21_lowfrac.json"


# ============================================================
# Helpers
# ============================================================

def safe_load_npz(path):
    """Safely load an NPZ, returning None on error."""
    if not os.path.exists(path):
        print(f"[WARN] NPZ file not found: {path}")
        return None
    try:
        d = np.load(path, allow_pickle=True)
        print(f"[INFO] Loaded NPZ: {path}")
        return d
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return None


def safe_load_json(path):
    """Safely load a JSON file, returning None on error."""
    if not os.path.exists(path):
        print(f"[WARN] JSON file not found: {path}")
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        print(f"[INFO] Loaded JSON: {path}")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return None


# ============================================================
# 1) Amplitude & PSD metrics
# ============================================================

def summarize_amp_psd():
    print("\n============================================================")
    print("[SECTION] Amplitude & PSD metrics (gamma)")
    print("============================================================")

    d = safe_load_npz(AMP_PSD_PATH)
    if d is None:
        return

    keys = set(d.files)

    def maybe_print(key, fmt="{:.4f}"):
        if key in keys:
            val = float(d[key])
            print(f"  {key:25s}: " + fmt.format(val))
        else:
            print(f"  {key:25s}: [missing]")

    # Global amplitude / std
    maybe_print("amp_std_real_mean")
    maybe_print("amp_std_synth_mean")
    maybe_print("amp_std_ratio_mean")
    maybe_print("amp_std_ratio_min")
    maybe_print("amp_std_ratio_max")

    # Band-wise PSD metrics
    for band in ["delta", "theta", "alpha", "beta", "gamma"]:
        k_mse = f"band_{band}_mse"
        k_rel = f"band_{band}_rel_err"
        maybe_print(k_mse)
        maybe_print(k_rel)


# ============================================================
# 2) Supernova metrics
# ============================================================

def summarize_supernova():
    print("\n============================================================")
    print("[SECTION] Supernova-style fidelity metrics (gamma)")
    print("============================================================")

    d = safe_load_npz(SUPERNOVA_PATH)
    if d is None:
        return

    print("[INFO] Keys in supernova metrics:", list(d.files))

    for key in d.files:
        arr = np.array(d[key])

        if arr.size == 1 and arr.dtype.kind in "if":
            print(f"  {key:25s}: {float(arr):.4f}")
        elif arr.ndim == 1 and arr.size <= 10:
            print(f"  {key:25s}: values={arr}")
        else:
            print(
                f"  {key:25s}: "
                f"mean={arr.mean():.4f}, std={arr.std():.4f}, shape={arr.shape}"
            )


# ============================================================
# 3) Downstream classifier (global)
# ============================================================

def summarize_downstream_global():
    print("\n============================================================")
    print("[SECTION] Global downstream pain classifier (gamma)")
    print("============================================================")

    d = safe_load_npz(DOWNSTREAM_GLOBAL_PATH)
    if d is None:
        return

    print("[INFO] Keys in downstream global NPZ:", list(d.files))

    # Basic info (handle numeric vs string safely)
    for key in ["n_train_real", "n_test_real", "n_synth_total", "sfreq", "timesteps", "beta_schedule"]:
        if key in d.files:
            val = d[key]
            arr = np.array(val)
            if arr.size == 1 and arr.dtype.kind in "if":
                print(f"  {key:25s}: {float(arr):.4f}")
            else:
                # string or non-scalar; just print raw
                print(f"  {key:25s}: {arr}")

    # Summarize key regimes
    regimes = [
        ("real_only",       "real_train_test_"),
        ("synth_only",      "synth_train_test_"),
        ("real_plus_synth", "realsynth_train_test_"),
    ]

    print("\nRegime             Variant       ACC        AUC        F1")
    print("------------------------------------------------------------")

    for regime_name, prefix in regimes:
        for variant, label in [("final", "test"), ("best", "best_test")]:
            k_acc = f"{prefix}{label}_acc"
            k_auc = f"{prefix}{label}_auc"
            k_f1  = f"{prefix}{label}_f1"

            if all(k in d.files for k in [k_acc, k_auc, k_f1]):
                acc = float(np.array(d[k_acc]))
                auc = float(np.array(d[k_auc]))
                f1  = float(np.array(d[k_f1]))
                print(f"{regime_name:17s} {variant:9s} {acc:9.3f} {auc:10.3f} {f1:9.3f}")
            # If keys missing, skip silently


# ============================================================
# 4) Subjectwise classifier (mean ± SD)
# ============================================================

def summarize_downstream_subjectwise():
    print("\n============================================================")
    print("[SECTION] Subjectwise downstream pain classifier (gamma)")
    print("============================================================")

    d = safe_load_npz(DOWNSTREAM_SUBJECTWISE_PATH)
    if d is None:
        return

    print("[INFO] Keys in subjectwise NPZ:", list(d.files))

    # Basic counts and flags
    for key in ["n_train_real", "n_test_real", "n_synth_total", "match_amp"]:
        if key in d.files:
            print(f"  {key:25s}: {d[key]}")

    regimes = [
        ("real",  "acc_real",  "auc_real",  "f1_real"),
        ("synth", "acc_synth", "auc_synth", "f1_synth"),
        ("mix",   "acc_mix",   "auc_mix",   "f1_mix"),
    ]

    print("\nRegime     Metric    n_folds   mean     std      min      max")
    print("----------------------------------------------------------------")

    for regime_name, k_acc, k_auc, k_f1 in regimes:
        for metr_name, key in [("ACC", k_acc), ("AUC", k_auc), ("F1", k_f1)]:
            if key not in d.files:
                continue
            arr = np.array(d[key]).astype(float).ravel()
            if arr.size == 0:
                continue
            mean = float(arr.mean())
            std = float(arr.std())
            mn = float(arr.min())
            mx = float(arr.max())
            print(f"{regime_name:8s} {metr_name:7s} {arr.size:8d} "
                  f"{mean:7.3f} {std:7.3f} {mn:7.3f} {mx:7.3f}")


# ============================================================
# 5) Label-control: group_flip & pain_sweep
# ============================================================

def summarize_label_control():
    print("\n============================================================")
    print("[SECTION] Label-control: group_flip")
    print("============================================================")

    d = safe_load_npz(LABEL_GROUPFLIP_PATH)
    if d is not None:
        print("[INFO] Keys:", list(d.files))
        if "X_synth" in d.files:
            X = d["X_synth"]
            print(f"  X_synth shape: {X.shape} (Ns, C, T)")
        if "group_labels" in d.files:
            g = d["group_labels"]
            uniq = np.unique(g)
            try:
                counts = np.bincount(g.astype(int))
            except Exception:
                counts = "n/a"
            print(f"  group_labels: unique={uniq}, counts={counts}")
        if "real_std" in d.files and "synth_std" in d.files:
            print(f"  real_std:  {float(d['real_std']):.4f}")
            print(f"  synth_std: {float(d['synth_std']):.4f}")
        if "match_amp" in d.files:
            print(f"  match_amp: {bool(d['match_amp'])}")

    print("\n============================================================")
    print("[SECTION] Label-control: pain_sweep")
    print("============================================================")

    d2 = safe_load_npz(LABEL_PAINSWEEP_PATH)
    if d2 is not None:
        print("[INFO] Keys:", list(d2.files))
        if "X_synth" in d2.files:
            X = d2["X_synth"]
            print(f"  X_synth shape: {X.shape} (Ns, C, T)")
        if "pain_labels" in d2.files:
            p = d2["pain_labels"]
            uniq = np.unique(p)
            print(f"  pain_labels: unique={uniq}")
            for u in uniq:
                count = int(np.sum(p == u))
                print(f"    pain={u}: n={count}")
        if "real_std" in d2.files and "synth_std" in d2.files:
            print(f"  real_std:  {float(d2['real_std']):.4f}")
            print(f"  synth_std: {float(d2['synth_std']):.4f}")
        if "match_amp" in d2.files:
            print(f"  match_amp: {bool(d2['match_amp'])}")


# ============================================================
# 6) Privacy & artifact metrics
# ============================================================

def summarize_privacy_artifacts():
    print("\n============================================================")
    print("[SECTION] Privacy & artifact metrics (v2.1 auto)")
    print("============================================================")

    d = safe_load_npz(PRIVACY_PATH)
    if d is None:
        return

    keys = set(d.files)
    print("[INFO] Keys in privacy NPZ:", list(d.files))

    def summarize_vec(name):
        if name not in keys:
            print(f"{name}: [missing]")
            return
        arr = np.array(d[name]).astype(float).ravel()
        q = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"{name}:")
        print(f"  mean={arr.mean():.4f}, std={arr.std():.4f}")
        print(
            "  q05={:.4f}, q25={:.4f}, q50={:.4f}, q75={:.4f}, q95={:.4f}".format(
                q[0], q[1], q[2], q[3], q[4]
            )
        )
        print()

    summarize_vec("d_synth_to_train")
    summarize_vec("d_val_to_train")
    summarize_vec("max_amp_real")
    summarize_vec("max_amp_synth")
    summarize_vec("std_real")
    summarize_vec("std_synth")
    summarize_vec("ln_real")
    summarize_vec("ln_synth")


# ============================================================
# 7) Classifier grids (v2.1 only)
# ============================================================

def summarize_classifier_grid(path, title):
    print("\n============================================================")
    print(f"[SECTION] Classifier grid summary: {title}")
    print("============================================================")

    data = safe_load_json(path)
    if data is None:
        return

    if not isinstance(data, list):
        print("[WARN] Expected a list of runs in JSON.")
        return

    groups = defaultdict(list)
    for r in data:
        try:
            regime = r["regime"]
            frac = float(r["real_fraction"])
            mets = r["metrics"]
            groups[(regime, frac)].append(mets)
        except Exception as e:
            print(f"[WARN] Skipping one entry due to missing keys: {e}")

    print("{:<15} {:>11} {:>10} {:>10} {:>10}".format(
        "Regime", "Real_frac", "ACC", "AUC", "F1"
    ))

    for (regime, frac), mets_list in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        accs = [m.get("test_acc", np.nan) for m in mets_list]
        aucs = [m.get("test_auc", np.nan) for m in mets_list]
        f1s  = [m.get("test_f1", np.nan) for m in mets_list]

        accs = np.array(accs, dtype=float)
        aucs = np.array(aucs, dtype=float)
        f1s  = np.array(f1s, dtype=float)

        acc_mean = np.nanmean(accs)
        auc_mean = np.nanmean(aucs)
        f1_mean  = np.nanmean(f1s)

        print("{:<15} {:>11.2f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
            regime, frac, acc_mean, auc_mean, f1_mean
        ))


# ============================================================
# main
# ============================================================

def main():
    print("############################################################")
    print("# EEG diffusion v2.1 gamma – consolidated summary")
    print("############################################################")
    print(f"[INFO] MODEL_DIR = {MODEL_DIR}")

    summarize_amp_psd()
    summarize_supernova()
    summarize_downstream_global()
    summarize_downstream_subjectwise()
    summarize_label_control()
    summarize_privacy_artifacts()
    summarize_classifier_grid(CLF_GRID_MAIN_JSON, "v2.1 only (0.25 / 0.50 / 1.00)")
    summarize_classifier_grid(CLF_GRID_LOW_JSON, "v2.1 only (0.10 / 0.20)")


if __name__ == "__main__":
    main()

