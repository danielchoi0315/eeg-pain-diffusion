#!/usr/bin/env python

import numpy as np

PATH =  "results/privacy_artifacts_v2_1_notch60.npz"

def main():
    d = np.load(PATH, allow_pickle=True)
    print(f"[INFO] Loaded {PATH}")
    print("[INFO] Keys:", d.files)

    d_synth = d["d_synth_to_train"]
    d_val   = d["d_val_to_train"]

    max_amp_real  = d["max_amp_real"]
    max_amp_synth = d["max_amp_synth"]
    std_real      = d["std_real"]
    std_synth     = d["std_synth"]
    ln_real       = d["ln_real"]
    ln_synth      = d["ln_synth"]

    def summarize(name, arr):
        q = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"{name}:")
        print(f"  mean={arr.mean():.4f}, std={arr.std():.4f}")
        print(f"  q05={q[0]:.4f}, q25={q[1]:.4f}, q50={q[2]:.4f}, q75={q[3]:.4f}, q95={q[4]:.4f}")
        print()

    print("=== Nearest-neighbor distances in feature space ===")
    summarize("d_synth_to_train", d_synth)
    summarize("d_val_to_train", d_val)

    print("=== Max |amplitude| per epoch ===")
    summarize("max_amp_real", max_amp_real)
    summarize("max_amp_synth", max_amp_synth)

    print("=== Std per epoch ===")
    summarize("std_real", std_real)
    summarize("std_synth", std_synth)

    print("=== Line-noise bandpower (log10) ===")
    summarize("ln_real", ln_real)
    summarize("ln_synth", ln_synth)


if __name__ == "__main__":
    main()

