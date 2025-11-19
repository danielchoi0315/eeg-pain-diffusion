#!/usr/bin/env python

import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def summarize_results(results):
    """
    Group by (regime, real_fraction) and print a summary table.
    If you ever rerun with multiple seeds/runs per config, this will
    compute mean ± std across them automatically.
    """
    groups = defaultdict(list)
    for r in results:
        key = (r["regime"], float(r["real_fraction"]))
        groups[key].append(r["metrics"])

    print("=== Classifier grid (v2.1 only) summary ===")
    print("{:<15} {:>11} {:>10} {:>10} {:>10}".format(
        "Regime", "Real_frac", "ACC", "AUC", "F1"
    ))

    summary = []
    for (regime, frac), mets_list in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        accs = [m["test_acc"] for m in mets_list]
        aucs = [m["test_auc"] for m in mets_list]
        f1s  = [m["test_f1"] for m in mets_list]

        acc_mean, auc_mean, f1_mean = np.mean(accs), np.mean(aucs), np.mean(f1s)
        acc_std, auc_std, f1_std = np.std(accs), np.std(aucs), np.std(f1s)

        print("{:<15} {:>11.2f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
            regime, frac, acc_mean, auc_mean, f1_mean
        ))

        summary.append({
            "regime": regime,
            "real_fraction": frac,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
        })

    return summary


def plot_data_efficiency(summary, out_path="results/clf_grid_v21_only_auc.png"):
    """
    Make a simple data-efficiency plot:
    x = % real data, y = test AUC,
    curves: real_only, real_classical, real_plus_v2_1.
    """
    # Organize by regime
    regimes = ["real_only", "real_classical", "real_plus_v2_1"]
    frac_sorted = sorted({s["real_fraction"] for s in summary})

    plt.figure(figsize=(6, 4))

    for regime in regimes:
        xs, ys = [], []
        for frac in frac_sorted:
            matches = [s for s in summary if s["regime"] == regime and s["real_fraction"] == frac]
            if not matches:
                continue
            xs.append(frac * 100.0)
            ys.append(matches[0]["auc_mean"])
        if not xs:
            continue
        plt.plot(xs, ys, marker="o", label=regime)

    plt.xlabel("Percentage of real training data (%)")
    plt.ylabel("Test AUC (pain vs control)")
    plt.title("Data-efficiency curves (v2.1 only)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved data-efficiency plot to {out_path}")


def main():
    path = "results/classifier_grid_pain_v21_only.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}. Make sure the grid has finished.")

    results = load_results(path)
    print(f"[INFO] Loaded {len(results)} runs from {path}")

    summary = summarize_results(results)
    plot_data_efficiency(summary)


if __name__ == "__main__":
    main()

