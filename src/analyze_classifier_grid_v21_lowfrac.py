#!/usr/bin/env python

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_results(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # In case we stored as { "results": [...] }
        if "results" in data:
            data = data["results"]
        else:
            data = [data]
    return data


def get_real_fraction(r: dict) -> float:
    """
    Robustly extract the 'real_fraction' value from a result dict.
    Handles multiple possible key names.
    """
    if "real_fraction" in r:
        return float(r["real_fraction"])
    if "real_frac" in r:
        return float(r["real_frac"])
    if "real_train_fraction" in r:
        return float(r["real_train_fraction"])
    raise KeyError(
        f"Could not find a real_fraction key in result: "
        f"available keys = {list(r.keys())}"
    )


def get_metrics(r: dict):
    """
    Robustly extract acc / auc / f1 from a result dict.
    Tries multiple possible field names.
    """
    # accuracy
    acc = r.get("acc", None)
    if acc is None:
        acc = r.get("test_acc", None)
    if acc is None:
        acc = r.get("clf_acc", None)

    # auc
    auc = r.get("auc", None)
    if auc is None:
        auc = r.get("test_auc", None)
    if auc is None:
        auc = r.get("clf_auc", None)

    # f1
    f1 = r.get("f1", None)
    if f1 is None:
        f1 = r.get("test_f1", None)
    if f1 is None:
        f1 = r.get("clf_f1", None)

    if acc is None or auc is None or f1 is None:
        return None, None, None

    return float(acc), float(auc), float(f1)


def summarize_results(results):
    """
    Aggregate over runs with the same (regime, real_fraction).
    Returns a list of dicts with mean acc/auc/f1 and n_runs.
    """
    agg = defaultdict(lambda: {"acc": [], "auc": [], "f1": []})
    skipped = 0

    for r in results:
        regime = r.get("regime", r.get("regime_name", "UNKNOWN"))
        try:
            real_frac = get_real_fraction(r)
        except KeyError:
            skipped += 1
            continue

        acc, auc, f1 = get_metrics(r)
        if acc is None:
            # Missing metrics; skip
            skipped += 1
            continue

        key = (regime, real_frac)
        agg[key]["acc"].append(acc)
        agg[key]["auc"].append(auc)
        agg[key]["f1"].append(f1)

    summary = []
    for (regime, real_frac), vals in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
        acc_mean = float(np.mean(vals["acc"])) if len(vals["acc"]) > 0 else float("nan")
        auc_mean = float(np.mean(vals["auc"])) if len(vals["auc"]) > 0 else float("nan")
        f1_mean = float(np.mean(vals["f1"])) if len(vals["f1"]) > 0 else float("nan")
        summary.append(
            {
                "regime": regime,
                "real_fraction": real_frac,
                "acc": acc_mean,
                "auc": auc_mean,
                "f1": f1_mean,
                "n_runs": len(vals["acc"]),
            }
        )

    return summary, skipped


def print_summary(summary):
    print("=== Classifier grid (v2.1 only, low fractions) summary ===")
    print(f"{'Regime':16s} {'Real_frac':>10s} {'ACC':>10s} {'AUC':>10s} {'F1':>10s}")

    # Sort by regime then real_fraction
    summary_sorted = sorted(summary, key=lambda r: (r["regime"], r["real_fraction"]))

    for row in summary_sorted:
        print(
            f"{row['regime']:16s} "
            f"{row['real_fraction']:10.2f} "
            f"{row['acc']:10.3f} "
            f"{row['auc']:10.3f} "
            f"{row['f1']:10.3f}"
        )


def plot_auc(summary, out_path: str):
    """
    Plot AUC vs real_fraction for each regime.
    """
    if not summary:
        print("[WARN] No summary rows to plot.")
        return

    regimes = sorted(list({row["regime"] for row in summary}))

    plt.figure()
    for regime in regimes:
        rows = [r for r in summary if r["regime"] == regime]
        rows = sorted(rows, key=lambda r: r["real_fraction"])
        fracs = [r["real_fraction"] for r in rows]
        aucs = [r["auc"] for r in rows]
        plt.plot(fracs, aucs, marker="o", label=regime)

    plt.xlabel("Real fraction in training set")
    plt.ylabel("AUC (pain classifier)")
    plt.title("v2.1 gamma – data-efficiency (low real fractions)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved low-fraction data-efficiency plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize low-fraction classifier grid results for v2.1 gamma."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="results/classifier_grid_pain_v21_lowfrac.json",
        help="Path to classifier grid JSON file.",
    )
    parser.add_argument(
        "--out_fig",
        type=str,
        default="results/clf_grid_v21_lowfrac_auc.png",
        help="Output path for AUC figure.",
    )
    args = parser.parse_args()

    results = load_results(args.json_path)
    print(f"[INFO] Loaded {len(results)} runs from {args.json_path}")

    summary, skipped = summarize_results(results)
    if skipped > 0:
        print(f"[INFO] Skipped {skipped} runs with missing keys/metrics.")

    print_summary(summary)
    plot_auc(summary, args.out_fig)


if __name__ == "__main__":
    main()

