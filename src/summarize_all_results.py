# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Summarize all DDPM training and evaluation results.

Looks for:
  - Training logs: eeg_ddpm_cond_*.out
  - Model dirs:    eeg_model_cond/<CONFIG_NAME>/
  - Eval files inside each model dir:
        eval_*.npz
        metrics_*.npz

For each config:
  * Parse training log:
      - config name
      - number of epochs logged
      - final epoch and loss
      - best epoch and loss
      - ddpm_last.pt checkpoint existence and size
  * Parse eval npz files:
      - for each key:
          - scalar -> value
          - small 1D array (<=16) -> mean +- std
          - larger arrays -> just show shape
"""

import glob
import os
import re
from typing import Dict, List, Optional

import numpy as np

# Patterns for training logs
LOG_PATTERN = "eeg_ddpm_cond_*.out"
CONFIG_RE = re.compile(r"\[INFO\]\s+Config\s*=\s*(.+)")
EPOCH_RE = re.compile(r"\[EPOCH\s+(\d+)\]\s+loss\s*=\s*([0-9.eE+-]+)")


def parse_training_log(path: str) -> Dict:
    """
    Parse a single training log file and return a dict with summary info.
    """
    config_name: Optional[str] = None
    epochs: List[int] = []
    losses: List[float] = []

    with open(path, "r") as f:
        for line in f:
            m_cfg = CONFIG_RE.search(line)
            if m_cfg:
                config_name = m_cfg.group(1).strip()

            m_ep = EPOCH_RE.search(line)
            if m_ep:
                ep = int(m_ep.group(1))
                loss = float(m_ep.group(2))
                epochs.append(ep)
                losses.append(loss)

    summary: Dict[str, Optional[object]] = {
        "log_file": os.path.basename(path),
        "config": config_name,
        "n_epochs": 0,
        "final_epoch": None,
        "final_loss": None,
        "best_epoch": None,
        "best_loss": None,
        "ckpt_exists": False,
        "ckpt_size_mb": None,
        "model_dir": None,
    }

    if not epochs or config_name is None:
        return summary

    summary["n_epochs"] = len(epochs)
    summary["final_epoch"] = epochs[-1]
    summary["final_loss"] = losses[-1]

    best_loss = min(losses)
    best_idx = losses.index(best_loss)
    summary["best_loss"] = best_loss
    summary["best_epoch"] = epochs[best_idx]

    model_dir = os.path.join("eeg_model_cond", config_name)
    summary["model_dir"] = model_dir

    ckpt_path = os.path.join(model_dir, "ddpm_last.pt")
    if os.path.exists(ckpt_path):
        summary["ckpt_exists"] = True
        summary["ckpt_size_mb"] = os.path.getsize(ckpt_path) / (1024.0 ** 2)

    return summary


# Eval metrics parsing

def summarize_array(arr: np.ndarray) -> str:
    """
    Turn an array into a compact human-readable summary string.
    - scalar: value
    - small 1D (<=16): mean +- std, n
    - else: just shape
    """
    if arr.ndim == 0:
        return "{:.6g}".format(float(arr))
    if arr.ndim == 1 and arr.size <= 16:
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        return "mean={:.4f}, std={:.4f}, n={}".format(mean, std, arr.size)
    return "array shape={}".format(arr.shape)


def parse_eval_npz(path: str) -> Dict[str, str]:
    """
    Parse an eval .npz file and return {metric_name: summary_string}.
    """
    out: Dict[str, str] = {}
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        out["__ERROR__"] = "Failed to load {}: {}".format(os.path.basename(path), e)
        return out

    for key in data.files:
        try:
            arr = data[key]
            out[key] = summarize_array(arr)
        except Exception as e:
            out[key] = "ERROR summarizing ({})".format(e)
    return out


def collect_eval_metrics(model_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Look for eval_*.npz and metrics_*.npz in model_dir.
    Returns:
      { filename: {metric_name: summary_string} }
    """
    metrics_by_file: Dict[str, Dict[str, str]] = {}

    if not os.path.isdir(model_dir):
        return metrics_by_file

    patterns = [
        os.path.join(model_dir, "eval_*.npz"),
        os.path.join(model_dir, "metrics_*.npz"),
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    for fpath in sorted(files):
        fname = os.path.basename(fpath)
        metrics_by_file[fname] = parse_eval_npz(fpath)

    return metrics_by_file


# Pretty printing helpers

def print_summary_table(summaries: List[Dict]):
    """
    Print a table of training summaries (one row per config).
    """
    if not summaries:
        print("[WARN] No training logs found.")
        return

    def sort_key(s: Dict) -> float:
        if s["final_loss"] is None:
            return float("inf")
        return float(s["final_loss"])

    summaries_sorted = sorted(summaries, key=sort_key)

    print("Found {} training logs".format(len(summaries_sorted)))
    print("=" * 120)
    header = (
        "{:40s}  {:26s}  {:>4s}  {:>7s}  {:>10s}  {:>7s}  {:>10s}  {:>5s}  {:>7s}"
        .format("Config", "Log", "#Ep", "FinalEp", "FinalLoss",
                "BestEp", "BestLoss", "Ckpt?", "CkptMB")
    )
    print(header)
    print("-" * 120)

    for s in summaries_sorted:
        cfg = s["config"] or "UNKNOWN"
        log = s["log_file"]
        n_ep = s["n_epochs"]
        fe = s["final_epoch"] if s["final_epoch"] is not None else "-"
        fl = "{:.6f}".format(s["final_loss"]) if s["final_loss"] is not None else "-"
        be = s["best_epoch"] if s["best_epoch"] is not None else "-"
        bl = "{:.6f}".format(s["best_loss"]) if s["best_loss"] is not None else "-"
        ck = "Y" if s["ckpt_exists"] else "N"
        mb = "{:.1f}".format(s["ckpt_size_mb"]) if s["ckpt_size_mb"] is not None else "-"

        print(
            "{:40s}  {:26s}  {:4d}  {:>7s}  {:>10s}  {:>7s}  {:>10s}  {:>5s}  {:>7s}"
            .format(cfg, log, n_ep, str(fe), fl, str(be), bl, ck, mb)
        )

    print("=" * 120)
    print("Configs are sorted by final loss (lowest = best at top).")
    print("")


def print_eval_details(config_summary: Dict, eval_metrics: Dict[str, Dict[str, str]]):
    """
    For one config, print all eval metrics found in its model dir.
    """
    cfg = config_summary["config"] or "UNKNOWN"
    model_dir = config_summary.get("model_dir") or "(no model_dir)"
    print("=" * 80)
    print("CONFIG: {}".format(cfg))
    print("  Model dir: {}".format(model_dir))
    print("  Log file:  {}".format(config_summary["log_file"]))
    print("  Epochs:    {}".format(config_summary["n_epochs"]))
    if config_summary["final_loss"] is not None:
        print(
            "  Final loss (epoch {}): {:.6f}".format(
                config_summary["final_epoch"], config_summary["final_loss"]
            )
        )
    if config_summary["best_loss"] is not None:
        print(
            "  Best loss  (epoch {}): {:.6f}".format(
                config_summary["best_epoch"], config_summary["best_loss"]
            )
        )
    if config_summary["ckpt_exists"]:
        print(
            "  Checkpoint: ddpm_last.pt ({:.1f} MB)".format(
                config_summary["ckpt_size_mb"]
            )
        )
    else:
        print("  Checkpoint: NOT FOUND")

    if not eval_metrics:
        print("  Eval metrics: none found (no eval_*.npz or metrics_*.npz)")
        print("")
        return

    print("  Eval metrics:")
    for fname, metrics in eval_metrics.items():
        print("    File: {}".format(fname))
        if not metrics:
            print("      (no metrics found)")
            continue
        for key, summary in metrics.items():
            print("      {}: {}".format(key, summary))
    print("")


def main():
    # 1) Training summaries
    log_paths = sorted(glob.glob(LOG_PATTERN))
    if not log_paths:
        print("[WARN] No log files matching pattern: {}".format(LOG_PATTERN))
        return

    summaries = [parse_training_log(p) for p in log_paths]

    # Print global table first
    print_summary_table(summaries)

    # 2) Per-config eval details
    print("DETAILED PER-CONFIG EVAL METRICS")
    print("================================")
    print("")

    for s in summaries:
        cfg = s["config"]
        if cfg is None:
            continue
        model_dir = s.get("model_dir")
        if model_dir is None:
            model_dir = os.path.join("eeg_model_cond", cfg)
            s["model_dir"] = model_dir

        eval_metrics = collect_eval_metrics(model_dir)
        print_eval_details(s, eval_metrics)


if __name__ == "__main__":
    main()

