#!/usr/bin/env python
"""
Gamma evaluation suite wrapper.

TASK 0: REAL vs SYNTH amp/PSD diagnostics
TASK 1: Supernova-style metrics
TASK 2: Downstream pain-vs-control classifier

This script is a thin wrapper that calls:

  - compare_real_vs_synth_stats.py
  - eval_ddpm_supernova.py
  - eval_ddpm_downstream_pain_classifier.py

NOTE:
- We only forward --match_amp to the *downstream classifier* script,
  because the other two scripts do not yet support that flag.
"""

import os
import argparse
import subprocess
import shlex


def log(msg: str) -> None:
    print(msg, flush=True)


def run_cmd(cmd: str) -> None:
    """Run a shell command with logging."""
    log(f"[CMD] {cmd}")
    result = subprocess.run(shlex.split(cmd))
    if result.returncode != 0:
        raise SystemExit(f"Command failed with return code {result.returncode}: {cmd}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="ddpm_best.pt")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sfreq", type=float, default=250.0)

    parser.add_argument(
        "--n_synth",
        type=int,
        default=512,
        help="Number of synthetic epochs for amp/PSD + Supernova.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size for sampling from DDPM.",
    )

    parser.add_argument(
        "--clf_batch_size",
        type=int,
        default=128,
        help="Batch size for downstream classifier.",
    )
    parser.add_argument(
        "--clf_epochs",
        type=int,
        default=20,
        help="Number of epochs for downstream classifier.",
    )
    parser.add_argument("--pain_scale_max", type=float, default=10.0)
    parser.add_argument(
        "--n_synth_per_class",
        type=int,
        default=2000,
        help="Synthetic epochs per class for downstream classifier.",
    )

    parser.add_argument(
        "--match_amp",
        action="store_true",
        help="Apply global amplitude matching in the downstream classifier script.",
    )

    # If not running as SLURM array, allow manual override
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="0 = amp/PSD, 1 = Supernova, 2 = downstream clf (only used if SLURM_ARRAY_TASK_ID not set).",
    )

    args = parser.parse_args()

    # Determine task from SLURM_ARRAY_TASK_ID or --task_id
    slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if slurm_task is not None:
        task_id = int(slurm_task)
        log("============================================================")
        log(f"[INFO] SLURM_ARRAY_TASK_ID = {task_id}")
    else:
        if args.task_id is None:
            raise SystemExit(
                "You must specify --task_id when SLURM_ARRAY_TASK_ID is not set."
            )
        task_id = int(args.task_id)
        log("============================================================")
        log(f"[INFO] Using --task_id = {task_id}")

    log(f"[INFO] Host                = {os.uname().nodename}")
    log("============================================================")
    log(f"[INFO] DATA      = {args.data_path}")
    log(f"[INFO] META      = {args.meta_path}")
    log(f"[INFO] MODEL_DIR = {args.model_dir}")
    log(f"[INFO] CKPT      = {args.ckpt_name}")

    # Common bits
    base_flags = [
        f"--data_path {args.data_path}",
        f"--meta_path {args.meta_path}",
        f"--model_dir {args.model_dir}",
        f"--ckpt_name {args.ckpt_name}",
        f"--device {args.device}",
        f"--sfreq {args.sfreq}",
    ]
    if args.use_ema:
        base_flags.append("--use_ema")

    if task_id == 0:
        # ------------------------------------------------------------
        # TASK 0: REAL vs SYNTH amp/PSD diagnostics
        # ------------------------------------------------------------
        log("[TASK 0] REAL vs SYNTH amp/PSD diagnostics")

        flags = base_flags + [
            f"--n_synth {args.n_synth}",
            f"--sample_batch_size {args.sample_batch_size}",
            "--metrics_name metrics_amp_psd_compare_gamma.npz",
        ]
        # IMPORTANT: do NOT add --match_amp here (script doesn't support it)

        cmd = "python compare_real_vs_synth_stats.py " + " ".join(flags)
        run_cmd(cmd)
        log("[DONE] Task 0 finished.")

    elif task_id == 1:
        # ------------------------------------------------------------
        # TASK 1: Supernova-style metrics
        # ------------------------------------------------------------
        log("[TASK 1] Supernova-style metrics")

        flags = base_flags + [
            f"--n_synth {args.n_synth}",
            f"--sample_batch_size {args.sample_batch_size}",
            "--metrics_name metrics_supernova_gamma.npz",
        ]
        # IMPORTANT: do NOT add --match_amp here (script doesn't support it)

        cmd = "python eval_ddpm_supernova.py " + " ".join(flags)
        run_cmd(cmd)
        log("[DONE] Task 1 finished.")

    elif task_id == 2:
        # ------------------------------------------------------------
        # TASK 2: Downstream pain-vs-control classifier
        # ------------------------------------------------------------
        log("[TASK 2] Downstream pain-vs-control evaluation")

        flags = base_flags + [
            f"--pain_scale_max {args.pain_scale_max}",
            f"--n_synth_per_class {args.n_synth_per_class}",
            f"--sample_batch_size {args.sample_batch_size}",
            f"--clf_batch_size {args.clf_batch_size}",
            f"--clf_epochs {args.clf_epochs}",
            "--metrics_name metrics_downstream_pain_clf_gamma.npz",
        ]
        # Here it *is* safe to pass --match_amp (this script supports it)
        if args.match_amp:
            flags.append("--match_amp")

        cmd = "python eval_ddpm_downstream_pain_classifier.py " + " ".join(flags)
        run_cmd(cmd)
        log("[DONE] Task 2 finished.")

    else:
        raise SystemExit("Unknown task_id={task_id}; expected 0, 1, or 2.")


if __name__ == "__main__":
    main()

