#!/bin/bash
#SBATCH --job-name=privacy_artifacts_auto
#SBATCH --partition=cpu2019
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/privacy_artifacts_auto_%A.out
#SBATCH --error=logs/privacy_artifacts_auto_%A.err

module load python/3.10
source ~/eegdiff/bin/activate

cd ~/eeg_diffusion

mkdir -p logs
mkdir -p results

DATA=preprocessed_eeg_minimal.npy
META=preprocessed_eeg_metadata.npz

# This file will be created if it doesn't exist (in the Python script)
SPLITS=results/subject_splits_pain_auto.npz

# Use v2.1 label-control pain_sweep npz as the synthetic pool
SYNTH_V2_1=eeg_model_cond_v2_gamma/T400_lr2e-4_linear_psd0.1_gamma0.1/label_control_pain_sweep_gamma.npz

OUT=results/privacy_artifacts_v2_1_auto.npz

python analyze_privacy_and_artifacts_autokeys.py \
  --data_path "${DATA}" \
  --meta_path "${META}" \
  --splits_path "${SPLITS}" \
  --synth_path "${SYNTH_V2_1}" \
  --output_path "${OUT}" \
  --sfreq 250.0 \
  --line_freq 60.0 \
  --band_width 2.0 \
  --seed 0

