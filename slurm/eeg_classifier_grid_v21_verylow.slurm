#!/bin/bash
#SBATCH --job-name=eeg_clf_grid_v21_low
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/clf_grid_v21_low_%A_%a.out
#SBATCH --error=logs/clf_grid_v21_low_%A_%a.err

module load python/3.10
source ~/eegdiff/bin/activate
cd ~/eeg_diffusion

DATA=preprocessed_eeg_minimal.npy
META=preprocessed_eeg_metadata.npz

SYNTH_V2_1=eeg_model_cond_v2_gamma/T400_lr2e-4_linear_psd0.1_gamma0.1/label_control_pain_sweep_gamma_notch60.npz

python classifier_grid_pain_binary_autokeys.py \
  --data_path "${DATA}" \
  --meta_path "${META}" \
  --synth_v2_1 "${SYNTH_V2_1}" \
  --config_set "v21_lowfrac" \
  --output_json "results/classifier_grid_pain_v21_lowfrac_notch60.json" \
  --seed_array_index "${SLURM_ARRAY_TASK_ID}"

