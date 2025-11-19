#!/bin/bash
#SBATCH --job-name=ddpm_subj_gamma
#SBATCH --output=eeg_subj_gamma_%j.out
#SBATCH --error=eeg_subj_gamma_%j.err
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

cd #file

# Activate your env (adjust if needed)
source #env

python eval_ddpm_downstream_pain_classifier_subjectwise_v2.py \
  --data_path preprocessed_eeg_minimal.npy \
  --meta_path preprocessed_eeg_metadata.npz \
  --model_dir eeg_model_cond_v2_gamma/T400_lr2e-4_linear_psd0.1_gamma0.1 \
  --ckpt_name ddpm_best.pt \
  --device cuda \
  --sfreq 250.0 \
  --pain_scale_max 10.0 \
  --n_synth_per_class 2000 \
  --sample_batch_size 4 \
  --clf_batch_size 128 \
  --clf_epochs 20 \
  --match_amp \
  --metrics_name metrics_downstream_pain_clf_subjectwise_gamma.npz \
  --seed 123

