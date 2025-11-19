#!/bin/bash
#SBATCH --job-name=eeg_gamma_suite
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=eeg_gamma_suite_%A_%a.out
#SBATCH --error=eeg_gamma_suite_%A_%a.err

module load python/3.10

# Activate your venv (not conda)
source /home/daniel.choi2/eegdiff/bin/activate

cd /home/daniel.choi2/eeg_diffusion

python eval_ddpm_gamma_suite.py \
  --data_path preprocessed_eeg_minimal.npy \
  --meta_path preprocessed_eeg_metadata.npz \
  --model_dir eeg_model_cond_v2_gamma/T400_lr2e-4_linear_psd0.1_gamma0.1 \
  --ckpt_name ddpm_best.pt \
  --use_ema \
  --device cuda \
  --sfreq 250.0 \
  --n_synth 512 \
  --sample_batch_size 4 \
  --clf_batch_size 128 \
  --clf_epochs 20 \
  --pain_scale_max 10 \
  --n_synth_per_class 2000 \
  --match_amp

