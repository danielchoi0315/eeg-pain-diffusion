#!/bin/bash
#SBATCH --job-name=eeg_ddpm_v2_best
#SBATCH --output=eeg_ddpm_v2_best_%j.out
#SBATCH --error=eeg_ddpm_v2_best_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00

module load cuda   # or whatever your ARC uses
source ~/eegdiff/bin/activate

cd ~/eeg_diffusion

python train_ddpm_cond_v2.py \
  --data_path preprocessed_eeg_minimal.npy \
  --meta_path preprocessed_eeg_metadata.npz \
  --out_dir eeg_model_cond_v2/best_T400_lr2e-4_linear_psd0.1_cov0.1 \
  --timesteps 400 \
  --beta_schedule linear \
  --base_channels 64 \
  --time_emb_dim 256 \
  --dropout 0.1 \
  --lambda_psd 0.1 \
  --lambda_cov 0.1 \
  --epochs 200 \
  --batch_size 64 \
  --lr 2e-4 \
  --device cuda

