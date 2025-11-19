#!/bin/bash
#SBATCH --job-name=eeg_clf_vae_low
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/eeg_clf_vae_lowfrac_%A_%a.out
#SBATCH --error=logs/eeg_clf_vae_lowfrac_%A_%a.err

set -euo pipefail

module load python/3.10.4
source ~/eegdiff/bin/activate

cd ~/eeg_diffusion

DATA="preprocessed_eeg_minimal.npy"
META="preprocessed_eeg_metadata.npz"
SPLITS="results/subject_splits_pain_auto.npz"

# VAE synthetic file from train_vae_eeg.py
SYNTH_VAE="results/synth_vae_baseline.npz"

# New JSON just for the VAE runs
OUT_JSON="results/classifier_grid_pain_vae_lowfrac.json"

SEED=$((1234 + SLURM_ARRAY_TASK_ID))

echo "============================================================"
echo "[INFO] VAE classifier grid (low fractions)"
echo "============================================================"
echo "[INFO] DATA      = ${DATA}"
echo "[INFO] META      = ${META}"
echo "[INFO] SPLITS    = ${SPLITS}"
echo "[INFO] SYNTH_VAE = ${SYNTH_VAE}"
echo "[INFO] OUT_JSON  = ${OUT_JSON}"
echo "[INFO] SEED      = ${SEED}"
echo "============================================================"

python classifier_grid_pain_binary_autokeys.py \
  --data_path "${DATA}" \
  --meta_path "${META}" \
  --splits_path "${SPLITS}" \
  --synth_v2_1 "${SYNTH_VAE}" \
  --out_json "${OUT_JSON}" \
  --fractions 0.05 0.10 0.15 0.20 \
  --n_repeats 1 \
  --seed "${SEED}" \
  --regimes real_only real_classical real_plus_v2_1

echo "[DONE] VAE classifier grid (low fractions) finished."

