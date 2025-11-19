#!/bin/bash
#SBATCH --job-name=clf_grid_v21
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/clf_grid_v21_%A_%a.out
#SBATCH --error=logs/clf_grid_v21_%A_%a.err

module load python/3.10
source ~/eegdiff/bin/activate

cd ~/eeg_diffusion

mkdir -p logs
mkdir -p results

DATA=preprocessed_eeg_minimal.npy
META=preprocessed_eeg_metadata.npz

OUTPUT_PATH=results/classifier_grid_pain_v21_only.json
SPLITS_PATH=results/subject_splits_pain_auto.npz

# v2.1 synthetic pool: label-control group_flip (has binary group labels)
SYNTH_V2_1=eeg_model_cond_v2_gamma/T400_lr2e-4_linear_psd0.1_gamma0.1/label_control_group_flip_gamma.npz

# We won't use v1/v2 here, but the script expects args – harmless to pass placeholders
SYNTH_V1=/dev/null
SYNTH_V2=/dev/null

# 3 real fractions × 3 regimes = 9 tasks
REAL_FRACS=(1.0 1.0 1.0   0.5 0.5 0.5   0.25 0.25 0.25)
REGIMES=("real_only" "real_classical" "real_plus_v2_1"
         "real_only" "real_classical" "real_plus_v2_1"
         "real_only" "real_classical" "real_plus_v2_1")

IDX=${SLURM_ARRAY_TASK_ID}
REAL_FRAC=${REAL_FRACS[$IDX]}
REGIME=${REGIMES[$IDX]}

echo "============================================================"
echo "[INFO] v2.1 classifier grid job index ${IDX}"
echo "[INFO] REAL_FRAC=${REAL_FRAC}, REGIME=${REGIME}"
echo "============================================================"

python classifier_grid_pain_binary_autokeys.py \
  --data_path "${DATA}" \
  --meta_path "${META}" \
  --output_path "${OUTPUT_PATH}" \
  --splits_path "${SPLITS_PATH}" \
  --real_fraction "${REAL_FRAC}" \
  --regime "${REGIME}" \
  --synth_v1 "${SYNTH_V1}" \
  --synth_v2 "${SYNTH_V2}" \
  --synth_v2_1 "${SYNTH_V2_1}" \
  --device "cuda" \
  --seed 0 \
  --n_epochs 20 \
  --batch_size 64

