#!/bin/bash
#SBATCH --job-name=eeg_label_ctrl
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=eeg_label_ctrl_%A_%a.out
#SBATCH --error=eeg_label_ctrl_%A_%a.err

module load python/3.10
# add your CUDA / cuDNN modules here if needed

source ~/eegdiff/bin/activate

cd ~/eeg_diffusion

DATA=preprocessed_eeg_minimal.npy
META=preprocessed_eeg_metadata.npz
MODEL_DIR=eeg_model_cond_v2_gamma/T400_lr2e-4_linear_psd0.1_gamma0.1
CKPT=ddpm_best.pt

# Just in case (should already exist, but harmless)
mkdir -p "${MODEL_DIR}"

if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
  # 0 = group_flip
  MODE="group_flip"
  OUT_NAME="label_control_group_flip_gamma.npz"
  EXTRA_ARGS="--n_pairs 512"
elif [ "${SLURM_ARRAY_TASK_ID}" -eq 1 ]; then
  # 1 = pain_sweep
  MODE="pain_sweep"
  OUT_NAME="label_control_pain_sweep_gamma.npz"
  EXTRA_ARGS="--n_pain_steps 6 --n_per_pain 256"
else
  echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

python eval_ddpm_label_control_gamma.py \
  --data_path "${DATA}" \
  --meta_path "${META}" \
  --model_dir "${MODEL_DIR}" \
  --ckpt_name "${CKPT}" \
  --device cuda \
  --sfreq 250.0 \
  --pain_scale_max 10.0 \
  --mode "${MODE}" \
  ${EXTRA_ARGS} \
  --sample_batch_size 4 \
  --match_amp \
  --out_name "${OUT_NAME}" \
  --seed 0

