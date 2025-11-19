#!/bin/bash
#SBATCH --job-name=preproc_osf
#SBATCH --partition=cpu2023            # <-- change if your CPU partition is different
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=preproc_osf_%j.out
#SBATCH --error=preproc_osf_%j.err

set -euo pipefail

module purge
module load python/3.10.4
source ~/eegdiff/bin/activate   # <-- your venv with mne, numpy, pandas

cd ~/eeg_diffusion

echo "[$(date)] Starting preprocessing from data/osf_raw/data"

python preprocess_eeg.py \
  --root_dir data/osf_raw/data \
  --out_data preprocessed_eeg_minimal.npy \
  --out_meta preprocessed_eeg_metadata.npz \
  --sfreq 250 \
  --epoch_len_sec 3.0

echo "[$(date)] Done preprocessing"

