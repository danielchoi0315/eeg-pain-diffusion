# Conditional EEG Diffusion for Chronic Pain (v2.1 gamma)

This repository contains the code and reproducible pipelines for a **conditional, physiology-aware diffusion model** trained on chronic pain EEG, plus baselines and evaluation scripts.

The core model is a **label-controlled 1D UNet DDPM (v2.1, gamma-weighted loss)** trained on 3-second epochs from a chronic pain dataset, with:

- **Conditioning** on pain vs control and pain intensity.
- **Physiology-aware training & evaluation**, including band-limited PSD metrics and “supernova-style” fidelity score suite.
- **Subject-wise and global downstream classifiers** (pain vs control).
- **Label controllability tests** (`group_flip`, `pain_sweep`).
- **Privacy & artifact analysis** (nearest neighbor distances, amplitude/stability, line noise).
- A **CNN-VAE baseline** for comparison at low data fractions.

All code has been tested on the **UCalgary ARC cluster** with Python 3.10 and CUDA-enabled PyTorch.

---

## 1. Repository structure

```text
eeg-pain-diffusion/
├─ README.md                      # This file
├─ .gitignore
├─ env/
│   └─ requirements.txt           # Environment (exported from working env)
├─ src/
│   ├─ diffusion_1d.py            # DDPM core
│   ├─ eeg_unet.py                # CondUNet1D architecture
│   ├─ datasets.py                # EEG dataset loader helpers
│   ├─ preprocess_eeg.py          # Preprocessing -> preprocessed_eeg_minimal.npy/.npz
│   ├─ save_real_channel_stats.py # Saves real_channel_stats.npz
│   ├─ train_ddpm_cond_v2_gamma.py    # v2.1 gamma DDPM training script
│   ├─ train_ddpm_cond_v2.py          # (optional) v2 with PSD+cov loss
│   ├─ train_ddpm_cond.py             # (optional) v1 baseline
│   ├─ compare_real_vs_synth_stats.py # Amp/PSD metrics
│   ├─ eval_ddpm_supernova.py         # Supernova-style fidelity metrics
│   ├─ eval_ddpm_gamma_suite.py       # Gamma suite wrapper (v2.1)
│   ├─ eval_ddpm_downstream_pain_classifier.py
│   ├─ eval_ddpm_downstream_pain_classifier_subjectwise_v2.py
│   ├─ eval_ddpm_label_control_gamma.py
│   ├─ analyze_privacy_and_artifacts_autokeys.py
│   ├─ analyze_privacy_artifacts_v21.py
│   ├─ classifier_grid_pain_binary_autokeys.py
│   ├─ analyze_classifier_grid_v21_only.py
│   ├─ analyze_classifier_grid_v21_lowfrac.py
│   ├─ summarize_v21_gamma_results.py
│   ├─ train_vae_eeg.py               # CNN-VAE baseline + synthetic generator
│   └─ summarize_all_results.py       # (optional) global summarizer
├─ slurm/
│   ├─ run_preprocess_osf.slurm
│   ├─ run_ddpm_train_v21.slurm           # Trains the v2.1 gamma DDPM
│   ├─ slurm_gamma_suite.slurm            # Runs the v2.1 gamma suite
│   ├─ ddpm_subjectwise_gamma.slurm       # Subject-wise CV downstream classifier
│   ├─ eeg_label_ctrl_gamma.slurm         # group_flip + pain_sweep
│   ├─ eeg_privacy_artifacts_autokeys.slurm
│   ├─ eeg_classifier_grid_v21_only.slurm # 0.25/0.5/1.0 real-frac grid
│   ├─ eeg_classifier_grid_v21_lowfrac.slurm # 0.05/0.10/0.15/0.20 real-frac grid
│   ├─ eeg_vae_train.slurm                # Trains VAE baseline and saves synth NPZ
│   └─ eeg_classifier_grid_vae_lowfrac.slurm # VAE augmentation grid
├─ configs/
│   ├─ configs_eval.txt
│   ├─ config_eval.txt
│   └─ (optional) JSON config sets
├─ data/
│   └─ README_DATA.md                 # How to obtain raw EEG and preprocessed files
└─ results/
    ├─ example_metrics/
    │   ├─ metrics_amp_psd_compare_gamma.npz
    │   ├─ metrics_supernova_gamma.npz
    │   ├─ metrics_downstream_pain_clf_gamma.npz
    │   ├─ metrics_downstream_pain_clf_subjectwise_gamma.npz
    │   ├─ label_control_group_flip_gamma.npz
    │   ├─ label_control_pain_sweep_gamma.npz
    │   ├─ label_control_pain_sweep_gamma_notch60.npz
    │   ├─ privacy_artifacts_v2_1_notch60.npz
    │   ├─ classifier_grid_pain_v21_only.json
    │   ├─ classifier_grid_pain_v21_lowfrac.json
    │   └─ classifier_grid_pain_vae_lowfrac.json
    └─ plots/
        ├─ clf_grid_v21_lowfrac_auc.png
        └─ (other figures)

