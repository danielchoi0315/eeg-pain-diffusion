import argparse
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Utility functions
# -------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: Optional[str] = None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------
# Dataset
# -------------------------


class EEGDataset(Dataset):
    """
    Simple EEG dataset for (N, C, T) arrays.
    Labels / subjects are optional and only used for bookkeeping.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        subjects: Optional[np.ndarray] = None,
    ) -> None:
        assert data.ndim == 3, f"Expected data shape (N, C, T), got {data.shape}"
        self.data = torch.from_numpy(data.astype(np.float32))
        self.labels = labels
        self.subjects = subjects

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        x = self.data[idx]
        return x


def load_split(
    data_path: str,
    meta_path: Optional[str],
    splits_path: Optional[str],
    split_name: str,
    label_key: Optional[str],
    subject_key: Optional[str],
) -> Tuple[EEGDataset, Optional[np.ndarray], Optional[np.ndarray], Optional[dict]]:
    """
    Load one split (train/val/test) of the EEG dataset.

    Robust behaviour:
    - If splits_path is provided, tries several candidate keys:
        * split_name
        * f"{split_name}_idx"
        * f"{split_name}_indices"
    - If none found, logs a warning and uses ALL samples for that split.
    - If splits_path is None, also uses ALL samples.
    """

    # Load main data
    data = np.load(data_path)  # (N, C, T)
    meta = np.load(meta_path, allow_pickle=True) if meta_path is not None else None

    labels = None
    subjects = None
    idx = None  # indices for this split, if we find them

    if splits_path is not None and os.path.exists(splits_path):
        splits = np.load(splits_path, allow_pickle=True)
        available_keys = list(splits.keys())

        candidate_keys = [
            split_name,
            f"{split_name}_idx",
            f"{split_name}_indices",
        ]

        found_key = None
        for k in candidate_keys:
            if k in splits:
                found_key = k
                break

        if found_key is None:
            print(
                f"[WARN] Could not find any of {candidate_keys} in splits file "
                f"{splits_path}. Available keys: {available_keys}. "
                f"Falling back to using ALL samples for split='{split_name}'."
            )
        else:
            idx = splits[found_key]
            print(
                f"[INFO] Using split key '{found_key}' from {splits_path} "
                f"for split='{split_name}' (n={len(idx)})"
            )
            data = data[idx]
            if meta is not None:
                if label_key is not None and label_key in meta:
                    labels = meta[label_key][idx]
                if subject_key is not None and subject_key in meta:
                    subjects = meta[subject_key][idx]
    else:
        if splits_path is not None:
            print(
                f"[WARN] splits_path='{splits_path}' does not exist. "
                f"Using ALL samples for split='{split_name}'."
            )

    # If we didn't have a splits file or didn't find a key, just use all data
    if idx is None and meta is not None:
        if label_key is not None and label_key in meta:
            labels = meta[label_key]
        if subject_key is not None and subject_key in meta:
            subjects = meta[subject_key]

    dataset = EEGDataset(data=data, labels=labels, subjects=subjects)

    meta_dict = None
    if meta is not None:
        meta_dict = {k: meta[k] for k in meta.files}

    return dataset, labels, subjects, meta_dict


# -------------------------
# CNN-VAE model
# -------------------------


class Conv1dVAE(nn.Module):
    """
    Minimal but reasonably robust 1D CNN-VAE for EEG (N, C, T).

    - Encoder: several Conv1d blocks (stride=1, padding=1) so that T is preserved.
      Then flatten and project to latent mean/logvar.
    - Decoder: project latent back to (C_hidden, T) and use Conv1d blocks
      to reconstruct to original (C_in, T).
    """

    def __init__(
        self,
        in_channels: int,
        in_length: int,
        latent_dim: int = 64,
        hidden_channels: Tuple[int, ...] = (64, 128, 128),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_length = in_length
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels

        # Encoder
        enc_layers = []
        ch_prev = in_channels
        for ch in hidden_channels:
            enc_layers.append(
                nn.Conv1d(ch_prev, ch, kernel_size=3, stride=1, padding=1)
            )
            enc_layers.append(nn.BatchNorm1d(ch))
            enc_layers.append(nn.GELU())
            ch_prev = ch
        self.encoder_cnn = nn.Sequential(*enc_layers)

        # After encoder, shape is (B, ch_prev, in_length)
        enc_feat_dim = ch_prev * in_length
        self.fc_mu = nn.Linear(enc_feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_feat_dim, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, enc_feat_dim)

        dec_layers = []
        hidden_channels_rev = list(hidden_channels[::-1])
        ch_prev = hidden_channels_rev[0]
        for ch in hidden_channels_rev[1:]:
            dec_layers.append(
                nn.Conv1d(ch_prev, ch, kernel_size=3, stride=1, padding=1)
            )
            dec_layers.append(nn.BatchNorm1d(ch))
            dec_layers.append(nn.GELU())
            ch_prev = ch

        # Final conv back to original channels
        dec_layers.append(
            nn.Conv1d(ch_prev, in_channels, kernel_size=3, stride=1, padding=1)
        )
        self.decoder_cnn = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_cnn(x)  # (B, C_enc, T)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h_flat = self.fc_dec(z)
        # reshape back to (B, C_enc, T)
        B = z.size(0)
        C_enc = self.hidden_channels[-1]
        T = self.in_length
        h = h_flat.view(B, C_enc, T)
        x_recon = self.decoder_cnn(h)
        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# -------------------------
# Loss
# -------------------------


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standard VAE loss: MSE reconstruction + beta * KL.
    """
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta * kld
    return loss, recon_loss, kld


# -------------------------
# Training
# -------------------------


def train_vae(
    model: Conv1dVAE,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    n_epochs: int = 100,
    lr: float = 1e-3,
    beta: float = 1.0,
    grad_clip: Optional[float] = 1.0,
    save_dir: str = "eeg_vae_model",
) -> Conv1dVAE:
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_path = os.path.join(save_dir, "vae_best.pt")

    model.to(device)

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        recon_loss_sum = 0.0
        kld_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            x_recon, mu, logvar = model(x)
            loss, recon_loss, kld = vae_loss(x_recon, x, mu, logvar, beta=beta)
            loss.backward()

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            train_loss_sum += loss.item()
            recon_loss_sum += recon_loss.item()
            kld_sum += kld.item()
            n_batches += 1

        avg_train_loss = train_loss_sum / max(n_batches, 1)
        avg_recon = recon_loss_sum / max(n_batches, 1)
        avg_kld = kld_sum / max(n_batches, 1)

        log_msg = (
            f"[Epoch {epoch:03d}] "
            f"train_loss={avg_train_loss:.4f} "
            f"recon={avg_recon:.4f} "
            f"kld={avg_kld:.4f}"
        )

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch.to(device)
                    x_recon, mu, logvar = model(x)
                    loss, _, _ = vae_loss(x_recon, x, mu, logvar, beta=beta)
                    val_loss_sum += loss.item()
                    val_batches += 1
            avg_val_loss = val_loss_sum / max(val_batches, 1)
            log_msg += f"  val_loss={avg_val_loss:.4f}"

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                    },
                    best_path,
                )
        else:
            # If no val loader, keep last epoch as "best"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                },
                best_path,
            )

        print(log_msg)

    # Load best weights
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# -------------------------
# Sampling & NPZ export
# -------------------------


@torch.no_grad()
def sample_synthetic(
    model: Conv1dVAE,
    n_synth: int,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    latent_dim = model.latent_dim
    synth_list = []

    remaining = n_synth
    while remaining > 0:
        cur_bs = min(batch_size, remaining)
        z = torch.randn(cur_bs, latent_dim, device=device)
        x_recon = model.decode(z)
        synth_list.append(x_recon.cpu().numpy())
        remaining -= cur_bs

    synth = np.concatenate(synth_list, axis=0)
    return synth.astype(np.float32)


def save_synth_npz(
    synth: np.ndarray,
    out_path: str,
    meta: Optional[dict],
    label_key: Optional[str],
    subject_key: Optional[str],
) -> None:
    """
    Save synthetic data to NPZ with a schema that matches the DDPM-style
    evaluation scripts as closely as possible:

        - synth: (N, C, T) float32
        - label_key: (str) name of label field in real meta npz
        - subject_key: (str) name of subject field in real meta npz
        - sfreq, ch_names (if present in meta)
    """
    save_dict = {
        "synth": synth.astype(np.float32),
        "label_key": label_key if label_key is not None else "",
        "subject_key": subject_key if subject_key is not None else "",
    }
    if meta is not None:
        if "sfreq" in meta:
            save_dict["sfreq"] = meta["sfreq"]
        if "ch_names" in meta:
            save_dict["ch_names"] = meta["ch_names"]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **save_dict)
    print(f"[INFO] Saved synthetic NPZ to: {out_path}")
    print(f"[INFO] synth shape: {synth.shape}")


# -------------------------
# Main CLI
# -------------------------


def main():
    parser = argparse.ArgumentParser(
        description="1D CNN-VAE baseline for chronic pain EEG (unconditional)."
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to preprocessed_eeg_minimal.npy")
    parser.add_argument("--meta_path", type=str, default=None,
                        help="Path to preprocessed_eeg_metadata.npz (optional but recommended)")
    parser.add_argument("--splits_path", type=str, default=None,
                        help="Path to subject_splits_pain_auto.npz for train/val splits")
    parser.add_argument("--train_split_name", type=str, default="train",
                        help="Split name for training indices (default: train)")
    parser.add_argument("--val_split_name", type=str, default="val",
                        help="Split name for validation indices (default: val)")
    parser.add_argument("--no_val", action="store_true",
                        help="If set, do not use a validation split (all data is train).")

    parser.add_argument("--label_key", type=str, default=None,
                        help="Name of label field in meta npz (for downstream scripts).")
    parser.add_argument("--subject_key", type=str, default=None,
                        help="Name of subject field in meta npz (for downstream scripts).")

    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0,
                        help="KL weight in VAE loss.")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default=None,
                        help="Override device, e.g. 'cuda', 'cpu', 'mps'.")

    parser.add_argument("--save_dir", type=str, default="eeg_vae_model",
                        help="Directory to save VAE checkpoints.")
    parser.add_argument("--out_npz", type=str, default="results/synth_vae_baseline.npz",
                        help="Output path for synthetic NPZ.")
    parser.add_argument("--n_synth", type=int, default=4096,
                        help="Number of synthetic samples to generate.")
    parser.add_argument("--sample_batch_size", type=int, default=128,
                        help="Batch size used during sampling.")

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")

    # -------------------------
    # Load data
    # -------------------------
    train_dataset, _, _, meta_train = load_split(
        data_path=args.data_path,
        meta_path=args.meta_path,
        splits_path=args.splits_path,
        split_name=args.train_split_name,
        label_key=args.label_key,
        subject_key=args.subject_key,
    )

    if args.no_val or args.splits_path is None:
        val_loader = None
    else:
        val_dataset, _, _, _ = load_split(
            data_path=args.data_path,
            meta_path=args.meta_path,
            splits_path=args.splits_path,
            split_name=args.val_split_name,
            label_key=args.label_key,
            subject_key=args.subject_key,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # Infer shape
    example = train_dataset[0]
    assert example.ndim == 2, f"Expected (C, T) sample, got {example.shape}"
    in_channels, in_length = example.shape
    print(f"[INFO] Input shape: C={in_channels}, T={in_length}, N_train={len(train_dataset)}")

    # -------------------------
    # Init and train VAE
    # -------------------------
    model = Conv1dVAE(
        in_channels=in_channels,
        in_length=in_length,
        latent_dim=args.latent_dim,
    )

    model = train_vae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_epochs=args.n_epochs,
        lr=args.lr,
        beta=args.beta,
        grad_clip=args.grad_clip,
        save_dir=args.save_dir,
    )

    # -------------------------
    # Sampling & NPZ save
    # -------------------------
    synth = sample_synthetic(
        model=model,
        n_synth=args.n_synth,
        device=device,
        batch_size=args.sample_batch_size,
    )

    save_synth_npz(
        synth=synth,
        out_path=args.out_npz,
        meta=meta_train,
        label_key=args.label_key,
        subject_key=args.subject_key,
    )


if __name__ == "__main__":
    main()

