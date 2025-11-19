"""
datasets.py

EEGNPYDataset for (N, C, T) preprocessed EEG stored in .npy
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class EEGNPYDataset(Dataset):
    """
    Dataset wrapping a (N, C, T) EEG .npy file.
    """

    def __init__(self, npy_path: str):
        super().__init__()
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"EEG data file not found: {npy_path}")
        arr = np.load(npy_path)  # (N, C, T)
        if arr.ndim != 3:
            raise ValueError(f"Expected data of shape (N, C, T), got {arr.shape}")
        self.data = arr.astype("float32")

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.data[idx]  # (C, T)
        return torch.from_numpy(x)
