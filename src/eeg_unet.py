"""
eeg_unet.py

Deep 1D UNet for multichannel EEG with sinusoidal time embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) or (B, 1) timesteps
        returns: (B, dim)
        """
        if t.dim() == 1:
            t = t[:, None]
        device = t.device
        half_dim = self.dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / (half_dim - 1))
        )
        args = t * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class TimeMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(32, c), num_channels=c)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = _gn(out_ch)
        self.norm2 = _gn(out_ch)
        self.act = nn.SiLU()
        self.time_fc = nn.Linear(t_dim, out_ch)
        self.drop = nn.Dropout(dropout)
        self.skip = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), t_emb: (B, t_dim)
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_fc(t_emb).unsqueeze(-1)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(h + self.skip(x))


class Downsample1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.convT = nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convT(x)


class EEGUNetDeep(nn.Module):
    """
    Deep 1D UNet suitable for multichannel EEG diffusion.

    Args:
        ch: number of input/output channels (EEG channels)
        base_channels: width of the first conv
        channel_mults: multipliers for each UNet level
        num_res_blocks: residual blocks per level (plus extra after first)
        t_dim: time-embedding dimension
        dropout: dropout in ResBlocks
    """

    def __init__(
        self,
        ch: int,
        base_channels: int = 64,
        channel_mults=(1, 2, 4, 8, 8),
        num_res_blocks: int = 2,
        t_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_ch = ch
        self.in_conv = nn.Conv1d(ch, base_channels, kernel_size=3, padding=1)

        self.time_emb = SinusoidalPosEmb(t_dim)
        self.time_mlp = TimeMLP(t_dim, t_dim)

        in_ch = base_channels
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels = []

        # Down path
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            num_blocks = num_res_blocks + (0 if i == 0 else 1)
            for _ in range(num_blocks):
                blocks.append(ResBlock1D(in_ch, out_ch, t_dim, dropout))
                in_ch = out_ch
                self.skip_channels.append(out_ch)
            self.down_blocks.append(blocks)
            if i != len(channel_mults) - 1:
                self.downsamples.append(Downsample1D(in_ch))
            else:
                self.downsamples.append(nn.Identity())

        # Bottleneck
        mid_ch = in_ch
        self.mid_blocks = nn.ModuleList(
            [
                ResBlock1D(mid_ch, mid_ch, t_dim, dropout),
                ResBlock1D(mid_ch, mid_ch, t_dim, dropout),
            ]
        )

        # Up path
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_channels = list(reversed(self.skip_channels))

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            num_blocks = num_res_blocks + (0 if i == len(channel_mults) - 1 else 1)
            for _ in range(num_blocks):
                skip_ch = self.skip_channels.pop(0)
                blocks.append(ResBlock1D(in_ch + skip_ch, out_ch, t_dim, dropout))
                in_ch = out_ch
            self.up_blocks.append(blocks)
            if i != 0:
                self.upsamples.append(Upsample1D(in_ch))
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = _gn(in_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv1d(in_ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dtype != torch.float32:
            t = t.float()
        t_emb = self.time_mlp(self.time_emb(t))  # (B, t_dim)

        skips = []
        x = self.in_conv(x)

        # Down
        for blocks, down in zip(self.down_blocks, self.downsamples):
            for blk in blocks:
                x = blk(x, t_emb)
                skips.append(x)
            x = down(x)

        # Mid
        for blk in self.mid_blocks:
            x = blk(x, t_emb)

        # Up
        for blocks, up in zip(self.up_blocks, self.upsamples):
            for blk in blocks:
                skip = skips.pop()
                if x.shape[-1] != skip.shape[-1]:
                    diff = skip.shape[-1] - x.shape[-1]
                    if diff > 0:
                        x = F.pad(x, (0, diff))
                    elif diff < 0:
                        skip = F.pad(skip, (0, -diff))
                x = torch.cat([x, skip], dim=1)
                x = blk(x, t_emb)
            x = up(x)

        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.out_conv(x)
        return x
