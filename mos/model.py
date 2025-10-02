# mos/model.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyPointNetSeg(nn.Module):
    """
    Minimal point MLP + simple global pooling (mean/max) broadcast back to points.
    Features per point: in_channels (e.g., x,y,z,intensity,range,[norm_dmin])
    """
    def __init__(
        self,
        in_channels: int = 6,      # 5 if no prev, 6 if range+norm_dmin
        h1: int = 64,
        h2: int = 256,
        head_hidden: int = 128,
    ):
        super().__init__()
        # per-point encoder
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, h1),
            nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
        )
        # head takes [x, g_mean, g_max] -> 3*h2 channels
        self.head = nn.Sequential(
            nn.Linear(3 * h2, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, points: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        points:    [N, in_channels]
        batch_idx: [N] int64 in [0..B-1]
        returns:   [N] logits
        """
        # per-point MLPs
        x = self.mlp1(points)          # [N, h1]
        x = self.mlp2(x)               # [N, h2]
        N, C = x.shape

        # batch size
        B = int(batch_idx.max().item()) + 1 if N > 0 else 0

        # dtype/device-safe accumulators
        dev, dt = x.device, x.dtype

        # --- SUM and MEAN pooling ---
        glob_sum = torch.zeros(B, C, device=dev, dtype=dt)
        glob_sum.index_add_(0, batch_idx, x)

        ones = torch.ones_like(x)
        glob_cnt = torch.zeros(B, C, device=dev, dtype=dt)
        glob_cnt.index_add_(0, batch_idx, ones)
        glob_mean = glob_sum / glob_cnt.clamp_min(1)

        # --- MAX pooling (index_reduce_ if present; fallback to scatter_reduce_) ---
        try:
            glob_max = torch.full((B, C), -float("inf"), device=dev, dtype=dt)
            # index_reduce_ is available in recent PyTorch
            glob_max.index_reduce_(0, batch_idx, x, "amax")
        except Exception:
            # Fallback path
            glob_max = torch.full((B, C), -float("inf"), device=dev, dtype=dt)
            # scatter_reduce_ introduced in 2.0; include_self=True preserves initial -inf
            glob_max.scatter_reduce_(0, batch_idx.view(-1, 1).expand(-1, C), x, reduce="amax", include_self=True)

        # broadcast global features to each point
        g_mean = glob_mean[batch_idx]   # [N, C]
        g_max  = glob_max[batch_idx]    # [N, C]

        h = torch.cat([x, g_mean, g_max], dim=1)  # [N, 3C]
        logits = self.head(h).squeeze(-1)         # [N]
        return logits
