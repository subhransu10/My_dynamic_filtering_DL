from __future__ import annotations
import torch
import torch.nn as nn

class TinyPointNetSeg(nn.Module):
    """
    Lightweight PointNet-style per-point segmentation network.
    Input features: [x,y,z,intensity,(dmin_prev)].
    Uses per-batch global feature via mean pooling.
    """
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU()
        )
        self.global_fc = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(256 + 256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)  # binary logit
        )

    def forward(self, points: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        # points: [M,F], batch_idx: [M]
        x = self.mlp1(points)
        x = self.mlp2(x)  # [M,256]

        # global mean per batch
        B = int(batch_idx.max().item()) + 1
        device = points.device
        glob_sum = torch.zeros(B, 256, device=device)
        glob_sum = glob_sum.index_add(0, batch_idx, x)
        counts = torch.bincount(batch_idx, minlength=B).clamp(min=1).unsqueeze(1).to(device=device, dtype=glob_sum.dtype)
        glob_mean = glob_sum / counts

        g = glob_mean[batch_idx]           # [M,256]
        z = torch.cat([x, g], dim=1)       # [M,512]
        logit = self.head(z).squeeze(1)    # [M]
        return logit
