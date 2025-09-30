from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalBCE(nn.Module):
    """
    Binary focal-BCE with:
      - pos_weight (class weight for positives)
      - gamma (focal focusing)
      - reduction: "mean" | "sum" | "none"
    You can pass per-sample weights via `sample_weight` to forward().
    """
    def __init__(self, pos_weight: float = 1.0, gamma: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.register_buffer("pos_w", torch.tensor(float(pos_weight)))
        self.gamma = float(gamma)
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor | None = None):
        # logits, target: [N]
        # BCE with logits
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_w, reduction="none")
        if self.gamma > 0:
            # p = sigmoid(logits), pt = p if y=1 else (1-p)
            p = torch.sigmoid(logits)
            pt = torch.where(target > 0.5, p, 1.0 - p)
            bce = bce * ((1.0 - pt) ** self.gamma)

        if sample_weight is not None:
            bce = bce * sample_weight

        if self.reduction == "mean":
            return bce.mean()
        elif self.reduction == "sum":
            return bce.sum()
        else:
            return bce  # [N]
