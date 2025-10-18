from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalBCE(nn.Module):
    """
    Binary focal-BCE with optional pos_weight and gamma.
    Accepts optional per-sample weights via sample_weight.
    """
    def __init__(self, pos_weight: float = 1.0, gamma: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.register_buffer("pos_w", torch.tensor(float(pos_weight)))
        self.gamma = float(gamma)
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor | None = None):
        # logits, target: [N]
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_w, reduction="none")
        if self.gamma > 0:
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


# ----------------------------- New losses -----------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.smooth, self.eps = smooth, eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)
        t = targets.float()
        inter = (probs * t).sum()
        denom = probs.sum() + t.sum() + self.smooth
        dice = (2.0 * inter + self.smooth) / (denom + self.eps)
        return 1.0 - dice


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.eps = alpha, beta, gamma, eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        p = torch.sigmoid(logits)
        t = targets.float()
        tp = (p * t).sum()
        fp = (p * (1.0 - t)).sum()
        fn = ((1.0 - p) * t).sum()
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return (1.0 - tversky) ** self.gamma


class ComboLoss(nn.Module):
    """
    ComboLoss = w_ft * FocalTversky + w_dice * Dice.
    Supports deep supervision via aux logits list.
    """
    def __init__(self, w_ft: float = 0.5, w_dice: float = 0.5):
        super().__init__()
        self.ft = FocalTverskyLoss()
        self.dice = DiceLoss()
        self.w_ft, self.w_dice = float(w_ft), float(w_dice)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, aux_logits: list[torch.Tensor] | None = None):
        loss = self.w_ft * self.ft(logits, targets) + self.w_dice * self.dice(logits, targets)
        if aux_logits:
            aux = 0.0
            for a in aux_logits:
                aux += 0.5 * (self.ft(a, targets) + self.dice(a, targets))  # light weight
            loss = loss + 0.25 * aux
        return loss
