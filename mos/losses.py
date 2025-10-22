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
    ComboLoss = weighted focal+CE + Dice, with optional auxiliary logits and sample weights.
    Matches the constructor/forward used by train.py.
    """
    def __init__(
        self,
        pos_weight: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: float | None = 0.25,
        dice_eps: float = 1e-6,
        label_smooth: float = 0.0,
        ce_weight: float = 0.6,
        dice_weight: float = 0.4,
        aux_weight: float = 0.2,
        reduction: str = "mean",
    ):
        super().__init__()
        self.pos_weight_val = float(pos_weight)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = None if focal_alpha is None else float(focal_alpha)
        self.dice_eps = float(dice_eps)
        self.label_smooth = float(label_smooth)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.aux_weight = float(aux_weight)
        assert reduction in ("mean", "sum")
        self.reduction = reduction

    def _smooth(self, y):
        if self.label_smooth <= 0: return y
        e = self.label_smooth
        return y * (1 - 2*e) + e

    def _bce(self, logits, y):
        pos_w = torch.tensor(self.pos_weight_val, device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w, reduction="none")

    def _focalize(self, ce, logits, y):
        out = ce
        if self.focal_alpha is not None:
            a = self.focal_alpha
            alpha_t = torch.where(y > 0.5, torch.as_tensor(a, device=ce.device, dtype=ce.dtype),
                                         torch.as_tensor(1.0 - a, device=ce.device, dtype=ce.dtype))
            out = out * alpha_t
        if self.focal_gamma > 0:
            p = torch.sigmoid(logits)
            pt = torch.where(y > 0.5, p, 1.0 - p)
            out = (1.0 - pt).pow(self.focal_gamma) * out
        return out

    def _dice(self, logits, y_hard):
        p = torch.sigmoid(logits)
        inter = (p * y_hard).sum()
        denom = p.sum() + y_hard.sum()
        return 1.0 - (2.0 * inter + self.dice_eps) / (denom + self.dice_eps)

    def _reduce(self, x):
        return x.mean() if self.reduction == "mean" else x.sum()

    def forward(self, logits, targets, *, aux_logits=None, sample_weight=None):
        y = self._smooth(targets)
        ce = self._bce(logits, y)
        ce = self._focalize(ce, logits, y)
        if sample_weight is not None:
            ce = ce * sample_weight.to(device=ce.device, dtype=ce.dtype)
        ce_term = self._reduce(ce)

        dice_term = self._dice(logits, targets)
        loss = self.ce_weight * ce_term + self.dice_weight * dice_term

        if aux_logits:
            for a in aux_logits:
                ace = self._bce(a, y)
                ace = self._focalize(ace, a, y)
                if sample_weight is not None:
                    ace = ace * sample_weight.to(device=ace.device, dtype=ace.dtype)
                ace = self._reduce(ace)
                ad = self._dice(a, targets)
                loss = loss + self.aux_weight * (self.ce_weight * ace + self.dice_weight * ad)
        return loss
