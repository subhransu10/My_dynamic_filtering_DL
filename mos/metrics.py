# mos/metrics.py
from __future__ import annotations
import torch

@torch.no_grad()
def binary_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresh: float = 0.5,
) -> dict:
    """
    Compute TP/FP/FN/TN for binary segmentation given raw logits and 0/1 targets.
    Returns a dict of Python ints: {'tp', 'fp', 'fn', 'tn'}.

    Args:
        logits: shape [N] or [N,1] (raw scores, NOT sigmoids)
        targets: shape [N] or [N,1] with values in {0,1}
        thresh: probability threshold applied after sigmoid
    """
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets[:, 0]

    # ensure tensors on CPU for safe .item() aggregation later
    logits = logits.detach().cpu()
    targets = targets.detach().cpu().float()

    probs = torch.sigmoid(logits)
    preds = (probs >= float(thresh)).float()

    tp = torch.sum((preds == 1) & (targets == 1)).item()
    fp = torch.sum((preds == 1) & (targets == 0)).item()
    fn = torch.sum((preds == 0) & (targets == 1)).item()
    tn = torch.sum((preds == 0) & (targets == 0)).item()
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


@torch.no_grad()
def precision_recall_f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Convenience helper if you need PR/F1 from raw counts."""
    tp, fp, fn = float(tp), float(fp), float(fn)
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1
