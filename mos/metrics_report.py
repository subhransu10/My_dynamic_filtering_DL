from __future__ import annotations
import os, argparse, csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from mos.train import load_cfg, make_datasets
from mos.dataset import mos_collate
from mos.model import TinyPointNetSeg


def stable_sigmoid_np(x: np.ndarray) -> np.ndarray:
    # avoid overflow in exp
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def gather_logits_targets(model, ds_val, batch_size: int, num_workers: int, device: torch.device):
    dl = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=mos_collate,
        pin_memory=(device.type == "cuda"),
    )
    logits_all, targets_all = [], []
    with torch.no_grad():
        for batch in dl:
            pts  = batch["points"].to(device, non_blocking=True)
            y    = batch["label"].to(device, non_blocking=True)
            bidx = batch["batch_idx"].to(device, non_blocking=True)
            logits = model(pts, bidx)
            logits_all.append(logits.detach().cpu().numpy().reshape(-1))
            targets_all.append(y.detach().cpu().numpy().reshape(-1))
    logits = np.concatenate(logits_all, axis=0)
    targets = np.concatenate(targets_all, axis=0)
    return logits, targets


def compute_confusion(pred: np.ndarray, gt: np.ndarray):
    tp = int(np.logical_and(pred,  gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred,  gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return tp, fp, fn, tn


def print_report(tp, fp, fn, tn, thr: float, total: int):
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    iou  = tp / (tp + fp + fn + 1e-8)
    bal_acc = 0.5 * (tp / (tp + fn + 1e-8) + tn / (tn + fp + 1e-8))
    acc  = (tp + tn) / max(total, 1)
    true_frac = (tp + fn) / max(total, 1)
    pred_frac = (tp + fp) / max(total, 1)

    print("\n=== Point-wise Metrics @ threshold {:.2f} ===".format(thr))
    print(f"Counts: TP={tp}  FP={fp}  FN={fn}  TN={tn}  (total={total})")
    print(f"Frac:   true_moving={true_frac:.4f}  pred_moving={pred_frac:.4f}")
    print(f"Precision (moving): {prec:.3f}")
    print(f"Recall    (moving): {rec:.3f}")
    print(f"F1        (moving): {f1:.3f}")
    print(f"IoU       (moving): {iou:.3f}")
    print(f"Balanced Accuracy : {bal_acc:.3f}")
    print(f"Plain Accuracy    : {acc:.3f}  (imbalanced; mostly TNs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=os.path.join("mos", "config.yaml"))
    ap.add_argument("--ckpt", type=str, default=os.path.join("checkpoints", "mos_last.pt"))
    ap.add_argument("--batch_size", type=int, default=None, help="override batch size just for evaluation")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--temp_path", type=str, default=None, help="optional path to temperature scaling file (*.pt with {'T': float})")
    ap.add_argument("--prcurve", action="store_true", help="also print a small PR sweep (10 thresholds)")
    args = ap.parse_args()

    # config / device
    cfg = load_cfg(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")

    # dataset
    _, ds_val = make_datasets(cfg)

    # model
    in_ch = 5 if cfg["use_prev"] else 4
    if cfg.get("use_flow_vec", False) and cfg["use_prev"]:
        in_ch += 3
    model = TinyPointNetSeg(in_channels=in_ch).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()

    # temperature (optional)
    T = 1.0
    if args.temp_path and os.path.isfile(args.temp_path):
        try:
            T = float(torch.load(args.temp_path, map_location="cpu")["T"])
            print(f"[metrics] using temperature T={T:.4f}")
        except Exception as e:
            print(f"[metrics] warning: failed to read temp file '{args.temp_path}': {e}")

    # gather logits/targets once
    bs = int(args.batch_size) if args.batch_size is not None else int(cfg["batch_size"])
    logits, targets = gather_logits_targets(model, ds_val, bs, cfg["num_workers"], device)

    # main threshold report
    probs = stable_sigmoid_np(logits / float(T))
    pred  = probs >= float(args.threshold)
    gt    = targets >= 0.5
    tp, fp, fn, tn = compute_confusion(pred, gt)
    total = tp + fp + fn + tn
    print_report(tp, fp, fn, tn, float(args.threshold), total)

    # optional PR sweep (quick glance)
    if args.prcurve:
        print("\n--- Quick PR sweep ---")
        for thr in np.linspace(0.1, 0.9, 10):
            p = probs >= float(thr)
            tp, fp, fn, tn = compute_confusion(p, gt)
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2 * prec * rec / (prec + rec + 1e-8)
            print(f"thr={thr:0.2f}  P={prec:0.3f}  R={rec:0.3f}  F1={f1:0.3f}")

if __name__ == "__main__":
    main()
