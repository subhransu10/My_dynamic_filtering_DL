# mos/eval_only.py
from __future__ import annotations
import os, torch, argparse
from torch.utils.data import DataLoader

# Reuse helpers from your package
from .train import load_cfg, make_datasets, evaluate
from .model import build_model
from .dataset import mos_collate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["val","train"])
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")

    # Build model + load weights
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={len(missing)} unexpected={len(unexpected)}")

    # Datasets / loader
    ds_train, ds_val = make_datasets(cfg)
    ds = ds_val if args.split == "val" else ds_train
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                    num_workers=cfg["num_workers"], collate_fn=mos_collate, drop_last=False)

    # Eval
    use_sweep = bool(cfg.get("eval_sweep", True))
    fixed_t = float(cfg.get("eval_threshold", 0.5))
    with torch.no_grad():
        m = evaluate(model, dl, device, fixed_thresh=fixed_t, do_sweep=use_sweep)

    fx, bs, ps = m["fixed"], m["best"], m["pred_summary"]
    print(f"\nFixed@t={fx['thresh']:.2f}  P={fx['precision']:.3f}  R={fx['recall']:.3f}  F1={fx['f1']:.3f}  IoU={fx['iou_moving']:.3f}")
    if use_sweep:
        print(f"Best @t={bs['thresh']:.2f}  P={bs['precision']:.3f}  R={bs['recall']:.3f}  F1={bs['f1']:.3f}  IoU={bs['iou_moving']:.3f}")
    print(f"[pred stats] mean={ps['mean']:.4f} p10={ps['p010']:.4f} p90={ps['p900']:.4f} max={ps['max']:.4f}  (pos_rate≈{m['pos_rate']:.4f})")

if __name__ == "__main__":
    main()
