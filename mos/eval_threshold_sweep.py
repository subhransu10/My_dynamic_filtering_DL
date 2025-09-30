# mos/eval_threshold_sweep.py
import argparse, torch, numpy as np
from torch.utils.data import DataLoader
from mos.train import make_datasets, evaluate, load_cfg
from mos.model import TinyPointNetSeg
from mos.dataset import mos_collate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/mos_last.pt",
                    help="Path to checkpoint (e.g., checkpoints/mos_last.pt or mos_best.pt)")
    ap.add_argument("--tmin", type=float, default=0.20)
    ap.add_argument("--tmax", type=float, default=0.60)
    ap.add_argument("--tsteps", type=int, default=9)
    args = ap.parse_args()

    dev = torch.device("cpu")
    ckpt = torch.load(args.ckpt, map_location=dev)
    cfg = ckpt["cfg"]

    # dataloader (validation)
    _, ds_val = make_datasets(cfg)
    dl = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False, num_workers=0, collate_fn=mos_collate)

    # model
    in_ch = 5 if cfg["use_prev"] else 4
    model = TinyPointNetSeg(in_channels=in_ch)
    model.load_state_dict(ckpt["model"])

    best = None
    for th in np.linspace(args.tmin, args.tmax, args.tsteps):
        met = evaluate(model, dl, dev, thresh=float(th))
        print(f"th={th:.2f} -> F1={met['f1']:.3f}  P={met['precision']:.3f}  R={met['recall']:.3f}  IoU={met['iou_moving']:.3f}")
        if best is None or met["f1"] > best[0]:
            best = (met["f1"], th, met)

    print("\nBEST THRESHOLD")
    print(f"th={best[1]:.2f} -> F1={best[0]:.3f}  P={best[2]['precision']:.3f}  R={best[2]['recall']:.3f}  IoU={best[2]['iou_moving']:.3f}")

if __name__ == "__main__":
    main()
