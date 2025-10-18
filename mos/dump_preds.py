# mos/dump_preds.py
import os, torch, argparse
from torch.utils.data import DataLoader
from .train import load_cfg, make_datasets
from .model import build_model
from .dataset import mos_collate
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max_batches", type=int, default=50)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd.get("model", sd), strict=False)

    _, ds_val = make_datasets(cfg)
    dl = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0, collate_fn=mos_collate)

    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= args.max_batches: break
            pts = batch["points"].to(device)      # [N, F]
            bidx = batch["batch_idx"].to(device)  # [N]
            out = model(pts, bidx)
            logits = out["logits"].reshape(-1) if isinstance(out, dict) else out.reshape(-1)
            probs = torch.sigmoid(logits).float().cpu().numpy()  # [N]
            xyz = batch["points"][:, :3].cpu().numpy()           # [N,3]
            np.savez(os.path.join(args.outdir, f"frame_{i:06d}.npz"),
                     xyz=xyz, prob=probs)
            if "label" in batch:
                y = batch["label"].cpu().numpy().astype(np.uint8)
                np.savez(os.path.join(args.outdir, f"frame_{i:06d}_gt.npz"),
                         y=y)
            print(f"wrote frame_{i:06d}.npz")

if __name__ == "__main__":
    main()
