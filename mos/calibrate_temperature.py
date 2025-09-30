from __future__ import annotations
import os, argparse
import torch
from torch.utils.data import DataLoader
from .train import load_cfg, make_datasets
from .dataset import mos_collate
from .model import TinyPointNetSeg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=os.path.join("mos","config.yaml"))
    ap.add_argument("--ckpt",   type=str, default=os.path.join("checkpoints","mos_best.pt"))
    ap.add_argument("--out",    type=str, default=os.path.join("checkpoints","temp_scale.pt"))
    ap.add_argument("--max_batches", type=int, default=400, help="speed cap (None=all)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")

    _, ds_val = make_datasets(cfg)
    dl = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False,
                    num_workers=cfg["num_workers"], collate_fn=mos_collate)

    in_ch = 5 if cfg["use_prev"] else 4
    if cfg.get("use_flow_vec", False) and cfg["use_prev"]:
        in_ch += 3
    model = TinyPointNetSeg(in_channels=in_ch).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"]); model.eval()

    logits_all, targets_all = [], []
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            pts  = batch["points"].to(device, non_blocking=True)
            y    = batch["label"].to(device, non_blocking=True).float()
            bidx = batch["batch_idx"].to(device, non_blocking=True)
            logits = model(pts, bidx)  # (N,)
            logits_all.append(logits.detach().cpu())
            targets_all.append(y.detach().cpu())
            if args.max_batches and (bi+1) >= args.max_batches:
                break

    logits = torch.cat(logits_all, 0)
    targets = torch.cat(targets_all, 0)

    # learn temperature T>0: minimize BCE with logits/T
    T = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=80, line_search_fn="strong_wolfe")
    bce = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def closure():
        opt.zero_grad()
        loss = bce(logits / T.clamp_min(1e-3), targets)
        loss.backward()
        return loss

    for _ in range(8):
        opt.step(closure)

    T_star = float(T.clamp_min(1e-3).item())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"T": T_star}, args.out)
    print(f"[calibrate] temperature T={T_star:.4f} -> saved to {args.out}")

if __name__ == "__main__":
    main()
