from __future__ import annotations
import os, math, yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SemanticKITTIMOS, mos_collate
from .model import TinyPointNetSeg
from .losses import WeightedFocalBCE
from .metrics import binary_stats
from .utils import set_seed

torch.backends.cudnn.benchmark = True

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_datasets(cfg):
    root = cfg["DATASET_ROOT"]
    train_seqs = cfg["SEQUENCES_TRAIN"]
    val_seq = [cfg["SEQUENCE_VAL"]]
    common = dict(
        root=root,
        semantic_kitti_yaml=cfg.get("semantic_kitti_yaml", None),
        max_points=cfg["max_points"],
        use_prev=cfg["use_prev"],
        n_prev=int(cfg.get("n_prev", 1)),
        max_prev_points=cfg["max_prev_points"],
        frame_stride=int(cfg.get("frame_stride", 1)),         # NEW
        prev_voxel_size=float(cfg.get("prev_voxel_size", 0.5)),# NEW
        dmin_chunk=int(cfg.get("dmin_chunk", 4096)),           # NEW
    )
    ds_train = SemanticKITTIMOS(sequences=train_seqs, aug=cfg["aug"], split="train", **common)
    ds_val   = SemanticKITTIMOS(sequences=val_seq,   aug=None,       split="val",   **common)
    return ds_train, ds_val

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n,p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for n,p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(d).add_(p.data, alpha=1.0-d)
    def apply_to(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])
    def restore(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])
        self.backup = {}

def build_scheduler(optimizer, cfg):
    if cfg.get("scheduler", "none") != "cosine":
        return None
    total_epochs = int(cfg["epochs"]); warmup = int(cfg.get("warmup_epochs", 0))
    def lr_lambda(epoch_idx):
        e = max(0, epoch_idx - 1)
        if e < warmup and warmup > 0:
            return float(e + 1) / float(warmup)
        T = max(1, total_epochs - warmup)
        t = min(T, e - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t / T))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def build_range_weights(points: torch.Tensor, cfg) -> torch.Tensor:
    rw_cfg = cfg.get("range_weight", {})
    if not rw_cfg or not rw_cfg.get("enable", False):
        return torch.ones(points.shape[0], device=points.device)
    alpha = float(rw_cfg.get("alpha", 0.5))
    r0 = float(rw_cfg.get("r0", 20.0)); r1 = float(rw_cfg.get("r1", 50.0))
    r = points[:, -1]
    t = torch.clamp((r - r0) / max(1e-6, (r1 - r0)), 0.0, 1.0)
    return 1.0 + alpha * t

def train_one_epoch(model, loader, optim, loss_fn, device, cfg, ema: EMA | None, scaler, log_interval=50):
    model.train()
    running = 0.0; n = 0
    for it, batch in enumerate(tqdm(loader, desc="train")):
        pts = batch["points"].to(device, non_blocking=True)
        y   = batch["label"].to(device, non_blocking=True)
        bidx= batch["batch_idx"].to(device, non_blocking=True)
        sample_w = build_range_weights(pts, cfg)

        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type=="cuda" else torch.bfloat16):
            logits = model(pts, bidx).reshape(-1)
            loss = loss_fn(logits, y, sample_weight=sample_w)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        if ema is not None:
            ema.update(model)

        running += float(loss.detach().cpu())
        n += 1
        if (it + 1) % log_interval == 0:
            with torch.no_grad():
                pmov = (torch.sigmoid(logits) > 0.5).float().mean().item()
                ppos = (y > 0.5).float().mean().item()
            print(f"  it {it+1}: loss={running/n:.4f} | pos_rate={ppos:.3f} pred_rate={pmov:.3f}")
            running, n = 0.0, 0

@torch.no_grad()
def evaluate(model, loader, device, thresh: float = 0.5):
    model.eval()
    agg = dict(tp=0, fp=0, fn=0, tn=0)
    for batch in tqdm(loader, desc="val"):
        pts = batch["points"].to(device, non_blocking=True)
        y   = batch["label"].to(device, non_blocking=True)
        bidx= batch["batch_idx"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=False):
            logits = model(pts, bidx).reshape(-1)
        s = binary_stats(logits.cpu(), y.cpu(), thresh=thresh)
        for k in agg.keys():
            agg[k] += s[k]
    prec = agg["tp"] / (agg["tp"] + agg["fp"] + 1e-8)
    rec  = agg["tp"] / (agg["tp"] + agg["fn"] + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    iou  = agg["tp"] / (agg["tp"] + agg["fp"] + agg["fn"] + 1e-8)
    return dict(precision=prec, recall=rec, f1=f1, iou_moving=iou)

def save_checkpoint(out_dir: str, fname: str, payload: dict):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    torch.save(payload, path)
    print(f"Saved {path}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--resume_last", action="store_true")
    ap.add_argument("--resume_path", type=str, default=None)
    ap.add_argument("--fresh", action="store_true", help="Ignore optimizer/EMA from checkpoint (shape-safe resume).")
    ap.add_argument("--no_compile", action="store_true", help="Disable torch.compile()")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(cfg.get("random_seed", 42))

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    print(f"[device] using: {device}")
    if device.type == "cuda":
        print(f"[device] name: {torch.cuda.get_device_properties(device).name}")

    # data
    ds_train, ds_val = make_datasets(cfg)
    dl_train = DataLoader(
        ds_train, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], collate_fn=mos_collate, drop_last=False,
        pin_memory=True
    )
    dl_val = DataLoader(
        ds_val, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], collate_fn=mos_collate, drop_last=False,
        pin_memory=True
    )

    # model
    in_ch = 6 if cfg["use_prev"] else 5
    model = TinyPointNetSeg(in_channels=in_ch).to(device)
    if not args.no_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[compile] torch.compile enabled")
        except Exception as e:
            print(f"[compile] disabled: {e}")

    # loss & optim
    loss_fn = WeightedFocalBCE(
        pos_weight=cfg["class_weight_moving"],
        gamma=cfg["focal_gamma"],
        reduction="mean",
    )
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = build_scheduler(optim, cfg)
    ema = EMA(model, decay=float(cfg.get("ema_decay", 0.999))) if cfg.get("ema_enable", False) else None
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    # resume (weights only if --fresh)
    if args.resume_path or args.resume_last:
        path = args.resume_path or os.path.join("checkpoints", "mos_last.pt")
        if os.path.isfile(path):
            print(f"Resuming weights from: {path}")
            ckpt = torch.load(path, map_location=device)
            model.load_state_dict(ckpt["model"], strict=False)
            if not args.fresh and "optim" in ckpt:
                try:
                    optim.load_state_dict(ckpt["optim"])
                except Exception as e:
                    print(f"(warn) optimizer state not loaded: {e}")
            best_f1 = float(ckpt.get("best_f1", -1.0))
            start_epoch = int(ckpt.get("ep", 0)) + 1
        else:
            print("(info) resume path missing; starting fresh")
            best_f1, start_epoch = -1.0, 1
    else:
        best_f1, start_epoch = -1.0, 1

    epochs = int(cfg["epochs"])
    eval_thresh = float(cfg.get("eval_threshold", 0.5))

    for ep in range(start_epoch, epochs + 1):
        print(f"\nEpoch {ep}/{epochs}")
        train_one_epoch(model, dl_train, optim, loss_fn, device, cfg, ema, scaler, log_interval=cfg["log_interval"])
        if scheduler is not None: scheduler.step()

        if ep % cfg["val_interval_epochs"] == 0:
            if ema is not None: ema.apply_to(model)
            metrics = evaluate(model, dl_val, device, thresh=eval_thresh)
            if ema is not None: ema.restore(model)

            print(f"Val: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} IoU={metrics['iou_moving']:.3f}")

            state_to_save = model.state_dict() if ema is None else (ema.apply_to(model) or model.state_dict())
            if ema is not None: ema.restore(model)

            save_checkpoint("checkpoints", "mos_last.pt",
                            {"ep": ep, "model": state_to_save, "optim": optim.state_dict(), "cfg": cfg, "best_f1": best_f1})
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                save_checkpoint("checkpoints", "mos_best.pt",
                                {"ep": ep, "model": state_to_save, "optim": optim.state_dict(), "cfg": cfg, "best_f1": best_f1})

if __name__ == "__main__":
    main()
