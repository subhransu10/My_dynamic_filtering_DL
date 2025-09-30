# mos/train.py
from __future__ import annotations
import os
import yaml
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SemanticKITTIMOS, mos_collate
from .model import TinyPointNetSeg
from .losses import WeightedFocalBCE
from .metrics import binary_stats  # used only for eval
from .utils import set_seed


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
    )
    ds_train = SemanticKITTIMOS(sequences=train_seqs, aug=cfg["aug"], split="train", **common)
    ds_val   = SemanticKITTIMOS(sequences=val_seq,   aug=None,       split="val",   **common)
    return ds_train, ds_val


# ---------------------- Optimizer shape repair ----------------------
def repair_optimizer_state_shapes(optim: torch.optim.Optimizer, model: torch.nn.Module):
    """
    If any Adam/AdamW buffer (exp_avg, exp_avg_sq, max_exp_avg_sq) shape mismatches
    its parameter (e.g., after widening first layer), reset that buffer.
    Also drop state for params no longer in any param_group.
    """
    param_to_state = optim.state
    model_params = set(p for g in optim.param_groups for p in g["params"])
    for p in list(param_to_state.keys()):
        if p not in model_params:
            del param_to_state[p]
            continue
        st = param_to_state[p]
        for k in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if k in st:
                buf = st[k]
                if tuple(buf.shape) != tuple(p.shape):
                    st[k] = torch.zeros_like(p, dtype=buf.dtype, device=p.device)


# ---------------------- Optional ckpt in-ch upgrade ----------------------
def _auto_upgrade_in_channels(ckpt_state: dict, new_in: int) -> dict:
    """
    Pads the first [out,in] weight matrix to match `new_in` if it's smaller.
    Returns possibly-modified state dict (wrapped).
    """
    has_wrapper = "model" in ckpt_state and isinstance(ckpt_state["model"], dict)
    sd = ckpt_state["model"] if has_wrapper else ckpt_state
    changed = False
    for name, W in list(sd.items()):
        if isinstance(W, torch.Tensor) and W.ndim == 2 and name.endswith(".weight"):
            out, inn = W.shape
            likely_first = ("mlp1.0.weight" in name) or (inn < new_in and out <= 128)
            if likely_first and inn != new_in:
                W_new = torch.zeros((out, new_in), dtype=W.dtype)
                W_new[:, :min(inn, new_in)] = W[:, :min(inn, new_in)]
                sd[name] = W_new
                print(f"[upgrade] adapted {name}: {tuple(W.shape)} -> {tuple(W_new.shape)}")
                changed = True
                break
    if has_wrapper:
        ckpt_state["model"] = sd
        return ckpt_state
    else:
        return {"model": sd} if changed else ckpt_state


# ---------------------- EMA ----------------------
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(d).add_(p.data, alpha=1.0 - d)

    def apply_to(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.data.clone()
            p.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self.backup[name])
        self.backup = {}


# ----------------- LR schedule (warmup+cosine) -----------------
def build_scheduler(optimizer, cfg):
    if cfg.get("scheduler", "none") != "cosine":
        return None, lambda ep: None

    total_epochs = int(cfg["epochs"])
    warmup = int(cfg.get("warmup_epochs", 0))

    def lr_lambda(epoch_idx):
        # epoch_idx starts at 1 in our loop; map to 0-based
        e = max(0, epoch_idx - 1)
        if e < warmup and warmup > 0:
            return float(e + 1) / float(warmup)
        # cosine over remaining epochs
        T = max(1, total_epochs - warmup)
        t = min(T, e - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t / T))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler, lr_lambda


# ----------------- range-based sample weights -----------------
def build_range_weights(points: torch.Tensor, cfg) -> torch.Tensor:
    """
    points: [N, D]; last column is 'range' feature in our dataset pipeline.
    returns [N] weights in [1 .. 1+alpha]
    """
    rw_cfg = cfg.get("range_weight", {})
    if not rw_cfg or not rw_cfg.get("enable", False):
        return torch.ones(points.shape[0], device=points.device)

    alpha = float(rw_cfg.get("alpha", 0.5))
    r0 = float(rw_cfg.get("r0", 20.0))
    r1 = float(rw_cfg.get("r1", 50.0))
    r = points[:, -1]  # last feature is range
    t = torch.clamp((r - r0) / max(1e-6, (r1 - r0)), min=0.0, max=1.0)
    return 1.0 + alpha * t


def train_one_epoch(model, loader, optim, loss_fn, device, cfg, ema: EMA | None, log_interval=50):
    model.train()
    running = 0.0
    n = 0
    for it, batch in enumerate(tqdm(loader, desc="train")):
        pts = batch["points"].to(device)
        y   = batch["label"].to(device)
        bidx = batch["batch_idx"].to(device)

        sample_w = build_range_weights(pts, cfg)  # [N]

        optim.zero_grad(set_to_none=True)
        logits = model(pts, bidx).reshape(-1)
        loss = loss_fn(logits, y, sample_weight=sample_w)
        loss.backward()
        optim.step()
        if ema is not None:
            ema.update(model)

        running += loss.item()
        n += 1
        if (it + 1) % log_interval == 0:
            with torch.no_grad():
                ppos = (y > 0.5).float().mean().item()
                pmov = (torch.sigmoid(logits) > 0.5).float().mean().item()
            print(f"  it {it+1}: loss={running/n:.4f}")
            print(f"  it {it+1}: pos_rate={ppos:.3f}  pred_rate={pmov:.3f}")
            running, n = 0.0, 0


@torch.no_grad()
def evaluate(model, loader, device, thresh: float = 0.5):
    model.eval()
    agg = dict(tp=0, fp=0, fn=0, tn=0)
    for batch in tqdm(loader, desc="val"):
        pts = batch["points"].to(device)
        y   = batch["label"].to(device)
        bidx = batch["batch_idx"].to(device)
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


def try_resume(
    model,
    optim,
    cfg,
    device,
    *,
    resume_path: str | None,
    resume_last: bool,
    reset_optim: bool
):
    start_epoch = 1
    best_f1 = -1.0

    path = None
    if resume_path:
        path = resume_path
    elif resume_last:
        candidate = os.path.join("checkpoints", "mos_last.pt")
        if os.path.isfile(candidate):
            path = candidate

    if path and os.path.isfile(path):
        print(f"Resuming from: {path}")
        ckpt = torch.load(path, map_location=device)

        # ensure first layer matches new in_channels count if older ckpt
        need_in = 6 if cfg.get("use_prev", True) else 5
        ckpt = _auto_upgrade_in_channels(ckpt, new_in=need_in)

        model.load_state_dict(ckpt["model"], strict=False)

        if "optim" in ckpt and not reset_optim:
            try:
                optim.load_state_dict(ckpt["optim"])
                repair_optimizer_state_shapes(optim, model)
            except Exception as e:
                print(f"  (warn) couldn't load optimizer state: {e} -> resetting optimizer.")
        else:
            print("  (info) skipping optimizer state (reset_optim on or no state in ckpt).")

        best_f1 = float(ckpt.get("best_f1", best_f1))
        start_epoch = int(ckpt.get("ep", 0)) + 1
        print(f"  start_epoch={start_epoch}, best_f1={best_f1:.4f}")
    else:
        if resume_path or resume_last:
            print("  (info) no valid checkpoint found to resume; starting fresh.")
    return start_epoch, best_f1


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--resume_last", action="store_true")
    ap.add_argument("--resume_path", type=str, default=None)
    ap.add_argument(
        "--reset_optim",
        action="store_true",
        help="Ignore optimizer state in the checkpoint (use after changing input features / shapes).",
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(cfg.get("random_seed", 42))

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    print(f"[device] using: {device}")
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        print(f"[device] name: {prop.name}")

    # data
    ds_train, ds_val = make_datasets(cfg)
    dl_train = DataLoader(
        ds_train, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], collate_fn=mos_collate, drop_last=False,
    )
    dl_val = DataLoader(
        ds_val, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], collate_fn=mos_collate, drop_last=False,
    )

    # model
    # features: 6 if use_prev else 5 (xyz, intensity, range, dmin_norm)
    in_ch = 6 if cfg["use_prev"] else 5
    model = TinyPointNetSeg(in_channels=in_ch).to(device)
    print(f"[sanity] model device: {next(model.parameters()).device}")

    # loss & optim
    loss_fn = WeightedFocalBCE(
        pos_weight=cfg["class_weight_moving"],
        gamma=cfg["focal_gamma"],
        reduction="mean",
    )
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # resume BEFORE building scheduler (scheduler depends on optimizer state sometimes)
    start_epoch, best_f1 = try_resume(
        model, optim, cfg, device,
        resume_path=args.resume_path, resume_last=args.resume_last, reset_optim=args.reset_optim
    )

    # scheduler + warmup
    scheduler, _ = build_scheduler(optim, cfg)

    # EMA
    ema = EMA(model, decay=float(cfg.get("ema_decay", 0.999))) if cfg.get("ema_enable", False) else None

    epochs = int(cfg["epochs"])
    eval_thresh = float(cfg.get("eval_threshold", 0.5))
    for ep in range(start_epoch, epochs + 1):
        print(f"\nEpoch {ep}/{epochs}")
        train_one_epoch(model, dl_train, optim, loss_fn, device, cfg, ema, log_interval=cfg["log_interval"])

        if scheduler is not None:
            scheduler.step()

        # ---- VALIDATION (on EMA if enabled) ----
        if ep % cfg["val_interval_epochs"] == 0:
            if ema is not None:
                ema.apply_to(model)
            metrics = evaluate(model, dl_val, device, thresh=eval_thresh)
            if ema is not None:
                ema.restore(model)

            print(f"Val: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} IoU={metrics['iou_moving']:.3f}")

            # always save "last" (EMA weights if enabled)
            state_to_save = model.state_dict()
            if ema is not None:
                ema.apply_to(model)
                state_to_save = model.state_dict()
                ema.restore(model)

            save_checkpoint(
                "checkpoints",
                "mos_last.pt",
                {"ep": ep, "model": state_to_save, "optim": optim.state_dict(), "cfg": cfg, "best_f1": best_f1},
            )

            # Save "best" when F1 improves
            improved = metrics["f1"] > best_f1
            if improved:
                best_f1 = metrics["f1"]
                save_checkpoint(
                    "checkpoints",
                    "mos_best.pt",
                    {"ep": ep, "model": state_to_save, "optim": optim.state_dict(), "cfg": cfg, "best_f1": best_f1},
                )


if __name__ == "__main__":
    main()
