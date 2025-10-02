# mos/train.py
from __future__ import annotations
import os, math, yaml, warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SemanticKITTIMOS, mos_collate
from .model import TinyPointNetSeg
from .losses import WeightedFocalBCE
from .metrics import binary_stats
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
    if cfg.get("scheduler", "cosine") != "cosine":
        return None, lambda ep: None
    total_epochs = int(cfg["epochs"])
    warmup = int(cfg.get("warmup_epochs", 0))

    def lr_lambda(epoch_idx):
        e = max(0, epoch_idx - 1)
        if warmup > 0 and e < warmup:
            return float(e + 1) / float(warmup)
        T = max(1, total_epochs - warmup)
        t = min(T, e - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t / T))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler, lr_lambda


# ----------------- range-based sample weights -----------------
def build_range_weights(points: torch.Tensor, cfg) -> torch.Tensor:
    rw_cfg = cfg.get("range_weight", {})
    if not rw_cfg or not rw_cfg.get("enable", False):
        return torch.ones(points.shape[0], device=points.device, dtype=torch.float32)
    alpha = float(rw_cfg.get("alpha", 0.5))
    r0 = float(rw_cfg.get("r0", 20.0))
    r1 = float(rw_cfg.get("r1", 50.0))
    r = points[:, -1]  # last feature is range
    t = torch.clamp((r - r0) / max(1e-6, (r1 - r0)), min=0.0, max=1.0)
    w = 1.0 + alpha * t
    return w.to(torch.float32).clamp_(1.0, 1.5)


# ----------------- forgiving checkpoint load -----------------
def load_forgiving(model: torch.nn.Module, state_dict: dict) -> tuple[int, int]:
    msd = model.state_dict()
    new_sd = {}
    loaded = skipped = 0
    for k, v in state_dict.items():
        if k in msd and msd[k].shape == v.shape:
            new_sd[k] = v
            loaded += 1
        else:
            skipped += 1
    missing_before = set(msd.keys()) - set(new_sd.keys())
    model.load_state_dict({**msd, **new_sd}, strict=False)
    if skipped > 0:
        print(f"[resume] forgiving load: loaded={loaded}, skipped={skipped}, still_missing={len(missing_before)}")
    else:
        print(f"[resume] forgiving load: loaded={loaded}, skipped=0")
    return loaded, skipped


def try_resume(model, optim, cfg, device, resume_path: str | None, resume_last: bool, fresh: bool):
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
        print(f"Resuming weights from: {path}")
        ckpt = torch.load(path, map_location=device)
        sd = ckpt.get("model", ckpt)
        load_forgiving(model, sd)

        if (not fresh) and ("optim" in ckpt):
            try:
                optim.load_state_dict(ckpt["optim"])
            except Exception as e:
                print(f"  (warn) couldn't load optimizer state (resetting): {e}")

        best_f1 = float(ckpt.get("best_f1", best_f1))
        start_epoch = int(ckpt.get("ep", 0)) + 1
        print(f"  start_epoch={start_epoch}, best_f1={best_f1:.4f}")
    else:
        if resume_path or resume_last:
            print("  (info) no valid checkpoint found to resume; starting fresh.")
    return start_epoch, best_f1


# ----------------- loss plumbing -----------------
def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_w: torch.Tensor,
    focal_loss: WeightedFocalBCE | None,
    bce_loss_none: nn.BCEWithLogitsLoss | None,
    use_bce: bool,
) -> torch.Tensor:
    sample_w = sample_w.to(dtype=torch.float32)
    if use_bce:
        per = bce_loss_none(logits, targets)  # [N]
        per = per.to(torch.float32)
        loss = (per * sample_w).mean()
    else:
        loss = focal_loss(logits, targets, sample_weight=sample_w)
    return loss


# ----------------- prediction stats -----------------
@torch.no_grad()
def pred_stats(logits: torch.Tensor):
    s = torch.sigmoid(logits)
    return dict(
        mean=float(s.mean().cpu()),
        min=float(s.min().cpu()),
        max=float(s.max().cpu()),
        p001=float(torch.quantile(s, 0.01).cpu()),
        p010=float(torch.quantile(s, 0.10).cpu()),
        p900=float(torch.quantile(s, 0.90).cpu()),
        p999=float(torch.quantile(s, 0.99).cpu()),
    )


def train_one_epoch(
    model,
    loader,
    optim,
    device,
    cfg,
    ema: EMA | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    focal_loss: WeightedFocalBCE,
    bce_loss_none: nn.BCEWithLogitsLoss,
    use_bce: bool,
    log_interval=50,
):
    model.train()
    running = 0.0
    n = 0
    for it, batch in enumerate(tqdm(loader, desc="train")):
        pts = batch["points"].to(device)
        y   = batch["label"].to(device)
        bidx = batch["batch_idx"].to(device)

        pts = torch.nan_to_num(pts, nan=0.0, posinf=1e6, neginf=-1e6)
        sample_w = build_range_weights(pts, cfg)

        optim.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
            logits = model(pts, bidx).reshape(-1)
            loss = compute_loss(
                logits, y, sample_w,
                focal_loss=focal_loss,
                bce_loss_none=bce_loss_none,
                use_bce=use_bce,
            )

        if not torch.isfinite(loss):
            print("  (warn) non-finite loss, skipping step")
            continue

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

        if ema is not None:
            ema.update(model)

        running += float(loss.detach().cpu())
        n += 1
        if (it + 1) % log_interval == 0:
            with torch.no_grad():
                ppos = (y > 0.5).float().mean().item()
                s = pred_stats(logits.detach())
                pmov = float((torch.sigmoid(logits.detach()) > 0.5).float().mean().cpu())
            print(f"  it {it+1}: loss={running/n:.4f}")
            print(f"  it {it+1}: pos_rate={ppos:.3f}  pred_rate@0.5={pmov:.3f}  sigm(mean/min/p10/p90/max)={s['mean']:.4f}/{s['min']:.4f}/{s['p010']:.4f}/{s['p900']:.4f}/{s['max']:.4f}")
            running, n = 0.0, 0


# ----------------- evaluation (with threshold sweep) -----------------
@torch.no_grad()
def evaluate(model, loader, device, fixed_thresh: float = 0.5, do_sweep: bool = True):
    model.eval()
    all_logits = []
    all_y = []
    for batch in tqdm(loader, desc="val"):
        pts = batch["points"].to(device)
        y   = batch["label"].to(device)
        bidx = batch["batch_idx"].to(device)
        logits = model(pts, bidx).reshape(-1)
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())
    logits = torch.cat(all_logits, 0)
    y = torch.cat(all_y, 0)

    # fixed-threshold metrics
    s_fixed = binary_stats(logits, y, thresh=fixed_thresh)
    prec_f = s_fixed["tp"] / (s_fixed["tp"] + s_fixed["fp"] + 1e-8)
    rec_f  = s_fixed["tp"] / (s_fixed["tp"] + s_fixed["fn"] + 1e-8)
    f1_f   = 2 * prec_f * rec_f / (prec_f + rec_f + 1e-8)
    iou_f  = s_fixed["tp"] / (s_fixed["tp"] + s_fixed["fp"] + s_fixed["fn"] + 1e-8)

    best = dict(thresh=fixed_thresh, precision=prec_f, recall=rec_f, f1=f1_f, iou_moving=iou_f)

    if do_sweep:
        # try a grid of thresholds
        ths = torch.linspace(0.01, 0.90, steps=30)
        best_f1 = -1.0
        for t in ths:
            st = binary_stats(logits, y, thresh=float(t))
            p = st["tp"] / (st["tp"] + st["fp"] + 1e-8)
            r = st["tp"] / (st["tp"] + st["fn"] + 1e-8)
            f1 = 2 * p * r / (p + r + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best = dict(thresh=float(t), precision=p, recall=r, f1=f1,
                            iou_moving=st["tp"] / (st["tp"] + st["fp"] + st["fn"] + 1e-8))

    return dict(
        fixed=dict(thresh=fixed_thresh, precision=prec_f, recall=rec_f, f1=f1_f, iou_moving=iou_f),
        best=best,
        pred_summary=pred_stats(logits),
        pos_rate=float(y.float().mean().item()),
    )


def save_checkpoint(out_dir: str, fname: str, payload: dict):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    torch.save(payload, path)
    print(f"Saved {path}")


# ----------------- prior bias init -----------------
@torch.no_grad()
def init_head_bias_from_prior(model: nn.Module, loader: DataLoader, device, max_batches: int = 8):
    """
    Estimate P(y=1) from a few batches and set final layer bias to logit(P).
    Works if model has attribute .head with last Linear out_features=1.
    """
    # find last linear with out_features=1
    last_lin = None
    if hasattr(model, "head") and isinstance(model.head, nn.Sequential):
        for m in reversed(model.head):
            if isinstance(m, nn.Linear) and m.out_features == 1:
                last_lin = m
                break
    if last_lin is None or last_lin.bias is None:
        return  # nothing to do

    total = 0
    pos = 0.0
    iters = 0
    for batch in loader:
        y = batch["label"].to(device)
        pos += float(y.sum().item())
        total += int(y.numel())
        iters += 1
        if iters >= max_batches:
            break
    if total == 0:
        return
    p = max(1e-5, min(1 - 1e-5, pos / total))
    bias = math.log(p / (1.0 - p))
    last_lin.bias.data.fill_(bias)
    print(f"[init] set final bias to prior logit={bias:.4f} (p≈{p:.6f}, from {total} labels)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--resume_last", action="store_true")
    ap.add_argument("--resume_path", type=str, default=None)
    ap.add_argument("--fresh", action="store_true", help="do not load optimizer state even if present")
    ap.add_argument("--no_compile", action="store_true", help="disable torch.compile even if available")
    ap.add_argument("--no_amp", action="store_true", help="disable AMP/mixed precision")
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

    # model
    in_ch = 6 if cfg.get("use_prev", True) else 5
    base_model = TinyPointNetSeg(in_channels=in_ch)
    if (device.type == "cuda") and (not args.no_compile):
        try:
            base_model = torch.compile(base_model)
            print("[compile] torch.compile enabled")
        except Exception as e:
            print(f"[compile] disabled ({e})")
    model = base_model.to(device)

    # losses
    focal_loss = WeightedFocalBCE(
        pos_weight=cfg["class_weight_moving"],
        gamma=cfg["focal_gamma"],
        reduction="mean",
    )
    bce_pos_weight = torch.tensor([cfg["class_weight_moving"]], device=device, dtype=torch.float32)
    bce_loss_none = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight, reduction="none")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler, _ = build_scheduler(optim, cfg)
    ema = EMA(model, decay=float(cfg.get("ema_decay", 0.999))) if cfg.get("ema_enable", False) else None

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

    # optional: initialize final bias from label prior to avoid "all zeros" at start
    if cfg.get("init_prior_bias", True):
        init_head_bias_from_prior(model, dl_train, device, max_batches=int(cfg.get("prior_scan_batches", 8)))

    # resume
    start_epoch, best_f1 = try_resume(
        model, optim, cfg, device,
        resume_path=args.resume_path,
        resume_last=args.resume_last,
        fresh=args.fresh,
    )

    # AMP
    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = int(cfg["epochs"])
    eval_thresh = float(cfg.get("eval_threshold", 0.5))
    burnin_epochs = int(cfg.get("bce_burnin_epochs", 0))
    sweep_eval = bool(cfg.get("eval_sweep", True))

    for ep in range(start_epoch, epochs + 1):
        print(f"\nEpoch {ep}/{epochs}")
        use_bce = ep <= burnin_epochs  # BCE burn-in, then focal

        train_one_epoch(
            model=model,
            loader=dl_train,
            optim=optim,
            device=device,
            cfg=cfg,
            ema=ema,
            scaler=scaler,
            use_amp=use_amp,
            focal_loss=focal_loss,
            bce_loss_none=bce_loss_none,
            use_bce=use_bce,
            log_interval=cfg["log_interval"],
        )

        if scheduler is not None:
            scheduler.step()

        # ---- VALIDATION ----
        if ep % cfg["val_interval_epochs"] == 0:
            if ema is not None:
                ema.apply_to(model)
            metrics = evaluate(model, dl_val, device, fixed_thresh=eval_thresh, do_sweep=sweep_eval)
            if ema is not None:
                ema.restore(model)

            fx = metrics["fixed"]
            bs = metrics["best"]
            ps = metrics["pred_summary"]
            print(
                f"Val (fixed t={fx['thresh']:.2f}): P={fx['precision']:.3f} R={fx['recall']:.3f} "
                f"F1={fx['f1']:.3f} IoU={fx['iou_moving']:.3f}"
            )
            if sweep_eval:
                print(
                    f"Val (best  t={bs['thresh']:.2f}): P={bs['precision']:.3f} R={bs['recall']:.3f} "
                    f"F1={bs['f1']:.3f} IoU={bs['iou_moving']:.3f}"
                )
            print(
                f"[pred stats] sigmoid mean={ps['mean']:.4f} min={ps['min']:.4f} p10={ps['p010']:.4f} "
                f"p90={ps['p900']:.4f} max={ps['max']:.4f}  (val pos_rate≈{metrics['pos_rate']:.4f})"
            )

            # save "last" (EMA weights if enabled)
            state_to_save = model.state_dict()
            if ema is not None:
                ema.apply_to(model)
                state_to_save = model.state_dict()
                ema.restore(model)

            save_checkpoint(
                "checkpoints",
                "mos_last.pt",
                {"ep": ep, "model": state_to_save, "optim": (optim.state_dict() if not args.fresh else {}), "cfg": cfg, "best_f1": best_f1},
            )

            # save "best" by F1 (use swept F1 if enabled, else fixed)
            curr_f1 = bs["f1"] if sweep_eval else fx["f1"]
            if curr_f1 > best_f1:
                best_f1 = curr_f1
                save_checkpoint(
                    "checkpoints",
                    "mos_best.pt",
                    {"ep": ep, "model": state_to_save, "optim": (optim.state_dict() if not args.fresh else {}), "cfg": cfg, "best_f1": best_f1},
                )
#end of the training file

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="index_reduce\\(\\) is in beta", category=UserWarning)
    main()
