# mos/train.py
from __future__ import annotations
import math, yaml, warnings
import torch
import torch.nn as nn
import os, sys
from torch._dynamo import config as dynamo_config

# Try to disable Inductor's CPU C++ path (which looks for cl.exe on Windows)
try:
    import torch._inductor.config as inductor_config
    inductor_config.cpp.enable = False
except Exception:
    pass

# Also disable via env (PyTorch checks this)
os.environ.setdefault("TORCHINDUCTOR_DISABLE_CPP", "1")

# Avoid graph breaks from .item() in compiled graphs
dynamo_config.capture_scalar_outputs = True

from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SemanticKITTIMOS, mos_collate
from .model import build_model, TinyPointNetSeg  # TinyPointNetSeg kept for bias init scan
from .losses import WeightedFocalBCE
from .metrics import binary_stats
from .utils import set_seed

# Optional ComboLoss (if you add it later)
try:
    from .losses import ComboLoss  # type: ignore
except Exception:
    ComboLoss = None  # pyright: ignore[reportConstantRedefinition]


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
        self.shadow, self.backup = {}, {}
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


# ----------------- prediction stats (safe for huge tensors) -----------------
@torch.no_grad()
def pred_stats(logits: torch.Tensor, max_elements: int = 2_000_000):
    s = torch.sigmoid(logits.detach()).view(-1).to(dtype=torch.float32)
    n = s.numel()
    if n == 0:
        return dict(mean=0.0, min=0.0, max=0.0, p010=0.0, p900=0.0)
    if n > max_elements:
        idx = torch.randperm(n, device=s.device)[:max_elements]
        s = s.index_select(0, idx)
    out = dict(
        mean=float(s.mean().cpu()),
        min=float(s.min().cpu()),
        max=float(s.max().cpu()),
    )
    try:
        out["p010"] = float(torch.quantile(s, 0.10).cpu())
        out["p900"] = float(torch.quantile(s, 0.90).cpu())
    except RuntimeError:
        m = min(int(256_000), s.numel())
        if s.numel() > m:
            idx = torch.randperm(s.numel(), device=s.device)[:m]
            t = s.index_select(0, idx)
        else:
            t = s
        k10 = max(1, int(0.10 * (t.numel() - 1)))
        k90 = max(1, int(0.90 * (t.numel() - 1)))
        out["p010"] = float(t.kthvalue(k10).values.cpu())
        out["p900"] = float(t.kthvalue(k90).values.cpu())
    return out


# ----------------- simple BCE/focal loss -----------------
def compute_loss_basic(
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
        assert focal_loss is not None
        loss = focal_loss(logits, targets, sample_weight=sample_w)
    return loss


# ----------------- optional combo loss path -----------------
def compute_loss_combo(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_w: torch.Tensor,
    focal_loss: WeightedFocalBCE | None,
    bce_loss_none: nn.BCEWithLogitsLoss | None,
    use_bce: bool,
    combo_loss: "ComboLoss | None",
    aux_logits: list[torch.Tensor] | None,
) -> torch.Tensor:
    # If combo not provided, fall back to basic
    if combo_loss is None:
        return compute_loss_basic(logits, targets, sample_w, focal_loss, bce_loss_none, use_bce)

    # Otherwise: combo loss branch
    sample_w = sample_w.to(dtype=torch.float32)
    if use_bce:
        per = bce_loss_none(logits, targets)
        per = per.to(torch.float32)
        return (per * sample_w).mean()

    core = combo_loss(logits, targets, aux_logits=aux_logits or [])
    if focal_loss is not None:
        fb = focal_loss(logits, targets, sample_weight=sample_w)
        core = core + 0.1 * fb
    return core


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
    ema: EMA | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    focal_loss: WeightedFocalBCE | None,
    bce_loss_none: nn.BCEWithLogitsLoss | None,
    use_bce: bool,
    combo_loss: "ComboLoss | None",
    log_interval: int,
    clip_model: nn.Module | None = None,  # clip grads on this module (prefer raw_model)
    ema_model: nn.Module | None = None,   # update EMA from this module (prefer raw_model)
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
            # Support both tensor and dict outputs from the model
            out = model(pts, bidx)
            if isinstance(out, dict):
                logits = out["logits"].reshape(-1)
                aux = [a.reshape(-1) for a in out.get("aux", [])]
            else:
                logits = out.reshape(-1)
                aux = []

            loss = compute_loss_combo(
                logits, y, sample_w,
                focal_loss=focal_loss,
                bce_loss_none=bce_loss_none,
                use_bce=use_bce,
                combo_loss=combo_loss,
                aux_logits=aux,
            )

        if not torch.isfinite(loss):
            print("  (warn) non-finite loss, skipping step")
            continue

        to_clip = clip_model if clip_model is not None else model
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(to_clip.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(to_clip.parameters(), max_norm=1.0)
            optim.step()

        if ema is not None:
            ema.update(ema_model if ema_model is not None else to_clip)

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
        out = model(pts, bidx)
        logits = out["logits"].reshape(-1) if isinstance(out, dict) else out.reshape(-1)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.cpu())
    logits = torch.cat(all_logits, 0)
    y = torch.cat(all_y, 0)

    s_fixed = binary_stats(logits, y, thresh=fixed_thresh)
    prec_f = s_fixed["tp"] / (s_fixed["tp"] + s_fixed["fp"] + 1e-8)
    rec_f  = s_fixed["tp"] / (s_fixed["tp"] + s_fixed["fn"] + 1e-8)
    f1_f   = 2 * prec_f * rec_f / (prec_f + rec_f + 1e-8)
    iou_f  = s_fixed["tp"] / (s_fixed["tp"] + s_fixed["fp"] + s_fixed["fn"] + 1e-8)

    best = dict(thresh=fixed_thresh, precision=prec_f, recall=rec_f, f1=f1_f, iou_moving=iou_f)

    if do_sweep:
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
    Works if model has any Linear with out_features=1 in its heads.
    """
    last_lin = None
    for m in model.modules():
        if isinstance(m, nn.Linear) and getattr(m, "out_features", None) == 1:
            last_lin = m
    if last_lin is None or last_lin.bias is None:
        return

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

    # ---------------- model ----------------
    raw_model = build_model(cfg).to(device)  # always keep an uncompiled base model

    # Choose a safe backend for Windows (no Triton needed), and allow override via cfg.
    compile_backend = str(cfg.get("compile_backend", "auto")).lower()
    backend = None
    if compile_backend == "auto":
        # On Windows, avoid inductor (needs Triton). Use aot_eager instead.
        backend = "aot_eager" if os.name == "nt" else None  # None => default inductor on Linux
    else:
        backend = compile_backend  # e.g., "eager", "aot_eager", or "inductor"

    model_train = raw_model
    if (device.type == "cuda") and (not args.no_compile):
        try:
            if backend is None:
                model_train = torch.compile(raw_model)  # default (inductor) on Linux
                print("[compile] torch.compile enabled (backend=inductor)")
            else:
                model_train = torch.compile(raw_model, backend=backend)
                print(f"[compile] torch.compile enabled (backend={backend})")
        except Exception as e:
            print(f"[compile] disabled (fallback due to: {e})")
            model_train = raw_model

    # ------------- losses/optim/data -------------
    focal_loss = WeightedFocalBCE(
        pos_weight=cfg["class_weight_moving"],
        gamma=cfg["focal_gamma"],
        reduction="mean",
    )
    bce_pos_weight = torch.tensor([cfg["class_weight_moving"]], device=device, dtype=torch.float32)
    bce_loss_none = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight, reduction="none")

    # Optional: instantiate ComboLoss only if available + enabled in cfg
    combo_loss = None
    if ComboLoss is not None and cfg.get("combo_loss", {}).get("enable", False):
        cl = cfg["combo_loss"]
        combo_loss = ComboLoss(**{k: v for k, v in cl.items() if k != "enable"})  # type: ignore
        print("[loss] Using ComboLoss with params:", cl)

    optim = torch.optim.AdamW(raw_model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler, _ = build_scheduler(optim, cfg)
    ema = EMA(raw_model, decay=float(cfg.get("ema_decay", 0.999))) if cfg.get("ema_enable", False) else None

    ds_train, ds_val = make_datasets(cfg)
    dl_train = DataLoader(
        ds_train, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], collate_fn=mos_collate, drop_last=False,
    )
    dl_val = DataLoader(
        ds_val, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], collate_fn=mos_collate, drop_last=False,
    )

    if cfg.get("init_prior_bias", True):
        init_head_bias_from_prior(raw_model, dl_train, device, max_batches=int(cfg.get("prior_scan_batches", 8)))

    start_epoch, best_f1 = try_resume(
        raw_model, optim, cfg, device,
        resume_path=args.resume_path,
        resume_last=args.resume_last,
        fresh=args.fresh,
    )

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = int(cfg["epochs"])
    eval_thresh = float(cfg.get("eval_threshold", 0.5))
    burnin_epochs = int(cfg.get("bce_burnin_epochs", 0))
    sweep_eval = bool(cfg.get("eval_sweep", True))

    for ep in range(start_epoch, epochs + 1):
        print(f"\nEpoch {ep}/{epochs}")
        use_bce = ep <= burnin_epochs

        train_one_epoch(
            model=model_train,          # compiled wrapper (or raw_model if compile disabled)
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
            combo_loss=combo_loss,
            log_interval=cfg["log_interval"],
            clip_model=raw_model,       # clip on raw_model params
            ema_model=raw_model,        # update EMA from raw_model params
        )

        if scheduler is not None:
            scheduler.step()

        if ep % cfg["val_interval_epochs"] == 0:
            # Evaluate directly on the RAW (uncompiled) model.
            if ema is not None:
                ema.apply_to(raw_model)

            metrics = evaluate(raw_model, dl_val, device, fixed_thresh=eval_thresh, do_sweep=sweep_eval)

            if ema is not None:
                ema.restore(raw_model)

            fx = metrics["fixed"]; bs = metrics["best"]; ps = metrics["pred_summary"]
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

            # Save checkpoints from raw_model weights
            state_to_save = raw_model.state_dict()
            save_checkpoint(
                "checkpoints",
                "mos_last.pt",
                {"ep": ep, "model": state_to_save, "optim": (optim.state_dict() if not args.fresh else {}),
                 "cfg": cfg, "best_f1": best_f1},
            )

            curr_f1 = bs["f1"] if sweep_eval else fx["f1"]
            if curr_f1 > best_f1:
                best_f1 = curr_f1
                save_checkpoint(
                    "checkpoints",
                    "mos_best.pt",
                    {"ep": ep, "model": state_to_save, "optim": (optim.state_dict() if not args.fresh else {}),
                     "cfg": cfg, "best_f1": best_f1},
                )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="index_reduce\\(\\) is in beta", category=UserWarning)
    main()
