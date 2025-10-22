from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- grid & sampling -----------------------------

def make_bev_grid(cfg):
    bev = cfg["bev"]
    vx = float(bev["voxel_size_xy"])
    x0, x1 = float(bev["x_min"]), float(bev["x_max"])
    y0, y1 = float(bev["y_min"]), float(bev["y_max"])
    nx = int(math.ceil((x1 - x0) / vx))
    ny = int(math.ceil((y1 - y0) / vx))
    return dict(v=vx, x0=x0, y0=y0, nx=nx, ny=ny)

def bilinear_sample(feat: torch.Tensor, xy_norm: torch.Tensor) -> torch.Tensor:
    # feat: [B,C,H,W], xy_norm: [N,2] in [-1,1]
    B, C, H, W = feat.shape
    N = int(xy_norm.shape[0])
    grid = xy_norm.view(1, 1, N, 2)
    out = F.grid_sample(feat, grid, mode="bilinear", align_corners=True, padding_mode="zeros")
    out = out.view(B, C, -1).permute(0, 2, 1).reshape(-1, C)
    return out

def coord_conv(H, W, device):
    ys = torch.linspace(-1, 1, steps=H, device=device).view(1,1,H,1).expand(1,1,H,W)
    xs = torch.linspace(-1, 1, steps=W, device=device).view(1,1,1,W).expand(1,1,H,W)
    return torch.cat([xs, ys], dim=1)  # [1,2,H,W]

# ----------------------------- blocks -----------------------------

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        mid = max(4, ch // r)
        self.fc1 = nn.Conv2d(ch, mid, 1)
        self.fc2 = nn.Conv2d(mid, ch, 1)
    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w

class CBAM(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        mid = max(4, ch // r)
        self.mlp  = nn.Sequential(nn.Conv2d(ch, mid, 1), nn.ReLU(True), nn.Conv2d(mid, ch, 1))
        self.spat = nn.Conv2d(2, 1, kernel_size=7, padding=3)
    def forward(self, x):
        maxp = F.adaptive_max_pool2d(x, 1)
        avgp = F.adaptive_avg_pool2d(x, 1)
        w = torch.sigmoid(self.mlp(maxp) + self.mlp(avgp))
        x = x * w
        m = torch.cat([x.max(1, keepdim=True)[0], x.mean(1, keepdim=True)], dim=1)
        s = torch.sigmoid(self.spat(m))
        return x * s

def conv3x3(cin, cout, s=1, g=1): return nn.Conv2d(cin, cout, 3, s, 1, bias=False, groups=g)

class ResBlock(nn.Module):
    def __init__(self, cin, cout, s=1):
        super().__init__()
        self.proj = nn.Identity() if (s==1 and cin==cout) else nn.Conv2d(cin, cout, 1, s, bias=False)
        self.bn0  = nn.BatchNorm2d(cin)
        self.conv1= conv3x3(cin, cout, s)
        self.bn1  = nn.BatchNorm2d(cout)
        self.conv2= conv3x3(cout, cout, 1)
        self.cbam = CBAM(cout)
    def forward(self, x):
        h = F.relu(self.bn0(x), inplace=True)
        h = self.conv1(h)
        h = F.relu(self.bn1(h), inplace=True)
        h = self.conv2(h)
        h = self.cbam(h)
        return h + self.proj(x)

class ASPP(nn.Module):
    def __init__(self, cin, cout, rates=(1,6,12,18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(cin, cout, 3, padding=rates[1], dilation=rates[1], bias=False), nn.BatchNorm2d(cout), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(cin, cout, 3, padding=rates[2], dilation=rates[2], bias=False), nn.BatchNorm2d(cout), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(cin, cout, 3, padding=rates[3], dilation=rates[3], bias=False), nn.BatchNorm2d(cout), nn.ReLU(True)),
        ])
        self.fuse = nn.Sequential(nn.Conv2d(cout*4, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True), SEBlock(cout))
    def forward(self, x):
        xs = [b(x) for b in self.branches]
        x  = torch.cat(xs, dim=1)
        return self.fuse(x)

# ----------------------------- UNet (residual, attentional) -----------------------------

class ResUNetBEV(nn.Module):
    def __init__(self, in_ch: int, base: int, aspp_rates=(1,6,12,18), deep_sup=True, dropout=0.1):
        super().__init__()
        b = base
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(True),
            ResBlock(b, b), SEBlock(b)
        )
        self.e2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(b, 2*b), ResBlock(2*b, 2*b))
        self.e3 = nn.Sequential(nn.MaxPool2d(2), ResBlock(2*b, 4*b), ResBlock(4*b, 4*b))
        self.e4 = nn.Sequential(nn.MaxPool2d(2), ResBlock(4*b, 8*b))

        self.bott = nn.Sequential(ASPP(8*b, 8*b, aspp_rates), nn.Dropout2d(dropout))

        self.up3 = nn.ConvTranspose2d(8*b, 4*b, 2, 2)
        self.d3  = nn.Sequential(ResBlock(8*b, 4*b), ResBlock(4*b, 4*b))
        self.up2 = nn.ConvTranspose2d(4*b, 2*b, 2, 2)
        self.d2  = nn.Sequential(ResBlock(4*b, 2*b), ResBlock(2*b, 2*b))
        self.up1 = nn.ConvTranspose2d(2*b,   b, 2, 2)
        self.d1  = nn.Sequential(ResBlock(2*b, b), ResBlock(b, b))

        self.out      = nn.Conv2d(b, 1, 1)   # main MOS
        self.center_h = nn.Conv2d(b, 1, 1)   # self-supervised center (NEW)

        self.deep_sup = deep_sup
        if deep_sup:
            self.aux2 = nn.Conv2d(2*b, 1, 1)   # H/2, W/2
            self.aux3 = nn.Conv2d(4*b, 1, 1)   # H/4, W/4
    
    def _match(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=True)
        return x

    def forward(self, x, *, temporal_fuse=None):
        # temporal_fuse: optional function that mutates e4/bott for short-range fusion
        s1 = self.stem(x)          # [B,b,H,W]
        e2 = self.e2(s1)           # [B,2b,H/2,W/2]
        e3 = self.e3(e2)           # [B,4b,H/4,W/4]
        e4 = self.e4(e3)           # [B,8b,H/8,W/8]

        if temporal_fuse is not None:
            e4 = temporal_fuse(e4)

        bt = self.bott(e4)         # [B,8b,H/8,W/8]

        u3 = self.up3(bt)                         # ~[B,4b,H/4, W/4]
        u3 = self._match(u3, e3)
        d3 = self.d3(torch.cat([u3, e3], 1))     # [B,4b,H/4,W/4]

        u2 = self.up2(d3)                         # ~[B,2b,H/2, W/2]
        u2 = self._match(u2, e2)
        d2 = self.d2(torch.cat([u2, e2], 1))     # [B,2b,H/2,W/2]

        u1 = self.up1(d2)                         # ~[B,b,H, W]
        u1 = self._match(u1, s1)
        d1 = self.d1(torch.cat([u1, s1], 1))     # [B,b,H,W]
        bev_logits = self.out(d1)
        center_logits = self.center_h(d1)

        aux = []
        if self.deep_sup:
            aux2 = F.interpolate(self.aux2(d2), size=bev_logits.shape[-2:], mode="bilinear", align_corners=True)
            aux3 = F.interpolate(self.aux3(d3), size=bev_logits.shape[-2:], mode="bilinear", align_corners=True)
            aux = [aux2, aux3]

        return bev_logits, center_logits, aux

# ----------------------------- top-level model -----------------------------

class BEVResUNetSegModel(nn.Module):
    """
    Stronger BEV model with residual U-Net, ASPP, CBAM, deep supervision,
    + self-supervised center head and optional short-range temporal fusion.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.grid = make_bev_grid(cfg)
        g = self.grid
        self.register_buffer("x0", torch.tensor(g["x0"]))
        self.register_buffer("y0", torch.tensor(g["y0"]))
        self.v = g["v"]; self.nx = g["nx"]; self.ny = g["ny"]

        # rasterizer inputs: [dens, mean_i, mean_h, dmin_min, dmin_max, range_comp, coord_x, coord_y]
        in_ch = 8
        base  = int(cfg["bev"].get("base_ch", 64))
        rates = tuple(cfg["bev"].get("aspp_rates", [1,6,12,18]))
        deep_sup = bool(cfg.get("deep_supervision", True))
        dropout  = float(cfg.get("dropout", 0.10))
        self.backbone = ResUNetBEV(in_ch=in_ch, base=base, aspp_rates=rates, deep_sup=deep_sup, dropout=dropout)

        # simple temporal FIFO per forward() call (stub; only used if cfg.TEMPORAL.enable)
        tcfg = cfg.get("TEMPORAL", {"enable": False})
        self.temporal_on = bool(tcfg.get("enable", False))
        self.temporal_decay = float(tcfg.get("decay", 0.7))
        self.temporal_fuse_mode = str(tcfg.get("fuse", "ema")).lower()
        self.register_buffer("_prev_e4", None, persistent=False)   # [B,8b,H/8,W/8] cached per batch

    def _rasterize(self, pts: torch.Tensor, bidx: torch.Tensor) -> torch.Tensor:
        device = pts.device
        H, W = self.ny, self.nx
        v = self.v
        feats = []
        B = int(bidx.max().item() + 1) if bidx.numel() else 1
        for b in range(B):
            m = (bidx == b)
            p = pts[m]
            if p.numel() == 0:
                feats.append(torch.zeros(1, 8, H, W, device=device)); continue
            x, y, z = p[:,0], p[:,1], p[:,2]
            inten   = p[:,3] if p.shape[1] >= 4 else torch.zeros_like(x)
            rng     = p[:,4] if p.shape[1] >= 5 else torch.sqrt(x*x + y*y)
            dmin    = p[:,5] if p.shape[1] >= 6 else torch.zeros_like(x)

            ix = torch.clamp(((x - self.x0) / v).floor().long(), 0, W-1)
            iy = torch.clamp(((y - self.y0) / v).floor().long(), 0, H-1)
            idx = iy * W + ix

            one = torch.ones_like(x)
            dens = torch.zeros(H*W, device=device).index_add(0, idx, one)
            isum = torch.zeros(H*W, device=device).index_add(0, idx, inten)
            hsum = torch.zeros(H*W, device=device).index_add(0, idx, z)

            rc = ((rng - 15.0) / 45.0).clamp(-1.0, 1.0)
            rc_grid = torch.zeros(H*W, device=device).index_add(0, idx, rc)

            # dmin pseudo min/max (scatter-ish)
            dmin_min = torch.full((H*W,), float("inf"), device=device)
            dmin_min.index_put_((idx,), dmin, accumulate=False)
            dmin_min = torch.where(torch.isfinite(dmin_min), dmin_min, torch.zeros_like(dmin_min))
            dmin_max = torch.full((H*W,), float("-inf"), device=device)
            dmin_max.index_put_((idx,), dmin, accumulate=False)
            dmin_max = torch.where(torch.isfinite(dmin_max), dmin_max, torch.zeros_like(dmin_max))

            dens_safe = dens.clamp_min(1.0)
            mean_i = (isum / dens_safe)
            mean_h = (hsum / dens_safe)
            rc_grid = (rc_grid / dens_safe).clamp(-1.0, 1.0)

            to_hw = lambda t: t.view(H, W).unsqueeze(0).unsqueeze(0)
            feat = torch.cat([
                to_hw(dens), to_hw(mean_i), to_hw(mean_h),
                to_hw(dmin_min), to_hw(dmin_max), to_hw(rc_grid),
                coord_conv(H, W, device)
            ], dim=1)  # [1,8,H,W]
            feats.append(feat)
        return torch.cat(feats, dim=0)  # [B,8,H,W]

    def _points_to_norm_xy(self, pts: torch.Tensor):
        g = self.grid
        x = pts[:,0]; y = pts[:,1]
        xn = 2.0 * ((x - g["x0"]) / (g["nx"] * g["v"])) - 1.0
        yn = 2.0 * ((y - g["y0"]) / (g["ny"] * g["v"])) - 1.0
        return torch.stack([xn, yn], dim=1)

    def _temporal_fuser(self):
        if not self.temporal_on:
            return None
        mode = self.temporal_fuse_mode
        decay = self.temporal_decay
        def fuse(e4: torch.Tensor) -> torch.Tensor:
            if self._prev_e4 is None or self._prev_e4.shape != e4.shape:
                self._prev_e4 = e4.detach()
                return e4
            if mode == "max":
                out = torch.max(e4, self._prev_e4)
            else:  # EMA
                out = decay * self._prev_e4 + (1.0 - decay) * e4
            self._prev_e4 = out.detach()
            return out
        return fuse

    def forward(self, points: torch.Tensor, batch_idx: torch.Tensor):
        bev = self._rasterize(points, batch_idx)                      # [B,8,H,W]
        bev_logits, center_logits, aux_maps = self.backbone(
            bev, temporal_fuse=self._temporal_fuser()
        )                                                             # [B,1,H,W], [B,1,H,W], [...]
        B = int(batch_idx.max().item() + 1) if batch_idx.numel() else 1
        out_logits = torch.empty(points.shape[0], device=points.device, dtype=torch.float32)
        for b in range(B):
            m = (batch_idx == b)
            if not m.any(): continue
            xy_norm = self._points_to_norm_xy(points[m])
            l_b = bilinear_sample(bev_logits[b:b+1], xy_norm)         # [Mb,1]
            out_logits[m] = l_b.squeeze(1)

        # Deep supervision at points
        aux_logits = []
        for am in aux_maps:
            a = torch.empty(points.shape[0], device=points.device, dtype=torch.float32)
            for b in range(B):
                m = (batch_idx == b)
                if not m.any(): continue
                xy_norm = self._points_to_norm_xy(points[m])
                a_b = bilinear_sample(am[b:b+1], xy_norm).squeeze(1)
                a[m] = a_b
            aux_logits.append(a)

        return {
            "logits": out_logits,         # [N]
            "aux": aux_logits,            # list([N])
            "bev_logits": bev_logits,     # [B,1,H,W]  (for self-supervised center target)
            "center_logits": center_logits # [B,1,H,W]
        }

# ----------------------------- factory -----------------------------

class TinyPointNetSeg(nn.Module):
    def __init__(self, in_ch=5, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, 1),
        )
    def forward(self, pts, bidx): return self.mlp(pts).squeeze(-1)

def build_model(cfg) -> nn.Module:
    name = str(cfg.get("model_name", "bev_resunet")).lower()
    if name in ("bev_resunet", "bev_unet_plus"):
        return BEVResUNetSegModel(cfg)
    else:
        in_ch = 6 if cfg.get("use_prev", False) else 5
        return TinyPointNetSeg(in_ch=in_ch)
