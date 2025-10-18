# mos/model.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------ Tiny MLP baseline (unchanged) ------------------------
class TinyPointNetSeg(nn.Module):
    """
    Minimal point MLP + simple global pooling (mean/max) broadcast back to points.
    Features per point: in_channels (e.g., x,y,z,intensity,range,[norm_dmin])
    """
    def __init__(
        self,
        in_channels: int = 6,      # 5 if no prev, 6 if range+norm_dmin
        h1: int = 64,
        h2: int = 256,
        head_hidden: int = 128,
    ):
        super().__init__()
        # per-point encoder
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, h1),
            nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
        )
        # head takes [x, g_mean, g_max] -> 3*h2 channels
        self.head = nn.Sequential(
            nn.Linear(3 * h2, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, points: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        points:    [N, in_channels]
        batch_idx: [N] int64 in [0..B-1]
        returns:   [N] logits
        """
        # per-point MLPs
        x = self.mlp1(points)          # [N, h1]
        x = self.mlp2(x)               # [N, h2]
        N, C = x.shape

        # batch size
        B = int(batch_idx.max().item()) + 1 if N > 0 else 0

        # dtype/device-safe accumulators
        dev, dt = x.device, x.dtype

        # --- SUM and MEAN pooling ---
        glob_sum = torch.zeros(B, C, device=dev, dtype=dt)
        glob_sum.index_add_(0, batch_idx, x)

        ones = torch.ones_like(x)
        glob_cnt = torch.zeros(B, C, device=dev, dtype=dt)
        glob_cnt.index_add_(0, batch_idx, ones)
        glob_mean = glob_sum / glob_cnt.clamp_min(1)

        # --- MAX pooling (index_reduce_ if present; fallback to scatter_reduce_) ---
        try:
            glob_max = torch.full((B, C), -float("inf"), device=dev, dtype=dt)
            glob_max.index_reduce_(0, batch_idx, x, "amax")
        except Exception:
            glob_max = torch.full((B, C), -float("inf"), device=dev, dtype=dt)
            glob_max.scatter_reduce_(0, batch_idx.view(-1, 1).expand(-1, C), x, reduce="amax", include_self=True)

        # broadcast global features to each point
        g_mean = glob_mean[batch_idx]   # [N, C]
        g_max  = glob_max[batch_idx]    # [N, C]

        h = torch.cat([x, g_mean, g_max], dim=1)  # [N, 3C]
        logits = self.head(h).squeeze(-1)         # [N]
        return logits


# ------------------------ BEV U-Net MOS (fixed scales) ------------------------
def conv3x3(in_ch, out_ch, stride=1, bias=False):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=bias)

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super().__init__()
        layers = [conv3x3(in_ch, out_ch, 1, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    """Downscale + 2 convs."""
    def __init__(self, in_ch, out_ch, bn=True):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, bn=bn),
            ConvBNReLU(out_ch, out_ch, bn=bn),
        )

    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    """Upscale, concat with skip (same spatial size), then 2 convs."""
    def __init__(self, in_ch, skip_ch, out_ch, bn=True):
        """
        in_ch:  channels of the incoming (lower) feature BEFORE upsample
        skip_ch: channels of the skip connection at this scale
        out_ch: output channels after this block
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBNReLU(out_ch + skip_ch, out_ch, bn=bn),
            ConvBNReLU(out_ch, out_ch, bn=bn),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # size guard for odd dims
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh != 0 or dw != 0:
            x = F.pad(x, (0, max(0, dw), 0, max(0, dh)))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class OutHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.head = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        return self.head(x)

class BEVUNetMOS(nn.Module):
    """
    Fast BEV U-Net for MOS. Projects point features to a fixed grid,
    runs a small 2D U-Net, samples logits back to points.
    """
    def __init__(
        self,
        in_channels: int = 6,   # points channels incl. xyz
        voxel_size: float = 0.20,
        bev_xyrange: tuple[float, float, float, float] = (-50.0, 50.0, -50.0, 50.0),
        base_ch: int = 32,
        depth: int = 4,
        bn: bool = True,
    ):
        super().__init__()
        assert depth == 4, "Depth 4 is implemented (C,2C,4C,8C enc; 16C bottleneck)."

        self.voxel_size = float(voxel_size)
        self.xmin, self.xmax, self.ymin, self.ymax = bev_xyrange
        self.in_channels = int(in_channels)
        self.feat_dim = max(1, self.in_channels - 3)  # exclude xyz
        C = int(base_ch)

        # point -> BEV encoder (produce C channels per voxel)
        self.pt_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, C),
            nn.ReLU(inplace=True),
        )

        # U-Net encoder
        self.enc1 = nn.Sequential(ConvBNReLU(C, C, bn=bn), ConvBNReLU(C, C, bn=bn))  # -> C
        self.down1 = Down(C, 2*C, bn=bn)      # -> 2C
        self.down2 = Down(2*C, 4*C, bn=bn)    # -> 4C
        self.down3 = Down(4*C, 8*C, bn=bn)    # -> 8C

        # bottleneck
        self.bot = nn.Sequential(ConvBNReLU(8*C, 16*C, bn=bn), ConvBNReLU(16*C, 16*C, bn=bn))

        # U-Net decoder: three ups (pair with s3, s2, s1)
        self.up1 = Up(in_ch=16*C, skip_ch=4*C, out_ch=8*C, bn=bn)   # H/8 -> H/4, skip s3(4C) -> 8C
        self.up2 = Up(in_ch=8*C,  skip_ch=2*C, out_ch=4*C, bn=bn)   # H/4 -> H/2, skip s2(2C) -> 4C
        self.up3 = Up(in_ch=4*C,  skip_ch=C,   out_ch=C,   bn=bn)   # H/2 -> H,   skip s1(C)  -> C

        self.out_head = OutHead(C)  # -> 1

    # ---------- BEV projection helpers ----------
    def _make_grid(self, device):
        vx = self.voxel_size
        H = int((self.xmax - self.xmin) / vx + 1e-6)
        W = int((self.ymax - self.ymin) / vx + 1e-6)
        return H, W

    def _points_to_indices(self, xyz: torch.Tensor):
        vx = self.voxel_size
        ix = torch.clamp(((xyz[:, 0] - self.xmin) / vx).floor().to(torch.long), 0,
                         int((self.xmax - self.xmin) / vx) - 1)
        iy = torch.clamp(((xyz[:, 1] - self.ymin) / vx).floor().to(torch.long), 0,
                         int((self.ymax - self.ymin) / vx) - 1)
        return ix, iy

    def _scatter_mean(self, feats: torch.Tensor, ix: torch.Tensor, iy: torch.Tensor, H: int, W: int):
        """
        feats: [N,C], ix/iy in [0..H/W-1]
        returns [1,C,H,W] mean-pooled features
        """
        C = feats.size(1)
        dev = feats.device
        bev = torch.zeros(C, H, W, device=dev, dtype=feats.dtype)
        cnt = torch.zeros(1, H, W, device=dev, dtype=feats.dtype)

        bev.index_put_((torch.arange(C, device=dev).unsqueeze(1), ix, iy),
                       feats.t(), accumulate=True)
        cnt.index_put_((torch.zeros(1, dtype=torch.long, device=dev), ix, iy),
                       torch.ones_like(ix, dtype=feats.dtype), accumulate=True)

        cnt = cnt.clamp_min_(1.0)
        bev = bev / cnt  # broadcast on C
        return bev.unsqueeze(0)  # [1,C,H,W]

    # ---------- UNet core ----------
    def _unet(self, bev: torch.Tensor) -> torch.Tensor:
        """
        bev: [1,C,H,W]  (C = base channels from pt_mlp)
        returns: [1,1,H,W] logits
        """
        s1 = self.enc1(bev)     # [1, C, H,   W]
        s2 = self.down1(s1)     # [1, 2C, H/2, W/2]
        s3 = self.down2(s2)     # [1, 4C, H/4, W/4]
        s4 = self.down3(s3)     # [1, 8C, H/8, W/8]

        xm = self.bot(s4)       # [1,16C, H/8, W/8]

        # decoder with matching scales
        x  = self.up1(xm, s3)   # [1, 8C, H/4, W/4]
        x  = self.up2(x,  s2)   # [1, 4C, H/2, W/2]
        x  = self.up3(x,  s1)   # [1,  C, H,   W]

        logit_map = self.out_head(x)  # [1,1,H,W]
        return logit_map

    def forward(self, points: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        points:    [N, in_channels] (xyz first)
        batch_idx: [N] (we expect a single frame per batch here; use zeros)
        returns:   [N] point logits
        """
        assert points.dim() == 2 and points.size(1) >= 4, "points must be [N, in_ch] with xyz+features"
        xyz = points[:, :3]
        feats = points[:, 3:3 + self.feat_dim]

        # Encode point features to C channels
        pf = self.pt_mlp(feats)  # [N, C]

        # Build BEV
        H, W = self._make_grid(points.device)
        ix, iy = self._points_to_indices(xyz)
        bev = self._scatter_mean(pf, ix, iy, H, W)  # [1, C, H, W]

        # U-Net over BEV
        logit_map = self._unet(bev)  # [1,1,H,W]

        # Sample back to each point's cell
        logits = logit_map[0, 0, ix, iy]  # [N]
        return logits


# --------- factory ---------
def build_model(cfg) -> nn.Module:
    """
    cfg keys:
      - model_name: "tiny_mlp" | "bev_unet"
      - in_channels: optional override; default matches your dataset (6 if use_prev else 5)
      - voxel_size, bev_xyrange, base_ch, depth, me_* ignored here
    """
    name = str(cfg.get("model_name", "tiny_mlp")).lower()
    use_prev = bool(cfg.get("use_prev", True))
    in_ch = 6 if use_prev else 5
    in_ch = int(cfg.get("in_channels", in_ch))

    if name in ("bev_unet", "bev", "unet2d"):
        return BEVUNetMOS(
            in_channels=in_ch,
            voxel_size=float(cfg.get("voxel_size", 0.20)),
            bev_xyrange=tuple(cfg.get("bev_xyrange", (-50.0, 50.0, -50.0, 50.0))),
            base_ch=int(cfg.get("bev_base_ch", 32)),
            depth=int(cfg.get("bev_depth", 4)),
            bn=bool(cfg.get("bev_bn", True)),
        )
    else:
        # default tiny MLP
        return TinyPointNetSeg(in_channels=in_ch)
