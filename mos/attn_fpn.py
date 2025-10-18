# mos/models/attn_fpn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# --- small helpers ---
class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None, groups=1):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False, groups=groups)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class SCSE(nn.Module):
    # Squeeze & channel excitation + spatial excitation
    def __init__(self, c, r=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//r, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c//r, c, 1),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(c, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class FPNBlock(nn.Module):
    def __init__(self, c_in, c_lat):
        super().__init__()
        self.lateral = nn.Conv2d(c_in, c_lat, 1, bias=False)
        self.out     = ConvBNAct(c_lat, c_lat, 3)
        self.scse    = SCSE(c_lat)
    def forward(self, x, up=None):
        h = self.lateral(x)
        if up is not None:
            h = h + F.interpolate(up, size=h.shape[-2:], mode='bilinear', align_corners=False)
        h = self.out(h)
        h = self.scse(h)
        return h

class SegHead(nn.Module):
    def __init__(self, c_in, c_mid=64, out_ch=1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(c_in, c_mid, 3),
            nn.Dropout2d(0.1),
            nn.Conv2d(c_mid, out_ch, 1)
        )
    def forward(self, x): return self.block(x)

# --- main model ---
class AttnFPN(nn.Module):
    """
    Encoder: any timm CNN/ConvNeXt/EfficientNet
    Decoder: 4-level FPN with SCSE attention
    Deep supervision: aux heads at all pyramid outputs
    Optional image-level presence gate
    """
    def __init__(self,
                 encoder_name='convnext_tiny',
                 in_ch=3,
                 out_ch=1,
                 fpn_dim=128,
                 use_gate=True,
                 pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name, features_only=True, in_chans=in_ch,
            pretrained=pretrained, out_indices=(1,2,3,4)
        )
        feats = self.encoder.feature_info.channels()  # [C2,C3,C4,C5]
        self.use_gate = use_gate

        # top-down FPN
        self.top = nn.Conv2d(feats[-1], fpn_dim, 1, bias=False)
        self.p4  = FPNBlock(feats[-2], fpn_dim)
        self.p3  = FPNBlock(feats[-3], fpn_dim)
        self.p2  = FPNBlock(feats[-4], fpn_dim)

        # fuse pyramid to a common stride (p2 resolution)
        self.fuse = ConvBNAct(fpn_dim*4, fpn_dim, 3)

        # heads (deep supervision)
        self.head_p5 = SegHead(fpn_dim, 64, out_ch)
        self.head_p4 = SegHead(fpn_dim, 64, out_ch)
        self.head_p3 = SegHead(fpn_dim, 64, out_ch)
        self.head_p2 = SegHead(fpn_dim, 64, out_ch)

        # image-level presence gate (binary)
        if self.use_gate:
            gate_dim = 256
            self.gap  = nn.AdaptiveAvgPool2d(1)
            self.gate = nn.Sequential(
                nn.Conv2d(fpn_dim, gate_dim, 1), nn.SiLU(inplace=True),
                nn.Dropout(0.2),
                nn.Conv2d(gate_dim, 1, 1), nn.Sigmoid()
            )

    def forward(self, x):
        c2, c3, c4, c5 = self.encoder(x)
        p5 = self.top(c5)
        p4 = self.p4(c4, p5)
        p3 = self.p3(c3, p4)
        p2 = self.p2(c2, p3)

        # upsample all to p2 size and fuse
        u5 = F.interpolate(p5, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        u4 = F.interpolate(p4, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        u3 = F.interpolate(p3, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat([u5,u4,u3,p2], dim=1))

        # deep supervision logits (all at input scale)
        logit_p5 = F.interpolate(self.head_p5(u5), size=x.shape[-2:], mode='bilinear', align_corners=False)
        logit_p4 = F.interpolate(self.head_p4(u4), size=x.shape[-2:], mode='bilinear', align_corners=False)
        logit_p3 = F.interpolate(self.head_p3(u3), size=x.shape[-2:], mode='bilinear', align_corners=False)
        logit_p2 = F.interpolate(self.head_p2(fused), size=x.shape[-2:], mode='bilinear', align_corners=False)

        if self.use_gate:
            g = self.gate(self.gap(fused))  # [B,1,1,1] in [0,1]
            logit_p2 = logit_p2 * g

        return {
            'logits': logit_p2,
            'aux': [logit_p5, logit_p4, logit_p3],
        }
