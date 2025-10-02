from __future__ import annotations
import os, glob
from typing import List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import (
    read_bin_points, read_labels, read_poses_txt, read_calib_velo_to_cam,
    se3_inv, transform_points, random_sample_indices
)
from .labels import load_moving_ids_from_yaml

def _voxel_downsample_np(xyz: np.ndarray, voxel: float) -> np.ndarray:
    if voxel is None or voxel <= 0:
        return xyz
    gx = np.floor(xyz[:,0] / voxel).astype(np.int32)
    gy = np.floor(xyz[:,1] / voxel).astype(np.int32)
    gz = np.floor(xyz[:,2] / voxel).astype(np.int32)
    key = (gx << 42) ^ (gy << 21) ^ gz
    _, idx = np.unique(key, return_index=True)
    return xyz[idx]

@torch.no_grad()
def _dmin_chunked(a_xyz: torch.Tensor,
                  b_xyz: torch.Tensor,
                  chunk: int = 4096) -> torch.Tensor:
    """
    a_xyz: [Na,3] current; b_xyz: [Nb,3] previous (already ego-comped into t)
    Returns dmin: [Na] using chunked cdist on GPU (fast, low mem).
    """
    device = a_xyz.device
    Na = a_xyz.shape[0]
    mins = []
    for s in range(0, Na, chunk):
        e = min(Na, s + chunk)
        da = a_xyz[s:e]                       # [S,3]
        # compute pairwise distances chunk-vs-all, but in sub-batches to keep mem stable
        d = torch.cdist(da, b_xyz, p=2)       # [S,Nb]
        mins.append(d.min(dim=1).values)      # [S]
    return torch.cat(mins, dim=0)             # [Na]

class SemanticKITTIMOS(Dataset):
    def __init__(
        self,
        root: str,
        sequences: List[str],
        semantic_kitti_yaml: str | None,
        max_points: int = 8192,
        use_prev: bool = True,
        n_prev: int = 1,
        max_prev_points: int = 8192,
        aug: Dict | None = None,
        split: str = "train",
        # --- speed knobs ---
        frame_stride: int = 1,           # NEW: skip frames for speed
        prev_voxel_size: float = 0.5,    # NEW: voxel downsample prev before dmin
        dmin_chunk: int = 4096,          # NEW: chunk size for cdist
    ):
        super().__init__()
        self.root = root
        self.sequences = sequences
        self.max_points = max_points
        self.use_prev = use_prev
        self.n_prev = int(n_prev)
        self.max_prev_points = max_prev_points
        self.aug = aug or {}
        self.split = split
        self.frame_stride = max(1, int(frame_stride))
        self.prev_voxel_size = float(prev_voxel_size or 0.0)
        self.dmin_chunk = int(dmin_chunk)

        self.moving_ids = load_moving_ids_from_yaml(semantic_kitti_yaml)

        self.items = []
        for seq in sequences:
            seq_dir = os.path.join(root, "sequences", seq)
            velo_dir = os.path.join(seq_dir, "velodyne")
            label_dir = os.path.join(seq_dir, "labels")
            pose_path = os.path.join(seq_dir, "poses.txt")
            calib_path = os.path.join(seq_dir, "calib.txt")
            assert os.path.isdir(velo_dir), f"Missing {velo_dir}"
            assert os.path.isdir(label_dir), f"Missing {label_dir}"
            assert os.path.isfile(pose_path), f"Missing {pose_path}"
            assert os.path.isfile(calib_path), f"Missing {calib_path}"
            frames = sorted([
                os.path.splitext(os.path.basename(p))[0]
                for p in glob.glob(os.path.join(velo_dir, "*.bin"))
            ])
            # need at least n_prev history
            start = self.n_prev if self.use_prev else 0
            frames = frames[start::self.frame_stride]  # NEW: stride
            for f in frames:
                self.items.append((seq, f))

        # caches
        self._poses_lidar_cache: Dict[str, np.ndarray] = {}  # [T,4,4] in LiDAR frame

    def _load_lidar_poses(self, seq: str) -> np.ndarray:
        if seq not in self._poses_lidar_cache:
            seq_dir = os.path.join(self.root, "sequences", seq)
            pose_cam = read_poses_txt(os.path.join(seq_dir, "poses.txt"))         # T_w_cam0[t]
            Tr_velo_to_cam = read_calib_velo_to_cam(os.path.join(seq_dir, "calib.txt"))  # T_cam0<-velo
            T_w_velo = pose_cam @ Tr_velo_to_cam
            self._poses_lidar_cache[seq] = T_w_velo.astype(np.float64)
        return self._poses_lidar_cache[seq]

    def __len__(self): return len(self.items)

    def _read_frame(self, seq: str, frame: str):
        base = os.path.join(self.root, "sequences", seq)
        pts = read_bin_points(os.path.join(base, "velodyne", f"{frame}.bin"))
        labels = read_labels(os.path.join(base, "labels", f"{frame}.label"))
        return pts, labels

    def _augment(self, pts: np.ndarray) -> np.ndarray:
        xyz, feat = pts[:, :3], pts[:, 3:]
        lim = float(self.aug.get("rot_limit_deg", 0.0))
        if lim > 0:
            import math, random
            a = np.deg2rad(random.uniform(-lim, lim))
            ca, sa = math.cos(a), math.sin(a)
            R = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], dtype=np.float32)
            xyz = xyz @ R.T
        if float(self.aug.get("flip_prob", 0.0)) > 0:
            import random
            if random.random() < float(self.aug["flip_prob"]): xyz[:,0] *= -1.0
        std = float(self.aug.get("jitter_std", 0.0))
        if std > 0: xyz = xyz + np.random.normal(scale=std, size=xyz.shape).astype(np.float32)
        return np.concatenate([xyz, feat], axis=1)

    def __getitem__(self, idx: int):
        seq, f = self.items[idx]
        f_id = int(f)
        pts_t, lab_t = self._read_frame(seq, f)

        # sample current
        if self.max_points is not None:
            sel = random_sample_indices(len(pts_t), self.max_points)
            pts_t, lab_t = pts_t[sel], lab_t[sel]

        moving = np.isin(lab_t, list(self.moving_ids)).astype(np.float32)

        # features: [x,y,z,intensity, dmin_norm, range]
        feat_extra = None
        if self.use_prev:
            T_w_velo = self._load_lidar_poses(seq)  # [T,4,4]
            # accumulate min over n_prev frames
            a = torch.from_numpy(pts_t[:, :3]).float().cuda() if torch.cuda.is_available() else torch.from_numpy(pts_t[:, :3]).float()
            dmin_all = None
            for h in range(1, self.n_prev + 1):
                pts_tm1, _ = self._read_frame(seq, f"{f_id-h:06d}")
                T_tm1 = T_w_velo[f_id - h]
                T_t   = T_w_velo[f_id]
                T_rel = se3_inv(T_t) @ T_tm1  # bring tm1 -> t
                prev_in_t = transform_points(T_rel, pts_tm1[:, :3])
                # voxel downsample previous
                prev_in_t = _voxel_downsample_np(prev_in_t, self.prev_voxel_size)
                # subsample previous
                if self.max_prev_points is not None and prev_in_t.shape[0] > self.max_prev_points:
                    sel2 = random_sample_indices(len(prev_in_t), self.max_prev_points)
                    prev_in_t = prev_in_t[sel2]
                b = torch.from_numpy(prev_in_t).float().to(a.device)
                dmin_h = _dmin_chunked(a, b, chunk=self.dmin_chunk)  # [Na]
                dmin_all = dmin_h if dmin_all is None else torch.minimum(dmin_all, dmin_h)

            # clamp + log1p + range-normalize
            dmin = torch.clamp(dmin_all, max=2.0)
            rng = torch.from_numpy(np.linalg.norm(pts_t[:, :3], axis=1)).float().to(dmin.device)
            dmin_norm = torch.log1p(dmin / (rng + 1e-3))
            feat_extra = dmin_norm.unsqueeze(1).cpu().numpy().astype(np.float32)

        rng_col = np.linalg.norm(pts_t[:, :3], axis=1, keepdims=True).astype(np.float32)
        feats = np.concatenate([pts_t, feat_extra if feat_extra is not None else np.zeros_like(rng_col), rng_col], axis=1)
        if self.split == "train":
            feats = self._augment(feats)

        return {
            "points": feats.astype(np.float32),  # [N, 6] if use_prev else [N,5]
            "label": moving.astype(np.float32),  # [N]
            "seq": seq,
            "frame": f,
        }

def mos_collate(batch: list[dict]) -> dict:
    import numpy as np, torch
    pts_list, y_list, bidx_list = [], [], []
    for i, b in enumerate(batch):
        P, Y = b["points"], b["label"]
        pts_list.append(P); y_list.append(Y)
        bidx_list.append(np.full((len(P),), i, dtype=np.int64))
    pts = torch.from_numpy(np.concatenate(pts_list, axis=0))
    y   = torch.from_numpy(np.concatenate(y_list,  axis=0))
    bidx= torch.from_numpy(np.concatenate(bidx_list,axis=0))
    return {"points": pts, "label": y, "batch_idx": bidx}
