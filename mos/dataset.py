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


def _range_from_xyz(xyz: np.ndarray) -> np.ndarray:
    return np.linalg.norm(xyz, axis=1).astype(np.float32)


class SemanticKITTIMOS(Dataset):
    """
    Returns dict:
      points: float32 [N, D]
        if use_prev: D=6 -> [x,y,z,intensity,dmin_norm,range]
        else:        D=5 -> [x,y,z,intensity,range]
      label:  float32 [N]  (binary moving{0,1})
      seq:    str
      frame:  str (zero-padded)
    """
    def __init__(
        self,
        root: str,
        sequences: List[str],
        semantic_kitti_yaml: str | None,
        max_points: int = 8192,
        use_prev: bool = True,
        n_prev: int = 1,                  # <— NEW
        max_prev_points: int = 8192,
        aug: Dict | None = None,
        split: str = "train",
    ):
        super().__init__()
        assert n_prev >= 0
        self.root = root
        self.sequences = sequences
        self.max_points = max_points
        self.use_prev = use_prev
        self.n_prev = n_prev if use_prev else 0
        self.max_prev_points = max_prev_points
        self.aug = aug or {}
        self.split = split

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
            # if using previous frames, we need to start at index n_prev
            if self.n_prev > 0:
                frames = frames[self.n_prev:]
            for f in frames:
                self.items.append((seq, f))

        # caches
        self._poses_lidar_cache: Dict[str, np.ndarray] = {}  # [T,4,4] in LiDAR frame

    def _load_lidar_poses(self, seq: str) -> np.ndarray:
        if seq not in self._poses_lidar_cache:
            seq_dir = os.path.join(self.root, "sequences", seq)
            pose_cam = read_poses_txt(os.path.join(seq_dir, "poses.txt"))               # T_w_cam0[t]
            Tr_velo_to_cam = read_calib_velo_to_cam(os.path.join(seq_dir, "calib.txt")) # T_cam0←velo
            # Convert to LiDAR poses: T_w_velo[t] = T_w_cam0[t] @ T_cam0←velo
            T_w_velo = pose_cam @ Tr_velo_to_cam
            self._poses_lidar_cache[seq] = T_w_velo.astype(np.float64)
        return self._poses_lidar_cache[seq]

    def __len__(self): return len(self.items)

    def _read_frame(self, seq: str, frame: str):
        base = os.path.join(self.root, "sequences", seq)
        pts = read_bin_points(os.path.join(base, "velodyne", f"{frame}.bin"))  # [N,4] xyz + intensity
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
        pts_t, lab_t = self._read_frame(seq, f)     # pts_t: [N,4]

        # subsample current frame
        if self.max_points is not None:
            sel = random_sample_indices(len(pts_t), self.max_points)
            pts_t, lab_t = pts_t[sel], lab_t[sel]

        xyz_t = pts_t[:, :3].astype(np.float32)
        intensity = pts_t[:, 3:4].astype(np.float32)
        rng = _range_from_xyz(xyz_t).reshape(-1,1)  # [N,1]

        moving = np.isin(lab_t, list(self.moving_ids)).astype(np.float32)

        # build temporal cue across n_prev frames: dmin = min_k dmin_k
        dmin_norm = None
        if self.n_prev > 0:
            T_w_velo = self._load_lidar_poses(seq)  # [T,4,4]
            xyz_prev_list = []
            for k in range(1, self.n_prev + 1):
                f_prev = f_id - k
                pts_tm, _ = self._read_frame(seq, f"{f_prev:06d}")
                # transform t-k points into t frame: x_t = (T_t^-1 T_tm) x_tm
                T_tm = T_w_velo[f_prev]
                T_t  = T_w_velo[f_id]
                T_rel = se3_inv(T_t) @ T_tm
                xyz_tm_in_t = transform_points(T_rel, pts_tm[:, :3])
                # subsample
                if self.max_prev_points is not None and len(xyz_tm_in_t) > self.max_prev_points:
                    sel2 = random_sample_indices(len(xyz_tm_in_t), self.max_prev_points)
                    xyz_tm_in_t = xyz_tm_in_t[sel2]
                xyz_prev_list.append(torch.from_numpy(xyz_tm_in_t).float())

            a = torch.from_numpy(xyz_t).float()  # [Na,3]
            # compute min distance to any previous cloud (across k)
            dmins = []
            for b in xyz_prev_list:
                d = torch.cdist(a, b, p=2).min(dim=1).values
                dmins.append(d)
            dmin = torch.stack(dmins, dim=1).min(dim=1).values  # [Na]

            # cap + log for stability, then normalize by range
            dmin = torch.clamp(dmin, max=2.0)
            dmin = torch.log1p(dmin)  # ~[0, log1p(2)]
            # normalize by (range + eps) to reduce near/far bias
            eps = 0.5
            dmin_norm = (dmin.numpy().astype(np.float32) / (rng.reshape(-1) + eps)).reshape(-1,1).astype(np.float32)

        # features
        if dmin_norm is not None:
            feats = np.concatenate([xyz_t, intensity, dmin_norm, rng], axis=1)  # [N,6]
        else:
            feats = np.concatenate([xyz_t, intensity, rng], axis=1)             # [N,5]

        if self.split == "train":
            feats = self._augment(feats)

        return {
            "points": feats.astype(np.float32),
            "label": moving.astype(np.float32),
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
