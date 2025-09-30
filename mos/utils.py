from __future__ import annotations
import numpy as np
import torch

def read_bin_points(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

def read_labels(path: str) -> np.ndarray:
    labels = np.fromfile(path, dtype=np.uint32)
    return (labels & 0xFFFF).astype(np.int32)

def read_poses_txt(path: str) -> np.ndarray:
    mats = []
    with open(path, "r") as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            assert len(vals) == 12
            M = np.eye(4, dtype=np.float64)
            M[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
            mats.append(M)
    return np.stack(mats, axis=0)  # camera-0 poses T_w_cam0

def read_calib_velo_to_cam(path: str) -> np.ndarray:
    """
    Parse calib.txt to get Tr (velo->cam) as 4x4.
    Lines look like: 'Tr: r11 r12 ... r34'
    """
    Tr = None
    with open(path, "r") as f:
        for ln in f:
            if ln.startswith("Tr:"):
                vals = [float(x) for x in ln.split()[1:]]
                assert len(vals) == 12
                M = np.eye(4, dtype=np.float64)
                M[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
                Tr = M
                break
    if Tr is None:
        # Some files use 'Tr_velo_to_cam:'
        with open(path, "r") as f:
            for ln in f:
                if ln.startswith("Tr_velo_to_cam:"):
                    vals = [float(x) for x in ln.split()[1:]]
                    assert len(vals) == 12
                    M = np.eye(4, dtype=np.float64)
                    M[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
                    Tr = M
                    break
    if Tr is None:
        raise FileNotFoundError(f"No Tr in calib file {path}")
    return Tr  # T_cam0 <- T_velo

def se3_inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv

def transform_points(T: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return (pts_xyz @ R.T) + t

def random_sample_indices(n_total: int, n_keep: int) -> np.ndarray:
    if n_total <= n_keep:
        return np.arange(n_total, dtype=np.int64)
    return np.random.choice(n_total, n_keep, replace=False)

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
