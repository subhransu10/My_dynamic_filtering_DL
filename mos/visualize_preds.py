# mos/visualize_preds.py
from __future__ import annotations
import os, argparse, time
import numpy as np
import torch

from mos.train import load_cfg, make_datasets
from mos.model import build model

try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False


# ------------------------ coloring helpers ------------------------

def colors_from_prob(probs: np.ndarray) -> np.ndarray:
    """Blue→Red ramp. Red = high prob."""
    p = np.clip(probs.reshape(-1, 1), 0.0, 1.0).astype(np.float32)
    # R = p, G = small for contrast, B = (1-p)
    return np.concatenate([p, 0.2 * (1.0 - p), (1.0 - p)], axis=1)

def colors_pred_only(xyz: np.ndarray, pred_moving: np.ndarray) -> np.ndarray:
    col = np.full((xyz.shape[0], 3), 0.55, dtype=np.float32)
    col[pred_moving] = np.array([1.0, 0.15, 0.15], dtype=np.float32)  # red
    return col

def colors_with_gt(xyz: np.ndarray, gt_moving: np.ndarray, pred_moving: np.ndarray) -> np.ndarray:
    # TN = gray, TP = red, FP = yellow, FN = cyan
    col = np.full((xyz.shape[0], 3), [0.35, 0.35, 0.35], dtype=np.float32)
    tp =  pred_moving &  gt_moving
    fp =  pred_moving & ~gt_moving
    fn = ~pred_moving &  gt_moving
    col[tp] = [1.00, 0.15, 0.15]
    col[fp] = [1.00, 0.90, 0.20]
    col[fn] = [0.10, 0.85, 1.00]
    return col


# ------------------------ io helpers ------------------------

def save_ply(path: str, xyz: np.ndarray, rgb01: np.ndarray):
    rgb255 = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb255):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r:d} {g:d} {b:d}\n")


# ------------------------ dataset helpers ------------------------

def collect_seq_items(ds_val, seq: str, max_frames: int | None):
    idxs = [i for i, (s, _) in enumerate(ds_val.items) if s == seq]
    if max_frames is not None:
        idxs = idxs[:max_frames]
    return [ds_val[i] for i in idxs]

def apply_crop(xyz: np.ndarray, keep_front: bool, rmax: float | None) -> np.ndarray:
    m = np.ones((xyz.shape[0],), dtype=bool)
    if keep_front:
        m &= (xyz[:, 0] > 0.0)  # keep points in front of the car
    if rmax is not None:
        m &= (np.linalg.norm(xyz[:, :2], axis=1) <= float(rmax))
    return m


# ------------------------ viewer ------------------------

def run_o3d_viewer(
    items,
    model,
    device,
    thresh: float,
    show_gt: bool,
    do_cluster: bool,
    eps: float,
    min_pts: int,
    out_png_dir: str | None,
    prob_mode: bool,
    point_size: float,
    crop_front: bool,
    rmax: float | None,
    ckpt_path: str | None = None,
):
    assert HAS_O3D, "Open3D not available"

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"MOS viz (th={thresh:.2f})", width=1280, height=800)
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0], dtype=np.float32)
    opt.point_size = float(point_size)
    geom = None

    state = {"i": 0, "show_gt": show_gt, "do_cluster": do_cluster,
             "eps": eps, "min_pts": min_pts, "thresh": thresh,
             "prob_mode": prob_mode}

    def fit_camera_to(xyz: np.ndarray):
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(xyz))
        center = bbox.get_center()
        extent = np.linalg.norm(bbox.get_extent()) + 1e-6
        cam = vis.get_view_control()
        front = np.array([0.0, -1.0, 0.0])
        up    = np.array([0.0,  0.0, 1.0])
        cam.set_lookat(center); cam.set_front(front); cam.set_up(up)
        cam.set_zoom(0.35 if extent > 40 else 0.50)

    def cluster_mask(xyz: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not state["do_cluster"]:
            return mask
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return mask
        sub = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[idx]))
        labels = np.array(sub.cluster_dbscan(eps=state["eps"], min_points=state["min_pts"], print_progress=False))
        keep = labels >= 0
        out = np.zeros_like(mask)
        out[idx[keep]] = True
        return out

    @torch.no_grad()
    def forward_probs(points_np: np.ndarray) -> np.ndarray:
        pts = torch.from_numpy(points_np).to(device)
        pts = torch.nan_to_num(pts, nan=0.0, posinf=1e6, neginf=-1e6)
        bidx = torch.zeros((len(pts),), dtype=torch.long, device=device)
        # our model returns logits by default
        logits = model(pts, bidx)
        return torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

    def render(i: int):
        nonlocal geom
        item = items[i]
        full_xyz = item["points"][:, :3].astype(np.float32)
        crop_mask = apply_crop(full_xyz, crop_front, rmax)
        xyz = full_xyz[crop_mask]

        probs_full = forward_probs(item["points"])
        probs = probs_full[crop_mask]

        if state["prob_mode"]:
            col = colors_from_prob(probs)
        else:
            pred = (probs >= state["thresh"])
            if state["do_cluster"]:
                pred = cluster_mask(xyz, pred)
            if state["show_gt"]:
                gt = (item["label"] >= 0.5)[crop_mask]
                col = colors_with_gt(xyz, gt, pred)
                # per-frame stats (cropped)
                tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum())
                fn = int((~pred & gt).sum())
                prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                print(f"   frame stats (cropped): TP={tp} FP={fp} FN={fn} | P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
            else:
                col = colors_pred_only(xyz, pred)

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz)
        pc.colors = o3d.utility.Vector3dVector(col)

        if geom is None:
            geom = pc
            vis.add_geometry(geom)
            fit_camera_to(xyz)
        else:
            geom.points = pc.points
            geom.colors = pc.colors
            vis.update_geometry(geom)

        vis.update_renderer()
        title = f"{item['seq']}_{item['frame']} (N={len(xyz)})"
        print(f"[view] {title} | thresh={state['thresh']:.2f} | GT={state['show_gt']} | cluster={state['do_cluster']} ({state['eps']},{state['min_pts']}) | prob_mode={state['prob_mode']}")
        return title

    # ---------- hot reload ----------
    def reload_ckpt(_):
        if not ckpt_path:
            print("[reload] No --ckpt provided.")
            return True
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            sd = ckpt.get("model", ckpt)
            model.load_state_dict(sd, strict=False)
            model.eval()
            print(f"[reload] reloaded {ckpt_path} (ep={ckpt.get('ep','?')})")
            render(state["i"])
        except Exception as e:
            print(f"[reload] failed: {e}")
        return True

    # ---------- key bindings ----------
    def go_next(_): state["i"] = (state["i"] + 1) % len(items); render(state["i"]); return True
    def go_prev(_): state["i"] = (state["i"] - 1) % len(items); render(state["i"]); return True
    def toggle_gt(_): state["show_gt"] = not state["show_gt"]; render(state["i"]); return True
    def toggle_cluster(_): state["do_cluster"] = not state["do_cluster"]; render(state["i"]); return True
    def inc_thresh(_): state["thresh"] = min(0.99, state["thresh"] + 0.05); render(state["i"]); return True
    def dec_thresh(_): state["thresh"] = max(0.01, state["thresh"] - 0.05); render(state["i"]); return True
    def toggle_prob(_): state["prob_mode"] = not state["prob_mode"]; render(state["i"]); return True
    def save_png(_):
        if out_png_dir is None:
            print("[save] set --save_png_dir to enable screenshots"); return True
        os.makedirs(out_png_dir, exist_ok=True)
        path = os.path.join(out_png_dir, f"{items[state['i']]['seq']}_{items[state['i']]['frame']}.png")
        vis.capture_screen_image(path, do_render=True); print(f"[save] {path}"); return True

    vis.register_key_callback(ord("N"), go_next)
    vis.register_key_callback(ord("B"), go_prev)
    vis.register_key_callback(ord("G"), toggle_gt)
    vis.register_key_callback(ord("C"), toggle_cluster)
    vis.register_key_callback(ord("]"), inc_thresh)
    vis.register_key_callback(ord("["), dec_thresh)
    vis.register_key_callback(ord("P"), toggle_prob)
    vis.register_key_callback(ord("S"), save_png)
    vis.register_key_callback(ord("R"), reload_ckpt)

    render(0)
    print("Keys: N=next, B=prev, G=toggle GT, C=toggle clusters, [ / ] threshold, P=prob heatmap, S=screenshot, R=reload ckpt, Q/ESC=quit")
    vis.run()
    vis.destroy_window()


# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob", action="store_true", help="Color by probability instead of binary mask")
    ap.add_argument("--config", type=str, default=os.path.join("mos", "config.yaml"))
    ap.add_argument("--ckpt", type=str, default=os.path.join("checkpoints", "mos_best.pt"))
    ap.add_argument("--seq", type=str, default=None, help="Sequence to visualize (e.g., 08)")
    ap.add_argument("--frames", type=int, default=50, help="Visualize first N frames (None=all)")
    ap.add_argument("--thresh", type=float, default=None, help="Override threshold (default: cfg.eval_threshold)")
    ap.add_argument("--show_gt", action="store_true", help="Overlay ground truth (TP=red, FP=yellow, FN=cyan, TN=gray)")
    ap.add_argument("--cluster", action="store_true", help="Drop tiny FP clusters (DBSCAN)")
    ap.add_argument("--eps", type=float, default=0.5, help="DBSCAN radius (m)")
    ap.add_argument("--min_pts", type=int, default=20, help="Min points per cluster")
    ap.add_argument("--save_png_dir", type=str, default=None, help="Allow 'S' to save screenshots here")
    ap.add_argument("--save_ply_dir", type=str, default=None, help="If not using Open3D, write PLYs here")
    ap.add_argument("--point_size", type=float, default=2.5, help="Open3D point size")
    ap.add_argument("--crop_front", action="store_true", help="Keep only points with x>0 (front FOV)")
    ap.add_argument("--rmax", type=float, default=None, help="Keep only points within XY radius (meters)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")

    _, ds_val = make_datasets(cfg)
    seq = args.seq or cfg["SEQUENCE_VAL"]
    items = collect_seq_items(ds_val, seq, args.frames)
    if not items:
        print(f"[viz] no frames for sequence {seq}"); return

    # build the same model as training
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model", ckpt)

    # forgiving load for shape-safe init (in case you change channels/ranges)
    msd = model.state_dict()
    loadable = {k: v for k, v in state.items() if k in msd and msd[k].shape == v.shape}
    miss = len(msd) - len(loadable)
    model.load_state_dict({**msd, **loadable}, strict=False)
    if miss:
        print(f"[viz] forgiving load: used {len(loadable)} / {len(msd)} tensors (missing {miss})")
    model.eval()

    thresh = float(args.thresh if args.thresh is not None else cfg.get("eval_threshold", 0.5))
    print(f"[viz] seq={seq} frames={len(items)} thresh={thresh:.2f} HAS_O3D={HAS_O3D}")

   