# mos/debug_check_labels.py
import os, glob, numpy as np, sys, json

ROOT = r"D:\Subhransu workspace\Dataset\my_kitti_dataset\dataset"  # <-- set to DATASET_ROOT
SEQ  = "08"

MOVING_IDS = {252,253,254,255,256,257,258,259,260}

def read_labels(p):
    lab = np.fromfile(p, dtype=np.uint32)
    return (lab & 0xFFFF).astype(np.int32)

lab_dir = os.path.join(ROOT, "sequences", SEQ, "labels")
paths = sorted(glob.glob(os.path.join(lab_dir, "*.label")))[:50]  # sample first 50
uniq = set()
moving_pct_list = []
for p in paths:
    L = read_labels(p)
    uniq |= set(np.unique(L).tolist())
    moving = np.isin(L, list(MOVING_IDS)).mean()
    moving_pct_list.append(float(moving))
print("unique_ids_sample:", sorted(list(uniq))[:40], ("... total:", len(uniq)))
print("mean_moving_fraction_over_50:", float(np.mean(moving_pct_list)))
