# mos/upgrade_ckpt_inchannels.py
from __future__ import annotations
import os, argparse, torch

def pad_first_in(state, new_in: int, candidate_names=("mlp1.0.weight",)):
    sd = state["model"] if "model" in state else state
    changed = False
    for name, W in list(sd.items()):
        if not isinstance(W, torch.Tensor):
            continue
        if W.ndim == 2 and name.endswith(".weight"):
            out, inn = W.shape
            # heuristic: upgrade likely first linear layer
            if name in candidate_names or inn < new_in:
                W_new = torch.zeros((out, new_in), dtype=W.dtype)
                W_new[:, :min(inn, new_in)] = W[:, :min(inn, new_in)]
                sd[name] = W_new
                print(f"[upgrade] {name}: {tuple(W.shape)} -> {tuple(W_new.shape)}")
                changed = True
                break
    if not changed:
        raise RuntimeError("Did not find a suitable [out,in] weight to upgrade.")
    state["model"] = sd
    return state

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--new_in", type=int, required=True)  # e.g., 6
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location="cpu")
    ckpt_up = pad_first_in(ckpt, args.new_in)
    torch.save(ckpt_up, args.dst)
    print(f"[done] wrote {args.dst}")

if __name__ == "__main__":
    main()
