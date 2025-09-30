# mos/labels.py
from __future__ import annotations
from typing import Set, Optional

_DEFAULT_MOVING_IDS = {252, 253, 254, 255, 256, 257, 258, 259, 260}

def load_moving_ids_from_yaml(yaml_path: Optional[str]) -> Set[int]:
    if yaml_path is None:
        return set(_DEFAULT_MOVING_IDS)
    try:
        import yaml
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        moving = set()
        labels = cfg.get("labels", {}) or {}
        for k, v in labels.items():
            try:
                kid = int(k)
            except Exception:
                continue
            name = (v or "").lower()
            if name.startswith("moving-") or "moving" in name:
                moving.add(kid)
        if not moving:
            moving = set(_DEFAULT_MOVING_IDS)
        return moving
    except Exception:
        return set(_DEFAULT_MOVING_IDS)
