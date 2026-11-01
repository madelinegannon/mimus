# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""App configuration: built-in defaults for a base on the floor at the world origin, overlaid by
`app.json` and then `app.local.json` in the config dir. Poses saved from the panel go to `app.json`."""
from __future__ import annotations

import copy
import json
import os
import shutil
import time

FILES = ("app.json", "app.local.json")
DEFAULTS = {
    "mount_z": 0.0,                       # base flange height above the floor (a pedestal); 0 = on the floor
    "base_xy": [0.0, 0.0],                # base position in the world frame the tracker uses
    "base_yaw_deg": 0.0,                  # base yaw about +z; the idle sweep is centred on it
    "home_q_rad": [0.0, 1.012291, -0.785398, 0.0, -0.226893, 0.0],   # arm folded over the base, tool level toward +x
    "sleep_q_rad": [0.0, 1.012291, -0.785398, 0.0, -1.797689, 0.0],  # same arm, tool straight down
    "unwind_rate_rad_s": 0.8,             # HOME / SLEEP: largest joint delta / rate = move time (peak 1.5x)
    "unwind_min_s": 0.5,                  # shortest move time, so small corrections still ease
    "startup_rate_rad_s": 0.3,            # slower than an unwind: at startup the arm's pose is whatever it is
    "speed_scale": 0.2,                   # the bridge's starting cap; it cannot start above 0.2, operators raise it live
    "speed_slew_per_s": 0.33,             # raising the cap ramps in over ~3 s; cuts apply at once
    "anchor_z_m": 1.1,                    # FollowTargetSensor rest anchor [base_xy, Z], a WORLD height
    "reach_max_m": 0.85,                  # every target stays within this of the shoulder: 0.85 = the arm's full reach
    "tracker_params": {},                 # starting values for the Attention sliders, e.g. {"near_pref": 0.5}
}
_LIST6 = ("home_q_rad", "sleep_q_rad")


def load_config(config_dir: str) -> dict:
    """DEFAULTS <- app.json <- app.local.json. Unknown keys are reported and ignored; a bad file is
    reported and skipped. `_source` names what was read."""
    cfg = copy.deepcopy(DEFAULTS)
    sources = []
    for name in FILES:
        path = os.path.join(config_dir, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError) as e:
            print(f"[config] WARN: {path}: {e}; skipped", flush=True)
            continue
        if not isinstance(d, dict):
            print(f"[config] WARN: {path}: not an object; skipped", flush=True)
            continue
        for k, v in d.items():
            if k not in DEFAULTS:
                print(f"[config] WARN: {path}: unknown key {k!r} ignored", flush=True)
                continue
            if k in _LIST6:
                if not (isinstance(v, list) and len(v) == 6):
                    print(f"[config] WARN: {path}: {k} needs six joint angles; kept", flush=True)
                    continue
                cfg[k] = [float(x) for x in v]
            elif k == "base_xy":
                cfg[k] = [float(v[0]), float(v[1])]
            elif k == "tracker_params":
                cfg[k] = {str(a): float(b) for a, b in dict(v).items()}
            else:
                cfg[k] = float(v)
        sources.append(name)
    cfg["_source"] = " + ".join(sources) if sources else "built-in"
    return cfg


def save_param(config_dir: str, name: str, value, backup: bool = False) -> str:
    """Write one key into app.json (created if missing). `backup` keeps a timestamped copy first."""
    if name not in DEFAULTS:
        raise KeyError(name)
    path = os.path.join(config_dir, FILES[0])
    d = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            d = {}
        if backup:
            shutil.copy(path, path.replace(".json", f"_{time.strftime('%Y%m%d_%H%M%S')}.json"))
    d[name] = value
    os.makedirs(config_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    print(f"[config] saved {name} -> {path}", flush=True)
    return path
