# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""App config: defaults, the app.json + app.local.json overlay, saving a parameter, the shipped example."""
import json
import os
import tempfile

import _common as T
from _common import check

from mimus.config import DEFAULTS, FILES, load_config, save_param

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("== defaults ==")
with tempfile.TemporaryDirectory() as d:
    cfg = load_config(d)
    check("built-in when no file", cfg["_source"] == "built-in", cfg["_source"])
    check("base on the floor at the origin", cfg["mount_z"] == 0.0 and cfg["base_xy"] == [0.0, 0.0] and cfg["base_yaw_deg"] == 0.0)
    check("speed cap starts at 0.2", cfg["speed_scale"] == 0.2)
    check("six-joint poses", len(cfg["home_q_rad"]) == 6 and len(cfg["sleep_q_rad"]) == 6)
    check("every default key present", set(cfg) == set(DEFAULTS) | {"_source"})

print("== overlay ==")
with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(d, FILES[0]), "w", encoding="utf-8") as f:
        json.dump({"speed_scale": 0.15, "mount_z": 1.0, "home_q_rad": [0, 1, -1, 0, 0, 0], "bogus": 1,
                   "sleep_q_rad": [1, 2, 3], "tracker_params": {"near_pref": 0.7}}, f)
    with open(os.path.join(d, FILES[1]), "w", encoding="utf-8") as f:
        json.dump({"mount_z": 1.2}, f)
    cfg = load_config(d)
    check("source names both files", cfg["_source"] == "app.json + app.local.json", cfg["_source"])
    check("local overlay wins", cfg["mount_z"] == 1.2, str(cfg["mount_z"]))
    check("app.json value taken", cfg["speed_scale"] == 0.15)
    check("pose coerced to floats", cfg["home_q_rad"] == [0.0, 1.0, -1.0, 0.0, 0.0, 0.0])
    check("a short pose is refused, default kept", cfg["sleep_q_rad"] == DEFAULTS["sleep_q_rad"])
    check("unknown key ignored", "bogus" not in cfg)
    check("tracker params coerced", cfg["tracker_params"] == {"near_pref": 0.7})
    with open(os.path.join(d, FILES[1]), "w", encoding="utf-8") as f:
        f.write("{not json")
    cfg = load_config(d)
    check("a broken overlay is skipped", cfg["mount_z"] == 1.0 and cfg["_source"] == "app.json", cfg["_source"])

print("== save_param ==")
with tempfile.TemporaryDirectory() as d:
    path = save_param(d, "home_q_rad", [0.1] * 6)
    check("creates app.json", os.path.basename(path) == FILES[0] and os.path.exists(path))
    check("round-trips", load_config(d)["home_q_rad"] == [0.1] * 6)
    save_param(d, "anchor_z_m", 1.3, backup=True)
    check("keeps the earlier key", load_config(d)["home_q_rad"] == [0.1] * 6)
    check("backup written", any(n.startswith("app_") for n in os.listdir(d)), str(os.listdir(d)))
    try:
        save_param(d, "bogus", 1)
        check("unknown key refused", False)
    except KeyError:
        check("unknown key refused", True)

print("== shipped example ==")
with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(ROOT, "config", "app.example.json"), "r", encoding="utf-8") as f:
        example = json.load(f)
    with open(os.path.join(d, FILES[0]), "w", encoding="utf-8") as f:
        json.dump(example, f)
    cfg = load_config(d)
    check("example equals the built-in defaults", all(cfg[k] == DEFAULTS[k] for k in DEFAULTS), str({k: (cfg[k], DEFAULTS[k]) for k in DEFAULTS if cfg[k] != DEFAULTS[k]}))

T.finish("test_config")
