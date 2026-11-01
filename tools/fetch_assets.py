# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Fetch the stock UFACTORY xArm 850 USD and a grid environment from the Isaac Sim assets into usd/,
then bake the joint drive maxForce the behaviours need. Run with Isaac Sim's Python:

  python.bat tools/fetch_assets.py [--robot Isaac/Robots/UFactory/uf850] [--env Isaac/Environments/Grid/default_environment.usd]

The assets stay NVIDIA's (their license applies); this repo never ships them."""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--robot", default="Isaac/Robots/UFactory/uf850", help="robot folder under the assets root")
ap.add_argument("--env", default="Isaac/Environments/Grid/default_environment.usd", help="environment USD under the assets root")
ap.add_argument("--assets-root", default=None, help="override the assets root (default: Isaac's configured root)")
ap.add_argument("--max-force", type=float, default=1e4, help="joint drive maxForce to bake (Nm)")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
try:
    import omni.client  # noqa: E402
    from isaacsim.storage.native import get_assets_root_path  # noqa: E402

    root = args.assets_root or get_assets_root_path()
    if not root:
        print("[fetch] no assets root: pass --assets-root or configure Isaac's asset location", flush=True)
        sys.exit(2)
    root = root.rstrip("/")
    robot_src = f"{root}/{args.robot.strip('/')}"
    robot_dst = os.path.join(ROOT, "usd", "uf850").replace("\\", "/")
    env_src = f"{root}/{args.env.strip('/')}"
    env_dst = os.path.join(ROOT, "usd", "environment", os.path.basename(args.env)).replace("\\", "/")
    os.makedirs(os.path.dirname(env_dst), exist_ok=True)
    for src, dst in ((robot_src, robot_dst), (env_src, env_dst)):
        print(f"[fetch] {src} -> {dst}", flush=True)
        res = omni.client.copy(src, dst, behavior=omni.client.CopyBehavior.OVERWRITE)
        if res != omni.client.Result.OK:
            print(f"[fetch] FAILED ({res}); check the path with Isaac's asset browser and pass --robot/--env", flush=True)
            sys.exit(1)
    usd = os.path.join(robot_dst, "uf850.usd")
    if not os.path.exists(usd):
        print(f"[fetch] copied, but {usd} is missing: the robot folder layout differs; point app.py at the right file", flush=True)
        sys.exit(1)
    sys.path.insert(0, HERE)
    from bake_usd_forces import bake  # noqa: E402

    bake(usd, args.max_force)
    print("[fetch] done", flush=True)
finally:
    app.close()
