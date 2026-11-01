# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Bake the joint drive maxForce into the robot USD. The stock asset ships low wrist torques that make
the PD drives lag RMPflow; the behaviours expect the drives never to be torque-limited (speed and
smoothness live in the RMPflow layer). Run with Isaac Sim's Python:

  python.bat tools/bake_usd_forces.py [--usd usd/uf850/uf850.usd] [--max-force 1e4]
"""
from __future__ import annotations

import argparse
import os

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def bake(usd_path: str, max_force: float = 1e4) -> dict:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    done = {}
    for prim in stage.Traverse():
        if prim.GetName() in JOINT_NAMES:
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            before = drive.GetMaxForceAttr().Get() if drive.GetMaxForceAttr() else None
            drive.CreateMaxForceAttr().Set(float(max_force))
            done[prim.GetName()] = (before, float(max_force))
    stage.GetRootLayer().Save()
    for name, (before, after) in done.items():
        print(f"[bake] {name}: maxForce {before} -> {after}", flush=True)
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", default=os.path.join(os.path.dirname(__file__), "..", "usd", "uf850", "uf850.usd"))
    ap.add_argument("--max-force", type=float, default=1e4)
    a = ap.parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        bake(a.usd, a.max_force)
    finally:
        app.close()
