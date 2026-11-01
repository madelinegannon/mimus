# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""The collision model RMPflow runs on, checked against the arm's own meshes (numpy + yaml; no Isaac).

Spheres: rmpflow/uf850_robot_description.yaml against fixtures/uf850_surface.npz (tools/sample_collision_mesh.py).
Self-collision: the body capsule and wrist controllers in rmpflow/uf850_rmpflow_config.yaml; RMPflow has no other.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mimus import motion as M  # noqa: E402
from mimus.robot_model import RobotModel  # noqa: E402
from rt_xarm_bridge import URDF_JOINT_LIMITS_RAD as FALLBACK_JOINT_LIMITS_RAD  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

PASS = 0
FAIL = 0


def check(name, cond, info=""):
    global PASS, FAIL
    PASS, FAIL = (PASS + 1, FAIL) if cond else (PASS, FAIL + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {info if not cond else ''}")


RM = RobotModel(mount_z=0.0)                     # base frame: what the capsule is written in
METAL = np.load(Path(__file__).parent / "fixtures" / "uf850_surface.npz")
LIMITS = np.array(FALLBACK_JOINT_LIMITS_RAD)
POSES = json.loads((REPO / "config" / "app.example.json").read_text(encoding="utf-8"))
HOME_Q, SLEEP_Q = np.array(POSES["home_q_rad"]), np.array(POSES["sleep_q_rad"])
rng = np.random.default_rng(0)


def world(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def seg_dist(P, a, b):
    """Distance from each point to the segment a-b."""
    P, ab = np.atleast_2d(P), b - a
    t = np.clip((P - a) @ ab / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


print("== spheres cover the metal ==")
check("the fixture and the YAML name the same links", set(METAL.files) == set(RM.links) == set(RM.spheres),
      f"{sorted(METAL.files)} vs {sorted(RM.spheres)}")
configs = [np.zeros(6)] + [rng.uniform(LIMITS[:, 0], LIMITS[:, 1]) for _ in range(40)]
worst = {link: 0.0 for link in RM.links}
for q in configs:
    Ts = RM.link_transforms(q)
    cs, rs = RM.sphere_world(q)
    for link in RM.links:
        P = world(Ts[link], METAL[link].astype(np.float64))
        out = (np.linalg.norm(P[:, None, :] - cs[None, :, :], axis=-1) - rs[None, :]).min(axis=1)
        worst[link] = max(worst[link], float(out.max()))
for link in RM.links:
    check(f"{link}: no metal more than 12 mm outside the spheres", worst[link] <= 0.012, f"{worst[link] * 1000:.0f} mm")
qd = np.array([0.0, 0.26, -0.9, 0.0, 0.6, 0.0])
diag = world(RM.link_transforms(qd)["link3"], np.array([[0.075, 0.213, 0.0]]))
check("the elbow->wrist diagonal is air: no sphere on its midpoint", not RM.inside_mask(diag, qd, 0.0, 0.0)[0])
tube = world(RM.link_transforms(qd)["link3"], np.array([[0.15, 0.213, 0.0]]))
check("the forearm tube is covered where it actually runs", RM.inside_mask(tube, qd, 0.0, 0.0)[0])

print("== self-collision: wrist controllers against the body capsule ==")
with open(REPO / "rmpflow" / "uf850_rmpflow_config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
CAPS = [(np.array(c["pt1"], float), np.array(c["pt2"], float), float(c["radius"])) for c in CFG["body_cylinders"]]
CTRL = [(c["name"], float(c["radius"])) for c in CFG["body_collision_controllers"]]
check("every controller sits on a URDF frame", all(n in RM.links for n, _ in CTRL), str(CTRL))
check("the wrist and the flange both carry a controller", {"link5", "link6"} <= {n for n, _ in CTRL}, str(CTRL))


def clearance(q):
    """Smallest gap between a controller sphere and a body capsule (negative: RMPflow is pushing back)."""
    Ts = RM.link_transforms(q)
    return min(float(seg_dist(Ts[n][:3, 3], a, b)[0]) - R - r for n, r in CTRL for a, b, R in CAPS)


def in_capsule(P):
    return np.min([seg_dist(P, a, b) - R for a, b, R in CAPS], axis=0)


T1 = RM.link_transforms(np.zeros(6))["link1"]          # J1 only spins the housing about the capsule's own axis
housing = np.vstack([METAL["link_base"].astype(np.float64), world(T1, METAL["link1"].astype(np.float64))])
check("the capsule holds the base and the whole J1/J2 housing", in_capsule(housing).max() <= 0.0,
      f"{in_capsule(housing).max() * 1000:.0f} mm outside, metal top z={housing[:, 2].max():.3f}")

check("HOME is clear of the keep-out by 5 cm", clearance(HOME_Q) >= 0.05, f"{clearance(HOME_Q) * 1000:.0f} mm")
check("SLEEP is clear of the keep-out by 5 cm", clearance(SLEEP_Q) >= 0.05, f"{clearance(SLEEP_Q) * 1000:.0f} mm")
params = M.BUILTIN_PRESETS["Curious"]
breath = [clearance(M.sleep_joint_pose(t, SLEEP_Q, params)) for t in np.linspace(0.0, params["sleep_breath_period"], 60)]
check("the sleep breath never dips toward it", min(breath) >= clearance(SLEEP_Q) - 1e-9, f"{min(breath) * 1000:.0f} mm")
mv = M.JointMove(HOME_Q, SLEEP_Q, POSES["unwind_rate_rad_s"], POSES["unwind_min_s"], 0.0)
path = [clearance(mv.at(t)[0]) for t in np.linspace(0.0, mv.duration, 60)]
check("home -> sleep stays clear by 5 cm", min(path) >= 0.05, f"{min(path) * 1000:.0f} mm")

for j5 in (-1.57, 1.57):
    q = np.array([0.0, 0.0, 0.0, 0.0, j5, 0.0])
    check(f"folded elbow, wrist bent {j5:+.2f}: inside the keep-out", clearance(q) < 0.0, f"{clearance(q) * 1000:.0f} mm")

# the property the radii exist for: whatever the keep-out allows, the real metal is nowhere near
l2, l4 = METAL["link2"], METAL["link4"]


def some(P, n):
    return P[rng.choice(len(P), min(n, len(P)), replace=False)].astype(np.float32)


shoulder = {"link_base": some(METAL["link_base"], 200), "link1": some(METAL["link1"], 300), "link2": some(l2[l2[:, 0] < 0.15], 150)}
wrist = {"link4": some(l4[l4[:, 2] > -0.14], 200), "link5": some(METAL["link5"], 150), "link6": some(METAL["link6"], 100)}
gaps, allowed = [], []
for q in rng.uniform(LIMITS[:, 0], LIMITS[:, 1], (6000, 6)):
    c = clearance(q)
    if c > 0.15:                                       # the wrist is nowhere near the base
        continue
    Ts = RM.link_transforms(q)
    A = np.vstack([world(Ts[l], P) for l, P in shoulder.items()]).astype(np.float32)
    B = np.vstack([world(Ts[l], P) for l, P in wrist.items()]).astype(np.float32)
    gaps.append(float(np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)).min()))
    allowed.append(c >= 0.0)
gaps, allowed = np.array(gaps), np.array(allowed)
touching = gaps < 0.005
check("the sample does fold the wrist into the shoulder", touching.sum() > 50, f"{touching.sum()} contacts")
check("every such contact is inside the keep-out", not (touching & allowed).any(), f"{(touching & allowed).sum()} allowed")
check("what the keep-out allows leaves 25 mm of air between the metal", gaps[allowed].min() >= 0.025,
      f"{gaps[allowed].min() * 1000:.0f} mm over {allowed.sum()} poses")

print(f"\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
