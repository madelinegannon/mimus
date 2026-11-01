# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Sample the uf850 collision meshes into the fixture test_collision_model.py checks the sphere set against.

Points are area-weighted on each link's collision mesh, in that link's own frame (metres). Needs pxr: run under
Isaac's Python, or under the venv, which borrows pxr from Isaac's omni.usd.libs. Re-run only if the USD changes.

Usage:  .venv\\Scripts\\python tools/sample_collision_mesh.py [--n 1500] [--isaac C:\\isaacsim]
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--usd", default=os.path.join(HERE, "..", "usd", "uf850", "uf850.usd"))
ap.add_argument("--out", default=os.path.join(HERE, "..", "tests", "fixtures", "uf850_surface.npz"))
ap.add_argument("--n", type=int, default=1500, help="points per link")
ap.add_argument("--isaac", default=r"C:\isaacsim", help="Isaac Sim root, for pxr when it is not importable")
args = ap.parse_args()

try:
    from pxr import Usd, UsdGeom
except ImportError:
    libs = sorted(glob.glob(os.path.join(args.isaac, "extscache", "omni.usd.libs-*")))
    if not libs:
        raise SystemExit(f"no pxr, and no omni.usd.libs under {args.isaac}")
    sys.path.insert(0, libs[-1])
    for d in (os.path.join(libs[-1], "bin"), libs[-1]):
        if os.path.isdir(d):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
    from pxr import Usd, UsdGeom

LINKS = ["link_base", "link1", "link2", "link3", "link4", "link5", "link6"]


def link_mesh(stage, xf, link):
    """(points in the link frame, triangles) of one link's collision mesh."""
    root = "/UF_ROBOT/world" if link == "link_base" else f"/UF_ROBOT/{link}"
    link_prim = stage.GetPrimAtPath(f"{root}/link_base" if link == "link_base" else root)
    mesh_prim = stage.GetPrimAtPath(f"{root}/collisions/{link}/mesh")
    if not (link_prim and mesh_prim and mesh_prim.IsA(UsdGeom.Mesh)):
        raise SystemExit(f"{link}: no collision mesh at {root}/collisions/{link}/mesh")
    mesh = UsdGeom.Mesh(mesh_prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    # USD matrices are row-vector; transpose to the column convention used everywhere else here
    T = np.linalg.inv(np.array(xf.GetLocalToWorldTransform(link_prim)).T) @ np.array(xf.GetLocalToWorldTransform(mesh_prim)).T
    tris, o = [], 0
    for c in counts:
        f = idx[o:o + c]
        o += c
        tris += [(f[0], f[k], f[k + 1]) for k in range(1, c - 1)]
    return pts @ T[:3, :3].T + T[:3, 3], np.array(tris)


def sample(P, tris, n, rng):
    a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    i = rng.choice(len(tris), n, p=area / area.sum())
    s, v = np.sqrt(rng.random(n)), rng.random(n)
    return a[i] * (1 - s)[:, None] + b[i] * (s * (1 - v))[:, None] + c[i] * (s * v)[:, None]


def main():
    stage = Usd.Stage.Open(os.path.abspath(args.usd))
    if not stage:
        raise SystemExit(f"cannot open {args.usd}")
    xf = UsdGeom.XformCache()
    rng = np.random.default_rng(0)
    out = {}
    for link in LINKS:
        P, tris = link_mesh(stage, xf, link)
        out[link] = sample(P, tris, args.n, rng).astype(np.float32)
        lo, hi = out[link].min(0), out[link].max(0)
        print(f"{link}: {len(tris)} triangles, x {lo[0]:+.3f}..{hi[0]:+.3f}  y {lo[1]:+.3f}..{hi[1]:+.3f}  z {lo[2]:+.3f}..{hi[2]:+.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"wrote {os.path.abspath(args.out)} ({os.path.getsize(args.out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
