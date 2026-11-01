# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Plain-Python unit tests for helpers (numpy only; no Isaac)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mimus import helpers as H  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, info=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {info}")


print("== quaternion round-trips ==")
for q in [
    np.array([1.0, 0, 0, 0]),
    H.quat_from_axis_angle([0, 0, 1], 0.7),
    H.quat_from_axis_angle([1, 1, 0], 1.2),
]:
    R = H.matrix_from_quat(q)
    q2 = H.quat_from_matrix(R)
    # quaternion double-cover: compare up to sign
    same = min(np.linalg.norm(q - q2), np.linalg.norm(q + q2)) < 1e-9
    check("matrix<->quat round-trip", same, f"{q} vs {q2}")
    check("matrix orthonormal", np.allclose(R @ R.T, np.eye(3), atol=1e-9))

print("== aim_at: local +Z points at target ==")
for ee, tgt in [
    (np.array([0.3, 0.0, 0.5]), np.array([0.6, 0.0, 0.5])),
    (np.array([0.0, 0.0, 0.3]), np.array([0.4, 0.3, 0.6])),
    (np.array([0.2, -0.1, 0.4]), np.array([0.2, -0.1, 0.9])),  # straight up (near up_ref)
]:
    q = H.aim_at(ee, tgt, approach_axis="z")
    R = H.matrix_from_quat(q)
    z_world = R @ np.array([0, 0, 1.0])
    fwd = (tgt - ee) / np.linalg.norm(tgt - ee)
    check("tool +Z aligns to forward", np.allclose(z_world, fwd, atol=1e-6), f"{z_world} vs {fwd}")

print("== calibration Y-up -> Z-up ==")
q_cal = H.yup_to_zup_quat()
world = H.compose_point([0, 0, 0], q_cal, [0, 1.0, 0])  # sensor +Y
check("sensor +Y -> world +Z", np.allclose(world, [0, 0, 1.0], atol=1e-6), str(world))
world_x = H.compose_point([0, 0, 0], q_cal, [1.0, 0, 0])  # sensor +X stays +X
check("sensor +X -> world +X", np.allclose(world_x, [1.0, 0, 0], atol=1e-6), str(world_x))
world_off = H.compose_point([0.5, 0, 0.2], q_cal, [0, 1.0, 0])
check("translation composes", np.allclose(world_off, [0.5, 0, 1.2], atol=1e-6), str(world_off))

print(f"\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
