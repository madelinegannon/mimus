# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Pure-numpy helpers for the motion layer.

NO Isaac imports here on purpose: this module is unit-testable with plain Python
(numpy only), so the math is validated without launching the simulator.

Quaternions are [w, x, y, z] (scalar-first), matching Isaac Sim's convention.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Quaternion / rotation utilities (scalar-first [w, x, y, z])
# --------------------------------------------------------------------------- #


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion [w, x, y, z] (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a*b, both [w, x, y, z]."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q [w, x, y, z]."""
    return matrix_from_quat(q) @ np.asarray(v, dtype=np.float64)


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic angle (degrees) between two orientations given as quaternions."""
    d = abs(float(np.dot(np.asarray(q1), np.asarray(q2))))
    d = min(1.0, max(-1.0, d))
    return float(np.degrees(2.0 * np.arccos(d)))


def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    h = 0.5 * angle_rad
    s = np.sin(h)
    return np.array([np.cos(h), axis[0] * s, axis[1] * s, axis[2] * s])


# --------------------------------------------------------------------------- #
# Aim-at-person (look-at) orientation
# --------------------------------------------------------------------------- #

# Map of approach-axis name -> local unit axis the tool "points" along.
_AXES = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}


def aim_at(ee_pos, target_pos, up_ref=(0.0, 0.0, 1.0), approach_axis: str = "z") -> np.ndarray:
    """Quaternion [w,x,y,z] orienting the tool's `approach_axis` from ee_pos toward target_pos.

    Builds an orthonormal frame whose `approach_axis` column = the forward (look) direction.
    Falls back to an alternate up-reference when forward is near-parallel to up_ref.
    """
    ee_pos = np.asarray(ee_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    fwd = target_pos - ee_pos
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])  # degenerate: identity
    fwd = fwd / n
    up = np.asarray(up_ref, dtype=np.float64)
    up = up / (np.linalg.norm(up) + 1e-12)
    if abs(float(np.dot(fwd, up))) > 0.95:  # near-parallel -> pick a different up
        up = np.array([0.0, 1.0, 0.0]) if abs(fwd[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(up, fwd)
    right = right / (np.linalg.norm(right) + 1e-12)
    up2 = np.cross(fwd, right)
    # Columns of basis = where local x/y/z map to. We want local approach_axis -> fwd.
    if approach_axis == "z":
        R = np.column_stack([right, up2, fwd])
    elif approach_axis == "x":
        R = np.column_stack([fwd, right, up2])
    elif approach_axis == "y":
        R = np.column_stack([up2, fwd, right])
    else:
        raise ValueError(f"approach_axis must be x/y/z, got {approach_axis}")
    return quat_from_matrix(R)



# --------------------------------------------------------------------------- #
# Calibration math (sensor frame -> world via a parent transform)
# --------------------------------------------------------------------------- #


def yup_to_zup_quat() -> np.ndarray:
    """Default sensor->world rotation: +90 deg about X maps sensor +Y -> world +Z."""
    return quat_from_axis_angle([1.0, 0.0, 0.0], np.pi / 2.0)


def compose_point(parent_pos, parent_quat, local_point) -> np.ndarray:
    """World point = parent_pos + R(parent_quat) @ local_point. (USD does this for us in-app;
    this mirrors it for unit tests of the calibration default.)"""
    return np.asarray(parent_pos, dtype=np.float64) + quat_rotate(parent_quat, local_point)


