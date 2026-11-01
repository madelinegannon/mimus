# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Unit tests for mimus.motion (pure numpy; no Isaac). Mode-dispatched Behavior."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mimus import motion as M  # noqa: E402
from mimus import helpers as H  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, info=""):
    global PASS, FAIL
    PASS, FAIL = (PASS + 1, FAIL) if cond else (PASS, FAIL + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {info if not cond else ''}")


BASE_XY = (0.0, 0.0)
HOME = np.array([0.4, 0.0, 1.0])
DT = 1.0 / 60.0
IDENT = np.array([1.0, 0.0, 0.0, 0.0])
BASE_Z = 1.01        # a raised test base; the shipped default scene is the floor (0.0)
ANCHOR_Z = BASE_Z + 1.04   # 0.67 above the shoulder: inside the reach sphere


def new_behavior(**over):
    b = M.Behavior(BASE_XY, HOME, params=dict(M.BUILTIN_PRESETS["Curious"], **over), base_z=BASE_Z)
    b.seed(HOME)
    return b


print("== SpringAgent arrival: converge + settle ==")
a = M.SpringAgent([0.3, 0, 1.0], kp=1.2, kd=2.4, steering=0.6)
tgt = np.array([0.55, 0.1, 0.9])
maxd = 0.0
for _ in range(1200):
    a.update(tgt, DT)
    maxd = max(maxd, np.linalg.norm(a.get_position()))
err = np.linalg.norm(a.get_position() - tgt)
check("converges to static target (<0.01)", err < 0.01, f"err={err:.4f}")
check("no runaway (stays bounded)", maxd < 3.0, f"maxd={maxd:.2f}")
check("velocity settled", np.linalg.norm(a._vel) < 0.05, f"|v|={np.linalg.norm(a._vel):.4f}")

print("== reach_cone_point (horizontal radius from base axis) ==")
anchor = np.array([0, 0, 1.0])
# Person far away horizontally (d_h = 0.6, reach = 0.5 → t_reach = 0.833)
p, rl = M.reach_cone_point([0.6, 0.0, 0.9], BASE_XY, height_z=1.0, reach=0.5)
horiz = np.sqrt(p[0]**2 + p[1]**2)
check("default lerp=0.5 gives half of reach", abs(horiz - 0.25) < 1e-6, f"horiz={horiz:.4f}")
rl_horiz = np.sqrt(rl[0]**2 + rl[1]**2)
check("reach_limit at horizontal distance == reach", abs(rl_horiz - 0.5) < 1e-6, f"rl_horiz={rl_horiz:.4f}")

print("== reach_cone_point lerp ==")
p0, _ = M.reach_cone_point([2.0, 0.0, 1.0], BASE_XY, height_z=1.0, reach=0.5, lerp=0.0)
check("lerp=0 -> at anchor", np.linalg.norm(p0 - anchor) < 1e-9, str(p0))
p1, rl1 = M.reach_cone_point([2.0, 0.0, 1.0], BASE_XY, height_z=1.0, reach=0.5, lerp=1.0)
check("lerp=1 far person -> horiz == reach", abs(np.sqrt(p1[0]**2 + p1[1]**2) - 0.5) < 1e-6, str(p1))
ph, _ = M.reach_cone_point([2.0, 0.0, 1.0], BASE_XY, height_z=1.0, reach=0.5, lerp=0.5)
check("lerp=0.5 far person -> half of horiz reach", abs(np.sqrt(ph[0]**2 + ph[1]**2) - 0.25) < 1e-6, str(ph))
p_close, _ = M.reach_cone_point([0.3, 0.0, 1.0], BASE_XY, height_z=1.0, reach=0.5, lerp=1.0)
check("lerp=1 close person -> at reach limit (horiz == reach)",
      abs(np.sqrt(p_close[0]**2 + p_close[1]**2) - 0.5) < 1e-6, str(p_close))

print("== modes constant set ==")
check("MODES has the 6 dev states",
      set(M.MODES) == {M.FOLLOW_TARGET, M.FOLLOW_SENSOR, M.IDLE, M.SLEEP, M.HOME, M.HOLD}, str(M.MODES))

print("== hint -> mode (the tracker owns IDLE / SLEEP / follow) ==")
check("idle -> IDLE", M.mode_for_hint("idle") == M.IDLE)
check("sleep -> SLEEP", M.mode_for_hint("sleep") == M.SLEEP)
check("watch / engage / reach -> follow",
      all(M.mode_for_hint(h) == M.FOLLOW_SENSOR for h in ("watch", "engage", "reach")))
check("an unknown hint means a person, never a fold", M.mode_for_hint("bogus") == M.FOLLOW_SENSOR)
check("AUTO_MODES is exactly the tracker-owned pair", set(M.AUTO_MODES) == {"idle", "sleep"}, str(M.AUTO_MODES))

print("== FollowTarget: passthrough of cube pose (agent OFF) ==")
b = new_behavior()
cube_pos = np.array([0.5, 0.1, 0.95])
cube_quat = H.quat_from_axis_angle([0, 0, 1], 0.6)
pos, quat, dbg = b.update(M.FOLLOW_TARGET, (cube_pos, cube_quat), HOME, ANCHOR_Z, False, HOME, DT)
check("FollowTarget pos == cube pos", np.linalg.norm(pos - cube_pos) < 1e-9, str(pos))
check("FollowTarget quat == cube quat", np.linalg.norm(quat - cube_quat) < 1e-9, str(quat))
check("state == FollowTarget", b.state == M.FOLLOW_TARGET, b.state)
check("agent kept synced when OFF (no lunge on toggle)",
      np.linalg.norm(dbg["agent_pos"] - cube_pos) < 1e-9, str(dbg["agent_pos"]))
check("dbg has the viz keys", all(k in dbg for k in ("agent_pos", "seek_target", "anchor", "lookat", "reach_limit")))

print("== FollowTarget: agent ON seeks the cube ==")
b = new_behavior()
cube_pos = np.array([0.55, -0.1, 1.0])      # 0.67 from the shoulder: inside the reach sphere
last = None
for _ in range(800):
    last, _, _ = b.update(M.FOLLOW_TARGET, (cube_pos, IDENT), HOME, ANCHOR_Z, True, HOME, DT)
check("agent ON converges toward cube", np.linalg.norm(last - cube_pos) < 0.02, str(last))

print("== reach sphere: no target ever asks the arm to straighten out ==")
b = new_behavior()
far_cube = np.array([0.9, 0.0, 0.9])        # 0.99 from the shoulder: beyond the 0.72 sphere
for _ in range(800):
    last, _, dbg_c = b.update(M.FOLLOW_TARGET, (far_cube, IDENT), HOME, ANCHOR_Z, True, HOME, DT)
check("an unreachable cube is pulled onto the sphere about the shoulder", abs(np.linalg.norm(last - b.shoulder) - b.reach_max) < 0.02, f"{np.linalg.norm(last - b.shoulder):.3f}")
check("and the dbg says so", dbg_c["reach_clamped"] > 0, str(dbg_c["reach_clamped"]))
check("the shoulder sits 0.364 above the flange", abs(b.shoulder[2] - (BASE_Z + 0.364)) < 1e-9, str(b.shoulder))
b_off = M.Behavior(BASE_XY, HOME, params=M.BUILTIN_PRESETS["Curious"], base_z=BASE_Z, reach_max=None)
b_off.seed(HOME)
for _ in range(800):
    last_off, _, _ = b_off.update(M.FOLLOW_TARGET, (far_cube, IDENT), HOME, ANCHOR_Z, True, HOME, DT)
check("reach_max=None switches the clamp off (tests of the raw geometry)", np.linalg.norm(last_off - far_cube) < 0.02, str(last_off))
for mode_name, look in ((M.FOLLOW_SENSOR, np.array([3.0, 0.0, 1.6])), (M.FOLLOW_SENSOR, np.array([0.3, 0.0, 1.0])), (M.IDLE, HOME)):
    bm = new_behavior()
    worst = 0.0
    for _ in range(600):
        pm, _, _ = bm.update(mode_name, (np.zeros(3), IDENT), look, ANCHOR_Z + 0.9, True, HOME, DT)   # anchor pushed high on purpose
        worst = max(worst, float(np.linalg.norm(pm - bm.shoulder)))
    check(f"{mode_name} with a too-high anchor never leaves the sphere", worst <= bm.reach_max + 1e-6, f"worst {worst:.3f}")

print("== FollowTargetSensor: proximity-responsive reach cone ==")
b = new_behavior()
# Lookat at 1.5m horizontal — inside the working range (0.5–3.0)
lookat_mid = np.array([1.5, 0.0, 1.2])
pos, quat, dbg = b.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), lookat_mid, ANCHOR_Z, False, HOME, DT)
check("sensor dbg has reach_limit", dbg.get("reach_limit") is not None)
check("sensor dbg has retreating", "retreating" in dbg)
check("not retreating at 1.5m", not dbg["retreating"])
horiz_dist = np.sqrt(pos[0]**2 + pos[1]**2)
check("sensor pos horiz within reach", horiz_dist <= b.params["reach"] + 1e-6, f"horiz={horiz_dist:.4f}")
# orientation: tool approach axis (z) must point from eff_anchor toward lookat
eff_anchor = dbg["anchor"]
fwd = (lookat_mid - eff_anchor) / np.linalg.norm(lookat_mid - eff_anchor)
zaxis = H.matrix_from_quat(quat) @ np.array([0.0, 0.0, 1.0])
check("aim ray = anchor->lookat (z axis aligned)", np.linalg.norm(zaxis - fwd) < 1e-6, str(zaxis))

print("== FollowTargetSensor: proximity modulation (close vs far) ==")
p = new_behavior().params
lookat_near = np.array([p["dist_near"], 0.0, 1.0])
lookat_far = np.array([p["dist_far"], 0.0, 1.0])
ee_far = np.array([0.0, 0.0, ANCHOR_Z])  # ee far from lookat so retreat never triggers
b_near = new_behavior()
pos_near, _, dbg_near = b_near.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), lookat_near, ANCHOR_Z, False, ee_far, DT)
b_far = new_behavior()
pos_far, _, dbg_far = b_far.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), lookat_far, ANCHOR_Z, False, ee_far, DT)
check("near: eff_height ~ base + height_near_dz", abs(dbg_near["anchor"][2] - (BASE_Z + p["height_near_dz"])) < 1e-6,
      f"eff_h={dbg_near['anchor'][2]:.3f}")
check("far: eff_height ~ anchor_z", abs(dbg_far["anchor"][2] - ANCHOR_Z) < 1e-6,
      f"eff_h={dbg_far['anchor'][2]:.3f}")
check("near pos lower than far pos", pos_near[2] < pos_far[2],
      f"near_z={pos_near[2]:.3f} far_z={pos_far[2]:.3f}")
horiz_near = np.sqrt(pos_near[0]**2 + pos_near[1]**2)
horiz_far = np.sqrt(pos_far[0]**2 + pos_far[1]**2)
check("near reaches out further than far", horiz_near > horiz_far + 1e-3,
      f"near_h={horiz_near:.3f} far_h={horiz_far:.3f}")

print("== FollowTargetSensor: heights follow the base flange ==")
UP = np.array([0.0, 0.0, 1.5 - BASE_Z])
b_mus = M.Behavior(BASE_XY, HOME + UP, params=M.BUILTIN_PRESETS["Curious"], base_z=1.5)
_, _, dbg_mus = b_mus.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), lookat_near, ANCHOR_Z + UP[2], False, ee_far + UP, DT)
check("raised base: near height = base_z + height_near_dz", abs(dbg_mus["anchor"][2] - (1.5 + p["height_near_dz"])) < 1e-6,
      f"eff_h={dbg_mus['anchor'][2]:.3f}")
b_mus_far = M.Behavior(BASE_XY, HOME + UP, params=M.BUILTIN_PRESETS["Curious"], base_z=1.5)
_, _, dbg_mus_far = b_mus_far.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), lookat_far, ANCHOR_Z + UP[2], False, ee_far + UP, DT)
check("raised base: far height = anchor_z", abs(dbg_mus_far["anchor"][2] - (ANCHOR_Z + UP[2])) < 1e-6,
      f"eff_h={dbg_mus_far['anchor'][2]:.3f}")
b_dz = M.Behavior(BASE_XY, HOME, params=dict(M.BUILTIN_PRESETS["Curious"], height_near_dz=-0.1), base_z=BASE_Z)
_, _, dbg_dz = b_dz.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), lookat_near, ANCHOR_Z, False, ee_far, DT)
check("height_near_dz overrides the near height", abs(dbg_dz["anchor"][2] - (BASE_Z - 0.1)) < 1e-6,
      f"eff_h={dbg_dz['anchor'][2]:.3f}")

print("== FollowTargetSensor: anchor_z changes the rest height ==")
b_lo = new_behavior()
pos_lo, _, d_lo = b_lo.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), lookat_far, 1.2, False, ee_far, DT)
b_hi = new_behavior()
pos_hi, _, d_hi = b_hi.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), lookat_far, 2.0, False, ee_far, DT)
check("anchor_z shifts far-rest height", abs(pos_lo[2] - pos_hi[2]) > 0.1,
      f"{pos_lo[2]:.3f} vs {pos_hi[2]:.3f}")

print("== FollowTargetSensor: retreat drives to anchor ==")
b = new_behavior()
b.seed(HOME)
ee_here = HOME.copy()
lookat_touch = ee_here + np.array([0.05, 0.0, 0.0])
anchor_expected = np.array([0.0, 0.0, ANCHOR_Z])
pos_r, _, dbg_r = b.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), lookat_touch, ANCHOR_Z, False, ee_here, DT)
check("retreat triggered (ee-to-lookat < retreat_dist)", dbg_r["retreating"])
d_before, d_after = np.linalg.norm(ee_here - anchor_expected), np.linalg.norm(pos_r - anchor_expected)
check("retreat drives most of the way to the anchor (retreat_frac)", d_after < 0.4 * d_before and pos_r[2] > ee_here[2], f"{d_after:.3f} of {d_before:.3f}")
check("retreat timer set", b._retreat_until > b._phase - DT)
check("preset retreat_dist clears the robot mask zone (12.5 cm) with margin", 0.2 <= b.params["retreat_dist"] <= 0.5, str(b.params["retreat_dist"]))
b2 = new_behavior()
b2.seed(HOME)
_, _, dbg_02 = b2.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), ee_here + np.array([0.2, 0.0, 0.0]), ANCHOR_Z, False, ee_here, DT)
check("a hand 0.2 m from the tool (visible outside the mask) triggers", dbg_02["retreating"])
b3 = new_behavior()
b3.seed(HOME)
_, _, dbg_05 = b3.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), ee_here + np.array([0.5, 0.0, 0.0]), ANCHOR_Z, False, ee_here, DT)
check("0.5 m does not", not dbg_05["retreating"])

print("== FollowTargetSensor: retreat timer holds and expires ==")
for _ in range(10):
    _, _, dbg_hold = b.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT),
                               np.array([2.0, 0.0, 1.0]), ANCHOR_Z, False, ee_here, DT)
check("still retreating during timer (person moved away)", dbg_hold["retreating"])
for _ in range(200):
    b.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT),
             np.array([2.0, 0.0, 1.0]), ANCHOR_Z, False, ee_here, DT)
_, _, dbg_expired = b.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT),
                              np.array([2.0, 0.0, 1.0]), ANCHOR_Z, False, ee_here, DT)
check("retreat expires after duration", not dbg_expired["retreating"])

print("== FollowTargetSensor: manual mode switch clears retreat ==")
b = new_behavior()
b.seed(HOME)
ee_here = HOME.copy()
lookat_touch = ee_here + np.array([0.05, 0.0, 0.0])
b.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), lookat_touch, ANCHOR_Z, False, ee_here, DT)
check("retreat active", b._phase < b._retreat_until)
b.update(M.IDLE, (np.zeros(3), IDENT), HOME, ANCHOR_Z, False, HOME, DT)
check("timer cleared on mode switch", b._retreat_until == 0.0)

print("== FollowTargetSensor: no re-trigger during active retreat ==")
b = new_behavior()
b.seed(HOME)
ee_here = HOME.copy()
lookat_touch = ee_here + np.array([0.05, 0.0, 0.0])
b.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), lookat_touch, ANCHOR_Z, False, ee_here, DT)
original_until = b._retreat_until
b.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), lookat_touch, ANCHOR_Z, False, ee_here, DT)
check("timer not reset by second trigger", abs(b._retreat_until - original_until) < 1e-9)

print("== FollowTargetSensor: one sniff per encounter (no approach/retreat cycle) ==")


def sniff_run(hand, once, secs=30.0, tid=7, hint="reach"):
    """Stationary close hand, EE fed back in as the app does. -> (arms, gap, behavior, ee, min held z)"""
    bb = M.Behavior(BASE_XY, HOME, params=dict(M.BUILTIN_PRESETS["Curious"], sniff_once=once, sway_amp=0.0), base_z=BASE_Z)   # the hold pose is asserted to 2 mm; the sway has its own block
    bb.seed(HOME)
    ee, arms, prev, gap, latched, zmin = HOME.copy(), 0, False, 1e9, False, 1e9
    for _ in range(int(secs / DT)):
        ee, _, d = bb.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), hand, ANCHOR_Z, True, ee, DT,
                             track_id=tid, hint=hint)
        arms += 1 if (d["retreating"] and not prev) else 0
        prev = d["retreating"]
        if latched and not d["retreating"]:
            gap = min(gap, float(np.linalg.norm(hand - ee)))
        latched = latched or d["retreating"]
        if latched and not d["retreating"]:
            zmin = min(zmin, float(ee[2]))
    return arms, gap, bb, ee.copy(), zmin


HAND = np.array([0.85, 0.0, BASE_Z + 0.15])
arms_off, _, _, _, _ = sniff_run(HAND, 0.0)
check("sniff_once=0 reproduces the old approach/retreat cycle", arms_off >= 2, f"arms={arms_off}")
arms_on, gap_on, b_sn, sniff_ee, zmin_on = sniff_run(HAND, 1.0)
check("exactly one retreat in 30 s of a stationary close hand", arms_on == 1, f"arms={arms_on}")
check("holds clear of retreat_dist after the sniff", gap_on > b_sn.params["retreat_dist"], f"gap={gap_on:.3f}")
check("latch set and held on the sniffed track", b_sn._sniffed and b_sn._sniff_track == 7)
check("holds the high retreat pose, not the low near-height cone",
      np.allclose(sniff_ee, HAND + b_sn.params["sniff_hold_frac"] * (np.array([0.0, 0.0, ANCHOR_Z]) - HAND), atol=2e-3),
      str(sniff_ee))
check("held wrist stays above the flange (not folded in beside the column)", zmin_on > BASE_Z, f"z={zmin_on:.3f}")
_p = M.BUILTIN_PRESETS["Curious"]
_near_h = BASE_Z + _p["height_near_dz"]
check("held well above the old near-anchor height", zmin_on > _near_h + 0.5, f"{zmin_on:.3f} vs near {_near_h:.3f}")
check("held radius clears the column axis", np.hypot(sniff_ee[0], sniff_ee[1]) > 0.15, str(sniff_ee))
check("dbg reports the latch", b_sn.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), HAND, ANCHOR_Z,
                                           True, HOME.copy(), DT, track_id=7, hint="reach")[2]["sniffed"])

print("== FollowTargetSensor: holding is not freezing ==")
_, _, b_h, _, _ = sniff_run(HAND, 1.0)
hand2 = np.array([0.85, 0.5, BASE_Z + 0.15])
pos_h = HOME.copy()
for _ in range(int(4.0 / DT)):
    pos_h, quat_h, dbg_h = b_h.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), hand2, ANCHOR_Z, True,
                                      pos_h, DT, track_id=7, hint="reach")
check("body tracks the hand sideways, not frozen", abs(pos_h[1]) > 0.05, str(pos_h))
fwd = hand2 - pos_h
fwd = fwd / np.linalg.norm(fwd)
zax = H.matrix_from_quat(quat_h) @ np.array([0.0, 0.0, 1.0])
check("gaze still aims at the hand while holding", float(np.dot(zax, fwd)) > 0.99, str(zax))

print("== FollowTargetSensor: sniff latch lifecycle ==")
_, _, b_l, _, _ = sniff_run(HAND, 1.0)
b_l.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), HAND, ANCHOR_Z, True, HOME.copy(), DT,
           track_id=-1, hint="reach")
check("a track dropout does not clear the latch", b_l._sniffed)
b_l.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), HAND, ANCHOR_Z, True, HOME.copy(), DT,
           track_id=7, hint="engage")
check("a hint change alone does not clear the latch", b_l._sniffed)
b_l.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), HAND, ANCHOR_Z, True, HOME.copy(), DT,
           track_id=8, hint="engage")
check("a new track clears the latch", not b_l._sniffed)
_, _, b_m, _, _ = sniff_run(HAND, 1.0)
far = HAND + np.array([0.0, 1.5, 0.0])
for _ in range(int(3.0 / DT)):
    b_m.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), far, ANCHOR_Z, True, HOME.copy(), DT,
               track_id=7, hint="reach")
check("look-at travel past sniff_move_m clears the latch", not b_m._sniffed)
ee_rel = HOME.copy()
for _ in range(int(6.0 / DT)):
    ee_rel, _, _ = b_m.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), far, ANCHOR_Z, True, ee_rel, DT,
                              track_id=7, hint="reach")
_held_z = HAND[2] + b_m.params["sniff_hold_frac"] * (ANCHOR_Z - HAND[2])
check("after release the height drops back to the interpolated anchor", ee_rel[2] < _held_z - 0.3,
      f"z={ee_rel[2]:.3f} vs held {_held_z:.3f}")
_, _, b_s, _, _ = sniff_run(HAND, 1.0)
for _ in range(int(3.0 / DT)):
    b_s.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), HAND + np.array([0.0, 0.05, 0.0]), ANCHOR_Z,
               True, HOME.copy(), DT, track_id=7, hint="reach")
check("small jitter does not clear the latch", b_s._sniffed)

print("== presets: standoff clears the retreat trigger ==")
for _nm, _pp in M.BUILTIN_PRESETS.items():
    check(f"{_nm}: sniff_standoff > retreat_dist", _pp["sniff_standoff"] > _pp["retreat_dist"],
          f"{_pp['sniff_standoff']} vs {_pp['retreat_dist']}")

print("== FollowTargetSensor: lerp [0, 1] range ==")
p = new_behavior().params
lookat_near = np.array([p["dist_near"], 0.0, 1.0])
ee_safe = np.array([0.0, 0.0, ANCHOR_Z])
b_ln = new_behavior()
pos_near, _, _ = b_ln.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), lookat_near, ANCHOR_Z, False, ee_safe, DT)
horiz_near = np.sqrt(pos_near[0]**2 + pos_near[1]**2)
check("lerp=1 at dist_near: reaches out as far as the sphere allows (the cone alone would be reach = 1.0)",
      abs(np.linalg.norm(pos_near - b_ln.shoulder) - b_ln.reach_max) < 1e-6 and horiz_near < p["reach"], f"horiz={horiz_near:.4f}")
lookat_far = np.array([p["dist_far"], 0.0, 1.0])
b_lf = new_behavior()
pos_far, _, _ = b_lf.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), lookat_far, ANCHOR_Z, False, ee_safe, DT)
horiz_far = np.sqrt(pos_far[0]**2 + pos_far[1]**2)
check("lerp=0 at dist_far: at anchor", horiz_far < 1e-6, f"horiz={horiz_far:.4f}")

print("== FollowTargetSensor: the lean eases in and out along the ramp ==")


def lean_at(d):
    """(dbg lean, horizontal offset / Reach) for a visitor d metres out, raw geometry (no reach sphere)."""
    bb = M.Behavior(BASE_XY, HOME, params=M.BUILTIN_PRESETS["Curious"], base_z=BASE_Z, reach_max=None)
    pos_d, _, dbg_d = bb.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), np.array([d, 0.0, 1.0]), ANCHOR_Z, False, ee_safe, DT)
    return dbg_d["lean"], np.sqrt(pos_d[0] ** 2 + pos_d[1] ** 2) / p["reach"]


lin = lambda d: 1.0 - (d - p["dist_near"]) / (p["dist_far"] - p["dist_near"])   # the old straight line
mid = (p["dist_near"] + p["dist_far"]) / 2
check("midpoint of the ramp: lean 0.5", abs(lean_at(mid)[0] - 0.5) < 1e-6, f"{lean_at(mid)[0]:.4f}")
check("close in (1.2 m): more lean than the straight line", lean_at(1.2)[0] > lin(1.2) + 0.05, f"{lean_at(1.2)[0]:.3f} vs linear {lin(1.2):.3f}")
check("far out (2.5 m): less lean than the straight line", lean_at(2.5)[0] < lin(2.5) - 0.05, f"{lean_at(2.5)[0]:.3f} vs linear {lin(2.5):.3f}")
leans = [lean_at(d)[0] for d in np.linspace(p["dist_near"], p["dist_far"], 26)]
check("lean falls monotonically with distance, 1 at dist_near, 0 at dist_far", all(a >= b - 1e-9 for a, b in zip(leans, leans[1:]))
      and abs(leans[0] - 1.0) < 1e-9 and abs(leans[-1]) < 1e-9, str(np.round(leans, 3)))
check("dbg['lean'] is the horizontal offset over Reach", all(abs(lean_at(d)[0] - lean_at(d)[1]) < 1e-6 for d in (0.8, 1.2, 2.0, 2.8)))

print("== FollowTargetSensor: a still visitor never faces a still arm (follow sway) ==")


def sway_run(amp, secs=30.0, look=np.array([2.0, 0.4, 1.3])):
    """Fixed look-at, EE parked far from it (no retreat), agent ON. -> (behavior, [(pos, quat, dbg)])"""
    bb = new_behavior(sway_amp=amp)
    ee_far = np.array([0.0, 0.0, ANCHOR_Z])
    out = []
    for _ in range(int(secs / DT)):
        pp, qq, dd = bb.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), look, ANCHOR_Z, True, ee_far, DT)
        out.append((pp.copy(), qq.copy(), dd))
    return bb, out


AMP = M.BUILTIN_PRESETS["Curious"]["sway_amp"]
check("shipped sway is barely there (1..2 cm)", 0.01 <= AMP <= 0.02, str(AMP))
b_sw, run_sw = sway_run(AMP)
b_st, run_st = sway_run(0.0)
P, P0 = np.array([r[0] for r in run_sw]), np.array([r[0] for r in run_st])
i0, win = int(6.0 / DT), int(2.0 / DT)
p2p = [float(np.max(np.ptp(P[i:i + win], axis=0))) for i in range(i0, len(P) - win, win)]
check("never still: every 2 s window moves > 2 mm", min(p2p) > 0.002, f"min p2p={min(p2p) * 1e3:.2f} mm")
check("and it shows: > 1 cm of travel over the run", float(np.max(np.ptp(P[i0:], axis=0))) > 0.01)
dev = np.linalg.norm(P - P0, axis=1)
check("bounded by sway_amp about the no-sway path", float(dev.max()) <= AMP + 1e-3, f"max={dev.max():.4f}")
check("always inside the reach sphere", max(float(np.linalg.norm(pp - b_sw.shoulder)) for pp in P) <= b_sw.reach_max + 1e-9)
check("quaternion stays unit", max(abs(float(np.linalg.norm(r[1])) - 1.0) for r in run_sw) < 1e-6)
check("no jump on entry: first frame equals the no-sway first frame (fade-in)", float(np.linalg.norm(P[0] - P0[0])) < 1e-4)
steps = np.linalg.norm(np.diff(P[i0:], axis=0), axis=1)
check("smooth: no frame steps more than 2 mm", float(steps.max()) < 0.002, f"max={steps.max() * 1e3:.2f} mm")
_, run_again = sway_run(AMP)
check("deterministic: a second run is identical", all(np.array_equal(a[0], c[0]) for a, c in zip(run_sw, run_again)))
b_re, run_re = sway_run(AMP, secs=10.0)
b_re.seed(run_re[-1][0])                      # what _handoff_from_joints does after a sleep
p_re, _, _ = b_re.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), np.array([2.0, 0.4, 1.3]), ANCHOR_Z, True, run_re[-1][0], DT)
check("a re-seed (waking from SLEEP) fades the sway back in, no step", float(np.linalg.norm(p_re - run_re[-1][0])) < 1e-3)

print("== FollowTargetSensor: sway_amp=0 is the old exact behaviour ==")
b0, run0 = sway_run(0.0, secs=15.0)
check("sway off: the output is the settled spring position itself, every frame",
      all(np.array_equal(r[0], r[2]["agent_pos"]) for r in run0))
check("sway off: converges onto the clamped cone point (< 1 mm)",
      float(np.linalg.norm(run0[-1][0] - run0[-1][2]["seek_target"])) < 1e-3,
      f"{np.linalg.norm(run0[-1][0] - run0[-1][2]['seek_target']):.4f}")
check("sway on: the output is not the spring position", any(not np.array_equal(r[0], r[2]["agent_pos"]) for r in run_sw))

print("== IDLE: wanders within reach, smooth gaze drift ==")
b = new_behavior()
np.random.seed(42)
positions = []
for _ in range(300):
    pos, quat, dbg = b.update(M.IDLE, (np.zeros(3), IDENT), HOME, ANCHOR_Z, True, HOME, DT)
    positions.append(pos.copy())
check("IDLE state", b.state == M.IDLE, b.state)
horiz = np.sqrt(pos[0]**2 + pos[1]**2)
check("IDLE within reach", horiz <= b.params["reach"] + 1.0, f"horiz={horiz:.3f}")
deltas = [np.linalg.norm(positions[i+1] - positions[i]) for i in range(len(positions)-1)]
check("IDLE motion is smooth (no jumps)", max(deltas) < 0.1, f"max_delta={max(deltas):.4f}")
check("IDLE quat is unit", abs(np.linalg.norm(quat) - 1.0) < 1e-6, f"|q|={np.linalg.norm(quat):.6f}")

print("== IDLE: cone in the base frame, starts where the arm looks, heights at head height ==")
b_yaw = M.Behavior(BASE_XY, HOME, params=M.BUILTIN_PRESETS["Curious"], base_z=BASE_Z, base_yaw_deg=90.0)
ee_side = np.array([BASE_XY[0] + 0.3, BASE_XY[1] + 0.9, BASE_Z + 0.7])          # arm looking ~72 deg to the left
b_yaw.update(M.IDLE, (np.zeros(3), IDENT), HOME, ANCHOR_Z, False, ee_side, DT)
az0 = b_yaw._idle_azimuth
d_az = abs((az0 - np.arctan2(0.9, 0.3) + np.pi) % (2 * np.pi) - np.pi)
check("sweep starts near the arm's current azimuth (no snap to the centre)", d_az <= np.radians(36), f"{np.degrees(az0):.1f} deg")
ys = []
for _ in range(2000):
    pos_y, quat_y, dbg_y = b_yaw.update(M.IDLE, (np.zeros(3), IDENT), HOME, ANCHOR_Z, False, ee_side, DT)
    ys.append(b_yaw._idle_lookat[1] - BASE_XY[1])
check("base yaw 90: the cone (heading 0 +-90) faces +y, look-at never behind the base", min(ys) > -0.05, f"min y {min(ys):.3f}")
check("IDLE arm height is a world height (idle_ee_z), not a flange offset", abs(dbg_y["anchor"][2] - b_yaw.params["idle_ee_z"]) < 1e-6, f"{dbg_y['anchor'][2]:.3f}")
lz = b_yaw._idle_lookat[2]
check("look-at height is at head height (idle_look_z_lo..hi, world)", b_yaw.params["idle_look_z_lo"] - 1e-6 <= lz <= b_yaw.params["idle_look_z_hi"] + 1e-6, f"{lz:.3f}")
b_hi = M.Behavior(BASE_XY, HOME, params=M.BUILTIN_PRESETS["Curious"], base_z=1.5)
_, _, dbg_hi = b_hi.update(M.IDLE, (np.zeros(3), IDENT), HOME, ANCHOR_Z, False, HOME + UP, DT)
check("raised base: idle height stays at head height, it does not rise with the mount", abs(dbg_hi["anchor"][2] - b_hi.params["idle_ee_z"]) < 1e-6, f"{dbg_hi['anchor'][2]:.3f}")
check("IDLE has gaze agent", hasattr(b, 'gaze'))
check("IDLE dbg has reach_limit", dbg["reach_limit"] is not None)
check("IDLE dbg lookat is smoothed gaze", np.allclose(dbg["lookat"], b.gaze.get_position()))

print("== JointMove: the timed joint-space move (HOME / SLEEP): smoothstep ease-in-out, all joints arrive together ==")
import json  # noqa: E402
from pathlib import Path  # noqa: E402
REPO = Path(__file__).resolve().parents[1]
POSES = json.loads((REPO / "config" / "app.example.json").read_text(encoding="utf-8"))
q0 = np.array([1.0, -0.5, 0.3, 2.0, -1.0, 0.1])
tgt = np.array([0.0, 1.0123, -0.7854, 0.0, -0.2269, 0.0])
mv = M.JointMove(q0, tgt, 0.8, 0.5, 10.0)
check("duration = largest delta / rate (2.0 rad on J4 at 0.8 rad/s -> 2.5 s)", abs(mv.duration - 2.5) < 1e-9, str(mv.duration))
qs = [mv.at(10.0 + i * DT)[0] for i in range(int(mv.duration / DT) + 1)]
q_end, done_end = mv.at(10.0 + mv.duration)
check("starts at q0, not done before the duration", np.allclose(qs[0], q0) and not mv.at(10.0 + mv.duration - 1e-3)[1])
check("ends exactly at the target and reports done", done_end and np.array_equal(q_end, tgt) and mv.at(99.0)[1])
steps = np.diff(np.array(qs + [q_end]), axis=0)
check("monotone toward the target, never overshoots", bool(np.all(steps * np.sign(tgt - q0) >= -1e-12)))
speed = np.abs(steps[:, 3]) / DT
check("ease-in-out: first and last frame under 5% of the peak; peak = 1.5 x rate (1.2 rad/s +-3%)",
      speed[0] < 0.05 * speed.max() and speed[-1] < 0.05 * speed.max() and abs(speed.max() - 1.2) < 0.036, f"peak={speed.max():.3f}")
mid = (qs[len(qs) // 2] - q0) / (tgt - q0)
check("all joints share one progress (arrive together)", float(np.ptp(mid)) < 1e-9, str(mid))
check("a tiny move still takes min_s", abs(M.JointMove(q0, q0 + 0.01, 0.8, 0.5, 0.0).duration - 0.5) < 1e-9)
hq, sq0 = np.asarray(POSES["home_q_rad"]), np.asarray(POSES["sleep_q_rad"])
t_hs = M.JointMove(hq, sq0, POSES["unwind_rate_rad_s"], POSES["unwind_min_s"], 0.0).duration
check("configured home -> sleep takes about 2 s", 1.5 <= t_hs <= 2.5, f"{t_hs:.2f} s")

print("== sway_uv: bounded, slow, never a beat ==")
ts = np.arange(0.0, 600.0, DT)
uv = np.array([M.sway_uv(t, 6.0) for t in ts])
check("|u|, |v| <= 1", float(np.abs(uv).max()) <= 1.0 + 1e-12)
check("both axes use their range (> 0.8)", float(np.abs(uv).max(axis=0).min()) > 0.8)
check("slow: a frame never steps more than 2 pi dt / period", float(np.abs(np.diff(uv, axis=0)).max()) < 2 * np.pi * DT / 6.0)
n6 = int(6.0 / DT)
check("does not repeat at the base period or its multiples",
      all(float(np.abs(uv[k * n6:] - uv[:-k * n6]).max()) > 0.3 for k in (1, 2, 5)))

print("== sleep_joint_pose: the sleep pose is the bottom of the breath; the head rises straight up and stays bowed ==")
p = dict(M.BUILTIN_PRESETS["Curious"])
sq = np.asarray(POSES["sleep_q_rad"], dtype=np.float64)
check("phase 0 is the sleep pose exactly", np.allclose(M.sleep_joint_pose(0.0, sq, p), sq))
qs = [M.sleep_joint_pose(i * DT, sq, p) for i in range(int(2 * p["sleep_breath_period"] / DT))]
amp = p["sleep_breath_rad"]
unit = M.SLEEP_BREATH_DIR / float(np.dot(M.SLEEP_BREATH_DIR, M.SLEEP_BREATH_DIR))
lifts = [float(np.dot(qq - sq, unit)) for qq in qs]
check("one-sided: never below the sleep pose, peak == sleep_breath_rad", min(lifts) >= -1e-9 and abs(max(lifts) - amp) < 1e-6,
      f"min={min(lifts):.4f} max={max(lifts):.4f}")
check("elbow unfolds and the wrist follows it, other joints still",
      all(abs((qq[2] - sq[2]) - (qq[4] - sq[4])) < 1e-9 and np.allclose(qq[[0, 1, 3, 5]], sq[[0, 1, 3, 5]]) for qq in qs))
deltas = [float(np.max(np.abs(qs[i + 1] - qs[i]))) for i in range(len(qs) - 1)]
check("SLEEP motion is smooth", max(deltas) < 0.01, f"max_delta={max(deltas):.4f}")
check("breath actually moves (elbow range == amplitude)", abs((max(qq[2] for qq in qs) - min(qq[2] for qq in qs)) - amp) < 1e-6)
# forward kinematics on the Lula URDF (mimus.robot_model: numpy + yaml, no Isaac) checks the configured pose, not just the law
from mimus.robot_model import RobotModel  # noqa: E402
_rm = RobotModel(mount_z=0.0)


def _flange(q):
    T = _rm.link_transforms(q)["link6"]
    return T[:3, 3], T[:3, 2]


p_sleep, tool_sleep = _flange(sq)
fk = [_flange(qq) for qq in qs]
check("sleep pose: tool points down at the pedestal (within 0.1 rad of -z)", float(np.linalg.norm(tool_sleep - [0, 0, -1.0])) < 0.1,
      f"tool={tool_sleep.round(3)} flange={p_sleep.round(3)}")
check("breath lifts the flange in +Z only (never below the sleep pose, > 1.5 cm at the peak)",
      min(pp[2] for pp, _ in fk) >= p_sleep[2] - 1e-6 and max(pp[2] for pp, _ in fk) - p_sleep[2] > 0.015,
      f"dz min={min(pp[2] for pp, _ in fk) - p_sleep[2]:.4f} max={max(pp[2] for pp, _ in fk) - p_sleep[2]:.4f}")
check("the head stays bowed through the breath (tool axis unchanged within 1e-3)",
      max(float(np.linalg.norm(tt - tool_sleep)) for _, tt in fk) < 1e-3)
pos, quat, dbg = b.update(M.SLEEP, (HOME, IDENT), HOME, ANCHOR_Z, True, HOME, DT)
check("Behavior SLEEP holds the current pose (the joint routine lives in the app)", np.allclose(pos, HOME) and abs(np.linalg.norm(quat) - 1.0) < 1e-6)
check("SLEEP in MODES", M.SLEEP in M.MODES)

print("== HOME: eases to home and holds ==")
b = M.Behavior(BASE_XY, HOME, params=M.BUILTIN_PRESETS["Curious"])
b.seed(np.array([0.6, 0.2, 1.2]))
pos = None
for _ in range(600):
    pos, _, _ = b.update(M.HOME, (np.zeros(3), IDENT), HOME, ANCHOR_Z, False, HOME, DT)
check("HOME state", b.state == M.HOME, b.state)
check("HOME eases to home (<0.02)", np.linalg.norm(pos - HOME) < 0.02, str(pos))

print("== HOLD: holds current pose unchanged ==")
b = new_behavior()
ee_here = np.array([0.45, 0.1, 0.85])
pos, _, _ = b.update(M.HOLD, (np.zeros(3), IDENT), HOME, ANCHOR_Z, False, ee_here, DT)
check("HOLD returns ee_now", np.allclose(pos, ee_here))
check("HOLD state", b.state == M.HOLD)
check("HOLD syncs agent", np.allclose(b.body.get_position(), ee_here))
# run 100 frames with a DIFFERENT cube pose and agent_on=True: HOLD must ignore both
for _ in range(100):
    pos, _, _ = b.update(M.HOLD, (np.array([0.6, 0.2, 1.1]), IDENT), HOME, ANCHOR_Z, True, ee_here, DT)
check("HOLD ignores cube + agent_on", np.allclose(pos, ee_here))

print("== FollowTargetSensor: agent ON orients from agent pos ==")
b = new_behavior()
b.seed(HOME)
lookat_side = np.array([2.0, 1.0, 1.0])
ee_far = np.array([0.0, 0.0, ANCHOR_Z])
for _ in range(60):
    pos, quat, dbg = b.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), lookat_side, ANCHOR_Z, True, ee_far, DT)
fwd_expected = (lookat_side - pos) / np.linalg.norm(lookat_side - pos)   # the output is the swayed agent position
zaxis = H.matrix_from_quat(quat) @ np.array([0.0, 0.0, 1.0])
check("agent ON: aim from the output pos->lookat", np.linalg.norm(zaxis - fwd_expected) < 1e-4,
      f"zaxis={zaxis} expected={fwd_expected}")
# agent OFF: aim from anchor
b2 = new_behavior()
_, quat_off, dbg_off = b2.update(M.FOLLOW_SENSOR, (np.zeros(3), IDENT), lookat_side, ANCHOR_Z, False, ee_far, DT)
anchor = dbg_off["anchor"]
fwd_anchor = (lookat_side - anchor) / np.linalg.norm(lookat_side - anchor)
zaxis_off = H.matrix_from_quat(quat_off) @ np.array([0.0, 0.0, 1.0])
check("agent OFF: aim from anchor->lookat", np.linalg.norm(zaxis_off - fwd_anchor) < 1e-6,
      f"zaxis={zaxis_off} expected={fwd_anchor}")
check("dbg has trail", "trail" in dbg and isinstance(dbg["trail"], list))
check("dbg has agent_on", dbg.get("agent_on") is True)

print("== default_mode selection ==")
check("mock -> FollowTarget", M.default_mode("mock", False, False, None) == M.FOLLOW_TARGET)
check("xarm -> HOLD", M.default_mode("xarm", False, False, None) == M.HOLD)
check("xarm+engage -> FollowTarget", M.default_mode("xarm", True, False, None) == M.FOLLOW_TARGET)
check("test -> FollowSensor", M.default_mode("xarm", False, True, None) == M.FOLLOW_SENSOR)
check("explicit overrides", M.default_mode("xarm", False, False, "idle") == M.IDLE)
check("explicit hold", M.default_mode("mock", False, False, "hold") == M.HOLD)
check("explicit sleep", M.default_mode("mock", False, False, "sleep") == M.SLEEP)

print("== preset persistence round-trip ==")
pf = os.path.join(tempfile.gettempdir(), "behavior_presets_test.json")
if os.path.exists(pf):
    os.remove(pf)
params = dict(M.BUILTIN_PRESETS["Tired"]); params["body_kp"] = 0.42
M.save_preset(pf, "Tired", params)
loaded = M.load_presets(pf)
check("override persists", abs(loaded["Tired"]["body_kp"] - 0.42) < 1e-9, str(loaded["Tired"]["body_kp"]))
check("other presets intact", loaded["Curious"]["body_kp"] == M.BUILTIN_PRESETS["Curious"]["body_kp"])
os.remove(pf)

print("== speed setting persists once the drag settles ==")
sv = M.SettledValue(0.1, after_s=2.0)
check("no write while it matches disk", not sv.update(0.1, 0.0))
check("no write the instant it changes", not sv.update(0.95, 0.0))
check("no write mid-drag", not sv.update(0.95, 1.0))
# still dragging: every new value restarts the clock, so a slow sweep writes nothing
for t, v in ((1.5, 0.80), (2.0, 0.70), (2.5, 0.60)):
    check(f"a moving slider does not write (t={t})", not sv.update(v, t))
check("still unsaved after the sweep", sv.saved == 0.1, str(sv.saved))
check("writes once it holds", sv.update(0.60, 4.6))
check("saved value updated", sv.saved == 0.60, str(sv.saved))
check("does not write twice for one change", not sv.update(0.60, 9.0))
check("back to the saved value is not a change", not sv.update(0.60, 20.0))
sv2 = M.SettledValue(0.1, after_s=2.0)
sv2.update(0.95, 0.0)
check("returning to the old value cancels the pending write", not sv2.update(0.1, 0.5))
check("... and nothing is written later", not sv2.update(0.1, 10.0) and sv2.saved == 0.1, str(sv2.saved))

print("== startup pose check ==")
HOME = [0.0, 1.012291, -0.785398, 0.0, -0.226893, 0.0]
LIM = [(-3.1101767, 3.1101767), (-2.3038346, 2.3038346), (-3.1101767, 0.061087),
       (-3.1101767, 3.1101767), (-2.1642, 2.1642), (-3.1101767, 3.1101767)]
ok, why = M.startup_pose_ok(HOME, LIM)
check("home itself passes", ok, why)
ok, why = M.startup_pose_ok([0.0, -1.05, -1.84, 0.0, -0.2, 0.0], LIM)
check("the USD start pose passes", ok, why)
# anywhere the arm can physically hold is accepted: it got there under its own power
ok, why = M.startup_pose_ok([3.10, -2.30, -3.10, -3.10, 2.16, 3.10], LIM)
check("a pose at the far end of every joint passes", ok, why)
ok, why = M.startup_pose_ok([2.9, -2.0, -2.9, 1.5, 1.9, -2.5], LIM)
check("a wildly folded but legal pose passes", ok, why)
ok, why = M.startup_pose_ok(None, LIM)
check("no reading is refused", not ok and "controller" in why, why)
ok, why = M.startup_pose_ok([0.0, 1.0, -0.8], LIM)
check("a short vector is refused", not ok and "expected 6" in why, why)
ok, why = M.startup_pose_ok([0.0, float("nan"), -0.8, 0.0, 0.0, 0.0], LIM)
check("NaN is refused", not ok and "non-finite" in why, why)
ok, why = M.startup_pose_ok([0.0, 1.0, -0.8, 0.0, float("inf"), 0.0], LIM)
check("infinity is refused", not ok and "non-finite" in why, why)
# only a reading the joint cannot hold is refused: that means bad data, not a badly parked arm
ok, why = M.startup_pose_ok([0.0, -2.9, -0.785398, 0.0, -0.226893, 0.0], LIM)
check("an impossible joint angle is refused", not ok and "J2" in why, why)
check("the reason gives the angle and its limits", "deg" in why and ".." in why, why)
ok, why = M.startup_pose_ok([0.0, 1.012291, 0.9, 0.0, -0.226893, 0.0], LIM)
check("an asymmetric limit is respected (J3 max 3.5 deg)", not ok and "J3" in why, why)
ok, why = M.startup_pose_ok([0.0, 2.3038346 + 0.01, -0.8, 0.0, 0.0, 0.0], LIM)
check("a hair over the limit still passes on margin", ok, why)

print("== startup move: slower than an unwind, and it starts where the arm is ==")
REAL = np.array([0.0, -1.05, -1.84, 0.0, -0.2, 0.0])
mv_up = M.JointMove(REAL, np.asarray(HOME), 0.3, 0.5, 0.0)
expect = float(np.max(np.abs(np.asarray(HOME) - REAL))) / 0.3
check("duration = largest delta / startup rate", abs(mv_up.duration - expect) < 1e-9,
      f"{mv_up.duration:.2f} vs {expect:.2f}")
mv_un = M.JointMove(REAL, np.asarray(HOME), 0.8, 0.5, 0.0)
check("slower than an unwind", mv_up.duration > mv_un.duration, f"{mv_up.duration:.2f} vs {mv_un.duration:.2f}")
q0, done0 = mv_up.at(0.0)
check("starts at the measured pose", np.allclose(q0, REAL) and not done0, str(q0))
q1, done1 = mv_up.at(mv_up.duration)
check("ends at home", np.allclose(q1, HOME) and done1, str(q1))

print("== panel sliders: Advanced rows start at the Curious preset; both sway knobs sit under Motion Parameters ==")
from mimus import ui as UI  # noqa: E402
from unittest import mock  # noqa: E402
for key, _, lo, hi, dv in UI._ADV_PARAMS:
    check(f"{key} slider inits at the preset default, inside its range",
          abs(dv - M.BUILTIN_PRESETS["Curious"][key]) < 1e-9 and lo <= dv <= hi, f"{dv} vs {M.BUILTIN_PRESETS['Curious'].get(key)}")
for key, _, lo, hi, dv in UI._TRACKER_PARAMS:
    check(f"{key} Attention slider default inside its range", lo <= dv <= hi, f"{dv}")
# build the panel against a recording stub of omni.ui: the rows and their starting values, no Isaac needed
_rows = []


def _frame(title, **kw):
    _rows.append(("frame", title))
    return mock.MagicMock()


def _label(text, **kw):
    _rows.append(("label", text))
    return mock.MagicMock()


def _slider(**kw):
    w = mock.MagicMock()
    w.model.set_value.side_effect = lambda v: _rows.append(("value", round(float(v), 6)))
    _rows.append(("slider", (kw.get("min"), kw.get("max"))))
    return w


_ui = mock.MagicMock()
_ui.CollapsableFrame.side_effect = _frame
_ui.Label.side_effect = _label
_ui.FloatSlider.side_effect = _slider
UI.ui, UI._HAS_UI = _ui, True
UI.LiveDemoUI(UI.UIState()).build()
_i = _rows.index(("frame", "Motion Parameters"))
_j = next(k for k in range(_i + 1, len(_rows)) if _rows[k][0] == "frame")
_motion = _rows[_i + 1:_j]
_labels = [t for k, t in _motion if k == "label"]
check("Motion Parameters ends with the two sway knobs, amplitude first", _labels[-2:] == ["Sway amp (cm)", "Sway period (s)"], str(_labels))
check("sway amp slider starts at the preset, in cm", ("value", round(M.BUILTIN_PRESETS["Curious"]["sway_amp"] * 100.0, 6)) in _motion, str(_motion))
check("sway period slider starts at the preset", ("value", float(M.BUILTIN_PRESETS["Curious"]["sway_period"])) in _motion, str(_motion))
check("no sway row hides in Advanced", not any(k.startswith("sway_") for k, *_ in UI._ADV_PARAMS), str(UI._ADV_PARAMS))
_a = _rows.index(("frame", "Attention (tracker)"))
_b = next(k for k in range(_a + 1, len(_rows)) if _rows[k][0] == "frame")
_att = _rows[_a + 1:_b]
check("Attention rows: Near, Kids, Kid height, each starting at the tracker default",
      [t for k, t in _att if k == "label"][1:] == [p[1] for p in UI._TRACKER_PARAMS] and all(("value", p[4]) in _att for p in UI._TRACKER_PARAMS), str(_att))

print(f"\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
