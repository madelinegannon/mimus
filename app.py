# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Mimus: tracker presence -> behaviour -> RMPflow -> (mock or real) xArm 850, in Isaac Sim.

Three loops: the tracker process (UDP /tracker/target, see ASSEMBLY.md), this SIM/control loop at
60 fps or better, and the rt_xarm_bridge servo thread at 125 Hz. `--test` runs headless with a
synthetic tracker and the bridge's MockArm and checks the whole chain.

Run with Isaac Sim's Python:
  GUI, mock arm:   python.bat app.py [--config-dir config]
  Self-check:      python.bat app.py --test
  Real arm:        python.bat app.py --arm xarm --arm-ip <controller-ip> --allow-motion --safety safety.yaml
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)      # the mimus package next to this script, installed or not

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--test", action="store_true", help="headless self-check of the full chain (synthetic tracker + MockArm)")
ap.add_argument("--arm", choices=["mock", "xarm"], default="mock", help="mock (default) or a real controller")
ap.add_argument("--arm-ip", default="", help="controller IP; required with --arm xarm, no default")
ap.add_argument("--allow-motion", action="store_true",
                help="let the bridge arm and stream to a REAL arm (a mock arm is always allowed); needs --safety")
ap.add_argument("--safety", default=None, help="declared controller safety config (YAML) the bridge verifies before arming")
ap.add_argument("--engage", action="store_true", help="arm and start streaming at launch instead of monitoring only")
ap.add_argument("--auto-recover", action="store_true",
                help="supervised runs only: auto-clear transient controller faults, reconnect and re-arm")
ap.add_argument("--static-target", default=None, help="diagnostic: drive a fixed world x,y,z (position only)")
ap.add_argument("--mode", choices=["follow_target", "follow_sensor", "idle", "sleep", "home", "hold"], default=None,
                help="initial state (headless/CLI; the GUI sets it live). Default: follow_sensor under --test, else follow_target")
ap.add_argument("--agent", action="store_true", help="start with the Follow-Target spring agent ON")
ap.add_argument("--arm-start-offset", type=float, default=0.0,
                help="mock arm only: start it this many rad off the sim pose, to exercise the startup reconciliation")
ap.add_argument("--config-dir", default=os.path.join(THIS_DIR, "config"),
                help="dir holding app.json (+ app.local.json overlay) and behavior_presets.json")
ap.add_argument("--tracker-port", type=int, default=7001, help="OSC /tracker/target input port")
ap.add_argument("--hb-port", type=int, default=9000, help="/hb heartbeat UDP port on localhost (for a supervisor, if any)")
ap.add_argument("--q-port", type=int, default=7011, help="tracker robot-pose feed (/robot/q) UDP port on localhost")
ap.add_argument("--tracker-ctrl-port", type=int, default=7012, help="tracker control port (/tracker/set from the Attention sliders)")
ap.add_argument("--no-auto-mode", action="store_true", help="tracker hints do not switch IDLE/SLEEP/follow")
ap.add_argument("--no-sniff-latch", action="store_true", help="re-approach after every retreat")
args, _ = ap.parse_known_args()
if args.test:                          # the self-check never shares a port with a running stack
    if args.tracker_port == 7001:
        args.tracker_port = 7009
    if args.q_port == 7011:
        args.q_port = 7019
    if args.hb_port == 9000:
        args.hb_port = 9009
    if args.tracker_ctrl_port == 7012:
        args.tracker_ctrl_port = 7020
if args.arm == "xarm" and not args.arm_ip:
    ap.error("--arm xarm needs --arm-ip <controller-ip>; there is no default")
if args.arm == "xarm" and args.allow_motion and not args.safety:
    ap.error("--allow-motion on a real arm needs --safety <file> (see safety.example.yaml in rt_xarm_bridge)")
if args.arm == "xarm" and args.engage and not args.allow_motion:
    ap.error("--engage on a real arm needs --allow-motion")

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": bool(args.test)})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt  # noqa: E402

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import cylinder  # noqa: E402
from isaacsim.core.api.robots import Robot  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats  # noqa: E402
from isaacsim.robot_motion.motion_generation.articulation_motion_policy import ArticulationMotionPolicy  # noqa: E402
from isaacsim.robot_motion.motion_generation.interface_config_loader import _process_policy_config  # noqa: E402
from isaacsim.robot_motion.motion_generation.lula import RmpFlow  # noqa: E402

from rt_xarm_bridge import (DISCONNECTED, ENABLED, MotionNotAllowed, RecoveryPolicy, RobotBridge,  # noqa: E402
                            SafetyConfig, make_arm)
from mimus import motion as LM  # noqa: E402
from mimus.config import load_config, save_param  # noqa: E402
from mimus.osc import ControlLink, TelemetryOut  # noqa: E402
from mimus.tracker_link import TrackerLink  # noqa: E402

try:
    from isaacsim.util.debug_draw import _debug_draw
    _draw = _debug_draw.acquire_debug_draw_interface()
except Exception:  # noqa: BLE001
    _draw = None

# ----------------------------- configuration -------------------------------- #
USD_PATH = os.path.join(THIS_DIR, "usd", "uf850", "uf850.usd").replace("\\", "/")
ENV_USD_PATH = os.path.join(THIS_DIR, "usd", "environment", "default_environment.usd").replace("\\", "/")
RMPFLOW_CONFIG_JSON = os.path.join(THIS_DIR, "rmpflow", "config.json")
CONFIG_DIR = args.config_dir
if not os.path.exists(USD_PATH):
    print(f"[app] robot USD missing: {USD_PATH}\n      fetch the stock asset first: python.bat tools/fetch_assets.py", flush=True)
    simulation_app.close()
    sys.exit(2)


def _log_event(ev):
    """Bridge events -> one JSON line on stdout (a supervisor can tail this)."""
    print("[robot-event] " + json.dumps(ev, default=str), flush=True)


PRIM_PATH = "/World/uf850"
EE_FRAME = "link6"
AIM_AXIS = "z"                       # link6 approach axis


def _apply_config(cfg):
    global HOME_Q, SLEEP_Q, DEFAULT_Q, UNWIND_RATE_RAD_S, UNWIND_MIN_S, STARTUP_RATE_RAD_S
    global SPEED_SCALE, SPEED_SLEW_PER_S, REACH_MAX_M, DEFAULT_ANCHOR_Z
    HOME_Q = np.asarray(cfg["home_q_rad"], dtype=np.float64)
    SLEEP_Q = np.asarray(cfg["sleep_q_rad"], dtype=np.float64)
    DEFAULT_Q = HOME_Q
    UNWIND_RATE_RAD_S = float(cfg["unwind_rate_rad_s"])
    UNWIND_MIN_S = float(cfg["unwind_min_s"])
    STARTUP_RATE_RAD_S = float(cfg["startup_rate_rad_s"])
    SPEED_SCALE = max(0.0, min(1.0, float(cfg["speed_scale"])))
    SPEED_SLEW_PER_S = float(cfg["speed_slew_per_s"])
    REACH_MAX_M = float(cfg["reach_max_m"])
    DEFAULT_ANCHOR_Z = float(cfg["anchor_z_m"])     # a world height: it does not move with the mount


CFG = load_config(CONFIG_DIR)
_apply_config(CFG)
print(f"[app] config ({CFG['_source']}): home={np.round(HOME_Q, 3).tolist()} sleep={np.round(SLEEP_Q, 3).tolist()} "
      f"unwind {UNWIND_RATE_RAD_S} rad/s startup {STARTUP_RATE_RAD_S} rad/s speed {SPEED_SCALE:.0%} "
      f"anchor_z {DEFAULT_ANCHOR_Z} reach_max {REACH_MAX_M}", flush=True)

# The robot base: on the floor at the world origin unless the config says otherwise (a pedestal height, a yaw).
MOUNT = np.array([float(CFG["base_xy"][0]), float(CFG["base_xy"][1]), float(CFG["mount_z"])])
_yaw = np.radians(float(CFG["base_yaw_deg"]))
MOUNT_QUAT = np.array([np.cos(_yaw / 2), 0.0, 0.0, np.sin(_yaw / 2)])   # (w, x, y, z)
BASE_XY = (float(MOUNT[0]), float(MOUNT[1]))
print(f"[app] geometry: mount={np.round(MOUNT, 3).tolist()} yaw={CFG['base_yaw_deg']} deg (from {CONFIG_DIR})", flush=True)

STALE_S = 0.5
STATIC_TARGET = (np.array([float(v) for v in args.static_target.split(",")]) if args.static_target else None)
CLI_MODE = LM.default_mode(args.arm, args.engage, args.test, args.mode)
CLI_AGENT = bool(args.agent) or (args.test and CLI_MODE == LM.FOLLOW_SENSOR)


# ------------------------------ scene build --------------------------------- #
add_reference_to_stage(usd_path=USD_PATH, prim_path=PRIM_PATH)
world = World(stage_units_in_meters=1.0)
if os.path.exists(ENV_USD_PATH):
    add_reference_to_stage(usd_path=ENV_USD_PATH, prim_path="/World/environment")
else:
    world.scene.add_default_ground_plane()
robot = world.scene.add(Robot(prim_path=PRIM_PATH, name="uf850"))
stage = omni.usd.get_context().get_stage()

_GRAY = np.array([0.55, 0.55, 0.58])
if MOUNT[2] > 0.01:      # a raised base gets a visual pedestal from the floor to just under the flange
    cylinder.VisualCylinder("/World/pedestal", position=np.array([MOUNT[0], MOUNT[1], (MOUNT[2] - 0.01) / 2]),
                            radius=0.075, height=MOUNT[2] - 0.01, color=_GRAY)


def _set_refinement(prim_path, level):
    """Increase render tessellation so analytic cylinders look round, not faceted."""
    p = stage.GetPrimAtPath(prim_path)
    if p.IsValid():
        p.CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)
        p.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(level)


_set_refinement("/World/pedestal", 2)

# Lookat sphere: world-parented so the operator drags it in world coordinates.
_look = UsdGeom.Sphere.Define(stage, "/World/lookat")
_look.GetRadiusAttr().Set(0.05)
_look.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.1, 0.9, 0.3)]))
lookat_translate_op = UsdGeom.Xformable(_look.GetPrim()).AddTranslateOp()
lookat_translate_op.Set(Gf.Vec3d(1.5, 0.0, float(MOUNT[2]) + 1.0))
lookat_prim = _look.GetPrim()

# Tracker target sphere: the raw point the tracker reports (before behaviour processing).
_stgt = UsdGeom.Sphere.Define(stage, "/World/tracker_target")
_stgt.GetRadiusAttr().Set(0.03)
_stgt.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.5, 0.0)]))
tracker_target_translate_op = UsdGeom.Xformable(_stgt.GetPrim()).AddTranslateOp()
tracker_target_translate_op.Set(Gf.Vec3d(0, 0, -10))

# Gaze agent viz: the smoothed lookat position (orientation reference).
_gaze_viz = UsdGeom.Sphere.Define(stage, "/World/gaze_viz")
_gaze_viz.GetRadiusAttr().Set(0.04)
_gaze_viz.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.9, 1.0)]))
gaze_viz_translate_op = UsdGeom.Xformable(_gaze_viz.GetPrim()).AddTranslateOp()
gaze_viz_translate_op.Set(Gf.Vec3d(0, 0, -10))

# Target cube: world-parented, draggable in FollowTarget, app-written in FollowTargetSensor.
_raw_cube = UsdGeom.Cube.Define(stage, "/World/target_raw")
_raw_cube.GetSizeAttr().Set(0.08)
_raw_cube.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.1, 0.1)]))
_raw_xf = UsdGeom.Xformable(_raw_cube.GetPrim())
raw_translate_op = _raw_xf.AddTranslateOp()
raw_translate_op.Set(Gf.Vec3d(0.4, 0.0, float(MOUNT[2]) + 0.5))
raw_orient_op = _raw_xf.AddOrientOp()
raw_orient_op.Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
raw_prim = _raw_cube.GetPrim()

_xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())


def read_world_point(prim) -> np.ndarray:
    t = _xf_cache.GetLocalToWorldTransform(prim).ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=np.float64)


def read_world_quat(prim) -> np.ndarray:
    """World orientation of `prim` as [w, x, y, z]."""
    q = _xf_cache.GetLocalToWorldTransform(prim).ExtractRotationQuat().GetNormalized()
    im = q.GetImaginary()
    return np.array([q.GetReal(), im[0], im[1], im[2]], dtype=np.float64)


def write_cube_world_pos(world_pos):
    raw_translate_op.Set(Gf.Vec3d(*[float(v) for v in world_pos]))


def set_cube_orientation(aim_q):
    raw_orient_op.Set(Gf.Quatf(float(aim_q[0]), Gf.Vec3f(float(aim_q[1]), float(aim_q[2]), float(aim_q[3]))))


_LOCKED_PRIMS = ["/World/environment", "/World/pedestal"]


def lock_environment():
    """Ground and pedestal are not selectable in the viewport (runtime UsdContext state; re-applied after a reset)."""
    if args.test:
        return
    try:
        ctx = omni.usd.get_context()
        for p in _LOCKED_PRIMS:
            if stage.GetPrimAtPath(p).IsValid():
                ctx.set_pickable(p, False)
    except Exception as e:  # noqa: BLE001
        print(f"[app] WARN: set_pickable failed ({e})", flush=True)


# ------------------------------ RMPflow ------------------------------------- #
rmp_config = _process_policy_config(RMPFLOW_CONFIG_JSON)
rmpflow = RmpFlow(**rmp_config)
rmpflow.set_robot_base_pose(MOUNT, MOUNT_QUAT)
amp = ArticulationMotionPolicy(robot, rmpflow, 1.0 / 60.0)
art_ctrl = robot.get_articulation_controller()
kin = LulaKinematicsSolver(robot_description_path=rmp_config["robot_description_path"], urdf_path=rmp_config["urdf_path"])
kin.set_robot_base_pose(MOUNT, MOUNT_QUAT)

# ------------------------------ networking ---------------------------------- #
tracker_link = TrackerLink(port=args.tracker_port)
telemetry = TelemetryOut(hb_addr=("127.0.0.1", args.hb_port), q_addr=("127.0.0.1", args.q_port),
                         ctrl_addr=("127.0.0.1", args.tracker_ctrl_port))

# ------------------------------ test rig ------------------------------------ #
fake_tracker = None
if args.test:
    from mimus.fake_tracker import FakeTracker
    fake_tracker = FakeTracker(port=args.tracker_port, hz=60.0)

# ------------------------------- runtime ------------------------------------ #
world.reset()
robot.set_world_pose(position=MOUNT, orientation=MOUNT_QUAT)
robot.set_default_state(position=MOUNT, orientation=MOUNT_QUAT)
robot.get_articulation_controller().set_gains(kps=np.array([1.0e6] * 6), kds=np.array([1.0e5] * 6))
_stage = omni.usd.get_context().get_stage()
_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
_max_forces = {}
for _jn in _joint_names:
    _jp = _stage.GetPrimAtPath(f"{PRIM_PATH}/{_jn}")
    if not _jp.IsValid():
        for _c in _stage.Traverse():
            if _c.GetName() == _jn:
                _jp = _c
                break
    _drv = UsdPhysics.DriveAPI.Get(_jp, "angular")
    _max_forces[_jn] = _drv.GetMaxForceAttr().Get() if _drv else None
print(f"[app] USD max_forces: {_max_forces}", flush=True)
init_q = np.asarray(robot.get_joint_positions()[:6], dtype=np.float64)
print("[app] robot root world pos:", np.round(robot.get_world_pose()[0], 3), "  init_q:", np.round(init_q, 2), flush=True)

lock_environment()

if not args.test:
    try:
        import carb
        vp = __import__("omni.kit.viewport.utility", fromlist=["get_active_viewport"]).get_active_viewport()
        if vp:
            carb.settings.get_settings().set_bool(f"/persistent/app/viewport/{vp.id}/fillViewport", True)
    except Exception:  # noqa: BLE001
        pass

# ------------------------------ the arm bridge ------------------------------ #
_mock_q0 = (init_q + float(args.arm_start_offset)).tolist()   # 0 = the mock starts where the sim does
arm = make_arm(args.arm, ip=args.arm_ip, initial_q=_mock_q0)
ALLOW_MOTION = bool(args.allow_motion) or args.arm == "mock"   # a mock arm cannot hurt anyone
if args.safety:
    SAFETY = SafetyConfig.load(args.safety)
elif args.arm == "mock":
    SAFETY = arm.safety_config()
else:
    SAFETY = None
bridge = RobotBridge(arm, allow_motion=ALLOW_MOTION, safety=SAFETY, initial_speed_scale=SPEED_SCALE,
                     rate_hz=125.0, watchdog_s=STALE_S, speed_slew_per_s=SPEED_SLEW_PER_S,
                     auto_arm=bool(args.auto_recover) and ALLOW_MOTION,
                     recovery=RecoveryPolicy() if args.auto_recover else None,
                     reconnect=bool(args.auto_recover), on_event=_log_event)
if args.arm == "mock" or args.engage:
    bridge.connect()          # MONITORING + armed (CONNECTED) when the safety check passes
else:
    bridge.soft_connect()     # a real arm without --engage: monitor until the operator arms it in the panel

enabled = args.engage or args.test


def _reconcile_with_arm():
    """The sim starts at whatever pose the USD authored; the real arm starts wherever it was left. Enable at
    the arm's MEASURED pose (not the sim's) or the first servo packet teleports it, then ease to home in
    `control_step` before anything else drives. Returns the startup move, or None."""
    rq, deadline = None, time.monotonic() + 2.0    # the bridge thread reads joints at 125 Hz
    while time.monotonic() < deadline:
        rq = bridge.latest_real_q()
        if rq is not None and rq[1] <= 1.0:
            break
        rq = None
        time.sleep(0.05)
    real_q = None if rq is None else rq[0]
    ok, why = LM.startup_pose_ok(real_q, bridge.joint_limits)     # the controller's limits intersected with the URDF table
    if not ok:
        bridge.halt(f"startup pose check: {why}")
        return None
    _snap_sim_to_real()                            # twin, RMPflow seed and cube all follow the arm
    if not bridge.enable(sync_q=list(real_q[:6])):
        return None
    mv = LM.JointMove(np.asarray(real_q[:6], dtype=np.float64), HOME_Q, STARTUP_RATE_RAD_S, UNWIND_MIN_S, time.perf_counter())
    print(f"[app] startup move -> home over {mv.duration:.1f} s at {STARTUP_RATE_RAD_S} rad/s", flush=True)
    return mv


if fake_tracker is not None:
    fake_tracker.start()

_home_pos, _home_rot = rmpflow.get_end_effector_pose(HOME_Q)
HOME_EE = np.asarray(_home_pos, dtype=np.float64).ravel()
HOME_QUAT = rot_matrices_to_quats(_home_rot)

# ---- motion behaviour: spring agent + mode-dispatched FSM ----
BEHAVIOR_PRESETS_FILE = os.path.join(CONFIG_DIR, "behavior_presets.json")
_presets = LM.load_presets(BEHAVIOR_PRESETS_FILE)
_preset_name = "Curious"
if args.no_sniff_latch:
    for _p in _presets.values():
        _p["sniff_once"] = 0.0
behavior = LM.Behavior(BASE_XY, HOME_EE, params=_presets[_preset_name], aim_axis=AIM_AXIS, base_z=float(MOUNT[2]),
                       base_yaw_deg=float(CFG["base_yaw_deg"]), reach_max=REACH_MAX_M)
print(f"[app] heights: anchor={DEFAULT_ANCHOR_Z:.3f} (world; reach_max {REACH_MAX_M:.2f} about the shoulder) "
      f"near={behavior.base_z + behavior.params['height_near_dz']:.3f} idle={float(behavior.params.get('idle_ee_z', 1.75)):.3f} "
      f"sleep_ee_z={float(np.asarray(rmpflow.get_end_effector_pose(SLEEP_Q)[0]).ravel()[2]):.3f} "
      f"(base flange at {MOUNT[2]:.3f}: near is an offset from it, idle a world height)", flush=True)
_q0 = amp.get_active_joints_subset().get_joint_positions()
_ee0, _ = rmpflow.get_end_effector_pose(_q0)
behavior.seed(np.asarray(_ee0, dtype=np.float64))

_spheres_shown = False

# GUI control panel; headless --test skips it.
ui_state = None
ui_panel = None
control = None
if not args.test:
    from mimus.ui import LiveDemoUI, _TRACKER_PARAMS
    from mimus.ui_state import UIState
    ui_state = UIState()
    ui_state.anchor_z = DEFAULT_ANCHOR_Z
    ui_state.reach_max = REACH_MAX_M
    ui_state.arm_ip = args.arm_ip or "(mock)"
    ui_state.mode = CLI_MODE
    ui_state.speed_scale = bridge.speed_scale   # the bridge capped the configured value: the slider starts there
    ui_state.tracker_params = {k: float(CFG["tracker_params"][k]) for k in [p[0] for p in _TRACKER_PARAMS] if k in CFG.get("tracker_params", {})}
    control = ControlLink(ui_state)
    ui_panel = LiveDemoUI(ui_state, control=control).build(joint_limits=bridge.joint_limits)
    if bridge.state != DISCONNECTED:
        ui_panel._conn_toggle_v["v"] = True
        ui_panel._conn_toggle_btn.text = "Connected: ON"
        ui_panel._conn_toggle_btn.style = {"background_color": 0xFF2E7D32, "color": 0xFFFFFFFF}

_hold_q = None   # joint positions to hold at (set by snap or HOLD mode entry)
_home_q = None   # current interpolated joints during HOME (joint-space unwind)
_sleep_q = None  # current joints during SLEEP (home -> settle -> breathe), same law as HOME
_sleep_stage = "home"
_sleep_t0 = None
_home_move = None
_sleep_move = None
_startup_move = None  # JointMove from the arm's measured startup pose to home; owns the arm until it ends


def _snap_sim_to_real():
    """Read real joint positions and apply them to the sim robot. Returns the pose, or None without a readback."""
    global _hold_q
    rq = bridge.latest_real_q()
    if rq is None:
        return None
    q_real = np.array(rq[0], dtype=np.float64)
    robot.set_joint_positions(q_real)
    try:
        robot.set_joint_velocities(np.zeros(6))
    except Exception:  # noqa: BLE001  the position snap is what matters
        pass
    ee_snap, ee_snap_rot = rmpflow.get_end_effector_pose(q_real)
    ee_snap = np.asarray(ee_snap, dtype=np.float64).ravel()
    ee_snap_quat = rot_matrices_to_quats(ee_snap_rot)
    behavior.seed(ee_snap)
    _hold_q = q_real.copy()
    write_cube_world_pos(ee_snap)
    set_cube_orientation(ee_snap_quat)
    print(f"[app] SNAP sim->real: {np.round(np.degrees(q_real), 1)} deg", flush=True)
    return q_real


if enabled:
    _startup_move = _reconcile_with_arm()
    enabled = bridge.state == ENABLED      # a refused startup pose or safety check leaves us connected but not driving


def _save_pose(name, q):
    save_param(CONFIG_DIR, name, [float(v) for v in np.asarray(q, dtype=np.float64).ravel()[:6]])
    _reload_config()


def _poll_ui():
    """Apply UI flags to bridge/scene each frame (GUI mode)."""
    global enabled, _spheres_shown, _preset_name, _presets, _hold_q, _last_fwd_error
    if ui_state is None:
        return
    bridge.speed_scale = ui_state.speed_scale      # the slider is the operator's runtime cap; never persisted
    if ui_state.consume_connect():
        try:
            bridge.soft_connect()
            _snap_sim_to_real()
        except Exception as e:  # noqa: BLE001
            bridge._capture_error("connect", e)
    if ui_state.consume_disconnect():
        bridge.disconnect()
        enabled = False
    if ui_state.consume_arm():
        try:
            if bridge.arm_robot():
                _snap_sim_to_real()
                cur = amp.get_active_joints_subset().get_joint_positions()
                enabled = bridge.enable(sync_q=np.asarray(cur, dtype=np.float64).tolist())
        except MotionNotAllowed as e:
            bridge._capture_error("arm", e)      # started without --allow-motion: the panel shows why
    if ui_state.consume_disarm():
        bridge.stop()
        enabled = False
    if ui_state.consume_recover():
        bridge.recover()
    if ui_state.consume_stop_arm():
        bridge.stop()
        enabled = False
    if ui_state.consume_estop():
        bridge.emergency_stop()
        enabled = False
    if ui_state.consume_snap_to_real():
        if bridge.state == DISCONNECTED:
            bridge._capture_error("snap", RuntimeError("Not connected"))
        else:
            _snap_sim_to_real()
    if ui_state.consume_stop():
        bridge.stop()
        enabled = False
    cube_req = ui_state.consume_cube()
    if cube_req is not None:
        write_cube_world_pos(np.asarray(cube_req, dtype=np.float64))
    pn = ui_state.consume_preset()
    if pn:
        _preset_name = pn
        behavior.set_params(_presets.get(pn, LM.BUILTIN_PRESETS[pn]))
    adv = ui_state.consume_params()
    if adv:
        if "reach_max" in adv:                      # the sphere lives on the behaviour, not in the preset
            behavior.reach_max = float(adv["reach_max"])
        behavior.set_params({k: v for k, v in adv.items() if k != "reach_max"})
        print(f"[app] params {adv} (slider)", flush=True)
    tune = ui_state.consume_tracker_params()
    if tune:
        for _k, _v in tune.items():                 # the tracker applies it at once
            telemetry.push_tracker_set(_k, _v)
        print(f"[app] tracker {tune} (slider)", flush=True)
    if ui_state.consume_save_preset():
        LM.save_preset(BEHAVIOR_PRESETS_FILE, _preset_name, behavior.params)
        _presets = LM.load_presets(BEHAVIOR_PRESETS_FILE)
    if ui_state.consume_reset_preset():
        behavior.set_params(LM.BUILTIN_PRESETS[_preset_name])
    for _pose_name, _consume in (("home_q_rad", ui_state.consume_set_home_pose), ("sleep_q_rad", ui_state.consume_set_sleep_pose)):
        if _consume():
            _save_pose(_pose_name, amp.get_active_joints_subset().get_joint_positions())
    if ui_state.show_collision_spheres != _spheres_shown:
        _spheres_shown = ui_state.show_collision_spheres
        if _spheres_shown:
            rmpflow.visualize_collision_spheres()
        else:
            rmpflow.stop_visualizing_collision_spheres()
    if control is not None and bridge.last_error and bridge.last_error != _last_fwd_error:
        control.push_error(bridge.last_error)
        _last_fwd_error = bridge.last_error


_last_fwd_error = None
_fps_ema = None
_cs_ema = None


def _draw_debug(ee_world, target_world, dbg, mode=None):
    """Batched debug overlay: always the EE->target line; magenta anchor lines in sensor mode;
    plus (toggle-gated) the spring agent and seek target."""
    if _draw is None or args.test:
        return
    _draw.clear_lines()
    _draw.clear_points()
    va = ui_state.viz_agent if ui_state is not None else False
    vt = ui_state.viz_targets if ui_state is not None else False
    vn = ui_state.viz_anchor if ui_state is not None else False
    starts = [tuple(float(v) for v in ee_world)]
    ends = [tuple(float(v) for v in target_world)]
    lcol = [(1.0, 1.0, 0.0, 1.0)]
    lwid = [2.0]
    pts, pcol, psz = [], [], []
    _MAG = (1.0, 0.0, 1.0, 1.0)
    _CYAN = (0.0, 1.0, 1.0, 1.0)
    if mode in (LM.FOLLOW_SENSOR, LM.IDLE):
        anchor = tuple(float(v) for v in dbg["anchor"])
        ground = (anchor[0], anchor[1], 0.0)
        starts.append(ground); ends.append(anchor); lcol.append(_MAG); lwid.append(2.0)
        starts.append(anchor); ends.append(tuple(float(v) for v in dbg["lookat"]))
        lcol.append(_MAG); lwid.append(2.0)
        pts.append(anchor); pcol.append(_MAG); psz.append(9.0)
        if dbg.get("reach_limit") is not None:
            pts.append(tuple(float(v) for v in dbg["reach_limit"]))
            pcol.append(_CYAN); psz.append(13.0)
    elif vn:
        starts.append(tuple(float(v) for v in dbg["anchor"]))
        ends.append(tuple(float(v) for v in dbg["lookat"]))
        lcol.append((0.2, 0.8, 1.0, 1.0)); lwid.append(2.0)
        pts.append(tuple(float(v) for v in dbg["anchor"])); pcol.append((1.0, 1.0, 1.0, 1.0)); psz.append(9.0)
    if dbg.get("retreating"):
        pts.append(tuple(float(v) for v in ee_world)); pcol.append((1.0, 0.2, 0.2, 1.0)); psz.append(18.0)
    _YELLOW = (1.0, 1.0, 0.0, 1.0)
    _ORANGE = (1.0, 0.5, 0.0, 1.0)
    if va:
        pts.append(tuple(float(v) for v in dbg["agent_pos"])); pcol.append(_YELLOW); psz.append(13.0)
        trail = dbg.get("trail", [])
        for i in range(len(trail) - 1):
            starts.append(tuple(float(v) for v in trail[i]))
            ends.append(tuple(float(v) for v in trail[i + 1]))
            lcol.append(_YELLOW); lwid.append(1.5)
        if "gaze_lookat" in dbg:
            gp = tuple(float(v) for v in dbg["gaze_lookat"])
            pts.append(gp); pcol.append(_CYAN); psz.append(13.0)
            gaze_trail = dbg.get("gaze_trail", [])
            for i in range(len(gaze_trail) - 1):
                starts.append(tuple(float(v) for v in gaze_trail[i]))
                ends.append(tuple(float(v) for v in gaze_trail[i + 1]))
                lcol.append(_CYAN); lwid.append(1.5)
            _WHITE = (1.0, 1.0, 1.0, 1.0)
            starts.append(tuple(float(v) for v in dbg["agent_pos"]))
            ends.append(gp)
            lcol.append(_WHITE); lwid.append(3.0)
    if dbg.get("agent_on") and mode in (LM.FOLLOW_SENSOR, LM.IDLE):
        starts.append(tuple(float(v) for v in dbg["agent_pos"]))
        ends.append(tuple(float(v) for v in dbg.get("gaze_lookat", dbg["lookat"])))
        lcol.append(_ORANGE); lwid.append(2.0)
    if vt:
        pts.append(tuple(float(v) for v in dbg["seek_target"])); pcol.append(_ORANGE); psz.append(11.0)
    _draw.draw_lines(starts, ends, lcol, lwid)
    if pts:
        _draw.draw_points(pts, pcol, psz)


_last_action_q = None
_prev_mode = None


def _handoff_from_joints(q):
    """Leaving a joint-space mode (HOLD / HOME / SLEEP): RMPflow, the behaviour and the cube all restart at q."""
    _ee_pos, _ee_rot = rmpflow.get_end_effector_pose(q)
    _ee_pos = np.asarray(_ee_pos, dtype=np.float64).ravel()
    _ee_quat = rot_matrices_to_quats(_ee_rot)
    robot.set_joint_positions(q)
    rmpflow.set_internal_robot_joint_states(np.asarray(q, dtype=np.float64).copy(), np.zeros(6), np.empty(0), np.empty(0))
    rmpflow.set_end_effector_target(target_position=_ee_pos, target_orientation=_ee_quat)
    rmpflow.update_world()
    behavior.seed(_ee_pos)
    write_cube_world_pos(_ee_pos)
    set_cube_orientation(_ee_quat)


def _reload_config():
    global CFG, HOME_EE, HOME_QUAT
    CFG = load_config(CONFIG_DIR)
    _apply_config(CFG)
    behavior.reach_max = REACH_MAX_M
    if ui_state is not None:
        ui_state.anchor_z = DEFAULT_ANCHOR_Z
        ui_state.reach_max = REACH_MAX_M
    _p, _r = rmpflow.get_end_effector_pose(HOME_Q)
    HOME_EE, HOME_QUAT = np.asarray(_p, dtype=np.float64).ravel(), rot_matrices_to_quats(_r)


def _finish_step(t_cs, ee):
    global _cs_ema
    cs_ms = (time.perf_counter() - t_cs) * 1e3
    _cs_ema = cs_ms if _cs_ema is None else 0.9 * _cs_ema + 0.1 * cs_ms
    return ee


def control_step(dt: float):
    global _hold_q, _home_q, _sleep_q, _sleep_stage, _sleep_t0, _home_move, _sleep_move, _last_action_q, _prev_mode, _startup_move, _mode_override
    _xf_cache.Clear()
    t_cs = time.perf_counter()

    # --- startup: ease from the pose the arm was actually in to home before any mode drives it ---
    if _startup_move is not None:
        q_up, done = _startup_move.at(t_cs)
        art_ctrl.apply_action(ArticulationAction(joint_positions=q_up))
        if enabled:
            bridge.publish_desired_q(q_up[:6])
        ee_up, _ = rmpflow.get_end_effector_pose(q_up)
        ee_up = np.asarray(ee_up, dtype=np.float64).ravel()
        _draw_debug(ee_up, ee_up, dict(agent_pos=ee_up, seek_target=ee_up, anchor=np.array([0.0, 0.0, 1.0]), lookat=ee_up))
        if done:
            _startup_move = None
            _mode_override = None          # release: the configured mode and the tracker take over now
            _handoff_from_joints(q_up)
            print(f"[app] startup move done, at home; mode -> {ui_state.mode if ui_state else CLI_MODE}", flush=True)
        return _finish_step(t_cs, ee_up)

    if STATIC_TARGET is not None:
        rmpflow.set_end_effector_target(target_position=STATIC_TARGET, target_orientation=None)
        rmpflow.update_world()
        action = amp.get_next_articulation_action()
        art_ctrl.apply_action(action)
        if action.joint_positions is not None:
            bridge.publish_desired_q(np.asarray(action.joint_positions, dtype=np.float64)[:6])
        return _finish_step(t_cs, STATIC_TARGET)

    mode = _mode_override or (ui_state.mode if ui_state is not None else CLI_MODE)
    mode_reason = "override" if _mode_override is not None else "command"

    # --- tracker-driven attention: the tracker owns IDLE / SLEEP / FollowTargetSensor (it decides both who to
    # watch and whether anyone is worth being awake for); operator states (HOLD, HOME, FollowTarget) are never touched ---
    trk = tracker_link.latest_target()
    trk_fresh = trk is not None and trk[4] <= STALE_S
    auto = (ui_state.auto_mode if ui_state is not None else True) and not args.no_auto_mode
    if trk_fresh and auto and _mode_override is None and mode in (LM.IDLE, LM.FOLLOW_SENSOR, LM.SLEEP):
        want = LM.mode_for_hint(trk[1])
        if want != mode:
            mode = want
            mode_reason = f"tracker hint '{trk[1]}'"
            if ui_state is not None:
                ui_state.mode = want

    # --- mode changed: drop the joint state of the mode we left, so re-entering it eases from where we are ---
    if mode != _prev_mode:
        print(f"[app] mode {_prev_mode} -> {mode} ({mode_reason})", flush=True)
        left = {LM.HOLD: _hold_q, LM.HOME: _home_q, LM.SLEEP: _sleep_q}.get(_prev_mode)
        if left is not None:
            _handoff_from_joints(left)
        _hold_q = None
        _home_q = _home_move = None
        _sleep_q = _sleep_move = _sleep_stage = _sleep_t0 = None
        _prev_mode = mode

    # --- HOLD: bypass RMPflow, apply stored joints directly via the PD controller ---
    if mode == LM.HOLD:
        if _hold_q is None:
            _hold_q = np.asarray(amp.get_active_joints_subset().get_joint_positions(), dtype=np.float64)
        ee_hold, _ = rmpflow.get_end_effector_pose(_hold_q)
        ee_hold = np.asarray(ee_hold, dtype=np.float64).ravel()
        art_ctrl.apply_action(ArticulationAction(joint_positions=_hold_q))
        if enabled:
            bridge.publish_desired_q(_hold_q[:6])
        _draw_debug(ee_hold, ee_hold, dict(agent_pos=ee_hold, seek_target=ee_hold, anchor=np.array([0.0, 0.0, 1.0]), lookat=ee_hold))
        return _finish_step(t_cs, ee_hold)

    # --- HOME: bypass RMPflow, one timed joint-space move (smoothstep ease-in-out) to the home pose ---
    if mode == LM.HOME:
        if _home_q is None:
            _home_q = np.asarray(amp.get_active_joints_subset().get_joint_positions(), dtype=np.float64)
            _home_move = LM.JointMove(_home_q, HOME_Q, UNWIND_RATE_RAD_S, UNWIND_MIN_S, time.perf_counter())
            write_cube_world_pos(HOME_EE)
            set_cube_orientation(HOME_QUAT)
            rmpflow.set_end_effector_target(target_position=HOME_EE, target_orientation=HOME_QUAT)
        _home_q, _ = _home_move.at(time.perf_counter())
        ee_home, _ = rmpflow.get_end_effector_pose(_home_q)
        ee_home = np.asarray(ee_home, dtype=np.float64).ravel()
        art_ctrl.apply_action(ArticulationAction(joint_positions=_home_q))
        if enabled:
            bridge.publish_desired_q(_home_q[:6])
        _draw_debug(ee_home, ee_home, dict(agent_pos=ee_home, seek_target=ee_home, anchor=np.array([0.0, 0.0, 1.0]), lookat=ee_home))
        return _finish_step(t_cs, ee_home)

    # --- SLEEP: joint space, staged: home (unwind J4/J6 every time) -> settle into the sleep pose -> breathe ---
    if mode == LM.SLEEP:
        if _sleep_q is None:
            _sleep_q = np.asarray(amp.get_active_joints_subset().get_joint_positions(), dtype=np.float64)
            _sleep_stage, _sleep_t0 = "home", None
            _sleep_move = LM.JointMove(_sleep_q, HOME_Q, UNWIND_RATE_RAD_S, UNWIND_MIN_S, time.perf_counter())
            _sp, _sr = rmpflow.get_end_effector_pose(SLEEP_Q)
            write_cube_world_pos(np.asarray(_sp, dtype=np.float64).ravel())
            set_cube_orientation(rot_matrices_to_quats(_sr))
        if _sleep_stage == "home":
            _sleep_q, _done = _sleep_move.at(time.perf_counter())
            if _done:
                _sleep_stage = "settle"
                _sleep_move = LM.JointMove(HOME_Q, SLEEP_Q, UNWIND_RATE_RAD_S, UNWIND_MIN_S, time.perf_counter())
        elif _sleep_stage == "settle":
            _sleep_q, _done = _sleep_move.at(time.perf_counter())
            if _done:
                _sleep_stage, _sleep_t0 = "breathe", time.perf_counter()
        else:
            _sleep_q = LM.sleep_joint_pose(time.perf_counter() - _sleep_t0, SLEEP_Q, behavior.params)
        ee_sleep, _ = rmpflow.get_end_effector_pose(_sleep_q)
        ee_sleep = np.asarray(ee_sleep, dtype=np.float64).ravel()
        art_ctrl.apply_action(ArticulationAction(joint_positions=_sleep_q))
        if enabled:
            bridge.publish_desired_q(_sleep_q[:6])
        _draw_debug(ee_sleep, ee_sleep, dict(agent_pos=ee_sleep, seek_target=ee_sleep, anchor=np.array([0.0, 0.0, 1.0]), lookat=ee_sleep))
        return _finish_step(t_cs, ee_sleep)

    q = amp.get_active_joints_subset().get_joint_positions()
    ee_now, _ = rmpflow.get_end_effector_pose(q)
    ee_now = np.asarray(ee_now, dtype=np.float64)

    agent_on = ui_state.agent_on if ui_state is not None else CLI_AGENT
    anchor_z = ui_state.anchor_z if ui_state is not None else DEFAULT_ANCHOR_Z

    # --- the tracker's target is already WORLD metres (base-frame origin on the floor, Z-up): no transform ---
    if trk_fresh and trk[1] not in LM.AUTO_MODES:
        target = np.asarray(trk[0], dtype=np.float64)
        lookat_translate_op.Set(Gf.Vec3d(*target))
        tracker_target_translate_op.Set(Gf.Vec3d(*target))
    lookat = read_world_point(lookat_prim)

    cube_pose = (read_world_point(raw_prim), read_world_quat(raw_prim))
    if ui_state is not None:
        behavior.params["gaze_steer"] = ui_state.gaze_steer

    ee_target, aim_q, dbg = behavior.update(mode, cube_pose, lookat, anchor_z, agent_on, ee_now, dt,
                                            track_id=(trk[2] if trk_fresh else None), hint=(trk[1] if trk_fresh else None))

    if mode == LM.FOLLOW_SENSOR:
        write_cube_world_pos(ee_target)
        set_cube_orientation(aim_q)
    elif mode == LM.IDLE:
        lookat_translate_op.Set(Gf.Vec3d(*[float(v) for v in dbg["lookat"]]))
    if "gaze_lookat" in dbg:
        gaze_viz_translate_op.Set(Gf.Vec3d(*[float(v) for v in dbg["gaze_lookat"]]))

    ee_target = LM.clamp_reach(ee_target, behavior.shoulder, behavior.reach_max)[0]   # nothing beyond the sphere reaches RMPflow
    rmpflow.set_end_effector_target(target_position=ee_target, target_orientation=aim_q)
    rmpflow.update_world()
    action = amp.get_next_articulation_action()
    art_ctrl.apply_action(action)
    _last_action_q = np.asarray(action.joint_positions, dtype=np.float64)[:6] if action.joint_positions is not None else None
    if action.joint_positions is not None:
        bridge.publish_desired_q(np.asarray(action.joint_positions, dtype=np.float64)[:6])

    _draw_debug(ee_now, ee_target, dbg, mode)
    return _finish_step(t_cs, ee_target)


# ------------------------------ run plumbing -------------------------------- #
# HOME while the startup move runs, so the panel and the heartbeat say HOME until the arm is actually there
_mode_override = LM.HOME if _startup_move is not None else None
_override_until = 0.0
_quit_requested = False
_exited = False
POST_RECOVERY_HOME_S = 6.0      # a full unwind at 0.8 rad/s takes under 4 s


def _safe_exit():
    """Whatever kills us (quit, Ctrl-C, taskkill, crash): stop streaming first so the arm holds where it is,
    then drop the link. Idempotent."""
    global _exited
    if _exited:
        return
    _exited = True
    try:
        if bridge.state == ENABLED:
            bridge.stop()
        bridge.shutdown()
    except Exception as e:  # noqa: BLE001
        print(f"[app] safe_exit: {e}", flush=True)


def _on_signal(signum, frame_):
    global _quit_requested
    print(f"[app] signal {signum}: quitting", flush=True)
    _quit_requested = True


atexit.register(_safe_exit)
for _sig in [s for s in ("SIGINT", "SIGTERM", "SIGBREAK") if hasattr(signal, s)]:
    try:
        signal.signal(getattr(signal, _sig), _on_signal)
    except (ValueError, OSError):
        pass


def _push_robot_q():
    """The tracker masks the robot out of its depth clouds from this feed, so it has to be where the arm
    ACTUALLY is: the measured joints, every frame. No readback means the tracker widens its mask, which is
    the honest answer."""
    rq = bridge.latest_real_q()
    if rq is not None:
        telemetry.push_q(rq[0][:6])


def _apply_recovery_hint(now_s):
    """After the bridge re-enabled the arm at its MEASURED pose (an auto-recovery or a reconnect) the twin, RMPflow
    and every joint-mode move restart from that pose; a 'home' hint then settles via HOME for a few seconds."""
    global _mode_override, _override_until, _startup_move, _hold_q, _home_q, _home_move, _sleep_q, _sleep_move, _sleep_stage, _sleep_t0
    hint = bridge.pop_recovery_hint()
    if hint is not None:
        q_real = _snap_sim_to_real()
        if q_real is not None:
            _handoff_from_joints(q_real)
            _hold_q = q_real.copy()
            _home_q = _home_move = None
            _sleep_q = _sleep_move = _sleep_stage = _sleep_t0 = None
            if _startup_move is not None:
                _startup_move = LM.JointMove(q_real, HOME_Q, STARTUP_RATE_RAD_S, UNWIND_MIN_S, time.perf_counter())
        if hint == "home" and _startup_move is None:
            _mode_override, _override_until = LM.HOME, now_s + POST_RECOVERY_HOME_S
            print("[app] post-recovery HOME (from the arm's measured pose)", flush=True)
        else:
            print(f"[app] re-seeded from the arm's measured pose ({hint})", flush=True)
    if _mode_override is not None and _startup_move is None and now_s >= _override_until:
        _mode_override = None
        print("[app] post-recovery HOME done, resuming", flush=True)


def _ops_state():
    """(GREEN|AMBER|RED, message, needs_recover) for the control link banner and the heartbeat."""
    if bridge.halt_reason:
        return "RED", f"RECOVER NEEDED: {bridge.halt_reason}", True
    st = bridge.state
    if st == ENABLED:
        if tracker_link.age() is not None and tracker_link.age() > 5.0:
            return "AMBER", "Tracker not reporting", False
        return "GREEN", "Running", False
    if st == DISCONNECTED:
        return ("RED", "Robot controller unreachable", False) if args.engage else ("AMBER", "Robot not connected", False)
    return "AMBER", f"Robot {st.lower()}", False


def _push_heartbeat(now_s):
    state, msg, needs_recover = _ops_state()
    m = {
        "fps": round(_fps_ema or 0.0, 1), "cs_ms": round(_cs_ema or 0.0, 2),
        "bridge": bridge.state, "hw_state": bridge.arm_hw_state, "err": bridge.arm_error_code,
        "warn": bridge.arm_warn_code, "mode": (_mode_override or (ui_state.mode if ui_state else CLI_MODE)),
        "sleep_stage": _sleep_stage,
        "real_q_age": (None if bridge.real_q_age() is None else round(bridge.real_q_age(), 3)),
        "tracker_age": (None if tracker_link.age() is None else round(tracker_link.age(), 3)),
        "control_peer": bool(control.connected) if control is not None else False,
        "recoveries": bridge.recoveries_total, "halted": bridge.halt_reason,
        "safety_ok": (None if bridge.safety_report is None else bridge.safety_report.ok),
        "speed_scale": round(bridge.speed_scale, 3), "speed_effective": round(bridge.speed_effective, 3),
        "ops": state, "ops_msg": msg, "last_error": bridge.last_error or "",
    }
    telemetry.push_hb(m)
    if control is not None:
        control.push_ops(state, msg, needs_recover)


# ------------------------------ main loop ----------------------------------- #
frame = 0
ee_trace = []
last_t = time.perf_counter()
reset_needed = False
_follow_err = None               # --test: 900 frames follow + retreat, then sleep / follow 2 s / sleep, by tracker hints
_sleep_min_home = 1e9
_sleep_settled = []
_sleep_cmd_settled = []
_sleep_elbow = []
_sleep_settle_t = None
_sleep_t_start = None
_follow_until = None
_test_t0 = None
_test_done = False
_retreat_done_frame = None
_retreat_ee0 = None
ee_target = HOME_EE
while simulation_app.is_running() and not _quit_requested:
    world.step(render=not args.test)
    now = time.perf_counter()
    loop_dt = now - last_t
    last_t = now
    fps = 1.0 / loop_dt if loop_dt > 0 else 0.0
    _fps_ema = fps if _fps_ema is None else 0.9 * _fps_ema + 0.1 * fps

    if world.is_stopped() and not reset_needed:
        reset_needed = True
    if world.is_playing():
        if reset_needed:
            world.reset()
            lock_environment()
            reset_needed = False
        if ui_state is not None:
            _poll_ui()
            if ui_state.consume_quit():
                _quit_requested = True
        _apply_recovery_hint(now)
        if bridge.state == ENABLED:
            enabled = True
        ee_target = control_step(loop_dt)
        _push_robot_q()
        _push_heartbeat(now)
        if _startup_move is not None:
            continue          # the scripted self-check starts at frame 0 once the arm has reached home
        frame += 1
        if ui_panel is not None and frame % 6 == 0:
            _trk = tracker_link.latest_target()
            _rq = bridge.latest_real_q()
            ui_panel.set_status(state=bridge.state, fps=_fps_ema or 0, cs_ms=_cs_ema or 0,
                                target_age=(_trk[4] if _trk else None), robot_hz=125.0,
                                behavior_state=behavior.state, halt_reason=bridge.halt_reason,
                                speed_effective=bridge.speed_effective,
                                arm_error_code=bridge.arm_error_code, arm_warn_code=bridge.arm_warn_code,
                                arm_hw_state=bridge.arm_hw_state, arm_mode=bridge.arm_mode,
                                arm_connected=bridge.arm_connected, last_error=bridge.last_error,
                                real_q=(_rq[0] if _rq else None), arm_position_m=bridge.arm_position_m)
            if control is not None:
                ui_panel.set_status(control_connected=control.connected, control_ip=control.peer_ip,
                                    control_ping_age=control.last_ping_age)
        if control is not None:
            _ee_q = amp.get_active_joints_subset().get_joint_positions()
            _ee_p, _ = rmpflow.get_end_effector_pose(_ee_q)
            control.push_status(bridge_state=bridge.state, mode=ui_state.mode if ui_state else "HOLD",
                                agent_on=ui_state.agent_on if ui_state else False,
                                speed_scale=ui_state.speed_scale if ui_state else 0.0,
                                arm_error=bridge.arm_error_code, arm_warn=bridge.arm_warn_code,
                                arm_hw_state=bridge.arm_hw_state, last_error=bridge.last_error or "",
                                ee_pos=np.asarray(_ee_p, dtype=np.float64).ravel())
        if args.test and frame % 60 == 0 and frame <= 600:
            q = amp.get_active_joints_subset().get_joint_positions()
            ee_pos, _ = rmpflow.get_end_effector_pose(q)
            ee_trace.append(np.asarray(ee_pos))
            _la = read_world_point(lookat_prim)
            _d_la = float(np.linalg.norm(np.asarray(_la) - np.asarray(ee_pos)))
            _retr = behavior._phase < behavior._retreat_until
            print(f"[app][test] f={frame} fps~{_fps_ema:5.1f} cs={_cs_ema:4.2f}ms ee={np.round(ee_pos, 3)} "
                  f"tgt={np.round(ee_target, 3)} lookat={np.round(_la, 3)} d_la={_d_la:.3f} retr={_retr} "
                  f"servo_cmds={len(arm.servo_cmds)}", flush=True)

        # --- retreat timing test: force a retreat at frame 600, measure retraction ---
        if args.test and frame == 600:
            behavior._retreat_until = behavior._phase + 10.0
            _rq = amp.get_active_joints_subset().get_joint_positions()
            _ree, _ = rmpflow.get_end_effector_pose(_rq)
            _retreat_ee0 = np.asarray(_ree).ravel()
            _retreat_anchor = np.array([*behavior.base_xy, DEFAULT_ANCHOR_Z])
            _retreat_done_frame = None
            print(f"\n[retreat-test] START  ee={np.round(_retreat_ee0, 3)}  anchor={np.round(_retreat_anchor, 3)}  "
                  f"dist={np.linalg.norm(_retreat_ee0 - _retreat_anchor):.3f}m", flush=True)

        if args.test and 600 < frame <= 900 and frame % 10 == 0:
            _rq_act = np.asarray(robot.get_joint_positions()[:6], dtype=np.float64)
            _rq_ee, _ = rmpflow.get_end_effector_pose(_rq_act)
            _rq_ee = np.asarray(_rq_ee).ravel()
            _ee_err = float(np.linalg.norm(np.asarray(ee_target) - _rq_ee))
            _q_gap = float(np.linalg.norm(_last_action_q - _rq_act)) if _last_action_q is not None else -1.0
            print(f"[retreat-test] f={frame}  ee_err={_ee_err:.3f}m  q_gap={_q_gap:.3f}rad", flush=True)
            if _ee_err < 0.05 and _retreat_done_frame is None:
                _retreat_done_frame = frame

    # --- the sleep part of the self-check is driven through the tracker's own hints ---
    if args.test and frame == 900 and _test_t0 is None:
        _fq = amp.get_active_joints_subset().get_joint_positions()
        _fee, _ = rmpflow.get_end_effector_pose(_fq)
        _follow_err = float(np.linalg.norm(np.asarray(_fee).ravel() - np.asarray(ee_target)))
        fake_tracker.hint = "sleep"
        _test_t0 = _sleep_t_start = now
        print(f"[sleep-test] f={frame} hint -> sleep #1 (follow err {_follow_err:.3f} m)", flush=True)
    if args.test and _prev_mode == LM.SLEEP and _sleep_q is not None:
        _sq_act = np.asarray(robot.get_joint_positions()[:6], dtype=np.float64)
        _sleep_min_home = min(_sleep_min_home, float(np.max(np.abs(_sq_act - HOME_Q))))
        if _sleep_stage == "breathe":
            if _sleep_settle_t is None:
                _sleep_settle_t = now
                _sleep_cmd_settled.append(np.asarray(_sleep_q, dtype=np.float64).copy())
                print(f"[sleep-test] f={frame} settled {now - _sleep_t_start:.1f} s after entry: cmd={np.round(_sleep_q, 4).tolist()} "
                      f"measured={np.round(_sq_act, 3).tolist()}", flush=True)
            if now - _sleep_settle_t >= 0.5 and len(_sleep_settled) < len(_sleep_cmd_settled):
                _sleep_settled.append(_sq_act.copy())
            _sleep_elbow.append(float(_sleep_q[2]))
            if now - _sleep_settle_t >= 4.0:               # a full breath observed
                if len(_sleep_cmd_settled) == 1:
                    fake_tracker.hint = "engage"
                    _follow_until, _sleep_settle_t = now + 2.0, None
                    print(f"[sleep-test] f={frame} hint -> engage for 2 s", flush=True)
                else:
                    _test_done = True
        elif frame % 300 == 0:
            print(f"[sleep-test] f={frame} stage={_sleep_stage} |q-home|={np.max(np.abs(_sq_act - HOME_Q)):.3f} "
                  f"|q-sleep|={np.max(np.abs(_sq_act - SLEEP_Q)):.3f}", flush=True)
    if args.test and _follow_until is not None and now >= _follow_until:
        fake_tracker.hint = "sleep"
        _follow_until, _sleep_t_start = None, now
        print(f"[sleep-test] f={frame} hint -> sleep #2", flush=True)
    if args.test and (_test_done or (_test_t0 is not None and now - _test_t0 > 120.0)):
        break

# ------------------------------ self-check ---------------------------------- #
if args.test:
    ok = True
    q = amp.get_active_joints_subset().get_joint_positions()
    ee_pos, _ = rmpflow.get_end_effector_pose(q)
    track_err = float(np.linalg.norm(np.asarray(ee_pos) - np.asarray(ee_target)))
    moved = float(np.std(np.asarray(ee_trace)[:, 0])) if len(ee_trace) > 3 else 0.0
    n_cmds = len(arm.servo_cmds)

    def report(name, cond, info=""):
        global ok
        ok = ok and cond
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {info if not cond else ''}")

    print(f"\n== self-check (mode={CLI_MODE} agent={'ON' if CLI_AGENT else 'OFF'}) ==")
    report("tracker target received", tracker_link.latest_target() is not None)
    report("safety config verified before arming", bridge.safety_report is not None and bridge.safety_report.ok)
    report("MockArm received servo cmds (bridge)", n_cmds > 0, f"n={n_cmds}")
    trk_tol = 0.12
    _ferr = track_err if _follow_err is None else _follow_err
    report(f"RMPflow tracks commanded EE (<{trk_tol} m)", _ferr < trk_tol, f"err={_ferr:.4f}")
    report("EE responds to the moving person (motion)", moved > 0.01, f"std_x={moved:.4f}")
    report("sleep goes through home first (|q-home| < 0.02)", _sleep_min_home < 0.02, f"min={_sleep_min_home:.3f}")
    _still = lambda q: np.array([q[0], q[1], q[3], q[5], q[2] - q[4]])   # noqa: E731
    _meas_err = [float(np.max(np.abs(_still(q) - _still(SLEEP_Q)))) for q in _sleep_settled]
    report("sleep settles at the sleep pose, both times (measured < 0.02 rad)", len(_meas_err) == 2 and max(_meas_err) < 0.02,
           f"n={len(_meas_err)} err={np.round(_meas_err, 4).tolist()}")
    report("sleep pose repeats (commanded settle pose identical)",
           len(_sleep_cmd_settled) == 2 and float(np.max(np.abs(_sleep_cmd_settled[0] - _sleep_cmd_settled[1]))) < 1e-6)
    report("sleep breathes (elbow nods)", bool(_sleep_elbow) and (max(_sleep_elbow) - min(_sleep_elbow)) > 0.03)
    report("sleep test ran to completion (two sleeps within 120 s)", _test_done)
    report(f"EE is at mount height (z>{MOUNT[2] - 0.21:.2f})", ee_pos[2] > MOUNT[2] - 0.21, f"z={ee_pos[2]:.3f}")
    report("perf: avg loop fps > 30 (headless)", (_fps_ema or 0) > 30, f"fps={_fps_ema:.1f}")
    if _retreat_ee0 is not None:
        if _retreat_done_frame is not None:
            retreat_secs = (_retreat_done_frame - 600) / 60.0
            report("retreat completes in <4s", retreat_secs < 4.0, f"{retreat_secs:.2f}s")
        else:
            report("retreat completes in <4s", False, "did not reach anchor in 300 frames (5s)")
    print(f"\nRESULT: {'ALL PASS' if ok else 'FAILURES PRESENT'} (fps~{_fps_ema:.1f}, cs~{_cs_ema:.2f}ms)")

# cleanup
if fake_tracker is not None:
    fake_tracker.stop()
_safe_exit()
tracker_link.stop()
telemetry.stop()
if control is not None:
    control.stop()
simulation_app.close()
if args.test:
    sys.exit(0 if ok else 1)
