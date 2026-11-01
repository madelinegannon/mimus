# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Motion behaviour for the arm: operator modes, the spring agent and the orientation ray.

  - SpringAgent : damped-spring "seek & arrive" steering
  - Behavior    : MODE-dispatched FSM (FollowTarget / FollowTargetSensor / IDLE / HOME). The operator
                  picks the active mode by button (hard manual override, no presence auto-switching).
                  An `agent_on` toggle layers the SpringAgent on top of the two follow modes: when on
                  the agent seeks the active target and its position is the output (orientation still
                  from the mode); when off the target pose drives the robot directly.
  - orientation : a pure geometric ray from a vertical anchor [0,0,Z] (a point above the robot base)
                  to the look-at point -- no gaze smoothing, no reorienting to the target's own pose.

Pure numpy (only imports helpers) -> unit-testable with no Isaac. The app applies the
workspace/reach clamps to whatever this returns, so "organic" never means "unsafe". USD I/O lives
in the app (app.py); this stays USD-free and is handed plain numpy.
"""
from __future__ import annotations

import copy
import json
import os
import time

import numpy as np

from . import helpers as H

# operator modes (hard manual override; no presence auto-switching)
FOLLOW_TARGET = "FollowTarget"          # follow the movable target cube's full world pose
FOLLOW_SENSOR = "FollowTargetSensor"    # cube pose computed from anchor[0,0,Z] -> look-at
IDLE = "IDLE"                           # look-around sweep of the empty room (tracker-owned, see AUTO_MODES)
SLEEP = "SLEEP"                         # compact rest position with sinusoidal Z breathing
HOME = "HOME"                           # ease to HOME_EE and hold
HOLD = "HOLD"                           # hold current pose (no autonomous motion)
MODES = [FOLLOW_TARGET, FOLLOW_SENSOR, IDLE, SLEEP, HOME, HOLD]

# ----- built-in personality presets (param dicts) --------------------------- #
# body/gaze: spring gains. reach/height: reach-cone framing (height_near_dz is an offset from the base flange;
# idle_ee_z and idle_look_z_lo/hi are WORLD heights, head height, so a taller mount does not lift the idle sweep).
# sway_*: the follow sway (amp m, period s). sniff_*: one approach per encounter -- after a retreat the body holds
# sniff_standoff off the look-at until it travels sniff_move_m (sniff_once=0 restores the old cycle).
BUILTIN_PRESETS = {
    "Curious":  dict(body_kp=1.2, body_kd=2.4, body_steer=0.6, gaze_kp=1.2, gaze_kd=2.4, gaze_steer=0.6,
                     reach=1.0, lerp=0.5, height=1.0, bow_depth=0.25, bow_lean=0.15,
                     retreat_dist=0.3, retreat_duration=3.0, retreat_frac=0.75,
                     sniff_once=1.0, sniff_standoff=0.55, sniff_move_m=0.4, sniff_cooldown=0.0, sniff_hold_frac=0.75,
                     dist_near=0.5, dist_far=3.0,
                     height_near_dz=-0.21,
                     idle_attn_lo=1.5, idle_attn_hi=5.0, idle_ee_z=1.75,
                     idle_gaze_heading=0.0, idle_gaze_angle=90.0, idle_gaze_step=35.0,
                     idle_gaze_dist_lo=1.5, idle_gaze_dist_hi=3.5,
                     idle_look_z_lo=1.35, idle_look_z_hi=1.80,
                     sleep_breath_amp=0.035, sleep_breath_rad=0.05, sleep_breath_period=3.0, sleep_breath_power=1.8,
                     sway_amp=0.015, sway_period=6.0),
    "Playful":  dict(body_kp=1.6, body_kd=1.8, body_steer=0.85, gaze_kp=1.6, gaze_kd=1.8, gaze_steer=0.85,
                     reach=1.0, lerp=0.5, height=1.05, bow_depth=0.3, bow_lean=0.2,
                     retreat_dist=0.3, retreat_duration=3.0, retreat_frac=0.75,
                     sniff_once=1.0, sniff_standoff=0.5, sniff_move_m=0.3, sniff_cooldown=0.0, sniff_hold_frac=0.75,
                     dist_near=0.5, dist_far=3.0,
                     height_near_dz=-0.21,
                     idle_attn_lo=0.8, idle_attn_hi=3.5, idle_ee_z=1.75,
                     idle_gaze_heading=0.0, idle_gaze_angle=90.0, idle_gaze_step=45.0,
                     idle_gaze_dist_lo=1.0, idle_gaze_dist_hi=4.0,
                     idle_look_z_lo=1.30, idle_look_z_hi=1.85,
                     sleep_breath_amp=0.04, sleep_breath_rad=0.05, sleep_breath_period=3.5, sleep_breath_power=2.0,
                     sway_amp=0.02, sway_period=4.5),
    "Tired":    dict(body_kp=0.7, body_kd=3.0, body_steer=0.35, gaze_kp=0.7, gaze_kd=3.0, gaze_steer=0.35,
                     reach=1.0, lerp=0.5, height=0.85, bow_depth=0.18, bow_lean=0.1,
                     retreat_dist=0.3, retreat_duration=3.0, retreat_frac=0.75,
                     sniff_once=1.0, sniff_standoff=0.8, sniff_move_m=0.5, sniff_cooldown=0.0, sniff_hold_frac=0.75,
                     dist_near=0.5, dist_far=3.0,
                     height_near_dz=-0.21,
                     idle_attn_lo=3.0, idle_attn_hi=8.0, idle_ee_z=1.75,
                     idle_gaze_heading=0.0, idle_gaze_angle=90.0, idle_gaze_step=25.0,
                     idle_gaze_dist_lo=2.0, idle_gaze_dist_hi=3.0,
                     idle_look_z_lo=1.40, idle_look_z_hi=1.75,
                     sleep_breath_amp=0.025, sleep_breath_rad=0.05, sleep_breath_period=5.0, sleep_breath_power=1.5,
                     sway_amp=0.01, sway_period=9.0),
    "Cautious": dict(body_kp=1.0, body_kd=3.0, body_steer=0.4, gaze_kp=1.0, gaze_kd=3.0, gaze_steer=0.4,
                     reach=1.0, lerp=0.5, height=0.95, bow_depth=0.2, bow_lean=0.12,
                     retreat_dist=0.3, retreat_duration=3.0, retreat_frac=0.75,
                     sniff_once=1.0, sniff_standoff=0.9, sniff_move_m=0.5, sniff_cooldown=0.0, sniff_hold_frac=0.75,
                     dist_near=0.5, dist_far=3.0,
                     height_near_dz=-0.21,
                     idle_attn_lo=2.0, idle_attn_hi=6.0, idle_ee_z=1.75,
                     idle_gaze_heading=0.0, idle_gaze_angle=90.0, idle_gaze_step=20.0,
                     idle_gaze_dist_lo=2.0, idle_gaze_dist_hi=3.0,
                     idle_look_z_lo=1.40, idle_look_z_hi=1.75,
                     sleep_breath_amp=0.02, sleep_breath_rad=0.05, sleep_breath_period=4.5, sleep_breath_power=1.6,
                     sway_amp=0.01, sway_period=7.0),
}
PRESET_NAMES = list(BUILTIN_PRESETS.keys())


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


# --------------------------------------------------------------------------- #
class SpringAgent:
    """Damped-spring seek-and-arrive on a 3D point."""

    def __init__(self, pos, kp=1.0, kd=2.5, steering=0.5, slow_radius=0.3, trail_len=50):
        self.kp, self.kd, self.steering = float(kp), float(kd), float(steering)
        self.slow_radius = float(slow_radius)
        self._pos = np.asarray(pos, dtype=np.float64).copy()
        self._vel = np.zeros(3)
        self.trail = []
        self.trail_len = trail_len

    def update(self, target, dt):
        target = np.asarray(target, dtype=np.float64)
        heading = target - self._pos
        n = np.linalg.norm(heading)
        # ARRIVE: desired speed scales down within slow_radius so it settles (not just seek)
        vel_desired = (heading / n) * min(1.0, n / self.slow_radius) if n > 1e-9 else np.zeros(3)
        accel = self.kp * (target - self._pos) + self.kd * (vel_desired - self._vel)
        self._vel = self._vel + accel * dt
        self._pos = self._pos + self._vel * dt * self.steering
        self.trail.append(self._pos.copy())
        if len(self.trail) > self.trail_len:
            self.trail.pop(0)
        return self._pos.copy()

    def get_position(self):
        return self._pos.copy()

    def set_position(self, p):
        self._pos = np.asarray(p, dtype=np.float64).copy()
        self._vel = np.zeros(3)

    def reset(self, p=None):
        if p is not None:
            self._pos = np.asarray(p, dtype=np.float64).copy()
        self._vel = np.zeros(3)
        self.trail = []

    def set_gains(self, kp, kd, steering):
        self.kp, self.kd, self.steering = float(kp), float(kd), float(steering)


def reach_cone_point(person, base_xy, height_z, reach, lerp=0.5):
    """Point along anchor→person ray, clamped by horizontal reach from the base axis.

    reach = max horizontal distance (radial) from the vertical axis through base_xy.
    lerp=0 → anchor, lerp=1 → reach limit.
    Returns (target_point, reach_limit_point) — reach_limit is always on the ray at the reach boundary.
    """
    anchor = np.array([base_xy[0], base_xy[1], height_z], dtype=np.float64)
    d = np.asarray(person, dtype=np.float64) - anchor
    n = np.linalg.norm(d)
    if n < 1e-9:
        rl = anchor + np.array([reach, 0.0, 0.0])
        return anchor + np.array([reach * lerp, 0.0, 0.0]), rl
    d_h = np.sqrt(d[0] ** 2 + d[1] ** 2)
    if d_h < 1e-9:
        rl = anchor + np.array([reach, 0.0, 0.0])
        return anchor + np.array([reach * lerp, 0.0, 0.0]), rl
    t_reach = reach / d_h
    point = anchor + d * (lerp * t_reach)
    reach_limit = anchor + d * t_reach
    return point, reach_limit


def clamp_reach(p, centre, r_max):
    """Pull a target inside the sphere of radius r_max about `centre` (the shoulder axis): the arm is never asked
    to straighten out. Beyond it the elbow locks, the wrist folds to keep aiming and J5 runs into its limit (the
    C23 fault on a real arm). Returns (point, clamped). r_max None or 0 disables the clamp."""
    p = np.asarray(p, dtype=np.float64)
    if not r_max or r_max <= 0:
        return p, False
    c = np.asarray(centre, dtype=np.float64)
    d = p - c
    n = float(np.linalg.norm(d))
    if n <= r_max:
        return p, False
    return c + d * (r_max / n), True


# --------------------------------------------------------------------------- #
class JointMove:
    """One timed joint-space move (HOME and SLEEP): every joint follows the same smoothstep ease-in-out from q0 to q1,
    so they arrive together. The largest joint delta over rate_rad_s sets the duration (never under min_s); the peak
    joint speed is 1.5 x rate_rad_s. at(t) is a function of wall time, so a stalled frame never changes the path."""

    def __init__(self, q0, q1, rate_rad_s, min_s, t0):
        self.q0 = np.asarray(q0, dtype=np.float64).copy()
        self.q1 = np.asarray(q1, dtype=np.float64).copy()
        self.t0 = float(t0)
        self.duration = max(float(min_s), float(np.max(np.abs(self.q1 - self.q0))) / max(float(rate_rad_s), 1e-6))

    def at(self, t):
        """(q, done) at wall time t."""
        u = (float(t) - self.t0) / self.duration
        if u >= 1.0:
            return self.q1.copy(), True
        s = max(0.0, u)
        s = s * s * (3.0 - 2.0 * s)
        return self.q0 + (self.q1 - self.q0) * s, False


def breath_value(phase_s, period_s, power):
    """-1..1 breathing curve: a sine whose inhale half is shaped by `power` (slower rise)."""
    t_norm = (phase_s % period_s) / period_s
    val = np.sin(2.0 * np.pi * t_norm)
    if np.cos(2.0 * np.pi * t_norm) > 0:
        val_01 = ((val + 1.0) / 2.0) ** power
        val = val_01 * 2.0 - 1.0
    return float(val)


def breath_lift(phase_s, period_s, power):
    """0..1 one-sided breath: 0 at phase 0 (the rest pose), 1 at period/2, zero slope at both turning points."""
    return (breath_value(phase_s + 0.75 * period_s, period_s, power) + 1.0) / 2.0


SLEEP_BREATH_DIR = np.array([0.0, 0.0, -1.0, 0.0, -1.0, 0.0])   # elbow unfolds, wrist follows by the same angle (J3 and J5 pitch the same way): the head rises and stays bowed


def sleep_joint_pose(phase_s, sleep_q, params):
    """Joint-space sleep pose with breathing: sleep_q is the bottom of the breath; the lift (0..sleep_breath_rad along
    SLEEP_BREATH_DIR) raises the flange straight up and returns to exactly sleep_q every cycle."""
    period = float(params["sleep_breath_period"])
    amp = float(params.get("sleep_breath_rad", 0.05))
    v = breath_lift(phase_s, period, float(params["sleep_breath_power"]))
    return np.asarray(sleep_q, dtype=np.float64) + amp * v * SLEEP_BREATH_DIR


# ----- follow sway: the arm is never perfectly still in front of a still visitor ----- #
UP = np.array([0.0, 0.0, 1.0])
SWAY_RATIOS = (1.0, 1.618, 2.618)      # golden-ratio periods: the sum never lines up into a beat
SWAY_WEIGHTS = (0.5, 0.3, 0.2)         # sum to 1, so |sway_value| <= 1 exactly
SWAY_PHASES = ((0.0, 2.1, 4.2), (1.0, 3.3, 5.5))
SWAY_VERT = 0.6                        # up/down share of the lateral amplitude
SWAY_VERT_RATIO = 1.31                 # the vertical runs slower than the lateral so the pair never repeats
SWAY_FADE_S = 2.0


def sway_value(t, period, phases):
    """-1..1: three sines at golden-ratio multiples of `period`."""
    return float(sum(w * np.sin(2.0 * np.pi * t / (period * k) + ph)
                     for w, k, ph in zip(SWAY_WEIGHTS, SWAY_RATIOS, phases)))


def sway_uv(t, period):
    """(lateral, vertical) in -1..1 at time t."""
    return sway_value(t, period, SWAY_PHASES[0]), sway_value(t, period * SWAY_VERT_RATIO, SWAY_PHASES[1])


class Behavior:
    """Mode-dispatched FSM producing (target_pos, target_quat, dbg) each frame.

    The operator selects `mode` each frame (hard manual override). `agent_on` layers the body
    SpringAgent on the two FOLLOW modes. Orientation in FollowTargetSensor is the pure geometric
    ray from the vertical anchor [0,0,anchor_z] toward the look-at point.
    """

    def __init__(self, base_xy, home_ee, params=None, aim_axis="z", base_z=0.0, base_yaw_deg=0.0,
                 shoulder_dz=0.364, reach_max=0.85):
        self.base_xy = np.asarray(base_xy, dtype=np.float64)
        self.base_z = float(base_z)          # base flange height: preset heights are offsets from it
        # every target stays inside a sphere about the shoulder (on the UF850 the J1/J2 axes cross 0.364 above the
        # flange): 0.85 m is the arm's full reach, an operator's choice; the bridge's joint-limit net is
        # the backstop when the tool points back toward the shoulder
        self.shoulder = np.array([self.base_xy[0], self.base_xy[1], self.base_z + float(shoulder_dz)])
        self.reach_max = None if reach_max is None else float(reach_max)
        self._clamped = 0
        self.base_yaw = np.radians(float(base_yaw_deg))   # J1 = 0 direction in the world: the idle cone is centred on it
        self._fwd = np.array([np.cos(self.base_yaw), np.sin(self.base_yaw), 0.0])
        self.home_ee = np.asarray(home_ee, dtype=np.float64)
        self.aim_axis = aim_axis
        self.params = dict(BUILTIN_PRESETS["Curious"])
        self.body = SpringAgent(self.home_ee)
        self.gaze = SpringAgent(self.home_ee + self._fwd)
        if params:
            self.set_params(params)
        self.state = FOLLOW_TARGET
        self._phase = 0.0
        self._retreat_until = 0.0
        self._sniffed = False        # sniffed once: watch from standoff instead of driving in again
        self._sniff_track = None     # None also means "caller gave no identity", so the latch still holds
        self._sniff_ref = None
        self._sniff_hint = None
        self._sniff_at = 0.0
        self._prev_agent_on = False
        self._idle_lookat = self.home_ee + self._fwd
        self._idle_azimuth = 0.0
        self._idle_attn_phase = 0.0
        self._idle_attn_limit = 3.0
        self._sleep_phase = 0.0
        self._sway_t0 = 0.0

    def set_params(self, params: dict):
        self.params.update(params)
        p = self.params
        self.body.set_gains(p["body_kp"], p["body_kd"], p["body_steer"])
        self.gaze.set_gains(p["gaze_kp"], p["gaze_kd"], p["gaze_steer"])

    def seed(self, ee_now):
        self.body.set_position(ee_now)
        self._sway_t0 = self._phase   # a re-seeded spring starts still: the sway fades back in

    def _clamp(self, p):
        """Reach clamp for one target; counts hits for dbg['reach_clamped']."""
        q, hit = clamp_reach(p, self.shoulder, self.reach_max)
        self._clamped += int(hit)
        return q

    def _settle(self, pos):
        """The spring's output, clamped; a clamped spring is moved onto the sphere so it never winds up against it."""
        q, hit = clamp_reach(pos, self.shoulder, self.reach_max)
        if hit:
            self._clamped += 1
            self.body.set_position(q)
        return q

    def _swayed(self, pos):
        """pos plus the follow sway, tangential to the reach sphere (the clamp cannot eat it), faded in after a (re)seed."""
        p = self.params
        amp = float(p.get("sway_amp", 0.0))
        env = smoothstep((self._phase - self._sway_t0) / SWAY_FADE_S)
        if amp <= 0.0 or env <= 0.0:
            return pos
        r = pos - self.shoulder
        n = float(np.linalg.norm(r))
        r = r / n if n > 1e-6 else self._fwd
        side = np.cross(UP, r)
        ns = float(np.linalg.norm(side))
        side = side / ns if ns > 1e-6 else np.cross(UP, self._fwd)   # straight above the shoulder: sway across the base's forward
        u, v = sway_uv(self._phase, float(p.get("sway_period", 6.0)))
        d = (u * side + SWAY_VERT * v * np.cross(r, side)) * (amp * env / np.hypot(1.0, SWAY_VERT))
        return clamp_reach(pos + d, self.shoulder, self.reach_max)[0]

    def _idle_bounds(self):
        """The idle sweep is a cone in the base frame: centre = base yaw + idle_gaze_heading, half-angle idle_gaze_angle."""
        p = self.params
        return self.base_yaw + np.radians(p["idle_gaze_heading"]), np.radians(p["idle_gaze_angle"])

    def _idle_start_azimuth(self, ee_now):
        """Start the sweep where the arm is looking now (no snap back to the centre), clipped into the cone."""
        centre, half = self._idle_bounds()
        d = np.asarray(ee_now, dtype=np.float64)[:2] - self.base_xy
        if np.hypot(d[0], d[1]) < 0.05:          # on the base axis the azimuth is undefined
            return float(centre)
        az = np.arctan2(d[1], d[0])
        az = centre + (az - centre + np.pi) % (2 * np.pi) - np.pi      # wrap to the turn nearest the centre
        return float(np.clip(az, centre - half, centre + half))

    def _release_sniff(self, tid, hint, gaze_lookat):
        """Clear the one-approach latch when this stops being the same still encounter."""
        if tid is not None and self._sniff_track is not None and tid != self._sniff_track:
            self._sniffed, self._sniff_track = False, None       # a different person earns a fresh approach
        elif hint != self._sniff_hint:
            # head->hand is the tracker changing which point it reports, not the person moving
            self._sniff_ref, self._sniff_hint = gaze_lookat.copy(), hint
        elif self._sniff_ref is not None:
            cool = float(self.params.get("sniff_cooldown", 0.0))
            moved = float(np.linalg.norm(gaze_lookat - self._sniff_ref))
            if (moved > float(self.params.get("sniff_move_m", 0.4))
                    or (cool > 0.0 and self._phase - self._sniff_at > cool)):
                self._sniffed, self._sniff_track = False, None

    def update(self, mode, cube_pose, lookat, anchor_z, agent_on, ee_now, dt, track_id=None, hint=None):
        """mode: one of MODES. cube_pose: (pos[3], quat[4]) WORLD pose of the target cube.
        lookat: WORLD look-at point. Returns (target_pos[3], target_quat[4], dbg) where
        dbg = {agent_pos, seek_target, anchor, lookat} for the debug viz.
        track_id / hint: tracker identity for the sniff latch (None or -1 = unknown)."""
        self._phase += dt
        self._clamped = 0
        lean = None                      # follow mode: the lean fraction, for the overlay and the tests
        prev_mode = self.state
        self.state = mode
        if prev_mode in (IDLE, SLEEP) and mode not in (IDLE, SLEEP):
            p = self.params
            self.body.set_gains(p["body_kp"], p["body_kd"], p["body_steer"])
        cube_pos = np.asarray(cube_pose[0], dtype=np.float64)
        cube_quat = np.asarray(cube_pose[1], dtype=np.float64)
        lookat = np.asarray(lookat, dtype=np.float64)
        ee_now = np.asarray(ee_now, dtype=np.float64)
        anchor_z = float(anchor_z)
        anchor = np.array([self.base_xy[0], self.base_xy[1], anchor_z], dtype=np.float64)

        if agent_on and not self._prev_agent_on:
            self.body.reset(cube_pos)
        self._prev_agent_on = agent_on

        # Smooth lookat for orientation (aim_at) only — geometry uses raw lookat
        if mode == FOLLOW_SENSOR:
            if prev_mode != FOLLOW_SENSOR:
                p = self.params
                self.gaze.set_gains(p["gaze_kp"], p["gaze_kd"], p["gaze_steer"])
                self.gaze.reset(lookat)
                self._sway_t0 = self._phase
            gaze_lookat = self.gaze.update(lookat, dt)
        else:
            gaze_lookat = lookat

        # Retreat: timed partial retract along lookat→anchor slope
        if mode == FOLLOW_SENSOR:
            tid = int(track_id) if track_id is not None and int(track_id) >= 0 else None
            if self._sniffed:
                self._release_sniff(tid, hint, gaze_lookat)
            if self._phase >= self._retreat_until:
                d_ee = float(np.linalg.norm(lookat - ee_now))
                if d_ee < self.params.get("retreat_dist", 0.1):
                    self._retreat_until = self._phase + self.params.get("retreat_duration", 3.0)
                    self._sniffed, self._sniff_track, self._sniff_at = True, tid, self._phase
                    self._sniff_ref, self._sniff_hint = gaze_lookat.copy(), hint
            if self._phase < self._retreat_until:
                frac = self.params.get("retreat_frac", 0.75)
                seek_target = self._clamp(lookat + frac * (anchor - lookat))
                quat = H.aim_at(seek_target, gaze_lookat, approach_axis=self.aim_axis)
                pos = seek_target.copy()
                self.body.set_position(pos)
                dbg = dict(agent_pos=self.body.get_position(), seek_target=seek_target.copy(),
                           anchor=anchor.copy(), lookat=lookat.copy(), reach_limit=None,
                           reach_clamped=self._clamped, retreating=True, sniffed=self._sniffed, agent_on=agent_on,
                           trail=list(self.body.trail),
                           gaze_lookat=gaze_lookat.copy(), gaze_trail=list(self.gaze.trail))
                return pos, quat, dbg
        elif self._retreat_until > 0.0 or self._sniffed:
            self._retreat_until = 0.0
            self._sniffed, self._sniff_track = False, None

        if mode == FOLLOW_TARGET:
            seek_target = self._clamp(cube_pos)
            quat = cube_quat.copy()
            if agent_on:
                pos = self._settle(self.body.update(seek_target, dt))
            else:
                pos = seek_target.copy()
                self.body.set_position(pos)   # keep agent synced for a smooth toggle-on hand-off

        elif mode == FOLLOW_SENSOR:
            p = self.params
            self.body.set_gains(p["body_kp"], p["body_kd"], p["body_steer"])
            self.gaze.set_gains(p["gaze_kp"], p["gaze_kd"], p["gaze_steer"])
            d_horiz = np.sqrt((gaze_lookat[0] - self.base_xy[0]) ** 2
                              + (gaze_lookat[1] - self.base_xy[1]) ** 2)
            d_near = p.get("dist_near", 0.5)
            d_far = p.get("dist_far", 3.0)
            d_range = d_far - d_near
            d_norm = float(np.clip((d_horiz - d_near) / d_range, 0.0, 1.0)) if d_range > 1e-9 else 0.0
            # ease-in-out along the ramp (tuned on a real arm): the lean stays near full close in and lets go
            # smoothly toward dist_far, instead of a straight line that had a visitor at the rail get only 0.7 of
            # the reach. dist_near / dist_far keep their meaning: full lean and near height at one end, the anchor
            # at the other.
            frac = smoothstep(d_norm)
            sniff_hold = self._sniffed and p.get("sniff_once", 0.0) > 0.0
            # held: stay at the retreat's high anchor, not the near-visitor height
            eff_height = anchor_z if sniff_hold else float(np.interp(
                frac, [0, 1], [self.base_z + p.get("height_near_dz", -0.21), anchor_z]))
            eff_lerp = 1.0 - frac
            lean = eff_lerp
            anchor_eff = np.array([self.base_xy[0], self.base_xy[1], eff_height], dtype=np.float64)
            seek_target, reach_limit = reach_cone_point(
                lookat, self.base_xy, eff_height, p["reach"], eff_lerp)
            if sniff_hold:
                # the reach cone at the near height folds the wrist down beside the column: hold the retreat pose
                d_a = float(np.linalg.norm(anchor_eff - lookat))
                stand = max(float(p.get("sniff_standoff", 0.6)), 1.5 * float(p.get("retreat_dist", 0.3)))
                frac = max(float(p.get("sniff_hold_frac", 0.75)), (stand / d_a) if d_a > 1e-6 else 1.0)
                seek_target = lookat + min(frac, 1.0) * (anchor_eff - lookat)
            seek_target = self._clamp(seek_target)
            if agent_on:
                pos = self._swayed(self._settle(self.body.update(seek_target, dt)))
                quat = H.aim_at(pos, gaze_lookat, approach_axis=self.aim_axis)
            else:
                pos = seek_target.copy()
                self.body.set_position(pos)
                quat = H.aim_at(anchor_eff, gaze_lookat, approach_axis=self.aim_axis)

        elif mode == IDLE:
            p = self.params
            if prev_mode != IDLE:
                self.body.set_gains(kp=0.7, kd=3.0, steering=0.35)
                self.gaze.set_gains(kp=1.0, kd=2.0, steering=0.05)
                self.body.reset(ee_now)
                self._idle_azimuth = self._idle_start_azimuth(ee_now)
                self.gaze.reset(ee_now + np.array([np.cos(self._idle_azimuth), np.sin(self._idle_azimuth), 0.0]))
                self._idle_attn_phase = self._idle_attn_limit

            self._idle_attn_phase += dt
            if self._idle_attn_phase >= self._idle_attn_limit:
                heading_rad, max_angle_rad = self._idle_bounds()
                step_rad = np.radians(p["idle_gaze_step"])
                delta = np.random.uniform(-step_rad, step_rad)
                self._idle_azimuth = np.clip(self._idle_azimuth + delta,
                                             heading_rad - max_angle_rad,
                                             heading_rad + max_angle_rad)
                dist = np.random.uniform(p["idle_gaze_dist_lo"], p["idle_gaze_dist_hi"])
                self._idle_lookat = np.array([
                    self.base_xy[0] + dist * np.cos(self._idle_azimuth),
                    self.base_xy[1] + dist * np.sin(self._idle_azimuth),
                    np.random.uniform(p["idle_look_z_lo"], p["idle_look_z_hi"]),   # world heights: where visitors' heads are
                ])
                self._idle_attn_phase = 0.0
                self._idle_attn_limit = np.random.uniform(p["idle_attn_lo"], p["idle_attn_hi"])

            idle_height = float(p.get("idle_ee_z", 1.75))     # a world height just above heads: the same on any mount
            seek_target, reach_limit = reach_cone_point(
                self._idle_lookat, self.base_xy, idle_height, p["reach"], p.get("lerp", 0.5))
            seek_target = self._clamp(seek_target)
            if agent_on:
                pos = self._settle(self.body.update(seek_target, dt))
            else:
                pos = seek_target.copy()
                self.body.set_position(pos)
            gaze_pos = self.gaze.update(self._idle_lookat, dt)
            quat = H.aim_at(pos, gaze_pos, approach_axis=self.aim_axis)
            lookat = gaze_pos.copy()
            anchor_eff = np.array([self.base_xy[0], self.base_xy[1], idle_height], dtype=np.float64)

        elif mode == SLEEP:
            # the sleep routine is joint space and lives in the app (home -> settle -> breathe); hold here
            pos = ee_now.copy()
            quat = H.aim_at(pos, pos + np.array([0.3, 0.0, -0.3]), approach_axis=self.aim_axis)
            self.body.set_position(pos)
            seek_target = pos.copy()

        elif mode == HOME:
            pos = ee_now.copy()
            quat = H.aim_at(pos, pos + np.array([0.3, 0.0, -0.3]), approach_axis=self.aim_axis)
            self.body.set_position(pos)
            seek_target = pos.copy()

        elif mode == HOLD:
            pos = ee_now.copy()
            quat = H.aim_at(pos, pos + np.array([0.3, 0.0, -0.3]), approach_axis=self.aim_axis)
            self.body.set_position(pos)
            seek_target = pos.copy()

        else:
            raise ValueError(f"unknown mode: {mode}")

        reach_lim = reach_limit if mode in (FOLLOW_SENSOR, IDLE) else None
        eff_anchor = anchor_eff if mode in (FOLLOW_SENSOR, IDLE) else anchor
        dbg = dict(agent_pos=self.body.get_position(), seek_target=seek_target.copy(),
                   anchor=eff_anchor, lookat=lookat.copy(), reach_limit=reach_lim,
                   reach_clamped=self._clamped, lean=lean, retreating=(self._phase < self._retreat_until), sniffed=self._sniffed,
                   agent_on=agent_on, trail=list(self.body.trail),
                   gaze_lookat=gaze_lookat.copy(), gaze_trail=list(self.gaze.trail))
        return pos, quat, dbg


# ----- tracker-driven mode selection ----------------------------------------- #
# The tracker owns these two: "idle" is nobody engaged (sweep for passersby), "sleep" is its presence
# gate saying nobody is worth being awake for. Every other hint means a person to follow.
AUTO_MODES = {"idle": IDLE, "sleep": SLEEP}


def mode_for_hint(hint):
    """Mode the tracker's hint asks for; unknown or attentive hints mean follow."""
    return AUTO_MODES.get(hint, FOLLOW_SENSOR)


# ----- persisting a live-adjusted value -------------------------------------- #
class SettledValue:
    """Decides when a value a human is dragging is worth writing to disk: once it has held steady for
    `after_s`, so one drag is one save and not one per pixel."""

    def __init__(self, value, after_s):
        self.saved = value
        self.after_s = float(after_s)
        self._pending = None
        self._t = 0.0

    def update(self, value, now):
        """True exactly once, when `value` has settled and differs from what was last saved."""
        if value == self.saved:
            self._pending = None
            return False
        if value != self._pending:
            self._pending, self._t = value, float(now)
            return False
        if float(now) - self._t >= self.after_s:
            self.saved, self._pending = value, None
            return True
        return False


# ----- startup pose check ---------------------------------------------------- #
def startup_pose_ok(real_q, limits_rad, margin_rad=0.02):
    """(ok, reason) for the measured pose the arm is in when the app starts. The bound is the robot's
    own joint limits, not a distance from home: any pose inside them is one the arm reached under its own
    power, so refusing it would strand the app for no reason. What this catches is a reading that
    cannot be true - no answer, a short frame, NaN, or an angle the joint cannot physically hold - because
    the startup move interpolates straight through joint space and garbage in is a swing through the rail."""
    if real_q is None:
        return False, "no joint angles from the controller"
    q = np.asarray(real_q, dtype=np.float64).ravel()
    if q.size < 6:
        return False, f"controller returned {q.size} joint angles, expected 6"
    q = q[:6]
    if not np.all(np.isfinite(q)):
        return False, f"controller returned a non-finite joint angle: {q.tolist()}"
    for i, (lo, hi) in enumerate(list(limits_rad)[:6]):
        if not (float(lo) - margin_rad <= q[i] <= float(hi) + margin_rad):
            return False, (f"J{i + 1} reads {np.degrees(q[i]):.0f} deg, outside its limits "
                           f"{np.degrees(float(lo)):.0f}..{np.degrees(float(hi)):.0f} deg")
    return True, ""


# ----- startup mode selection ------------------------------------------------ #
_MODE_MAP = {"follow_target": FOLLOW_TARGET, "follow_sensor": FOLLOW_SENSOR,
             "idle": IDLE, "sleep": SLEEP, "home": HOME, "hold": HOLD}


def default_mode(arm, engage, test, explicit_mode):
    """Return the startup mode string given CLI args."""
    if explicit_mode:
        return _MODE_MAP[explicit_mode]
    if test:
        return FOLLOW_SENSOR
    if arm == "xarm" and not engage:
        return HOLD
    return FOLLOW_TARGET


# ----- preset persistence (mirrors calibration ground-truth) ---------------- #
def load_presets(path):
    """Return {name: params}, built-ins overlaid with any saved overrides."""
    presets = copy.deepcopy(BUILTIN_PRESETS)
    if os.path.exists(path):
        try:
            with open(path) as f:
                saved = json.load(f)
            for k, v in saved.items():
                if k in presets:
                    presets[k].update(v)
        except Exception:
            pass
    return presets


def save_preset(path, name, params):
    """Override one preset in the on-disk file (timestamped backup kept)."""
    import shutil
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            data = {}
        shutil.copy(path, path.replace(".json", f"_{time.strftime('%Y%m%d_%H%M%S')}.json"))
    data[name] = {k: float(v) for k, v in params.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
