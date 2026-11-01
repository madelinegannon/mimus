# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Touch-first omni.ui control panel. Every toggle is a large button (>= 44 px, ON/OFF label and a
green/grey state colour) so it works on a tablet over a remote desktop. The panel only mutates a plain
`UIState`; the control loop polls it (no async, no cross-thread UI calls). Import-safe headless."""
from __future__ import annotations

import math

from .motion import BUILTIN_PRESETS
from .ui_state import UIState  # noqa: F401  re-exported for callers and tests

try:
    import omni.ui as ui
    _HAS_UI = True
except Exception:  # noqa: BLE001
    _HAS_UI = False

try:
    from rt_xarm_bridge import URDF_JOINT_LIMITS_RAD as FALLBACK_JOINT_LIMITS_RAD
except Exception:  # noqa: BLE001
    FALLBACK_JOINT_LIMITS_RAD = [(-3.14, 3.14)] * 6

_ON = 0xFF2E7D32     # green
_OFF = 0xFF555555    # grey
_ACT = 0xFF8A5A00    # amber (momentary)
_STOP = 0xFF2222CC   # red
_PRESETS = ["Curious", "Playful", "Tired", "Cautious"]
_MODES = [("Follow Target", "FollowTarget"), ("Follow Sensor", "FollowTargetSensor"),
          ("Idle", "IDLE"), ("Sleep", "SLEEP"), ("Home", "HOME"), ("Hold", "HOLD")]
_ADV_PARAMS = [  # (key, label, min, max, default[Curious]): spring agent + framing
    ("body_kp", "Body kp", 0.0, 3.0, 1.2), ("body_kd", "Body kd", 0.0, 5.0, 2.4),
    ("body_steer", "Body steer", 0.0, 1.5, 0.6),
    ("idle_gaze_heading", "Idle heading °", -180.0, 180.0, 0.0),
    ("sniff_standoff", "Sniff standoff m", 0.3, 1.5, 0.55),
    ("sniff_move_m", "Sniff move m", 0.1, 1.5, 0.4),
    ("sniff_hold_frac", "Sniff hold frac", 0.2, 0.95, 0.75),
]
_TRACKER_PARAMS = [  # (key, label, min, max, default): who the tracker follows; sent to the tracker, never into a preset
    ("near_pref", "Near", 0.0, 1.0, 0.5),
    ("kid_pref", "Kids", 0.0, 1.0, 0.4),
    ("kid_h", "Kid height m", 0.8, 1.6, 1.2),
]


class LiveDemoUI:
    def __init__(self, state: UIState, control=None):
        self.state = state
        self._control = control
        self._window = None
        self._labels = {}
        self._joint_sliders = []
        self._arm_toggle_btn = None
        self._arm_toggle_v = {"v": False}
        self._conn_toggle_btn = None
        self._conn_toggle_v = {"v": False}
        self._speed_model = None
        self._mode_highlight = None
        self._agent_btn = None
        self._agent_v = None

    def _toggle(self, label, initial, on_change):
        s = {"v": bool(initial)}
        btn = ui.Button(f"{label}: {'ON' if s['v'] else 'OFF'}", height=46)

        def click():
            s["v"] = not s["v"]
            btn.text = f"{label}: {'ON' if s['v'] else 'OFF'}"
            btn.style = {"background_color": _ON if s["v"] else _OFF, "color": 0xFFFFFFFF}
            on_change(s["v"])
        btn.set_clicked_fn(click)
        btn.style = {"background_color": _ON if s["v"] else _OFF, "color": 0xFFFFFFFF}
        return btn, s

    def _action(self, label, on_click, color=_ACT, height=46):
        b = ui.Button(label, height=height, clicked_fn=on_click)
        b.style = {"background_color": color, "color": 0xFFFFFFFF}
        return b

    def _radio(self, options, initial, on_change, height=46):
        btns = {}

        def select(val):
            for v, b in btns.items():
                b.style = {"background_color": _ON if v == val else _OFF, "color": 0xFFFFFFFF}
            on_change(val)

        def highlight(val):
            for v, b in btns.items():
                b.style = {"background_color": _ON if v == val else _OFF, "color": 0xFFFFFFFF}

        for label, val in options:
            b = ui.Button(label, height=height)
            b.set_clicked_fn(lambda v=val: select(v))
            b.style = {"background_color": _ON if val == initial else _OFF, "color": 0xFFFFFFFF}
            btns[val] = b
        return btns, highlight

    def build(self, joint_limits=None):
        if not _HAS_UI:
            return self
        self._joint_limits = joint_limits or list(FALLBACK_JOINT_LIMITS_RAD)
        self._window = ui.Window("Mimus", width=380, height=860, dockPreference=ui.DockPreference.RIGHT_TOP)
        with self._window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6, height=0):
                    self._robot_management_group()
                    self._devmode_group()
                    self._motion_params_group()
                    self._robot_group()
                    self._behavior_group()
                    self._control_group()
                    self._attention_group()
                    self._advanced_group()
                    self._status_group()
        try:
            self._window.deferred_dock_in("Property", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
        except Exception:  # noqa: BLE001
            pass
        return self

    def _robot_management_group(self):
        with ui.CollapsableFrame("Robot Management", collapsed=False):
            with ui.VStack(spacing=6, height=0):
                ui.Label("Disconnect UFACTORY Studio before connecting", height=18, style={"color": 0xFFAAAAFF})
                self._labels["arm_ip"] = ui.Label(f"IP: {self.state.arm_ip}", height=18)
                self._conn_toggle_v = {"v": False}
                conn_btn = ui.Button("Connected: OFF", height=46)

                def conn_click():
                    self._conn_toggle_v["v"] = not self._conn_toggle_v["v"]
                    v = self._conn_toggle_v["v"]
                    conn_btn.text = f"Connected: {'ON' if v else 'OFF'}"
                    conn_btn.style = {"background_color": _ON if v else _OFF, "color": 0xFFFFFFFF}
                    if v:
                        self.state.request_connect()
                    else:
                        self.state.request_disconnect()
                conn_btn.set_clicked_fn(conn_click)
                conn_btn.style = {"background_color": _OFF, "color": 0xFFFFFFFF}
                self._conn_toggle_btn = conn_btn
                self._arm_toggle_v = {"v": False}
                arm_btn = ui.Button("Robot Enabled: OFF", height=46)

                def arm_click():
                    self._arm_toggle_v["v"] = not self._arm_toggle_v["v"]
                    v = self._arm_toggle_v["v"]
                    arm_btn.text = f"Robot Enabled: {'ON' if v else 'OFF'}"
                    arm_btn.style = {"background_color": _ON if v else _OFF, "color": 0xFFFFFFFF}
                    if v:
                        self.state.request_arm()
                    else:
                        self.state.request_disarm()
                arm_btn.set_clicked_fn(arm_click)
                arm_btn.style = {"background_color": _OFF, "color": 0xFFFFFFFF}
                self._arm_toggle_btn = arm_btn
                self._action("Snap Sim → Real Joints", self.state.request_snap_to_real, color=_ACT, height=46)
                ui.Label("Poses (app.json): shape the arm with FollowTarget, then save the joints", height=20)
                with ui.HStack(spacing=4, height=0):
                    self._action("Set HOME = joints", self.state.request_set_home_pose, color=_OFF, height=40)
                    self._action("Set SLEEP = joints", self.state.request_set_sleep_pose, color=_OFF, height=40)
                self._action("Recover from Stop", self.state.request_recover, color=_ACT, height=46)
                self._action("STOP", self.state.request_stop_arm, color=_ACT, height=46)
                self._action("EMERGENCY STOP", self.state.request_estop, color=_STOP, height=54)
                with ui.HStack(height=0):
                    ui.Label("Speed %", width=120)
                    spd = ui.IntSlider(min=0, max=100, height=28)
                    spd.model.set_value(int(self.state.speed_scale * 100))
                spd.model.add_value_changed_fn(
                    lambda m: setattr(self.state, "speed_scale", max(0.0, min(1.0, int(m.get_value_as_int()) / 100.0))))
                self._speed_model = spd.model
                self._labels["speed_eff"] = ui.Label("effective speed: -", height=16)
                self._labels["halt"] = ui.Label("", height=16, style={"color": 0xFF5555FF})
                with ui.CollapsableFrame("Diagnostics", collapsed=True):
                    with ui.VStack(spacing=4, height=0):
                        for k, t in [("mgmt_conn", "connection: -"), ("mgmt_state", "arm state: -"), ("mgmt_mode", "arm mode: -"),
                                     ("mgmt_error", "error: 0 (OK)"), ("mgmt_warn", "warn: 0 (OK)"),
                                     ("mgmt_last_err", "last error: none"), ("mgmt_ee_pos", "EE pos: -")]:
                            self._labels[k] = ui.Label(t, height=16)
                        with ui.HStack(height=0, spacing=4):
                            for i in range(3):
                                self._labels[f"jval_{i}"] = ui.Label(f"j{i+1}: --- °", height=16)
                        with ui.HStack(height=0, spacing=4):
                            for i in range(3, 6):
                                self._labels[f"jval_{i}"] = ui.Label(f"j{i+1}: --- °", height=16)
                        ui.Label("Joint Readback", height=16)
                        self._joint_sliders = []
                        for i in range(6):
                            lo_r, hi_r = self._joint_limits[i] if i < len(self._joint_limits) else (-3.14, 3.14)
                            with ui.HStack(height=0):
                                ui.Label(f"j{i+1}", width=30)
                                s = ui.FloatSlider(min=math.degrees(lo_r), max=math.degrees(hi_r), height=22)
                                s.model.set_value(0.0)
                                self._joint_sliders.append(s.model)

    def _devmode_group(self):
        with ui.CollapsableFrame("Program State", collapsed=False):
            with ui.VStack(spacing=6, height=0):
                self._labels["mode"] = ui.Label(f"mode: {self.state.mode}", height=22)
                _, self._mode_highlight = self._radio(_MODES, self.state.mode, self.state.request_mode, height=48)
                self._agent_btn, self._agent_v = self._toggle("Follow-Target Agent", False,
                                                              lambda v: setattr(self.state, "agent_on", v))
                with ui.CollapsableFrame("Debug viz", collapsed=True):
                    with ui.VStack(spacing=4, height=0):
                        self._toggle("Show Agent", False, lambda v: setattr(self.state, "viz_agent", v))
                        self._toggle("Show Seek Target", False, lambda v: setattr(self.state, "viz_targets", v))
                        self._toggle("Show Anchor + Ray", False, lambda v: setattr(self.state, "viz_anchor", v))

    def _motion_params_group(self):
        with ui.CollapsableFrame("Motion Parameters", collapsed=False):
            with ui.VStack(spacing=6, height=0):
                with ui.HStack(height=0):
                    ui.Label("Height (m)", width=120)
                    az = ui.FloatSlider(min=0.5, max=3.5, height=28)
                    az.model.set_value(self.state.anchor_z)
                az.model.add_value_changed_fn(lambda m: setattr(self.state, "anchor_z", float(m.get_value_as_float())))
                with ui.HStack(height=0):
                    ui.Label("Reach (m)", width=120)
                    rc = ui.FloatSlider(min=0.0, max=1.0, height=28)
                    rc.model.set_value(1.0)
                rc.model.add_value_changed_fn(lambda m: self.state.set_param("reach", m.get_value_as_float()))
                with ui.HStack(height=0):
                    ui.Label("Reach ramp (m)", width=120)    # dist_far: how far away a visitor is before the arm sits back at the anchor
                    rf = ui.FloatSlider(min=1.0, max=10.0, height=28)
                    rf.model.set_value(3.0)
                rf.model.add_value_changed_fn(lambda m: self.state.set_param("dist_far", m.get_value_as_float()))
                with ui.HStack(height=0):
                    ui.Label("Reach max (m)", width=120)   # the sphere about the shoulder: 0.85 is the arm fully straight
                    rm = ui.FloatSlider(min=0.5, max=0.85, height=28)
                    rm.model.set_value(self.state.reach_max)
                rm.model.add_value_changed_fn(lambda m: self.state.set_param("reach_max", m.get_value_as_float()))
                with ui.HStack(height=0):
                    ui.Label("Retreat (s)", width=120)
                    rd = ui.FloatSlider(min=0.5, max=10.0, height=28)
                    rd.model.set_value(3.0)
                rd.model.add_value_changed_fn(lambda m: self.state.set_param("retreat_duration", m.get_value_as_float()))
                with ui.HStack(height=0):
                    ui.Label("Sway amp (cm)", width=120)     # the barely-there wobble while following (0 = still)
                    sw = ui.FloatSlider(min=0.0, max=5.0, height=28)
                    sw.model.set_value(BUILTIN_PRESETS["Curious"]["sway_amp"] * 100.0)
                sw.model.add_value_changed_fn(lambda m: self.state.set_param("sway_amp", m.get_value_as_float() / 100.0))
                with ui.HStack(height=0):
                    ui.Label("Sway period (s)", width=120)
                    sp = ui.FloatSlider(min=3.0, max=15.0, height=28)
                    sp.model.set_value(BUILTIN_PRESETS["Curious"]["sway_period"])
                sp.model.add_value_changed_fn(lambda m: self.state.set_param("sway_period", m.get_value_as_float()))

    def _robot_group(self):
        with ui.CollapsableFrame("Visualization", collapsed=False):
            with ui.VStack(spacing=6, height=0):
                self._toggle("Show Collision Spheres", False, lambda v: setattr(self.state, "show_collision_spheres", v))

    def _behavior_group(self):
        with ui.CollapsableFrame("Personality (agent feel)", collapsed=False):
            with ui.VStack(spacing=6, height=0):
                ui.Label("Personality")
                with ui.HStack(spacing=4, height=0):
                    for name in _PRESETS:
                        self._action(name, (lambda n=name: self.state.request_preset(n)), color=_OFF, height=42)
                with ui.HStack(spacing=4, height=0):
                    self._action("Save Preset", self.state.request_save_preset, height=40)
                    self._action("Reset Preset", self.state.request_reset_preset, color=_OFF, height=40)
                with ui.HStack(height=0):
                    ui.Label("Gaze smoothing", width=120)
                    gs = ui.FloatSlider(min=0.01, max=1.0, height=28)
                    gs.model.set_value(self.state.gaze_steer)
                gs.model.add_value_changed_fn(lambda m: setattr(self.state, "gaze_steer", float(m.get_value_as_float())))

    def _control_group(self):
        with ui.CollapsableFrame("Control Link (OSC)", collapsed=False):
            with ui.VStack(spacing=6, height=0):
                self._labels["osc_status"] = ui.Label("control: waiting...", height=18)
                self._labels["osc_ping"] = ui.Label("last ping: --", height=18)
                if self._control is not None:
                    self._action("Restart Control Link", lambda: self._control.restart(), color=_ACT, height=40)

    def _attention_group(self):
        with ui.CollapsableFrame("Attention (tracker)", collapsed=True):
            with ui.VStack(spacing=4, height=0):
                ui.Label("Who gets followed when several people are active: activity first, then these", height=18)
                for key, label, lo, hi, dv in _TRACKER_PARAMS:
                    with ui.HStack(height=0):
                        ui.Label(label, width=120)
                        s = ui.FloatSlider(min=lo, max=hi, height=28)
                        s.model.set_value(float(self.state.tracker_params.get(key, dv)))
                    s.model.add_value_changed_fn(lambda m, k=key: self.state.set_tracker_param(k, m.get_value_as_float()))

    def _advanced_group(self):
        with ui.CollapsableFrame("Advanced (raw gains)", collapsed=True):
            with ui.VStack(spacing=4, height=0):
                for key, label, lo, hi, dv in _ADV_PARAMS:
                    with ui.HStack(height=0):
                        ui.Label(label, width=120)
                        s = ui.FloatSlider(min=lo, max=hi, height=28)
                        s.model.set_value(dv)
                    s.model.add_value_changed_fn(lambda m, k=key: self.state.set_param(k, m.get_value_as_float()))

    def _status_group(self):
        with ui.CollapsableFrame("Status", collapsed=False):
            with ui.VStack(spacing=3, height=0):
                for k, t in [("state", "robot: -"), ("perf", "fps: -  ctrl: - ms"), ("target", "target: -"),
                             ("robot_hz", "robot loop: - Hz")]:
                    self._labels[k] = ui.Label(t)

    def set_status(self, **kw):
        if not _HAS_UI or not self._labels:
            return
        if "behavior_state" in kw and "mode" in self._labels:
            self._labels["mode"].text = f"mode: {kw['behavior_state']}"
        if self._mode_highlight is not None:          # sync the radio with UIState (the control link may have changed it)
            self._mode_highlight(self.state.mode)
        if self._agent_btn is not None and self._agent_v is not None:
            if self._agent_v["v"] != self.state.agent_on:
                self._agent_v["v"] = self.state.agent_on
                v = self._agent_v["v"]
                self._agent_btn.text = f"Follow-Target Agent: {'ON' if v else 'OFF'}"
                self._agent_btn.style = {"background_color": _ON if v else _OFF, "color": 0xFFFFFFFF}
        if "state" in kw and self._arm_toggle_btn is not None:
            s = kw["state"]
            if s in ("DISCONNECTED", "MONITORING") and self._arm_toggle_v["v"]:
                self._arm_toggle_v["v"] = False
                self._arm_toggle_btn.text = "Robot Enabled: OFF"
                self._arm_toggle_btn.style = {"background_color": _OFF, "color": 0xFFFFFFFF}
            elif s in ("CONNECTED", "ENABLED") and not self._arm_toggle_v["v"]:
                self._arm_toggle_v["v"] = True
                self._arm_toggle_btn.text = "Robot Enabled: ON"
                self._arm_toggle_btn.style = {"background_color": _ON, "color": 0xFFFFFFFF}
        if "state" in kw and self._conn_toggle_btn is not None:
            s = kw["state"]
            if s == "DISCONNECTED" and self._conn_toggle_v["v"]:
                self._conn_toggle_v["v"] = False
                self._conn_toggle_btn.text = "Connected: OFF"
                self._conn_toggle_btn.style = {"background_color": _OFF, "color": 0xFFFFFFFF}
            elif s != "DISCONNECTED" and not self._conn_toggle_v["v"]:
                self._conn_toggle_v["v"] = True
                self._conn_toggle_btn.text = "Connected: ON"
                self._conn_toggle_btn.style = {"background_color": _ON, "color": 0xFFFFFFFF}
        if "state" in kw:
            self._labels["state"].text = f"robot: {kw['state']}"
        if "fps" in kw or "cs_ms" in kw:
            self._labels["perf"].text = f"fps: {kw.get('fps', 0):.1f}   ctrl: {kw.get('cs_ms', 0):.2f} ms"
        if "target_age" in kw:
            a = kw["target_age"]
            self._labels["target"].text = "target: STALE" if a is None or a > 0.5 else f"target age: {a*1e3:.0f} ms"
        if "robot_hz" in kw:
            self._labels["robot_hz"].text = f"robot loop: {kw['robot_hz']:.0f} Hz"
        if "speed_effective" in kw and kw["speed_effective"] is not None:
            self._labels["speed_eff"].text = f"effective speed: {kw['speed_effective']:.0%}"
        if "halt_reason" in kw:
            self._labels["halt"].text = f"HALTED: {kw['halt_reason']}" if kw["halt_reason"] else ""
        if "arm_connected" in kw and kw["arm_connected"] is not None:
            self._labels["mgmt_conn"].text = f"connection: {'YES' if kw['arm_connected'] else 'NO'}"
        if "arm_hw_state" in kw and kw["arm_hw_state"] is not None:
            s = kw["arm_hw_state"]
            _snames = {0: "normal", 1: "pause", 2: "stop", 3: "offline", 4: "stopped"}
            self._labels["mgmt_state"].text = f"arm state: {s} ({_snames.get(s, '?')})"
        if "arm_mode" in kw and kw["arm_mode"] is not None:
            m = kw["arm_mode"]
            _mnames = {0: "position", 1: "servo", 2: "joint_teach", 4: "joint_vel", 5: "cart_vel", 6: "joint_online", 7: "cart_online"}
            self._labels["mgmt_mode"].text = f"arm mode: {m} ({_mnames.get(m, '?')})"
        if "arm_error_code" in kw and kw["arm_error_code"] is not None:
            ec = kw["arm_error_code"]
            self._labels["mgmt_error"].text = f"error: {ec} ({'OK' if ec == 0 else 'ERROR'})"
        if "arm_warn_code" in kw and kw["arm_warn_code"] is not None:
            wc = kw["arm_warn_code"]
            self._labels["mgmt_warn"].text = f"warn: {wc} ({'OK' if wc == 0 else 'WARN'})"
        if "last_error" in kw:
            self._labels["mgmt_last_err"].text = f"last error: {kw['last_error'] or 'none'}"
        if "arm_position_m" in kw and kw["arm_position_m"] is not None:
            p = kw["arm_position_m"]
            self._labels["mgmt_ee_pos"].text = f"EE pos: x={p[0]:.3f} y={p[1]:.3f} z={p[2]:.3f} m"
        if self._speed_model is not None:              # sync the slider with external writes (OSC)
            ext = int(self.state.speed_scale * 100)
            if self._speed_model.get_value_as_int() != ext:
                self._speed_model.set_value(ext)
        if "control_connected" in kw and "osc_status" in self._labels:
            if kw["control_connected"]:
                self._labels["osc_status"].text = f"control: connected ({kw.get('control_ip') or '?'})"
            else:
                self._labels["osc_status"].text = "control: waiting..."
        if "control_ping_age" in kw and "osc_ping" in self._labels:
            pa = kw["control_ping_age"]
            self._labels["osc_ping"].text = "last ping: --" if pa is None else f"last ping: {pa:.1f}s ago"
        if "real_q" in kw and kw["real_q"] is not None:
            rq = kw["real_q"]
            for i in range(min(6, len(rq))):
                deg = math.degrees(rq[i])
                if f"jval_{i}" in self._labels:
                    self._labels[f"jval_{i}"].text = f"j{i+1}: {deg:.1f}°"
                if i < len(self._joint_sliders):
                    self._joint_sliders[i].set_value(deg)

    def destroy(self):
        if self._window is not None:
            self._window.destroy()
            self._window = None
