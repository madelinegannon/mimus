# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Shared flags between the panel / control link (writers) and the control loop (reader). Plain
Python, no omni imports, so it is usable headless and in tests."""
from __future__ import annotations


class UIState:
    def __init__(self):
        self.mode = "FollowTarget"          # one of motion.MODES
        self.agent_on = True                # Follow-Target spring agent overlay
        self.anchor_z = 1.1                 # vertical anchor [base, Z] for the FollowTargetSensor ray (world height)
        self.reach_max = 0.85               # reach sphere about the shoulder; the app sets both from app.json
        self.tracker_params = {}            # starting values for the Attention sliders (sent to the tracker)
        self.viz_agent = False
        self.viz_targets = False
        self.viz_anchor = False
        self.speed_scale = 0.2              # the operator's runtime cap; the app seeds it from the bridge
        self.show_collision_spheres = False
        self.gaze_steer = 1.0
        self._stop = False
        self.arm_ip = ""
        self._want_connect = False
        self._want_disconnect = False
        self._want_arm = False
        self._want_disarm = False
        self._want_recover = False
        self._want_stop_arm = False
        self._want_estop = False
        self._want_snap_to_real = False
        self._set_home_pose = False
        self._set_sleep_pose = False
        self.auto_mode = True               # tracker hints may switch IDLE / SLEEP / FollowTargetSensor
        self._want_quit = False
        self._cube_target = None
        self._preset = None
        self._save_preset = self._reset_preset = False
        self._params = {}
        self._tracker_params = {}

    def _pop_bool(self, attr):
        v = getattr(self, attr)
        setattr(self, attr, False)
        return v

    def request_stop(self):
        self._stop = True

    def consume_stop(self):
        return self._pop_bool("_stop")

    def request_mode(self, name):
        self.mode = name

    def request_connect(self):
        self._want_connect = True

    def consume_connect(self):
        return self._pop_bool("_want_connect")

    def request_disconnect(self):
        self._want_disconnect = True

    def consume_disconnect(self):
        return self._pop_bool("_want_disconnect")

    def request_arm(self):
        self._want_arm = True

    def consume_arm(self):
        return self._pop_bool("_want_arm")

    def request_disarm(self):
        self._want_disarm = True

    def consume_disarm(self):
        return self._pop_bool("_want_disarm")

    def request_recover(self):
        self._want_recover = True

    def consume_recover(self):
        return self._pop_bool("_want_recover")

    def request_stop_arm(self):
        self._want_stop_arm = True

    def consume_stop_arm(self):
        return self._pop_bool("_want_stop_arm")

    def request_estop(self):
        self._want_estop = True

    def consume_estop(self):
        return self._pop_bool("_want_estop")

    def request_snap_to_real(self):
        self._want_snap_to_real = True

    def consume_snap_to_real(self):
        return self._pop_bool("_want_snap_to_real")

    def request_set_home_pose(self):
        self._set_home_pose = True

    def consume_set_home_pose(self):
        return self._pop_bool("_set_home_pose")

    def request_set_sleep_pose(self):
        self._set_sleep_pose = True

    def consume_set_sleep_pose(self):
        return self._pop_bool("_set_sleep_pose")

    def request_quit(self):
        self._want_quit = True

    def consume_quit(self):
        return self._pop_bool("_want_quit")

    def request_cube(self, x, y, z):
        self._cube_target = (float(x), float(y), float(z))

    def consume_cube(self):
        c, self._cube_target = self._cube_target, None
        return c

    def request_preset(self, name):
        self._preset = name

    def consume_preset(self):
        p, self._preset = self._preset, None
        return p

    def request_save_preset(self):
        self._save_preset = True

    def consume_save_preset(self):
        return self._pop_bool("_save_preset")

    def request_reset_preset(self):
        self._reset_preset = True

    def consume_reset_preset(self):
        return self._pop_bool("_reset_preset")

    def set_param(self, key, val):
        self._params[key] = float(val)

    def consume_params(self):
        if not self._params:
            return None
        p, self._params = self._params, {}
        return p

    def set_tracker_param(self, key, val):
        self._tracker_params[key] = float(val)

    def consume_tracker_params(self):
        if not self._tracker_params:
            return None
        p, self._tracker_params = self._tracker_params, {}
        return p
