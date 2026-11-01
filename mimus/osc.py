# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""OSC control surface for the app (any client: a console, a tablet app, a supervisor) and the
telemetry senders. Raw UDP sockets + python-osc message parsing. No Isaac imports.

client --> ControlLink (port 8100): /arm/mode, /arm/agent, /arm/stop, /arm/estop, /arm/speed, ...
ControlLink --> client (port 8101): /arm/status (pushed), /arm/ee, /arm/error, /arm/ops, /arm/pong
"""
from __future__ import annotations

import json
import socket
import threading
import time

from pythonosc.osc_message import OscMessage
from pythonosc.osc_message_builder import OscMessageBuilder

DEFAULT_IN_PORT = 8100
DEFAULT_OUT_PORT = 8101
DEFAULT_STATUS_HZ = 30.0
_STALE_S = 5.0

MODE_MAP = {
    "follow_target": "FollowTarget",
    "follow_sensor": "FollowTargetSensor",
    "idle": "IDLE",
    "sleep": "SLEEP",
    "home": "HOME",
    "hold": "HOLD",
}
REVERSE_MODE_MAP = {v: k for k, v in MODE_MAP.items()}


def _build(*args_list, address="/"):
    b = OscMessageBuilder(address=address)
    for a in args_list:
        if isinstance(a, bool):
            b.add_arg(int(a), OscMessageBuilder.ARG_TYPE_INT)
        elif isinstance(a, int):
            b.add_arg(a, OscMessageBuilder.ARG_TYPE_INT)
        elif isinstance(a, float):
            b.add_arg(a, OscMessageBuilder.ARG_TYPE_FLOAT)
        elif isinstance(a, str):
            b.add_arg(a, OscMessageBuilder.ARG_TYPE_STRING)
        else:
            b.add_arg(a)
    return b.build().dgram


class ControlLink:
    """OSC control link. Non-blocking, always-on; the first client to send becomes the status peer."""

    def __init__(self, ui_state=None, in_port=DEFAULT_IN_PORT, out_port=DEFAULT_OUT_PORT, status_hz=DEFAULT_STATUS_HZ):
        self._ui = ui_state
        self._in_port = in_port
        self._out_port = out_port
        self._status_hz = status_hz
        self._lock = threading.Lock()
        self._peer_addr = None
        self._last_rx_t = 0.0
        self._last_ping_t = 0.0
        self._last_status_t = 0.0
        self._last_ops_t = 0.0
        self._disabled = False
        self._running = False
        self._rx = None
        self._tx = None
        self._thread = None
        self._start()

    def _start(self):
        try:
            rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            rx.bind(("0.0.0.0", self._in_port))
            rx.settimeout(0.2)
        except OSError as e:
            print(f"[control] WARNING: cannot bind port {self._in_port}: {e}", flush=True)
            self._disabled = True
            return
        self._rx = rx
        self._tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._disabled = False
        self._running = True
        self._thread = threading.Thread(target=self._rx_loop, name="ControlLinkRx", daemon=True)
        self._thread.start()
        print(f"[control] listening on port {self._in_port}", flush=True)

    def restart(self):
        """Tear down and rebind. Returns True on success."""
        self.stop()
        with self._lock:
            self._peer_addr = None
            self._last_rx_t = 0.0
            self._last_ping_t = 0.0
        self._start()
        return not self._disabled

    def stop(self):
        self._running = False
        for s in (self._rx, self._tx):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        self._rx = self._tx = None

    def _rx_loop(self):
        while self._running:
            try:
                data, addr = self._rx.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            with self._lock:
                self._last_rx_t = time.monotonic()
                if self._peer_addr is None:
                    self._peer_addr = (addr[0], self._out_port)
                    print(f"[control] peer discovered: {addr[0]}", flush=True)
            try:
                self._dispatch(OscMessage(data), addr)
            except Exception:  # noqa: BLE001
                pass

    def _dispatch(self, msg, addr):
        a, params, ui = msg.address, msg.params, self._ui
        if a == "/arm/mode" and params:
            mapped = MODE_MAP.get(str(params[0]).lower())
            if mapped and ui is not None:
                ui.request_mode(mapped)
        elif a == "/arm/agent" and params:
            if ui is not None:
                ui.agent_on = bool(int(params[0]))
        elif a == "/arm/stop":
            if ui is not None:
                ui.mode = "HOLD"
                ui.request_stop_arm()
        elif a == "/arm/estop":
            if ui is not None:
                ui.request_estop()
        elif a == "/arm/speed" and params:
            if ui is not None:
                ui.speed_scale = max(0.0, min(1.0, float(params[0])))
        elif a == "/arm/connect":
            if ui is not None:
                ui.request_connect()
        elif a == "/arm/disconnect":
            if ui is not None:
                ui.request_disconnect()
        elif a == "/arm/arm":
            if ui is not None:
                ui.request_arm()
        elif a == "/arm/disarm":
            if ui is not None:
                ui.request_disarm()
        elif a == "/arm/recover":
            if ui is not None:
                ui.request_recover()
        elif a == "/arm/snap":
            if ui is not None:
                ui.request_snap_to_real()
        elif a == "/arm/ping":
            self._send(_build(address="/arm/pong"), addr=(addr[0], self._out_port))
            with self._lock:
                self._last_ping_t = time.monotonic()
        elif a == "/arm/quit":
            if ui is not None:
                ui.request_quit()
        elif a == "/arm/cube" and len(params) >= 3:
            if ui is not None:
                ui.request_cube(float(params[0]), float(params[1]), float(params[2]))
        elif a == "/arm/auto" and params:
            if ui is not None:
                ui.auto_mode = bool(int(params[0]))

    def _send(self, dgram, addr=None):
        if self._tx is None:
            return
        target = addr
        if target is None:
            with self._lock:
                target = self._peer_addr
        if target is None:
            return
        try:
            self._tx.sendto(dgram, target)
        except OSError:
            pass

    def push_status(self, bridge_state="DISCONNECTED", mode="HOLD", agent_on=False, speed_scale=0.0,
                    arm_error=0, arm_warn=0, arm_hw_state=0, last_error="", ee_pos=None):
        if self._disabled:
            return
        now = time.monotonic()
        if now - self._last_status_t < 1.0 / self._status_hz:
            return
        self._last_status_t = now
        osc_mode = REVERSE_MODE_MAP.get(mode, mode.lower())
        self._send(_build(str(bridge_state), str(osc_mode), int(agent_on), float(speed_scale),
                          int(arm_error or 0), int(arm_warn or 0), int(arm_hw_state or 0), str(last_error or ""),
                          address="/arm/status"))
        if ee_pos is not None:
            self._send(_build(float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2]), address="/arm/ee"))

    def push_error(self, error_msg):
        if self._disabled:
            return
        self._send(_build(str(error_msg), address="/arm/error"))

    def push_ops(self, state, message, needs_recover=False, hz=1.0):
        """Plain-language banner: state GREEN|AMBER|RED, message, needs_recover flag."""
        if self._disabled:
            return
        now = time.monotonic()
        if now - self._last_ops_t < 1.0 / hz:
            return
        self._last_ops_t = now
        self._send(_build(str(state), str(message), int(bool(needs_recover)), address="/arm/ops"))

    @property
    def connected(self):
        if self._disabled:
            return False
        with self._lock:
            if self._peer_addr is None:
                return False
            return (time.monotonic() - self._last_rx_t) < _STALE_S

    @property
    def peer_ip(self):
        with self._lock:
            return self._peer_addr[0] if self._peer_addr else None

    @property
    def last_ping_age(self):
        with self._lock:
            if self._last_ping_t == 0.0:
                return None
            return time.monotonic() - self._last_ping_t

    @property
    def disabled(self):
        return self._disabled


class TelemetryOut:
    """Bind-free senders: /hb heartbeat (name + JSON metrics) for a supervisor, /robot/q for the tracker's
    robot self-mask, /tracker/set for the tracker's tuning sliders. Always constructed; never fails loudly."""

    def __init__(self, name="mimus", hb_addr=("127.0.0.1", 9000), q_addr=("127.0.0.1", 7011),
                 hb_hz=1.0, q_hz=30.0, ctrl_addr=("127.0.0.1", 7012)):
        self._name = name
        self._hb_addr = tuple(hb_addr)
        self._q_addr = tuple(q_addr)
        self._ctrl_addr = tuple(ctrl_addr)
        self._hb_dt = 1.0 / hb_hz
        self._q_dt = 1.0 / q_hz
        self._last_hb = 0.0
        self._last_q = 0.0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def push_hb(self, metrics: dict, force=False):
        now = time.monotonic()
        if not force and now - self._last_hb < self._hb_dt:
            return False
        self._last_hb = now
        try:
            self._sock.sendto(_build(self._name, json.dumps(metrics, separators=(",", ":"), default=str), address="/hb"),
                              self._hb_addr)
            return True
        except OSError:
            return False

    def push_q(self, q):
        now = time.monotonic()
        if now - self._last_q < self._q_dt:
            return False
        self._last_q = now
        try:
            self._sock.sendto(_build(*[float(v) for v in list(q)[:6]], address="/robot/q"), self._q_addr)
            return True
        except (OSError, TypeError):
            return False

    def push_tracker_set(self, key: str, value: float):
        """A tracker tuning slider: /tracker/set <key> <value> to the tracker's control port."""
        try:
            self._sock.sendto(_build(str(key), float(value), address="/tracker/set"), self._ctrl_addr)
            return True
        except (OSError, TypeError, ValueError):
            return False

    def stop(self):
        try:
            self._sock.close()
        except OSError:
            pass
