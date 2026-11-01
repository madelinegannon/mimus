# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Tests for the control-link extras: /arm/quit, /arm/cube, /arm/auto, /arm/ops,
TelemetryOut (/hb, /robot/q) and TrackerLink (/tracker/target). No Isaac required."""
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mimus.osc import ControlLink, TelemetryOut, _build  # noqa: E402
from mimus.tracker_link import TrackerLink  # noqa: E402
from mimus.ui_state import UIState  # noqa: E402
from pythonosc.osc_message import OscMessage  # noqa: E402

PASS = FAIL = 0


def check(name, cond, info=""):
    global PASS, FAIL
    PASS, FAIL = (PASS + 1, FAIL) if cond else (PASS, FAIL + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {info if not cond else ''}")


def send(dgram, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(dgram, ("127.0.0.1", port))
    s.close()


def listener(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.settimeout(1.0)
    return s


def recv_msg(sock, address, tries=20):
    for _ in range(tries):
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            return None
        m = OscMessage(data)
        if m.address == address:
            return m
    return None


IN_PORT, OUT_PORT, HB_PORT, Q_PORT, TRK_PORT, CTRL_PORT = 8300, 8301, 9300, 7311, 7301, 7312

print("== ControlLink new commands ==")
ui = UIState()
ex = ControlLink(ui, in_port=IN_PORT, out_port=OUT_PORT, status_hz=100.0)
peer = listener(OUT_PORT)
time.sleep(0.1)
send(_build(address="/arm/ping"), IN_PORT)          # discover the peer
time.sleep(0.1)
send(_build(address="/arm/quit"), IN_PORT)
time.sleep(0.1)
check("quit -> ui.consume_quit", ui.consume_quit() is True)
check("quit consumed once", ui.consume_quit() is False)
send(_build(0.5, 1.4, 1.6, address="/arm/cube"), IN_PORT)
time.sleep(0.1)
c = ui.consume_cube()
check("cube -> ui.consume_cube", c is not None and abs(c[1] - 1.4) < 1e-6, str(c))
check("cube consumed once", ui.consume_cube() is None)
check("auto_mode defaults on", ui.auto_mode is True)
send(_build(0, address="/arm/auto"), IN_PORT)
time.sleep(0.1)
check("auto 0 -> off", ui.auto_mode is False)
send(_build(1, address="/arm/auto"), IN_PORT)
time.sleep(0.1)
check("auto 1 -> on", ui.auto_mode is True)
ui.set_param("body_kp", 1.0)
ui.set_tracker_param("near_pref", 0.7)
check("tracker sliders queue apart from behaviour sliders (never into a preset)",
      ui.consume_params() == {"body_kp": 1.0} and ui.consume_tracker_params() == {"near_pref": 0.7})
check("tracker params consumed once", ui.consume_tracker_params() is None and ui.consume_params() is None)

print("== /arm/ops push ==")
ex.push_ops("RED", "PRESS RESET: E-stop C2", True)
m = recv_msg(peer, "/arm/ops")
check("ops received", m is not None and m.params[0] == "RED" and m.params[2] == 1, str(m and m.params))
check("ops rate-limited", ex.push_ops("GREEN", "x", False) is None and recv_msg(peer, "/arm/ops", 3) is None)
ex.stop()
peer.close()

print("== TelemetryOut ==")
hb = listener(HB_PORT)
qs = listener(Q_PORT)
t = TelemetryOut(name="isaac", hb_addr=("127.0.0.1", HB_PORT), q_addr=("127.0.0.1", Q_PORT), hb_hz=1000, q_hz=1000)
check("hb sent", t.push_hb({"fps": 60, "bridge": "ENABLED", "halted": None}) is True)
m = recv_msg(hb, "/hb")
check("hb name + json", m is not None and m.params[0] == "isaac" and json.loads(m.params[1])["fps"] == 60, str(m and m.params))
check("q sent", t.push_q([0, 0.26, -0.9, 0, 0.6, 0]) is True)
m = recv_msg(qs, "/robot/q")
check("q 6 floats", m is not None and len(m.params) == 6 and abs(m.params[2] + 0.9) < 1e-6, str(m and m.params))
ctrl = listener(CTRL_PORT)
t3 = TelemetryOut(hb_addr=("127.0.0.1", HB_PORT), q_addr=("127.0.0.1", Q_PORT), ctrl_addr=("127.0.0.1", CTRL_PORT))
check("tracker set sent", t3.push_tracker_set("near_pref", 0.5) is True)
m = recv_msg(ctrl, "/tracker/set")
check("/tracker/set carries key + float", m is not None and m.params[0] == "near_pref" and abs(m.params[1] - 0.5) < 1e-6, str(m and m.params))
t3.stop()
ctrl.close()
t2 = TelemetryOut(hb_addr=("127.0.0.1", HB_PORT), q_addr=("127.0.0.1", Q_PORT), hb_hz=1.0)
t2.push_hb({"a": 1})
check("hb rate-limited at 1 Hz", t2.push_hb({"a": 2}) is False)
check("hb force bypasses limit", t2.push_hb({"a": 3}, force=True) is True)
t.stop()
t2.stop()
hb.close()
qs.close()

print("== TrackerLink ==")
tl = TrackerLink(port=TRK_PORT)
time.sleep(0.1)
check("no target initially", tl.latest_target() is None and tl.age() is None)
send(_build(0.3, 1.5, 1.6, "engage", 7, 0.83, address="/tracker/target"), TRK_PORT)
time.sleep(0.1)
r = tl.latest_target()
check("target parsed", r is not None and abs(r[0][1] - 1.5) < 1e-6 and r[1] == "engage" and r[2] == 7
      and abs(r[3] - 0.83) < 1e-6 and r[4] < 0.5 and r[5] == 1, str(r))
send(_build(0.0, 0.0, 0.0, "bogus", 1, 0.1, address="/tracker/target"), TRK_PORT)
time.sleep(0.1)
check("bad hint ignored", tl.latest_target()[5] == 1)
send(_build(0.0, 0.0, 0.0, "idle", -1, 0.0, address="/tracker/target"), TRK_PORT)
time.sleep(0.1)
r = tl.latest_target()
check("idle target updates (staleness != nobody)", r[1] == "idle" and r[5] == 2)
send(_build(1.8, 0.0, 1.6, "sleep", -1, 0.0, address="/tracker/target"), TRK_PORT)
time.sleep(0.1)
r = tl.latest_target()
check("sleep hint accepted (unknown hints are dropped, so this must be listed)", r[1] == "sleep" and r[5] == 3, str(r))
tl.stop()

print(f"\ntest_osc_ops: {PASS} passed, {FAIL} failed")
sys.exit(0 if FAIL == 0 else 1)
