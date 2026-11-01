# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Unit tests for ControlLink (the OSC control surface). No Isaac required."""
import os
import socket
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mimus.osc import ControlLink, MODE_MAP, REVERSE_MODE_MAP, _build  # noqa: E402
from mimus.ui_state import UIState  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, info=""):
    global PASS, FAIL
    PASS, FAIL = (PASS + 1, FAIL) if cond else (PASS, FAIL + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {info if not cond else ''}")


# non-default ports to avoid clashes
IN_PORT = 8200
OUT_PORT = 8201


def send_osc(address, *args, port=IN_PORT):
    """Send a single OSC message via raw UDP."""
    dgram = _build(*args, address=address)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(dgram, ("127.0.0.1", port))
    s.close()


def make_listener(port):
    """Create a UDP socket that listens for incoming OSC messages."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.settimeout(1.0)
    return s


# ---- setup ----
ui = UIState()
link = ControlLink(ui, in_port=IN_PORT, out_port=OUT_PORT, status_hz=100.0)
time.sleep(0.15)

# ---- construction ----
print("construction")
check("not disabled", not link.disabled)
check("not connected initially", not link.connected)
check("peer_ip is None", link.peer_ip is None)
check("last_ping_age is None", link.last_ping_age is None)

# ---- mode command ----
print("\nmode commands")
for osc_name, internal_name in MODE_MAP.items():
    ui.mode = "FollowTarget"
    send_osc("/arm/mode", osc_name)
    time.sleep(0.1)
    check(f"mode {osc_name} -> {internal_name}", ui.mode == internal_name, f"got {ui.mode}")

# invalid mode
ui.mode = "HOLD"
send_osc("/arm/mode", "nonexistent_mode")
time.sleep(0.1)
check("invalid mode ignored", ui.mode == "HOLD", f"got {ui.mode}")

# ---- IP learning ----
print("\nIP learning")
check("peer_ip learned", link.peer_ip == "127.0.0.1", f"got {link.peer_ip}")
check("connected after first packet", link.connected)

# ---- agent toggle ----
print("\nagent toggle")
ui.agent_on = False
send_osc("/arm/agent", 1)
time.sleep(0.1)
check("agent on", ui.agent_on is True)

send_osc("/arm/agent", 0)
time.sleep(0.1)
check("agent off", ui.agent_on is False)

# ---- speed ----
print("\nspeed")
send_osc("/arm/speed", 0.42)
time.sleep(0.1)
check("speed set", abs(ui.speed_scale - 0.42) < 0.01, f"got {ui.speed_scale}")

send_osc("/arm/speed", 1.5)
time.sleep(0.1)
check("speed clamped high", ui.speed_scale == 1.0, f"got {ui.speed_scale}")

send_osc("/arm/speed", -0.5)
time.sleep(0.1)
check("speed clamped low", ui.speed_scale == 0.0, f"got {ui.speed_scale}")

# ---- stop ----
print("\nstop")
ui.mode = "FollowTarget"
send_osc("/arm/stop")
time.sleep(0.1)
check("stop sets HOLD", ui.mode == "HOLD", f"got {ui.mode}")
check("stop fires stop_arm", ui.consume_stop_arm())

# ---- estop ----
print("\nestop")
send_osc("/arm/estop")
time.sleep(0.1)
check("estop fires", ui.consume_estop())

# ---- robot management ----
print("\nrobot management")
send_osc("/arm/connect")
time.sleep(0.1)
check("connect fires", ui.consume_connect())

send_osc("/arm/disconnect")
time.sleep(0.1)
check("disconnect fires", ui.consume_disconnect())

send_osc("/arm/arm")
time.sleep(0.1)
check("arm fires", ui.consume_arm())

send_osc("/arm/disarm")
time.sleep(0.1)
check("disarm fires", ui.consume_disarm())

send_osc("/arm/recover")
time.sleep(0.1)
check("recover fires", ui.consume_recover())

send_osc("/arm/snap")
time.sleep(0.1)
check("snap fires", ui.consume_snap_to_real())

# ---- ping/pong ----
print("\nping/pong")
listener = make_listener(OUT_PORT)
send_osc("/arm/ping")
try:
    data, _ = listener.recvfrom(4096)
    from pythonosc.osc_message import OscMessage
    pong = OscMessage(data)
    check("pong received", pong.address == "/arm/pong")
except socket.timeout:
    check("pong received", False, "timeout")
listener.close()

check("last_ping_age set", link.last_ping_age is not None)
check("last_ping_age recent", link.last_ping_age < 1.0,
      f"got {link.last_ping_age}")

# ---- status push ----
print("\nstatus push")
listener = make_listener(OUT_PORT)
link.push_status(
    bridge_state="ENABLED", mode="FollowTarget", agent_on=True,
    speed_scale=0.5, arm_error=0, arm_warn=0, arm_hw_state=0,
    last_error="", ee_pos=[0.1, 0.2, 0.3],
)
try:
    data, _ = listener.recvfrom(4096)
    msg = OscMessage(data)
    check("status address", msg.address == "/arm/status")
    p = msg.params
    check("status bridge_state", p[0] == "ENABLED", f"got {p[0]}")
    check("status mode (osc name)", p[1] == "follow_target", f"got {p[1]}")
    check("status agent_on", p[2] == 1, f"got {p[2]}")
    check("status speed_scale", abs(p[3] - 0.5) < 0.01, f"got {p[3]}")

    # also receive /arm/ee
    data2, _ = listener.recvfrom(4096)
    ee = OscMessage(data2)
    check("ee address", ee.address == "/arm/ee")
    check("ee x", abs(ee.params[0] - 0.1) < 0.01, f"got {ee.params[0]}")
    check("ee y", abs(ee.params[1] - 0.2) < 0.01, f"got {ee.params[1]}")
    check("ee z", abs(ee.params[2] - 0.3) < 0.01, f"got {ee.params[2]}")
except socket.timeout:
    check("status push received", False, "timeout")
listener.close()

# ---- error push ----
print("\nerror push")
listener = make_listener(OUT_PORT)
link.push_error("test error message")
try:
    data, _ = listener.recvfrom(4096)
    msg = OscMessage(data)
    check("error address", msg.address == "/arm/error")
    check("error content", msg.params[0] == "test error message")
except socket.timeout:
    check("error push received", False, "timeout")
listener.close()

# ---- status rate limiting ----
print("\nrate limiting")
listener = make_listener(OUT_PORT)
# status_hz=100 so rate limiting is very permissive for tests above,
# but let's verify the mechanism works with a tight limit
link2 = ControlLink(ui, in_port=IN_PORT + 10, out_port=OUT_PORT + 10, status_hz=2.0)
send_osc("/arm/ping", port=IN_PORT + 10)
time.sleep(0.15)
listener2 = make_listener(OUT_PORT + 10)
count = 0
for _ in range(10):
    link2.push_status(bridge_state="ENABLED", mode="HOLD")
count_received = 0
listener2.settimeout(0.3)
while True:
    try:
        listener2.recvfrom(4096)
        count_received += 1
    except socket.timeout:
        break
check("rate limited (<=3 of 10 sent)", count_received <= 3, f"got {count_received}")
listener2.close()
link2.stop()

# ---- restart ----
print("\nrestart")
ok = link.restart()
check("restart succeeds", ok)
check("connected reset after restart", not link.connected)
send_osc("/arm/mode", "idle")
time.sleep(0.15)
check("works after restart", ui.mode == "IDLE", f"got {ui.mode}")

# ---- cleanup ----
link.stop()
print(f"\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
