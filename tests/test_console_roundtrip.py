# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Integration test: OscConsole <-> ControlLink roundtrip. No Isaac required.

Usage:
    python tests/test_console_roundtrip.py [--loop N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

from mimus.osc import ControlLink, MODE_MAP  # noqa: E402
from mimus.ui_state import UIState  # noqa: E402
from osc_console import OscConsole  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, info=""):
    global PASS, FAIL
    PASS, FAIL = (PASS + 1, FAIL) if cond else (PASS, FAIL + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {info if not cond else ''}")


def run_once(cmd_port, status_port):
    global PASS, FAIL
    PASS = FAIL = 0

    ui = UIState()
    link = ControlLink(ui, in_port=cmd_port, out_port=status_port, status_hz=100.0)
    console = OscConsole(dest_ip="127.0.0.1", cmd_port=cmd_port, status_port=status_port)
    time.sleep(0.15)

    # ping/pong
    print("ping/pong")
    console.send_ping()
    time.sleep(0.2)
    check("pong roundtrip", console.pong_received)

    # mode commands
    print("mode commands")
    for osc_name, internal_name in MODE_MAP.items():
        console.send_mode(osc_name)
        time.sleep(0.1)
        check(f"mode {osc_name}", ui.mode == internal_name, f"got {ui.mode}")

    # agent
    print("agent")
    console.send_agent(True)
    time.sleep(0.1)
    check("agent on", ui.agent_on is True)
    console.send_agent(False)
    time.sleep(0.1)
    check("agent off", ui.agent_on is False)

    # speed
    print("speed")
    console.send_speed(0.75)
    time.sleep(0.1)
    check("speed 0.75", abs(ui.speed_scale - 0.75) < 0.01, f"got {ui.speed_scale}")

    # stop
    print("stop")
    ui.mode = "FollowTarget"
    console.send_stop()
    time.sleep(0.1)
    check("stop -> HOLD", ui.mode == "HOLD")
    check("stop_arm fired", ui.consume_stop_arm())

    # estop
    print("estop")
    console.send_estop()
    time.sleep(0.1)
    check("estop fired", ui.consume_estop())

    # status push
    print("status push")
    link.push_status(bridge_state="ENABLED", mode="FollowTarget", agent_on=True,
                        speed_scale=0.5, arm_error=0, arm_warn=0, arm_hw_state=0,
                        last_error="", ee_pos=[1.0, 2.0, 3.0])
    time.sleep(0.3)
    with console._lock:
        check("status received", console.last_status is not None)
        if console.last_status:
            check("status bridge", console.last_status[0] == "ENABLED")
            check("status mode", console.last_status[1] == "follow_target")
        check("ee received", console.last_ee is not None)
        if console.last_ee:
            check("ee z", abs(console.last_ee[2] - 3.0) < 0.01)

    # error push
    print("error push")
    link.push_error("test error")
    time.sleep(0.2)
    with console._lock:
        check("error received", console.last_error == "test error", f"got {console.last_error}")

    console.stop()
    link.stop()
    return PASS, FAIL


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=1, help="number of iterations")
    a = ap.parse_args()

    total_pass = total_fail = 0
    for i in range(a.loop):
        if a.loop > 1:
            print(f"\n===== iteration {i + 1}/{a.loop} =====")
        p, f = run_once(cmd_port=8300 + i * 2, status_port=8301 + i * 2)
        total_pass += p
        total_fail += f

    print(f"\nTOTAL: {total_pass} passed, {total_fail} failed")
    sys.exit(1 if total_fail else 0)
