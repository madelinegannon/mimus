# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""OSC console for the app's control link: sends commands, prints status pushes.

Usage:
    python examples/osc_console.py [--ip 127.0.0.1] [--cmd-port 8100] [--status-port 8101]

Interactive: cycles through modes every 3 seconds, prints status pushes.
Ctrl-C to stop.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pythonosc.osc_message import OscMessage  # noqa: E402
from mimus.osc import _build, MODE_MAP  # noqa: E402


class OscConsole:
    """Sends OSC commands to ControlLink, receives status pushes."""

    def __init__(self, dest_ip="127.0.0.1", cmd_port=8100, status_port=8101):
        self._dest = (dest_ip, cmd_port)
        self._tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self._lock = threading.Lock()
        self.last_status = None
        self.last_ee = None
        self.last_error = None
        self.pong_received = False

        self._rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._rx.bind(("0.0.0.0", status_port))
        self._rx.settimeout(0.5)
        self._running = True
        self._thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._thread.start()

    def _rx_loop(self):
        while self._running:
            try:
                data, _ = self._rx.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = OscMessage(data)
                with self._lock:
                    if msg.address == "/arm/status":
                        self.last_status = msg.params
                    elif msg.address == "/arm/ee":
                        self.last_ee = msg.params
                    elif msg.address == "/arm/error":
                        self.last_error = msg.params[0] if msg.params else ""
                    elif msg.address == "/arm/pong":
                        self.pong_received = True
            except Exception:
                pass

    def _send(self, address, *args):
        self._tx.sendto(_build(*args, address=address), self._dest)

    def send_mode(self, name):
        self._send("/arm/mode", name)

    def send_agent(self, on):
        self._send("/arm/agent", int(on))

    def send_stop(self):
        self._send("/arm/stop")

    def send_estop(self):
        self._send("/arm/estop")

    def send_speed(self, val):
        self._send("/arm/speed", float(val))

    def send_connect(self):
        self._send("/arm/connect")

    def send_disconnect(self):
        self._send("/arm/disconnect")

    def send_arm(self):
        self._send("/arm/arm")

    def send_disarm(self):
        self._send("/arm/disarm")

    def send_recover(self):
        self._send("/arm/recover")

    def send_snap(self):
        self._send("/arm/snap")

    def send_ping(self):
        with self._lock:
            self.pong_received = False
        self._send("/arm/ping")

    def stop(self):
        self._running = False
        for s in (self._rx, self._tx):
            try:
                s.close()
            except OSError:
                pass


def _fmt_status(params):
    if not params or len(params) < 8:
        return str(params)
    return (f"bridge={params[0]}  mode={params[1]}  agent={'ON' if params[2] else 'OFF'}  "
            f"speed={params[3]:.0%}  err={params[4]}  warn={params[5]}  "
            f"hw={params[6]}  last_err={params[7] or '-'}")


def _fmt_ee(params):
    if not params or len(params) < 3:
        return str(params)
    return f"x={params[0]:.3f}  y={params[1]:.3f}  z={params[2]:.3f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OSC console")
    ap.add_argument("--ip", default="127.0.0.1", help="app IP")
    ap.add_argument("--cmd-port", type=int, default=8100)
    ap.add_argument("--status-port", type=int, default=8101)
    a = ap.parse_args()

    console = OscConsole(dest_ip=a.ip, cmd_port=a.cmd_port, status_port=a.status_port)
    modes = list(MODE_MAP.keys())
    mode_idx = 0

    print(f"[console] sending to {a.ip}:{a.cmd_port}, listening on :{a.status_port}")
    print("[console] ping...")
    console.send_ping()
    time.sleep(0.5)
    print(f"[console] pong: {'YES' if console.pong_received else 'NO'}")

    try:
        while True:
            m = modes[mode_idx % len(modes)]
            print(f"\n[console] -> /arm/mode {m}")
            console.send_mode(m)
            mode_idx += 1

            for _ in range(6):
                time.sleep(0.5)
                with console._lock:
                    if console.last_status:
                        print(f"  status: {_fmt_status(console.last_status)}")
                    if console.last_ee:
                        print(f"  ee:     {_fmt_ee(console.last_ee)}")
                    if console.last_error:
                        print(f"  ERROR:  {console.last_error}")
                        console.last_error = None
    except KeyboardInterrupt:
        print("\n[console] stopped")
    finally:
        console.stop()
