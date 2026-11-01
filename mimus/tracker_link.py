# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Receiver for the tracker's /tracker/target feed (see ASSEMBLY.md). No Isaac imports."""
from __future__ import annotations

import socket
import threading
import time

import numpy as np
from pythonosc.osc_message import OscMessage

DEFAULT_TRACKER_PORT = 7001
TRACKER_HINTS = ("idle", "watch", "engage", "reach", "sleep")


class TrackerLink:
    """Receives /tracker/target "fffsif" (x y z metres WORLD, hint, track_id, score). Non-blocking,
    latest-wins. The tracker sends every frame even when nobody is present ("idle" = nobody engaged,
    "sleep" = nobody worth being awake for), so staleness means the tracker is down, not an empty room."""

    def __init__(self, port: int = DEFAULT_TRACKER_PORT):
        self._lock = threading.Lock()
        self._xyz: np.ndarray | None = None
        self._hint = "idle"
        self._track_id = -1
        self._score = 0.0
        self._stamp = 0.0
        self._seq = 0
        self._port = port
        self._rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._rx.bind(("0.0.0.0", port))
        self._rx.settimeout(0.2)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="TrackerRx", daemon=True)
        self._thread.start()
        print(f"[tracker-link] listening on port {port}", flush=True)

    def _loop(self):
        while self._running:
            try:
                data, _ = self._rx.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = OscMessage(data)
                if msg.address == "/tracker/target" and len(msg.params) >= 6:
                    p = msg.params
                    hint = str(p[3])
                    if hint not in TRACKER_HINTS:
                        continue
                    with self._lock:
                        self._xyz = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float64)
                        self._hint = hint
                        self._track_id = int(p[4])
                        self._score = float(p[5])
                        self._stamp = time.monotonic()
                        self._seq += 1
            except Exception:  # noqa: BLE001
                pass

    def latest_target(self):
        """(xyz world metres, hint, track_id, score, age_s, seq) or None if never received."""
        with self._lock:
            if self._xyz is None:
                return None
            return (self._xyz.copy(), self._hint, self._track_id, self._score, time.monotonic() - self._stamp, self._seq)

    def age(self):
        with self._lock:
            return None if self._xyz is None else time.monotonic() - self._stamp

    def stop(self):
        self._running = False
        try:
            self._rx.close()
        except OSError:
            pass
