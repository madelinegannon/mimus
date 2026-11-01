# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""A synthetic tracker for the headless self-check and for trying the app with no camera: one person
walking a slow figure in front of the base, emitted as /tracker/target in the world frame. `hint` is
settable live, so a test can send the arm to sleep the way the real tracker does. Deterministic."""
from __future__ import annotations

import argparse
import socket
import threading
import time

import numpy as np
from pythonosc.osc_message_builder import OscMessageBuilder

from .tracker_link import DEFAULT_TRACKER_PORT, TRACKER_HINTS


def person_at(t: float) -> np.ndarray:
    """World-frame head position of the synthetic visitor at time t: 1.3-1.9 m out, swaying."""
    x = 1.6 + 0.3 * np.sin(2 * np.pi * 0.05 * t)
    y = 0.6 * np.sin(2 * np.pi * 0.10 * t)
    z = 1.5 + 0.05 * np.sin(2 * np.pi * 0.20 * t)
    return np.array([x, y, z], dtype=np.float64)


def pack_target(xyz, hint: str, track_id: int, score: float) -> bytes:
    b = OscMessageBuilder(address="/tracker/target")
    for v in xyz:
        b.add_arg(float(v), OscMessageBuilder.ARG_TYPE_FLOAT)
    b.add_arg(str(hint), OscMessageBuilder.ARG_TYPE_STRING)
    b.add_arg(int(track_id), OscMessageBuilder.ARG_TYPE_INT)
    b.add_arg(float(score), OscMessageBuilder.ARG_TYPE_FLOAT)
    return b.build().dgram


class FakeTracker:
    def __init__(self, port: int = DEFAULT_TRACKER_PORT, host: str = "127.0.0.1", hz: float = 60.0,
                 track_id: int = 1, hint: str = "engage"):
        self._dest = (host, int(port))
        self._dt = 1.0 / float(hz)
        self._track_id = int(track_id)
        self._hint = hint
        self._lock = threading.Lock()
        self._tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._running = False
        self._t0 = None
        self._thread = threading.Thread(target=self._loop, name="FakeTrackerTx", daemon=True)
        self.sent = 0

    @property
    def hint(self) -> str:
        with self._lock:
            return self._hint

    @hint.setter
    def hint(self, value: str):
        if value not in TRACKER_HINTS:
            raise ValueError(f"hint {value!r} not in {TRACKER_HINTS}")
        with self._lock:
            self._hint = value

    def elapsed(self) -> float:
        return 0.0 if self._t0 is None else time.monotonic() - self._t0

    def start(self):
        self._t0 = time.monotonic()
        self._running = True
        self._thread.start()

    def _loop(self):
        while self._running:
            t = self.elapsed()
            hint = self.hint
            xyz = person_at(t)
            tid = self._track_id if hint not in ("idle", "sleep") else -1
            try:
                self._tx.sendto(pack_target(xyz, hint, tid, 0.9), self._dest)
                self.sent += 1
            except OSError:
                pass
            time.sleep(self._dt)

    def stop(self):
        self._running = False
        try:
            self._tx.close()
        except OSError:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="emit a synthetic /tracker/target stream")
    ap.add_argument("--port", type=int, default=DEFAULT_TRACKER_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--hz", type=float, default=60.0)
    ap.add_argument("--hint", default="engage", choices=TRACKER_HINTS)
    a = ap.parse_args(argv)
    ft = FakeTracker(port=a.port, host=a.host, hz=a.hz, hint=a.hint)
    ft.start()
    print(f"[fake-tracker] /tracker/target @{a.hz:.0f} Hz -> {a.host}:{a.port} hint={a.hint}; Ctrl-C to stop", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        ft.stop()


if __name__ == "__main__":
    main()
