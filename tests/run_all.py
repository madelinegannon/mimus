# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Runs every Isaac-free suite as a subprocess and aggregates. The end-to-end check needs Isaac Sim:
  python.bat app.py --test
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ["test_helpers.py", "test_config.py", "test_motion.py", "test_collision_model.py", "test_osc.py",
          "test_osc_ops.py", "test_console_roundtrip.py", "test_fake_tracker.py"]

results = []
for s in SUITES:
    print(f"\n========== {s} ==========", flush=True)
    rc = subprocess.call([sys.executable, os.path.join(HERE, s)])
    results.append((s, rc))

print("\n========== SUMMARY ==========")
allok = True
for s, rc in results:
    allok = allok and rc == 0
    print(f"  {'PASS' if rc == 0 else 'FAIL'}  {s}")
print("\nNOTE: the end-to-end headless check runs under Isaac Sim's Python: python.bat app.py --test")
sys.exit(0 if allok else 1)
