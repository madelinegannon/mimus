# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""The synthetic tracker feeds a TrackerLink like the real one: world-frame targets, live hint changes."""
import time

import numpy as np

import _common as T
from _common import check, wait_for

from mimus.fake_tracker import FakeTracker, person_at
from mimus.tracker_link import TrackerLink

PORT = 17911

print("== synthetic path ==")
p0, p1 = person_at(0.0), person_at(5.0)
check("in front of the base, head height", 1.2 < p0[0] < 2.0 and 1.3 < p0[2] < 1.7, str(p0))
check("moves", float(np.linalg.norm(p1 - p0)) > 0.1)
check("deterministic", np.array_equal(person_at(3.3), person_at(3.3)))

print("== stream ==")
link = TrackerLink(port=PORT)
ft = FakeTracker(port=PORT, hz=120.0)
ft.start()
check("targets arrive", wait_for(lambda: link.latest_target() is not None, 2.0))
r = link.latest_target()
check("engage hint with a track id", r[1] == "engage" and r[2] == 1 and r[3] > 0.5, str(r))
check("position matches the path at its time", float(np.linalg.norm(r[0] - person_at(ft.elapsed()))) < 0.05, str(r[0]))
ft.hint = "sleep"
check("hint change reaches the link", wait_for(lambda: link.latest_target()[1] == "sleep", 2.0), str(link.latest_target()))
check("sleep carries no track id", link.latest_target()[2] == -1)
try:
    ft.hint = "bogus"
    check("bad hint refused", False)
except ValueError:
    check("bad hint refused", True)
n = ft.sent
time.sleep(0.2)
check("keeps streaming at the asked rate", ft.sent - n >= 10, str(ft.sent - n))
ft.stop()
link.stop()

T.finish("test_fake_tracker")
