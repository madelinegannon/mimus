# Assembling the three packages

```
 depth cameras ──> rt_depth_tracker ──/tracker/target (UDP 7001)──> mimus (Isaac Sim) ──> rt_xarm_bridge ──> xArm 850
                         ▲                                                 │                          (125 Hz servo stream,
                         └───────────── /robot/q (UDP 7011) ───────────────┘                           gated)
```

One world frame for all three: Z up, origin on the floor under the robot base, +x toward the
people. The tracker reports targets in it; the app places the robot base in it from `app.json`
(`mount_z`, `base_xy`, `base_yaw_deg`); the tracker needs the same three values in its own config so
it can mask the arm out of its depth clouds at the pose the app reports on `/robot/q`.

## Ports (defaults; every one is a flag)

| Port | Direction | Message | Flag |
|---|---|---|---|
| 7010 | camera process → tracker, loopback | `/depth`, `/intrinsics` | tracker |
| 7001 | tracker → app | `/tracker/target` `fffsif`: x y z (m, world), hint, track id, score; every frame, `idle`/`sleep` when nobody is there | `--tracker-port` |
| 7011 | app → tracker, loopback | `/robot/q`: six measured joint angles (rad) at 30 Hz | `--q-port` |
| 7012 | app → tracker | `/tracker/set key value`: the Attention sliders | `--tracker-ctrl-port` |
| 8100 / 8101 | any client ↔ app | `/arm/*` commands in, status out | `ControlLink` |
| 9000 | app → a supervisor, loopback | `/hb name json`: once a second | `--hb-port` |

The hints the app understands: `idle` and `sleep` (nobody worth being awake for: the tracker owns
these and the app folds accordingly), `watch`, `engage` and `reach` (a person; `reach` means the
target is a hand reaching toward the arm, and the behaviour retreats once from it).

## Launch order

1. The tracker's camera capture, then the tracker itself (`rt_depth_tracker`, see its README).
   Confirm `/tracker/target` arrives: the app's panel shows "target age" instead of STALE.
2. `rt-xarm-bridge check --arm xarm --arm-ip <controller-ip> --safety safety.yaml` until it says
   SAFETY OK.
3. `python.bat app.py --arm xarm --arm-ip <controller-ip>` to monitor, then with
   `--allow-motion --safety safety.yaml` to move. Port 7001 must be free: a second app on the same
   machine would need `--tracker-port`.

Each step can be tested without the others: `python.bat -m mimus.fake_tracker` stands in
for step 1, `--arm mock` stands in for the arm, and `python.bat app.py --test` runs the whole chain
headless with both stand-ins.

## Supervision

Nothing in these packages restarts itself or runs without an operator unless `--auto-recover` is
asked for explicitly. The `/hb` heartbeat on port 9000 exists for whatever supervises them.
