# Mimus

Mimus is a robot arm with a curious personality, for NVIDIA Isaac Sim and a UFACTORY xArm 850. It has
been exhibited publicly since 2016; this is the 2026 generation of its software, released as open source. A depth
tracker reports who is in the room and where their head or reaching hand is; a behaviour layer
turns that into an end-effector target with a spring-damped "seek and arrive" character (lean in,
back off, sniff once, idle sweeps, sleep and breathe); Lula RMPflow turns the target into joint
motion on a digital twin; and `rt_xarm_bridge` streams the twin's joints to the real arm behind
explicit safety gates. Everything runs with a mock arm and a synthetic tracker out of the box.

> **Read [SAFETY.md](SAFETY.md) before connecting a real arm.** The behaviours move the arm
> toward people on purpose. The gates in `rt_xarm_bridge` make accidental motion hard; they do not
> replace a risk assessment, a hardware e-stop, guarding, or a correctly configured controller.

This is one of three packages that assemble into the installation; [ASSEMBLY.md](ASSEMBLY.md)
wires them together:

| Package | Role |
|---|---|
| [`rt_depth_tracker`](https://github.com/madelinegannon/rt_depth_tracker) | perception: depth cameras in, `/tracker/target` out; never commands a robot |
| [`rt_xarm_bridge`](https://github.com/madelinegannon/rt_xarm_bridge) | the 125 Hz servo bridge and every motion gate |
| `mimus` (this repo) | behaviour + RMPflow in Isaac Sim; consumes the tracker, drives the bridge |

## Prerequisites

- NVIDIA Isaac Sim 5.x. Every command below runs with Isaac Sim's own Python (`python.bat` on
  Windows, `python.sh` on Linux), never the system interpreter.
- A UFACTORY xArm 850 only when you want to move a real arm; the mock arm needs nothing.

## Install

```bash
python.bat -m pip install git+https://github.com/madelinegannon/rt_xarm_bridge.git
python.bat -m pip install -e .
python.bat tools/fetch_assets.py
```

`fetch_assets.py` copies NVIDIA's stock xArm 850 USD and a grid environment from the Isaac Sim
assets into `usd/` (git-ignored) and bakes the joint drive `maxForce` the behaviours expect. If
your Isaac version keeps the robot elsewhere, pass `--robot <path under the assets root>`.

## Run

```bash
python.bat app.py --test                                   # headless self-check: synthetic tracker + MockArm
python.bat app.py                                          # GUI with the mock arm, FollowTarget mode (drag the red cube)
python.bat -m mimus.fake_tracker                   # in another shell: a synthetic visitor on /tracker/target
python.bat app.py --mode follow_sensor                     # ... and the arm follows it
```

With the real tracker running (see ASSEMBLY.md) the same `--mode follow_sensor` follows people.

### A real arm

1. Set up the controller and write `safety.yaml` as described in `rt_xarm_bridge`'s README and
   SAFETY.md, and prove it with `rt-xarm-bridge check --arm xarm --arm-ip <controller-ip> --safety safety.yaml`.
2. Monitor first, with no motion possible:
   ```bash
   python.bat app.py --arm xarm --arm-ip <controller-ip>
   ```
   The panel shows the measured joints; "Robot Enabled" refuses, because the app was started
   without `--allow-motion`.
3. Then, with a hand on the e-stop:
   ```bash
   python.bat app.py --arm xarm --arm-ip <controller-ip> --allow-motion --safety safety.yaml
   ```
   Press **Robot Enabled** in the panel. The bridge verifies the controller against `safety.yaml`,
   enables at the arm's *measured* pose, eases to home at the startup rate, and ramps the speed cap
   up from zero to the configured 0.2. Raise the Speed slider only once you have watched it move.
4. `--engage` arms at launch instead of waiting for the button; `--auto-recover` lets the bridge
   clear transient controller faults and reconnect by itself. Both are for supervised runs only.

## Configuration

`config/app.example.json` is the built-in default: the robot base on the floor at the world origin.
Copy it to `config/app.json` (git-ignored) and edit; `config/app.local.json` overlays it for
per-machine values. Keys:

| Key | Meaning |
|---|---|
| `mount_z`, `base_xy`, `base_yaw_deg` | where the base flange is in the world frame; must match the tracker's config |
| `home_q_rad`, `sleep_q_rad` | the two resting poses (the panel can save the current joints into them) |
| `unwind_rate_rad_s`, `unwind_min_s`, `startup_rate_rad_s` | timed joint moves: HOME/SLEEP, and the slower startup ease |
| `speed_scale` | the bridge's starting cap, at most 0.2; the panel slider raises it live, never persisted |
| `speed_slew_per_s` | how fast a raised cap ramps in (cuts apply at once) |
| `anchor_z_m` | the rest height of the follow behaviour, a world height |
| `reach_max_m` | the sphere about the shoulder nothing is allowed outside of (0.85 = the arm straight) |
| `tracker_params` | starting values for the Attention sliders, forwarded to the tracker |

Personality presets (Curious, Playful, Tired, Cautious) live in `mimus/motion.py`; the
panel's "Save Preset" writes overrides to `config/behavior_presets.json`. Their idle and look-at
heights are world heights written for heads at 1.3 to 1.8 m; on a floor-mounted base the reach
sphere clamps them, so lower `idle_ee_z` and friends or raise `mount_z` to taste.

## Modes

`FollowTarget` (drag the cube, full 6-DOF), `FollowTargetSensor` (the tracker's person: reach
cone, lean, retreat, one sniff per encounter, follow sway), `IDLE` (a slow look-around sweep),
`SLEEP` (home, settle, breathe), `HOME`, `HOLD`. The tracker's hints switch between
`FollowTargetSensor`, `IDLE` and `SLEEP` on their own unless `--no-auto-mode`; the operator's
modes (`HOLD`, `HOME`, `FollowTarget`) are never touched by the tracker.

## Control link (OSC)

Any client can drive the app over UDP (`examples/osc_console.py` is one): commands on port 8100,
status pushed back to the sender on 8101.

| Address | Args | Effect |
|---|---|---|
| `/arm/mode` | `follow_target\|follow_sensor\|idle\|sleep\|home\|hold` | set the mode |
| `/arm/agent` | `0\|1` | spring agent on the cube target |
| `/arm/speed` | `0..1` | the runtime speed cap |
| `/arm/stop`, `/arm/estop`, `/arm/recover` | | as the panel buttons |
| `/arm/connect`, `/arm/disconnect`, `/arm/arm`, `/arm/disarm`, `/arm/snap` | | robot management |
| `/arm/cube` | `x y z` | move the target cube (world metres) |
| `/arm/auto` | `0\|1` | let tracker hints switch modes |
| `/arm/ping`, `/arm/quit` | | `/arm/pong` reply; shut the app down |
| pushed: `/arm/status`, `/arm/ee`, `/arm/error`, `/arm/ops` | | state, end-effector, errors, a GREEN/AMBER/RED banner |

The app also sends a `/hb` heartbeat (name + JSON metrics) to port 9000 once a second, for any
supervisor you write, and `/robot/q` (the measured joints) to the tracker on 7011 so it can mask
the arm out of its own clouds.

## Tuning

The motion pipeline is behaviour → RMPflow → PhysX PD drives, each with its own gains.

- `rmpflow/uf850_rmpflow_config.yaml`: `target_rmp.accel_p_gain` (stiffness, 80), `accel_d_gain`
  (damping, 120) and `damping_rmp.inertia` (virtual mass, 100) set the feel. RMPflow runs with
  `ignore_robot_state_updates: true`, so inertia retards its own rollout, not just feedback noise:
  inertia 0 / P 120 / D 30 is snappy and mechanical, the shipped values are organic and ~2x slower
  to retreat. `joint_velocity_cap_rmp.max_velocity` (3.014 rad/s) is the per-joint ceiling.
- PhysX gains in `app.py` (`kps=1e6`, `kds=1e5`) are stiff on purpose so the twin tracks RMPflow;
  `tools/bake_usd_forces.py` raises the drives' `maxForce` so they are never torque-limited.
- Arm won't retract or retreats slowly: lower `inertia`, check no joint is capped below 1 rad/s,
  re-run the bake. Approaches and retreats over and over: `sniff_once` is 0 in your preset.

## Tests

```bash
python tests/run_all.py                  # eight Isaac-free suites (any Python 3.10+ with the deps)
python.bat app.py --test                 # the end-to-end chain, headless, under Isaac Sim
```

## License

Apache-2.0, Copyright 2024-2026 ATONATON, LLC. See LICENSE, and NOTICE for third-party attributions.
The name Mimus is a trademark and is not licensed with the code: TRADEMARKS.md says what you may
and may not call things. The NVIDIA assets the app loads are not part of this work.
