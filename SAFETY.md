# Safety

This app moves an industrial robot arm toward people on purpose. It is **not a safety-rated
system** and nothing in it has been certified. A safe cell is your responsibility. Read
`rt_xarm_bridge`'s SAFETY.md too: that is where the motion gates live.

## What you own

- **A risk assessment** before anything moves. ISO 10218-2 and ISO/TS 15066 are the references
  for arms that share space with people; your jurisdiction may add requirements. Decide what the
  arm may touch, at what speed, and what happens when the tracker is wrong.
- **A hardware emergency-stop circuit** wired to the controller, tested before each session,
  within reach of whoever is near the arm. The EMERGENCY STOP button in the panel and `/arm/estop`
  send a stop over the network; they are not a substitute.
- **Guarding and distance.** The reach sphere (`reach_max_m`) bounds where targets can be; it does
  not bound where people stand. Decide what is inside the arm's reach and keep people out of it
  unless your risk assessment accepts contact at the configured collision sensitivity and speed.
- **The controller's own safety configuration** (collision sensitivity, reduced mode, boundary,
  speed limits), set in UFACTORY Studio and declared in `safety.yaml`. The bridge refuses to arm
  until they match. The arm is safer than this software in exactly those settings.
- **Supervision.** Nothing here is designed to run without someone able to stop it.

## What bounds the motion, and what does not

| Layer | Bounds | Does not bound |
|---|---|---|
| Reach sphere about the shoulder | every target the behaviour emits | people inside the sphere |
| Retreat and sniff | the arm backs off when the look-at point comes within `retreat_dist` | a person the tracker cannot see |
| RMPflow velocity cap, joint-limit buffers, self-collision model | joint speed, joint limits, the arm hitting its own base | obstacles in the room (none are modelled by default) |
| Bridge speed cap and ramp | how fast the real arm follows the twin; starts at 0.2 and from zero on every enable | motion once an operator raises the cap |
| Bridge watchdog, joint-limit net, safety check | stale targets, controller limit faults, a misconfigured controller | anything the controller itself does not enforce |
| Tracker staleness | the mode stays where it is when the tracker stops (the app does not guess) | a tracker that reports the wrong person or position |

Starting with `--arm xarm` and no `--allow-motion` reaches `MONITORING` only: joints are read,
nothing is armed, the panel's "Robot Enabled" refuses. That is the recommended first run.

## Operating procedure

1. Before each session: e-stop test, cell clear, `rt-xarm-bridge check` against your `safety.yaml`.
2. Run the app with the mock arm, then monitor-only against the real arm.
3. Start with `--allow-motion --safety safety.yaml`, keep a hand on the e-stop, press Robot
   Enabled, and watch the startup ease to home at the startup rate.
4. Raise the Speed slider in steps while watching. Cuts apply at once; use them.
5. After any halt (the panel shows HALTED and why), read the controller's error code before
   Recover. An E-stop must be released at the hardware first.
6. `--auto-recover` is for supervised runs only. Leave it off unless someone is watching the arm
   and can reach the e-stop; the bridge never clears an E-stop code by itself.
7. Shut down with the panel's Stop or `/arm/quit`, which stops streaming before the link drops.

## Known limitations

- No room obstacles are modelled by default. If the arm shares its reach with furniture or a
  wall, add them to RMPflow (the bridge's joint-limit net does not know about them).
- Heights in the presets are world heights for a mounted base; on a floor-mounted base the idle
  sweep sits at the top of the reach sphere until you lower them.
- The mock arm tracks perfectly; a real arm lags the twin by whatever the speed cap allows, which
  is why the cap ramps and why raising it is an operator action.
