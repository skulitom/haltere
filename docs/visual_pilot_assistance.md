# Visual pilot assistance for the newest brain

The visual runner can wrap a trained gate/scene brain in the existing Rabbit
pilot. It handles gate selection, target smoothing, speed scheduling and yaw.
The trained brain still supplies throttle, roll and pitch, with its learned
visual inputs intact. This is an explicit complete-system experiment under the
[generalization-first direction](project_direction.md).

The default `--pilot-assistance none` retains the unassisted comparison.
Assistance does not change checkpoint weights. It currently requires a
checkpoint with a `gate_sensor` contract; the earliest raw-image-only brains
have no calibrated gate detector for this pilot.

## Run

Run Liftoff, capture and controller processes **inside Anode**, keeping its
viewer hidden unless the user asks to watch. Keep the original `[Copy] New Drone`
and its measured mapping. See [game setup](liftoff_setup.md).

For this development workspace, the newest completed scene candidate reviewed
in this update is `runs/scene-brain-09-navigation/last.pt` (SHA256
`3789e33b8bd5bbc0fa504fd9c3365ab5a5e897451d14bc5ce3c042c091543a6e`).
Despite its directory name, this is a connectome with learned visual sensory
currents, not the separate `NavigationNet` path predictor. The candidate and
mapping below are local research outputs and are not included in a fresh clone.

Shadow mode sends no controls. Use a new output path each time:

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/scene-brain-09-navigation/last.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --pilot-assistance rabbit --assist-speed 2 --seconds 15 --log runs/assisted-visual-01/shadow.csv
```

To test live, keep a throttle-low bridge running in a separate Anode process:

```powershell
.venv/Scripts/python.exe -m haltere.cli liftoff pad --udp-in 9003 --seconds 900 --control-file runs/assisted-visual-01/pad-command.txt
```

Verify the game's processed controls on the ground before enabling output.
Then use the shadow command with a fresh log path and add
`--udp-out 127.0.0.1:9003 --pause-on-stop`. Add `--record PATH.mp4` for footage.
The bridge's 900-second lifetime must outlast the attempt: pause before it
expires or before disconnecting it. Keep Liftoff open between attempts.

The runner retains bounded duration, fresh-image/telemetry requirements,
impact detection and default limits of 8 m height, 10 m/s speed and 20 m
horizontal distance. Those defaults are for initial bounded tests, not a full
course. Set explicit test bounds for longer attempts. The assisted mode uses
Rabbit search instead of the neural search timeout; stale-input and flight-limit
stops still apply. `--assist-speed` is a nominal pilot speed, not a hard limit
on actual drone velocity. Choose it per vehicle/controller before held-out tests.

## What is integrated

- The same `SightPilot` used by the older visual stack tracks gates and produces
  a smooth moving target. It receives the new checkpoint's calibrated detector
  points and original capture times, aligned with already observed poses.
- The detector's saved centre offset is respected. Opening-centred detectors
  use zero offset. The older detector's empirical range correction is disabled
  for this adapter, so it does not distort already calibrated range estimates.
- Rabbit schedules horizontal velocity senses and supplies yaw. The current
  brain's altitude convention, retinal features, recurrent circuit and motor
  mapping remain in use. Actual telemetry remains unchanged for limits and logs.
- No course file, route or navigation predictor is loaded. That describes this
  implementation, not a ban on future causal learned planners.

The pilot still carries assumptions about motion, gate appearance/range and
course layout. Reusing it does not establish transfer. In particular, Rabbit
is a race-gate pilot; it does not supply arbitrary freestyle objectives.

## Evidence and attribution

The CSV keeps `thr/roll/pitch/yaw` as the brain's own output.
`command_*` includes the pilot's yaw, `processed_*` describes that command in
game units, and `raw_*` includes the actual arming hold/ramp when output is on.
`in_*` remains the game's observed processed input. Pilot mode, speed-sense
gain and estimated passes are logged separately. Estimated passes require
independent HUD/geometry/video confirmation.

The standard `haltere liftoff score PATH.csv` command accepts these logs and
labels shadow/live control and pilot assistance. It scores observed motion and
processed inputs, not the brain's proposed actions. `phase` records seconds
since the first telemetry frame; older visual logs infer it per reset. The first
three seconds cover the live arming hold/ramp. Motion scores and estimated gate
passes do not prove completion of an ordered race.

The JSON sidecar records `pilot_assistance`, `yaw_assistance`, the complete
pilot parameters, checkpoint hash and whether this was shadow or live control.
An optional neural replay preserves brain outputs before yaw assistance and
the assisted sensory inputs; its sidecar identifies that distinction.

Tests exercise delayed-frame alignment, one-use detection intake, detector
offsets, unmodified real telemetry, motor-axis ownership, retinal preservation,
default unassisted behavior and the existing pilot/guard tests. Live flight is
still unverified: Anode reported a Windows sign-in failure (reason 2055) during
this update. No assisted lap, transfer improvement or new qualified weights
are claimed. Resume with a bounded comparison once the Anode seat is available.
