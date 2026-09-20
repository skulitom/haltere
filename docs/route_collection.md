# First bot-guided collection experiment

Prepared and first flown 2026-09-20 from the [verified Pine Valley bot recording](bot_routes.md).
The experiment uses the shipped `artifacts/ftPath2_best.pt` as a controller and
the bot's positions as an explicit route teacher. It does not imitate the bot's
stick values and is not a test of flying by sight.

## Offline result and chosen first live test

The frozen brain was rehearsed in the checkpoint's training simulator with its
configured control delay, a stationary start at the source heading, the normal
arming ramp and a 2 m/s target path speed. There was **no Pine terrain, tree or
gate collision geometry**. The waypoint target slows when the drone falls behind,
so nominal moving time is not a prediction of flight duration.

| Prefix | Endpoint reached within 0.8 m | Mean / p95 distance to route | Median drone speed |
|---|---:|---:|---:|
| 80 m | 52.65 s | 0.79 / 1.91 m | 1.67 m/s |
| 30 m | 21.67 s | 0.47 / 0.83 m | 1.49 m/s |

The 80 m rehearsal overshot the recorded altitude by **2.60 m** near target
progress 47.5 m. Reaching its endpoint is insufficient for accepting that route
in a course with arches and trees. The first live attempt is therefore restricted
to **30 m**, checking takeoff, alignment and the start arch before the rising
section. Extending the route requires investigating that vertical error.

Artifacts (ignored local research outputs):

- `runs/pine-route-collection-01/route.json`: 80 m investigation route.
- `runs/pine-route-collection-01/route-30m.json`: selected first live route.
- `runs/pine-route-collection-01/rehearsal-overview.png`: 80 m plots, including the overshoot.
- `runs/pine-route-collection-01/sim-flight.csv` and `sim-report.json`: 80 m results.
- `runs/pine-route-collection-01/short-rehearsal/`: 30 m results.
- `runs/pine-route-collection-01/rehearse.py` and `plot.py`: local reproduction scripts.

The old `pine1`/`pine2` indices contain reset-relative positions without the
absolute reset origin. They cannot by themselves certify world-coordinate
alignment. The live command binds world waypoints to the **actual first reset
telemetry position**, preserving world axes and altitude. It rejects a start more
than 2 m from the source spawn, a moving drone or a tipped-over drone. This guard
does not identify the course; verify the selected map and race in the game too.

## Commands for one live attempt

Use the user's **[Copy] New Drone** (`85e3d358-5b42-4f16-b3d3-398a7d18f650`),
with its existing in-game settings. The user explicitly authorized taking the
Steam session for these experiments. Keep Liftoff open and paused on this course
between experiments; do not repeatedly close and relaunch it.

Keep a persistent virtual-pad bridge alive across consecutive tests. Liftoff can
return to mid-throttle when the virtual pad disconnects. Pause the game before
ending the bridge; do not infer successful control from lift-off alone. Verify
the game's processed inputs against commands on the ground.

```powershell
.venv/Scripts/python.exe -m haltere.cli liftoff pad --udp-in 9003 --seconds 900 --control-file runs/pine-route-collection-01/pad-command.txt
```

For the live original-drone test, the existing mapping was copied into
`runs/pine-route-collection-01/liftoff-original-drone.yaml`; only the measured
hover point and motor-RPM normalization were adapted. That file is provisional,
and does not change the game, the shipped mapping, or the brain weights.

Prepare a new bounded route if needed:

```powershell
.venv/Scripts/python.exe -m haltere.cli liftoff prepare-route runs/pine-bot-route-01/source.recording --out runs/pine-route-collection-02/route.json --length 30 --speed 2
```

For the existing prepared route, select **Pine Valley / 01 - Forest For The Trees**
and reset. Run both processes in the same Windows session as Liftoff. The camera
file must match the game field of view; `camera_seat.yaml` is the existing Anode
calibration, which still needs an image/pose alignment check on this new capture.

Start the passive recorder first, then start the pilot promptly in a separate
process. Use new output paths for every attempt:

```powershell
.venv/Scripts/python.exe -m haltere.cli liftoff capture --out data/vision/pine_teacher_01 --course "Pine Valley / Forest For The Trees / 30 m teacher prefix" --camera configs/camera_seat.yaml --port 9011 --seconds 45 --teacher-route runs/pine-route-collection-01/route-30m.json
```

```powershell
.venv/Scripts/python.exe -m haltere.cli liftoff fly artifacts/ftPath2_best.pt --liftoff-config runs/pine-route-collection-01/liftoff-original-drone.yaml --udp-out 127.0.0.1:9003 --world-route runs/pine-route-collection-01/route-30m.json --seconds 30 --face-travel .8 --face-max .25 --face-ahead 1.5 --reset-key r --telemetry-copy-port 9011 --log runs/pine-route-collection-01/live-01.csv
```

The pilot receives on 9001 and copies valid telemetry packets to 9011 for capture,
so two receivers do not compete for the game's stream. Forwarding starts only
after the initial reset backlog is drained and the route start is validated.
The copy's receive age includes the forwarding step; it is not a measurement of
physical display latency. Capture remains outside the control loop, checks the
foreground game window and refuses stale image/pose pairs.

The route mode requires a bounded duration and a new flight log. It stops at the
endpoint, on a game reset, or when the existing grounded/stuck detector fires;
it does not automatically retry a failed attempt. The persistent bridge returns
to throttle-low within 0.5 s of the pilot stopping and stays connected.
`live-01.csv.route.json` records the route hash and actual Unity reset origin.
`capture.json` marks the images `route_teacher_live` with `oracle_route: true` and
the route hash. Review synchronization, the start-arch crossing and the HUD before
counting these as useful data. A short prefix is not a completed lap.

## Implementation checks

The ordinary non-looping waypoint path previously included a return segment to
its first point. That is fixed: open paths clamp both target and progress at their
last point. Closed paths retain their return segment.

Tests cover preserving source corners and altitude, exact prefix clipping, live
origin changes without rotating world axes, rejecting invalid starts, endpoint
clamping, telemetry copying, teacher provenance, and command cleanup on a reset
or invalid spawn. Existing by-sight and Liftoff tests passed as well. Model and
mapping hashes remain unchanged; no training or weight promotion took place.

## Live result on the original drone

Steam and Liftoff were moved to Anode session 2 with the user's authorization.
The earlier Borrum was not used for the live tests: the user selected their
**[Copy] New Drone** (game UI: 1,541 g, 4.95 kg thrust). Its existing PID, rates,
camera tilt and components were left as configured.

| Check | Result |
|---|---|
| Ground input verification | All four axes received on the expected channels; neutral throttle -1.000, speed below 0.011 m/s |
| Brain hover, 2 m target | Second-half mean 3D position error 0.34 m over a 12 s run |
| Source spawn alignment | 0.026 m from archived source start |
| 30 m live prefix | Endpoint reached in 18.12 s of telemetry; final endpoint distance 0.46 m |
| Nearest 3D route distance | Mean 0.376 m, p95 0.597 m, including arming and takeoff |
| Median speed | 1.86 m/s |
| Start arch | Passage visible in captured frames; HUD race timer starts afterwards |
| Capture | 165 frames, zero age/foreground rejections; p95 receive age 21.0 ms |

Distance to the moving target printed by `fly` is about 1.54 m because the target
leads the drone. The 0.376 m result instead measures nearest distance to the
recorded polyline in aligned world coordinates. The receive-age statistic is
not a measurement of physical image/pose latency.

Artifacts under `runs/pine-route-collection-01/`: `live-report.json`,
`live-01.csv` and its route sidecar, `live-overview.png`, `live-contact.png`,
`live-flight.mp4`, `live-flight.gif`, and `analyze_live.py`. Source frames and
telemetry are in `data/vision/pine_teacher_01/`. These are local research outputs.

The preceding open-loop vehicle identification probe drifted out of the allowed
area and reset before completing all axes. Its altitude-hold segment supplied a
provisional processed hover value 0.13566 and RPM normalization 40,278, subsequently
checked by the successful brain hover and prefix flight. Ground verification
was repeated with a persistent bridge after the user identified the Xbox
midpoint/disconnection concern. `pad-verification.csv/json` record the evidence.

The hover's image recording initially stopped on the pilot's startup reset,
leaving one frame; it is not usable hover imagery. This exposed and fixed a
forwarding bug: ordinary hover now also waits for the first drained post-reset
telemetry before forwarding. The regression test and route/capture/Liftoff suite
passed (33 tests; one existing training warning).

### Gate-clearance fix after the initial prefix

The user correctly pointed out that crossing the start line clears no course
gates. Gate 1 is about 54 m along the source route. The first 65 m attempt
(`live-02.csv`) hit a tree at about 34 m with a steady 0.39 m lateral bias.
Proportional correction alone passed the gate but introduced weaving (live-03).
Damping removed that weaving but left a 0.17 m bias and another tree contact
(live-04). Bounded lateral proportional/velocity/integral correction with gains
1.0 / 1.0 / 0.4 reduced approach bias to 0.022 m (8-18 s interval) and cleared the
first course gate in **two consecutive 65 m runs**, live-05 and live-06. Source
route geometry and brain weights were unchanged. Captured images show the arch
passage and subsequent checkpoint markers. This still does not complete a lap.

The tested route is `runs/pine-route-collection-01/route-65m-centred.json`.
Optional JSON fields `cross_track_gain`, `cross_track_damping`, and
`cross_track_integral` default to zero, preserving previous pilot behaviour.
Correction is capped at 0.8 m, acts horizontally across the route, freezes
integration on the ground/saturation, and clears its accumulator on reset.
All 73 route, Liftoff and by-sight regression tests passed (one existing warning).
The game stays open and paused between tests; keep the persistent pad for reuse.

### What this does and does not establish

The single prefix flight demonstrates that archived bot geometry can guide this
drone and produce new live images. It is neither a completed lap nor flying by
sight. The weights and shipped mapping hashes remain unchanged.

The inherited camera file is **not validated** for this capture. A rotation fit
on the route yielded f=175 px, tilt=30 degrees, 0.56-degree median residual from
only 14 pairs, versus the inherited f=200 px. The earlier probe fit hit the search
boundary and was rejected. Translation, HUD matches, and limited rotation make
these insufficient to promote a calibration. Keep `labels_reviewed=false` and
`course_complete=false`; no frames have been used for training.

Next: repeat this prefix with the same drone/bridge, validate camera and timing,
then extend in short increments toward the next arch. Investigate the known
80 m rehearsal climb overshoot before collecting through that section.
