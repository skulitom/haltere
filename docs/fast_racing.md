# Fast racing stack

Added on 2026-09-23 to make race flight faster and smoother while staying
course-agnostic. It sits beside the unchanged standard race-cue pilot and the
brain's PD teacher contract; every part is opt-in.

| Part | Flag | What it does |
|---|---|---|
| Fast race-cue pilot | `--pilot-profile fast` | Requests an acceleration-limited world velocity along the filtered bearing of the visible next-checkpoint ring, slowing continuously with the turn still required. Clipped markers set bounded climbs/descents (the descent follows a slope just steeper than the clipped edge ray), corner clamps turn and climb/descend together, a descent the vehicle cannot achieve is treated as support by terrain, a brief cue dropout coasts, and yaw uses the measured rate curve with throttle priority on the shared stick. |
| Fast PD motors | `--pd-profile fast --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json` | Velocity command plus feedforward on the measured full-throttle thrust curve and post-expo rate curve of the original `[Copy] New Drone`; 15 m/s² horizontal, 60° tilt cone, stick low-pass. A matched baseline and the brain's distillation teacher, not a fly brain. |
| Fast brain contract | a checkpoint with `fast_motor_tracking` metadata, `--motor-controller brain` | The brain receives the pilot's velocity request as a body-frame goal and its horizontal velocity senses scaled by a declared factor, so a nominal-speed flight looks like its familiar regime. Speeds above the trained nominal are refused. |
| Looming brake (experimental, off) | `--looming-brake` | Fly-style time-to-contact from image expansion around the focus of expansion, computed in the camera process with de-rotated optical flow, caps speed toward surfaces ahead. |

Nothing in the stack reads course files, routes or per-course parameters. The
checkpoint ring is a disclosed generic Liftoff race cue; it gives a bearing, not
range or free space, and does not exist in freestyle.

## Brain training

```powershell
.venv/Scripts/python.exe -m haltere.train.fast_motor_tracking runs/motor-brain-10-tracking-05/candidate.pt --out runs/fast-brain-NN --speed 6 --scaled-speed 2.4 --steep 0.4 --rounds 5 --courses 10
```

DAgger in the measured-drone surrogate (`IdentifiedSim`) on seeded synthetic
checkpoint courses with the fast pilot and a synthetic HUD marker: one round under
the fast PD, then rounds under the brain with PD labels on the states it visits.
A parent-centred ridge fit changes only the throttle, roll and pitch readout rows
and their biases; the script refuses any other change. `--steep` adds 15-35°
climbing/descending legs (needed for hills); `--scaled-speed` maps the nominal
request to a slower apparent speed so the brain's saturating (tanh) velocity
senses stay informative. CPU only, about 45 minutes for five rounds; it pauses on
GPU heat if run on CUDA. Surrogate results are development checks, not flight
evidence.

## Offline rehearsal

`python -m haltere.liftoff.fast_rehearsal --speeds 6 8 --seeds 0 1 2 --baseline`
flies the pilot and fast PD through synthetic courses in the surrogate with
camera latency, dropout and command delay. It found every pilot bug fixed before
the first live flight, but it has flat ground, no obstacles and an assumed HUD
clamp rule, so it cannot predict terrain or obstacle failures.

## Results so far

See the [flight card](flight_cards/2026-09-23_fast_stack.md). With the fast PD at
6 m/s: Straw Bale full race **5:03.066** (previous best 13:04.047) and the
generated loop in 1:20.925; Minus Two and Pine Valley crashed into obstacles on
the direct line to the ring (a hairpin wall and a mound). Brain-motor attempts
under the same pilot have not yet finished these races. Obstacle perception is
the main open problem; looming is a weak cue on its own.
