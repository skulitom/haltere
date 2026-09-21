# Human demonstrations train the fly brain

The navigation predictor supplies an auxiliary training target. It is absent
from the exported controller. Images are masked and sampled on a fixed 20 by 12
RGB grid, then passed through a sensory encoder into the existing LPTC population.
The connectome's recurrent circuit produces all four motor outputs. Its wiring
and known transmitter signs remain fixed; recurrent gains, neuron parameters,
sensory encoders and motor readout receive gradient updates.

Training combines observed human controls with a small path loss. A linear
training-only decoder reads the brain's goal-population activity and learns to
match the frozen predictor's paths. This decoder and the predictor are discarded
at export. No predicted path is fed into the student's senses. The external goal
and global compass channels are zero, and no external yaw controller is used.
This is a first, coarse visual sensory encoding, not a biological retinal model.

## Reproduce

Run game tests, capture, controller and the following associated jobs in Anode.
Use fresh output directories; preparation verifies hashes of the source takes.

```powershell
.venv/Scripts/python.exe -m haltere.train.human_brain prepare --dataset data/vision/human_demonstrations_v3 --teacher artifacts/experimental/navigation_human_v2_residual.pt --brain artifacts/ftPath2_best.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --out data/vision/human_brain_v1b
.venv/Scripts/python.exe -m haltere.train.human_brain train --prepared data/vision/human_brain_v1b --initial artifacts/ftPath2_best.pt --config configs/train_human_brain.json --out runs/human-brain-01
```

The mapping is the measured original-drone mapping, not a generic Xbox profile.
Supervision uses processed Liftoff inputs in throttle/roll/pitch/yaw order, after
converting the brain's throttle and axis conventions. It never treats these as
raw gamepad commands. Sources have variable telemetry cadence; training advances
the brain at 100 Hz using causal sample-and-hold, with a maximum 120 ms data gap.
Images become available only after their recorded capture end. Physical display
delay remains uncalibrated. Neuron normalization statistics are kept fixed.

Train: Straw Bale race, Pine Valley take 2, fence take 1. Validation: Minus Two
take 2 and fence take 2. Whole-take membership is preserved. Neither validation
take contributes optimizer updates. Fence take 3 and the mistake recordings are
not read by this workflow. Fixed validation windows are chosen before training;
initial, constant-training-mean and blank-image comparisons are retained.

`parameter_changes.json` records actual brain changes, `first_gradients.json`
checks that learning reaches recurrent parameters, and `result.json` evaluates
the exported brain without loading the predictor. Checkpoint selection uses
validation action error. These short, teacher-forced replay windows do not
establish closed-loop stability, gate completion, or generalization.

## Teacher-free live runner

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/human-brain-01/best.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --seconds 15 --log runs/human-brain-01/shadow-01.csv
```

The default is **shadow mode**, with no controller opened. To fly a reviewed
candidate, explicitly add `--udp-out 127.0.0.1:9003` for an existing throttle-low
bridge. The runner requires fresh game images and telemetry, bounds each attempt,
and stops on a reset or stale input. It does not load GateNet, the navigation
predictor, a course route or a yaw assistant. Logs identify the exact checkpoint
and whether any controls were sent. Videos of shadow runs must be labelled as
shadow runs. Live qualification remains separate from offline imitation.

## First experiment, 2026-09-21

The 120-update run changed recurrent gains, neuron gains/biases, sensory encoders
and motor readout. The exported candidate has SHA256
`98ea6a1fba89b5c3045c50a6efac42c58ae25e13b84a97a76167f93ebe91e700`.
The learned time constants stayed unchanged (their minimum clamp is active).

On continuous validation replay, normalized control MSE was 0.499 for Minus Two
and 0.613 for fence take 2, versus 0.810 and 1.575 for constant training-mean
controls. Blanking images changed these to 0.517 and 0.619: visual dependence is
still weak. A matched 120-update run with path-loss weight zero produced almost
identical short-window errors (differences below 0.0001). The connection carries
gradients, but this experiment shows **no meaningful benefit from the predictor**.

All execution tests ran in Anode session 2, on `[Copy] New Drone`. A 15-second
shadow run completed 1,491 updates. Ground axis verification confirmed actual
processed inputs within 0.00005 of the commands and speed below 0.005 m/s.
A 12-second live control test completed 1,194 updates with the teacher absent.
The drone took off and climbed uncontrollably to 67.4 m: **flight qualification
failed**. Requested and observed mean processed throttle after arming were 0.415
and 0.414, so this was a controller failure, not a disconnected gamepad.

The candidate is not promoted or published as an improved pilot. The next
training work must preserve stabilization and teach recovery on states caused
by the policy itself; short imitation errors alone cannot select a flight model.
Experimental live attempts now also stop at configurable height, speed and
distance limits (defaults 8 m, 10 m/s and 20 m). These abort limits do not steer
the drone. The recorded first attempt predates those limits. Its first-second
raw-stick CSV columns describe the requested output before the arming hold;
later runs log the actual held-low command. The video is only five seconds long
and is incomplete; telemetry covers the full attempt.

Local outputs: `runs/human-brain-01/` and `runs/human-brain-no-teacher-01/`.
Weights remain experimental local outputs; the existing deployed checkpoint
has not been replaced.

## Recovery training

The first visual candidate reacts to independently increased motor RPM by
raising throttle, and its climb-braking response has degraded. The exported
brain can now mask the RPM component of `wing_cs` with
`brain.mask_motor_feedback`; the mask is applied inside the model, identically
in replay, simulation and live flight. It is off for existing checkpoints.

`train_human_brain_recovery.json` adds fixed, counterfactual recovery examples
from the original motor brain. `train_human_brain_onpolicy.json` continues that
run with student-driven simulator rollouts. The frozen motor teacher labels
the student's resulting states; gradients update the student, and both brain
states persist between training windows. Physics and teacher outputs are
detached. Navigation path supervision still comes exclusively from the cached,
training-only predictor. Neither teacher is exported or loaded for live flight.

```powershell
.venv/Scripts/python.exe -m haltere.train.human_brain train --prepared data/vision/human_brain_v1b --initial artifacts/ftPath2_best.pt --config configs/train_human_brain_recovery.json --out runs/human-brain-recovery-01
.venv/Scripts/python.exe -m haltere.train.human_brain train --prepared data/vision/human_brain_v1b --initial artifacts/ftPath2_best.pt --config configs/train_human_brain_onpolicy.json --out runs/human-brain-onpolicy-01
.venv/Scripts/python.exe -m haltere.train.recovery CHECKPOINT --out NEW_RESULT.json
```

With `qualify_motor` enabled, each validation checkpoint also runs 24
closed-loop simulator recovery episodes with blank images, control delay and
physical variation. `best.pt` still denotes lowest replay error;
`motor-qualified.pt` is saved only after the reflex check passes. This is a
simulator check, **not live flight or navigation qualification**. Failure leaves
no qualified checkpoint; there is no automatic switch to an older flight model.

This check rejects the first visual candidate (maximum height 32.9 m in six
seconds) and accepts the original motor reference (maximum 5.3 m). Fixed
recovery examples plus RPM masking alone did not suffice: the 240-update run
still failed the check, including a large tilt. Student-driven recovery is the
next training stage. Live recordings now wait for their first encoded frame
before the flight timer starts, so startup does not omit most of a short test.

The student-driven 240-update run passed the airborne simulator check. Its
actual Anode flights, with full images and with images blanked, both moved into
the nearby barrier below 0.8 m and reset. The runaway climb was absent, but
neither run qualified as sustained flight. Blanking images did not remove the
launch bias. Both used the same new checkpoint, SHA256
`dd019c9fbe3e72a673e12e54f7590f5c3d1def382619fcedabf0858e345c9c70`.

## Takeoff curriculum

An optional `altitude` sensory channel carries `tanh(height / 3 m)` into the
wing sensory population. Height is relative to the recording/launch origin,
**not terrain clearance**. It remains observable at rest, when the existing
velocity/height optic-flow channel carries no height information. Older brains
ignore this new observation. The takeoff curriculum starts half its simulated
episodes at rest and gives the training-only motor teacher a two-metre vertical
goal. The student still receives zero external goal and compass inputs. Ground
contact before first lift is treated as resting contact; later impacts are
crashes. This simplified contact model requires independent Liftoff testing.

```powershell
.venv/Scripts/python.exe -m haltere.train.human_brain add-altitude --prepared data/vision/human_brain_v1b --dataset data/vision/human_demonstrations_v3 --out data/vision/human_brain_v2_altitude
.venv/Scripts/python.exe -m haltere.train.human_brain train --prepared data/vision/human_brain_v2_altitude --initial artifacts/ftPath2_best.pt --config configs/train_human_brain_takeoff.json --out runs/human-brain-takeoff-01
.venv/Scripts/python.exe -m haltere.train.recovery runs/human-brain-takeoff-01/last.pt --takeoff --seconds 30 --out NEW_RESULT.json
```

The cache extension verifies source hashes and uses each existing causal raw
row; it neither changes raw recordings nor regenerates the navigation targets.
Takeoff qualification additionally requires settling near two metres, including
the worst remaining vertical speed. `--blank-retina` is an explicit diagnostic
ablation in the live runner and is identified in the result metadata.

Final training selection prefers a simulator-qualified trained checkpoint when
requested. If none qualifies, it reports the last trained candidate as
unqualified. It never silently selects `initial.pt`; that file is a baseline.
Qualification still does not establish navigation, gate completion or readiness
to publish an improved pilot.

`train_human_brain_damping.json` continues the takeoff run with a differentiable
height, velocity, attitude and action-change cost through the simulated vehicle
and the student's senses. Teacher outputs remain detached. This differs from
the earlier recovery stage, which learned control labels only. The new altitude
encoder has a separate learning rate; existing recurrent weights still update.
The takeoff examples include a noiseless gyro to match the quiet launch state.
The first takeoff model failed a 30-second check because one case kept
oscillating vertically, despite its good median height; it was not flight-tested
or promoted. Qualification checks the worst remaining vertical speed as well
as the median. A passing short test must be followed by longer simulator and
independent live evaluation.

The damping candidate at update 40 (SHA256
`750fec168032ba9789341e5366a01e6891cb10fb0b4c4f17597f43090af74f3c`)
passed 24 simulated takeoff/recovery cases for 30 seconds on seed 882. In Anode,
it then completed a 30-second camera-enabled flight on the original drone:
maximum height 2.066 m, maximum speed 1.462 m/s, minimum upright cosine 0.997.
It drifted 12.54 m, so this demonstrates takeoff and vertical stabilization,
not position holding or navigation. A separate blank-image run also completed
30 seconds and drifted 8.26 m with a similar slow yaw. Both tests had no teacher,
external goal or yaw assistant. Local evidence is under
`runs/human-brain-damping-01/`; no gate completion is claimed.

The stationary recovery curriculum increases horizontal-velocity and angular-rate
costs and supervises zero yaw only in recovery training. The deployed brain still
produces every motor axis. `--stationary` additionally requires worst-case final
horizontal speed below 0.25 m/s and angular speed below 0.1 rad/s. The 80-update
stationary candidate held height in 24 further 30-second simulations but retained
0.316 m/s drift in the worst case, so it failed that stricter check.

## Slow navigation curriculum

The older `pine_lap_02` capture has roughly 9 fps imagery and 2 m/s flight speed.
It can supply control labels closer to this brain's motor training range. The
3 fps `pine_three_laps_01` capture is excluded by the existing 120 ms image-age
contract. `slow_navigation_demonstrations.json` explicitly labels the Pine source
as `route_teacher_live`; raw metadata stays unchanged. A route teacher supplies
training examples, never a deployed route. Human fence take 1 remains training;
Minus Two take 2 and fence take 2 remain validation. The fast Minus Two flight is
a demanding transfer check, not a matched-speed navigation benchmark.

```powershell
.venv/Scripts/python.exe -m haltere.vision.demonstrations configs/slow_navigation_demonstrations.json --out data/vision/slow_navigation_v1
.venv/Scripts/python.exe -m haltere.train.human_brain prepare --dataset data/vision/slow_navigation_v1 --teacher artifacts/experimental/navigation_human_v2_residual.pt --brain artifacts/ftPath2_best.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --out data/vision/slow_brain_v1
.venv/Scripts/python.exe -m haltere.train.human_brain train --prepared data/vision/slow_brain_v1 --initial artifacts/ftPath2_best.pt --config configs/train_slow_visual_brain.json --out runs/slow-visual-brain-01
```

Changing curricula requires the prior replay and source manifests. Continuation
verifies hashes, sensory/calibration contracts, and absence of prior training
telemetry or images in the new holdout. Earlier training fingerprints persist in
checkpoint provenance across further curricula. Normalized replay errors use
each curriculum's training-control standard deviation; compare processed RMSE,
not normalized scores, between curricula. Gate completion still needs live tests.

`train_navigation_motor_brain.json` adds direct path-to-control distillation.
Only during training, the frozen motor teacher receives the navigation
predictor's one-second path as its goal (limited to 3 m). Training yaw labels
face along that path, with a bounded rate. The visual student receives neither
the path nor this goal: its own motor output learns the resulting control
targets. The original auxiliary path decoder remains a small extra loss.
Both teachers and the decoder are absent from the exported pilot.

This run fixes processed-control normalization to the earlier human training
standard deviations. Otherwise, the much smaller control variation in the slow
route dataset greatly increases imitation losses relative to physical recovery.
The exact scale is saved with the checkpoint and reused for continuous replay
evaluation. This is a new training objective; it needs its own live evaluation
and does not establish that the navigation predictor improves flight.

The 120-update direct-distillation candidate has SHA256
`33ec2e20534c13abec3074f9e0a63142fd4b9b099f84ae97883e1cfa955065d0`.
Recurrent edge gains, neuron parameters and motor weights changed; fixed wiring
and transmitter constraints remain. Replay error worsened and the strict
simulator stationary check failed, so this candidate is not release-qualified.
Bounded Anode tests on the original drone nevertheless demonstrate improved
takeoff and vertical stability with both teachers absent:

| Live attempt | Result |
| --- | --- |
| 15 s, camera enabled | Completed; maximum speed 0.622 m/s; drift 2.448 m |
| First 60 s attempt | Stopped at 18.6 s for stale imagery; height 2.450 m before stop |
| Repeat 60 s attempt | Completed; maximum speed 0.627 m/s; drift 13.210 m |

In the completed minute, height during the last 30 seconds stayed between
2.4718 and 2.4809 m; maximum absolute vertical speed in that interval was
0.00237 m/s. Slow yaw and horizontal drift remained. All four motor axes came
from the exported brain with camera input, no route, no external goal and no
yaw assistant. Evidence and synchronized brain/flight videos are under
`runs/navigation-motor-brain-01/`. This is a stability milestone, not evidence
of gate navigation, generalization or a benefit attributable to the predictor.
The next navigation experiment should train corrections on states visited by
the visual student and qualify one gate before attempting laps. These weights
have not been promoted to GitHub/Hugging Face model releases.

## Camera gate measurements and learned approach

`haltere.train.gate_brain` continues the newest human-trained brain with
differentiable gate approaches. GateNet supplies a camera-derived relative point
to the brain's existing goal population during flight. This is a change in the
sensory interface: the earlier raw-pixel channel is inactive in this stage,
and the detector is a required visual frontend. It is distinct from the
training-only navigation predictor, which is still absent at runtime.

All four control axes come from the recurrent brain. The optional frozen motor
teacher and camera-facing yaw labels are used only during training; there is
no deployed yaw helper or route. Checkpoints record the parent and detector
hashes and changes to recurrent, sensory and motor weights. Aperture crossings
and camera visibility are evaluated separately: flying sideways to a synthetic
point is insufficient when the real camera loses the gate. Live testing must
use a distance boundary beyond the intended gate (the two Field Day launch
arches are approximately 23 and 27 m from spawn).

The second gate candidate (SHA256
`648127de98937cfc7a7ed315c4584856c9681629d389dc1bc4dac97df3d9483a`)
cleared both launch arches in three recorded Anode attempts on the original
drone. Start-gate lateral errors were -4.2, -3.0 and +1.7 cm, at heights of
0.82, 0.67 and 0.90 m above the gate base. These are physical crossings in free
flight, checked against video and track planes, not a completed timed race.
A fourth attempt stopped before reaching the arches. All four were eventually
stopped by the 120 ms camera freshness check; faster sampling and a GPU detector
trial did not eliminate that limitation. The longest reached 49.8 m.

The final simulator check recorded 8/12 aperture crossings, no crashes and no
prolonged loss of gate visibility. This is an initial approach milestone, not
lap qualification or generalization. The navigation predictor and motor teacher
were absent in every live attempt, and no yaw override was applied. Raw evidence
and synchronized brain/flight videos are in `runs/gate-brain-live-01/`; these
weights remain local and unqualified for a model release.

The next live test exposed a different stop: a close arch filled the camera and
its detector confidence fell. The pilot now retains a measured gate for at most
2 seconds while it remains within 8 m and no more than 1 m behind the drone.
Other missing targets still expire after 0.5 seconds; image freshness remains
120 ms. With this change, the same brain cleared gate 2 at 67.55 seconds, with
2.9 cm lateral error and 1.11 m height. It then missed gate 3 by 2.86 m on the
first substantial turn. Evidence is in `runs/camera-profile-01/occlusion.*`.
Neither of the two diagnostic flights stopped for stale imagery; camera timing
is now recorded so an intermittent recurrence can be diagnosed rather than
masked by a looser deadline.

Camera measurements now use interpolated poses from telemetry receipt times at
capture, avoiding use of the later processing-time orientation. This does not
calibrate the game's display latency. `haltere.liftoff.gate_evaluation` checks
rotated/tilted track planes offline and reports offsets; it deliberately does
not infer opening dimensions or certify a lap from plane intersections alone.

`--turns` adds moving approaches, randomized world headings, camera-facing yaw
supervision and a differentiable penalty that slows approach while misaligned.
These velocity/heading targets remain training-only. The exported controller
still supplies every motor axis from the recurrent brain. Each training run
saves an identically seeded parent evaluation before updating weights:

```powershell
.venv/Scripts/python.exe -m haltere.train.gate_brain runs/gate-brain-02/last.pt --out runs/gate-brain-03a --iters 200 --lr 0.00005 --motor-teacher runs/gate-brain-02/last.pt --turns
```

The turn candidate remains experimental until matched simulator checks and
recorded live flights establish an improvement. Release qualification still
requires repeatable full courses and transfer testing with the navigation
predictor absent.

The 200-update turn candidate (`gate-brain-03a`, SHA256
`eca5257ab7e5b6fefc7dcca9453bf68523c7bcabe154b9aa2926bbd3a77b1ccf`)
improved the synthetic turn check from 11/12 to 12/12, with no simulated crashes.
Its live test regressed: it cleared the launch arches faster, then hit gate 2's
post. It is rejected for release. `runs/gate-brain-live-02/turns.*` retains the
failed attempt. Do not use the perfect synthetic score to claim flight readiness.

The simulator still had the earlier drone's rates and much higher horizontal
drag. `original_drone_gate_dynamics.json` records the original drone's configured
rates/PID gains and a partial horizontal drag correction from two recorded
flights, with source hashes. Gravity-corrected body acceleration perpendicular
to the rotor thrust identifies this drag without assuming a throttle-to-thrust
curve. Fixed corrected coefficients gave 0.0152 m/s² forward residual RMS on a
third recording that was not used to choose them. This is a dynamics check,
not a held-out navigation result. Mass, inertia, vertical drag and the existing
hover adapter remain uncalibrated by this procedure.

The corrected simulator predicts 2.59 m/s for the rejected turn brain, close to
its 2.60 m/s live peak, versus 1.81 m/s in the old simulator. The next experiment
continues that newest brain with corrected dynamics and stronger roll/pitch
stabilization supervision from the earlier motor teacher:

```powershell
.venv/Scripts/python.exe -m haltere.train.gate_brain runs/gate-brain-03a/last.pt --out runs/gate-brain-04 --iters 200 --lr 0.00005 --motor-teacher runs/gate-brain-02/last.pt --turns --motor-anchor 1 --dynamics configs/original_drone_gate_dynamics.json
.venv/Scripts/python.exe -m haltere.liftoff.fit_drag runs/gate-brain-live-01/repeat.csv --end 51 --reference .01 .004 --out drag-validation.json
```

Live attempts can now use `--pause-on-stop` so a terminal stop pauses the still
active game before the recorder closes. The flag sends no key in shadow mode
or when the game is hidden; verify the pause before disconnecting the bridge.
The bounded duration limit is 1800 seconds to permit eventual full-lap checks.
