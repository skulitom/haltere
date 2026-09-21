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
