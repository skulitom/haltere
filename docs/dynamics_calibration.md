# Original-drone dynamics measurements

The deployed throttle curve was fitted near hover. Full-throttle and high-rate
pulses have now been measured on the original drone; the broader response fits
remain experimental. Independent validation precedes changes to training or
faster live operation.

`haltere.liftoff.visual_brain --dynamics-calibration` reuses the visual runner's
workload check, telemetry/camera freshness checks, impact detection, recording,
deadline checks and pause-on-stop. It requires explicit PD motors, UDP control
and an operator-verified empty arena. Visual race guidance, oracle routes and
geometry control are incompatible with this mode. Keep the original calibrated
`[Copy] New Drone`; select The Drawing Board, free flight, no course, in Anode.

Start with `--dynamics-calibration hover`. The PD baseline climbs to 15 m above
launch and must remain within 0.6 m, below 0.35 m/s and below 0.25 rad/s for two
continuous seconds. Only then does a pulse sequence begin. `throttle` requests
processed inputs 0.25, 0.5, 0.75 and 1.0, three times each, for 0.25/0.15 seconds.
`roll`, `pitch` and `yaw` request both signs at those amplitudes, three times each.
Their durations are limited to 20 degrees of requested rotation under the
checkpoint's calibrated rate curve, or 0.25 seconds, whichever is shorter. Each
angular pulse also ends if measured attitude change plus 40 ms of continued
measured rotation reaches 35 degrees. Every pulse requires a new
stable recovery before the next; missing recovery stops the experiment.

The calibration-specific limits are 35 m above launch, 15 m horizontal radius,
15 m/s, 55 degrees of tilt and 1,800 degrees/s. Falling below 3 m after departure
also stops it. Existing stricter runner limits remain effective. Use an adequate
declared duration and explicit height/speed limits when preparing a flight card.
Pause before releasing the independently held pad bridge.

The standard video labels PD plus pulses and the shadow brain. CSV records
requested/processed/raw commands, the game's actual processed inputs, pose,
velocity, derived body rates and all four measured rotor RPMs. Sidecar pulse
events identify timing and termination. Radial stick coupling can prevent a
requested angular endpoint from reaching exactly 1.0; measure actual inputs.
Short-pulse peak rates are not steady-state rate measurements.

Calibration is excluded from autonomous race/freestyle acceptance. This mode
does not change brain weights or drone settings. Preserve raw attempts, fit on
declared windows and validate on separate pulses before changing training
dynamics. Hover qualification and wider measurements remain live checks, not
capabilities implied by the automated tests.

The first measured batch at `9220911` qualified hover and completed all 12
throttle pulses, including three at actual processed input 0.99996. No impact,
camera failure or controller deadline failure occurred. The first roll sequence
completed the 18 lower-amplitude pulses, then exceeded the tilt limit during the
first full-input pulse. It paused without a detected impact; pitch and yaw were
not attempted on that revision. Command-to-game input lag was about 30 ms. This
failure motivated the advance rotation/time bounds above. Preserve it alongside
subsequent attempts.

The revised batch at `22c3014` completed **24/24 roll, 24/24 pitch and 24/24 yaw
pulses**, with stable recovery after every pulse and no detected impact, camera
failure or controller deadline failure. All three complete standard videos
decoded successfully. Roll/pitch reached actual processed input 0.99996; yaw
reached 0.99291 because it shares radial travel with throttle. The measured peak
tilts were 20.1 degrees in roll and 29.4 degrees in pitch. These are bounded
identification pulses, not completed freestyle tasks or race results.

The initial response analysis underestimated short full-input pulses. Native
telemetry intervals must be retained: hold the actual game input forward and
compare predicted **interval-average** rate to quaternion differences. Linear
interpolation before a nonlinear stick curve attenuates narrow pulses. The
new `haltere.liftoff.fit_rates` module implements this observation contract.
An empirical curve with saturation applied after expo fits this drone better
than the simulator's existing rate curve. Its model form was selected after
examining these recordings; independent amplitudes are still needed. Response
times below the roughly 10 ms telemetry interval are unresolved, not precise
measurements of the physical motor/controller lag.

Use `--calibration-amplitudes 0.15 0.35 0.65 0.85` to declare that independent
batch. The same repetition, recovery and motion limits remain in force. Freeze
the fitted coefficients and source hashes before flying; score new amplitudes
without refitting. Existing checkpoints and deployed controls remain unchanged.

That independent batch at `4fb5fda` completed **all 72 pulses** without detected
impact, camera/deadline failure or flight intervention. It passed the frozen
prediction criteria (overall RMSE at most 10 degrees/s and each amplitude at
most 20 degrees/s), without refitting:

| Axis | All new amplitudes, RMSE | Worst amplitude, RMSE |
| --- | ---: | ---: |
| Roll | 6.20 degrees/s | 11.34 degrees/s |
| Pitch | 7.68 degrees/s | 15.81 degrees/s |
| Yaw | 5.69 degrees/s | 8.55 degrees/s |

Evidence is in `runs/dynamics-independent-20260923`; all three videos decoded.
The [original-drone response release](https://github.com/skulitom/haltere/releases/tag/original-drone-response-20260923)
contains all nine calibration videos, including the initial failed roll pulse,
raw telemetry, flight cards, fitted profiles and independent predictions.
All 12 uploaded assets matched their local hashes and sizes.
This validates short angular-pulse predictions on this drone. It does not
validate high-speed translation, obstacle clearance or acrobatics.

`haltere.sim.identified` is an explicit experimental training surrogate using
these angular fits and the broader throttle measurements. It reproduces the
deployed brain-to-game mapping, radial stick saturation and filtered gyro input.
Horizontal drag remains an uncertain assumption, and approximate rotor RPMs
cannot be used as brain input. `haltere.train.flight_cost` therefore requires
masked motor feedback and trains with varied dynamics, command delay, initial
motion and separate recorded retinal streams. Its synthetic local targets are
a motor-training curriculum, not an autonomous navigation evaluation. Flight
cost gradients may change recurrent edge magnitudes, neuron parameters and the
motor readout; graph wiring, signs, encoders and normalization remain frozen.
The export audits every changed tensor and remains unqualified until live tests.

The first 400-update flight-cost attempt (`motor-brain-11-flight-cost-01`,
revision `4e89e21`) changed 2,767,316 recurrent edge magnitudes, 29,869 neuron
gains, 29,912 neuron biases and the first three motor readout rows. Wiring,
transmitter signs, encoders and normalization stayed unchanged. It was rejected:
development flight cost worsened from 3.016 to 7.100 and velocity RMSE from
1.094 to 1.629 m/s. Neither model crashed in that short evaluation, but training
was unstable. This is retained failed brain training, not a promoted checkpoint.

Protocol 2 uses smaller, separate edge/neuron/readout learning rates and saves
intermediate snapshots. Evaluation runs every declared maneuver to its fixed
duration: a crash cannot skip later phases or change the following task's random
start. Snapshot selection prioritizes fewer crashed tasks, then lower flight
cost, on a development seed. A separate seed compares the selected snapshot
with the unchanged parent after selection. These are still motor simulations;
successful live flight and full-race comparisons are required before promotion.

For matched motor checks in an empty arena, the visual runner accepts
`--collection-route PATH --motor-controller brain --oracle-motor-diagnostic`.
This explicit diagnostic runs the same brain, camera, mapping, recorder and
assisted yaw as other flights, but supplies a declared stored trajectory. Its
video and metadata identify privileged guidance; it cannot count as autonomous
navigation or a race result. PD remains the default permitted route-collection
motor. Optional route fields `finish_speed_mps` and `finish_hold_s` require a
settled endpoint before completion. Routes for these checks are generated in
code, not manually designed maps.

The first generated S-bend motor check (`552fe9d`) finished with PD in 29.515 s
(path RMS 0.173 m). The parent brain followed the path (RMS 0.655 m) but did not
settle by 120 s. Flight-cost candidate02, selected at update 150, never left the
launch phase: its maximum height was 5.846 m against a 5.85 m launch threshold.
All three videos decoded, with no detected impacts or runtime failures. The
candidate had reduced separate-seed simulated cost from 4.954 to 2.366, with
0/16 crashed tasks for either brain; this did not establish a live improvement.
The recordings remain in `runs/motor-tracking-live-20260923`.

The revised route diagnostic holds launch heading, applies the same horizontal
velocity damping as the training/race pilot, and requires a measured launch
hover within 0.6 m at less than 0.6 m/s for one second. Endpoint criteria and
weights remain unchanged. This is a generic guidance revision for another
declared comparison, not a reinterpretation of the failed first batch.

That revised comparison (`d6cc806`) completed the generated trajectory with all
three controllers and no detected impacts or runtime failures:

| Motor | Complete task time | Moving path RMS |
| --- | ---: | ---: |
| PD | 28.505 s | 0.165 m |
| motor10 candidate05 | 29.315 s | 0.299 m |
| motor11 flight-cost candidate02 | 39.557 s | 0.396 m |

All three recordings decoded. These are three different controllers completing
one stored motor task, **not 3/3 autonomous races**. The new weights were slower
and less precise than their parent and remain unpromoted. Evidence is in
`runs/motor-tracking-stable-guidance-20260923`.
The [motor comparison release](https://github.com/skulitom/haltere/releases/tag/motor-tracking-comparison-20260923)
contains all six untrimmed standard brain/gameplay videos from both batches,
their telemetry, cards, frozen manifests and the training audits. All eight
uploaded assets matched their local hashes and sizes. Rejected weights were
retained locally rather than published as a new model download.

Native-interval analysis of the PD recording estimated forward drag at
0.0275/s, versus the surrogate's assumed 0.1/s. A small thrust-scale correction
(3.1561 to 3.1378) reduced its vertical acceleration fit RMSE from 0.0592 to
0.0125 m/s². These are development fits at roughly 3 m/s; lateral drag is weakly
identified and high-speed transfer remains unvalidated. The next profile keeps
explicit drag uncertainty, uses 5% variation of the measured parameters, and
retains 20--60 ms command delay uncertainty. It is not installed in the live PD.

Training protocol 3 adds sustained airborne climbs, descents and stops, longer
episodes and separate height/velocity metrics. Selection rejects a regression
greater than 5% in either metric even if aggregate cost decreases. Distinct
recorded takes supply additional empty-arena retinal currents; they are sensory
perturbations independent of the simulated pose. New development and final-test
seeds are declared for each subsequent run. Ground contact and autonomous visual
navigation still require live evidence.

The third run (`motor-brain-11-flight-cost-03-20260923`, revision `432b390`)
selected update 50 from 200 updates. On the separate seed 17493, its 16 tasks
reduced cost from 4.461 to 3.109, velocity RMSE from 0.951 to 0.905 m/s and
height RMSE from 1.245 to 0.989 m; neither brain crashed. The audit found changes
to 2,767,278 edge magnitudes, 29,825 neuron gains, 29,864 neuron biases and the
first three readout rows/biases. Wiring, transmitter signs, encoders and sensory
normalization stayed unchanged. Its checkpoint SHA256 is
`2729587ec0e3474280307e0f7ba16fb01dac67c324e7ec2f5c91ec1dd837071b`.

The unchanged live S-bend comparison then produced:

| Motor | Complete task time | Moving path RMS |
| --- | ---: | ---: |
| PD | 28.507 s | 0.164 m |
| motor10 candidate05 | 29.319 s | 0.300 m |
| motor11 flight-cost candidate03 | 29.929 s | 0.361 m |

All three tasks completed with settled endpoints, no detected impacts or
runtime failures, and fully decoded standard videos. The new weights improved
substantially over candidate02, but were still slower and less precise than
their unchanged parent. They remain unpromoted. This is one stored-route motor
task per controller, not three autonomous races. Raw evidence and frozen cards
are in `runs/motor-tracking-flightcost03-20260923`.
All three [candidate03 comparison videos and their evidence](https://github.com/skulitom/haltere/releases/tag/motor-flightcost03-comparison-20260923)
are published; all five remote asset hashes and sizes were verified.

Local raw evidence: `runs/dynamics-calibration-20260923` and
`runs/dynamics-rate-calibration-20260923`. Ground-check binding failures and
workload refusals are retained separately from flown attempts. Fresh project
family resolution now handles the user-authorized rotating LitHarness jobs;
busy or unknown-load jobs still fail the preflight. The calibration flights
themselves did not train brain weights.
