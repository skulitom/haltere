# Motor-control comparison and faster tracking

The current race-cue pilot supplies a local target and yaw assistance. The fly
brain controls throttle, roll and pitch. Increasing the old `--assist-speed`
above 2 m/s did not train a faster controller: the old pilot's velocity-sensing
gain bottoms out at one, and its parent motor objective targets about 2 m/s.

## Matched motor diagnostic

`--motor-controller pd` substitutes a conventional position/velocity/attitude
controller under the same race-cue guidance. It consumes the same causal local
target and assisted velocity measurements, with no course geometry. It uses
the original drone's rate curves and measured near-hover throttle calibration.
The brain continues in shadow for inspection. CSV, metadata, scorer and video
identify which controller actually commands the motors.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/scene-brain-09-navigation/last.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --motor-controller pd --pilot-assistance race-cue --assist-speed 2 --device cpu --seconds 180 --log runs/comparison-pd.csv --record runs/comparison-pd.mp4 --udp-out 127.0.0.1:9003 --pause-on-stop --max-height 250 --max-speed 12 --max-distance 2000
```

Run flight processes inside Anode with its viewer hidden, an already verified
throttle-low pad bridge and the game ready to fly. Use fresh output paths and
the same parameters/start for the brain comparison. `pd` is a diagnostic mode,
not a fly-brain flight. A bounded segment is not a full-race finish.

The 2026-09-22 comparison batch is frozen in
`runs/motor-comparison-20260922/batch-manifest.json`: generated loop, Straw Bale
and Minus Two, PD then scene09 brain, 180 seconds each, 2 m/s, identical guidance.
Every launch/stop is retained. All three are development courses.

| Course | Brain median speed | PD median speed | Brain / PD horizon shake | Outcome |
|---|---:|---:|---:|---|
| Generated loop | 1.74 m/s | 1.98 m/s | 1.01 / 0.67 degrees | Both reached the time limit |
| Straw Bale | 1.70 m/s | 1.97 m/s | 1.06 / 0.74 degrees | Both reached the time limit |
| Minus Two | 1.68 m/s | 1.95 m/s | 0.93 / 1.39 degrees | Brain reached the limit; PD stopped on a control deadline |

No contact was detected in these logs. No full finish was established. The PD
indoor attempt stopped after 112.17 seconds of telemetry with a 288.5 ms late
control loop; its shorter segment is not directly comparable to a full window.
CPU simulation collection ran alongside parts of this development batch, so
timing differences cannot be attributed solely to controller design. Subsequent
candidate transfer runs must run without simultaneous training/benchmarking.
Source hashes remained frozen across all six launches; their exact source is
preserved in `runs/motor-comparison-20260922/frozen-source`.

`score-timestamp-corrected.json` uses telemetry timestamps for elapsed flight
time. The older sample-count estimate undercounted the slower PD indoor loop.
Horizon/rate shake remains a sampled high-pass diagnostic, not a direct measure
of gate clearance or task completion.

## Calibration correction

The prior equivalent thrust fit omitted the simulator's motor idle mapping.
`equivalent_power_curve(..., idle=cfg.ctl.idle)` now preserves both the measured
hover input and the local acceleration slope after the mixer applies idle.
The old checkpoint and its live input mapping remain unchanged. New motor
experiments use the corrected simulation dynamics; this is a local near-hover
fit, not a validated physical full-throttle model.

## Motor readout training

`haltere.train.motor_tracking` evaluates straight acceleration, opposing turns,
altitude changes and braking, using randomized dynamics and global headings.
It generates ideal local targets in simulation. There are no rendered images,
obstacles or route decisions in this test; it cannot establish camera navigation,
collision avoidance or race completion.

```powershell
.venv/Scripts/python.exe -m haltere.train.motor_tracking runs/scene-brain-09-navigation/last.pt --out runs/motor-tracking-new --speed 3 --device cpu --train --ridge 10
```

The training-only PD teacher supplies desired motor outputs. A regularized fit
updates the throttle/roll/pitch rows of `readout.weight` and `readout.bias` from
the connectome's individual motor-neuron activity. Recurrent wiring, synaptic
weights, transmitter signs, visual encoder, normalization and yaw readout are
unchanged. The exported checkpoint contains no PD teacher. These are brain motor
readout changes, not predictor training or a claim of changed recurrent synapses.

The first candidate is `runs/motor-brain-10-tracking-02/candidate.pt`. Its local
training data, source snapshot, parent hash, changed-parameter audit and simulator
evaluations are preserved alongside it. It improved the 3 m/s simulation cases
but retained failures at the 2 m/s setting. The broader `tracking-03` candidate
also retained low-speed failures. Neither is a qualified replacement. Corrective
collection with `--data-controller brain` samples the student's own visited
states and excludes post-crash samples. `--reuse-data` retains earlier teacher
examples only after checking that their frozen neural representation matches.
The source snapshot avoids confusing
later tooling changes with the code that produced these weights.
The recovery-trained `tracking-04` candidate passed all 72 CPU development
cases (seeds 8292/8293 at 1.5, 2 and 3 m/s), followed by 36 cases with the fresh
seed 9167. At 3 m/s on that fresh seed, median speed increased from scene09's
1.61 to 2.29 m/s, mean turn-phase speed from 1.62 to 2.63 m/s, and simulated
crashes fell from 3/12 to 0/12. These are 16-second ideal-target simulations with
15% dynamics/controller randomization, not complete races. The earlier seeds
were used to diagnose preceding candidates and are development evidence.

Candidate04 retains candidate03's teacher data and adds corrective labels from
its own visited states. Its SHA256 is
`c64dd2a815d3eeab7afd6425aaa9e7f6a342892a2214988a08f4d876daf7ab59`.
An audit against scene09 confirms that every saved model tensor except the first
three rows of `readout.weight` and `readout.bias` is identical. The source,
configuration, data, evaluations and audit are preserved in the local run folder.
The checkpoint declares a 3 m/s motor reference; race-cue assistance reads that
reference when scaling velocity observations for slower requested settings.

The first recorded transfer batch uses the same candidate at a 3 m/s setting
on all three development courses. Straw Bale reached a 2.61 m/s median but hit
a flag after 151.5 seconds of control; Minus Two reached 2.93 m/s but hit a
pillar after 75.9 seconds. Neither had a control-deadline failure. Those failed
flights are retained in `runs/motor10-transfer-20260922`, including their terminal
impact records. The generated loop finished its single lap in **2:20.759**,
confirmed by the game results screen, with a 3.04 m/s median and no detected
impact. This is the first full playability validation of the generated v2 loop.
It has checkpoint volumes and scenery outside the route, without physical gate
frames. It is an open development course, not a held-out obstacle race.
Faster motor tracking has not established safer navigation.
The 3 m/s stack is not qualified for promotion. Scene09 remains the published
reference while further transfer checks are evaluated.
The subsequent common 2.5 m/s full-race batch was withdrawn after its first
Straw Bale attempt developed a large motor excursion and hit the ground at
150.15 seconds. The other two planned launches were not run. Candidate04 was
not published or promoted. This withdrawal is recorded in
`runs/motor10-full-25-20260922/outcomes.json`.

A paired CPU diagnostic then added recorded validation-scene currents to the
same ideal-target simulations. At 2.5 m/s, mean velocity error rose from 0.60
to 1.30 m/s and the 90th-percentile speed from 2.55 to 5.15 m/s; at 3 m/s the
error rose from 0.75 to 1.42 m/s. None of these short simulations crashed, so
the input mismatch is a demonstrated robustness gap, not proof of the complete
live failure mechanism. The blank-retina results alone were too optimistic.

`--retina-data` now injects recorded training-scene currents and configurable
100 ms missing-image intervals during motor training. `--validation-retina-data`
supplies a separate stream used only for evaluation. These images are unrelated
to the simulated pose; this is motor robustness training, not learned visual
navigation. `--data-controller mixed` collects PD trajectories with seed 1921
and corrective labels on brain trajectories with seeds 1922/1923.
`--training-speeds 1.5 2 2.5 3` covers the intermediate speed explicitly.
Recorded image sampling uses a separate random generator so paired runs keep
the same physical randomization. Each new run saves its exact training source.
The loader also checks the recording's detector, scene projection and camera
calibration against the checkpoint before using its retinal features.

Candidate05 (`runs/motor-brain-10-tracking-05/candidate.pt`, SHA256
`64444a628c58b2c1615086bc84f8c5d3c0a0a12667df31d888c4587a802288b2`)
uses these recorded currents and missing-image intervals. It passed all 96 CPU
development cases: 12 randomized worlds at each of 1.5, 2, 2.5 and 3 m/s,
with either continuous images or 25% missing 100 ms image blocks. The physical
seed was 9167 with 15% dynamics/controller randomization; that seed has already
been used for development, so these are not sealed generalization results.
In a paired check at 2.5 m/s with image gaps, mean velocity error fell from
candidate04's 0.82 to 0.78 m/s, 90th-percentile speed from 3.16 to 2.63 m/s,
and speed at the end of braking from 0.32 to 0.20 m/s. Both had zero simulated
crashes. The improvement is principally less overshoot and better stopping.
The preservation audit again finds only the first three motor-readout rows
changed; yaw, all other saved neural tensors and the connectome are identical
to scene09. Full-race transfer is evaluated separately.

### Independent vision device

The runner accepts `--vision-device cuda` while retaining `--device cpu` for
the brain. The detector and scene projection are unchanged; the worker returns
CPU tensors through the existing camera transport. CPU remains the default.
A saved-gameplay-frame check of the actual model found maximum absolute
CPU/CUDA differences of 0.00000382 in detection output and 0.00000036 in
retinal features. An 18-second camera-only Anode check passed with 510 observed
frames and a 30.5 ms median camera cycle. This preflight check includes no
active brain controller or video recorder and is not a matched flight timing
comparison. Full-flight timing is saved in each run's JSON sidecar.

The frozen candidate05 batch at a common 2.5 m/s setting finished **0/3**
development races. Straw Bale hit a flag at 185.03 s of control (median speed
1.92 m/s); Minus Two hit a pillar at about 90.3 s (2.25 m/s). The generated loop
stopped on stale telemetry at 159.87 s (2.21 m/s), before any finish screen.
Its game resumed after the shutdown pause check and the drone dropped after
throttle-low shutdown; it was manually paused during review. This is not a
completed race or a clean landing. All three videos fully decode, and their
exact causal sensory replays and source hashes are retained in
`runs/motor10-visual-full-20260922`.

Straw Bale's full-flight median camera cycle was 59.7 ms; stale-image ticks
were 6.7%, compared with 73.3 ms and 37.9% in candidate04's CPU-camera attempt.
Both the weights and camera device changed, so this is a complete-stack
development comparison, not an isolated estimate of the weights' benefit.
Candidate05's median horizon/rate shake on Straw Bale was 0.81 degrees and
6.98 degrees/s, versus candidate04's 1.30 degrees and 12.48 degrees/s at the
same requested speed. Different flight lengths and visited states limit that
comparison; smoother control did not prevent the flag collision.

The telemetry-stop issue prompted a shutdown-only fix: wait up to 1.4 seconds
for fresh telemetry progress before sending Escape, while keeping throttle low.
Never toggle a paused/results screen based on a cached frame. Stale-input stops
now preserve receipt age, simulation-progress age, pose validity and packet
counts. The live telemetry threshold, weights and steering are unchanged.
The full suite passed 342 checks before this fix; 34 relevant checks passed
afterwards, including delayed telemetry recovery and paused/results cases.

On 2026-09-23, a separate generated-loop repeat with the same candidate05
weights, guidance and 2.5 m/s setting finished in **3:01.576**, confirmed by
the results screen. Median speed was 2.31 m/s, horizon shake 0.77 degrees and
rate shake 4.86 degrees/s, with no detected contact, reset or flight intervention.
The standard brain/gameplay video fully decodes. Logs, exact sensory replay,
finish screenshot and unchanged-source verification are in
`runs/motor10-loop-repeat-20260923`. The original frozen batch remains **0/3**;
across that batch and this repeat, candidate05 has **1/4** full finishes, all
on development courses. The open generated loop does not establish obstacle
avoidance, unseen racing or freestyle. Candidate05 is an experimental download,
not a promoted replacement for scene09. See the [motor10 bundle](motor_brain_10_release.md).

The broader [generalization program](generalization_program.md) still requires
free-space perception, full races on frozen held-out batches and separate
freestyle evaluation.
