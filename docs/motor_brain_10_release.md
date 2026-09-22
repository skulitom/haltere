# Motor10: experimental motor-readout weights

Reviewed on 2026-09-23. Download from [GitHub Releases](https://github.com/skulitom/haltere/releases/tag/motor-brain-10-experimental) or [Hugging Face](https://huggingface.co/Skulitom/haltere/tree/main/motor10).
The GitHub assets' stored SHA256 digests and a downloaded Hugging Face archive
were verified against the local files after publication.

Candidate05 improves simulated braking and reduces overshoot with visual inputs.
It completed one open development course, but **is not promoted over scene09**:
its first frozen race batch finished 0/3, followed by one successful loop repeat.
Unseen-race completion, general obstacle avoidance and freestyle remain unproven.

## What changed

A PD teacher supplies offline motor labels on randomized straight, turning,
reverse-turn and braking trajectories. Training covers 1.5/2/2.5/3 m/s, retains
previous examples, and adds recorded training-scene currents and missing-image
intervals. The images are unrelated to simulated poses; this is motor robustness
training, not visual navigation or obstacle learning.

Only the throttle/roll/pitch rows of `readout.weight` and `readout.bias` changed.
The audit against scene09 confirms that yaw, all other saved neural tensors,
connectome wiring and known transmitter signs are unchanged. The PD teacher is
absent at inference. These are brain motor-readout updates, not learned recurrent
synapses or navigation-predictor training.

Live race-cue assistance supplies a bounded target, speed-sense scaling and yaw
from current checkpoint rings and visible route arrows. The brain controls
throttle, roll and pitch. No course XML, stored route, teacher or separate learned
navigation predictor is loaded. The standard video shows the actual controlling
brain beside gameplay.

## Evidence

The candidate passed **96 CPU development simulation cases** without a crash:
12 worlds at four speeds, with continuous images or 25% missing 100 ms blocks,
15% dynamics/controller randomization, and physical seed 9167. Each case lasts
16 seconds. That seed was already used for development. At 2.5 m/s with image
gaps, paired velocity error fell from candidate04's 0.82 to 0.78 m/s, 90th-percentile
speed from 3.16 to 2.63 m/s, and final braking speed from 0.32 to 0.20 m/s.

All live attempts used the original `[Copy] New Drone`, CPU brain, CUDA vision,
race-cue guidance and a common 2.5 m/s requested speed. All courses are development
data. Every launch is retained:

| Attempt | Median speed | Outcome |
|---|---:|---|
| Straw Bale / Field Day | 1.92 m/s | Flag collision after 185 s; lap 1/3. |
| Minus Two / Turn Signals | 2.25 m/s | Pillar collision after 90 s; lap 1/3. |
| Generated loop, first attempt | 2.21 m/s | Telemetry stop at 160 s; no finish. Missed shutdown pause caused a subsequent drop. |
| Generated loop, separate repeat | 2.31 m/s | **Full finish, 3:01.576**, no detected contact, reset or flight intervention. |

The original frozen batch is **0/3**; including the repeat gives **1/4** finishes.
The repeat changed shutdown handling and diagnostics only. The generated loop
has open checkpoint volumes and scenery outside the route, without physical
gate frames. It is not an unseen obstacle race. Candidate04's earlier 2:20.759
loop finish used different weights and must not be attributed to candidate05.

On Straw Bale, median camera-cycle time fell from 73.3 to 59.7 ms and stale-image
ticks from 37.9% to 6.7% relative to candidate04's CPU-camera run. Horizon/rate
shake fell from 1.30 degrees / 12.48 degrees/s to 0.81 / 6.98. Weights and vision
device both changed, and the flights cover different durations/states; these
are descriptive complete-system measurements, not isolated causal estimates.
All four brain/gameplay videos fully decode. The full suite passed 342 tests;
34 focused checks passed after the shutdown-only fix.

## Download and use

Extract `motor10-candidate05-inference.zip` into the repository root. Use code
`ca40862` or a later compatible revision. The archive preserves checkpoint,
detector and mapping paths and includes source snapshots, configuration,
individual hashes, audits and evaluation records. The graph is already tracked
in the repository. Raw training examples, original recordings and optimizer
state are not included; the archive alone does not reproduce training.

| Artifact | SHA256 |
|---|---|
| Archive | `40387f90376fbd4dc293a773d3bb8c5417fc0f862e98c1436e1d5e9503d7ee42` |
| `runs/motor-brain-10-tracking-05/candidate.pt` | `64444a628c58b2c1615086bc84f8c5d3c0a0a12667df31d888c4587a802288b2` |
| Original scene09 parent | `3789e33b8bd5bbc0fa504fd9c3365ab5a5e897451d14bc5ce3c042c091543a6e` |

Follow [game setup](https://github.com/skulitom/haltere/blob/main/docs/liftoff_setup.md)
and verify the real processed controls, including throttle-low. This example
runs in shadow mode and sends no controls:

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/motor-brain-10-tracking-05/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --assist-speed 2.5 --seconds 60 --max-height 250 --max-speed 12 --max-distance 2000 --log runs/my-motor10-shadow.csv --record runs/my-motor10-shadow.mp4
```

CUDA vision needs a compatible NVIDIA/PyTorch installation. `--vision-device cpu`
retains the original CPU camera path, with different live timing. For authorized
live simulator tests inside hidden Anode, use the verified pad bridge and add
`--udp-out 127.0.0.1:9003 --pause-on-stop`. Use new output paths, allow enough time
for the race, and pause before disconnecting the pad. A finish can stop live
telemetry: require the game results screen, not a successful runner exit code.

See the [motor training record](https://github.com/skulitom/haltere/blob/main/docs/motor_tracking.md)
and [generalization program](https://github.com/skulitom/haltere/blob/main/docs/generalization_program.md)
for the remaining obstacle-planning and freestyle work.
