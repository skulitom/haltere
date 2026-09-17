---
license: mit
tags:
  - connectome
  - drosophila
  - computational-neuroscience
  - drone
  - fpv
  - liftoff
  - pytorch
  - recurrent-neural-network
  - imitation-learning
  - reinforcement-learning
library_name: pytorch
pipeline_tag: reinforcement-learning
---

# Haltere: a fruit-fly connectome brain that flies an FPV drone

A recurrent network whose 30,000 neurons and 2.77 million synaptic connections are copied from the
[Janelia male CNS connectome v1.0](https://www.janelia.org/project-team/flyem/male-cns-connectome)
(FlyEM, Cambridge Connectomics, Google Connectomics), trained to fly a quadcopter in the FPV
simulator [Liftoff](https://store.steampowered.com/app/410340/). The drone's senses are written into
the fly's own sensory neurons (haltere, wing campaniform, optic-flow, ocellar, Johnston's organ,
compass and goal cells) and the four stick commands are read out of the wing motor neurons and their
premotor partners. Synaptic structure and sign are fixed by the connectome; per-synapse gains,
neuron gains, biases, time constants, sensory encoders and the readout are trained.

Code, training pipeline, Liftoff integration and videos: https://github.com/skulitom/haltere

![The fly brain racing the Straw Bale lap in Liftoff](liftoff_fast_lap.gif)

*The lap brain on a taught race line in Liftoff: gates 0 to 6 in 38 s at 4.8 m/s, peaks of 8.3 m/s,
with a steady horizon (roll and pitch rate shake 2 deg/s, against 26 before the pilot inverted Liftoff's
radial stick deadzone exactly).*

![Flying by sight through the two hill gates](liftoff_sight_hill.gif)

*By sight: the gate detector finds the arches in the FPV image and the rabbit pilot turns them into a
smooth line the brain follows; six of the seven gates in one run, both hill gates included.*

![The same stretch before and after the stick-path fix](liftoff_stickfix.gif)

![The first flight: a hover in Liftoff](liftoff_hover.gif)

![Orbit in Liftoff](liftoff_orbit.gif)

![Climb and dive in Liftoff](liftoff_climbdive.gif)

![Following a taught race lap in Liftoff, nose along the path](liftoff_race.gif)

## Files

| file | what | use |
|---|---|---|
| `ftPath2_best.pt` | the smooth brain fine-tuned on moving targets (two stages): races a taught lap at 4.8 m/s and flies by sight | `haltere liftoff fly ftPath2_best.pt --liftoff-config liftoff.yaml --waypoints-file track_strawbale.yaml --path-speed 8 --lookahead 6 --z-lead 1.5 --flow-gain 0.5 --face-travel 0.8 --face-ahead 6 --throttle-scale 0.8 --gyro telemetry` |
| `ftSmooth_best.pt` | fine-tuned with 50 ms extra latency and a smoothness penalty: hover, orbit, climb-and-dive | `haltere liftoff fly ftSmooth_best.pt` |
| `ftRobust_best.pt` | wide domain randomization; first brain that flew in Liftoff | |
| `imJ_best.pt` | imitation of the MLP controller with the premotor readout | |
| `mlp_baseline.pt` | the MLP teacher (no connectome) | control experiment |
| `gatenet_best.pt` | gate detector (5 M parameters) on the FPV image, 42k labelled frames | `--vision gatenet_best.pt --camera camera_seat.yaml --sight rabbit` |
| `gatenet_colourblind.pt` | the same flights, three held out whole, with hue/saturation/gamma/sharpness/scale augmentation | studying generalisation, not flying — see the results table |
| `liftoff.yaml` | the Liftoff mapping: stick and gyro signs, hover point, and the radial stick-deadzone model | `--liftoff-config liftoff.yaml` |
| `track_strawbale.yaml` | the taught Straw Bale Field Day lap (174 waypoints) | `--waypoints-file`, `liftoff score --track` |
| `gates_strawbale.json` | the lap's seven gates (position, heading) | `liftoff score --gates` |
| `camera_seat.yaml` | FPV camera calibration (focal length, tilt) | `--camera` with `--vision` |
| `flight.npz`, `flight.nodes.parquet`, `flight.meta.json` | the built flight graph: 30,000 neurons, signed synapse counts, named populations | required by every brain checkpoint |

The checkpoints are slim (parameters only, about 12 MB); the graph is loaded next to them. Full
checkpoints with optimizer state are on the
[GitHub releases](https://github.com/skulitom/haltere/releases) (v0.4.0 has the videos of the fast lap and of
the flight by sight).

## Results

| controller | simulator, full difficulty | Liftoff |
|---|---|---|
| MLP baseline | 0.05 m mean error, 100% within 0.5 m | not flown |
| `imJ_best` (imitation) | 0.22 m, 95% | drifts 1.4 m on the physics stand-in |
| `ftRobust_best` (+ domain randomization) | 0.20 m, 99.6% | 2 m hover, 0.34 m mean error over 40 s; 3 m square pattern |
| `ftSmooth_best` (+ latency, smoothness) | 0.30 m, 95% (50 ms delay) | 2 m hover, 0.35 m mean error with a quarter of the stick jitter; orbit (0.75 m tracking error at 0.8 m/s) and climb-and-dive (1.1 m at about 1 m/s), no crashes; taught lap at 1.2 m/s with 0.9-1.0 m error |
| `ftPath2_best` (+ moving targets) | 0.37 m, 80% static; 0.76 m following a 1 m/s target, 3.1 m at 2 m/s | taught race lap: the seven gates in 38 s at 4.8 m/s (peaks 8.0 m/s) with `--flow-gain 0.5`, no contact, roll and pitch rate shake 2 deg/s; **by sight with the rabbit pilot, all seven gates in 64 s at 3.19 m/s**, every arch within 0.4 m of centre, no contacts between gates 0 and 6, yaw shake 2.0 deg/s |
| `gatenet_best` (gate detector, 5 M parameters) | on whole flights recorded AFTER it was trained: 85.6% recall, centre error 2.9 px at 320 wide, 11.6% false positives on gate-less frames (9.6% pooled over all the gate-less frames of those flights). (An earlier card said 98% on "a held-out tenth" — that split took single frames from the same flights, and frames 130 ms apart are the same picture, so it could not fall) | flies by sight: five clean 7/7 laps of Straw Bale Field Day (clean between gates 0 and 6), the shipped-defaults one gates 0-6 in 64 s at 3.19 m/s with every arch within 0.4 m of centre and no contacts; the fastest 58 s at 3.35 m/s |
| `gatenet_colourblind` (same frames, strong augmentation) | 78.7% recall as itself and **77-80% through the whole colour battery** — grayscale, desaturation, hue 60 and 180, darkness, gamma, blur — where `gatenet_best` falls to 43.1% in grayscale and 51.1% at hue+180 with 40.6 px of centre error | **not for flying** (7 points of in-domain recall). On an unseen map (Pine Valley) it fires on 2.7% of frames against the shipped one's 3.0%, both below their false-positive rates at home (8.7% and 11.6%): colour invariance was necessary and is not sufficient |

Full difficulty: 25 degrees of tilt, 90 deg/s rotation, 1 m/s velocity and 1 m offset at the start,
targets anywhere in a 6 x 6 x 2 m box, physics jittered by 35%.

## How to use

```bash
git clone https://github.com/skulitom/haltere && cd haltere
uv venv --python 3.13 .venv && uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
# put the files of this repo into artifacts/ and data/built/, then:
haltere eval artifacts/ftSmooth_best.pt          # simulator evaluation
haltere render artifacts/ftSmooth_best.pt        # neural activity + drone video
haltere liftoff doctor                           # everything needed to fly it in Liftoff
```

## Data

Connectome: male CNS v1.0, Janelia FlyEM, CC BY 4.0, downloaded by `haltere fetch` from
`gs://flyem-male-cns` (neuron annotations, neurotransmitter predictions, connection weights). Not
redistributed here.

## Method in one paragraph

Lappalainen et al. 2024-style connectome-constrained RNN (rate units, weights proportional to
synapse counts with fixed neurotransmitter signs) on a 30k-neuron subgraph of the male CNS (all
neurons within two synapses of the flight senses and the wing motor neurons, plus the central
complex and all descending neurons). Trained in a batched differentiable quadrotor simulator with
Betaflight-style rates and rate PID (Liftoff's own Zetaflight gains) by imitation of an MLP
controller, then fine-tuned by back-propagation through the simulator with domain randomization.
Flown in Liftoff through its UDP telemetry and a virtual Xbox controller (ViGEmBus). The lap brain
was fine-tuned on targets drifting at up to 3 m/s (`configs/train_path.yaml`), then on a mix with
30% static targets, a Huber position cost and a stronger smoothness penalty (`train_path2.yaml`).
The brain has no camera and no heading objective; the pilot's `--face-travel` yaws the nose toward
the next point on the path so the FPV view looks where it flies. Two pilot-side findings made the
flight smooth and fast without retraining: Liftoff applies its gamepad deadzone to each stick's
two-axis vector, so the sticks are now inverted as vectors (the per-axis inverse had distorted the
brain's small corrections into a 2.4 Hz wobble); and the brain cruises at the speed its optic-flow
and airflow senses report, so scaling the horizontal velocity written into those senses
(`--flow-gain`) sets its speed, much as a fly speeds up when its visual feedback gain is lowered.
