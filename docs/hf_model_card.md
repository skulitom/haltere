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

![The fly brain flying the drone in Liftoff](liftoff_hover.gif)

![Orbit in Liftoff](liftoff_orbit.gif)

![Climb and dive in Liftoff](liftoff_climbdive.gif)

![Following a taught race lap in Liftoff, nose along the path](liftoff_race.gif)

## Files

| file | what | use |
|---|---|---|
| `ftPath2_best.pt` | the smooth brain fine-tuned on moving targets (two stages): follows a taught race lap at 1.5 m/s | `haltere liftoff fly ftPath2_best.pt --waypoints-file track.yaml --path-speed 1.5 --face-travel 0.8` |
| `ftSmooth_best.pt` | fine-tuned with 50 ms extra latency and a smoothness penalty: hover, orbit, climb-and-dive | `haltere liftoff fly ftSmooth_best.pt` |
| `ftRobust_best.pt` | wide domain randomization; first brain that flew in Liftoff | |
| `imJ_best.pt` | imitation of the MLP controller with the premotor readout | |
| `mlp_baseline.pt` | the MLP teacher (no connectome) | control experiment |
| `flight.npz`, `flight.nodes.parquet`, `flight.meta.json` | the built flight graph: 30,000 neurons, signed synapse counts, named populations | required by every brain checkpoint |

The checkpoints are slim (parameters only, about 12 MB); the graph is loaded next to them. Full
checkpoints with optimizer state are on the
[GitHub release](https://github.com/skulitom/haltere/releases/tag/v0.1.0).

## Results

| controller | simulator, full difficulty | Liftoff |
|---|---|---|
| MLP baseline | 0.05 m mean error, 100% within 0.5 m | not flown |
| `imJ_best` (imitation) | 0.22 m, 95% | drifts 1.4 m on the physics stand-in |
| `ftRobust_best` (+ domain randomization) | 0.20 m, 99.6% | 2 m hover, 0.34 m mean error over 40 s; 3 m square pattern |
| `ftSmooth_best` (+ latency, smoothness) | 0.30 m, 95% (50 ms delay) | 2 m hover, 0.35 m mean error with a quarter of the stick jitter; orbit (0.75 m tracking error at 0.8 m/s) and climb-and-dive (1.1 m at about 1 m/s), no crashes; taught lap at 1.2 m/s with 0.9-1.0 m error |
| `ftPath2_best` (+ moving targets) | 0.37 m, 80% static; 0.76 m following a 1 m/s target, 3.1 m at 2 m/s | taught race lap at 1.5 m/s with 0.67 m mean error, 46% of the time within 0.5 m, flying through the gates nose first |
| `gatenet_best` (gate detector, 5 M parameters) | agrees with the projected gate labels on 97% of a held-out tenth of 8.7k flight frames, centre error 7 px at 320 wide; no false positives left on the by-sight flight's gate-less views | flies by sight: with `--vision` the goal comes from this network's view of the game, and the brain flew through a gate it saw on the first such flight |

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
the next point on the path so the FPV view looks where it flies.
