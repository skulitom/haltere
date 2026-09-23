---
license: mit
library_name: pytorch
pipeline_tag: reinforcement-learning
tags:
  - connectome
  - drosophila
  - computational-neuroscience
  - learned-control
  - drone
  - fpv
  - liftoff
  - recurrent-neural-network
  - imitation-learning
  - differentiable-simulation
---

# Haltere — connectome-constrained drone control

A **30,000-neuron recurrent network** built from a subgraph of the fruit-fly male CNS connectome and trained to control a quadcopter in the **Liftoff simulator**. Synaptic structure and signs follow the connectome; connection gains, neuron dynamics, sensory encoders and motor readout are learned.

This repository contains controller checkpoints, gate detectors and the derived flight graph. The controller receives telemetry-derived sensory inputs. For flights guided by camera images, a separate gate detector and hand-engineered pilot provide the goal it follows. Training uses imitation and back-propagation through a differentiable simulator; the Hub's reinforcement-learning label is a broad task category.

**For the current experimental visual stack, use the self-contained [scene09](https://huggingface.co/Skulitom/haltere/tree/main/scene09) or [motor10](https://huggingface.co/Skulitom/haltere/tree/main/motor10) bundles and their versioned instructions.** Motor10 candidate05 is the newest published motor experiment, not a promoted replacement. Its release evaluation recorded one generated-loop finish in four development attempts, including flag and pillar impacts on Straw Bale and Minus Two and a separate telemetry stop. The successful repeat took 3:01.576. No qualifying unseen-course or freestyle result is claimed. See the [motor10 provenance and limits](https://github.com/skulitom/haltere/blob/main/docs/motor_brain_10_release.md) and [continuing flight evaluation](https://github.com/skulitom/haltere/blob/main/docs/flight_cards/README.md).

Only the three motor readout rows and their biases changed in motor10 relative to its parent; the recurrent connectome is unchanged. Scene09 learned its visual goal adapter. These bundles preserve their exact sensory, mapping and source contracts. A readout fitted to PD commands is an initialization experiment, not evidence that recurrent brain learning outperforms PD.

**The quickstart, video and root-level checkpoint descriptions below cover the older release.** Start with `ftSmooth_best.pt` for those hover and movement experiments, or `ftPath2_best.pt` for their taught-path and by-sight pipeline. The pinned source and `download_assets.py` workflow below do not install the newer visual bundles. All are research prototypes evaluated in simulation, with no demonstrated real-drone deployment or generalisation to arbitrary maps.

[Code](https://github.com/skulitom/haltere) · [Quickstart](#quickstart) · [Results](#results) · [Related cursor readouts](https://huggingface.co/Skulitom/ganglion-haltere-cursor)

<video controls preload="none" width="100%" poster="https://huggingface.co/Skulitom/haltere/resolve/main/preview.jpg" src="https://huggingface.co/Skulitom/haltere/resolve/main/replay.mp4"></video>

[Watch or download the recorded by-sight flight](https://huggingface.co/Skulitom/haltere/resolve/main/replay.mp4). The video shows the controller, gate detector and pilot working together in Liftoff. It is a recording, not an interactive demo. The reported clean-run window is gates 0 through 6; it is not a claim that the entire recording is contact-free.

## Quickstart

The commands below use **Windows PowerShell**, Python 3.13 and CUDA-enabled PyTorch 2.11.0. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git first. An NVIDIA GPU is recommended; the short simulator check can also use `--device cpu`. Liftoff and its virtual-controller setup are only needed for game integration.

```powershell
mkdir haltere-demo
cd haltere-demo
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe "haltere @ git+https://github.com/skulitom/haltere@94c3e53c12ae0a1aa4cf9cdc6657d08d652d9829" huggingface_hub
.venv/Scripts/python.exe -c "from huggingface_hub import hf_hub_download; from shutil import copyfile; copyfile(hf_hub_download('Skulitom/haltere', 'download_assets.py'), 'download_assets.py')"
.venv/Scripts/python.exe download_assets.py
.venv/Scripts/haltere.exe eval artifacts/ftSmooth_best.pt --steps 100 --batch 4 --device cuda
```

The last command runs a short **installation check**, not the published benchmark. It prints a JSON result with tracking and crash statistics. The download helper reads [config.json](config.json), resolves one Hub revision, downloads the selected checkpoint and shared assets, and verifies each file's SHA-256. Existing files with different contents are left untouched; use a fresh `--output` directory for another bundle.

The resulting layout is:

```text
haltere-demo/
  artifacts/ftSmooth_best.pt
  data/built/flight.npz
  data/built/flight.nodes.parquet
  data/built/flight.meta.json
  configs/liftoff.yaml
  configs/track_strawbale.yaml
  configs/gates_strawbale.json
  configs/camera_seat.yaml
```

Download another controller or the optional gate detector into the same bundle:

```powershell
.venv/Scripts/python.exe download_assets.py --checkpoint ftPath2_best.pt
.venv/Scripts/python.exe download_assets.py --checkpoint gatenet_best.pt
```

Run commands from the bundle root so `data/built/flight` resolves consistently. On Linux, use `.venv/bin/python` and `.venv/bin/haltere` for simulator work; the Liftoff integration documented here targets Windows. The source revision is pinned for this quickstart, not asserted to be the original training revision.

### Fly in Liftoff

Follow the [Liftoff setup guide](https://github.com/skulitom/haltere/blob/94c3e53c12ae0a1aa4cf9cdc6657d08d652d9829/README.md#3-fly-in-liftoff) for the game, UDP telemetry, virtual gamepad and controller calibration. The simulator quickstart does not install or configure those components.

```powershell
.venv/Scripts/haltere.exe liftoff doctor
.venv/Scripts/haltere.exe liftoff fly artifacts/ftSmooth_best.pt --liftoff-config configs/liftoff.yaml
```

For a taught path, after downloading `ftPath2_best.pt` and completing that setup:

```powershell
.venv/Scripts/haltere.exe liftoff fly artifacts/ftPath2_best.pt --liftoff-config configs/liftoff.yaml --waypoints-file configs/track_strawbale.yaml --path-speed 8 --lookahead 6 --z-lead 1.5 --flow-gain 0.5 --face-travel 0.8 --face-ahead 6 --throttle-scale 0.8 --gyro telemetry
```

## Checkpoints and files

| File | Purpose |
|---|---|
| `ftSmooth_best.pt` | Recommended starting controller: hover, orbit and climb-and-dive; fine-tuned for latency and smoothness. |
| `ftPath2_best.pt` | Moving-target fine-tune for taught paths and the by-sight pipeline. |
| `ftRobust_best.pt` | Earlier controller with wide domain randomisation; first successful Liftoff flights. |
| `imJ_best.pt` | Imitation-trained connectome controller. |
| `mlp_baseline.pt` | Flight-control MLP teacher; not the Ganglion cursor-suite baseline. |
| `gatenet_best.pt` | Separate gate detector used by the by-sight pipeline. |
| `gatenet_colourblind.pt` | Augmentation experiment for studying generalisation; not the recommended detector for flying. |
| `flight.npz`, `flight.nodes.parquet`, `flight.meta.json` | Derived flight graph, placed together in `data/built/`. |
| `liftoff.yaml`, `track_strawbale.yaml`, `gates_strawbale.json`, `camera_seat.yaml` | Game mapping, taught path, gate locations and camera calibration, placed in `configs/`. |
| `config.json`, `download_assets.py` | Versioned artifact manifest and checksum-verifying downloader. The manifest is not a Transformers configuration. |

The connectome checkpoints contain parameters and configuration (about 12 MB each); they load the separate graph. Full checkpoints with optimiser state and additional videos are available in [GitHub releases](https://github.com/skulitom/haltere/releases).

## Results

These are the previously reported experiments, not results from the short installation check. Simulator and Liftoff measurements refer to different tasks and conditions.

| controller | simulator, full difficulty | Liftoff |
|---|---|---|
| MLP baseline | 0.05 m mean error, 100% within 0.5 m | not flown |
| `imJ_best` (imitation) | 0.22 m, 95% | drifts 1.4 m on the physics stand-in |
| `ftRobust_best` (+ domain randomization) | 0.20 m, 99.6% | 2 m hover, 0.34 m mean error over 40 s; 3 m square pattern |
| `ftSmooth_best` (+ latency, smoothness) | 0.30 m, 95% (50 ms delay) | 2 m hover, 0.35 m mean error with a quarter of the stick jitter; orbit (0.75 m tracking error at 0.8 m/s) and climb-and-dive (1.1 m at about 1 m/s), no crashes; taught lap at 1.2 m/s with 0.9-1.0 m error |
| `ftPath2_best` (+ moving targets) | 0.37 m, 80% static; 0.76 m following a 1 m/s target, 3.1 m at 2 m/s | taught race lap: the seven gates in 38 s at 4.8 m/s with `--flow-gain 0.5`, no contact between gates 0 and 6, roll and pitch rate shake 2 deg/s; **by sight with the rabbit pilot, all seven gates in 64 s at 3.19 m/s**, every arch within 0.4 m of centre, no contacts between gates 0 and 6, yaw shake 2.0 deg/s |
| `gatenet_best` (gate detector, 5 M parameters) | on whole flights recorded AFTER it was trained: 85.6% recall, centre error 2.9 px at 320 wide, 11.6% false positives on gate-less frames (9.6% pooled over all the gate-less frames of those flights). (An earlier card said 98% on "a held-out tenth" — that split took single frames from the same flights, and frames 130 ms apart are the same picture, so it could not fall) | flies by sight: five clean 7/7 laps of Straw Bale Field Day (clean between gates 0 and 6), the shipped-defaults one gates 0-6 in 64 s at 3.19 m/s with every arch within 0.4 m of centre and no contacts; the fastest 58 s at 3.35 m/s |
| `gatenet_colourblind` (same frames, strong augmentation) | 78.7% recall as itself and **77-80% through the whole colour battery** — grayscale, desaturation, hue 60 and 180, darkness, gamma, blur — where `gatenet_best` falls to 43.1% in grayscale and 51.1% at hue+180 with 40.6 px of centre error | **not for flying** (7 points of in-domain recall). On an unseen map (Pine Valley) it fires on 2.7% of frames against the shipped one's 3.0%, both below their false-positive rates at home (8.7% and 11.6%): colour invariance was necessary and is not sufficient |

Full difficulty: 25 degrees of tilt, 90 deg/s rotation, 1 m/s velocity and 1 m offset at the start,
targets anywhere in a 6 x 6 x 2 m box, physics jittered by 35%.

## Limitations

- A rate-network abstraction of a selected connectome subgraph, not a complete biological simulation of a fruit fly.
- The controller, sensory encoding, pilot and detector all contribute to the displayed behaviour. By-sight flight still uses telemetry for flight state.
- Published successful runs are from a specific simulator, map and configuration. Gate detection is sensitive to visual changes and does not establish unseen-map generalisation.
- Clean gate-to-gate traversal does not imply a crash-free run before take-off, after gate 6 or across arbitrary restarts.
- No claim of validated real-world drone control, biological equivalence or superiority to all conventional controllers.

## Method

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

## Data and attribution

The source is the [Janelia male CNS connectome v1.0](https://www.janelia.org/project-team/flyem/male-cns-connectome), from Janelia FlyEM, Cambridge Connectomics and Google Connectomics, identified by the project as CC BY 4.0. `haltere fetch` retrieves the source annotations, neurotransmitter predictions and connection weights from `gs://flyem-male-cns`.

The full raw source tables are **not** distributed in this repository. The included `flight.npz`, `flight.nodes.parquet` and `flight.meta.json` are a **derived subgraph**: 30,000 neurons, signed synapse counts and selected populations. Preserve the source attribution and consult the upstream terms for those data. The model repository's MIT label describes the project's code/checkpoint release; it does not replace the upstream data attribution.

## More recordings

Earlier experiments are linked here rather than loaded as seven consecutive animations:

- [Taught fast lap](liftoff_fast_lap.gif): earlier path-following recording; separate from the by-sight video above.
- [Earlier by-sight hill-gate run](liftoff_sight_hill.gif): the six-gate run described in the original card, preceding the later seven-gate results.
- [Before and after the stick-path fix](liftoff_stickfix.gif).
- [Hover](liftoff_hover.gif), [orbit](liftoff_orbit.gif), and [climb-and-dive](liftoff_climbdive.gif).
- [Earlier taught lap](liftoff_race.gif).
