# Haltere

A connectome-constrained fruit-fly brain controlling an FPV drone in
[Liftoff](https://store.steampowered.com/app/410340/Liftoff_FPV_Drone_Racing/).
The goal is a system that can complete **unfamiliar races and freestyle tasks**.
Visual pilot assistance and learned navigation are welcome when they improve
that goal; the fly brain remains a meaningful motor controller.

[Quickstart](#quickstart) · [Current status](#current-status) ·
[Models](#models-and-data) · [Game setup](docs/liftoff_setup.md) ·
[Training](#training) · [For agents and contributors](#for-agents-and-contributors)

![Connectome activity beside an earlier assisted Straw Bale flight](docs/liftoff_by_sight_v05.gif)

*Historical v0.5.0 recording, shown at 4× speed: the published motor brain with
GateNet and the Rabbit visual pilot. The reported seven-gate segment took 64 s.
This is an assisted system result on a seen course, not a recording of the newest
experimental brain or evidence of arbitrary-course flight.*
[Release and videos](https://github.com/skulitom/haltere/releases/tag/v0.5.0).

## Current status

**Reviewed against code and flight evidence on 2026-09-23.** Haltere can finish
specific seen races with assistance, but reliable, fast general race and
freestyle flight remains unsolved. The newest published weights are
[motor10 candidate05](docs/motor_brain_10_release.md), an experimental download,
not a promoted replacement for [scene09](docs/scene_brain_09_release.md).

The current acceptance target is **three clean full races on each of five
tracks, within 20% of the user's matching full-race time**, on one frozen stack
using the same original `[Copy] New Drone`:

| Three-lap race | User's full-race time | Maximum target time |
|---|---:|---:|
| Straw Bale / Field Day | 1:19.006 | 1:34.807 |
| Pine Valley / Forest For The Trees | 2:07.049 | 2:32.459 |
| Minus Two / Turn Signals | 1:29.277 | 1:47.132 |
| Autumn Fields / Walk In The Park | 1:04.494 | 1:17.392 |
| Hangar C03 / Shipments | 1:24.076 | 1:40.891 |

These are saved **race**, not single-lap, times. Exact IDs, values and provenance
are in [main_track_targets.json](configs/main_track_targets.json). They are
scoring metadata and are never supplied to the flight controller.

The latest [frozen full-race comparison](docs/flight_cards/2026-09-23_matched_full_races.md)
used motor10 and a corrected PD baseline under the same visual pilot at 2.5 m/s:
**brain 0/2 finishes; PD 1/2**. PD finished Straw Bale in **13:04.047**;
both motors hit a pillar on Minus Two. There were no camera or control-deadline
stops. All four untrimmed standard brain/gameplay videos fully decode; PD footage
explicitly labels the brain as running in shadow. This diagnostic supports
geometry as the next priority and also exposes a motor-tracking gap.

| Component | Current evidence and limits |
|---|---|
| Scene09 + visible race cues | Earlier full three-lap finishes: Straw Bale **14:05.703**, Minus Two **9:27.415**. Both are seen courses. |
| Motor10 candidate05 | Only three motor-readout rows/biases changed; recurrent weights are unchanged. Its release batch finished 0/3; a separate open development-loop repeat finished in **3:01.576**. Later diagnostic results are retained separately. |
| Full Rabbit visual assistance | Gate selection, target smoothing, speed and heading run with the visual brain. Its frozen five-attempt baseline completed no laps. [Guide](docs/visual_pilot_assistance.md). |
| Visible race cues | Causal checkpoint-marker guidance is disclosed. It supplies a direction, not free space or a freestyle objective. [Inputs](docs/race_cue_assistance.md). |
| Navigation predictor | Runtime use is allowed. No current flight runner loads the motion-forecasting predictor; it lacks task-directed flight evidence. [Direction](docs/project_direction.md). |
| Generalization | Five first-exposure courses failed across successive development revisions. A completed open generated loop does not establish obstacle avoidance or unseen racing. |

The [development program](docs/generalization_program.md) uses generated and
Workshop courses, frozen comparisons, camera-derived geometry, then broader
brain training. Manual map design is unnecessary. The
[obstacle-section generator and offline scorer](docs/challenge_sections.md)
include physical gate frames, descents and occlusions. The first box-calibration
autonomous flight hit the last wall. A separate **oracle-guided PD collection**
finished that course in **3:35.704**, qualifying its playability; this is not an
autonomous result. A causal image/motion geometry prototype detects the wall in
the failed-flight replay. On the complete collection trajectory it produced no
warnings, with 6.7% mean relative depth error on matched primitive surfaces, but
coverage remains sparse. Published geometry flights are passive; experimental
steering integration has not yet established an autonomous improvement.
[Attempts, diagnostic and limits](docs/flight_cards/2026-09-23_obstacle_geometry.md).
Course geometry is excluded from autonomous runtime; oracle collection is
explicitly labelled and scored separately.

Validation: **408 automated tests passed** (three existing warnings). The CPU
quickstart and checkpoint loading were checked in this checkout, and recorded
flights verified actual processed controls. A fresh installation has not been
revalidated. See the [flight index](docs/flight_cards/README.md) for historical
attempts and [project direction](docs/project_direction.md) for acceptance rules.

## Quickstart

The commands below are **PowerShell, run from the repository root**. Python
3.11+ is declared in [pyproject.toml](pyproject.toml); this checkout was checked
with Python 3.13.2 and PyTorch 2.11.0+cu128 on Windows with an RTX 4090.
Install [uv](https://docs.astral.sh/uv/) and Git first.

```powershell
git clone https://github.com/skulitom/haltere.git
Set-Location haltere
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -e ".[dev,vision]"
.venv/Scripts/python.exe -m haltere.cli --help
.venv/Scripts/python.exe -m haltere.cli eval artifacts/ftSmooth_best.pt --device cpu --batch 1 --steps 30
```

The last command is a short **loading and simulator smoke check** and prints
JSON metrics. It needs no game, gamepad or raw connectome download. It is too
short to assess flight quality. The graph and published inference checkpoints
are already tracked in this repository.

PyTorch is installed separately so you can select the build for your machine;
use the [official installer selector](https://pytorch.org/get-started/locally/)
for a different CUDA or CPU setup. Training is designed for an NVIDIA GPU.
The live game/capture/controller workflow is Windows-specific; other platforms
have not been verified in this update.

All examples use the environment's Python directly, so activation is optional.
With that environment activated, `haltere` is shorthand for
`python -m haltere.cli`. Optional extras are `liftoff` (virtual pad),
`fast-capture` (Windows DXGI), `neuprint`, `replays` and `publish`.
Install the gamepad driver before adding `liftoff`; see [game setup](docs/liftoff_setup.md).

## Models and data

| Checkpoint | Role |
|---|---|
| [Motor10 candidate05 bundle](docs/motor_brain_10_release.md) | New experimental motor-readout weights with recorded-scene robustness training. One development-loop finish; obstacle races failed. Download separately; not promoted over scene09. |
| [Scene09 bundle](docs/scene_brain_09_release.md) | Published learned-scene reference with its exact detector and original-drone mapping. Download separately; evaluated with explicit visual assistance. |
| `artifacts/ftSmooth_best.pt` | Published connectome controller for hover and movement patterns. |
| `artifacts/ftPath2_best.pt` | Published connectome controller for taught paths and the older visual-pilot stack. |
| `artifacts/ftRobust_best.pt` | Earlier controller trained with broad physics randomization. |
| `artifacts/imJ_best.pt` | Imitation-trained connectome controller; simulator reference. |
| `artifacts/mlp_baseline.pt` | Non-connectome motor teacher/control baseline. |
| `artifacts/gatenet_best.pt` | Published gate detector for the older visual stack. |
| `artifacts/gatenet_colourblind.pt` | Detector augmentation experiment; not a demonstrated transfer improvement. |
| `artifacts/experimental/navigation_*.pt` | Offline path predictors; [versions and limits](artifacts/experimental/README.md). |

Published assets are also available on
[GitHub Releases](https://github.com/skulitom/haltere/releases) and
[Hugging Face](https://huggingface.co/Skulitom/haltere).
Inference checkpoints omit optimizer state and depend on the matching
`data/built/flight.npz`, `flight.nodes.parquet` and `flight.meta.json`.
Use full training checkpoints from a run for `train --resume` / `imitate --resume`.

`runs/`, raw connectome tables and recordings under `data/vision/` and
`data/liftoff/` are local and ignored by Git. Research guides refer to those
local experiments; those paths will not exist in a fresh clone. Dataset plans
pin particular source takes, not downloadable public training data.

## How it works

The checked-in graph selects **30,000 neurons and 2,767,698 directed edges**
from the male CNS v1.0 connectome. It is a rate-network abstraction of a selected
subgraph, not the entire biological fly. Graph selection and transmitter-sign
rules are in [configs/flight.yaml](configs/flight.yaml); actual graph counts
are in [flight.meta.json](data/built/flight.meta.json).

Telemetry is converted into gyro, load, flow, attitude and other sensory
channels. Encoders drive selected fly populations; the recurrent network
produces motor activity. Published working brains read from wing motor **and
premotor** populations. Visual variants add image or frozen scene features;
the checkpoint specifies the exact sensory encoding and detector calibration.

The deployed arrangement can include a visual pilot or learned planner before
the brain's goal inputs, and explicit heading assistance after its output.
The [assisted visual mode](docs/visual_pilot_assistance.md) records which commands
come from each component. Teacher models and known routes can also supply
training labels. Distinguish training-only labels from the declared runtime inputs.

| Location | Responsibility |
|---|---|
| `haltere/connectome/` | Source tables, population selection and graph construction. |
| `haltere/brain/` | Sparse recurrent network, sensory encoders and motor readout. |
| `haltere/sim/` | Differentiable quadrotor, rates, PID, mixer and training tasks. |
| `haltere/train/` | Motor imitation, flight-cost training, human/gate/scene learning and evaluation. |
| `haltere/vision/` | Gate detection, datasets, scene features and path-prediction experiments. |
| `haltere/liftoff/` | Telemetry, control mapping, capture, pilots, recording and scoring. |
| `configs/`, `tests/`, `docs/` | Configurations, automated checks and evidence/workflow guides. |

## Training

Start motor imitation with the included MLP teacher and the premotor readout:

```powershell
.venv/Scripts/python.exe -m haltere.cli imitate --config configs/train_premotor.yaml --imitate-config configs/imitate_premotor.yaml --teacher artifacts/mlp_baseline.pt --run runs/imitate-new
.venv/Scripts/python.exe -m haltere.cli eval runs/imitate-new/best.pt --batch 8 --steps 400
```

The explicit teacher path avoids the configuration's historical
`runs/mlp300/best.pt` dependency. Use a new run directory. This is a full training
job, not part of the quickstart. Flight-cost fine-tuning uses `haltere train`;
`configs/train_premotor_smooth.yaml` and `configs/train_path2.yaml` describe
smoothing and moving-target curricula. Basic `configs/train.yaml` is an
experimental starting configuration, not a recipe that reproduces the shipped brain.

| Task | Guide / entry point |
|---|---|
| Record and prepare human demonstrations | [Human recordings](docs/human_demonstrations.md). |
| Train the actual fly brain from recordings | [Human, gate and scene learning](docs/human_brain_training.md). |
| Train/evaluate the separate predictor | [Navigation training](docs/navigation_training.md), [v2 evidence](docs/navigation_release_v02.md). |
| Collect oracle-guided demonstrations | [Route collection](docs/route_collection.md), [bot-route extraction](docs/bot_routes.md). |
| Test unfamiliar courses and tasks | [Project acceptance criteria](docs/project_direction.md), [race protocol](docs/generalisation.md). |

Training a predictor alone does not update the fly brain. Report the actual
changed parameters for every brain-training claim. Evaluate the intended full
stack, and use no-predictor/no-assistance comparisons to attribute changes.
Preserve raw recordings and whole-take/course holdouts.

To rebuild a graph, download the public tables and write to a new output stem:

```powershell
.venv/Scripts/python.exe -m haltere.cli fetch
.venv/Scripts/python.exe -m haltere.cli populations
.venv/Scripts/python.exe -m haltere.cli build --out runs/flight-rebuilt
```

The raw download is approximately 570 MB and needs no login. A rebuilt or
modified graph must not silently replace the graph used by an existing model.
Optional neuPrint access requires the `neuprint` extra and
`NEUPRINT_APPLICATION_CREDENTIALS`; `haltere neuprint check` checks access.

## Flying and recording

Follow [Liftoff setup](docs/liftoff_setup.md) for telemetry, driver installation,
calibration and the persistent pad bridge. Run the game and its capture/controller
processes in **Anode**, with the viewer hidden unless requested. Retain the original
`[Copy] New Drone`, verify throttle-low and the game's processed controls, pause
before disconnecting the pad, and keep Liftoff open between tests.

- Published motor checkpoints use `haltere liftoff fly`.
- New visual checkpoints use `python -m haltere.liftoff.visual_brain` and their
  recorded detector/mapping contract. The legacy `fly` command rejects these.
- Add `--pilot-assistance rabbit` for the new [assisted visual mode](docs/visual_pilot_assistance.md).
- `--record PATH.mp4` produces the standard brain-activity panel beside gameplay.
  Use fresh log/video paths and label shadow footage as shadow.

The [flight cards](docs/flight_cards/README.md) record predictions, complete
commands, model provenance and observed outcomes. Verify race completion from
the game, not just the pilot's estimated gate count. Scripted patterns and
known-route demonstrations are useful tracking tests, but do not establish
unseen freestyle or race navigation.

## For agents and contributors

Read [AGENTS.md](AGENTS.md) and [project_direction.md](docs/project_direction.md)
first. Inspect the working tree before editing and preserve unrelated changes.
Use the current command help and checkpoint metadata as the implementation
reference; historical guides document particular experiments, not universal defaults.

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m haltere.cli liftoff fly --help
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain --help
```

For changes to live control, verify the actual processed input and record the
whole stack's assistance mode, checkpoint hashes, code revision, camera/mapping
and outcome. Publish completed, verified code to GitHub. Publish new weights
only with accurate provenance and evaluation limits; offline and synthetic
checks alone do not qualify an improved flight controller.

The previous long experiment narrative remains in the
[historical README at 6e86b8b](https://github.com/skulitom/haltere/blob/6e86b8b82f77228ec71919a5c50583ad83258d2b/README.md).
Its dates, recommendations and training-only policy are historical.

## Attribution

Project code is [MIT licensed](LICENSE). The source connectome is the
[Janelia male CNS dataset](https://www.janelia.org/project-team/flyem/male-cns-connectome),
created by Janelia FlyEM, Cambridge Connectomics and Google collaborators and
released under CC BY. The included flight graph is derived data; retain source
attribution. The full raw tables are downloaded separately.
