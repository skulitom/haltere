# Human flight examples

The manual recorder saves FPV frames, flight poses and Liftoff's observed control
inputs. `haltere.vision.demonstrations` converts selected intervals into temporal
examples for learning a local flight path. It does not run a pilot or train a model.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.manual_recording
.venv/Scripts/python.exe -m haltere.vision.demonstrations configs/human_demonstrations_v1.json --out data/vision/human_demonstrations_v1
```

The recorder starts idle. Ctrl+Alt+F9 starts/stops a take; Ctrl+Alt+F10 changes the
profile while stopped. Each take has a new timestamped directory. The recorder
captures only the active Liftoff window and ends a file on a reset or results
telemetry. Keep the original files: preparation references them without copying
or editing them. A prepared output directory must be new.

## Selection and split

The checked-in plan describes the recordings from 20 September 2026:

| Recording | Use |
|---|---|
| Straw Bale race, take 1 | Training |
| Pine Valley race, take 2 | Training |
| Straw Bale fence freestyle, take 1 | Training |
| Minus Two race, take 2 | Validation on an unseen course |
| Pine Valley and Minus Two, take 1 | Review only: user reported mistakes |

The first preparation produced 5,093 training examples (632 overlapping
16-frame sequences at stride 8), 1,643 validation examples (204 sequences),
and 4,312 review-only examples. Additional recordings do not enter this version
automatically; add their explicit paths to a new plan and prepare a new output.

The plan removes idle starts and finish/landing tails. The longest future-path
horizon also removes one second of example starting points at each selected
segment's end. Whole takes stay in one split. Nonuniform images are hashed across
whole source takes to reject exact copies between training and validation.
Uniform transition images are excluded. All image hashes and source-file hashes
are retained in `manifest.json`; the loader checks prepared arrays and loaded images.

The first Pine Valley take has a rapid slowdown around 54.17 seconds; the first
Minus Two take has one around 87.10 seconds. These are **review candidates**, not
automatic collision labels. The clean replacements show the three-lap finish in
sampled visual review. This does not certify every frame as collision-free.

## Example format

Each NPZ contains source filenames/row numbers, original telemetry times,
relative world positions, body-to-world quaternions, body velocity, observed
controls, sequence group IDs, and future positions at 0.25, 0.5 and 1 second.
Future displacement is expressed in the current drone's forward/left/up frame,
in metres. Interpolation uses the irregular telemetry timestamps, not the preview
video's nominal 20 fps. Sequences never bridge a trimmed interval, excluded image
or time gap above 120 ms. Future paths also stop at telemetry gaps and segment ends.

```python
from torch.utils.data import DataLoader
from haltere.vision.demonstrations import DemonstrationSequences

train = DemonstrationSequences('data/vision/human_demonstrations_v1',
                               split='train', length=16, stride=8)
batch = next(iter(DataLoader(train, batch_size=8, shuffle=True)))
# images:       [batch, time, 3, 90, 160], RGB in [0, 1]
# future_body:  [batch, time, 3 horizons, 3 coordinates], metres
# controls:     [batch, time, 4], throttle/yaw/pitch/roll
# velocity_body, attitude, time_s and take_id are also available.
```

Use `future_body` as the navigation target and `controls` as recorded supervision,
not as an input when evaluating control prediction. Liftoff Input is **not** a raw
radio or Xbox command; the existing stick mapping must not be applied blindly.
The current simulator imitation trainer consumes synthetic controller rollouts
and does not yet consume these examples. The sequence loader is the handoff for
an offline navigation learner; preparation itself does not improve the pilot.

The inherited camera calibration and physical display delay remain unvalidated.
Therefore these examples have no projected gate labels, and the fence is not
labelled as an arch. The existing course gate files are incomplete for these full
races. Model weights stay unchanged until a learner has been evaluated. There is
only one fence take, so this set cannot measure independent fence generalization.
