# Second-course flight: collection and acceptance plan

Started 2026-09-20. The next milestone is repeatable Pine Valley flight while retaining
the shipped Straw Bale result. Pine Valley is a development map, not a sealed test.

**Update from the bot investigation:** Pine's installed race uses 28 checkpoint-box
objects and only two distinct arches. A detector-only improvement cannot be assumed
to solve its navigation. We have now extracted a three-lap bot-pool recording; see
[the verified findings and revised next experiment](bot_routes.md). The collection
and dataset checks below still apply, but detector training is conditional on
collecting and reviewing the appropriate visual targets.

## Frozen starting point

Keep the brain, flight mapping and rabbit pilot unchanged during detector experiments.
Do not overwrite the shipped detector; candidates belong in new `runs/` directories.
The starting pilot revision is `29e50da`.

| Artifact | SHA-256 |
|---|---|
| `artifacts/ftPath2_best.pt` | `fecdd56bb0ce45ec800c21d1aafbc10954eba6bac604eff388aa25395db494e5` |
| `artifacts/gatenet_best.pt` | `a00d0252e08011349061123484406dfcbcff26c4c7506b48a8295f75f1376cd2` |
| `configs/liftoff.yaml` | `4d71aeb7d5546e6c3afbc08706913194c1323a77a5884b7e9433621a6b4a9b82` |

## Current data audit

These are label counts, not verified detections or race scores.

| Recording | Frames | Positive labels | Gate IDs represented |
|---|---:|---:|---|
| pine1 | 1,392 | 578 | 0 only |
| pine2 | 2,092 | 356 | 0 only |
| run29 | 1,400 | 1,155 | 0–6 |
| run30 | 1,607 | 901 | 0–6 |
| run31 | 1,418 | 948 | 0–6 |

The Pine gate file describes only one arch. Its projected negative labels do not
establish that other gates are absent. Both datasets also contain the same all-black
JPEG (`pine1/001362.jpg`, `pine2/002039.jpg`), verified by hash and visual inspection.
Retain the originals; exclude black, countdown, reset and unsuitable frames when
building reviewed training copies. Do not begin a full-course fit from these labels.
The local inventory with hashes is `runs/second-course-preflight/inventory.json`.

## Collect a complete player flight

The new passive command captures only the foreground Liftoff window, stores the
player's original telemetry, and writes the same image/pose index the existing
calibration, beacon, labelling and inspection commands consume. It sends no input,
does not load the brain, and does not change focus. A game reset ends the recording;
use a new directory per flight. Existing directories are refused.

1. Select Pine Valley's development race. Reset before starting collection; begin
   recording while stationary so the first captured pose is a useful origin.
2. Select the calibration matching the actual capture setup. `camera.yaml` is the
   normal-desktop calibration; `camera_seat.yaml` is the previously measured Anode
   setup. A changed field of view requires recalibration. In Anode, run the recorder
   in that session too.
3. Start a bounded recording, bring Liftoff to the foreground, and fly a complete lap:

```powershell
.venv/Scripts/python.exe -m haltere.cli liftoff capture --out data/vision/pine_manual_01 --course "Pine Valley / Forest For The Trees" --camera configs/camera.yaml --seconds 180
```

Outputs: `frames/`, `index.csv`, `telemetry.csv`, `camera.yaml`, `capture.json`.
The index includes capture start/end, UDP-receive age, simulator-frame pose relative
to the first accepted image, and telemetry Input values. Those inputs are the game's
processed inputs, not raw radio commands. Display latency is not measured; the
reported receive age alone is not proof of exact image/pose synchronization.

4. Check the first short recording with `vision calibrate` and inspect image/pose
   alignment before collecting longer flights. Then collect at least separate training
   and validation flights, including turns and approaches. Save a game recording or
   review the HUD to establish course completion; the capture command never infers it.
5. Use `vision beacon` and inspect its candidate clusters. Green ground strips also
   produce clusters. Recover and verify every gate before projecting labels. An
   exploration spiral which never advances a checkpoint cannot reveal later beacons.
6. Run `vision label` and `vision inspect` using each recording's own coordinate origin
   and camera. Check occlusion and false negative labels as well as visible arches.
   Record the review and HUD result with the flight card; `capture.json` starts with
   `labels_reviewed` and `course_complete` false.

## Train and compare

Create a reviewed dataset copy for each flight, keeping its frame index consistent
with the selected labels. Preserve a whole Pine flight for validation, alongside a
home validation flight that the candidate and its parent have not trained on.
Never split neighbouring frames into train and validation. Exact-image hashing catches
byte-identical copies; it cannot certify independence of similar scenes.

```powershell
.venv/Scripts/python.exe -m haltere.cli vision audit data/vision/pine_train_reviewed data/vision/home_train_reviewed --holdout data/vision/pine_val_reviewed data/vision/home_val_reviewed --out runs/two-course-audit.json
.venv/Scripts/python.exe -m haltere.cli vision train data/vision/pine_train_reviewed data/vision/home_train_reviewed --holdout data/vision/pine_val_reviewed data/vision/home_val_reviewed --out runs/gatenet_two_course_v1 --augment strong
```

These reviewed directories are future outputs, not existing approved data. Training
now rejects overlap, seeds NumPy augmentation as well as Torch/Python, and embeds
the dataset audit in the checkpoints and `datasets.json`. Warnings in an inventory
remain review items; the audit does not certify visual correctness. A legacy random
frame split remains available but is explicitly identified in provenance.

Evaluate recall, false positives and localization separately on home and Pine. Fit
the candidate's range correction with `vision rangefit`, keep the calibration with
the candidate, and test it as a distinct change. Gate-width-independent ranging is
the next separate experiment: the shipped range estimate assumes 4 m gates.

Proposed live acceptance: at least 4/5 complete Pine runs and 5/5 clean home runs,
recording every attempt, HUD checkpoints, contacts and flight-card verdicts. This is
a development threshold, not a statistical reliability claim. After passing, freeze
the complete stack before testing an untouched map under `generalisation.md`.

## Can Liftoff bots provide the data?

They may provide useful route shapes and gate approaches, but a route is not a set
of stick commands. Keep these possible uses separate:

- **FPV images:** potentially useful gate-training examples after manual annotation,
  even without poses. Spectator/free-camera views have a different camera contract.
- **Timed poses:** useful path targets, speed/curvature profiles and posed images if
  the actual replay-camera pose can be aligned. They do not supply action labels.
- **Inputs plus state:** could support imitation only if the controls, timing and
  vehicle dynamics are verified. Do not infer that a moving ghost is physics-driven.

The developer says bot flight paths are generated from data bundled with modern
versions of Liftoff ([developer explanation](https://steamcommunity.com/app/410340/discussions/0/3811782223865983050/)).
The official telemetry guide explicitly excludes spectated multiplayer drones and
replay sessions ([telemetry contract](https://steamcommunity.com/sharedfiles/filedetails/?id=3160488434)).
Thus ordinary UDP collection is not a verified way to obtain bot poses or actions.

Read-only inspection of installed build `25118475` subsequently located the actual
recording register and 275 entries for Forest For The Trees. A selected recording
has now been exported and checked for timestamp, quaternion and content-ID
consistency. See [bot_routes.md](bot_routes.md) for commands, evidence and remaining
alignment work. No game assets were modified. Any course used for bot-derived
training becomes development data and cannot remain a sealed test.

No detector or controller was retrained or promoted during this preparation; the
tiny CPU training test only validates the software pipeline. The new bot export is
an offline route reference, pending live alignment and reviewed image collection.
