# Flying races and freestyle tasks the system has never seen

**Current direction, 2026-09-22:** evaluate the whole deployed system, including
any visual pilot or learned navigation predictor. Assistance is allowed when
it helps complete unseen races and freestyle tasks. See
[project direction and acceptance criteria](project_direction.md) for the
authoritative goal, component roles and separate freestyle protocol.

Generic visible race cues may be used by an explicitly declared mode; route
lookups and hand-entered geometry do not establish unseen-course navigation.
Record the cue set and test freestyle without relying on race markers. The
older by-sight implementation described below does not explicitly read markers.

The results and status table below are the **historical September 2026 race
baseline**, not a current full-system qualification. The seven-gate score is a
measured segment; verify complete race finish separately.

The by-sight stack flies clean 7/7 laps on Straw Bale "Field Day". That number says nothing about
any other course, and the first flight on an unseen map showed why: over the 1392 Pine Valley frames
the detector fired on 3.0%, median confidence 0.010, with arches plainly in view - against its own
11.6% false-positive rate on gate-less frames at home. Detection on an unseen course was worse than
its own noise floor, and the pilot never got a gate to fly at.

The same detector loses half its skill when the colour goes: on held-out home flights, recall
85.6% -> 43.1% in grayscale and -> 51.1% at hue+180, where the centre error goes 2.9 -> 40.6 px and
false positives 11.6% -> 56.4%. Brightness, gamma and blur barely touch it. What it learned is a
palette.

This file is the discipline that keeps that from being discovered late again. It is about
measurement and hold-out rules, not about the fixes.

## 1. "Did it complete the race?" — four instruments, all four every flight

On the home course `liftoff score` answers from `configs/gates_strawbale.json`. On an unseen track
there is no gate list, and a default gate list scores a foreign flight against Straw Bale's seven
gates and prints confident nonsense. So:

| Instrument | What it gives | How |
|---|---|---|
| The game's race HUD | Ground truth: checkpoint counter, splits, finish screen | Fly in race mode, capture with `--record`; the HUD is in the MP4. **The primary number.** |
| Video review | *Why* it failed: laterals by eye, contacts, near misses the counter hides | Same MP4, scrubbed at each split |
| The pilot's own pass events | What the pilot *believed*: `n_passes`, `pass_kind`, `mode`, `det_gap` | `fly --log` columns. Never trust alone — it records a clean pass 3 m wide of a narrow gate |
| Gate-list-free telemetry | Speed, shake, collisions, height range, resets; comparable across any map | `haltere liftoff score <log> --gates ""` |

Report **pass-event precision and recall against the HUD** every flight. At home both are 1.0. On a
track with smaller or larger gates they will not be, and that gap is exactly the failure that lets a
pilot report "7/7" while missing most of the race.

A gate list built from a track's own flight (`vision sheet` → click both posts → `vision triangulate`)
is a measuring stick for laterals, measured *after*. Feed it back into a run on that track and the
run is no longer zero-shot.

## 2. What counts as zero-shot — declare the rung per flight

| Rung | Definition | Status |
|---|---|---|
| Z0 | Same map, same gates, in the detector's training set | 7/7, done |
| Z1 | Same map and gate type, different race line | untested |
| Z2 | Unseen map, same gate family | untested |
| Z3 | Seen map, unseen gate type (Straw Bale's dark truss cubes, labelled "nothing" in 43k frames) | expected to fail |
| Z4 | Unseen map *and* unseen gate type — the goal | measured at noise on `pine1` |

The original development milestone was **Z4 on three sealed maps**, with at
least one full finish and ≥80% of checkpoints on the others on first attempts.
Report that as partial transfer evidence. It does not establish completion of
all three races, arbitrary races, or freestyle. Current acceptance uses full
task completion, repeated trials and the [cross-task protocol](project_direction.md#what-qualifies-an-improvement).

## 3. Hold-out rules

**The pool.** *Home*: Straw Bale — training data and the regression test, never a generalisation
number. *Dev*: Pine Valley (`pine1`, `pine2`) and one or two more — collect, label, train and tune
freely. *Sealed*: at least four maps, sealed simply by never flying them. Sealing costs nothing
today and cannot be bought back later.

- **R1.** A track leaves the sealed pool permanently the first time a by-sight flight touches it,
  including a crash 5 s in. No "that one didn't count".
- **R2.** On a sealed track the command line is a template diff — map, log path, dataset path,
  `--seconds`. **No course parameter may appear on it.** If you had to type one, the run is tuned,
  not zero-shot.
- **R3.** Freeze and identify the full stack before flight: brain, detector,
  optional predictor, pilot mode/parameters, calibration and declared runtime
  cues. Hash checkpoints and record the code revision in the flight card.
- **R4.** Three attempts per sealed track per session; all three reported. Attempt 1 is the
  zero-shot number. **No parameter change between attempts.**
- **R5.** Any change made *because of* what a sealed track showed converts that track to dev. The
  next headline number needs a fresh sealed track — which is why you seal several.
- **R6.** One flight card per flight, written *before* it: command, checkpoint hashes, rung,
  predicted outcome, pass criterion; then the HUD count, splits and verdict. A prediction written
  after the flight does not exist. Cards live in `docs/flight_cards/` (version controlled - a record
  that is not kept is not a record).

## 4. Which parameters may be touched

| Class | Meaning | Rule |
|---|---|---|
| **V — vehicle / brain** | Transfers with the drone and the fixed brain: `yaw_rate`, `kappa_max`, `a_lat`, `goal_max`, `flow_ref` | Set once per drone. Never per track |
| **S — setup** | Per capture rig and measurable: camera file, capture rect, `--vision-fps` | *Verified* per session with `vision calibrate`, not chosen. A wrong camera file is a silent 1.7x range error |
| **D — detector** | Belongs to the checkpoint: `RANGE_CORR`, the noise model, `p_min` | Ships with the `.pt`, refitted whenever the detector changes, frozen before a sealed flight |
| **C — course** | Gate width, gate spacing, gate height profile, turn direction | **Zero hand-set values on a sealed track.** Each of these is a bug until it estimates itself |

**The D rule is not met by what ships.** `sightpilot.RANGE_CORR` was fitted to gatenet8, and v0.5.0
flies gatenet9 on it. The five clean laps were flown with exactly that pairing, so it is the
measured-good combination - but it is an inherited table, not a refitted one, and `vision rangefit`
gives gatenet9's own (it reads 0.90 at 13 m and 0.89 at 18 m where the shipped table reads 0.99 and
1.08). Refitting it is a flight-tested change, not a free one.

The course-knowledge register — every place a Straw Bale fact is currently baked in, and what has
to replace it — is the odd-course bench (`haltere vision oddcourse`), one synthetic course per
assumption, rather than a list here, because it shrinks as the work lands.

## 5. Training-side hold-out

Two things made the detector's own numbers flattering, both fixed in `vision train`:

- The validation split was a random slice of frames pooled across flights. Frames 130 ms apart are
  the same picture, so a held-out frame was in the training set in all but name; the reported 97%
  could not fall. `--holdout DATASET...` validates on whole datasets instead — and once there is
  more than one course, on a whole course.
- The augmentation was brightness and contrast only, which leaves a track's palette intact and lets
  the network key on it. `--augment strong` varies hue over the whole circle, saturation, gamma,
  sharpness, noise, bank angle and apparent size.

Checkpoints trained since this change record what they trained on, and `vision eval` marks their
lines `TRAINED ON` or `held out`. A checkpoint without that record - including the gatenet9 that
ships as `artifacts/gatenet_best.pt`, which predates it - is marked `provenance unknown`, and its
numbers still have to be checked by hand.

## 6. What the first round measured (2026-09-16)

`gatenet10`: 43,263 frames (gatenet9 saw 41,842), `--augment strong`, `--holdout run29 run30 run31`,
16 epochs from scratch. The augmentation did exactly what it was aimed at, and it was not enough.

| on held-out home flights | gatenet9 | gatenet10 |
|---|---|---|
| identity | 85.6% recall, 2.9 px | 78.7%, 4.6 px |
| grayscale | 43.1% | 79.9% |
| hue+180 | 51.1%, 40.6 px | 79.2%, 4.4 px |
| spread over all eight | 43-86% | 77-80% |

Colour dependence is gone, for about seven points of in-domain recall. **And on Pine Valley it still
sees nothing: 2.7% of frames fire (gatenet9: 3.0%), median confidence 0.093, against its own 8.7%
false-positive rate at home.** Its nine most confident Pine Valley frames are three pictures of the
game's countdown ring - a white circle with a number in it - and six crashes into foliage. It has
learned "white round thing" well enough to fire on a HUD overlay, and Pine Valley's dark truss
arches are not white round things.

The conclusion is not subtle: **colour invariance was necessary and is not sufficient.** A detector
that has seen one gate type cannot recognise another, however the pixels are jittered. The next
round needs a second gate *type* in the training set, not more augmentation.

The hold-out tagging also shows what the old split was hiding, on the same checkpoint:
`run29 [held out]` 88.5% visibility accuracy against `run10 [TRAINED ON]` 98.2%.

Range table refitted for gatenet10 (`vision rangefit`, held-out flights); the 35-60 m bin has 32
samples and is not trustworthy, so it keeps the old 45 m anchor:
`((2.8, 0.665), (6.0, 0.79), (9.2, 0.874), (13.0, 0.937), (18.0, 0.986), (24.1, 1.021), (30.3, 1.058), (45.0, 1.0))`

**Which detector flies today: still gatenet9.** gatenet10 is better everywhere except the one course
we can currently fly, and neither can see Pine Valley. Swapping now would trade 7 points of recall
and 1.7 px of bearing for nothing that can be flown yet.
