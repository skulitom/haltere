# Flying a race the drone has never seen

The by-sight stack flies clean 7/7 laps on Straw Bale "Field Day". That number says nothing about
any other course, and the first flight on an unseen map showed why: over the 1392 Pine Valley frames
the detector fired on 3.0%, median confidence 0.010, with arches plainly in view - against its own
8.7% false-positive rate on gate-less frames at home. Detection on an unseen course was worse than
its own noise floor, and the pilot never got a gate to fly at.

The same detector loses half its skill when the colour goes: on held-out home flights, recall
85.6% -> 43.1% in grayscale and -> 51.1% at hue+180, where the centre error goes 2.9 -> 40.6 px and
false positives 8.7% -> 50.2%. Brightness, gamma and blur barely touch it. What it learned is a
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

The headline claim — "completes an unseen race in an unseen environment" — needs **Z4 on three
sealed maps, each on the first attempt**, with at least one full finish and ≥80% of checkpoints on
the others.

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
- **R3.** The detector checkpoint, its range table and noise model are frozen and hashed into the
  flight card before the flight.
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

The course-knowledge register — every place a Straw Bale fact is currently baked in, and what has
to replace it — is tracked in the issue list rather than here, because it shrinks as the work lands.

## 5. Training-side hold-out

Two things made the detector's own numbers flattering, both fixed in `vision train`:

- The validation split was a random slice of frames pooled across flights. Frames 130 ms apart are
  the same picture, so a held-out frame was in the training set in all but name; the reported 97%
  could not fall. `--holdout DATASET...` validates on whole datasets instead — and once there is
  more than one course, on a whole course.
- The augmentation was brightness and contrast only, which leaves a track's palette intact and lets
  the network key on it. `--augment strong` varies hue over the whole circle, saturation, gamma,
  sharpness, noise, bank angle and apparent size.

Checkpoints now record what they trained on, and `vision eval` marks every line `TRAINED ON` or
`held out`, so an in-domain number can no longer be quoted as a generalisation number by accident.
