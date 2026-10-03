# Main-track race eval

**Status:** cases approved by the user on 2026-10-02 (all five races; baseline = brain-11 and the
fast PD on the m6 stack; attempt cap about 6x the user's race time). Grader approved on
2026-10-03 after the pilot below (finishes judged clean by a video contact count; tracks weighted
equally). Nothing has been flown for this eval yet; the baseline batch is
[batch_m6_baseline.json](../configs/race_eval/batch_m6_baseline.json).

## Baseline result: m6 stack, 2026-10-03 (30 of 30 attempts)

Batch `runs/race-eval/m6-baseline-20261003` (plan frozen at `ddf9df5`; raw logs, videos, finish
evidence, `results.jsonl`, `errors.jsonl`, `summary.md` and per-attempt video notes in
`observations.md` there). Flown 00:08-03:22 in the hidden Anode seat with the original
`[Copy] New Drone`, including a one-hour pause the user requested after attempt 6. Every crash
cause was checked on the video.

| Variant | Mean progress (95% CI) | Straw | Minus | Pine | Autumn | Hangar | Finishes | Clean |
|---|---|---|---|---|---|---|---|---|
| brain-11 | **0.139** [0.085, 0.190] | 0.313 | 0.088 | 0.018 | 0.247 | 0.030 | 0/15 | 0/15 |
| fast PD | **0.199** [0.117, 0.302] | 0.486 | 0.222 | 0.007 | 0.247 | 0.030 | 1/15 | 0/15 |

Paired difference (fast PD - brain-11): **+0.059, 95% CI -0.039 to +0.171**, so the batch does
not resolve a motor difference overall. Per track: Straw +0.174, Minus +0.134, Pine -0.011,
Autumn 0, Hangar 0.

- **One finish:** fast PD on Straw, game race time **4:34.661** (1:30.171, 1:31.213, 1:31.762),
  the fastest autonomous Straw finish so far (3.48x the user's time). Not clean: the telemetry
  shows two belly scrapes on the same downhill crest (laps 1 and 3), which the up-tilted camera
  cannot show.
- **Brain bests:** Straw 0.563 (27/48: lap 1 and most of lap 2), Minus 0.140 (to arch 287 by
  the garage wall), Autumn 0.296.
- **The blockers are shared by both motors, at the same places**, which points at
  perception/pilot rather than motor control (the program's step-2 reading):
  - Straw: descending onto the next arch's top banner with the ring clipped below (PD 2/3,
    identical; brain-11 r6 too); FAT SHARK arch entered obliquely (brain 2/3); hillside ground
    after the ImmersionRC arch (brain 1).
  - Minus: hairpin exit into the dark pillar at (78, 23) (one each); low arches 243/99/287
    struck while passing; the PD's one full lap ended beside arch 8 on lap 2.
  - Pine: a pine trunk on the hillside at (70.5, -8.5) (4 of 6, both motors), other trees
    (2). No run passed checkpoint 3 of 90.
  - Autumn: the red tree just before the lap line at (-9.2, -9.3) (4 of 6, both motors,
    8 of 9 lap-1 checkpoints); the tree stand before box 140 (2).
  - Hangar: the dark container stack at (15.2, -64.3) right after checkpoint 1 (6 of 6).
- **The fast PD is nearly deterministic** (attempts 3/6, 14/15, 19-24, 26/27/30 crash at the same
  point and time), so its repetitions add little information; brain-11 varies more.
- **Runtime:** two runner preflight refusals (attempts 16 and 25) from other agents' jobs on the
  desktop (an eval batch, a test suite, .NET builds); nothing flew, both retried unchanged.
- **Grader note for the next version:** in attempt 18 the PD overflew the start arch, came back
  through it backwards, and Liftoff started the race timer on that reverse pass. The approved rule
  (start from the race side only) scored it 0/90; a reverse-allowed start would give 1/90.

## Running a batch

```powershell
# once per game version: the race XML, checked against the case file's hashes
.venv/Scripts/python.exe -m haltere.liftoff.race_eval export-geometry --out runs/main-race-geometry-<date>
# freeze identity, order and the 30 commands (refuses uncommitted runtime changes and an existing plan)
.venv/Scripts/python.exe -m haltere.liftoff.race_eval plan configs/race_eval/batch_m6_baseline.json --out runs/race-eval/m6-baseline-<date>
# after each attempt (and its video check): grade it into results.jsonl, or errors.jsonl for a runtime failure
.venv/Scripts/python.exe -m haltere.liftoff.race_eval record runs/race-eval/m6-baseline-<date> LOG.csv --geometry runs/main-race-geometry-<date>
.venv/Scripts/python.exe -m haltere.liftoff.race_eval summary runs/race-eval/m6-baseline-<date>
```

Fly `plan.json`'s attempts in order with the round-6 pre-launch checks before each. `record`
refuses an attempt flown with another checkpoint or after a runtime change, a duplicate, an
operator pause without a review note, and a finish without a video contact count. Evidence files
next to the log: `<stem>-finish.json` (the results screen: `game_confirmed_finish`, `race_time`,
`lap_times`, `flight_interventions`) and `<stem>-review.json` (`contacts_on_video`,
`interventions`, and `end_cause` when the sidecar cannot tell, e.g. `operator_stop`). A runtime
error is retried once, unchanged, as `<stem>-retry1.csv`.

Purpose: a standing, matched comparison of **stack variants** (a new brain, a pilot rule on/off,
brain vs fast PD) on the user's five acceptance races. All five tracks are **development data**
(each has been flown), so this eval measures progress on known courses, not transfer. Transfer
needs the separate held-out batch in [generalization_program.md](generalization_program.md).

Case file: [configs/race_eval/main_tracks_v1.json](../configs/race_eval/main_tracks_v1.json).
Scorer: [haltere/liftoff/race_eval.py](../haltere/liftoff/race_eval.py), tests in
`tests/test_race_eval.py`.

## Cases

One case = one race from the game's standard spawn, the declared task "fly the race, three laps,
following the game's visible race cues", the original `[Copy] New Drone`, and one frozen stack.
Targets come from [main_track_targets.json](../configs/main_track_targets.json): evaluation
metadata only, never a runtime input.

| id | Environment / race | Checkpoints (3 laps) | Your race time | 120% limit | `--seconds` | `--max-distance` |
|---|---|---|---|---|---|---|
| `straw` | Straw Bale / 01 - Field Day | 48 | 79.0 s | 94.8 s | 480 | 3200 m |
| `pine` | Pine Valley / 01 - Forest For The Trees | 90 | 127.0 s | 152.5 s | 760 | 5000 m |
| `minus` | Minus Two / 01 - Turn Signals | 57 | 89.3 s | 107.1 s | 540 | 2000 m |
| `autumn` | Autumn Fields / 01 - Walk In The Park | 27 | 64.5 s | 77.4 s | 390 | 1900 m |
| `hangar` | Hangar C03 / 01 - Shipments | 33 | 84.1 s | 100.9 s | 500 | 2200 m |

`--seconds` is about 6x your race time; a finish slower than the 120% limit still counts as a
finish, but not as a pass. `--max-distance` is about twice the race length between checkpoints.
**Round 6 flew with `--max-distance 2000`, which would stop a perfect Pine attempt before its
finish** (the checkpoints alone are 2461 m apart over three laps).

**Repetitions.** 3 attempts per case per variant (15 per variant), matching the 3/3 requirement.
Two variants fly each track in A-B-B-A-A-B order on the same setup, with no changes between them.

**Attempt boundaries.** An attempt starts at Réinitialiser plus the standard pre-launch checks
(quiet machine, `seatOnly` pad, ground check, telemetry `LIVE`). It ends at the first of: the
results screen (finish), an impact that ends flight, a declared operator stop (parked over 3 s,
runaway climb, search over 30 s), or a runner limit.

**Race geometry** is the game's own race and track XML, copied read-only out of the installed
bundles (`python -m haltere.liftoff.race_eval export-geometry --out runs/main-race-geometry-<date>`).
The XML stays out of the repository; the case file pins its SHA-256 and the scorer refuses a
mismatch. It is used only to score a flight afterwards.

## Grader

| Metric | Kind | Definition |
|---|---|---|
| `progress` | primary, 0-1 | Checkpoints passed in race order, from the correct side and through the opening, divided by the race's checkpoints after the start |
| `finish` | binary | Game-confirmed: the results screen, recorded as `<log>-finish.json`. Telemetry alone never confirms a finish |
| `clean_finish` | binary | `finish`, with zero contacts counted on the video, no reset and no intervention. Pending (blank) until the video is reviewed |
| `within_target` | binary | `clean_finish` and the game's race time is within the 120% limit |
| `time_ratio` | finishes only | Game race time divided by your race time |
| `end_cause` | class | From the sidecar: finish, impact, limit, runtime error; an operator pause needs a review note |

**How a checkpoint is credited.** Only the next checkpoint in race order counts. The flight
must cross its plane in the race direction and pass through the opening (plus 0.25 m for
interpolation). Checkpoint boxes are sized from the game file. Arch openings are approximate:
3.0 x 2.4 m for Straw's and Pine's large arches (asset name `240H300B`; a recorded top-banner
impact at 2.39 m) and 1.8 x 1.3 m for Minus arches (from recorded passes and a leg strike).
Passes within 0.3 m of an edge are listed for a video check. No crossing is credited across a
telemetry gap over 0.25 s, and scoring stops at a reset.

**Runtime failures are not scored.** A controller deadline, camera failure or zero-tick start is
an error with one unchanged retry, never a navigation failure.

**Reported, not graded:** contact-audit count (its video false-positive check still fails),
edge crossings, time to the end, the geometric race time (a cross-check on the game's).

**Headline per variant.** Mean `progress` over the five tracks (each track weighted equally),
with a per-track breakdown, then finishes/attempts and clean finishes/attempts. Variant
comparisons use the per-track paired difference in mean `progress`, with a bootstrap interval
over attempts.

**Resolution.** From the spread of the recorded attempts below (Straw progress varies by about
0.12 between attempts of one stack, Minus by about 0.06), the five-track mean-progress
difference between two variants should be resolvable to about +-0.05: two or three Straw
checkpoints. A single track's difference is much coarser (about +-0.2 on Straw). A finish rate
over 15 attempts is good only to about +-25 points.

## Grader pilot on recorded flights (2026-10-02; no new flights)

| Log | What the card says (video-checked) | End cause | Progress | Checkpoints | Finish | Clean | Time ratio | Geometric race time |
|---|---|---|---|---|---|---|---|---|
| `straw-brain08-04` | Finish 5:17.898 | finish | 1.0 | 48/48 | yes | pending | 4.02 | 317.959 s |
| `straw-brain08-06` | Finish 5:17.805 | finish | 1.0 | 48/48 | yes | pending | 4.02 | 317.874 s |
| `straw-cue-04` | Finish 14:05.703 (09-22 stack) | finish | 1.0 | 48/48 | yes | pending | 10.70 | 846.425 s |
| `minus-cue-01` | Finish 9:27.415 (09-22 stack) | finish | 1.0 | 57/57 | yes | pending | 6.36 | 567.799 s |
| `straw-brain11cw13-r4b-noassist-02` | Lap 1 in 1:42.988; start arch 8 s into lap 2 | impact | 0.333 | 16/48 | no | no | - | lap 1: 102.939 s |
| `straw-brain11cw13-r5-noassist-01` | FAT SHARK arch leg at 19.4 s | impact | 0.042 | 2/48 | no | no | - | - |
| `straw-brain11cw13-r5-noassist-04` | Next arch's top bar at 81.9 s | impact | 0.208 | 10/48 | no | no | - | - |
| `straw-brain11cw13-r6-01` | Next arch's top banner at 87.3 s | impact | 0.229 | 11/48 | no | no | - | - |
| `minus-fast6-r4b-01` | Arch leg beside a ring, 44.8 s | impact | 0.158 | 9/57 | no | no | - | - |
| `minus-fast6-r5-01` | Hairpin stop, then a pillar at 26.3 s | impact | 0.053 | 3/57 | no | no | - | - |
| `minus-brain11cw13-r5-02` | Assist v3 crawl into the first arch | impact | 0.0 | 0/57 | no | no | - | - |
| `minus-fast6-r6-01` | Arch ring bump at 41.4 s | impact | 0.140 | 8/57 | no | no | - | - |
| `minus-brain11cw13-r6-01` | Through the hairpin; floor arch leg at 29.6 s | impact | 0.070 | 4/57 | no | no | - | - |
| `pine-fast6-r6-01` | Hillside at 14.8 s | impact | 0.011 | 1/90 | no | no | - | - |
| `hangar-01`, `hangar-02` | Roof impact at 30.5 s (09-22 scene brain) | impact | 0.0 | 0/33 | no | no | - | - |
| `straw-brain08-05` | Controller deadline in lap 2 | runtime error | - | - | - | - | - | - |
| `pine-brain05-01` | Camera measurements unavailable | runtime error | - | - | - | - | - | - |

Checks through the same scorer: every game-confirmed finish scores 1.0, and its geometric race
time is within 0.07% of the game's (clock drift). Pine's built-in bot lap scores 90/90 in
137.916 s against its reported 137.915 s. A drone parked at the spawn scores 0. Both runtime
failures are errors, not zeros. The r4b lap 1 ends 0.049 s from the game's lap time.

Known limits seen in the pilot:

- **A crash inside a gate is not credited for that gate.** `minus-brain11cw13-r6-01` hit the
  leg of arch 243 0.15 m before its plane; the game may already have switched its marker. The
  geometric score can lag the game's count by one checkpoint at the crash.
- **Two edge passes:** Straw arch 13 at 2.19 m (the top banner that ended r6 is at 2.4 m) and
  a Minus arch at 1.03 m. They stay credited and are listed for the video check.
- **Finishes stay `clean: pending`** until someone counts contacts on the video. brain-08's
  finishes had contact-support firings, so they cannot be called clean from telemetry.
- Autumn Fields has no scorable log yet (its 09-20 flights used the older `liftoff fly` format).
  Its checkpoint boxes are the same types that score Pine's bot lap 90/90.
