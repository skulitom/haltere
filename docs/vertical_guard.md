# Obstacle stack: vertical guard (round 3)

**Status: not flown.** Only open-loop replays of logged flights and unit tests exist.
The guard is part of the obstacle stack (`--obstacle-stack on|shadow`; `--vertical-guard off`
removes it). `shadow` computes and logs it without applying it. Nothing changes without the
stack. The declaration is `configs/obstacles/vertical_guard.json` **version 2**. Version 1
is kept and refused.

It answers three findings of round 2 (see the
[M2 flight card](flight_cards/2026-09-26_obstacles_m2.md) and
[obstacle_gap_pilot.md](obstacle_gap_pilot.md#wall-pilot-rules-round-2)):

- **Minus Two.** The looming governor's fixed 3.5 m/s terrain climb drove three drones
  into the ~2.2 m garage ceiling: brain-08 (`minus-brain08-gapon-01/02`) and the fast PD
  (`minus-fast6-wall-01`). The PD was sinking at 0.3-0.37 m/s from 0.7 m with a
  below-path TTC of 0.24 s, and the climb held for about 0.8 s.
- **Straw Bale downhill.** In both fast-brain-08 finishes the drone followed the pilot's
  line to a ring below a convex hill and skimmed the straw (4 and 6 support climbs at
  x ~ -37, y 145-168). The user asked for fewer ground touches: on downhills, keep speed
  and height instead of sinking into the hill.
- **Pine Valley.** The mound climb (`pine-fast6-ttc-01`) must keep working.

## The rules (version 2)

All in `haltere/liftoff/fast_race_cue.py` (`VerticalGuardConfig`, `TtcClearanceGovernor`,
`FastRaceCue`). The guard is **scale-free**:

- It reads only the looming samples the governor already receives: `below_fraction`,
  `ttc_lower` (the time to contact with the surface fitted below the flight path) and the
  alarm `ttc`.
- It also reads the measured vertical speed.
- It uses no height above ground and no metric distance. The only height is a
  telemetry difference, the episode bound, as in the TTC policy.

Descending means measured vz < -0.3 m/s, climbing means vz > +0.3 m/s, and level is in
between.

1. **Sink margin.** The pilot's own requested sink is scaled by
   `clip((crossing - 0.6) / (1.5 - 0.6), 0, 1)`. That is all of the sink at a crossing TTC
   of 1.5 s or more, and none at 0.6 s or less.
   - The crossing TTC is `max(alarm ttc, ttc_lower)` of the latest sample that has a
     `ttc_lower`, aged by the time since its capture and kept for 1 s. Both TTCs must be
     short: the path itself must head into the surface below it.
   - Over flat ground both TTCs of a descending path equal height / sink rate, so the
     allowed sink shrinks as the ground approaches.
   - The factor falls at up to 4 /s and recovers at up to 1 /s. The command keeps the
     pilot's 5 m/s² slew, so there is no jump. A requested climb is never scaled.
2. **Descent first.** A below-path alarm (`below_fraction >= 0.7`, crossing TTC < 1.2 s)
   received while descending holds the vertical request at least level
   (`max(pilot, 0)`) for 0.5 s, brought there at up to 10 m/s².
   - No alarm received while descending starts a climb.
   - While the drone descends, no terrain climb is applied at all.
3. **Terrain climb only for rising ground.**
   - A climb needs two below-path alarms (`ttc_lower < 1.2 s`) received while level or
     climbing within 0.3 s.
   - Its rate is graded by urgency on `ttc_lower`: 0 at 1.2 s, the full 3.5 m/s at
     0.6 s.
   - It is bounded to **1 m/s and 1 m** above where the episode began until rising ground
     is confirmed. Confirmation needs two alarms within 0.5 s, received while the drone
     already climbs faster than 0.5 m/s *and* the guard's own climb is the binding
     vertical request (above the pilot's own request).
   - After confirmation the graded rate applies, up to 3.5 m/s and the policy's 2.5 m
     bound.
   - The hold (0.5 s) and release (3 m/s²) are the TTC policy's. The ceiling guard
     (wall-pilot v3) still cuts climbs under overhead evidence.
4. **Keep speed.** The descent-path governor's shortfall is not fed in any of these
   cases:
   - the guard withholds part of the pilot's sink;
   - it arrests a descent;
   - it withholds a climb because the drone descends;
   - the pilot's support timers run (contact: the requested sink is not achieved while
     resting on the terrain).

   So the horizontal request is not cut for a sink that was withheld or that the terrain
   prevents. The guard itself never changes the horizontal request.

No other rule of the fast pilot climbs faster than about 1 m/s without the pilot's own
target: the support climb (1 m/s), the search climb (0.5 m/s) and the ceiling guard's weak
climbs (1 m/s). The launch and a ring clamped at the top edge are the pilot's own.

## Flag, logs and sidecar

| Flag | Default | Effect |
|---|---|---|
| `--vertical-guard on\|off` | on inside the stack | Component override. `on` is refused without `--obstacle-stack`. In `shadow` the flown governor is unguarded and a guarded copy fed the same samples logs the request the guard would make. |

CSV columns appended at the end of each row. They are NaN without the guard or before the
first looming sample:

- `vertical_pilot`: the pilot's own vertical request;
- `vertical_target`: the guard's request, applied or, in shadow, intended;
- `vertical_factor`: the sink factor;
- `vertical_arrest`: 1 while descent first holds the request level;
- `vertical_stage`: 0 no climb, 1 gentle, 2 rising ground confirmed;
- `vertical_climb`: the guard's climb request.

The sidecar records these under `pilot_assistance`:

- `vertical_guard_declaration`: path, content and file sha256, version and `applied`;
- `vertical_guard`: the rule, parameters and counts, and the seconds spent limiting,
  arresting and climbing.

`obstacle_stack.components.vertical_guard` says whether the guard was part of the stack.

## Declarations

| File | Version | sha256 (content) | Frozen |
|---|---|---|---|
| `configs/obstacles/vertical_guard.json` | 2 | `e06b690d0d4f...` | after the replay of v1, before any replay of v2 (commit `61f10f4`) |
| `configs/obstacles/vertical_guard_v1.json` | 1 | `703f60e33aa0...` | before any replay of the guard (commit `1861e8e`); kept verbatim, refused |
| `configs/obstacles/vertical_guard_gates.json` | 2 | `53926ceada10...` | v1's definitions, scoring guard v2, before its replay |
| `configs/obstacles/vertical_guard_gates_v1.json` | 1 | `977740fbc0f5...` | with guard v1, before any replay |

**Where the values come from.**

- Task values: the margin (1.5 s -> 0.6 s) and about 1 m/s without rising-ground evidence.
- Declared values reused from the existing rules:
  - 0.3 m/s is the ceiling guard's `overhead_min_rise`;
  - 1.2 s is the TTC policy's `climb_on_s`;
  - 10 m/s² is the policy's terrain climb acceleration;
  - 1 m is the ceiling guard's `weak_climb_max_m`;
  - the hold, release and 2.5 m bound are the policy's own.
- They were set after inspecting the incidents' logs and looming samples.

**Version 2 keeps every value and changes three rules** after the v1 replay:

- the crossing TTC for rules 1 and 2;
- the binding test for rising ground;
- keep speed on contact.

Every replayed flight is therefore development evidence for version 2, not held-out
evidence.

## Gates (open-loop replays)

The harness is `haltere/obstacles/vertical_replay.py`, a committed port of the m2r2 / m3-pilot
replay without the planner. Run it as a script: `python haltere/obstacles/vertical_replay.py
--tree TREE --out PREFIX --stack none|flown|on|shadow [--vertical off] [--looming-stream
NPZ] flight ...`, then `... vertical_replay.py score --out PREFIX --baseline BASEPREFIX --json
OUT`.

- **Inputs:**
  - each logged tick goes through `FastRaceCue` as the runner fed it: pose, ring cue,
    logged looming sample and gap sample;
  - the Straw Bale laps flew without looming, so they use the offline looming2 stream of
    M2 round 2, recomputed from the recorded video;
  - the recorded motion does not respond to the requests.
- **Variants:**
  - "guard" is `--stack on`: the full stack of the flight's motor contract with the
    guard;
  - "control" is `--stack on --vertical off`: the same stack without it, which is
    m2-hairpin's stack.
- **Definitions** are in the gates file and were frozen with each version.

Scores: `docs/experiments/vertical_guard_v1_scores.json` and
`docs/experiments/vertical_guard_v2_scores.json`.

| Gate | Threshold | v1 | v2 | Control (m2-hairpin stack) |
|---|---|---|---|---|
| **Identity**: stack off and shadow bit-identical to m2-hairpin `4083454` on 9 flights (20 gate pairs; 21 report pairs: stack as flown + wall rules on/shadow, stack on without the guard) | all | **pass** (41/41) | **pass** (41/41) | - |
| **V-Minus** wall-01: max requested vz, logged climb onset -1 s to impact | <= 1 m/s | **0.97** | **0.97** | 3.44 |
| V-Minus wall-01: floor sink requested >= 0 before 0.3 m height loss | yes | **yes** (level from the sink's start, 0.02 m lost) | **yes** | yes (the PD already requested level) |
| V-Minus gapon-01: max requested vz | <= 1 m/s | **0.88** | **0.88** | 0.79 |
| V-Minus gapon-02: max requested vz | <= 1 m/s | **0.98** | **0.98** | 3.46 |
| **V-Pine**: seconds of the logged climb with requested vz >= 1 m/s | >= 80% | 73.8% | 73.8% | 70.8% (the flown log itself: 72.4%) |
| V-Pine: no requested descent in the last 2 s before contact | min >= 0 | **0.0** | -0.70 | -0.72 |
| **V-Straw**: guard limits the sink before contact | >= 80% of 10 | 20% | 20% | - |
| V-Straw: request above max(pilot, 0) while descending at the spots | 0 ticks | **0** | **0** | - |
| V-Straw: horizontal request below the control's at the spots | 0 ticks | 79 (max 0.28 m/s) | **0** | - |
| V-Straw whole lap: guard terrain climb above 1 m/s | 0 ticks | 4930 / 5494 ticks | 2107 / 2126 | control climbs 3.3 / 3.0 m/s max |
| **Result** | | Identity, V-Minus pass; V-Straw, V-Pine fail | Identity, V-Minus pass; V-Straw, V-Pine fail | |

Other numbers (per lap, 5.45 min):

| Straw Bale (04 / 06) | v1 | v2 | control |
|---|---|---|---|
| sink limiting | 20.2 / 15.1 s | 7.1 / 4.2 s | - |
| descent-first arrests | 3.1 / 2.0 s | 1.1 / 0.0 s | - |
| guard climb (all / escalated) | 64 / 55 s, 68 / 62 s | 57 / 23 s, 57 / 24 s | 12.5 / 10.4 s of climb |
| sink withheld in the arch zone (x ~ -37, 90 < y < 140) | 3.4 / 1.7 m | 0 / 0.27 m | - |
| descent_scale minimum during the 10 contacts | 0.44-0.53 | 0.65-1.0 | 0.44-0.53 |

Pine Valley per climb episode (logged climb seconds answered with >= 1 m/s):

| Episode | Logged | v1 = v2 | Control |
|---|---|---|---|
| 4.3-6.0 s, the mound | 1.72 s | 1.47 s (max 3.5) | 1.27 s |
| 15.2-16.9 s, hillside | 1.71 s | 0.38 s (gentle 1.0 only) | 1.14 s |
| 17.6-20.5 s, hillside | 2.91 s | 2.83 s | 2.08 s |

## What the replays show

- **The Straw Bale contacts cannot be seen by looming.** No looming sample carried
  below-path evidence in the 3 s before any of the 10 contacts (0 of 38-50 samples each).
  In the two approaches re-run with looming2 on the recorded frames (straw-brain08-04,
  69.5-71.6 s and 285.8-287.8 s), the flight path pointed below the camera's field of
  view for the whole approach. The descent was 20-29 deg at 2.8-3.4 m/s with the camera
  28-30 deg up. The focus of expansion lay 1.05-1.38 image heights down, and every
  looming window was outside the image or without texture.
  - The two episodes counted as "limited" were limited exactly 3.0 s before contact, from
    the previous hill, not from this contact.
  - No looming-driven rule can meet V-Straw's first criterion on these laps.
  - What the guard does there is keep speed. During the contacts the control's
    descent-path governor cut the horizontal request to 44-53%; v2 keeps it at 65-100%.
- **v1 arrested the descent in front of both descending arches.**
  - The lower window read 0.8-1.3 s on the ground in front of the arch while the alarm
    read 1.8-9.9 s through the opening.
  - v1 withheld up to 1.0 m/s of the pilot's sink for about 2 s before the first arch and
    0.7 m/s for about 1 s before the second.
  - Earlier flights show the top beam only 0.3-0.5 m above clean crossings.
  - v2's crossing TTC removes this: 0 m withheld in lap set 04, 0.27 m in 06, where lap 3
    still withheld 0.3-0.4 m/s for about 1.2 s at y 123-117 near the first arch.
- **v1 escalated on every Straw Bale uphill.** There the pilot itself climbs toward rings
  up the hill. v2's binding test cuts the escalated seconds from 55-62 to 23-24 per lap,
  but v2 still escalates. These are the uphill legs (y 93-193) where the pilot's own
  request is 0.3-0.9 m/s, the gentle climb binds and the recorded (as-flown) climb of
  0.5-1 m/s meets repeated below-path alarms. Closed loop, the guard's 1 m/s climb would
  meet the same alarms, so this is expected in flight too: **escalated requests of up to
  3.5 m/s for 1.8-6.3 s on the Straw Bale uphill legs**, bounded by the policy's 2.5 m
  above the episode start (plus the release).
- **Minus Two.**
  - The ceiling climbs are gone in every window (max 0.88-0.98 m/s against 3.44-3.5 as
    flown).
  - wall-01's floor sink is answered with a level request from its start. Two below-path
    alarms while level then allow the gentle 1 m/s climb for up to 1 m, from 0.69 m: a
    request that tops out about 0.35 m under the ~2.2 m ceiling (1 m plus the release).
  - gapon-02's arch below the path still reads as terrain, but only gently (0.98 m/s),
    and the ceiling guard cuts it.
- **Pine Valley.** The mound (the first episode) is climbed as before. The escalation
  reaches 3.5 m/s, and the episode is answered on 85% of its seconds against the
  control's 74%.
  - The second episode starts while the PD sinks at 0.3 m/s. The guard arrests first,
    then climbs gently at 1 m/s for 0.4 s, where the flight climbed at 3.5 m/s. Its
    samples during the flown climb (bf 0.49-0.77, lower TTC 0.4-1.8 s) never confirm
    rising ground, so whether 1 m/s is enough there is not shown.
  - The 80% threshold is above what the flown log itself requested (72.4%). The logged
    climb seconds include its own release below 1 m/s.
  - The end of the flight was a boulder beside the line: lower TTC 0.49-0.62 s, alarm
    1.9-2.8 s. v1 held the request level for its last 0.7 s. v2's crossing rule trusts the
    long alarm and keeps the pilot's -0.7 m/s descent there. This is the price of not
    arresting in front of arch openings.

## Limits

- **Open loop.** Every number above is the request a variant would have made at the
  recorded states, not a flight. It does not show whether brain-08 follows a withheld sink
  or a gentle climb.
- **Development evidence.** Every flight here was inspected while the rules were designed.
  Version 2 was changed after the v1 replay of the same flights.
- **The Straw Bale laps use an offline looming stream,** not the camera's own samples.
- **Straw Bale contacts remain.** The camera (30 deg uptilt, 42 deg vertical half field of
  view) cannot see a descent steeper than about 12-14 deg below the horizon, and the pilot's
  bottom-clamped descents are 20-29 deg by design (to pass under the descending arches).
  Options, none taken here:
  - keep the descent within view (at the price of the arch clearance);
  - a lower camera uptilt (needs the brain's sensory contract re-trained);
  - publish looming2's upper-window TTC, which sees the ground ahead of a steep descent.
- **Uphill escalation (v2).** Expect extra climbing on Straw Bale uphills.
- **Pine end.** v2 does not arrest a descent toward a surface that only the lower window
  sees.

## Tests

- **`tests/test_fast_race_cue_vertical.py`**, 25 tests:
  - the declaration is frozen and refused when edited or of another version; v1 is kept;
  - the sink factor's margin, ramp and memory;
  - descent first and the wall-01 shape;
  - confirmation, the gentle bound and its height limit;
  - escalation, graded by urgency;
  - the binding test (a pilot climbing up a hill does not escalate);
  - the arch pattern (a long alarm limits nothing);
  - the ceiling guard still cuts;
  - unchanged governor without the guard;
  - the pilot's margin, arrest, mound climb and keep speed (withheld sink and contact);
  - shadow flies the unguarded pilot bit for bit;
  - the runner's columns, flags and refusals.
- **`tests/test_vertical_replay.py`**, 7 tests: the gates declaration and the scoring
  functions on synthetic arrays.
- **`tests/test_gap_pilot.py`** covers the new stack field.
- **The full suite passes:** 952 tests.
- **CPU wiring check (plumbing only, no pad, no flight).** `VisualController` was built as
  `run()` does for four cases:
  - brain-08 with the stack on;
  - brain-08 in shadow;
  - the fast PD with the stack on;
  - the fast PD with `--vertical-guard off`.

  Each pilot received the version 2 declaration (`applied` true, false, true and absent),
  the sidecar component and the six log columns.
