# Obstacle stack: vertical guard (rounds 3-4)

**Status: version 3 is not flown.** Only open-loop replays of logged flights, an idealised
closed-loop check and unit tests exist for it. Version 2 flew three Minus Two attempts in
round 3 (`minus-fast6-vg-02`, `minus-brain08-vg-01`, `minus-brain09b-vg-01`: no ceiling
climb). The guard is part of the obstacle stack (`--obstacle-stack on|shadow`;
`--vertical-guard off` removes it). `shadow` computes and logs it without applying it.
Nothing changes without the stack. The declaration is
`configs/obstacles/vertical_guard.json` **version 3**. Versions 1 and 2 are kept and
refused.

![Guard v2 and v3 on a Straw Bale lap and on Pine Valley (open-loop replays)](vertical_guard_v3.png)

*Open-loop replays (development evidence, not flights). Top: the first uphill legs of
`straw-brain08-04`, which flew without looming and without contact there. Version 2
escalated three times, to 3.0-3.5 m/s, while the pilot itself climbed toward rings up the
hill; version 3 follows the pilot. Bottom: `pine-fast6-ttc-01`. Both versions climb the
mound at up to 3.5 m/s; on the hillside at 15.2 s both answer the flown 3.5 m/s climb with
only 1 m/s.*

**Round 4 in short.** Version 3 removes the escalated climbs on the Straw Bale uphills: 0 s
on nine laps, against 108 s for version 2. It keeps the Minus Two ceiling fix (at most
0.98 m/s) and the Pine mound climb (106% of the flown height request). It still fails two
gates:

- V-Straw downhill: looming cannot see the downhill contacts.
- V-Pine: the hillside at 15.2 s gets 1 m/s, and the end of the flight is a descent.

The idealised check still gives too little clearance when the drone sinks into a steep
ramp. See [Round 4](#round-4-version-3) and [Limits](#limits-and-risks).

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

## The rules (version 3)

Rules 1, 2 and 4 are version 2's. Version 3 changes two parts of rule 3 (marked **v3**);
no value changes.

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
   - A climb needs two below-path alarms (`below_fraction >= 0.7`, `ttc_lower < 1.2 s`)
     received while level or climbing within 0.3 s.
   - **v3:** a climb that is not already running starts only if at least one of them is a
     *path alarm*: its crossing TTC `max(alarm ttc, ttc_lower)` is also under 1.2 s, so the
     flight path itself heads into the surface below it (the crossing TTC of rules 1 and
     2). A short `ttc_lower` with a long alarm (the lower window sees the slope below a
     crest, or below the pilot's line up a hill, while the focus of expansion sees over it)
     starts no climb. It only sustains a running one. The sidecar counts such alarm pairs
     as `no_path_onsets`.
   - Its rate is graded by urgency on `ttc_lower`: 0 at 1.2 s, the full 3.5 m/s at
     0.6 s.
   - It is bounded to **1 m/s and 1 m** above where the episode began until rising ground
     is confirmed. Confirmation needs two alarms within 0.5 s, received while the drone
     already climbs faster than 0.5 m/s *and* the guard's own climb binds.
   - **v3:** the guard's climb binds when it exceeds *every* vertical request the pilot
     made itself in the last 0.5 s by at least 0.5 m/s (`rising_window_s`,
     `rising_min_rise`). The drone then climbs at least 0.5 m/s faster because of the
     guard, and the ground keeps looming. A pilot that follows a ring up a hill climbs by
     its own request, and a one-tick dip of that request at a checkpoint switch does not
     make the guard's climb the reason. Version 2 compared the guard's climb only with the
     pilot's request of the same tick.
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
| `configs/obstacles/vertical_guard.json` | 3 | `b70e263ccdc5...` | after the replays of v2 (21 logs) and its round-3 flights, before any replay of v3 (commit `79895b7`) |
| `configs/obstacles/vertical_guard_v2.json` | 2 | `e06b690d0d4f...` | after the replay of v1, before any replay of v2 (commit `61f10f4`); kept verbatim, refused |
| `configs/obstacles/vertical_guard_v1.json` | 1 | `703f60e33aa0...` | before any replay of the guard (commit `1861e8e`); kept verbatim, refused |
| `configs/obstacles/vertical_guard_gates.json` | 3 | `689635881467...` | with guard v3, before its replay (commit `79895b7`) |
| `configs/obstacles/vertical_guard_gates_v2.json` | 2 | `53926ceada10...` | v1's definitions, scoring guard v2, before its replay; kept verbatim |
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

**Version 3 keeps every value and changes two parts of rule 3** after the replays of
version 2 (see [Round 4](#round-4-version-3)). Both changes reuse declared values
(`climb_on_s`, `rising_window_s`, `rising_min_rise`). Every replayed flight is development
evidence for version 3.

## Round 4: version 3

### Diagnosis (version 2, open-loop replays of 21 logs)

Round 4 replayed version 2 on every log named below before version 3 was written. Seven
more Straw Bale laps and `pine-brain08-01`, which flew without looming, got offline looming2
streams recomputed from their videos, as in M2 round 2 (see the gates' `inputs`).

**Straw Bale uphills: the climbs were not needed, and version 2 started and escalated them
on the lower window alone.**

| Straw Bale lap (offline stream) | min | v2 climb s | v2 escalated s | v2 escalations | v2 extra height asked | v3 climb s | v3 escalated s | v3 extra height asked |
|---|---|---|---|---|---|---|---|---|
| straw-brain08-04 (finish) | 5.45 | 57.0 | 23.2 | 6 | 43.6 m | 9.6 | 0 | 1.2 m |
| straw-brain08-06 (finish) | 5.45 | 56.7 | 23.8 | 6 | 44.5 m | 11.2 | 0 | 1.1 m |
| straw-brain08-01 | 1.27 | 15.5 | 8.6 | 2 | 13.3 m | 4.6 | 0 | 0.6 m |
| straw-brain08-02 | 2.74 | 38.8 | 15.2 | 4 | 30.3 m | 7.4 | 0 | 0.7 m |
| straw-brain08-03 | 0.99 | 18.7 | 8.1 | 2 | 13.8 m | 4.1 | 0 | 0.6 m |
| straw-fast6-01 | 1.04 | 11.4 | 0.8 | 1 | 0.9 m | 4.4 | 0 | 0 |
| straw-fast6-02 | 5.20 | 36.7 | 10.2 | 5 | 12.7 m | 13.4 | 0 | 0.1 m |
| straw-fast6-03 | 5.19 | 34.7 | 9.5 | 4 | 16.1 m | 14.4 | 0 | 0.4 m |
| straw-fast6-arc-01 | 4.60 | 34.4 | 9.0 | 4 | 16.0 m | 19.3 | 0 | 1.6 m |
| **all 9 laps** | **31.9** | **304** | **108** | **34** | **191 m** | **88** | **0** | **6.4 m** |

"Extra height asked" integrates the guard's request above the pilot's own request while the
guard climbs. The laps flew without looming, so none of them had a terrain climb.

- No lap touched the ground on the uphill legs. The support climbs of the brain-08 laps are
  all at the downhill spots (x ~ -37, y 144-166).
- Five laps ended in an impact:
  - three on the downhill (x ~ -36, y 121-175);
  - two brain-08 laps (02, 03) at the hilltop (x 0-5, y 193-198), at 1-3 m/s horizontal
    during a turn. Those samples carried no below-path evidence (`below_fraction` unknown,
    alarm 0.9-1.2 s).

- **Onsets.** v2 started 208 climbs on these laps. Only 6 of them had a sample with both
  raw TTCs under 1.2 s in the 0.3 s before. The others followed samples whose lower TTC
  read 0.2-1.2 s while the alarm read 1.3-10 s or had no evidence. These are the crests
  (y 150-194) and the climb out of the valley (x ~ 38, y 94 on the brain-08 laps; y 72 on
  the PD laps). The lower window sees the slope below the pilot's line. Its fitted plane
  crosses the extended flight path, but the path passes over the crest.
- **Escalations.** All 34 happened while the pilot itself climbed toward a ring up the
  hill. Its own request reached at least 0.61 m/s in the 0.5 s before each escalation:
  - 0.85-1.05 m/s at the valley exits (the guard's 1 m/s exceeded it by at most 0.15 m/s);
  - one-tick dips to 0-0.25 m/s after 0.9-1.05 m/s at checkpoint switches;
  - 0.6-1.5 m/s on the crests.

  Version 2's binding test compared the guard's climb with the pilot's request of the same
  tick only, so these counted as "the drone climbs because of the guard".
- **Pine and Minus.** The three escalations on Pine Valley had the pilot's own request at
  most 0.44 m/s (the mound, 4.55 s), 0.13 m/s (the hillside, 18.4 s) and 0.39 m/s
  (`pine-brain08-01`, 17.0 s) in the 0.5 s before. On Minus Two, 2 of version 2's 8 climb
  onsets had a path sample: the floor sink of `minus-fast6-wall-01` and the arch of
  `minus-brain08-gapon-02`. The other six followed the lower window alone and stayed gentle
  (the floor or a wall foot ahead of a level path).

So the two tests that version 3 changes separate the groups in these logs. The margin is
small: the mound's pilot request (0.44 m/s) is 0.06 m/s under the 0.5 m/s the binding test
allows, and the lowest Straw Bale escalation (0.61 m/s) is 0.11 m/s over it.

**V-Pine was ill-posed.** Version 2's gate asked for a request of at least 1 m/s during 80%
of the seconds in which the logged governor climbed. Those seconds include:

- the governor's release tail (3 m/s² down from its hold);
- graded climbs below 1 m/s.

The flown pilot itself asked for less than 1 m/s in those seconds. The flown log therefore
reached 72.4%, under its own gate. Counting only the seconds in which the logged governor
itself asked for at least 1 m/s, the flown log answers 91.4% (of 5.02 s).

Version 2's real shortfall on Pine is the hillside at 15.2 s:

- The PD sank at 0.32-0.34 m/s while accelerating, although the pilot asked +0.3 m/s. A path
  alarm (alarm 0.85 s, lower 0.34 s) arrived while it sank, so descent first arrested and
  started no climb.
- A gentle climb began at 15.45 s.
- The recorded motion then climbed at the flown 3.5 m/s. The next samples read
  `below_fraction` 0.63-0.69, just under 0.7, so rising ground was never confirmed.

An open-loop replay cannot show whether a staged climb would have met more evidence there:
the recorded climb removes it.

### Gates version 3

`configs/obstacles/vertical_guard_gates.json` version 3 was frozen with guard v3, before any
replay of v3. Every version-2 definition is kept in `old_definitions` and scored on the same
replays. Where a v2 gate is redefined, `change` gives the reason:

- **Identity**:
  - baseline: m2-vertical (`935cfdb`, guard v2);
  - 21 logs: stack off (with and without the offline stream) and shadow must be bitwise
    identical.
- **V-Straw downhill**: v2's first three V-Straw criteria, unchanged. The first one (limit
  the sink before contact) is out of reach of looming, because the path points below the
  camera's view. It is expected to fail until the pilot keeps its descents in view, which is
  another change of round 4.
- **V-Straw uphill**: escalated seconds pooled over all nine laps, at most 0.2 s per minute
  (about 1 s per lap: "near zero"). This replaces v2's whole-lap criterion (zero ticks of
  guard climb above 1 m/s, on two laps). It is a disclosed loosening. The old criterion is
  still scored below.
- **V-Minus**: v2's windows and floor-sink criterion, unchanged, plus *no escalation*. The
  guard's climb must stay at most 1 m/s at every tick of all nine Minus Two logs.
- **V-Pine** (`pine-fast6-ttc-01`):
  - climb: at least 1 m/s during at least 80% of the seconds in which the logged governor
    itself asked for at least 1 m/s (min_vz and 80% unchanged; the flown log: 91.4%);
  - no descent in the last 2 s (unchanged);
  - the mound (new): the first episode, whose evidence precedes any climb, so the open-loop
    replay is fair up to its onset. The issued height request must reach at least 90% of
    the flown one (3.42 m).

The replays use the round-3 harness (`haltere/obstacles/vertical_replay.py`, see below), with
five variants per log:

- `--stack none`, with and without the stream;
- `--stack shadow`;
- `--stack on`;
- `--stack on --vertical off`;
- `--stack flown --wall on|shadow`.

Each tree was a `git archive` of its commit: m2-vertical `935cfdb` and v3's freeze commit
`79895b7`. `vertical_replay.py score` recognises gates version 3. The offline streams and
their alignment reports are named, with sha256, in the gates' `inputs`. Like round 3's
streams, the files stay in the session scratchpad, not in the repository.

Scores: `docs/experiments/vertical_guard_v3_scores.json`, and version 2 on the same replays
under the same gates: `docs/experiments/vertical_guard_v2_under_gates_v3_scores.json`.

| Gate (v3) | Threshold | **v3** | v2 (same replays) | Control (m2-hairpin stack) |
|---|---|---|---|---|
| **Identity**: stack off and shadow bitwise identical to m2-vertical, 21 logs | all | **pass** 52/52 pairs (report pairs 43/43: the stack without the guard, flown with wall rules on/shadow) | - | - |
| **V-Straw downhill**: sink limited before contact | >= 80% of 10 | 20% | 20% | - |
| V-Straw downhill: request above max(pilot, 0) while descending | 0 ticks | **0** | 0 | - |
| V-Straw downhill: horizontal request below the control's | 0 ticks | **0** (descent_scale at the contacts 0.65-1.0) | 0 | descent_scale 0.44-0.53 |
| **V-Straw uphill**: escalated s per min, 9 laps pooled | <= 0.2 | **0.0** (0 s in 31.9 min; guard climb never above 1 m/s) | 3.39 (108 s) | climbs 3.4-34 s per lap, up to 3.5 m/s |
| **V-Minus** max requested vz, wall-01 / gapon-01 / gapon-02 windows | <= 1 m/s | **0.97 / 0.12 / 0.98** | 0.97 / 0.88 / 0.98 | 3.44 / 0.79 / 3.46 |
| V-Minus wall-01: request level before 0.3 m height lost | yes | **yes** (from the sink's start, 0.02 m lost) | yes | - |
| V-Minus no escalation: guard climb <= 1 m/s, all 9 Minus logs | all | **pass** (max 1.0) | pass (max 1.0) | - |
| **V-Pine** climb: >= 1 m/s during >= 80% of the seconds the logged governor asked >= 1 m/s | >= 80% | 77.7% | 77.7% | 89.4% (flown log 91.4%) |
| V-Pine: no requested descent in the last 2 s | min >= 0 | -0.70 | -0.70 | -0.72 |
| V-Pine mound: height request vs the flown one | >= 90% | **106%** | 106% | 92% |
| **Result** | | Identity, V-Straw uphill, V-Minus pass; V-Straw downhill, V-Pine fail | V-Minus passes; V-Straw downhill, V-Straw uphill, V-Pine fail | |

The version-2 definitions on the same replays (`old_definitions`):

| Old gate (v2 definitions) | v3 | v2 |
|---|---|---|
| V-Straw (04/06): limited >= 80%, 0 raised, 0 reduced, 0 ticks of guard climb above 1 m/s | 20%, 0, 0, **0 ticks**: fails on the first criterion only | 20%, 0, 0, 4233 ticks |
| V-Pine: >= 1 m/s during >= 80% of all logged climb seconds; no descent in the last 2 s | 71.8%, -0.70 | 73.8%, -0.70 |

Pine Valley per logged climb episode (`pine-fast6-ttc-01`; seconds with the logged governor
at >= 1 m/s answered with >= 1 m/s, and the height request):

| Episode | Logged (>= 1 m/s) | v3 | Control | Flown height request | v3 height request |
|---|---|---|---|---|---|
| 4.3-6.0 s, the mound | 1.38 s | 1.27 s (max 3.5 m/s) | 1.27 s | 3.42 m | 3.63 m |
| 15.2-16.9 s, hillside | 1.39 s | 0.38 s (gentle 1.0 only) | 1.14 s | 3.17 m | 0.51 m |
| 17.6-20.5 s, hillside | 2.25 s | 2.25 s (max 3.3 m/s) | 2.08 s | 4.53 m | 5.43 m |

### What the version-3 replays show

- **Straw Bale uphills: no escalation left.** On all nine laps the guard never asks more
  than 1 m/s. It still climbs gently for 4-19 s per lap (7-17 onsets, each started by a
  path alarm), mostly under the pilot's own climb. The extra height it asks above the pilot
  is 0-1.6 m per lap (6.4 m in all), against 0.9-44 m per lap (191 m) for version 2. For
  comparison, the stack without the guard (as flown on Minus Two and Pine, with looming)
  would climb for 3.4-34 s per lap at up to 3.5 m/s on these laps.
- **Straw Bale downhill: unchanged from version 2.** It keeps speed at the contacts
  (descent_scale 0.65-1.0, against the control's 0.44-0.53). It never adds a climb while
  descending, and it does not see the contacts coming (2 of 10 "limited", from the previous
  hill). That part is for the pilot change that keeps descents in view.
- **Minus Two: the ceiling fix is kept.** The requests stay at most 0.98 m/s in the ceiling
  windows, and no Minus log escalates.
  - Version 3 drops the gentle climbs that version 2 started on the lower window alone:
    `minus-brain08-gapon-01`, `minus-fast6-vg-02`, and one of `minus-fast6-gapon-01`'s two.
  - It starts the hairpin climb of `minus-brain08-vg-01` later (0.75 s of climb against
    0.88 s).
  - With fewer climbs under a 2.2 m ceiling, nothing new goes up. These are the requests at
    the recorded states. The round-3 flight `minus-fast6-vg-02` flew with version 2's two
    gentle climbs (at 8.5 s, and at 21.6 s, 1.8 s before the pillar C impact); version 3
    would not have made them.
- **Pine Valley: the mound is kept, the first hillside is not answered.**
  - The mound escalates at 4.55 s exactly as with version 2. Its height request is 106% of
    the flown one.
  - The hillside at 17.6 s escalates at 18.44 s. Its onset is 0.3 s later than version 2's
    (17.62 s against 17.33 s), because the lower window alone no longer starts it.
  - The hillside at 15.2 s gets 1 m/s where the flight climbed 3.5 m/s (see the diagnosis).
    This fails V-Pine's climb criterion (77.7%) for version 3 and version 2 alike.
  - The brain-08 hillside of `pine-brain08-01` escalates at 16.96 s, as with version 2.
- **The hillside at 15.2 s and the Minus Two floor sink look the same.** Both are a path
  alarm (alarm 0.71-0.85 s, lower 0.24-0.42 s, `below_fraction` 0.78-1.0), received while
  the PD sinks at 0.31-0.34 m/s with the pilot asking level or a slight climb:
  - `minus-fast6-wall-01` at 12.51 s must not climb above 1 m/s (the ceiling);
  - `pine-fast6-ttc-01` at 15.19 s got 3.5 m/s as flown, and after that climb the
    hillside loomed again at TTC 0.54 s.

  A rule that reads only these samples, the vertical speed and the pilot's request cannot
  meet V-Minus and V-Pine's climb criterion at once in open loop. Only what happens after
  the descent is stopped separates them, which needs a closed-loop test or a new causal
  signal.

### Idealised closed-loop check (development evidence)

Round 3's verifier model (a point drone with a 0.25 s vertical lag at 6 m/s over synthetic
terrain, ideal looming at 18 Hz, 0.08 s late) gives the same result for version 3 as for
version 2 in all 20 cases. There every sample is a path alarm, and the pilot's request is
level or a sink:

| Case | v2 = v3 min clearance | m2-hairpin |
|---|---|---|
| ramp slope 0.20, level from 1.5 m | 0.43 m | 0.56 m |
| ramp slope 0.35, from 1.0 m sinking 0.5 m/s (pilot -0.8) | **-0.08 m** | 0.67 m |
| ramp slope 0.50, same start | **-1.14 m** | 0.54 m |
| ramp slope 0.50, level from 1.5 m | -0.43 m | -0.60 m |
| garage step 0.8 m, sinking 0.35 m/s, pilot -0.5: ceiling clearance | 0.63 m (max 1.0 m/s) | 0.20 m (max 3.5 m/s) |

This is the upslope risk of round 3: a descent into a steep ramp is stopped first and climbed
gently. Version 3 does not change it (see Limits).

## Round 3: versions 1 and 2 (open-loop replays)

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

### What the round-3 replays showed (v1, v2)

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

## Limits and risks

- **Open loop.** Every replay number is the request a variant would have made at the
  recorded states, not a flight. It does not show whether brain-08 follows a withheld sink
  or a gentle climb.
- **Development evidence.** Every log here was inspected, and replayed through version 2,
  before version 3 was written. Versions 2 and 3 were changed after replays of the same
  logs.
- **Offline looming streams.** The Straw Bale laps and `pine-brain08-01` use streams
  recomputed from video, not the camera's own samples. The alignment residuals of the
  round-4 streams are 0.38-0.87 px, against 0.31-0.36 px for the two laps of round 3.
- **A narrow binding margin (v3).** On the logs, the mound's escalation passes the binding
  test by 0.06 m/s: the pilot asked 0.44 m/s, and the test allows up to 0.5 m/s. The lowest
  Straw Bale escalation of version 2 is blocked by 0.11 m/s (0.61 m/s). A mound approached
  toward a higher ring (the pilot asking 0.5-1 m/s) would get only the gentle 1 m/s, for at
  most 1 m. That is exactly the case of version 2's Straw Bale escalations, which version 3
  removes.
- **Descending into an upslope is not fixed (v2 and v3).** In the idealised closed-loop
  check, a drone that sinks into a 19-27 deg ramp gets less clearance than with the
  m2-hairpin governor. Examples:
  - slope 0.35, from 1.0 m sinking at 0.5 m/s: -0.08 m against +0.67 m;
  - slope 0.5, same start: -1.14 m against +0.54 m.

  Descent first arrests to level before any climb, and the ramp and the flat garage floor
  of `minus-fast6-wall-01` give the same samples until the arrest has taken effect. The
  Pine hillside at 15.2 s is such a case: version 3 answers the flown 3.5 m/s climb with
  1 m/s. A fix needs a causal test of "the ground keeps looming although the descent was
  stopped", for example the vertical speed at each sample's capture time. It is not in
  version 3.
- **Straw Bale contacts remain.** The camera (30 deg uptilt, 42 deg vertical half field of
  view) cannot see a descent steeper than about 12-14 deg below the horizon, and the pilot's
  bottom-clamped descents are 20-29 deg by design (to pass under the descending arches).
  Options, none taken here:
  - keep the descent within view (at the price of the arch clearance; another round-4
    change);
  - a lower camera uptilt (needs the brain's sensory contract re-trained);
  - publish looming2's upper-window TTC, which sees the ground ahead of a steep descent.
- **Pine end.** Versions 2 and 3 do not arrest a descent toward a surface that only the
  lower window sees (the boulder at 21.2 s).
- **Gentle climbs remain on Straw Bale.** Version 3 still climbs gently (at most 1 m/s, 1 m)
  for 4-19 s per lap. Mostly this sits under the pilot's own climb. The extra height it asks
  is 0-1.6 m per lap.

## Tests

- **`tests/test_fast_race_cue_vertical.py`**, 31 tests:
  - the declaration is frozen and refused when edited or of another version; v1 and v2 are
    kept verbatim and refused;
  - the sink factor's margin, ramp and memory;
  - descent first and the wall-01 shape;
  - confirmation, the gentle bound and its height limit;
  - escalation, graded by urgency;
  - the binding test (a pilot climbing up a hill does not escalate);
  - **v3:** a crest (lower TTC 0.4 s, alarm 6 s) starts no climb, and a path alarm among the
    confirming alarms does (the lower window then sustains it);
  - **v3:** the binding test over the pilot's recent requests: a one-tick dip at a
    checkpoint switch and a steady 0.6 m/s do not escalate, while a pilot that stopped
    climbing more than 0.5 s ago and the mound's 0.4 m/s do;
  - the arch pattern (a long alarm limits nothing);
  - the ceiling guard still cuts;
  - unchanged governor without the guard;
  - the pilot's margin, arrest, mound climb and keep speed (withheld sink and contact);
  - shadow flies the unguarded pilot bit for bit;
  - the runner's columns, flags and refusals.
- **`tests/test_vertical_replay.py`**, 10 tests: the gates declarations (v3 frozen, v1 and v2
  kept, v2 definitions carried unchanged) and the scoring functions on synthetic arrays,
  including the v3 V-Pine, uphill and no-escalation scores.
- **`tests/test_gap_pilot.py`** covers the stack field.
- **The full suite passes:** 961 tests (round 4).
- **CPU wiring check (plumbing only, no pad, no flight).** `VisualController` was built as
  `run()` does for four cases:
  - brain-08 with the stack on;
  - brain-08 in shadow;
  - the fast PD with the stack on;
  - the fast PD with `--vertical-guard off`.

  Each pilot received the version 3 declaration (`b70e263ccdc5...`; `applied` true, false,
  true and absent), the sidecar component and the six log columns.
