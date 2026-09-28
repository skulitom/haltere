# Motor assist for lagging brains (round 4b)

**Status: not flown.** Off by default (`--motor-assist on|off|DECLARATION`). The declaration is
`configs/pilot/motor_assist.json` **version 1** (sha256 `eefb4a42...`), declared per motor contract: the
brain contract (`fast_velocity_brain_v1`: fast-brain-08, brain-09b, fast-brain-10b) has an entry, the fast
PD has none and flies bit-identically with the flag on. Its frozen gates are
`configs/pilot/motor_assist_gates.json` version 1 (sha256 `9abfb80d...`), frozen and committed
(`7a29724`) before any gate run; scored once (`0c3f56c`). Everything below is surrogate or open-loop
replay evidence, never flight evidence.

**Version 1 passes 12 of its 16 frozen gates.** It stops the braking brains at the held-out hairpin
walls (brain-09b clean passes 1 -> 8 of 12, fast-brain-10b 0 -> 7, wall contacts 11 -> 0 and 12 -> 1),
holds brain-09b's height in the reproduced live stop-then-accelerate window (lowest point 0.06 -> 0.38 m),
slows fast-brain-10b in its reproduced live hairpin window (4.85 -> 3.26 m/s), leaves the 16-course
development gate and the descent hills alone (no finish lost, course time +0.3..+1.2%), and is
bit-identical to m4 when off (57/57 replay arrays). It fails:

- **hairpin fast-brain-10b**: two ceiling contacts (a braking brain climbs; limit baseline + 1 = 1);
- **accelerate brain-08**: its height loss is unchanged (a slow sink at cruise speed, below the rule's
  deadband, not the low-speed acceleration sag the rule targets);
- **accelerate brain-09b**: the loss fell to 61% of the baseline's (gate: 60%) and it touched the ceiling
  once more;
- **quiet**: on the clean Straw Bale laps the stopping bound acts 1.8-3.3 s per minute (gate: 1) at the
  uphill rings, and the open-loop request removed reaches 4.1% of the flown distance (gate: 3%).

It does not make any brain a release brain. The hairpin gain is the clearest result; the ceiling and the
floor after a stop remain the flight risks in the 2.2 m Minus Two garage (see [Risks](#risks-for-live-flight)).

## What the live flights showed (development cases)

Both logs were read to design the rule, so every replay of them is development evidence.

- **minus-brain10b-r4-02** (fast-brain-10b): at the hairpin it slowed to 2.5-2.8 m/s, then flew
  5.1-5.2 m/s under 3.6-4.0 m/s governor caps (the pilot's request) into the wall at (81.9, 19.0). It never
  converged to the cap: a brain that follows 0.3 s late and weakly overshoots every cap unless the request
  goes below it. From 20.2 s the vertical guard also asked for a 1 m/s gentle climb (it read the looming
  ahead as terrain, below fraction 0.76; the round-4b guard branch changes that rule).
- **minus-brain09b-r4-01** (brain-09b): turn-first v4 stopped it at (81.1, 20.0); released aligned 0.41 s
  later, it accelerated toward a 4.9 m/s request, tilted up to 57 deg and sank from 0.78 to 0.06 m at up to
  1.9 m/s while its vertical request was +0.06..+0.16 m/s.

**The surrogate reproduces both** (the logged request replayed from the logged state, the brain warmed on
the recorded inputs; `haltere.train.brake_gates.Flight`): fast-brain-10b's speed at the end of the window
from 20.2 s is 4.85 m/s (live 4.93), from 17.8 s 5.00 (live 5.03) with the lowest point 0.71 m (live
0.72); brain-09b's lowest point from 22.95 s is 0.06 m (live 0.08). Replaying the logged stick commands
open loop gives 4.87 m/s and 0.07 m, so the plant is also faithful here. The window gates check this again
(within 0.08 m/s and 0.02 m).

## Which lever: bound the acceleration, or hold a climb bias?

The task asked which lever the data supports for the sag out of a stop.

- **Live logs** (all brain and PD logs of the fast-stack folder; `sag_logs.json`): at less than 2.5 m/s the
  vertical shortfall (the request 0.25 s earlier minus the measured vertical speed) grows with the lead the
  pilot asks the motor to accelerate by: brain-08 +0.56-0.73 m/s at 0.5-3 m/s of lead (1.7 above 3),
  brain-09b 0.16 at 1-2 and 0.52 at 2-3, the fast PD 0.17 and 0.47. Every launch loses 0.2-0.35 m the same
  way; brain-09b's r4 episode lost 0.71 m at 57 deg of tilt.
- **Surrogate** (`accel.json`, attempt 1): from the logged turn-first release of brain-09b, a horizontal
  request that grows at 1.5 / 2.5 / 4 m/s^2 instead of the pilot's slew loses 0.86 / 1.35 / 1.59 m instead
  of 1.61 m and reaches 4 m/s 0.4-2 s later or not at all; a request held at most 0.75 / 1.5 m/s above the
  measured speed loses 1.15 / 1.21 m (0.75 stalls the brain); a climb bias of 0.3 / 0.6 / 1.0 m/s at the
  pilot's own slew loses 1.19 / 0.85 / 0.33 m. From a hover and for brain-08 and fast-brain-10b the bias cuts
  the loss just as monotonically, while the slew bound is inconsistent.

So the rule holds a climb bias; it does not bound the growth of the horizontal request.

## The rule (version 1)

`haltere.liftoff.fast_race_cue.MotorAssistConfig`. Both parts act on the pilot's final request, after every
other rule and the command slews. The motor receives the assisted request; the pilot keeps its own request
as its state (its rules read that next tick). The support rule and the descent-path shortfall compare the
measured vertical speed with the vertical request the motor received: a sink the assist withheld is not the
vehicle failing to descend.

1. **Cap tracking.** For each binding cap, the measured horizontal speed along its direction beyond the bound
   plus 0.3 m/s lowers the bound by 2 x the excess beyond that deadband (at most 3 m/s, moving at up to
   8 m/s^2). The sources are:
   - the pilot's own final request in `cue`, `below`, `below_weak` and `side` (the speed schedule, the edge
     speeds and every cap already applied);
   - the looming governor's cap along the horizontal part of its ray;
   - turn-first's wall ray (no speed toward the wall) and its creep bound;
   - a **stopping** source: while 2 wall samples (not below-path terrain) with a TTC under 1.3 s arrived
     within 0.25 s and the newest looming sample is one of them, the speed toward the governor's latest wall
     sample is bounded by the speed from which the motor stops within the remaining distance
     (v x 0.3 s + v^2 / (2 x 3.5 m/s^2) + 0.5 m: the wall pilot v4's measured model of this contract). The
     governor's own TTC thresholds were chosen for the fast PD; a brain that follows 0.3 s late must start
     braking about 1 s before a wall at 5-6 m/s. A clear sample (a gate arch flown through) ends the bound.

   The request never reverses along its own direction and points away from a wall by at most 1 m/s.
2. **Sag compensation.** A climb bias of 1 x the vertical shortfall beyond 0.3 m/s (the brain sinks faster
   than asked), plus 0.3 x the horizontal lead beyond 0.5 m/s while the drone is at most 2.5 m/s fast (an
   acceleration from low speed: its sink starts before any feedback sees it). At most 1 m/s, rising at
   5 m/s^2 and falling at 2 m/s^2; nothing while the drone already climbs more than 0.3 m/s faster than
   asked, during the launch or a support climb, and never above the pilot's `vertical_up` or an overhead
   bound of the ceiling guard.

It reads only the measured velocity, the pilot's own rule states and the governor's looming samples: no
course geometry, route, logged position or per-course value.

**Left out** (declaration `not_included`, with the development numbers): the governor cap along the
horizontal part of its ray (the round-4b contact fix, owned by another branch; not duplicated), raising
turn-first's 3.5 m/s engagement limit (with the stopping source it changed nothing or added floor contacts; a
stop from race speed makes a brain climb, then sink), a climb compensation and a tilt term (mixed or worse
in development), and the horizontal slew bound (see above).

**Ablations** on the development hairpins (clean passes of 12, fast-brain-10b / brain-09b / brain-08):
this version 8 / 11 / 4; without the stopping source 0 / 0 / 0 (every wall hit); without the lead bias
7 / 12 / 3; without any sag compensation 4 / 5 / 1 (floor contacts 6 / 7 / 9).

## Frozen gates and results (version 1)

Scored on held-out sets: 12 hairpin scenarios (turn 45/75/105 deg x arch 8/13 m x wall 2.3/3.0 m, sim seed
23; development: 20/60/90, 7/11, 2.1/2.6, seed 17), 6 stop-then-accelerate scenarios (ring height
1.0/1.5 m x bearing 45/120/180 deg, seed 23; development 0.8/1.2 x 0/90/150, seed 17), the 16-course
development gate (flat and steep 3000-3007) and the descent hills (6000-6011), all under the round-4 pilot
(`--obstacle-stack on --descent-view on`) with and without the assist. The scenario sets have a 2.2 m
scoring ceiling (the Minus Two garage). Scores: `docs/experiments/motor_assist_v1_scores.json`.

| Gate | Threshold | brain-08 | brain-09b | fast-brain-10b |
|---|---|---|---|---|
| Hairpin (12) | braking brains 09b/10b: clean >= base + 6, walls <= 25% of base; all: any-contact scenarios <= base, ceiling <= base + 1 (08: clean and walls not worse) | clean 0 -> 1, walls 10 -> 3, floor 3 -> 9, ceiling 0 -> 0: **pass** | clean 1 -> 8, walls 11 -> 0, floor 0 -> 3, ceiling 0 -> 1: **pass** | clean 0 -> 7, walls 12 -> 1, floor 0 -> 2, ceiling 0 -> 2: **fail** |
| Accelerate (6) | loss <= 60% of base if base >= 0.2 m, else <= base + 0.05; floor, ceiling, crashes <= base; time <= +5% | loss 0.40 -> 0.40, ceiling 3 -> 3: **fail** | loss 0.34 -> 0.21 (61%), floor 1 -> 0, ceiling 1 -> 2: **fail** | loss 0.15 -> 0.11, ceiling 3 -> 3, time +2.2%: **pass** |
| 16-course gate | finishes, crashes not worse; chatter <= 1.15x; time <= +3%; contacts <= base + 1; high passes <= base + 2 | 15 -> 16 finished, crash 1 -> 0, time +0.7%, chatter -0.1%: **pass** | 16/16, time +1.2%, chatter +5.3%, contacts 7 -> 7: **pass** | 16/16, time +0.3%, contacts 3 -> 1: **pass** |
| Descent hills (12) | as above | contacts 6 -> 6, high 15 -> 14, time -0.2%: **pass** | contacts 7 -> 7, high 11 -> 10, time +0.6%: **pass** | contacts 9 -> 9, high 18 -> 15, time +0.7%: **pass** |

| Gate | Threshold | Result | Pass |
|---|---|---|---|
| Window minus-brain10b-r4-02 from 20.2 s (development) | reproduced within 0.5 m/s; end speed <= unassisted - 1.0 | reproduced 4.85 vs live 4.93; 4.85 -> 3.26 m/s | yes |
| Window minus-brain09b-r4-01 from 22.95 s (development) | reproduced within 0.15 m; lowest point >= unassisted + 0.2 m | reproduced 0.06 vs live 0.08; 0.06 -> 0.38 m | yes |
| Identity (replays vs `git archive 3decaac`) | all bit-identical | 57/57 (24 logs x full stack and default pilot, 9 fast PD logs with `--motor-assist on`) | yes |
| Quiet (clean Straw/Pine brain-08 laps, open loop) | no more turn-first episodes; removed <= 3% of distance; stopping bound <= 1 s/min | turn-first 0 -> 0; removed 0.3-4.1%; stopping 1.8-3.3 s/min on Straw | **no** |

Chatter (roll/pitch command change per tick) rises in the hairpins with the harder braking (brain-09b
0.0082 -> 0.0108, fast-brain-10b 0.0037 -> 0.0057) and by 0-5% on the courses.

**Why the failures happen** (post-scoring diagnosis of the same scenarios; nothing changed v1):

- The **ceiling contacts** come from braking climbs (a brain pitched back hard climbs 0.5-1.5 m/s) and from
  the lead bias after one: fast-brain-10b stopped at a wall, climbed to 1.9 m, and the bias for the
  re-acceleration then carried it to 2.57 m, although the pilot wanted to descend toward a ring below (its
  final request stayed level after the rules near the wall). brain-09b's extra accelerate ceiling contact is a
  marginal case (2.16 m against 2.06 m without the assist: the bias added 0.1 m/s to the launch climb); in
  the 180 deg case both variants touch the ceiling, the assist higher (2.87 m against 2.39 m) because the
  bias added to the pilot's climb request toward a ring clamped at the top edge.
- The **floor contacts after a stop** are mostly the looming governor's stand-off cap along a ray captured
  while the braking brain climbed (ray z up to 0.72): as the pilot re-accelerates toward the next ring, that
  cap cuts its own vertical request to -1.8 m/s and the brain follows it down. This is the round-4b contact
  fix's case, not the assist's.
- **brain-08's** accelerate loss is a slow sink at 5-6 m/s (0.1-0.2 m/s below its request, inside the
  0.3 m/s deadband), not an acceleration sag.
- The **Straw stopping episodes** are the uphill rings (x about 35, y 95-167) on every lap: looming samples
  of the rising slope ahead with a TTC of 0.8-1.0 s. Two thirds carry no below fraction (like the Minus
  walls), a quarter one under the terrain fraction, and 40% a lower-surface TTC (median 0.7-0.8 of the TTC:
  the lower window explains them).

**Report-only diagnostics** (the same held-out scenarios; not gates):

- *Live-like wall samples.* The live Minus Two wall samples carry no below fraction (78-100% of the
  short-TTC samples) and the lower window explains 0-7% of the brains' ones, while the gate's synthetic
  samples carried 0.5 and a lower TTC equal to the TTC. With live-like samples
  (`motor_assist_eval.run_scenarios(live_wall_samples=True)`) the assist gives fast-brain-10b 7 clean
  passes with no ceiling contact (floor 3) and brain-09b 11 (floor 1); the baselines still hit every wall.
- *With the contact fix's stand-in* (the governor cap along the horizontal part of its ray,
  `hairpin_flatray`): with the gate's samples brain-09b 10 of 12 clean, fast-brain-10b 6, brain-08 3, and
  no floor contact for the two braking brains; with live-like samples as well brain-09b 12 of 12,
  fast-brain-10b 7 (2 wall contacts, no floor or ceiling), brain-08 3.

## Explored after scoring, not adopted

After v1 was scored, three changes were tried on development data only (attempt 1's development sets and
v1's gate sets, now seen; the Straw replays): the sag bias referenced to min(request, 0) so that it never
raises a requested climb; the lead bias only while the pilot's own request before the governor is not a
descent; and stopping samples that the lower window explains counted as terrain (the ceiling guard's test).
The first two moved the hairpin results by +-2 clean passes of 12 per brain in both directions (fewer
ceiling, more floor contacts); the third cut the Straw stopping activity to 1.2-2.3 s per minute (still
above the gate's 1) and left the Minus logs unchanged. None was a clear improvement, so no version 2 was
frozen. The next version should be frozen with fresh held-out sets after the contact fix and guard v4 are
merged, because both change exactly the vertical interactions these changes target. The explored diff and
its results are in the session scratchpad (`m4b/assist/r2/v2_explored.patch`, `devcheck2.jsonl`,
`quiet2.json`) and summarised in `docs/experiments/motor_assist_v1_development.json`.

## Open-loop replays of the Minus Two brain logs (development)

At the recorded states (`vertical_replay.py --motor-assist`), the stopping source would have acted 1.4-3.5 s
on every Minus Two brain log, the assist removing 7-19 m of requested travel in all and adding 0.1-1.2 m of
climb. On minus-brain10b-r4-02 the stopping source cuts the request from 6 to about
0.2 m/s at x 67-70 (17.6-18.1 s), where the governor itself capped 3.8-5 m/s, and again at the wall from
21.0 s; on minus-brain09b-r4-01 it starts at 21.6 s, 1.2 s before turn-first engaged live, and the sag
compensation asks for about +1 m/s of climb from 23.2 s, while the brain sank. The recorded motion does not
respond, so these are the requests the assist would have made, not flights.

## On the integration branch `m4b` (round 4b)

`m4b` merges this branch with vertical guard v4, wall pilot v5 (the clearance brake never adds sink:
the "contact fix" this document waits for, as a sink floor rather than a horizontal-only ray) and
descent view v2 (contact support). With the assist on, contact support also compares the measured
vertical speed with the request the motor received. The harness's sets flown under the merged pilot
(development report, not a re-score; flight card, Round 4b):

- hairpin, clean of 12 with the assist: brain-09b 9, fast-brain-10b 7, fast-brain-11-b-cw13 7; without
  it 0 for every brain (the fast PD 10). The floor contacts after a stop are gone (brain-09b 3 -> 0,
  fast-brain-10b 2 -> 0).
- every hairpin run has 2 ceiling contacts that come from contact support firing falsely on the two
  drones whose randomised thrust is about 15% above the declared curve, with or without the assist.

## Risks for live flight

- **Not flight evidence.** The surrogate's walls, looming and contact scoring are synthetic; its brains
  reproduce the two live failures, but Liftoff's walls, ceiling and looming will differ.
- **Ceiling.** In the 2.2 m garage, braking climbs plus the lead bias can reach the ceiling (hairpin 2 of 12
  for fast-brain-10b; the reproduced 10b window climbs to 2.57 m with the assist against 1.92 m without,
  because the guard also asked for a 1 m/s climb there). The ceiling guard only bounds the vertical
  request after overhead evidence.
- **Floor after a stop.** Until the contact fix is merged, the governor's cap along an up-tilted ray can pull
  a stopped brain down as it re-accelerates (brain-09b 3, fast-brain-10b 2 floor contacts of 12).
- **Slower and stop-and-go.** The stopping source brakes about 1 s before any wall-like surface: on the
  Minus hairpin approach it asks for near zero speed where the pilot would have turned, and on the Straw
  uphill rings it acted 1.8-3.3 s per minute (open loop: up to 4% of the requested travel removed).
- **brain-08** does not brake for caps; with the assist it stops at walls but then touches the floor
  (hairpin floor contacts 3 -> 9 of 12). Fly the assist only with a braking brain (brain-09b or
  fast-brain-10b), as a disclosed development deviation, and preferably after the contact fix and guard
  v4 are merged and the next assist version is frozen.
- **Support rule.** With the assist on, a sink the assist withheld no longer counts as ground contact. If
  the assist raises the vertical request while the drone rests on a slope, the support climb may start later.

## Files

- Rule: `haltere/liftoff/fast_race_cue.py` (`MotorAssistConfig`, `FastRaceCue._assist_sources`,
  `FastRaceCue._motor_assist`); runner flag and CSV columns (`assist_pilot_vx/vy/vz`, `assist_horizontal`,
  `assist_vertical`, `assist_source`, only with the flag on) in `haltere/liftoff/visual_brain.py`; replay
  option `--motor-assist` in `haltere/obstacles/vertical_replay.py`.
- Declaration and gates: `configs/pilot/motor_assist.json`, `configs/pilot/motor_assist_gates.json`.
- Surrogate scenarios, live windows and diagnostics: `haltere/liftoff/motor_assist_eval.py`; gates runner
  and scorer: `haltere/liftoff/motor_assist_gates.py`.
- Tests: `tests/test_fast_race_cue_motor_assist.py` (rule, declaration, runner wiring, identity digest of the
  full round-4 stack against `git archive 3decaac`).
- Results: `docs/experiments/motor_assist_v1_scores.json`, `docs/experiments/motor_assist_v1_development.json`.

Reproduce (CPU, one process, 2 threads):

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.motor_assist_gates run --dir OUT
.venv/Scripts/python.exe haltere/obstacles/vertical_replay.py --tree . --out PREFIX --stack on --near-on-path --descent-view configs/pilot/descent_view.json --motor-assist configs/pilot/motor_assist.json FLIGHT ...
.venv/Scripts/python.exe -m haltere.liftoff.motor_assist_gates score --dir OUT --replays PREFIX --baseline-replays M4PREFIX --json SCORES
```
