# Obstacle milestone 2: GapPilot on Minus Two — 2026-09-25/26

Branch `m2-wire` (not merged). Plan: a judged design round chose GapPilot: a frozen
pretrained relative-depth gap cue (DA-V2-Small) that shifts the aim to the free side
of the ring, lag-aware turns after a checkpoint switch, and the existing looming
(TTC) governor; no GPU training, no brain retraining
([gap pilot](../obstacle_gap_pilot.md), [gap cue](../gap_cue.md)).

## Failure anatomy (logs and videos of all earlier runs)

- **Minus Two pillar A:** the next ring appears 5.9 m before a dark garage pillar;
  the line to the ring passes 0.42 m beside it. brain-08's velocity lags the pilot's
  request by 0.32-0.39 s (the PD 0.11-0.14 s), so it cuts into the pillar.
- **Minus Two hairpins:** the next ring appears 1.4 m before a wall; only slowing
  before the gate works.
- **Pine Valley:** the ring is drawn through a mound (humans fly 8-10 m to the
  right); trees can also stand on the line to the ring.

## Baselines (main, no new code)

| Run | Outcome |
|---|---|
| `pine-brain08-loom-01` (brain-08, `--looming-brake`) | Tree trunk on the line to the ring at 9.6 s; TTC fell from 1.8 to 0.5 s in 0.25 s, climb too late |
| `minus-brain08-slow35-01` (brain-08, `--assist-speed 3.5`) | Flew 5.1 m/s instead of 3.5; pillar A |
| `minus-brain08-loom-01` (brain-08, `--looming-brake`) | False brake at the arch (the brain did not slow); looming read ~14 m at the moment it hit the dark pillar |
| `minus-fast6-cur-01` (fast PD, current arc-turn pilot) | Pillar A at its left edge (y 4.41); the PD passed at y~4.9 with the pre-arc pilot |

## Offline gates (frozen before scoring)

Gap cue v1/v2 (336x602 depth input): G2 pillar B passes; G1 pillar A fails (left
confirmed at >= 5.5 m on 4 of 8 approaches), G3 clean Straw Bale fails (9.3
confirmed episodes/min against 6), G4 leak test fails narrowly (94.8% against 95%).
Lag-aware turns: surrogate brain-08 15/16 -> 15/16, post-switch lateral deviation
1.72 -> 1.54 m; PD 14/16 -> 14/16. Runtime bench: separate depth process 17.7 Hz,
gap sample age p95 117 ms, checkpoint-cue latency unchanged.

**Deviation:** the plan blocked authority flights of the gap cue after a G1
failure. It was flown with authority anyway on this development course after the
live shadow run below confirmed the left shift at 5.8 m; these are development
results, not a pass of the frozen gates.

## Live (m2-wire; `--looming-brake --obstacle-stack on|shadow`)

| Run | Mode | Outcome |
|---|---|---|
| `minus-brain08-gapshadow-01` | brain-08, shadow | Pillar A, as without the stack; the gap cue confirmed a left (ring-side) shift 5.8 m out, growing to 12 degrees; camera ~16.5 Hz, gap sample age p50 96 ms / p95 135 ms |
| `minus-brain08-gapon-01` | brain-08, on | **Passed pillar A** (y 4.65; edge 4.45). At the next 90-degree arch turn the brain did not slow for the governor's caps (3.7-4 m/s asked, ~6 flown), then the terrain climb (floor looming below) drove it into the garage ceiling at (80.7, 19.2, 2.1), 19.1 s |
| `minus-brain08-gapon-02` | brain-08, on | **Passed pillar A** (y 4.71); same ceiling climb at (78.1, 19.0, 2.2), 18.6 s |
| `minus-fast6-gapon-01` | fast PD, on | **Passed pillar A** (y 5.18). Braked to a stand-off at the hairpin wall; with the next ring clamped at the left edge the side rule pushed it along the wall; low-speed impact at (80.0, 18.1), 21.6 s |

First time any brain run passed pillar A at 6 m/s.

## Round 2: wall pilot rules and brain braking

Branch `m2-hairpin` adds two rules inside the stack (`--wall-pilot`, frozen
`configs/obstacles/wall_pilot.json` v3): turn toward the ring before translating
after a stand-off stop, and a ceiling guard on the governor's terrain climb. Open
-loop replays of the logs: the PD's push into the hairpin wall and brain-08's 3.5
m/s ceiling climb are removed, the Pine mound climbs are unchanged, and nothing
changes with the stack off or in shadow.

| Run | Mode | Outcome |
|---|---|---|
| `minus-fast6-wall-01` | fast PD, stack + wall pilot | Crashed before pillar A at (54.0, 4.8, 2.25): the governor false-braked on the arch, then the PD, sinking at 0.37 m/s from 0.7 m, saw the floor loom (below-path TTC 0.24 s) and the governor climbed at 3.5 m/s for 0.8 s into the ceiling. The ceiling guard leaves climbs with below-path evidence alone (for the Pine mound). |

Brain braking diagnosis: brain-08 does not brake for 3-4.5 m/s requests in the
surrogate either (the slow-request metric hid it). Its pitch response to the
speed error is 1-6% of the teacher's between 3 and 7 m/s; it brakes hard only for
requests below about 2 m/s. Its distillation data had 0.13% of samples in that
regime and no sustained governor-like caps. brain-09 recipe: synthetic governor
caps and slow legs in the DAgger rollouts, a braking weight, and a
ridge/smoothing refit sweep, with cap-step, sustained-request and live-state
swap gates frozen first.

## Round 3 (not flown): vertical guard

Branch `m2-vertical` adds a scale-free [vertical guard](../vertical_guard.md) to the stack
(`configs/obstacles/vertical_guard.json` v2; v1 kept). It keeps a time margin to the ground
below the path, stops a descent before any terrain climb, and climbs above 1 m/s only for
confirmed rising ground. It also keeps speed at contacts.

Open-loop replays of the logs (development evidence):

- The Minus Two ceiling climbs fall from 3.4-3.5 to at most 0.98 m/s.
- The Pine mound is still climbed at up to 3.5 m/s.
- Stack off and shadow are bit-identical to `m2-hairpin`.
- It fails two of its frozen gates:
  - **V-Straw:** looming cannot see the Straw Bale contacts, because the descent points
    below the camera's view. Also, about 23 s per lap of escalated climb remain on the
    uphill legs.
  - **V-Pine:** 73.8% against 80%; the flown log itself reaches 72.4%.

## Next (revised)

1. Graded terrain climb: when the floor looms because the drone is descending,
   stop the descent first; climb hard only if the below-path TTC stays short while
   level (rising ground).
2. brain-09 with the braking recipe above. Done offline ([brain-09](../brain09_braking.md)): with
   synthetic caps, slow legs and a brake weight the best candidate brakes for 1-4.5 m/s requests
   almost like the PD on straight legs (cap excess +0.19 against brain-08's +1.20), but it slows
   1.5 m/s too much in the arch-turn windows and chatters 0.0051 (limit 0.0035); none of 22
   candidates passed the frozen gates, so there is no brain-09 to fly yet.
3. Then Minus Two again (PD and brain), a Straw Bale regression lap and Pine
   Valley with the stack on.

## Round 3: graded vertical guard (branch m2-vertical) and a braking brain

The 2-D free-space planner v1 (branches m3-freespace/m3-pilot) failed 6 of 9 frozen
offline gates (wrong side at pillar A on 5/5 approaches, 7.8 false blocked episodes
per minute on clean Straw Bale, floor height 1.85x) and was not flown; a diagnosis
recommends extending the gap cue instead. A scale-free vertical guard on the
looming below-path time to contact (v2, frozen `configs/obstacles/vertical_guard.json`)
passed the Minus ceiling replays but not Straw Bale or Pine: on the Straw downhill the
flight path points 5-15 degrees below the bottom of the (30-degree up-tilted) camera
image, so no camera cue sees the ground there; its verifier found upslope and Straw
uphill risks, so it is not for Straw or Pine yet. brain-09 (branch m3-brain09): 22
candidates, none passed all frozen gates; the best (`fast-brain-09b-caps-r0001m100`,
synthetic governor caps in DAgger, only readout rows 0-2 changed) brakes on straight
legs and was flown as a disclosed development deviation.

| Run | Mode | Outcome |
|---|---|---|
| `minus-fast6-vg-01` | fast PD | Not flown: preflight refused (an orphaned worker from an agent run used 1.15 cores; stopped) |
| `minus-fast6-vg-02` | fast PD, stack + vertical guard | Pillar A (y 5.22), **through the hairpin** at (80,19), max height 1.4 m (no ceiling climb); impact at pillar C (78.3, 31.0), 23.4 s: the ring sits at the pillar's edge, the gap cue shifted left for 0.5 s, decayed, then flipped right 0.2 s before impact |
| `minus-brain08-vg-01` | brain-08, stack + vertical guard | Pillar A (y 4.51, 6 cm clear); no ceiling climb; at the hairpin it did not slow (caps 3.9, flew ~6.4 m/s) and hit the wall at (78.1, 19.2), 18.5 s |
| `minus-brain09b-vg-01` | brain-09 candidate, stack + vertical guard | Pillar A (y 5.77 at the pillar edge x 54.7, 1.3 m clear); **braked to 2-4 m/s** into the hairpin; grazed the wall at (81.8, 19.9) at ~2.7 m/s, 21.9 s: turn-first only engages below 1.5 m/s caps (here ~3 m/s). Stick change 0.0056/tick (brain-08 0.0031) |

Next: gap cue side commitment near an obstacle (pillar C); turn-first for side-clamped
rings at a wall at governor caps up to ~3.5 m/s; smoother brain-09 (chatter); keep
descents inside the camera's view on Straw Bale (fly the downhill with the nose down
and speed kept, as the user suggested).

## Round 4 (not flown): view-keeping descent for the Straw Bale downhill

Branch `m4-descent` answers the user's request to touch the Straw Bale hill less: keep
throttle up and pitch forward instead of dropping. See [descent_view.md](../descent_view.md).

- **Diagnosis** of straw-brain08-04/-06/-01. All 10 contacts came after bottom-clip brakes
  to half speed. The brakes lifted the nose 10-20 degrees and the lower image edge with it.
  The pilot then asked for 16-30 degree descents at 3 m/s into a hill of about 12 degrees,
  and the path pointed below the image 70-79% of the last 3 s. During contact the
  descent-path governor cut the speed to 0.35-0.53x. The rings lay only 5-10 degrees down.
- **Rule** (`--descent-view`, `configs/pilot/descent_view.json` v1, off by default): keep the
  requested path 3 degrees inside the lower image edge at the measured attitude, keep speed,
  more speed rather than less, gentle sink onset, and steep only late for rings that stay
  clipped below.
- **Frozen surrogate gates** (held-out seeds, scoring-only hills): **failed for all three
  motors.**
  - Passed: no crash or finish changed.
  - Contacts: down 50% (PD), 69% (brain-08) and 38% (brain-09b) against 75%; contact time
    down 65-84%.
  - The path stayed below the image for 36-53% of the descent time.
  - Passes more than 1.5 m above a checkpoint rose from 1-3 to 13-24.
  - The brains do not fly 6 m/s descents (brain-08 keeps about 3 m/s and sinks less than
    asked).
- **Open-loop replay** of the Straw downhill: the rule's request points into the image before
  all 10 contacts, against 58-83% below with the default pilot. Development evidence only.
- **Default pilot**: bit-identical to `935cfdb` (52/52 replayed command arrays, golden-digest
  test).

Next: distil a brain under this pilot (6 m/s in-view descents), or fly the fast PD on Straw
Bale with `--descent-view on` as a disclosed development test.

## Round 4: brain-10, a smoother braking brain (branch m4-brain10, not flown)

[brain-10](../fast_brain_10_candidate.md) was distilled under the current pilot. The descent branch
was not flight ready and was not merged.

Why brain-09b chatters:

- Its pitch hunts at about 0.8 Hz in steady cruise.
- The brain lags its teacher label by 70-90 ms at about half its amplitude.
- The PD teacher is not stable with that much latency: +60 ms already doubles its chatter.

The brain-10 recipe:

- A label teacher with a slower attitude loop (gain 4) and a stronger vertical loop (gain 5).
- Labels paired with the teacher's output 60 ms later (label lead), to anticipate the latency.
- Longer synthetic caps, like the live governor's.

Six candidates were scored on the frozen gates (`configs/brain10_gates.json` v1: brain-09's G1-G6
plus smoothness G7 and regressions G8). All fail G3; none is selected for flight.

The best-ranked is `fast-brain-10b` (sha256 `0ccf1161...`). It passes all six other gates:

- It brakes for caps: in-course cap excess +0.35 m/s (brain-08 +1.20).
- Its chatter is 0.00226 per tick, 24% below brain-08 and 56% below brain-09b.
- It finishes 16/16 courses with no high descending pass.
- Only readout rows 0-2 changed.

G3 is the closest of any brain so far: worst window 0.89 m/s against brain-09b's 1.56. It is still
latency-limited: even the PD with 80 ms of added delay fails G3.

Flying brain-10b would be a disclosed development deviation, like brain-09b's flight.

## Round 4 (offline): the integrated stack on branch m4, gate tables and the live plan

**Nothing in this section has flown.** Branch `m4` is `m2-vertical` (`935cfdb`) with the five
round-4 branches merged: `m4-vguard`, `m4-hairpin`, `m4-pillar`, `m4-descent` and `m4-brain10`
(which carries `m3-brain09`'s training code and gates). The merges conflicted only in the replay
harness (`haltere/obstacles/vertical_replay.py`), the runner's CSV columns (`gap_commit` then, with
`--descent-view on`, the three view columns) and the docs. Every frozen declaration kept its
content and hash. The runner's own loaders on the merged tree load wall pilot v4, vertical guard
v3, gap pilot v5 (gap cue v2), lag turn v2 and descent view v1, and refuse wall pilot v3,
vertical guard v2 and gap pilot v4. The replay harness gained `--descent-view` so that the whole
stack can be replayed as `--obstacle-stack on --descent-view on` would fly it.

The full suite passes: 1051 tests right after the merges, and 1054 with this round's three new
tests. A CPU wiring check parsed each planned command with the runner's own parser. It built the
controller and the camera-side gap spec, but no camera, pad or flight, and every run declared
the versions in the plan's table. Results:
`docs/experiments/round4_integration.json`.

**Can the new brain release graduate from pre-release soon? Not yet.** The proposed bar is one
frozen stack (brain weights, pilot and obstacle configs, nothing per course) that finishes Straw
Bale 3/3 with no ground contact and Minus Two 3/3, with Pine Valley attempted. After round 4:

- **No brain is selected.** Every brain-10 candidate fails G3 of its frozen gates (fast-brain-10b
  fails only G3), and brain-09b fails its own gates and chatters. The stack cannot be frozen
  without a brain chosen by its gates. Two routes exist: a justified G3 v2, frozen before any
  scoring, or a candidate that passes G3.
- **The Straw Bale downhill is not solved offline.** Descent view v1 cuts surrogate contacts by
  38-69%, against the 75% the gate asks, and brains pass descending rings high. It is the only
  rule aimed at the user's downhill request, and no brain has been distilled under it.
- **Minus Two has development fixes only.** Gap pilot v5 holds the free side at pillar C in
  replay (not held-out; pillar B still fails at the pilot level). Turn-first v4 engages 0.61 s
  before brain-09b's hairpin graze, but the estimates range from a 0.29 m miss to a 1.27 m/s
  graze.
- **The vertical guard v3 is quiet on Straw** (0 escalated seconds on 9 laps) **but fails
  V-Pine.** The Pine hillside at 15.2 s gets 1 m/s where the flight climbed 3.5 m/s.

The next flights (below) are development flights of this stack. They can show whether the fixes
work live. They cannot graduate the release, because the brain they fly is not a selected one.

### Frozen gates of the five round-4 branches (as scored by each branch)

Every rule and gate file was frozen and committed before its candidates were scored. The
disclosed revisions are listed with each branch. Open-loop replays and the surrogate are
development evidence, not flight evidence.

**Vertical guard v3** (`vertical_guard.json` v3 `b70e263c`, gates v3 `68963588`, commit
`79895b7`; v1/v2 kept and refused). Gates v3 loosened V-Straw uphill from "0 ticks above
1 m/s" to "at most 0.2 s/min escalated" (disclosed; v3 also meets the old rule on 04/06).

| Gate | Threshold | v2 | v3 | Pass |
|---|---|---|---|---|
| Identity (stack off and shadow vs `935cfdb`, 21 logs) | all identical | - | 52/52 | yes |
| V-Straw downhill: sink limited before contact | >= 80% | 20% | 20% | **no** |
| V-Straw downhill: raised above pilot / horizontal cut | 0 / 0 ticks | 0 / 0 | 0 / 0 | yes |
| V-Straw uphill escalated (9 laps pooled; gates v3) | <= 0.2 s/min | 3.39 s/min | 0.0 | yes |
| V-Minus windows max vz (wall-01 / gapon-01 / gapon-02) | <= 1 m/s | 0.97/0.88/0.98 | 0.97/0.12/0.98 | yes |
| V-Minus wall-01 floor sink levelled before 0.3 m lost | yes | yes | yes (0.02 m) | yes |
| V-Minus no escalation (9 logs) | <= 1 m/s | pass | max 1.0 | yes |
| V-Pine climb >= 1 m/s while the logged governor asked >= 1 m/s | >= 80% | 77.7% | 77.7% | **no** |
| V-Pine no descent in the last 2 s | min >= 0 | -0.70 | -0.70 | **no** |
| V-Pine mound height request vs flown | >= 90% | 106% | 106% | yes |

**Wall pilot v4, turn-first at stopping distance** (`wall_pilot.json` v4 `92f842a5`, gates v1
`6418aea5`, commit `b36bab4`; v3 kept and refused).

| Gate | Threshold | Result | Pass |
|---|---|---|---|
| W-Identity: stack off / wall off / shadow vs m2-vertical | all identical | 42/42 | yes |
| W-Shadow: `--wall shadow` commands equal `--wall off` | all identical | 12/12 | yes |
| W-B09: brain-09b engages >= 0.5 s before the graze, nothing toward the wall | lead >= 0.5 s; <= 0.1 m/s | 0.61 s (coast trigger); 0.00 m/s | yes |
| W-B08: brain-08 engages >= 0.5 s before its impact | lead >= 0.5 s | no engagement: an overshoot beside the arch at 6.4-7 m/s, predicted before the freeze | **no** |
| W-PD: vg-02 hairpin not pushed toward the wall, engaged <= 1 s, delay <= 1 s | <= 0.05 m/s; <= 1 s; <= 1 s | 0.00; 0.44 s; 0.38 s | yes |
| W-V3case: gapon-01 side push still removed | engaged; <= 0.1 m/s | 20.98 s; 0.04 m/s | yes |
| W-Quiet: no episodes on clean Straw and Pine | 0 | 0 | yes |

Report only: with a motor fitted to brain-09b's log v4 stops 0.29 m short; with the measured
braking it still grazes, at 1.27 m/s instead of 2.73. A lagged-motor surrogate reaches the
wall with every variant (v4 0.3-1.0 m less deep than v3).

**Gap pilot v5, side commitment** (`gap_pilot.json` v5 `43c30420`, gates v3 `e520b64e`, commit
`e51d3cf`; v2-v4 kept and refused). v3 was replaced unscored; v4 was frozen, scored and failed
A, B and S; v5 changes two v4 values after reading those results. v5 is therefore not held-out
evidence.

| Gate | Threshold | v2 | v5 | Pass |
|---|---|---|---|---|
| C pillar C (vg-02, development case) | no left shift; right >= 0.9 s before impact, held | 6 deg left for 103 ticks; right 0.39 s before | no left; right 0.99 s before, held to -12 deg | yes |
| A pillar A (25 approaches) | no later first left, no more right or into-pillar ticks than v2 | - | 25/25 (v4: 24/25) | yes |
| B pillar B (store runs 123, 127) | first -x target >= 1.0 s before impact | 1.25 / 0.95 s | 1.25 / 0.95 s (unchanged by design) | **no** |
| P pillar C passes (8 store runs) | no more left or into-pillar ticks than v2 | 0 / 0 | 0 / 0 | yes |
| S clean Straw laps (4 laps, 22.6 min) | episodes/min <= v2; quiet switches >= min(v2, 95%) | 8.87/min; 120/120 | 7.45/min; 115/120 (v4: 101/120) | yes |
| L leak test (G4 re-scored) | cue config unchanged, G4 reproduced | - | identical | yes |
| O identity (stack off, shadow) | bit-identical | - | 44/44 | yes |

The per-frame cue's own gates G1, G3 and G4 still fail (unchanged).

**View-keeping descent v1** (`descent_view.json` v1 `8afb64d7`, gates v1 `d2b1e4bb`, commits
`67d013b`/`7e96146`; off by default, `--descent-view on`). Surrogate with scoring-only hills,
held-out seeds, rule vs the current pilot on the same seeds.

| Gate | Threshold | PD | brain-08 | brain-09b |
|---|---|---|---|---|
| Contacts reduced, none new | >= 75% | -50% **no** | -69% **no** | -38% **no** |
| Crashes not more | <= baseline | 0/0 yes | 0/0 yes | 0/0 yes |
| Finishes not fewer | >= baseline | 28/28 yes | 28/28 yes | 28/28 yes |
| Path below the image while descending | <= 5% | 36.5% **no** | 53.3% **no** | 45.8% **no** |
| ... on hills <= 12 deg | <= 5% | 5.2% **no** | 20.0% **no** | 28.7% **no** |
| Passes > 1.5 m above a checkpoint | <= baseline + 1 | 2 -> 13 **no** | 1 -> 24 **no** | 3 -> 17 **no** |
| Course time (paired) | <= +3% | -9.1% yes | +3.5% **no** | +0.7% yes |
| Flat courses unchanged | +-2%, same finishes | -4.3% **no** | +0.1% yes | -0.5% yes |
| Default identity (rule off vs `935cfdb`) | bit-identical | 52/52 arrays, golden digest | | yes |

**brain-10** (`brain10_gates.json` v1 `7736f465`, commit `c51c2af`, before any candidate; G1-G6
copied from brain-09's gates). Six candidates; the rule ranks by gates passed, then chatter.

| Gate (fast-brain-10b) | Threshold | brain-08 / brain-09b | fast-brain-10b | Pass |
|---|---|---|---|---|
| G1 cap step | within cap+0.5 in <= 1 s; settled <= 0.3 | fail / pass | 0.79-0.89 s; -0.43..+0.05 | yes |
| G2 sustained | rollout(S) 3.5/4.5 within 0.3, 6 m/s in [-0.6, 0.3] | +1.30/+0.75/-0.47 / +0.16/+0.18/-0.54 | +0.05/+0.10/-0.44 | yes |
| G3 live swaps (9 Minus windows) | speed within 0.5 m/s of the PD's at +1 s | max 1.68 / 1.56 | max 0.89 (4/9 within) | **no** |
| G5 in-course caps | excess <= 0.4 m/s | +1.20 / +0.19 | +0.354 | yes |
| G6 regressions (16 courses) | >= 15 finished, <= 1 crash, chatter <= 0.0035, ... | 15/1, 0.00298 | 16/0, 0.00226 | yes |
| G7 smoothness | <= 1.15 x brain-08 (0.003427 / 0.005612 / 0.006727) | - / 0.00508 | 0.00226 / 0.00187 / 0.00179 | yes |
| G8 regressions | >= 15 finished; sink shortfall <= 0.25 | 15, 0.157 | 16, 0.127 | yes |

All six candidates fail G3, so none is selected. `fast-brain-10b` (sha256 `0ccf1161...`) passes
the other six. Only readout rows 0-2 and their biases changed (weights audit).

### Combined open-loop replays (development evidence)

Every listed Minus Two, Straw Bale and Pine Valley log was replayed three ways:

- through `m2-vertical`'s stack (gap pilot v2, wall pilot v3, vertical guard v2);
- through `m4`'s stack (`--stack on --near-on-path`, as the live camera's samples carry
  `near_on_path`);
- through the full round-4 stack (`m4` plus `--descent-view`).

The `m2-vertical` replays are round 4's vertical-guard baseline, taken read-only; their tree
matches `git archive m2-vertical` file for file. The Straw laps and `pine-brain08-01` flew
without looming and use the same offline looming stream in every variant. The recorded motion
does not respond to the requests: these are the requests each stack would have made at the
recorded states, not flights.

**Default and shadow unchanged:** 52 of 52 command-array pairs are bit-identical to
`m2-vertical`. That covers the default pilot on all 21 logs and on the 10 stream logs, and the
shadow stack on all 21.

Scripts and outputs are in the session scratchpad under `m4/integrate/`:
`run_replays.py`, `analyse_replays.py`, `downhill_view.py`, `replay_analysis.json` and
`downhill_view.json`.

| Log | Motor | Turn-first v4 (s, onsets) | Gap side commitment v5 | Guard climb s, v2 -> v3 (escalated s) | Descent view: limiting s / boost s / sink withheld m | Requests changed vs m2-vertical (s) |
|---|---|---|---|---|---|---|
| `minus-fast6-wall-01` | PD | - | - | 0.83 -> 0.83 (0 -> 0) | 0 / 0 / 0 | 0.11 |
| `minus-brain08-gapon-01` | brain-08 | - | left 11.98 s (1.28 s) | 0.84 -> 0 (0 -> 0) | 0.19 / 0 / 0.04 | 3.43 |
| `minus-brain08-gapon-02` | brain-08 | - | left 12.26 s, left 16.5 s (2.15 s) | 0.37 -> 0.37 (0 -> 0) | 0 / 0 / 0 | 3.85 |
| `minus-fast6-gapon-01` | PD | 1.07 (17.04, 18.48, 19.04, 20.98, 21.17) | left 12.46 s (1.19 s) | 2.22 -> 1.38 (0 -> 0) | 0 / 0 / 0 | 5.6 |
| `minus-brain08-loom-01` | brain-08 | - | - | 0 -> 0 (0 -> 0) | 0 / 0 / 0 | 0 |
| `pine-fast6-ttc-01` | PD | - | - | 7.77 -> 5.74 (4.02 -> 3.97) | 1.3 / 0.12 / 1.75 | 6.2 |
| `pine-brain08-loom-01` | brain-08 | - | - | 1.76 -> 0 (0 -> 0) | 0 / 0 / 0 | 1.61 |
| `minus-fast6-vg-02` | PD | 0.44 (20.56) | left 8.66 s, left 13.49 s, right 22.45 s (2.99 s) | 2.33 -> 0 (0.83 -> 0) | 0.28 / 0 / 0.19 | 7.08 |
| `minus-brain08-vg-01` | brain-08 | - | left 12.06 s (1.06 s) | 0.88 -> 0.75 (0 -> 0) | 0 / 0 / 0 | 1.97 |
| `minus-brain09b-vg-01` | brain-09b | 0.61 (21.29) | - | 0 -> 0 (0 -> 0) | 0 / 0 / 0 | 1.4 |
| `minus-brain08-gapshadow-01` | brain-08 | - | left 12.06 s (1.03 s) | 0 -> 0 (0 -> 0) | 0 / 0 / 0 | 0.16 |
| `straw-brain08-04` | brain-08 | - | - | 56.99 -> 9.55 (23.15 -> 0) | 46.26 / 7.81 / 57.84 | 93.66 |
| `straw-brain08-06` | brain-08 | - | - | 56.67 -> 11.21 (23.75 -> 0) | 46.17 / 8.93 / 58.09 | 98.5 |
| `straw-brain08-01` | brain-08 | - | - | 15.5 -> 4.61 (8.61 -> 0) | 7.69 / 1.16 / 4.27 | 18.6 |
| `straw-brain08-02` | brain-08 | - | - | 38.79 -> 7.44 (15.19 -> 0) | 15 / 2.62 / 19.28 | 44.91 |
| `straw-brain08-03` | brain-08 | - | - | 18.7 -> 4.07 (8.08 -> 0) | 0 / 0 / 0 | 10.5 |
| `straw-fast6-01` | PD | - | - | 11.36 -> 4.44 (0.81 -> 0) | 0.98 / 0.2 / 0.7 | 4.71 |
| `straw-fast6-02` | PD | - | - | 36.69 -> 13.39 (10.2 -> 0) | 55.76 / 11.39 / 71.52 | 103.13 |
| `straw-fast6-03` | PD | - | - | 34.73 -> 14.35 (9.53 -> 0) | 56.12 / 11.99 / 72.22 | 100.15 |
| `straw-fast6-arc-01` | PD | - | - | 34.39 -> 19.27 (8.98 -> 0) | 44.62 / 8.05 / 59.03 | 82.44 |
| `pine-brain08-01` | brain-08 | - | - | 6.16 -> 4.17 (1 -> 1) | 0 / 0 / 0 | 2.02 |

What changes per log:

- **Minus Two (9 logs).**
  - **Turn-first v4** engages only on the approaches where the motor slowed:
    - `vg-02` (PD): 0.44 s from 20.56 s;
    - `brain09b-vg-01`: 0.61 s from 21.29 s, on the coast trigger;
    - `fast6-gapon-01` (PD): five holds, 1.07 s in all. Three are short coast holds before the
      arch (17.0-19.1 s), exactly as in the hairpin branch's own replay.

    The brain-08 logs never slowed enough for turn-first to engage.
  - **Gap v5** commits left at pillar A (x about 49) on six of the seven logs that carry live gap
    samples and reach it. The seventh, `brain09b-vg-01` (1.2 m clear when flown), shifts without
    committing. `vg-02` also commits left for 1 s at x 24, and commits right at pillar C
    (22.45 s).
  - **Vertical guard v3** drops the gentle climbs that version 2 started on the lower window alone:
    - `vg-02`: 2.3 s -> 0;
    - `gapon-01`: 0.8 s -> 0;
    - `fast6-gapon-01`: 2.2 s -> 1.4 s.

    It keeps the 1 m/s climb at `wall-01`'s floor sink.
  - **Descent view** acts for at most 0.28 s per log and never boosts speed on Minus Two.
- **Pine Valley.**
  - On `pine-fast6-ttc-01` the mound escalation is unchanged: 3.97 s against 4.02 s, up to
    3.5 m/s. The guard's total climb falls from 7.8 s to 5.7 s. The descent view reshapes the
    mound's backside: 1.3 s limiting and 1.75 m of sink withheld (see below).
  - `pine-brain08-01` keeps its 1 s escalation.
  - `pine-brain08-loom-01` loses a 1.8 s gentle climb.
- **Straw Bale (9 laps).**
  - Turn-first never engages.
  - These laps flew without the stack, so they carry no live gap samples. The gap aim's Straw
    behaviour is only the pillar branch's gate S: 7.45 episodes per minute, and 5 of 120
    switches with more than 4 deg.
  - The guard climbs 4-19 s per log instead of 11-57 s, never above 1 m/s (0 s escalated).
  - The descent view limits sink for 45-56 s of each three-lap log. It raises speed for 8-12 s
    and withholds 58-72 m of requested sink.

**The Straw Bale downhill with the whole stack** (x -50 to -25, y 120 to 195, flying toward -y;
recorded attitude; exact projection through the calibrated camera; `descent_replay`'s
definitions):

- A contact here is a support-climb onset of the logged pilot on the downhill.
- The fast-PD laps and `straw-brain08-01`/`-02` flew earlier pilots, so their replay does not
  reproduce the log exactly.
- `straw-brain08-03` never reached the downhill.

| Log | Downhill s (contacts) | Request below the image: m2-vertical stack / m4 stack / view rule alone / full m4 | 3 s before contacts: m2-vertical -> full m4 | Requested speed m/s | Requested vz m/s | Requested sink m |
|---|---|---|---|---|---|---|
| `straw-brain08-04` | 55.1 (4) | 51% / 51% / 8% / 8% | 59-82% -> 0-12% | 4.38 -> 5.87 | -0.74 -> -0.48 | 43.7 -> 29.3 |
| `straw-brain08-06` | 54.9 (6) | 51% / 51% / 7% / 7% | 58-82% -> 0-15% | 4.33 -> 5.83 | -0.74 -> -0.49 | 44.2 -> 30.3 |
| `straw-brain08-01` | 11 (3) | 12% / 12% / 0% / 0% | 0-45% -> 0% | 5.31 -> 5.78 | -0.35 -> -0.47 | 5.1 -> 5.4 |
| `straw-brain08-02` | 18 (2) | 53% / 53% / 8% / 8% | 61-82% -> 0-13% | 4.34 -> 5.84 | -0.77 -> -0.51 | 14.8 -> 10.2 |
| `straw-fast6-01` | 3.5 (0) | 24% / 24% / 22% / 22% | - | 5.04 -> 5.56 | -0.36 -> -0.29 | 1.2 -> 1.0 |
| `straw-fast6-02` | 53.7 (0) | 62% / 62% / 34% / 35% | - | 3.42 -> 5.85 | -1.06 -> -0.88 | 56.9 -> 47.1 |
| `straw-fast6-03` | 54.6 (0) | 66% / 66% / 39% / 39% | - | 3.22 -> 5.85 | -1.13 -> -0.96 | 61.8 -> 52.5 |
| `straw-fast6-arc-01` | 51.8 (0) | 61% / 61% / 34% / 34% | - | 3.52 -> 5.93 | -1.07 -> -0.87 | 55.5 -> 45.1 |

**Interactions between the new rules** (open-loop replays of the same 21 logs; each overlap is
measured in the full-stack arrays):

- **Hairpin turn-first and pillar side commitment: never at the same time** (0 s on every log).
  On `minus-fast6-vg-02` turn-first holds 20.56-21.0 s and the right commitment starts at
  22.45 s. Both happen exactly as each branch reported alone: turn-first 0.44 s, and the right
  commitment 0.99 s before the impact, held to -12 deg. On `minus-brain09b-vg-01` turn-first
  engages at 21.29 s (lead 0.61 s) and no commitment is active.
- **Vertical guard v3 and the gap aim's terrain votes.** v3 no longer starts the gentle floor
  climb that let v2's terrain vote steer toward pillar C. Its guard climb on `vg-02` falls from
  2.3 s to 0. With `terrain_rising_only`, gap terrain votes were 0 s on every log.
  - Replaying `vg-02` with gap pilot **v2** under guard v3 gives the same pillar-C requests: no
    terrain vote, no left shift, and the right side from 22.45 s (0.99 s before the impact).
  - Gap pilot v5 under guard v2 also removed the left shift (the pillar branch's gate C).
  - The pillar-C fix is therefore over-determined in replay. A live pass would not show which
    change it owes to.
- **Descent view and the vertical guard on the Straw downhill.** Both only reduce the pilot's
  sink, and the smaller sink wins. They bound it together for 0-2.5 s of the 52-55 s that each
  three-lap log spends on the downhill. No tick had one climbing while the other bounded sink.
  On the downhill, the lowest descent-governor scale is 0.75-1.0 with the view rule (its floor
  is 0.75) and 0.35-1.0 without it.
- **Descent view, governor caps and turn-first on Minus Two.**
  - The view rule acts at most 0.28 s per Minus log, and its speed boost never acts there.
  - It never acts inside a turn-first episode.
  - Where a looming cap was active, it asked for up to 0.55-0.68 m/s more horizontal speed
    (`gapon-02`, `vg-02`, x 71-78 before the hairpin). That is keep-speed: no half-speed brake
    for a ring clipped at the bottom, near the floor.
  - The cap itself and turn-first act after it in the pilot, so they still bound the request.
- **Descent view on Pine Valley.** This is the largest change outside Straw. On the mound's
  backside (`pine-fast6-ttc-01`, 6.3-7.4 s) the stack without the view rule asks for up to
  2.1 m/s of sink at about 2.5 m/s; the full stack asks for 0.3 m/s at 5.2-5.7 m/s. The mound
  climb, the hillside at 15.2 s and the end are unchanged.

### Surrogate (development evidence)

The loop is the descent gates' surrogate: IdentifiedSim, 10% per-drone randomisation, a
synthetic HUD marker, a three-tick command delay, 150 s per course, sim seed 17 and the gates'
profile. Three course sets were flown:

- the 16-course development gate: flat and steep seeds 3000-3007, the steep ones with
  scoring-only hills;
- the descent terrain courses: hill seeds 6000-6011.

Four pilot variants were compared:

- the current pilot;
- the current pilot with the descent view;
- the round-4 stack as the runner builds it for the motor contract;
- the full round-4 pilot (stack plus descent view).

The surrogate has no looming or gap samples. The gap aim, turn-first and the vertical guard
therefore stay idle there; only the lag-aware turns and the descent view act.

The current-pilot and descent-view runs of the PD, brain-08 and brain-09b are the descent
branch's gate runs, reused read-only. This tree reproduces them exactly: a rerun of the PD's
flat baseline matched. The metrics and thresholds are the frozen descent gates' definitions.
They are the frozen gates only for the descent-view-vs-current comparisons of the three motors
the gates name. Everything else, including fast-brain-10b, is a report.

| Motor | Variant | 16-course dev gate (flat+steep): finished / crashed | Chatter (16) | Terrain sets (steep+hill): contacts / contact s | Path below image while descending | High passes (all) | Mean finish s (steep / hill) |
|---|---|---|---|---|---|---|---|
| fast PD | current pilot | 16/16, 0 | 0.00576 | 22 / 97.2 | 48.6% | 2 | 68.99 / 55.19 |
| fast PD | + descent view | 16/16, 0 | 0.00577 | 11 / 15.2 | 36.5% | 13 | 64.04 / 49.31 |
| fast PD | + round-4 stack | 16/16, 0 | 0.00646 | 21 / 60.8 | 45.0% | 6 | 65.07 / 54.34 |
| fast PD | full round-4 pilot | 16/16, 0 | 0.00634 | 11 / 15.6 | 33.5% | 14 | 66.93 / 49.25 |
| brain-08 | current pilot | 16/16, 0 | 0.00315 | 26 / 105.2 | 56.0% | 1 | 71.6 / 61.25 |
| brain-08 | + descent view | 16/16, 0 | 0.00334 | 8 / 21.5 | 53.3% | 24 | 76.56 / 61.81 |
| brain-08 | full round-4 pilot | 15/16, 1 | 0.00373 | 8 / 22.0 | 53.9% | 26 | 79.96 / 62.74 |
| brain-09b | current pilot | 16/16, 0 | 0.00533 | 26 / 66.0 | 53.5% | 3 | 68.98 / 58.5 |
| brain-09b | + descent view | 16/16, 0 | 0.00533 | 16 / 23.2 | 45.8% | 17 | 71.2 / 57.75 |
| brain-09b | full round-4 pilot | 16/16, 0 | 0.00561 | 14 / 19.2 | 45.7% | 18 | 70.29 / 58.0 |
| fast-brain-10b | current pilot | 16/16, 0 | 0.00240 | 23 / 104.2 | 64.1% | 1 | 73.59 / 62.67 |
| fast-brain-10b | + descent view | 16/16, 0 | 0.00260 | 9 / 19.0 | 50.0% | 26 | 73.1 / 60.24 |
| fast-brain-10b | + round-4 stack | 16/16, 0 | 0.00267 | 25 / 96.4 | 62.9% | 1 | 72.63 / 62.58 |
| fast-brain-10b | full round-4 pilot | 16/16, 0 | 0.00289 | 12 / 19.7 | 49.6% | 29 | 72.75 / 60.28 |

Findings:

- **The 16-course development gate is almost unchanged.** Every motor finishes 16/16 with every
  variant, with one exception: brain-08 under the full pilot crashed on steep seed 3003, after
  44 s of search. brain-08 had no crash with the descent view alone. The PD, brain-09b and
  fast-brain-10b have no crash with any variant.
- **The stack alone changes little in the surrogate, because only the lag-aware turns can act
  there.** Stick chatter rises 11-12% (PD 0.00576 -> 0.00646; fast-brain-10b 0.00240 ->
  0.00267). Contacts barely move (PD 22 -> 21; fast-brain-10b 23 -> 25).
- **The full pilot keeps the descent view's effect and its failures on every motor.**
  - Contacts fall by 46-69%, against the 75% the gate asks.
  - Passes more than 1.5 m above a checkpoint rise from 1-3 to 14-29.
  - The path points below the image for 33-54% of the descent time.
  - brain-08's paired course time is +5.2%.
  - No motor passes the descent gates' definitions with any variant.
- **fast-brain-10b under the full pilot** finishes 16/16 with no crash. Contacts fall from 23 to
  12 and high passes rise from 1 to 29. Its chatter, 0.00289, is still below brain-08's 0.00315
  and brain-09b's 0.00533 on the current pilot. Like the other brains, it does not fly the rule's
  in-view descents: it sinks less than asked and passes rings high.
- **Search time rises** on the terrain sets for fast-brain-10b (hill: 88 s -> 130 s) and
  brain-08. These are missed rings and turn-backs, as the flight plan warns for Straw.

Scripts and results are in the session scratchpad under `m4/integrate/`: `surrogate.py`,
`score_surrogate.py`, `surrogate/` and `surrogate_scores.json`. They are also in
`docs/experiments/round4_integration.json`.

Each row below compares two pilot variants using the descent gates' definitions: "full" is the
round-4 stack plus the descent view, "dv" is the descent view alone, and "stack" is the stack
without it. Only the dv-versus-current rows of the fast PD, brain-08 and brain-09b are the frozen
gates' own scoring.

| Motor | Comparison | contacts | crashes | finishes | below view | below view <= 12 deg | high passes | course time | flat | all |
|---|---|---|---|---|---|---|---|---|---|---|
| fast PD | dv vs baseline | 22->11 (-50%) **no** | 0->0 yes | 28->28 yes | 36.5% **no** | 5.2% **no** | 2->13 **no** | -9.1% yes | -4.3% **no** | **no** |
| fast PD | stack vs baseline | 22->21 (-4%) **no** | 0->0 yes | 28->28 yes | 45.0% **no** | 4.3% yes | 2->6 **no** | -3.4% yes | -1.8% yes | **no** |
| fast PD | full vs baseline | 22->11 (-50%) **no** | 0->0 yes | 28->28 yes | 33.5% **no** | 5.2% **no** | 2->14 **no** | -7.2% yes | -4.6% **no** | **no** |
| fast PD | full vs stack | 21->11 (-48%) **no** | 0->0 yes | 28->28 yes | 33.5% **no** | 5.2% **no** | 6->14 **no** | -3.9% yes | -2.8% **no** | **no** |
| brain-08 | dv vs baseline | 26->8 (-69%) **no** | 0->0 yes | 28->28 yes | 53.3% **no** | 20.0% **no** | 1->24 **no** | +3.5% **no** | +0.1% yes | **no** |
| brain-08 | full vs baseline | 26->8 (-69%) **no** | 0->1 **no** | 28->27 **no** | 53.9% **no** | 23.3% **no** | 1->26 **no** | +5.2% **no** | +0.8% yes | **no** |
| brain-09b | dv vs baseline | 26->16 (-38%) **no** | 0->0 yes | 28->28 yes | 45.8% **no** | 28.6% **no** | 3->17 **no** | +0.7% yes | -0.5% yes | **no** |
| brain-09b | full vs baseline | 26->14 (-46%) **no** | 0->0 yes | 28->28 yes | 45.7% **no** | 29.0% **no** | 3->18 **no** | +0.4% yes | -0.4% yes | **no** |
| fast-brain-10b | dv vs baseline | 23->9 (-61%) **no** | 0->0 yes | 28->28 yes | 50.0% **no** | 18.3% **no** | 1->26 **no** | -2.5% yes | -0.2% yes | **no** |
| fast-brain-10b | stack vs baseline | 23->25 (+9%) **no** | 0->0 yes | 28->28 yes | 62.9% **no** | 28.2% **no** | 1->1 yes | -0.6% yes | -0.0% yes | **no** |
| fast-brain-10b | full vs baseline | 23->12 (-48%) **no** | 0->0 yes | 28->28 yes | 49.6% **no** | 22.1% **no** | 1->29 **no** | -2.6% yes | -0.4% yes | **no** |
| fast-brain-10b | full vs stack | 25->12 (-52%) **no** | 0->0 yes | 28->28 yes | 49.6% **no** | 22.1% **no** | 1->29 **no** | -2.0% yes | -0.3% yes | **no** |

### Live flight plan (for the main session; development flights)

Fly from `C:\DEV\Haltere` with branch `m4` checked out. The runner loads its declarations from
the checkout it runs from and refuses other versions. Keep the untracked
`configs/explore_spiral.yaml`.

The flight procedure is unchanged:

- Liftoff and every capture and controller process run inside Anode, with the viewer hidden.
- The pad must report `seatOnly`.
- Run the ground check, and verify throttle-low and a real processed control response.
- Nothing else heavy may run: no training, pytest or agent job. The preflight refuses a busy
  machine.
- Use a new log name for every attempt, and keep Liftoff open between runs.

Every run declares the same stack; nothing is set per course. Check these in each sidecar
(`pilot_assistance.*_declaration` and `obstacle_stack.gap_cue`) before reading the result.

| Declaration | Version | Content sha256 |
|---|---|---|
| `configs/obstacles/lag_turn.json` | 2 | `d4eb83da51ab...` |
| `configs/obstacles/gap_pilot.json` (gap cue `gap_cue.json` v2 `284b3c46a819...`) | 5 | `43c304204f93...` |
| `configs/obstacles/wall_pilot.json` (stopping model: the motor's contract) | 4 | `92f842a54e56...` |
| `configs/obstacles/vertical_guard.json` | 3 | `b70e263ccdc5...` |
| `configs/pilot/descent_view.json` (`--descent-view on`) | 1 | `8afb64d730ad...` |

**Every run below is a disclosed development deviation:**

- descent view v1 failed its frozen surrogate gates;
- gap pilot v5 was chosen after its scored v4 failed, and it fails B;
- turn-first v4 fails W-B08;
- vertical guard v3 fails V-Pine and V-Straw's limited-before-contact gate;
- no brain passed its own gates.

The brain candidate is brain-09b, because brain-10 was not selected. fast-brain-10b is the
alternative if the user accepts it as a deviation. Its surrogate chatter is about half of
brain-09b's, but its brake onset is about 0.3 s later.

The matched control for any run is the same command with `--obstacle-stack shadow`, keeping
`--descent-view on`. It is not required for this development round.

Run the four flights in this order. Each command is one line; run it from `C:\DEV\Haltere`. The
limits are the ones earlier fast-stack flights recorded, and the video uses NVENC as before.

**(a) Minus Two, fast PD.** This is the lowest-risk first look at every round-4 rule together.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-08-vgs04-s10r03m30/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller pd --pd-profile fast --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --descent-view on --seconds 150 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/minus-fast6-r4-01.csv --record runs/fast-stack-20260923/minus-fast6-r4-01.mp4 --video-encoder h264_nvenc
```

What to look for. The requests below come from the open-loop replay of `minus-fast6-vg-02` at its
recorded states. Where the drone ends up is a flight question.

- **Pillar A:** passed on the left, with a left gap commitment (`gap_commit` 1) near x 49
  (13.5 s). The replay also commits left for 1 s at x 24 (8.7 s).
- **Hairpin:** a turn-first episode of about 0.4 s near 20.5 s (`turn_first` 1), at creep, then
  the turn north with no contact.
- **Floor after the hairpin:** no vertical-guard climb (`vertical_climb` 0; v2 climbed 1 m/s
  here) and no left terrain steer.
- **Pillar C:** a right commitment (`gap_commit` -1) about 1 s before the pillar, holding up to
  -12 deg. The drone should pass on the right, where the store runs passed (x 79.3-80.7).
- **Garage floor:** the descent view may bound sink to about 0.3 m/s and keep 0.2-0.7 m/s more
  speed where the ring clips below. It must never act inside a turn-first episode (it did not in
  any replay).
- Beyond pillar C, the course has not been flown with this stack.

**(b) Minus Two, brain-09b** (the brain candidate; brain-10 was not selected).

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-09b-caps-r0001m100/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller brain --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --descent-view on --seconds 150 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/minus-brain09b-r4-01.csv --record runs/fast-stack-20260923/minus-brain09b-r4-01.mp4 --video-encoder h264_nvenc
```

To fly fast-brain-10b instead (a deviation the user must accept), use
`runs/fast-brain-10b/candidate.pt` and the log stem `minus-brain10b-r4-01`. It has the same
contract, and the same stack is declared.

What to look for:

- **Hairpin arrival:** the speed toward the wall (+x) when passing the arch near (79.7, 18.7).
  About 2.6-3.0 m/s or less is needed to stop in the 2.1 m after it; brain-09b arrived at 3.2.
- **Turn-first:** it should engage with the coast trigger about 0.6 s before the old graze point
  (replay: 21.29 s) and hold the horizontal request at 0.
- **Graze or pass** at about (81.8, 19.9).
- **Stick change per tick:** brain-09b logged 0.0056 live, brain-08 0.0031.
- If it gets through: pillar C as in (a).

**(c) Straw Bale, brain-09b, full stack.** This is the Straw regression and the user's downhill
request.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-09b-caps-r0001m100/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller brain --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --descent-view on --seconds 480 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/straw-brain09b-r4-01.csv --record runs/fast-stack-20260923/straw-brain09b-r4-01.mp4 --video-encoder h264_nvenc
```

What to look for:

- **Ground contacts:** `support_climb` onsets, especially on the downhill (x about -37, y
  144-166). The two brain-08 finishes had 4 and 6.
- **Downhill speed:** the drone should now hold about 5.5-6 m/s instead of braking to about 3.
- **Downhill risks:**
  - rings passed high, or the top frame of a descending ring hit. In the surrogate under the
    full pilot, brain-09b's contacts fell from 26 to 14 but its high passes rose from 3 to 18.
  - misses and turn-backs.

  If these appear, the next step is a brain distilled under the view rule, not more flights of
  this one.
- **The stack must stay quiet on Straw.** In replay:
  - turn-first had 0 episodes;
  - the vertical guard never climbed above 1 m/s (gentle climbs of 4-19 s per three-lap log);
  - in the pillar branch's cue-level replay of clean Straw laps, the gap aim had 7.45 episodes
    per minute, and 5 of 120 checkpoint switches kept more than 4 deg (arch or bale clips are
    possible).
- **Hilltop:** the brain-08 hilltop crashes came after the ring was lost there. No bottom-clip
  brake happens there any more.
- The lap times against brain-08's 5:17.

**(d) Pine Valley, fast PD.**

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-08-vgs04-s10r03m30/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller pd --pd-profile fast --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --descent-view on --seconds 150 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/pine-fast6-r4-01.csv --record runs/fast-stack-20260923/pine-fast6-r4-01.mp4 --video-encoder h264_nvenc
```

What to look for:

- **Mound climb:** must still happen, with the guard escalating near 4.5 s and climbing up to
  3.5 m/s.
- **Mound backside:** the descent view changes this most on Pine. At 6.3-7.4 s the default stack
  asked for up to 2.1 m/s of sink at about 2.5 m/s; the full stack asks for 0.3 m/s at 5.2-5.7
  m/s and steepens only after 0.75 s of bottom clip. Watch for a ring below being overflown and a
  turn-back.
- **Hillside at 15.2 s:** v3 answers with only 1 m/s, where the flight climbed at 3.5 m/s.
  Hillside contact is possible.
- **Boulder at the end:** no rule arrests that descent.

**Stop criteria.**

- **Before a run:** do not start if any of these hold:
  - the preflight is not quiet;
  - the pad is not `seatOnly`, or the guard unplugs it (a user's game alone is not a reason);
  - the ground check shows no real processed response or no throttle-low.
- **After a run:** stop the series if any of these hold:
  - the sidecar declares any other version or hash than the table above;
  - a controller deadline or camera failure was recorded.
- **During a run:** the runner ends it on impact, on stale telemetry and at the limits, and
  `--pause-on-stop` pauses the game on exit. Stop the runner if any of these happen:
  - the drone climbs into a ceiling or keeps climbing above about 4 m on Minus Two;
  - a turn-first hold keeps the drone parked for more than about 3 s;
  - it circles in search for more than about 30 s.
- **Between runs:** stop and analyse instead of flying the next run if a round-4 rule causes a
  new failure: a hold or commitment that steers into an obstacle, a climb the old stack did not
  make, or a stall. Replay the log first with
  `haltere/obstacles/vertical_replay.py --stack flown ...`. A crash at a place the old stack
  also crashed does not stop the series: (b) and (c) test different rules from (a).
- **Graduation evidence:** only complete runs count. None of these runs does, because the brain
  is not selected.

### Blockers for graduation, and what would clear them

| Blocker | State after round 4 | What would clear it |
|---|---|---|
| A selected brain | None: brain-10 candidates all fail G3; brain-09b fails G3/G6 | A G3 v2 justified from the latency evidence and frozen before any scoring, or a candidate that passes G3; then flights |
| Straw downhill ground contact | Descent view v1 fails its gates (contacts -38..-69% vs -75%, high passes up); replay puts requests in view before every logged contact; brains do not fly its 6 m/s descents | A brain distilled with the view rule in its DAgger rollouts (scored on the frozen descent gates), then Straw 3/3 with no support climb on the downhill |
| Minus pillar C | Gap v5 holds the right side 0.99 s early in replay (development data); pillar B fails at pilot level (0.95 vs 1.0 s) | Live passes of (a) and (b); a held-out obstacle set for the next gap-pilot version |
| Minus hairpin for braking brains | Turn-first v4 engages 0.61 s before brain-09b's graze; the lagged surrogate still reaches the wall | Approach speed: a governor cap that accounts for motor delay, or a brain that follows caps within about 0.3 s |
| Vertical guard on Straw and Pine | Straw escalations 0 s (fixed); V-Pine fails (hillside 1 m/s vs 3.5 flown; boulder descent not arrested) | Closed-loop evidence or a new causal signal (vertical speed at each sample's capture) |
| brain-09 chatter | fast-brain-10b 0.00226 per tick in the surrogate (brain-09b 0.00508); not flown | A live flight of the selected brain |

## Round 4 (live), 2026-09-27: Minus Two with the m4 stack

Branch `m4` at `9a41acb`: `--looming-brake --obstacle-stack on --descent-view on` (wall pilot v4,
vertical guard v3, gap pilot v5, lag turn v2, descent view v1), original `[Copy] New Drone`, Anode
seat with the viewer hidden by us, pad `seatOnly` (preflight and postflight `pads_seat_only` true),
ground check `ground-check-30`/`-31` on pad 29. Every run is a disclosed development deviation: no
round-4 rule passed all of its gates, and no brain is selected.

| Run | Motor | Outcome |
|---|---|---|
| `minus-fast6-r4-01` | fast PD | Runtime stop, 0 ticks: "No fresh live image/telemetry". The ground-check script pauses the game when it ends and the run was started without resuming (operator error; the camera delivered 213 frames). Retried unchanged |
| `minus-fast6-r4-02` | fast PD | **Pillar A** (y 5.48), **hairpin** (exit at 20.4 s), **pillar C** (y 31.5 at 22.6 s, where round 3 ended) and on to (79, 58): the furthest any 6 m/s run has flown on Minus Two. From 26.5 s the pilot followed rings standing on the garage floor down to z 0.01-0.3 m (floor contact at 27.4-28.9 s: z < 0.1 m for 1.5 s; the support climb never fired because the descent view held the sink request above -0.8 m/s, the review's major finding). At (75.9, 64.3), 0.5 m above the floor, the looming lower window read the nearby floor as rising ground (ttc_lower 0.44-1.0 s, below fraction 0.85-1.0): the guard's gentle climb (1 m/s) was escalated at 33.0 s to 3.5 m/s, the overhead cut came at 33.4 s at 2.1 m/s of climb, and the drone struck the ~2.2 m garage ceiling at (73.5, 69.4, 2.13), 34.1 s |
| `minus-brain10b-r4-01` | fast-brain-10b | Runtime stop, 0 ticks: same paused-game operator error. Retried unchanged |
| `minus-brain10b-r4-02` | fast-brain-10b (`0ccf1161…`) | Pillar A (y 5.84). Stick change 0.0031 per tick (brain-08's live value). It braked to 2.8-2.9 m/s under 2.7-2.9 m/s caps at both arches. At the hairpin, while a false vertical-guard terrain climb (vertical_stage 1, 20.15-21.4 s, set off by the floor below the path: below fraction 0.84, ttc_lower 0.48-0.62 s at 0.85 m) overrode the pilot's descent and lifted it from 0.85 to 1.9 m over the ring, it held 4.9-5.3 m/s under 3.6-4.0 m/s caps; above turn-first's 3.5 m/s limit, so no episode; hairpin wall at (81.9, 19.0), 21.6 s, ~4.9 m/s. Its cap miss is therefore confounded with the guard climb |
| `minus-brain09b-r4-01` | fast-brain-09b (round-3 candidate) | Pillar A (y 5.66). **Turn-first v4 engaged live** (coast + stopping triggers) at 22.8 s and stopped the drone at (81.1, 20.0), 0.5 m/s, short of the wall where brain-09b grazed in round 3; released aligned after 0.41 s. Accelerating out of the turn toward a 4.9 m/s request, the brain sank from 0.78 to 0.06 m at up to 1.9 m/s while the vertical request was +0.06..+0.16 m/s, and hit the floor at (80.4, 20.9), 23.7 s. Stick change 0.0063 per tick |

What this shows:

- The pillar C fix and the hairpin rule work live with the fast PD, and turn-first v4 stops a braking
  brain at the hairpin wall. Neither is yet a clean pass for a brain.
- New failure: the guard's rising-ground confirmation is fooled by a flat floor close below the path
  (the looming lower-window TTC of a floor 0.5-0.9 m below a level path at 4-5 m/s is 0.6-1 s, and its
  own gentle climb satisfies the "climbing" condition). Escalated climbs need a test that the floor
  keeps approaching as the drone rises, or a bound under an overhead.
- The descent view's suppression of the support climb (review finding) showed up live on the Minus
  floor, not only on Straw.
- fast-brain-10b does not follow governor caps live (+1.5 m/s); fast-brain-09b brakes but loses height
  when it accelerates hard from low speed. Neither is a release brain.

Procedure note: the ground-check script pauses the game at exit; reset (Réinitialiser) or resume, and
confirm telemetry is streaming, before launching a run.
