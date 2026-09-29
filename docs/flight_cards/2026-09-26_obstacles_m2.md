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

[brain-10](../fast_brain_10_candidate.md) was distilled under the current pilot, without the descent
view: the descent branch was not flight ready and was not merged on brain-10's branch. `m4` now merges
it, off by default (`--descent-view on`).

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
Gates v3 also redefined the V-Pine climb criterion in a more lenient direction and added the
mound gate (both disclosed in the gates file and in `docs/vertical_guard.md`): the denominator
narrowed from every second the logged governor climbed to the seconds it asked for at least
1 m/s, which moves the flown log itself from 72.4% to 91.4%. V-Pine fails under both definitions
(v3: 77.7% new, 71.8% old; v2: 77.7% new, 73.8% old), so no verdict changes. v3's binding
threshold of 0.5 m/s lies between the Pine mound's 0.44 m/s and the lowest Straw escalation's
0.61 m/s, measured on the same logs that score V-Straw uphill and the V-Pine mound: those two
passes are in-sample, with small margins. The score file of v2 under gates v3
(`vertical_guard_v2_under_gates_v3_scores.json`) named guard v3 in its header; round 4b corrected
the header to the guard actually replayed (v2, `e06b690d`).

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
`6418aea5`, commit `b36bab4`; v3 kept and refused). The scorer's key fix
(`wall_pilot_gates.py`: `gates['flights']['logged_looming']` -> `['logged']`, the only key the
frozen gates have) went into the results commit `0d1010a` after the scores were produced; the
scorer committed at the freeze would have stopped with a KeyError. Thresholds and definitions are
unchanged, and the audit's re-score with m4's code reproduced every verdict and value.

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
`e51d3cf`; v2-v4 kept and refused). v3 was replaced unscored: a unit test found a flaw before
any gate was scored, but its candidate replay files had already been generated (about
01:45-01:47) and were deleted, by the pillar branch's account unread (this cannot be verified;
v3 and its gates v1 were first committed together with v4). v4 was frozen, scored and failed
A, B and S; v5 changes two v4 values after reading those results. v5 is therefore not held-out
evidence. The notes of gates v2 and v3 still say they were frozen "together with the version 3
declaration" (a copy slip in frozen files; their `candidate_version` fields, 4 and 5, are
right).

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
  - **Descent view.** Its bound limits sink for at most 0.28 s per log and its speed boost never
    acts on Minus Two, but it changes the request for longer: by more than 0.05 m/s for 0.85 s
    on `vg-02` (17.85-18.64 s in `below`, up to 0.72 m/s horizontal and 0.58 m/s vertical),
    0.79 s on `brain09b-vg-01` (7.02-7.63 s in `cue` near x 19, up to 0.46 m/s, with both flags
    at 0) and 0.60 s on `gapon-02` (up to 0.56 m/s), through its gentler sink onset and the
    keep-speed floor of the descent-path governor. Inside turn-first episodes the differences
    stay below 0.005 m/s.
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
  - The view bound limits sink at most 0.28 s per Minus log and its speed boost never acts
    there; the request itself changes by more than 0.05 m/s for up to 0.85 s per log (above).
  - It never acts inside a turn-first episode.
  - Where a looming cap was active, it asked for up to 0.55-0.68 m/s more horizontal speed
    (`gapon-02`, `vg-02`, x 71-78 before the hairpin). That is keep-speed: no half-speed brake
    for a ring clipped at the bottom, near the floor.
  - The cap itself and turn-first act after it in the pilot, so they still bound the request.
- **Descent view on Pine Valley.** This is the largest change outside Straw. On the mound's
  backside (`pine-fast6-ttc-01`, 6.3-7.4 s) the stack without the view rule asks for up to
  2.1 m/s of sink at about 2.6 m/s; the full stack asks for 0.3 m/s at 5.1-5.7 m/s until 7.17 s,
  then steepens to 0.54 m/s at 5.1 m/s by 7.37 s (0.3-0.54 m/s of sink at 4.9-5.7 m/s over the
  window). The mound climb, the hillside at 15.2 s and the end are unchanged.

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
| brain-09b | dv vs baseline | 26->16 (-38%) **no** | 0->0 yes | 28->28 yes | 45.8% **no** | 28.7% **no** | 3->17 **no** | +0.7% yes | -0.5% yes | **no** |
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
`--descent-view on`, so that it isolates the obstacle stack. It is not required for this
development round. (The round-4 wiring check ran this control without `--descent-view`:
`round4_integration.json` `plan_wiring.shadow_control_minus_pd` has `descent_view` null. The
text above is the intended control; round 4b's wiring check covers it with the view rule on.)

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
  asked for up to 2.1 m/s of sink at about 2.6 m/s; the full stack asks for 0.3-0.54 m/s at
  4.9-5.7 m/s, steepening after 7.2 s (0.75 s of bottom clip). Watch for a ring below being overflown and a
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
| Straw downhill ground contact | Descent view v1 fails its gates (contacts -38..-69% vs -75%, high passes up); in replay the requests point below the image 0-15% of the 3 s before the logged contacts (default pilot 58-83%); brains do not fly its 6 m/s descents | A brain distilled with the view rule in its DAgger rollouts (scored on the frozen descent gates), then Straw 3/3 with no support climb on the downhill |
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
| `minus-fast6-r4-02` | fast PD | **Pillar A** (y 5.48), **hairpin** (exit at 20.4 s), **pillar C** (y 31.5 at 22.6 s, where round 3 ended) and on to (79, 58): the furthest any 6 m/s run has flown on Minus Two. From 26.5 s the pilot followed rings standing on the garage floor down to z 0.01-0.3 m (floor contact at 27.4-28.9 s: z < 0.1 m for 1.5 s; the support climb never fired because the sink request stayed above -0.8 m/s; round 4b traced that sink to the clearance brake, not the descent view, see below). At (75.9, 64.3), 0.5 m above the floor, the looming lower window read the nearby floor as rising ground (ttc_lower 0.44-1.0 s, below fraction 0.85-1.0): the guard's gentle climb (1 m/s) was escalated at 33.0 s to 3.5 m/s, the overhead cut came at 33.4 s at 2.1 m/s of climb, and the drone struck the ~2.2 m garage ceiling at (73.5, 69.4, 2.13), 34.1 s |
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
  floor, not only on Straw. **Corrected in round 4b:** the replay shows the pilot asked for +0.1 to
  +0.23 m/s there and the descent view withheld nothing. The sink (-0.3 to -0.5 m/s) came from the
  clearance brake along a stand-off ray tilted 11 deg up. With a sink that small, no support rule could
  fire. See `docs/clearance_brake.md` and Round 4b below.
- fast-brain-10b braked for the caps at both arches but missed them at the hairpin (+1.0 to +1.7 m/s)
  during a false guard climb, so whether it misses caps in general is untested; fast-brain-09b brakes but
  loses height when it accelerates hard from low speed. Neither is a release brain.

Procedure note: the ground-check script pauses the game at exit; reset (Réinitialiser) or resume, and
confirm telemetry is streaming, before launching a run.

## Round 4b (offline): the merged m4b stack, replays, surrogate and the live plan

**Nothing in this section has flown.** Branch `m4b` is `m4` (`c953758`) with the four round-4b
branches merged: `m4b-guard` (vertical guard v4), `m4b-contact` (descent view v2 with contact
support, wall pilot v5 with the clearance-brake sink floor, contact audit v1), `m4b-assist` (motor
assist v1, off by default) and `m4b-brain11` (brain-11 training options, frozen gates, five
candidates, none selected). Every round-4b branch read the three round-4 live logs
(`minus-fast6-r4-02`, `minus-brain10b-r4-02`, `minus-brain09b-r4-01`) to design its rule, so every
replay of them below is **development evidence**, not held-out evidence.

**Merges.** `m4b-guard` and `m4b-contact` merged without conflicts. `m4b-assist` conflicted with
`m4b-contact` in `fast_race_cue.py`, `visual_brain.py` and the replay harness. Both sides were kept:

- `FastRaceCue` takes `contact_support`, `clearance_brake` and `motor_assist`.
- The runner's CSV ends with the view columns, then the contact columns, then the assist columns.
- The harness takes `--throttle-column`, `--wall-pilot` and `--motor-assist` (file tags
  `-cthr -dv -wp -ma`).

`m4b-brain11` merged cleanly. Two integration choices, both inert unless a new rule is on:

- With `--motor-assist on`, contact support compares the measured vertical speed with the request
  the motor received. The assist's declaration already has the older support rules do this.
  Without the assist it is the pilot's own request, so nothing else changes.
- `haltere.train.deployed_pilot` (brain-11's offline copy of the runner's stack) now follows the
  merged runner: wall pilot v5's floor, descent view v2's contact support, and `motor_assist=True`
  for `--motor-assist on`. brain-11's committed scores are m4-stack scores and stay as they are.

Every frozen declaration and gate file under `configs/` (41 files) kept its content and hash: the
canonical hash equals the file's own `sha256`, and the bytes equal the branch that changed them.

The runner's own loaders on `m4b` load:

- lag turn v2 `d4eb83da`;
- gap pilot v5 `43c30420` (gap cue v2);
- wall pilot v5 `1f37d702`;
- vertical guard v4 `409d06f9`;
- descent view v2 `7dc36efc`;
- motor assist v1 `eefb4a42`.

They refuse wall pilot v3/v4, vertical guard v2/v3, descent view v1 and gap pilot v4.

Tests: the full suite passed after the merges and again on the final tree: 1117 tests, including
every test the four branches added. Scripts and raw outputs
are in the session scratchpad under `m4b/integrate/`. Results:
`docs/experiments/round4b_integration.json`.

**Can the brain release graduate from pre-release now? No.** The bar is one frozen stack (weights,
pilot and obstacle configs, nothing per course) that finishes Straw Bale 3/3 with no ground contact
and Minus Two 3/3, with Pine Valley attempted. After round 4b:

- **No brain is selected.** The best-ranked brain-11 candidate, `fast-brain-11-b-cw13`, passes 8 of
  its 12 frozen gates. It fails G3v2, G6, G10 and G11: it over-brakes for requests left of its heading.
- **Every round-4b rule fails at least one of its frozen gates.**
  - Vertical guard v4 fails V-Straw downhill and both V-Pine checks.
  - Wall pilot v5 fails B-Quiet.
  - The contact audit fails its video false-positive check, so it cannot yet gate a run.
  - Motor assist v1 fails 4 of its 16 gates.
  - Descent view v2 carries version 1's view rule, which failed its own surrogate gates.
- **Nothing of round 4b has flown.**
- **New offline finding (below): contact support false-fires in the motor-assist harness.** It fires
  on 2 of 12 hairpin drones for every motor, and each of those drones then touches the ceiling.

The flights planned below are development flights. They can show whether the fixes work live. They
cannot graduate the release.

### Frozen gates of the four round-4b branches (as scored by each branch)

| Branch | Declaration (content sha256) | Gates, freeze and scores | Result |
|---|---|---|---|
| `m4b-guard` | `vertical_guard.json` v4 `409d06f9` (v1-v3 kept, refused) | gates v4 `7901b154`; `70e0918` / `7a9545c` | Identity 61/61; V-R4 pass (no escalation over the r4-02 floor and arch, request <= 0.96 m/s); V-Straw uphill 0 s/min; V-Minus pass; Pine mound pass. **Fails** V-Straw downhill (20%), V-Pine climb (74.3%, v3 77.7%) and V-Pine no-descent (-0.70) |
| `m4b-contact` | `descent_view.json` v2 `7dc36efc` (v1 kept); `wall_pilot.json` v5 `1f37d702` (v4 kept); `contact_audit.json` v1 `1c82c7f4` | contact-support gates v1 `a7d033ce`, wall gates v2 `b6ce9d6f`; `d15faeb` / `cc7d0f3` | CS-Identity 96/96, CS-Rest (0.19-0.47 s), CS-R402, CS-Surrogate (0 onsets on 112 gate courses), CS-Clean pass; B-Identity, B-NoSink, B-R402, B-Horizontal pass; **B-Quiet fails** (1.0-2.9 s of withheld sink on 4 Straw laps); audit: every labelled contact found, **video false positives 0.32/min vs 0.05** |
| `m4b-assist` | `motor_assist.json` v1 `eefb4a42` (brain contract only) | gates v1 `9abfb80d`; `7a29724` / `0c3f56c` | 12 of 16: hairpin 09b and 08, accelerate 10b, dev16 and hills for three brains, both live windows, identity 57/57. **Fails** hairpin 10b (2 ceiling contacts), accelerate 08 and 09b, quiet (stopping bound 1.8-3.3 s/min on Straw uphill rings) |
| `m4b-brain11` | candidates `runs/fast-brain-11-*` | `brain11_gates.json` v1 `78a528ab`; `4ab221b` / `4ee71c3` | None selected. `fast-brain-11-b-cw13` (`44cca3c4`) 8/12: passes G9 (r4 hairpin +0.38 m/s over the request vs 10b +1.23; stop-and-accelerate 0.05 m lost vs 09b 1.41) and G13; **fails** G3v2, G6, G10, G11 |

### Identity: the new rules off are m4

These comparisons use open-loop replays of the 24 logs through a `git archive` of `m4` and of
`m4b`, both with the `m4b` harness. **81 of 81 command-array pairs are bit-identical:**

- the default pilot on all 24 logs;
- the stack in shadow on all 24;
- the descent view v1 rule alone on all 24 (the kept `descent_view_v1.json` on `m4b` against m4's
  declaration);
- `--motor-assist` on the 9 fast-PD logs against the default pilot.

The fast PD has no assist entry. The unit tests' golden digests of the default, version-1 and
round-4-stack pilots also match.

In the descent surrogate, the full `m4b` pilot equals the full `m4` pilot course by course for all
four motors (84 courses). Contact support never fires there (its hills are scoring-only), and the
guard and the clearance brake have no looming samples to act on.

### Open-loop replays: what the m4b stack changes, per log (development evidence)

Each log was replayed at its recorded states through two stacks:

- `m4`: `--stack on --near-on-path --descent-view` (v1), with guard v3 and wall v4.
- `m4b`: the same flags with guard v4, wall v5 and descent view v2.

Both use `--throttle-column command_thr`: the issued throttle the runner's pilot saw. The Straw
laps and `pine-brain08-01` use the offline looming stream.

Three more `m4b` variants each put back one kept declaration, to give each rule's share: wall pilot
v4, descent view v1, and both (the last isolates guard v4). A last variant adds
`--motor-assist on`. The recorded motion does not respond to the requests.

| Log | Motor | Request changed vs m4, s (guard / wall floor / contact) | Guard escalations m4 -> m4b | Brake sink withheld: s, max m/s | Contact-support onsets (audit contacts caught) | Motor assist (brain logs): s changed; stopping s; climb bias s, max |
|---|---|---|---|---|---|---|
| `minus-fast6-wall-01` | fast PD | 0 (0 / 0 / 0) | - -> - | - | 0 | none (no PD entry; identical) |
| `minus-brain08-gapon-01` | brain-08 | 1.04 (0 / 1.04 / 0) | - -> - | 1.3, 0.22 | 0 | 6.89; 2.19; 2.3, 0.61 |
| `minus-brain08-gapon-02` | brain-08 | 0.56 (0 / 0.56 / 0) | - -> - | 0.6, 0.13 | 0 | 8.77; 2.37; 2.29, 0.64 |
| `minus-fast6-gapon-01` | fast PD | 0.55 (0 / 0.55 / 0) | - -> - | 1.58, 0.33 | 0 | none (no PD entry; identical) |
| `minus-brain08-loom-01` | brain-08 | 0 (0 / 0 / 0) | - -> - | - | 0 | 2.8; 1.46; 0.91, 0.54 |
| `minus-fast6-vg-02` | fast PD | 2.04 (0 / 2.04 / 0) | - -> - | 3.7, 0.19 | 0 | none (no PD entry; identical) |
| `minus-brain08-vg-01` | brain-08 | 0.59 (0 / 0.59 / 0) | - -> - | 0.54, 0.14 | 0 | 6.33; 2.18; 2.38, 0.66 |
| `minus-brain09b-vg-01` | brain-09b | 1.37 (0 / 1.37 / 0) | - -> - | 3.59, 0.11 | 0 | 6.77; 3.12; 2.42, 0.7 |
| `minus-brain08-gapshadow-01` | brain-08 | 0 (0 / 0 / 0) | - -> - | - | 0 | 3.73; 1.44; 1.07, 0.6 |
| `minus-fast6-r4-02` | fast PD | 6.66 (0.77 / 5.86 / 0) | 32.9 -> - | 7.1, 0.8 | 0 (0 of 3) | none (no PD entry; identical) |
| `minus-brain10b-r4-02` | fast-brain-10b | 0.66 (0 / 0.66 / 0) | - -> - | 0.67, 0.07 | 0 | 5.51; 2.48; 0.55, 0.45 |
| `minus-brain09b-r4-01` | brain-09b | 2.12 (0 / 2.12 / 0) | - -> - | 3.66, 0.13 | 0 | 8.5; 3.51; 4.04, 1 |
| `pine-fast6-ttc-01` | fast PD | 1.49 (0.68 / 0.81 / 0) | 4.55, 18.44 -> 4.55, 19.05 | 0.78, 0.43 | 0 | none (no PD entry; identical) |
| `pine-brain08-loom-01` | brain-08 | 0 (0 / 0 / 0) | - -> - | - | 0 | 0.21; 0; 0, 0.01 |
| `pine-brain08-01` | brain-08 | 1 (1 / 0 / 0) | 16.96 -> - | - | 0 (0 of 1) | 5.75; 0.29; 4.85, 1 |
| `straw-brain08-04` | brain-08 | 12.65 (0 / 0.14 / 12.51) | - -> - | 1.37, 0.07 | 10 (7 of 7) | 79.69; 9.89; 69.59, 1 |
| `straw-brain08-06` | brain-08 | 13.57 (0 / 1.07 / 12.5) | - -> - | 2.89, 0.41 | 6 (6 of 6) | 86.5; 11.05; 69.59, 1 |
| `straw-brain08-01` | brain-08 | 4.38 (0 / 0.13 / 4.25) | - -> - | 1.1, 0.08 | 4 (1 of 1) | 18.64; 3.47; 12.81, 1 |
| `straw-brain08-02` | brain-08 | 4.51 (0 / 0 / 4.51) | - -> - | 0.11, 0.04 | 2 (2 of 2) | 41.92; 4.88; 29.79, 1 |
| `straw-brain08-03` | brain-08 | 0 (0 / 0 / 0) | - -> - | 0.87, 0.04 | 0 | 12.41; 3.29; 9.19, 1 |
| `straw-fast6-01` | fast PD | 0 (0 / 0 / 0) | - -> - | - | 0 | none (no PD entry; identical) |
| `straw-fast6-02` | fast PD | 1.25 (0 / 1.25 / 0) | - -> - | 1, 0.29 | 0 | none (no PD entry; identical) |
| `straw-fast6-03` | fast PD | 0 (0 / 0 / 0) | - -> - | - | 0 | none (no PD entry; identical) |
| `straw-fast6-arc-01` | fast PD | 0 (0 / 0 / 0) | - -> - | - | 0 | none (no PD entry; identical) |

- **Vertical guard v4** changes only three logs.
  - `minus-fast6-r4-02`: the escalation at 32.9 s is gone. The request at 32.3-33.5 s is at most
    0.96 m/s instead of 2.74, with 0 escalated ticks.
  - `pine-fast6-ttc-01`: the mound escalation is unchanged (4.55 s, 145 ticks, 3.5 m/s). The
    end-of-course hillside escalation comes 0.61 s later (19.05 s against 18.44 s), with 168
    escalated ticks instead of 229 at 17.6-21.3 s.
  - `pine-brain08-01`: it loses its escalation at 16.96 s (3.46 -> 1.0 m/s). The contact audit shows
    that flight already in contact with the hillside from 16.72 s.
- **Wall pilot v5's floor** withholds brake-made sink on 16 of the 24 logs: 0.1-7.1 s per log, up to
  0.8 m/s.
  - `minus-fast6-r4-02`, garage floor at 26.4-29.7 s: the lowest request goes from -0.47 to
    0.0 m/s.
  - `minus-fast6-r4-02`, the hairpin exit at 20.1-21.3 s: from -0.45 to -0.02 m/s.
  - `pine-fast6-ttc-01`, the hillside at 15.2-16.9 s: from -0.43 to 0.0 m/s.
  - Straw: 0.1-2.9 s per log at up to 0.41 m/s. This is the B-Quiet failure: on the uphill it
    withholds sink the brake had added toward rising ground.
- **Contact support** fires on no Minus, Pine or fast-PD Straw log.
  - On the brain-08 Straw laps it fires in all 16 downhill slides that the contact audit finds, and
    nowhere else. It fires 0.36-1.18 s after each slide starts (3.2 s into the long slide of
    `straw-brain08-01`).
  - The logged support rule caught 12 of those 16 slides, 0.85-1.11 s after they started.
  - It does not fire on the three `minus-fast6-r4-02` floor touches (27.3, 29.0 and 30.7 s): with
    wall v5 there is no sink request left there. In closed loop the brake-made sink that put the
    drone on the floor is gone.
  - It adds support climbs on the downhill: 3-11 per brain-08 Straw log against 0-5 before.
- **Motor assist**: bit-identical on every fast-PD log. On the brain logs:
  - The stopping source acts 1.4-3.5 s per Minus log and 3.3-11 s per Straw log.
  - The climb bias acts 0.5-4.9 s per Minus or Pine log and 9-70 s per Straw log, up to 1 m/s.
    On `straw-brain08-04` and `-06`, 62.5 of the 69.6 s come from sink shortfall: the recorded
    drone sinks more than 0.3 m/s faster than the request. About half of those 69.6 s (31-32 s)
    are an open-loop artifact: the recorded drone was following the older pilot's steeper request.
    The rest is brain-08 sinking faster than it was asked to.
  - `minus-brain10b-r4-02`, hairpin at 20.0-21.6 s: the lowest horizontal request falls from 3.62 to
    0.33 m/s.
  - `minus-brain09b-r4-01`, after the turn-first release at 22.8-23.7 s: the vertical request rises
    to +1.29 m/s, where it was at most +0.29.

### Contact audit of the logs (`contact_audit.json` v1; report, not a gate)

- `minus-fast6-r4-02`: three floor supports (27.33-27.99, 29.02-29.55 and 30.68-31.06 s, lowest
  0.01-0.08 m) and the ceiling impact at 34.1 s.
- Every other Minus log: only its terminal impact. `minus-brain08-gapon-02`'s impact was on the
  ceiling, at 2.13 m.
- `straw-brain08-04` and `-06` (three laps each): 7 and 6 downhill slides at x -36..-39,
  y 136-169.
- `straw-brain08-01`: the 68.1-75.8 s hillside slide.
- `straw-brain08-02`: 2 slides.
- **The fast-PD Straw laps have no contact:** `straw-fast6-01`, `-02`, `-03` and `-arc-01`, 1.0-5.1
  scored minutes each.
- `pine-fast6-ttc-01`: the tree hit at 16.7 s.
- `pine-brain08-01`: a hillside support at 16.7-17.9 s.

### Surrogate (development evidence)

The loop is round 4's descent surrogate: IdentifiedSim, 10% per-drone randomisation, 150 s per
course, sim seed 17, flat and steep seeds 3000-3007 and hill seeds 6000-6011. The m4 rows are round
4's `full` runs; brain-11's are its G13 report. The m4b rows are new.

| Motor | Stack | Finished / crashed (28) | Terrain contacts / s | High passes | Chatter (16) | Mean terrain finish s |
|---|---|---|---|---|---|---|
| fast PD | m4 = m4b | 28 / 0 | 11 / 15.6 | 14 | 0.00634 | 56.32 |
| brain-09b | m4 = m4b | 28 / 0 | 14 / 19.2 | 18 | 0.00561 | 62.92 |
| brain-09b | m4b + assist | 28 / 0 | 14 / 19.0 | 15 | 0.00591 | 63.67 |
| fast-brain-10b | m4 = m4b | 28 / 0 | 12 / 19.7 | 29 | 0.00289 | 65.27 |
| fast-brain-10b | m4b + assist | 28 / 0 | 10 / 19.5 | 27 | 0.00291 | 65.53 |
| fast-brain-11-b-cw13 | m4 = m4b | 28 / 0 | 11 / 26.1 | 12 | 0.00239 | 61.86 |
| fast-brain-11-b-cw13 | m4b + assist | 28 / 0 | 11 / 25.2 | 14 | 0.00247 | 62.07 |

- Under the deployed pilot, fast-brain-11-b-cw13 has the fewest high passes and the lowest chatter
  of the brains, with contacts level with the PD's.
- The assist changes little on these courses: course time +0.3..+1.2%, and high passes -3..+2.
- A checkpoint pass counts anywhere within 3 m, so 28/28 does not mean completion-safe descents.

### Hairpin and stop-then-accelerate scenarios under m4b (development evidence)

These are the motor-assist harness's sets that its frozen gates v1 scored: 12 hairpins, a wall
2.3-3 m behind the arch, a 2.2 m ceiling, and 6 stop-then-accelerate courses. Here they are flown
under the `m4b` pilot, with and without the assist. This is a report, not a re-score. "Live-like"
means wall samples without a below fraction, as the recorded Minus walls carry.

| Motor | Hairpin, clean of 12 (wall / floor / ceiling): m4b | m4b + assist | Live-like samples: m4b / + assist | Accelerate, clean of 6 and mean height loss: m4b / + assist | m4 stack, assist gates: baseline / assist |
|---|---|---|---|---|---|
| fast PD | 10 (2 / 0 / 2) | no entry | 9 (1 / 0 / 3) / - | - | - |
| brain-08 | 0 (10 / 3 / 2) | 0 (3 / 9 / 2) | - | 3, 0.40 m / 3, 0.40 m | 0 (10 / 3 / 0) / 1 (3 / 9 / 0) |
| brain-09b | 0 (12 / 0 / 2) | 9 (0 / 0 / 2) | 0 / 10 | 4, 0.34 m / 3, 0.21 m | 1 (11 / 0 / 0) / 8 (0 / 3 / 1) |
| fast-brain-10b | 0 (12 / 0 / 2) | 7 (0 / 0 / 2) | 0 / 7 | 3, 0.15 m / 3, 0.11 m | 0 (12 / 0 / 0) / 7 (1 / 2 / 2) |
| fast-brain-11-b-cw13 | 0 (12 / 0 / 2) | 7 (2 / 0 / 3) | 0 / 6 | 3, 0.56 m (1 crash) / 3, 0.39 m | not run |

- **Without the assist, no brain stops at these hairpin walls**, fast-brain-11-b-cw13 included.
  The fast PD passes 9-10 of 12, so the set is passable. With the assist, the braking brains pass
  6-10 of 12.
- **Wall v5 removes the floor contacts the assist brought after a stop on m4**: brain-09b 3 -> 0 and
  fast-brain-10b 2 -> 0. The assist branch's horizontal-ray stand-in predicted this.
- **New: contact support false-fires in this harness.** Every hairpin run with contact support has a
  support climb in the same 2 of 12 drones, for every motor (the PD included). Each of those drones
  then touches the 2.2 m ceiling; without contact support none does (runs that each drop one rule).
  - Cause (fast PD, full batch): those two drones' randomised thrust is about 15% above the declared
    curve; their learned gain ends at 1.13-1.17.
  - At arming (3.2 s) the gain is still 1.0. The rule reads 1.2-1.5 m/s² of unexplained upward
    force while the pilot asks for a 0.1 m/s sink and the drone still rises after launch.
  - It fires at 3.4-3.5 s and again at 4.8 and 6.6-7.2 s.
  - On the logged flights the gain stays within 0.95-1.10, and the contact branch found no onset
    outside a contact on 58 logs. A drone whose thrust runs about 10% above the curve at arming
    could still trigger it.
  - It is a frozen rule, so it is not changed here. The live plan watches for it.

### Round-4 audit and review items fixed on m4b (docs only)

Each item below is a doc fix. Everything else the audit found matched.

- **Minus descent view.** The limiting time is now reported alongside the time the request changes:
  up to 0.85 s per log.
- **"Before every contact".** The Straw downhill claim now reads "below the image 0-15% of the 3 s
  before the contacts" (here and in `docs/descent_view.md`).
- **Gates v3 caveats.** The V-Pine redefinition and the in-sample threshold note were added to the
  gates v3 paragraph.
- **Shadow control.** The intended control keeps `--descent-view on`; round 4b's wiring check covers
  it.
- **Score header.** `vertical_guard_v2_under_gates_v3_scores.json` now names the guard actually
  replayed (v2 `e06b690d`), with a correction note. Only that entry changed.
- **Pine backside figures.** They now read "0.3-0.54 m/s at 4.9-5.7 m/s, steepening after 7.2 s".
- **Stale "not merged".** The brain-10 sentences here and in `docs/fast_brain_10_candidate.md` were
  corrected.
- **Wall scorer fix.** The fix committed after the scores (`0d1010a`) is disclosed.
- **Gap v3.** The card now says v3's replays were generated and then deleted, and notes the copy
  slip in the frozen notes of gates v2/v3.
- **Rounding.** brain-09b's value on hills of 12 deg or less now reads 28.7% in both tables (exact
  value 0.28651).
- **Other stale text.** Also updated:
  - brain-11's "not merged" section;
  - the gap-pilot flag table (guard version);
  - the guard doc's note that wall v5 now covers the r4-02 floor skim.

### Live flight plan (for the main session; development flights)

Fly from `C:\DEV\Haltere` with `m4b` checked out. It is a local branch and is not pushed. While
the integration worktree still holds the branch, use `git switch --detach m4b` in the main checkout,
or remove that worktree first. The runner loads its declarations from the checkout it runs from and
refuses other versions. Keep the untracked `configs/explore_spiral.yaml`.

The flight procedure is unchanged:

- Liftoff and every capture, pad and controller process run inside Anode, with the viewer hidden.
- The pad must report `seatOnly`.
- Run the ground check, and verify throttle-low and a real processed control response.
- Nothing else heavy may run. The preflight refuses a busy machine.
- Use a new log name for every attempt, and keep Liftoff open between runs.

**One stack for every course:**

```
--looming-brake --obstacle-stack on --descent-view on --motor-assist on
```

The fast PD has no motor-assist entry: its commands are bit-identical with and without the flag,
and its sidecar records `applied: false`. A wiring check parsed every command below with the
runner's own parser and built the controller (no camera, pad or flight). Each run declares:

| Declaration | Version | Content sha256 |
|---|---|---|
| `configs/obstacles/lag_turn.json` | 2 | `d4eb83da51ab...` |
| `configs/obstacles/gap_pilot.json` (gap cue `gap_cue.json` v2 `284b3c46a819...`) | 5 | `43c304204f93...` |
| `configs/obstacles/wall_pilot.json` (stopping model: the motor's contract; clearance-brake floor applied) | 5 | `1f37d7024ee4...` |
| `configs/obstacles/vertical_guard.json` | 4 | `409d06f9ded7...` |
| `configs/pilot/descent_view.json` (with contact support; CSV adds `contact_unexplained`, `contact_gain`, `contact_fire`) | 2 | `7dc36efc6377...` |
| `configs/pilot/motor_assist.json` (applied to the brain contract only; CSV adds the six `assist_*` columns) | 1 | `eefb4a42613c...` |

**Every run is a disclosed development deviation:**

- no brain is selected;
- guard v4, wall v5 and the motor assist fail frozen gates;
- descent view v2 carries version 1's failed view rule;
- the contact audit cannot gate.

**Brain: `fast-brain-11-b-cw13`** (`runs/fast-brain-11-b-cw13/candidate.pt`, sha256 `44cca3c4...`),
the best-ranked brain-11 candidate. It is the same brain on Minus Two and on Straw Bale, so that one
stack is tested. Why this one:

- It passes the r4 live-window gate G9. By G9's thresholds fast-brain-10b would fail it (+1.23 m/s
  over the hairpin request) and so would brain-09b (1.41 m of height lost after the stop).
- Under the m4b pilot in the surrogate it has the fewest high passes and the lowest chatter of the
  brains.
- The alternatives:
  - brain-09b stops at the harness hairpins most often with the assist (9-10 of 12), but chatters
    (0.010 per tick there, 0.0063 live) and has the most surrogate contacts;
  - fast-brain-10b needs the assist just as much and passes rings high (27-29 high passes).

Its known faults:

- over-braking for requests left of its heading (G3v2, G6, G10);
- height lost in hard accelerations at 60 deg (G11; 0.56 m mean in the harness without the assist,
  0.39 m with it);
- the gap pilot's commitment timing uses brain-08's response model for it.

**Assist:** without it no brain stopped at the harness hairpin walls; with it this brain passes 6-7
of 12, at the cost of 3 of 12 ceiling contacts. The optional diagnostic below flies the same brain
without it.

**Before every launch** (in the seat, from `C:\DEV\Haltere`):

```powershell
.venv/Scripts/python.exe runs/fast-stack-20260923/ground_check.py runs/fast-stack-20260923/ground-check-32.json
```

The ground-check script **pauses the game when it exits**, and pausing inside it cannot be
disabled. So after it, and after every run (the runner's `--pause-on-stop` pauses the game too):

1. Click **Réinitialiser** (646,277) in the pause menu, or resume.
2. Confirm that telemetry is live. The one-liner below prints `LIVE` when two frames 0.5 s apart
   show an advancing game timestamp:

```powershell
.venv/Scripts/python.exe -c "import time; from haltere.liftoff.telemetry import TelemetryReceiver,read_config,DEFAULT_STREAM; rx=TelemetryReceiver(port=9001,stream=(read_config() or {}).get('StreamFormat',DEFAULT_STREAM)); a=rx.wait(1.0); time.sleep(0.5); b=rx.wait(1.0); rx.close(); print('LIVE' if a is not None and b is not None and b.timestamp > a.timestamp else 'NOT LIVE', None if b is None else [round(float(v), 2) for v in b.position])"
```

Launch only after `LIVE`. Otherwise the run stops at 0 ticks with "No fresh live image/telemetry",
as `minus-fast6-r4-01` and `minus-brain10b-r4-01` did. Increment the ground-check number for every
check.

Run the four flights in this order. Each command is one line.

**(a) Minus Two, fast PD.**

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-08-vgs04-s10r03m30/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller pd --pd-profile fast --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --descent-view on --motor-assist on --seconds 150 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/minus-fast6-r4b-01.csv --record runs/fast-stack-20260923/minus-fast6-r4b-01.mp4 --video-encoder h264_nvenc
```

What to look for. The replay figures below are open-loop requests at the recorded states of
`minus-fast6-r4-02`.

- **Pillar A, hairpin and pillar C** as in r4-02: a left commitment near x 49, about 0.5 s of
  turn-first at the hairpin, a right commitment before pillar C.
- **Hairpin exit (20-21 s):** the lowest vertical request is -0.02 m/s instead of -0.45; the brake
  adds no sink.
- **Garage floor after pillar C (26-30 s):**
  - no brake-made sink (the lowest request is 0.0 instead of -0.47 m/s);
  - the drone should stay above the rings' floor instead of skimming it;
  - the audit found three floor touches on r4-02.
- **Arch at about (75.9, 64.3):**
  - no escalated climb: the guard stays at 1 m/s or less, and the request was at most 0.96 m/s in
    the replay;
  - the gentle climb still starts on the floor misread, so the drone may rise over the arch rather
    than pass through it;
  - the ceiling guard is the only backstop under the 2.2 m ceiling.
- Beyond (79, 58) nothing has been flown.

**(b) Minus Two, fast-brain-11-b-cw13.**

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-11-b-cw13/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller brain --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --descent-view on --motor-assist on --seconds 150 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/minus-brain11cw13-r4b-01.csv --record runs/fast-stack-20260923/minus-brain11cw13-r4b-01.mp4 --video-encoder h264_nvenc
```

What to look for:

- **Hairpin arrival speed toward the wall (+x) near (79.7, 18.7).** About 2.6-3.0 m/s or less is
  needed. The assist's stopping source and cap tracking should lower the request before the arch;
  in the r4-02 replay of fast-brain-10b the request dropped to 0.33 m/s.
- **Turn-first engagement and release**, then the acceleration out: no sink to the floor. In G9 this
  brain lost 0.05 m where brain-09b lost 1.41 m.
- **Left capped turns** (pillar A's left commitment, the turn after the hairpin): slow or stalled
  turns are this brain's known fault.
- **Stick change per tick:** fast-brain-10b 0.0031 live, brain-09b 0.0063.
- **Ceiling:** support climbs (`contact_fire` 1) or assist climbs near the 2.2 m ceiling.
- Then the floor and the arch as in (a).

Optional diagnostic: the same brain without the assist. It is not the deployed stack, so fly it only
after (b):

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-11-b-cw13/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller brain --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --descent-view on --seconds 150 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/minus-brain11cw13-r4b-noassist-01.csv --record runs/fast-stack-20260923/minus-brain11cw13-r4b-noassist-01.mp4 --video-encoder h264_nvenc
```

**(c) Straw Bale, fast-brain-11-b-cw13, full stack.** This is the graduation course and the user's
downhill request.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-11-b-cw13/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller brain --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --descent-view on --motor-assist on --seconds 480 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/straw-brain11cw13-r4b-01.csv --record runs/fast-stack-20260923/straw-brain11cw13-r4b-01.mp4 --video-encoder h264_nvenc
```

After the run, score ground contact from telemetry with the contact audit, not from support-climb
onsets:

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.contact_audit runs/fast-stack-20260923/straw-brain11cw13-r4b-01.csv --json runs/fast-stack-20260923/straw-brain11cw13-r4b-01.contact-audit.json
```

Its false-positive check failed on video, so report its contacts next to the video; they do not
decide pass or fail. The same command works for every run.

What to look for:

- **Downhill (x about -37, y 135-170):**
  - the audit's slides against brain-08's 6-7 per three laps and the fast PD's 0;
  - contact support should start a climb 0.4-1.2 s into any slide (`contact_fire`);
  - more support climbs mean a risk of rings passed high, missed rings and turn-backs.
- **Uphill:**
  - wall v5 withholds brake-made sink there (1-3 s per three laps in replay);
  - the assist's stopping source may slow the drone at the uphill rings (1.8-3.3 s/min open loop;
    the quiet failure).
- **Hilltop:** the ring marker drops out there; the gentle search must hold height.
- **Lap times** against brain-08's 5:17.8.

**(d) Pine Valley, fast PD.**

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-08-vgs04-s10r03m30/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller pd --pd-profile fast --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --descent-view on --motor-assist on --seconds 150 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/pine-fast6-r4b-01.csv --record runs/fast-stack-20260923/pine-fast6-r4b-01.mp4 --video-encoder h264_nvenc
```

What to look for:

- **Mound:** the escalation near 4.55 s, climbing up to 3.5 m/s, unchanged in replay.
- **Backside at 6.3-7.4 s:** 0.3-0.54 m/s of sink at about 5-5.7 m/s.
- **Hillside at 15.2 s:** the guard asks for 1 m/s (as v3). The brake no longer adds sink there: the
  lowest request is 0.0 instead of -0.43 m/s.
- **End hillside:** v4 escalates 0.61 s later than v3 (19.05 s) and asks for less height.
- **Boulder:** no rule arrests the descent into it.

Optional matched control for (a): the same stack in shadow, which isolates the obstacle stack
(the descent view and motor assist stay on; contact support stays applied even in shadow, see the review):

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-08-vgs04-s10r03m30/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller pd --pd-profile fast --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack shadow --descent-view on --motor-assist on --seconds 150 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/minus-fast6-r4b-shadow-01.csv --record runs/fast-stack-20260923/minus-fast6-r4b-shadow-01.mp4 --video-encoder h264_nvenc
```

**Stop criteria.**

- **Before a run, do not start if:**
  - the preflight is not quiet;
  - the pad is not `seatOnly`, or the guard unplugs it (a user's game alone is not a reason);
  - the ground check fails;
  - telemetry is not `LIVE`.
- **After a run, stop the series if:**
  - the sidecar declares any other version or hash than the table above, or
    `motor_assist_declaration.applied` is wrong for the motor;
  - a controller deadline or camera failure was recorded.
- **During a run, stop the runner if:**
  - the drone climbs above about 1.9 m in the Minus garage, or keeps climbing above 4 m elsewhere;
  - a support climb starts with the drone clearly airborne (`contact_fire` 1 while the video shows
    no ground), the contact-support false positive above;
  - a turn-first hold or an assist stop parks the drone for more than about 3 s;
  - it circles in search for more than about 30 s.
- **Between runs:** stop and analyse instead of flying the next run if a round-4b rule causes a new
  failure. First replay the log through `haltere/obstacles/vertical_replay.py` with
  `--stack flown --descent-view configs/pilot/descent_view.json --throttle-column command_thr
  --motor-assist configs/pilot/motor_assist.json`.
- **Graduation evidence:** none of these runs counts, because no brain is selected and the rules fail
  gates.

### Blockers for graduation, and what would clear them

| Blocker | State after round 4b | What would clear it |
|---|---|---|
| A selected brain | None. fast-brain-11-b-cw13 8/12 (left-turn over-braking, 60 deg acceleration); no brain stops at the harness hairpins without the assist | A brain-12 fixing the left asymmetry, scored on frozen gates with fresh held-out sets; then flights |
| Minus hairpin for brains | Needs the motor assist (harness 6-10 of 12 with it, 0 without); assist v1 fails 4/16 gates | An assist v2 frozen with fresh held-out sets on the merged stack (the branch's own next step), or a brain that follows caps; live (b) |
| Contact-support false positives | Fires on the 2 of 12 harness drones with ~15% more thrust, at arming; ceiling contacts follow | A contact-support v3 that learns the gain before arming, or needs the drone not climbing away, frozen and scored on a thrust-randomised set; live (b)/(c) `contact_gain` |
| Straw downhill ground contact | Contact support reacts 0.4-1.2 s into every audited slide; nothing prevents the slides; brains do not fly the view rule's 6 m/s descents; the audit cannot gate (video FP 0.32/min) | A brain distilled under the view rule; a contact-audit v2 validated with terrain-height or chase-view labels, frozen before scoring; then Straw 3/3 with no audited contact |
| Vertical guard on Pine | v4 keeps the mound, costs hillside climb (V-Pine 74.3%; one escalation lost on `pine-brain08-01`) | Closed-loop evidence or a causal signal the camera does not publish yet |
| Minus floor and arch | Wall v5 removes the brake-made floor skim in replay; the guard's gentle climb still starts on the floor misread | Live (a); a guard rule that uses a looming quality signal |

## Round 4b (live), 2026-09-28: m4b stack on Minus Two and Straw Bale

Branch `m4b` at `b4d9e2c`: `--looming-brake --obstacle-stack on --descent-view on` (wall pilot v5,
vertical guard v4, descent view v2 with contact support, gap pilot v5, lag turn v2). The motor assist
(v1) was flown only as a no-op with the fast PD: the round-4b review found its request steps by up to
4.5 m/s per tick and nearly stops brains before pillar A and the arches, so the brain runs flew
without it (`--motor-assist` off, the plan's no-assist diagnostic). Original `[Copy] New Drone`, pad 30
`seatOnly` in every preflight/postflight, ground checks 32 (Minus) and 33 (Straw, after the level change).
Every run is a disclosed development deviation: every round-4b rule fails at least one of its frozen
gates and no brain is selected (fast-brain-11-b-cw13 passes 8/12). Contacts are from the offline
`haltere.liftoff.contact_audit` (its video false-positive check failed, so counts are indicative).

| Run | Motor | Outcome |
|---|---|---|
| `minus-fast6-r4b-01` | fast PD | **44.8 s, the longest 6 m/s Minus Two run.** Pillar A (y 5.47), hairpin (exit 23.7 s), pillar C (25.8 s), past round 4's floor-and-ceiling spot (y 69.4 at 35.8 s) with no floor contact (min z 0.16 m, audit 0 contacts) and no terrain climb (max z 1.13 m), on to (59, 97). Then, turning toward a ring clamped at the left edge, it clipped the leg of an arch whose ring sits near the arch edge, at ~6 m/s, (51.1, 93.5); the governor held a stand-off status with a 1 m/s cap along a stale ray while the pilot asked 6 m/s, and the gap cue did not shift the aim |
| `minus-brain11cw13-r4b-noassist-01` | fast-brain-11-b-cw13 (`44cca3c4…`), no assist | Pillar A (y 5.64); braked to ~3.5 m/s under 3.8 m/s caps at the arch. At the hairpin the looming TTC fell from ~1 s to 0.2 s while the governor stayed armed, the cap dropped to 3.5 m/s only ~0.5 s before the wall, turn-first engaged for 0.05 s, and it hit the wall at (81.9, 20.0) at 3.7 m/s, 23.6 s. Stick change 0.0038 per tick |
| `straw-brain11cw13-r4b-noassist-01` | fast-brain-11-b-cw13 | Not flown: preflight refused (a `rustc.exe` compile in the user's desktop session used 5.3 cores) |
| `straw-brain11cw13-r4b-noassist-02` | fast-brain-11-b-cw13, no assist | **Lap 1 in 1:42.988** (fast-brain-08's first laps: 1:46.358, 1:46.072), including the hill and the downhill: **one audited ground contact on the downhill** (79.4-80.0 s at (-36.6, 132.8), 5.2 m/s, a support climb; fast-brain-08: 6-7 audited slides per 3-lap race). Stick change 0.0030 per tick (fast-brain-08 0.0031). 8 s into lap 2 the next ring marker dropped out after the lap arch; coasting at 5.6 m/s on a request that turned right, it clipped the leg of the second start arch at (27.1, -0.5), 112.4 s |

What this shows:

- The round-4b guard and the brake-without-sink fix removed round 4's floor skim and ceiling climb live.
- fast-brain-11 is the first braking brain to fly a full Straw Bale lap, faster than fast-brain-08's laps
  and with fewer downhill contacts in this one lap; one lap is not a repeatability result.
- New failures: arch legs beside rings (Minus at 44.8 s, Straw start arch in lap 2 while coasting), and
  the brain hairpin still needs earlier braking (the looming warning comes ~0.5 s before the wall).
  Round 5 (below) found that the Minus failure was the wall behind the arch, not an arch leg. The Straw failure began
  with a false ring reading about 0.3 s before the lap arch, and the impact came about 0.8 s into lap 2 (the "8 s"
  above is a slip).

## Round 5 (offline): brain-12 (branch `m5-brain12`, not flown)

`m5-brain12` merges `m5-brake` (motor assist v3; the only round-5 pilot branch returned flight ready) and distils
brain-12 under that round-5 pilot. Its gates are frozen before any candidate existed: `configs/brain12_gates.json` v1
`8fb1b1a0c412...`, with fresh held-out course seeds, G13 terrain contact seconds, G14 (the r4b live windows, development
cases), G15 (right-hand mirrors of G10/G11) and G16 (harness hairpins).

**No candidate passes.** The best-ranked is `fast-brain-12-b-cw26d3` (`2ecf3f1f9b38...`), with 6 of 15 primary gates;
fast-brain-11-b-cw13 also passes 6 of 15 on the same gates. Compared with brain-11:

- **Better:**
  - left 60-degree re-acceleration (G11 passes);
  - harness hairpins with the assist: 8 of 12 clean, no wall (G16);
  - the Straw downhill replay: it passes 1.22 m above the live contact onset.
- **Worse:**
  - it overshoots moderate requests and caps (G1, G2, G5);
  - the live r4b hairpin: capped excess 1.50 against 1.17.
- **Terrain contacts are not lower:** 23 in 34.88 s against 21 in 33.29 s.

The left over-braking comes from a weak roll toward lateral requests that arrive without a yaw rate; synthetic
body-frame turns made it symmetric but did not fix it. Details: `docs/fast_brain_12_candidate.md`. Graduation is not
advanced by this round.

## Round 5 (offline): safety fixes of the round-4b review (branch `m5-safety`)

**Nothing in this section has flown.** Branch `m5-safety` is `m4b` (`2a5bccb`) with two rule changes
from the round-4b review. Both were frozen with their gates before any gate run (commit `68446ee`):

- **Wall pilot version 6** (`configs/obstacles/wall_pilot.json`, `fe65951d3d68...`).
  - The ceiling guard's overhead cut now bounds every climb while the measured vz exceeds 0.3 m/s:
    support, contact-support, search, motor-assist sag, top-edge and coast climbs. Before, it acted
    only on the governor's own terrain climb.
  - The pilot's own climb toward the ring in view is exempt: a literal draft cut the Straw uphill
    climbs on 5 of the 10 Straw development logs.
  - Details: [obstacle_gap_pilot.md](../obstacle_gap_pilot.md#ceiling-guard-any_climb-wall-pilot-version-6-round-5).
- **Descent view version 3** (`configs/pilot/descent_view.json`, `2bdb17fc2479...`): contact support
  version 3.
  - Windows with a pitch/roll body rate above 2.5 rad/s or a horizontal brake above 4 m/s² are not
    used.
  - The thrust gain is learnt before arming: the median of quiet or rising windows from 3 s, arming
    after 0.5 s of them.
  - Its climb never ends turn-first.
  - `--contact-support on|off|shadow` isolates it.
  - Details: [descent_view.md](../descent_view.md#version-3-round-5-contact-support-version-3).

Versions 5 and 2 are kept verbatim (`wall_pilot_v5.json`, `descent_view_v2.json`), and the runner
refuses them. With the new rules off the pilot is `m4b`'s bit for bit: 162 of 162 replay pairs, and
the golden digests of the default, view-rule and version-2 pilots.

### Gates (`configs/pilot/safety_gates.json` v1 `c6dfc88cae88...`; open-loop, synthetic and surrogate evidence)

Held out: the three round-4b live logs, harness sim seeds 101 and 102, and surrogate course seeds
3100-3103 and 6200-6207. Scores: `docs/experiments/round5_safety_scores.json`. Full table and the
post-scoring diagnoses: [descent_view.md](../descent_view.md#round-5-gates-configspilotsafety_gatesjson-version-1).

| Gate | Result | Pass |
|---|---|---|
| Identity (new rules off, and shadow = off), 27 logs | 162 of 162 pairs bit-identical | yes |
| Ceiling cut without a governor climb (review's case) | version 6 bounds the request after 0.05 s; version 5 never; a surface below: never | yes |
| Pilot climbing under the garage-ceiling pattern | support / contact / search levelled after 0.11 / 0.11 / 0.08 s (version 5: not cut) | yes |
| Turn-first during a contact-support climb | kept, 0.0 m/s toward the wall (version 2: handed off, 0.969 m/s) | yes |
| Pine mound climb | identical | yes |
| Ceiling quiet on Straw/Pine (13 logs, held-out lap included) | 0.0 s/min lowered | yes |
| Contact rest (review scenario), 5 motors x 0-6 m/s | fast PD 0.28 s, brain-09b 0.41-0.48 s, fast-brain-10b 0.39-0.41 s; **brain-08 (0-2 m/s) and fast-brain-11 (0-2 m/s) never armed** in the scripted prologue | **no** |
| `minus-fast6-r4-02` floor (round-4 stack + version 3) | onset 27.718 s; shadow marks it | yes |
| Detection where version 2 fires (46 audited contacts) | 45 of 45 development contacts; **the one held-out contact (Straw downhill touch, 79.35 s) missed** | **no** |
| Clean: onsets outside audited contacts, 61 logs, 90.38 min | 0 | yes |
| Held-out false reads, command-agnostic | the fast-PD flare of `minus-fast6-r4b-01` removed; **a commanded climb at the Straw hilltop crest (62.43 s) remains, as with version 2** | **no** |
| Arming false fires, harness, fresh seeds, 5 motors | 0 (round-4b stack: 3-5 per motor); ceiling contacts equal to contact support off | yes |
| Descent surrogate, fresh seeds, 3 motors | 0 onsets, 48 of 48 courses identical to off | yes |

**What the failures mean.**
- **Version 3 missed the only held-out real contact that version 2 caught.**
  - It was an impact-style touchdown on the Straw downhill at 5.2 m/s. The knock pitched the body at
    up to 3.9 rad/s for 0.05 s and friction braked the drone.
  - Version 3 excluded the windows around it for 0.33 s. After that the drone sank faster than asked,
    so the rule no longer suspected contact.
  - The manoeuvre exclusion cannot tell a commanded flare, where the rate comes before the residual,
    from a knock, where both come at once.
- **Version 3 is blind in 6.7% of the logged time** (361 s of 90.4 logged minutes). For the fast PD in the Minus
  garage it is up to 21.9 s per minute.
- **The rest scenario holds the motion level whatever the motor issues.**
  - brain-08 and fast-brain-11 at 0-2 m/s issue less thrust than the curve's hover there, so version 3
    never arms.
  - On the 61 live logs it armed at 3.82-4.83 s every time.

**Can the brain release graduate now? No.** This round changes two safety rules and nothing that
flies. No brain is selected, and none of this is flight evidence. Contact support version 3 fails 3
of its gates. The ceiling guard version 6 passes its gates, but on the 27 recorded flights it changes
no request: its effect shows only in the synthetic scenarios.

### Live plan changes (for the main session; development flights)

The procedure, the commands and the stop criteria of the round-4b plan are unchanged. A checkout of
`m5-safety` loads the round-5 declarations by default, so the same command line flies wall pilot 6
and descent view 3 with contact support on. The sidecar must declare:

| Declaration | Version | Content sha256 |
|---|---|---|
| `configs/obstacles/wall_pilot.json` | 6 | `fe65951d3d68...` |
| `configs/pilot/descent_view.json` (contact support: on / off / shadow as flown) | 3 | `2bdb17fc2479...` |

The other declarations are the round-4b ones. The CSV gains `contact_armed` and `contact_excluded`.
A wiring check parsed the plan's command lines with the runner's own parser and built its controller on
the CPU (no camera, pad or flight) for the fast PD (`on`, `shadow`) and fast-brain-11-b-cw13 (`on`,
`off`). Every case declared wall pilot 6 and descent view 3 with the mode given, and the eight view and
contact columns (three with `off`).

- **Recommended order.**
  1. A Minus Two run with the fast PD and `--contact-support shadow`. It measures live arming,
     excluded time and would-be fires without acting.
  2. Then the brain runs with `on`.
  3. On Straw Bale, compare `contact_fire` against the contact audit. Version 3 may miss a hard
     downhill touch that version 2 caught.
- **Watch:**
  - `ceiling_status` `overhead` outside a governor climb, with the vertical request cut under the
    garage ceiling;
  - `contact_armed` turning 1 at about 3.8-4.8 s. If it stays 0, the rule never armed;
  - `contact_excluded` during hard brakes and turns.
- **Stop the series** if an overhead hold engages on the Straw uphill and the drone loses height into
  the hill. A pilot's own ring climb is exempt, but the hold also brakes on below-path samples.

### Risks

- Contact support version 3 can miss a hard touchdown (held-out case). A contact inside 3.8-5 s
  after the first tick is not detected by it at all. Before arming, a drone resting on something
  after the launch could be learnt as thrust gain, but only in quiet or rising windows.
- The manoeuvre exclusion uses the runner's smoothed quaternion-derived body rate. A different
  drone, rate filter or frame rate changes what 2.5 rad/s excludes.
- The ceiling cut's exemption trusts the pilot's in-view ring climb. A ring close under the garage
  ceiling approached with lag could still overshoot. Top-edge (above) climbs are cut, so a ring above
  the image under a ceiling is climbed toward only after the hold.
- The overhead hold also brakes on below-path samples for 1 s. On a hill, a support or search climb
  with overhead evidence brakes on the rising ground instead of climbing.
- The harness and surrogate have no ceiling looming samples or ground reaction. The ceiling cut was
  shown only in synthetic scenarios and has never acted in closed loop.
- Every live log except the three round-4b ones was development evidence for version 3's values.

## Round 5 (offline): arches (branch `m5-arches`; details in [arches.md](../arches.md))

**Nothing in this section has flown.** The two round-4b clips are development cases. They were re-read from the logs,
the recorded frames and depth replays.

- **`minus-fast6-r4b-01`, 44.8 s: not an arch leg.**
  - The PD flew through the arch (checkpoint taken at 44.55 s) and hit the garage wall 1.5 m behind it.
  - Looming saw the wall from 43.76 s, 1.04 s before the impact (TTC 0.86 -> 0.35 s along the travel direction).
  - The governor's only cap lay along 28 deg, 180 deg from the flight: a stand-off from the first wall at 39.5 s.
  - Wall samples along 139 deg, then -151..-162 deg, renewed that stand-off. Their own targets were above the held
    1.03 m/s, so none of them braked.
  - The gap cue had nothing to see: the dark wall filled the band behind the arch.
- **`straw-brain11cw13-r4b-noassist-02`, 112.4 s: a false marker.**
  - At 111.34 and 111.41 s (captures) the live ring reader read a dark logo on a white fence banner beside the second
    start arch as the marker (u 0.605, 16.6 deg right). The recorded frames show the marker elsewhere.
  - The pilot and the lag turn swung the request from -3.6 to -20.8 deg.
  - The marker was then lost, and the coast held the turn into the arch's right leg.
  - The leg was visible in the depth (ratio 3.9-8.9 about the flown course from 5.6 m); looming read no evidence
    there.

| Rule | Declaration | Gates | Result |
|---|---|---|---|
| Stale evidence v1 | `stale_evidence_v1.json` `afcda589` (kept, refused) | v1 `81c36bf6` (`371c28d`, re-frozen `b835950`, scorer fix `68a090f`, scores `dcebaf8`) | Identity 27/27 x3. Minus development case: pass (brake 0.75 s before the impact). **Fails** the held-out logs (+2.43 and +1.93 m/s along the travel direction) and one fast-PD harness hairpin (a wall contact) |
| Stale evidence v2 | `stale_evidence.json` `4a951606`; opt-in `--stale-evidence on` | v2 `da57c268` (`6e58ff8`, scores `3c5f9d7`) | **Passes.** Identity 27/27 x3; version 1 reproduced 4/4. Minus clip (development, open loop): re-seat at 42.01 s; brake for the wall behind the arch from 43.76 s, 1.04 s before the impact (5.9 -> 4.24 m/s along the travel direction by 43.90 s). Other logs unchanged (development). Fresh held-out harness hairpins: identical for all 5 motors (the rule did not act) |
| Ring-marker reader v1 | `ring_marker.json` `09824e47`; `--ring-marker on` | v1 | Identity 8,193/8,193 frames. **Fails:** retention of overlay-confirmed markers 92.0-100% per held-out flight (pooled 95.1%); the dropped markers lie low in the image. Quiet Straw laps drop 6.3-6.6% of detections. The development false detections were not matched on the aligned recorded frames. **Do not fly** |

The default obstacle stack is unchanged (m4b, bit for bit).

**Live plan (development).** Fly the round-4b Minus Two fast-PD run with `--stale-evidence on` (command in
[arches.md](../arches.md#live-flight-plan-for-the-main-session-development-flights)). Watch for:

- a re-seat (`cap_reseat`) after the first garage wall;
- braking about 1 s before the wall behind the arch at (52.6, 94.1).

**Still open: the Straw Bale false marker.** Next: log the live reader's candidates in the next flights, then design a
reader rule on live frames.

## Round 5 (offline): the merged m5 stack, replays, surrogate and the live plan

**Nothing in this section has flown.** Branch `m5` is `m4b` (`2a5bccb`) with the four round-5 branches merged, in
this order: `m5-brake` (motor assist v3), `m5-brain12` (brain-12 gates, options and candidates; it already contains
`m5-brake`), `m5-arches` at `3c5f9d7` (stale-evidence rule v2, ring-marker reader rule v1), `m5-safety` (wall pilot v6,
descent view v3 with contact support v3) and, last, `m5-arches` again at `2b80b18` (its write-up, figures and card
section, and post-scoring fixes to comments, docstrings, sidecar metadata text and tests). Every live log replayed below
was read by at least one round-5 branch while it designed its rule, so **every replay here is development evidence**.
The surrogate and harness sets are development sets, except the fresh sets named below.

**Merges.** `m5-brake` and `m5-brain12` merged cleanly. The first `m5-arches` merge and `m5-safety` conflicted with the
branches before them in `fast_race_cue.py`, `visual_brain.py`, the replay harness, `deployed_pilot.py` and this card.
Every conflict was two independent additions, and both sides were kept:

- `FastRaceCue` takes `motor_assist` (v3), `clearance_ray`/`stale_apply` (stale evidence) and `contact_apply`
  (contact support shadow). `contact_support_config` parses version 3; the motor-assist constants are version 3's.
- The runner resolves `--ring-marker` and `--contact-support` and passes `stale_evidence`/`stale_apply` and
  `contact_support` to the controller. The CSV ends with the view and contact columns (8), the assist columns (8),
  then the stale-evidence columns (2).
- The replay harness takes `--stale-evidence`, `--cue-drop` and `--contact-support` (tags `-se2`, `-cd`,
  `-csoff`/`-csshadow`) beside `--motor-assist`.
- `deployed_pilot_kwargs` takes `stale_evidence=False` and `contact_support='on'`.

At `3c5f9d7` two tests failed on `m5-arches` itself (a comment in `fast_race_cue.py` named the declaration file, which
the no-file-reading test rejects, and the CSV-tail test predated the stale-evidence columns). The first merge fixed
both without a behaviour change; `m5-arches` then fixed them itself in `2b80b18`, and the second merge (`ac16bb8`)
took that branch's wording. Its other conflicts were comment hunks and this card's round-5 sections, which are kept in
the order brain-12, safety, arches. `2b80b18` changed no control code: the 27 flight-stack replays below, re-run on
the `ac16bb8` tree, equal the `96750a3` tree's replays in all 72 arrays of every log (27 of 27). `docs/arches.md` is
that branch's write-up; its live-plan command is replaced on `m5` by plan (a) below.

Every frozen declaration and gate file under `configs/` (54 files) kept its content and hash: the canonical hash equals
the file's own `sha256`, and the bytes equal the round-5 branch that changed it (or `m4b`). The runner's own loaders on
`m5` load:

| Declaration | Version | Content sha256 |
|---|---|---|
| `configs/obstacles/lag_turn.json` | 2 | `d4eb83da51ab...` |
| `configs/obstacles/gap_pilot.json` (gap cue `gap_cue.json` v2 `284b3c46a819...`) | 5 | `43c304204f93...` |
| `configs/obstacles/wall_pilot.json` (ceiling guard `any_climb`; clearance-brake floor) | 6 | `fe65951d3d68...` |
| `configs/obstacles/vertical_guard.json` | 4 | `409d06f9ded7...` |
| `configs/obstacles/stale_evidence.json` (opt-in: `--stale-evidence on`) | 2 | `4a9516068f1f...` |
| `configs/pilot/descent_view.json` (contact support v3; `--contact-support on\|off\|shadow`) | 3 | `2bdb17fc2479...` |
| `configs/pilot/motor_assist.json` (brain contract only) | 3 | `7c3b49e7bcc7...` |
| `configs/pilot/ring_marker.json` (off by default; **failed its gates, not flown**) | 1 | `09824e473659...` |

They refuse the kept wall pilot v1-v5, vertical guard v1-v3, descent view v1-v2, motor assist v1-v2, stale evidence v1
and gap pilot v1-v4 (checked with each loader).

Tests: the full suite passed after the first four merges (1203 passed in 372.32 s) and on `ac16bb8`, the code of the
final tree (**1204 passed in 242.02 s**); the integration commit after it adds only documents, which no test reads.
Scripts and raw outputs are in the session scratchpad under `m5/integrate/`. Results:
`docs/experiments/round5_integration.json`.

**Can the brain release graduate from pre-release now? No.** The proposed bar is one frozen stack that finishes Straw
Bale 3/3 with no ground contact and Minus Two 3/3. After round 5:

- **No brain is selected.** No brain-12 candidate passes its frozen gates. The best-ranked, `fast-brain-12-b-cw26d3`,
  passes 6 of 15, and so does `fast-brain-11-b-cw13` on the same gates.
- **Not every rule of the stack passes its own frozen gates.** Contact support v3 fails 3 of the 13 safety gates
  (rest, held-out detection, held-out false read). Motor assist v3 fails its quiet gate on Straw Bale (3.30-4.18% of the
  request travel removed on three laps; bound 3%). Vertical guard v4, wall pilot v5's floor (kept in v6) and descent
  view's view rule carry their round-4b failures. The contact audit's video false-positive check still fails.
- The stale-evidence rule v2 and the ceiling guard of wall pilot v6 pass their own gates. Neither has acted in closed
  loop: the stale-evidence rule changes only its development log, and the ceiling cut changes no recorded request
  except one merge interaction (below).
- **Nothing of round 5 has flown.**

The flights below are development flights. They can show whether the round-5 fixes work live; they cannot graduate
the release.

### Frozen gates of the four round-5 branches (as scored by each branch)

| Branch | Declaration (content sha256) | Gates; freeze / scores | Result |
|---|---|---|---|
| `m5-brake` | `motor_assist.json` v3 `7c3b49e7` (v1 `eefb4a42`, v2 `f3f35022` kept, refused) | gates v3 `f75a45e4`; `420b839` / `7003741` (v2: `7a15f25` / `87514c9`, 8 of 11) | **10 of 11.** Held-out hairpins (seed 41) with the assist: 7 / 9 / 10 of 12 clean (fast-brain-11-b-cw13 / brain-09b / fast-brain-10b); held-out hills time +0.04 / +0.41 / -0.79%; identity 102/102; no planned stop before the Minus hairpin; warning 1.20 / 1.11 s; slew 15.0 m/s². **Fails quiet** (Straw removal 3.30-4.18%) |
| `m5-safety` | `wall_pilot.json` v6 `fe65951d` (v5 kept); `descent_view.json` v3 `2bdb17fc` (v2 kept) | `safety_gates.json` v1 `c6dfc88c`; `68446ee` / `467ad37` | **10 of 13.** Every ceiling-guard gate passes; identity 162/162. Contact support v3 **fails** rest (brain-08 and fast-brain-11 never arm at 0-2 m/s), held-out detection (misses the Straw downhill touch v2 caught at +0.30 s) and held-out false reads (hilltop crest residual) |
| `m5-arches` | `stale_evidence.json` v2 `4a951606` (v1 `afcda589` kept, refused); `ring_marker.json` v1 `09824e47` | gates v2 `da57c268` (v1 `81c36bf6`); v1 `371c28d`+`b835950` / `dcebaf8` (scorer fix `68a090f` before scoring); v2 `6e58ff8` / `3c5f9d7`; write-up and post-scoring fixes `2b80b18` | **Stale evidence v2 passes every gate**: identity 27/27 x3, v1 reproduced 4/4, the Minus clip (first brake 2.79 s before the impact), 26 other logs unchanged, and the fresh held-out hairpin set (seed 29) unchanged for 5 motors (it never acted there). Stale evidence v1 and the **ring-marker rule failed** (the reader kept 95.1% of the held-out overlay-confirmed markers, 68,335 of 71,859, against a 99.5% bound per flight) |
| `m5-brain12` | candidates `runs/fast-brain-12-*` | `brain12_gates.json` v1 `8fb1b1a0`; `f0222ca` / `1836b79` | **None selected.** `fast-brain-12-b-cw26d3` (`2ecf3f1f`) 6/15: passes G7, G8, G9, G11, G12, G16 (8 of 12 hairpins clean with the assist); fails G1, G2, G3v2, G5, G6, G10, G13 (23 contacts in 34.88 s), G14 (hairpin capped excess 1.50), G15 |

What the two live clips were, after `m5-arches` looked at the recorded frames (details in `docs/arches.md`):

- **Minus Two, `minus-fast6-r4b-01`:** the fast PD flew **through** the arch at (52.6, 94.1) and hit the garage wall
  1.5 m behind it, at (51.09, 93.51) and 5.34 m/s. It did not clip an arch leg. A stand-off left from the first garage
  wall kept the governor's only cap along a ray 180 deg from the flight, so the looming samples of that wall never
  lowered it.
- **Straw Bale, `straw-brain11cw13-r4b-noassist-02`:**
  - The live ring reader read a dark logo on a white fence banner, just right of the second start arch's right leg, as
    the marker (captures at 111.34 and 111.41 s; u 0.605, 16.6 deg right).
  - The pilot and the lag turn swung the request from -3.6 to -20.8 deg.
  - The marker was then lost until 112.11 s, and the coast held the turn into the arch's right leg (112.376 s,
    5.45 m/s).
  - The ring-marker reader rule v1 failed its gates. The two false readings could not be matched on the aligned
    recorded frames, which are a separate 18 fps capture, and the held-out retention also failed.
  - So **this failure is not addressed in m5**.

### Identity: the round-5 rules off are m4b

Open-loop replays of the 27 logs (the 24 of round 4b and the three round-4b live flights) through a `git archive` of
`m4b` and of `m5` (`96750a3`), both with the `m5` harness: **91 of 91 command-array pairs are bit-identical**:

- the default pilot on all 27 logs;
- the stack in shadow on all 27;
- the m5 stack with every round-5 rule off (the kept `wall_pilot_v5.json` and `descent_view_v2.json`, no assist, no
  stale-evidence rule) against the `m4b` stack as flown in round 4b live (`--stack on --near-on-path --throttle-column
  command_thr --descent-view`), on all 27;
- the full m5 stack with and without `--motor-assist` on the 10 fast-PD logs (no PD entry).

In the harness (below), the `m4b` tree and the m5 tree's m4b-equivalent pilot (the kept v5 and v2 declarations, no
round-5 rule) give the same rows on **15 of 15** scenario sets (five sets for each of the fast PD,
fast-brain-11-b-cw13 and fast-brain-12-b-cw26d3, compared on the `m4b` rows' fields; the m5 rows add a contact-support
summary). The recorded logs were not re-flown, and the two r4-01 logs (`minus-fast6-r4-01`, `minus-brain10b-r4-01`)
have no ticks to replay.

### Open-loop replays: what the m5 stack changes, per log (development evidence)

Each log was replayed at its recorded states through:

- `m4b`: `--stack on --near-on-path --throttle-column command_thr --descent-view configs/pilot/descent_view.json`
  (wall pilot v5, descent view v2 with contact support v2, no assist), the stack flown live in round 4b;
- `m5`: the same flags on the m5 tree plus `--motor-assist configs/pilot/motor_assist.json --stale-evidence
  configs/obstacles/stale_evidence.json` (wall pilot v6, descent view v3 with contact support v3, motor assist v3,
  stale evidence v2): the stack the plan below flies.

Four more m5 variants each remove or put back one round-5 rule, to give each rule's share: no stale-evidence rule, no
assist, wall pilot v5 (isolates the ceiling cut) and descent view v2 (contact support v2 in place of v3). A last variant
runs contact support v3 in shadow. The Straw laps and `pine-brain08-01` use the offline looming stream. The recorded
motion does not respond to the requests, so the assist's cap tracking keeps pushing a drone that cannot slow; its
removal figures are an upper bound of what closed loop would show.

| Log | Motor | Request changed vs m4b, s (stale / assist / ceiling v6 / contact v3 vs v2) | Audited contacts | Contact-support onsets: m4b -> m5 (caught) | Assist: removed / stopping model, % of own travel; approach s/min; lowest plan (before x 73 on Minus) | Contact v3 armed at, s; excluded s/min |
|---|---|---|---|---|---|---|
| `minus-fast6-wall-01` | fast PD | 0 (0 / 0 / 0 / 0) | 0 | 0 | no PD entry (identical) | 4.4; 8.72 |
| `minus-brain08-gapon-01` | brain | 7.12 (0 / 7.12 / 0 / 0) | 0 | 0 | 15.40 / 6.09; 6.42; 2.71 | 4.2; 1.82 |
| `minus-brain08-gapon-02` | brain | 8.77 (0 / 8.77 / 0 / 0) | 0 | 0 | 23.28 / 7.74; 7.08; 2.6 | 4.19; 0.81 |
| `minus-fast6-gapon-01` | fast PD | 0 (0 / 0 / 0 / 0) | 0 | 0 | no PD entry (identical) | 4.4; 13.7 |
| `minus-brain08-loom-01` | brain | 2.85 (0 / 2.85 / 0 / 0) | 0 | 0 | 12.10 / 7.96; 6.73; 2.92 | 4.18; 0.0 |
| `minus-fast6-vg-02` | fast PD | 0 (0 / 0 / 0 / 0) | 0 | 0 | no PD entry (identical) | 4.75; 13.0 |
| `minus-brain08-vg-01` | brain | 6.49 (0 / 6.49 / 0 / 0) | 0 | 0 | 15.56 / 6.85; 6.63; 2.61 | 4.19; 0.65 |
| `minus-brain09b-vg-01` | brain | 7.1 (0 / 7.1 / 0 / 0) | 0 | 0 | 11.31 / 8.30; 7.62; 2.5 | 4.32; 3.62 |
| `minus-brain08-gapshadow-01` | brain | 3.73 (0 / 3.73 / 0 / 0) | 0 | 0 | 15.40 / 7.61; 6.68; 2.5 | 4.17; 0.0 |
| `minus-fast6-r4-02` | fast PD | 0 (0 / 0 / 0 / 0) | 3 | 0 | no PD entry (identical) | 4.39; 21.98 |
| `minus-brain10b-r4-02` | brain | 5.53 (0 / 5.53 / 0 / 0) | 0 | 0 | 11.30 / 6.13; 5.15; 2.5 | 3.82; 0.0 |
| `minus-brain09b-r4-01` | brain | 8.5 (0 / 8.5 / 0 / 0) | 0 | 0 | 12.52 / 8.12; 6.58; 2.5 | 4.35; 7.12 |
| `pine-fast6-ttc-01` | fast PD | 0 (0 / 0 / 0 / 0) | 1 | 0 | no PD entry (identical) | 4.41; 9.38 |
| `pine-brain08-loom-01` | brain | 0.21 (0 / 0.21 / 0 / 0) | 0 | 0 | 0.28 / 0.00; 0.00; None | 4.23; 0.0 |
| `pine-brain08-01` | brain | 5.76 (0 / 5.76 / 0 / 0) | 1 | 0 | 1.59 / 0.21; 0.94; 3.41 | 4.24; 2.17 |
| `straw-brain08-04` | brain | 82.91 (0 / 82.91 / 0 / 0) | 7 | 10 -> 10 (7 -> 7 of 7) | 1.98 / 0.89; 1.70; 2.5 | 4.21; 1.9 |
| `straw-brain08-06` | brain | 90.58 (0 / 90.58 / 1.28 / 0) | 6 | 6 -> 6 (6 -> 6 of 6) | 2.43 / 1.07; 1.94; 0.0 | 4.2; 1.8 |
| `straw-brain08-01` | brain | 19.63 (0 / 19.63 / 0 / 0) | 1 | 4 -> 4 (1 -> 1 of 1) | 3.30 / 1.55; 2.61; 2.5 | 4.22; 2.94 |
| `straw-brain08-02` | brain | 43.31 (0 / 43.31 / 0 / 0) | 2 | 2 -> 2 (2 -> 2 of 2) | 3.56 / 0.87; 1.74; 1.46 | 4.21; 2.12 |
| `straw-brain08-03` | brain | 13.64 (0 / 13.64 / 0 / 0) | 0 | 0 | 4.18 / 1.98; 3.29; 2.5 | 4.21; 2.98 |
| `straw-fast6-01` | fast PD | 0 (0 / 0 / 0 / 0) | 0 | 0 | no PD entry (identical) | 4.37; 5.93 |
| `straw-fast6-02` | fast PD | 0 (0 / 0 / 0 / 0) | 0 | 0 | no PD entry (identical) | 4.4; 8.24 |
| `straw-fast6-03` | fast PD | 0 (0 / 0 / 0 / 0) | 0 | 0 | no PD entry (identical) | 4.39; 7.51 |
| `straw-fast6-arc-01` | fast PD | 0 (0 / 0 / 0 / 0) | 0 | 0 | no PD entry (identical) | 4.4; 8.77 |
| `minus-fast6-r4b-01` | fast PD | 2.65 (2.65 / 0 / 0 / 0) | 0 | 0 | no PD entry (identical) | 4.39; 18.47 |
| `minus-brain11cw13-r4b-noassist-01` | brain | 7.78 (0 / 7.78 / 0 / 0) | 0 | 0 | 13.66 / 10.08; 7.98; 2.5 | 3.84; 1.6 |
| `straw-brain11cw13-r4b-noassist-02` | brain | 12.9 (0 / 11.99 / 0 / 1.5) | 1 | 1 -> 1 (1 -> 1 of 1) (79.65 -> 80.08 s) | 0.52 / 0.37; 0.94; 3.41 | 3.84; 0.52 |

"Caught" counts audited contacts with an onset from 0.3 s before to 1.5 s after them. The lowest plan is the stopping
model's lowest bound before x 73 m (the Minus hairpin); on Straw and Pine x 73 m lies beyond the whole course.

- **Stale-evidence rule v2** changes only `minus-fast6-r4b-01`, from 42.01 s to the impact (2.65 s of changed
  requests). All figures are open-loop requests at the recorded states:
  - At 42.01 s a wall sample along 139 deg re-seats the cap (the old cap lay along 28 deg, 1.12 m/s in this replay). The
    horizontal request falls to 4.13 m/s by 42.30 s (lowest 3.79 m/s at 42.79 s, where `m4b` asks 4.51); `m4b` asks
    6.0 m/s until 42.5 s.
  - From 43.76 s (the first sample of the wall behind the arch, TTC 0.86 s; 1.04 s before the impact at 44.797 s) the
    cap on the travel ray falls. The request falls from 5.9 to 4.25 m/s by 43.90 s and stays at 4.00-4.25 m/s to the
    impact. Over 43.90 s to the impact `m4b` asks 4.74-5.99 m/s. At the last tick the request along the travel
    direction is 3.75 m/s (`m4b` 4.52).
    The recorded drone flew 5.3-6.1 m/s over that interval and does not slow in a replay.
  - It changes nothing on the other 26 logs, the PD's first garage wall included.
- **Motor assist v3** changes every brain log and no fast-PD log.
  - Minus Two: no planned stop before the hairpin; the lowest plan before x 73 m is 2.5-2.92 m/s on the nine brain logs.
    The final continuous cut of at least 1 m/s starts 1.20 s (fast-brain-11) and 1.11 s (fast-brain-10b) before the
    hairpin impact, as `m5-brake` scored on its own branch. On `minus-brain11cw13-r4b-noassist-01` the approach bound
    acts from 22.33 s (x 76.9) and the stopping source from 23.02 s (x 79.4); the impact was at 23.596 s. In open loop
    the assist removes 11.3-23.3% of the pilot's own request travel on the Minus brain logs (the stopping model
    6.1-10.1%).
  - Straw Bale: 1.98-4.18% of the request travel on the brain-08 laps (the quiet failure) and 0.52% on fast-brain-11's
    lap; the approach acts 1.70-3.29 and 0.94 s per minute. On `straw-brain08-06` the marker was lost at the start arch
    (7.71-7.82 s, x 23.6, y 0.4) and the stopping source planned a stop (0 m/s) for 0.11 s; the slewed request fell to
    4.19 m/s.
  - At the lap-2 start arch of `straw-brain11cw13-r4b-noassist-02` (110.8-112.4 s), where the looming read no
    evidence, it changes no horizontal request; it adds up to 0.04 m/s of vertical request for three ticks
    (112.08-112.10 s, search).
- **Wall pilot v6's ceiling cut** changes one request window, a **merge interaction** that neither branch could see: on
  `straw-brain08-06` at 37.49-38.77 s (x 36.6, y 102.6, z 7.6 m, Straw uphill), the assist's sag climb removes the
  exemption of the pilot's own ring climb (`state == 'cue'` without a sag climb), and an overhead sample then cuts the
  vertical request. It is held at 0 m/s from 37.60 to 38.49 s and is 0-1.43 m/s over the window, where wall pilot v5
  asks 1.47-1.76 m/s. The safety branch scored the ceiling cut without the assist (no change on any Straw log) and the
  brake branch had no v6. It is the only change of the ceiling cut on the 27 logs.
- **Contact support v3 against v2**, both inside the m5 stack:
  - the four brain-08 Straw logs with audited slides: identical onsets (22), catching every one of the 16 audited
    slides, as `m4b`;
  - the held-out Straw touch of `straw-brain11cw13-r4b-noassist-02` (audited 79.354-80.044 s): v3 fires at 80.08 s,
    0.73 s after the touch began and 0.04 s after it ended, with or without the assist; v2 fires at 79.65 s. The
    safety gate's detector, fed the logged command, recorded no v3 fire for this touch; here the rule sees the
    replayed stack's own requests;
  - no onset on any other log; arming at 3.82-4.75 s on all 27 logs; excluded (blind) 0-7.12 s per minute on the brain
    logs and 5.9-22.0 s per minute on the fast-PD logs.
- The vertical guard, the lag turn and the gap aim are `m4b`'s.

### Contact audit of the 27 logs (`contact_audit.json` v1 `1c82c7f4`; report, not a gate)

38.18 scored minutes, 25 contacts. The 24 earlier logs give exactly round 4b's contacts.

- **Minus Two:** `minus-fast6-r4-02`'s three floor supports (27.33-31.06 s) and its ceiling impact (34.08 s);
  `minus-brain08-gapon-02`'s ceiling impact (2.13 m). Every other Minus log has only its terminal impact, including
  `minus-fast6-r4b-01` (garage wall behind the arch, 44.797 s, 5.34 m/s) and `minus-brain11cw13-r4b-noassist-01`
  (hairpin wall, 23.596 s, 3.73 m/s).
- **Pine Valley:** `pine-fast6-ttc-01`'s tree impact (16.7 s) and `pine-brain08-01`'s hillside support (16.72-17.95 s).
- **Straw Bale:**
  - fast-brain-08: 16 downhill slides (`-04` 7, `-06` 6, `-01` one 7.7 s slide, `-02` 2) and `-02`'s impact at 164.44 s;
  - fast PD: no contact in 15.8 scored minutes, including the two 3-lap finishes `straw-fast6-02` and `-03`;
  - fast-brain-11-b-cw13: one support on the downhill in lap 1 (79.354-80.044 s at (-36.58, 132.78), 5.16 m/s), then
    the start-arch impact (112.376 s).

### Harness and surrogate (closed loop in the identified simulator; not flight evidence)

**Harness** (`motor_assist_eval.run_scenarios`: synthetic walls, looming samples and ceiling, 12 scenarios per hairpin
set). The sets:

- the v1 hairpins (seed 23) and the accelerate set of motor-assist gates v1 (development);
- the v3 held-out hairpins (seed 41), which `m5-brake` scored;
- a **fresh** hairpin set chosen here before any run and used by no gate or screening: turn 55/85/120 deg x arch
  9.5/13.5 m x wall 2.2/2.65 m, sim seed 67. It is also run with live-like wall samples (no below fraction and no
  lower-surface TTC, as the recorded Minus walls carry);
- the v3 pass-through report set (8 arches with a looming surface beyond the ring; its looming is not validated
  against Liftoff).

Cells read: clean passes (wall / floor / ceiling contacts). The `m4b` rows are the `m4b` tree. For the pass-through
set, which the `m4b` harness lacks, they are the m5 tree's m4b-equivalent pilot. The m5 rows ran on the `96750a3`
tree; `2b80b18` changed no control code.

| Motor | Stack | v1 hairpins (seed 23) | v3 hairpins (seed 41) | Fresh hairpins (seed 67) | Fresh, live-like samples | Accelerate (6): clean, mean height loss | Pass-through (8): finished; lowest speed at the arch |
|---|---|---|---|---|---|---|---|
| fast PD | m4b | 10 (2 / 0 / 2) | 8 (3 / 0 / 2) | 9 (2 / 0 / 3) | 8 (2 / 0 / 3) | 4, 0.48 m | 5; 0.42-2.48 m/s |
| fast PD | m5 | 11 (0 / 0 / 1) | 8 (2 / 0 / 2) | 10 (1 / 0 / 1) | 10 (1 / 0 / 1) | 4, 0.48 m | 5; 0.42-2.48 m/s |
| fast-brain-11-b-cw13 | m4b | 0 (12 / 0 / 2) | 0 (12 / 0 / 2) | 0 (12 / 0 / 3) | 0 (12 / 0 / 3) | 3, 0.56 m | 8; 4.04-4.65 m/s |
| fast-brain-11-b-cw13 | m5 without the assist | 0 (12 / 0 / 0) | 0 (12 / 0 / 0) | 0 (12 / 0 / 0) | 0 (12 / 0 / 0) | 3, 0.56 m | 8; 4.04-4.65 m/s |
| fast-brain-11-b-cw13 | m5 | 7 (1 / 0 / 2) | 9 (0 / 0 / 1) | 10 (0 / 0 / 0) | 10 (0 / 0 / 2) | 3, 0.39 m | 6; 0.05-1.06 m/s |
| fast-brain-12-b-cw26d3 | m4b | 0 (12 / 0 / 2) | 0 (12 / 1 / 2) | 0 (12 / 0 / 3) | 0 (12 / 0 / 3) | 3, 0.12 m | 8; 4.46-4.88 m/s |
| fast-brain-12-b-cw26d3 | m5 without the assist | 0 (12 / 0 / 0) | 0 (12 / 1 / 0) | 0 (12 / 0 / 0) | 0 (12 / 0 / 0) | 3, 0.12 m | 8; 4.46-4.88 m/s |
| fast-brain-12-b-cw26d3 | m5 | 9 (3 / 0 / 1) | 12 (0 / 0 / 0) | 12 (0 / 0 / 0) | 12 (0 / 0 / 0) | 3, 0.05 m | 7; 0.29-1.15 m/s |

| Motor | Stack | Dev sets (28): finished / crashed | Terrain contacts / s | High passes | Support climbs | Chatter (16) | Mean terrain finish s | Fresh sets (20): finished / crashed | Terrain contacts / s | High passes | Mean finish s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fast PD | m4b | 28 / 0 | 11 / 15.64 | 14 | 13 | 0.00634 | 56.32 | 20 / 0 | 15 / 21.59 | 16 | 52.4 |
| fast PD | m5 | 28 / 0 | 11 / 15.64 | 14 | 13 | 0.00634 | 56.32 | 20 / 0 | 15 / 21.59 | 16 | 52.4 |
| fast-brain-11-b-cw13 | m4b | 28 / 0 | 11 / 26.11 | 12 | 6 | 0.00239 | 61.86 | 20 / 0 | 14 / 35.69 | 8 | 60.38 |
| fast-brain-11-b-cw13 | m5 without the assist | - | - | - | - | - | - | 20 / 0 | 14 / 35.69 | 8 | 60.38 |
| fast-brain-11-b-cw13 | m5 | 28 / 0 | 11 / 25.15 | 14 | 7 | 0.00247 | 62.07 | 20 / 0 | 14 / 34.25 | 7 | 60.44 |
| fast-brain-12-b-cw26d3 | m4b | - | - | - | - | - | - | 20 / 0 | 18 / 30.3 | 11 | 60.09 |
| fast-brain-12-b-cw26d3 | m5 without the assist | - | - | - | - | - | - | 20 / 0 | 18 / 30.3 | 11 | 60.09 |
| fast-brain-12-b-cw26d3 | m5 | 28 / 0 | 16 / 24.12 | 14 | 8 | 0.00259 | 61.39 | 20 / 0 | 16 / 27.7 | 11 | 60.31 |

- **Brains need the assist at hairpins.** Without motor assist v3, both brains hit the wall in all 12 scenarios of
  every hairpin set, under `m4b` and `m5` alike. With it:
  - fast-brain-11-b-cw13: 7 / 9 / 10 / 10 clean of 12 (v1 / v3 / fresh / fresh with live-like samples);
  - fast-brain-12-b-cw26d3: 9 / 12 / 12 / 12 clean.

  Stick change per tick on the hairpin sets is 33-48% higher with the assist (fast-brain-11 on the fresh set: 0.00363
  -> 0.00501). The runs without the assist end at the wall, so the runs are not like for like.
- **Crawls at pass-through arches.** With the assist, the lowest speed near the arch falls:
  - fast-brain-11: from 4.04-4.65 to 0.05-1.06 m/s, with 2 of the 8 scenarios unfinished in the scenario time;
  - fast-brain-12: from 4.46-4.88 to 0.29-1.15 m/s, with 1 of 8 unfinished.

  This is `m5-brake`'s reported risk: the governor's stand-off holds a slowed brain. The fast PD is unchanged (5 of 8
  finished, 3 crashed, in both stacks).
- **Fast PD, m4b -> m5:**
  - hairpins: v1 10 -> 11 clean (walls 2 -> 0), v3 8 -> 8 (walls 3 -> 2), fresh 9 -> 10, fresh live-like 8 -> 10;
  - on the hairpin sets the highest climb falls from 3.5-4.1 m to 2.3-2.5 m (the accelerate set is unchanged, 4.49 m).

  Report-only attribution: putting contact support v2 back into m5 gives `m4b`'s counts on the v3, fresh and fresh
  live-like sets (v1: 1 wall contact against 2). Removing the stale-evidence rule or wall pilot v6 alone changes no
  count. The gain comes from contact support v2's false fires, which v3 does not make: under v2, every hairpin
  scenario with a ceiling contact had 2-4 support onsets, and v3 made no onset in any harness scenario. The accelerate
  set's ceiling contacts come without onsets and are unchanged. The fast PD's remaining ceiling contacts under `m5`
  (1-2 per hairpin set) come from other climbs. The brains' 2-3 ceiling contacts per hairpin set under `m4b` were
  v2's too: they drop to 0 under `m5` without the assist.
- **Surrogate, fast PD:** the descent surrogate has no looming or gap samples, so the gap aim, turn-first, the ceiling
  guard, the vertical guard, the clearance brake and the stale-evidence rule stay idle there. `m5` equals `m4b` course
  for course. Every row field agrees except the contact-support summary; neither version makes an onset in any surrogate
  course.
- **Surrogate, brains:** `m5` without the assist equals `m4b` course for course on the fresh sets, for both brains.
  With the assist:
  - fast-brain-11: 14 contacts in 35.69 s -> 14 in 34.25 s on the fresh sets; mean finish 60.38 -> 60.44 s;
  - fast-brain-12: 18 in 30.3 s -> 16 in 27.7 s.
- **The user's ground-contact request is not met by any motor in the surrogate.** On the fresh sets under `m5`:
  - fast PD: 15 contacts, 21.59 s;
  - fast-brain-11: 14 contacts, 34.25 s;
  - fast-brain-12: 16 contacts, 27.7 s.

  Round 5 changed no rule that requests less sink near terrain.

### Live flight plan (for the main session; development flights)

Fly from `C:\DEV\Haltere` with `m5` checked out. It is a local branch and is not pushed. While the integration worktree
still holds the branch, use `git switch --detach m5` in the main checkout, or remove that worktree first. The runner
loads its declarations from the checkout it runs from and refuses other versions. Keep the untracked
`configs/explore_spiral.yaml`.

This plan replaces the live plan of the arches section above and of `docs/arches.md`. Its run (a) uses the same log
name and adds `--contact-support on`.

The flight procedure is unchanged:

- Liftoff and every capture, pad and controller process run inside Anode, with the viewer hidden.
- The pad must report `seatOnly`.
- Run the ground check, and verify throttle-low and a real processed control response.
- Nothing else heavy may run. The preflight refuses a busy machine.
- Use a new log name for every attempt, and keep Liftoff open between runs.

**One stack for every course:**

```
--looming-brake --obstacle-stack on --stale-evidence on --descent-view on --contact-support on --motor-assist on
```

`--contact-support on` is the default and is written out so the log records the choice. The fast PD has no
motor-assist entry: its commands are bit-identical with and without the flag, and its sidecar records
`applied: false`. `--ring-marker` stays off: its rule failed its held-out gates. A wiring check parsed every command
below with the runner's own argparse and built what `run()` builds before the camera starts, with `run()`'s own keyword
arguments (no camera, pad, preflight record or flight). Each run declares:

| Declaration | Version | Content sha256 | Applied |
|---|---|---|---|
| `configs/obstacles/lag_turn.json` (the contract's entry) | 2 | `d4eb83da51ab...` | yes |
| `configs/obstacles/gap_pilot.json` (gap cue v2) | 5 | `43c304204f93...` | yes |
| `configs/obstacles/wall_pilot.json` (stopping model: the motor's contract; ceiling `any_climb`; brake floor) | 6 | `fe65951d3d68...` | yes |
| `configs/obstacles/vertical_guard.json` | 4 | `409d06f9ded7...` | yes |
| `configs/obstacles/stale_evidence.json` (CSV adds `cap_ray_deg`, `cap_reseat`) | 2 | `4a9516068f1f...` | yes |
| `configs/pilot/descent_view.json` (contact support `on`; CSV adds `contact_unexplained`, `contact_gain`, `contact_fire`, `contact_armed`, `contact_excluded`) | 3 | `2bdb17fc2479...` | yes |
| `configs/pilot/motor_assist.json` (CSV adds the eight `assist_*` columns, with `assist_plan` and `assist_wall_ahead`) | 3 | `7c3b49e7bcc7...` | brain: yes; fast PD: no |

The camera reads the ring with the earlier reader (no ring-marker rule).

**Every run is a disclosed development deviation:**

- no brain is selected;
- contact support v3 and motor assist v3 fail frozen gates, and guard v4, the wall pilot's brake floor and the
  descent view's view rule carry their round-4b failures;
- the stale-evidence rule's only evidence of an effect is one development log, open loop;
- the contact audit cannot gate.

**Brain: `fast-brain-11-b-cw13`** (`runs/fast-brain-11-b-cw13/candidate.pt`, sha256 `44cca3c4...`), not a brain-12
candidate, on both Minus Two and Straw Bale, so that one stack is tested. Why:

- No brain-12 candidate is selected. The best-ranked, `fast-brain-12-b-cw26d3`, ties fast-brain-11-b-cw13 at 6 of 15
  on the brain-12 gates.
- In the round-5 harness under the m5 stack, brain-12 does better at hairpins. It has 12 of 12 clean on the v3,
  fresh and fresh live-like sets, where brain-11 has 9 / 10 / 10. It also finishes 7 of 8 pass-through arches, where
  brain-11 finishes 6. It is worse where Minus asks for cap following: it overshoots sustained 3-3.5 m/s requests by
  0.63-0.72 m/s (G2), has a cap excess of 0.585 (G5; brain-11 0.413), and follows the late live hairpin cap worse
  without the assist (G14: 1.50 against 1.17). Its surrogate terrain contacts are not fewer: 16 in 24.12 s against 11
  in 25.15 s on the dev sets under m5, and 16 in 27.7 s against 14 in 34.25 s on the fresh sets. Neither brain is
  clearly better, and neither passes.
- The same brain on both courses makes (b) and (c) direct comparisons with the round-4b live runs of this brain
  (`minus-brain11cw13-r4b-noassist-01`, `straw-brain11cw13-r4b-noassist-02`). The difference is then the round-5
  stack, not the motor.
- fast-brain-11-b-cw13 is the only braking brain with live evidence under this pilot: Straw Bale lap 1 in 1:42.988
  with one audited touch, and its known faults showed live (the Minus hairpin).

Its known faults: over-braking for requests left of its heading (G3v2, G6, G10), a slow 60-degree left
re-acceleration (G11), and no brake on the late live hairpin cap without the assist (G14).

**Motor assist v3 is on for the brain runs** (`--motor-assist on`; for the fast PD it declares v3 with no entry and
changes nothing). Why:

- Without it the brains do not stop at hairpins. In every harness hairpin set, both brains hit the wall in 12 of 12
  scenarios without the assist, under `m4b` and `m5` alike. With it fast-brain-11 has 7-10 of 12 clean. The r4b live
  hairpin crash of this brain flew without it.
- It passes 10 of its 11 frozen gates, among them the held-out hairpins, the held-out hills, identity, the slew bound
  and no planned stop before the Minus hairpin.
- It fails its Straw quiet gate on fast-brain-08's laps (3.30-4.18% of the request travel removed; bound 3%). On
  fast-brain-11's lap it removes 0.52% (open loop).

Its known costs:

- crawls behind the governor's stand-off at arches: in the pass-through report fast-brain-11's lowest speed falls from
  4.04-4.65 to 0.05-1.06 m/s, and 2 of 8 do not finish in the scenario time;
- 33-48% more stick change per tick in the harness hairpins;
- the merge interaction with the ceiling cut on the Straw uphill (below).

These are watch items in (b) and (c).

**Before every launch** (in the seat, from `C:\DEV\Haltere`):

1. The machine must be quiet. The runner's preflight refuses competing workloads, for example the user's `rustc.exe`
   compiles (5.3 cores refused `straw-brain11cw13-r4b-noassist-01`). A refusal writes `<log>.preflight.json`, so that
   log name cannot be reused: wait until the compile has finished and use the next number (`-02`). This prints the
   same check without writing anything:

   ```powershell
   .venv/Scripts/python.exe -m haltere.liftoff.preflight
   ```

2. Ground check (it sends throttle-low, then throttle, roll, pitch and yaw, and checks the processed controls):

   ```powershell
   .venv/Scripts/python.exe runs/fast-stack-20260923/ground_check.py runs/fast-stack-20260923/ground-check-34.json
   ```

   Increment the number for every check (34, 35, ...; 33 was the last).

3. The ground-check script **pauses the game when it exits**, and the pause inside it cannot be disabled; the runner's
   `--pause-on-stop` pauses it after every run too. So after the ground check and after every run, click
   **Réinitialiser** (646,277) in the pause menu, or resume.
4. Confirm that telemetry is live. This prints `LIVE` when two frames 0.5 s apart show an advancing game timestamp:

   ```powershell
   .venv/Scripts/python.exe -c "import time; from haltere.liftoff.telemetry import TelemetryReceiver,read_config,DEFAULT_STREAM; rx=TelemetryReceiver(port=9001,stream=(read_config() or {}).get('StreamFormat',DEFAULT_STREAM)); a=rx.wait(1.0); time.sleep(0.5); b=rx.wait(1.0); rx.close(); print('LIVE' if a is not None and b is not None and b.timestamp > a.timestamp else 'NOT LIVE', None if b is None else [round(float(v), 2) for v in b.position])"
   ```

   Launch only after `LIVE`. Otherwise the run stops at 0 ticks with "No fresh live image/telemetry", as
   `minus-fast6-r4-01` and `minus-brain10b-r4-01` did.

Run the flights in this order. Each command is one line.

**(a) Minus Two, fast PD.**

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-08-vgs04-s10r03m30/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller pd --pd-profile fast --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --stale-evidence on --descent-view on --contact-support on --motor-assist on --seconds 150 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/minus-fast6-r5-01.csv --record runs/fast-stack-20260923/minus-fast6-r5-01.mp4 --video-encoder h264_nvenc
```

What to look for (replay figures are open-loop requests at the recorded states of `minus-fast6-r4b-01`):

- **Pillar A, the hairpin, pillar C and round 4's floor-and-ceiling spot** as in r4b-01: no round-5 rule changed any
  request there.
- **The first garage wall (38.8-39.6 s in r4b-01):** the stand-off as before. Version 2 keeps it while it is active.
- **After that wall:** `cap_reseat` should step to 1 at the first confirmed wall sample along the new flight direction
  (42.01 s in the replay), and `cap_ray_deg` should follow the travel. The status `standoff` should not persist while
  the drone flies at 6 m/s.
- **The arch at about (52.6, 94.1) and the garage wall 1.5 m behind it:** in the replay the cap on the travel ray falls
  from 43.76 s, 1.04 s before the r4b impact. The request is 4.25 m/s by 43.90 s and 4.00-4.25 m/s to the impact, where
  `m4b` asks 4.74-5.99 m/s. Live, each new sample should lower it further as the drone slows, so the PD should stop
  short of the wall or turn left along it, as at the first wall.
- **Contact support v3:** `contact_armed` should turn 1 at about 3.8-4.8 s (4.39 s in the replay). For the fast PD in
  the garage the rule is blind (`contact_excluded`) up to 18-22 s per minute.
- **Ceiling:** under the 2.2 m garage ceiling, `clearance_status` `overhead` outside a governor climb should cut the
  vertical request (wall pilot v6); it changed no recorded Minus request.
- The sidecar records `motor_assist_declaration.applied: false`. Beyond (51, 93.5) nothing has been flown.

**(b) Minus Two, fast-brain-11-b-cw13.**

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-11-b-cw13/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller brain --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --stale-evidence on --descent-view on --contact-support on --motor-assist on --seconds 150 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/minus-brain11cw13-r5-01.csv --record runs/fast-stack-20260923/minus-brain11cw13-r5-01.mp4 --video-encoder h264_nvenc
```

What to look for (replay figures from `minus-brain11cw13-r4b-noassist-01`, open loop):

- **First arch and pillar A:** the assist's approach bound slows the brain toward every short-TTC surface, never below
  2.5 m/s (lowest plan before x 73: 2.5 m/s on this log). Watch for crawls: once a brain is slowed, the governor's own
  stand-off can hold it near 1 m/s for about 2 s at an arch (the pass-through report; r4b showed it at x 22-26 without
  any assist). More than about 3 s parked is a stop criterion.
- **Hairpin (wall near (81.9, 20.0)):** in the replay the approach bound acts from 22.33 s (x 76.9) and the stopping
  source, under wall-ahead conditions, from 23.02 s (x 79.4). The final continuous cut of at least 1 m/s starts 1.20 s
  before the r4b impact (the governor alone: 0.72 s). About 2.6-3.0 m/s or less toward +x at the arch is needed.
- **Turn-first engagement and release**, then the acceleration out: no sink to the floor.
- **Left capped turns** (pillar A's left commitment, the turn after the hairpin): this brain's known fault.
- **Ceiling:** support or assist climbs near the 2.2 m ceiling (`contact_fire`, `assist_vertical`).
- **Stick change per tick:** 0.0038 in r4b.
- Then the floor, the garage walls and the arch as in (a).

**(c) Straw Bale, fast-brain-11-b-cw13, three laps.** This is the graduation course and the user's downhill request.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-11-b-cw13/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller brain --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --stale-evidence on --descent-view on --contact-support on --motor-assist on --seconds 480 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/straw-brain11cw13-r5-01.csv --record runs/fast-stack-20260923/straw-brain11cw13-r5-01.mp4 --video-encoder h264_nvenc
```

After the run, score ground contact from telemetry with the contact audit, not from support-climb onsets:

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.contact_audit runs/fast-stack-20260923/straw-brain11cw13-r5-01.csv --json runs/fast-stack-20260923/straw-brain11cw13-r5-01.contact-audit.json
```

Its false-positive check failed on video, so report its contacts next to the video; they do not decide pass or fail.
The same command, with the run's name, works for every run.

What to look for (replay figures from `straw-brain11cw13-r4b-noassist-02`, open loop):

- **Downhill (x about -37, y 130-170):** r4b lap 1 had one audited touch (79.35-80.04 s, 5.2 m/s). In the replay
  contact support v3 fires at 80.08 s, just after the touch ended, with or without the assist (v2: 79.65 s). The safety
  gate's detector, fed the logged command, recorded no v3 fire there. A knock-style touchdown may get no support climb.
- **Uphill rings:** the assist's approach acts 0.94 s per minute on this brain's lap and removes 0.52% of its request
  travel (brain-08 laps: 1.7-3.3 s per minute, 2.0-4.2%). Compare the lap times with 1:42.988.
- **Uphill, a merge interaction:** while the assist adds a sag climb (`assist_vertical` > 0), the pilot's own ring
  climb loses its exemption from the ceiling cut. On `straw-brain08-06` (37.49-38.77 s, x 36.6, y 102.6) an overhead
  sample then cut the vertical request to 0-1.43 m/s, and to 0 m/s from 37.60 to 38.49 s, where wall pilot v5 asked
  1.47-1.76 m/s. It happened on no other log.
- **Hilltop:** the ring marker drops out there; the gentle search must hold height.
- **The start arches in laps 2 and 3** (x 23-27, y -0.5 to 0.4): the false marker on the fence banner is not
  addressed, and the looming reads no evidence there against the bright sky. If the marker is lost, the assist's
  stopping source may act (on `straw-brain08-06` it planned a stop for 0.11 s at 7.71 s; the request fell to 4.19 m/s).

**(d) Pine Valley, fast PD.**

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-08-vgs04-s10r03m30/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller pd --pd-profile fast --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --stale-evidence on --descent-view on --contact-support on --motor-assist on --seconds 150 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/pine-fast6-r5-01.csv --record runs/fast-stack-20260923/pine-fast6-r5-01.mp4 --video-encoder h264_nvenc
```

What to look for: as round 4b's plan (d). No round-5 rule changed any request of `pine-fast6-ttc-01` in the replay.

- **Mound:** the escalation near 4.55 s, climbing up to 3.5 m/s.
- **Backside at 6.3-7.4 s:** 0.3-0.54 m/s of sink at about 5-5.7 m/s.
- **Hillside at 15.2 s:** the guard asks for 1 m/s; the brake adds no sink there.
- **End hillside:** the guard escalates at about 19.05 s.
- **Boulder:** no rule arrests the descent into it.

**(e) Optional diagnostic after (b): Minus Two with `fast-brain-12-b-cw26d3`** (`2ecf3f1f...`, not selected; same
stack). It shows whether brain-12's surrogate hairpin gain holds live. Its gates predict +0.6-0.7 m/s over sustained
3-3.5 m/s requests and slower cap following.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-12-b-cw26d3/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller brain --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --stale-evidence on --descent-view on --contact-support on --motor-assist on --seconds 150 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/minus-brain12cw26d3-r5-01.csv --record runs/fast-stack-20260923/minus-brain12cw26d3-r5-01.mp4 --video-encoder h264_nvenc
```

**Stop criteria.**

- **Before a run, do not start if:**
  - the preflight is not quiet;
  - the pad is not `seatOnly`, or the guard unplugs it (a user's game alone is not a reason);
  - the ground check fails;
  - telemetry is not `LIVE`.
- **After a run, stop the series if:**
  - the sidecar declares any other version or hash than the table above, `motor_assist_declaration.applied` is wrong
    for the motor, `stale_evidence_declaration` is missing, or the descent view's `contact_support` is not `on`;
  - a controller deadline or camera failure was recorded.
- **During a run, stop the runner if:**
  - the drone climbs above about 1.9 m in the Minus garage, or keeps climbing above 4 m elsewhere;
  - a support climb starts with the drone clearly airborne (`contact_fire` 1 while the video shows no ground);
  - a turn-first hold, an assist stop or a governor stand-off parks the drone for more than about 3 s;
  - it circles in search for more than about 30 s;
  - on the Straw uphill, an overhead hold (`clearance_status` `overhead`, vertical request 0) lets the drone lose height
    into the hill;
  - a stale-evidence re-seat is followed by a contact with the earlier wall.
- **Between runs:** stop and analyse instead of flying the next run if a round-5 rule causes a new failure. First
  replay the log (development evidence), from `C:\DEV\Haltere` with `m5` checked out; `LOGNAME` is the log's name
  without `.csv`, and the output folder must exist:

  ```powershell
  New-Item -ItemType Directory -Force C:/Users/artem/AppData/Local/Temp/haltere-replay | Out-Null
  .venv/Scripts/python.exe haltere/obstacles/vertical_replay.py LOGNAME --out C:/Users/artem/AppData/Local/Temp/haltere-replay/r5 --stack on --near-on-path --throttle-column command_thr --descent-view configs/pilot/descent_view.json --motor-assist configs/pilot/motor_assist.json --stale-evidence configs/obstacles/stale_evidence.json
  ```

- **Graduation evidence:** none of these runs counts, because no brain is selected and rules of the stack fail gates.

### Blockers for graduation, and what would clear them

| Blocker | State after round 5 | What would clear it |
|---|---|---|
| A selected brain | None. brain-12's best candidate ties brain-11 at 6 of 15; every candidate keeps the left/right fault (weak roll toward lateral requests without a yaw rate), and none has fewer surrogate terrain contacts | A brain whose training reaches the fault (more than the readout rows, or yaw-coupled requests), scored once on fresh frozen gates; then flights |
| Minus hairpin (brains) | Motor assist v3: held-out harness 7-10 of 12 with the assist, 0 without; replay warning 1.20 s; not flown. Crawls behind the governor's stand-off at arches remain | Live (b); a wall-pilot version whose stand-off acts only under wall-ahead conditions, frozen and scored with the pass-through scenarios |
| Minus garage wall behind an arch (fast PD) | Stale evidence v2 lowers the cap on the travel ray from 1.04 s before the r4b impact; the request is 4.00-4.25 m/s from 43.90 s where `m4b` asks 4.74-5.99 m/s (open loop, development); not flown | Live (a) |
| Straw start-arch false marker | Not addressed: the ring-marker rule kept 95.1% of held-out overlay-confirmed markers (bound 99.5% per flight) | Log the reader's candidates from the live capture; a rule scored on live frames |
| Straw downhill ground contact | Contact support reacts after a touch (v3 fires 0.73 s after the held-out touch began, 0.04 s after it ended, with or without the assist); nothing prevents touches; brain-11 had 1 touch in its lap; brain-12 has no fewer surrogate contacts | A pilot-side lever (less sink requested near terrain: the brains follow the sink request within 0.004 m/s); a contact audit v2 validated on video, frozen before scoring; then Straw 3/3 with no audited contact |
| Contact support v3 | Fails 3 of 13 safety gates; blind in 6.7% of 90.4 logged minutes (safety branch) and in 8.8% of the 27 replayed logs inside the m5 stack (210 of 2,387 s); brain-08 and fast-brain-11 never armed in the scripted rest | A v4 that orders body rate before the residual (flare) against both together (knock), frozen and scored |
| Ceiling cut x assist sag (new, merge interaction) | On one Straw uphill window of 27 logs (`straw-brain08-06`, 37.49-38.77 s), the pilot's own ring climb lost its exemption: 0 m/s for 0.89 s where wall pilot v5 asks 1.47-1.76 m/s (open loop) | A wall-pilot version that cuts only the assist's share of a climb, frozen and scored with the assist on |
| Straw lap time under the assist | Quiet gate fails on brain-08 laps (3.30-4.18% removed); 0.52% on brain-11's lap | Live (c) lap times against 1:42.988 |
| Vertical guard on Pine, the view rule, the contact audit | Unchanged from round 4b | As round 4b |

### Round 5 live plan: amendments after the combined review (2026-09-29)

The combined review (fly-with-fixes, no blocker) found that wall pilot v6's exemption of the pilot's own
in-view ring climb from the ceiling cut is lost whenever motor assist v3 adds a sag climb, and the cut then
zeroes the whole climb request: open loop on straw-brain08-06 the climb was held at 0 m/s for 0.89 s on the
uphill (0.238 s per scored minute against the safety gates' 0.2 s/min quiet limit), and on fast-brain-11's
r4b Straw lap the exemption is off for 10.2 of 28.9 s of its own ring climbs. Therefore:

- **(c) Straw Bale flies without `--motor-assist`** (Straw has no hairpin; the assist also fails its Straw
  quiet gate; the r4b lap flew without it). Log stem `straw-brain11cw13-r5-noassist-01`.
- **(a) flies `--contact-support shadow`**, following the safety section's recommended order (measure live
  arming, excluded time and would-be fires with the fast PD first); (b) and (c) fly `on`.
- A fix that bounds only the assist's share of a climb with the ceiling cap needs a new frozen
  wall-pilot/motor-assist version before any Straw run with the assist.


## Round 5 (live), 2026-09-29: m5 stack

Branch `m5` at `3b2fe2f`: `--looming-brake --obstacle-stack on --stale-evidence on --descent-view on`
(wall pilot v6, descent view v3 with contact support v3, vertical guard v4, gap pilot v5, lag turn v2,
stale evidence v2), per the amended plan: Straw without the motor assist, the Minus PD with
`--contact-support shadow`, the Minus brain with `--contact-support on --motor-assist on` (assist v3).
Liftoff was relaunched in a restarted seat (session 4); the viewer was hidden; pad 31 `seatOnly`;
ground checks 34 (Straw) and 35 (Minus). Every run is a disclosed development deviation (no brain is
selected; every round-4b/5 rule fails at least one frozen gate). Stick change = mean |d roll|, |d pitch|
command per 10 ms tick after 3 s.

| Run | Motor | Outcome |
|---|---|---|
| `straw-brain11cw13-r5-noassist-01` | fast-brain-11-b-cw13 | Lap 1, 19.4 s: at the tall "FAT SHARK" arch the marker switched to the next ring (far left) while the drone was still inside the arch; the pilot turned hard left (request (4.3, 4.0) -> (1.0, 4.5) m/s in 0.2 s); the drone yawed left faster than its path turned and its momentum carried its right side into the arch's right leg at 5.2 m/s, (80.1, 15.6) (corrected from 'left leg' after the release check of the video). Its line was ~0.4 m off the r4b lap, which passed. Stick change 0.0025 |
| `straw-brain11cw13-r5-noassist-02`, `-03` | | Not flown: preflight refused (`rustc.exe` builds in the user's session, 8.0 and 1.5+0.6 cores) |
| `straw-brain11cw13-r5-noassist-04` | fast-brain-11-b-cw13 | Passed the arch and the hill; on the downhill the ring stayed bottom-clipped and the view rule's steep-late bound allowed -2.2..-2.5 m/s of sink at 5.2-5.7 m/s (~24 deg); audited ground contact at 79.43-79.97 s (-36.5, 133.8), answered by a support climb (both this run and the r4b lap recovered from it); 1.9 s later, flying 0.5-0.7 m higher than the r4b lap at the next (ImmersionRC) arch, it descended at ~12 degrees onto the arch's top bar at 81.86 s, (-36.4, 120.7), 5.5 m/s; the r4b lap passed under the bar (corrected from 'into the hillside' after the release check of the video). Stick change 0.0027 |
| `minus-fast6-r5-01` | fast PD, contact support shadow | Pillar A and the hairpin, but this time the wall brake stopped it at the hairpin (stand-off, 1.0 m/s at 24.7 s); after the turn it accelerated to 5.3 m/s on a new line and hit a dark pillar at (78.3, 23.2), 26.3 s. The r4b PD took the hairpin without stopping and passed here. The stale-evidence rule did not re-seat |
| `minus-brain11cw13-r5-01` | | Not flown: preflight refused (`rustc.exe`, 0.67 core) |
| `minus-brain11cw13-r5-02` | fast-brain-11-b-cw13, motor assist v3 | **Assist-caused crash at the first arch.** At 6.59 s the assist's approach source cut the horizontal request from 6.0 to 1.27 m/s (4.73 m/s removed) and then to -0.21 m/s at 6.99 s in front of the first arch, which every earlier brain run passed at ~5.5 m/s; the brain pitched up and climbed (vz +0.93), crawled at 1-2 m/s and bumped the arch at (19.0, -0.5), 8.1 s, ~1 m/s. Motor assist v3 is not flyable (its declaration says the approach bound never plans below 2.5 m/s; the live request went to 1.27 and below 0) |

What this shows:

- Straw repeatability is the blocker: the same brain and stack flew a full lap (r4b), crashed at the arch after
  a checkpoint switch inside the arch (r5-01), and, after the same downhill contact the full lap survived, descended onto the next arch's
  top bar (r5-04). Generic faults: turning toward the next ring while still inside a gate structure; descending
  onto a gate's top bar with the ring clipped at the bottom of the image; and ~24-deg descents that touch terrain the
  camera cannot measure (no looming evidence on the straw; both runs recovered from that touch).
- Motor assist v3's approach source must not act at pass-through arches (a live counterpart of the review's
  harness crawl finding); the brain hairpin remains unsolved live.
- The PD's hairpin now ends in a stand-off stop, after which its new line met a pillar: stop-and-turn exits need
  the same obstacle care as through-flight.

## Round 6 (offline): gate clearance around checkpoint switches (branch `m6-gates`, not flown)

Details, figures and tables: [gate_clearance.md](../gate_clearance.md). The three Straw Bale crashes of
fast-brain-11-b-cw13 are the development cases. Their causes were re-checked on the recorded video and in the
semi-closed-loop surrogate (development evidence):

- **FAT SHARK (`r5-noassist-01`):** not the post-switch turn. The gap aim shifted the aim up to 8.0 deg right (away
  from the arch's near left leg) from 17.4 to 18.7 s. Both round-5 passes crossed 0.5-1.1 m right of each of the 23
  other logged passes, and after the switch the lagging path could not leave that line within the 0.37 s before the
  right leg. Surrogate distance from the contact point: logged 0.023 m; without the lag turn 0.011 m; post-switch
  heading/yaw holds 0.016-0.021 m; **without the gap aim 0.638 m**.
- **Start arch (`r4b-noassist-02`):** the false marker (a banner logo read for two captures after 0.55 s unread); the
  lag turn's lead on it accounts for 0.27 m. The marker-jump rule (below) clears the contact point by 1.03 m in the
  surrogate (0.02 m without).
- **Top bar (`r5-noassist-04`):** contact support v3 started its climb at 80.08 s, after the downhill touch had ended
  (79.97 s; its manoeuvre exclusion blinded it during the touch). The climb left the drone 0.5-0.7 m above the lap
  run's line 1.8 s before the next arch. Surrogate height at the bar: logged 13.48 m; horizontal request at 60% from
  the end of the climb 13.16 m; **without the late climb 12.65 m** (the lap run: about 12.8 m).

Turn timing after the 57 switches of the fast-brain-08 finishes (no stack) and the 13 of the fast-brain-11 runs: the
request turns about 80 deg/s from the first tick in both. The lag turn adds 4.4 deg (p50) by 0.3 s, and the path has
turned only 2-4 deg by then.

Rules:

| Rule | State | Result |
|---|---|---|
| Post-switch clearance (hold the request within a small angle until clear of the gate) | Tested on its development case before a freeze, **not frozen** | No effect on FAT SHARK (the line was set before the switch) |
| Marker-jump confirmation (`configs/pilot/marker_jump.json` v1 `8752cd7e`, `--marker-jump on\|off\|shadow`, off by default) | **Frozen with its gates** (`marker_jump_gates.json` v1 `4eb99684`, `f4b2ebd`); scorer fix before scoring `1249e77` | Development case passes (the false readings held; surrogate 1.034 m from the contact point). **Fails held-out**: H2 (34 of 40 changed log windows more than 0.3 m from the base's surrogate path, worst 5.41 m: real rings reacquired after a gap were held while the pilot coasted and searched), H1 (6 new stops, three of them Minus Two fast-PD re-accelerations after a stand-off) and HG2 (fast-brain-08 28 -> 27 finishes); identity 126/126 and 126/126, H3, HG1 (fast-PD post contacts 7 -> 0), HG3 and HG4 pass (`58c3d48`). A development variant that keeps the bearing during a hold removed the new stops but still moved 33 of 40 windows more than 0.3 m: not frozen. **Do not fly `on`**; `shadow` is identical to off |
| Gate below (keep the path under a gate's top bar when the ring is bottom-clipped) | Tested on its development case before a freeze, **not frozen** | At most 0.32 m lower at the bar; no causal cue measured the arch's distance; the late support climb was the cause |

What this means for the next flights:

- No rule of this round should fly with authority.
- The largest Straw Bale lever found is the gap aim at oblique arches: a gap-pilot revision with its own frozen gates
  (Minus Two pillars A and C, every Straw arch pass). A component-off diagnostic (`--gap-cue off` inside the stack) on
  Straw Bale would show it live.
- The second lever is contact support that climbs only while the contact lasts (the ground work of round 6).
- `--marker-jump shadow` is safe to add to any run (identical requests); it logs where the rule would have held.

## Round 6 (offline): early brake and motor assist v4 (branch `m6-brake`, not flown)

Details, gates and risks: [early_brake.md](../early_brake.md). Nothing here has flown. Every live log named here is a
development case, and the held-out evidence is the fresh harness sets and the 20 untouched logs.

- **Early brake** (`configs/obstacles/early_brake.json` v1, `ecf76971...`; `--early-brake on`, off by default, inside
  the obstacle stack).
  - The looming governor engages when a wall sample lies within the brain contract's stopping distance (0.3 s,
    3.5 m/s^2, 0.5 m), not only at TTC < 0.8 s.
  - The episode is floored at 2.5 m/s while the pilot sees its ring ahead.
  - Samples the lower window explains and governor climbs do not vote.
  - The fast PD has no entry.
- **Motor assist v4** (`configs/pilot/motor_assist.json`, `8954a798...`; v3 kept, refused by the runner).
  - No approach source.
  - Cap tracking floored at 2.5 m/s unless a wall is confirmed ahead under a wall-ahead condition.
  - The ceiling cut bounds only the assist's share of a climb.
- **Gates** `configs/obstacles/early_brake_gates.json` v1 (`3b029e32...`): frozen `33a27dd`, one scorer fix before scoring
  `b4d0e3b`. **16 of 21 pass.**
  - Identity: 31/31 bit-identical to `m5` with the rules off; the fast PD is unchanged.
  - Development: the live hairpin cap binds 1.18 s before the wall (round 5: 0.76 s); with v4 the brain reaches the
    wall at 1.00 m/s in the surrogate window (3.88 without).
  - Held-out hairpins (16, live-like samples): fast-brain-10b 14 clean, fast-brain-11-b-cw13 10 clean / 2 walls (v3: 14
    and 11).
  - Held-out pass-through arches: no crawl like v3's 0.09-1.03 m/s. Lowest 0.66-2.38 m/s (fast-brain-11, one below the
    0.75 bound) and 1.03-2.28 (fast-brain-10b).
  - Fails:
    - fast-brain-11's hairpin ceiling contacts (2; base 0) and its one slow arch;
    - the quiet bounds on the held-out brain-08 Straw/Pine laps (early brake 1.9-4.0%, with v4 4.2-7.7%; their offline
      looming stream lacks vertical evidence at the uphill rings);
    - a 0.38 s ceiling-share window that v3 shares (not the cut).
- **The fast PD's hairpin stop was not new** (`minus-fast6-r4b-01` also came to rest there, 0.12 m/s at 20.04 s; video:
  1 km/h on the HUD), and the round-5 rules changed none of `minus-fast6-r5-01`'s requests (m4b = m5 on all 2608
  ticks).
  - Its pillar came from the exit line: the governor's stand-off cap along the hairpin wall's ray (24.4 deg) removed the
    request's component toward the wall after turn-first had released aligned. The request pointed 102-106 deg with the
    ring at the centre of the image (85-90 deg), 13-18 deg left, into the dark pillar (video 25.5-26.1 s).
  - Diagnostic: ending the cap at the aligned release crosses the pillar's row 2.0 m to its right in the surrogate.
  - A wall-pilot fix, not the gap aim or turn-first, is the next step; not done here.

### Live flight plan (for the main session; development flights)

Fly from `C:\DEV\Haltere` with `m6-brake` checked out (local branch, not pushed; see the worktree note in the round-5
plan). The flight procedure, preflight, ground check (next number: 36), resuming after every stop, the `LIVE` telemetry
check and the stop criteria are the round-5 plan's. **Every run is a disclosed development deviation:** no brain is
selected, the new rules fail 5 of 21 frozen gates, and the rules of round 4b/5 keep their failures.

**(r6-a) Minus Two, fast-brain-11-b-cw13, early brake + assist v4.** It is the direct comparison with
`minus-brain11cw13-r4b-noassist-01` (hairpin wall) and `minus-brain11cw13-r5-02` (the v3 crawl at the first arch).

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-11-b-cw13/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller brain --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --stale-evidence on --early-brake on --descent-view on --contact-support on --motor-assist on --seconds 150 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/minus-brain11cw13-r6-01.csv --record runs/fast-stack-20260923/minus-brain11cw13-r6-01.mp4 --video-encoder h264_nvenc
```

The sidecar must declare:

- `early_brake_declaration`: version 1, `ecf76971...`, `applied: true`;
- `motor_assist_declaration`: version 4, `8954a798...`, `applied: true`;
- the round-5 declarations as before (wall pilot v6, stale evidence v2, descent view v3 with contact support `on`).

The CSV ends with `early_brake` after the stale-evidence columns.

What to look for:

- **First arch (x 15-19):** `early_brake` turns 1 about 0.1-0.25 s before the governor's own brake would (open-loop replays). With the ring in view
  the flown request should stay at or above 2.5 m/s (`cmd_v*`; the assist's `assist_wall_ahead` is 0). The r5-02 crawl
  (6 -> 0.2 m/s) must not recur.
- **Pillar A and the 90-degree arch:** slower than round 4b (about 2.5-4 m/s), no stop.
- **Hairpin (wall near (81.9, 20.0)):** the cap should bind about 1.2 s before the wall (r4b: 0.77 s). After the arch
  pass (marker to the side) the stopping source and turn-first take over. Speed at the wall well below r4b's 3.7 m/s;
  the surrogate predicts about 1 m/s from the logged state.
- **Ceiling:** braking climbs under the 2.2 m ceiling (the held-out harness had 2 ceiling contacts of 16).
- **Stop criteria as round 5.** In addition, stop if the drone is parked for more than about 3 s at an arch with the
  ring in view (that is the failure this round fixes).

**(r6-b) Optional: Straw Bale, fast-brain-11-b-cw13, three laps, early brake without the assist.** The same command
with `--seconds 480`, without `--motor-assist on`, log `straw-brain11cw13-r6-01`. The early brake slows the brain at
the FAT SHARK arch (where r5-01 crashed at 5.2 m/s after a checkpoint switch inside the arch), at (43.8, 63.5) and at
(-30.5, 28.4). Compare lap times with 1:42.988.

- The quiet gate failed on the brain-08 laps.
- The false marker at the start arches, the top bar after the downhill and the downhill contact are not addressed.

**The fast PD:** unchanged by these rules; the pillar exit after the hairpin stand-off is open.

### Blockers after round 6 (brake)

| Blocker | State | What would clear it |
|---|---|---|
| Brain hairpin | Early brake + v4: 10-14 of 16 held-out hairpins clean for fast-brain-11/10b, the live window at 1.00 m/s; not flown | Live (r6-a); a wall-pilot stand-off that holds a stopped brain without backing it off |
| Crawls at pass-through arches with the assist | v4 plans none with the ring in view (0 s on 20 logs); held-out lowest 0.66-2.38 m/s against v3's 0.09-1.03 | Live (r6-a) at the first arch |
| Ceiling cut x assist sag | Fixed in v4 (development window: v3 1.27 s below the unassisted climb, v4 0.00); the frozen metric was blind | A correct metric on fresh logs |
| Early brake on Straw/Pine | 0.8% of request travel on live brain-11 laps (development); 1.9-4.0% on the held-out brain-08 laps (failed) | Straw live lap times; a version that also leaves out samples without vertical evidence while the pilot climbs toward its ring, frozen and scored |
| Wall behind a ring in view | The early floor holds until the marker switches | A version lifting the floor at the urgent TTC, frozen and scored |
| Fast PD's exit after a stand-off | Deflected by the old wall's cap (diagnosed, not fixed) | A wall-pilot version, frozen and scored with PD sets and logs |

## Round 6 (offline): ground clearance on descents, the sighted descent (branch `m6-ground`, not flown)

**Nothing here has flown.** Details, figures and every number: [sighted_descent.md](../sighted_descent.md). Branch
`m6-ground` from `m5` (`bc7c71c`).

**The Straw downhill touch.** Both fast-brain-11 runs that reached the downhill (`straw-brain11cw13-r4b-noassist-02`,
`-r5-noassist-04`; development logs of this rule) touched the straw at (-36.5, 133-134) after a 24-27 degree descent.
Checked on the recorded frames and telemetry:
- the next ring (ImmersionRC arch) lay 13.6-15.1 degrees below the drone from the hilltop on (offline triangulation of
  the in-view marker rays, residuals 0.08-0.13 m), and the marker was read in view at 13.4-14.6 degrees before it
  stayed clipped for 4.2-4.8 s;
- the view rule's steep late then asked for 20-24 degrees (flown 24-27), about 10 degrees below the ring;
- the camera never sees that hillside: a 9-15 degree slope under a lower image edge at 12-17 degrees is met, if at all,
  43-57 m ahead at grazing incidence (flow about 0.005 rad/s). The looming sampler read no evidence on the whole
  downhill, and the video shows no straw in front of the drone until the knock. No depth, flow or ring-size cue can
  measure the clearance there, so none was built.

**The rule** (`configs/pilot/sighted_descent.json` v1 `49b8d7a79f32...`; `--sighted-descent on|off|shadow`, off by
default, needs `--descent-view`): two agreeing in-view readings of the ring near the bottom edge set its line of sight;
it turns down while the flight path stays above it (as for a ring 20 m away) and is raised by later bottom clips; while
the ring is clipped at the bottom edge the pilot never requests a path more than 1 degree below it (never below the
in-view bound, never more sink than the view rule). Causal and course-agnostic: the ring cue, attitude, velocity and
camera calibration only. It keeps the view rule's speed parts (6 m/s kept on the downhill in the replays).

**Frozen gates** (`configs/pilot/sighted_descent_gates.json` v1 `46deedeeea36...`, frozen in `98c4609` before any gate
run; scored on `afd6277`): **10 of 12 pass.**

| Gate | Result | Pass |
|---|---|---|
| Identity: `m5` archive = this tree (default pilot, round-5 Straw stack), off = shadow, 31 logs | 93 of 93 bit-identical | yes |
| Replays: the first changed request of each log is a withheld tick (29 held-out logs + 2 development) | 11 logs changed, all causal | yes |
| Replays: horizontal request not lower by > 0.05 m/s where it withholds | fast-brain-08 laps `-04` 0.111 and `-06` 0.173 m/s (<= 0.25 s, via the descent-path governor's floor) | **no** |
| Development: steepest request before the two touches | 15.29 / 16.11 deg against 22.10 / 23.34 (ring 14.4 / 15.1) | yes |
| Straw variations 1001-1032 (held out), fast PD / fast-brain-11 | contacts 16 -> 11 / 15 -> 8, contact s 58.51 -> 42.97 / 36.86 -> 20.50, high passes 8 -> 5 / 4 -> 4, time -0.36 / -0.31% | yes / yes |
| Logged Straw geometry, fresh randomisation (held out) | 16 -> 12 / 6 -> 5 (63.67 -> 55.21 / 13.99 -> 11.04 s), high passes 7 -> 2 / 5 -> 4, time -2.36 / -0.06% | yes / yes |
| Synthetic hills 8500-8515 + steep 8600-8607 (held out) | contacts 17 -> 16 / 16 -> 16, **fast PD high passes 8 -> 9** / brain 9 -> 8, time +0.46 / +0.29% | **no** / yes |
| Flat 8700-8707 (held out) | unchanged counts, time 0.00 / +0.002% | yes / yes |

Post-scoring diagnosis: of the fast PD's three changed high passes on the synthetic hills, one course (8504) changed
although the rule never acted there: the surrogate draws all drones' HUD dropouts from one random stream, so a drone
that finishes at another time changes the others. The keep-speed failure and a shifted support-climb timing (up to
0.33 m/s more sink for up to 1 s after a round-5 support climb the rule's variant lacks) occur only in counterfactual
replays of fast-brain-08 laps flown with the default pilot.

**What it does not fix** (seen in the Straw rebuild and on the synthetic hills): the brain flies 2-4 degrees steeper
than it is asked while the ring is in view; the line from the hilltop checkpoint to the ring grazes the convex Straw
crest (0.4-0.6 m); a dive right after the hilltop turn (the fast PD); the chord through the shoulder of an S-shaped hill.
The residual Straw-rebuild touches are these.

**Live plan (development flight, for the main session):** the round-5 amended run (c) with `--sighted-descent on`
added (command and what to watch in [sighted_descent.md](../sighted_descent.md#live-plan-for-the-main-session-development-flights)).
It is a disclosed deviation: the rule fails 2 of 12 gates and no brain is selected. Watch the downhill (request held
near 15-16 degrees from about 77.5 s, 6 m/s kept, no touch at y 133) and the ImmersionRC arch's top bar.

**Blockers, updated:** the Straw downhill row of round 5 now has a pilot-side lever that holds the requested descent to
the ring's line of sight (offline evidence only); a live Straw lap with it, and a fix for the brain's sink overshoot
in view, remain. The contact audit's video validation is unchanged.

## Round 6 (offline): the merged m6 stack, replays, harness and surrogate, and the live plan

**Nothing in this section has flown.** Branch `m6` is `m5` (`bc7c71c`) with the three round-6 branches merged in this
order: `m6-gates` (`01f10c9`: marker-jump rule v1 and its gates, the gate-clearance analysis, replay-harness switches),
`m6-brake` (`13bacfe`: early brake v1, motor assist v4, their gates) and `m6-ground` (`1102c45`: sighted descent v1 and
its gates, the Straw downhill rebuild). Every live log replayed below was read by at least one round-6 branch while it
designed its rule, so **every replay here is development evidence**. The closed-loop sets are fresh (course seeds,
scenario values and sim seed 113 chosen before any run and used by nothing before); they are reports, not gates.

**Merges.** `m6-gates` merged cleanly. `m6-brake` and then `m6-ground` conflicted with the branches before them in
`fast_race_cue.py`, `visual_brain.py`, the replay harness, `deployed_pilot.py`, the motor-assist CSV-tail test and this
card. Every conflict was independent additions, and all sides were kept:

- `FastRaceCue` takes `marker_jump`/`marker_jump_apply`, `early_brake`/`early_apply` and
  `sighted_descent`/`sighted_apply`; its metadata gains each rule's record when it is declared.
- The runner resolves `--marker-jump`, `--early-brake` (inside the obstacle stack) and `--sighted-descent` and passes
  them to the controller. The CSV ends with the view and contact columns, the assist columns (8), the stale-evidence
  columns (2), then `marker_held`, `marker_candidates` (with `--marker-jump`), `early_brake` (with `--early-brake on`)
  and `sighted_los`, `sighted_bound`, `sighted_withheld` (with `--sighted-descent`). The column comments and the three
  CSV-tail tests (`test_fast_race_cue_motor_assist.py`, `test_round6_brake.py`, `test_sighted_descent.py`) now state
  that order.
- The replay harness takes `--marker-jump [--marker-jump-mode shadow]`, `--lag-turn off`, `--gap-aim off`,
  `--early-brake` and `--sighted-descent [--sighted-mode shadow]` together.
- `deployed_pilot_kwargs` takes `marker_jump=None`, `early_brake=False` and `sighted_descent='off'` beside
  `motor_assist` (`True`: version 4; `3`: the kept round-5 declaration).
- This card keeps the three round-6 branch sections above in the order gates, brake, ground.
- `docs/fast_racing.md` gained the early-brake and sighted-descent rows; its motor-assist and descent-view rows now name
  the current versions.

**Can the brain release graduate now? No.** The bar is one frozen stack that finishes Straw Bale 3/3 with no ground
contact and Minus Two 3/3. After round 6:

- **No brain is selected.** `fast-brain-11-b-cw13` is flown because it is the only braking brain with live Straw and
  Minus evidence, not because a gate selected it.
- **Every round-6 rule fails at least one of its own frozen gates**: the marker-jump rule fails its held-out gates
  (it is flown in shadow only), the early brake and motor assist v4 pass 16 of 21, the sighted descent 10 of 12. The
  round-4b/5 rules keep their failures (contact support v3 3 of 13 safety gates, guard v4, the view rule, the wall
  pilot's brake floor, the contact audit's video check).
- **Two of the three video-verified Straw crashes are not addressed:** the gap aim's line at the oblique FAT SHARK arch
  (`r5-noassist-01`) and the false marker at the lap-2 start arch (`r4b-noassist-02`). The third (`r5-noassist-04`,
  top bar after a late support climb) is addressed only upstream: the sighted descent is meant to prevent the downhill
  touch, and the plan below flies contact support in shadow so that a late climb cannot happen.
- **Nothing of round 6 has flown.**

The flights below are development flights. They can show whether the round-6 fixes work live; they cannot graduate the
release.

### Frozen gates of the three round-6 branches (as scored by each branch)

| Branch | Declaration (content sha256) | Gates; freeze / scores | Result |
|---|---|---|---|
| `m6-gates` | `configs/pilot/marker_jump.json` v1 `8752cd7e` | `marker_jump_gates.json` v1 `4eb99684`; `f4b2ebd` / `58c3d48` (scorer fix `1249e77` before scoring) | **Fails its held-out gates** H1 (6 new stops), H2 (34 of 40 surrogate windows moved more than 0.3 m) and HG2 (fast-brain-08 28 -> 27 finishes). Passes identity (126/126 off = m5, 126/126 shadow = off), its development case (surrogate 1.034 m from the start-arch contact point), H3, HG1, HG3, HG4. Not flown with authority; `shadow` is identical to off |
| `m6-brake` | `configs/obstacles/early_brake.json` v1 `ecf76971`; `configs/pilot/motor_assist.json` v4 `8954a798` (v3 kept as `motor_assist_v3.json` `7c3b49e7`, refused by the runner) | `early_brake_gates.json` v1 `3b029e32`; `33a27dd` / `13bacfe` (scorer fix `b4d0e3b` before scoring) | **16 of 21.** Fails H1 fast-brain-11 hairpins (10 clean, 2 walls, 2 ceiling contacts against the base's 0), H2 fast-brain-11 pass-through (lowest 0.66 m/s with v4, bound 0.75), H4b quiet early brake (Straw brain-08 laps 1.88-2.59%, `pine-brain08-01` 3.95%; bound 1.5%), H4d quiet with v4 (4.21-5.24%, 7.68%; bound 3%) and H5 ceiling share (0.38 s on `straw-brain08-01`, v3 identical) |
| `m6-ground` | `configs/pilot/sighted_descent.json` v1 `49b8d7a7` | `sighted_descent_gates.json` v1 `46deedee`; `98c4609` / `1102c45` (scored on `afd6277`; a report-only field's sign fixed after the first scoring, no gate changed) | **10 of 12.** Fails SD_Replay_KeepSpeed (horizontal request 0.111 / 0.173 m/s lower for at most 0.25 s on the counterfactual fast-brain-08 laps) and SD_Surrogate_Hills fast PD (high passes 8 -> 9). Held-out Straw variations: fast-brain-11 contacts 15 -> 8 (36.86 -> 20.50 s), fast PD 16 -> 11 |

Every frozen declaration and gate file under `configs/` (61 files) kept its content and hash: its canonical hash equals
its own `sha256`, and its bytes equal the round-6 branch that added or changed it (8 files, each changed by one branch
only) or `m5` (53 files). The runner's own loaders on `m6` load:

| Declaration | Version | Content sha256 |
|---|---|---|
| `configs/obstacles/lag_turn.json` | 2 | `d4eb83da51ab...` |
| `configs/obstacles/gap_pilot.json` (gap cue v2) | 5 | `43c304204f93...` |
| `configs/obstacles/wall_pilot.json` | 6 | `fe65951d3d68...` |
| `configs/obstacles/vertical_guard.json` | 4 | `409d06f9ded7...` |
| `configs/obstacles/stale_evidence.json` | 2 | `4a9516068f1f...` |
| `configs/obstacles/early_brake.json` (`--early-brake on`) | 1 | `ecf76971107f...` |
| `configs/pilot/descent_view.json` (contact support v3) | 3 | `2bdb17fc2479...` |
| `configs/pilot/sighted_descent.json` (`--sighted-descent on\|shadow`) | 1 | `49b8d7a79f32...` |
| `configs/pilot/motor_assist.json` (brain contract only) | 4 | `8954a798e127...` |
| `configs/pilot/marker_jump.json` (`--marker-jump on\|shadow`; **failed its held-out gates**) | 1 | `8752cd7e2860...` |
| `configs/pilot/ring_marker.json` (off by default; failed its gates in round 5) | 1 | `09824e473659...` |

They refuse the kept wall pilot v1-v5, vertical guard v1-v3, descent view v1-v2, motor assist v1-v3, stale evidence v1
and gap pilot v1-v4 (checked with each loader).

### Identity: the round-6 rules off (or in shadow) are m5

Open-loop replays of all 31 logs with ticks (the 27 of round 5 and the four round-5 live flights) through a `git
archive` of `m5` (`bc7c71c`) and of `m6` (`eb22ecd`, the merged code), each tree with its own harness, run from that
tree. **226 of 226 pairs are bit-identical**, in the command arrays and in every other array both replays carry:

- the default pilot (`--stack none`): 31 of 31;
- the stack in shadow: 31 of 31;
- the round-5 stack as flown on Straw Bale in round 5 (`--stack on --near-on-path --throttle-column command_thr
  --descent-view --stale-evidence`, no assist): 31 of 31;
- the same with motor assist v3 as flown on Minus Two in round 5 (on `m6` the kept `motor_assist_v3.json`): 31 of 31;
- `m6`'s round-5 stack with the marker-jump rule and the sighted descent both in shadow, against `m6`'s round-5 stack and
  against `m5`'s: 31 of 31 each;
- `m6`'s shadow stack with the early brake against `m5`'s shadow stack: 31 of 31;
- the three live fast-PD logs of rounds 4-5 through the full `m6` stack against the same stack without the sighted
  descent, without the assist and without the early brake: 3 of 3 each (the PD has no early-brake or assist entry, and
  the sighted descent changes no request on these Minus Two logs).

On the merged tree the brake branch's full-stack digest test (round-5 stack with assist v3 = `m5`) and the ground
branch's golden digests (default pilot, view rule, round-5 view rule with contact support v3; rule absent and in shadow)
still pass. The full suite passed on the merged code (`eb22ecd`): **1266 passed in 339.52 s**, and again on the
integration commit, which adds only documents (1266 passed in 291.89 s). Scripts and raw outputs
are in the session scratchpad under `m6/integrate/`. Results: `docs/experiments/round6_integration.json`.

### Open-loop replays: what the m6 stack changes, per log (development evidence)

The live logs of rounds 4, 4b and 5 that have ticks (the two r4-01 logs have none) and the two fast-brain-08 Straw
finishes, each replayed at its recorded states. The recorded motion does not respond to the requests, so every figure
is what a stack would have **requested** there; cap tracking keeps pushing a drone that cannot slow, so the removal
figures are upper bounds. The Straw finishes of fast-brain-08 flew without the obstacle stack and use the offline
looming stream, which carries no vertical evidence at the uphill rings.

- `m5`: the round-5 stack as flown on Straw Bale in round 5 (`--stack on --near-on-path --throttle-column command_thr
  --descent-view --stale-evidence`, contact support v3 on, no assist), on the `m5` archive; `m5ma3` adds motor assist v3
  (as flown on Minus Two in `minus-brain11cw13-r5-02`).
- `m6`: the `m5` flags plus `--early-brake`, `--sighted-descent`, `--motor-assist` (v4) and `--marker-jump` in shadow, on
  the `m6` archive. Its shares: the same without the assist, without the early brake, without the sighted descent.
- The stack of the live plan below is `m6` with `--contact-support shadow`. It equals `m6` on every log except where
  contact support v3 fires: 1.91 s of `straw-brain11cw13-r4b-noassist-02` and 12.6-13.4 s of the two fast-brain-08 laps.

| Log | Motor | Request changed vs `m5`, s (assist v4 / early brake / sighted descent) | Early brake, s; lowest request in an episode, m/s | Sighted descent withheld sink, s | Request travel removed vs `m5`, % (without the assist) | Contact-support v3 onsets: `m5` -> `m6` | Marker jump in shadow: would-hold ticks |
|---|---|---|---|---|---|---|---|
| `minus-fast6-r4-02` | fast PD | 0.0 (0.0 / 0.0 / 0.0) | - | 0.0 | 0.0 (0.0) | 0 -> 0 | 0 |
| `minus-brain10b-r4-02` | brain | 5.42 (5.34 / 1.46 / 0.0) | 6.97; 2.5 | 0.15 | 7.55 (1.96) | 0 -> 0 | 0 |
| `minus-brain09b-r4-01` | brain | 11.13 (8.33 / 6.44 / 0.0) | 10.25; 0.27 | 0.0 | 7.51 (3.63) | 0 -> 0 | 0 |
| `minus-fast6-r4b-01` | fast PD | 0.0 (0.0 / 0.0 / 0.0) | - | 0.0 | 0.0 (0.0) | 0 -> 0 | 10 |
| `minus-brain11cw13-r4b-noassist-01` | brain | 9.88 (8.27 / 7.99 / 0.0) | 8.5; 0.55 | 0.04 | 8.61 (3.14) | 0 -> 0 | 0 |
| `straw-brain11cw13-r4b-noassist-02` | brain | 16.19 (13.86 / 4.81 / 2.38) | 3.83; 2.5 | 2.76 | 1.31 (0.79) | 1 -> 1 (80.08 s) | 63 |
| `straw-brain11cw13-r5-noassist-01` | brain | 0.06 (0.06 / 0.0 / 0.0) | 0.0; - | 0.0 | 0.0 (0.0) | 0 -> 0 | 0 |
| `straw-brain11cw13-r5-noassist-04` | brain | 13.74 (12.1 / 3.36 / 4.11) | 2.44; 2.5 | 2.43 | 1.32 (0.76) | 1 (80.08 s) -> 0 | 20 |
| `minus-fast6-r5-01` | fast PD | 0.0 (0.0 / 0.0 / 0.0) | - | 0.0 | 0.0 (0.0) | 0 -> 0 | 12 |
| `minus-brain11cw13-r5-02` | brain | 1.64 (1.48 / 1.58 / 0.0) | 1.58; 1.75 | 0.0 | 4.79 (2.93) | 0 -> 0 | 0 |
| `straw-brain08-04` | brain | 105.58 (90.97 / 44.65 / 9.08) | 30.47; 2.5 | 6.74 | 4.57 (2.59) | 10 -> 10 | 66 |
| `straw-brain08-06` | brain | 97.13 (87.8 / 26.87 / 10.07) | 23.43; 0.69 | 5.14 | 2.99 (1.41) | 6 -> 6 | 123 |

The shares overlap (a tick can change for several rules). Against `m5ma3`, the Minus brain logs lose 8.28-13.66% of
their request travel to assist v3 and 4.79-8.61% to the `m6` stack.

**Straw Bale, fast-brain-11-b-cw13 (the three development crashes):**

- **Downhill touch (both laps that reached it).** The sighted descent withholds sink from 77.42 s (`r4b-noassist-02`)
  and 77.61 s (`r5-noassist-04`) to the touch, up to 1.08 and 0.94 m/s. The steepest requested path in 72 s to the
  touch is 16.06 and 15.2 deg (`m5`: 23.52 and 22.15; the ring lay 13.6-15.1 deg below the drone). The horizontal
  request stays 5.63-6.0 m/s, as in `m5`. The assist adds up to 0.23 m/s of climb there (its sag compensation: the
  recorded drone sinks faster than the shallower request); without it the path figures are 16.11 and 15.29 deg.
- **After the touch (`r5-noassist-04`, top bar at 81.86 s).** Contact support v3 does not fire in the `m6` replay
  (`m5`: 80.08 s); it still fires at 80.08 s without the assist or without the sighted descent, so the changed requests
  after the touch account for it, and it says nothing about live behaviour. In `r4b-noassist-02` it fires at 80.08 s in
  every variant, after the audited touch ended (80.044 s); in shadow that is logged and no climb starts. In open loop the
  pilot's older support rule then starts a climb at 80.48 s (`r4b-noassist-02`) and 80.76 s (`r5-noassist-04`),
  because the recorded drone climbed (it flew the round-5 contact climb) while the replay asks to sink. Open loop cannot
  say whether either would start live, or whether the touch itself would happen.
- **FAT SHARK (`r5-noassist-01`, 19.4 s).** The `m6` stack changes 0.06 s of requests before the crash (the assist, at
  most 0.06 m/s). The early brake has no episode there, and nothing in the stack acts on the gap aim's line, which the
  gates branch found on video and in the surrogate to be the cause. In `r4b-noassist-02`'s pass the early brake acts from
  18.83 to 20.14 s and the request reaches its 2.5 m/s floor (without the assist 3.83; `m5` 4.4).
- **Lap-2 start arch (`r4b-noassist-02`, 110.8-112.4 s).** No request changes (lowest 4.88 m/s, as `m5`; the looming
  reads no evidence against the sky). The marker jump in shadow would hold the false reading at 111.40 s; it would also
  have held real rings at 64.61, 70.60, 71.93 and 87.14 s, which is why it failed its held-out gates.
- **Early brake elsewhere on these laps:** `r4b-noassist-02` at (43.8, 63.5) and (-30.5, 28.4), `r5-noassist-04` at
  (43.7, 63.7) and (1.6, 194.8), each floored at 2.5 m/s. Assist v4's ceiling-share hold never acts on these laps (0 s).

**Straw Bale, fast-brain-08 finishes (flown without the stack; counterfactual):**

- The early brake acts 24 and 18 times (30.47 and 23.43 s), at the arches and, because the offline stream lacks
  vertical evidence, at the uphill rings; 4.57% and 2.99% of `m5`'s request travel is removed (2.59% and 1.41% without
  the assist).
- **An interaction of the two brake-branch rules at a lost marker.** At `straw-brain08-06`'s first start arch (7.82 s,
  the marker lost for about 0.1 s, pilot state coast) the early brake and assist v4 together ask 0.69 m/s, where the
  early brake alone asks 4.0 and the assist alone 4.19 (`m5` 6.0). At (-36.3, 98.5), 89.6 s, together 1.39 m/s (alone
  3.49 and 2.76). Outside the ring-in-view floor the early episode brakes as the governor does, and the assist's cap
  tracking then follows a drone that, in the replay, keeps flying 4.8-5.9 m/s. It is an upper bound, and it does not
  occur on the brain-11 laps (lowest 2.5 m/s), but it is a crawl risk at an arch where the marker drops out. The brake
  branch's crawl gate measured only ticks with the ring ahead.
- The round-5 ceiling-cut window (`straw-brain08-06`, 37.49-38.77 s): the pilot's climb request stays 1.47-1.62 m/s
  (assist v3: 0-1.47); the ceiling-share hold acts 1.01 s in the whole lap.
- The sighted descent withholds sink 6.74 and 5.14 s (at most 0.42 and 0.53 m/s); in `-04`'s first downhill (75-80.5 s,
  two audited slides) the steepest request falls from 10.12 to 9.51 deg.

**Minus Two:**

- **Brain hairpin (`minus-brain11cw13-r4b-noassist-01`, wall at 23.596 s).** The governor's cap first binds 1.18 s
  before the impact (`m5` 0.76 s). The request falls at least 1 m/s below `m5`'s from 1.15 s before the impact. At the
  last tick the request along the travel direction is 0.17 m/s (`m5` 2.17; `m5ma3` -0.02; `m6` without the assist
  2.28). The early brake alone does not make the brain follow; the assist's cap tracking does (the brake branch's
  surrogate window: 1.00 m/s at the wall with both, 3.74 with the early brake alone).
- **First arch (`minus-brain11cw13-r5-02`, the assist-v3 crash).** From 6.61 s (x 16.4, the arch at about x 19) the
  request falls 6.0 -> 4.17 -> 2.59 -> 2.50 m/s and stays at the 2.5 m/s floor while the ring is in view, where v3 asked
  1.02 -> 0.23 m/s (the live crawl) and `m5` without an assist kept 6.0 until 6.81 s. The window's lowest request,
  1.75 m/s, comes at 8.0-8.09 s, when the recorded drone had already crawled to 1-1.7 m/s and the pilot's own
  acceleration-limited request was below the floor.
- **Arches and pillars (brain logs):** early episodes at the first arch, around pillar A (x 43.8-44.6; lowest 1.78 m/s
  in `r4b-noassist-01`) and at the 90-degree arch (x 67), floored at 2.5 m/s while the ring is ahead. Episodes go below
  the floor only at the hairpin (0.27-0.55 m/s), around pillar A and at the end of `r5-02` (above).
- **Fast PD:** no request changes on its three logs (no early-brake or assist entry; the sighted descent does not act).
  The `r5-01` pillar exit (the stand-off cap along the old wall ray, diagnosed by the brake branch) is unchanged. The
  marker jump in shadow would hold at 39.65 s (`r4b-01`) and 23.2 s (`r5-01`), two of its held-out new-stop cases.

### Harness and surrogate (closed loop in the identified simulator; not flight evidence)

Fresh sets, chosen here before any run and used by no gate, screening or earlier integration: sim seed 113;
hairpins turn 50/90/130 deg x arch 9/13 m x wall 2.3/2.7 m and pass-through arches turn 25/75 deg x see-through
1.4/1.9 m x surface 0.05/0.3 m, both with live-like wall samples (no below fraction, as the recorded Minus walls); the
gate-post set of `m6-gates` (turn -60/-15/30/75 deg x half width 1.3/1.7 m x false marker 0/-17/+15 deg); descent
surrogate hill 9300-9311 and steep 9400-9407; the Straw downhill rebuild of `m6-ground` with its logged geometry (16
drones) and variations 2001-2016. All runs on the `m6` tree:

- `m5`: `deployed_pilot_kwargs(contract, stale_evidence=True)` (the round-6 rules off: bit-identical to `m5` in the
  replays); `m5ma3`: plus the kept motor assist v3;
- `m6`: plus the early brake, motor assist v4, the sighted descent and the marker jump in shadow;
- `m6` without the assist.

Contact support v3 was on and made **no onset in any run**, so these runs are also the plan's stack (contact support in
shadow). The surrogate has no looming or gap samples (the early brake, gap aim, turn-first and ceiling guard stay idle
there) and no contact physics; the harness walls and arches are synthetic.

Cells: clean (wall / floor / ceiling contacts; posts for the gate set); for pass-through, the lowest speed near the arch
(min / median / max).

| Motor | Stack | Hairpins (12) | Pass-through (8): finished; lowest speed | Gate posts (24): clean; post contacts (with a false marker); smallest post gap |
|---|---|---|---|---|
| fast-brain-11-b-cw13 | `m5` | 0 (12 / 0 / 0) | 8; 4.08 / 4.26 / 4.41 | 24; 0 (0); 0.549 m |
| fast-brain-11-b-cw13 | `m5ma3` | 9 (0 / 0 / 1), 2 unfinished | 5; 0.04 / 0.44 / 0.89 | 24; 0 (0); 0.564 m |
| fast-brain-11-b-cw13 | `m6` | 9 (0 / 0 / 0), 3 unfinished | 8; 0.92 / 1.71 / 2.38 | 24; 0 (0); 0.564 m |
| fast-brain-11-b-cw13 | `m6` without the assist | 0 (12 / 0 / 0) | 8; 1.37 / 2.43 / 2.86 | 24; 0 (0); 0.549 m |
| fast PD | `m5` | 10 (2 / 0 / 1) | 7 (1 crash); 0.41 / 0.94 / 2.38 | 17; 6 (6); 0.276 m |
| fast PD | `m6` | 10 (2 / 0 / 1) | 7 (1 crash); 0.41 / 0.94 / 2.38 | 17; 6 (6); 0.276 m |

| Motor | Stack | Hill (12): contacts / s; high passes | Steep (8): contacts / s; high | Straw variations (16): contacts / s; high | Straw logged (16): contacts / s; high |
|---|---|---|---|---|---|
| fast-brain-11-b-cw13 | `m5` | 10 / 18.33; 7 | 2 / 6.5; 2 | 7 / 19.58; 3 | 3 / 13.94; 4 |
| fast-brain-11-b-cw13 | `m5ma3` | 9 / 17.55; 7 | 2 / 5.96; 2 | 7 / 18.45; 2 | 3 / 13.77; 2 |
| fast-brain-11-b-cw13 | `m6` | 9 / 17.48; 7 | 2 / 5.95; 2 | **5 / 11.74; 1** | 3 / 13.65; 2 |
| fast-brain-11-b-cw13 | `m6` without the assist | 10 / 18.19; 7 | 2 / 6.48; 2 | 5 / 11.99; 2 | 3 / 13.79; 4 |
| fast PD | `m5` | 10 / 12.44; 9 | 8 / 6.36; 2 | 8 / 27.15; 2 | 7 / 29.73; 2 |
| fast PD | `m6` | 10 / 12.47; 9 | 7 / 6.32; 2 | **6 / 23.32; 1** | 6 / 29.09; 0 |

Every course finished and none crashed in every variant; mean finish times moved by at most 0.26 s.

- **The brain needs the assist at hairpins.** Without motor assist v4 fast-brain-11 hits the wall in 12 of 12 (as under
  `m5`); with it, 9 are clean, none touches, and 3 (arch 9 m, turns 90 and 130 deg) stop 0.40-1.33 m from the wall and
  do not finish within 16 s: the stand-off holds a stopped brain (a known blocker). Assist v3 gave 9 clean with one
  ceiling contact.
- **No v3-like crawl at pass-through arches,** but the brain is slower there: lowest 0.92-2.38 m/s with `m6` (without the
  assist 1.37-2.86; `m5` 4.08-4.41; v3 0.04-0.89 with 3 of 8 unfinished).
- **Gate posts:** fast-brain-11 is clean in all 24 scenarios under every variant. The fast PD touches a post in 6 of the
  16 false-marker scenarios under `m5` and `m6` alike (the marker jump is in shadow; its `on` mode removed these in the
  gates branch but failed elsewhere).
- **Ground contact:** on the fresh Straw variations the `m6` stack has fewer contacts than `m5` for both motors
  (fast-brain-11 7 -> 5, 19.58 -> 11.74 s; fast PD 8 -> 6, 27.15 -> 23.32 s) and fewer high passes (3 -> 1, 2 -> 1). On
  the logged geometry fast-brain-11 stays at 3 (13.94 -> 13.65 s) and the fast PD goes 7 -> 6; on the hills and steep
  courses the counts move by at most one. The
  assist's own share on Straw is small and not harmful here: 11.99 -> 11.74 s and 2 -> 1 high passes on the variations,
  4 -> 2 high passes on the logged geometry. **The user's request is still not met in the surrogate:** 5 of 16
  fast-brain-11 drones touch the Straw variations under `m6`.

### Live flight plan (for the main session; development flights)

Fly from `C:\DEV\Haltere` with `m6` checked out. It is a local branch and is not pushed. While the integration worktree
holds the branch, use `git switch --detach m6` in the main checkout, or remove that worktree first. The runner loads
its declarations from the checkout it runs from and refuses other versions. Keep the untracked
`configs/explore_spiral.yaml`. This plan replaces the live plans of the three round-6 branch sections above,
`docs/early_brake.md` and `docs/sighted_descent.md`.

**One stack for every course:**

```
--looming-brake --obstacle-stack on --stale-evidence on --early-brake on --descent-view on --contact-support shadow --sighted-descent on --motor-assist on --marker-jump shadow
```

- **Motor assist v4 is on for every run, the brain on Straw Bale included.** The brain does not stop at a hairpin
  without it: 0 of 12 fresh hairpins clean without it, 9 of 12 with it (above); on the live hairpin the open-loop
  request at the wall is 0.17 m/s with it and 2.28 without, and the brake branch's surrogate window ends at 1.00 m/s
  against 3.74. Version 4 has no approach source (the cause of the `r5-02` crash) and bounds only its own share of a
  climb under a ceiling (0 s of such holds on the brain-11 Straw laps). On Straw Bale it removes 0.5% more of the
  request travel on the brain-11 laps (1.31-1.32% with the early brake, open loop) and, in the surrogate, adds no
  contact (Straw variations 11.99 -> 11.74 s, high passes 2 -> 1). One stack means the Straw run needs it too. Its
  costs: the quiet gates failed on the fast-brain-08 laps (4.21-5.24% with the early brake), arches are slower (lowest
  0.92-2.38 m/s in the fresh pass-through set), and with the marker lost at a looming arch the early brake and the
  assist together can ask for a crawl (0.69 m/s in one fast-brain-08 replay; above). The fast PD has no entry: its
  commands are bit-identical with and without the flag and its sidecar records `applied: false`.
- **Contact support v3 runs in shadow** (`contact_fire` logs where it would climb; it starts no climb). Round 5 flew it
  `on` for the brain. Why the change:
  - corrected after the round-6 review: live, the two downhill touches differ. In `r4b-noassist-02` (descent view v2,
    contact support v2) contact support fired INSIDE the touch (79.654 s; audited touch 79.354-80.044 s) and the drone
    flew a support climb 79.654-80.254 s; that lap passed. In `r5-noassist-04` (contact support v3) the manoeuvre
    exclusion blinded v3 during the touch (79.61-79.90 s), it fired only at 80.078 s, after the touch had ended
    (79.968 s), and that late climb left the drone 0.5-0.7 m high onto the next arch's top bar (video-verified by the
    gates branch; surrogate: 13.48 m at the bar with the climb, 12.65 m without, the lap run about 12.8 m). The 80.08 s
    figure previously given for r4b-02 was an open-loop replay artifact;
  - the r5-04 touch itself ended without any support climb;
  - it fails 3 of its 13 safety gates, held-out detection of this touch among them;
  - no live run shows a v3 climb that helped (the only live v3 fire was the late one; the round-5 fast-PD run in shadow
    logged none). The surrogate's ground is scoring-only, so its 'no onset' is not evidence either way.

  The pilot's older support rule (a descent the drone cannot achieve while the issued throttle stays below hover) stays
  active. A contact-support version that climbs only while the contact lasts is the fix (a blocker below).
- **Marker jump in shadow:** identical requests (126/126 in the gates branch, 31/31 here); it logs where it would hold
  a marker (`marker_held`, `marker_candidates`), the data a reader-side rule needs. `on` failed its held-out gates.
- **Early brake and sighted descent on** for every run; the fast PD has no early-brake entry.

A wiring check parsed every command below with the runner's own argparse and built what `run()` builds before the
camera starts, with `run()`'s own calls and keyword arguments (no camera, pad, preflight record or flight). Each run
declares:

| Declaration | Version | Content sha256 | Applied |
|---|---|---|---|
| `configs/obstacles/lag_turn.json` (the contract's entry) | 2 | `d4eb83da51ab...` | yes |
| `configs/obstacles/gap_pilot.json` (gap cue v2) | 5 | `43c304204f93...` | yes |
| `configs/obstacles/wall_pilot.json` (the contract's stopping model) | 6 | `fe65951d3d68...` | yes |
| `configs/obstacles/vertical_guard.json` | 4 | `409d06f9ded7...` | yes |
| `configs/obstacles/stale_evidence.json` | 2 | `4a9516068f1f...` | yes |
| `configs/obstacles/early_brake.json` | 1 | `ecf76971107f...` | brain: yes; fast PD: no |
| `configs/pilot/descent_view.json` (contact support `shadow`) | 3 | `2bdb17fc2479...` | yes (contact support logged only) |
| `configs/pilot/sighted_descent.json` (mode `on`) | 1 | `49b8d7a79f32...` | yes |
| `configs/pilot/motor_assist.json` | 4 | `8954a798e127...` | brain: yes; fast PD: no |
| `configs/pilot/marker_jump.json` (shadow) | 1 | `8752cd7e2860...` | no (logged only) |

The CSV ends with the view and contact columns (8), the assist columns (8), `cap_ray_deg`, `cap_reseat`, `marker_held`,
`marker_candidates`, `early_brake`, `sighted_los`, `sighted_bound`, `sighted_withheld`. Limits for every run:
`--max-height 250 --max-speed 14 --max-distance 2000`. The camera reads the ring with the earlier reader (no
ring-marker rule). Checkpoints: fast-brain-11-b-cw13 `44cca3c4...`, the fast PD's calibration checkpoint `dad013ed...`.

**Every run is a disclosed development deviation:** no brain is selected; the marker jump, the early brake with assist
v4 and the sighted descent each fail frozen gates; the round-4b/5 rules keep their failures; the contact audit cannot
gate.

**Before every launch** (in the Anode seat, from `C:\DEV\Haltere`; viewer hidden; Liftoff stays open between runs):

1. **Liftoff in the seat.** If the seat was restarted and Liftoff is gone, relaunch it there with Anode's
   `steam_launch 410340`. A NordVPN window opens in the seat over the game: minimize it (about (966,16)). Then the
   original `[Copy] New Drone` on the course.
2. **The pad** must report `seatOnly` and `verifiedFromDesktop` (`anode gamepad state`). A user's game on the desktop is
   no reason to stop; a machine-wide pad or a guard unplug is.
3. **A quiet machine for 60 s.** The runner's preflight refuses competing workloads, for example the user's `rustc.exe`
   builds (they refused four round-4b/5 launches). A refusal writes `<log>.preflight.json`, so that log name cannot be
   reused: use the next number (`-02`). This loop prints each blocker and returns after six passing checks about 10 s
   apart (about 60 s of quiet); it writes nothing:

   ```powershell
   $q = 0; while ($q -lt 6) { $r = .venv/Scripts/python.exe -m haltere.liftoff.preflight | Out-String | ConvertFrom-Json; if ($r.passed) { $q++ } else { $q = 0; $r.blockers | ForEach-Object { "$($_.executable) pid $($_.pid) $($_.cpu_cores) cores" } }; Start-Sleep -Seconds 9 }; 'quiet for about 60 s'
   ```

4. **Ground check** (throttle-low, then throttle, roll, pitch and yaw, checking the processed controls; Xbox neutral
   throttle is not zero):

   ```powershell
   .venv/Scripts/python.exe runs/fast-stack-20260923/ground_check.py runs/fast-stack-20260923/ground-check-36.json
   ```

   Increment the number for every check (36, 37, ...; 35 was the last). **The ground check pauses the game when it
   exits** (the pause inside it cannot be disabled), and `--pause-on-stop` pauses it after every run: click
   **Réinitialiser** (646,277) in the pause menu, or resume, before the next step.
5. **Telemetry live.** This prints `LIVE` when two frames 0.5 s apart show an advancing game timestamp:

   ```powershell
   .venv/Scripts/python.exe -c "import time; from haltere.liftoff.telemetry import TelemetryReceiver,read_config,DEFAULT_STREAM; rx=TelemetryReceiver(port=9001,stream=(read_config() or {}).get('StreamFormat',DEFAULT_STREAM)); a=rx.wait(1.0); time.sleep(0.5); b=rx.wait(1.0); rx.close(); print('LIVE' if a is not None and b is not None and b.timestamp > a.timestamp else 'NOT LIVE', None if b is None else [round(float(v), 2) for v in b.position])"
   ```

   Launch only after `LIVE` (otherwise the run stops at 0 ticks, as both r4-01 runs did).

Run the flights in this order. Each command is one line. Change the course between (a) and (b) and before (d) from the
pause menu (`Changer de Niveau`); steps 1-5 apply before every launch.

**(a) Straw Bale, fast-brain-11-b-cw13, three laps (first).** The graduation course and the user's downhill request;
the direct comparison with `straw-brain11cw13-r4b-noassist-02` (lap 1 in 1:42.988) and the two round-5 laps.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-11-b-cw13/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller brain --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --stale-evidence on --early-brake on --descent-view on --contact-support shadow --sighted-descent on --motor-assist on --marker-jump shadow --seconds 480 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/straw-brain11cw13-r6-01.csv --record runs/fast-stack-20260923/straw-brain11cw13-r6-01.mp4 --video-encoder h264_nvenc
```

What to look for (replay figures are open-loop requests at the recorded states of the development laps):

- **FAT SHARK arch (about 17-19.5 s, x 77-80, y 13-16): not addressed.** The gap aim's right shift (`gap_applied`, up
  to 8.0 and 9.9 deg right at 17.4-18.7 s in the two round-5 laps) set the line that met the right leg. The path at
  y = 14 m: x 77.17-77.56 m in the 23 earlier passes, 78.30 m (passed) and 78.08 m (crashed) in round 5.
  `early_brake` may slow the approach (the 2.5 m/s floor in the r4b lap's replay; no episode in `r5-noassist-01`).
- **(43.8, 63.5), (-30.5, 28.4) and similar:** `early_brake` 1 with the request at 2.5 m/s or more while the ring is in
  view. A request below 2.5 m/s with `pilot_state` `coast` or `search` near an arch is the lost-marker interaction.
- **Uphill rings:** no ceiling holds expected (`assist_share_cap` never set on the replayed brain-11 laps); the climb
  request should not drop to 0.
- **Downhill after the hilltop (72-80 s):** `sighted_los` about 13.5-15 deg; from about 77.5 s `sighted_withheld` > 0
  and the requested path held near 15-16 deg (round 5 asked 22-24); horizontal request 5.6-6 m/s. **No touch at y
  133-134 (x -36.5).** `contact_fire` 1 there means contact support would have climbed (shadow: it does not).
- **ImmersionRC arch (y about 120.7):** pass under the top bar at about 12.8 m (the r4b lap; `r5-noassist-04` hit it at
  13.5 m after the late climb). A `support_climb` state right after a touch is the older support rule.
- **Start arches of laps 2 and 3 (x 23-27, y -0.5 to 0.4): the false marker is not addressed.** `marker_held` 1 marks a
  reading the rule would have held (the r4b replay: 111.40 s).
- **Lap times** against 1:42.988 (expect them longer: slower arches); stick change 0.0025-0.0030 in round 4b/5.

After the run, score ground contact with the contact audit (report next to the video; its false-positive check failed,
so it does not decide pass or fail). The same command, with the run's name, works for every run:

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.contact_audit runs/fast-stack-20260923/straw-brain11cw13-r6-01.csv --json runs/fast-stack-20260923/straw-brain11cw13-r6-01.contact-audit.json
```

**(b) Minus Two, fast PD.** The PD has no early-brake or assist entry and the sighted descent changed none of its
Minus requests, so this repeats `minus-fast6-r5-01` with contact support in shadow (as then).

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-08-vgs04-s10r03m30/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller pd --pd-profile fast --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --stale-evidence on --early-brake on --descent-view on --contact-support shadow --sighted-descent on --motor-assist on --marker-jump shadow --seconds 150 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/minus-fast6-r6-01.csv --record runs/fast-stack-20260923/minus-fast6-r6-01.mp4 --video-encoder h264_nvenc
```

What to look for: the sidecar's `early_brake_declaration` and `motor_assist_declaration` say `applied: false`.

- **The 90-degree arch and the hairpin:** `r4b-01` came to rest at the hairpin too (0.12 m/s at 20.04 s); `r5-01` also
  stopped past the 90-degree arch (0.26 m/s at 18.32 s).
- **The hairpin exit: not addressed.** In `r5-01` the governor's stand-off cap along the old wall ray (24.4 deg) stayed
  active after turn-first released aligned; the path ran 13-18 deg left of the ring's bearing into the dark pillar at
  (78.3, 23.2). Watch `clearance_status` `standoff` after `turn_first` ends while the ring sits at the image centre.
- **The garage wall behind the arch at (52.6, 94.1):** stale-evidence re-seat (`cap_reseat`) as in the round-5 plan.

**(c) Minus Two, fast-brain-11-b-cw13, with motor assist v4 (on, as in every run: see above).** The direct comparison
with `minus-brain11cw13-r4b-noassist-01` (hairpin wall) and `minus-brain11cw13-r5-02` (the v3 crawl at the first arch).

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-11-b-cw13/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller brain --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --stale-evidence on --early-brake on --descent-view on --contact-support shadow --sighted-descent on --motor-assist on --marker-jump shadow --seconds 150 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/minus-brain11cw13-r6-01.csv --record runs/fast-stack-20260923/minus-brain11cw13-r6-01.mp4 --video-encoder h264_nvenc
```

What to look for (open-loop figures from the two development logs):

- **First arch (x 15-19):** `early_brake` 1 from about x 16; the request settles at 2.5 m/s while the ring is in view
  (v3 asked 1.02 -> 0.23 m/s there and the brain crawled into the arch). No rule-caused request (early-brake floor or `assist_source`) below 2.5 m/s with the ring in
  view.
- **Pillar A and the 90-degree arch:** slower than round 4b (2.5-4 m/s); around pillar A the replay went to 1.78 m/s.
- **Hairpin (wall near (81.9, 20.0)):** the cap should bind about 1.2 s before the wall (r4b: 0.76 s), the request fall
  to about 0.2 m/s at the wall (open loop) and the brain arrive at about 1 m/s (surrogate). The harness also shows the
  stand-off parking a stopped brain 0.4-1.3 m from the wall.
- **Ceiling:** braking climbs under the 2.2 m ceiling (`assist_vertical`, `assist_share_cap`).
- **Stick change per tick:** 0.0038 in r4b.

**(d) Pine Valley, fast PD.** No round-6 rule acts for the PD except the sighted descent, which withheld sink for
0.15 s of `pine-fast6-ttc-01` without changing a request (the ground branch's replay).

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-08-vgs04-s10r03m30/candidate.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller pd --pd-profile fast --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake --obstacle-stack on --stale-evidence on --early-brake on --descent-view on --contact-support shadow --sighted-descent on --motor-assist on --marker-jump shadow --seconds 150 --max-height 250 --max-speed 14 --max-distance 2000 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/fast-stack-20260923/pine-fast6-r6-01.csv --record runs/fast-stack-20260923/pine-fast6-r6-01.mp4 --video-encoder h264_nvenc
```

What to look for: as round 5's plan (d): the mound escalation near 4.55 s, the backside sink at 6.3-7.4 s, the hillside
at 15.2 s, the end hillside escalation near 19.05 s; no rule arrests a descent into the boulder.

**Stop criteria.**

- **Before a run, do not start if** the machine was not quiet for 60 s, the pad is not `seatOnly`, the ground check
  fails, or telemetry is not `LIVE`.
- **After a run, stop the series if** the sidecar declares any other version or hash than the table above, the
  descent view's `contact_support` is not `shadow`, `sighted_descent_declaration` or `early_brake_declaration` is
  missing, `motor_assist_declaration.applied` or `early_brake_declaration.applied` is wrong for the motor, or a
  controller deadline or camera failure was recorded.
- **During a run, stop the runner if:**
  - the drone climbs above about 1.9 m in the Minus garage, or keeps climbing above 4 m elsewhere;
  - it is parked for more than about 3 s (an early-brake, assist, turn-first or governor stand-off hold), in particular
    at an arch with the ring in view or with the marker lost;
  - it circles in search for more than about 30 s;
  - on the Straw downhill the drone overflies the ImmersionRC arch or turns back for it (the sighted descent holding the
    path too shallow).
- **Between runs:** stop and analyse instead of flying the next run if a round-6 rule causes a new failure. First
  replay the log through this stack (development evidence), from `C:\DEV\Haltere` with `m6` checked out; `LOGNAME` is
  the log's name without `.csv`, and the output folder must exist:

  ```powershell
  New-Item -ItemType Directory -Force C:/Users/artem/AppData/Local/Temp/haltere-replay | Out-Null
  .venv/Scripts/python.exe haltere/obstacles/vertical_replay.py LOGNAME --out C:/Users/artem/AppData/Local/Temp/haltere-replay/r6 --stack on --near-on-path --throttle-column command_thr --descent-view configs/pilot/descent_view.json --contact-support shadow --stale-evidence configs/obstacles/stale_evidence.json --early-brake configs/obstacles/early_brake.json --sighted-descent configs/pilot/sighted_descent.json --motor-assist configs/pilot/motor_assist.json --marker-jump configs/pilot/marker_jump.json --marker-jump-mode shadow
  ```

- **Graduation evidence:** none of these runs counts (no brain is selected and rules of the stack fail gates). A clean
  Straw 3-lap run without an audited contact would be the first live evidence for the downhill fix; a crash cause must
  be checked on the video before it is written down.

### Blockers for graduation, and what would clear them

| Blocker | State after round 6 | What would clear it |
|---|---|---|
| A selected brain | None; fast-brain-11-b-cw13 flies as the only braking brain with live Straw and Minus evidence | A brain scored once on fresh frozen gates that passes them; then flights |
| Straw FAT SHARK arch (gap aim at an oblique arch) | Not addressed: the gap aim's pre-switch right shift set the crash line; the `m6` stack changes 0.06 s of that log's requests | A gap-pilot revision that separates a gate's own leg from a pillar (the gap cue's depth profile on both sides of the ring), frozen with gates on the gap-commit sets and every Straw arch pass |
| Straw start-arch false marker | Not addressed: the marker-jump rule failed its held-out gates (it holds real rings reacquired after a gap); flown in shadow to log live candidates | A reader-side rule scored on the live `marker_held` logs and the reader's candidates |
| Straw downhill ground contact | Sighted descent v1 (10 of 12 gates): the request held at 15-16 deg where round 5 asked 22-24 (open loop); fresh Straw variations 7 -> 5 contacts for fast-brain-11 (surrogate); not flown | Live (a) without an audited contact; a fix for the brain flying 2-4 deg steeper than asked in view and for crest-grazing lines |
| Late support climb (top-bar crash) | Contact support v3 fires after the touch has ended; flown in shadow in this plan | A contact-support version that climbs only while the contact lasts, frozen and scored on the two Straw touches and the safety gates |
| Brain hairpin | Early brake + assist v4: 9 of 12 fresh hairpins clean (0 without the assist), 3 parked 0.4-1.3 m from the wall; not flown | Live (c); a wall-pilot stand-off that holds a stopped brain without parking it |
| Slower arches, lost-marker crawl | Lowest pass-through speed 0.92-2.38 m/s; 0.69 m/s requested at a lost marker in one fast-brain-08 replay | Live lap times; an early-brake/assist version that keeps a floor while the marker is briefly lost, frozen and scored |
| Fast PD hairpin exit | The stand-off cap along the old wall ray deflects the exit (diagnosed, not fixed) | A wall-pilot version, frozen and scored with PD hairpin sets and the Minus PD logs |
| Quiet gates (early brake, assist v4) | Failed on the fast-brain-08 laps (offline stream without vertical evidence); 1.31-1.32% on the live brain-11 laps | Straw live lap times; a version that also leaves out samples without vertical evidence while the pilot climbs toward its ring |
| Contact audit | Its video false-positive check still fails | An audit v2 validated on video, frozen before scoring |

### Round 6 live plan: amendments after the combined review (2026-09-29)

- **Run (c), Minus brain, and run (b), Minus PD: watch the hairpin exit.** Once the looming governor holds a stand-off on
  one wall it keeps that wall's cap direction for `standoff_s` (2 s); a new obstacle in another direction cannot take
  over the cap meanwhile, and stale-evidence v2's keep_standoff blocks the re-seat (synthetic check in the review; the
  live precedent is `minus-fast6-r5-01`, whose cap stayed on the old ray while it flew into the pillar at (78.3, 23.2)).
  Stop criterion: `clearance_status` standoff/brake with `cap_ray_deg` on the old wall while the drone accelerates in
  another direction toward the pillar area. The blocker "hairpin exit deflected by a stale stand-off cap" covers the
  brain too.
- **Wall behind a ring in view** (m6-brake blocker, restored): the early brake's floor holds 2.5 m/s while the ring is
  ahead, even at an urgent TTC; the brain needs about 2.1 m to stop from 2.5 m/s (e.g. the Minus garage wall 1.5 m
  behind the arch at (52.6, 94.1)).
- The m6-brake branch's own live-plan command above (`--contact-support on`, without sighted descent and marker jump,
  log stem `minus-brain11cw13-r6-01`) is **superseded** by the merged plan's run (c).
- Watch-item wording: at the first Minus arch the pilot's own acceleration-limited request may dip below 2.5 m/s after
  the drone slowed (replay: 1.75 m/s at 8.06 s, state below); only a rule-caused dip (early-brake floor or
  `assist_source`) is a flag.


## Round 6 (live), 2026-09-29: m6 stack

Branch `m6` at `9269a62`: `--looming-brake --obstacle-stack on --stale-evidence on --early-brake on --descent-view on
--contact-support shadow --sighted-descent on --motor-assist on --marker-jump shadow` for every run (wall pilot v6,
vertical guard v4, descent view v3, early brake v1, sighted descent v1, motor assist v4 (no-op for the fast PD),
stale evidence v2, gap pilot v5, lag turn v2). Seat session 4 (hidden viewer), pad 32 `seatOnly`, ground checks 36
(Minus), 37 (Straw), 38 (Pine). Every run is a disclosed development deviation (no brain is selected; every round-4b..6
rule fails at least one frozen gate). The postflight checks of the three Minus/Straw runs found a `rustc.exe` compile
in the user's session (0.71-0.73 core; 6.22 cores at the end of the Minus brain run); no controller-deadline, camera or
telemetry failure was recorded. Crash causes were checked on the video.

| Run | Motor | Outcome |
|---|---|---|
| `minus-fast6-r6-01` | fast PD | Pillar A (y 5.35), hairpin (exit 20.1 s), pillar C (22.0 s), round 4's floor/ceiling spot (31.9 s), on to y 96.7. At the garage wall behind the arch (where `r4b-01` hit at ~6 m/s) the governor braked it from 6.0 to 1.4 m/s (40.4-40.9 s); maneuvering slowly toward the next ring it bumped that arch's ring structure at 2.1 m/s, 41.4 s, (54.8, 94.1) (impact 30.7 m/s^2, the gentlest yet) |
| `minus-brain11cw13-r6-01` | fast-brain-11-b-cw13 | **First brain run through the Minus Two hairpin and past pillar C.** Pillar A (y 5.76); the early brake engaged before the hairpin (early_brake 1 from ~27 s at the next wall), hairpin exit 23.9 s, pillar C 25.7 s. At 29.6 s, passing through a low floor-standing arch as the marker switched to the next ring, the pilot turned right and the arch's leg hit the drone's right side at 4.65 m/s, (76.0, 46.5). Stick change 0.0046 |
| `straw-brain11cw13-r6-01` | fast-brain-11-b-cw13 | **No ground contact on the downhill** (contact audit 0; every earlier brain-11 run touched at ~79.4 s): the sighted descent withheld 0.8-1.6 m/s of sink while the ring was bottom-clipped. Passed the ImmersionRC arch that ended `r5-04`. Further down, still descending at 1.6-2.3 m/s with the ring bottom-clipped, it came down onto the next arch's white top banner at 87.3 s, (-36.2, 95.9, 7.1), 5.2 m/s. Stick change 0.0034 |
| `pine-fast6-r6-01` | fast PD | The Pine hillside at 14.8 s, (59.1, -16.3, 4.1), 5.9 m/s: vertical guard v4 answered the hillside with its 1 m/s gentle climb (stage 1); the round-3 stack's 3.5 m/s climb carried `pine-fast6-ttc-01` to 21.2 s. The expected outcome of the failed V-Pine gate |

What this shows:

- The round-6 hairpin braking (early brake + assist v4) works live for the brain; the sighted descent removed the Straw
  downhill touch in this run.
- The common blocker now is gate structures around checkpoint switches: turning toward the next ring while still
  inside a gate (Minus floor arch, earlier FAT SHARK) and descending onto the next gate's top bar with the ring clipped
  below the image (Straw, twice). Pine needs a guard that answers real hillsides.
