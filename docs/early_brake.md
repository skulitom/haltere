# Early brake and motor assist v4 (round 6, branch `m6-brake`)

**Status: not flown.** Both rules are off by default:

- `--early-brake on` (inside `--obstacle-stack on|shadow`): `configs/obstacles/early_brake.json` **version 1**
  (`ecf76971...`), declared per motor contract. The brain contract has an entry; the fast PD has none and flies
  bit-identically with the flag.
- `--motor-assist on`: `configs/pilot/motor_assist.json` **version 4** (`8954a798...`). The runner refuses the kept
  versions 1-3 (`motor_assist_v3.json`, `7c3b49e7...`). The fast PD has no entry.

Without the new flags the round-5 stack is bit-identical to `m5` (identity gates below; the full-stack digest test).
Gates: `configs/obstacles/early_brake_gates.json` version 1 (`3b029e32...`), frozen and committed (`33a27dd`) before any
gate run, scored by `haltere.obstacles.early_brake_gates`. Scores: `docs/experiments/early_brake_scores.json`.
Development evidence: `docs/experiments/early_brake_development.json`.

**Scored once: 16 of 21 gates pass.** None of it is flight evidence.

**Passes:**

- **Identity:** 31 of 31 logs bit-identical to `m5` for the default pilot, the round-5 stack, the round-5 stack with
  assist v3, and the shadow stack with the early brake. The 11 fast-PD logs are unchanged with both new rules.
- **Development cases:**
  - on the live brain hairpin the governor's cap first binds 1.18 s before the impact (round-5 stack: 0.76 s);
  - with the early brake and assist v4 the final cut of at least 1 m/s starts 1.15 s before the impact (assist v3:
    1.20 s);
  - in the surrogate window the brain reaches the wall at 1.00 m/s (round-5 stack: 3.88);
  - no crawl is planned before the first arch while the ring is in view;
  - the ceiling-share case passes, but its frozen metric was blind (below).
- **Held out (fresh harness sets, seed 83):**
  - fast-brain-10b: 14 of 16 hairpins clean (base 0, assist v3 14), and every pass-through arch at 1.03 m/s or more
    (assist v3: 0.21-1.03);
  - hill courses: both gated brains within the bounds;
  - held-out logs: no new crawl with the ring ahead (0 s on every held-out brain log). The ring-in-view floor holds on
    all 20 brain logs, development and held out. The gates measure to the last logged tick, so D1's base reads 0.76 s
    where the impact time gives 0.77 s.

**Fails:**

- **fast-brain-11-b-cw13, hairpins:** 10 of 16 clean and 2 walls (assist v3: 11 and 2), but 2 ceiling contacts against
  the base's 0 (bound +1; assist v3 had 3).
- **fast-brain-11-b-cw13, pass-through:** one arch at 0.66 m/s with assist v4 (bound 0.75; median 1.79). Assist v3
  crawled to 0.09-0.82 m/s; the round-5 stack flew 3.65-4.55.
- **Quiet, early brake (held-out brain-08 laps):** 1.88-2.59% of the Straw Bale request travel removed and 3.95% on
  Pine Valley (bound 1.5%).
- **Quiet, with assist v4:** 4.21-5.24% on Straw and 7.68% on Pine (bound 3%; assist v3: 1.98-4.18% and 1.60%).
- **Ceiling share (held out):** 0.38 s on `straw-brain08-01` (bound 0.1 s). Assist v3 scores the same 0.38 s; it is not
  the ceiling cut (below).

**Neither rule makes a brain a release brain.** The live hairpin failure is addressed only in development evidence and
the surrogate.

## What failed live (development cases)

The three logs below were read to design the rules; every result on them is development evidence.

- **The brain hairpin (`minus-brain11cw13-r4b-noassist-01`, fast-brain-11-b-cw13, no assist).** The looming samples read
  the hairpin wall at a TTC of 1.01, 0.96, 0.96, 0.96, 1.02, 0.89, 0.89 s from 22.28 s (4.5-4.8 m/s), 1.3 s before the
  impact at 23.596 s. The governor engages on three samples below 0.8 s; they came at 22.75-22.90 s. Its cap first bound
  the request at 22.83 s, 0.77 s before the wall. A brain that follows about 0.3 s late needs about 1 s.
- **The assist-v3 crash at the first Minus arch (`minus-brain11cw13-r5-02`, video-verified in the release check).** In
  front of an arch that every earlier brain run passed at about 5.5 m/s, with the ring in view and centred, v3's approach
  source cut the request, and its cap-tracking extra took it below the approach floor (the logged request fell from 6.0
  to 0.23 m/s by 6.86 s). The brain pitched up, climbed and crawled. The governor's own 1.14 m/s cap then held a
  stand-off, and the brain bumped the arch at about 1 m/s (8.1 s). Open-loop replays show the same v3 mechanism at every
  Minus arch of every brain log: requests of 0.02-0.16 m/s at the first arch, pillar A and the 90-degree arch.
- **The ceiling-cut merge interaction (`straw-brain08-06`, round-5 integration, open loop).** While v3 added a sag
  climb, the pilot's own climb toward its ring lost its exemption from wall pilot v6's ceiling cut, and the cut held the
  whole climb at 0 m/s for 0.89 s on the Straw uphill.

## The rules

### Early brake (`EarlyBrakeConfig`, the looming governor)

A wall sample also votes for the governor's engagement when the distance left to it (its capture-time reach along its
ray, minus the odometry travelled along that ray since) is at most the contract's stopping distance at the closing
speed v along the ray:

v x 0.3 s + v^2 / (2 x 3.5 m/s^2) + 0.5 m

This is the brain contract's measured stopping model (wall pilot `turn_first_stopping`, motor assist). At 4.5 m/s it is
4.74 m (a TTC of 1.05 s); at 5.5 m/s 6.47 m (1.18 s). The governor's own condition waits for TTC < 0.8 s.

- **Confirmation.** The governor's own count (3 votes within 0.25 s) engages a governor that is not yet braking, with its
  own graded target (0.7 x the closing speed below a TTC of 1.03 s).
- **What does not vote:**
  - below-path terrain;
  - samples the lower window explains (a lower-surface TTC at most the alarm TTC): the Straw Bale uphill rings loom at
    0.9-1.1 s at 4.5 m/s on the development laps, with such a lower TTC;
  - samples during a governor terrain climb.
- **Floor.** While the pilot sees its next checkpoint ahead (none of turn-first's checkpoint conditions: side clamp, lost
  marker, bearing 50 deg or more off the heading, turn-first episode or side guard), every target of the episode is at
  least 2.5 m/s. So an early episode never plans a stop or a stand-off (2 m/s) toward a surface the ring is seen
  through. Past an arch with the next checkpoint to the side, the governor brakes exactly as without the rule.
- **End.** The episode ends when the cap has released back to the closing speed at its engagement.
- **Fast PD.** It has no entry. With its model (0.15 s, 6 m/s^2) the distance test would come about two samples
  (0.11 s) before the governor's own condition at 6 m/s.
- **Logs.** CSV column `early_brake` (last, only when declared); sidecar `pilot_assistance.early_brake` (rule,
  parameters, counts) and `early_brake_declaration`.

### Motor assist version 4

Version 3's rule (cap tracking of the request, governor, turn-first and stopping sources; slew 15 m/s^2; the stopping
source only under turn-first's wall-ahead conditions, with its 1 s memory; sag compensation), with three changes:

1. **No approach source.** The approach is what crashed r5-02 with the ring in view. Early braking toward a surface is
   now the governor's early brake, on the pilot's own request, floored while the ring is ahead.
2. **Tracking floor.** Unless the stopping source binds (a wall confirmed under a wall-ahead condition), cap tracking of
   the request and governor caps never lowers a bound below min(the bound, 2.5 m/s).
3. **Ceiling share.** While the pilot climbs toward its ring in view (state `cue`), the assist's sag climb no longer
   removes the pilot's exemption from the ceiling cut. Overhead evidence confirmed during the sag climb starts a
   separate hold (the ceiling guard's own evidence and confirmation) that bounds only the assist's share of the climb to
   the guard's `vertical_cap`. The pilot's own request is untouched. Wall pilot v6 is unchanged.

## Development evidence (before the freeze)

All of it is development evidence: the logs were read to design the rules, and the harness sets were used to choose
between variants.

**Open loop** (the recorded drone does not respond to the requests):

- **Hairpin (`minus-brain11cw13-r4b-noassist-01`).** With the early brake the cap binds the request from 22.42 s,
  1.18 s before the impact; the round-5 stack waits until 22.83 s (0.77 s). The request is 3.18 m/s where the round-5
  governor later asks 3.47.
- **Minus arches, brain logs.** Assist v3 asks 0.02-0.16 m/s at the first arch, pillar A and the 90-degree arch. The
  early brake with assist v4 asks at least 2.5 m/s there while the ring is in view. The early brake alone moves the
  arches' lowest request by about 0.1 m/s (it engages 0.1-0.25 s earlier there).
- **Straw Bale laps of fast-brain-11-b-cw13.** The early brake acts at three places, never at the uphill rings: the
  FAT SHARK arch (lowest request 2.8 m/s where the round-5 stack asks 3.04), (43.8, 63.5) and (-30.5, 28.4) (3.9 where
  4.9). It removes 0.76-0.79% of the request travel.

**Surrogate window** (fast-brain-11-b-cw13 in the surrogate from the logged state at 22.0 s, flying the replayed
requests). Speed 0.05 s before the logged impact:

| Requests | Speed (m/s) |
|---|---|
| Round-5 stack | 3.88 |
| Logged | 3.89 |
| Early brake alone | 3.74 |
| Early brake + assist v4 | 1.00 |
| Assist v3 | 0.55 |

The brain does not follow the governor's lower cap without the assist's cap tracking.

**Harness** (live-like wall samples: no below fraction, no lower-surface TTC, as the recorded Minus walls). Clean
passes / wall contacts of 12 hairpins; lowest speed (m/s) around pass-through arches:

| Brain, set | Round-5 stack | Assist v3 | Early brake | Early brake + v4 | Engagement floor + v4 |
|---|---|---|---|---|---|
| fast-brain-11-b-cw13, h41 | 0 / 12 | 8 / 0 | 0 / 12 | 8 / 1 | 7 / 2 |
| fast-brain-11-b-cw13, h67 | 0 / 12 | 10 / 0 (2 ceiling) | 0 / 12 | 10 / 0 | 9 / 1 |
| fast-brain-11-b-cw13, dev17 (14 s) | 0 / 12 | 3 / 1 | 0 / 12 | 3 / 3 | 3 / 5 |
| fast-brain-10b, h41 | 0 / 12 | 11 / 0 | 1 / 11 | 9 / 3 | - |
| brain-09b, h41 | 0 / 12 | 10 / 0 (2 floor) | 6 / 3 (3 floor) | 11 / 0 (1 floor) | - |
| fast-brain-11-b-cw13, pt41 (8 arches) | 3.97-4.61 | 0.05-1.06 (2 unfinished) | 1.82-3.08 | 0.92-2.24 | 0.17-1.6 (1 unfinished) |
| fast-brain-11-b-cw13, pt17 (6) | 4.01-4.43 | 0.44-1.01 | 1.11-2.96 | 1.05-2.32 | 0.11-1.69 (1 unfinished) |
| fast-brain-10b, pt41 | 3.33-5.05 | 0.33-1.12 (1 unfinished) | 1.04-3.96 | 1.05-2.35 | - |
| brain-09b, pt41 | 2.95-4.64 | 0.37-1.44 (2 floor) | 0.81-2.88 (3 floor) | 0.49-2.46 (2 floor) | - |

- **The early brake alone does not stop the brains at hairpins** (0-1 of 12 clean for fast-brain-11 and fast-brain-10b).
  The brains fly through the governor's caps; the assist's cap tracking makes them follow.
- **With assist v4 the brains stop about as often as with v3.** fast-brain-11-b-cw13: 8 and 10 clean against v3's 8
  and 10. fast-brain-10b: 9 against 11, with 3 wall contacts against 0. That is the cost of dropping the approach
  source.
- **At pass-through arches v4 no longer crawls to 0.05-0.4 m/s.** But every brain is slower there than with the
  round-5 stack. The lowest speeds come after the pass: the turn to the next ring at low speed, where turn-first can
  engage behind the arch.
- **brain-09b** sinks to the floor when it brakes (3 of 8 pass-through arches with the early brake, 2 with v3 and with
  v4). It is reported, not gated.
- **The floor while the ring is ahead beat the floor until the governor's own engagement** in every set. The
  engagement floor let the governor's cascade reach its stand-off before the arch.

The development tables are in `docs/experiments/early_brake_development.json`. Runs marked `[h1]` or `[h2]` used an
earlier tracking floor (off under any wall-ahead condition); `[h3]` is the declared rule.

## The fast PD's hairpin stop and the pillar (`minus-fast6-r5-01`; development)

The card said that the r4b fast PD took the hairpin without stopping. Its telemetry and video say otherwise.

**The hairpin stop is not new.** The r4b PD also came to rest at this hairpin: 0.12 m/s at 20.04 s at (79.1, 18.2).
The recorded frame at about 20.0 s shows 1 km/h on the HUD beside the arch. It left once its governor cap had
released (the cap rose from 4.25 m/s at 23.0 s) along x 80.1-80.4 at y 20-22.

**What differed in r5-01:**

1. **A stop at the 90-degree arch (x 67-72).** Just past the arch the looming read a surface 0.6-0.1 m ahead (TTC
   0.15-0.05 s at 17.59-17.81 s). The governor stopped the PD, 0.26 m/s at 18.32 s at (71.3, 11.1), and turn-first
   engaged while the marker was lost. The recorded frames show the drone passing through the arch and pitching up hard
   under the ceiling structure behind it. The r4b PD braked there only to 2.55 m/s.
2. **A slow approach to the hairpin.** It flew 1.3-3.7 m/s under caps of 0.75-1.89 m/s and came to rest at the wall at
   (79.9, 17.6): 0.94 m/s at 24.79 s, in the stand-off, with turn-first active from 24.14 s until it released aligned at
   24.94 s.
3. **The exit was deflected by the stand-off cap.** It accelerated out at 25.0 s while the stand-off along the wall's
   ray (24.4 deg, cap 0.92 m/s) was still active, until about 26.2 s. The brake removed the request's component along
   that ray:
   - the ring marker sat in the middle of the image (u 0.50-0.51) at a heading of 85-90 deg;
   - the request pointed 101.7-105.6 deg (mean 103.1);
   - the path ran 13-18 deg left of the heading.

   The recorded frames show the dark pillar growing on the left of the image from about 25.5 s until the image goes
   dark at about 26.0-26.1 s. The recording lags the log by about 0.15 s.

**The round-5 rules did not cause it.** r5-01 replayed through the `m4b` stack and the `m5` stack gives the same request
on all 2608 ticks: none differs by more than 0.05 m/s. The stop at the 90-degree arch and the hairpin stand-off are
closed-loop outcomes of rules `m4b` already had. A different line gave different looming.

**Does the exit line need the gap aim or turn-first to consider obstacles? Not first.** The line came from the
governor's stand-off cap along the old wall ray:

- turn-first had already released aligned (24.94 s);
- the gap aim shifted right only late, 1.0-7.1 deg from 25.83 s (it saw something near the path).

A report-only diagnostic (not a rule) ended the stand-off cap when turn-first released aligned. The request then pointed
72.7-98.7 deg (mean 81.7). Flown in the surrogate from the logged state at 24.9 s, that path crosses y = 23.2 m at
x = 80.29, 2.0 m right of the pillar's centre (r4b's line: x 80.1-80.4). The as-flown requests reproduce the logged path
in the same surrogate: it ends at (78.39, 22.86) against the logged (78.41, 22.82).

**The fix belongs in the wall pilot (next round; not done here).** A stand-off cap should not deflect the request once
turn-first has released aligned: end it, or scale the request along its own direction instead of removing its component
along the old ray. It changes the fast PD, so it needs its own frozen version with PD hairpin sets, the Minus PD logs
and quiet gates. The early brake and motor assist v4 have no fast-PD entry and change nothing here.

## Frozen gates and results

`configs/obstacles/early_brake_gates.json` version 1 (`3b029e32...`) was frozen with both declarations and the scorer
in `33a27dd`, before any gate run. One scorer fix was committed before scoring (`b4d0e3b`). D1 had taken the first
braking tick of the last 3 s, which a braking episode running from before the window fills at the window start; it
now takes the first braking onset. The replays ran through this tree and `git archive m5` (`bc7c71c`). Scores:
`docs/experiments/early_brake_scores.json`. Post-scoring diagnostics: `docs/experiments/early_brake_post_scoring.json`.

Development logs: the 10 round-4/4b/5 logs named above plus `straw-brain08-06` (the ceiling-share case). Held-out logs:
the other 20 of the 31 logs, which no replay of these rules touched before the freeze. The fresh harness sets use
parameter values and a seed used by no earlier set:

- hairpins: turn 40/70/100/125 deg x arch 8.5/12.5 m x wall 2.35/2.75 m, 16 scenarios;
- pass-through arches: turn 15/50/100 deg x see-through 1.25/1.75 m x surface 0.1/0.35 m, 12 scenarios;
- both with live-like wall samples, sim seed 83, 16 s;
- hill courses 6500-6511.

| Gate | Threshold (short) | Result | Pass |
|---|---|---|---|
| I1 default pilot | bit-identical to m5, 31 logs | 31/31 | yes |
| I2 round-5 stack (stale evidence v2, descent view v3) | bit-identical, 31 logs | 31/31 | yes |
| I3 round-5 stack + assist v3 (kept declaration) | bit-identical, 31 logs | 31/31 | yes |
| I4 shadow stack with the early brake | bit-identical to m5's shadow, 31 logs | 31/31 | yes |
| I5 fast PD with both rules | = the stack, 11 PD logs | 11/11 | yes |
| D1 hairpin warning (dev, open loop) | first cap binding >= 1.0 s before the impact | 1.18 s (base 0.76) | yes |
| D2 hairpin cut with assist v4 (dev) | final cut >= 1 m/s starts >= 1.0 s before | 1.15 s (assist v3 1.20) | yes |
| D3 hairpin window (dev, surrogate) | end speed <= base - 1.5 m/s | 1.00 (base 3.88, logged 3.89, assist v3 0.55) | yes |
| D4 first arch of r5-02 (dev) | no ring-in-view tick below min(own, 2.5) | 0.0 s | yes |
| D5 ceiling share, straw-brain08-06 (dev) | no own-climb loss with an assist climb | 0.0 s (assist v3 also 0.0: blind metric) | yes |
| H1 hairpins, fast-brain-11-b-cw13 | clean >= base + 6; walls <= 25% base; floor, ceiling <= base + 1; walls <= v3 + 4 | clean 10, walls 2, ceiling 2 (base 0 / 16 / 0; v3 11 / 2 / 3) | **no** |
| H1 hairpins, fast-brain-10b | as above | clean 14, walls 1, ceiling 1 (base 0 / 13 / 1; v3 14 / 1 / 1) | yes |
| H2 pass-through, fast-brain-11-b-cw13 | early brake and with v4: finished, contacts as base; lowest >= 0.75, median >= 1.5 m/s | early brake 1.75-2.86; + v4 0.66-2.38 (base 3.65-4.55; v3 0.09-0.82, 1 unfinished) | **no** |
| H2 pass-through, fast-brain-10b | as above | early brake 1.50-3.00; + v4 1.03-2.28 (base 2.39-4.66; v3 0.21-1.03, 1 unfinished) | yes |
| H3 hills, fast-brain-11-b-cw13 | finishes, crashes; time <= +2%; contacts <= +1; high <= +2; chatter <= 1.15x | 12/12, time +0.05%, contacts 8 -> 7 | yes |
| H3 hills, fast-brain-10b | as above | 12/12, time +1.52%, contacts 5 -> 5, high 23 -> 20 | yes |
| H4a no new crawl with the ring ahead (12 held-out brain logs) | <= 0.1 s per log | 0 s everywhere (both variants) | yes |
| H4b quiet, early brake (held-out Straw/Pine brain laps) | <= 1.5% of request travel | Straw 1.88-2.59%, pine-brain08-01 3.95%, pine-brain08-loom-01 0 | **no** |
| H4c ring-in-view floor (20 brain logs) | 0 s below min(own, 2.5) | 0 s | yes |
| H4d quiet, early brake + v4 | <= 3% | Straw 4.21-5.24%, pine-brain08-01 7.68% (v3 1.98-4.18%, 1.60%) | **no** |
| H5 ceiling share (held-out brain logs) | <= 0.1 s per log | 0.38 s on straw-brain08-01 (v3 the same), 0 elsewhere | **no** |

Reported, not gated:

- **brain-09b, hairpins:** 12 of 16 clean with 4 floor contacts (assist v3: 11 clean, 5 floor). It sinks when it
  brakes.
- **brain-09b, pass-through:** the early brake adds contacts (5 against the base's 3; with assist v4 3).
- **Gate-style wall samples** (below fraction 0.5, lower-surface TTC = TTC): the early brake never votes, so assist v4
  passes 1 of 16 hairpins for fast-brain-11-b-cw13 and fast-brain-10b, where assist v3 passes 11 and 14.

**What the failures are:**

- **The ceiling contacts (H1, fast-brain-11):** braking climbs in the 2.2 m scoring ceiling. Assist v3 had one more.
- **The pass-through minimum (H2):** a 100-degree turn right after the arch. The brain arrives at about 2.5 m/s instead
  of 4 and turns slowly; assist v3 crawled to 0.1 m/s in the same scenario.
- **The quiet failures (H4b, H4d):** on the brain-08 laps the offline looming stream carries no vertical evidence at the
  uphill rings. So the early brake engages there as well as at the arches: 24 episodes in the 5.4-minute
  `straw-brain08-04`, lowest request 3.0-5.1 m/s. On the live fast-brain-11 laps (development) the uphill samples
  carried a lower-surface TTC, and the early brake removed 0.76-0.79%.
- **The ceiling-share failure (H5):** a 0.07 m/s difference in the pilot's own sink request while the assist adds a
  climb. The assisted request feeds back into the pilot's own state through the issued request, and assist v3 shows
  exactly the same 0.38 s.
- **D5's metric was blind.** When version 3's cut zeroes the climb, the capped sag climb adds nothing, so the metric's
  condition (the assist adds a climb) never held. A post-scoring diagnostic without that condition finds version 3's
  own request below the unassisted stack's for 1.27 s in the development window, and version 4's for 0.00 s. It
  changes no score.

## Risks

- **Not flight evidence.** The harness walls, arches and looming are synthetic, and the replays are open loop.
- **The early brake depends on the wall samples carrying no lower-window explanation.** With the harness's gate-style
  samples it never votes, and assist v4, which has no approach, then stops almost no brain at a hairpin (1 of 16,
  where v3 stops 11-14). The recorded Minus Two walls carry no below fraction in 78-100% of their short-TTC samples.
  If Liftoff's walls read otherwise live, the hairpin reverts to round 5 without any approach braking.
- **A wall behind a ring still in view.** The floor holds while the checkpoint is ahead. A brain slowed to 2.5 m/s is
  not braked further by the early episode until the marker switches (the Minus garage wall 1.5 m behind the arch at
  (52.6, 94.1)); from 2.5 m/s it needs 2.14 m. The governor's own cascade (without the rule) would have gone on braking
  there. Next version: lift the floor at the governor's urgent TTC (0.4 s) too.
- **Hairpins (fast-brain-11-b-cw13).** 10 of 16 clean, 2 walls and 2 ceiling contacts in the held-out harness, against
  v3's 11, 2 and 3. The live hairpin window reaches the wall at 1.00 m/s, not 0.
- **Slower arches.** With the early brake the brains reach every gate arch at 2.5-3 m/s instead of 4-5. The lowest
  harness pass-through speed is 0.66-2.38 m/s with v4 (fast-brain-11). After the arch pass, turn-first can engage behind
  a slowed brain (its wall is the arch's looming surface, "reached" once passed). Minus Two and Straw Bale lap times
  will be longer.
- **Straw Bale.** On the live fast-brain-11 laps the early brake acts at the FAT SHARK arch (lowest request 2.8 m/s
  where round 5 asks 3.04), at (43.8, 63.5) and at (-30.5, 28.4), and removes 0.8% of the request travel
  (development). On the held-out brain-08 laps, whose offline looming stream lacks vertical evidence, it also acts at the
  uphill rings and removes 1.9-2.6% (quiet gate failed). With assist v4 it removes 4.2-5.2%.
- **brain-09b** sinks to the floor when it brakes (held-out hairpins: 4 floor contacts of 16). It is not the flown
  brain.
- **Ceiling share.** It is verified only on the development window and on synthetic states. The frozen metric could not
  see the failure, and the held-out ceiling gate failed on an unrelated 0.38 s feedback window that v3 shares.
- **The fast PD's pillar exit is not fixed.** The stand-off cap along the old wall ray still deflects its exit after a
  turn-first release. That needs a wall-pilot version.

## Files

- Rules: `haltere/liftoff/fast_race_cue.py`:
  - `EarlyBrakeConfig`, `early_brake_for_contract`;
  - `TtcClearanceGovernor(early=...)`, `_early_vote`, `ring_ahead`, `share_climb` / `share_cap`;
  - `FastRaceCue(early_brake=..., early_apply=...)`, `_checkpoint_beside`, `early_log`;
  - `MotorAssistConfig` (`track_floor`, `ceiling_share`, version 4), `MOTOR_ASSIST_V3_FIELDS`.
- Runner: `--early-brake on|off` (`haltere/liftoff/visual_brain.py`: `load_early_brake`, `EARLY_COLUMNS`); the runner
  loads motor assist version 4 only.
- Replay harness: `haltere/obstacles/vertical_replay.py --early-brake DECLARATION` (file tag `-eb1`; arrays gain
  `early_brake`, and `assist_share_cap` with a version-4 assist).
- Deployed pilot: `haltere.train.deployed_pilot.deployed_pilot_kwargs(early_brake=True)` (off by default).
  `motor_assist=True` now loads version 4. `motor_assist=3` loads the kept round-5 declaration, which the brain-12
  gates (`haltere.train.brake_gates`, `motor_assist: true`) and `fast_motor_tracking --motor-assist` keep using, so
  their records stay reproducible.
- Gates: `configs/obstacles/early_brake_gates.json`, `haltere/obstacles/early_brake_gates.py`.
- Tests: `tests/test_round6_brake.py`; `tests/test_fast_race_cue_motor_assist.py` (version 4 declaration, kept v3).
  - Full suite at the end: 1227 passed and 1 failed. The failure is the wall-clock budget guard
    `tests/test_obstacle_overlays.py::test_budget_regression_guard`, run while other sessions loaded the machine. It
    passes alone and passed in the previous full run, which failed only on two tests fixed since (the obstacle-stack
    flags gained `early_brake`; the brain-12 deployed-pilot test now names motor assist version 3).
- Scratch scripts (development and diagnostics): session scratchpad `m6/brake/`.
