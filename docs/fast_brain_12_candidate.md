# brain-12: a braking brain trained for left and right turns, hairpin braking and descents (surrogate) — 2026-09-28

Branch `m5-brain12` (from `m4b` 2a5bccb). Offline only: no Liftoff, Anode, pad or live run; CPU at 2 threads, one
heavy process at a time. Every new training option is off by default, and with them off a collection and a refit are
bit-identical to `m4b` ([experiments/brain12_identity.json](experiments/brain12_identity.json)).

**Result: no candidate passes every primary gate of the frozen `configs/brain12_gates.json` v1, so under its rule none
is selected. The brain release stays pre-release.** Five candidates were trained and each was scored once. The
best-ranked is **`fast-brain-12-b-cw26d3`**: 6 of 15 primary gates, sha256
`2ecf3f1f9b388c16f1ccbea3a9321aae15263a8e5369eadc090258d9f66728ec`.

Under the same gates, fast-brain-11-b-cw13 also passes 6 of 15. So brain-12 moved the faults around rather than
removing them. Compared with fast-brain-11-b-cw13, fast-brain-12-b-cw26d3:

- **Improves:**
  - G11: it re-accelerates to 4 m/s at 60 deg to the left within 1.31 s; brain-11 never does.
  - G16: 8 clean hairpins of 12 with the motor assist and no wall contact; brain-11 has 5 and 1.
  - G14 downhill: it passes 1.22 m above the live contact onset and keeps its speed (brain-11 -0.01 m, 0.51 m/s short).
  - Left turns are less one-sided: the worst settled deficit is -1.36 m/s against -1.75.
- **Regresses:**
  - G1: it follows cap steps more slowly (t_within up to 1.25 s, one never).
  - G2: it overshoots sustained 3-3.5 m/s requests by +0.63-0.72 m/s.
  - G5: in-course cap excess 0.585.
  - G14 hairpin: the capped excess is 1.50 against 1.17.
  - G15: right turns under-brake.
  - **G13 terrain contacts: 23 in 34.88 s, against brain-11's 21 in 33.29 s.** The gate asks for fast-brain-10b's 14
    and 21.76 s.

**The ground-contact direction the user asked for is not met by any brain-12 candidate.** The fewest contact seconds
come from `fast-brain-12-c` (20 contacts in 29.38 s), which fails 11 gates. Nothing here is flight evidence: the
surrogate is IdentifiedSim with synthetic courses, logged Minus Two / Straw Bale states and synthetic walls.

## Round-5 pilot branches: what was merged

The task was to merge the round-5 pilot branches that are flight ready, so that the DAgger rollouts and the gates fly
the round-5 pilot.

| Branch | Result returned | Merged |
|---|---|---|
| `m5-brake` (motor assist v3, `7003741`) | flight_ready true (10 of 11 frozen gates; the quiet gate fails on Straw request removal) | **yes** (`010daa3`) |
| `m5-safety` (wall pilot v6, descent view v3 / contact support v3, `d8f9ba3`) | flight_ready false (contact support v3 fails 3 of 13 frozen gates) | no |
| `m5-arches` (stale-evidence and ring-marker rules) | no result returned | no |

So the round-5 pilot here is the `m4b` stack (lag turn v2, gap pilot v5, wall pilot v5, vertical guard v4, descent
view v2 with contact support v2) **plus motor assist v3** for the brain contract (`--motor-assist on`,
`configs/pilot/motor_assist.json` v3 `7c3b49e7...`; the fast PD has no entry). `haltere.train.deployed_pilot` builds it
with `motor_assist=True`; the new `--motor-assist` training flag and the `motor_assist: true` field of G13 use it.
Contact support v2 is the version whose false fire the round-4b card reports (4 ceiling contacts of 12 in every
hairpin set below, for every motor); m5-safety's fix is not in this stack.

## Gates (frozen before any brain-12 candidate existed)

`configs/brain12_gates.json` version 1, sha256 `8fb1b1a0c412...`, frozen 2026-09-28 11:46:18 and committed in `f0222ca`
before any training. It keeps the brain-11 gate definitions (G1-G13; G3 v1 reported only) and changes three things:

- **Fresh held-out sets.** The round-4b audit found that brain-11's G13 and course gates reused the training evaluation
  seeds 3000-3007, which are also the DAgger round-3 collection seeds. brain-12's course gates use flat/steep 7300-7307,
  hill 7400-7411, sim seed 37 and cap seeds 9101/9102. No earlier training, training evaluation or gate used them, and
  none of brain-12's own development screening did either: the training evaluation used steep 7100-7107, the dev
  screening steep 7200-7207, hill 7210-7215 and sim seed 23. A unit test checks that the sets are disjoint.
- **Thresholds re-measured on the fresh sets with the brain-11 definitions** before the freeze:
  - G6 and G7 against brain-08: mean speed >= 0.95 x brain-08, lateral error at 1 s <= brain-08 + 0.3 m, chatter
    <= 1.15 x brain-08, and so on;
  - G13 against fast-brain-10b under the same pilot.
- **G13 flies the round-5 pilot** (motor assist on) and adds **terrain contact seconds** (<= fast-brain-10b's).

New gates:

| Gate | What it asks |
|---|---|
| G14 (development cases) | The two round-4b live windows of fast-brain-11-b-cw13, replayed from the logged state with the logged request. **Hairpin** (`minus-brain11cw13-r4b-noassist-01`, 22.0 s to 0.05 s before the wall): mean speed over the capped request from 0.3 s after the cap fell <= 0.5 m/s. With motor assist v3 applied to the logged pilot's request, the speed at the wall must be <= 1.0 m/s. **Downhill** (`straw-brain11cw13-r4b-noassist-02`, 76.0-80.1 s): pass >= 0.3 m above the live contact onset along the live path, and keep speed on the descent (mean request minus speed <= 0.3 m/s). Plant validity: logged commands open loop within 0.3 m/s and 0.15 m |
| G15 (held out) | The mirror images of G10 and G11 (capped turns of -45/-90 deg and accelerations toward -30/-60 deg, to the right), with G10/G11's thresholds |
| G16 (fresh set) | The motor-assist harness hairpins (turn 40/70/100 x arch 8.5/12.5 x wall 2.35/2.75 m, sim seed 59) under the round-5 pilot with the assist. Clean passes >= fast-brain-11-b-cw13's + 2. Wall contacts <= its; floor and ceiling contacts <= its + 1. Without the assist: report only |

Selection rule: every primary gate (G1, G2, G3v2, G5-G16) must pass, with a clean weights audit. Among those, the
fewest G13 contact seconds win, then the lowest 16-course chatter. Otherwise none is selected, and candidates are ranked
by gates passed, then contact seconds.

**Sim-to-real check of G14 (before the freeze).**

| Window | Live | Logged commands open loop | fast-brain-11-b-cw13 replayed | FastMotorPD | PD + 80 ms |
|---|---|---|---|---|---|
| Hairpin: speed at the wall (capped excess) | 3.768 m/s (1.225) | 3.712 (1.107) | 3.814 (1.166) | 3.271 (-0.107) | 3.088 (-0.194) |
| Downhill: height over the live contact onset (speed shortfall) | 0 m (0.467 m/s) | +0.084 m (0.585) | -0.012 m (0.514) | +1.111 m (0.137) | +1.125 m (0.136) |

The surrogate reproduces both live failures, so G14 carries weight; it is still a gate on development cases.
The replayed hairpin with motor assist v3 ends at 0.929 m/s for fast-brain-11-b-cw13.
Other brains on the same windows:

- fast-brain-10b: 4.033 m/s (excess 1.558) at the hairpin; it never reaches the downhill contact point in the window.
- brain-08: 3.741 m/s (excess 1.033); it never reaches the downhill contact point either.
- brain-09b: 2.861 m/s (excess 0.646) at the hairpin, and +0.446 m on the downhill.

**The reference controllers under these gates** (measured on this tree before the freeze):

- **fast-brain-11-b-cw13** passes 6 of 15 primary gates on the fresh sets: G1, G2, G7, G8, G9 and G12.
  - It also fails G5 (in-course cap excess 0.413 > 0.4) and G6 (one descending pass more than 1.5 m high; post-switch
    lateral error 3.027 m > 3.0). It passed both on the old sets.
  - G13: 21 terrain contacts in 33.29 s. For comparison, fast-brain-10b has 14 in 21.76 s, brain-08 8 in 16.05 s and
    FastMotorPD 17 in 24.73 s.
  - G15: it under-brakes in right turns, where it over-brakes in left ones (settled down to -0.959 m/s, t_within up
    to 1.21 s). Its right-hand accelerations reach 4 m/s in 0.83-1.02 s; the left 60-degree ones never do.
  - G16: 5 clean hairpins of 12 with the assist, 0 without.
- **fast-brain-10b** passes G13 and G16.
- **FastMotorPD** passes G15.

## Why the braking brains over-brake for requests left of their heading (development diagnosis)

All of this is development evidence (scratchpad `m5/brain12`, [experiments/brain12_development.json](experiments/brain12_development.json)).
The dev sets are disjoint from the gates: turns of ±60/±75 deg at 1.2 rad/s to 3.5 m/s, accelerations toward
±15/±45/±75 deg, and courses steep 7200-7207 and hill 7210-7215 at sim seed 23.

**1. The data are lopsided, but only after the first brain-flown round.** fast-brain-11-b's saved DAgger data were split
by the side of the request relative to the velocity (left = counterclockwise, z up). The first brain-flown round was the
one with 3 crashes of 10 drones.

| Round (flown by) | fast (>= 4.5 m/s) requests >= 30 deg off, left / right | 10-30 deg, left / right | over-speed >= 0.8 m/s, left / right |
|---|---|---|---|
| 0 (label teacher) | 21 / 19 | 190 / 192 | 316 / 324 |
| 1 (brain) | 230 / 1259 | 168 / 835 | 749 / 2788 |
| 2 (brain) | 95 / 58 | 549 / 307 | 977 / 2139 |
| 3 (brain) | 69 / 63 | 334 / 261 | 417 / 633 |
| 4 (brain) | 45 / 39 | 337 / 200 | 474 / 534 |

**2. Re-weighting the same data does not remove the fault.** Refits of that data were screened, and the settled speed
in the left capped turns stays 1.6-1.85 m/s under the cap in every one:

- side-balanced weights (every speed and angle bin weighted equally left and right);
- the roll row's smoothness relaxed (0.3 or 0.1);
- lateral samples up-weighted;
- ridge 1e-3 down to 1e-4.

Lower ridge slows both sides.

**3. The fault needs a lateral request without yaw.** A trace of the dev 60-degree capped turn with the yaw stick held
compares fast-brain-11-b-cw13 with its label teacher in the same states:

| | Label teacher roll | fast-brain-11-b-cw13 roll |
|---|---|---|
| Left | -0.25..-0.50 | -0.07..-0.11, decaying to 0 and then +0.1 |
| Right | +0.26..+0.44 | +0.06..+0.17 |

The same turn with the yaw stick following the request:

- fast-brain-11-b-cw13 settles at 3.51 m/s left (under a 3.5 cap) and 4.22 right;
- FastMotorPD settles at 3.48 both ways;
- starting headings of 0/90/180/270 deg give identical results, so the compass input plays no part.

So G10 and G11 test the brain's roll toward a body-frame lateral request that no yaw rate accompanies. In the course
data, left turns almost always come with the pilot's yaw.

**4. What moves it: data of yaw-free lateral requests on both sides.** Two extra DAgger rounds of *synthetic body-frame
turns* were flown by fast-brain-11-b-cw13 (the request turned 30-90 deg left or right with the yaw held) and added to
its data, with lateral weight 3. Settled speed (1-3 s after the cap):

| | Left, yaw held | Right, yaw held | Left, yaw following | Right, yaw following |
|---|---|---|---|---|
| fast-brain-11-b-cw13 | -1.82..-0.48 | -0.58..-0.25 | 3.51 m/s | 4.22 m/s |
| With the two synthetic-turn rounds | -1.07..-0.99 | -1.02..-0.68 | 3.94 m/s | 3.76 m/s |

The fault becomes symmetric but does not go away: both sides still settle about 1 m/s under the cap with the yaw
held. That is the basis of the `--synthetic-turns` option of candidates a and b.

## Recipe (`haltere/train/fast_motor_tracking.py`; every new option off by default)

The fast-brain-11-b-cw13 recipe is kept:

- parent `motor-brain-10-tracking-05`, brain-08's contract;
- 5 DAgger rounds x 10 courses x 110 s;
- synthetic caps on 60% of the drones, slow legs 30%, hill share 0.2, yaw holds 4/min;
- the label teacher (attitude gain 4, vertical velocity gain 7), turn relief 0.5, label lead 0.08 s;
- the deployed pilot on half the drones;
- balance-speed, sink weight 10, brake weight 5 (level 1.6, turn 90 deg);
- ridge 0.003, smoothness 100 (rows 0.3/1/1).

Only readout rows 0-2 and their biases change. New in brain-12:

| Option | What it does | Why |
|---|---|---|
| `--pilot deployed --motor-assist` | The deployed pilot of the rollouts is the round-5 pilot (motor assist v3 for the brain contract) | The stack the brain will fly |
| caps `configs/brain12_caps.json` (`brake_rate_max` 15, `release_max` 10) | Each synthetic cap falls at U(8, 15) m/s^2 and rises after its hold at U(3, 10) m/s^2 | The live hairpin cap fell 5.9 -> 3.5 m/s within 0.15 s, 0.7 s before the wall; the assist slews at 15 m/s^2; a quick release makes the pilot's own 10 m/s^2 slew shape a hard re-acceleration (G11) |
| `--synthetic-turns R` | R times per minute per drone the request turns 30-90 deg left or right in the body frame at 0.8-1.5 rad/s, holds 0.5-2 s and turns back, with the yaw stick held (training data only) | Diagnosis 3-4 above |
| `--lateral-weight W` | Up-weights samples whose request points >= 10 deg off the velocity (refit, every round) | Same |
| `--side-balance G` | Weights left and right requests alike per speed and angle bin, at most G x (refit, every round) | Diagnosis 1; neutral in the refits, kept so that an early lopsided round cannot dominate |
| `--sink-relief F` | The label teacher sees the sink beyond 0.5 m/s reduced by the fraction F (vz' = vz + F x max(0, -vz - 0.5)); training labels only | The user's direction: fewer ground touches, keep speed and nose forward on descents instead of sinking into the hill. On the fresh sets the brain that sinks less than asked (brain-08, 0.29 m/s) has the fewest terrain contacts, and the exact FastMotorPD more than fast-brain-10b; G8 still bounds the steep sink shortfall at 0.25 m/s |
| `--descent-weight W` | Up-weights aligned descents at >= 1 m/s with a horizontal request >= 0.8 x nominal (refit) | Keep speed on descents |
| `--mirror-courses` | Each round also flies the mirror images of its courses | Implemented and tested; not used by any candidate (diagnosis 1: the teacher-flown round is already balanced) |

`--evaluation-seeds 7100-7107` (steep): the training evaluation no longer uses the gate seeds.

## Candidates

Three DAgger collections (CPU, 2 threads, 35-55 min each) and two refits of their saved data were made, five
candidates in all under `C:/DEV/Haltere/runs/fast-brain-12-*` (never overwritten). Every candidate was scored once on
the frozen gates after all five existed.

| Candidate | Collection | Refit weights (besides brain-11's) |
|---|---|---|
| `fast-brain-12-a` | round-5 pilot on half the drones; brain-12 caps; synthetic turns 6/min | cruise 1.3, lateral 3, side balance 4 |
| `fast-brain-12-a-cw20` | a's data | cruise 2.0 |
| `fast-brain-12-b` | as a, plus sink relief 0.1; cruise 2.0 in the rounds | cruise 2.0, lateral 3, side balance 4 |
| `fast-brain-12-b-cw26d3` | b's data | cruise 2.6, descent 3 |
| `fast-brain-12-c` | round-5 pilot on half the drones; brain-12 caps; sink relief 0.1; no synthetic turns; cruise 2.6 and descent 3 in the rounds | cruise 2.6, descent 3, side balance 4, no lateral weight |

**Sequential design (disclosed).**

- The candidates were designed one after another on development screening only (`dev12.py`, `quick_turns.py`;
  [experiments/brain12_development.json](experiments/brain12_development.json)), never on the gates:
  - b's sink relief and cruise 2.0 were chosen after screening a and its refits (a had more dev contacts and a slower
    6 m/s cruise than brain-11);
  - c was a no-turns control;
  - the two refits were chosen from dev screening: a-cw20 directly, b-cw26d3 from its components (cruise 2.6 and
    descent 3 were screened separately on b's data).
- The two promoted refits reproduce their development refits bit for bit (readout equality).
- Several training options were added after the gate freeze and before the candidates existed; the gates and scorer
  are unchanged since the freeze (`f0222ca`):
  - `--synthetic-turns`, `--lateral-weight` and `--sink-relief` were committed in `d032c5a`;
  - `--side-balance` and `--mirror-courses` were in the freeze commit.

## Results (frozen gates v1, sha256 `8fb1b1a0c412...`)

Every report was made on this tree with the frozen file (`gates_frozen` true). The ranking is by primary gates passed,
then G13 contact seconds. fast-brain-11-b-cw13 is shown as the reference; it is not a candidate.

| candidate | G1 | G2 | G3v2 | G5 | G6 | G7 | G8 | G9 | G10 | G11 | G12 | G13 | G14 | G15 | G16 | passed | sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fast-brain-12-b-cw26d3` | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | pass | pass | pass | **FAIL** | pass | pass | **FAIL** | **FAIL** | **FAIL** | pass | 6/15 | `2ecf3f1f9b38...` |
| `fast-brain-12-b` | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | pass | pass | pass | **FAIL** | **FAIL** | pass | **FAIL** | **FAIL** | **FAIL** | pass | 5/15 | `8769b46fb3d2...` |
| `fast-brain-12-c` | pass | **FAIL** | **FAIL** | **FAIL** | **FAIL** | pass | pass | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | pass | 4/15 | `1beff2d0b809...` |
| `fast-brain-12-a` | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | pass | pass | pass | **FAIL** | **FAIL** | pass | **FAIL** | **FAIL** | **FAIL** | **FAIL** | 4/15 | `fa3618a51088...` |
| `fast-brain-12-a-cw20` | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | pass | pass | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | 2/15 | `e7b66b44ad0b...` |
| `fast-brain-11-b-cw13 (reference)` | pass | pass | **FAIL** | **FAIL** | **FAIL** | pass | pass | pass | **FAIL** | **FAIL** | pass | **FAIL** | **FAIL** | **FAIL** | **FAIL** | 6/15 | `44cca3c4fc3e...` |

| metric | `fast-brain-12-b-cw26d3` | `fast-brain-12-b` | `fast-brain-12-c` | `fast-brain-12-a` | `fast-brain-12-a-cw20` | `fast-brain-11-b-cw13` | limit |
|---|---|---|---|---|---|---|---|
| G1 cap t_within max (hover) | 1.25 (1 never) | 1.04 | 0.98 | 1.05 | 1.21 | 0.91 | <= 1.0 s |
| G2 sustained excess, hover (2/3/3.5/4.5 m/s) | -0.03/+0.63/+0.71/+0.38 | +0.01/+0.34/+0.43/+0.14 | +0.01/+0.39/+0.48/+0.32 | +0.07/+0.38/+0.40/+0.09 | +0.07/+0.50/+0.58/+0.34 | -0.05/+0.14/+0.27/+0.20 | +-0.3 |
| G2 cruise excess at 3.5/4.5/6 m/s | +0.72/+0.42/-0.50 | +0.43/+0.17/-0.72 | +0.39/+0.24/-0.51 | +0.39/+0.07/-0.82 | +0.54/+0.29/-0.56 | +0.26/+0.15/-0.55 | +-0.3; -0.6..+0.3 |
| G3v2 worst band distance | 0.93 | 0.73 | 0.88 | 0.65 | 0.88 | 0.78 | <= 0.5 |
| G5 / G12 cap excess | 0.585 / 0.375 | 0.457 / 0.300 | 0.413 / 0.458 | 0.401 / 0.297 | 0.543 / 0.468 | 0.413 / 0.340 | <= 0.4 |
| G6 lateral 40-75 at 1 s; high descending passes | 3.044 m; 0 | 2.898 m; 0 | 3.084 m; 2 | 2.913 m; 1 | 3.005 m; 1 | 3.027 m; 1 | <= 3.0; 0 |
| G6 16-course finished / chatter | 16 / 0.00219 | 16 / 0.00200 | 16 / 0.00190 | 16 / 0.00192 | 16 / 0.00199 | 16 / 0.00208 | >= 15 / <= 0.0035 |
| G8 steep sink shortfall | 0.041 | 0.214 | 0.096 | 0.029 | 0.017 | 0.093 | <= 0.25 |
| G9 hairpin excess / stop-accelerate loss | 0.25 / 0.30 m | 0.35 / 0.29 m | 0.58 / 0.26 m | 0.47 / 0.25 m | 0.53 / 0.25 m | 0.38 / 0.05 m | <= 0.5 / <= 0.4 |
| G10 left turns: settled excess (hover+live) | -1.36..+0.73 | -1.36..+0.52 | -1.81..+0.24 | -1.31..+0.41 | -1.30..+0.51 | -1.75..+0.22 | -0.6..+0.3 |
| G15 right turns: settled excess (hover+live) | -1.14..+0.73 | -1.22..+0.52 | -0.59..+0.33 | -1.17..+0.41 | -1.11..+0.51 | -1.00..+0.33 | -0.6..+0.3 |
| G10 / G15 excess 0.5-1.5 s max | 1.16 / 1.16 | 1.04 / 1.04 | 0.73 / 0.91 | 0.88 / 0.88 | 1.04 / 1.04 | 0.57 / 0.65 | <= 0.5 |
| G11 left: height loss max; time to 4 m/s max | 0.28 m; 1.31 s | 0.23 m; 1.19 s (1 never) | 0.52 m; 1.11 s (2 never) | 0.14 m; 1.13 s (2 never) | 0.14 m; 1.11 s (2 never) | 0.06 m; 1.08 s (2 never) | <= 0.4; <= 1.5 |
| G15 right accel: height loss max; time max | 0.26 m; 1.01 s (1 never) | 0.25 m; 1.03 s (1 never) | 0.70 m; 1.04 s | 0.13 m; 0.97 s | 0.13 m; 0.97 s | 0.36 m; 1.02 s | <= 0.4; <= 1.5 |
| G13 terrain contacts / seconds | 23 / 34.88 | 21 / 41.81 | 20 / 29.38 | 25 / 47.16 | 24 / 44.39 | 21 / 33.29 | <= 14 / <= 21.76 |
| G13 high passes / chatter / terrain finish | 9 / 0.00245 / 57.79 | 10 / 0.00222 / 61.14 | 14 / 0.00215 / 56.78 | 5 / 0.00213 / 59.52 | 5 / 0.00223 / 58.06 | 7 / 0.00232 / 58.39 | <= 20 / <= 0.003921 / <= 62.89 |
| G14 hairpin: capped excess; speed at wall with assist | 1.50; 0.55 m/s | 1.39; 0.51 m/s | 1.03; 1.13 m/s | 1.32; 0.38 m/s | 1.50; 0.41 m/s | 1.17; 0.93 m/s | <= 0.5; <= 1.0 |
| G14 downhill: height over contact onset; speed shortfall | +1.22 m; -0.42 | -0.17 m; 0.98 | +0.50 m; 0.38 | +0.04 m; 0.30 | +0.07 m; 0.28 | -0.01 m; 0.51 | >= 0.3; <= 0.3 |
| G16 hairpins with assist: clean / wall / ceiling (no assist: walls) | 8 / 0 / 4 (12) | 7 / 1 / 4 (12) | 8 / 0 / 4 (12) | 6 / 0 / 4 (12) | 5 / 1 / 4 (12) | 5 / 1 / 4 (12) | >= 7 / <= 1 / <= 5 |

What the round asked for, in the surrogate:

- **Over-braking for requests left of the heading.** The fault is real, but it is not about the data's side balance.
  It is the brain's weak roll toward a lateral request that no yaw rate accompanies (G10 holds the yaw stick at 0). The
  synthetic body-frame turns make it more symmetric, but they do not fix it:
  - G10 settled left: -1.30..-1.36 m/s (a, b, b-cw26d3) against -1.75 for brain-11 and -1.81 for c (no turns).
  - The right side gets worse: G15 settled -1.11..-1.22 against -1.00.
  - Every candidate fails G10 and G15.
  - With the yaw following the request, as the pilot flies, brain-11 turns left fine and under-brakes right; see the
    diagnosis.
- **Slow 60-degree re-acceleration.** Fixed only in fast-brain-12-b-cw26d3: G11 passes, with every case reaching 4 m/s
  within 1.31 s and 0.28 m of height lost. On the right (G15), one of its four accelerations never reaches 4 m/s.
- **Earlier and stronger hairpin braking.**
  - With the round-5 assist (G16), b-cw26d3 and c pass 8 of 12 clean with no wall contact (brain-11 5, 1 wall).
    Without the assist, every brain hits all 12 walls.
  - In the live r4b hairpin window with the logged request (G14), no candidate follows the late cap better than
    brain-11: capped excess 1.03-1.50 against 1.17. FastMotorPD gives -0.11.
  - The faster synthetic caps did not teach a faster response to a cap that falls while the brain is still
    accelerating. G1 cap steps got slower in the four candidates with synthetic turns (t_within 1.04-1.25 s against
    0.91); c passes G1 (0.98 s).
- **Ground contact (count and seconds).**
  - G13 on the fresh sets under the round-5 pilot: 20-25 contacts and 29.4-47.2 s. The references are fast-brain-10b
    14 / 21.76, brain-08 8 / 16.05, FastMotorPD 17 / 24.73 and brain-11 21 / 33.29.
  - Only c (no synthetic turns) has fewer contact seconds than brain-11.
  - The replayed Straw downhill passes above the live contact only for the two candidates with cruise 2.6 and
    descent weight 3: b-cw26d3 (+1.22 m) and c (+0.50 m). b, with the sink relief alone, is at -0.17 m and 0.98 m/s
    short.
  - The sink relief does not reduce the surrogate's terrain contacts: b 41.8 s, b-cw26d3 34.9 s.

## Weights audit

Every candidate changed only readout rows 0-2 (throttle, roll, pitch) and their biases against the parent
`motor-brain-10-tracking-05` (sha256 `64444a62...`). The yaw row and the other 38 tensors are bit-identical to it:
connectome wiring, weights, transmitter signs, time constants and encoders. They are therefore also bit-identical to
brain-08, brain-09b, fast-brain-10b and fast-brain-11-b-cw13. The audits are in `runs/fast-brain-12-*/weights-audit.json`.

| Candidate | sha256 | Largest change: throttle / roll / pitch |
|---|---|---|
| `fast-brain-12-b-cw26d3` | `2ecf3f1f9b388c16f1ccbea3a9321aae15263a8e5369eadc090258d9f66728ec` | 0.265 / 0.125 / 0.380 |
| `fast-brain-12-b` | `8769b46fb3d275465fec96d5675a9a362ab9224c8e3bafa9e5b8eb1fb327df8c` | 0.288 / 0.132 / 0.358 |
| `fast-brain-12-c` | `1beff2d0b809761d8ac1b98c037107b8ab43cdc8449411c5e42d55f63f5200b2` | 0.261 / 0.083 / 0.292 |
| `fast-brain-12-a` | `fa3618a510887cf79e415c45333d3ab5adca037ed749b111cdebd23c9706cf4b` | 0.241 / 0.138 / 0.312 |
| `fast-brain-12-a-cw20` | `e7b66b44ad0baeeab2bb4c59fc754e9e34a1e47c715a2a65b49a22b0535ef757` | 0.239 / 0.138 / 0.310 |

The synthetic turns roughly double the roll row's largest change (0.13 against 0.067 for fast-brain-11-b-cw13 and
0.083 for c).

## Limits and risks

- **Surrogate and replay only; nothing flew.**
  - G14 uses two live windows that were looked at before the freeze (development cases). The replays hold the logged
    request, so the pilot does not react to the new brain.
  - G16's walls, looming and ceiling are synthetic. Its 4 ceiling contacts of 12 are the contact-support v2 false fire
    (m5-safety's fix is not merged).
- **No brain-12 candidate is a graduation candidate.** Flying fast-brain-12-b-cw26d3 would be a disclosed development
  deviation, like the brain-11 flights. Its gates predict:
  - better hairpin stops with the assist, and keeping height and speed on the Straw downhill replay;
  - over-speed under moderate requests and caps (G1, G2, G5): about 0.6-0.7 m/s over sustained 3-3.5 m/s requests.
    This matters at the Minus arches, where the governor caps to 3.5-4 m/s;
  - no fewer terrain contacts than fast-brain-11-b-cw13 in the surrogate.
- **The fresh sets are harsher than the old ones.** On them fast-brain-11-b-cw13 also fails G5 (0.413) and G6
  (lateral 3.027 m, one high descending pass), which it passed on the seeds it was evaluated on in training.
- **Terrain contacts in the surrogate depend strongly on the pilot's descent requests.**
  - The brains track the requested sink within 0.004 m/s on the dev hills.
  - The brain that sinks least faithfully (brain-08, 0.29 m/s short) has the fewest contacts, and the exact FastMotorPD
    more than fast-brain-10b.
  - A brain-only fix pulls against G8 (steep sink tracking). The descent view's requests near terrain are the more
    direct lever.
- **The yaw-free lateral fault may be a readout limit.** Weighting, regularisation and extra yaw-free turn data moved
  it only part of the way. The connectome's lateral goal response without a yaw rate is weak, most of all to the left.

## Next

- The descent view (pilot) should ask for less sink near the terrain. brain-12's data say the brain already does what
  it is asked, and asking it to sink less than requested trades against G8.
- Hairpins: the brains stop only with the assist (G16 8 of 12 at best, 0 without). The live cap in the r4b window came
  0.7 s before the wall, which no brain here follows (G14). The assist's approach bound is the working lever.
- The left/right fault: G10 and G15 hold the yaw stick at 0. A pilot rule that yaws before a lateral capped request
  would avoid the state. A training route would need the readout to use the lateral goal without a yaw rate, and five
  rounds of synthetic turns were not enough.
- The graduation bar (Straw Bale 3/3 with no ground contact, Minus Two 3/3) stays out of reach for the brain release
  after this round.

## Files

- **Training options:** `haltere/train/fast_motor_tracking.py`:
  - `SyntheticTurnConfig`, `SyntheticTurns`, `rotate_request`;
  - `side_balance_weights`, `lateral_mask`, `descent_mask`;
  - `descent_sink_relief` / `LabelTeacher(sink_relief=)`;
  - `mirror_course`, `SyntheticCapsConfig.brake_rate_max` / `release_max`;
  - CLI `--synthetic-turns --lateral-weight --side-balance --sink-relief --descent-weight --mirror-courses --motor-assist`.
- **Caps:** `configs/brain12_caps.json`.
- **Gates:** `configs/brain12_gates.json`. The runner is `haltere/train/brake_gates.py`, with the parts
  `capped_turns_right`, `accelerate_right`, `r4b_windows` and `hairpins`, `full_pilot` with `motor_assist`, and
  `regression16` with declared seeds.
- **Tests:** `tests/test_fast_motor_tracking_brain12.py`, `tests/test_brake_gates_brain12.py`; the brain-11 caps-record
  test was updated for the two optional fields.
- **Results:**
  - [experiments/brain12_gate_results.json](experiments/brain12_gate_results.json): every candidate's gates, failing
    checks, metrics and ranking;
  - [experiments/brain12_development.json](experiments/brain12_development.json);
  - [experiments/brain12_identity.json](experiments/brain12_identity.json).
- **Raw reports:** session scratchpad `m5/brain12/cands/` (candidates) and `refs/` (references).

Reproduce a score (CPU, one process, 2 threads):

```powershell
.venv/Scripts/python.exe -m haltere.train.brake_gates C:/DEV/Haltere/runs/fast-brain-12-b-cw26d3/candidate.pt --gates configs/brain12_gates.json --out REPORT.json
```
