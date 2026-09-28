# brain-11: a braking brain that follows caps in turns and keeps height when it accelerates (surrogate) — 2026-09-27

Branch `m4b-brain11` (from `m4`). Offline only: no Liftoff, no pad, CPU at 2 threads. Everything new is off
by default.

## Round-4b pilot branches: not merged

The task was to merge `m4b-contact` (contact support in descent view v2, wall pilot v5) and `m4b-assist`
(motor assist v1) only if they were flight ready. Neither is:

- `m4b-contact` fails its frozen gates B-Quiet (wall pilot v5) and the contact audit's video false-positive check.
- `m4b-assist` fails 4 of its 16 frozen gates.

So neither branch is merged. The DAgger rollouts use the **m4 deployed pilot**: `--obstacle-stack on --descent-view on`
for the brain contract, which is lag turn v2, gap pilot v5, wall pilot v4 (brain stopping model), vertical guard v3 and
descent view v1. The motor assist is not used.

## What the round-4 live flights showed, replayed (development cases)

The two brain failures of round 4 were replayed in the surrogate from their logged states, with the logged request
(all three axes) and the logged yaw stick (`brake_gates.r4_window_tests`). Having been looked at, these two logs are
**development cases**, not held-out evidence.

| Window | Live | Logged commands, open loop | Flown brain, replayed | FastMotorPD |
|---|---|---|---|---|
| `minus-brain10b-r4-02` from 20.2 s (hairpin: 75 deg capped turn, 3.6-4.0 m/s caps, guard climb): mean speed over the request from +0.5 s | +1.30 m/s | +1.25 | fast-brain-10b +1.23 (end 4.85 m/s, live 4.93) | -0.10 |
| `minus-brain09b-r4-01` from 22.9 s (turn-first release at 0.8 m, accelerating to 4.9 m/s): lowest height before the floor impact | 0.06 m | 0.05 | brain-09b 0.04 (1.41 m lost over 1.5 s) | 0.70 (0.23 m lost) |

**The surrogate reproduces both live failures** (sim-to-real check), so the gates built on these windows (G9) carry
full weight. What else the replays show:

- **fast-brain-10b's hairpin miss is the capped turn, not the guard climb.** With the vertical request replaced by the
  pilot's own (no climb) it still overshoots by 1.00 m/s, and level by 1.17. With the request direction held (no turn)
  it brakes to -0.49 m/s below the request by the wall.
- **Scripted capped turns** (45/90 deg at 1 rad/s to 3-4 m/s): fast-brain-10b over-brakes in level turns (1-3 s after
  the cap: 0.5-1.7 m/s below it) and does not brake while a 1 m/s climb is requested (up to +1.29 m/s over it, 1.7 s to
  get within 0.5). brain-09b over-brakes in level turns by up to 2.1 m/s. The PD and the label teacher, even with
  80 ms of added delay, track all of them within 0.2 m/s.
- **Hard acceleration from low speed**: brain-09b loses 0.37-0.61 m in the scripted tests and brain-08 up to 2.5 m;
  fast-brain-10b keeps height (at most 0.25 m) but does not accelerate at 60 deg off its heading (below 2.7 m/s after
  1.5 s). The label teacher with 80 ms of delay loses at most 0.38 m; the PD with 80 ms of delay loses up to 1.0 m, so
  this is latency-sensitive and the teacher's stronger vertical loop matters.

## Gates (frozen before any brain-11 candidate existed)

`configs/brain11_gates.json` version 1, sha256 `78a528ab8501...`, frozen 2026-09-27 19:58:40 (commit `4ab221b`). It
keeps brain-10's tests and gates G1-G8 verbatim. G3 v1 is kept and reported, but not primary. The round-4 audit allows a
G3 v2 only for candidates scored after its freeze, and G3v2 replaces G3 v1 here.

| Gate | What it asks | Why |
|---|---|---|
| G1, G2, G5-G8 | brain-10's (cap steps, sustained speeds, in-course caps, regressions, smoothness <= 1.15 x brain-08) | unchanged |
| G3v2 | G3's 9 logged Minus Two windows: the speed at +1 s lies within 0.5 m/s of the band between FastMotorPD and FastMotorPD with 80 ms of added delay | The PD with +80 ms (the brain's measured lag) is up to 0.555 m/s off the PD in these windows, so G3 v1's 0.5 m/s is below any brain's latency floor. A deviation away from both (over-braking where the delayed PD is faster) still fails. fast-brain-10b would fail it too |
| G9 | the two round-4 live failure windows, replayed (development cases): hairpin speed excess <= 0.5 m/s; stop-and-accelerate height loss <= 0.4 m and >= 3.5 m/s at 1.5 s; plant validity | the live failures themselves |
| G10 | scripted caps during 45/90 deg turns and 1 m/s climbs, from a hover and from 6 logged cruise states: within X + 0.5 in <= 1 s, excess over 0.5-1.5 s <= 0.5, settled excess over 1-3 s in [-0.6, 0.3] | held-out form of the hairpin |
| G11 | scripted hard acceleration from a 0.5 m/s creep or a stop, at 0/30/60 deg off the heading: height loss <= 0.4 m and 4 m/s within 1.5 s | held-out form of brain-09b's sink |
| G12 | G5 with rays up to 45 deg off the track and climbs on half the caps (held-out cap seed 9002): excess <= 0.4, >= 15 finished, <= 1 crash | caps in turns and climbs in course |
| G13 | the deployed pilot in the descent surrogate (flat, steep 3000-3007, hill 6000-6011): no worse than fast-brain-10b under it in finishes, crashes, terrain contacts (12), high passes (29) and terrain time (+3%); chatter <= 1.15 x brain-08 under it | the stack the brain will fly |

Scripted G10/G11 hold the yaw stick at 0, so their turns and accelerations are also body-frame lateral requests.

Before the freeze, the reference controllers were measured on the new tests. No brain-11 candidate existed then. The PD
passes G3v2, G9, G10 and G11, and the G12 cap excess. brain-08, brain-09b and fast-brain-10b fail G3v2, G9, G10 and G11.
This tree reproduces the round-4 integration surrogate for fast-brain-10b under the full pilot exactly. Details are in
the file's `baselines` and `diagnostics`.

**Identity.** With the new options off, a collection, a refit and the brain-10 gate parts are bit-identical to `m4`.
That covers the collected data, the labels, the brake mask, the refitted readout, and the hover and swaps rows. See
[experiments/brain11_identity.json](experiments/brain11_identity.json); the check was re-run on the final code.

## Recipe (`haltere/train/fast_motor_tracking.py`, all off by default)

brain-10b's recipe is kept: parent `motor-brain-10-tracking-05`, brain-08's contract, 5 DAgger rounds x 10 courses x
110 s, synthetic caps on 60% of the drones, slow legs 30%, balance-speed, sink weight 10, steep 0.4, the label teacher
(attitude gain 4) and turn relief 0.5. Only readout rows 0-2 and their biases change. New options:

- `--pilot deployed [--pilot-share F]`: DAgger rollouts fly the deployed pilot (`haltere/train/deployed_pilot.py`,
  loaded through the runner's own hash-checked loaders). The surrogate has no looming or gap samples, so the lag-aware
  turns, the descent view, and turn-first on a synthetic cap act. With F < 1, only a seeded share of each round's drones
  fly it; the rest fly the default pilot that G2/G6/G8 use.
- Climbing synthetic caps (`configs/brain11_caps.json`): rays up to 45 deg off the track, targets down to 0.3 m/s, and
  a 0.5-1.5 m/s climb for 0.5-2 s on 40% of the caps.
- `--brake-level 1.6 --brake-turn-deg 90`: the brake weights also select climbing brakes and capped turns, where the
  speed is over the request's magnitude.
- `--hill-share 0.2`: hill courses (long descents) in the DAgger rounds.
- `--yaw-holds R`: training only. The yaw stick is held at 0 for 0.5-2 s at R per minute, so requests turn in the body
  frame.
- `--cruise-weight W`: refit only. It up-weights aligned cruise samples at 4.8 m/s or more that are not over-speed.

## Candidates

There were three DAgger collections (CPU, 2 threads, about 40 minutes each) and two refits of collection b's saved data,
so five candidates in all under `C:/DEV/Haltere/runs/`. Every candidate was scored once on the frozen gates.

| Candidate | Collection | Refit |
|---|---|---|
| `fast-brain-11-a` | deployed pilot on every drone; teacher attitude 4 / vertical 5; lead 0.06 s | brake weight 5, level 1.6, turn 90; ridge 0.003, smooth 100, rows 0.3/1/1 |
| `fast-brain-11-b` | deployed pilot on half the drones; yaw holds 4/min; teacher vertical 7; lead 0.08 s | as a |
| `fast-brain-11-c` | as b, with yaw holds 12/min | as a |
| `fast-brain-11-b-cw16` | b's data | as a, plus cruise weight 1.6 |
| `fast-brain-11-b-cw13` | b's data | as a, plus cruise weight 1.3 |

All five also use brain-11 caps, hill share 0.2 and turn relief 0.5. b-e's pilot share, yaw holds and cruise weight were
added after screening a. Between the candidates, 8 development refits were made in the scratchpad. They were checked on
development tests only (`dev.py`, `dev_cruise.py`: other angles, heights and course seeds), never on the gates. The two
promoted refits reproduce their development refits bit for bit. See
[experiments/brain11_development.json](experiments/brain11_development.json).

### Results (frozen gates v1, sha256 `78a528ab8501...`)

**No candidate passes every primary gate, so under the frozen rule none is selected.** Ranked by gates passed, then
16-course chatter, the best is **`fast-brain-11-b-cw13`**. It passes 8 of 12 primary gates. sha256
`44cca3c4fc3e40f8ab1cc9cb5e94337edf0229dc10d1611e271e640e5627c11d`.

| candidate | G1 | G2 | G3v2 | G5 | G6 | G7 | G8 | G9 | G10 | G11 | G12 | G13 | passed | sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fast-brain-11-b-cw13` | pass | pass | **FAIL** | pass | **FAIL** | pass | pass | pass | **FAIL** | **FAIL** | pass | pass | 8/12 | `44cca3c4fc3e...` |
| `fast-brain-11-b` | pass | **FAIL** | **FAIL** | pass | pass | pass | pass | pass | **FAIL** | **FAIL** | pass | **FAIL** | 7/12 | `723d46b3019a...` |
| `fast-brain-11-c` | pass | **FAIL** | **FAIL** | pass | **FAIL** | pass | pass | pass | **FAIL** | **FAIL** | pass | **FAIL** | 6/12 | `f5889ddbf8e7...` |
| `fast-brain-11-b-cw16` | **FAIL** | **FAIL** | **FAIL** | pass | **FAIL** | pass | pass | pass | **FAIL** | **FAIL** | pass | pass | 6/12 | `74658a4d9532...` |
| `fast-brain-11-a` | pass | **FAIL** | **FAIL** | pass | **FAIL** | pass | **FAIL** | pass | **FAIL** | **FAIL** | **FAIL** | **FAIL** | 4/12 | `e3721542b77a...` |

G3 v1 (reported only) fails for every candidate, with worst windows of 0.72-1.0 m/s.

| Metric | fast-brain-11-b-cw13 | fast-brain-11-b | fast-brain-10b | Limit |
|---|---|---|---|---|
| G9 hairpin: speed over the capped request (live 1.30) | **+0.38** | +0.36 | +1.23 | <= 0.5 |
| G9 stop-and-accelerate: height lost in 1.5 s (live: 0.74 to the floor) | **0.05 m** | 0.05 | 0.48 | <= 0.4 |
| G11 height lost, hard acceleration (worst of 6) | 0.06 m | 0.06 | 0.25 | <= 0.4 |
| G11 4 m/s reached within 1.5 s | 4 of 6 (not at 60 deg) | 4 of 6 | 4 of 6 | 6 of 6 |
| G10 capped turns: settled speed minus the cap | -1.75..+0.22 | -1.79..+0.15 | -1.65..+0.29 | -0.6..+0.3 |
| G5 / G12 in-course cap excess | 0.29 / 0.25 | 0.26 / 0.25 | 0.35 / 0.40 | <= 0.4 |
| G2 cruise excess at 3.5 / 4.5 / 6 m/s | +0.27 / +0.17 / -0.55 | +0.22 / +0.07 / -0.68 | +0.05 / +0.10 / -0.44 | +-0.3; -0.6..+0.3 |
| G6 16-course finished / crashed | 16 / 0 | 16 / 0 | 16 / 0 | >= 15 / <= 1 |
| G6 post-switch lateral error at 1 s (40-75 deg) | 3.14 m | <= 3.08 | 3.04 | <= 3.08 |
| Stick chatter, 16 / 8 courses / caps | 0.00202 / 0.00161 / 0.00184 | 0.00197 / 0.00159 / 0.00182 | 0.00226 / 0.00187 / 0.00179 | brain-08 x 1.15: 0.00343 / 0.00561 / 0.00673 |
| G8 steep sink shortfall | 0.057 | 0.056 | 0.127 | <= 0.25 |
| G13 deployed pilot: finished, contacts, high passes, terrain time | 28/28, **11**, **12**, 61.9 s | 28/28, 14, 12, 62.5 s | 28/28, 12, 29, 65.3 s | >= 28, <= 12, <= 29, <= 67.2 |
| G13 chatter under the deployed pilot | 0.00239 | 0.00234 | 0.00289 | <= 0.00429 |

What the round asked for, in the surrogate:

- **Follow governor caps.** Done where the capped request points ahead or to the right. The replayed live hairpin
  window drops from +1.23 to +0.38 m/s over the request. In-course caps are +0.29 m/s, and +0.25 with climbs and rays
  up to 45 deg off the track.
- **Keep height while accelerating hard from low speed.** Done. brain-09b's live window loses 0.05 m instead of 1.41,
  and the scripted accelerations lose at most 0.06 m. The label teacher's stronger vertical loop (gain 7) and a
  0.08 s lead kept it: fast-brain-11-a (vertical 5, lead 0.06) crashed twice in the 16-course gate from exactly this
  sink.
- **Stay smooth.** Done: 16-course chatter 0.00202, 32% below brain-08.
- **No worse on the 16-course gate or steep sink.** Done: 16/16, sink shortfall 0.057.
- **Under the deployed pilot** (a brain distilled with the view rule in its rollouts): 11 terrain contacts against
  fast-brain-10b's 12, 12 passes more than 1.5 m above a checkpoint against 29, and 5% faster. On Straw's downhill this
  is the direction the user asked for, though the surrogate's terrain is scoring-only.

### Why every candidate fails G3v2, G10 and G11: left body-frame requests

These failures, and very likely fast-brain-11-b-cw13's 0.054 m miss on G6's post-switch lateral error, come mostly from
one fault. **The braking brains over-brake when the request points left of their heading.**

- **Scripted, yaw stick at 0.** A 45 deg change at 5.5 m/s: FastMotorPD rolls +-0.50 symmetrically and holds
  5.4 m/s either way. fast-brain-11-b rolls only -0.15 to the left, then reverses and pitches back, and slows to
  2.4 m/s. To the right it holds 5.2 m/s.
- **G10 and G11.** The failing G10 cases are the 45 and 90 deg turns, which are left turns, down to 1.8 m/s below the
  cap. The failing G11 cases are the 60 deg accelerations, also to the left.
- **G3's logged windows.** The over-braking windows are exactly those where the logged request pointed 13-30 deg left
  of the heading: gapon-01 at 11.6 and 16.1 s, gapon-02 at 11.9 and 16.2 s. There fast-brain-11-b-cw13 is 0.33-0.78 m/s
  under the band, and three of those four windows fail. Its fourth G3v2 miss is the other way: gapon-02 at 6.7 s, where
  it is 0.02 m/s too fast.
- **Origin.** fast-brain-10b and brain-09b show the same fault. brain-08, distilled without braking data, flies the
  left change at 5-6.6 m/s. So it comes from the braking recipe (synthetic caps and the DAgger data around them). It
  is not the brake weights: a refit without them keeps it. It is not a lack of lateral data either: 12 yaw holds per
  minute (fast-brain-11-c) do not remove it.
- **The live stack mostly hides it.** The pilot yaws toward the checkpoint at up to 3 rad/s, so body-frame requests
  are only transiently lateral. But the G3 windows show that a 13-30 deg lag is enough to cost 0.8 m/s in left arch
  turns.

## Weights audit

Every candidate changed only readout rows 0-2 (throttle, roll, pitch) and their biases. The yaw row and the other 38
tensors (connectome wiring, weights, transmitter signs, time constants, encoders) are bit-identical to the parent
`motor-brain-10-tracking-05` (sha256 `64444a62...`), and so to brain-08, brain-09b and fast-brain-10b. The audits are in
`runs/fast-brain-11-*/weights-audit.json`.

For fast-brain-11-b-cw13 the largest weight changes per row are throttle 0.276, roll 0.067 and pitch 0.290
(fast-brain-10b: 0.154 / 0.068 / 0.262).

## Limits and next

- **Surrogate only.** IdentifiedSim, synthetic courses and logged Minus Two states. Nothing was flown.
- **Not selected.** Flying fast-brain-11-b-cw13 would be a disclosed development deviation, as brain-09b's and
  fast-brain-10b's flights were. Its gates predict:
  - it brakes at the Minus hairpin and keeps height out of turn-first;
  - it may be slow in left capped turns, where the yaw lags the request.

  Use the m4 stack with `--descent-view on`; the round-4b pilot branches are not merged.
- **G9 is not held-out.** The two r4 windows were looked at before the freeze and screened during development. G10-G13
  are held-out in parameters and seeds, but were designed from the same diagnosis.
- **Post-freeze code.** Three training options were added after the freeze and after scoring fast-brain-11-a:
  `--yaw-holds`, `--pilot-share` and `--cruise-weight`. They are training code, not rules or gates, and the gate file is
  unchanged.
  - A logging-only slip in that edit dropped the `synthetic_cap_events` count from rollout rows while four candidates
    were scored. No gate reads it. It is fixed, and the default-off identity was re-checked on the final code.
- **Next.**
  - Find the left asymmetry in the braking data: per-side fit residuals, and caps and labels by side.
  - Then consider a mirrored-course collection, or a pilot rule that yaws before a lateral capped request.
  - The contact-support and motor-assist branches need their failed gates fixed before a stack with them can be
    frozen for graduation.
