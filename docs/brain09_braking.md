# brain-09: teaching the brain motors to brake when the pilot asks (surrogate) — 2026-09-26

Branch `m3-brain09` (from `m2-hairpin`). Offline only: no Liftoff flight, no pad,
CPU at 2 threads. Everything new is off by default.

## Why

The obstacle stack can only steer around or stop before an obstacle if the motors
follow a slower request. brain-08 does not: its pitch response to a 3-4.5 m/s
request is 1-6% of the PD teacher's, in Liftoff and in the surrogate
([round-2 diagnosis](flight_cards/2026-09-26_obstacles_m2.md)). On Minus Two it
kept flying 5.2-6 m/s under the looming governor's 3.6-4.4 m/s caps, and 5.1 m/s
with `--assist-speed 3.5`. Its distillation data had almost no sustained caps
(0.13% of samples in the live regime).

## Recipe (`haltere/train/fast_motor_tracking.py`, all off by default)

brain-08's contract and parent are kept: `motor-brain-10-tracking-05`, 6 m/s
nominal, scaled speed 2.4, vertical goal 0.4 s, retina blanked, 5 DAgger rounds x
10 courses x 110 s, rest 15 s, steep 0.4, balance-speed, sink weight 10. Only the
throttle/roll/pitch readout rows and biases change.

- `--synthetic-caps 0.6`: in every collection round (the PD round and all brain
  rounds) 60% of the drones get a `SyntheticCaps` object in the place of the
  pilot's looming governor (`pilot.clearance`). About 8 caps a minute while the
  pilot is in `cue`/`side`/`coast` at 3 m/s or more; the cap is U(1, 4) m/s or
  0.2-0.7 x the speed, along the travel direction +-15 deg; it falls at the TTC
  governor's 8 m/s^2, holds U(0.5, 3) s and releases at 3 m/s^2. The pilot's own
  cap path (brake_slew 15 m/s^2, no taper) shapes the request and the PD labels
  the braking. Training data only; nothing at runtime reads it.
- `--slow-legs 0.3 --slow-leg-speed 2.5 4.5`: 3 of the 10 courses of each round
  are flown at a sustained pilot speed of U(2.5, 4.5) m/s (at or below nominal).
- `--brake-weight 5`: samples that are aligned (< 20 deg), level (|vz request| <
  0.5), at 3 m/s or more and over-speed along the track by 0.8 m/s or more count 5x.
- `rollout(record_brake=True)` adds braking metrics whose classes come from the
  request side (cap events, cruise ticks, slow requests split by cap/aligned/turn),
  never from the excess itself.

Default rollouts are bit-identical to the previous module (checked on brain-08).

## Gates (frozen before any brain-09 data)

`configs/brain09_gates.json` version 1, sha256 `21ba2fcb2e32...`, frozen
2026-09-26 12:55, run by `python -m haltere.train.brake_gates CHECKPOINT|pd`. They
follow the recipe with the verifier's corrections (request-side cap metrics, a
height loop instead of the below-ground 15 s swaps, the rollout(S) assist-speed
test, regression baselines from the same gate on this tree, a one-sided speed
rule). brain-08 and FastMotorPD were measured with the same runner before the
thresholds were written.

| Gate | What it asks |
|---|---|
| G1 cap step | a governor-like cap to 1/2/3/3.5/4 m/s: within X + 0.5 in <= 1 s and settled excess <= 0.3 m/s, from a hover and from 6 logged Minus Two cruise states |
| G2 sustained | 2/3/3.5/4.5 m/s requests: \|excess\| <= 0.3 from a hover and from 7 logged states, the two within 0.3 of each other; rollout(S) cruise excess \|.\| <= 0.3 at S = 3.5 and 4.5, -0.6..+0.3 at 6, <= 1 crash of 8 |
| G3 live swaps | the logged request replayed from 9 logged Minus Two states: speed at +1 s within 0.5 m/s of FastMotorPD in the same plant; plant validity; 3.5 m/s from slow35/loom settles <= 3.9 |
| G5 in-course caps | synthetic caps on all 16 standard courses, held-out cap seed: excess along the cap ray over the request <= 0.4 m/s |
| G6 regressions | 16-course harness >= 15 finishes, <= 1 crash, chatter <= 0.0035, mean speed >= 0.95 x brain-08, steep sink shortfall <= 0.3, no descending pass > 1.5 m high, Straw-like 45-75 deg switches (speed_min, regain, stick_after, lateral) near brain-08; 8-course velocity error <= 1.05 x brain-08; weights audit |

Reference verdicts: brain-08 passes G6 only (cap excess +1.20 m/s, rollout(S)
+1.30/+0.75/-0.47); the PD teacher passes G1-G5 and fails the brain regressions
(chatter 0.0057, 14/16 finished).

## Runs

Two DAgger collections, then refits of their saved data (`--resolve`, no new
rollouts). Runs are under the `m3-brain09` worktree's `runs/` (git-ignored).

- `fast-brain-09-caps`: the recipe as written (DAgger refits at ridge 0.3, smooth
  30). It brakes, but it cruises 1.36 m/s under a 6 m/s request (brain-08 0.48):
  the readout pulls every fast state toward braking. The recipe's refit grid
  (ridge 0.03-0.3 x smooth 10-30) did not fix that; lower ridge (0.01-0.001) with
  more smoothing (100-300) cut it to 0.4-0.6 m/s, and a lower brake weight (1-2)
  gave the braking back.
- `fast-brain-09b-caps`: the same data recipe with the DAgger refits at ridge
  0.003 / smooth 100 (the best refit of the first run), so the brain rounds visit
  the states of a brain that brakes; then ridge 0.001-0.01 x smooth 100-300 and
  brake weight 10 refits.

## Results (frozen gates v1)

**No candidate passes every primary gate, so none is selected as brain-09 for
flight** (selection rule in the gate file). All 22 candidates fail G3. Ranked by
primary gates passed, the best is `fast-brain-09b-caps-r0001m100` (ridge 0.001,
smooth 100, brake weight 5): G1, G2 and G5 pass; G3 and G6 fail.

| | brain-08 | 09b r0.001/s100 | PD teacher |
|---|---:|---:|---:|
| Cap from hover to 1/2/3/3.5/4 m/s, settled excess | +0.68 / +0.91 / +1.19 / +0.97 / +0.62 | -0.33 / -0.09 / -0.04 / +0.01 / -0.02 | -0.03 to -0.08 |
| Within X + 0.5 m/s after the cap (3 / 3.5 m/s) | never / never | 0.59 / 0.56 s | 0.35 / 0.32 s |
| Cap from logged Minus cruise, 3.5 m/s: settled excess | +1.79 | +0.15 | -0.07 |
| slow35 state, 3.5 m/s request, steady speed | 5.29 | 3.57 | 3.44 |
| rollout(S) cruise excess at 3.5 / 4.5 / 6 m/s | +1.30 / +0.75 / -0.47 | +0.16 / +0.18 / -0.54 | -0.04 / -0.06 / -0.10 |
| In-course synthetic caps: excess over the request along the cap | +1.20 | +0.19 | -0.19 |
| Logged-request swaps, speed at +1 s minus PD | +0.42 to +1.68 | -1.56 to +0.72 | 0 |
| 16-course harness: finished / crashed | 15 / 1 | 16 / 0 | 14 / 2 |
| mean speed, stick chatter | 4.80, 0.0030 | 4.93, **0.0051** | 5.12, 0.0057 |
| steep sink shortfall; descending passes > 1.5 m high | 0.16; 0/10 | 0.14; **1/10** | -0.13; 1/7 |
| 45-75 deg switches: speed_min, regain, stick_after | 2.97, 2.00 s, 0.0113 | 3.32, 1.19 s, 0.0123 | 3.46, 1.24 s, 0.0155 |
| 8-course velocity error | 0.947 | 0.755 | 0.427 |

What fails:

- **G3, turning windows.** In the four Minus Two arch-turn windows (the request
  turns 24-54 deg in 1 s under a 3.7-4.1 m/s governor cap, with a small climb) the
  best candidate is 1.47-1.56 m/s slower than the PD at +1 s, and so is every
  candidate that brakes (0.6-2.5 m/s); on the straight gapon-02 8.2 s window it is
  still 0.72 m/s slow to decelerate. The only refits that stay within 0.5 m/s in
  the turns (brake weight 1-2) are again up to 1.35-1.56 m/s too fast on the
  straight windows, as brain-08 was. The static-gain diagnostic agrees: at the
  logged over-speed states the best candidate's pitch response to the request is 16-29 deg/s per m/s between 4 and 6
  m/s (brain-08 1.4-6, PD 73-132), and in the turning windows it still pitches back
  at 13-44 deg/s when asked for 6-7 m/s, where the PD accelerates. A linear
  readout of the compressed features cannot separate "the request turned" from
  "the request dropped" well enough.
- **G6, smoothness.** The refits that brake hard enough chatter above 0.0035
  (0.0038-0.0054 at smooth 100); smooth 300 brings chatter to 0.0033-0.0038, but
  three of the four such refits crash on 3 of 16 courses and their 6 m/s cruise
  falls 0.54-1.10 m/s short. The PD teacher itself chatters 0.0057, as the
  verifier warned.
- **G2 at 6 m/s.** Most fully gated candidates cruise 0.65-1.10 m/s under a 6
  m/s request (the gate allows 0.6); inside are the 09b ridge 0.001 / smooth 100
  refit (-0.54) and the first run's ridge 0.003 refits at smooth 100 (-0.39) and
  300 (-0.54).

Every checkpoint changes only readout rows 0-2 and their biases (the gate audit;
`runs/fast-brain-09b-caps-r0001m100/weights-audit.json` for the best-ranked one:
yaw row, connectome wiring and signs identical to the parent and to brain-08).
Its readout moved much more than brain-08's (max weight change 0.67 against 0.026
at ridge 0.3), so it relies on low-variance feature directions.

Every candidate, in rank order ("braking only": the quick G1-G3 tests were run
first and had already failed, at least G3, so the slower parts were not run):

| candidate | data | ridge / smooth / brake w. | G1 | G2 | G3 | G5 | G6 | cap excess | rollout(S) 3.5 / 4.5 / 6 | 16-course fin/crash, chatter | parts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| brain08 | reference | - | FAIL | FAIL | FAIL | FAIL | pass | +1.20 | +1.30 / +0.75 / -0.47 | 15/1, 0.0030 | all |
| fast_pd | reference | - | pass | pass | pass | pass | FAIL | -0.19 | -0.04 / -0.06 / -0.10 | 14/2, 0.0057 | all |
| fast-brain-09b-caps-r0001m100 | 09b-caps | 0.001 / 100 / 5 | pass | pass | FAIL | pass | FAIL | +0.19 | +0.16 / +0.18 / -0.54 | 16/0, 0.0051 | all |
| fast-brain-09-caps-r0003m100b10 | 09-caps | 0.003 / 100 / 10 | pass | FAIL | FAIL | pass | FAIL | +0.13 | +0.08 / +0.01 / -0.70 | 15/1, 0.0039 | all |
| fast-brain-09b-caps-r0001m300 | 09b-caps | 0.001 / 300 / 5 | pass | FAIL | FAIL | pass | FAIL | +0.14 | +0.13 / +0.10 / -0.66 | 13/3, 0.0038 | all |
| fast-brain-09b-caps-r0003m100b10 | 09b-caps | 0.003 / 100 / 10 | pass | FAIL | FAIL | pass | FAIL | +0.18 | +0.01 / -0.15 / -0.97 | 16/0, 0.0054 | all |
| fast-brain-09b-caps-r0003m300 | 09b-caps | 0.003 / 300 / 5 | pass | FAIL | FAIL | pass | FAIL | +0.26 | +0.15 / +0.06 / -0.76 | 13/3, 0.0035 | all |
| fast-brain-09-caps-r0003m300 | 09-caps | 0.003 / 300 / 5 | pass | FAIL | FAIL | pass | FAIL | +0.19 | +0.23 / +0.20 / -0.54 | 13/3, 0.0036 | all |
| fast-brain-09b-caps | 09b-caps | 0.003 / 100 / 5 | pass | FAIL | FAIL | pass | FAIL | +0.26 | +0.20 / +0.14 / -0.65 | 16/0, 0.0049 | all |
| fast-brain-09b-caps-r001m100 | 09b-caps | 0.01 / 100 / 5 | pass | FAIL | FAIL | pass | FAIL | +0.36 | +0.20 / +0.05 / -0.80 | 16/0, 0.0046 | all |
| fast-brain-09b-caps-r0003m300b10 | 09b-caps | 0.003 / 300 / 10 | pass | FAIL | FAIL | pass | FAIL | +0.09 | -0.06 / -0.25 / -1.10 | 15/1, 0.0033 | all |
| fast-brain-09-caps-r0003m100 | 09-caps | 0.003 / 100 / 5 | FAIL | FAIL | FAIL | pass | FAIL | +0.26 | +0.28 / +0.31 / -0.39 | 15/1, 0.0038 | all |
| fast-brain-09b-caps-r001m300 | 09b-caps | 0.01 / 300 / 5 | FAIL | not run | FAIL | not run | not run | not run | not run | not run | braking only |
| fast-brain-09-caps-r0001m300 | 09-caps | 0.001 / 300 / 5 | FAIL | FAIL | FAIL | not run | not run | not run | not run | not run | braking only |
| fast-brain-09-caps-r001m100 | 09-caps | 0.01 / 100 / 5 | FAIL | FAIL | FAIL | not run | not run | not run | not run | not run | braking only |
| fast-brain-09-caps-r001m30 | 09-caps | 0.01 / 30 / 5 | FAIL | FAIL | FAIL | not run | not run | not run | not run | not run | braking only |
| fast-brain-09-caps-r003m10 | 09-caps | 0.03 / 10 / 5 | FAIL | FAIL | FAIL | not run | not run | not run | not run | not run | braking only |
| fast-brain-09-caps-r003m30 | 09-caps | 0.03 / 30 / 5 | FAIL | FAIL | FAIL | not run | not run | not run | not run | not run | braking only |
| fast-brain-09-caps-r01m10 | 09-caps | 0.1 / 10 / 5 | FAIL | FAIL | FAIL | not run | not run | not run | not run | not run | braking only |
| fast-brain-09-caps-r01m30 | 09-caps | 0.1 / 30 / 5 | FAIL | FAIL | FAIL | not run | not run | not run | not run | not run | braking only |
| fast-brain-09-caps-r03m10 | 09-caps | 0.3 / 10 / 5 | FAIL | FAIL | FAIL | not run | not run | not run | not run | not run | braking only |
| fast-brain-09-caps | 09-caps | 0.3 / 30 / 5 | FAIL | FAIL | FAIL | not run | not run | not run | not run | not run | braking only |
| fast-brain-09-caps-r003m30b2 | 09-caps | 0.03 / 30 / 2 | FAIL | FAIL | FAIL | not run | not run | not run | not run | not run | braking only |
| fast-brain-09-caps-r003m30b1 | 09-caps | 0.03 / 30 / 1 | FAIL | FAIL | FAIL | not run | not run | not run | not run | not run | braking only |

All gate reports and the ranking: [experiments/brain09_gate_results.json](experiments/brain09_gate_results.json).

## Limits and next

- Surrogate only (IdentifiedSim, synthetic courses, logged states of six brain-08
  flights on one course, Minus Two). Nothing here was flown; the gates are development checks.
- The best-ranked candidate brakes for caps and slow requests almost like the
  teacher on straight legs, but it is not selected: it over-brakes in turns under
  a cap and chatters more than brain-08. Flying it anyway would be a deviation
  from the frozen gates.
- The recipe's next lever for G3 is a declared goal-contract change for the
  along-track component (the connectome compresses 3-7 m/s requests 5-8x); that
  needs its own contract, gates and runtime support. More DAgger rounds or finer
  refit grids on this contract did not move G3.
