# brain-10: a braking brain motor as smooth as brain-08 (surrogate) — 2026-09-27

Branch `m4-brain10` (from `m3-brain09`, with `m2-vertical` merged for the pilot code the
flights use). Offline only: no Liftoff, no pad, CPU at 2 threads. Everything new is off
by default. The view-keeping descent (`m4-descent`) failed its frozen gates and was not
flight ready, so it is **not** merged: brain-10 is distilled under the current pilot.

## Why

brain-09b (`fast-brain-09b-caps-r0001m100`) brakes for governor caps on straight legs, in the
surrogate and live on Minus Two, but its sticks chatter (0.0051 per tick in the 16-course
harness, 0.0056 live; brain-08 0.0030 and 0.0031) and it over-brakes in capped turns
(1.5 m/s under the PD in the logged arch-turn windows). The graduation bar needs one frozen
stack that brakes and flies smoothly.

## Diagnosis (existing traces and teacher-only runs; no candidate)

- **The chatter is a pitch limit cycle in cruise, not steering.** In the 16-course traces
  brain-09b's extra chatter is almost all pitch (0.0033 against brain-08's 0.0016; roll
  0.0018 against 0.0014), below 2 Hz, and in steady 6 m/s cruise (0.0042 per tick at
  requests of 5 m/s or more, against 0.0021 for brain-08 and 0.0027 for the PD). The time
  series shows a sustained ~0.8 Hz pitch oscillation (stick about +-0.1) with the speed
  hunting between 5.3 and 5.75 m/s under a constant 6 m/s request, where the PD holds its
  pitch still.
- **The brain is a delayed, weaker copy of its teacher.** On its own trajectory the brain's
  roll/pitch output lags the teacher's label by 7-9 ticks (70-90 ms) at 0.5-0.6 of its
  amplitude (brain-08 and brain-09b alike).
- **The PD teacher does not tolerate that latency.** FastMotorPD with extra command delay
  (8 capped steep courses): chatter 0.0072 at +0, 0.0095 at +40 ms, 0.0103 and a crash at
  +60 ms, 0.0134 and 6 of 8 crashed at +100 ms. Its attitude loop (gain 8) is too fast for a
  student with 70-90 ms of latency; with attitude gain 4 the chatter at +60 ms halves
  (0.0054), and attitude 3 / velocity 2 gives 0.0041 at +100 ms.
- **The capped-turn over-braking is the same latency.** In the brain-09 G3 windows the PD
  delayed by 80 ms is already 0.55 m/s off the PD at +1 s, and teachers with a slower
  attitude loop are 0.9-1.2 m/s slow: a later brake bottoms out nearer +1 s. G3 compares a
  speed at one instant of a transient, so it strongly penalises latency.
- **Live caps last longer than brain-09's synthetic ones.** In the logged flights the
  governor's caps last 1.5-17 s (median 7.3 s) at 0.4-0.7 of the speed and continue
  through 13-136 deg turns; brain-09 held them 0.5-3 s.
- **GPU vs CPU.** One identical collection iteration (10 drones, 20 s, then a refit):
  39.5 s on the CPU at 2 threads, 36.8 s on the RTX 4090 through the thermal wrapper (19.1
  vs 18.3 ms per tick; refit 1.2 vs 0.1 s). A profile of the tick: the connectome step is
  about a third; the per-drone pilot, the teacher's rate-curve inversion and the simulator
  (Python, small tensors) are the rest and stay on the CPU either way. Training therefore
  stays on the CPU, which also avoids the GPU lock and 5-minute chunking.

## Recipe (`haltere/train/fast_motor_tracking.py`, all off by default)

brain-09's data recipe is kept (brain-08's contract and parent, 5 DAgger rounds x 10
courses x 110 s, synthetic caps on 60% of the drones, slow legs 30%, brake weight 5,
balance-speed, sink weight 10, steep 0.4); only the throttle/roll/pitch readout rows and
biases change. New:

- `--teacher-gains attitude_gain=4 ...`: the label teacher (which also flies the first
  round) is FastMotorPD with a slower attitude loop the student's latency can follow. The
  deployed FastMotorPD and the pilot are unchanged.
- `--turn-relief F` (`capped_turn_relief`): in a capped turn (request magnitude 0.3-1 m/s
  or more below the flown speed, 0-45 deg off the velocity, fading to 0 at 90 deg) the
  label teacher sees F x the along-track braking that comes only from the turn geometry
  removed; aligned caps, hairpins and uncapped turns are unchanged.
- `--caps-config configs/brain10_caps.json`: caps at 6 per minute held 1-5 s (about the
  same capped share as brain-09, each cap longer, like the live governor's).
- `--label-lead S`: each sample's features are paired with the teacher's label S seconds
  later on the same drone, so the readout learns to anticipate its own latency.
- `--smooth-rows T R P`: per-row multipliers of the smoothness penalty (refit only).
- `--sag-weight W`: up-weights samples sinking at least 1 m/s below a request that is not a
  fast descent. It was tried in the development screening only; no candidate uses it,
  because it did not help.

With every new option off, a collection and refit are bit-identical to the module before
brain-10 (`73c4c78`). This was checked on a brain-controlled and a PD-controlled collection:
all metrics, features, labels, requests, velocities, the step Gram and the refitted readout
match. The tests in `tests/test_fast_motor_tracking_teacher.py` pin it.

## Gates (frozen before any brain-10 candidate existed)

`configs/brain10_gates.json` version 1, sha256 `7736f465ae49...`, frozen 2026-09-27
04:27:59 (commit `c51c2af`). The brain-09 v1 tests and gates G1-G6 are copied verbatim;
none was shown ill-posed, so all are kept (the G3 latency finding above is recorded in
the file's diagnostics). New:

| Gate | What it asks |
|---|---|
| G7 smoothness | stick chatter <= 1.15 x brain-08 in the 16-course harness (<= 0.003427), the 8-course evaluation (<= 0.005612) and the in-course caps rollout (<= 0.006727) |
| G8 regressions | 16-course finishes >= brain-08's (15); steep sink shortfall <= 0.25 m/s |

brain-08 was re-measured on this tree before freezing and reproduced its brain-09 baseline
exactly (so the merged pilot's defaults are unchanged). Reference verdicts: brain-08 passes
G6, G7, G8 (3 of 7); the PD teacher passes G1, G2, G3, G5 (4 of 7); brain-09b passes G1, G2,
G5, G8 (4 of 7; G3, G6 and G7 fail).

## Runs

Three DAgger collections (CPU, 2 threads, about 45 minutes each), with refits of their saved
data. Six candidates in all, under `C:/DEV/Haltere/runs/` (git-ignored).

| Collection | Label teacher | Relief | Lead | DAgger refit | Candidates |
|---|---|---|---|---|---|
| `fast-brain-10a` | attitude gain 4 | 0.5 | - | ridge 0.003, smooth 100 | `10a`; refits `10a-t03` (throttle smoothing x0.3), `10a-r00001t03` (ridge 0.0001, throttle x0.3) |
| `fast-brain-10b` | attitude 4, vertical velocity gain 5 | 0.5 | 0.06 s | ridge 0.003, smooth 100, rows 0.3/1/1 | `10b`; refit `10b-r0001` (ridge 0.001) |
| `fast-brain-10c` | attitude 5, vertical 5 | 1.0 | 0.08 s | ridge 0.001, smooth 100, rows 0.3/1/1 | `10c` |

All three use the brain10 caps, brake weight 5, synthetic caps 60% and slow legs 30%.

`--label-lead` and `--sag-weight` were added during the round, after the gates were frozen.
They are training options only; the gates did not change.

The capped-turn relief did not do what it was meant to. It barely changes the teacher in the G3
windows (within 0.01 m/s). 10c, with full relief, a faster teacher and a longer lead, over-braked
in the turn windows more than 10b did.

The refits to promote were chosen by development screening, not by the gates:

- Development courses (steep and flat seeds 5000-5007, disjoint from the gate courses).
- Two scripted diagnostics:
  - Height kept while re-accelerating after a turn at 1.5 m.
  - 18 held-out logged windows from `minus-brain08-vg-01` and `minus-brain09b-vg-01`, as a G3 analogue.

What the diagnostics showed:

- **10a crashed from height loss.** 10a was smooth and braked, but lost 2 of 16 courses in the gate harness. Each time it sank 1-2 m/s below a level request while re-accelerating after a slow turn at low height.
- **The re-acceleration test reproduced it.** Lowest height, starting from 1.5 m, after a 90 deg turn and re-acceleration:

  | Controller | Lowest height |
  |---|---|
  | PD | 1.33 m |
  | brain-08 | 0.61 m |
  | brain-09b | 0.45 m |
  | 10a | 0.43 m |
  | 10a at ridge 0.0001 | below the ground |
  | 10b | 1.34 m |
  | 10b-r0001 | 1.33 m |

- **Sag weight did not help.** Up-weighting sagging samples did not fix the height loss.
- **What did help.** A stronger vertical loop in the label teacher (the delayed teacher's lowest height 1.02 -> 1.14 m) together with the label lead.
- **On the held-out windows every brain stays far from the PD at +1 s.**

  | Controller | Windows within 0.5 m/s (of 18) |
  |---|---|
  | brain-08 | 6 |
  | brain-09b | 4 |
  | 10a | 3 |
  | 10b | 5 |
  | 10b at ridge 0.001 | 7 |

  Every controller's worst window was 1.5-2.6 m/s off.

## Results (frozen gates v1, sha256 `7736f465ae49...`)

**No candidate passes every primary gate: all six fail G3. Under the frozen selection rule none
is selected for flight.**

Ranked by gates passed, then by 16-course chatter, the best is **`fast-brain-10b`**. It passes
G1, G2, G5, G6, G7 and G8 and fails only G3. `fast-brain-10b-r0001` also passes six gates. It
ranks second on chatter (0.00227 against 0.00226) and has more braking margin.

| candidate | G1 | G2 | G3 | G5 | G6 | G7 | G8 | cap excess | chatter 16 / 8 / caps | 16-course fin/crash | G3 max \|c-PD\| (within 0.5) | sink shortfall | high passes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **fast-brain-10b** | pass | pass | FAIL | pass | pass | pass | pass | +0.35 | 0.00226 / 0.00187 / 0.00179 | 16 / 0 | 0.89 (4/9) | 0.127 | 0 |
| fast-brain-10b-r0001 | pass | pass | FAIL | pass | pass | pass | pass | +0.26 | 0.00227 / 0.00189 / 0.00189 | 16 / 0 | 0.75 (5/9) | 0.117 | 0 |
| fast-brain-10c | pass | pass | FAIL | pass | FAIL | pass | pass | +0.27 | 0.00222 / 0.00185 / 0.00191 | 16 / 0 | 1.25 (3/9) | 0.140 | 2 |
| fast-brain-10a-t03 | pass | FAIL | FAIL | pass | FAIL | pass | pass | +0.26 | 0.00254 / 0.00208 / 0.00189 | 15 / 1 | 1.75 (3/9) | 0.203 | 2 |
| fast-brain-10a | pass | pass | FAIL | pass | FAIL | pass | FAIL | +0.27 | 0.00256 / 0.00214 / 0.00192 | 14 / 2 | 1.76 (3/9) | 0.204 | 1 |
| fast-brain-10a-r00001t03 | pass | pass | FAIL | pass | FAIL | pass | FAIL | +0.19 | 0.00261 / 0.00207 / 0.00206 | 13 / 3 | 1.61 (4/9) | 0.218 | 1 |
| brain-08 (reference) | FAIL | FAIL | FAIL | FAIL | pass | pass | pass | +1.20 | 0.00298 / 0.00488 / 0.00585 | 15 / 1 | 1.68 (1/9) | 0.157 | 0 |
| brain-09b (reference) | pass | pass | FAIL | pass | FAIL | FAIL | pass | +0.19 | 0.00508 / 0.00445 / 0.00417 | 16 / 0 | 1.56 (2/9) | 0.143 | 1 |
| PD teacher (reference) | pass | pass | pass | pass | FAIL | FAIL | FAIL | -0.19 | 0.00571 / 0.00424 / 0.00454 | 14 / 2 | 0 | -0.132 | 1 |

Other failed checks:

- `10a-t03` fails G2 on the rollout(S) 6 m/s cruise excess: -0.603 against the -0.6 limit.
- The G6 failures are descending passes more than 1.5 m above the gate, plus the crashes.

Every report is in [experiments/brain10_gate_results.json](experiments/brain10_gate_results.json).

### fast-brain-10b against brain-08 and brain-09b (surrogate)

- **Brakes like a braking brain.**
  - Cap steps from a hover settle within 0.05 m/s of 2-4 m/s caps. The 1 m/s cap undershoots by 0.4 m/s, which the gate's one-sided rule allows.
  - It reaches the cap + 0.5 m/s in 0.79-0.89 s (brain-09b 0.50-0.71 s, PD 0.28-0.50 s). From logged Minus Two cruise states a 3.5 m/s cap is reached in 0.82 s.
  - rollout(S) cruise excess at 3.5 / 4.5 / 6 m/s is +0.05 / +0.10 / -0.44 (brain-09b +0.16 / +0.18 / -0.54, brain-08 +1.30 / +0.75 / -0.47).
  - In-course cap excess is +0.35 m/s (brain-09b +0.19, brain-08 +1.20; limit 0.4).
- **Smoother than brain-08.**
  - 16-course chatter is 0.00226: 24% below brain-08 (0.00298) and 56% below brain-09b (0.00508).
  - Pitch chatter is 0.00128 (brain-08 0.00157, brain-09b 0.00325). Roll is 0.00093 (brain-08 0.00140).
  - Chatter in steady cruise (requests of 5 m/s or more) is 0.00176 (brain-08 0.00207, brain-09b 0.00415, PD 0.00267). The cruise hunting is about half brain-09b's, not gone (figure in the outputs).
- **Regressions pass.**
  - 16/16 courses finished (brain-08 15/1), at mean speed 4.80 m/s (brain-08 4.80).
  - Steep sink shortfall 0.127 (brain-08 0.157). No descending pass more than 1.5 m high.
  - Post-switch lateral error 3.04 m at 1 s (limit 3.08).
- **G3 still fails, but is the closest of any brain so far.**
  - Speed at +1 s minus the PD's in the 9 windows: -0.12, -0.50, -0.76, +0.55, +0.89, -0.26, -0.66, +0.26, +0.26.
  - It still over-brakes in the arch turns and is slow to brake on the straight gapon-02 windows.
  - Worst window: brain-09b 1.56 m/s, brain-08 1.68 m/s.
- **Only readout rows 0-2 changed.** The weight audit is in `runs/fast-brain-10b/weights-audit.json`.
  - Readout rows 0-2 and their biases changed. The yaw row is unchanged, and the other 38 tensors (connectome wiring, weights, signs, time constants, encoders) are bit-identical to the parent `motor-brain-10-tracking-05` (sha256 `64444a62...`).
  - Maximum weight change per row: throttle 0.154, roll 0.068, pitch 0.262. Bias changes are at most 0.006. brain-09b's largest change was 0.667.
  - sha256 `0ccf116190f5135220dbd2cc20a6dd9948654d07509eaa388a2286d8d4640bcd`.

## Limits and next

- **Surrogate only.** IdentifiedSim, synthetic courses and logged Minus Two states. Nothing was flown. These gates are development checks, not flight evidence.
- **The G3/G7 conflict.** The brain's 70-90 ms latency and about 0.5-0.6 amplitude make G3 and the new smoothness gate G7 pull against each other:
  - The PD itself chatters at 1.9x brain-08 in the harness even with no added delay. With 80 ms of added delay it fails G3 by 0.05 m/s.
  - Teachers whose inner loop is slow enough for the brain's latency (attitude gain 4 or less) miss G3 by 0.64 m/s with no delay and by 0.9-1.2 m/s with 80 ms of delay.
  - Latency compensation (the label lead) moved G3 the most: 10a -> 10b, worst window 1.76 -> 0.89 m/s. It was not enough.
  - A G3 v2 would have to be justified from this evidence and frozen before any scoring. It was not done in this round.
- **Flying it would be a deviation.** Flying `fast-brain-10b` would be a disclosed development deviation from the frozen gates, as brain-09b's flight was.
- **Worth checking live:**
  - Whether brain-10b's chatter reduction holds in Liftoff. brain-09b's surrogate-to-live chatter ratio matched: 1.7 vs 1.8 times brain-08.
  - Whether its slower brake onset (about 0.3 s later than brain-09b's) still stops it before the Minus Two hairpin wall.
- **Not in this brain.** The descent pilot was not merged on brain-10's branch, so brain-10 was not distilled with the view-keeping descent (`m4` merges it now, off by default). Its Straw Bale downhill contacts are unaddressed here.
