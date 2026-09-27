# fast-brain-10b and fast-brain-09b: experimental braking brain motors (development candidates)

Published on 2026-09-27. Download from
[GitHub Releases](https://github.com/skulitom/haltere/releases/tag/fast-brain-10-experimental)
or [Hugging Face](https://huggingface.co/Skulitom/haltere/tree/main/fast-brain10).

![fast-brain-09b stops short of the Minus Two hairpin wall and turns, 1x speed](https://raw.githubusercontent.com/skulitom/haltere/main/docs/liftoff_fast_brain09b_hairpin_stop.gif)

*Minus Two hairpin, real time: connectome activity (left) beside gameplay (right).
fast-brain-09b brakes, the round-4 turn-first rule stops it short of the wall and
turns it toward the next ring; accelerating out of the turn it then loses height and
hits the floor.*

**Brain motors that brake for the obstacle stack, not yet a race result.**
fast-brain-08 (the published Straw Bale finisher) does not slow down when the
obstacle governor asks for 3-4.5 m/s, so it cannot make the Minus Two hairpins. These
two readouts were distilled with synthetic governor speed caps so that they brake:

- **fast-brain-10b** is as smooth as fast-brain-08 in flight (roll/pitch command
  change 0.0031 per 10 ms tick\*; fast-brain-08's published race value is also 0.0031). Live
  it braked for the governor's caps at both arches before the hairpin (to 2.8-2.9 m/s
  under 2.7-2.9 m/s caps, somewhat later than fast-brain-09b). At the hairpin itself,
  while the vertical guard's false 1 m/s terrain climb overrode the pilot's descent
  (0.85 to 1.9 m, over the ring), it held 4.9-5.3 m/s under 3.6-4.0 m/s caps and hit
  the wall.
- **fast-brain-09b** brakes live: it slowed to 2.2-2.5 m/s at the arch before the
  hairpin in both flights. In round 3 it slowed to the ~3 m/s cap before grazing the
  hairpin wall; in round 4 it was braking (5.1 to 3.5 m/s in 0.5 s, still 0.6-1.6 m/s
  above the 2.8-3.6 m/s caps) when the new turn-first rule engaged and stopped it at
  0.5 m/s short of the wall. It is about twice as rough (0.0056-0.0063 per tick) and
  loses height when it accelerates hard from low speed.

\*Stick change: the mean of |Δroll| and |Δpitch| of the roll/pitch commands per 10 ms
tick, after the first 3 s of the log (launch excluded); over these short flights the
launch would otherwise dominate.

**Neither checkpoint finished any race, and neither passed all of its frozen gates**
(fast-brain-10b fails 1 of 7, fast-brain-09b 2 of 5; see below), so neither is
selected. They are published as development candidates with their full evidence;
**fast-brain-08 remains the published race result** (Straw Bale 5:17.805). Neither has
flown Straw Bale or Pine Valley. Minus Two is a seen development course; there is no
unseen-race or freestyle result.

## What they are

Both keep fast-brain-08's parent (motor10 candidate05), contract (6 m/s nominal,
scaled speed 2.4, vertical goal = request x 0.4 s) and blanked scene currents. Only
`readout.weight` and `readout.bias` rows 0-2 (throttle, roll, pitch) changed; the yaw
readout, connectome wiring, transmitter signs and all 38 other tensors are
bit-identical to the parent. The PD teacher is used only offline.

| | fast-brain-09b | fast-brain-10b |
|---|---|---|
| Recipe | DAgger distillation of the fast PD with synthetic governor caps (60% of rollouts, holds 0.5-3 s), slow legs (30%) and braking samples x5, refit with ridge 0.001 and smoothing 100 | brain-09's data recipe with longer synthetic caps (configs/brain10_caps.json, holds 1-5 s), a label teacher with a softer attitude loop (gain 4, default 8) and a stiffer vertical loop (gain 5, default 3), capped-turn relief, a 60 ms label lead, per-row smoothing and ridge 0.003 (no refit) |
| Largest readout change vs parent | 0.667 | 0.262 (fast-brain-08: 0.026) |
| Surrogate gates | brain-09 v1: 3 of 5 primary; fails G3 (speed vs the PD 1 s into 9 replayed logged Minus Two requests, 7 of them governor cap onsets) and G6 (stick chatter; descending passes high) | brain-10 v1: 6 of 7 primary; fails only G3 (worst 0.89 m/s against 0.5; 4 of 9 windows within) |
| Surrogate stick chatter (16 courses) | 0.00508 | **0.00226** (fast-brain-08 0.00298) |
| Surrogate cap excess | +0.19 m/s | +0.35 m/s (limit 0.4; fast-brain-08 +1.20) |
| Branch / commit | `m3-brain09` `0c634e6` | `m4-brain10` `7490a6b` |

The gates were frozen before any candidate was scored (brain-09: 22 candidates;
brain-10: 6). The frozen brain-10 rule allows the best-ranked failing candidate
(fast-brain-10b) to fly only as a disclosed development deviation; fast-brain-09b
flew under the same disclosure. Training ran on the CPU: a benchmark DAgger iteration
(10 drones, 20 s of flight, then a refit) took 39.5 s on two CPU threads and 36.8 s
with CUDA, because most of each tick is per-drone Python (pilot, teacher, simulator),
not the connectome; each full collection took about 45 min on the CPU.

## Live evidence (Minus Two / Turn Signals, 6 m/s)

Original `[Copy] New Drone`, fast race-cue pilot with the obstacle stack
(`--looming-brake --obstacle-stack on`), hidden Anode desktop, virtual pad kept inside
the seat (`pads_seat_only` true in every preflight and postflight), processed-control
ground check per pad.

| Run | Checkpoint | Code | Outcome |
|---|---|---|---|
| `minus-brain09b-vg-01` | fast-brain-09b | `m2-vertical` `f895c73` (round 3) | Pillar A clear (y 5.77, 1.3 m); slowed to 2.5 m/s at the arch, then from 5.2 to about 3 m/s under ~3 m/s caps at the hairpin, and grazed its wall at 21.9 s. Stick change 0.0056 per tick |
| `minus-brain10b-r4-01` | fast-brain-10b | `m4` `9a41acb` + descent view | Runtime stop, 0 ticks: the game had been left paused by the operator |
| `minus-brain10b-r4-02` | fast-brain-10b | `m4` `9a41acb` + descent view | Pillar A clear (y 5.84). Braked to 2.8-2.9 m/s under 2.7-2.9 m/s caps at both arches. At the hairpin a false vertical-guard terrain climb (20.15-21.4 s, 0.85 to 1.9 m, over the ring) overrode the pilot's descent while it held 4.9-5.3 m/s under 3.6-4.0 m/s caps, too fast for turn-first (limit 3.5 m/s); hairpin wall at 21.6 s. Stick change 0.0031 per tick |
| `minus-brain09b-r4-01` | fast-brain-09b | `m4` `9a41acb` + descent view | Pillar A clear (y 5.66); slowed to 2.2 m/s at the arch. At the hairpin it was braking (5.1 to 3.5 m/s, still above the 2.8-3.6 m/s caps) when turn-first v4 engaged, stopped it at 0.5 m/s short of the wall and turned it (the GIF above); accelerating toward a 4.9 m/s request it sank from 0.78 to 0.06 m and hit the floor at 23.7 s. Stick change 0.0063 per tick |

Times are from the start of the flight log. The round-4 stack's rules each failed at
least one of their own frozen offline gates, so these flights are development
evidence ([flight card](https://github.com/skulitom/haltere/blob/m4/docs/flight_cards/2026-09-26_obstacles_m2.md)).
In the same round the fast PD passed pillar C for the first time at 6 m/s (after
pillar A and the hairpin), then touched the garage floor for 1.5 s and struck the
ceiling after a false terrain climb. Fixes for the false climb, contact detection and a
brain that both brakes and holds height are in progress.

## Limits

- No race finish with either checkpoint; one development course (Minus Two) flown.
- fast-brain-10b did not follow the caps at the hairpin (+1.0 to +1.7 m/s, mean +1.5)
  during a simultaneous false guard climb. This is one episode, so it is untested
  whether it misses caps in general or only while climbing; its surrogate cap excess is
  +0.35 m/s.
- fast-brain-09b brakes but is rough and loses height when it accelerates hard from low
  speed near the floor.
- The readout moved much further from its parent than fast-brain-08's did; how that
  transfers across courses is untested.
- The obstacle stack they were flown with lives on branch `m4` and is not merged into
  `main`; its rules are off by default.

## Download and use

Extract `fast-brain10-inference.zip` into the repository root of branch
[`m4`](https://github.com/skulitom/haltere/tree/m4) (the round-4 flights ran at
`9a41acb`). It contains both checkpoints, their configs, weight audits, training
source, logs and the surrogate evaluations; fast-brain-09b's collection run; the
frozen gate files and scored results; the GateNet detector, stick mapping and
measured dynamics profile the runs used; the four flight sidecars with their
preflights; the ground checks; and per-file hashes (`fast-brain10-manifest.json`).
It also contains the 16-course gate harness the gate files pin by hash
(`tools/brain_gates/harness16.py`; the gate files record the absolute path it had
when the gates were scored). Training examples (~1.1 GB each) and raw flight CSVs are
kept locally, not included.

Prerequisites for the obstacle stack: its gap cue needs a CUDA GPU, the
`transformers` package and Depth-Anything-V2-Small
([depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf),
revision `5426e4f0f36572d16453bbda7a8389317b1bef99`, Apache-2.0; `model.safetensors`
sha256 `3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`). Put its
`config.json`, `preprocessor_config.json`, `model.safetensors` and model card
`README.md` in `runs/dense-depth-probe-20260923/model`, or set
`HALTERE_RELATIVE_DEPTH_MODEL_DIR`; nothing is downloaded automatically. The flight
limits below are the ones every published run used; without them the runner's
defaults stop a flight after 20 m.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-10b/candidate.pt `
  --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda `
  --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller brain `
  --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json `
  --looming-brake --obstacle-stack on --descent-view on `
  --seconds 150 --max-height 250 --max-speed 14 --max-distance 2000 `
  --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/my-flight.csv
```

For fast-brain-09b use `runs/fast-brain-09b-caps-r0001m100/candidate.pt`.

| Artifact | SHA256 |
|---|---|
| Archive `fast-brain10-inference.zip` | `d89a249d8b5202d5452abc4b8dae44b292b5105013e747f90a07c77e8a0855df` |
| `runs/fast-brain-10b/candidate.pt` | `0ccf116190f5135220dbd2cc20a6dd9948654d07509eaa388a2286d8d4640bcd` |
| `runs/fast-brain-09b-caps-r0001m100/candidate.pt` | `a222544cd617452088c54123f6b250e8d6b6ede859109437ad50b8ea97efc8d1` |
| Parent motor10 candidate05 | `64444a628c58b2c1615086bc84f8c5d3c0a0a12667df31d888c4587a802288b2` |

Videos (GitHub release assets): the three brain flights above, each with connectome
activity beside gameplay. The fast-brain-09b hairpin stop is also in the Hugging Face
folder.
