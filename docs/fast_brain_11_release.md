# fast-brain-11: first braking brain motor to fly a full Straw Bale lap (experimental development candidate)

Published on 2026-09-29. Download from
[GitHub Releases](https://github.com/skulitom/haltere/releases/tag/fast-brain-11-experimental)
or [Hugging Face](https://huggingface.co/Skulitom/haltere/tree/main/fast-brain11).

![fast-brain-11 over the Straw Bale hilltop and downhill, lap 1, 4x speed](https://raw.githubusercontent.com/skulitom/haltere/main/docs/liftoff_fast_brain11_straw_downhill.gif)

*Straw Bale lap 1, hilltop and downhill, 4x speed: connectome activity (left) beside gameplay
(right). Brain motors, fast race-cue pilot with the obstacle stack, assisted yaw.*

**The first brain motor that both brakes for the obstacle stack and flies a full Straw Bale lap.**
`fast-brain-11-b-cw13` flew lap 1 of the Straw Bale / Field Day race in **1:42.988**. fast-brain-08's first
laps took 1:46.358 and 1:46.072, but with a different pilot, so this is a whole-stack comparison. Lap 1 had one
audited ground contact on the downhill, and so did the other brain-11 run that reached it. fast-brain-08 had 3 and 2
in the first laps of its two finishes (7 and 6 per race). Audit counts are indicative only: the audit's video
false-positive check failed. brain-11's roll/pitch commands are as smooth as fast-brain-08's: 0.0025-0.0030
change per 10 ms tick on Straw, against 0.0031\*. Unlike fast-brain-08 it slows down for the obstacle governor's
speed caps.

**It has not finished a race, and it is not selected by its frozen gates** (brain-11 gates v1: 8 of 12). Every
flight below ended in a crash, and the release is published as a development candidate with all of them.
**fast-brain-08 remains the published race result** (Straw Bale 5:17.805). Straw Bale and Minus Two are seen
development courses; there is no unseen-race or freestyle result.

\*Stick change: the mean of |Δroll| and |Δpitch| of the roll/pitch commands per 10 ms tick after the
first 3 s of the log.

## Every flight with this checkpoint

All runs used the original `[Copy] New Drone` at 6 m/s. The pilot was the fast race-cue pilot with the obstacle
stack (`--looming-brake --obstacle-stack on --descent-view on`), running on a hidden Anode desktop. The virtual
pad stayed inside the seat (`pads_seat_only` true in every preflight), and a processed-control ground check ran
for each pad.

The runs used two stacks:

- **Round 4b** (`m4b` `b4d9e2c`): wall pilot v5, descent view v2, vertical guard v4.
- **Round 5** (`m5` `3b2fe2f`): adds `--stale-evidence on --contact-support on`, with wall pilot v6, descent
  view v3 with contact support v3, and stale evidence v2.

The motor assist was off unless stated. Times are from the start of the flight log.

| Run | Course, stack | Outcome |
|---|---|---|
| `minus-brain11cw13-r4b-noassist-01` | Minus Two, round 4b | Pillar A cleared (y 5.64), then braked to ~3.5 m/s at the arch. At the hairpin, looming read a TTC of ~1 s from 1.3 s before the wall, but the governor stayed armed until 0.8 s before it (cap 3.5 m/s at 0.6 s). It hit the wall at 3.7 m/s, 23.6 s. Stick change 0.0038 |
| `straw-brain11cw13-r4b-noassist-02` | Straw Bale, round 4b | **Lap 1 in 1:42.988**, with one audited downhill contact (79.4-80.0 s, 5.2 m/s), answered by a support climb. About 0.3 s before the lap arch (111.3 s), the ring reader took a dark logo on a fence banner beside the second start arch for the marker. The pilot turned toward it, and about 0.8 s into lap 2 the drone clipped that arch's right leg (112.4 s, 5.45 m/s). Stick change 0.0030 |
| `straw-brain11cw13-r5-noassist-01` | Straw Bale, round 5 | At the tall "FAT SHARK" arch the marker switched to the next ring while the drone was still inside the arch. The pilot turned hard left, and the drone yawed left faster than its path turned. Its momentum carried its right side into the arch's right leg (5.2 m/s, 19.4 s). Stick change 0.0025 |
| `straw-brain11cw13-r5-noassist-04` | Straw Bale, round 5 | Got past that arch and the hill. It had the same downhill contact as the lap run (79.4-80.0 s, after a ~24-degree descent), again answered by a support climb. 1.9 s later it was flying 0.5-0.7 m higher than the lap run at the next arch, and it descended at ~12 degrees onto the arch's top bar (81.9 s, 5.5 m/s); the lap run passed under the bar. Stick change 0.0027 |
| `minus-brain11cw13-r5-02` | Minus Two, round 5, motor assist v3 on | In front of the first arch, the motor assist's approach rule cut the request from 6 m/s to 1.3 m/s and then briefly backwards. Every earlier brain run passed this arch at ~5.5 m/s. The obstacle governor's looming brake then engaged at 6.84 s, and its 1.1 m/s cap held the crawl after the assist released (7.3 s). It bumped the arch at ~1 m/s, 8.1 s. The assist started this crash; it is withdrawn. Stick change 0.0059 |

Four further launches never flew: `straw-brain11cw13-r4b-noassist-01`, `straw-brain11cw13-r5-noassist-02` and
`-03`, and `minus-brain11cw13-r5-01`. The preflight refused them because compilers were running on the same PC.
The archive includes their preflight records.

The postflight checks of `straw-brain11cw13-r4b-noassist-02`, `straw-brain11cw13-r5-noassist-04` and
`minus-brain11cw13-r5-02` found a `rustc.exe` compile using 0.6-0.7 of a CPU core at the end of the flight. No
controller-deadline, camera or telemetry failures were recorded.

What the crashes point to (open problems, not fixed in this release):

- **Gate structures after a checkpoint switch:** the pilot turns toward the next ring while the drone may still
  be inside or beside a tall arch (r5-01), and a false ring-marker reading turned it into an arch leg (r4b-02).
- **Descending onto a gate's top bar:** with the ring clipped at the bottom of the image, the descent line
  carried the drone onto the next arch's bar (r5-04).
- **Straw downhill contact:** both runs that reached the downhill touched it at the same place after a ~24-degree
  descent, and both recovered. No camera cue measures the straw there.
- **Minus hairpin:** a brain that follows about 0.3 s late needs about 1 s of braking. The governor started
  braking only ~0.75 s before the wall, although looming had read a TTC of ~1 s earlier.

## What it is

Only `readout.weight` and `readout.bias` rows 0-2 (throttle, roll, pitch) changed relative to the parent, motor10
candidate05, which is also the parent of fast-brain-06 to -10. The largest changes are 0.290 (weights) and 0.0102
(biases). The yaw readout, connectome wiring, transmitter signs and the 38 other tensors are bit-identical. The PD
teacher is used only offline, and scene currents are blanked.

**Training.** DAgger distillation of the fast PD in the measured-drone surrogate:

- the rollouts were flown by the round-4 deployed pilot (descent view v1, wall pilot v4, vertical guard v3, gap
  pilot v5, lag turn v2);
- synthetic governor speed and climb caps (configs/brain11_caps.json), yaw holds, a braking weight and a sag
  weight;
- a label teacher with a softer attitude loop and a stiffer vertical loop;
- a refit with a cruise weight of 1.3 (the "cw13" refit).

The exact training file is `runs/fast-brain-11-b-cw13/training-source.py`. It differs from
`haltere/train/fast_motor_tracking.py` at `4ee71c3` only in where one reporting metric (`synthetic_cap_events`) is
recorded. Training ran on the CPU. In brain-10's benchmark a DAgger iteration took 39.5 s on the CPU and 36.8 s on
the GPU, because most of each tick is per-drone Python. Details and every candidate:
[fast_brain_11_candidate.md](https://github.com/skulitom/haltere/blob/m5/docs/fast_brain_11_candidate.md).

**Surrogate gates** (surrogate only, not flight evidence):

- **brain-11 gates v1** (frozen before any brain-11 candidate was scored): 8 of 12. It fails:
  - G3v2: speed against the PD after logged cap changes;
  - G6: lateral error;
  - G10: settling in left turns;
  - G11: reaching 4 m/s within 1.5 s when accelerating from a stop or creep toward 60 degrees left of the heading.
- **Later brain-12 gates:** 6 of 15, the same as the best brain-12 candidate. brain-11 was the reference there, not
  a candidate, and these gates are not independent of it:
  - they were frozen after its round-4b flights and after it was measured;
  - G14 replays its own live windows;
  - G16 is defined as its own score plus 2;
  - only the course gates use fresh held-out seeds.

## Download and use

Extract `fast-brain11-inference.zip` into the repository root. It contains:

- the checkpoint, config, weight audit, training source, log and evaluation;
- the collection run's config, log, evaluation and source;
- the frozen gate files, the synthetic-cap config (configs/brain11_caps.json), the scored results and the candidate
  doc;
- the GateNet detector, stick mapping and dynamics profile the runs used;
- the flight sidecars, the preflights (including the refusals), the contact audits and the ground checks;
- per-file hashes (`fast-brain11-manifest.json`).

The training examples (~1.3 GB) and raw flight CSVs are kept locally, not included.

The obstacle stack's gap cue also needs:

- a CUDA GPU;
- the `transformers` package;
- Depth-Anything-V2-Small ([depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf),
  revision `5426e4f0f36572d16453bbda7a8389317b1bef99`, Apache-2.0; `model.safetensors` sha256
  `3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`), placed in
  `runs/dense-depth-probe-20260923/model` or pointed to by `HALTERE_RELATIVE_DEPTH_MODEL_DIR`.

Nothing is downloaded automatically. Keep the flight limits in the commands below; without them the runner stops a
flight after 20 m.

**The stack of the 1:42.988 lap** (round 4b): check out branch [`m4b`](https://github.com/skulitom/haltere/tree/m4b)
at `b4d9e2c`, extract the archive, then:

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-11-b-cw13/candidate.pt `
  --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda `
  --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller brain `
  --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json `
  --looming-brake --obstacle-stack on --descent-view on `
  --seconds 480 --max-height 250 --max-speed 14 --max-distance 2000 `
  --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/my-flight.csv
```

**The round-5 stack** (branch [`m5`](https://github.com/skulitom/haltere/tree/m5) at `3b2fe2f`): the same command
plus `--stale-evidence on --contact-support on`. Both Straw runs with it crashed. m5 refuses to load the round-4b
declarations.

| Artifact | SHA256 |
|---|---|
| Archive `fast-brain11-inference.zip` | `baa93dd4b8f78d9e39b661ed74a192775e32fe885e61a4b012898a5af126bbd8` |
| `runs/fast-brain-11-b-cw13/candidate.pt` | `44cca3c4fc3e40f8ab1cc9cb5e94337edf0229dc10d1611e271e640e5627c11d` |
| Parent motor10 candidate05 | `64444a628c58b2c1615086bc84f8c5d3c0a0a12667df31d888c4587a802288b2` |

**Videos** (GitHub release assets, with connectome activity beside gameplay):

- the Straw Bale lap-1 run;
- the r5-04 run, which ended on the arch's top bar;
- the Minus Two hairpin.

The lap-1 video is also in the Hugging Face folder.
