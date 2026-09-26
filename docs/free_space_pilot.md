# Free-space corridor planner: the pilot half (M3)

The user asked for the model to adapt its flight path around obstacles: steer
sideways or up/down toward the checkpoint, rather than only brake or climb. The
free-space corridor planner does this in two parts. Perception (branch
`m3-freespace`) turns each frame into a planner sample: which offsets beside or
above the ring have a free corridor, plus floor and ceiling clearance in metres.
This page covers the other half, branch `m3-pilot`: how the fast pilot uses those
samples. The fixed interface between the two halves is section 5 of the
specification.

**Status.** Everything is off by default. The pilot half is built and tested
against synthetic samples. On its own it cannot fly: `--obstacle-planner` refuses
to start until the camera stack publishes planner samples (`plan_out`), which is
the perception half's job. No planner flight and no gate result exist. The only
replays so far are identity checks and a clearly labelled plumbing check with
synthetic samples (below).

## What the pilot does with a sample

The pilot only uses a sample that is valid (kind `clear`, `aperture`, `shift` or
`blocked`) and no more than 0.2 s old at the controller. Each seq is used once.
The rules live in `haltere/liftoff/corridor_aim.py` and read their values from the
frozen `configs/obstacles/free_space.json` v1 (`pilot` and `vertical` sections).

**Aim (CorridorAim).**
- A class (left, right or up) is confirmed when 2 of the last 3 samples, captured
  within 0.25 s, are `shift` samples of that class.
- Once confirmed, the class stays latched for at least 0.6 s and for as long as
  the ring path stays blocked.
- The target is that class's option in the newest sample. If a sample has no
  option for the class, the previous target is kept.
- The latch flips to another class only after its own class had no option for 3
  samples in a row. So it never swaps to the far side while its own side is still
  open.
- The offset moves toward the target at up to 40 deg/s sideways and 20 deg/s up.
  It is bounded to |az| <= 20 deg and 0 <= el <= 12 deg.
- Release: after 2 `clear`/`aperture` samples in a row, or after 0.3 s without a
  sample. The offset then falls to 0 linearly over 0.3 s.
- Conflicts: a sample of another ring (azimuth more than 6 deg off), or a flag
  clearance on the other side, drops the evidence. The pilot then holds the ring
  cue's own aim for 0.6 s.
- While turn-first is active, the sideways target is 0. Search, launch, wait and
  support climb reset the aim.

**Applying the offset (FastRaceCue).**
- The ring cue's aim ray is rotated about world z by the offset's az, then tilted
  up by its el. The filtered direction is moved with every change of the offset,
  so yaw, the speed schedule, the vertical request and switch detection all see
  one bearing.
- An up-shift therefore adds only speed x tan(el), at most about 1.3 m/s at 6 m/s.
- With a flag clearance on the same side, the aim keeps the larger of the two
  offsets, as the gap aim does.
- The lag-turn lead is computed on the ring bearing without the offset, and the
  offset is added after it. The flown aim is therefore within 15 + 20 deg of the
  ring.

**Speed.** A confirmed blocked path, or a confirmed class whose newest sample is
an urgent shift, caps the request along the applied aim at max(v_cap, 1 m/s).
The cap is brought in at up to 15 m/s^2. When the TTC governor's own cap is
lower, the governor's cap wins. brain-08 ignores slow requests, so this acts for
the fast PD (and a future brain-09).

**Vertical guard.** The guard uses the newest fresh valid sample, with heights
dead-reckoned by the measured vertical speed. Its rules act on the vertical
request in this order:
1. **Floor bound.** vz >= -(h_floor - 0.5)/0.5. A positive bound is capped at
   1 m/s. A sink toward the floor is slowed and stopped, never answered with a
   hard climb.
2. **Descent first.** While the drone sinks faster than 0.3 m/s, rule 3 is
   withheld.
3. **Terrain climb.** Needs rising ground (`rise` finite in 2 of the last 3
   samples) while level. vz >= min(3.5, v_h tan(gamma+) + rise / max(rise_x/v_h -
   0.3, 0.3)), where gamma+ is the flight-path angle at the sample's capture.
   Held 0.5 s, then released at 3 m/s^2, and ended 2.5 m above its start.
4. **Ceiling bound** (applied last). vz <= max((h_ceil - 0.5)/0.5, -1). If the
   ceiling bound is below the floor bound, the request is their midpoint (a
   squeeze).

The ramps are 5 m/s^2 (floor bound), 10 m/s^2 (climb) and 15 m/s^2 (down, at the
ceiling).

**What it replaces, and what stays.**
- The gap cue's aim shift, including its terrain side-steer, is still computed
  and logged but never applied.
- The TTC governor's terrain climb is replaced by rules 1-4 while the newest
  planner sample is fresh and valid. Its value is logged as
  `governor_climb_shadow`.
- When the planner is stale or invalid (`no_ring`, `no_scale`, `slow`...), the
  governor's climb and ceiling guard v3 apply as before. With the planner on, a
  sink is levelled first: while vz < -0.3 m/s the governor's climb floor becomes 0.
- These stay unchanged: the wall brake, urgent brake, stand-off and cap machinery,
  the ceiling guard's overhead hold, turn-first, lag-aware turns, search, launch,
  support climb and the surface rules.

## Flags and modes

| Flag | Default | Effect |
|---|---|---|
| `--obstacle-planner off\|shadow\|on` | off | Inside `--obstacle-stack` only (refused without it, and refused with `--gap-cue off`, because the planner runs in the depth process). `on` needs `--obstacle-stack on`. |

- **Shadow** computes and logs every planner quantity and applies nothing. Its
  commands are bitwise those of the same stack with the planner off and the gap
  aim unapplied.
- **Planner off:** the stack is exactly as before.
- The runner accepts only the frozen v1 `free_space.json` (same canonical hash as
  `gap_pilot.json`). The spec it passes to the depth process carries
  `free_space` (config path, sha256, version, motor response model).

Live command (after the offline gates, with the user's go-ahead only):
`--looming-brake --obstacle-stack on --wall-pilot on --lag-turn on --obstacle-planner shadow|on`.

## Logs

**CSV.** Each row carries `plan_<field>` for every planner field of the newest
sample the pilot received, and these pilot columns: `plan_fresh`,
`plan_cls_confirmed` (-1/+1/+2/0, 9 = blocked), `plan_intended_az/el` (the latched
option), `plan_applied_az/el` (0 in shadow), `plan_v_cap_intended/applied`,
`plan_vz_lo/hi`, `plan_climb`, `plan_vz_before/after`, `governor_climb_shadow`,
`plan_squeeze`, `plan_conflict`, `plan_episode` and `plan_flip`.

**Sidecar** (`obstacle_stack.planner`):
- the mode and the declaration (content, sha256, file sha256, version);
- the `response_models.json` sha, the depth weights sha prefix (`3152477c`),
  placement, motor model;
- the worker counters (from the depth process) and the controller's busy-lock
  skips;
- per-kind counts, held-scale samples and p50/p95 of `age`, `lk_ms`, `plan_ms`,
  `depth_ms` and `scale_n` for the samples received;
- the corridor aim and vertical guard counts.

## Interpretation choices

These are places where the specification left a detail open. Each choice is
recorded here and in the code.
- The 0.3 s "no fresh sample" release has no key in `free_space.json` v1. It is
  the declared constant `CorridorAimConfig.stale_release_s` and is written to the
  sidecar.
- Rise confirmation reuses the pilot's `confirm`/`window`/`confirm_window_s` (2
  of 3 within 0.25 s).
- `gamma+` in the terrain-climb formula is the flight-path angle at the sample's
  capture time (the path the rise was measured from). With the current angle, the
  climb would feed on itself between samples.
- "The lower cap wins": the planner cap is applied only when it is below the
  governor cap. The governor cap always stays.
- `--obstacle-planner shadow` is accepted under `--obstacle-stack shadow` as well
  as `on`. `on` is accepted only under `on`.
- Rule 4 (ceiling) is also applied while descent-first holds rule 3.
- `plan_cls_confirmed` shows 9 while a blocked path is confirmed, even if a class
  is latched.

## Offline replay harness

`haltere/obstacles/pilot_replay.py` is the committed port of the M2 round-2
open-loop harness (an offline module; no runtime code can import it). It feeds
every logged controller tick of a flight to the `FastRaceCue` of a chosen code
tree: the recorded pose, the ring cue, and the logged looming, gap and planner
samples, each published as the runner published it. It writes the per-tick
requests.

- **Variants:** `--stack none|flown|on|shadow` (with `--wall off|on|shadow` for
  `flown`) and `--planner off|shadow|on`. `--gap-apply off` builds the planner's
  matched control.
- **Planner stream:** `--plan-stream <npz>` takes one array per planner field
  plus `pub_time` (capture + 0.10 s). A sample is published at the first tick
  whose clock has reached it. A stream whose `source` is not `perception` is
  reported as a plumbing check, never as a gate result.
- **Gates:** `--score-v` scores V1-V3 from `configs/obstacles/free_space_gates.json`
  and refuses an unfrozen file.

```powershell
.venv/Scripts/python.exe haltere/obstacles/pilot_replay.py --out <prefix> --stack on --planner on `
  --plan-stream <dir>/{flight}.plan.npz --score-v minus-brain08-gapon-01 minus-brain08-gapon-02 minus-fast6-wall-01
```

The gates declaration `configs/obstacles/free_space_gates.json` v1 was frozen
before anything was scored: sha256 `626c4b7606ee...`, 2026-09-26T12:27:24,
commit `cc74270`. It holds the specification's thresholds for S1-S3, A1, B1, T1,
Q1, L1, V1-V3, R1 and I1, the splits, and what each failure means. One reading is
recorded in the file: the five S1/S2/A1 approaches are the held-out Minus Two
pillar-A approaches flown without gap-cue authority (`minus-brain07-01`,
`minus-fast6-cur-01`, `minus-brain08-slow35-01`, `minus-brain08-loom-01`,
`minus-brain08-gapshadow-01`).

## Evidence so far (none of it is flight evidence)

### Identity against m2-hairpin (open-loop replays, 2026-09-26)

**Method.** Each logged tick was replayed through the old tree (`m2-hairpin`
4083454, exported with `git archive`) and through the new tree, each in its own
process. The per-tick requests were compared bitwise: velocity x/y/z, yaw stick,
pilot state, governor cap and climb.

**Coverage.** 12 flights (18,385 ticks):
- the five stack flights: `minus-brain08-gapshadow-01`, `-gapon-01`, `-gapon-02`,
  `minus-fast6-gapon-01`, `minus-fast6-wall-01`;
- `minus-brain08-loom-01`, `pine-brain08-loom-01`, `pine-fast6-ttc-01`,
  `minus-brain08-01`, `minus-fast6-cur-01`, `minus-brain07-01` and
  `minus-brain08-slow35-01`.

Straw Bale laps `straw-brain08-04` and `-06` were added for four variants
(65,284 more ticks), with their offline looming streams.

| Variants, planner off (old tree vs new tree) | Result |
|---|---|
| default pilot; stack as flown; as flown + wall rules on; as flown + wall rules in shadow; stack on; stack on with the gap aim unapplied; stack in shadow | bitwise identical on every flight and tick |

**Planner shadow control (new tree only).** The planner was fed the synthetic
streams below, and its shadow was compared with its matched control:
- stack on + planner shadow == stack on with the gap aim unapplied;
- stack shadow + planner shadow == stack shadow;
- as flown + planner shadow == as flown with the gap aim unapplied.

All three are bitwise identical on all 12 flights. The shadow planner was not
idle during these runs: it held an intended offset on 1,100 ticks and would have
displaced the governor climb on about 850 ticks. The same identity check was
repeated on the committed code (commit `48f12d8`) without the Straw laps, with the same result.
Reports: `docs/experiments/free_space_pilot_identity_full.json` (with the Straw laps) and
`docs/experiments/free_space_pilot_identity_48f12d8.json`.

### Plumbing check with synthetic samples (NOT evidence)

This check only shows that the samples reach the pilot and that it acts on them
as specified.
- **The samples are synthetic.** They are derived from the M2 anatomy
  coordinates: the pillar-A box, the Minus Two floor at z 0 and a ceiling at
  2.3 m, a static 2-D corridor test and the logged telemetry. They are not derived
  from images.
- **The replay is open loop.** The recorded motion never responds to the requests.
- **Labels.** The harness labels every score on such a stream "PLUMBING CHECK".
- **Report:** `docs/experiments/free_space_pilot_PLUMBING_synthetic_not_evidence.json`.

What the pilot did with the samples (stack on + planner on):
- It confirmed left and right classes and blocked paths at pillar A, with applied
  offsets up to 20 deg and speed caps on the blocked samples.
- It displaced the governor's climb whenever the newest sample was valid.
- It lowered the peak vertical request in the last 3 s of `minus-brain08-gapon-02`
  from 3.46 to 1.54 m/s, and of `minus-fast6-wall-01` from 3.44 to 2.98 m/s.

**The design finding.** Whenever the samples turned `no_ring` (the ring marker
clamped at an image edge while pitching up or turning), the planner was invalid
and the governor's 3.5 m/s climb came back.

The logs confirm this independently of any perception quality. During the logged
governor climbs, the ring was in view and the drone above 1.5 m/s on only this
share of the ticks:

| Flight | Ticks with ring in view and > 1.5 m/s |
|---|---|
| `minus-brain08-gapon-01` | 36% |
| `minus-brain08-gapon-02` | 18% |
| `minus-fast6-wall-01` | 56% |
| `minus-fast6-gapon-01` | 98% |
| `pine-fast6-ttc-01` | 79% |

The ring was mostly clamped at an edge on the rest. Under the v1 rule ("the
vertical guard uses the newest fresh valid sample"), the planner cannot displace
the governor climb on the other ticks, however good the perception. V2's
"displaced on >= 90% of the logged climb ticks" is therefore out of reach by
construction on gapon-01 and wall-01. V1 then rests on the fallback: the
governor climb, ceiling guard v3 and descent first. Floor and ceiling profiles
do not depend on the ring. A v2 would use `h_floor`, `h_ceil` and `rise` from
`no_ring` samples too, but that needs a specification change, a new version and a
new freeze. It is not implemented here.

**A second note from the same check.** On `pine-fast6-ttc-01` the synthetic
samples were valid `clear` samples without floor or rise values: the stream
has no terrain model.
- The planner therefore displaced the governor's mound climb (513 ticks), and
  the guard had nothing to put in its place.
- The V3 plumbing score was 0.13 of the climb seconds answered.
- Under the v1 rule, a valid sample with neither `h_floor` nor `rise` removes
  the governor climb and leaves no vertical protection.

Real perception should report the mound's floor bins and rise. V3 (dev-flagged)
tests exactly this on the real stream.

## Open items

- **Perception interface.** `camera_process.py` carries `PLAN_FIELDS`,
  `PLAN_KINDS`, `plan_sample` and the controller's non-blocking read of
  `plan_out`, exactly as section 5 of the specification defines them.
  - The perception branch (`m3-freespace`) owns the same interface and the depth
    process side (creating `plan_out`, running the planner in the worker). Expect
    to reconcile this file on merge.
  - The runner passes the planner request to the depth process as
    `gap_spec['free_space']` (mode, config path, sha256, version, motor model).
- **The declaration.** `configs/obstacles/free_space.json` v1 is frozen by the
  perception side. Until it exists and is frozen, the runner refuses
  `--obstacle-planner`.
- **Gates.** V1-V3 wait for the perception plan streams
  (`<flight>.plan.npz`). The perception gates use the real
  `haltere.liftoff.corridor_aim.CorridorAim` for confirmation.
- **Flights.** No flight has been made with the planner. Live stages follow
  section 10 of the specification and need the offline gates and the user's
  go-ahead.
