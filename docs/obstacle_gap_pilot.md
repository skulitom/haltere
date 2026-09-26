# Obstacle stack: gap pilot wiring (M2)

This page describes how the [gap cue](gap_cue.md) and the lag-aware turns are wired
into the live flight runner. The target is the dark pillar on Minus Two, which
looming misses: the aim should move beside a near object on the line to the ring
early enough for the lagging motor to follow.

**Status: nothing has been flown.** The stack is off by default. The runtime
bench (G8) passes with the depth model in its own process. The gap cue itself
still fails three of its four frozen offline gates (G1, G3, G4). Enabling it for
a live test is an explicit choice; run `shadow` first.

## Flags

| Flag | Default | Effect |
|---|---|---|
| `--obstacle-stack on\|shadow` | off | Needs `--pilot-profile fast` and `--looming-brake`. Runs the gap cue and the lag-aware turns. `shadow` runs the same processes and computations and logs them, but applies no aim shift and no lag-turn lead or heading change. It is the matched control. |
| `--gap-cue on\|off` | on inside the stack | Component override. `on` is refused without `--obstacle-stack`. |
| `--lag-turn [on\|off\|DECLARATION]` | on inside the stack, off outside | Component override. Outside the stack it keeps its earlier meaning (a bare flag means on). |

There is no speed cap: the live runs showed that brain-08 ignores slow requests
(asked for 3.5 m/s, it flew 5.1 m/s). The stack changes only the aim bearing and,
through the lag turn, the heading taper.

Example, brain-08 shadow run (not flown yet):

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/fast-brain-08-vgs04-s10r03m30/candidate.pt `
  --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --vision-device cuda `
  --pilot-assistance race-cue --pilot-profile fast --assist-speed 6 --motor-controller brain `
  --dynamics-profile runs/measured-dynamics-low-speed-20260923/profile.json --looming-brake `
  --obstacle-stack shadow --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/<new>.csv
```

## Data flow

1. **Camera process** (unchanged order). It captures a frame, preprocesses it,
   runs GateNet and the retina, detects the checkpoint ring and publishes the cue
   first, then runs looming.
   - With the stack on, it first copies the raw captured frame into a one-frame
     shared slot (`gap_stack.FrameSlot`), right after capture. The copy takes
     0.1-0.3 ms.
   - An mss screen grab arrives as a strided view of its BGRA buffer, which
     would take 2-3 ms to copy directly. It is instead converted from that
     buffer with one cv2 call (0.14 ms p50). The benched replay frames were
     already contiguous.
2. **Depth process** (`gap_stack.gap_process_worker`, normal priority). While the
   camera computes the cue, it:
   - resizes the frame to 448x252 with INTER_AREA (the same path as the obstacle
     store and the gates);
   - computes the HUD, ring and ghost overlay masks, which become block validity;
   - runs Depth-Anything-V2-Small (frozen, fp16 on CUDA, 336x602 input) to get
     36x64 block relative disparity.

   It then waits for that frame's ring cue in the camera's shared memory (median
   wait 9 ms). With the pose at capture time, taken from telemetry the controller
   has already observed, it runs `GapCue.update`. It publishes one sample into
   new shared slots after the looming slots (`camera_process.GAP_FIELDS`):
   - capture time;
   - the per-frame shift (unconfirmed);
   - kind, r_peak, r_ring, near_on_path and lr;
   - age;
   - the ring's world azimuth;
   - valid;
   - the cue's own two-frame confirmation (a diagnostic only);
   - stage timings.
3. **Pilot** (`FastRaceCue.update(..., gap=sample)`, `haltere/liftoff/gap_aim.py`).
   It confirms and applies the shift:
   - A sample counts once, and only if it is at most 0.2 s old.
   - A shift is confirmed when 2 of the last 3 samples, captured within 0.25 s,
     have |shift| >= 2 deg on the same side. The target is the newest confirming
     shift, clipped to 12 deg.
   - Side latch: the other side cannot engage until 0.6 s after the last
     confirmation of the first side.
   - The applied shift slews at up to 40 deg/s. Without fresh confirmation it
     decays to 0 over 0.3 s.
   - In `_ingest` it rotates the ring ray about world z (positive = left). The
     filtered direction is rotated by every change of the applied shift, so yaw,
     speed schedule and switch detection all see one consistent bearing.
   - The lag-turn trigger compares bearings with the shift removed, so a shift
     never looks like a checkpoint switch.
   - Conflicts:
     - A sample whose ring azimuth differs from the ring cue's by more than 6 deg
       describes another ring, for example right after a checkpoint switch.
     - The ring cue's flag clearance (`aim_u`) may point to the other side of the
       ring than the shift.

     Either way the evidence is dropped, the pilot holds the ring cue's own aim
     for 0.6 s, and the conflict is logged. When the flag clearance and the shift
     point the same way, the larger of the two offsets is used.
   - Terrain side steer: only while the looming TTC governor requests a climb
     (expansion below the path). |ln(L/R)| >= ln 1.5 votes 6 deg toward the
     farther side. Obstacle votes take priority.

The CPU fallback is refused: at about 0.3 s per frame the samples would always be
stale. Without an in-view ring the cue publishes `no_ring` and the aim decays.

## Logs and sidecar

CSV columns added at the end of each row. The columns are present in every
flight and are NaN or empty when the stack is off.

- **Gap columns:**
  - `gap_shift`, `gap_kind`, `gap_r_peak`, `gap_r_ring`, `gap_age`,
    `near_on_path`, `gap_lr`;
  - `gap_applied`: the rotation actually flown, 0 in shadow;
  - `gap_conflict`: `ring` or `flag`;
  - `gap_intended`: the shift the gap aim computed, also in shadow;
  - `gap_target`, `gap_valid`, `gap_ring_deg`, `gap_confirmed_cue`, `gap_seq`;
  - `gap_overlay_ms`, `gap_depth_ms`, `gap_decide_ms`, `gap_latency_ms`.
- **Camera stage columns** for the latest frame:
  - `cam_frame_time`;
  - `cam_capture_ms`, `cam_preprocess_ms`, `cam_inference_ms`, `cam_publish_ms`,
    `cam_looming_ms`, `cam_gap_ms`, `cam_total_ms`;
  - `cam_cue_latency_ms`: capture to the checkpoint cue in shared memory.
  - `publish` now covers only the cue write. Looming was part of `publish`
    before this change.

The sidecar's `obstacle_stack` records:

- the mode, the components and whether they were applied;
- the depth weights sha256 (`3152477c...`, reported by the worker at load);
- the content and file sha256 of `gap_cue.json`, `response_models.json` and
  `gap_pilot.json`;
- the placement, the motor response model and the worker's timings.

`pilot_assistance.gap_aim` holds the pilot counts: samples, stale samples,
episodes, latch blocks, conflicts and engaged seconds. `lag_turn` and
`lag_turn_declaration` record `applied`.

## Frozen configs

| File | Version | sha256 (content) | Frozen |
|---|---|---|---|
| `configs/obstacles/gap_cue.json` | 2 | `284b3c46a819...` | before any wiring result |
| `configs/obstacles/gap_bench_gates.json` (G8) | 1 | `db551b8813a3...` | before the first bench run |
| `configs/obstacles/gap_pilot.json` | 1 | `e704a3ba0d3d...` | after the bench, before any flight |
| `configs/obstacles/lag_turn.json` (from `m2-lagturn`) | 1 | `94315b4ddc4a...` | unchanged |

About these versions:

- **`gap_cue.json` version 2** changes only `relative_depth.input_hw`, from
  252x448 to 336x602. That is the teacher preprocessing, the primary basis of the
  v1 gates. On that basis the 252x448 input confirmed pillar A 0.3-3.2 m later.
  Because only the input changed, the v1 primary-basis gate results are the v2
  results. v1 is kept verbatim as `gap_cue_v1.json`, and the runtime reads the
  input size from this file.
- **`gap_pilot.json`** holds a-priori values, fitted to no flight: the M2 plan
  values plus the declared conflict and terrain thresholds. Its only choice made
  after the bench is `placement: process`, and that choice is not a threshold.

## Gates and results

### Offline gap-cue gates (v1, carried to v2), from `m2-gapcue`

- **G1 (pillar A):** fails. 4 of 8 approaches confirmed left at >= 5.5 m; 6 are
  needed.
- **G2 (pillar B):** passes, 1.35 s and 1.04 s before impact.
- **G3 (clean Straw Bale laps):** fails. 9.3 confirmed episodes per minute
  (limit 6), p90 |shift| 12 deg.
- **G4 (leak tests):** fails. Removed ring 94.8% (limit 95%).

Details are in [gap_cue.md](gap_cue.md) and
`docs/experiments/gap_cue_v1_results.json`.

### Lag-aware turns, from `m2-lagturn`

- **G5 surrogate:** both motors keep their finishes. The fast PD scores 14/16 and
  brain-08 15/16, with the same crashes. Brain-08 chatter is 0.00337 against a
  limit of 0.0035.
- **Lateral lag:** 1 s after a 20-40 deg switch the median lag falls from 0.96 to
  0.80 m (PD) and from 1.72 to 1.54 m (brain).
- **Pillar A kinematic replay:** the crossing moves 0.35-0.5 m toward the ring
  side but still hits at 6 m/s for both response models. The lag turn alone does
  not solve pillar A.

### G8 runtime bench

The bench is `haltere/obstacles/gap_bench.py`. It runs the real flight camera
process, including GateNet on CUDA, the retina, the ring cue, looming and the gap
cue, on `straw-brain08-06` from 30 to 90 s. The video is replayed at real time
through `camera_replay.py`, with the capture padded to 20 ms (the live mss median)
and Liftoff idle. A 100 Hz controller stand-in publishes the recorded pose, polls
the camera and burns a 6 ms brain-step load. Each run is one GPU chunk; the GPU
peaked at 42-45 C.

| 60 s, Straw Bale | baseline (looming, no gap) | gap in camera process | gap in depth process |
|---|---|---|---|
| camera rate | 17.8 Hz | **14.6 Hz** | 17.9 Hz |
| camera loop p50 / p95 | 60.4 / 68.3 ms | 78.6 / 93.2 ms | 60.6 / 68.3 ms |
| checkpoint-cue latency p50 / p95 | 49.5 / 56.4 ms | 46.9 / 54.6 ms | 49.9 / 56.5 ms (+0.1) |
| cue age at the controller p50 / p95 | 72.6 / 111.2 ms | 76.8 / 132.2 ms | 72.2 / 111.3 ms |
| gap work inside the camera loop p50 | - | 17.7 ms | 0.2 ms |
| gap sample age at the controller p50 / p95 | - | 131 / **173 ms** | 76 / 113 ms |
| gap capture-to-publish p50 / p95 | - | 88 / 95 ms | 51 / 58 ms |
| depth model p50 / p95 | - | 16.7 / 21.8 ms | 12.4 / 17.5 ms |
| **G8** | | **fail** (rate, gap age) | **pass** |

- **fp16 vs fp32** (336x602 input, 120 frames of the same flight): per-frame mean
  relative error p50 0.06%, p95 0.14%, max 0.26% (gate 1%). The largest single
  block differs by 2.5% at p95. Correlation >= 0.99999.
- **The camera placement fails**, as the task anticipated. The fallback placement,
  a separate depth process fed by a frame slot, passes every gate and is the
  declared runtime.
- **Smoke runs before scoring:** there were two 10 s smoke runs, after the gates
  were frozen and before the scored runs.
  - The first one handed the resized frame to the depth process *after* the cue
    and looming. Its gap age p95 was 150 ms and its rate 14.9 Hz, at or past the
    gates.
  - The hand-off then moved to right after capture, where it copies the full
    frame; the resize, masks and depth run in parallel with the cue. The second
    smoke run gave 15.5 Hz and a gap age p95 of 114 ms.
  - After the scored runs the frame slot gained two changes: the mss BGRA fast
    path and a resize for windows larger than the slot. Neither touches the
    benched path, because the replay frames are contiguous 1280x720. A 10 s
    smoke of the final code ran at 15.2 Hz with a gap age p95 of 118 ms.
  - All smoke runs are in `docs/experiments/obstacle_gap_pilot_bench.json`.
  - The 10 s smoke segment (30-40 s) is heavier than the 60 s mean. Short
    segments ran at 14.9-15.5 Hz, and so did the Minus Two baselines below, so the
    15 Hz margin depends on the scene more than on the stack.

### Wiring checks (report only, not gates)

- **Decision parity.** The bench ran two Minus Two clips (the first 13 s of
  `minus-brain08-01` and of `minus-fast6-cur-01`) through the depth-process
  placement. Its live-wired samples were compared with `gap_cue.decide` run
  offline on the same recorded frames, with the gap-cue evaluation's
  teacher-basis disparity, pose and logged cue:

  | | minus-brain08-01 | minus-fast6-cur-01 |
  |---|---|---|
  | frames matched | 195 | 199 |
  | ring-bearing difference, median / p95 | 0.02 / 0.49 deg | 0.003 / 0.31 deg |
  | shift difference p95 | 0.28 deg | 0.01 deg |
  | same kind | 96% | 96% |
  | active frames on the same side | 13 of 14 | 15 of 17 |

  Sign conventions, attitude, masks and the 336x602 input are therefore wired as
  they were scored.
- **Minus Two clips against a matched baseline** (10 s each):
  - The camera ran at 15.5 vs 15.6 Hz and at 14.9 vs 15.1 Hz. The baseline
    itself is near 15 Hz there, because GateNet and the cue take about 30 ms per
    frame.
  - Checkpoint-cue latency p95 was 59.0 vs 63.8 ms and 60.1 vs 62.9 ms.
  - Gap age p95 was 118 and 121 ms.
- **Wired pilot on pillar A.** `gap_aim` was replayed over those samples at the
  bench's 100 Hz ticks, using each sample's receipt time; hindsight geometry was
  used only for scoring.
  - It first confirmed a left shift inside the G1 approach region at **5.78 m**
    (brain-08) and **5.95 m** (fast PD), with no confirmed right shift, and
    applied up to 12 deg.
  - The cue's own two-frame rule offline reached 6.98 m and 6.45 m for these two
    approaches.
  - The recorded flights hit the pillar: the shift never flew. Whether 5.8-6 m
    is early enough for brain-08's 0.55 s response at 5-6 m/s is exactly the
    open question for a live test.
- **Wired pilot on a clean Straw Bale minute** (the G8 samples): 6 confirmed
  episodes per minute, engaged 8% of the time, applied shift p99 9.5 deg. The
  cue's own two-frame rule gave 10 per minute on the same frames. That is a
  1-minute sample; G3 used 22.6 minutes.
- **Runner dry run.** `visual_brain.run` ran in both `shadow` and `on` modes,
  20 s each, without a pad. Telemetry came from the fake-Liftoff simulator and
  frames from the replayed video, with the game-window, preflight and
  Liftoff-config lookups patched. That exercised:
  - the flag resolution and `wait_ready` (the depth model is ready in about 7 s);
  - 2000 CSV rows with 113 columns;
  - the sidecar hashes, the depth weights sha and the placement.

  The fake pose is unrelated to the video, so its decisions mean nothing. This
  was a plumbing check only.
- **Unit tests:** `tests/test_gap_pilot.py`, 35 tests. The label-isolation test
  also covers `gap_stack`, `gap_aim`, `camera_replay`, `camera_process`,
  `fast_race_cue` and `visual_brain`, and checks that no runtime module reaches
  the bench. The full suite passes: 882 tests.

## Limits

- **Nothing here is flight evidence.** A decision to keep the stack needs
  complete-system flights compared with it disabled (`shadow` is the matched
  control), on held-out courses too (`docs/project_direction.md`).
- **The gap cue fails G1, G3 and G4.** Expect false episodes near gate arch legs,
  round bales and hillsides (G3). A removed ring can still leak into the
  decision (G4).
- **The Pine trunk is seen only 0.77 s out.** That is too late for brain-08.
- **Liftoff was idle during the bench.** Live flights also carry the game's GPU
  and CPU load: the live camera ran at about 13 Hz with looming alone, with GateNet
  and the cue taking 40 ms against 27-30 ms here. The depth-process placement adds
  almost nothing to the camera loop. Depth inference, however, shares the GPU
  with the game, so live depth time and gap age will be higher than benched. The
  `gap_*_ms` and `cam_*_ms` columns record them.
- **The depth process runs on every frame,** also without an in-view ring,
  because the cue is not known yet when depth starts.
- **The conflict thresholds and the terrain side steer are a-priori.** Terrain
  votes never occurred in these replays, because the governor was not consulted
  there.
- **Brain-08 does not slow down on request.** There is therefore no speed cap:
  the shift must be early enough on its own.
