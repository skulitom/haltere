# Generalization program

Direction revised on 2026-09-22 after the user's review of the flight evidence.
The 2026-09-23 acceptance scope is Straw Bale / Field Day, Pine Valley / Forest
For The Trees, Minus Two / Turn Signals, Autumn Fields / Walk In The Park and
Hangar C03 / Shipments, all three laps. The user requires full-race time within
20% of their matching completion. Saved personal race times and thresholds are
in [main_track_targets.json](../configs/main_track_targets.json). They are
evaluation metadata, never runtime guidance. Preserve the user's earlier 3/3
completion requirement per track; speed without clean finishes does not pass.
The user confirmed that these personal times used `[Copy] New Drone`; autonomous
runs retain that same original calibrated drone.

The corrected PD wiring and workload checks are implemented. The four full-race
comparisons (PD and motor10 candidate05 on Straw Bale and Minus Two at 2.5 m/s)
finished **brain 0/2, PD 1/2**. Both motors hit the Minus Two pillar; PD alone
finished Straw Bale in **13:04.047**. There were no camera/deadline failures.
The [complete batch record](flight_cards/2026-09-23_matched_full_races.md)
retains all attempts. Current priority is obstacle geometry, while retaining the
motor-tracking gap as a separate problem to address through later flight-cost training.
Use unscaled measured velocity, the effective requested speed and
`position_gain=max(.8, speed/3)` for PD, matching its training-teacher contract.
The brain alone retains its speed-dependent sensory scaling. Report finishes,
full-race times and impacts for both motors as the standing comparison; 180-second
windows do not answer the full-race question. Runtime stops remain visible and
receive one unchanged retry after fixing the runtime issue; do not count them as
clean navigation failures or quietly replace the original attempt.

The visual runner now refuses known training/evaluation/benchmark jobs in any
Windows session and other busy compute processes before starting capture/control.
The inventory is a snapshot, not a machine-wide reservation or exhaustive GPU
detector. Arrange exclusive flight time, retain the preflight report and do not
start training or benchmarking during a race. Compare recorder-off, bounded CPU
libx264 and explicit `--video-encoder h264_nvenc` on the same scene before choosing
the encoder for the frozen batch. NVENC must actually encode in Anode; it never
falls back silently. New CPU recordings cap ffmpeg at two encoder threads.
The user subsequently authorized leaving the separate LitHarness benchmark
running while we use Anode. Its explicit PID exceptions are recorded; the guard
still rejects any exception measured at half a CPU core or more, and retains
preflight and postflight snapshots. This is a disclosed exception to the original
"nothing else running" condition, not evidence that Windows sessions isolate CPU/GPU resources.

Challenge sections with nearby obstacles and per-section outcomes now support
the first frozen geometry on/off comparison. Extend the complete-system evidence
across layouts before promotion. Do not substitute another readout fit for
obstacle clearance. Full-throttle/high-rate measurements and brain training
through flight costs follow this comparison.
The [section generator and offline scorer](challenge_sections.md) now implement
that course format, including explicit unknown geometry and a separate box-only
depth-calibration variant. The box variant rendered in game and its first PD
attempt crossed five of six sections geometrically before hitting the last
wall. Separate privileged PD collection subsequently qualified that box course
with a game-confirmed **3:35.704** finish. The flag variant remains unqualified;
this oracle finish is not an autonomous score. The subsequent
[matched live geometry comparison](flight_cards/2026-09-23_geometry_control.md)
finished **0/1 off versus 1/1 on**, with a **2:26.657** autonomous PD finish on
the same known course. This is initial development evidence; the brain was in
shadow, and main-track speed, transfer and reliable completion remain unmet.
Two unchanged assisted repeats added one finish and one wall impact (2/3 overall).
The failed repeat lost nearby obstacle memory while nearly stationary. Retaining
both points and interpolated surfaces subsequently finished 0/3 (one geometry
timing stop and two operator-stopped stalls). Short-lived interpolation and
faster queries also finished 0/3, without timing failures. Explicit overlap
recovery and feasible motion before braking then finished 2/3, with one wall
stall and no detected impacts or runtime failures. A subsequent minimum-motion
revision finished 1/3 (2:31.664), with one wall impact and one declared wall stall.
These failures prevent a reliability claim; broader geometry and motor work
remain required. Do not promote a local planner change from a successful replay.
The [obstacle/depth record](flight_cards/2026-09-23_obstacle_geometry.md) preserves
the failure and causal replay limits.

The race-cue stack has two seen-course full finishes and **zero full finishes on
five first-exposure courses**: Hannover, The Pit, Paris, Hall 26 and The Green.
These attempts used successive development revisions, not one frozen test batch.
The Green repeat also hit an overhang after the horizontal-hold change. Stop
using successive official maps to justify another marker-specific steering rule.

## 1. Prepare courses in code, then freeze batches

Use procedural courses and existing Workshop content. Do not manually design
evaluation maps. A small editor check is acceptable to verify the file format;
it is not the course-production workflow.

**First check passed:** Liftoff 1.7.6 loaded a code-edited local copy of
`[Honk] Zoomies I` on 2026-09-22. The copy has new track/race/passage IDs, a
two-metre translation of every object and spawn, and one required lap instead
of three. The original Workshop files retained their hashes. The game log
identified both new IDs and the rendered race showed `1/1`. This establishes
file compatibility, not a completed flight or verified checkpoint crossings.

Local evidence: `runs/course-pool-load-check-20260922/load-evidence.json`,
`loaded-content-log.txt`, `loaded.png`, and the bundle's `manifest.json`.
The loaded IDs are `b0964726-0254-4e2a-b6c7-169a46ad7718` (track) and
`96e0fbcb-40e9-4416-9cb5-62560628094d` (race). This entire source family is
development data. Translation or renamed IDs do not make it unseen.

The offline `haltere.liftoff.course_pool` tool clones Workshop pairs, generates
simple seeded loops, validates references and installs new local content without
overwriting existing courses. Example commands from the repository root:

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.course_pool generate --seed 100 --gates 10 --obstacles 12 --out runs/course-dev-100
.venv/Scripts/python.exe -m haltere.liftoff.course_pool install runs/course-dev-100 --local-root "$env:USERPROFILE/AppData/LocalLow/LuGus Studios/Liftoff"
.venv/Scripts/python.exe -m haltere.liftoff.course_pool clone --track PATH_TO_WORKSHOP_TRACK --race PATH_TO_WORKSHOP_RACE --name "Local development copy" --translate 2 0 0 --laps 1 --out runs/course-copy
```

Install a batch before starting Liftoff. The running game did not discover
externally added files until a content-refresh restart; launch through Steam
inside Anode, then keep Liftoff open between flights. The native editor stores
local files under `Tracks/<ID>/<ID>_0001.track` and the analogous `Races` folder.
An unchanged empty editor track can report a successful save without writing
a file. Changing its settings produced a native file at the expected location.

The initial `ellipse-v2` generator uses built-in asset IDs in The Drawing Board.
It varies loop size and height, with scenery outside the route. **Generated
courses are development candidates, not a qualified obstacle benchmark.**
Seed `20260922` with eight checkpoints and six scenery objects loaded in game
with a `1/1` race HUD. Evidence is in
`runs/generated-course-dev-20260922-v2/load-evidence.json` and `loaded.png`.
These checkpoint assets define passage volumes; this initial generator does not
yet create a varied set of physical gate frames. Candidate04 subsequently
completed this development loop in **2:20.759**, confirmed by the game results
screen in `runs/motor10-transfer-20260922/loop-finish.jpg`. That verifies this
one generated course's full playability, not other seeds or obstacle layouts.
Placing objects in other environments needs a checked free-space envelope;
random coordinates can put gates inside existing buildings or below terrain.

The first generated draft failed to load because the wall asset requires
`TrackBlueprintFlag`, not the base XML type. That failed bundle/log remain under
`runs/generated-course-dev-20260922`; the validator now rejects this mismatch.
The corrected generator uses new IDs and a distinct `Haltere loop v2` name.
The failed draft remains installed after automated cleanup was blocked; choose
the v2 course for development.

Before a headline evaluation:

- Assemble approximately 20 courses across several layout families, environments,
  gate appearances, elevation profiles and obstacle arrangements. Keep Workshop
  attribution and source hashes. Do not republish creators' geometry as ours.
- Split by original Workshop family and generated layout family/recipe, not file
  name, seed alone, translation or individual recordings. Distinguish unseen
  layouts in a known family from genuinely held-out families and environments.
- Check playability independently before controller evaluation; record rejected
  candidates and fixed rejection rules. Never discard a course because our
  controller failed it. Keep oracle clearance checks separate from runtime.
- Freeze course hashes, order, full stack hashes, parameters, duration limits and
  repetitions before any evaluated attempt. Run the entire batch unchanged.
  Count every attempt, including impacts, stalls, resets and timing stops.
- Review failures after the batch. A family used for tuning becomes development
  data; a subsequent headline comparison needs another untouched batch.
- Report full finishes/attempts first, then race times, contacts, interventions,
  invalid launches and timing failures. Summarize both first attempts and repeats.
  Compare variants on the same preregistered batch without tuning between them.

For autonomous evaluation, course geometry remains offline. The live controller
gets causal camera frames, telemetry and a declared task, never the XML, a route
lookup or a known waypoint. Explicit oracle qualification/collection may use
geometry but is labelled privileged and excluded from autonomous scores.
XML supplies object transforms and directed checkpoint labels. **It does not by
itself supply exact image depth.** Depth ground truth also needs object meshes,
terrain, occlusion, camera calibration and synchronized pose/rendering. Primitive
approximations must be labelled as approximations and checked against images.

### Automatic flight cards

Generate the preflight card from the run manifest before flying. Afterwards,
add scorer output and a separate HUD/video review to a new result card:

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.flight_card --manifest runs/attempt-manifest.json --out runs/attempt-preflight.md
.venv/Scripts/python.exe -m haltere.liftoff.flight_card --manifest runs/attempt-manifest.json --score runs/attempt-score.json --review runs/attempt-review.json --out runs/attempt-result.md
```

The review records `finish_confirmed`, `finish_evidence` (saved image/video paths),
observed laps/time, resets, interventions and observations. The tool preserves
the manifest and source hashes, retains terminal impacts even when the CSV
contact list is empty, and refuses to overwrite prior cards. It does not infer
finishes from telemetry or invent a preflight prediction after a result. Without
game evidence a result cannot become a confirmed finish. The Green repeat was
re-rendered from its preserved preflight manifest and score as a regression check.

## 2. Isolate motor control from guidance

Compare the current brain with a calibrated PD or trained MLP motor controller
on 3–5 development courses under the same frozen race-cue pilot. Match the
camera, detector, target limits, speed setting, yaw assistance, vehicle, starts,
run limits and scoring. Preserve actual motor commands and controller identity.
Tune a baseline only on simulation/development cases before freezing the run.

Record target-tracking error as well as finishes, speeds and impacts. Similar
results would suggest perception/planning is the dominant bottleneck; improved
baseline finishes would implicate motor control or its operating range. Neither
outcome proves a universal architectural conclusion from a few flights. A
baseline flight must not be displayed or published as brain-controlled flight.

## 3. Give the system a representation of free space

Evaluate causal FPV depth fused with recent telemetry into a short-range obstacle
map, including uncertainty and unknown space. Plan a feasible local trajectory
toward the task, accounting for vehicle size, braking distance and directed gate
crossing. A marker bearing does not establish free space, range or gate attitude.
Develop on existing failure courses, then compare geometry enabled/disabled on
one frozen batch. Check depth scale, edge errors, thin obstacles and latency
offline before granting live authority. Preserve the same runtime safety limits.

The current channels called optic flow use angular velocity and velocity divided
by an altitude proxy in `haltere/sim/tasks.py`; they are **not image optical flow**.
Test real image motion and looming as a separate causal sensory experiment,
compensating camera rotation and logging freshness. Do not silently change the
sensory contract of existing trained weights or assume metric depth from flow.

## 4. Broaden training after the comparison

Randomize layouts, obstacles, starts, dynamics, image appearance and disturbances
in simulation. Train a task-conditioned local planner to propose roughly one
second of trajectory, and train the connectome to track it. Extend the verified
speed/attitude envelope toward 5–10 m/s and sharp turns before adding acrobatics.
Keep wiring and known transmitter signs; report exactly which brain weights
changed. Planner-only learning is not brain learning. Verify transfer in stages
on the original calibrated Liftoff drone before claiming those speeds in game.

The existing motion predictor remains permitted at runtime but is deprioritized.
It has no task input and only small recorded path-error gains over motion-only
prediction. No live benefit has been established. A task-conditioned planner is
the proposed next experiment, not a capability already implemented.

## 5. Declared freestyle tasks

These are initial acceptance targets, **not demonstrated capabilities**. Freeze
task parameters, starts and layout family before each trial; no race markers or
stored routes are allowed. Use video plus synchronized telemetry. Scene targets
must be chosen from current observations. Offline geometry may score clearance.
All tasks require no contact, reset or human flight intervention and a stable
recovery: five seconds upright within 15 degrees, speed below 1 m/s and vertical
speed below 0.5 m/s. Report failures to enter that recovery explicitly.

| Task | Declared request and pass condition |
|---|---|
| Hover and recover | Hold a requested visible open region for 20 s; stay within a 2 m radius and ±1 m altitude after settling, then recover. Repeat from varied initial speeds/headings. |
| Fly a gap | Cross one visually selected opening with at least 0.5 m vehicle-envelope clearance; stop in a visible free region within 10 s after crossing and recover. |
| Underpass | Select and traverse beneath a visible bridge/overhang, preserving 0.5 m clearance above and below, without flying around it; exit into free space and recover within 15 s. |
| Orbit | Complete one continuous 360-degree orbit around a selected visible object, retain at least 1 m clearance and finish within 2 m of the starting orbit position; recover. |
| Climb and dive | In visible free space, climb 10 m, descend at least 8 m with a sustained nose-down segment of at least 45 degrees, then arrest descent with at least 3 m ground clearance and recover. |
| Roll | Complete one unwrapped 360-degree roll within 3 s while airborne in a declared clear volume; lose at most 3 m altitude and recover. |
| Flip | Complete one unwrapped 360-degree pitch rotation within 3 s in a declared clear volume; lose at most 3 m altitude and recover. |

Ground clearance is relative to observed terrain, not launch altitude. If it
cannot be measured reliably, the clearance criterion is unverified. Liftoff's
aggregate freestyle score is supplemental; a score increase alone does not
identify the requested maneuver, its clearance or a clean recovery.
