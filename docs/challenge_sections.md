# Procedural obstacle sections

`challenge-sections-v1` creates Drawing Board courses in code. Each section has
an entry and exit checkpoint, a 5 × 5 m opening with physical cube frames, and
one of six challenges: a nearby flag, pillar, descent below the launch height,
high checkpoint, overhang or wall hiding the next marker. Seeds vary lateral
offsets and obstacle side; difficulty changes clearance. All seeds and the
calibration variant belong to the **same development family**.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.challenge_courses --seed 123 --difficulty 0.4 --out runs/sections-123
.venv/Scripts/python.exe -m haltere.liftoff.course_pool install runs/sections-123 --local-root "$env:USERPROFILE/AppData/LocalLow/LuGus Studios/Liftoff"
```

Generation validates XML references and writes deterministic new track/race IDs,
file hashes and a separate `offline-geometry.json`. Installation refuses to
overwrite existing content. Install before the game's next content refresh,
then verify rendering, directed checkpoint progression and playability in Anode.
**Passing file and geometry tests does not qualify a course as playable.** The
initial six-section development bundle is installed locally. The box-calibration
variant rendered and started correctly in Anode. Its first PD run crossed five
sections geometrically, then hit the final wall: **0/1 autonomous completions**.
A separate oracle-guided PD run finished the box course in **3:35.704**, qualifying
that exact course as playable. The flag variant remains unqualified. Do not
report these seeds as unseen layout families.

## Score attempts without inventing finishes

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.section_scoring --bundle runs/sections-123 --log runs/section-attempt.csv --out runs/section-attempt-score.json
```

The scorer checks the frozen geometry hash, restores the log's absolute world
origin and tests directed checkpoint-plane crossings in order. It distinguishes
estimated successful sections, impact failures, incomplete sections, runtime
censoring and sections never attempted after a crash. Telemetry gaps stop
geometric scoring; resets require separate attempts. Report whole-course results
alongside obstacle-type results because sections within a flight are correlated.
These geometric estimates **never confirm a game finish or checkpoint count**;
retain a separate HUD/video review and finish screenshot.

## Offline depth calibration

`section_geometry.py` casts rays against measured box colliders. The dimensions
were inspected in the installed Liftoff 1.7.6 prefabs: one-metre cubes, ten-metre
cubes and `DrawingBoardWall5mx5m05`. Despite its name, that wall's collider is
**10 × 5 × 1 m**, with its pivot at the base. This matters for clearance labels.
The ground plane and collider surfaces still require comparison with rendered
images; collision geometry is not automatically exact rendered depth.

Unknown flag meshes remain explicit in the scene contract. Complete depth
generation refuses those scenes rather than treating missing obstacles as free
space. Generate a distinct box-only calibration scene when testing projection:

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.challenge_courses --seed 123 --box-calibration --out runs/boxes-123
```

This variant has a separately identified box obstacle in place of the flag and
new IDs. It is not a flag test. `camera_collision_depth` uses the calibrated
camera, logged absolute position, attitude and camera offset to return both
distance along each ray and optical-axis depth. Pixel centers, camera tilt and
coordinate conversion are explicit. Unhit rays remain unknown; neither they
nor unvalidated primitive boundaries establish safe space.

Autonomous visual modes do not import these geometry files. They are for
post-flight scoring, checking causal image-depth predictions, and the explicitly
privileged qualification/collection mode below. A live visual geometry
layer must derive obstacles from camera observations, account for uncertainty,
vehicle clearance and braking distance, and pass a frozen on/off flight comparison
before any improvement is claimed. See the [generalization program](generalization_program.md).

For aligned passive image collection, the visual runner can copy its telemetry
to another local port with `--telemetry-copy-port 9011`. Start `haltere liftoff
capture` on that port; two receivers must not compete for the game's 9001 stream.
The visual runner records the forwarding port. Keep the collection labelled
with its actual controller and assistance mode; image/UDP receipt alignment does
not by itself measure physical display latency.

## Current camera-geometry experiment

The [2026-09-23 obstacle diagnostic](flight_cards/2026-09-23_obstacle_geometry.md)
retains the failed flight, video and 599 passive images. A pretrained metric-depth
candidate produced inconsistent scale and remains outside flight control.

`temporal_depth.py` instead tracks actual image features and triangulates them
from measured motion. It rejects insufficient translation, incompatible rays,
behind-camera points and tracks that fail a third-view reprojection check.
`geometry_mask.py` excludes the configured HUD and original drone's propellers;
it also sacrifices some bright/green scene features. `surface_memory.py` forms
short-lived finite surface hypotheses. Neither missing points nor positive
clearance certify free space; triangulation noise, pose timing, thin obstacles
and holes between samples remain limitations. Early geometry flights were
passive. The optional `--geometry-shadow` runner mode below measures live
timing in a separate process; experimental control is a separate explicit mode.

Reproduce the causal replay, optionally scoring predictions against the separate
offline collider file after each frame's prediction. Use a new output directory:

```powershell
.venv/Scripts/python.exe -m haltere.vision.geometry_replay --dataset data/vision/challenge_boxes_pd_20260923 --out runs/geometry-replay-check --offline-labels runs/challenge-box-calibration-20260923/offline-geometry.json --stop-timestamp 136.57421875
```

The stop timestamp excludes the terminal impact sample from this particular
recording; it is evaluation metadata. Omitting `--offline-labels` produces the
same geometry and warnings without reading course geometry. The 1.2-second
constant-velocity query is a diagnostic, not a dynamically feasible planner.
On this development replay it warns about the wall 1.98 s before impact, but
also warns during successful passages. Broader calibration and a frozen live
on/off comparison are still required.

## Privileged course qualification and collection

`qualification_route.py` computes a route through the generated checkpoints and
around known box colliders. It rejects unknown meshes and changed geometry
hashes, checks directed gate order and samples the simplified path for model
clearance. This is an **oracle**, used to check course playability and collect
images; it is never an autonomous visual race result. Geometry clearance alone
does not account for all motor tracking error or qualify the course in game.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.qualification_route --bundle runs/boxes-123 --out runs/boxes-123-collection-route.json
```

The visual runner's explicit `--collection-route` option requires
`--motor-controller pd` with the visual pilot mode left at `none`. It uses the
newest compatible brain only in shadow, labels the video **ORACLE ROUTE**, marks
metadata and neural replay as privileged, and validates the stationary reset
against the route's original coordinates. Passive capture must also receive
`--teacher-route` pointing to that same file. Keep this dataset in development
and separate from autonomous scores. Existing visual modes still load no route.

The first qualification attempt stopped before takeoff on a 4.9 s controller
stall; the unchanged route/motor retry stopped at 32.7 s on a 129 ms telemetry
gap. Neither establishes a finish. A 20 s stationary shadow recording passed
between them. Loop-stage diagnostics and an explicit camera-rate cap support
runtime investigation; neither weakens the existing freshness or stop limits.
A subsequent **60 s shadow/video check at 16 camera fps passed 5,994 ticks**
without camera or controller-deadline failure. This motivates a reduced-load
collection attempt; it does not establish the cause of either prior stall.

The third attempt, at 16 camera fps with the same route and PD settings,
**finished the complete one-lap box course in 3:35.704**. The game results screen
confirms the finish, with no detected impact, reset or flight intervention. A
134 ms telemetry gap triggered the guard at the finish transition; retain that
runtime stop. Across the series there was one game finish in three attempts,
with two earlier runtime-censored attempts, not a frozen performance comparison.
The endpoint beyond the finish line was not reached because the game ended.
Only five sections are credited by the geometric scorer because the controller
CSV ends just before the last plane; the independent game result establishes
whole-course completion. See the [evidence record](flight_cards/2026-09-23_obstacle_geometry.md).

The complete run supplied 1,123 passive images. Applying the unchanged geometry
prototype to the 1,122 pre-transition images gave 213 frames with accepted
points and no warnings. On 1,292 primitive-collider matches, median range ratio
was 1.000 and mean absolute relative error was 6.7%; 0.93% overestimated by more
than 25%. This tests a second trajectory in the same development course, not
an unseen family or live obstacle avoidance. Sparse observations remain a limit.

## Passive local-planner development

`local_trajectory.py` proposes velocities using a bounded-acceleration point-mass
rollout, a reaction interval, and a braking tail. Alternatives are checked against
observed surfaces and the camera's current view; missing geometry never certifies
free space. Its response assumptions are not yet validated motor dynamics.

`trajectory_replay.py` reads a matching causal geometry cache and past controller
goals. It never changes the recorded states or claims a counterfactual finish:

```powershell
.venv/Scripts/python.exe -m haltere.vision.trajectory_replay --dataset data/vision/challenge_boxes_pd_20260923 --geometry-replay runs/depth-calibration-20260923/reproducible-v5 --log runs/challenge-box-pd-20260923/flight.csv --out runs/trajectory-check
```

On the failed obstacle trajectory, the first prototype proposed changes in 62
frames. Offline collider checking found 12 proposals with positive observed
clearance that still intersected an unobserved surface. Restricting new detours
to the camera view reduced that count to one. A subsequent review located that
overlap at the initially occupied launch platform: the conservative sphere
starts 0.258 m inside it, exits after 0.75 s, and never reenters. Both original
audits and this correction are retained. Sparse coverage and the surrogate
motion model still prevent a safety or closed-loop improvement claim. These
proposals are correlated development diagnostics, not successful flight attempts.

The visual runner's explicit `--geometry-shadow` option is currently restricted
to PD diagnostics. A separate process captures at 5 fps, reads a bounded live
pose/goal history, and writes `.geometry.jsonl` and `.geometry.json` beside the
flight log. Sharing never waits on a lock, and **no proposed action returns to
the motor controller**. This permits timing and perception checks during a
flight without interpreting them as geometry-enabled navigation. Oracle source
goals remain explicitly labelled privileged.

The first live observer check passed 60 s on the ground. A separate complete
oracle flight then finished in **3:35.708**, with no detected impact or camera/
controller-deadline failure. The observer processed 1,130 frames, recovered
current points in 211, and held recent points in 572. It made 14 uncommanded
detour proposals. Total update time was 46.3 ms median, 69.6 ms p95 and 284.5 ms
maximum; sparse coverage and occasional slow proposals remain visible. The
telemetry guard stopped on `live_pose=false` at the finish transition. This
checks runtime alongside one slow oracle flight, not autonomous obstacle avoidance.

Subsequent batched rollout evaluation preserved every proposed velocity and
status across the failed trajectory's 598 frames while reducing replay planner
time from 133.6 to 41.3 ms p95 (211.0 to 114.5 ms maximum). A second 1,122-frame
replay also retained its decision counts. These are replay measurements; the
optimized worker was subsequently measured in the matched live comparison below.

## Experimental control boundary (not promoted)

`--geometry-control` explicitly enables causal image-geometry guidance around
the PD diagnostic motor controller. It requires `--pilot-assistance race-cue`,
refuses an oracle collection route, and cannot be combined with shadow mode.
This remains an experiment with limited development evidence. The newest brain
runs in shadow and the video labels both PD control and visual geometry.

The control boundary accepts only finite proposals with current image/task
timestamps, matching position and a compatible current task velocity. Rejected
or unavailable proposals request braking; prolonged unavailability pauses the
attempt. Existing image, telemetry, impact and flight-limit guards remain.
The unmodified task goal is recorded and supplied to perception even while a
detour is commanded, preventing a detour from becoming its own objective.
The worker archives its exact resized input images losslessly, capture poses,
accepted depth points and proposed velocities for diagnosis. No route, course
geometry or future pose enters perception or planning.
Use `--geometry-shadow --geometry-record-images` for a passive comparison with
the same input archival workload as the control experiment.

The first [frozen off/on comparison](flight_cards/2026-09-23_geometry_control.md)
finished 0/1 off and 1/1 on, with a 2:26.657 full development-course finish.
Both conditions retained exact worker input images and the same recording work.
Two unchanged assisted repeats left 2/3 finishes overall; obstacle memory expired
before the failed repeat had passed the wall. The failure remains in the record.
Passing the control-boundary tests and one course does not validate sparse free
space, motor-model accuracy, reliable transfer or main-track speed.

The next experimental revision retains observed static surfaces while they are
within 12 m, instead of deleting them after a three-second depth gap. Braking
can remove the parallax needed for new depth, so elapsed time alone is not
evidence that a nearby obstacle disappeared. Original observation ages remain
logged. The map is bounded to 512 voxels, evicts farthest points first when full,
and reports capacity evictions. Identical surface patches are reused until the
points or uncertainties change. This assumes stationary Liftoff scenery in the
telemetry frame; false points can persist, and no free-space certificate or new
flight success follows from the memory change alone. The older time-limited
memory remains the default for historical offline replays. That first persistent
revision finished 0/3: one freshness stop and two operator-stopped stalls.

The transient-patch revision kept measured points but required support within
three seconds for interpolated patches. It also finished 0/3: two declared stall
stops and one wall impact. No geometry freshness, camera or controller deadline
failure occurred. Expiry alone did not solve the coverage/recovery problem.

The current experiment retains local static surfaces and adds explicit recovery
from an initially violated clearance constraint: each overlapping observation
must recede within 1 cm slack, other obstacles retain their required clearance,
and the full braking endpoint must be clear. Initial overlap remains reported;
the observations are not deleted. When nominal travel is blocked, a feasible
moving detour takes precedence over permanent braking. Vertical alternatives,
existing speed/view limits and original receipt ages remain intact. This is
still incomplete observed geometry, not a free-space certificate. Its frozen
batch finished 2/3 (2:28.763 and 2:39.492), with one declared wall stall and no
detected impacts or runtime failures. The subsequent minimum-motion revision
finished 1/3 (2:31.664), with one wall stall and one wall impact. It did not
demonstrate an improvement.

### Consistent detours and longer image baselines

The stalled flight's last minute contained 107 reversals between upward and
downward proposals. The drone remained almost stationary despite the 0.3 m/s
minimum command. The revised planner treats clearance as a required constraint
and penalizes changing a recently feasible detour when the task remains similar.
Every continuation still receives fresh reaction, surface, braking and view
checks. A short 0.5 m/s retreat is also considered when forward choices cannot
leave an uncertain overlap. No route or course geometry supplies these choices.
On the same recorded poses, the consistency change reduced vertical reversals
from 107 to 1. This replay does not simulate the resulting motion or prove a
completed flight.

The image worker retains its existing 0.7 s tracker and adds a second reference
of up to 2 s. Additional points require three-view agreement, uncertainty below
10% of range and below 0.5 m; duplicate corners retain the more precise estimate.
Across five recorded development approaches, the combined tracker added points
with similar collider-relative depth error. Its 95th-percentile tracking time
was 25--40 ms in that offline check. Missing pixels still remain unknown.
Logs identify the added points and both reference-image limits. The full suite
passed 452 tests; complete-system flight qualification remains separate.

A parallel offline check of frozen Depth Anything V2 Small relative and metric
models did not justify runtime integration. The strict relative-depth alignment
had insufficient reliable anchors on all 25 selected frames. The unscaled
outdoor metric model substantially overestimated some wall distances; the indoor
model also retained material errors. These models have no control authority.
Local probes and model revisions are retained in
`runs/dense-depth-probe-20260923`, `runs/depth-keyframe-probe-20260923` and
`runs/depth-multibaseline-probe-20260923`.
