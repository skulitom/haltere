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
and holes between samples remain limitations. **These modules have no live
steering authority.** The optional `--geometry-shadow` runner mode below can
measure their live timing in a separate passive process.

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
optimized worker still needs its own live timing check before control integration.
