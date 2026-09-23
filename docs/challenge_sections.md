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
sections geometrically, then hit the final wall: **0/1 complete courses**. Full
playability and the flag variant remain unqualified. Do not report these seeds
as unseen layout families.

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

The runtime controller does not import these geometry files. They are for
post-flight scoring and checking causal image-depth predictions. A live geometry
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
steering authority and are not imported by the flight controller.**

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
