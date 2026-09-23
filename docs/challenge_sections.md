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
initial six-section development bundle is installed locally; game verification
is pending. Do not report its seeds as unseen layout families.

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
