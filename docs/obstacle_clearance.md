# Obstacle clearance: milestone 1 (offline bench and first model)

Goal: a causal, learned "free distance per direction" signal that lets the fast
race-cue pilot steer around obstacles on or beside the line to a checkpoint
(the Minus Two pillar, the Pine Valley boulder and trees), for PD and brain
motors alike. Plan chosen by a judged design review (learned clearance fan with
offline hindsight labels, beating self-supervised depth and a faster sparse
geometry map on expected benefit and honesty of evaluation).

`haltere/obstacles/` is offline tooling only; nothing in the flight runner uses
it yet, and runtime modules never import the label code (tested).

## What exists

- **Frame store** (`store.py`, `store_build.py`, `timing.py`): 135,538 posed
  448x252 frames from recorded flights in 11 environments at
  `runs/obstacle-store-v1` (not in git). 59 of 74 good/fair videos align to
  telemetry within 1 px median residual.
- **Folds** (`splits.py`, `configs/obstacles/folds.json`): leave-environment-out.
  F12 holds out Minus Two, Pine Valley and Autumn Fields; The Green and Hall 26
  are sealed for a final check.
- **Overlay masks** (`overlays.py`): HUD, propellers, checkpoint ring and other
  racers' ghost trails, which are never used as inputs or labels.
- **Offline labels** (`labels/`): flown-volume free bounds, hindsight
  triangulation, generated-course colliders, impact events, a pretrained
  relative-depth teacher. On the box course, hindsight labels match the colliders
  within 5.1% of range (median).
- **Bench** (`evaluate.py`, `baselines.py`, `leaks.py`): distance at impact
  points, blocked-direction accuracy, correct-side lead, clean-flight false
  alarms, shortcut tests, deployed 65 ms latency, frozen thresholds and a
  score-once ledger.
- **Model** (`model.py`, `train.py`): ClearanceNet v0 (Depth-Anything-V2-Small
  relative backbone, Apache-2.0) with a distance grid and a 36-direction fan.

## Result: the first model is not usable

Held out on Minus Two and Pine Valley (scored once, thresholds `8469884606a7`):

| | Range ratio 2-4 m | Flown path read as blocked | Correct side (13 events) | Clean activations/min |
|---|---:|---:|---:|---:|
| ClearanceNet v0 | 0.63 | 42% | 0 | 11.6 |
| ResNet18 control | 1.90 | 36% | 0 | 9.4 |
| Pretrained metric depth (B2) | 2.74 | 0.9% | 2 | 8.9 |
| B2 + global correction (B3) | 3.21 | 0.2% | 2 | 6.1 |

With the target environments held out, F12's training side is essentially Straw
Bale and the Drawing Board courses: the model learns their distance scale and
does not transfer. The shortcut tests also failed (painting a ring changes the
output), traced to a validity input channel frozen with the backbone.

An independent audit found the bench sound (no leakage, results reproduce,
every held-out result scored once) with two fixes needed before the next
milestone: impact reference points must lie on the obstacle surface (most were
placed in free space beside side-clipped obstacles), and held-out label
coverage is limited (Pine Valley 24% of near fan cells decided; impact
consistency 57% Minus, 40% Pine).

## Next

1. Fix the impact points and occlusion check; re-freeze thresholds and rescore
   the baselines.
2. Broaden training environments: data from more Liftoff maps is the main lever
   (a scripted collection campaign, only when the user is not gaming: the virtual
   gamepad is machine-wide and GPU training slows their games).
3. Un-freeze the validity channel; re-run the shortcut tests.
