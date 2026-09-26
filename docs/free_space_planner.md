# FreeSpace corridor planner v1 (perception half)

Branch `m3-freespace` (from `m2-hairpin`). The user asked to "configure our model to
adapt the flight path to avoid obstacles": steer around obstacles toward the
checkpoint (sideways or up) instead of only braking or climbing. The judged merge
(ScaledCorridor base with RSCP-2D grafts) splits the work into a perception half
(this branch) and a pilot half (`CorridorAim`, `VerticalGuard`, FastRaceCue wiring,
pilot replay: branch `m3-pilot`). Everything is off by default.

## What is built here

- `haltere/vision/free_space.py`: `FreeSpaceConfig`, `ScaleTracker`,
  `FreeSpacePlanner.process` (spec section 4, P1-P8). Inputs are causal only: the
  current 448 x 252 frame, its DA-V2-Small block disparity, the overlay masks (hud,
  ring, ghost, propeller), attitude and world velocity at capture, the checkpoint-ring
  cue of the same frame and the declared motor. Numpy at import, OpenCV only for tracks.
- `haltere/liftoff/gap_stack.py`: `plan_spec` (refuses anything but the frozen v1,
  and `on` under `--obstacle-stack shadow`), `PlanStage` (worker counters and
  timing percentiles for the sidecar), `pose_for_plan` (`no_pose` when the nearest
  observed pose is more than 0.05 s from the capture). The depth process runs the
  planner after the gap decision on the same frame, depth and cue, and writes its own
  shared array `plan_out`; a cue that does not arrive within 0.15 s gives `stale`.
- `haltere/liftoff/camera_process.py`: `PLAN_FIELDS`, `PLAN_KINDS`, `plan_sample`,
  `plan_values_of`, and a non-blocking controller read (`ProcessRetinaCamera.plan`,
  busy-lock skips counted). The planner is refused in the camera placement.
- `haltere/obstacles/free_space_eval.py` (offline only): decoding and depth of the
  flights not cached yet, plan streams (`streams/<flight>.plan.npz`, one array per
  field plus `pub_time = time + 0.10 s` for the pilot replay), and gates S1-S3, A1,
  B1, T1, Q1, L1.
- Configs: `configs/obstacles/free_space.json` v1 and
  `configs/obstacles/free_space_gates.json` v1, frozen before any held-out frame was
  scored (hashes below).
- Tests: `tests/test_free_space.py` (synthetic scenes: floor, pillar, wall, ceiling,
  mound / pillar / masked-base wall, HUD-only disparity; pure rotation; synthetic
  translation; a textured LK wall; the scale guard; the interface round trip; the
  depth-process worker) and `tests/test_obstacle_label_isolation.py` (free_space,
  the worker and the controller fields reach no label, torch, evaluation or bench
  code; `corridor_aim` is checked once it is on the branch).

## Implementation choices where the spec leaves room

- Inverse depth of a track uses the derotated previous position q (the current
  frame's depth, exact for a static scene) and the pair's mean world velocity.
- Scale-fit weights are normalised per frame (as the design study's `lkscale.py`)
  before the normal equations are pooled; a frame contributes its own fit only with
  at least 8 anchors.
- The ring corridor's clear test (P6.1) is the corridor test alone. Eligibility
  (image margin, footprint validity) applies to the escape candidates and to the
  blocked `v_cap`: the ring overlay masks the ring's own footprint.
- Paths are parameterised by the course coordinate x, with headings clipped at 80
  degrees; lateral profiles depend on the candidate azimuth only, which the corridor
  test uses to halve its work.
- Rising ground walks the data bins (bins with at least 2 points) in x order; a step
  over 0.5 m between consecutive data bins ends the reachable chain; `rise` is the
  maximum over the chain from the first rising bin.
- `h_floor`, `h_ceil` and `rise` are also published for `no_ring` and `stale`
  samples when a scale exists (diagnostics); the pilot uses valid samples only.
- `looming2.hud_mask` is reproduced with OpenCV channel min/max (identical output,
  tested) to keep the tracks inside the time budget.

## Frozen versions

| Config | Version | sha256 | Frozen |
|---|---|---|---|
| `configs/obstacles/free_space.json` | 1 | `532c2eddd37e72f12862b70551a80668f04f3fba7e8713905d397c6f3b13dd8d` | 2026-09-26T12:35:03, commit 8c936f2 |
| `configs/obstacles/free_space_gates.json` | 1 | `8c2ccf247bc7b88d5ea500fb51e60f0a1eb26c83b7a2d4111a9f99ab20570305` | same commit |

Before the freeze the planner ran on dev data only (minus-brain08-01,
straw-brain08-04, store runs 118 and 122). Nothing was tuned after scoring; v1 is the
only version. No version has been flown.

## Offline gate results (v1, each gate scored once)

| Gate | Result | Verdict |
|---|---|---|
| S1 floor scale | h_floor / telemetry height: median 1.85 (p10 1.03, p90 2.31), 30 % in [0.67, 1.5]; only 40 samples (none on minus-brain07-01 and minus-fast6-cur-01) | **fail** |
| S2 pillar range | pooled median 0.93; per-frame medians in [0.67, 1.5] on 95.1 % of 81 frames (2159 blocks) | pass |
| S3 coverage | valid share: Minus Two 0.989 (1507 frames), Straw Bale 0.936 (12,486), Pine Valley 1.000 (77) | pass |
| A1 pillar A | 1 of 5 approaches pass (minus-brain08-slow35-01: LEFT at 13.7 m); a RIGHT confirmation beyond 2 m on all 5; the applied aim inside the pillar's extent on 12 ticks (slow35-01) | **fail** |
| B1 pillar B | a -x shift confirmed 1.07 s (store 123) and 1.15 s (store 127) before impact | pass |
| T1 Pine trunk | first confirmed RIGHT 0.11 s before impact (first raw RIGHT 0.55 s); LEFT confirmed in the last 2 s (the gap cue: 0.66 s) | **fail** |
| Q1 clean Straw Bale | 8.68 episodes/min (limit 6), vertical 0.76 (limit 2), blocked/urgent 7.81 (limit 1), terrain climbs 6.70 (limit 1), applied az p90 20 deg (limit 10), quiet checkpoint switches 93.3 % (limit 95 %); floor bound active 0.07 s/min (report) | **fail** |
| L1 leaks | same decision on 96.3 % of 4158 perturbed frames (limit 97 %; painted 97.2 %, removed 95.3 %), 68.5 % where either decision acts (limit 90 %); HUD-only disparity: 0 of 21,837 frames blocked or shifted | **fail** |
| R1 runtime and parity | straw-brain08-06 30-90 s, below-normal parent, Liftoff running idle in the Anode seat: camera 16.65 Hz, plan age at the controller p95 131 ms, checkpoint-cue latency p95 -0.2 ms against the baseline, plan_ms p95 3.4, no camera skips; **lk_ms p95 15.5 (limit 8)**; **parity: same kind 81.1 % (limit 95 %), angle p95 20 deg (limit 1)** | **fail** |

V1-V3 (pilot replay) and I1 belong to the pilot half. Plan streams for them
(`<flight>.plan.npz`: one array per PLAN_FIELD plus `pub_time`, frozen v1) are in the
evaluation scratch directory `m3/impl-a/streams/` for every gate flight, including
minus-brain08-gapon-01/-02, minus-fast6-wall-01 and pine-fast6-ttc-01. The report-only
store environments were not scored: store frames are kept at stride 3, so Hannover
and Paris have no LK pair inside 0.03-0.15 s (0 of 1749 and 0 of 646), The Pit 566 of
7286 and Hangar C03 263 of 535; their rates would measure frame spacing, not the
planner. Full numbers: `docs/experiments/free_space_v1_results.json`,
`free_space_v1_r1.json`, `free_space_v1_r1_parity.json`.

What the failures mean (spec section 9): S1 failed, so no authority flights, only
shadow flights. A1 failed (no authority on Minus Two), T1 failed (none on Pine
Valley), Q1 failed (v2 adds RSCP's expansion veto under a new version and freeze). Any
authority flight of v1 would be a deviation that needs the user's explicit go-ahead.

## Why it failed (diagnosis on the scored data; v1 unchanged)

- **A1: the left escape is unobserved, not blocked.** On the five approaches, at 3-9 m
  from the pillar, a static-free LEFT candidate exists in 49 of 67 frames but is
  eligible in 19: the propeller-zone and HUD masks (10.9 % and 12.5 % of all blocks)
  cover the lower left of the image, where a left escape from the low-left ring points.
  The median valid footprint of those candidates is 0.04-0.41 against the 0.6 rule, so
  the planner proposes a saturated RIGHT shift. The pillar itself is seen (S2 passes).
  The gap cue does not mask the propeller zone; the spec added it for this planner.
  `free_space_v1_a1_diagnosis.json` has the per-frame counts.
- **S1: the near floor reads too far.** With the scale fitted to the whole frame's
  anchors, DA's disparity of the floor 1-3 m ahead at the image bottom reads 1.5-2x the
  telemetry height (on dev the track-only floor height was 0.8x). The floor inside 3 m
  is also rarely visible and unmasked (no sample on two approaches).
- **Q1: blocked and saturated proposals, and the rise rule.** Most Straw Bale episodes
  are blocked or urgent (7.8/min), and rising ground fires on the Straw hills (6.7/min).
- **T1:** the trunk blocks the ring corridor only 0.94 s before impact; the first
  proposal is a saturated LEFT, flipping to RIGHT at 0.55 s (thin trunks are the known
  v1 limit).
- **L1:** removing or painting a ring changes the masks around the ring, and with them
  the footprint validity and the cloud, in frames where the planner acts.
- **R1:** 600 LK corners on textured Straw Bale frames cost 15.5 ms at p95; the live
  planner pairs each frame with the previous bench frame and uses the live cue, so
  its decisions differ from the offline stream on 19 % of frames.

Candidate v2 levers (not implemented; each needs a new version and a new freeze):
drop the propeller layer from eligibility or use sub-block masks; RSCP's expansion
veto; a separate floor model or no bottom rows in `h_floor`; fewer LK corners.

## Dev data (report only, run before the freeze)

minus-brain08-01: first confirmed LEFT at 3.7 m (a saturated RIGHT first); store 118
(minus-brain03-01): no LEFT; store 122 (minus-fast6-01): a blocked state with an urgent
LEFT option at 10.9 m; straw-brain08-04: 8.3 episodes/min. The spec's dev numbers
(ScaledCorridor: straight corridor confirmed at 8.8 m) did not carry over once masks,
eligibility and the lag-aware push-out were applied as specified.

## Reproduce

```
OUT=<scratch>/m3/impl-a  CACHE=<scratch>/m2impl/gapcue  LOCK=<flight lock path>
python -m haltere.obstacles.free_space_eval frames --out $OUT --flight-lock $LOCK
python -m haltere.obstacles.free_space_eval depth  --out $OUT --flight-lock $LOCK --max-s 300
python -m haltere.obstacles.free_space_eval run    --out $OUT --cache $CACHE --names <gate sequences>
python -m haltere.obstacles.free_space_eval score  --out $OUT --cache $CACHE
python -m haltere.obstacles.gap_bench run --out $OUT/bench --flight-lock $LOCK --condition baseline|plan \
    --align-npz $CACHE/flights/straw-brain08-06.npz --parent-priority below-normal
python -m haltere.obstacles.gap_bench plan_parity --out $OUT/bench --tag straw-brain08-06_plan \
    --eval-dir $CACHE --stream $OUT/streams/straw-brain08-06.plan.npz
python -m haltere.obstacles.free_space_eval r1 --out $OUT --bench $OUT/bench
```

The decode took one CPU pass (six flights, about 30 s); the new depth one GPU chunk of
12 s (GPU 45-50 C); the R1 bench two 60 s replays (GPU peak 50 C).

## Offline confirmation

A1, B1, T1 and Q1 are confirmed by the section-6 CorridorAim. The pilot half is on
another branch, so the evaluation steps `free_space_eval.SpecCorridorAim`, a replica
of section 6 (confirm 2 of 3 fresh samples within 0.25 s, latch at least 0.6 s while
the ring path is blocked, flip after 3 samples without the latched option, release
after 2 clear samples or 0.3 s without a fresh sample, slew 40 / 20 deg/s, decay
0.3 s, 6-degree ring conflicts), on a 50 Hz tick grid with samples published at
capture + 0.10 s. Flag clearance, turn-first and suspension are not modelled offline.
