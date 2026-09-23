# Generated obstacle and camera-geometry diagnostic

[Untrimmed standard brain/gameplay video and reproducible evidence archive](https://github.com/skulitom/haltere/releases/tag/obstacle-geometry-diagnostic-20260923).

**Result: 0/1 complete courses.** The PD baseline crossed five of six sections
geometrically, then hit the wall hiding the last marker. The game showed lap
1/1 and **1:55.611**; no finish was observed. There were no resets, operator
flight interventions, camera failures or control-deadline failures. The impact
guard correctly stopped the attempt 6.94 m below the launch elevation.

This is a development course, `Haltere box calibration v1 20260923`, in the
Drawing Board. All generated section variants belong to one layout family.
The nearby-box variant does not test a flag, and five geometric crossings alone
do not qualify the entire course as playable. No runtime course geometry was
read in this autonomous attempt. The subsequent oracle qualification is below.

The recorded stack was commit `fd7cbce079c874cebcfba3d1df11c9c1e1f91137`, original
`[Copy] New Drone`, visible race-cue pilot at 2.5 m/s, corrected PD motors and
motor10 candidate05 running **in shadow**. Its standard brain/gameplay video
fully decodes. The run was entirely inside Anode 0.8.0, session 5, with NVENC.
No new neural weights were trained or promoted.

Local evidence is preserved under `runs/challenge-box-pd-20260923`: the frozen
preflight card/manifest, CSV, sidecar, video, neural replay, section score and
terminal screenshot. The generated bundle is under
`runs/challenge-box-calibration-20260923`. Passive images and aligned telemetry
are under `data/vision/challenge_boxes_pd_20260923` (599 accepted, 3 rejected;
telemetry receipt age p95 38.7 ms; physical display delay unmeasured).

## Offline geometry findings

Depth Anything V2 Metric Hypersim Small was checked on static and moving frames.
Its moving-frame median prediction/collider ratios ranged approximately
0.73–6.57; a global scale correction is inadequate. It remains outside steering.
Code revision `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`, model revision
`3bc65d4e14a6786a61acec16453c50e12bf5f338`; pretrained weight SHA-256
`b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545`.
These are third-party pretrained weights, not learned fly-brain weights.

The causal image/motion prototype was improved by excluding moving white HUD
marks and propeller regions and requiring a third view to agree with each depth
hypothesis. The reproducible replay processes 598 pre-impact frames:

- 156 frames supplied accepted points; coverage remains sparse.
- 856 points matched the primitive collider model: median depth ratio **0.998**,
  mean absolute relative error **11.0%**, and **3.74%** overestimated by more
  than 25%. Points are correlated and labels approximate rendered surfaces.
- The 1.2-second constant-velocity query warned in **23 frames**. Ten were in
  the final wall approach, starting **1.979 s before impact**; 13 occurred during
  earlier successful passages. These are frame counts, not independent trials.
- Offline use of the subsequently flown path still produced 20 warnings; none
  remained when uncertainty was removed. Dropping uncertainty would therefore
  also lose the wall warning. Future poses were used only for this audit.

Older replay variants are preserved, including the initial false obstacles
across open gates. The latest replay and code hashes are under
`runs/depth-calibration-20260923/reproducible-v5`. Inclusive tracking/masking time
was 13.9 ms median / 18.6 ms p95; surfaces/query added 2.9 / 12.1 ms on this
paused-game CPU replay. This is not a live controller latency measurement.

Surface patches may interpolate small unobserved holes; missing surfaces never
establish clear space. The prototype has **no live authority**, and this replay
does not demonstrate obstacle avoidance, a feasible local planner, unseen-course
success or an improvement in race time. Next qualify uncertainty/coverage and
live timing before a frozen geometry on/off comparison.

## Subsequent oracle qualification and second-trajectory check

The box course is now game-qualified: **one full lap in 3:35.704** with the same
original `[Copy] New Drone`, PD motors and motor10 candidate05 **in shadow**.
The route was generated in code from the frozen box colliders and checkpoints,
including a path over the final wall. This is privileged collection, explicitly
ineligible for autonomous evaluation. It does not change the 0/1 autonomous
result above or the five-track acceptance status. No neural weights changed.

All three qualification attempts are retained:

| Attempt | Camera rate | Result |
|---|---:|---|
| Initial | 48 fps | 4.9 s controller stall before takeoff; no finish. |
| Same route/motor retry | 48 fps | 129 ms telemetry gap at 32.7 s; no finish. |
| Reduced capture workload | 16 fps | Game-confirmed **3:35.704** finish, no detected impact/reset/flight intervention; 134 ms telemetry gap at finish transition. |

This is **1/3 game finishes across differing runtime configurations**, with
the first two attempts censored by runtime stops. A 20 s ground shadow check
passed between the first two; a 60 s, 5,994-tick shadow/video check passed before
the third. Reducing camera work does not establish the cause of earlier stalls.
Freshness limits were unchanged. Preflight/postflight workload checks passed.

The completed flight used commit `dd09e5b`, candidate SHA-256
`64444a628c58b2c1615086bc84f8c5d3c0a0a12667df31d888c4587a802288b2`,
route SHA-256 `edde5445538c56daa747fbec070fa28cde8a9953c193f5a4ac30e5a6af8647fb`,
2 m/s requested speed and 1.5 m route lookahead. Actual median horizontal speed
was 1.19 m/s. The controller CSV stops just before the last checkpoint plane;
the last forwarded passive image crosses it, and the game results screen
confirms completion. Therefore the geometric scorer credits only five sections;
the independent finish evidence establishes the complete course. The waypoint
past the finish was not reached because the game ended.

Local raw evidence is in `runs/challenge-box-qualification-20260923`,
`runs/challenge-box-qualification-repeat-20260923` and
`runs/challenge-box-qualification-16fps-20260923`. Preflight manifests retain an
inherited `visible_race_cues: true` field; actual oracle control did not parse or
use those cues, as recorded in the flight sidecars. Those frozen manifests are
preserved with this correction. The original derived `sections-score.json`
incorrectly hard-coded no runtime geometry; the retained, corrected
`sections-score-privilege-aware.json` explicitly marks oracle use and ineligibility.

Passive collection from the completed run accepted 1,123 frames and rejected
three, with telemetry receipt age p95 36.0 ms. Physical display delay is still
unmeasured. The frozen geometry prototype from the failed-flight replay was
applied unchanged to 1,122 pre-transition images:

- 213 frames supplied accepted points; there were **zero warning frames**.
- 1,292 primitive-collider matches gave median range ratio **1.000**, mean
  absolute relative error **6.7%**, and **0.93%** overestimates above 25%.
- Tracking/masking took 14.8 ms median / 19.7 ms p95; surface/query took
  0.17 / 10.4 ms. This is CPU replay timing, not a live deadline measurement.

Results and source hashes are under
`runs/depth-calibration-20260923/oracle-trajectory-v5`. The final image at the
finish transition was excluded using a declared timestamp cutoff. This is a
second trajectory on the same known course family. Sparse coverage, uncertain
patch interpolation and earlier false warnings still prevent claims of safe
free space or autonomous obstacle avoidance.
