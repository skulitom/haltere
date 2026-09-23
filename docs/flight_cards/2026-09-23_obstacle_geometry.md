# Generated obstacle and camera-geometry diagnostic

[Untrimmed standard brain/gameplay video and reproducible evidence archive](https://github.com/skulitom/haltere/releases/tag/obstacle-geometry-diagnostic-20260923).

**Result: 0/1 complete courses.** The PD baseline crossed five of six sections
geometrically, then hit the wall hiding the last marker. The game showed lap
1/1 and **1:55.611**; no finish was observed. There were no resets, operator
flight interventions, camera failures or control-deadline failures. The impact
guard correctly stopped the attempt 6.94 m below the launch elevation.

This is a development course, `Haltere box calibration v1 20260923`, in the
Drawing Board. All generated section variants belong to one layout family.
The nearby-box variant does not test a flag, and five geometric crossings do
not qualify the entire course as playable. No runtime course geometry was read.

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
