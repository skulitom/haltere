# Visual geometry control: first matched flight comparison

[All four complete brain/gameplay videos and raw evidence](https://github.com/skulitom/haltere/releases/tag/visual-geometry-control-20260923).

Source `732e140`, original `[Copy] New Drone`, motor10 candidate05 in shadow,
PD motors, causal visible race cues at 2.5 m/s. This is the already observed
generated box course, one lap, with a 500-second limit. It is development
evidence, outside the five main-track acceptance results.

The two conditions were declared together and flown off then on, without
changing source, weights, motor settings or recording workload. Both ran the
same geometry worker and archived its exact 640×360 inputs. Only the on
condition could apply its proposals. Neither consumed a stored route or course
geometry. Offline section scoring uses the generated colliders/checkpoints
after flight and cannot establish a finish by itself.

| Condition | Game result | Offline sections | Detected impact | Camera/deadline failure |
|---|---|---:|---|---|
| Geometry off | Incomplete; wall collision at race clock 1:56.918 | 5/6 | Yes | None |
| Geometry on | **Full finish, 2:26.657** | 6/6 | None | None |

The on condition stopped when the game changed to its results screen:
`live_pose=false`, with zero telemetry receipt/progress age. This is not an
in-race telemetry outage. Both complete standard brain/gameplay recordings
decode. No reset or human flight intervention occurred in either attempt.

This first comparison is **0/1 off versus 1/1 on** on one known layout. It
supports trying further complete-system tests; it does not establish statistical
reliability, unseen-course transfer, freestyle capability or a brain-controlled
improvement. No neural weights changed. The recurrent network remains in
shadow during these PD diagnostic flights.

## What the experimental assistance does

An isolated worker tracks image features across causal camera frames, combines
them with observed motion, and retains short-lived, uncertain surface patches.
A local planner checks the requested velocity plus bounded braking/turning/
climbing alternatives against those observations. Its proposals include a
reaction interval and braking tail. No map or future pose enters control.

The control boundary checks observation age, current position and task changes.
It brakes for stale or mismatched proposals and pauses if fresh proposals remain
unavailable. This run applied 759 detour ticks and 91 obstacle-brake ticks;
2,233 other ticks braked because the nominal task velocity had changed since
the proposal was calculated. These are correlated control ticks, not independent
obstacle detections. Missing surfaces remain unknown rather than certified free.

The worker processed 602 images off and 762 on, all preserved as lossless PNGs.
Total worker time, including capture and archival, was 129.9 ms p95 off and
105.0 ms p95 on. Proposal age was 110.0 ms p95 off and 84.4 ms p95 on. Different
trajectories change the worker's workload; these timings do not establish that
granting authority makes perception faster. Pose receipt alignment is causal,
but physical display delay, sparse coverage and the motion-model approximation
remain limitations.

## Evidence and provenance

Local raw data, frozen batch/condition manifests, automatic preflight and result
cards, controller logs, replay inputs, worker observations and terminal images:
`runs/geometry-on-off-20260923/{01-off,02-on}`. The preceding 60-second
ground-only proposal-boundary check is under
`runs/geometry-control-ground-20260923`; it sent no motor commands.

The brain SHA-256 is
`64444a628c58b2c1615086bc84f8c5d3c0a0a12667df31d888c4587a802288b2`.
The course geometry SHA-256 is
`03437a7e5a223d63404743aa79b80ad4c78cf8d38c3f6115d8c36487e01be432`.
All runtime source, detector, mapping and graph hashes are pinned in the
condition manifests. The inherited `runtime_geometry_allowed=false` field means
course geometry; `geometry_control` and the runtime sidecars separately record
camera-derived geometry authority. Raw manifests are preserved.

Liftoff, capture and controller ran inside hidden Anode. Preflight and postflight
checks passed with the user-authorized, idle LitHarness process family explicitly
recorded as an exception. GPU temperature stayed around 44°C.

Two additional unchanged geometry-on attempts were declared together after this
comparison, under `runs/geometry-repeat-20260923`. Their outcomes must remain
separate from the original predeclared two-condition batch.

| Unchanged follow-up | Game result | Impact | Camera/deadline failure |
|---|---|---|---|
| Repeat 1 | **Full finish, 2:24.148** | None detected | None |
| Repeat 2 | Incomplete at 2:17.030 | Final wall | None |

Combined assisted results are therefore **2/3 finishes** on this one development
course. All four flight videos decode; failures and terminal screenshots are
preserved. The second repeat saw the wall, climbed and braked, but its sparse
depth stopped refreshing while nearly stationary. All local obstacle points
expired at game time 154.96 s; the wall was still nearby. The nominal forward/
descent commands then resumed, followed by impact at 158.80 s. This exposes a
memory-lifetime failure, rather than supporting promotion from the two finishes.
