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

## Persistent local memory revision

The three attempts frozen at `9bf30fd` retained nearby points and their
interpolated patches instead of expiring them after three seconds. All used the
same original drone and settings as before. **None finished (0/3)**:

| Attempt | Stop | Race clock | Detected impact |
|---|---|---|---|
| 1 | Automatic geometry-freshness pause near final wall | 2:28.395 | None |
| 2 | Operator ended prolonged stationary braking before start | 0:00.000 | None |
| 3 | Operator ended prolonged stationary braking beneath overhang | 3:56.684 | None |

The two operator stops occurred before the declared 500-second cap; no stall
cutoff had been preregistered. They are retained as incomplete, operator-censored
attempts, not proof that the race could never finish within that cap. There was
no reset, camera failure or controller-deadline failure. All three complete
videos decode. Local evidence: `runs/geometry-static-memory-20260923`.

The first run's geometry worker reached 447.5 ms per update; the unchanged
freshness boundary correctly refused late proposals. In attempt 2, offline
reconstruction gives current-position clearance of +0.470 m against measured
points but −0.369 m after interpolating patches. The inferred interior crosses
the gate opening. Attempt 3 also has a predicted conflict with uncertain points,
so expiring patches alone is not a complete solution to both stalls.

Exact computational pruning subsequently preserved all velocities, statuses
and margins on 2,809 earlier recorded frames while reducing geometry/planning
p95 to 18–22 ms. It skips surfaces whose conservative distance bound cannot
affect the result, and candidates that cannot improve the current cost or lie
outside the current view. This is replay timing, not live performance proof.

The next experimental revision keeps measured points locally, requires recent
support for interpolated patches, and samples shallow vertical alternatives
as well as maximum-rate climbing/descending. Those changes require their own
frozen complete-flight batch; none of the earlier attempts is reclassified.

## Transient surfaces and faster planning: 0/3

Source `0bfa419`, frozen batch `runs/geometry-two-tier-20260923`, same vehicle,
course, motor baseline and 500-second cap. A stop after at least 60 seconds
within 1 m was declared before flying and checked from causal logged telemetry.

| Attempt | Result | Clock at stop |
|---|---|---|
| 1 | Declared stall stop near gate; retained uncertain point overlaps current position | 2:22.540 |
| 2 | Final wall impact after clearing overhang | 2:13.870 |
| 3 | Declared stall stop before gate; retained point overlaps current position | 2:13.884 |

All three videos fully decode. There were no camera, controller deadline or
geometry-freshness stops; worker p95 was 117.6, 95.2 and 123.8 ms respectively.
No reset occurred during an attempt. The two stall stops were operator pauses
under the preregistered rule and are explicitly counted as incomplete. PD
controlled the motors; the newest brain remained in shadow. No weights changed.

Offline reconstruction identifies a planning trap: every candidate includes
the occupied starting position, so an initial uncertain overlap makes all
ordinary clearance checks fail. Separately, a feasible downward path beneath
the overhang was being outscored by a stationary hold. The next revision adds
explicit receding-clearance recovery and treats braking as a fallback when no
feasible moving option exists. It retains static surface memory rather than
inferring that old surfaces disappeared. These changes require new complete
flights. The sparse wall coverage remains uncertain; no replay is a finish.

## Explicit recovery and moving detours: 2/3

Frozen runtime `946056e`, `runs/geometry-escape-20260923`. Same original drone,
one-lap development course, PD motors, 2.5 m/s assistance, camera/video settings
and 500-second/60-second-stall limits. The newest brain remained in shadow.

| Attempt | Result | Full race time |
|---|---|---|
| 1 | Declared stall stop at final wall, clock 4:09.926 | Incomplete |
| 2 | Game-confirmed full finish | 2:28.763 |
| 3 | Game-confirmed full finish | 2:39.492 |

All three full videos decode. No detected impacts, resets, camera failures,
controller deadline failures or geometry-freshness stops occurred. Worker p95
was 204.2, 100.2 and 101.6 ms. The stall was an operator pause under the declared
rule. The other terminal telemetry stops correspond to the saved game results.

One launch was refused before any flight because a new, idle LitHarness process
family was missing from the existing user-authorized exception list. The list
was amended uniformly before the batch flew. Original manifests, cards and
refusal evidence are retained; controller source/settings/order did not change.

The stalled attempt repeatedly chose about 0.02 m/s vertical motion as a detour.
A subsequent selection fix requires pure vertical alternatives to reach the
existing sampling increment; ordinary unobstructed nominal commands are
unchanged. The completed recovery batch remains 2/3, and
neither main-track speed, reliable transfer nor brain-controlled obstacle
avoidance is established. No neural weights changed.

All three full recovery videos, including the stall, and their frozen evidence
are in the [recovery release](https://github.com/skulitom/haltere/releases/tag/visual-geometry-recovery-20260923).
The release targets runtime `946056e`; its five asset hashes and sizes were
checked against GitHub, and the evidence archive's 2,971 entries passed CRC checks.

## Minimum detour motion: 1/3

Frozen runtime `abe1d83`, `runs/geometry-motion-floor-20260923`. The same original
`[Copy] New Drone`, course, motor baseline, speed, camera and stop rules were
retained for all three attempts without intervening changes.

| Attempt | Result | Full race time |
|---|---|---|
| 1 | Declared wall stall; clock 4:14.302 | Incomplete |
| 2 | Detected final wall impact; clock 2:19.074 | Incomplete |
| 3 | Game-confirmed full finish | 2:31.664 |

All three videos fully decode. There were no resets, camera failures, controller
deadline failures or geometry-freshness stops. Worker p95 was 235.7, 106.0 and
107.6 ms. Attempt 1 was stopped by the operator under the declared 60-second/
1-metre stall rule; attempt 2 stopped automatically on impact. PD controlled the
motors and the newest brain ran in shadow. No neural weights changed.

Offline reconstruction from the saved causal points shows that the stall now
alternates meaningful upward/downward commands, while its observed surface
constraints block further climbing. The impact's final nominal path was reported
clear despite a hole in observed wall coverage. Fixing tiny motions did not solve
either underlying limitation. The local audit is descriptive, never a
counterfactual finish or proof of safe free space.

## Consistent detours and dual keyframes: 2/3

Frozen runtime `c76722d`, `runs/geometry-consistent-depth-20260923`. All three
attempts retained the original `[Copy] New Drone`, PD motors, motor10 in shadow,
the same known one-lap course, 2.5 m/s guidance and the existing stop rules.

| Attempt | Result | Full race time |
|---|---|---|
| 1 | Game-confirmed full finish | 2:31.958 |
| 2 | Game-confirmed full finish | 6:51.857 |
| 3 | Detected final wall impact; clock 2:18.872 | Incomplete |

All three standard brain/gameplay videos fully decode. There were no resets,
operator flight interventions, camera failures or controller deadline failures.
Worker p95 was 104.2, 307.8 and 137.5 ms. Attempt 2 frequently rejected aging
proposals and spent several minutes near the wall before climbing over it.
Attempt 3 descended until a hole in the observed wall coverage admitted forward
motion, then hit the wall. Its final images still show the wall filling the view.
This batch does not establish reliable obstacle avoidance or faster racing.
The local runtime received only causal images, telemetry and visible race cues;
no course geometry or learned depth model was loaded. No neural weights changed.

The next integration exposes the same local target/velocity boundary to a
motor-tracking brain. Its training reference speed and the visual assistant's
velocity scaling determine target conversion; the connectome still supplies
throttle, roll and pitch, with assisted yaw. PD remains a separate comparison.
The footage and metadata distinguish these motor roles. Unit checks establish
the coordinate/authority contract only; complete brain-controlled flights and
transfer remain required. The incomplete geometry is shared by both motors.
