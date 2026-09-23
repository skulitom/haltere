# Project direction: generalization first

Clarified by the user on **2026-09-22**. This supersedes the earlier rule that
navigation prediction must be training-only or that the deployed brain must
make every navigation and yaw decision unaided.

The goal is a fly-brain-based drone system that can complete unfamiliar races
and freestyle tasks. The connectome remains a meaningful motor controller,
with its recurrent wiring and known transmitter signs preserved. Perception,
memory, task planning, learned path prediction, speed control and heading
assistance may surround it. Choose the combination by complete-system flight
performance and transfer to new tasks, rather than architectural purity.

“Any race or freestyle” is the intended scope, not a capability established by
the current checkpoints. Finite tests support a stated range of conditions;
they cannot establish universal success.

## Component roles

| Component | Allowed role | Evidence needed |
|---|---|---|
| Connectome brain | Learned stabilization and motor control | Identify changed weights and check stability in the deployed stack. |
| Visual pilot | Gate tracking, smooth targets, speed, heading and search | Compare complete flights with assistance on/off; disclose the assisted axes and inputs. |
| Navigation predictor | Training teacher, offline diagnostic, or runtime planner input | Compare against the same stack without it, including a motion-only baseline and unseen tasks. |
| Generic visible race cues | Task guidance where the game provides it | Declare which cues are used and test freestyle separately without race cues. |
| Routes, replay trajectories and track geometry | Demonstration collection, training targets, offline scoring | Label runtime route-following as taught/oracle-guided; it does not establish unfamiliar-course navigation. |

Assisted flight can be autonomous: assistance here means software running in
the deployed system, not a human flying the controls. Brain-only and
teacher-absent evaluations remain useful ablations. They are not mandatory
product architectures. A checkpoint's `runtime_requires_teacher=False` means
its training teacher is not required to execute that brain; it does not ban
adding an independently evaluated planner around it.

## What qualifies an improvement

1. Freeze the brain, perception, predictor, pilot parameters and input contract
   before testing. Separate whole recordings, courses and tasks into training,
   development and untouched evaluation pools. Once a result informs tuning,
   that course/task is development data.
2. For races, measure full completion in the correct checkpoint order using the
   game's HUD/finish evidence, plus time, contacts, resets and interventions.
   Report every attempt. Partial gate sequences and a pilot's estimated pass
   counter are diagnostics, not completed races.
3. For freestyle, declare the requested maneuver or task before flight, including
   its success conditions, duration, clearance and acceptable interventions.
   Test different layouts, obstacles and starts. A scripted orbit demonstrates
   trajectory tracking, not general scene-aware freestyle.
4. Test the same configuration across unseen maps, gate appearances and sizes,
   direction changes, elevation changes, sparse gates and recovery states.
   Vehicle and camera calibration are legitimate setup; hand-entered race
   geometry, per-course tuning and known-route lookup do not count as transfer.
5. Compare useful variants under matched conditions: brain alone, visual pilot
   plus brain, and that same stack plus a predictor. Include repeats and failures.
   Preserve causal timestamps and actual processed controls. A predictor's
   offline path error is not an autonomous-flight metric.

The [race evaluation protocol](generalisation.md) adds instrumentation and
flight cards. Its older numerical milestones are historical development gates,
not proof that the system can complete arbitrary races or freestyle tasks.

## Repository audit, 2026-09-22

| Area | Finding and disposition |
|---|---|
| `AGENTS.md`, README, navigation docs | Removed the blanket training-only rule and made the product goal explicit. |
| `liftoff/pilot.py`, `sightpilot.py` | Published assisted flights use external guidance and yaw. Their results are evidence for that complete stack. |
| `liftoff/visual_brain.py` | Retained the unassisted comparison and added explicit Rabbit assistance with separate brain/command logging. This integration is experimental. |
| `vision/navigation.py` | Existing predictor has causal image/motion inputs and body-relative outputs, making it a possible transferable component. It has no task/goal input and forecasts only 0.25–1 s. It is not yet a race/freestyle planner. |
| Predictor results | V2 improves one-second error over its motion base by 0.3%, 3.2% and 3.6% on three recorded takes. These small, mixed comparisons do not yet justify live authority or establish transfer. See the [release evidence](navigation_release_v02.md). |
| Human/gate/scene training | Training-only labels, teacher-free exports and frozen sensory contracts describe specific experiments. Preserve those contracts and results; do not reinterpret them as a project-wide ban on helpers. |
| By-sight tests and label tools | Keep the current runners' declared no-route/no-explicit-marker contract. Generic cue-assisted implementations are allowed when explicitly labelled and evaluated. Offline label extraction remains separate from flight. |
| Generalization tests | Historical race tests do not cover open-ended freestyle or all unseen gate types. Added task-specific acceptance criteria above; no universal success claim. |

## Current development sequence

The user's subsequent review replaced the one-course-at-a-time recovery-fix
sequence with the [generalization program](generalization_program.md). Prepare
courses in code or reuse Workshop content; do not manually design maps. Freeze
whole batches before evaluation and retain every attempt. Small variations of
one source layout stay in one split.

First compare the same race-cue pilot with the newest brain and a calibrated
PD or trained MLP motor baseline on 3–5 development courses. Then evaluate
camera-derived free space and task-directed local planning. Broaden simulation
training and the brain's tracking envelope before claiming high-speed or
acrobatic capability. The program declares seven freestyle tasks separately.

The current motion predictor is allowed but deprioritized: its inputs lack a
task and its offline gains do not establish a closed-loop benefit. A future
task-conditioned planner may use prediction if whole-system evidence supports
it. This is a prioritization decision, not a renewed training-only restriction.

The [flight index](flight_cards/README.md) retains the Rabbit baseline's five
failed attempts, the race-cue stack's Straw Bale and Minus Two full finishes,
and all subsequent failures. Five first-exposure courses produced no full
finishes across successive revisions; this was not one frozen benchmark.
Hannover, The Pit, Paris and The Green informed changes and are development
data. Hall 26's duct failure is observed evidence, not an untouched course for
future headline claims. The Green repeat failed even with horizontal holding.

On 2026-09-22 Liftoff loaded a programmatically edited Workshop copy with new
IDs, translated geometry and a changed lap count. That verifies the course-file
workflow only. A seeded open generated loop subsequently finished. On 2026-09-23,
a frozen main-track comparison finished brain 0/2 and corrected PD 1/2, identifying
both obstacle and motor-tracking failures. A generated box-obstacle course has
since been qualified by a separate oracle-guided PD finish in 3:35.704; its
autonomous visual baseline hit the last wall. A subsequent frozen comparison
finished **0/1 with camera geometry off and 1/1 with it on**, in **2:26.657**.
Those flights used PD motors with the newest brain in shadow, on the same known
development layout. Broader brain training has not yet addressed the motor gap,
and this initial geometry result does not establish transfer or main-track speed.
Two unchanged assisted repeats subsequently finished once and hit the wall once,
leaving 2/3 assisted finishes overall; nearby obstacle memory expired in the failure.
The subsequent persistent-memory revision finished 0/3: one geometry timing stop
and two operator-stopped stalls. Short-lived interpolation and faster queries
then finished 0/3 with two declared stall stops and a wall impact, without timing
failures. Retaining static surfaces with explicit recovery and feasible motion
before braking then finished 2/3 (2:28.763 and 2:39.492), with one declared wall
stall and no detected impacts or runtime failures. This is development PD
evidence, not main-track, transfer or brain-control acceptance.
See the [live geometry comparison](flight_cards/2026-09-23_geometry_control.md).
See the [current program](generalization_program.md) and
[obstacle evidence](flight_cards/2026-09-23_obstacle_geometry.md).
