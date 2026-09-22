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

## Next integration decisions

First compare the restored visual pilot around the newest visual brain with
its unassisted baseline. Keep the same checkpoint, calibrated detector, drone,
camera and starts; only assistance changes. Do not silently replace the newest
brain with the older successful motor checkpoint.

The [2026-09-22 Rabbit baseline recordings](flight_cards/README.md#2026-09-22-assisted-scene-brain)
establish live compatibility on three maps, with zero completed laps in
five attempts and repeated gate-acquisition/search collisions. That batch has
no matched unassisted comparison. Use these failures to guide the next fixes;
Hangar C03 can no longer be treated as untouched if its results inform tuning.
Its images have now informed cue recognition, so it is development data; choose
a new untouched course for the next frozen evaluation. The separately declared
[race-cue experiment](race_cue_assistance.md) is allowed, but its checkpoint
markers and route arrows cannot establish freestyle capability. It subsequently
completed full three-lap races on Straw Bale and Minus Two with unchanged settings.
Its first unseen Hannover attempt failed to recover a descent from a rooftop
start. That result informed a launch-clearance fix, so Hannover is also now
development data. Preserve these failures alongside successful recordings.
The Pit's later high-checkpoint stall also informed a generic perception and
climb-recovery fix, making it development data. Paris informed bounded camera
outage recovery. Hall 26 exposed a remaining obstacle-planning failure: the
next-checkpoint marker was rendered through an overhead duct. These attempts
are all indexed with the successful seen-course recordings; none is a full
unseen-race finish.

The predictor is eligible for a subsequent runtime experiment, not prohibited.
Before giving it control authority, establish how a prediction becomes a
task-directed target, measure its behavior at launch and on states caused by
the controller itself, and handle stale images and prediction failure. Its
motion-dominated forecast can otherwise encourage continuing the current motion
without selecting the next race gate or a freestyle objective. That is an
architectural inference from the current inputs and training objective, not a
measured closed-loop failure of this predictor.

Start with causal shadow predictions beside the assisted stack, then bounded
live comparisons if that evidence supports them. Promote runtime prediction
only when it improves the declared tasks without losing demonstrated stability
or transfer. No new predictor deployment result is claimed by this audit.
