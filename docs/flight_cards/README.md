# Flight cards

Evaluate the complete deployed stack under the [current project direction](../project_direction.md).
Record assistance and predictor use explicitly. A software-assisted flight may
be autonomous; oracle-route collection and human-controlled shadow recordings
must remain separately labelled. Freestyle cards must state the requested task
and success criterion rather than substitute a gate count.

One card per flight, written **before** it and appended to after. A prediction written after the
flight does not exist. Rules: `docs/generalisation.md`.

## 2026-09-23 live visual geometry

The first [matched geometry off/on comparison](2026-09-23_geometry_control.md)
finished **0/1 off versus 1/1 on** on the known generated box course. The on
condition completed its full lap in **2:26.657**, with no detected impact or
human flight intervention. Both conditions used PD motors, the newest brain
in shadow, and identical capture/recording work. This is initial development
evidence, outside the five main-track speed/completion target.
Two unchanged on-condition repeats subsequently finished once and hit the wall
once: **2/3 assisted finishes overall**, with a nearby-obstacle memory expiry
identified in the failure. All attempts remain in the linked record.
The subsequent persistent-memory revision finished **0/3**, with one geometry
timing stop and two operator-stopped stalls. The current revision has passed
414 automated tests and causal replay checks; flight validation remains pending.

## 2026-09-23 full-race motor comparison

The subsequent [generated obstacle/depth diagnostic](2026-09-23_obstacle_geometry.md)
completed five sections geometrically, then hit the final wall (**0/1 whole
courses**). It supplied 599 passive images for causal geometry development;
the geometry prototype was passive during that initial attempt.
A separate oracle-guided qualification series retained two runtime-censored
attempts and then a **3:35.704 full box-course finish**, with the brain in shadow.
That establishes course playability and supplies a second perception trajectory;
it does not count as autonomous race success.

[The corrected, frozen four-flight comparison](2026-09-23_matched_full_races.md)
finished **0/2 with motor10 candidate05 and 1/2 with PD**. PD completed all three
Straw Bale laps in **13:04.047**; both motors hit a Minus Two pillar. No camera or
control-deadline stops occurred. All four standard brain/gameplay videos fully
decode. These results prioritize geometry while preserving the observed motor
tracking gap; they do not meet the user's five-track speed target.

## 2026-09-22 assisted scene brain

Newest completed local brain `runs/scene-brain-09-navigation/last.pt`, full
Rabbit gate selection, smoothing, speed and yaw; trained neural throttle,
roll and pitch. Frozen detector, mapping and pilot parameters across the three
maps; no navigation predictor, route, course geometry or explicit cue parser.
The cards pin hashes and commands. Liftoff 1.7.6 / build 25441586 ran entirely
inside hidden Anode with the original `[Copy] New Drone`.

**Zero laps or races completed in five attempts.** Times below are seconds
since the first controller telemetry frame, including its arming period;
the game's race clock starts later, on crossing the start line.

| Map / race | Exposure | Control duration | Observed result | Card / local recording stem |
|---|---|---:|---|---|
| Straw Bale / Field Day | Seen | 96.2 s | Initial arch sequence, lost next gate, hay-bale collision; pilot estimated 2 passes | [Attempt 1](2026-09-22_strawbale_assisted_1.md), `straw-01` |
| Minus Two / Turn Signals | Seen | 17.4 s | Search and pillar collision before race timer started | [Attempt 1](2026-09-22_minustwo_assisted_1.md), `minus-01` |
| Hangar C03 / Shipments | First project exposure | 30.5 s | Timer started, poor gate acquisition, roof-structure collision | [Attempt 1](2026-09-22_hangarc03_assisted_1.md), `hangar-01` |
| Hangar C03 / Shipments | Unchanged repeat | 30.5 s | Same failure | [Attempt 2](2026-09-22_hangarc03_assisted_2.md), `hangar-02` |
| Hangar C03 / Shipments | Unchanged repeat | 30.5 s | Same failure | [Attempt 3](2026-09-22_hangarc03_assisted_3.md), `hangar-03` |

A preceding [bounded preflight](2026-09-22_strawbale_assisted_preflight.md)
passed controlled takeoff and recording, stopping at the planned 20 m distance
bound. That is not counted as a completed race. Both ground control checks
confirmed throttle-low and all four processed axes.

Raw recordings live in the ignored local folder `runs/assisted-races-20260922/`.
Each stem has `.mp4`, `.csv` and `.json`; videos show the standard live brain
panel beside gameplay, at 1928×720, H.264, 18 fps. All six videos (including
preflight) fully decode without errors. They are not included in a fresh clone.
The folder also preserves the model manifest, ground checks, pad logs and
`*-score.json` outputs. Keep CSV and JSON together: impact guards may stop
before the final CSV sample, and the scorer reads the sidecar to retain that
terminal event. CSV contact estimates and terminal impacts may overlap.

No runtime tuning occurred between these race attempts. Hangar was absent
from the project training/flight inventories inspected before its first test;
later attempts are repeats of that test, not additional unseen maps. There is
no live unassisted comparison or freestyle evaluation in this batch, so it
does not establish an assistance improvement or freestyle generalization.
No new weights were trained or promoted. Gate acquisition, turn recovery and
obstacle-aware search remain visible problems to fix.

## 2026-09-22 visible race-cue development

Separate experimental mode using the same newest scene brain, with visible
checkpoint guidance. No new neural weights, predictor or stored route. Raw
standard brain/gameplay videos and telemetry are in
`runs/race-completion-20260922/`.

| Race-cue attempt | Result | Evidence |
|---|---|---|
| 90-second preflight | No contact; partial lap 1/3 | [Preflight](2026-09-22_strawbale_cue_preflight.md), `straw-cue-01` |
| First full-race attempt | Flag collision at 164.6 s; no completed lap | [Attempt 2](2026-09-22_strawbale_cue_02.md), `straw-cue-02` |
| Flag-clearance attempt | Passed previous flag; manually stopped in downhill recovery stall, lap 1/3 | [Attempt 3](2026-09-22_strawbale_cue_03.md), `straw-cue-03` |
| Downhill-recovery attempt | **Full three-lap finish, 14:05.703; no detected contact or flight intervention** | [Attempt 4](2026-09-22_strawbale_cue_04.md), `straw-cue-04` |
| Minus Two / Turn Signals, unchanged controller | **Full three-lap finish, 09:27.415; no detected contact or intervention** | [Attempt 1](2026-09-22_minustwo_cue_01.md), `minus-cue-01` |
| Hannover / The Biggest Yet, first unseen attempt | No completed lap; manually stopped after launch-height floor prevented downhill recovery; now development data | [Attempt 1](2026-09-22_hannover_cue_01.md), `hannover-cue-01` |
| Hannover, temporary launch clearance regression | Planned 120 s completed without detected contact; descended 45.7 m below launch; not a full race | [Regression](2026-09-22_hannover_cue_01.md), `hannover-cue-02` |
| The Pit / course 01, first exposure | Distance bound stopped the run at 242.46 s, no detected contact; incomplete | [Attempt 1](2026-09-22_pit_cue_01.md), `pit-cue-01` |
| Paris Drone Festival / City Trip, first exposure | Camera freshness guard stopped the run during lap 1/3; no detected contact; incomplete | [Attempt 1](2026-09-22_paris_cue_01.md), `paris-cue-01` |
| Paris, unchanged repeat | Camera freshness stop before the race timer started; no detected contact | [Repeat](2026-09-22_paris_cue_01.md), `paris-cue-02` |
| Paris, camera recovery regression | Planned 200 s completed without detected contact or camera outage; not a full race | [Regression](2026-09-22_paris_cue_01.md), `paris-cue-03` |
| Hall 26 / course 01, first exposure | Impact into overhead duct at 77.11 s, lap 1/2; checkpoint marker visible through obstruction | [Attempt 1](2026-09-22_hall26_cue_01.md), `hall26-cue-01` |
| The Pit, expanded-envelope repeat | High-checkpoint stall; manually paused at game time 16:09.841, lap 1/1; now development data | [Attempt 2](2026-09-22_pit_cue_02.md), `pit-cue-02` |
| The Green / course 01, first exposure | Impact during upward recovery after a building underpass, lap 1/2; now development data | [Attempt 1](2026-09-22_green_cue_01.md), `green-cue-01` |
| The Green / course 01, development repeat | Horizontal-hold change still hit the overhang at 453.86 s, lap 1/2 | [Attempt 2](2026-09-22_green_cue_02.md), `green-cue-02` |

## 2026-09-22 motor-control development batches

The [motor comparison and training record](../motor_tracking.md) covers six
matched 180-second development segments on the generated loop, Straw Bale and
Minus Two. Scene09 and a conventional PD motor controller shared the same visible
cue guidance. PD videos explicitly label the brain as running in shadow. The
indoor PD attempt stopped on a control deadline; no complete race was established
in that initial comparison. All six videos fully decode.

The recovery-trained candidate04 then flew the same courses at a common 3 m/s
setting. It finished the generated v2 loop in **2:20.759**, but hit a flag on
Straw Bale and a pillar on Minus Two. All three attempts are retained, with
brain/gameplay videos, terminal-impact sidecars, source hashes, scores and the
generated-course finish screenshot in `runs/motor10-transfer-20260922`.
This is development evidence and does not establish unseen-race capability.

Candidate04's subsequent 2.5 m/s batch was withdrawn after a large motor
excursion and impact on its first Straw Bale attempt. Candidate05 added
recorded scene currents and image gaps to motor training. Its frozen 2.5 m/s
batch then finished **0/3**: a flag impact on Straw Bale, a pillar impact on
Minus Two, and a telemetry stop on the generated loop. All three standard
brain/gameplay videos decode, and exact sensory replays accompany the logs in
`runs/motor10-visual-full-20260922`. The loop's missed shutdown pause was fixed
and tested before a separately recorded repeat. Keep that repeat separate
from the original batch; neither batch is unseen-course evidence.

The separate 2026-09-23 repeat completed the generated loop in **3:01.576**
at a 2.31 m/s median, without detected contact, reset or flight intervention.
It used the same candidate05 weights and steering; only shutdown handling and
diagnostics changed. The finish screenshot, full brain/gameplay video, exact
sensory replay and source verification are in `runs/motor10-loop-repeat-20260923`.
Candidate05 therefore has **1/4** finishes including the failed frozen batch.
The [experimental bundle](../motor_brain_10_release.md) preserves these limits.

## Card template

Name: `<date>_<track>_<n>.md`. Template:

```markdown
# 2026-09-16 pinevalley 1
- **Rung**: Z4 (unseen map, unseen gate type)   <!-- Z0-Z4, see docs/generalisation.md -->
- **Track**: Pine Valley / "1 - Forest For The Trees"  (pool: dev)
- **Detector**: runs/gatenet10/best.pt  sha256 <first 12>
- **Brain / code revision**: <checkpoint hash and commit>
- **Assistance / predictor / visible cues**: <mode, parameters, predictor hash or absent, cue set>
- **Recording mode**: <live autonomous, human demonstration, oracle-guided collection, or shadow>
- **Command**: <the exact line, copy-pasteable>
- **Course parameters typed on the command line**: none   <!-- must be "none" on a sealed track -->
- **Prediction**: <what you expect, as a number, before flying>
- **Pass criterion**: <decided now, not after>

## Result
- HUD checkpoints: n/m, splits ...
- Pilot's own `n_passes`: ...  (precision/recall against the HUD: ...)
- `liftoff score --gates ""`: speed, shake, contacts
- Verdict against the criterion: pass / fail
- What it changed: <if this was a sealed track and anything here changes the code, the track
  becomes dev - record that>
```
