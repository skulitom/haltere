# Fast race-cue pilot and fast PD — 2026-09-23

**Straw Bale full race: 5:03.066** (laps 1:40.855, 1:39.939, 1:40.097) with no
detected impact, reset or flight intervention. The previous autonomous bests
on the same race were 13:04.047 (PD, race-cue pilot, 2.5 m/s) and 14:05.703
(scene brain, race-cue pilot). The generated loop v2 finished in **1:20.925**
against its previous best 2:20.759.

These are development results on previously flown courses with PD motors and
the brain in shadow. They are not brain-control, unseen-course or target-speed
results; the user's Straw Bale target is 1:34.807.

| Course | Setting | Result | Evidence |
|---|---|---|---|
| Haltere loop v2 (1 lap) | fast pilot + fast PD, 5 m/s | **Finish 1:20.925**, median 4.93 m/s, 0 contacts | `loop-fast5-02`, finish screenshot |
| Straw Bale / Field Day | fast, 6 m/s | Impact on the hill after the hilltop arch, race clock 0:54.6, lap 1/3 | `straw-fast6-01` |
| Straw Bale / Field Day | fast, 6 m/s, slope-bounded bottom-edge descent | **Finish 5:03.066**, median 5.89 m/s, 0 contacts | `straw-fast6-02`, finish screenshot |
| Minus Two / Turn Signals | fast, 6 m/s | Wall impact after an arch at a hairpin, race clock 0:28.5, lap 1/3 | `minus-fast6-01` |
| Pine Valley / Forest For The Trees | fast, 6 m/s | Rising-terrain impact, race clock 0:09.5, lap 1/3 | `pine-fast6-01` |

The only code change between the two Straw attempts was generic: a marker
clipped at the bottom edge now requests a descent along a slope just steeper
than the clipped edge ray, at half speed, instead of a 3 m/s dive at 1.2 m/s.
The first attempt dived into the hill; the old pilot's gentler version of the
same rule rested on that hill for 40 s per lap. No per-course parameter exists.

## Stack

- `--pilot-profile fast` (`haltere/liftoff/fast_race_cue.py`): world velocity
  along the filtered bearing of the visible checkpoint ring, continuous turn
  slowdown, acceleration-limited commands with a tapered end, evidence-weighted
  edge handling, a sink limit near the launch plane, support detection and yaw
  through the measured rate curve with throttle priority on the shared stick.
- `--pd-profile fast` (`FastMotorPD`): velocity command plus feedforward using
  the measured full-throttle thrust curve and post-expo rate curve of the
  original `[Copy] New Drone`. The brain's teacher contract `MotorPD` is unchanged.
- Offline surrogate rehearsal (`haltere.liftoff.fast_rehearsal`) and an
  adversarial review preceded any live flight; the rehearsal is not evidence.

Stick chatter on the loop was 0.002–0.0035 per 10 ms tick against 0.010–0.014
for the previous PD run and 0.025–0.040 for the brain. Both failures on Minus
Two and Pine Valley were obstacles directly on the flight path; the pilot has no
obstacle sense beyond the existing flag clearance. A causal looming
(time-to-contact) cue is being evaluated offline on these recordings before any
live use.

Game, pad, camera and controller ran in hidden Anode (session 5) with the original
drone. Each pad session passed a processed-control ground check (idle, throttle,
roll, pitch, yaw). Raw CSV/JSON, full brain/gameplay videos and screenshots are in
`runs/fast-stack-20260923/`. The first loop launch was refused by the workload
guard (a transient Adobe process) before takeoff; the unchanged retry is `-02`.

## Brain motors under the fast pilot (same day)

The fast PD was distilled into the connectome's throttle/roll/pitch readout
(`haltere.train.fast_motor_tracking`, DAgger in the measured surrogate; only
readout rows 0-2 and their biases change; wiring, transmitter signs, yaw readout
and all other tensors are unchanged and audited). A declared contract scales the
velocity request and horizontal velocity senses so the brain operates in its
familiar range. All attempts below used 6 m/s, the fast pilot and brain motors,
and one configuration across the three tracks. **0/4 finishes.**

| Brain | Course | Result |
|---|---|---|
| fast-brain-02 (flat training) | Straw Bale | Hill-climb impact at ~0:36, lap 1: climbed at 1.2 m/s where 2.0-2.2 was requested |
| fast-brain-03 (steep legs added) | Straw Bale | Climbed the hill; impact on the steep descent at ~1:25, lap 1: flew 4.4-5.4 m/s horizontally where 3 was requested and dithered over a checkpoint below |
| fast-brain-03 | Minus Two | Pillar impact at 0:07.8 (a pillar stood between the drone and the ring) |
| fast-brain-03 | Pine Valley | Rising-mound impact at ~0:10 at the same place as the PD |

In surrogate evaluation fast-brain-03 finished 5/8 steep held-out courses (PD 6/8)
and 7/8 ordinary ones, 10-20% slower than the PD. Its speed tracking below the
nominal speed is loose; the next contract maps the nominal speed to 2.4 m/s
instead of 3 so the brain's saturating velocity senses resolve it better.

## First live looming test

PD motors, 6 m/s, `--looming-brake`, Pine Valley: impact on the same mound.
Looming produced a sample every camera frame (~0.1 s old, camera loop p50 60 ms)
and time-to-contact stayed below ~1.5 s for over 2 s before impact, but its
distance estimate read 4-9 m, so the distance-based cap braked only ~0.3 s before
contact. A time-to-contact-driven slow-down with a terrain climb is being
developed offline next. The looming brake remains off by default.

## 2026-09-24: brain-05 on the three races, terrain climb, and why the brain descends badly

Same hidden Anode seat after a PC restart (Liftoff relaunched in the seat, original
`[Copy] New Drone`, processed-control ground check before each pad session).

| Run | Stack | Result |
|---|---|---|
| `straw-brain05-01` | fast-brain-05 (speed-balanced), fast pilot, 6 m/s | Hill descent: sank 0.4 m/s where 0.9 was requested, passed ~4 m above the lower checkpoint, turned back, hit a tree at race 1:34.5 (lap 1) |
| `minus-brain05-01` | same | Pillar on the line to the ring at race 0:07.6 (same place as brain-03/04) |
| `pine-brain05-01` | same | Stopped before takeoff: one camera frame 490 ms old (camera inference had slowed from 32 to 47 ms per frame); not a flight |
| `pine-brain05-02` | same, unchanged retry | Slid up the rising mound and hit it at ~0:15 |
| `pine-fast6-ttc-01` | fast PD, `--looming-brake` with the time-to-contact policy | **Climbed over the mound** (twice, up to 8 m) and reached (79, -2); hit a boulder beside the line to the next checkpoint at race 0:13.7 (earlier PD attempts ended at 0:09.5 on the mound) |
| `straw-brain05-trim-01/-02` | brain-05 + vertical integral trim (diagnostic, removed) | Same overshoot and crash (race ~1:30) |
| `straw-brain05-gov-01` | brain-05 + descent path governor | Governor cut the horizontal request to 1.1-1.4 m/s; the brain kept flying 3.2-4.2 m/s and overshot again |

brain-05 is 0/3 on the main races. Diagnosis from exact open-loop replays of the
recorded brain inputs (`--replay-out`):

- The brain's throttle hardly responds to a more negative vertical goal (encoded
  goal -0.4 -> -0.9 changed throttle by -0.014, while -0.4 -> +0.1 changed it by
  +0.25), so live it cannot command the below-hover throttle a fast descent
  needs; its noisier throttle and higher forward speed give ~6% more rotor thrust
  than the PD at the same mean stick. A vertical integral on the goal cannot help
  (the goal saturates at its 3 m clamp) and was removed.
- The recorded scene currents (retina) move the brain's mean throttle by up to
  0.2 and pitch by 0.1 depending only on which images are shown: live Straw images
  bias it toward less braking and more thrust than the random training footage.
  Under the fast pilot the goal comes from the pilot, so the scene input is not
  needed for motor control. fast-brain-06 is trained with scene currents blanked,
  and the runtime blanks them whenever the checkpoint says it was trained that way.

The time-to-contact policy (`TtcClearanceGovernor`, previously reviewed offline,
now merged behind `--looming-brake`) is the first thing that got the PD over the
Pine Valley mound.

### fast-brain-06: scene currents blanked — first brain finishes at race speed

Same recipe as brain-05 (DAgger distillation of the fast PD into readout rows 0-2,
6 m/s nominal, scaled speed 2.4, steep legs, speed-balanced weights) but trained
and flown with the recorded scene currents blanked (`recorded_scene_currents:
false` in the checkpoint; the runtime then zeroes the retina input to the brain;
checkpoint-ring detection still uses the camera). Weight audit: only
`readout.weight`/`readout.bias` rows 0-2 changed (max |change| 0.0061 / 0.0005);
yaw readout, wiring, transmitter signs and all other tensors are identical to
motor10 candidate05. One declared configuration on all three races: fast pilot,
6 m/s, brain motors, no looming.

| Run | Course | Result |
|---|---|---|
| `straw-brain06-01` | Straw Bale | Clean through lap 1 and into lap 2; stopped at 129 s by the 120 ms telemetry-gap guard (a single 127 ms game hitch; other flights show <= 70 ms). Not a finish. The guard is now 250 ms; the 0.5 s no-progress pause check is unchanged |
| `straw-brain06-02` | Straw Bale | **Finish 5:50.489** (1:56.209, 1:56.246, 1:56.418), no detected contact |
| `straw-brain06-03` | Straw Bale | **Finish 5:50.204** (1:56.531, 1:55.813, 1:56.252), no detected contact |
| `minus-brain06-01` | Minus Two | Pillar on the line to the ring at 0:07, 5.5 m/s (lateral response still ~0.3 s behind the request) |
| `pine-brain06-01` | Pine Valley | Climbed along the mound surface, then a tree trunk at 0:11.6 |
| `pine-brain06-ttc-01` | Pine Valley, `--looming-brake` (diagnostic) | Overshot a low checkpoint, turned back and crashed at 0:03 race clock; the looming policy had no evidence there |
| `straw-fast6-03` | Straw Bale, fast PD (matched baseline, current code) | **Finish 5:02.933** (1:40.734, 1:40.396, 1:40.275) |

The brain-06 Straw finishes are 2.4x faster than the previous brain best
(14:05.703) and 16% slower than the matched fast PD. Median speed 4.87 m/s
(PD 5.89); roll/pitch command change per tick 0.0050 (PD 0.0055); body-rate RMS
0.66-0.69 rad/s (PD 1.04-1.09). On the replayed Straw descent brain-06 sits at a
lower throttle than brain-05 and its pitch response to a slow-down request is five
times larger. The main-race batch is **1/3** (Straw Bale); Minus Two and Pine
Valley also defeat the fast PD (obstacles on or beside the line to the ring).
These are previously flown development courses, not unseen tracks.

### Lateral obstacle cue: negative offline result, not merged

A workflow labelled all 15 recorded impacts and compared three causal side cues
on aligned frames (split-field looming, lateral optic-flow balance, monocular
metric depth). None warned about the Minus Two pillar or the Pine boulder: 1/11,
0/11 and 1/10 lateral impacts with a correct-side lead of at least 1 s, the one
split-field "success" being an arch trigger 13 m before the pillar. Two independent
reviewers found that the same trigger would push the PD's clean Minus Two pass into
the pillar. The monocular depth network's metric scale is compressed 2.6-5x in
Liftoff. With the camera tilted 30 degrees up, level or descending flight at 3-6 m/s
often projects the flight path below the image, so an obstacle on the path is not
visible to any image cue. The patch stays out of the codebase; the study is kept
for reference.

### Arc turns after a checkpoint (2026-09-24)

After a checkpoint the ring jumps to the next one; when it was clamped at a
side edge the pilot asked for 30% speed and slewed the request along a straight
line through low speeds, so the drone braked hard, yawed, then re-accelerated
(pitch -0.64 -> +0.61 -> -0.44, ~1.5 s lost per wide switch). The fast pilot now
changes direction as a coordinated turn (heading rotation with at most 8 m/s^2
centripetal, speed change within the rest of the 10 m/s^2 budget) and follows a
side-clamped ring at 65% speed, level, 10 degrees beyond the clamped edge ray.
Three turn strategies were compared in the surrogate on 16 synthetic courses with
the PD and brain-06; the arc variant was the only one with no added crashes and
a lower time loss. A proposed rule that read a low side-clamped marker as a gate
below was dropped: live Liftoff places side markers near the lower corners even
for rings above.

| Run | Result |
|---|---|
| `straw-fast6-arc-01` (fast PD, arc turns) | Laps 1-2 clean and ~2 s ahead of `straw-fast6-03`; **impact on the descending ImmersionRC arch's top beam in lap 3** (race 4:27.9) |

Live, 45-75 degree switches kept 3.55 m/s minimum speed (baseline 3.03) and side
time fell from 4.2 to 2.7 s; the fast part of the post-switch stick change is
unchanged (it is not caused by the slew). The arch crash exposes a margin that
was already thin: across all fast PD laps the drone crosses that arch at
12.9-13.3 m, near the top of the opening (13.32 m on the impact lap), because the
bottom-edge descent aims just steeper than the clipped ray and approaches the
ring from above; brain-06 crosses at 12.4 m. brain-06 itself learned the old
braking during distillation, so it needs re-distillation under the new pilot.

### fast-brain-07: distilled under the arc-turn pilot (2026-09-24)

Same recipe as brain-06 (readout rows 0-2 only, scene currents blanked), re-distilled
with the arc-turn pilot so the brain learns to carve instead of brake after a
checkpoint. Surrogate: 7/8 steep held-out courses, 0 crashes (brain-06 6/8). Weight
audit: only `readout.weight`/`readout.bias` rows 0-2 changed (max 0.0060 / 0.0005).

| Run | Result |
|---|---|
| `straw-brain07-02` | **Finish 5:46.846** (1:55.128, 1:54.764, 1:55.355), no detected contact |

Live against `straw-brain06-02` (same course, pilot differences as described):

| | brain-06 | brain-07 |
|---|---:|---:|
| Minimum speed after 45-75 degree switches | 2.50 m/s | **3.75 m/s** |
| Stick change per tick after a switch | 0.0133 | **0.0092** |
| Roll/pitch command change per tick, whole race | 0.0049 | **0.0037** |
| Body-rate RMS | 0.66 rad/s | **0.54 rad/s** |
| Bearing switches counted | 60 | 48 |

brain-07 re-accelerates more gently after a switch (time to regain 90% of the
pre-switch speed is longer) but carries much more speed through it.
