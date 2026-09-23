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
