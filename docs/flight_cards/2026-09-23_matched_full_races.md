# Full-race brain versus PD comparison — 2026-09-23

**Brain: 0/2 finishes. PD: 1/2 finishes.** Both motors hit a pillar on Minus Two;
PD alone completed Straw Bale. These four development attempts support doing
geometry next and also expose a brain tracking gap. They do not establish a
statistical ranking, unseen-course performance or the requested speed target.

| Course | Motor | Full finish | Result | Median speed |
|---|---|---|---|---:|
| Straw Bale / Field Day | Motor10 candidate05 | No | Starting-arch impact, 23.20 s of control; race clock had not started | 1.52 m/s |
| Straw Bale / Field Day | PD; brain in shadow | **Three laps, 13:04.047** | No detected impact, reset or flight intervention | 2.46 m/s |
| Minus Two / Turn Signals | PD; brain in shadow | No | Pillar impact, race clock 1:15.868, lap 1/3 | 2.47 m/s |
| Minus Two / Turn Signals | Motor10 candidate05 | No | Pillar impact, race clock 1:19.388, lap 1/3 | 2.22 m/s |

The full-race time is transcribed from the game results screen. Its separately
displayed lap times do not sum to that total; the acceptance comparison uses the
displayed full-race time. The user's Straw Bale target is **1:34.807**, using the
same original `[Copy] New Drone`. No result here meets the five-track target.

## Frozen contract and retained failures

All four attempts used revision `fa5e6d6fc6d5e69edfa786721ddb6ef0d8e855ea`,
candidate05 SHA-256
`64444a628c58b2c1615086bc84f8c5d3c0a0a12667df31d888c4587a802288b2`,
the same detector, original-drone mapping, race-cue pilot and 2.5 m/s setting.
Each allowed up to 1,200 seconds to finish three laps. Sources and asset hashes
were checked before each launch. Order: Straw brain, Straw PD, Minus PD, Minus
brain. No tuning occurred between attempts. PD uses unscaled measured velocity
and the requested effective speed, correcting the earlier teacher mismatch.

The declared rule was to prioritize geometry if both motors hit the same
obstacle and investigate tracking if PD alone finished. Both observations
occurred. The starting-arch brain failure counts as a failed flight, not an
excluded launch. There were **no camera or control-deadline failures**. Straw
PD's stale-telemetry stop followed its verified finish screen, so it is not an
unfinished runtime failure. All preflight/postflight workload snapshots passed.
The user-authorized quiet LitHarness workload exception was recorded explicitly;
no training or benchmark job was started by this task during these flights.

Game, pad, camera, brain and recording ran in hidden Anode 0.8.0, session 5.
Brain inference used CPU, vision CUDA, recording explicit NVENC. Both ground
checks verified throttle-low and all four processed control axes. The visible
checkpoint marker supplied the task direction; no runtime course geometry,
stored route or navigation predictor was used. PD recordings visibly identify
the brain as shadow activity, not the source of motor commands.

All four untrimmed standard brain/gameplay recordings fully decode. The local
folder `runs/matched-full-races-20260923` preserves the manifest, frozen sources,
preflight cards, CSV/JSON, causal neural replays, terminal screenshots, scores,
HUD reviews and result cards. `batch-results.json` pins their hashes. The
earlier candidate05 release batch remains a separate **0/3**, plus its separate
generated-loop finish; this diagnostic does not overwrite those results.

## Recording headroom check

Before this batch, six 35-second ground-shadow runs compared recording off,
two-thread CPU encoding and NVENC in the order off/CPU/NVENC/NVENC/CPU/off.
Mean repeat medians for camera cycle were **59.93 / 64.28 / 63.17 ms**; mean
repeat p95 values were **67.96 / 79.18 / 78.63 ms**. These are averages of
per-run percentiles, not pooled percentiles. All four videos fully decoded.

NVENC passed the preregistered non-regression criterion and was selected. Its
benefit over bounded CPU encoding was small; recording does not explain the
whole camera-cycle cost. These were ground checks, not flight-performance
evidence or proof that there is sufficient headroom for a new depth model.
Raw evidence is in `runs/runtime-headroom-20260923`.
