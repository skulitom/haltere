# Hannover / The Biggest Yet: first unseen attempt

The frozen controller from the successful Straw Bale and Minus Two flights
was tested on Hannover. Before selection, an inventory search found no
Hannover/Hanover references in project documentation, configurations or local
JSON/JSONL/YAML training and flight inventories. No course geometry was inspected
or supplied. This establishes first project exposure within that inventory,
not proof of absence from every upstream image.

The preflight manifest is
`runs/race-completion-20260922/hannover-cue-01-manifest.json`. It pins the same
brain, runtime source hashes and settings as the two successful seen races,
with a 1700-second allowance. Hidden Anode 0.6.0; original drone; causal visible
checkpoint cues; no predictor, saved route or human flight controls.

## Result

**Failed: no completed lap.** Manually paused during lap 1/3 at game time
03:11.470 after prolonged failed downhill recovery. No detected impact or
estimated collision. The takeoff-clearance rule incorrectly continued to
treat launch elevation as a height floor. With the next cue below the screen,
descent requests were repeatedly reversed near 0.6 m above the rooftop start.

The recording, telemetry, metadata, manifest and score are preserved under
`runs/race-completion-20260922/hannover-cue-01*`. This result informed a generic
fix: release initial launch clearance after the first 0.6 m ascent. Hannover
is now development data and cannot serve as the untouched final evaluation.

A separately manifested 120-second follow-up, `hannover-cue-02`, tests descent
below launch elevation. It is a bounded regression, not a full-race result.
It passed: stopped at its planned duration, minimum relative height -45.685 m,
no impact, estimated collision, camera outage or controller deadline failure.
The game was still on lap 1/3 at 01:50.200 when automatically paused.
