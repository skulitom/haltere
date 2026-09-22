# Straw Bale / Field Day: first full race-cue attempt

Seen development course. Prediction and exact source hashes were saved before
flight in `runs/race-completion-20260922/straw-cue-02-manifest.json`.
Same brain, frozen detector, drone, mapping and 2 m/s settings as the
[90-second preflight](2026-09-22_strawbale_cue_preflight.md). Anode 0.6.0,
session 2, viewer hidden. Live checkpoint ring, neural throttle/roll/pitch,
assisted yaw; no predictor, saved route, local flag clearance or new weights.

Prediction: stable initial gates and turns; complete race unproven. Criterion:
game-confirmed three-lap finish within 900 seconds without contact, reset or
intervention. Command is the preflight command with `--seconds 900` and
recording/log stem `straw-cue-02`.

## Result

Failed: impact after 164.6 seconds of controller time; HUD lap 1/3, 02:23.417.
The approach toward the checkpoint ring intersected a racing flag. No completed
lap or race. The impact guard stopped and paused the game. No camera outage or
control deadline failure. Median horizontal speed 1.68 m/s, p90 1.94 m/s.

Preserved standard brain/gameplay video, CSV, JSON and score:
`runs/race-completion-20260922/straw-cue-02.*` and `straw-cue-02-score.json`.
This result motivated local image-based flag clearance in the next candidate;
it does not establish that the clearance fix works.
