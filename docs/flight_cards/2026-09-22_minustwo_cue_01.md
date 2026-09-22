# Minus Two / Turn Signals: full three-lap finish

Seen course; first race-cue attempt. Preflight prediction and exact hashes:
`runs/race-completion-20260922/minus-cue-01-manifest.json`, code `807319c`.
Runtime source hashes, brain, detector, mapping, original drone and nominal
2 m/s setting are identical to the successful [Straw Bale run](2026-09-22_strawbale_cue_04.md).
Only the game course, output paths and time allowance changed. Hidden Anode
0.6.0, session 2; Liftoff 1.7.6. No predictor, stored route or new weights.

Prediction: test transfer of the frozen system into indoor turns and pillars;
completion unproven. Criterion: game-confirmed three-lap finish within
1200 seconds, without contact, reset or intervention.

Command is the Straw Bale command with `--seconds 1200` and log/video stem
`runs/race-completion-20260922/minus-cue-01`. Controls were rechecked on the
ground and the drone was reset before takeoff.

## Result

**Passed: full three-lap finish, 09:27.415.** Displayed lap times:
03:08.596, 03:09.098, 03:09.766. No intervention or reset, detected impact,
estimated collision, camera outage or control deadline failure. Median
horizontal speed 1.73 m/s; p90 1.83 m/s. The game showed the player as finished;
the runner stopped automatically when telemetry left live flight.

Standard brain/gameplay video, CSV, JSON, score, finish screenshot and observed
HUD values are preserved under `runs/race-completion-20260922/minus-cue-01*`.
The complete video decoded without errors. This is a successful seen-course
flight; it does not establish unseen-course or freestyle completion.
