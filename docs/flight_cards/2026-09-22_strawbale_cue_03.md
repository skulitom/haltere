# Straw Bale / Field Day: flag-clearance attempt

Seen development course. Preflight prediction, criterion and exact source
hashes: `runs/race-completion-20260922/straw-cue-03-manifest.json`. Same
scene-09 brain, detector, mapping, drone, 2 m/s setting and hidden Anode session
as [attempt 2](2026-09-22_strawbale_cue_02.md). Added local flag clearance,
using current image size and nearby visible route arrows for the passing side.
No new neural weights, predictor or stored route.

Prediction: avoid the previous flag impact; full race unproven. Criterion:
game-confirmed three-lap finish within 900 seconds, without contact, reset or
intervention. Command is attempt 2 with recording/log stem `straw-cue-03`.

## Result

Failed full-race criterion: manually paused after about 252 seconds, at HUD
lap 1/3, 03:51.134. No impact. It cleared the flag that stopped attempt 2, then
held altitude and repeatedly turned with the downhill checkpoint clamped to
the bottom of the screen. This was an operator stop, not an autonomous finish.
The runner correctly stopped on non-progressing paused telemetry and preserved
the video. Next candidate adds bounded descent for a bottom-edge target.

Raw standard brain/gameplay video, CSV, JSON and score are preserved under
`runs/race-completion-20260922/straw-cue-03*`.
