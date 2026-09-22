# Straw Bale / Field Day: race-cue preflight

Prediction and configuration saved before flight in
`runs/race-completion-20260922/straw-cue-01-manifest.json`.
Seen development track. Anode 0.6.0, session 2, viewer hidden;
Liftoff 1.7.6, original `[Copy] New Drone`. Ground checks verified all four
processed controls and throttle-low after the reboot. Reset before takeoff.

Unchanged scene-09 brain and its frozen scene detector, as in the
[previous assisted race](2026-09-22_strawbale_assisted_1.md).
New explicitly labelled [race-cue guidance](../race_cue_assistance.md): live
visible next-checkpoint ring, bounded goal, yaw and braking during search.
No route, course coordinates, navigation predictor or new neural weights.
The brain supplies throttle/roll/pitch and retains its learned scene input.

- Prediction: takeoff and initial ordered gates; full race not established.
- Criterion: no impact during a bounded 90-second development flight.
- Source: base `724e59a`; exact modified runtime-file hashes in the manifest.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/scene-brain-09-navigation/last.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --pilot-assistance race-cue --assist-speed 2 --capture-backend dxgi --seconds 90 --max-height 40 --max-speed 10 --max-distance 400 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/race-completion-20260922/straw-cue-01.csv --record runs/race-completion-20260922/straw-cue-01.mp4
```

## Result

Passed the bounded criterion: 90 seconds of flight, no impact or camera/control
deadline stop. Maximum image age 111.6 ms. It passed the race start and handled
the early left turn that the Rabbit-only attempt lost. Final HUD: lap 1/3,
01:09.959; no completed lap or race. No exact checkpoint total claimed.
Stopped on the planned duration and automatically paused.

Full standard live brain/gameplay video, CSV and JSON are preserved as
`runs/race-completion-20260922/straw-cue-01.*`. The next attempt keeps these
controller parameters and extends the time allowance for a full three-lap race.
