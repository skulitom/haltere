# Hangar C03 / Shipments: unseen assisted scene brain, attempt 1

Written before flight on 2026-09-22. First autonomous exposure to Hangar C03 /
01 - Shipments in this project. No Hangar C03 take or experiment was found in
the training/flight inventories (`docs`, `configs`, `runs`, `data/vision`).
This is held out from this project, irrespective of the user's personal game
history. Gate-family transfer is not classified before inspecting the flight.

- Same frozen stack as [Straw Bale](2026-09-22_strawbale_assisted_1.md) and
  [Minus Two](2026-09-22_minustwo_assisted_1.md), including their failed attempts.
- Brain SHA256 `3789e33b8bd5bbc0fa504fd9c3365ab5a5e897451d14bc5ce3c042c091543a6e`;
  detector `2dfcfed54381a18270a6681be76debff84469bb38867c0880bea5e10090a1207`.
- Code `a6057aee42120c459729684131d0afc6db4e52cc`; Liftoff build 25441586;
  original `[Copy] New Drone`, same mapping/camera contract, Anode viewer hidden.
- Full Rabbit assistance at nominal 2 m/s. Neural throttle/roll/pitch, active
  scene input. No predictor, course geometry, route, explicit cue parser or
  per-course tuning. Live autonomous brain/gameplay recording.
- Prediction: 0–1 ordered checkpoints; full race completion unlikely given
  visual acquisition and recovery failures on the preceding seen courses.
- Success: full ordered race finish within 300 s without contact/reset or
  intervention. Report every attempt, including any guard-triggered stop.
- Three attempts with unchanged settings; attempt 1 is the first-exposure
  result. Later repeats do not replace it or restore an untouched holdout.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/scene-brain-09-navigation/last.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --pilot-assistance rabbit --assist-speed 2 --capture-backend dxgi --seconds 300 --max-height 40 --max-speed 10 --max-distance 400 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/assisted-races-20260922/hangar-01.csv --record runs/assisted-races-20260922/hangar-01.mp4
```

## Result

Failed to complete a lap or race. The HUD race timer started, but the pilot
recorded zero estimated gate passages and never established useful gate
tracking. Search steering brought the drone into the hangar roof structure.
HUD at the paused stop: lap 1/3, 00:19.922. No exact ordered checkpoint count
is claimed from that timer alone.

Impact guard stopped at 30.5 s of control (31.9 s wall including shutdown),
then paused the game. Terminal acceleration 34.0 m/s², unexplained component
29.8 m/s². Median ground speed 1.83 m/s, p90 2.21 m/s. Maximum image age
116.7 ms; no camera or controller-deadline failure and no stale-image memory
ticks. The first-exposure success criterion failed.

Raw standard brain/gameplay video, CSV and JSON:
`runs/assisted-races-20260922/hangar-01.*`. Video: 31.56 s, H.264, 1928×720,
18 fps, fully decoded without errors. The terminal impact is in the sidecar;
zero CSV contact estimates do not make the flight collision-free.

Two unchanged repeats also failed at the same structure, without training or
parameter changes. All three remain reported; future tuning informed by these
results makes this track development data, not a fresh holdout.
