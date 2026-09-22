# Straw Bale / Field Day: assisted scene brain, attempt 1

Written before flight on 2026-09-22. Seen development course; standard three-lap
race, original `[Copy] New Drone`, Anode viewer hidden.

- Brain SHA256: `3789e33b8bd5bbc0fa504fd9c3365ab5a5e897451d14bc5ce3c042c091543a6e`.
- Detector SHA256: `2dfcfed54381a18270a6681be76debff84469bb38867c0880bea5e10090a1207`.
- Mapping SHA256: `a8892a6b87f6cdf00ba5b6ac1f92438a0d88c65551809c749313be9ddbc37d61`.
- Code: `a6057aee42120c459729684131d0afc6db4e52cc`; Liftoff 1.7.6 / build 25441586.
- Rabbit gate tracking/selection, smoothing, nominal 2 m/s and yaw assistance;
  trained brain throttle/roll/pitch, scene features active. No predictor, route,
  course geometry or explicit marker parser. Live autonomous recording.
- Reset after successful bounded preflight; only duration and general flight
  bounds increase. No per-course pilot parameters.
- Prediction: 1–5 ordered checkpoints; full race completion remains unproven.
- Success: game's full ordered race finish within 300 s without contact/reset
  or intervention. Report partial progression separately and retain aborts.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/scene-brain-09-navigation/last.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --pilot-assistance rabbit --assist-speed 2 --capture-backend dxgi --seconds 300 --max-height 40 --max-speed 10 --max-distance 400 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/assisted-races-20260922/straw-01.csv --record runs/assisted-races-20260922/straw-01.mp4
```

## Result

Failed to complete a lap or race. Video shows two initial arch passages,
followed by loss of the next gate, search and collision with a hay bale. Pilot
estimated passages: 2. HUD remains lap 1/3, timer 01:17.178 at the stop; it does
not display a checkpoint total, so no exact HUD checkpoint count is claimed.

Controller stopped on impact at approximately 96.2 s of telemetry (97.6 s wall
including shutdown), then paused the game. Median ground speed 1.95 m/s,
90th percentile 2.37 m/s. No controller deadline failure; maximum image age
183 ms, 81 control ticks using memory during discarded stale images.

The CSV-only offline collision heuristic reports zero: the stop-triggering
sample is checked before the command/log row is emitted. The JSON sidecar
records the terminal impact (48.4 m/s² acceleration) and video confirms it;
this attempt is not collision-free. Preserve that distinction when scoring.

Raw standard brain/game video, CSV and JSON: `runs/assisted-races-20260922/straw-01.*`.
Same controller settings retained for the following seen and unseen tests.
