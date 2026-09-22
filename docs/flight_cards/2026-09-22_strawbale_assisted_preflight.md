# Straw Bale assisted scene brain: bounded preflight

Written before flight on 2026-09-22.

- Track: Straw Bale / 01 - Field Day, seen development course, standard race.
- Brain: `runs/scene-brain-09-navigation/last.pt`, SHA256 `3789e33b8bd5bbc0fa504fd9c3365ab5a5e897451d14bc5ce3c042c091543a6e`.
- Detector: `runs/gatenet-opening-08-spatial/best.pt`, SHA256 `2dfcfed54381a18270a6681be76debff84469bb38867c0880bea5e10090a1207`.
- Mapping: `runs/pine-route-collection-01/liftoff-original-drone.yaml`, SHA256 `a8892a6b87f6cdf00ba5b6ac1f92438a0d88c65551809c749313be9ddbc37d61`.
- Code: `a6057aee42120c459729684131d0afc6db4e52cc`; Liftoff 1.7.6, Steam build 25441586.
- Original `[Copy] New Drone`; Anode session 3, viewer hidden. Ground input check passed on all axes, throttle-low confirmed. Ten-second camera/brain shadow run completed without a guard stop.
- Live autonomous software assistance: Rabbit gate selection, smoothed target, nominal 2 m/s speed, yaw. Brain supplies throttle/roll/pitch and retains scene input. No navigation predictor, course geometry, known route or explicit race-marker parser.
- Prediction: takeoff and 0–1 checkpoint in 20 seconds; no race completion expected in this bounded preflight.
- Pass criterion: controlled takeoff with recording and real input response; no impact, stale-input or control-deadline stop. Race completion is a separate criterion for longer runs.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/scene-brain-09-navigation/last.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --pilot-assistance rabbit --assist-speed 2 --capture-backend dxgi --seconds 20 --max-height 8 --max-speed 10 --max-distance 20 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/assisted-races-20260922/straw-preflight.csv --record runs/assisted-races-20260922/straw-preflight.mp4
```

## Result

Controlled takeoff; stopped by the planned 20 m distance bound after 16.9 s
(1,551 neural ticks). No impact or controller deadline failure. Maximum image
age 106.9 ms, brain-step median 5.46 ms. Game automatically paused. The race
timer remained at zero before the start gate; no race completion claimed.

Video verified as H.264, 1928×720, 18 fps, 16.94 s, live brain beside gameplay.
Raw video/log/sidecar: `runs/assisted-races-20260922/straw-preflight.*`.
