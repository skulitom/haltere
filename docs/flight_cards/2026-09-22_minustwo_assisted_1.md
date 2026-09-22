# Minus Two / Turn Signals: assisted scene brain, attempt 1

Written before flight on 2026-09-22. Seen map, 01 - Turn Signals, standard race.
Original `[Copy] New Drone`, Anode session 3 and viewer hidden. Same frozen
brain, detector, mapping and full Rabbit assistance as the
[Straw Bale attempt](2026-09-22_strawbale_assisted_1.md). No predictor, route,
geometry or explicit marker parser; no parameter changes after Straw Bale.

- Brain SHA256: `3789e33b8bd5bbc0fa504fd9c3365ab5a5e897451d14bc5ce3c042c091543a6e`.
- Code: `a6057aee42120c459729684131d0afc6db4e52cc`; Liftoff build 25441586.
- Prediction: 0–2 ordered checkpoints; turn recovery is an observed weakness.
- Success: full ordered race finish within 300 s without contact, reset or
  intervention. Report partial progress separately; preserve every stop.
- Recording: autonomous assisted flight, standard brain activity beside gameplay.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/scene-brain-09-navigation/last.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --pilot-assistance rabbit --assist-speed 2 --capture-backend dxgi --seconds 300 --max-height 40 --max-speed 10 --max-distance 400 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/assisted-races-20260922/minus-01.csv --record runs/assisted-races-20260922/minus-01.mp4
```

## Result

Failed before the race start gate. Zero pilot-estimated passages; HUD timer
remained 00:00.000 on lap 1/3. Video shows low-confidence gate acquisition,
search steering and a pillar collision. Impact guard stopped at approximately
17.4 s of control (18.7 s wall including shutdown), then paused the game.
No controller deadline failure; maximum image age 113 ms. The recorded input
check passed after the persistent pad was refreshed in the paused menu.

Terminal impact: 60.4 m/s², recorded in JSON before the stop sample can enter
the CSV. Do not interpret a zero CSV-only collision estimate as collision-free.
Raw standard brain/game video, CSV and JSON: `runs/assisted-races-20260922/minus-01.*`.
The controller remains unchanged for the held-out test.
