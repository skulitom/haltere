# Hangar C03 / Shipments: assisted scene brain, attempt 3

Written before flight on 2026-09-22. An unchanged repeat of
[attempt 1](2026-09-22_hangarc03_assisted_1.md), with its exact brain, detector,
mapping, code, drone, capture and flight bounds. Attempt 1 remains the
first-exposure result. No training or course-specific tuning between attempts.

- Prediction: 0–1 ordered checkpoints, likely loss of route and collision.
- Success: full ordered three-lap race finish within 300 s without contact,
  reset or intervention. Keep the full video and any guard-triggered stop.
- Live autonomous assisted recording, standard brain activity beside gameplay.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/scene-brain-09-navigation/last.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --pilot-assistance rabbit --assist-speed 2 --capture-backend dxgi --seconds 300 --max-height 40 --max-speed 10 --max-distance 400 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/assisted-races-20260922/hangar-03.csv --record runs/assisted-races-20260922/hangar-03.mp4
```

## Result

Failed in the same way as attempts 1 and 2: race timer started, search without
useful gate tracking, then collision with the hangar roof structure. Zero
pilot-estimated passages. HUD at stop: lap 1/3, 00:19.902. No lap or race finish.

Impact stop at 30.5 s of control (31.8 s wall including shutdown), then
automatic pause. Terminal acceleration 36.9 m/s², unexplained component
32.5 m/s². Median ground speed 1.86 m/s, p90 2.21 m/s. Maximum image age
235.2 ms and 54 control ticks using memory during discarded stale images;
no camera or controller-deadline failure. The criterion failed in all three
unchanged Hangar trials.

Raw standard brain/gameplay video, CSV and JSON:
`runs/assisted-races-20260922/hangar-03.*`. Video: 31.39 s, H.264, 1928×720,
18 fps, fully decoded without errors. The CSV estimates one contact near the
stop, and the sidecar records a terminal impact. These are overlapping
evidence, not a claim of two separate collisions.

Verified the game was paused before disconnecting the pad. Liftoff remains
open in Anode with its viewer hidden and the original drone retained.
