# Hangar C03 / Shipments: assisted scene brain, attempt 2

Written before flight on 2026-09-22. An unchanged repeat of
[attempt 1](2026-09-22_hangarc03_assisted_1.md), with its exact brain, detector,
mapping, code, drone, capture and flight bounds. Attempt 1 remains the
first-exposure result. No training or course-specific tuning between attempts.

- Prediction: 0–1 ordered checkpoints, likely loss of route and collision.
- Success: full ordered three-lap race finish within 300 s without contact,
  reset or intervention. Keep the full video and any guard-triggered stop.
- Live autonomous assisted recording, standard brain activity beside gameplay.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/scene-brain-09-navigation/last.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --pilot-assistance rabbit --assist-speed 2 --capture-backend dxgi --seconds 300 --max-height 40 --max-speed 10 --max-distance 400 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/assisted-races-20260922/hangar-02.csv --record runs/assisted-races-20260922/hangar-02.mp4
```

## Result

Failed in the same way as attempt 1: race timer started, no useful gate
tracking, search steering and impact with the hangar roof structure. Zero
pilot-estimated passages. HUD at stop: lap 1/3, 00:19.891. No lap or race finish.

Impact stop at 30.5 s of control (31.8 s wall including shutdown), then
automatic pause. Terminal acceleration 32.1 m/s², unexplained component
27.8 m/s². Median ground speed 1.84 m/s, p90 2.21 m/s. Maximum image age
120.1 ms; no camera or controller-deadline failure and no stale-image memory
ticks. The success criterion failed; no tuning before attempt 3.

Raw standard brain/gameplay video, CSV and JSON:
`runs/assisted-races-20260922/hangar-02.*`. Video: 31.39 s, H.264, 1928×720,
18 fps, fully decoded without errors. The sidecar records the terminal impact
even though the CSV contact estimator reports zero.
