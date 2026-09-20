# Autumn Fields transfer — attempt 3

Written before the flight. Frozen setup; all three attempts retained.

- Track: Autumn Fields / 01 - Walk In The Park. New map in this project's flight/data inventory; first attempt is the first-exposure result.
- Drone: [Copy] New Drone. Existing motor brain and gate detector; no route, bot data, checkpoint coordinates or navigation-v0.2 prediction model supplied.
- Pilot source: `51a0c84ed97d005efc523e54b0ac238c277b976c`. All command settings stay fixed across attempts.
- Prediction: 0-2 race checkpoints per attempt; complete lap unlikely given prior unseen-map detector failures.
- Pass criterion: At least three consecutive HUD-confirmed race checkpoints within 90 s without contact or reset; record lap completion separately; no claim of full-course generalization without a finish.
- Setup limitation: Original-drone camera f=175 px, tilt=30 from prior Pine capture; provisional 14-pair calibration. This is exploratory transfer, not a fully setup-validated benchmark.
- Reset manually before the run; no automatic reset key configured. Stop after 90 seconds. If the game resets itself, report it.

## Frozen assets

- `artifacts/ftPath2_best.pt`: `fecdd56bb0ce45ec800c21d1aafbc10954eba6bac604eff388aa25395db494e5`
- `artifacts/gatenet_best.pt`: `a00d0252e08011349061123484406dfcbcff26c4c7506b48a8295f75f1376cd2`
- `runs/pine-route-collection-01/liftoff-original-drone.yaml`: `a8892a6b87f6cdf00ba5b6ac1f92438a0d88c65551809c749313be9ddbc37d61`
- `runs/pine-route-collection-01/camera-route.yaml`: `470e01925788058f72523821484278deae7c592822c99e6c96bf2c821acd2b55`
- `haltere/liftoff/sightpilot.py`: `1c5655b90475a354e678cd865725ac6db20a09aefeee13afe39cd3b21691c872`

## Command

```powershell
.venv/Scripts/python.exe -m haltere.cli liftoff fly artifacts/ftPath2_best.pt --liftoff-config runs/pine-route-collection-01/liftoff-original-drone.yaml --udp-out 127.0.0.1:9003 --vision artifacts/gatenet_best.pt --camera runs/pine-route-collection-01/camera-route.yaml --sight rabbit --sight-speed 3.5 --sight-gate-speed 3.2 --sight-turn-gate-speed 2.8 --sight-flow-min .6 --sight-flow-alt ground --throttle-scale .8 --gyro quat --seconds 90 --fps 20 --log runs/autumn-transfer-20260920/autumn-03.csv --record runs/autumn-transfer-20260920/autumn-03.mp4 --dataset data/vision/autumn_transfer_03 --dataset-every 1
```

## Result

Pending. Use game progress/video and gate-list-free telemetry; do not score against Straw Bale geometry.
