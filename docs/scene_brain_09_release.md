# Scene-09 visual brain: inference bundle

Download the experimental bundle from
[GitHub Releases](https://github.com/skulitom/haltere/releases/tag/scene-brain-09-experimental)
or [Hugging Face](https://huggingface.co/Skulitom/haltere/tree/main/scene09).
Both published archives were checked against the local SHA256. This is an
experimental download, not a qualified general race/freestyle controller.

This bundle contains the newest completed scene brain reviewed on 2026-09-22,
its exact frozen detector and the mapping used for the original `[Copy] New Drone`.
It preserves the original checkpoint bytes and relative paths. Extract it into
the repository root; do not substitute a different detector with the same name.

| File in the bundle | SHA256 |
|---|---|
| `runs/scene-brain-09-navigation/last.pt` | `3789e33b8bd5bbc0fa504fd9c3365ab5a5e897451d14bc5ce3c042c091543a6e` |
| `runs/gatenet-opening-08-spatial/best.pt` | `2dfcfed54381a18270a6681be76debff84469bb38867c0880bea5e10090a1207` |
| `runs/pine-route-collection-01/liftoff-original-drone.yaml` | `a8892a6b87f6cdf00ba5b6ac1f92438a0d88c65551809c749313be9ddbc37d61` |

The graph remains the repository's `data/built/flight.npz`, SHA256
`db429c4d4ed151d905467ac0ff474c3cb11f4dbe6b42bb5f490e1d438b2e64b7`.
The bundle includes training configuration and a manifest with individual hashes.
No route, gate coordinates, teacher or separate NavigationNet predictor is needed.

## What was trained

The parent was `runs/gate-brain-readout-02-spatial-08-replay/candidate.pt`.
Scene-09 learned `encoders.retina__goal.U` and
`encoders.retina__goal.log_gain` using `data/vision/observed_scene_v1`,
400 updates, seed 1839, batch size 8 and 128-tick windows. These parameters
inject learned visual currents into existing goal neurons. Original parent
weights, connectome wiring and known transmitter signs were retained; no motor
refit was performed. The original recordings are local and are not included.

The race-cue integration did not retrain these weights. In that mode, the brain
retains throttle, roll and pitch, while software supplies a bounded goal,
velocity-sense scaling and yaw from current visible checkpoint cues. Blue-tipped
flag clearance also uses nearby visible route arrows. This limited heuristic
does not establish general obstacle avoidance or freestyle planning.

## Running the evaluated stack

Follow [game setup](liftoff_setup.md) and [the assisted-run guide](visual_pilot_assistance.md).
Use code `de744e8` or a later compatible revision. It includes temporary launch
clearance, bounded camera-outage braking, high-checkpoint recovery and
two-thread image processing. The prior full suite passed 313 checks; the
subsequent top-edge horizontal-hold change passed 51 focused checks.
Use hidden Anode, the original drone, a 30-degree camera tilt and the calibrated
camera geometry embedded in the checkpoint. Verify the real processed controls,
including throttle-low, before connecting a live runner to a pad bridge.

This command defaults to **shadow mode**, so it observes without sending controls:

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/scene-brain-09-navigation/last.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --pilot-assistance race-cue --assist-speed 2 --capture-backend dxgi --seconds 60 --max-height 250 --max-speed 10 --max-distance 2000 --log runs/my-scene09-shadow.csv --record runs/my-scene09-shadow.mp4
```

For authorized live simulation flight, run the command inside Anode with the
verified bridge and add `--udp-out 127.0.0.1:9003 --pause-on-stop`. Choose a fresh
log/video path and a duration adequate for the full race. Keep the game open and
pause before disconnecting the pad. A finished race can stop live telemetry;
judge completion from game finish evidence, not the runner's exit code alone.

## Evidence and limits

The [flight index](flight_cards/README.md) preserves every attempt. The earlier
Rabbit integration failed all five race attempts. The separately declared
race-cue mode completed Straw Bale in 14:05.703 and Minus Two in 09:27.415 with
the same controller settings, no detected contact, resets or flight intervention.
Those are seen courses. A subsequent first-exposure Hannover test failed because
launch clearance prevented descent below a rooftop start. That course became
development data; a generic temporary-clearance fix passed a bounded descent
regression. It does not turn Hannover into a successful full-race test.
The Pit's first attempt was stopped by the original 400 m distance bound without
detected contact; it is also incomplete. The common full-race envelope was then
expanded to the bounds in the command above, without changing steering.
A later Pit attempt stalled at a high checkpoint masked as HUD. It informed
the high-marker and steep-bearing fixes, making The Pit development data.
Paris exposed camera timing interruptions; both failed attempts and the
successful bounded regression are retained. Hall 26 subsequently hit an overhead
duct while following a checkpoint marker rendered through it. This is a concrete
remaining obstacle-planning failure, not a timing problem or successful race.
The Green subsequently hit an overhang after traversing a building underpass.
It informed horizontal holding during top-edge recovery and is now development
data. The development repeat also hit the overhang on lap 1/2 at 453.86 s;
horizontal holding did not resolve the obstacle-planning failure. See the
[revised generalization program](generalization_program.md).

No universal race capability, freestyle completion or improvement from a separate
navigation predictor is established by these results. The visual helpers are
part of the evaluated controller, not evidence of unaided brain navigation.
