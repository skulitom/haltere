# Liftoff setup and flight operation

Use the [README quickstart](../README.md#quickstart) for Python and the included
models first. Liftoff, screen capture and controller processes run in an
[Anode background Windows desktop](https://github.com/skulitom/Anode). Keep
the viewer hidden unless the user asks to watch; the game itself must remain
foreground within that desktop. Moving Steam to that session is authorized
for this project. Retain the original `[Copy] New Drone`.

Examples below are PowerShell commands from the repository root **inside
Anode**. A terminal on the main desktop does not move its process into Anode.
Agents should check seat readiness/capabilities and hold an Anode desktop lease.

## One-time setup

Install [ViGEmBus 1.22.0](https://github.com/nefarius/ViGEmBus/releases/tag/v1.22.0)
using its administrator installer, then install the Python pad/capture extras:

```powershell
$env:VGAMEPAD_SKIP_VIGEMBUS_INSTALL = 'true'
uv pip install --python .venv/Scripts/python.exe -e ".[liftoff,vision]"
.venv/Scripts/python.exe -m haltere.cli liftoff setup
```

`setup` writes the telemetry configuration for the Windows account running
Liftoff. Reset a flight to make the game reread it. Disable Steam Input for
Liftoff. Before opening the game's controller wizard, create the virtual pad:

```powershell
.venv/Scripts/python.exe -m haltere.cli liftoff calibrate
```

Select that Xbox/XInput device in Liftoff and follow the axis prompts. Pause
the game before finishing calibration and releasing the pad. A physical radio
can remain mapped for human recording, but select the virtual pad again before
autonomous tests. The game's selected profile can change between sessions.

Check dependencies and telemetry while no other process owns the receiver port:

```powershell
.venv/Scripts/python.exe -m haltere.cli liftoff doctor
.venv/Scripts/python.exe -m haltere.cli liftoff listen --seconds 10
```

`doctor` checks installed components and files; it does not establish that the
game is responding correctly to controls. Inspect observed processed inputs.
The simulator commands use throttle/roll/pitch/yaw; Liftoff telemetry `Input`
uses throttle/yaw/pitch/roll. Recorded processed input is not raw Xbox input.

## Mapping and the persistent bridge

`configs/liftoff.yaml` contains a historical calibration. For a different
vehicle/setup, copy it to a fresh run directory, record manual flight and fit
that copy. Do not overwrite the shipped or previously measured mapping.

```powershell
New-Item -ItemType Directory runs/liftoff-calibration-new -ErrorAction Stop
Copy-Item configs/liftoff.yaml runs/liftoff-calibration-new/liftoff.yaml
.venv/Scripts/python.exe -m haltere.cli liftoff record --seconds 120 --out runs/liftoff-calibration-new/manual.csv
.venv/Scripts/python.exe -m haltere.cli liftoff fit --csv runs/liftoff-calibration-new/manual.csv --out runs/liftoff-calibration-new/liftoff.yaml
```

The fit alone is insufficient: retain/verify radial stick processing and the
measured hover point. `liftoff sticktest --help` describes the ground test.
New visual brains require the exact mapping recorded during training and reject
an incompatible mapping. The original-drone development mapping is currently
`runs/pine-route-collection-01/liftoff-original-drone.yaml`, a local file.

Keep one bridge alive between attempts, in its own Anode process:

```powershell
.venv/Scripts/python.exe -m haltere.cli liftoff pad --udp-in 9003 --seconds 900 --control-file runs/liftoff-session-new/pad-command.txt
```

Select this pad in the game and verify throttle-low and actual processed axis
response on the ground. The bridge defaults to throttle -1 and returns there
after 0.5 s without UDP commands. A centered Xbox stick is not throttle-low.
**Pause before the bridge expires or before disconnecting it.** With no pad,
Liftoff may report mid-throttle. Keep Liftoff open between tests.

## Choose the correct runner

For the published motor brains, use `haltere liftoff fly`. For example, after
verifying the mapping and bridge, a bounded hover test is:

```powershell
.venv/Scripts/python.exe -m haltere.cli liftoff fly artifacts/ftSmooth_best.pt --liftoff-config runs/liftoff-calibration-new/liftoff.yaml --udp-out 127.0.0.1:9003 --offset 0,0,2 --seconds 20 --pause-on-stop --log runs/liftoff-session-new/hover.csv --record runs/liftoff-session-new/hover.mp4
```

The older visual stack adds `--vision artifacts/gatenet_best.pt --camera
configs/camera_seat.yaml --sight rabbit` to a `ftPath2` flight. A taught path
uses `--waypoints-file`; neither mode's success establishes unseen-map transfer.

For new visual checkpoints, use [the visual runner and assisted-mode guide](visual_pilot_assistance.md).
It preserves their retinal inputs, detector fingerprint and calibration.
The older `fly` command deliberately rejects them because it cannot provide
their full sensory contract.

`--record` captures gameplay beside the live brain activity. `--show` opens a
display window and is not needed for recording. For the visual runner,
`--capture-backend dxgi` optionally uses Windows presentation timestamps after
installing `.[fast-capture]`. Match camera calibration to the actual game view.

Use fresh output paths for each attempt. Checkpoint hashes, the complete command,
assistance mode and prediction belong in a [flight card](flight_cards/README.md)
before flight. Include the complete attempt and report resets or aborts.

## Common blockers

| Symptom | Check |
|---|---|
| No telemetry | Reset after `setup`; check UDP port 9001 and competing receivers. |
| Commands have no effect | Verify focus within Anode, selected controller profile and processed inputs. |
| Throttle is unexpectedly centered | Keep the bridge connected and verify throttle -1; pause before pad release. |
| Video is blank or stale | The game must be visible and foreground within Anode, even with its outer viewer hidden. |
| New visual checkpoint rejected | Check its detector, graph, completed training metadata and mapping; do not bypass the sensory contract. |
| Anode sign-in error 2055 | Use the tray icon's **Sign in…** action and enter credentials only in Windows' native dialog. Do not reset the session or close apps to work around missing sign-in. |

For simultaneous route collection, the pilot can forward telemetry to a second
port. See [route collection](route_collection.md) rather than making two
receivers compete for the primary stream.
