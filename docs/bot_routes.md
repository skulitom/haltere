# Liftoff bot routes: verified local extraction

Follow-up: the [collection experiment](route_collection.md) now clears the first
course gate on **[Copy] New Drone** in two consecutive 65 m runs, after correcting
lateral tracking bias that caused a tree collision. Bot geometry is useful as a
route teacher; camera/label validation and a full lap remain unfinished.

On 2026-09-20, read-only inspection of installed Steam build **25118475** found a
bundled recording register used by the race ghost loader. It contains **275 entries
for Pine Valley / 01 - Forest For The Trees**, race ID
`58e6df20-7715-47ca-814d-05c57406a454`. Only this development course was selected for
trajectory export. Original game assets and decompiled research stay local under
ignored directories; they are not project source or training fixtures.

## What we have

The reproducible first export is `runs/pine-bot-route-01/`:

| Field | Result |
|---|---|
| Selection | Median reported race time from the 275 matching entries |
| Recording key | `2jAFMa9gyE5jLAsqGvLLlNsUrNTMzgqxfKJ05jdHyUVmeFMgNOfTwb5DenE1RTLu` |
| Original recording version | 1.2.11, bundled in the current installation |
| Source SHA-256 | `62fabec7fa749dfaf4167d49ec257655f3b0eb80fa209ff1426bd16a78941e0b` |
| Samples | 1,405 source rows; 1,402 unique timestamps |
| Sample interval | Approximately 0.1 s |
| Recorded race time | 137.915 s |
| Full source duration | 139.714 s, including lead-in |
| Source lap times after lead-in | 40.585, 46.815, 50.513 s |
| Sampled path length | 2,604.56 m across the recording |
| Median / 95th-percentile speed | 19.11 / 24.00 m/s |
| Signals | Unity world position, orientation, four varying input channels, time |

The three zero-duration intervals were identical repeated states at segment ends.
The export retains the last copy, preserves its source index, and maps segment
indices using the original lap-start array. Other timestamp collisions, reversed
time, non-finite data and invalid quaternions are rejected. Source race time is
retained separately from timestamp duration; neither is silently rescaled.

`route-overview.png` compares the three source laps with the installed course's
object pivots. Their horizontal shapes agree visually. Object pivots are not gate
openings, and this is not a collision check or a current HUD completion result.

## An important change to the second-course plan

The installed race has **31 checkpoint passage records referencing 30 distinct
objects: 28 checkpoint boxes and two arches**. The start and finish reference the
same arch. The first visible arch is not representative of the whole race.

This explains why the route problem cannot be reduced to finding successive arch
detections. The next bounded experiment should use one bot route as a **training
teacher for navigation between sparse gates**:

1. Verify the archived route against the current course and align its Unity world
   coordinates with the player's measured reset origin. Keep source altitude; do
   not assume the first recorded pose is the current spawn point.
2. Use a separate, slow route-following experiment to collect complete player FPV
   and telemetry. Start near the established controller speed, not the source's
   19 m/s. Slowing timestamps does not produce valid slow-flight stick labels.
3. Use those frames to investigate perception and route progress between visible
   gates. Score the route-following experiment separately from flying by sight.
   Oracle route access is a collection/teacher condition, not evidence of visual
   generalisation. A gate-focused second course may be a cleaner detector test.
4. Only then consider training from several curated routes. Hold out whole source
   recordings and entire courses; laps from one recording are not independent.

The current brain, detector and rabbit pilot remain frozen. The initial extraction was read-only; the linked collection experiment subsequently
used route positions to guide live flights, without replaying extracted stick inputs.

## Reproduce an export

Install the optional asset reader into the chosen research environment:

```powershell
uv pip install --python .venv/Scripts/python.exe "UnityPy==1.25.3"
.venv/Scripts/python.exe -m haltere.cli liftoff bot-route --game-dir "C:\SteamLibrary\steamapps\common\Liftoff" --race-id 58e6df20-7715-47ca-814d-05c57406a454 --recording-key 2jAFMa9gyE5jLAsqGvLLlNsUrNTMzgqxfKJ05jdHyUVmeFMgNOfTwb5DenE1RTLu --out runs/pine-bot-route-reproduction
```

Omitting `--recording-key` selects the median by reported race time, with filename
as the tie breaker. Existing output directories are refused. The command resolves
`RecordingRegister` through the installed Addressables catalogue, reads its
metadata, selects only the requested race, checks the selected XML against the
register, and saves source bytes, `trajectory.csv` and `report.json`. It does not
launch Liftoff or execute its managed assemblies. UnityPy may report a fallback
version warning for stripped bundles; the fallback is read from `resources.assets`.
The register format is based on this verified build and may need updating after a
game format change. UnityPy is optional and not needed to fly or import plain XML.

For an already exported XML or gzip recording:

```powershell
.venv/Scripts/python.exe -m haltere.cli liftoff import-replay runs/pine-bot-route-01/source.recording --out runs/pine-bot-reimport
```

The trajectory contains original world position, source timestamps and indices,
plus positions in Haltere's frame relative to the **source's first pose** and
body-to-world quaternions in `wxyz` order. Inputs are named `recorded_input_0..3`.
They are deliberately not labelled throttle/yaw/pitch/roll: replay channel order,
scaling, vehicle configuration and compatibility with our action mapping still
need verification. A separate inspected memorial recording had all-zero inputs,
so the mere presence of input fields is insufficient.

The loader can adjust ghost playback speed based on player race times. A bot seen
on screen therefore need not move at the source recording's timing. The official
[telemetry contract](https://steamcommunity.com/sharedfiles/filedetails/?id=3160488434)
excludes replay sessions: player UDP poses must not be paired with replay footage.
FPV camera alignment remains unverified, so this export is **pose data, not a
labelled vision dataset or a verified action-imitation dataset**.
