# Flight cards

Evaluate the complete deployed stack under the [current project direction](../project_direction.md).
Record assistance and predictor use explicitly. A software-assisted flight may
be autonomous; oracle-route collection and human-controlled shadow recordings
must remain separately labelled. Freestyle cards must state the requested task
and success criterion rather than substitute a gate count.

One card per flight, written **before** it and appended to after. A prediction written after the
flight does not exist. Rules: `docs/generalisation.md`.

## 2026-09-22 assisted scene brain

Newest completed local brain `runs/scene-brain-09-navigation/last.pt`, full
Rabbit gate selection, smoothing, speed and yaw; trained neural throttle,
roll and pitch. Frozen detector, mapping and pilot parameters across the three
maps; no navigation predictor, route, course geometry or explicit cue parser.
The cards pin hashes and commands. Liftoff 1.7.6 / build 25441586 ran entirely
inside hidden Anode with the original `[Copy] New Drone`.

**Zero laps or races completed in five attempts.** Times below are seconds
since the first controller telemetry frame, including its arming period;
the game's race clock starts later, on crossing the start line.

| Map / race | Exposure | Control duration | Observed result | Card / local recording stem |
|---|---|---:|---|---|
| Straw Bale / Field Day | Seen | 96.2 s | Initial arch sequence, lost next gate, hay-bale collision; pilot estimated 2 passes | [Attempt 1](2026-09-22_strawbale_assisted_1.md), `straw-01` |
| Minus Two / Turn Signals | Seen | 17.4 s | Search and pillar collision before race timer started | [Attempt 1](2026-09-22_minustwo_assisted_1.md), `minus-01` |
| Hangar C03 / Shipments | First project exposure | 30.5 s | Timer started, poor gate acquisition, roof-structure collision | [Attempt 1](2026-09-22_hangarc03_assisted_1.md), `hangar-01` |
| Hangar C03 / Shipments | Unchanged repeat | 30.5 s | Same failure | [Attempt 2](2026-09-22_hangarc03_assisted_2.md), `hangar-02` |
| Hangar C03 / Shipments | Unchanged repeat | 30.5 s | Same failure | [Attempt 3](2026-09-22_hangarc03_assisted_3.md), `hangar-03` |

A preceding [bounded preflight](2026-09-22_strawbale_assisted_preflight.md)
passed controlled takeoff and recording, stopping at the planned 20 m distance
bound. That is not counted as a completed race. Both ground control checks
confirmed throttle-low and all four processed axes.

Raw recordings live in the ignored local folder `runs/assisted-races-20260922/`.
Each stem has `.mp4`, `.csv` and `.json`; videos show the standard live brain
panel beside gameplay, at 1928×720, H.264, 18 fps. All six videos (including
preflight) fully decode without errors. They are not included in a fresh clone.
The folder also preserves the model manifest, ground checks, pad logs and
`*-score.json` outputs. Keep CSV and JSON together: impact guards may stop
before the final CSV sample, and the scorer reads the sidecar to retain that
terminal event. CSV contact estimates and terminal impacts may overlap.

No runtime tuning occurred between these race attempts. Hangar was absent
from the project training/flight inventories inspected before its first test;
later attempts are repeats of that test, not additional unseen maps. There is
no live unassisted comparison or freestyle evaluation in this batch, so it
does not establish an assistance improvement or freestyle generalization.
No new weights were trained or promoted. Gate acquisition, turn recovery and
obstacle-aware search remain visible problems to fix.

## Card template

Name: `<date>_<track>_<n>.md`. Template:

```markdown
# 2026-09-16 pinevalley 1
- **Rung**: Z4 (unseen map, unseen gate type)   <!-- Z0-Z4, see docs/generalisation.md -->
- **Track**: Pine Valley / "1 - Forest For The Trees"  (pool: dev)
- **Detector**: runs/gatenet10/best.pt  sha256 <first 12>
- **Brain / code revision**: <checkpoint hash and commit>
- **Assistance / predictor / visible cues**: <mode, parameters, predictor hash or absent, cue set>
- **Recording mode**: <live autonomous, human demonstration, oracle-guided collection, or shadow>
- **Command**: <the exact line, copy-pasteable>
- **Course parameters typed on the command line**: none   <!-- must be "none" on a sealed track -->
- **Prediction**: <what you expect, as a number, before flying>
- **Pass criterion**: <decided now, not after>

## Result
- HUD checkpoints: n/m, splits ...
- Pilot's own `n_passes`: ...  (precision/recall against the HUD: ...)
- `liftoff score --gates ""`: speed, shake, contacts
- Verdict against the criterion: pass / fail
- What it changed: <if this was a sealed track and anything here changes the code, the track
  becomes dev - record that>
```
