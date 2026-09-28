# Contact audit (round 4b)

`haltere/liftoff/contact_audit.py`, declaration `configs/contact_audit.json` **version 1**. Offline scoring only:
no pilot reads it.

## Why

The ground contacts that earlier rounds counted were the pilot's own support-climb onsets. Those depend on the
pilot: the descent view keeps the sink request small, so the older support rules could not fire at all on the
Minus Two floor (`minus-fast6-r4-02`, 27.4-28.9 s). A graduation bar of "no ground contact" needs a count that
comes from the telemetry, not from the pilot.

## What it measures

From the 100 Hz CSV alone, one row per telemetry frame:

- **Support.** The external vertical specific force: the measured acceleration plus gravity, less the thrust of the
  game's processed throttle (`in_thr`, through the measured thrust curve of
  `runs/measured-dynamics-low-speed-20260923/profile.json`) along the drone's up axis, less the measured body drag.
  A surface can only push, so a ground reaction shows as an upward force. It counts when the 0.3 s mean is at least
  1 m/s^2 (more at body rates above 3 rad/s), with the drive (`(in_thr + 1)/2`) within 0.4-0.8 over the window.
  Outside that range the curve is not reliable (idle thrust, spin-down).
- **Arrest.** A descent of at least 0.3 m/s that turns upward by at least 1.5 m/s^2 while the throttle stayed below
  hover for the window and 0.1 s before it. Below hover the propellers cannot arrest a descent; this covers throttle
  cuts, where the curve is not used.
- **Impact.** The 3-frame acceleration above 20 m/s^2 with more than 6 m/s^2 off the thrust axis, the criterion of
  `flightlog.collisions`.
- **Height** (reported). The lowest height of the episode, and whether it is within 0.3 m of the launch plane. On
  Minus Two the launch plane is the garage floor.
- **Terminal impact.** The runner's impact record from the sidecar. The run stops on it, so its frames are not all
  in the CSV.

Frames within 0.3 s of each other form one contact. A contact needs 0.15 s of support or arrest, or one impact
frame. Each contact reports:

- start and end time;
- position, horizontal speed and vertical speed at the start;
- the peak and mean external force, and the vertical impulse (the integral of the upward external force, m/s);
- the lowest height;
- its kind: impact, support, or slide (support for at least 1 s at a median horizontal speed of 1 m/s or more).

Frames before 3.5 s (the arming hold and throttle ramp) and before the first departure from the launch plane are
not scored.

```
python -m haltere.liftoff.contact_audit runs/fast-stack-20260923/straw-brain08-04.csv --json OUT.json
python -m haltere.liftoff.contact_audit validate --out DIR --sheets            # review items and sheets
python -m haltere.liftoff.contact_audit validate --out DIR --labels LABELS.json   # the frozen validation
```

## Validation

The values and the validation were frozen together in `configs/contact_audit.json` version 1
(`1c82c7f4`, commit `d15faeb`), before the validation was run.

**The validation is not blind.** The values are the first round-4b attempt's draft, unchanged. That
attempt ran its draft on a subset of logs it did not record, and it may have included validation logs.
This attempt checked the draft on design logs only: `straw-brain06-02`, `straw-brain07-02`,
`straw-fast6-03`, `pine-fast6-ttc-01`, `pine-brain08-01`, `minus-fast6-vg-02` and
`minus-brain08-vg-01`. The labelled cases were confirmed on the videos before the freeze.

Scores: `docs/experiments/contact_audit_v1_validation.json`. Every log with at least 100 rows: 58 logs,
83.9 scored minutes, 72 contacts (41 slides, 23 supports, 8 impacts).

| Check (frozen) | Threshold | Result | Pass |
|---|---|---|---|
| The ten Straw Bale downhill contacts of `straw-brain08-04`/`-06` | >= 9 of 10 detected (a contact within 2 s before to 0.3 s after each support-climb onset) | 10 of 10; each contact starts 0.87-1.10 s before the onset and lasts 1.36-1.59 s | yes |
| The `straw-brain08-01` hillside slide (70.5-75.8 s on the video) | >= 50% covered, one contact of kind slide | 99.8%: one slide from 68.13 s to the impact | yes |
| The `minus-fast6-r4-02` garage floor (27.4-28.9 s, z < 0.1 m) | detected | 27.33-27.99 s (min z 0.02 m); the second touch at 29.1-29.6 s also (29.02-29.55 s) | yes |
| Terminal impacts | every impact-ended run listed | 46 of 46 (the CSV itself carries contact evidence in its last second for 13) | yes |
| False positives on video | <= 0.05 per scored minute | **0.32 per minute** (27 of 45 review items not confirmed; none clearly airborne) | **no** |

Details:

- **The 45 review items** are the contacts outside the labelled windows, not within 1 s of a terminal
  impact, and not confirmed by height on Minus Two. Each was labelled on a 12-frame contact sheet, after
  the freeze and after the audit had run. Labels and notes: `docs/experiments/contact_audit_v1_video_labels.json`
  and `contact_audit_v1_video_label_notes.json`.
  - **18 confirmed contacts:**
    - the uphill skims of the brain-06 laps (x 30-34, y 130-155) and `straw-brain6-02`'s uphill scrape;
    - two hilltop tumbles;
    - six downhill touches in other brain-08 laps;
    - Pine Valley hillside slides;
    - a tree hit on `pine-fast6-ttc-01` and a course flag on `straw-brain05-trim-02` (the propeller
      damage icon appears on both).
  - **27 ambiguous items.** All are at the Straw Bale downhill contact spot (x -36..-39, y 135-172),
    in the brain-06/07 laps and once in `straw-brain08-04`. The nose is level there, so the view shows the far field from the
    lower image edge up and never the ground in front. The frames cannot show a touch, and they do not
    show the drone clearly airborne either. The frozen rule counts ambiguous items as false positives,
    so the gate fails.
  - **None was labelled clearly airborne.**
- **Report only, not a frozen check: a terrain-height cross-check.** Terrain points are the drone's
  recorded positions during video-confirmed contacts (4048 points). 26 of the 27 ambiguous items fly
  within 0.3 m (median) of those heights, within 1.5 m horizontally; their median offsets are -0.26 to
  +0.16 m. The 27th, with only 5 matched frames, lies 0.37 m below them. The same height as the
  confirmed touches, a steady descent below hover thrust, and a vertical speed that does not follow the
  request: these point to ground contact the camera cannot see.
  `docs/experiments/contact_audit_v1_terrain_check.json` (script: session scratchpad `m4b/contact/a2/terrain_check.py`).
- **Recall report.** 14 of the 15 support-climb onsets the pilots logged themselves, on every log,
  have an audit contact within 2 s before to 0.3 s after. The one without is `straw-brain03-01` at
  85.98 s.

**What this means for graduation scoring.** The audit finds every video-labelled contact, the slide and
the floor. As frozen, its false-positive check fails. On video the failure is that a level-nose
downhill slide cannot be seen, not that a contact was reported in clear air. Before the audit decides
a graduation run, a version 2 of the validation would need one of these, frozen before it is scored:

- a labelling procedure that does not rest on the forward camera alone (the terrain-height evidence
  above, or a chase view);
- or an independent label for such slides.

Until then, report its contacts next to the support-climb onsets and the videos rather than as a gate.
