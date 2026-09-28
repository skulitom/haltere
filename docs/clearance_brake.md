# Clearance brake sink floor (wall pilot version 5, round 4b)

**Status: not flown.** `configs/obstacles/wall_pilot.json` **version 5** adds one rule,
`clearance_brake` (`haltere.liftoff.fast_race_cue.ClearanceBrakeConfig`). The rule is part of the
obstacle stack: `--obstacle-stack on` applies it, and `shadow` measures and logs it without applying
it. Turn-first, its stopping models and the ceiling guard are version 4's values, unchanged. Version 4
is kept as `wall_pilot_v4.json`. The runner refuses it, but the replay harness can rebuild it with
`--wall-pilot configs/obstacles/wall_pilot_v4.json`.

## The failure (`minus-fast6-r4-02`, development case)

The round-4b guard agent found it; this branch replayed it before the freeze:

- From 26.4 s the TTC governor's stand-off cap (0.53-0.62 m/s) acted along a looming ray tilted
  about 11 degrees up, (-0.80, 0.58, 0.19): the travel direction at the sample's capture, while the
  drone rose out of a dip. Wall samples renewed it until 29.7 s.
- The brake step subtracts `ray x (along - cap)` from the request. Its vertical part turned the pilot's
  and the guard's +0.1..+0.23 m/s into -0.3..-0.5 m/s. In the open-loop replay of the flown stack, the
  brake added more than 0.01 m/s of sink on 7.1 s of the flight, at most 0.68 m/s.
- The drone skimmed the garage floor: z < 0.1 m for 1.5 s at 27.4-28.9 s, and again at 29.1-29.6 s.
- The sink request below -0.3 m/s also fed the descent-path governor, which cut the horizontal request
  to 0.83-0.93x.

## The rule

After each of the two brake steps (on the request, and `_cap_command` on the command), the vertical
value is at least `min(its value before the step, 0) - max_added_sink`, with `max_added_sink` = 0 m/s.
So:

- the brake may reduce a climb to level;
- it may still raise the request when the ray points down (terrain ahead and below);
- it never adds sink;
- the pilot's own sink and the vertical guard's sink are kept;
- the horizontal part of both steps is unchanged.

The alternative, correcting only the horizontal part, would brake harder horizontally on every rising
ray and change every Minus Two log. The floor changes only the vertical request, and only on ticks
where the brake added sink. It was chosen before any replay of version 5 was scored.

With the stack on, the sidecar's `wall_pilot.clearance_brake` records the rule, its parameters, the
seconds on which the brake added, withheld and left sink, and the largest of each. The replay harness
logs the same per tick (`brake_added_sink`, `brake_sink_left`, `brake_sink_withheld`,
`brake_reference`, the ray and `braking`). The CSV columns are unchanged: `cmd_vz` against
`vertical_target` shows the effect.

## Gates (frozen) and scores

`configs/obstacles/wall_pilot_gates.json` **version 2** (`b6ce9d6f`) was frozen with version 5 (commit
`d15faeb`). Before the freeze only the development case, the `minus-fast6-r4-02` replay, had been
looked at. Version 1 of these gates (turn-first version 4) is kept as `wall_pilot_gates_v1.json`.

The replays are open-loop replays of 24 logged flights: 12 Minus Two flights with the stack's live
looming samples, 3 Pine Valley flights, and 9 Straw Bale laps. The Straw laps and `pine-brain08-01` use
the offline looming stream. Each flight ran through this tree (version 5, and the kept version 4) and
through a `git archive` of `m4` (`3decaac`). The recorded motion does not respond to the requests.
Scorer: `python -m haltere.obstacles.wall_pilot_gates --out PREFIX --baseline PREFIX --json SCORES`.
Scores: `docs/experiments/wall_pilot_v5_scores.json`.

| Gate | Threshold | Result | Pass |
|---|---|---|---|
| B-Identity: stack off, shadow, and the kept version 4 (also with descent view 1) against `m4` | every command array bit-identical | 96 of 96 | yes |
| B-NoSink: sink the brake leaves with version 5, every flight, with and without descent view 1 | 0 | 0 on every tick | yes |
| B-R402: `minus-fast6-r4-02`, the stack as flown (descent view 1) | no brake-made sink; version 4 must add sink for >= 1 s | version 4 adds sink for 7.1 s (up to 0.68 m/s); version 5: none. Lowest vertical request -0.50 -> -0.39 m/s (the rest is the pilot's own) | yes |
| B-Horizontal: every Minus Two flight, ticks where version 4 brakes | version 5's request along the wall ray's horizontal direction exceeds version 4's by <= 0.1 m/s | worst +0.069 m/s (`minus-fast6-r4-02`; every other flight <= 0.001) | yes |
| B-Quiet: Pine Valley and Straw Bale | <= 1 s per flight with sink withheld (> 0.01 m/s) | Pine 0-0.78 s; Straw **1.37 s** (08-04), **2.90 s** (08-06), **1.10 s** (08-01), **1.001 s** (fast6-02), others 0-0.87 s | **no** |

What the replays show:

- **Version 4's brake adds sink on most flights.** Along a rising travel ray it did so on 9 of the 12
  Minus Two flights. On `minus-fast6-r4-02` that was 7 s, up to 0.68 m/s; on the other eight, 0.5-3.7 s
  each, up to 0.07-0.33 m/s.
- **On Pine and Straw the floor acts where version 4 added sink during climbs:**
  - the Straw hilltop approach (`straw-brain08-06` at 59.8-61.1 s, climbing 0.5 m/s: up to 0.41 m/s
    withheld);
  - the Straw uphill (`straw-fast6-02` at 237.9 s: up to 0.29 m/s);
  - the Pine mound's tree line (`pine-fast6-ttc-01` at 16.2-17.1 s: up to 0.43 m/s);
  - brief withholds of 0.03-0.08 m/s at the Straw start (x 65, y 3).

  These are the same failure the rule fixes on Minus Two: a brake that sinks the drone toward rising
  ground. They still break the frozen quiet threshold of 1 s on four Straw laps, so **B-Quiet fails as
  frozen**. The rule was not changed after scoring.
- **The horizontal effect is kept.** Where version 4 brakes, version 5's horizontal request toward the
  wall is the same to within 0.001 m/s on 11 of 12 Minus flights. On `minus-fast6-r4-02` it is up to
  0.069 m/s higher: version 4's brake-made sink below -0.3 m/s had fed the descent-path governor, which
  cut the horizontal request, and version 5 no longer does.

## Risks

- **Not flown.** Open-loop replays cannot show where the drone would have gone. On `minus-fast6-r4-02`
  the drone would have stayed level instead of sinking, 0.7 m above the floor, and the next events
  (the guard's floor-read climb, the ceiling) would have happened elsewhere.
- **A climb the pilot asked for can still be reduced to level** when a wall lies along a rising ray.
  That is intended: a climb toward a wall closes on it. It is not a sink.
- **Seconds of change on Pine and Straw** (B-Quiet): the rule removes up to 0.43 m/s of sink during
  climbs toward the Straw hilltop, the Straw uphill and the Pine tree line.
