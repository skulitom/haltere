# View-keeping descent (round 4) and contact support (rounds 4b and 5)

**Status.** Off by default (`--descent-view on|off|DECLARATION`). The declaration is
`configs/pilot/descent_view.json` **version 3** (round 5). It keeps version 1's view rule unchanged
and carries contact support version 3; see [Version 3](#version-3-round-5-contact-support-version-3).
Round 6 adds an optional companion rule, the sighted descent (`--sighted-descent on|off|shadow`, off by default;
[sighted_descent.md](sighted_descent.md)): while the ring is clipped at the bottom edge, it lowers this rule's sink
bound (steep late included) to the ring's sighted line of sight; this declaration and its values are unchanged.
`--contact-support on|off|shadow` (default `on`) isolates the contact rule. Versions 1 and 2
(`descent_view_v1.json`, `descent_view_v2.json`, kept) are refused by the runner. Version 1 flew on
Minus Two in round 4 and version 2 in round 4b, both as disclosed development deviations. Frozen gates:
version 1's are `configs/pilot/descent_view_gates.json` version 1, version 2's
`configs/pilot/contact_support_gates.json` version 1, version 3's `configs/pilot/safety_gates.json`
version 1 (with the ceiling guard of wall pilot version 6).

## Version 3 (round 5): contact support version 3

**Not flown.** Version 3 (`2bdb17fc2479...`, frozen with the safety gates in commit `68446ee`) keeps
version 1's view rule and version 2's contact-rule values, and changes three things the round-4b review
found. With the new fields unset (version 2's declaration), the rule is version 2's bit for bit: the unit
test's golden digest of the version-2 pilot equals the one computed in a git archive of `m4b`
(`bd8ec886...`).

1. **Manoeuvre windows are not used.** A window whose highest pitch/roll body rate exceeds
   `max_body_rate` (2.5 rad/s) or whose horizontal speed along its first horizontal velocity falls
   faster than `max_braking` (4 m/s²) is treated like a window outside the drive range: no suspicion
   and no gain learning.
   - Why: the fast PD's airborne hard brakes read 1.0-1.8 m/s² of unexplained upward force for
     0.15-0.29 s (13 command-agnostic runs on the 58 earlier logs, 11 of them fast-PD brake flares).
     They start as the nose pitches up at 3.2-5.5 rad/s, before the horizontal speed falls: the
     thrust of the mean drive underestimates a motor set split hard for the pitch. No support climb
     came only because the pilot was not asking to sink there.
   - The rate is the pilot's input `omega` (the runner's quaternion-derived body rates; the logged
     `omega_x`/`omega_y`), so the replays and the runner see the same value.
2. **The gain is learnt before arming.** From `learn_after_s` (3.0 s after the first tick, when the
   runner's throttle ramp ends; every sample of the window at or after it), never while launching or in
   a support climb, every valid window that is quiet (at most 0.4 m/s² at the current gain) or rising
   (lowest vz at least 0.05 m/s) adds its observed gain. The gain is the median of those. The rule arms
   at 3.2 s as before, but only once `arm_learn_s` (0.5 s) of such windows were collected; then version
   2's slow learning continues.
   - Why: in the round-4b integration harness 2 of 12 hairpin drones whose thrust was about 15%
     above the curve read 1.2-1.5 m/s² at arming (gain still 1), fired, and rose into the 2.2 m
     ceiling.
   - A drone that never gives such a window never arms (the older support rules still act).
3. **Its climb never ends turn-first** (`turn_first_handoff` false). An older support climb still
   hands off as before.

**Switch.** `--contact-support on` (default) flies the rule; `off` leaves it out (the view rule alone,
version 1's pilot); `shadow` computes and logs it (`contact_fire` marks where it would start a climb,
then it waits `support_climb_s` as if climbing) and changes nothing: no climb, no reset of the older
rules' timers, no turn-first effect. Its commands equal `off` bit for bit. The deployed pilot
(`haltere.train.deployed_pilot`, `contact_support=`) and the replay harness (`--contact-support`, tags
`-csoff`/`-csshadow`) follow it.

**Logs.** With `on` or `shadow` the CSV adds `contact_unexplained`, `contact_gain`, `contact_fire`,
`contact_armed` and `contact_excluded` after the view columns. The sidecar's
`pilot_assistance.contact_support` adds `applied`, `armed_at_s` (seconds after the first tick) and
`prearm_gain`; `descent_view_declaration` records the `contact_support` mode.

**Development evidence (before the freeze, disclosed in the declaration).**
- The window features and a threshold sweep on the 58 earlier logs (the three round-4b live logs held
  out; session scratchpad `m5/safety/features.py`, `sweep_exclusion.py`). A window-max rate bound of
  2.5-3.5 rad/s removed 11 of the 13 false-read runs and changed no detection of an audited contact
  (45 detected, first fire 0.27 s median, 0.52 s at most, as without it); 2.0 rad/s delayed one to
  0.68 s. The braking bound alone removed 3 of the 13 and none beyond the rate bound.
- The draft rule on those logs (`dev_detector.py`): every audited contact version 2 fires in, version 3
  fires in too, at the same time within 0.01 s; no onset outside a contact; arming at 3.8-4.8 s; the
  two command-agnostic runs left are a brain-06 climb at +1.9 m/s (279.4 s) and a brain-07 hilltop
  (67.6 s, 0.97 m/s²).
- The motor-assist harness on sim seed 23 with the fast PD (where round 4b found the arming fires):
  version 2 fired 3 times on each of the same 2 of 12 hairpin drones and 2 drones touched the ceiling;
  the draft fired on none, armed at 3.8 s, and flew the same as `--contact-support off`.

Scores: see [Gates](#round-5-gates-configspilotsafety_gatesjson-version-1) below.

### Round 5 gates (configs/pilot/safety_gates.json version 1)

Frozen with both declarations (`c6dfc88cae88...`, commit `68446ee`) before any gate run;
`haltere.liftoff.safety_gates` runs and scores them. Scores: `docs/experiments/round5_safety_scores.json`.
Held out: the three round-4b live logs (`minus-fast6-r4b-01`, `minus-brain11cw13-r4b-noassist-01`,
`straw-brain11cw13-r4b-noassist-02`), harness sim seeds 101 and 102, and surrogate course seeds flat and
steep 3100-3103 and hill 6200-6207. Nothing here is flight evidence. **10 of 13 gates pass; the three
failures are all contact support version 3's.**

| Gate | Evidence | Threshold | Result | Pass |
|---|---|---|---|---|
| SG_Identity | deterministic replays; the 3 held-out logs included | 6 replay pairs x 27 logs bit-identical: default, shadow stack, stack with wall 5, with wall 5 + descent view 2, with wall 5 + descent view 3 `--contact-support off` (all against a git archive of `m4b`), and shadow against off | 162 of 162 | yes |
| SG_Ceiling_Governor | synthetic, development | version 6 bounds the request within 0.1 s with no governor climb; never for an explained sample; with a governor climb as version 5 | version 6: 0.05 s (above-path and live-garage samples), explained: none, case B equal to version 5 (0.05 s); version 5 without a governor climb: never | yes |
| SG_Ceiling_Pilot | synthetic, development | support, contact and search climbs levelled (<= 0.05 m/s) within 0.3 s, with an overhead engagement | version 6: 0.11 / 0.11 / 0.08 s; version 5: no engagement (level only when the climb ended, 0.49 / 0.49 / 1.13 s) | yes |
| SG_TurnFirst | synthetic, development | a contact-support climb keeps turn-first (toward the wall <= 0.05 m/s); older climbs and version 2 hand off | version 3: episode kept, 0.0 m/s toward the wall; older climb and version 2: handoff, 0.969 m/s toward the wall | yes |
| SG_Ceiling_Pine_Mound | development | `pine-fast6-ttc-01` vertical request 3.5-6.5 s identical with wall 6 and wall 5 | identical (up to 3.5 m/s) | yes |
| SG_Ceiling_Quiet | development; held out for the round-4b Straw lap | <= 0.2 s/min of vertical request lowered by version 6 on Straw and Pine | 0.0 s/min on all 13 logs | yes |
| SG_Ceiling_Minus_Report | report | - | version 6 changes no request on the 14 Minus logs (both held-out ones included) | - |
| SG_Contact_Rest | the review's scenario (development for the fast PD) | first support climb within 0.8 s of the rest at 0-6 m/s, none before, armed by 4.5 s | fast PD 0.28 s (armed 3.8 s); brain-09b 0.41-0.48 s; fast-brain-10b 0.39-0.41 s; brain-08 0.44 s at 3-5 m/s but **never armed at 0-2 m/s** and not by 4.5 s at 6 m/s (an older rule at 0.74 s); fast-brain-11-b-cw13 0.37 s at 4-6 m/s but **never armed at 0-2 m/s**, an older rule at 2.95 s at 3 m/s | **no** |
| SG_Contact_R402 | development | an onset and a shadow mark in 27.4-28.9 s | onsets 27.718 and 29.378 s; shadow marks the same | yes |
| SG_Contact_Detection | development (45 contacts); held out (1) | wherever version 2 fires in an audited contact, version 3 fires at most 0.1 s later | 45 of 45 development contacts, same times within 0.01 s; **the held-out Straw downhill touch (`straw-brain11cw13-r4b-noassist-02`, 79.35 s): version 2 at +0.30 s, version 3 none** | **no** |
| SG_Contact_Clean | development + held out | no onset outside the audit's contacts | 0 on 61 logs (90.38 min) | yes |
| SG_Contact_Flare | held out | no command-agnostic false-read run outside contacts | `minus-fast6-r4b-01`: version 2 one run (42.62-42.90 s), version 3 none; **`straw-brain11cw13-r4b-noassist-02`: one run for both (62.43-62.80 s)**; `minus-brain11cw13-r4b-noassist-01`: none | **no** |
| SG_Contact_Arming | held out (fresh seeds) | round-5 stack: no contact onset in the harness (no ground reaction there); ceiling contacts <= contact support off | 0 onsets for all 5 motors (180 drone runs; the round-4b stack: fast PD 3, brain-08 5, brain-09b 3, fast-brain-10b 5, fast-brain-11 5); ceiling contacts equal to off for every motor (fast PD 11, brain-08 8, brain-09b 6, fast-brain-10b 8, fast-brain-11 1; round-4b stack 11, 10, 8, 9, 2) | yes |
| SG_Contact_Surrogate | held out (fresh course seeds) | no onset; every course identical to contact support off | 0 onsets; 48 of 48 identical (fast PD, fast-brain-11-b-cw13, fast-brain-10b; armed at 3.8 s everywhere) | yes |

**Post-scoring diagnosis of the failures (not tuning; nothing was changed after the scores).**
- **SG_Contact_Detection.** The held-out downhill touch is an impact-style touchdown at 5.2 m/s: at
  79.50 s the vertical speed jumps from -2.6 to -1.3 m/s, the knock pitches the body at up to
  3.9 rad/s for 0.05 s, and friction brakes the horizontal speed from 5.2 to 4.6 m/s.
  - Version 3 excludes every window that contains that spike or the brake: 79.51-79.84 s, 0.33 s.
  - Afterwards the drone sank faster than the logged command (-1.02 m/s against -0.82), so the rule
    was no longer suspicious.
  - Version 2 fired at 79.654 s.
  - The exclusion cannot tell a commanded pitch (the flare: the rate comes first, then the residual)
    from a knock (both at once). A version 4 would need that ordering, or a rate spike shorter than
    the window, before it excludes.
- **SG_Contact_Flare.** The held-out run is a commanded climb (+1.05 m/s, vz up to +1.41 m/s) at the
  Straw Bale hilltop crest (x about 0-2, y 196, z 26.9-27.4) at a body rate of 0.13 rad/s: 3.15 m/s²
  of unexplained force, the same for versions 2 and 3.
  - It cannot start a climb (the command is not a sink request).
  - It is either a crest scrape the audit does not count, or a thrust-model error in a climb.
  - The development logs had the same kind at the crest (`straw-brain07-03` 67.6 s) and in a climb
    (`straw-brain06-03` 279.4 s); the rule was not designed to remove those.
- **SG_Contact_Rest.** The scenario holds the measured motion level whatever the motor issues.
  - At 0-2(3) m/s brain-08 and fast-brain-11 issue less thrust than the curve's hover. The held
    motion then reads as an upward residual in every window: no window is quiet or rising, and
    version 3 never arms. Version 2 armed on time alone.
  - In the live logs every flight armed, at 3.82-4.83 s (all 61 logs). In the harness it armed at
    3.8-8.26 s, and at 3.8 s in the surrogate.
  - It is also a real property: a vehicle whose free flight never matches the curve never arms.
- **Blind time.** Version 3 does not use 361 s of the 90.4 logged minutes (6.7%). For the fast PD in
  the Minus garage it is up to 21.9 s/min (`minus-fast6-r4-02`): the rule is blind in much of the
  garage flight, where brakes and turns are hard.

## Version 2 (round 4b): contact support

**Flown in round 4b** (a disclosed development deviation; superseded by version 3). Version 2 keeps
version 1's view rule, value for value (the `descent_view` block is identical), and adds
`contact_support` (`haltere.liftoff.fast_race_cue.ContactSupportConfig`). Its
frozen gates are `configs/pilot/contact_support_gates.json` version 1
(`haltere/liftoff/contact_support_eval.py`).

### Why

Both older support rules need a sink request below -0.8 m/s. Version 1's view bound keeps the request
above that below about 5 m/s. The round-4 review found this, and `minus-fast6-r4-02` showed it live.
The drone lay on the Minus Two garage floor at 27.4-28.9 s (z < 0.1 m) with a request of
-0.3..-0.5 m/s, and no support climb came. (That request was itself the clearance brake's; see
[clearance_brake.md](clearance_brake.md).) A drone resting on a floor or a hill at low speed got no
support climb at all.

### The rule

It reads the ground reaction instead of the size of the request:

- **Thrust.** The propellers push only along the drone's own up axis. The thrust comes from the issued
  throttle through the measured curve of `runs/measured-dynamics-low-speed-20260923/profile.json`:
  `g * 3.138 * drive^1.973`, with `drive = (processed + 1)/2` from the pad calibration, acting 0.03 s
  after it was issued. The measured body drag is subtracted.
- **Unexplained force.** Over the last 0.3 s it compares the change of the measured vertical speed
  with what that thrust, gravity and drag explain:
  `unexplained = dvz/dt - (gain * mean(thrust * up_z) - g - mean(drag_z))`. In free air it stays near
  0; on the logged flights the median is -0.03..-0.08 m/s^2 and p10-p90 about +-0.2 m/s^2 at a drive of
  0.5-0.7. A surface can only push, so a sustained positive value is a ground reaction. Windows with any
  drive outside 0.4-0.8 are not used (idle thrust and spin-down leave +0.4..+3.7 m/s^2 there).
- **Support climb.** The older rules' 0.6 s climb at 1 m/s or more starts after 0.15 s of all of these:
  - the command asks to sink (at most -0.1 m/s);
  - the measured vertical speed does not follow it (at least 0.1 m/s above the command);
  - the drone is not climbing away (vz at most 0.5 m/s);
  - the unexplained upward force is at least 0.8 m/s^2.
- **Arming.** It is armed 3.2 s after the pilot's first tick (the runner holds the throttle for 1 s and
  ramps it until 3 s), and never while launching.
- **Thrust gain.** `gain` starts at 1 (the measured vehicle). On armed, valid windows outside a contact
  and a support climb, it follows the observed gain with a 1 s time constant, each step clipped to 0.15,
  within 0.75-1.3. It learns only when the unexplained force is at most 0.4 m/s^2 or the drone rises.
  The surrogate randomises the thrust by up to about +-20% at hover. The gain absorbs that: without it,
  a strong drone flying level looks like a drone on the ground.

It reads the measured velocity and attitude, the throttle the motor issued and the pad calibration:
no height above ground and no course geometry. With `--descent-view on` the CSV gains
`contact_unexplained`, `contact_gain` and `contact_fire` after the three view columns. The sidecar gains
`pilot_assistance.contact_support` (rule, parameters, onsets, seconds valid, suspected and learning,
and the final gain). With the flag off, the CSV and the sidecar are unchanged.

### How it was developed (disclosed)

Round 4b ran in two attempts. The first was cut off by a usage limit before it committed; its draft was
reviewed and reused unchanged. Everything below was development evidence, looked at before the freeze
(`d15faeb`):

- **First attempt:**
  - the rule's code on every logged flight, including the round-4 live logs;
  - the resting and floor scenarios for all four motors;
  - the surrogate on development seeds only (flat and steep 5000-5007, hill 6100-6111);
  - contact sheets of the labelled contacts;
  - the draft contact audit on an unrecorded subset of logs.
- **Second attempt:**
  - the `minus-fast6-r4-02` replay (onsets at 27.72 s and 29.38 s);
  - the resting scenario for the fast PD and fast-brain-10b;
  - the gain trace of `straw-brain6-02`.

The gate seeds of the surrogate were never run with the rule before the freeze. CS_Rest, CS_R402 and
CS_Clean are therefore not blind; CS_Surrogate and CS_Identity are.

### Gates (frozen) and scores

`configs/pilot/contact_support_gates.json` version 1 (`a7d033ce`, commit `d15faeb`). Scores:
`docs/experiments/contact_support_v1_scores.json`. None of it is flight evidence.

```
python -m haltere.liftoff.contact_support_eval rest|clean|surrogate --out DIR
haltere/obstacles/vertical_replay.py ...      (the variants the gates file lists, this tree and a git archive of m4)
python -m haltere.liftoff.contact_support_eval score --out DIR --prefix THIS --baseline M4 --audit AUDIT --json SCORES
```

| Gate | Threshold | Result | Pass |
|---|---|---|---|
| CS-Identity | Command arrays bit-identical to `m4` (24 flights x 4 variants): stack off, shadow, stack with the kept wall pilot 4, and stack with the kept descent view 1 | 96 of 96; plus the unit tests' golden digests of the default and version-1 pilots | yes |
| CS-Rest | The review's resting case with the motor that would fly (fast PD, fast-brain-08, brain-09b, fast-brain-10b) at 0-6 m/s: a support climb within 0.8 s at every speed, none before the rest | fast PD 0.28 s at every speed; brain-08 0.19-0.44 s; brain-09b 0.42-0.47 s; fast-brain-10b 0.39-0.46 s | yes |
| CS-R402 | `minus-fast6-r4-02` replayed through the round-4 stack as flown with version 2: an onset during the floor contact, 27.4-28.9 s | onsets at 27.72 s and 29.38 s (the second touch) | yes |
| CS-Surrogate | Gate seeds (flat and steep 3000-3007, hill 6000-6011), four motors, the deployed pilot (stack with wall pilot 5, plus descent view 2): no onset, and every course identical to its round-4 run | 0 onsets on 112 courses; 112 of 112 identical. The thrust gain ranged 0.93-1.21 over the randomised drones | yes |
| CS-Clean | The rule on every logged flight (58 logs, 87.4 min): no onset outside the frozen contact audit's contacts (+-1 s) | 0 outside; all 90 onsets are inside audit contacts: the Straw Bale downhill spot of the brain-06/07/08 laps, the `straw-brain08-01` slide, and the `minus-fast6-r4-02` floor | yes |

Notes:

- **CS-Clean leans on the audit.** Its clean segments are defined by the contact audit, and 27 of that
  audit's contacts cannot be confirmed on the video, although their heights match video-confirmed
  touches (see [contact_audit.md](contact_audit.md)). 51 of the 90 onsets are in those ambiguous
  downhill slides. The other 39 are in video-confirmed contacts: the labelled Straw contacts, the slide,
  other confirmed downhill touches, and the Minus floor.
- **In the review's case the first support climb is the contact rule's, at every speed and for every
  motor.** It then fires again every 1.2-1.3 s while the drone stays resting. An older rule fired only
  with the fast PD at 5-6 m/s, once, 1.82 s after the rest began: there the view bound allows more than
  0.8 m/s of sink.
- **Report: the round-4b stack on the 24 replayed flights** (wall pilot 5, descent view 2, the motor's
  issued throttle):
  - 22 contact onsets, all on the Straw Bale downhill and all inside audit contacts;
  - none on Minus Two or Pine Valley. On `minus-fast6-r4-02` the floor sink was the clearance brake's,
    which version 5 removes.
  - `docs/experiments/round4b_stack_replay_report.json` (scripts: session scratchpad
    `m4b/contact/a2/run_replays.py`, `run_gates.sh`, `full_stack_report.py`).
- **Report: the floor scenario** (`contact_support_eval.floor_scenario`). The nominal surrogate with a
  flat floor: level flight, then a ring below the floor, so the drone descends onto it and slides.
  - The fast PD, brain-09b and fast-brain-10b get a support climb 0.16-0.17 s after touchdown at
    0.5-6 m/s.
  - fast-brain-08 gets one after 0.16-0.17 s at 4-6 m/s, 0.32 s at 3 m/s and 0.9 s at 2 m/s. At
    0.5-1 m/s it gets none: brain-08 already sits on the floor during the level phase, with its
    throttle at the idle floor (a drive below 0.4, where the rule does not look).
  - `docs/experiments/contact_support_v1_floor_scenario.json`.

- **Wiring.** A CPU check built the runner's controller as `run()` does, with no camera and no pad:
  `--obstacle-stack on|shadow --descent-view on` for the fast PD, and `on` for brain-09b and fast-brain-10b.
  Every case declared wall pilot 5 (sink floor applied only with `on`) and descent view 2 (contact
  support, with the pad calibration), and wrote the six view and contact columns
  (`docs/experiments/round4b_contact_wiring.json`).

### Risks of version 2 for live flight

- **It has not flown.** The gates are open-loop replays, synthetic states and a surrogate without
  ground reaction.
- **The drone must be asked to sink.** The rule needs a sink request of at least 0.1 m/s that the drone
  does not follow. A drone resting on the ground with a level or climbing request gets no support climb.
  That is intended: nothing needs to climb. On `minus-fast6-r4-02` the floor contact came from the
  clearance brake's sink. With wall pilot version 5 that sink is gone, so the round-4b stack's replay
  has no request to sink there and no contact onset.
- **Throttle ranges.** Windows with a drive below 0.4 are not used: idle thrust, and a hard throttle
  cut. A motor that cuts to idle while resting (fast-brain-08 at 0.5-2 m/s in the floor scenario) is
  detected late or not at all. The older support rules still cover sink requests below -0.8 m/s.
- **The thrust gain.** An uphill scrape can raise the gain (1.195 on `straw-brain6-02`). A contact in
  the next 1-1.5 s is then detected less readily.
- **The thrust model.** The curve and the drag are the low-speed measured profile's. At high speed,
  in hard braking flares or in turns steeper than about 45 degrees (a drive above 0.8), the rule is not
  used or its model is less certain. No false support was seen on any log, but the logs never flew the
  rule itself.
- **Support climbs change the flight.** A climb of 0.6 s at 1 m/s on the Straw downhill lifts the drone
  off the hill, and then the view rule's sink resumes. In the round-4b stack's replay of the four
  brain-08 Straw laps with contacts, support climbs rise from 0-5 per file (m4 stack with descent view 1)
  to 3-11. All 22 contact-rule onsets lie inside an audit contact. Whether that brings more ring misses
  or turn-backs is a flight question.
- **False support with a thrust above the curve at arming (round-4b integration finding).** In the
  motor-assist harness's hairpin set (IdentifiedSim, 10% per-drone randomisation) the rule fires on the
  same 2 of 12 drones for every motor, the fast PD included, and each then touches the 2.2 m ceiling;
  without contact support none does. Those drones' thrust is about 15% above the declared curve (their
  gain ends at 1.13-1.17). At arming (3.2 s) the gain is still 1.0, the unexplained force reads
  1.2-1.5 m/s^2 while the pilot asks for a 0.1 m/s sink and the drone still rises, so the rule fires at
  3.4-3.5 s and again later. The descent surrogate's courses never trigger it, and the logged flights'
  gain stays within 0.95-1.10, but a gain near 1.1 at arming would give about 1 m/s^2. Flight card,
  Round 4b; `docs/experiments/round4b_integration.json` (`scenarios`).

## Version 1 (round 4)

**Version 1 fails its frozen surrogate gates for all three motors.** On the held-out seeds it
cut ground contacts by only 50% (fast PD), 69% (fast-brain-08) and 38% (brain-09b) against the
75% gate, although contact time fell 65-84%. It kept the flight path below the image for
36-53% of the descent time against the 5% gate. Checkpoint passes more than 1.5 m above the
centre rose from 1-3 to 13-24. Crashes and finishes were unchanged. On Straw Bale the
open-loop replay keeps the requested path mostly in the image before the 10 logged contacts: in
the 3 s before each, it points below the image 0-15% of the time, against the default pilot's
58-83%. That is not flight evidence. The rule is not ready to fly as a fix; see
[Risks](#risks-for-live-flight).

It answers the user's request of 2026-09-26: on the Straw Bale downhill the drone drops and
touches the hill; it should rather keep throttle up and pitch forward, and touch the ground
less.

## Diagnosis (Straw Bale logs)

Scripts: session scratchpad `m4/descent/diag.py`, `downhill.py`, `pitch_speed.py`.

- **Where.** straw-brain08-04 and -06 had 10 support-climb contacts, always at two spots per
  lap: y 161-166 (z 21.5-22.3) and y 144-147 (z 17.4-18.2), at x about -37. The hill there
  falls about 12 degrees (13.6 m over 65 m of the downhill; about 10.6 degrees between the
  first two contact points).
- **The 3 s before each contact.**
  - The pilot was in `below` (ring clipped at the bottom edge) 56-80% of the time.
  - It requested 3.0-4.0 m/s horizontally (`below_speed_fraction` 0.5 of 6 m/s). Measured:
    2.9-3.4 m/s.
  - It requested 1.0-1.4 m/s of sink, a path of 16-30 degrees down (the clipped edge's
    depression plus a margin that grows from 5 to 20 degrees while the clip lasts).
  - The camera's lower image edge lay 13-16 degrees down. The measured flight path pointed
    below the image 70-79% of that time.
  - The brain's throttle averaged 0.09-0.13 below hover (brain units, hover -0.43), with
    cuts to 0.27-0.36 below it.
- **The mechanism** (confirmed): every bottom clip began with a brake to half speed. The
  brake pitched the nose 10-20 degrees up, which lifted the lower image edge to between -8
  and +11 degrees. Rings that lay only 5-10 degrees below the drone were then clipped. The
  growing margin asked for 20-30 degree descents at 3 m/s, far below the image, into a
  12 degree hill. When the ring came back into view the drone was already on the hillside.
- **During contact** the descent-path governor saw the sink the ground prevented and cut the
  horizontal request to 0.35-0.53x. In straw-brain08-01 (the hillside slide) the drone
  slid for 5 s at 2-3 m/s: requested 0.35 x 6 m/s and 1.25 m/s of sink, nose pitched 12
  degrees down by the slope, throttle 0.37 below hover, ring in view.
- **The rings were visible with a level nose.** When the nose was down enough to see them,
  the Straw rings lay 5-10 degrees below the drone, inside a level camera's view.
- **Speed alone does not lower the camera.** Steady cruise at 5-6 m/s holds the nose only
  about 2 degrees down (drag-light drone; median over 7000-9600 steady ticks per flight), so the
  lower edge sits at about -14 degrees. Only acceleration pitches the nose down.
- **The fast PD** (straw-fast6-03) porpoises on the same downhill: brake in `below` (nose
  +12 to +33 degrees), then accelerate in `cue` (nose -28 to -45 degrees) to see the ring.

So the hypothesis of the task holds, with one correction: more forward speed does not by
itself let the path go steeper in view; what matters is not braking (no nose-up) and not
diving faster than the view.

## The rule (version 1)

`haltere/liftoff/fast_race_cue.py` (`DescentViewConfig`, `FastRaceCue._view_sink_bound`). It
reads the measured attitude and velocity, the pilot's own ring-cue state and the calibrated
camera. It uses no height above ground and no course geometry.

1. **View bound.** The pilot's own requested sink is bounded so that the flight path points
   at least 3 degrees inside the camera's lower image edge.
   - The path is the measured horizontal velocity plus the requested vertical speed.
   - The edge comes from the measured attitude, through an exact projection of the
     calibrated camera, roll included.
   - With a level nose that is a 9 degree path at any speed. A braking (nose-up) drone may
     sink less; a hovering drone at most 0.3 m/s.
   - The attitude term is low-passed with 1 s. With the unfiltered attitude, and with 0.3 s,
     fast-brain-08's pitch and the bound formed a 0.6-1 Hz limit cycle.
   - Climbs are never changed.
2. **Keep speed.** A bottom-clipped ring keeps the speed schedule instead of half of it. The
   descent-path governor is not fed while the bound withholds sink, and it never cuts below
   0.75x (was 0.35x).
3. **More speed, not less.** While the bound withholds sink toward a ring ahead, the
   horizontal request rises toward the speed schedule, never above the declared speed.
4. **Throttle up.** A descent starts at up to 2.5 m/s² instead of 5, so the motors are not
   asked for a deep throttle cut.
5. **Steep late.** A ring that stays clipped below for more than 0.75 s lies more steeply
   below than the view allows. The margin then falls at 6 degrees/s, down to 20 degrees below
   the edge. A steep leg is flown shallow first (over a convex crest after the upper ring)
   and steep late, instead of steep from the start.
   - Without this step (a strict bound) the development runs overflew steep lower rings by
     up to 8 m and circled back.
   - It is the only part of the rule that points the path below the image.

A steady descent steeper than about 12-14 degrees cannot stay in this camera's view: only a
lower camera uptilt (which needs the brain's sensory contract retrained) or a ground sensor
would change that.

Order in `FastRaceCue.update`:
- the pilot's own request (search, launch and launch-plane rules first);
- the view bound;
- the descent-path governor, the speed rise and the support rule;
- the looming governor and the vertical guard, which see the bounded request as the pilot's
  own;
- turn-first and the command slews.

The looming wall cap and turn-first still bound the raised speed.

## Flag, logs, sidecar

| Flag | Default | Effect |
|---|---|---|
| `--descent-view on\|off\|DECLARATION` | off | Fast pilot only (`--pilot-profile fast --pilot-assistance race-cue`). `on` loads `configs/pilot/descent_view.json`; the runner refuses an unfrozen or edited declaration and another version. |

With the rule on, three CSV columns are appended to each row (version 2 adds three more, see
[Version 2](#version-2-round-4b-contact-support)):
- `view_sink_bound`: the largest sink that keeps the path in view;
- `view_withheld`: the sink withheld from the pilot's own request;
- `view_boost`: 1 while the horizontal request was raised.

The sidecar records `pilot_assistance.descent_view` (version, rule, parameters, seconds
limiting and boosting, metres of sink withheld) and `pilot_assistance.descent_view_declaration`
(path, content and file sha256, version). With the rule off, the CSV and sidecar are exactly
as before.

## Declarations

| File | Version | sha256 (content) | Frozen |
|---|---|---|---|
| `configs/pilot/descent_view_v1.json` (was `descent_view.json`) | 1 | `8afb64d730ad...` | after the design on the development seeds, before any gate-seed run (commit `67d013b`) |
| `configs/pilot/descent_view_gates.json` | 1 | `d2b1e4bb312f...` | with rule version 1, before any gate-seed run (commit `7e96146`) |
| `configs/pilot/descent_view.json` | 2 | `7dc36efc6377...` | round 4b, after the development listed above, before any gate run (commit `d15faeb`) |
| `configs/pilot/contact_support_gates.json` | 1 | `a7d033cee819...` | with version 2 (commit `d15faeb`) |

The values were chosen on development seeds of the surrogate (`hill:6100-6111`,
`steep:5000-5007`, fast PD and fast-brain-08; brain-09b once). The declaration lists the
variants compared and their development numbers. No gate seed was run before both freezes.

## Offline rehearsal with terrain

`haltere/liftoff/fast_rehearsal.py` has an optional, scoring-only terrain model:
- `hill_course(seed)` builds climbs to a crest, then 1-3 descending legs of 6-30 degrees.
- `CourseTerrain` puts a hill under every leg that descends 2 m or more. The ground lies a
  seeded 1.2-2.5 m below each of the leg's checkpoints and follows a smoothstep between them,
  with a flat crest over 0-30% of the leg and a flat toe over 0-15%. It depends only on the
  distance along the leg. It exists while that leg is flown and for 1 s after the lower
  checkpoint's pass.
- `DescentScore` counts ground contacts (separated by 0.3 s), contact time, penetration,
  minimum clearance and speed into the surface. It also measures the time with the velocity
  vector below the camera's lower image edge while descending (exact projection at the true
  attitude), overall and on hills no steeper than 12 degrees, and counts checkpoint passes
  more than 1.5 m above the centre.

The pilot and the motor never see the terrain; the drone flies through it. Without terrain
`rehearse` is unchanged.

`haltere/liftoff/descent_rehearsal.py` runs the fast-brain development gate's closed loop
(`fast_motor_tracking.rollout`: 10% per-drone randomisation, HUD latency and dropout, command
delay, sim seed 17) for the fast PD or a fast-contract brain, with pilot keyword arguments and
this scoring. `haltere/liftoff/descent_gates.py` runs and scores the frozen gates.

## Gates (surrogate, held-out seeds)

Frozen in `configs/pilot/descent_view_gates.json` with rule version 1, before any gate-seed
run. Scores: `docs/experiments/descent_view_v1_scores.json`. Setup:
- seeds: flat 3000-3007 (no terrain), steep 3000-3007 and hill 6000-6011 (with hills);
- 150 s per course, sim seed 17;
- the baseline is the current pilot with the rule off, on the same seeds and drones;
- terrain gates pool the steep and hill sets; crash, finish and high-pass gates use every set.

Commands:

```
python -m haltere.liftoff.descent_gates runall --dir OUT
python -m haltere.liftoff.descent_gates score --dir OUT --json SCORES
```

| Gate | Threshold | Fast PD | fast-brain-08 | brain-09b |
|---|---|---|---|---|
| Contacts | >= 75% fewer, none new | **fail**: 22 -> 11 (-50%); none new | **fail**: 26 -> 8 (-69%); none new | **fail**: 26 -> 16 (-38%); none new |
| (contact seconds, reported) | - | 97.2 -> 15.2 s | 105.2 -> 21.5 s | 66.0 -> 23.2 s |
| Crashes | not more | **pass** (0 -> 0) | **pass** (0 -> 0) | **pass** (0 -> 0) |
| Finishes | not fewer | **pass** (28 -> 28) | **pass** (28 -> 28) | **pass** (28 -> 28) |
| Path below the image while descending | <= 5% | **fail**: 48.6% -> 36.5% | **fail**: 56.0% -> 53.3% | **fail**: 53.5% -> 45.8% |
| ... on hills no steeper than 12 deg | <= 5% | **fail**: 6.3% -> 5.2% | **fail**: 15.1% -> 20.0% | **fail**: 19.0% -> 28.7% |
| Passes > 1.5 m above a checkpoint | <= baseline + 1 | **fail**: 2 -> 13 | **fail**: 1 -> 24 | **fail**: 3 -> 17 |
| Course time (paired, terrain sets) | <= +3% | **pass**: 60.7 -> 55.2 s (-9.1%) | **fail**: 65.4 -> 67.7 s (+3.5%) | **pass**: 62.7 -> 63.1 s (+0.7%) |
| Flat courses | same finishes and crashes, time within +-2% | **fail**: 52.8 -> 50.5 s (-4.3%, faster) | **pass** (+0.1%) | **pass** (-0.5%) |

Per set (baseline -> rule):

| Set | Motor | Contacts | Deepest penetration | Speed into the ground | High passes | Mean time |
|---|---|---|---|---|---|---|
| steep | fast PD | 7 -> 2 | 6.71 -> 0.50 m | 3.64 -> 1.84 m/s | 1 -> 6 | 69.0 -> 64.0 s |
| hill | fast PD | 15 -> 9 | 4.84 -> 1.82 m | 1.81 -> 1.65 m/s | 1 -> 5 | 55.2 -> 49.3 s |
| steep | brain-08 | 8 -> 2 | 3.62 -> 0.47 m | 2.94 -> 1.42 m/s | 1 -> 8 | 71.6 -> 76.6 s |
| hill | brain-08 | 18 -> 6 | 5.42 -> 1.28 m | 1.55 -> 0.99 m/s | 0 -> 14 | 61.3 -> 61.8 s |
| steep | brain-09b | 10 -> 7 | 2.39 -> 1.11 m | 3.62 -> 2.37 m/s | 1 -> 4 | 69.0 -> 71.2 s |
| hill | brain-09b | 16 -> 9 | 2.61 -> 1.31 m | 3.08 -> 1.65 m/s | 2 -> 11 | 58.5 -> 57.8 s |

Stick chatter (mean roll/pitch change per tick) changed from 0.0058 to 0.0053 (PD), from
0.0029 to 0.0032 (brain-08) and from 0.0053 to 0.0053 (brain-09b).

**What the gates show.**
- **Contacts.** The rule makes contacts rarer, shallower and slower. It does not remove
  enough of them for the gate.
- **High passes.** It does not reach steep lower checkpoints from the right height. The high
  passes are the cost of descending shallower than the pilot used to. In Liftoff a ring
  passed 1.5-3 m above its centre is missed or its top frame is hit.
- **The brains** do not follow a 6 m/s descent. In the development traces fast-brain-08 was
  asked for (6, -1.8) m/s and flew about (3.0, -1.05) m/s. Asked for (3, -1.8) m/s by the
  default pilot, it tracked the request. The rule therefore leaves them high over lower rings
  without keeping the path in view: on the viewable hills brain-08 and brain-09b spend
  more time below the image than with the default pilot.
- **The strict view limit is physical.** In steady flight this camera cannot see a descent
  steeper than about 12-14 degrees. The steep and hill sets have many descending legs of
  15-35 degrees, so no rule can both keep those descents in view and reach their rings.
  Steep late gives up the view on those legs; a strict bound (development) overflew their
  rings.

## Open-loop replay of the Straw Bale downhill (development only)

```
python -m haltere.obstacles.descent_replay --declaration configs/pilot/descent_view.json --out DIR straw-brain08-04 ...
```

Output: `docs/experiments/descent_view_v1_straw_replay.json`. Every logged tick goes through
the default pilot and through the pilot with the rule (`vertical_replay.replay`, `--stack
none`). The recorded motion does not respond. Scored on the downhill region of the logs
(x -50 to -25, y 120 to 195, flying toward -y; offline only), at the recorded attitude.

| Flight | Downhill time | Request below the image: default -> rule | Requested horizontal speed | Requested sink over the downhill | 10th-percentile requested path |
|---|---|---|---|---|---|
| straw-brain08-04 (3 laps) | 55.1 s | 54.4% -> 7.5% | 4.06 -> 5.77 m/s | 45.2 -> 31.1 m | -24.5 -> -10.7 deg |
| straw-brain08-06 (3 laps) | 54.9 s | 52.6% -> 6.8% | 4.02 -> 5.69 m/s | 44.6 -> 30.2 m | -26.4 -> -10.5 deg |
| straw-brain08-01 (hillside slide)* | 11.0 s | 19.3% -> 6.4% | 4.66 -> 5.72 m/s | 6.9 -> 7.1 m | -18.9 -> -11.2 deg |
| straw-fast6-03 (fast PD)* | 54.6 s | 65.9% -> 39.2% | 3.22 -> 5.81 m/s | 61.8 -> 52.5 m | -59.6 -> -16.7 deg |

\* flown with an earlier pilot, so the replay of the default pilot does not reproduce the log
exactly (states 97.6% / 100% equal, horizontal request p99 differences 3.7 / 2.2 m/s). The
brain-08 runs 04 and 06 flew the current pilot and replay exactly (states 100%, request
difference 0).

In the 3 s before each of the 10 contacts of runs 04 and 06:
- the default pilot's request pointed below the image 58-83% of the time; the rule's 0-15%;
- the rule would have asked for 5.7-6.0 m/s instead of 3.0-4.0 m/s;
- it would have asked for 1.2-2.4 m of sink instead of 2.9-4.1 m.

Over the whole downhill of 04 it withholds 41 m of requested sink and raises the speed for
7.2 s. The plot shows the first downhill pass of straw-brain08-04: the lower image edge at the
recorded attitude, and the requested path, horizontal speed and vertical speed of both
pilots. Dotted lines mark the contact.

![Open-loop replay of the first Straw Bale downhill pass of straw-brain08-04](descent_view_straw04_replay.png)

This shows where the requests would point at the recorded states, nothing more. A drone
flying the rule would be elsewhere: faster, higher on the hill, with another attitude, and
the brain may not fly the request (see the gates).

## Default behaviour unchanged

With the rule off the pilot is bit-identical to m2-vertical (`935cfdb`):
- **Replay.** Open-loop replays were run through an export of `935cfdb` and through this
  tree, comparing every command array bitwise: 52 of 52 identical
  (`docs/experiments/descent_view_v1_identity.json`). They covered:
  - straw-brain08-04 and -06 (default pilot);
  - minus-fast6-vg-02 and minus-brain09b-vg-01 (stack as flown, wall rules and vertical guard
    on);
  - pine-fast6-ttc-01 and minus-brain08-vg-01 (full stack on);
  - two 40 s `rehearse` runs without terrain, whose result dicts are also identical.
- **Unit test.** `test_default_pilot_is_bit_identical_to_m2_vertical` pins a golden digest of
  a scripted descent, clip and climb sequence, taken from the `935cfdb` tree.
- **Logs.** With the flag off, the CSV columns and the sidecar keys are unchanged.

## Risks for live flight

Version 1 failed its gates. Flying it would be a disclosed development deviation, not a
qualified fix. If it is flown, fly the fast PD first. A default-pilot flight can also be
replayed through the rule offline (`haltere.obstacles.descent_replay`); no live shadow mode
is needed.

- **Brain motors do not fly the request.** fast-brain-08 (and less so brain-09b) keeps about
  3 m/s and sinks less than asked when the rule asks for 6 m/s descents. The gates show more
  high passes (1 -> 24) and more time below the image on gentle hills (15% -> 20%) than with
  the default pilot.
  - On Straw Bale this means rings reached from above: the top frame, or a missed ring and a
    turn back.
  - The expected fix is a brain distilled under this pilot (the fast-brain-08 recipe with the
    rule in the DAgger rollouts), scored on these frozen gates, not another pilot value.
- **Straw Bale downhill.**
  - In the replay the requests point below the image 0-15% of the 3 s before each logged
    contact (default pilot 58-83%): mostly in view, not always.
  - The drone now enters the downhill at 6 m/s instead of braking to 3. Contacts that remain
    come at higher ground speed.
  - The slope-support rule needs a requested sink below -0.8 m/s, which the bound allows only
    above about 5 m/s. A slow drone resting on the hillside is recognised only by the
    ordinary support rule, and the descent-path governor's floor keeps it moving at 0.75x.
- **Straw Bale uphill and hilltop.** Climbs are never changed.
  - At the hilltop no bottom-clip brake happens any more. The side-clamped turn onto the
    downhill and the gentle search are unchanged.
  - Steep late starts after 0.75 s of unbroken bottom clip. A ring hidden below the crest
    would get the steepening late, over the convex crest, which is the part a looming sample
    cannot see either.
- **Minus Two garage floor** (about 1-2 m high).
  - When the looming governor or turn-first brakes, the nose-up attitude shrinks the allowed
    sink toward 0.3 m/s. This is gentler on the floor than the default pilot, which asks for
    edge + 5..20 degree descents at half speed.
  - The rise toward the speed schedule acts before the looming wall cap and turn-first, which
    still bound it; it was not replayed on Minus logs.
  - Steep late relative to a nose-up edge stays small.
- **Pine Valley mound.** Terrain climbs (looming governor, vertical guard) come after the
  bound and are unchanged. Descents after the mound stay near the view, where the vertical
  guard's lower looming window can see the ground. The boulder descent (-0.7 m/s at 6 m/s)
  is under the bound.
- **Surrogate limits.**
  - The hills are synthetic and scoring-only: no contact physics, no Liftoff hill shapes.
  - The HUD clamp rule differs from Liftoff (rings behind are clamped to the nearest edge,
    not to the top corners).
  - The brains' behaviour in the surrogate is itself a model.

## Tests

- `tests/test_fast_race_cue_descent_view.py`:
  - the declaration and gates are frozen; edits and other versions are refused;
  - the bound's geometry for four nose pitches, and its speed scaling and free sink;
  - keep speed for a bottom clip, with the path 3 degrees inside a -12 degree edge;
  - steep late: the rate and the bound;
  - the descent-path governor's floor;
  - the gentle sink onset;
  - climbs are unchanged, and the rule is off by default;
  - a golden digest of the default pilot equals the m2-vertical tree's;
  - the runner's flag, columns and refusals.
- `tests/test_descent_rehearsal.py`: the terrain model, the contact, view and high-pass
  scoring, and the course sets.
- `tests/test_fast_race_cue_contact_support.py` (round 4b):
  - version 2 and its gates are frozen, version 1 is kept and refused by the runner, and the declared
    thrust curve is the measured profile's;
  - a drone resting on a surface with a small, view-bounded sink request gets a support climb within
    0.8 s at 0, 3 and 6 m/s; the same thrust in free air gets none;
  - no onset before arming; the thrust gain learns a 15% stronger drone without an onset;
  - off by default, and the default and version-1 pilots keep m4's golden digests;
  - the runner's contact columns and metadata;
  - the clearance brake's sink floor (wall pilot version 5).
- `tests/test_contact_audit.py`: the audit on synthetic telemetry and its validation scoring.
- `tests/test_round5_safety.py` (round 5, 18 tests):
  - wall pilot 6, descent view 3 and the safety gates are frozen; versions 5 and 2 are kept and refused
    by the runner; contact support versions 2 and 3 are validated field by field;
  - the default, view-rule and version-2 pilots keep `m4b`'s golden digests (a git archive of
    `2a5bccb`);
  - the ceiling cut of a climb that is not the governor's, none for a sample the lower window explains,
    and the pilot's levelling for support, contact and search climbs, but not for its own ring climb;
  - version 3 arms after its gain learning (3.8-4.0 s) and detects a rest within 0.8 s; a rest that
    starts during the learning is not learnt;
  - a hard pitch (3 and 5 rad/s) or a 6 m/s² horizontal brake starts no support climb (version 2 fires),
    while a rest at 1 rad/s is still detected;
  - a drone whose motion implies a gain of 1.16 fires with version 2 after arming and not with version 3;
  - shadow commands equal off bit for bit and mark the tick where on fires;
  - a contact-support climb never ends turn-first; an older support climb and version 2 still hand off;
  - the runner switch, columns and metadata, the deployed pilot and the replay harness modes.
