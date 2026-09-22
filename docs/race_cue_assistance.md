# Visible race-cue experiment

`--pilot-assistance race-cue` is a separate experimental mode for the newest
visual brain. It reads the game's visible next-checkpoint ring, supplies a
bounded local goal and heading assistance, and brakes while searching for a
missing or off-screen target. The connectome retains throttle, roll and pitch,
its learned scene features, wiring and neurotransmitter signs.

This mode uses game-provided race guidance. It does not measure unaided arch
detection and is not a freestyle planner. The ordinary `rabbit` and `none`
modes retain their previous inputs. No saved route, checkpoint coordinates,
future frames or course-specific tuning are supplied to the cue mode.

The 2026-09-22 development candidate uses the unchanged scene brain
`runs/scene-brain-09-navigation/last.pt` (SHA256
`3789e33b8bd5bbc0fa504fd9c3365ab5a5e897451d14bc5ce3c042c091543a6e`)
and its exact frozen detector/scene-feature contract. Cue recognition runs
beside those features; it does not replace the detector inside that contract.
No new brain weights or navigation predictor are introduced by this change.

The cue has an image bearing, not a known metric distance. Side-clamped cues
are turn hints only. Missing or ambiguous detections request braking and
search rather than a forward search circle. The pilot estimates no checkpoint
count: full ordered laps and finish must be verified from the game. A bottom-edge
cue near the horizontal centre requests bounded descent with a shorter forward
goal; treating it only as a turn hint caused a downhill recovery stall.

The first longer attempt hit a racing flag after 164.6 seconds. A checkpoint
marker can lie on a solid object or behind an intervening object: centring it
does not establish free space. The next candidate adds local clearance for the
visible blue-tipped racing flag asset. Apparent flag size sets the image margin;
nearby green route arrows choose the passing side. This is a limited visual
heuristic, not a general obstacle detector. It uses no saved obstacle positions.
The cloth below the cap determines its apparent slant in the image, including
opposite views. The subsequent development flight passed the previous flag
without impact, but was manually stopped during the downhill recovery stall.

Runtime metadata declares `visible_race_cues` and `pilot_assistance.mode`.
CSV `pilot_kind` is 0 for none, 1 for Rabbit and 2 for race-cue; `cue_u`, `cue_v`
are normalized image coordinates and `cue_edge` marks an off-screen cue.
`cue_aim_u` records the horizontal bearing after local clearance; `cue_u`
continues to record the original ring. Missing values are -1. All recordings retain standard live brain/gameplay
panels, real telemetry, unmodified neural actions and assisted commands.

Run in Anode with the original drone and a separately verified throttle-low
pad bridge, using the same setup as [visual assistance](visual_pilot_assistance.md).
Use fresh paths for every attempt and preserve all failures. The fourth
Straw Bale development attempt [completed all three laps](flight_cards/2026-09-22_strawbale_cue_04.md)
in 14:05.703 without detected contact or intervention before finish. This
establishes one seen-course finish; unseen race and freestyle completion remain
unproven. Earlier partial attempts and their failures are retained in the
[flight index](flight_cards/README.md#2026-09-22-visible-race-cue-development).
