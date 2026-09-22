# Straw Bale / Field Day: full three-lap finish

Seen development course. Prediction and exact runtime source hashes saved
before flight in `runs/race-completion-20260922/straw-cue-04-manifest.json`.
Same scene-09 brain, detector, mapping, original drone and nominal 2 m/s as
the preceding attempts. Hidden Anode 0.6.0, session 2; Liftoff 1.7.6.
Added bounded descent for a bottom-edge cue and image-derived flag slant.
No stored route, predictor, new neural weights or human flight commands.

Prediction: retain flag clearance and recover the downhill checkpoint; full
race unproven. Criterion: game-confirmed three-lap finish within 900 seconds,
without contact, reset or intervention.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/scene-brain-09-navigation/last.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --device cpu --pilot-assistance race-cue --assist-speed 2 --capture-backend dxgi --seconds 900 --max-height 40 --max-speed 10 --max-distance 400 --udp-out 127.0.0.1:9003 --pause-on-stop --log runs/race-completion-20260922/straw-cue-04.csv --record runs/race-completion-20260922/straw-cue-04.mp4
```

## Result

**Passed: all three laps completed.** Game finish screen showed the player as
finished and race time **14:05.703**. Displayed lap times: **04:39.529**,
**04:40.329**, **04:38.977**. No intervention before finish, no reset, no
detected impact or estimated collision, no camera outage or control deadline
failure. Median horizontal speed 1.82 m/s; p90 2.25 m/s.

The recorder preserved the finish screen. The runner then stopped on telemetry
leaving live flight; Escape was sent only after finish had been observed.
This terminal stop is not a racing failure. Finish evidence is recorded in
`straw-cue-04-finish.json` beside the raw recording.

Full standard brain/gameplay MP4, CSV, runtime JSON, preflight manifest and
score are preserved in `runs/race-completion-20260922/straw-cue-04*`.
Video fully decoded without errors. Automated suite: 309 tests passed.

This is one successful **seen development** race after three preceding partial
attempts. It does not establish unseen-course or freestyle completion. The
brain retains throttle, roll and pitch; the explicitly disclosed race-cue
pilot supplies goals, velocity-sense scaling and yaw.
