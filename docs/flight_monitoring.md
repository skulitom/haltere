# Automatic flight monitoring

The read-only monitor follows the flight CSV incrementally, without looking at
screenshots or opening the Anode viewer. It reports elapsed time, position,
speed, controller status, telemetry gaps, resets and a declared movement stall.
It saves a live `status.json` and emits events when the status changes. Run it
inside Anode alongside the flight; it does not own a telemetry port or pad.

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.flight_monitor runs/ATTEMPT/flight.csv --watch --seconds 600 --out runs/ATTEMPT/monitor
```

The output directory must be new. Omit `--watch` for a single inspection of a
saved log. Defaults require 60 continuous seconds within 1 m of the window's
first position; missing telemetry or a reset restarts that window. A stale log
is reported separately from lack of motion. Controller shutdown metadata is
included when available. This is a development stall rule, not a universal
failure criterion for freestyle tasks that intentionally hold position.

**This version observes only.** It does not pause, restart, select a race or
certify a finish. `finish_confirmed` remains unknown until separate game result
evidence is available. Automatic pausing and reliable result-screen recognition
are the next integration work; do not infer a finish from a stopped controller.

Validation on 2026-09-23: four focused tests pass, including partial CSV writes,
reset/gap handling and stale telemetry. Replay of five development recordings
(119,688 rows) had no malformed rows. It flagged the known dense-PD ground stall
at 269.683 s and flagged none of the other four attempts, including both prior
finishes. All five replay checks together took under three seconds locally.
These are retrospective checks, not a measured live-flight overhead result.

A local Windows OCR prototype is retained under
`runs/monitoring-prototype-20260923/`. It recognizes result headings but did not
reliably extract tiny clock digits, and has an image-file lifetime race to fix.
It is deliberately outside the runtime and is not a validated finish detector.
