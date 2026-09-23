# Original-drone dynamics measurements

The current throttle curve was fitted near hover. Its extrapolated full-throttle
response and the high-rate turning response have not yet been measured. These
measurements precede wider flight-cost training and faster live operation.

`haltere.liftoff.visual_brain --dynamics-calibration` reuses the visual runner's
workload check, telemetry/camera freshness checks, impact detection, recording,
deadline checks and pause-on-stop. It requires explicit PD motors, UDP control
and an operator-verified empty arena. Visual race guidance, oracle routes and
geometry control are incompatible with this mode. Keep the original calibrated
`[Copy] New Drone`; select The Drawing Board, free flight, no course, in Anode.

Start with `--dynamics-calibration hover`. The PD baseline climbs to 15 m above
launch and must remain within 0.6 m, below 0.35 m/s and below 0.25 rad/s for two
continuous seconds. Only then does a pulse sequence begin. `throttle` requests
processed inputs 0.25, 0.5, 0.75 and 1.0, three times each, for 0.25/0.15 seconds.
`roll`, `pitch` and `yaw` request both signs at those amplitudes, three times each,
with shorter durations at larger inputs. Each angular pulse ends after its time
limit or 35 degrees of measured attitude change. Every pulse requires a new
stable recovery before the next; missing recovery stops the experiment.

The calibration-specific limits are 35 m above launch, 15 m horizontal radius,
15 m/s, 55 degrees of tilt and 1,800 degrees/s. Falling below 3 m after departure
also stops it. Existing stricter runner limits remain effective. Use an adequate
declared duration and explicit height/speed limits when preparing a flight card.
Pause before releasing the independently held pad bridge.

The standard video labels PD plus pulses and the shadow brain. CSV records
requested/processed/raw commands, the game's actual processed inputs, pose,
velocity, derived body rates and all four measured rotor RPMs. Sidecar pulse
events identify timing and termination. Radial stick coupling can prevent a
requested angular endpoint from reaching exactly 1.0; measure actual inputs.
Short-pulse peak rates are not steady-state rate measurements.

Calibration is excluded from autonomous race/freestyle acceptance. This mode
does not change brain weights or drone settings. Preserve raw attempts, fit on
declared windows and validate on separate pulses before changing training
dynamics. Hover qualification and wider measurements remain live checks, not
capabilities implied by the automated tests.
