# Project direction and working preferences

- The product goal is a fly-brain-based drone system that can complete unfamiliar
  races and freestyle tasks. Generalization and successful flight take priority
  over requiring every navigation or steering decision to come from the brain.
- Visual pilot assistance and learned navigation predictors are allowed at runtime
  when they help that goal. The navigation predictor is not restricted to training.
  Decide whether to integrate or promote it using complete-system flight evidence,
  including held-out courses/tasks and comparisons with the component disabled.
  Offline path accuracy alone is not evidence of improved autonomous flight.
- Keep the connectome as a meaningful motor controller and preserve its wiring
  and known neurotransmitter signs. When claiming brain learning from recordings,
  identify the brain weights that actually changed; predictor-only training does
  not train the brain. Evaluate the intended deployed stack and use brain-only,
  no-predictor and no-assistance runs as diagnostics, not mandatory product modes.
- Generalization runs must use causal runtime observations and a declared task,
  without hand-entered course geometry, route lookups or per-course tuning. Generic
  visible race cues may be used if disclosed; evaluate freestyle separately where
  race cues are absent. Keep oracle-guided collection and offline labels distinct
  from autonomous flight. See docs/project_direction.md for acceptance criteria.
- Run Liftoff flight tests and their capture/controller processes inside Anode.
  Keep its viewer hidden unless the user asks to watch. The user has authorized
  moving the Steam session there; retain the original `[Copy] New Drone`.
- Keep Liftoff open between tests. Verify throttle-low and real processed control
  response; Xbox neutral throttle is not zero. Pause before disconnecting a pad.
- Push completed, verified changes to GitHub. Publish new weights to GitHub and
  Hugging Face when ready, with accurate evaluation limits and model provenance.
- Prefer fixing visible flight problems and useful visual demonstrations over
  producing reports. Preserve raw recordings and unrelated working changes.

Flight-operation preferences were confirmed on 2026-09-21. The generalization-first
goal and permission for runtime pilot/predictor assistance were clarified by the
user on 2026-09-22 and supersede the earlier training-only predictor restriction.
