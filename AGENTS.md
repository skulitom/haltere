# Project direction and working preferences

- The intended product is the connectome-constrained fly brain flying the drone.
- The navigation predictor is **training-only**: use it as a teacher, auxiliary
  supervision, or an offline diagnostic. It is not an intended final product and
  must not be a dependency of the deployed controller. Do not promote its path
  prediction accuracy as evidence that autonomous flight improved.
- Training on human recordings must actually update the fly brain. Identify which
  weights changed and evaluate the exported brain with the navigation teacher
  absent. Preserve the connectome's wiring and known neurotransmitter signs.
- Run Liftoff flight tests and their capture/controller processes inside Anode.
  Keep its viewer hidden unless the user asks to watch. The user has authorized
  moving the Steam session there; retain the original `[Copy] New Drone`.
- Keep Liftoff open between tests. Verify throttle-low and real processed control
  response; Xbox neutral throttle is not zero. Pause before disconnecting a pad.
- Push completed, verified changes to GitHub. Publish new weights to GitHub and
  Hugging Face when ready, with accurate evaluation limits and model provenance.
- Prefer fixing visible flight problems and useful visual demonstrations over
  producing reports. Preserve raw recordings and unrelated working changes.

These directions were explicitly confirmed by the user on 2026-09-21.
