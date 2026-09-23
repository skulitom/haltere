# Experimental dense obstacle input

The sparse geometry trials can descend below their last observed wall points,
then move forward through an apparent gap. The latest frozen batch finished
2/3 on the known box course with PD motors; it did not establish reliability.
See the [complete recordings and flight record](flight_cards/2026-09-23_geometry_control.md).

`visual_brain --dense-obstacles PATH` adds an explicitly selected pretrained
metric-depth export to `--geometry-control` or `--geometry-shadow`. It is off
by default. It uses the vision device and loads before the timed controller
starts. There is no runtime download, course lookup, fitted per-course scale,
or replacement of the motor controller.

The initial export is Depth Anything V2 Metric Indoor Small from Hugging Face,
revision `8078d68a9c75a972131914f6afd0c1723be0da7f`. Its indoor metric scale is
**unqualified for Liftoff**. Offline development images show substantial distance
errors. The model can produce false or misplaced obstacles; its output never
certifies empty space. It supplies transient points beside the persistent
camera/telemetry triangulations, without deleting those direct observations.
The point layer lasts one second, contains at most 384 occupied voxels, and
does not interpolate triangles. Its uncertainty is a heuristic maximum of
0.25 m and 25% of range, not a statistical error bound. HUD, propeller and
colored task-cue masks remain; bright scene pixels are retained for white walls.

The TorchScript export keeps the original preprocessing and float32 computation
with TF32 disabled. Forty frozen inputs matched the source model within
0.000021 m. Early export checks exposed a torchvision preprocessing mismatch
and TF32 numerical differences; those failed checks are retained. The local
model and its hash/provenance sidecar are under
`runs/depth-completion-probe-20260923/metric-indoor-336.ts` and `.json`.
This is a repackaged pretrained perception model, not trained brain weights.

On saved actual motion, the dense input reports an obstacle at the final failed
forward path where the sparse margin was positive. This justifies a bounded
comparison, not a counterfactual finish. Full-system flights must compare sparse
PD, dense PD and dense brain under one frozen configuration before this input
is considered useful. Broader layouts and unseen tasks are required for promotion.

A separate relative-depth completion diagnostic uses strict triangulated anchors
to fit metric inverse depth. It produced supported estimates in 5/40 sampled
frames but still left the collision path open. That diagnostic is not loaded by
the flight runner. Looser sparse anchors gave a large extrapolation error despite
passing a same-image consistency split, so they are not accepted by the helper.
All calibration, masks, labels and replay outputs remain in the local probe folder;
collider geometry was used only for subsequent offline scoring.
