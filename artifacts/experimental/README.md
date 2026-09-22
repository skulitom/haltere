# Experimental navigation checkpoints

These are offline future-path predictors trained from the human recordings on
20 September 2026. They do not emit sticks and are not loaded by the current live runners.
Runtime use is allowed under the [generalization-first project direction](../../docs/project_direction.md);
these checkpoints need complete-system flight evaluation before promotion.
See [training, limitations and measured results](../../docs/navigation_training.md).

`navigation_human_v1_motion_only.pt` is the stronger overall validation result.
`navigation_human_v1_vision_motion.pt` helped on the fence test but generalized
poorly to Minus Two; it is retained as an experimental comparison.

Load with `NavigationNet` from `haltere.vision.navigation`. The checkpoint stores
`vision`, `hidden`, `horizons`, `model`, the training configuration, selected
epoch and dataset hash. Inputs are causal RGB sequences plus body velocity,
body-to-world wxyz attitude, and actual elapsed seconds. Outputs are three
future displacement vectors in the current body's forward/left/up frame.

`navigation_human_v2_residual.pt` includes the frozen motion base and a bounded
visual corrector. Load it with `load_navigation`, which handles both checkpoint
architectures. See the [v0.2.0 release notes](../../docs/navigation_release_v02.md)
for results, videos and the limits of the comparison.
