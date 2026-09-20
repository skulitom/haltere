# Human demonstrations train the fly brain

The navigation predictor supplies an auxiliary training target. It is absent
from the exported controller. Images are masked and sampled on a fixed 20 by 12
RGB grid, then passed through a sensory encoder into the existing LPTC population.
The connectome's recurrent circuit produces all four motor outputs. Its wiring
and known transmitter signs remain fixed; recurrent gains, neuron parameters,
sensory encoders and motor readout receive gradient updates.

Training combines observed human controls with a small path loss. A linear
training-only decoder reads the brain's goal-population activity and learns to
match the frozen predictor's paths. This decoder and the predictor are discarded
at export. No predicted path is fed into the student's senses. The external goal
and global compass channels are zero, and no external yaw controller is used.
This is a first, coarse visual sensory encoding, not a biological retinal model.

## Reproduce

Run game tests, capture, controller and the following associated jobs in Anode.
Use fresh output directories; preparation verifies hashes of the source takes.

```powershell
.venv/Scripts/python.exe -m haltere.train.human_brain prepare --dataset data/vision/human_demonstrations_v3 --teacher artifacts/experimental/navigation_human_v2_residual.pt --brain artifacts/ftPath2_best.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --out data/vision/human_brain_v1b
.venv/Scripts/python.exe -m haltere.train.human_brain train --prepared data/vision/human_brain_v1b --initial artifacts/ftPath2_best.pt --config configs/train_human_brain.json --out runs/human-brain-01
```

The mapping is the measured original-drone mapping, not a generic Xbox profile.
Supervision uses processed Liftoff inputs in throttle/roll/pitch/yaw order, after
converting the brain's throttle and axis conventions. It never treats these as
raw gamepad commands. Sources have variable telemetry cadence; training advances
the brain at 100 Hz using causal sample-and-hold, with a maximum 120 ms data gap.
Images become available only after their recorded capture end. Physical display
delay remains uncalibrated. Neuron normalization statistics are kept fixed.

Train: Straw Bale race, Pine Valley take 2, fence take 1. Validation: Minus Two
take 2 and fence take 2. Whole-take membership is preserved. Neither validation
take contributes optimizer updates. Fence take 3 and the mistake recordings are
not read by this workflow. Fixed validation windows are chosen before training;
initial, constant-training-mean and blank-image comparisons are retained.

`parameter_changes.json` records actual brain changes, `first_gradients.json`
checks that learning reaches recurrent parameters, and `result.json` evaluates
the exported brain without loading the predictor. Checkpoint selection uses
validation action error. These short, teacher-forced replay windows do not
establish closed-loop stability, gate completion, or generalization.

## Teacher-free live runner

```powershell
.venv/Scripts/python.exe -m haltere.liftoff.visual_brain runs/human-brain-01/best.pt --mapping runs/pine-route-collection-01/liftoff-original-drone.yaml --seconds 15 --log runs/human-brain-01/shadow-01.csv
```

The default is **shadow mode**, with no controller opened. To fly a reviewed
candidate, explicitly add `--udp-out 127.0.0.1:9003` for an existing throttle-low
bridge. The runner requires fresh game images and telemetry, bounds each attempt,
and stops on a reset or stale input. It does not load GateNet, the navigation
predictor, a course route or a yaw assistant. Logs identify the exact checkpoint
and whether any controls were sent. Videos of shadow runs must be labelled as
shadow runs. Live qualification remains separate from offline imitation.
