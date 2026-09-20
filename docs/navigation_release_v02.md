# Navigation v0.2.0 — experimental offline path prediction

This release adds a small visual correction to the frozen motion predictor.
It forecasts the human's future path at 0.25, 0.5 and 1 second; it does not
emit sticks, operate the game, or replace the connectome motor controller.

The 30-epoch experiment selected epoch 3 using validation only. The underlying
motion model stayed bitwise unchanged. A checkpoint was eligible only when
neither validation flight's one-second error regressed by more than 2% against
that base. Later checkpoints failed this check. The selected one improved both.

Mean one-second endpoint error in metres:

| Model | Minus Two validation | Fence take 2 validation | Fence take 3 regression |
|---|---:|---:|---:|
| Constant velocity | 2.022 | 1.165 | 2.018 |
| Frozen motion v1 | 0.930 | 0.744 | 1.176 |
| Joint vision v1 | 1.923 | **0.705** | **1.048** |
| Bounded visual correction v2 | **0.927** | 0.720 | 1.134 |

V2 improves on its motion base by 0.3%, 3.2% and 3.6%, respectively. Its main
benefit relative to joint vision v1 is preserving the strong Minus Two result.
V1's joint vision model remains better on the fence, so v2 is not a universal
replacement. These are single-run, small-dataset results, not statistical proof
of generalization. Fence take 3 was already inspected during v1; its reuse is a
regression check, not a new untouched test.

## Download and use

Published on the project's [GitHub release](https://github.com/skulitom/haltere/releases/tag/navigation-v0.2.0)
and [Hugging Face](https://huggingface.co/Skulitom/haltere/tree/main/navigation/v0.2.0).
The package includes v2, its frozen motion base, and the prior joint vision
model for comparison. `SHA256SUMS` covers the weights, code, metrics and videos.

With PyTorch and `huggingface_hub` installed, download the versioned package:

```python
from huggingface_hub import snapshot_download
folder = snapshot_download('Skulitom/haltere', revision='navigation-v0.2.0',
                           allow_patterns=['navigation/v0.2.0/*'])
```

The package's `navigation.py` is a standalone PyTorch module. The same code
also lives in this repository as `haltere.vision.navigation`:

```python
from haltere.vision.navigation import load_navigation
model, metadata = load_navigation('artifacts/experimental/navigation_human_v2_residual.pt')
# prediction = model(images, velocity_body, attitude_wxyz, time_s)
# images:        [B, 16, 3, 90, 160], RGB floats in [0, 1]
# velocity_body: [B, 16, 3], metres/second, forward/left/up
# attitude_wxyz: [B, 16, 4], body-to-world unit quaternion
# time_s:        [B, 16], actual elapsed seconds
# prediction:    [B, 16, 3 horizons, 3 coordinates], body-frame metres
```

The base is included inside the residual checkpoint. It needs no graph assets
or separate base checkpoint for inference. Camera display delay is uncalibrated.
The model masks the stick display, compass, timer and standings. In-scene racing
guidance remains in the footage. Corrections are bounded by 0.6 metres at one
second, scaled by horizon squared; that bounds prediction changes, not flight risk.
All-black images fall back exactly to the frozen motion output.

## Videos

The release includes `fence_forecasts_v2.mp4` and `minus_two_forecasts_v2.mp4`.
They show the prepared flights at 2x speed with human future positions, model
forecasts and constant-velocity forecasts. Every frame is labelled **offline
comparison — recorded human flight**. These are not new autonomous flights.
The videos use a rolling 16-frame context; benchmark metrics also include
shorter contexts after warmup, so per-video errors are illustrative rather than
an exact replay of the aggregate benchmark.

## Reproduce

```powershell
.venv/Scripts/python.exe -u -m haltere.vision.train_residual_navigation data/vision/human_demonstrations_v3 --out runs/navigation-human-02
.venv/Scripts/python.exe -m haltere.vision.navigation_video data/vision/human_demonstrations_v3 artifacts/experimental/navigation_human_v2_residual.pt --take straw_bale_fence_03 --out runs/navigation-human-02/fence_forecasts_v2.mp4
```

The raw recordings remain local. The pinned data plan is
`configs/human_demonstrations_v3.json`, the training configuration is
`configs/train_residual_navigation.json`, and full numerical results are in
`docs/experiments/navigation-human-02.json`.
