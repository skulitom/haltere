import numpy as np

from haltere.vision.camera import Camera
from haltere.vision.depth_completion import calibrate_relative_depth, supported_metric_points


def plane():
    yy, xx = np.mgrid[:96, :128]
    relative = (1+xx/128+yy/192).astype(np.float32)
    pixels = np.array([[x, y] for y in [12, 32, 60, 80] for x in [12, 32, 60, 90, 110]])
    d = relative[pixels[:, 1], pixels[:, 0]]
    anchors = np.column_stack([pixels, 1/(.15*d+.2), np.full(len(pixels), .015)])
    return relative, anchors


def test_supported_depth_fills_local_gaps_without_extrapolating_or_claiming_free_space():
    relative, anchors = plane()
    fit = calibrate_relative_depth(relative, anchors)
    assert fit is not None
    np.testing.assert_allclose(fit['coefficients'], [.15, .2], atol=1e-7)
    camera = Camera(128, 96, 80, 0)
    mask = np.ones(relative.shape, np.uint8); mask[:20] = 0
    points = supported_metric_points(relative, fit, mask, camera, [0, 0, 0], [1, 0, 0, 0])
    assert len(points['points']) > len(anchors)
    assert np.all(points['pixels'][:, 1] >= 20)
    assert not points['establishes_free_space']
    # Optical forward distance is the first FLU component with zero camera tilt.
    px = points['pixels'].astype(int)
    np.testing.assert_allclose(points['points'][:, 0], 1/(.15*relative[px[:, 1], px[:, 0]]+.2), rtol=1e-6)
    near = supported_metric_points(relative, fit, mask, camera, [0, 0, 0], [1, 0, 0, 0], max_support_pixels=2.)
    assert len(near['points']) < len(points['points'])
    hole = relative.copy(); hole[24:64, 40:80] = .01
    result = supported_metric_points(hole, fit, mask, camera, [0, 0, 0], [1, 0, 0, 0])
    xy = result['pixels']
    assert not np.any((xy[:, 0] >= 40) & (xy[:, 0] < 80) & (xy[:, 1] >= 24) & (xy[:, 1] < 64))


def test_uncertain_insufficient_or_inconsistent_anchors_supply_no_metric_depth():
    relative, anchors = plane()
    assert calibrate_relative_depth(relative, anchors[:7]) is None
    uncertain = anchors.copy(); uncertain[:, 3] = .11*uncertain[:, 2]
    assert calibrate_relative_depth(relative, uncertain) is None
    # Keep a perfect fitting set while contradicting the independent check set.
    inconsistent = anchors.copy(); inconsistent[::4, 2] *= .5
    assert calibrate_relative_depth(relative, inconsistent) is None
    result = supported_metric_points(relative, None, None, None, None, None)
    assert result['points'].shape == (0, 3)
