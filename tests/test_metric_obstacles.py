import numpy as np
import pytest

from haltere.vision.camera import Camera
from haltere.vision.geometry_mask import liftoff_geometry_mask
from haltere.vision.metric_obstacles import metric_obstacle_points


def test_dense_white_wall_retains_scene_samples_but_excludes_hud_and_colored_cues():
    image = np.full((360, 640, 3), 240, np.uint8)
    image[190:215, 300:330] = [0, 200, 120]
    assert not liftoff_geometry_mask(image).any()
    mask = liftoff_geometry_mask(image, exclude_white=False)
    assert mask[140, 320] and not mask[40, 320] and not mask[200, 310]
    camera = Camera(640, 360, 200, 30)
    result = metric_obstacle_points(np.full((360, 640), 2.), image, camera, [1, 2, 3], [1, 0, 0, 0])
    assert len(result['points']) > 100
    camera_points = (result['points']-[1, 2, 3])@camera.body_to_cam().T
    np.testing.assert_allclose(camera_points[:, 2], 2.)
    pixels = result['pixels']
    assert (mask[pixels[:, 1], pixels[:, 0]] > 0).all()
    assert not result['establishes_free_space']


def test_dense_invalid_values_do_not_silently_become_empty_space():
    camera = Camera(640, 360, 200, 30)
    image = np.full((360, 640, 3), 100, np.uint8)
    with pytest.raises(ValueError, match='finite'):
        metric_obstacle_points(np.full((360, 640), np.nan), image, camera, [0, 0, 0], [1, 0, 0, 0])
    result = metric_obstacle_points(np.full((360, 640), 100.), image, camera, [0, 0, 0], [1, 0, 0, 0])
    assert result['points'].shape == (0, 3)
    assert not result['establishes_free_space']
