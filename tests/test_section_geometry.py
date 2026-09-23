import numpy as np
import pytest

from haltere.liftoff.section_geometry import camera_collision_depth, collision_depth, ray_box
from haltere.vision.camera import Camera


def test_box_front_parallel_miss_inside_and_rotation():
    rays = np.array([[0, 0, 1], [1, 0, 0]], dtype=float)
    np.testing.assert_equal(ray_box([0, 0, 0], rays, [0, 0, 10], [2, 2, 2]), [9, np.inf])
    np.testing.assert_equal(ray_box([0, 0, 10], rays, [0, 0, 10], [2, 2, 2]), [0, 0])
    assert ray_box([0, 0, 0], rays[:1], [0, 0, 10], [8, 2, 2], 90)[0] == pytest.approx(6.)


def test_occlusion_ground_and_missing_mesh_are_not_free_space_labels():
    scene = dict(runtime_geometry_allowed=False, unknown_geometry=[], ground_plane_y=0.,
                 primitives=[dict(instance_id=4, kind='box', center=[0, 3, 10], size=[10, 5, 1], yaw_deg=0.)])
    result = collision_depth(scene, [0, 3, 0], [[0, 0, 1], [0, -1, 0], [0, 1, 0]])
    np.testing.assert_equal(result['range_m'], [9.5, 3., np.inf])
    np.testing.assert_equal(result['instance_id'], [4, 0, -1])
    assert not result['rendered_depth_validated']
    with pytest.raises(ValueError, match='Unmodelled'):
        collision_depth(dict(scene, unknown_geometry=[{'item_id':'LedFlag01'}]), [0, 3, 0], [[0,0,1]])


def test_metric_ranges_require_normalized_rays():
    with pytest.raises(ValueError, match='unit rays'):
        ray_box([0, 0, 0], [[0, 0, 2]], [0, 0, 10], [2, 2, 2])


def test_camera_depth_distinguishes_optical_depth_and_range_with_body_offset():
    scene = dict(runtime_geometry_allowed=False, unknown_geometry=[],
                 primitives=[dict(instance_id=4, kind='box', center=[0, 3, 10], size=[30, 30, 1], yaw_deg=0.)])
    camera = Camera(3, 3, 2., 0.)
    result = camera_collision_depth(scene, camera, [0, 0, 3], [1, 0, 0, 0],
                                    camera_offset_body=[1, 0, 0])
    np.testing.assert_allclose(result['optical_depth_m'], 8.5)
    assert result['range_m'][0, 0] > result['range_m'][1, 1]
    assert result['range_m'][1, 1] == pytest.approx(8.5)
    np.testing.assert_allclose(result['camera_origin_unity'], [0, 3, 1])


def test_camera_tilt_and_world_yaw_are_applied_before_raycasting():
    # Simulator yaw +90 degrees points toward Unity -x. A 30-degree camera
    # uptilt intersects the ground only through lower pixels, not the center.
    scene = dict(runtime_geometry_allowed=False, unknown_geometry=[], ground_plane_y=0.,
                 primitives=[dict(instance_id=4, kind='box', center=[-10, 3, 0], size=[1, 30, 30], yaw_deg=0.)])
    q = [2**-.5, 0, 0, 2**-.5]
    result = camera_collision_depth(scene, Camera(1, 1, 1., 30.), [0, 0, 3], q)
    assert result['range_m'][0, 0] == pytest.approx(9.5/np.cos(np.pi/6))
    assert result['instance_id'][0, 0] == 4
    with pytest.raises(ValueError, match='unit wxyz'):
        camera_collision_depth(scene, Camera(), [0, 0, 3], [2, 0, 0, 0])
