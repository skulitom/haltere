"""Looming v2 time-to-contact and the camera-process motion lookup; offline only."""
import numpy as np
import pytest

from haltere.liftoff.camera_process import pose_at
from haltere.vision.camera import Camera
from haltere.vision.looming2 import Looming2Config, LoomingEstimator2

cv2 = pytest.importorskip('cv2')
SENSOR = dict(focal_320=100., tilt_deg=30.)


def level_quaternion(yaw=0.):
    return np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])


def render_wall(distance, texture, config):
    """Grey image of a fronto-parallel textured wall `distance` ahead of a level drone."""
    camera = Camera(config.width, config.height, config.focal_320*config.width/320, config.tilt_deg)
    ys, xs = np.mgrid[0:config.height, 0:config.width].astype(np.float64)
    rays = camera.unproject_body(np.stack((xs.ravel(), ys.ravel()), 1))
    forward = np.clip(rays[:, 0], 1e-3, None)
    scale = distance/forward
    y, z = rays[:, 1]*scale, rays[:, 2]*scale
    h, w = texture.shape
    u = np.mod((y*40.).astype(int), w)
    v = np.mod((z*40.).astype(int), h)
    grey = texture[v, u].reshape(config.height, config.width)
    return np.repeat(grey[..., None], 3, -1).astype(np.uint8)


def test_approaching_textured_wall_gives_time_to_contact():
    # The deployed resolution; the 30 degree uptilt puts the focus of expansion low in the image.
    config = Looming2Config(width=240, height=135, median=1, sig_k=0.)
    estimator = LoomingEstimator2(config)
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur((rng.random((512, 512))*255).astype(np.uint8), (5, 5), 1.5)
    speed, dt, distance = 5., .06, 12.
    results = []
    for k in range(8):
        image = render_wall(distance-speed*dt*k, texture, config)
        result = estimator.update(image, 10.+k*dt, level_quaternion(), np.array([speed, 0., 0.]),
                                  usable=np.ones((config.height, config.width), bool))
        if result is not None and result['ttc'] is not None:
            results.append((result['ttc'], (distance-speed*dt*k)/speed))
    assert results, 'an approached textured wall must produce evidence'
    measured = np.median([m for m, _ in results[-3:]])
    truth = results[-1][1]
    assert measured == pytest.approx(truth, rel=.35)


def test_rotation_alone_raises_no_alarm():
    config = Looming2Config(width=160, height=90, median=1)
    estimator = LoomingEstimator2(config)
    rng = np.random.default_rng(1)
    texture = cv2.GaussianBlur((rng.random((256, 256))*255).astype(np.uint8), (5, 5), 1.5)
    image = render_wall(20., texture, config)
    shifted = np.roll(image, 3, axis=1)
    estimator.update(image, 1., level_quaternion(0.), np.array([3., 0., 0.]))
    result = estimator.update(shifted, 1.06, level_quaternion(np.radians(-1.9)), np.array([3., 0., 0.]))
    assert result is None or result['ttc'] is None or result['ttc'] > 2.


def test_uniform_image_is_no_evidence():
    config = Looming2Config(width=160, height=90)
    estimator = LoomingEstimator2(config)
    flat = np.full((90, 160, 3), 90, np.uint8)
    estimator.update(flat, 1., level_quaternion(), np.array([6., 0., 0.]))
    result = estimator.update(flat, 1.06, level_quaternion(), np.array([6., 0., 0.]))
    assert result is not None and not result['evidence'] and result['ttc'] is None


def motion_rows(times, now):
    rows = np.zeros((len(times), 16))
    rows[:, 0] = now
    rows[:, 1] = times
    rows[:, 6:10] = level_quaternion()
    rows[:, 10] = np.linspace(1., 2., len(times))
    return rows


def test_pose_lookup_interpolates_and_refuses_stale_or_future_motion():
    rows = motion_rows([1., 1.1, 1.2], now=1.25)
    quaternion, velocity = pose_at(rows, 1.05, now=1.25)
    assert velocity[0] == pytest.approx(1.25) and np.linalg.norm(quaternion) == pytest.approx(1.)
    assert pose_at(rows, 1.3, now=1.25) is None      # beyond the observed motion
    assert pose_at(rows, .9, now=1.25) is None       # before it
    assert pose_at(rows, 1.1, now=1.5) is None       # motion itself is stale
    assert pose_at(None, 1.1, now=1.25) is None
