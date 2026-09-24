"""Looming v2 time-to-contact and the camera-process motion lookup; offline only."""
import numpy as np
import pytest

from haltere.liftoff.camera_process import (LOOMING_SLOTS, SHARED_SIZE, ProcessRetinaCamera, clearance_sample,
                                             looming_values, pose_at)
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


def pitched(deg):
    """Level flight attitude pitched nose-down by deg (body x forward, z up), as at race speed."""
    return np.array([np.cos(np.radians(deg)/2), 0., np.sin(np.radians(deg)/2), 0.])


def render_scene(config, texture, hit, quaternion, scale):
    """Grey image of a textured surface hit(world rays) -> (valid, a, b); a uniform sky elsewhere."""
    from haltere.vision.looming2 import quat_wxyz_to_mat
    camera = Camera(config.width, config.height, config.focal_320*config.width/320, config.tilt_deg)
    ys, xs = np.mgrid[0:config.height, 0:config.width].astype(np.float64)
    rays = camera.unproject_body(np.stack((xs.ravel(), ys.ravel()), 1)) @ quat_wxyz_to_mat(quaternion).T
    valid, a, b = hit(rays)
    h, w = texture.shape
    grey = np.where(valid, texture[np.mod((np.nan_to_num(a)*scale).astype(int), h),
                                   np.mod((np.nan_to_num(b)*scale).astype(int), w)], 200)
    return np.repeat(grey.reshape(config.height, config.width)[..., None], 3, -1).astype(np.uint8)


def wall_ahead(distance):
    def hit(rays):
        with np.errstate(divide='ignore', invalid='ignore'):
            t = distance/rays[:, 0]
        ok = rays[:, 0] > 1e-3
        return ok, np.where(ok, rays[:, 2]*t, np.nan), np.where(ok, rays[:, 1]*t, np.nan)
    return hit


def rising_ground(crossing, slope_deg=30., crest=.3):
    """Ground rising at slope_deg that crosses the flight path `crossing` ahead; sky above `crest` m over the path."""
    a = np.tan(np.radians(slope_deg))

    def hit(rays):
        den = rays[:, 0]*a-rays[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(den > 1e-6, crossing*a/den, np.nan)
        ok = np.isfinite(t) & (t > 0) & (t*rays[:, 2] <= crest)
        return ok, np.where(ok, t*rays[:, 0], np.nan), np.where(ok, t*rays[:, 1], np.nan)
    return hit


def approach(scene, config, scale, speed=5., dt=.06, start=9., frames=9):
    estimator = LoomingEstimator2(config)
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur((rng.random((512, 512))*255).astype(np.uint8), (5, 5), 1.5)
    quaternion = pitched(25.)
    results = []
    for k in range(frames):
        image = render_scene(config, texture, scene(start-speed*dt*k), quaternion, scale)
        result = estimator.update(image, 10.+k*dt, quaternion, np.array([speed, 0., 0.]),
                                  usable=np.ones((config.height, config.width), bool))
        if result is not None:
            results.append(result)
    return results


def test_vertical_windows_report_where_the_expansion_is_without_changing_the_alarm():
    config = Looming2Config(width=240, height=135, median=1, sig_k=0.)
    wall = approach(wall_ahead, config, 40.)
    alarm_only = approach(wall_ahead, Looming2Config(width=240, height=135, median=1, sig_k=0., vertical_windows=()), 40.)
    assert [r['ttc'] for r in wall] == [r['ttc'] for r in alarm_only]
    assert 'below_fraction' not in alarm_only[-1]
    # a wall facing the drone: the surfaces above and below the path cross it together
    level = [r['below_fraction'] for r in wall if r['below_fraction'] is not None]
    assert len(level) >= .8*len(wall) and all(.3 <= b <= .7 for b in level)
    assert np.median([r['ttc_lower'] for r in wall]) == pytest.approx(np.median([r['ttc'] for r in wall]), rel=.25)
    # ground rising into the path under open sky: all urgent expansion lies below it
    ground = approach(rising_ground, config, 10.)
    below = [r for r in ground if r['below_fraction_1side'] is not None]
    assert len(below) >= .8*len(ground)
    assert all(r['below_fraction_1side'] == 1. and r['ttc_lower'] is not None for r in below)
    assert all(r['below_fraction'] is None and not r['evidence_upper'] for r in ground)   # strict: no upper evidence


def test_looming_slots_carry_below_fraction_and_lower_ttc_to_the_pilot():
    import multiprocessing as mp
    from queue import Queue
    from types import SimpleNamespace
    result = dict(ttc=.8, distance=4.8, evidence=True, below_fraction=.9, below_fraction_1side=.9, ttc_lower=.6)
    values = looming_values(12.5, result)
    assert clearance_sample(values) == dict(time=12.5, ttc=.8, distance=4.8, evidence=True, below_fraction=.9,
                                            ttc_lower=.6)
    one_sided = looming_values(12.55, dict(result, below_fraction=None, below_fraction_1side=1.))
    assert clearance_sample(one_sided)['below_fraction'] is None      # published: both windows need evidence
    blind = clearance_sample(looming_values(12.6, dict(ttc=None, distance=None, evidence=False)))
    assert blind == dict(time=12.6, ttc=None, distance=None, evidence=False, below_fraction=None, ttc_lower=None)
    assert clearance_sample([0.]*len(values)) is None
    camera = ProcessRetinaCamera.__new__(ProcessRetinaCamera)
    camera.queue, camera.process, camera.looming = Queue(maxsize=2), SimpleNamespace(exitcode=None), True
    camera.data = mp.get_context('spawn').Array('d', SHARED_SIZE, lock=True)
    camera._latest = camera._error = camera._clearance = None
    camera._diagnostics = {}
    camera.done = SimpleNamespace(is_set=lambda: False)
    np.frombuffer(camera.data.get_obj(), dtype=np.float64)[LOOMING_SLOTS] = values
    assert camera.clearance == clearance_sample(values)
