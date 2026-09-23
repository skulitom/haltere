import time

import numpy as np
import pytest

from haltere.liftoff.geometry_shadow import MotionBuffer, ShadowGeometry
from haltere.vision.camera import Camera


def test_motion_history_is_ordered_bounded_and_never_waits_on_lock():
    buffer=MotionBuffer(capacity=3)
    for i in range(5):
        assert buffer.publish(i+.01,i,i,[i,0,0],[1,0,0,0],[1,0,0],[2,0,0])
    np.testing.assert_allclose(buffer.read()[:,2],[2,3,4])
    with buffer.lock:
        assert buffer.read() is None
        assert not buffer.publish(6.,6.,6.,[0,0,0],[1,0,0,0],[0,0,0],[0,0,0])
    with pytest.raises(ValueError,match='observed'):
        buffer.publish(6.,7.,6.,[0,0,0],[1,0,0,0],[0,0,0],[0,0,0])


def test_shadow_waits_for_pose_alignment_and_keeps_unknown_space_explicit():
    diagnostic=ShadowGeometry(Camera(640,360,200,30))
    rgb=np.zeros((360,640,3),np.uint8)
    stamp=time.monotonic()
    assert diagnostic.observe(rgb,stamp,None,stamp)['status']=='waiting_for_motion'
    buffer=MotionBuffer()
    buffer.publish(stamp,stamp,10.,[0,0,0],[1,0,0,0],[0,0,0],[2,0,0])
    assert diagnostic.observe(rgb,stamp-.1,buffer.read(),stamp)['status']=='image_pose_unaligned'
    assert diagnostic.observe(rgb,stamp,buffer.read(),stamp+.2)['status']=='stale_motion'
    result=diagnostic.observe(rgb,stamp,buffer.read(),stamp)
    assert result['valid_points']==0 and not result['live_authority']
    assert not result['coverage_certified']
    assert result['status']=='nominal_unverified'


def test_dense_hypotheses_remain_separate_transient_and_never_claim_free_space():
    class Depth:
        def predict(self, rgb):
            return np.full(rgb.shape[:2], 2., np.float32)
    diagnostic = ShadowGeometry(Camera(640, 360, 200, 30), Depth())
    rgb = np.full((360, 640, 3), 240, np.uint8)
    stamp = time.monotonic(); buffer = MotionBuffer()
    buffer.publish(stamp, stamp, 1., [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [2, 0, 0])
    result = diagnostic.observe(rgb, stamp, buffer.read(), stamp)
    assert result['accepted_points'] == []
    assert len(result['dense_obstacle_points']) > 100
    assert len(diagnostic.memory.cells) == 0  # No model points become triangulations.
    assert len(diagnostic.dense_memory.cells) <= 384
    assert not diagnostic.dense_memory.metadata()['interpolated_patches']
    assert not result['coverage_certified']
    late = diagnostic.dense_memory.update([0, 0, 0], stamp+1.1)
    assert len(late['points']) == len(late['triangles']) == 0
