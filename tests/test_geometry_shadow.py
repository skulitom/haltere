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
