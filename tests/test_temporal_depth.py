import numpy as np
import pytest

from haltere.vision.camera import Camera,quat_wxyz_to_mat
from haltere.vision.temporal_depth import TemporalDepth,MultiBaselineDepth,triangulate_motion


def project(camera,world,position,q):
    return camera.project_body((world-position)@quat_wxyz_to_mat(q))[0]


def test_known_metric_translation_and_rotation_recover_surface_points():
    cam=Camera(320,180,100,30)
    world=np.array([[5,-1,2],[8,2,3],[12,0,4]],float)
    p0=np.zeros(3);p1=np.array([.3,.5,.1]);q0=np.array([1.,0,0,0])
    q1=np.array([np.cos(.05),0,0,np.sin(.05)])
    result=triangulate_motion(cam,project(cam,world,p0,q0),project(cam,world,p1,q1),p0,q0,p1,q1)
    assert result['valid'].all()
    np.testing.assert_allclose(result['position_world'],world,atol=1e-9)
    np.testing.assert_allclose(result['range_m'],np.linalg.norm(world-p1,axis=1),atol=1e-9)
    assert not result['establishes_free_space']


def test_rotation_alone_and_inconsistent_matches_do_not_supply_depth():
    cam=Camera(320,180,100,0);p=np.zeros(3);q0=np.array([1.,0,0,0])
    q1=np.array([np.cos(.1),0,0,np.sin(.1)]);world=np.array([[5,0,0]],float)
    a=project(cam,world,p,q0);b=project(cam,world,p,q1)
    assert not triangulate_motion(cam,a,b,p,q0,p,q1)['valid'].any()
    p1=np.array([0.,.5,0]);b=project(cam,world,p1,q0)+[0,20]
    assert not triangulate_motion(cam,a,b,p,q0,p1,q0)['valid'].any()
    # A HUD feature or nearly static ground capture can have apparent parallax
    # without enough measured translation to supply meaningful metric depth.
    assert not triangulate_motion(cam,a,a+[5,0],p,q0,[0,.0001,0],q0)['valid'].any()


def test_unknown_pixels_and_noncausal_frames_are_rejected():
    tracker=TemporalDepth(Camera(32,18,10,0))
    gray=np.zeros((18,32),np.uint8);mask=gray.copy()
    assert tracker.update(gray,[0,0,0],[1,0,0,0],1.,mask) is None
    assert tracker.update(gray,[1,0,0],[1,0,0,0],1.2,mask) is None
    with pytest.raises(ValueError,match='chronological'):
        tracker.update(gray,[1,0,0],[1,0,0,0],1.1,mask)
    with pytest.raises(ValueError,match='quaternion'):
        TemporalDepth(Camera(32,18,10,0)).update(gray,[0,0,0],[0,0,0,0],1.,mask)


def test_tracked_image_translation_has_the_expected_metric_depth():
    import cv2
    rng=np.random.default_rng(87)
    old=cv2.GaussianBlur(rng.integers(0,256,(90,160),dtype=np.uint8),(3,3),0)
    current=cv2.warpAffine(old,np.float32([[1,0,5],[0,1,0]]),(160,90))
    mask=np.zeros_like(old);mask[12:-12,12:-12]=255
    tracker=TemporalDepth(Camera(160,90,100,0))
    assert tracker.update(old,[0,0,0],[1,0,0,0],1.,mask) is None
    result=tracker.update(current,[0,.5,0],[1,0,0,0],1.2,mask)
    assert not result['valid'].any()  # two views are still only a hypothesis
    third=cv2.warpAffine(old,np.float32([[1,0,10],[0,1,0]]),(160,90))
    result=tracker.update(third,[0,1.,0],[1,0,0,0],1.4,mask)
    valid=result['valid']
    assert valid.sum()>20
    assert np.median(result['optical_depth_m'][valid])==pytest.approx(10.,rel=.02)


def test_third_view_rejects_a_surface_with_inconsistent_image_motion():
    import cv2
    rng=np.random.default_rng(87)
    old=cv2.GaussianBlur(rng.integers(0,256,(90,160),dtype=np.uint8),(3,3),0)
    mask=np.zeros_like(old);mask[15:-15,15:-15]=255
    tracker=TemporalDepth(Camera(160,90,100,0))
    tracker.update(old,[0,0,0],[1,0,0,0],1.,mask)
    # Two final rays suggest a 10 m surface, but the intervening observation
    # puts it somewhere incompatible. This must not become a solid obstacle.
    middle=cv2.warpAffine(old,np.float32([[1,0,12],[0,1,0]]),(160,90))
    tracker.update(middle,[0,.5,0],[1,0,0,0],1.2,mask)
    last=cv2.warpAffine(old,np.float32([[1,0,10],[0,1,0]]),(160,90))
    result=tracker.update(last,[0,1.,0],[1,0,0,0],1.4,mask)
    assert not result['valid'].any()


def test_longer_baseline_adds_precise_depth_during_slow_translation():
    import cv2
    rng = np.random.default_rng(19)
    original = cv2.GaussianBlur(rng.integers(0, 256, (180, 320), dtype=np.uint8), (3, 3), 0)
    mask = np.zeros_like(original)
    mask[20:-20, 40:-40] = 255
    camera = Camera(320, 180, 200, 0)
    fast, combined = TemporalDepth(camera), MultiBaselineDepth(camera)
    fast_precise = combined_precise = 0
    for i in range(20):
        frame = cv2.warpAffine(original, np.float32([[1, 0, 2*i], [0, 1, 0]]), (320, 180))
        arguments = (frame, [0, .02*i, 0], [1, 0, 0, 0], 1.+.1*i, mask)
        one, both = fast.update(*arguments), combined.update(*arguments)
        for result, name in ((one, 'fast'), (both, 'combined')):
            if result is None:
                continue
            good = result['valid'] & (result['range_sigma_m'] < .1*result['range_m'])
            if good.any():
                assert np.median(result['optical_depth_m'][good]) == pytest.approx(2., rel=.02)
            if name == 'fast':
                fast_precise += good.sum()
            else:
                combined_precise += good.sum()
                assert not result['establishes_free_space']
    assert combined_precise > fast_precise + 20


def test_precision_keyframe_limits_are_validated():
    for limits in ({'refresh_sigma_fraction': 0}, {'refresh_sigma_m': float('nan')}):
        with pytest.raises(ValueError, match='precision'):
            TemporalDepth(Camera(), **limits)
