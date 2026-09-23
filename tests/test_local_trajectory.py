import numpy as np
import pytest

from haltere.vision.local_trajectory import LocalTrajectoryPlanner, TrajectoryConfig, rollout
from haltere.vision.surface_memory import surface_patches
from haltere.vision.camera import Camera


def view(position=(0,0,0), quaternion=(1,0,0,0)):
    return Camera(640,360,200,30), np.asarray(position), np.asarray(quaternion)


def surfaces(points, timestamp=1.):
    points=np.asarray(points,float).reshape(-1,3)
    sigma=np.full(len(points),.05)
    triangles,errors=surface_patches(points,sigma)
    return dict(points=points,sigma=sigma,triangles=triangles,triangle_sigma=errors,timestamp=timestamp)


def test_rollout_preserves_initial_motion_and_checks_a_braking_tail():
    config=TrajectoryConfig()
    path=rollout([0,0,0],[4,0,0],[2,0,0],config)
    assert path['velocities'][0,0]==4  # Never invent compliance with a speed cap.
    np.testing.assert_allclose(path['positions'][5],[1.,0,0])
    acceleration=np.diff(path['velocities'],axis=0)/np.diff(path['times'])[:,None]
    assert np.linalg.norm(acceleration,axis=1).max() <= config.acceleration_mps2+1e-9
    assert np.linalg.norm(path['velocities'][-1]) < .01
    assert path['times'][-1] > config.reaction_s+config.horizon_s


def test_wall_requires_change_but_an_open_gate_does_not():
    planner=LocalTrajectoryPlanner()
    for opening in [False,True]:
        cloud=[[4,y,z] for y in np.arange(-5,5.1,.5) for z in np.arange(-5,5.1,.5)
               if not opening or abs(y)>=2.5 or abs(z)>=2.5]
        result=planner.propose([0,0,0],[2,0,0],[2,0,0],surfaces(cloud),1.)
        assert result['changed'] is (not opening)
        assert result['selected_margin_m'] >= planner.config.extra_margin_m
        assert not result['coverage_certified']


def test_pillar_detour_is_in_task_frame_and_unknown_is_not_certified():
    planner=LocalTrajectoryPlanner()
    cloud=[[4,y,z] for y in [-.4,0,.4] for z in np.arange(-4,4.1,.5)]
    original=planner.propose([0,0,0],[2,0,0],[2,0,0],surfaces(cloud),1.,view=view())
    assert original['velocity'][0] > 0 and abs(original['velocity'][1]) > .1
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    origin=np.array([7.,-12.,3.])
    turned=planner.propose(origin,rotation@[2,0,0],rotation@[2,0,0],
                           surfaces(np.asarray(cloud)@rotation.T+origin),1.,
                           view=view(origin,[np.sqrt(.5),0,0,np.sqrt(.5)]))
    np.testing.assert_allclose(turned['velocity'],rotation@original['velocity'],atol=1e-9)
    empty=planner.propose([0,0,0],[0,0,0],[2,0,0],surfaces([]),1.)
    assert not empty['changed'] and not empty['coverage_certified']
    assert empty['status']=='nominal_unverified'


def test_stale_geometry_and_too_late_braking_are_explicit():
    planner=LocalTrajectoryPlanner()
    result=planner.propose([0,0,0],[2,0,0],[2,0,0],surfaces([]),2.)
    assert result['status']=='stale_geometry_brake'
    with pytest.raises(ValueError,match='causal'):
        planner.propose([0,0,0],[2,0,0],[2,0,0],surfaces([]),.9)
    cloud=[[.2,y,z] for y in np.arange(-5,5.1,.5) for z in np.arange(-5,5.1,.5)]
    result=planner.propose([0,0,0],[3,0,0],[2,0,0],surfaces(cloud),1.)
    assert result['status']=='no_observed_clear_path_brake'
    assert result['selected_margin_m'] < 0


def test_no_new_detour_without_a_view_and_no_blind_dive():
    from haltere.vision.local_trajectory import detour_in_view
    assert not detour_in_view(np.array([[1,0,0],[4,0,-3.]]), view(), .35)
    assert detour_in_view(np.array([[1,0,0],[4,0,3.]]), view(), .35)
    cloud=[[4,y,z] for y in [-.4,0,.4] for z in np.arange(-4,4.1,.5)]
    result=LocalTrajectoryPlanner().propose([0,0,0],[2,0,0],[2,0,0],surfaces(cloud),1.)
    np.testing.assert_array_equal(result['velocity'],[0,0,0])
