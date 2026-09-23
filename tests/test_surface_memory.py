import numpy as np
import pytest

from haltere.vision.surface_memory import SurfaceMemory,observed_path_margin,surface_patches,triangle_distance


def test_finite_triangle_distance_includes_edges_and_vertices():
    triangle=np.array([[[0,0,0],[2,0,0],[0,2,0]]],float)
    result=triangle_distance([[.5,.5,3],[2,2,0],[-1,0,0]],triangle)[:,0]
    np.testing.assert_allclose(result,[3,np.sqrt(2),1])


def plane(opening=False):
    return np.array([[4,y,z] for y in np.arange(-4,4.1,.5) for z in np.arange(-4,4.1,.5)
                     if not opening or abs(y)>=2.5 or abs(z)>=2.5],float)


def test_surface_patches_detect_a_wall_without_filling_a_gate_opening():
    path=np.column_stack([np.linspace(0,6,61),np.zeros(61),np.zeros(61)])
    for opening in [False,True]:
        cloud=plane(opening);sigma=np.full(len(cloud),.05)
        triangles,uncertainty=surface_patches(cloud,sigma)
        surfaces=dict(points=cloud,sigma=sigma,triangles=triangles,triangle_sigma=uncertainty)
        result=observed_path_margin(path,surfaces)
        assert result['observed_collision'] is (not opening)
        assert not result['coverage_certified']


def test_surface_memory_expires_and_unknown_space_never_becomes_certified():
    memory=SurfaceMemory(lifetime=1.)
    memory.update([0,0,0],1.,[[2,0,0]],[.1])
    assert len(memory.update([0,0,0],1.5)['points'])==1
    empty=memory.update([0,0,0],2.1)
    result=observed_path_margin([[0,0,0],[3,0,0]],empty)
    assert result['margin_m']==np.inf and not result['coverage_certified']
    with pytest.raises(ValueError,match='causal'):
        memory.update([0,0,0],1.)


def test_static_obstacle_survives_stationary_depth_gap_without_refreshing_its_age():
    memory=SurfaceMemory(lifetime=None,max_points=32)
    cloud=np.array([[3,y,z] for y in [-1,0,1] for z in [-1,0,1]],float)
    memory.update([0,0,0],1.,cloud,np.full(len(cloud),.1))
    retained=memory.update([1,0,0],10.)
    path=np.column_stack([np.linspace(1,4,31),np.zeros(31),np.zeros(31)])
    assert observed_path_margin(path,retained)['observed_collision']
    assert retained['oldest_observation_age_s']==9.
    assert not retained['coverage_certified']
    removed=memory.update([20,0,0],11.)
    assert len(removed['points'])==0
    assert removed['oldest_observation_age_s'] is None


def test_static_memory_is_bounded_and_evicts_farthest_before_nearby_hazards():
    memory=SurfaceMemory(lifetime=None,max_points=2)
    retained=memory.update([0,0,0],1.,[[8,0,0],[2,0,0],[4,0,0]],[.1,.1,.1])
    np.testing.assert_allclose(retained['points'],[[2,0,0],[4,0,0]])
    assert retained['capacity_evictions']==1
    assert memory.metadata()['lifetime_s'] is None
    with pytest.raises(ValueError):
        SurfaceMemory(lifetime=None)


def test_reused_patches_match_reconstruction_and_do_not_alias_returned_geometry():
    memory=SurfaceMemory(lifetime=None,max_points=32)
    cloud=np.array([[3,y,z] for y in [-1,0,1] for z in [-1,0,1]],float)
    sigma=np.full(len(cloud),.1)
    first=memory.update([0,0,0],1.,cloud,sigma)
    expected=surface_patches(cloud,sigma)
    first['triangles'][:]=100  # A diagnostic caller cannot corrupt stored patches.
    repeated=memory.update([0,0,0],5.)
    np.testing.assert_array_equal(repeated['triangles'],expected[0])
    assert repeated['oldest_observation_age_s']==4.
    changed=memory.update([0,0,0],6.,cloud,sigma*.5)
    np.testing.assert_array_equal(changed['triangle_sigma'],surface_patches(cloud,sigma*.5)[1])


def test_broad_phase_preserves_exhaustive_uncertain_surface_margin():
    rng=np.random.default_rng(812)
    for _ in range(12):
        path=rng.normal(size=(35,3)).cumsum(axis=0)*.2
        triangles=rng.normal(size=(80,3,3))*3+rng.uniform(-15,15,(80,1,3))
        cloud=rng.uniform(-6,6,(40,3));errors=rng.uniform(0,1.5,40)
        triangle_errors=rng.uniform(0,1.5,80)
        surfaces=dict(points=cloud,sigma=errors,triangles=triangles,triangle_sigma=triangle_errors)
        expected=min(np.min(triangle_distance(path,triangles)-triangle_errors[None,:]-.35),
                     np.min(np.linalg.norm(path[:,None,:]-cloud,axis=2)-errors[None,:]-.35))
        assert observed_path_margin(path,surfaces)['margin_m']==pytest.approx(expected,abs=1e-12)


def test_inferred_patch_expires_without_deleting_its_measured_obstacle_points():
    memory=SurfaceMemory(lifetime=None,max_points=64,patch_lifetime=3.)
    cloud=np.array([[3,y,z] for y in [-1.25,0,1.25] for z in [-1.25,0,1.25]
                    if abs(y)==1.25 or abs(z)==1.25])
    early=memory.update([0,0,0],1.,cloud,np.full(len(cloud),.1))
    assert observed_path_margin([[3,0,0]],early)['observed_collision']
    late=memory.update([0,0,0],5.)
    assert len(late['points'])==len(cloud) and len(late['triangles'])==0
    assert not observed_path_margin([[3,0,0]],late)['observed_collision']
    assert observed_path_margin([[3,1.25,0]],late)['observed_collision']
    assert late['oldest_observation_age_s']==4. and not late['coverage_certified']
