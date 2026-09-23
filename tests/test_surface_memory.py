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
