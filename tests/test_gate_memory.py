import numpy as np
from haltere.liftoff.gate_memory import PassedGateMemory


def test_passage_inhibits_return_but_keeps_the_next_arch_and_expires():
    memory=PassedGateMemory();point=np.array([6.,0.,1.])
    memory.observe(point,0.,np.array([0.,0.,1.]),0.)
    assert memory.advance(np.array([5.,0.,1.]),2.) is None
    passed=memory.advance(np.array([6.8,0.,1.]),3.)
    np.testing.assert_array_equal(passed,point)
    position=np.array([7.,0.,1.])
    assert memory.rejects(point+np.array([.3,.2,0]),4.,position)
    assert memory.rejects(point-np.array([3.5,0,0]),4.,position)
    assert not memory.rejects(point+np.array([4.5,0,0]),4.,position)
    assert not memory.rejects(point,34.,position)


def test_camera_rotation_or_a_moving_estimate_cannot_simulate_passage():
    memory=PassedGateMemory();position=np.array([0.,0.,1.]);point=np.array([5.,0.,1.])
    memory.observe(point,0.,position,0.)
    for i in range(1,8):
        memory.observe(point-np.array([i,0.,0.]),i*.1,position,i*.1)
        assert memory.advance(position,i*.1) is None
    assert memory.passages==0 and not memory.rejects(point,1.,position)


def test_flying_beside_a_gate_does_not_mark_it_passed():
    memory=PassedGateMemory();point=np.array([5.,0.,1.])
    memory.observe(point,0.,np.array([0.,0.,1.]),0.)
    assert memory.advance(np.array([6.,3.,1.]),3.) is None
    assert not memory.rejects(point,3.,np.array([6.,3.,1.]))
    memory.advance(np.array([7.,0.,1.]),7.)
    assert memory.active is None and memory.passages==0


def test_landmark_memory_is_invariant_to_world_origin_and_heading():
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    shift=np.array([950.,-120.,27.]);point=np.array([5.,0.,1.])
    memory=PassedGateMemory()
    memory.observe(rotation@point+shift,0.,rotation@np.array([0.,0.,1.])+shift,0.)
    passed=memory.advance(rotation@np.array([6.,0.,1.])+shift,3.)
    np.testing.assert_allclose(passed,rotation@point+shift)
