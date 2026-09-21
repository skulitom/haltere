import numpy as np
import torch

from haltere.train.onpolicy_scene import next_gate_labels,continuous_groups,targets,bounded_route_lookahead


def test_offline_labels_require_forward_aperture_crossing_in_order():
    gates=[dict(id=i,base=np.array([x,0.,0.]),normal=np.array([1.,0.,0.]),
                side=np.array([0.,1.,0.]),up=np.array([0.,0.,1.])) for i,x in enumerate([1.,3.])]
    order=[dict(gate_id=i,normal_direction=1) for i in range(2)]
    # Beside gate 0, then backwards through it, then genuinely forward.
    positions=np.array([[0.,2.,1.],[2.,2.,1.],[2.,0.,1.],[0.,0.,1.],[2.,0.,1.],[4.,0.,1.]])
    centres,progress=next_gate_labels(positions,gates,order)
    np.testing.assert_array_equal(progress,[0,0,0,0,1,2])
    np.testing.assert_allclose(centres[4],[3,0,1.2])


def test_stale_frames_split_recurrent_windows():
    good=np.ones(140,dtype=bool);good[70]=False
    ids,groups=continuous_groups(good)
    assert len(ids)==139 and 70 not in ids
    assert groups[0]!=groups[-1]
    assert len(np.unique(groups))==2


def test_correction_can_change_throttle_and_never_changes_student_inputs(monkeypatch):
    def mock_rollout(parent,batch,blank=False):
        return batch['goal'].clone(),None,None
    monkeypatch.setattr('haltere.train.onpolicy_scene.rollout',mock_rollout)
    b=dict(goal=torch.ones(1,2,4),teacher_goal=torch.zeros(1,2,4),
           correct=torch.tensor([[[0.],[1.]]]))
    original=b['goal'].clone()
    base,target=targets(None,b)
    assert torch.equal(base,b['goal']) and torch.equal(b['goal'],original)
    assert torch.equal(target[:,0],base[:,0])
    assert target[0,1,0]!=base[0,1,0]  # descent was previously suppressed


def test_window_rollout_uses_causal_preceding_neural_state():
    from haltere.train.human_brain import rollout
    class Brain:
        channel_dims={'goal':1}
        def init_state(self,n):return dict(v=torch.zeros(1,n),act=torch.zeros(n,1))
        def weight_matrix(self):return None
        def __call__(self,obs,state,w):
            v=state['v']+obs['goal'].T
            return v.T,dict(v=v,act=v.T),{}
    batch=dict(goal=torch.ones(1,3,1),action=torch.zeros(1,3,1),
               initial_v=torch.tensor([[7.]]),initial_act=torch.tensor([[6.]]))
    actual,_,_=rollout(Brain(),batch)
    assert actual.flatten().tolist()==[8.,9.,10.]
    assert batch['initial_v'].item()==7.


def test_route_target_keeps_intervening_rise_before_downhill_gate():
    points=np.array([[0.,0.,1.],[5.,0.,3.],[10.,0.,1.]])
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    target=bounded_route_lookahead(np.array([[1.,0.,1.4]]),np.array([0]),points,distance,[distance[-1]])
    assert target[0,2]>2.5  # pointing directly to the distant gate would descend
    assert 4<target[0,0]<6
