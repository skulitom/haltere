import numpy as np
import torch
import pytest

from haltere.brain.retina import brain_to_processed
from haltere.train.demonstration_scene import causal_indices,measured_brain_actions
from haltere.train.onpolicy_scene import targets


def test_capture_must_finish_before_image_can_enter_student():
    available=np.array([1.02,1.10,1.18])
    np.testing.assert_array_equal(causal_indices(available,[1.,1.03,1.099,1.10,1.2]),[-1,0,0,1,2])


def test_observed_throttle_uses_deployed_not_teachers_calibration():
    calibration=dict(hover_processed=.13566,hover_stick_sim=-.4302,throttle_scale=.8,stick_sign=[-1,1,1])
    processed=np.array([[.12,.1,-.2,.3],[.25,-.3,.2,-.1]],dtype=np.float32)
    brain=measured_brain_actions(processed,calibration)
    np.testing.assert_allclose(brain_to_processed(torch.tensor(brain),calibration),processed,atol=1e-7)
    assert brain[0,1]<0 and brain[0,0]<calibration['hover_stick_sim']


def test_observed_labels_do_not_replace_student_senses(monkeypatch):
    def replay(parent,batch,blank=False):return batch['goal'].clone(),None,None
    monkeypatch.setattr('haltere.train.onpolicy_scene.rollout',replay)
    goal=torch.ones(1,3,4);teacher=torch.zeros(1,3,4)
    batch=dict(goal=goal,teacher_action=teacher,correct=torch.tensor([[[0.],[.5],[1.]]]))
    base,label=targets(None,batch)
    assert torch.equal(base,goal) and torch.equal(batch['goal'],torch.ones_like(goal))
    assert torch.equal(label[0,:,0],torch.tensor([1.,.5,0.]))


def test_motor_refit_reports_blank_drift_instead_of_claiming_exact_retention(monkeypatch):
    from haltere.train.onpolicy_scene import assess
    class Brain:
        device='cpu'
    parent,student=Brain(),Brain()
    class Replay:
        def batch(self,ids,device):
            return dict(action=torch.zeros(1,21,4),goal=torch.zeros(1,21,4),
                        retina=torch.zeros(1,21,2),teacher_action=torch.zeros(1,21,4),
                        correct=torch.ones(1,21,1))
    def replay(brain,batch,**kw):
        return torch.full_like(batch['action'],0. if brain is parent else 1.),None,None
    monkeypatch.setattr('haltere.train.onpolicy_scene.rollout',replay)
    with pytest.raises(RuntimeError,match='retention'):
        assess(student,parent,Replay(),[0])
    result=assess(student,parent,Replay(),[0],require_blank_parent=False)
    assert result['blank_parent_max_delta']==1.
    assert result['mse_axes']['blank_correction']==[1.]*4
