import copy
import numpy as np
import pytest
import torch

from haltere.liftoff.neural_replay import NeuralReplay,replay_camera_sensor


def test_passive_collection_does_not_enable_controller_retina():
    sensor=dict(raw_retina_active=False,retina_mode='gatenet_scene_v1',sha256='abc',
                scene_projection=dict(detector_sha256='abc'))
    before=copy.deepcopy(sensor)
    camera=replay_camera_sensor(sensor,True)
    assert camera['raw_retina_active'] and sensor==before
    with pytest.raises(ValueError,match='differ'):
        replay_camera_sensor({**sensor,'sha256':'other'},True)


def test_replay_preserves_causal_samples_without_future_labels(tmp_path):
    replay=NeuralReplay(tmp_path/'flight.npz',dict(retina=2,goal=4),1)
    retina=torch.tensor([[.25,-.5]])
    obs=dict(retina=torch.zeros(1,2),goal=torch.tensor([[1.,2.,3.,4.]]))
    replay.append(obs,retina,[0]*4,[0]*3,[1,0,0,0],4.,50001.12,50001.04,True)
    retina.fill_(9);obs['goal'].fill_(9)
    replay.save()
    with np.load(replay.path) as data:
        np.testing.assert_array_equal(data['retina'],[[.25,-.5]])
        np.testing.assert_array_equal(data['goal'],[[1,2,3,4]])
        assert data['clock'].dtype==np.float64
        assert abs(data['clock'][0,3]-.08)<1e-9
        assert not any('teacher' in k or 'route' in k for k in data.files)
    with pytest.raises(FileExistsError):
        replay.save()
