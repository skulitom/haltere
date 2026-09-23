import numpy as np
import pytest
import torch

from haltere.brain.motor_baseline import MotorPDConfig
from haltere.liftoff.guidance_contract import GuidanceVelocityContract
from haltere.sim.quad import quat_to_mat


@pytest.mark.parametrize('quaternion', [[1., 0., 0., 0.], [.7, .2, -.3, .5]])
def test_velocity_roundtrip_and_physical_axis_limits(quaternion):
    q = np.array(quaternion); q /= np.linalg.norm(q)
    rotation = quat_to_mat(torch.tensor(q)[None])[0].numpy()
    contract = GuidanceVelocityContract.pd(MotorPDConfig(), 2.5)
    velocity = np.array([1.2, -.7, .4])
    target = contract.body_target(velocity, rotation)
    np.testing.assert_allclose(contract.world_velocity(target, rotation), velocity, atol=1e-12)
    saturated = contract.world_velocity(rotation.T @ np.array([10., -10., -10.]), rotation)
    assert np.linalg.norm(saturated[:2]) == pytest.approx(2.5)
    assert saturated[2] == -1.2


@pytest.mark.parametrize('speed', [1., 2., 2.5, 3., 4.])
def test_brain_contract_accounts_for_training_speed_and_sensory_scaling(speed):
    reference = 3.
    contract = GuidanceVelocityContract.brain(speed, reference)
    desired = np.array([.4, -.2, .3])
    target = contract.body_target(desired, np.eye(3))
    scale = max(1., reference / speed)
    # The training teacher sees scaled horizontal velocity. Its zero-error
    # physical velocity is the target gain divided by that same scale.
    teacher_velocity = target * np.array([1., 1., .8])
    teacher_velocity[:2] /= scale
    np.testing.assert_allclose(teacher_velocity, desired)
    assert contract.horizontal_speed == min(speed, reference)
    assert not contract.metadata()['supplies_motor_commands']


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_invalid_guidance_contract_is_rejected(value):
    with pytest.raises(ValueError, match='finite and positive'):
        GuidanceVelocityContract(value, .8, 2.5, 1.2)
    with pytest.raises(ValueError, match='finite and positive'):
        GuidanceVelocityContract.brain(value, 3.)
    with pytest.raises(ValueError, match='finite and positive'):
        GuidanceVelocityContract.brain(2.5, value)
