import torch

from haltere.train.motor_tracking import local_guidance
from haltere.sim.quad import QuadSim, QuadParams, QuadState, quat_from_euler, quat_to_mat
from tests.test_visual_assistance import checkpoint


def test_retina_windows_are_repeatable_and_do_not_change_physics_rng():
    from haltere.train.motor_tracking import retina_sequence
    stream = torch.arange(8*720).reshape(8,720).float()+1
    before = torch.random.get_rng_state().clone()
    sequence = retina_sequence(stream, 100, 4, 62, .5)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(sequence, retina_sequence(stream, 100, 4, 62, .5))
    missing = sequence.eq(0).all(-1)
    assert missing.any() and (~missing).any()
    assert (missing.reshape(10,10,4) == missing.reshape(10,10,4)[:, :1]).all()
    valid = sequence[~missing]
    assert (valid[:, 1:]-valid[:, :-1]).eq(1).all()


def test_collected_features_reproduce_motor_readout_and_batch_axis(checkpoint):
    from haltere.train.bptt import load_checkpoint
    from haltere.train.motor_tracking import motor_features
    brain, _, _ = load_checkpoint(checkpoint[0], 'cpu')
    state = brain.init_state(3)
    obs = {k: torch.zeros(3, v) for k, v in brain.channel_dims.items()}
    with torch.no_grad():
        _, state, aux = brain(obs, state)
        x = motor_features(brain, state)
    assert x.shape == (3, brain.readout.in_features)
    torch.testing.assert_close(brain.readout(x).tanh(), aux['u'])


def test_tracking_targets_are_rotation_equivariant_and_brake():
    sim = QuadSim(QuadParams(), 'cpu')
    q = QuadState.hover(2, 'cpu', 5.)
    yaw = torch.tensor([0., 1.7])
    q.quat = quat_from_euler(yaw*0, yaw*0, yaw)
    R = quat_to_mat(q.quat)
    q.vel = torch.einsum('bij,bj->bi', R, torch.tensor([[3., 0., 0.], [3., 0., 0.]]))
    relative, rate, moving = local_guidance(sim.sensors(q), yaw+.3, q.pos[:, 2], q.pos,
                                           torch.zeros(2, dtype=torch.bool))
    torch.testing.assert_close(relative[0], relative[1])
    torch.testing.assert_close(rate[0], rate[1])
    assert moving.all()
    relative, _, moving = local_guidance(sim.sensors(q), yaw, q.pos[:, 2], q.pos,
                                         torch.ones(2, dtype=torch.bool))
    assert not moving.any()
    assert (relative[:, 0] < 0).all()
    assert (relative[:, :2].norm(dim=-1) <= 3.00001).all()
