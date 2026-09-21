"""Visual sensory currents into existing navigation neurons; no path decoder.

Only the new sensory tuning and gain may learn. The original recurrent graph,
motor readout and every original weight remain frozen. A missing image adds
exactly zero current, retaining the parent's complete dynamics.
"""
import copy
import torch

from ..brain.model import BrainConfig, ConnectomeRNN


KEY = 'retina__goal'
PREFIX = f'encoders.{KEY}.'
ADDED_PARAMETERS = {PREFIX + name for name in ('U', 'log_gain', 'threshold')}


def navigation_config(parent):
    cfg = copy.deepcopy(parent)
    if cfg.sensory.get('retina') != 'lptc' or cfg.blank_encoders or cfg.opponent_encoders:
        raise ValueError('Expected the original passive-retina sensory interface')
    cfg.sensory['retina'] = ['lptc', 'goal']
    cfg.blank_encoders = ('retina__lptc',)
    cfg.opponent_encoders = (KEY,)
    return cfg


def add_navigation_input(parent, cfg, graph):
    cfg = copy.deepcopy(cfg)
    cfg.brain = navigation_config(cfg.brain)
    brain = ConnectomeRNN(graph, parent.channel_dims, cfg.brain, parent.device)
    missing, unexpected = brain.load_state_dict(parent.state_dict(), strict=False)
    expected = ADDED_PARAMETERS | {PREFIX + 'group', f'idx_{KEY}'}
    if unexpected or set(missing) != expected:
        raise ValueError('Unexpected navigation sensory migration')
    brain.eval().requires_grad_(False)
    encoder = brain.encoders[KEY]
    parameters = [encoder.U, encoder.log_gain]
    for parameter in parameters:
        parameter.requires_grad_(True)
    return brain, cfg, parameters


def validate_navigation_retention(candidate, parent):
    """Reject changes outside the declared added input, including config changes."""
    expected = copy.deepcopy(parent['config'])
    # Normalize tuple fields to the list representation used by checkpoints.
    from ..config import dataclass_to_dict
    expected['brain'] = dataclass_to_dict(navigation_config(BrainConfig.from_dict(expected['brain'])))
    if candidate['config'] != expected or candidate['graph'] != parent['graph'] or candidate['channels'] != parent['channels']:
        raise ValueError('Navigation input changed the parent control configuration')
    original, new = parent['model'], candidate['model']
    if set(new) - set(original) != ADDED_PARAMETERS or set(original) - set(new):
        raise ValueError('Unexpected added or removed checkpoint weights')
    if any(not torch.equal(value, new[key]) for key, value in original.items()):
        raise ValueError('Navigation input changed an original brain weight')
