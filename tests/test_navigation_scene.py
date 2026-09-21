import copy

import pytest
import torch

from haltere.brain.model import BrainConfig
from haltere.config import dataclass_to_dict
from haltere.train.navigation_scene import (
    ADDED_PARAMETERS, navigation_config, validate_navigation_retention,
)


@pytest.mark.parametrize('mutation', ['weight', 'configuration', 'extra_parameter'])
def test_export_retention_rejects_undeclared_changes(mutation):
    cfg = BrainConfig(sensory={'retina':'lptc'})
    parent = dict(config=dict(brain=dataclass_to_dict(cfg),quad={'mass':1.}),
                  graph='fixed',channels={'retina':12},model={'bias':torch.ones(3)})
    candidate = copy.deepcopy(parent)
    candidate['config']['brain'] = dataclass_to_dict(navigation_config(cfg))
    candidate['model'].update({name:torch.ones(2) for name in ADDED_PARAMETERS})
    validate_navigation_retention(candidate,parent)
    if mutation == 'weight':candidate['model']['bias'][0] += 1
    elif mutation == 'configuration':candidate['config']['quad']['mass'] = 2.
    else:candidate['model']['readout.unexpected'] = torch.ones(1)
    with pytest.raises(ValueError):
        validate_navigation_retention(candidate,parent)
