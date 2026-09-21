"""Record the exact causal sensory samples used during an autonomous flight.

Passive retinal collection does not enable image currents in a gate-only brain.
Route geometry and corrective labels are deliberately absent from this module.
"""
import copy
from pathlib import Path

import numpy as np


def replay_camera_sensor(sensor, enabled):
    result = copy.deepcopy(sensor)
    if enabled:
        if not result or result.get('retina_mode') != 'gatenet_scene_v1':
            raise ValueError('Replay collection requires a frozen scene projection')
        projection = result.get('scene_projection', {})
        if projection.get('detector_sha256') != result['sha256']:
            raise ValueError('Replay projection and detector differ')
        result['raw_retina_active'] = True  # camera computation only, not controller input
    return result


class NeuralReplay:
    def __init__(self, path, channels, capacity):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.n = 0
        self.capacity = capacity
        self.channels = dict(channels)
        shapes = {**channels, 'action': 4, 'position': 3, 'quaternion': 4,
                  'clock': 4, 'fresh': 1}
        self.arrays = {k: np.empty((capacity, d), np.float32) for k, d in shapes.items()}
        self.arrays['clock'] = np.empty((capacity, 4), np.float64)

    def append(self, observation, retina, action, position, quaternion,
               timestamp, observed_at, captured_at, fresh):
        if self.n >= self.capacity:
            raise RuntimeError('Replay capacity exceeded')
        # Copy immediately: tensors and shared camera slots are reused by the runner.
        for k in self.channels:
            value = retina if k == 'retina' else observation[k]
            self.arrays[k][self.n] = value.detach().cpu().numpy().reshape(-1)
        for k, v in dict(action=action, position=position, quaternion=quaternion,
                         clock=[timestamp, observed_at, captured_at, observed_at-captured_at],
                         fresh=[fresh]).items():
            self.arrays[k][self.n] = v
        self.n += 1

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('xb') as f:
            np.savez_compressed(f, **{k: v[:self.n] for k, v in self.arrays.items()})
        return dict(path=str(self.path), ticks=self.n,
                    timing='Exact controller observations and original camera presentation timestamps',
                    passive_retina=True, route_labels_present=False)
