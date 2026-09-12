"""Select named neuron populations from the male-CNS annotation table.

A population spec is ``{name: [selector, selector, ...]}``; selectors are OR-ed, and the keys
inside a selector are AND-ed. Selector keys (all optional):

- ``types``: list of exact ``type`` values
- ``type_regex``: regular expression matched against ``type`` (use anchors yourself)
- ``superclass``, ``class``, ``subclass``, ``entry_nerve``, ``exit_nerve``, ``soma_side``,
  ``soma_neuromere``, ``status``: lists of exact values

The default spec below wires drone senses onto the fly's own sensory populations and reads the
drone commands out of the fly's wing/haltere motor neurons.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

COLUMN = {
    'superclass': 'superclass',
    'class': 'class',
    'subclass': 'subclass',
    'entry_nerve': 'entryNerve',
    'exit_nerve': 'exitNerve',
    'soma_side': 'somaSide',
    'soma_neuromere': 'somaNeuromere',
    'status': 'status',
}

# Cell-type names and annotations in male-cns v1.0 (MANC nomenclature for the nerve cord;
# hemibrain/FlyWire/optic-lobe nomenclature for the brain). Verified against the v1.0 annotation table.
DEFAULT_FLIGHT_POPULATIONS: dict[str, list[dict]] = {
    # --- senses (drone telemetry is written into these) ---
    'haltere': [  # angular-rate sense: every afferent entering through the haltere (dorsal metathoracic) nerve
        {'entry_nerve': ['DMetaN'], 'superclass': ['vnc_sensory', 'sensory_ascending']},
    ],
    'wing_cs': [  # wing campaniform sensilla (wing load / aerodynamic force), anterior dorsal mesothoracic nerve
        {'subclass': ['campaniform sensilla'], 'entry_nerve': ['ADMN']},
    ],
    'lptc': [  # lobula-plate tangential cells: wide-field optic flow (self-rotation and translation)
        {'type_regex': r'^(HSN|HSE|HSS|HST|VS|VSm|VST1|VST2|H1|H2|VCH|DCH)$'},
    ],
    'ocelli': [  # ocellar pathway: photoreceptor afferents of the ocellar nerve + OCG interneurons (horizon/attitude)
        {'entry_nerve': ['ON']},
        {'type_regex': r'^OCG\d'},
    ],
    'jo': [  # Johnston's organ wind/gravity neurons (antennal airflow)
        {'subclass': ['wind_gravity']},
    ],
    'compass': [  # head-direction cells of the central complex
        {'types': ['EPG', 'EPGt']},
    ],
    'goal': [  # goal-direction cells of the fan-shaped body
        {'type_regex': r'^(PFL3|FC2A|FC2B|FC2C)$'},
    ],
    # --- outputs (drone sticks are read out of these) ---
    'wing_mn': [  # wing (wm) and haltere (hm) muscle motor neurons
        {'superclass': ['vnc_motor'], 'subclass': ['wm', 'hm']},
    ],
    # --- circuits kept whole ---
    'descending': [{'superclass': ['descending_neuron']}],
    'cx': [{'class': ['CX']}],
}


def selector_mask(ann: pd.DataFrame, sel: dict) -> np.ndarray:
    m = np.ones(len(ann), dtype=bool)
    t = ann['type'].astype('string')
    if 'types' in sel:
        m &= t.isin(sel['types']).fillna(False).to_numpy()
    if 'type_regex' in sel:
        pat = re.compile(sel['type_regex'])
        m &= t.map(lambda s: bool(pat.search(s)) if isinstance(s, str) else False).fillna(False).to_numpy()
    for key, col in COLUMN.items():
        if key in sel:
            m &= ann[col].astype('string').isin(sel[key]).fillna(False).to_numpy()
    return m


def select_populations(ann: pd.DataFrame, spec: dict[str, list[dict]]) -> dict[str, np.ndarray]:
    """Return name -> boolean mask over ``ann`` rows."""
    out = {}
    for name, selectors in spec.items():
        m = np.zeros(len(ann), dtype=bool)
        for sel in selectors:
            m |= selector_mask(ann, sel)
        out[name] = m
    return out


def describe_populations(ann: pd.DataFrame, masks: dict[str, np.ndarray], top: int = 8) -> str:
    lines = []
    for name, m in masks.items():
        n = int(m.sum())
        types = ann.loc[m, 'type'].value_counts().head(top)
        summary = ', '.join(f'{t} x{c}' for t, c in types.items())
        lines.append(f'{name:12s} {n:6d} neurons  [{summary}]')
    return '\n'.join(lines)
