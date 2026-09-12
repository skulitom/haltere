"""Graph building on a tiny synthetic 'male-CNS' with the real file layout."""
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pytest

from haltere.connectome import sources
from haltere.connectome.graph import BrainGraph, GraphConfig, build_graph


def make_dataset(root):
    d = sources.raw_dir(root, 'v1.0')
    d.mkdir(parents=True)
    # 12 neurons: 3 haltere afferents -> 3 interneurons -> 2 motor neurons, plus extras
    ids = np.arange(1, 13) * 100
    ann = pd.DataFrame({
        'bodyId': ids,
        'type': ['SApp01', 'SApp01', 'SNpp14', 'IN1', 'IN2', 'IN3', 'b1 MN', 'DLMn c-f', 'EPG', 'DNa02', 'Mi1', 'frag'],
        'instance': [f't_{i}' for i in ids],
        'class': [None] * 8 + ['CX', None, 'visual', None],
        'superclass': ['vnc_sensory', 'vnc_sensory', 'sensory_ascending', 'vnc_intrinsic', 'vnc_intrinsic',
                       'vnc_intrinsic', 'vnc_motor', 'vnc_motor', 'cb_intrinsic', 'descending_neuron', 'ol_intrinsic', None],
        'subclass': ['haltere', 'campaniform sensilla', 'haltere', None, None, None, 'wm', 'wm', None, 'xl', None, None],
        'somaSide': ['L'] * 12, 'rootSide': [None] * 12, 'somaNeuromere': ['T3'] * 3 + ['T2'] * 5 + [None] * 4,
        'entryNerve': ['DMetaN', 'DMetaN', 'DMetaN'] + [None] * 9, 'exitNerve': [None] * 6 + ['ADMN', 'PDMNa'] + [None] * 4,
        'status': ['Traced'] * 11 + ['Orphan'], 'statusLabel': ['Roughly traced'] * 12,
        'flywireType': [None] * 12, 'hemibrainType': [None] * 12, 'mancType': [None] * 12,
        'group': ids.astype(float), 'dimorphism': [None] * 12,
    })
    feather.write_feather(ann, sources.file_path('annotations', root))
    nt = pd.DataFrame({
        'body': ids,
        'cell_type': ann['type'],
        'total_nt_predictions': [100] * 12,
        'predicted_nt_confidence': [0.9] * 12,
        'predicted_nt': ['acetylcholine'] * 3 + ['gaba', 'unclear', 'glutamate', 'acetylcholine', 'acetylcholine',
                                                'acetylcholine', 'acetylcholine', 'acetylcholine', 'unclear'],
        'ground_truth': [None] * 12,
        'celltype_total_nt_predictions': [100] * 12,
        'celltype_predicted_nt': ['acetylcholine'] * 4 + ['dopamine'] + ['glutamate'] + ['acetylcholine'] * 5 + ['unclear'],
        'celltype_predicted_nt_confidence': [0.9] * 12,
        'consensus_nt': ['acetylcholine'] * 3 + ['gaba', 'unclear', 'glutamate', 'acetylcholine', 'acetylcholine',
                                                'acetylcholine', 'acetylcholine', 'acetylcholine', 'unclear'],
    })
    feather.write_feather(nt, sources.file_path('neurotransmitters', root))
    edges = [(100, 400, 10), (200, 400, 5), (300, 500, 8), (400, 600, 12), (500, 700, 6), (600, 800, 9),
             (600, 700, 2), (900, 1000, 7), (1000, 600, 4), (1100, 400, 20), (1200, 400, 30), (400, 400, 3)]
    w = pd.DataFrame(edges, columns=['body_pre', 'body_post', 'weight'])
    w['type_pre'] = ''
    w['type_post'] = ''
    feather.write_feather(w, sources.file_path('weights', root))


@pytest.fixture
def dataset(tmp_path):
    make_dataset(tmp_path)
    return tmp_path


def test_build_graph_selects_paths_and_signs(dataset):
    cfg = GraphConfig(min_synapses=3, hops_forward=2, hops_backward=2, max_neurons=100)
    g = build_graph(cfg, root=dataset, verbose=False)
    types = set(g.nodes['type'])
    # haltere afferents, interneurons on the path, motor neurons, and the always-included CX/DN neurons
    assert {'SApp01', 'SNpp14', 'IN1', 'IN2', 'b1 MN', 'DLMn c-f', 'EPG', 'DNa02'} <= types
    assert 'Mi1' not in types          # ol_intrinsic excluded
    assert 'frag' not in types         # not Traced
    assert 'IN3' in types              # SApp01 -> IN1 -> IN3 -> DLMn is within 2 forward + 2 backward hops
    # with a single hop each way only IN2 (SApp -> IN2 -> b1 MN) survives among the interneurons
    g1 = build_graph(GraphConfig(min_synapses=3, hops_forward=1, hops_backward=1, max_neurons=100), root=dataset,
                     verbose=False)
    t1 = set(g1.nodes['type'])
    assert 'IN2' in t1 and 'IN1' not in t1 and 'IN3' not in t1
    # weak edge (2 synapses) dropped
    assert g.E == sum(1 for (a, b, w) in [(400, 600, 12), (500, 700, 6), (600, 800, 9), (100, 400, 10), (200, 400, 5),
                                          (300, 500, 8), (900, 1000, 7), (1000, 600, 4), (400, 400, 3)])
    node = g.nodes.set_index('type')
    assert node.loc['IN1', 'sign'] == -1          # gaba
    assert node.loc['IN2', 'sign'] == 0           # body unclear -> celltype dopamine -> unknown sign
    assert node.loc['IN2', 'nt'] == 'dopamine'
    assert node.loc['SApp01', 'sign'].iloc[0] == 1
    assert len(g.population('haltere')) == 3
    assert len(g.population('wing_mn')) == 2
    assert len(g.population('cx')) == 1


def test_graph_roundtrip(dataset, tmp_path):
    g = build_graph(GraphConfig(min_synapses=3, max_neurons=100), root=dataset, verbose=False)
    g.save(tmp_path / 'built' / 'g')
    h = BrainGraph.load(tmp_path / 'built' / 'g')
    assert h.N == g.N and h.E == g.E
    assert np.array_equal(h.pre, g.pre) and np.array_equal(h.post, g.post)
    assert set(h.populations) == set(g.populations)
    assert h.meta['n_nodes'] == g.N


def test_max_neurons_cap(dataset):
    g = build_graph(GraphConfig(min_synapses=3, max_neurons=5), root=dataset, verbose=False)
    # core (haltere 3 + motor 2 + cx 1 + dn 1 = 7) is always kept, cap applies beyond the core
    assert g.N == 7
