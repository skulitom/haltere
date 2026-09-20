"""Replay imports must not turn corrupt timing or unknown controls into training data."""
import base64
import csv
import gzip
import json
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from haltere.liftoff.bot_routes import _register_bundle, read_register
from haltere.liftoff.frames import quat_wxyz_to_mat, quat_xyzw_to_mat, M
from haltere.liftoff.replay import import_replay, read_replay


def recording(states=None, layout='POSITION_STICKS', version='1.2.11'):
    if states is None:
        states = np.array([[10, 20, 30, 0, 0, 0, 1, .4, .1, .2, .3, 100],
                           [12, 23, 34, 0, 2**-.5, 0, 2**-.5, .5, .2, .3, .4, 100.1],
                           [12, 23, 34, 0, 2**-.5, 0, 2**-.5, .5, .2, .3, .4, 100.1],
                           [14, 26, 38, 0, 1, 0, 0, .6, .3, .4, .5, 100.2]])
    root = ET.Element('LocalGhostStatesRecording')
    for key, text in [('gameVersion', version), ('stateLayout', layout), ('environment', 'PineValley'),
                      ('totalTime', '.1'), ('statesByte', base64.b64encode(np.asarray(states, dtype='<f4').tobytes()).decode())]:
        ET.SubElement(root, key).text = text
    starts = ET.SubElement(root, 'lapStartIndices')
    for i in (0, 2):
        ET.SubElement(starts, 'int').text = str(i)
    return ET.tostring(root)


def test_import_keeps_world_axes_origin_inputs_and_boundary_indices(tmp_path):
    source = tmp_path / 'source.xml'
    source.write_bytes(recording())
    out = tmp_path / 'route'
    report = import_replay(source, out)
    rows = list(csv.DictReader((out / 'trajectory.csv').open()))
    assert [int(r['source_index']) for r in rows] == [0, 2, 3]
    assert [int(r['segment']) for r in rows] == [0, 1, 1]
    np.testing.assert_allclose([float(rows[1][k]) for k in ['px', 'py', 'pz']], [4, -2, 3])
    np.testing.assert_allclose([float(rows[1][f'recorded_input_{i}']) for i in range(4)], [.5, .2, .3, .4])
    q = np.array([float(rows[1][k]) for k in ['qw', 'qx', 'qy', 'qz']])
    np.testing.assert_allclose(quat_wxyz_to_mat(q), M @ quat_xyzw_to_mat([0, 2**-.5, 0, 2**-.5]) @ M.T, atol=1e-6)
    assert report['reported_total_time_s'] == .1
    assert report['duration_s'] == pytest.approx(.2, abs=1e-5)
    assert report['duplicate_boundary_rows'] == 1
    assert report['input_channels_vary'] and not report['action_labels_verified']
    assert (out / 'source.recording').read_bytes() == source.read_bytes()
    assert json.loads((out / 'report.json').read_text())['source_sha256'] == report['source_sha256']
    with pytest.raises(FileExistsError):
        import_replay(source, out)


def test_gzip_and_positional_have_no_invented_actions(tmp_path):
    states, _ = read_replay(recording())
    raw = recording(states[:, [0, 1, 2, 3, 4, 5, 6, 11]], layout='POSITIONAL')
    source = tmp_path / 'flight.gz'
    source.write_bytes(gzip.compress(raw))
    report = import_replay(source, tmp_path / 'out')
    assert not report['input_channels_present']
    assert 'recorded_input' not in (tmp_path / 'out' / 'trajectory.csv').read_text().splitlines()[0]


@pytest.mark.parametrize('problem', ['reversed_time', 'conflicting_duplicate', 'nan', 'bad_quat', 'truncated', 'legacy', 'layout', 'lap_index'])
def test_invalid_replays_fail_before_output(tmp_path, problem):
    states, _ = read_replay(recording())
    if problem == 'reversed_time':
        states[-1, -1] = 99
    elif problem == 'conflicting_duplicate':
        states[2, 0] += 1
    elif problem == 'nan':
        states[0, 0] = np.nan
    elif problem == 'bad_quat':
        states[0, 6] = 0
    root = ET.fromstring(recording(states))
    if problem == 'truncated':
        root.find('statesByte').text = base64.b64encode(b'12345').decode()
    elif problem == 'legacy':
        root.find('gameVersion').text = '0.13.0'
    elif problem == 'layout':
        root.find('stateLayout').text = 'POSITION_STICKS_BUMPERS'
    elif problem == 'lap_index':
        root.find('lapStartIndices/int').text = '100'
    source = tmp_path / 'source.xml'
    source.write_bytes(ET.tostring(root))
    with pytest.raises(ValueError):
        import_replay(source, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_constant_sticks_do_not_imply_usable_actions():
    states, _ = read_replay(recording())
    states[:, 7:11] = 0
    # Absence of control movement is valid pose data, not corruption.
    parsed, _ = read_replay(recording(states))
    assert not parsed[:, 7:11].any()


def test_register_wire_layout():
    def string(s):
        b = s.encode()
        return struct.pack('<i', len(b)) + b + b'\0' * (-len(b) % 4)
    content = string('uuid') + struct.pack('<ii', 1, 2)
    raw = (bytes(28) + string('RecordingRegister') + struct.pack('<i', 1)
           + string('key') + string('name') + string('PineValley')
           + struct.pack('<ifi', 2, 42., 0) + content * 2)
    row, = read_register(raw)
    assert row['race_time_s'] == 42. and row['race_id']['str'] == 'uuid'
    with pytest.raises(ValueError, match='trailing'):
        read_register(raw + b'\0')


def test_catalog_resolves_register_dependency_and_refuses_escape(tmp_path):
    key = b'RecordingRegister'
    catalog = {
        'm_KeyDataString': base64.b64encode(b'\0' + struct.pack('<i', len(key)) + key).decode(),
        'm_BucketDataString': base64.b64encode(struct.pack('<7i', 2, 0, 1, 0, 0, 1, 1)).decode(),
        'm_EntryDataString': base64.b64encode(struct.pack('<15i', 2, 0, 1, 1, 0, -1, 0, 0,
                                                         1, 0, -1, 0, -1, 1, 0)).decode(),
        'm_InternalIds': ['register-guid', '0#test.bundle'],
        'm_InternalIdPrefixes': ['{UnityEngine.AddressableAssets.Addressables.RuntimePath}/StandaloneWindows64/'],
    }
    assert _register_bundle(catalog, tmp_path) == tmp_path / 'StandaloneWindows64' / 'test.bundle'
    catalog['m_InternalIds'][1] = '0#../../escape.bundle'
    with pytest.raises(ValueError, match='outside'):
        _register_bundle(catalog, tmp_path)
