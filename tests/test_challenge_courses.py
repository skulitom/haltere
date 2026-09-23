import json

from haltere.liftoff.challenge_courses import KINDS, generate_sections
from haltere.liftoff.course_pool import install_pair, read_xml, validate_pair


def test_sections_have_physical_frames_and_distinct_reproducible_obstacles(tmp_path):
    a = generate_sections(tmp_path/'a', 123)
    b = generate_sections(tmp_path/'b', 123)
    c = generate_sections(tmp_path/'c', 124)
    assert a == b and a['files'] != c['files']
    assert a['family'] == c['family']  # another seed is not a held-out family
    geometry = json.loads((tmp_path/'a/offline-geometry.json').read_text())
    assert [s['kind'] for s in geometry['sections']] == list(KINDS)
    assert len(geometry['checkpoints']) == 12
    assert all(p['half_width'] == 2.5 for p in geometry['checkpoints'])
    assert any(o['role'] == 'gate_frame' for o in geometry['primitives'])
    assert geometry['unknown_geometry']  # flags cannot silently acquire exact depth labels
    assert not geometry['render_alignment_verified']
    assert not geometry['runtime_geometry_allowed']
    track = read_xml(next((tmp_path/'a').rglob('*.track')))
    race = read_xml(next((tmp_path/'a').rglob('*.race')))
    assert validate_pair(track, race)['passages'] == 12
    assert len(install_pair(tmp_path/'a', tmp_path/'installed')) == 2


def test_descent_is_below_the_raised_start_and_wall_blocks_marker_line(tmp_path):
    generate_sections(tmp_path/'a', 4)
    g = json.loads((tmp_path/'a/offline-geometry.json').read_text())
    checkpoints = {p['instance_id']: p for p in g['checkpoints']}
    descent = next(s for s in g['sections'] if s['kind'] == 'descent')
    assert checkpoints[descent['exit_checkpoint']]['center'][1] < 11.616
    wall = next(p for p in g['primitives'] if p['item_id'] == 'DrawingBoardWall5mx5m05')
    assert wall['size'] == [10., 5., 1.]  # measured asset, despite its misleading name


def test_box_calibration_is_a_distinct_scene_in_the_same_development_family(tmp_path):
    original = generate_sections(tmp_path/'original', 4)
    calibration = generate_sections(tmp_path/'calibration', 4, box_calibration=True)
    g = json.loads((tmp_path/'calibration/offline-geometry.json').read_text())
    assert calibration['track_id'] != original['track_id']
    assert calibration['family'] == original['family']
    assert not g['unknown_geometry']
    assert g['sections'][0]['kind'] == 'box_near_line'
    assert all(s['kind'] != 'flag' for s in g['sections'])
    assert not g['render_alignment_verified']
