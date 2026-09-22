import copy
import json
from pathlib import Path

import pytest

from haltere.liftoff.course_pool import clone_pair, generate_loop, install_pair, read_xml, validate_pair


def fixture_pair(tmp_path):
    track = '''<Track><localID><str>source</str><version>3</version><type>TRACK</type></localID>
    <managedID><str>123</str></managedID><name>Workshop author</name><description/>
    <environment>TheDrawingBoard</environment><blueprints>
    <TrackBlueprint><instanceID>1</instanceID><itemID>Gate</itemID>
    <position><x>0</x><y>2</y><z>4</z></position><rotation><x>0</x><y>90</y><z>0</z></rotation></TrackBlueprint>
    </blueprints></Track>'''
    race = '''<Race><localID><str>race</str><version>1</version><type>RACE</type></localID>
    <managedID><str>456</str></managedID><name>Race author</name><description/>
    <dependencies><dependency><str>source</str><version>3</version><type>TRACK</type></dependency></dependencies>
    <requiredLaps>3</requiredLaps><spawnPointID>-1</spawnPointID><checkPointPassages>
    <RaceCheckpointPassage><uniqueId>a</uniqueId><checkPointID>1</checkPointID><passageType>Start</passageType>
    <directionality>RightToLeft</directionality><nextPassageIDs><string>b</string></nextPassageIDs></RaceCheckpointPassage>
    <RaceCheckpointPassage><uniqueId>b</uniqueId><checkPointID>1</checkPointID><passageType>Finish</passageType>
    <directionality>LeftToRight</directionality><nextPassageIDs/></RaceCheckpointPassage>
    </checkPointPassages></Race>'''
    a, b = tmp_path/'source.track', tmp_path/'source.race'
    # Reproduce the incorrect encoding declaration in installed Workshop assets.
    a.write_text('<?xml version="1.0" encoding="utf-16"?>'+track, encoding='utf-8')
    b.write_text(race, encoding='utf-16')
    return a, b


def test_clone_preserves_originals_directions_and_links_with_new_ids(tmp_path):
    a, b = fixture_pair(tmp_path)
    originals = [p.read_bytes() for p in (a, b)]
    out = tmp_path/'bundle'
    m = clone_pair(a, b, out, 'Load check', (2., 0., 0.), 1)
    assert [p.read_bytes() for p in (a, b)] == originals
    track = read_xml(next(out.rglob('*.track')))
    race = read_xml(next(out.rglob('*.race')))
    assert track.findtext('localID/str') != 'source'
    assert track.find('managedID') is None and race.find('managedID') is None
    assert track.findtext('./blueprints/TrackBlueprint/position/x') == '2'
    assert race.findtext('./dependencies/dependency/str') == track.findtext('localID/str')
    assert [p.text for p in race.findall('.//directionality')] == ['RightToLeft', 'LeftToRight']
    assert validate_pair(track, race)['laps'] == 1
    assert not m['validation']['game_load_verified']
    assert m['pool'] == 'development' and not m['runtime_geometry_allowed']
    targets = install_pair(out, tmp_path/'local')
    assert len(targets) == 2 and all(Path(p).is_file() for p in targets)
    with pytest.raises(FileExistsError):
        install_pair(out, tmp_path/'local')
    with pytest.raises(FileExistsError):
        clone_pair(a, b, out, 'Duplicate')


def test_validation_rejects_missing_objects_links_cycles_and_wrong_version(tmp_path):
    a, b = fixture_pair(tmp_path)
    track, race = read_xml(a), read_xml(b)
    for path, value, message in [('.//checkPointID','99','checkpoint object'),
                                 ('.//nextPassageIDs/string','absent','missing passage'),
                                 ('.//nextPassageIDs/string','a','cycle'),
                                 ('./dependencies/dependency/version','2','exact local track')]:
        invalid = copy.deepcopy(race)
        invalid.find(path).text = value
        with pytest.raises(ValueError, match=message):
            validate_pair(track, invalid)


def test_install_rejects_tampering_and_path_escape_before_writing(tmp_path):
    a, b = fixture_pair(tmp_path)
    out, local = tmp_path/'bundle', tmp_path/'local'
    m = clone_pair(a, b, out, 'Load check')
    source = out/next(iter(m['files']))
    source.write_bytes(source.read_bytes()+b' ')
    with pytest.raises(ValueError, match='hash'):
        install_pair(out, local)
    assert not local.exists()
    m['files'] = {'../outside.track': 'fake'}
    (out/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='inside'):
        install_pair(out, local)
    assert not local.exists()


def test_generated_courses_are_reproducible_and_close_directional_loop(tmp_path):
    a = generate_loop(tmp_path/'a', 19, gates=8, obstacles=6)
    b = generate_loop(tmp_path/'b', 19, gates=8, obstacles=6)
    c = generate_loop(tmp_path/'c', 20, gates=8, obstacles=6)
    assert a == b
    assert a['files'] != c['files']
    track = read_xml(next((tmp_path/'a').rglob('*.track')))
    race = read_xml(next((tmp_path/'a').rglob('*.race')))
    assert validate_pair(track, race)['objects'] == 15
    passages = race.findall('./checkPointPassages/RaceCheckpointPassage')
    assert len(passages) == 9
    assert passages[0].findtext('checkPointID') == passages[-1].findtext('checkPointID')
    assert {p.findtext('directionality') for p in passages} == {'RightToLeft'}
    assert float(track.find('./blueprints/TrackBlueprint/position/y').text) == 5.
    assert not a['validation']['game_load_verified']
    assert a['pool'] == 'development' and not a['runtime_geometry_allowed']
    assert len(install_pair(tmp_path/'a', tmp_path/'local')) == 2


def test_generated_courses_reject_degenerate_layout_before_writing(tmp_path):
    with pytest.raises(ValueError, match='gates'):
        generate_loop(tmp_path/'bad', 0, gates=2)
    assert not (tmp_path/'bad').exists()


def test_known_asset_blueprint_type_is_checked_before_game_loading(tmp_path):
    generate_loop(tmp_path/'bundle', 1, obstacles=1)
    track = read_xml(next((tmp_path/'bundle').rglob('*.track')))
    race = read_xml(next((tmp_path/'bundle').rglob('*.race')))
    wall = next(o for o in track.findall('./blueprints/TrackBlueprint')
                if o.findtext('itemID') == 'DrawingBoardWall5mx5m05')
    wall.attrib.clear()
    with pytest.raises(ValueError, match='requires TrackBlueprintFlag'):
        validate_pair(track, race)
