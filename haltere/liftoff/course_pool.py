"""Offline course preparation. Course geometry must never be a runtime pilot input.

Workshop originals are read-only. Clones receive new local IDs and retain source
attribution; installing adds only new folders in Liftoff's local content store.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import uuid
import xml.etree.ElementTree as ET


ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
XSI_TYPE = '{http://www.w3.org/2001/XMLSchema-instance}type'
BUILTIN_BLUEPRINT_TYPES = {
    'CheckpointBox10mX10m01': 'TrackBlueprintFlag',
    'DrawingBoardWall5mx5m05': 'TrackBlueprintFlag',
    'SpawnPointSingle02': 'TrackBlueprintSpawnpoint',
}


def read_xml(path):
    raw = Path(path).read_bytes()
    if raw.startswith((b'\xff\xfe', b'\xfe\xff')):
        text = raw.decode('utf-16')
    elif raw.startswith(b'<\x00'):
        text = raw.decode('utf-16-le')
    elif raw.startswith(b'\x00<'):
        text = raw.decode('utf-16-be')
    else:
        text = raw.decode('utf-8-sig')
    # Some installed Workshop files declare UTF-16 but contain UTF-8 bytes.
    text = re.sub(r'^\s*<\?xml[^?]*\?>', '', text, count=1)
    return ET.fromstring(text)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_pair(track, race):
    if track.tag != 'Track' or race.tag != 'Race':
        raise ValueError('Expected a Track and a Race')
    track_id = track.findtext('localID/str')
    deps = race.findall('./dependencies/dependency')
    if not any(d.findtext('str') == track_id and d.findtext('type') == 'TRACK'
               and d.findtext('version') == track.findtext('localID/version') for d in deps):
        raise ValueError('Race must depend on this exact local track version')
    objects = track.findall('./blueprints/TrackBlueprint')
    ids = [x.findtext('instanceID') for x in objects]
    if not ids or None in ids or len(set(ids)) != len(ids):
        raise ValueError('Track object IDs must be present and unique')
    for obj in objects:
        expected = BUILTIN_BLUEPRINT_TYPES.get(obj.findtext('itemID'))
        if expected and obj.get(XSI_TYPE) != expected:
            raise ValueError(f'Asset {obj.findtext("itemID")} requires {expected}')
        for group in ('position', 'rotation'):
            values = [float(obj.findtext(group+'/'+axis, 'nan')) for axis in 'xyz']
            if not all(math.isfinite(v) for v in values):
                raise ValueError('Track transforms must be finite')
    passages = race.findall('./checkPointPassages/RaceCheckpointPassage')
    by_id = {p.findtext('uniqueId'): p for p in passages}
    if not passages or None in by_id or len(by_id) != len(passages):
        raise ValueError('Race passage IDs must be present and unique')
    starts = [p for p in passages if p.findtext('passageType') == 'Start']
    finishes = [p for p in passages if p.findtext('passageType') == 'Finish']
    if len(starts) != 1 or not finishes:
        raise ValueError('Expected one start and at least one finish')
    for p in passages:
        if p.findtext('checkPointID') not in ids:
            raise ValueError('Race references a missing checkpoint object')
        if not p.findtext('directionality'):
            raise ValueError('Checkpoint directionality must be explicit')
        following = [x.text for x in p.findall('./nextPassageIDs/string')]
        if any(x not in by_id for x in following):
            raise ValueError('Race references a missing passage')
        if p.findtext('passageType') == 'Finish' and following:
            raise ValueError('Finish must terminate its passage chain')
        if p.findtext('passageType') != 'Finish' and not following:
            raise ValueError('Non-finish passage must have a successor')
    visiting, visited = set(), set()

    def visit(key):
        if key in visiting:
            raise ValueError('Passage graph has a cycle; laps belong in requiredLaps')
        if key in visited:
            return
        visiting.add(key)
        for link in by_id[key].findall('./nextPassageIDs/string'):
            visit(link.text)
        visiting.remove(key)
        visited.add(key)

    visit(starts[0].findtext('uniqueId'))
    if len(visited) != len(passages):
        raise ValueError('Passage graph contains unreachable entries')
    spawn = race.findtext('spawnPointID', '-1')
    if spawn != '-1' and spawn not in ids:
        raise ValueError('Race references a missing spawn object')
    laps = int(race.findtext('requiredLaps', '0'))
    if laps < 1:
        raise ValueError('Race must require at least one lap')
    return dict(environment=track.findtext('environment'), objects=len(objects),
                passages=len(passages), laps=laps, directionality_preserved=True,
                game_load_verified=False, geometry_clearance_verified=False)


def clone_pair(track_path, race_path, out, name, translation=(0., 0., 0.), laps=1):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if len(translation) != 3 or not all(math.isfinite(v) for v in translation):
        raise ValueError('Translation must contain three finite values')
    if not name.strip() or laps < 1:
        raise ValueError('Provide a nonempty name and positive lap count')
    track, race = read_xml(track_path), read_xml(race_path)
    validate_pair(track, race)
    source = dict(track=dict(path=str(Path(track_path).resolve()), sha256=digest(track_path),
                            name=track.findtext('name')),
                  race=dict(path=str(Path(race_path).resolve()), sha256=digest(race_path),
                           name=race.findtext('name')))
    old_id = track.findtext('localID/str')
    track_id, race_id = str(uuid.uuid4()), str(uuid.uuid4())
    for root, new_id, label in [(track, track_id, name), (race, race_id, name+' race')]:
        root.find('localID/str').text = new_id
        root.find('localID/version').text = '1'
        for managed in root.findall('managedID'):
            root.remove(managed)
        root.find('name').text = label
        description = root.find('description')
        if description is None:
            description = ET.SubElement(root, 'description')
        original_description = description.text or ''
        description.text = ('Private Haltere development copy of '+source['track']['name']+
                            '. Original Workshop authorship retained; not a published original course.\n'+
                            original_description)
    for d in race.findall('./dependencies/dependency'):
        if d.findtext('str') == old_id and d.findtext('type') == 'TRACK':
            d.find('str').text, d.find('version').text = track_id, '1'
    passage_ids = {p.findtext('uniqueId'): str(uuid.uuid4())
                   for p in race.findall('./checkPointPassages/RaceCheckpointPassage')}
    for p in race.findall('./checkPointPassages/RaceCheckpointPassage'):
        p.find('uniqueId').text = passage_ids[p.findtext('uniqueId')]
        for link in p.findall('./nextPassageIDs/string'):
            link.text = passage_ids[link.text]
    race.find('requiredLaps').text = str(laps)
    for obj in track.findall('./blueprints/TrackBlueprint'):
        for axis, delta in zip('xyz', translation):
            node = obj.find('position/'+axis)
            node.text = format(float(node.text)+delta, '.9g')
    report = validate_pair(track, race)
    manifest = dict(schema=1, purpose='offline local course load check', pool='development',
                    runtime_geometry_allowed=False, source=source, translation_xyz=list(translation),
                    track_id=track_id, race_id=race_id, validation=report, files={})
    out.mkdir(parents=True)
    for kind, root, uid, ext in [('Tracks', track, track_id, 'track'), ('Races', race, race_id, 'race')]:
        relative = Path(kind)/uid/(uid+'_0001.'+ext)
        target = out/relative
        target.parent.mkdir(parents=True)
        ET.indent(root)
        ET.ElementTree(root).write(target, encoding='utf-8', xml_declaration=True)
        manifest['files'][relative.as_posix()] = digest(target)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def install_pair(bundle, local_root):
    bundle, local_root = Path(bundle).resolve(), Path(local_root).resolve()
    manifest = json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    files = []
    for relative, expected in manifest['files'].items():
        source, target = (bundle/relative).resolve(), (local_root/relative).resolve()
        if not source.is_relative_to(bundle) or not target.is_relative_to(local_root):
            raise ValueError('Bundle paths must remain inside their roots')
        if Path(relative).parts[0] not in ('Tracks', 'Races'):
            raise ValueError('Only track/race content can be installed')
        if target.exists() or target.parent.exists():
            raise FileExistsError(target.parent)
        if digest(source) != expected:
            raise ValueError('Bundle hash does not match its manifest')
        files.append((source, target))
    if len(files) != 2:
        raise ValueError('Expected exactly one track and one race')
    tracks = [s for s, _ in files if s.suffix == '.track']
    races = [s for s, _ in files if s.suffix == '.race']
    if len(tracks) != 1 or len(races) != 1:
        raise ValueError('Expected exactly one track and one race')
    track, race = tracks[0], races[0]
    validate_pair(read_xml(track), read_xml(race))
    for source, target in files:
        target.parent.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(source, target)
    return [str(target) for _, target in files]


def generate_loop(out, seed, gates=10, obstacles=12):
    """Generate a development course from built-in asset IDs, without route data.

    XML coordinates are Unity x/right, y/up, z/forward. This first generator
    deliberately targets the empty Drawing Board environment. Other environments
    need an independently checked free-space envelope before placing objects.
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if not 6 <= gates <= 20 or not 0 <= obstacles <= 100:
        raise ValueError('Use 6..20 gates and 0..100 obstacles')
    rng = random.Random(seed)
    recipe = dict(generator='ellipse-v2', seed=seed, gates=gates, obstacles=obstacles)
    identity = json.dumps(recipe, sort_keys=True)
    uid = lambda label: str(uuid.uuid5(uuid.NAMESPACE_URL, 'haltere/'+identity+'/'+label))
    track_id, race_id = uid('track'), uid('race')
    name = f'Haltere loop v2 {seed}'
    track, race = ET.Element('Track'), ET.Element('Race')

    def field(root, tag, value):
        ET.SubElement(root, tag).text = str(value)

    for root, local_id, kind in [(track, track_id, 'TRACK'), (race, race_id, 'RACE')]:
        field(root, 'gameVersion', '1.7.6')
        local = ET.SubElement(root, 'localID')
        for k, v in [('str', local_id), ('version', 1), ('type', kind)]:
            field(local, k, v)
        field(root, 'name', name + (' race' if kind == 'RACE' else ''))
        field(root, 'description', 'Procedural Haltere development course; flight validation pending.')
    ET.SubElement(track, 'dependencies')
    deps = ET.SubElement(race, 'dependencies')
    dep = ET.SubElement(deps, 'dependency')
    for k, v in [('str', track_id), ('version', 1), ('type', 'TRACK')]:
        field(dep, k, v)
    field(track, 'environment', 'TheDrawingBoard')
    blueprints = ET.SubElement(track, 'blueprints')
    xsi = XSI_TYPE

    def object_at(item, instance, position, yaw, subtype=None):
        obj = ET.SubElement(blueprints, 'TrackBlueprint')
        if subtype:
            obj.set(xsi, subtype)
        field(obj, 'itemID', item)
        field(obj, 'instanceID', instance)
        for group, values in [('position', position), ('rotation', (0., yaw, 0.))]:
            node = ET.SubElement(obj, group)
            for axis, value in zip('xyz', values):
                field(node, axis, format(value, '.9g'))
        return obj

    radius_x, radius_z = rng.uniform(40., 65.), rng.uniform(45., 75.)
    height_amplitude = rng.uniform(1., 3.)
    for i in range(gates):
        theta = 2 * math.pi * i / gates
        # Smooth height changes; keep the first gate at ground level.
        position = (radius_x * math.cos(theta),
                    5. + height_amplitude * (1. - math.cos(theta)),
                    radius_z * math.sin(theta))
        tangent = (-radius_x * math.sin(theta), radius_z * math.cos(theta))
        yaw = math.degrees(math.atan2(tangent[0], tangent[1])) + 180.
        object_at('CheckpointBox10mX10m01', i + 1, position, yaw, 'TrackBlueprintFlag')
    spawn_id = gates + 1
    spawn = object_at('SpawnPointSingle02', spawn_id, (radius_x, 1.616, -8.), 0.,
                      'TrackBlueprintSpawnpoint')
    spawn_info = ET.SubElement(spawn, 'spawnpoint', {xsi: 'NamedDroneSpawnpoint'})
    field(spawn_info, 'name', 'Race Start')
    # Scenery lies outside the route. These placements are not an obstacle-
    # avoidance benchmark and are not claimed as exact rendered depth labels.
    for i in range(obstacles):
        theta = rng.uniform(0., 2 * math.pi)
        margin = rng.uniform(14., 25.)
        position = ((radius_x + margin) * math.cos(theta), 2.5,
                    (radius_z + margin) * math.sin(theta))
        object_at('DrawingBoardWall5mx5m05', gates + 2 + i, position,
                  math.degrees(theta), 'TrackBlueprintFlag')
    field(track, 'hideDefaultSpawnpoint', 'true')
    for k, v in [('requiredLaps', 1), ('validity', 'Valid'), ('spawnPointID', spawn_id)]:
        field(race, k, v)
    passages = ET.SubElement(race, 'checkPointPassages')
    # Cross the first gate again to finish the closed loop.
    for i in range(gates + 1):
        p = ET.SubElement(passages, 'RaceCheckpointPassage')
        for k, v in [('uniqueId', uid(f'passage-{i}')), ('checkPointID', i % gates + 1),
                     ('checkPointSubID', ''),
                     ('passageType', 'Start' if i == 0 else 'Finish' if i == gates else 'Pass'),
                     ('directionality', 'RightToLeft')]:
            field(p, k, v)
        following = ET.SubElement(p, 'nextPassageIDs')
        if i < gates:
            field(following, 'string', uid(f'passage-{i+1}'))
    manifest = dict(schema=1, purpose='procedural development course', pool='development',
                    family='ellipse', runtime_geometry_allowed=False, recipe=recipe,
                    track_id=track_id, race_id=race_id, validation=validate_pair(track, race),
                    source=dict(kind='code-generated; built-in Liftoff asset IDs'), files={})
    out.mkdir(parents=True)
    for kind, root, local_id, ext in [('Tracks', track, track_id, 'track'),
                                      ('Races', race, race_id, 'race')]:
        relative = Path(kind)/local_id/(local_id+'_0001.'+ext)
        target = out/relative
        target.parent.mkdir(parents=True)
        ET.indent(root)
        ET.ElementTree(root).write(target, encoding='utf-8', xml_declaration=True)
        manifest['files'][relative.as_posix()] = digest(target)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    q = sub.add_parser('clone')
    q.add_argument('--track', required=True)
    q.add_argument('--race', required=True)
    q.add_argument('--out', required=True)
    q.add_argument('--name', required=True)
    q.add_argument('--translate', nargs=3, type=float, default=(0., 0., 0.))
    q.add_argument('--laps', type=int, default=1)
    q = sub.add_parser('install')
    q.add_argument('bundle')
    q.add_argument('--local-root', required=True)
    q = sub.add_parser('generate')
    q.add_argument('--out', required=True)
    q.add_argument('--seed', type=int, required=True)
    q.add_argument('--gates', type=int, default=10)
    q.add_argument('--obstacles', type=int, default=12)
    a = parser.parse_args()
    if a.command == 'clone':
        result = clone_pair(a.track, a.race, a.out, a.name, a.translate, a.laps)
    elif a.command == 'install':
        result = install_pair(a.bundle, a.local_root)
    else:
        result = generate_loop(a.out, a.seed, a.gates, a.obstacles)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
