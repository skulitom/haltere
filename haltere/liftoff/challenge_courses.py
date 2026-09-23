"""Generate offline Drawing Board obstacle sections, never runtime guidance.

Each section has an entry and exit checkpoint. Physical cube frames make the
5 m checkpoint openings visible. Seeds vary offsets, obstacle side and difficulty;
all variants of this recipe belong to one development family.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import uuid
import xml.etree.ElementTree as ET

from .course_pool import XSI_TYPE, digest, validate_pair


KINDS = ('flag', 'pillar', 'descent', 'high_checkpoint', 'overhang', 'occluded_marker')
# Read from installed prefab BoxColliders, not inferred from asset names.
# In particular, the asset called Wall5mx5m05 is actually 10 x 5 x 1 m.
BOXES = {
    'DrawingBoardCube1mx1m01': ((1., 1., 1.), (0., .5, 0.)),
    'DrawingBoardCube10mx10m01': ((10., 10., 10.), (0., 5., 0.)),
    'DrawingBoardWall5mx5m05': ((10., 5., 1.), (0., 2.5, 0.)),
}


def generate_sections(out, seed, *, repeats=1, difficulty=.4, box_calibration=False):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if not 1 <= repeats <= 3 or not math.isfinite(difficulty) or not 0 <= difficulty <= 1:
        raise ValueError('Use 1..3 repeats and finite difficulty in [0, 1]')
    recipe = dict(generator='challenge-sections-v1', seed=seed, repeats=repeats, difficulty=difficulty)
    if box_calibration:
        recipe['variant'] = 'box-calibration-v1'
    identity = json.dumps(recipe, sort_keys=True)
    uid = lambda label: str(uuid.uuid5(uuid.NAMESPACE_URL, 'haltere/'+identity+'/'+label))
    rng = random.Random(seed)
    track, race = ET.Element('Track'), ET.Element('Race')

    def field(root, tag, value):
        ET.SubElement(root, tag).text = str(value)

    for root, kind in [(track, 'TRACK'), (race, 'RACE')]:
        field(root, 'gameVersion', '1.7.6')
        local = ET.SubElement(root, 'localID')
        for k, v in [('str', uid(kind)), ('version', 1), ('type', kind)]:
            field(local, k, v)
        name = 'Haltere box calibration v1' if box_calibration else 'Haltere sections v1'
        field(root, 'name', f'{name} {seed}'+(' race' if kind == 'RACE' else ''))
        field(root, 'description', 'Procedural development sections; offline geometry, flight qualification pending.')
    ET.SubElement(track, 'dependencies')
    dependency = ET.SubElement(ET.SubElement(race, 'dependencies'), 'dependency')
    for k, v in [('str', uid('TRACK')), ('version', 1), ('type', 'TRACK')]:
        field(dependency, k, v)
    field(track, 'environment', 'TheDrawingBoard')
    objects = ET.SubElement(track, 'blueprints')
    primitives, unknown_geometry, checkpoints, sections = [], [], [], []

    def asset(item, position, yaw=0., role='obstacle'):
        instance = len(objects)+1
        obj = ET.SubElement(objects, 'TrackBlueprint', {XSI_TYPE: 'TrackBlueprintFlag'})
        field(obj, 'itemID', item)
        field(obj, 'instanceID', instance)
        for tag, values in [('position', position), ('rotation', (0., yaw, 0.))]:
            node = ET.SubElement(obj, tag)
            for axis, value in zip('xyz', values):
                field(node, axis, format(value, '.9g'))
        if item in BOXES:
            size, offset = BOXES[item]
            # All measured box offsets are vertical; yaw leaves them unchanged.
            primitives.append(dict(instance_id=instance, item_id=item, kind='box',
                center=[a+b for a, b in zip(position, offset)], size=list(size), yaw_deg=yaw,
                role=role, provenance='installed prefab BoxCollider; rendered surface check pending'))
        elif role == 'obstacle':
            unknown_geometry.append(dict(instance_id=instance, item_id=item,
                                         reason='mesh collider not yet calibrated; depth here is not ground truth'))
        return obj, instance

    def checkpoint(position):
        _, instance = asset('CheckpointBox5mX5m01', position, 180., 'checkpoint')
        checkpoints.append(dict(instance_id=instance, center=list(position), normal=[0., 0., 1.],
                                half_width=2.5, half_height=2.5))
        # Cube pivots sit at their base. Inner opening is exactly 5 x 5 m in
        # the measured collider model. No invisible 10 m catch-all gate volumes.
        x, y, z = position
        for side in [-3., 3.]:
            for row in range(5):
                asset('DrawingBoardCube1mx1m01', (x+side, y-2.5+row, z), role='gate_frame')
        for col in range(-3, 4):
            asset('DrawingBoardCube1mx1m01', (x+col, y+2.5, z), role='gate_frame')
        return instance

    # Raised launch platform explicitly makes the descent lower than the start.
    asset('DrawingBoardCube10mx10m01', (0., 0., -12.), role='launch_platform')
    spawn, spawn_id = asset('SpawnPointSingle02', (0., 11.616, -12.), role='spawn')
    spawn.set(XSI_TYPE, 'TrackBlueprintSpawnpoint')
    field(ET.SubElement(spawn, 'spawnpoint', {XSI_TYPE: 'NamedDroneSpawnpoint'}), 'name', 'Race Start')
    clearance = 2.-1.1*difficulty
    for repetition in range(repeats):
        for kind in KINDS:
            index = len(sections)
            x = 0. if index == 0 else rng.uniform(-2., 2.)
            z = index*44.
            entry_height, exit_height = 12., 12.
            if kind == 'descent':
                exit_height = 3.
            elif kind == 'high_checkpoint':
                entry_height, exit_height = 3., 10.+4.*difficulty
            elif kind in ('overhang', 'occluded_marker'):
                entry_height = exit_height = 3.
            entry = checkpoint((x, entry_height, z))
            exit_gate = checkpoint((x, exit_height, z+28.))
            obstacle_ids = []
            side = rng.choice([-1., 1.])
            if kind == 'flag':
                if box_calibration:
                    # A new, explicitly named calibration scene, not edited
                    # geometry labels for a course containing an unknown flag.
                    kind = 'box_near_line'
                    _, obj_id = asset('DrawingBoardCube1mx1m01',
                                      (x+side*(clearance+.5), entry_height-.5, z+14.))
                else:
                    _, obj_id = asset(rng.choice(['LedFlag01', 'LedFlagBlue01']),
                                      (x+side*clearance, entry_height-2., z+14.), 90.)
                obstacle_ids.append(obj_id)
            elif kind == 'pillar':
                # A box pillar gives exact collision geometry without pretending
                # a mesh cylinder's radius/pivot has already been verified.
                for row in range(5):
                    _, obj_id = asset('DrawingBoardCube1mx1m01',
                                      (x+side*(clearance+.5), entry_height-2.5+row, z+14.))
                    obstacle_ids.append(obj_id)
            elif kind == 'overhang':
                _, obj_id = asset('DrawingBoardCube10mx10m01', (x, 5.5-1.5*difficulty, z+14.))
                obstacle_ids.append(obj_id)
            elif kind == 'occluded_marker':
                _, obj_id = asset('DrawingBoardWall5mx5m05', (x, 0., z+14.))
                obstacle_ids.append(obj_id)
            sections.append(dict(id=f'section-{index:02d}', kind=kind, repetition=repetition,
                                 entry_checkpoint=entry, exit_checkpoint=exit_gate,
                                 obstacle_ids=obstacle_ids, clearance_setting_m=clearance))
    field(track, 'hideDefaultSpawnpoint', 'true')
    for k, v in [('requiredLaps', 1), ('validity', 'Valid'), ('spawnPointID', spawn_id)]:
        field(race, k, v)
    passages = ET.SubElement(race, 'checkPointPassages')
    for i, point in enumerate(checkpoints):
        node = ET.SubElement(passages, 'RaceCheckpointPassage')
        for k, v in [('uniqueId', uid(f'passage-{i}')), ('checkPointID', point['instance_id']),
                     ('checkPointSubID', ''), ('passageType', 'Start' if i == 0 else
                       'Finish' if i == len(checkpoints)-1 else 'Pass'), ('directionality', 'RightToLeft')]:
            field(node, k, v)
        following = ET.SubElement(node, 'nextPassageIDs')
        if i < len(checkpoints)-1:
            field(following, 'string', uid(f'passage-{i+1}'))
    geometry = dict(schema=1, coordinate_frame='Unity x-right y-up z-forward',
                    runtime_geometry_allowed=False, primitives=primitives,
                    unknown_geometry=unknown_geometry, checkpoints=checkpoints, sections=sections,
                    ground_plane_y=0., source='prefab colliders, not a rendered depth image',
                    render_alignment_verified=False, checkpoint_progress_verified=False)
    manifest = dict(schema=1, purpose='procedural obstacle-section development', pool='development',
                    family='challenge-sections-v1', runtime_geometry_allowed=False, recipe=recipe,
                    track_id=uid('TRACK'), race_id=uid('RACE'), validation=validate_pair(track, race),
                    source=dict(kind='code-generated; built-in Liftoff assets'), files={},
                    decision_rule=dict(primary='next checkpoint reached without impact',
                      post_crash='all later sections unattempted', runtime_stop='current section censored; unchanged retry once',
                      reporting='paired per-course and per-obstacle-type results; do not treat sections as independent courses',
                      promotion='no promotion from this development recipe alone; freeze held-out family comparison'))
    out.mkdir(parents=True)
    for folder, root, kind, ext in [('Tracks', track, 'TRACK', 'track'), ('Races', race, 'RACE', 'race')]:
        relative = Path(folder)/uid(kind)/(uid(kind)+'_0001.'+ext)
        path = out/relative
        path.parent.mkdir(parents=True)
        ET.indent(root)
        ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
        manifest['files'][relative.as_posix()] = digest(path)
    geometry_path = out/'offline-geometry.json'
    geometry_path.write_text(json.dumps(geometry, indent=2), encoding='utf-8')
    manifest['offline_geometry'] = dict(path=geometry_path.name, sha256=digest(geometry_path))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--difficulty', type=float, default=.4)
    p.add_argument('--box-calibration', action='store_true',
                   help='Create a distinct box-only scene for rendered-depth calibration; not a flag test')
    a = p.parse_args()
    print(json.dumps(generate_sections(a.out, a.seed, repeats=a.repeats, difficulty=a.difficulty,
                                      box_calibration=a.box_calibration), indent=2))


if __name__ == '__main__':
    main()
