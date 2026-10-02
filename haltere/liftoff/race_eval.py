"""Main-track race eval: offline checkpoint progress and attempt grades (docs/race_eval.md).

Race geometry is read here only after a flight, to score it. It is never a runtime input,
and a geometric crossing never confirms a game finish: `finish` needs game evidence (the
results screen), recorded as `<log stem>-finish.json` in the format the 2026-09-22 races used.
The game's race XML stays outside the repository; the case file pins its hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# Unity world (right, up, forward) -> simulator (forward, left, up), as in gate_evaluation.track_planes.
C = np.array([[0., 0., 1.], [-1., 0., 0.], [0., 1., 0.]])
GAME_BUNDLES = {'races': '4348d1c3112b77608a3123fa5af4e371.bundle', 'tracks': '7799290bb12d767f065d97ae43f10abc.bundle'}
UNITY_VERSION = '2022.3.62f3'  # the stripped bundles carry none; read from resources.assets (docs/bot_routes.md)

# Arch openings (width, height) in metres, measured up from the arch's base. The track files do not
# carry them: these are approximations from the asset names and recorded flights, so crossings near
# an edge are listed for a video check. Checkpoint boxes are centred volumes sized by name or scale.
AIRGATE_OPENINGS = [
    ('AirgateBig240H300B', (3.0, 2.4)),  # 240 cm high, 300 cm broad by name; a Straw top-banner impact at 2.39 m
    ('AirgateBigLiftoff', (3.0, 2.4)),   # assumed the Big family's size; clean crossings at 0.96-1.40 m
    ('AirgateLiftoff', (1.8, 1.3)),      # Minus: clean crossings at 0.64-0.68 m, a leg strike 0.71 m off centre
]
TOLERANCE_M = .25  # interpolation and telemetry error around an opening
EDGE_M = .3        # credited crossings this close to an edge are listed for a video check
MAX_GAP_S = .25    # no crossing is credited across a longer telemetry gap
RESET_JUMP_M = 5.  # a larger single-sample jump is a reset; scoring stops there

RUNTIME_STOPS = ('No fresh live', 'Fresh camera', 'Controller missed')
METRICS = ('progress', 'finish', 'clean_finish', 'within_target')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_geometry(game_dir, out, targets):
    """Copy the race and track XML of each target race out of the installed game, read-only."""
    import UnityPy
    UnityPy.config.FALLBACK_UNITY_VERSION = UNITY_VERSION
    aa = Path(game_dir)/'Liftoff_Data/StreamingAssets/aa/StandaloneWindows64'

    def documents(bundle):
        found = {}
        for obj in UnityPy.load(str(aa/bundle)).objects:
            if obj.type.name == 'TextAsset':
                asset = obj.read()
                raw = asset.m_Script
                raw = raw.encode('utf-8', 'surrogateescape') if isinstance(raw, str) else bytes(raw)
                local = re.search(rb'<localID>\s*<str>([0-9a-f-]+)</str>', raw)
                if local:
                    found[local.group(1).decode()] = (asset.m_Name, raw)
        return found

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    races, tracks = documents(GAME_BUNDLES['races']), documents(GAME_BUNDLES['tracks'])
    exported = []
    for case in targets:
        name, raw = races[case['race_id']]
        track_id = re.search(rb'<str>([0-9a-f-]+)</str>\s*<version>\d+</version>\s*<type>TRACK</type>', raw).group(1)
        track_name, track_raw = tracks[track_id.decode()]
        (out/f"{case['id']}-race.xml").write_bytes(raw)
        (out/f"{case['id']}-track.xml").write_bytes(track_raw)
        exported.append(dict(id=case['id'], race_asset=name, track_asset=track_name,
                             race_sha256=hashlib.sha256(raw).hexdigest(),
                             track_sha256=hashlib.sha256(track_raw).hexdigest()))
    (out/'manifest.json').write_text(json.dumps(dict(source_bundles=GAME_BUNDLES, exported=exported), indent=2))
    return exported


def opening(item, scale):
    """(width, height, centred) of a checkpoint's opening, in metres."""
    box = re.match(r'CheckpointBox(\d+)mX(\d+)m', item)
    if box:
        return float(box[1])*scale['x'], float(box[2])*scale['y'], True
    if item.startswith('CheckpointBoxFlexible'):
        return max(scale['x'], scale['z']), scale['y'], True
    for prefix, (width, height) in AIRGATE_OPENINGS:
        if item.startswith(prefix):
            return width, height, False
    raise ValueError(f'No opening known for checkpoint item {item!r}')


def checkpoint_planes(track_path, origin):
    planes = {}
    for b in ET.parse(track_path).getroot().findall('./blueprints/TrackBlueprint'):
        item = b.findtext('itemID', '').strip()
        position = np.array([float(b.findtext('position/'+k, '0')) for k in 'xyz'])
        R = Rotation.from_euler('zxy', [float(b.findtext('rotation/'+k, '0')) for k in 'zxy'], degrees=True).as_matrix()
        scale = {k: float(b.findtext('scale/'+k, '1')) for k in 'xyz'}
        normal, side = C@R[:, 2], -C@R[:, 0]
        if item.startswith('CheckpointBoxFlexible') and scale['x'] < scale['z']:
            normal, side = C@R[:, 0], C@R[:, 2]  # the passage crosses the box's thin horizontal axis
        planes[b.findtext('instanceID')] = dict(item=item, base=C@position-np.asarray(origin, float), normal=normal,
                                                side=side, up=C@R[:, 1], scale=scale)
    return planes


def race_sequence(race_path, track_path, origin):
    """Every checkpoint passage of the whole race in order: the start, then each lap's chain."""
    race = ET.parse(race_path).getroot()
    passages = race.findall('./checkPointPassages/RaceCheckpointPassage')
    by_id = {p.findtext('uniqueId'): p for p in passages}
    starts = [p for p in passages if p.findtext('passageType') == 'Start']
    if len(starts) != 1:
        raise ValueError('Expected one start passage')
    chain, seen, current = [], set(), starts[0]
    while True:
        if current.findtext('uniqueId') in seen:
            raise ValueError('Cyclic passage graph')
        seen.add(current.findtext('uniqueId'))
        chain.append(current.findtext('checkPointID'))
        following = current.findall('./nextPassageIDs/string')
        if current.findtext('passageType') == 'Finish':
            break
        if len(following) != 1 or following[0].text not in by_id:
            raise ValueError('Expected a single connected race chain')
        current = by_id[following[0].text]
    if len(seen) != len(passages):
        raise ValueError('Disconnected race passages')
    planes = checkpoint_planes(track_path, origin)
    laps = int(race.findtext('requiredLaps'))
    ids = [chain[0]] + chain[1:]*laps
    sequence = []
    for i, gate_id in enumerate(ids):
        gate = planes[gate_id]
        towards = planes[ids[1]]['base']-gate['base'] if i == 0 else gate['base']-planes[ids[i-1]]['base']
        approach = float(towards@gate['normal'])
        if abs(approach) < .1:
            raise ValueError(f'Cannot infer the crossing direction of checkpoint {gate_id}')
        width, height, centred = opening(gate['item'], gate['scale'])
        sequence.append(dict(gate, id=gate_id, sign=1 if approach > 0 else -1, width=width, height=height,
                             centred=centred, lap=0 if i == 0 else (i-1)//(len(chain)-1)+1))
    return sequence, laps, len(chain)-1


def _window(gate, lateral, height):
    """Distance inside the opening (negative outside), from the nearest edge."""
    low, high = (-gate['height']/2, gate['height']/2) if gate['centred'] else (0., gate['height'])
    return min(gate['width']/2-abs(lateral), height-low, high-height)


def race_progress(times, positions, sequence):
    """Credit the race's checkpoints in order, from the correct side, through their openings."""
    t, P = np.asarray(times, float), np.asarray(positions, float)
    if t.ndim != 1 or P.shape != (len(t), 3) or len(t) < 2 or (np.diff(t) < 0).any():
        raise ValueError('Need chronological times and positions')
    k, crossed, outside, gaps, reset_at = 0, [], [], [], None
    for i in range(1, len(t)):
        if np.linalg.norm(P[i]-P[i-1]) > RESET_JUMP_M:
            reset_at = float(t[i])
            break
        if t[i]-t[i-1] > MAX_GAP_S:
            gaps.append([float(t[i-1]), float(t[i])])
            continue
        while k < len(sequence):
            gate = sequence[k]
            d0 = float((P[i-1]-gate['base'])@gate['normal'])*gate['sign']
            d1 = float((P[i]-gate['base'])@gate['normal'])*gate['sign']
            if not d0 < 0 <= d1:
                break
            f = -d0/(d1-d0)
            offset = P[i-1]+f*(P[i]-P[i-1])-gate['base']
            lateral, height = float(offset@gate['side']), float(offset@gate['up'])
            event = dict(index=k, gate_id=gate['id'], lap=gate['lap'], seconds=float(t[i-1]+f*(t[i]-t[i-1])),
                         lateral_m=round(lateral, 3), height_m=round(height, 3))
            margin = _window(gate, lateral, height)
            if margin < -TOLERANCE_M:
                if abs(lateral) < gate['width']/2+3 and -3 < height < gate['height']+3:
                    outside.append(event)  # through the plane beside or over the opening: not credited
                break
            crossed.append(dict(event, edge=margin < EDGE_M))
            k += 1
    total = len(sequence)-1
    done = max(0, len(crossed)-1)
    return dict(progress=done/total, crossed=done, total=total,
                race_time_geometric_s=crossed[-1]['seconds']-crossed[0]['seconds'] if done == total else None,
                next_expected=None if k == len(sequence) else sequence[k]['id'],
                crossings=crossed, outside_opening=outside, gaps=gaps, reset_at=reset_at,
                edge_crossings=sum(c['edge'] for c in crossed), race_completion_verified=False)


def end_cause(meta, finish=None, review=None):
    """finish / impact / limit / operator_stop / runtime_error / needs_review, from the sidecar and evidence."""
    if review and review.get('end_cause'):
        return review['end_cause']
    if finish and finish.get('game_confirmed_finish'):
        return 'finish'
    reason = str(meta.get('stop_reason') or '')
    if (not meta.get('ticks') or meta.get('camera_failure') or meta.get('controller_deadline_failure')
            or reason.startswith(RUNTIME_STOPS)):
        return 'runtime_error'
    if meta.get('impact') or reason.startswith('Impact'):
        return 'impact'
    if reason == 'duration' or 'limit exceeded' in reason:
        return 'limit'
    return 'needs_review'  # e.g. telemetry left live flight: an operator pause or the results screen


def race_seconds(text):
    minutes, seconds = str(text).split(':')
    return 60*int(minutes)+float(seconds)


def _evidence(log, suffix):
    path = Path(log).with_name(Path(log).stem+suffix)
    return json.loads(path.read_text()) if path.exists() else None


def grade_attempt(log, case, geometry_dir, *, finish=None, review=None):
    log = Path(log)
    meta = json.loads(log.with_suffix('.json').read_text())
    finish = finish if finish is not None else _evidence(log, '-finish.json')
    review = review if review is not None else _evidence(log, '-review.json')
    cause = end_cause(meta, finish, review)
    row = dict(prompt_id=case['id'], log=log.as_posix(), end_cause=cause,
               checkpoint_sha256=meta.get('checkpoint_sha256'),
               motor=(meta.get('motor_controller') or {}).get('kind'),
               stop_reason=meta.get('stop_reason'))
    if cause == 'runtime_error':
        return dict(row, status='error', failure_class='runtime')
    geometry_dir = Path(geometry_dir)
    race, track = geometry_dir/f"{case['id']}-race.xml", geometry_dir/f"{case['id']}-track.xml"
    if (sha256(race), sha256(track)) != (case['race_sha256'], case['track_sha256']):
        raise ValueError(f"{case['id']}: exported race geometry does not match the case file's hashes")
    data = pd.read_csv(log, usecols=['ts', 'x', 'y', 'z'])
    sequence, _, _ = race_sequence(race, track, meta['origin_sim'])
    progress = race_progress(data.ts.values, data[['x', 'y', 'z']].values, sequence)
    finished = cause == 'finish'
    race_s = race_seconds(finish['race_time']) if finished and finish.get('race_time') else None
    contacts = None if not review else review.get('contacts_on_video')
    interventions = (review or {}).get('interventions', (finish or {}).get('flight_interventions'))
    clean = None if finished and contacts is None else bool(
        finished and contacts == 0 and not interventions and progress['reset_at'] is None)
    ratio = race_s/case['personal_race_s'] if race_s else None
    audit = _evidence(log, '.contact-audit.json')
    return dict(row, status='ok',
                grade=dict(progress=round(progress['progress'], 4), finish=finished, clean_finish=clean,
                           within_target=None if clean is None else bool(clean and ratio and ratio <= case['max_time_ratio'])),
                time_ratio=None if ratio is None else round(ratio, 3), race_time_s=race_s,
                race_time_geometric_s=progress['race_time_geometric_s'],
                checkpoints=f"{progress['crossed']}/{progress['total']}", next_expected=progress['next_expected'],
                end_s=round(float(data.ts.iloc[-1]-(progress['crossings'][0]['seconds'] if progress['crossings']
                                                     else data.ts.iloc[0])), 2),
                edge_crossings=progress['edge_crossings'], outside_opening=progress['outside_opening'],
                telemetry_gaps=len(progress['gaps']), reset_at=progress['reset_at'],
                contact_audit=None if audit is None else len(audit.get('contacts', [])),
                contacts_on_video=contacts)


def load_cases(path):
    spec = json.loads(Path(path).read_text())
    return spec, {c['id']: dict(c, max_time_ratio=spec['max_time_ratio']) for c in spec['cases']}


def _git(*args):
    import subprocess
    return subprocess.run(['git', *args], capture_output=True, text=True, check=True).stdout.strip()


def stack_identity(batch):
    """The commit, the runtime files changed against it, and the hashes of every file the commands name."""
    files = sorted({a for v in batch['variants'].values() for a in [v['checkpoint']]+v['flags'] if Path(a).is_file()}
                   | {a for a in batch['stack_flags'] if Path(a).is_file()})
    changed = [f for f in _git('diff', '--name-only', 'HEAD', '--', 'haltere', 'configs').splitlines()
               if not f.startswith('configs/race_eval/')]
    return dict(git_head=_git('rev-parse', 'HEAD'), changed_runtime_files=changed,
                file_sha256={f: sha256(f) for f in files})


def plan_batch(batch_path, out):
    """Freeze a batch: identity, order and one command per attempt. Refuses to overwrite."""
    batch = json.loads(Path(batch_path).read_text())
    spec, cases = load_cases(batch['cases'])
    out = Path(out)
    if (out/'plan.json').exists():
        raise FileExistsError(out/'plan.json')
    identity = stack_identity(batch)
    if identity['changed_runtime_files']:
        raise RuntimeError(f"Commit the runtime changes first: {identity['changed_runtime_files']}")
    attempts, reps = [], {}
    for case_id in batch['case_order']:
        case = cases[case_id]
        for variant in batch['order']:
            reps[case_id, variant] = rep = reps.get((case_id, variant), 0)+1
            v = batch['variants'][variant]
            stem = (out/f'{case_id}-{variant}-{rep}').as_posix()
            command = ['.venv/Scripts/python.exe', '-m', 'haltere.liftoff.visual_brain', v['checkpoint'], *batch['stack_flags'],
                       *v['flags'], '--seconds', str(case['attempt_seconds']), '--max-distance', str(case['max_distance_m']),
                       '--log', stem+'.csv', '--record', stem+'.mp4', *batch['recording_flags']]
            attempts.append(dict(n=len(attempts)+1, case=case_id, variant=variant, rep=rep, log=stem+'.csv',
                                 command=' '.join(command)))
    plan = dict(batch=batch['name'], cases_file=batch['cases'], cases_sha256=sha256(batch['cases']),
                batch_file=str(batch_path), batch_sha256=sha256(batch_path), identity=identity,
                checkpoint_sha256={k: sha256(v['checkpoint']) for k, v in batch['variants'].items()},
                retry_rule='a runtime error is retried once, unchanged, as <stem>-retry1', attempts=attempts)
    out.mkdir(parents=True, exist_ok=True)
    (out/'plan.json').write_text(json.dumps(plan, indent=2))
    return plan


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def record_attempt(batch_dir, log, geometry_dir):
    """Grade one finished attempt of a frozen batch and append it, once, to results.jsonl or errors.jsonl."""
    batch_dir, log = Path(batch_dir), Path(log)
    plan = json.loads((batch_dir/'plan.json').read_text())
    stem = re.sub(r'-retry\d+$', '', log.with_suffix('').as_posix())
    attempt = next(a for a in plan['attempts'] if a['log'] == stem+'.csv')
    _, cases = load_cases(plan['cases_file'])
    meta = json.loads(log.with_suffix('.json').read_text())
    if meta.get('checkpoint_sha256') != plan['checkpoint_sha256'][attempt['variant']]:
        raise RuntimeError('The log was flown with a different checkpoint than the plan')
    if _git('rev-parse', 'HEAD') != plan['identity']['git_head'] or stack_identity(
            json.loads(Path(plan['batch_file']).read_text()))['changed_runtime_files']:
        raise RuntimeError('The stack changed since the plan was frozen')
    key = dict(case=attempt['case'], variant=attempt['variant'], rep=attempt['rep'])
    results, errors = batch_dir/'results.jsonl', batch_dir/'errors.jsonl'
    if any(all(r.get(k) == v for k, v in key.items()) for r in _rows(results)):
        raise FileExistsError(f'Attempt already recorded: {key}')
    row = dict(grade_attempt(log, cases[attempt['case']], geometry_dir), **key, n=attempt['n'])
    if row['end_cause'] == 'needs_review':
        raise RuntimeError(f'{log.name}: write {log.stem}-review.json with the end cause seen on the video first')
    if row['status'] == 'ok' and row['grade']['finish'] and row['contacts_on_video'] is None:
        raise RuntimeError(f'{log.name}: a finish needs contacts_on_video in {log.stem}-review.json')
    with open(errors if row['status'] == 'error' else results, 'a') as f:
        f.write(json.dumps(row, default=str)+'\n')
    return row


def _bootstrap(by_case, draws=4000, seed=0):
    """95% interval of the equal-weight mean over cases, resampling attempts within each case."""
    rng = np.random.default_rng(seed)
    means = [np.mean([rng.choice(v, len(v)).mean() for v in by_case.values()]) for _ in range(draws)]
    return [round(float(q), 3) for q in np.percentile(means, [2.5, 97.5])]


def summarise(batch_dir):
    batch_dir = Path(batch_dir)
    plan = json.loads((batch_dir/'plan.json').read_text())
    rows, errors = _rows(batch_dir/'results.jsonl'), _rows(batch_dir/'errors.jsonl')
    variants = list(dict.fromkeys(a['variant'] for a in plan['attempts']))
    case_ids = list(dict.fromkeys(a['case'] for a in plan['attempts']))
    progress = {v: {c: np.array([r['grade']['progress'] for r in rows if r['variant'] == v and r['case'] == c])
                    for c in case_ids} for v in variants}
    lines = [f"# {plan['batch']}: {len(rows)} of {len(plan['attempts'])} attempts scored, {len(errors)} runtime errors", '',
             '| Variant | Mean progress (95% CI) | ' + ' | '.join(case_ids) + ' | Finishes | Clean | Within 120% |',
             '|---|---|' + '---|'*len(case_ids) + '---|---|---|']
    for v in variants:
        mine = [r for r in rows if r['variant'] == v]
        done = {c: p for c, p in progress[v].items() if len(p)}
        headline = '-' if len(done) < len(case_ids) else '{:.3f} {}'.format(
            np.mean([p.mean() for p in done.values()]), _bootstrap(done))
        cells = ['-' if not len(p) else f"{p.mean():.3f} (n={len(p)})" for p in progress[v].values()]
        count = lambda m: f"{sum(bool(r['grade'][m]) for r in mine)}/{len(mine)}"
        lines.append(f"| {v} | {headline} | " + ' | '.join(cells) + f" | {count('finish')} | {count('clean_finish')} | "
                     f"{count('within_target')} |")
    if len(variants) == 2 and all(len(progress[v][c]) for v in variants for c in case_ids):
        a, b = variants
        diff = {c: progress[b][c].mean()-progress[a][c].mean() for c in case_ids}
        rng, draws = np.random.default_rng(1), []
        for _ in range(4000):
            draws.append(np.mean([rng.choice(progress[b][c], len(progress[b][c])).mean()
                                  - rng.choice(progress[a][c], len(progress[a][c])).mean() for c in case_ids]))
        lines += ['', f"Paired difference {b} - {a} in mean progress: {np.mean(list(diff.values())):+.3f} "
                  f"(95% CI {np.percentile(draws, 2.5):+.3f} to {np.percentile(draws, 97.5):+.3f}); per track: "
                  + ', '.join(f'{c} {d:+.3f}' for c, d in diff.items())]
    lines += ['', '| # | Case | Variant | Rep | End cause | Progress | Checkpoints | Finish | Clean | Time ratio |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: r['n']):
        g = r['grade']
        lines.append(f"| {r['n']} | {r['case']} | {r['variant']} | {r['rep']} | {r['end_cause']} | {g['progress']} | "
                     f"{r['checkpoints']} | {g['finish']} | {g['clean_finish']} | {r['time_ratio'] or '-'} |")
    text = '\n'.join(lines)+'\n'
    (batch_dir/'summary.md').write_text(text)
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export-geometry', help='copy the cases\' race/track XML out of the installed game (read-only)')
    e.add_argument('--cases', default='configs/race_eval/main_tracks_v1.json')
    e.add_argument('--game-dir', default=r'C:\SteamLibrary\steamapps\common\Liftoff')
    e.add_argument('--out', required=True)
    s = sub.add_parser('score', help='grade one visual_brain log against its case')
    s.add_argument('log')
    s.add_argument('--case', required=True)
    s.add_argument('--cases', default='configs/race_eval/main_tracks_v1.json')
    s.add_argument('--geometry', required=True)
    q = sub.add_parser('plan', help='freeze a batch: stack identity, order and the command of every attempt')
    q.add_argument('batch')
    q.add_argument('--out', required=True)
    r = sub.add_parser('record', help='grade one attempt of a frozen batch and append it to its results')
    r.add_argument('batch_dir')
    r.add_argument('log')
    r.add_argument('--geometry', required=True)
    m = sub.add_parser('summary', help='per-variant table, paired difference and every attempt')
    m.add_argument('batch_dir')
    args = p.parse_args(argv)
    if args.command == 'export-geometry':
        spec, _ = load_cases(args.cases)
        print(json.dumps(export_geometry(args.game_dir, args.out, spec['cases']), indent=2))
    elif args.command == 'score':
        _, cases = load_cases(args.cases)
        print(json.dumps(grade_attempt(args.log, cases[args.case], args.geometry), indent=2))
    elif args.command == 'plan':
        plan = plan_batch(args.batch, args.out)
        print(f"{len(plan['attempts'])} attempts frozen at {plan['identity']['git_head'][:7]} -> {args.out}/plan.json")
    elif args.command == 'record':
        print(json.dumps(record_attempt(args.batch_dir, args.log, args.geometry), indent=2, default=str))
    else:
        print(summarise(args.batch_dir))


if __name__ == '__main__':
    main()
