"""Prepare human flights as image sequences, observed controls and future paths.

No gate labels are inferred. Coordinates use the current drone's FLU body frame;
controls retain Liftoff's Input convention, not the virtual Xbox convention.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .camera import quat_wxyz_to_mat
from .datasets import sha256
from ..liftoff.frames import unity_vec_to_sim, unity_quat_to_sim

INPUTS = ('in_throttle', 'in_yaw', 'in_pitch', 'in_roll')


def columns(rows, names):
    return np.array([[float(r[k]) for k in names] for r in rows], dtype=np.float64)


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def load_capture(source):
    """Validate pose/control correspondence against the original UDP records."""
    import cv2
    source = Path(source).resolve()
    meta = json.loads((source / 'capture.json').read_text(encoding='utf-8'))
    if meta.get('pilot') != 'human' or meta.get('stop_reason') == 'recording':
        raise ValueError(f'{source}: requires a stopped human recording')
    rows, raw = read_csv(source / 'index.csv'), read_csv(source / 'telemetry.csv')
    if len(rows) < 2 or len(raw) < 2 or len(rows) != meta['frames']:
        raise ValueError(f'{source}: empty or incomplete capture index')
    x = {k: columns(rows, [k])[:, 0] for k in ('t', 'ts', 'wall_time', 'capture_end', 'telemetry_age_s')}
    x.update(position=columns(rows, ('px', 'py', 'pz')), attitude=columns(rows, ('qw', 'qx', 'qy', 'qz')),
             controls=columns(rows, INPUTS))
    ts, recv = columns(raw, ('timestamp', 'recv_time')).T
    pos = unity_vec_to_sim(columns(raw, ('px', 'py', 'pz'))) - np.asarray(meta['origin_sim'])
    vel = unity_vec_to_sim(columns(raw, ('vx', 'vy', 'vz')))
    raw_q, inputs = columns(raw, ('qx', 'qy', 'qz', 'qw')), columns(raw, INPUTS)
    if not all(np.isfinite(a).all() for a in [*x.values(), ts, recv, pos, vel, raw_q, inputs]):
        raise ValueError(f'{source}: nonfinite data')
    if any(np.any(np.diff(a) <= 0) for a in (x['t'], x['ts'], x['wall_time'], ts, recv)):
        raise ValueError(f'{source}: duplicate timestamps, reset or time reversal')
    if np.any(np.linalg.norm(np.diff(pos, axis=0), axis=1) > np.maximum(5., 50. * np.diff(ts))):
        raise ValueError(f'{source}: position discontinuity; split the take at the reset')
    if (not np.allclose(x['t'], x['ts'] - x['ts'][0], atol=1e-5)
            or not np.allclose(np.linalg.norm(x['attitude'], axis=1), 1, atol=.001)
            or not np.allclose(np.linalg.norm(raw_q, axis=1), 1, atol=.001)
            or np.any(np.abs(inputs) > 1.00001)):
        raise ValueError(f'{source}: invalid time, attitude or controls')
    j = np.searchsorted(ts, x['ts'])
    if np.any(j >= len(ts)) or not np.allclose(ts[j], x['ts'], rtol=0, atol=1e-5):
        raise ValueError(f'{source}: image telemetry is missing from the raw log')
    matched_q = np.array([unity_quat_to_sim(q) for q in raw_q[j]])
    if (not np.allclose(pos[j], x['position'], atol=1e-5)
            or not np.allclose(inputs[j], x['controls'], atol=1e-6)
            or not np.allclose(np.abs(np.sum(matched_q * x['attitude'], axis=1)), 1, atol=1e-5)):
        raise ValueError(f'{source}: image/telemetry pose or control mismatch')
    if (np.any(recv[j] > x['wall_time']) or np.any(x['capture_end'] < x['wall_time'])
            or np.any(x['telemetry_age_s'] < 0) or np.any(x['telemetry_age_s'] > .05)
            or not np.allclose(x['capture_end'] - recv[j], x['telemetry_age_s'], atol=1e-5)):
        raise ValueError(f'{source}: invalid capture timing')
    files, hashes, valid_images = [], [], []
    for row in rows:
        name = row['file']
        path = (source / 'frames' / name).resolve()
        if not path.is_relative_to(source / 'frames') or name in files:
            raise ValueError(f'{source}: duplicate or escaping image path')
        data = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (360, 640, 3):
            raise ValueError(f'{source}: invalid captured image {name}')
        hashes.append(hashlib.sha256(data).hexdigest())
        valid_images.append(bool(image.max() != image.min()))
        files.append(name)
    x.update(files=np.array(files), hashes=np.array(hashes), raw_ts=ts, raw_position=pos,
             velocity=vel[j], valid_images=np.array(valid_images), meta=meta)
    return x


def make_examples(capture, segments, horizons=(.25, .5, 1.), max_gap=.12):
    """Never interpolate across a gap, or look beyond a selected segment."""
    c = capture
    horizons = np.asarray(horizons, dtype=float)
    if (horizons.ndim != 1 or not len(horizons) or not np.isfinite(horizons).all()
            or np.any(horizons <= 0) or np.any(np.diff(horizons) <= 0)
            or not np.isfinite(max_gap) or max_gap <= 0):
        raise ValueError('Use increasing positive horizons and a positive gap limit')
    groups = np.full(len(c['t']), -1, dtype=np.int32)
    ends = np.zeros(len(groups))
    for group, segment in enumerate(segments):
        start, end = segment['start_s'], segment['end_s']
        if not np.isfinite([start, end]).all() or not 0 <= start < end <= c['t'][-1]:
            raise ValueError('Segment bounds must be within the recorded flight')
        mask = (c['t'] >= start) & (c['t'] <= end)
        if np.any(groups[mask] >= 0):
            raise ValueError('Selected segments overlap')
        groups[mask], ends[mask] = group, end
    ids, future, body_velocity, run_ids = [], [], [], []
    run, previous = -1, -2
    for i in np.flatnonzero(groups >= 0):
        if not c['valid_images'][i]:
            continue
        query = c['ts'][i] + horizons
        if c['t'][i] + horizons[-1] > ends[i] or query[-1] > c['raw_ts'][-1]:
            continue
        lo = np.searchsorted(c['raw_ts'], c['ts'][i])
        hi = np.searchsorted(c['raw_ts'], query[-1])
        if np.any(np.diff(c['raw_ts'][lo:hi+1]) > max_gap):
            continue
        p = np.stack([np.interp(query, c['raw_ts'], c['raw_position'][:, k]) for k in range(3)], axis=1)
        rotation = quat_wxyz_to_mat(c['attitude'][i])
        future.append((p - c['position'][i]) @ rotation)
        body_velocity.append(c['velocity'][i] @ rotation)
        if (i != previous + 1 or groups[i] != groups[previous]
                or c['ts'][i] - c['ts'][previous] > max_gap
                or c['wall_time'][i] - c['wall_time'][previous] > max_gap):
            run += 1
        ids.append(i)
        run_ids.append(run)
        previous = i
    if not ids:
        raise ValueError('No examples remain after segment and timing checks')
    ids = np.array(ids)
    return dict(filename=c['files'][ids], source_row=ids, t=c['t'][ids], timestamp=c['ts'][ids],
                run_id=np.array(run_ids), segment_id=groups[ids],
                position=c['position'][ids].astype('float32'), attitude=c['attitude'][ids].astype('float32'),
                velocity_body=np.array(body_velocity, dtype='float32'),
                controls=c['controls'][ids].astype('float32'),
                future_body=np.array(future, dtype='float32'), horizons_s=horizons.astype('float32'))


def prepare(plan_path, out, *, root='.'):
    """Build a new dataset, retaining hashes and rejecting whole-take/image leakage."""
    plan_path, out, root = Path(plan_path), Path(out).resolve(), Path(root).resolve()
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('schema') != 1 or not plan.get('takes'):
        raise ValueError('Expected a schema 1 plan with takes')
    prepared, seen_paths, seen_ids = [], set(), set()
    hash_splits, course_splits = {}, {}
    for take in plan['takes']:
        ident, split = take['id'], take['split']
        if (Path(ident).name != ident or any(ch not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for ch in ident)
                or not ident or ident in seen_ids or split not in ('train', 'validation', 'review')):
            raise ValueError('Invalid or repeated take ID/split')
        source = (root / take['source']).resolve()
        if source in seen_paths:
            raise ValueError('A whole take cannot be assigned more than once')
        seen_ids.add(ident)
        seen_paths.add(source)
        c = load_capture(source)
        if c['meta'].get('profile') != take['profile']:
            raise ValueError(f'{ident}: selected profile differs from capture')
        examples = make_examples(c, take['segments'], plan['horizons_s'], plan.get('max_gap_s', .12))
        if split != 'review':
            course_splits.setdefault(take['profile'], set()).add(split)
            # Check the whole source flight. Uniform transition frames have no
            # scene information and are excluded from both examples and overlap.
            for fingerprint in set(c['hashes'][c['valid_images']]):
                if fingerprint in hash_splits and hash_splits[fingerprint] != split:
                    raise ValueError('An exact image occurs in both training and validation')
                hash_splits[fingerprint] = split
        entry = dict(take, source=os.path.relpath(source, out).replace('\\', '/'),
                     examples=len(examples['t']), arrays=f'{ident}.npz',
                     source_hashes={name: sha256(source / name) for name in ('capture.json', 'index.csv', 'telemetry.csv', 'camera.yaml')},
                     frame_hashes=dict(zip(c['files'].tolist(), c['hashes'].tolist())),
                     excluded_uniform_images=c['files'][~c['valid_images']].tolist(),
                     raw_frames=len(c['t']),
                     telemetry_receive_age_p95_s=float(np.quantile(c['telemetry_age_s'], .95)))
        prepared.append((entry, examples))
    for profile in plan.get('holdout_profiles', []):
        if course_splits.get(profile) != {'validation'}:
            raise ValueError(f'{profile}: holdout course must occur only in validation')
    if not {'train', 'validation'}.issubset({entry['split'] for entry, _ in prepared}):
        raise ValueError('Both training and validation takes are required')
    manifest = dict(schema=1, plan_sha256=sha256(plan_path), plan=plan,
                    task='human_future_path_and_observed_controls', split_unit='whole_take',
                    body_frame='FLU: forward, left, up; metres', input_order=list(INPUTS),
                    input_meaning='Observed Liftoff Input; not raw radio or virtual Xbox commands',
                    timing='Original telemetry timestamps; screen/display latency remains uncalibrated',
                    gate_labels_available=False, collision_free_verified=False,
                    takes=[entry for entry, _ in prepared])
    out.mkdir(parents=True)
    for entry, examples in prepared:
        np.savez_compressed(out / entry['arrays'], **examples)
        entry['arrays_sha256'] = sha256(out / entry['arrays'])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


class DemonstrationSequences:
    """CPU/numpy sequence loader usable by a PyTorch DataLoader.

    Future paths and observed controls are targets. No mirror/crop augmentation
    is applied, because those operations would also require transforming targets.
    """

    def __init__(self, dataset, split='train', length=16, stride=8, image_size=(160, 90)):
        self.root = Path(dataset).resolve()
        if split not in ('train', 'validation', 'review') or length <= 0 or stride <= 0:
            raise ValueError('Use a valid split and positive sequence length/stride')
        self.length, self.image_size = length, image_size
        self.takes, self.windows = [], []
        manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        for entry in manifest['takes']:
            if entry['split'] != split:
                continue
            if sha256(self.root / entry['arrays']) != entry['arrays_sha256']:
                raise ValueError(f'{entry["id"]}: prepared arrays changed since preparation')
            with np.load(self.root / entry['arrays'], allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
            n = len(self.takes)
            self.takes.append((entry, arrays))
            for run in np.unique(arrays['run_id']):
                ids = np.flatnonzero(arrays['run_id'] == run)
                for start in range(int(ids[0]), int(ids[-1]) - length + 2, stride):
                    self.windows.append((n, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        import cv2
        n, start = self.windows[index]
        entry, arrays = self.takes[n]
        stop = start + self.length
        source = (self.root / entry['source']).resolve()
        images = []
        for name in arrays['filename'][start:stop]:
            data = (source / 'frames' / str(name)).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry['frame_hashes'][str(name)]:
                raise ValueError(f'{entry["id"]}: source image changed since preparation: {name}')
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'Cannot decode {source / "frames" / name}')
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
        return dict(images=np.stack(images).astype('float32') / 255,
                    velocity_body=arrays['velocity_body'][start:stop],
                    attitude=arrays['attitude'][start:stop],
                    controls=arrays['controls'][start:stop],
                    future_body=arrays['future_body'][start:stop],
                    time_s=(arrays['timestamp'][start:stop] - arrays['timestamp'][start]).astype('float32'),
                    take_id=entry['id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan')
    parser.add_argument('--out', required=True)
    parser.add_argument('--root', default='.')
    args = parser.parse_args()
    manifest = prepare(args.plan, args.out, root=args.root)
    for split in ('train', 'validation', 'review'):
        selected = [t for t in manifest['takes'] if t['split'] == split]
        print(f'{split}: {len(selected)} takes, {sum(t["examples"] for t in selected):,} examples')


if __name__ == '__main__':
    main()
