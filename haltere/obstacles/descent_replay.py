"""Open-loop replay of the Straw Bale downhill through the view-keeping descent rule (offline; development only).

Each logged tick of a flight goes through the default fast pilot (``--stack none``, as the Straw Bale laps flew) and
through the same pilot with the descent-view declaration (``haltere.obstacles.vertical_replay.replay``). The recorded
motion does not respond to the replayed requests: the output is the request each variant would have made at the
recorded states, never a flight.

Scored on the downhill of the logs, a region taken from the logged positions (offline scoring only; nothing at runtime
knows it): x in [-50, -25], y in [120, 195], flying toward -y. Per tick, at the recorded attitude:
- the camera's lower image edge ray (centre column) and its world elevation;
- whether each variant's requested velocity (cvx, cvy, cvz) points below the lower image edge (exact projection
  through the calibrated camera, roll included), and whether the path the rule bounds (the measured horizontal
  velocity with the requested vertical speed) does;
- the requested horizontal speed and sink of each variant;
and around every support-climb onset of the logged pilot (the contacts; 3 s before), the same numbers.

usage: python -m haltere.obstacles.descent_replay --declaration configs/pilot/descent_view.json --out DIR flight ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .vertical_replay import RUNS, durations, onsets, replay

REGION = dict(x=(-50., -25.), y=(120., 195.), vy_max=-.5)
BEFORE_S = 3.


def camera_geometry(sensor):
    from haltere.vision.camera import Camera
    camera = Camera(320, 180, sensor['focal_320'], sensor['tilt_deg'])
    return camera, camera.unproject_body(np.array([[camera.cx, float(camera.height)]]))[0]


def below_edge(camera, rotation, vector):
    """True when a world vector points below the camera's lower image edge (or behind the camera, downward)."""
    cam = camera.body_to_cam() @ (np.asarray(vector, float) @ rotation)
    if cam[2] <= .05*max(np.linalg.norm(cam), 1e-9):
        return bool(cam[1] > 0)
    return bool(cam[1]/cam[2] > camera.cy/camera.f)


def analyse(flight, declaration, runs=RUNS, tree=None):
    from haltere.liftoff.fast_race_cue import descent_view_config
    from haltere.vision.camera import quat_wxyz_to_mat
    import pandas as pd
    tree = tree or str(Path(__file__).resolve().parents[2])
    config = descent_view_config(declaration)
    base, _, info = replay(flight, tree, runs, stack='none')
    new, _, _ = replay(flight, tree, runs, stack='none', descent_view=config)
    side = json.loads((Path(runs)/f'{flight}.json').read_text(encoding='utf-8'))
    camera, lower = camera_geometry(side['gate_sensor'])
    d = pd.read_csv(Path(runs)/f'{flight}.csv', low_memory=False,
                    usecols=['capture_time', 'image_age', 'qw', 'qx', 'qy', 'qz', 'thr'])
    d = d[np.isfinite(d.capture_time) & np.isfinite(d.image_age)].reset_index(drop=True)
    if len(d) != len(base['t']):
        raise RuntimeError('replay ticks and log rows differ')
    rotations = np.stack([quat_wxyz_to_mat(q) for q in d[['qw', 'qx', 'qy', 'qz']].values])
    edge = np.degrees(np.arcsin(np.clip((rotations @ lower)[:, 2], -1, 1)))
    region = ((base['x'] >= REGION['x'][0]) & (base['x'] <= REGION['x'][1]) & (base['y'] >= REGION['y'][0])
              & (base['y'] <= REGION['y'][1]) & (base['vy'] < REGION['vy_max']))
    rows = {}
    for name, a in (('baseline', base), ('descent_view', new)):
        request = np.stack([a['cvx'], a['cvy'], a['cvz']], -1)
        bounded = np.stack([a['vx'], a['vy'], a['cvz']], -1)
        rows[name] = dict(
            request_below=np.array([below_edge(camera, r, v) and v[2] < 0 for r, v in zip(rotations, request)]),
            bounded_below=np.array([below_edge(camera, r, v) and v[2] < 0 for r, v in zip(rotations, bounded)]),
            request_hs=np.hypot(a['cvx'], a['cvy']), request_vz=np.asarray(a['cvz'], float),
            path_deg=np.degrees(np.arctan2(a['cvz'], np.maximum(np.hypot(a['cvx'], a['cvy']), 1e-6))))
    dt = durations(base['t'])
    w = dt*region
    out = dict(flight=flight, downhill_s=round(float(w.sum()), 2),
               replay_fidelity=dict(cmd_vs_log_horizontal_p99=round(float(np.nanpercentile(
                   np.hypot(base['cvx']-base['log_cvx'], base['cvy']-base['log_cvy']), 99)), 4),
                   state_match=round(float((base['state'] == base['log_state']).mean()), 4)),
               measured=dict(horizontal_speed=round(float((np.hypot(base['vx'], base['vy'])*w).sum()/w.sum()), 2),
                             vz=round(float((np.asarray(base['vz'])*w).sum()/w.sum()), 2),
                             lower_edge_deg=round(float((edge*w).sum()/w.sum()), 1),
                             path_below_edge_fraction=round(float((w*np.array([
                                 below_edge(camera, r, v) and v[2] < 0 for r, v in zip(
                                     rotations, np.stack([base['vx'], base['vy'], base['vz']], -1))])).sum()/w.sum()), 3)),
               variants={})
    for name, r in rows.items():
        out['variants'][name] = dict(
            request_below_edge_fraction=round(float((w*r['request_below']).sum()/w.sum()), 3),
            bounded_path_below_edge_fraction=round(float((w*r['bounded_below']).sum()/w.sum()), 3),
            request_horizontal_speed=round(float((r['request_hs']*w).sum()/w.sum()), 2),
            request_vz=round(float((r['request_vz']*w).sum()/w.sum()), 2),
            request_sink_m=round(float((-np.minimum(r['request_vz'], 0)*w).sum()), 1),
            request_path_p10_deg=round(float(np.percentile(r['path_deg'][region], 10)), 1),
            state_seconds={s: round(float(w[base['state'] == s].sum() if name == 'baseline'
                                          else w[new['state'] == s].sum()), 2)
                           for s in ('cue', 'below', 'below_weak', 'support_climb', 'coast', 'side')})
    out['variants']['descent_view'].update(
        limiting_s=round(float((w*(np.nan_to_num(new['view_withheld']) > 0)).sum()), 2),
        boost_s=round(float((w*(np.nan_to_num(new['view_boost']) > 0)).sum()), 2),
        withheld_sink_m=round(float((w*np.nan_to_num(new['view_withheld'])).sum()), 2))
    # contacts: support-climb onsets of the logged pilot (as replayed; it reproduces the logged states)
    contacts = []
    for i in onsets(np.asarray(base['state']) == 'support_climb'):
        t0 = base['t'][i]
        m = (base['t'] >= t0-BEFORE_S) & (base['t'] < t0)
        wm = dt*m
        entry = dict(t=round(float(t0), 2), x=round(float(base['x'][i]), 1), y=round(float(base['y'][i]), 1),
                     z=round(float(base['z'][i]), 2), downhill=bool(region[i]),
                     lower_edge_deg=round(float((edge*wm).sum()/wm.sum()), 1))
        for name, r in rows.items():
            entry[name] = dict(request_below_edge_fraction=round(float((wm*r['request_below']).sum()/wm.sum()), 2),
                               request_horizontal_speed=round(float((r['request_hs']*wm).sum()/wm.sum()), 2),
                               request_vz=round(float((r['request_vz']*wm).sum()/wm.sum()), 2),
                               request_sink_m=round(float((-np.minimum(r['request_vz'], 0)*wm).sum()), 2))
        contacts.append(entry)
    out['support_climb_onsets'] = contacts
    trace = dict(t=base['t'], region=region, edge=edge, x=base['x'], y=base['y'], z=base['z'],
                 hs=np.hypot(base['vx'], base['vy']), vz=base['vz'], state=np.asarray(base['state']),
                 **{f'{n}_{k}': v for n, r in rows.items() for k, v in r.items()},
                 view_bound=new['view_sink_bound'], view_withheld=new['view_withheld'], new_state=np.asarray(new['state']))
    return out, trace


def plot(trace, flight, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t, region = trace['t'], trace['region']
    runs = onsets(region)
    if not len(runs):
        return
    i = runs[0]
    j = i
    while j+1 < len(region) and region[j+1]:
        j += 1
    s = slice(i, j+1)
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(t[s], trace['edge'][s], color='0.4', label='camera lower image edge (recorded attitude)')
    ax[0].plot(t[s], trace['baseline_path_deg'][s], color='#c0392b', label='default pilot: requested path')
    ax[0].plot(t[s], trace['descent_view_path_deg'][s], color='#2471a3', label='descent view: requested path')
    ax[0].set_ylabel('elevation (deg)')
    ax[0].legend(fontsize=8, loc='lower left')
    ax[0].set_title(f'{flight}: first downhill pass (open-loop replay; recorded motion)')
    ax[1].plot(t[s], trace['baseline_request_hs'][s], color='#c0392b', label='default: requested horizontal speed')
    ax[1].plot(t[s], trace['descent_view_request_hs'][s], color='#2471a3', label='descent view: requested')
    ax[1].plot(t[s], trace['hs'][s], color='0.4', label='measured')
    ax[1].set_ylabel('m/s')
    ax[1].legend(fontsize=8, loc='lower left')
    ax[2].plot(t[s], trace['baseline_request_vz'][s], color='#c0392b', label='default: requested vz')
    ax[2].plot(t[s], trace['descent_view_request_vz'][s], color='#2471a3', label='descent view: requested vz')
    ax[2].plot(t[s], trace['vz'][s], color='0.4', label='measured vz')
    support = np.asarray(trace['state'][s]) == 'support_climb'
    for k in onsets(support):
        for a in ax:
            a.axvline(t[s][k], color='k', lw=.8, ls=':')
    ax[2].set_ylabel('m/s')
    ax[2].set_xlabel('flight time (s); dotted: support-climb onsets (contacts)')
    ax[2].legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('flights', nargs='+')
    parser.add_argument('--declaration', required=True)
    parser.add_argument('--runs', default=str(RUNS))
    parser.add_argument('--out', required=True, help='output directory')
    args = parser.parse_args(argv)
    declaration = json.loads(Path(args.declaration).read_text(encoding='utf-8'))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for flight in args.flights:
        result, trace = analyse(flight, declaration, args.runs)
        results.append(result)
        np.savez_compressed(out/f'{flight}_descent_replay.npz', **trace)
        plot(trace, flight, out/f'{flight}_downhill.png')
        print(json.dumps(result), flush=True)
    (out/'descent_replay.json').write_text(json.dumps(dict(
        declaration=args.declaration, version=declaration.get('version'), sha256=declaration.get('sha256'),
        scope='open-loop replay of logged flights (the recorded motion does not respond); development evidence, '
              'not flight evidence', region=REGION, flights=results), indent=1))


if __name__ == '__main__':
    main()
