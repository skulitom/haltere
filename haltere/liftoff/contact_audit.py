"""Pilot-independent ground-contact audit of a flight log (offline scoring only; configs/contact_audit.json).

Why: the support-climb onsets that earlier rounds counted as ground contacts depend on the pilot. With the
view-keeping descent the sink request stays small and the older support rules cannot fire, so their onsets
undercount contacts; a graduation bar of "no ground contact" needs a count that does not depend on the pilot.

What it measures, from telemetry alone (the 100 Hz CSV, one row per telemetry frame):
- The external specific force: the measured acceleration (central difference of the velocity over +-half_frames
  frames) plus gravity, less what the propellers and drag explain. The propellers push only along the drone's own up
  axis, with the thrust the measured curve gives for the game's processed throttle (g * thrust_twr * drive^
  thrust_exponent, drive = (processed + 1)/2; the logged ``in_thr`` readback); drag is the measured body-frame linear
  drag. In free air what is left is small (live logs: median about -0.05 m/s^2 vertically, p10..p90 about
  +-0.2 m/s^2 at a drive of 0.5-0.7). A surface can only push: a ground reaction shows as an upward external force.
- Support: the external force's world-vertical component, averaged over window_s (centred), at least support_on
  while the drive stays within [drive_min, drive_max] (outside, the thrust curve is not reliable: throttle cuts leave
  +0.4..+3.3 m/s^2 of median residual at a drive of 0.3-0.45 from motor idle and spin-down). In fast manoeuvres the
  model leaves more (the first round-4b attempt's design runs: the window mean's 99.9th percentile below 0.4 m/s^2 at
  body rates up to 3 rad/s, 0.6 at 3-4 and 1.4-1.6 above 4), so the threshold rises by support_rate_gain per rad/s of
  the window's highest body rate above rate_free.
- Arrest with low throttle: a drone descending at vz <= -arrest_descent whose vertical speed then rises by at least
  arrest_accel m/s^2 over window_s (drag removed) while the drive has not exceeded the hover drive for the window and
  arrest_lookback_s before it: below hover the propellers cannot arrest a descent (the lookback lets the motors spin
  down), so something else did (this catches contacts during throttle cuts, where the curve is not used).
- Impact: the acceleration magnitude (3-frame mean) above impact_accel with at least impact_offaxis of it off the
  thrust axis (haltere.liftoff.flightlog.collisions), whatever the throttle.
- Height evidence (reported, not required): the recorded height within launch_plane_m of the launch plane
  (launch-relative z; on courses whose floor is the launch plane, e.g. Minus Two, that is the floor), and the lowest
  height of the episode.
- The terminal impact the runner recorded in the sidecar (the run ends on it; its impulse is not in the CSV).

Frames within arming_s of the start (the arming hold and throttle ramp on the launch pad) and before the first
departure from the launch plane are not scored. Evidence frames closer than merge_s are merged into one contact;
a contact needs min_s of support or arrest evidence, or one impact frame. Per contact: start and end time (log
phase), position, speed and vertical speed at the start, peak and mean external force, the vertical impulse it
delivered (the integral of the vertical external force, m/s), the lowest height, the kind (impact, support,
slide: support for at least slide_s at >= slide_speed m/s) and the evidence.

The declaration (configs/contact_audit.json) also freezes the validation: the labelled contacts (the ten Straw Bale
downhill contacts of straw-brain08-04/-06 and the straw-brain08-01 slide, confirmed on the videos, and the
minus-fast6-r4-02 garage-floor contact from the recorded height), the terminal impacts, and the false-positive count:
every other contact is looked up on the video (contact sheets) and counts as a false positive unless the frames show
the drone touching the ground or an object.

usage: python -m haltere.liftoff.contact_audit LOG.csv [LOG.csv ...] [--config configs/contact_audit.json]
                                               [--json OUT.json]
       python -m haltere.liftoff.contact_audit validate --out DIR [--sheets] [--labels LABELS.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO/'configs'/'contact_audit.json'
AUDIT_VERSION = 1
G = 9.81
META_KEYS = ('frozen', 'frozen_at', 'sha256')


def content_sha256(obj):
    """Hash of a declaration without its freeze keys (canonical sorted, compact JSON; as configs/obstacles)."""
    body = {k: v for k, v in obj.items() if k not in META_KEYS}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode('utf-8')).hexdigest()


def load_config(path=CONFIG_PATH, *, require_frozen=True):
    """The audit declaration and its content hash; refuses an unfrozen or edited file and another version."""
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = content_sha256(config)
    if require_frozen and (config.get('frozen') is not True or config.get('sha256') != digest):
        raise ValueError(f'{path} is not a frozen contact-audit declaration, or it changed after the freeze')
    if config.get('version') != AUDIT_VERSION:
        raise ValueError(f'{path} declares contact-audit version {config.get("version")}; this code implements '
                         f'version {AUDIT_VERSION}')
    if set(config.get('audit') or {}) != set(DEFAULTS):
        raise ValueError(f'{path} must declare exactly the audit parameters {sorted(DEFAULTS)}')
    return config, digest


DEFAULTS = dict(thrust_twr=3.1378033647887618, thrust_exponent=1.9728633605611887,
                body_drag_s_inv=[0.02745813096840542, 0., 0.3490431637001165], half_frames=3, window_s=.3,
                drive_min=.4, drive_max=.8, support_on=1., rate_free=3., support_rate_gain=.5, arrest_descent=.3,
                arrest_accel=1.5, arrest_lookback_s=.1, impact_accel=20.,
                impact_offaxis=6., merge_s=.3, min_s=.15, slide_s=1., slide_speed=1., launch_plane_m=.3,
                arming_s=3.5, depart_m=.5)


def _rotations(q):
    w, x, y, z = np.asarray(q, float).T
    R = np.empty((len(w), 3, 3))
    R[:, 0, 0] = 1-2*(y*y+z*z)
    R[:, 0, 1] = 2*(x*y-z*w)
    R[:, 0, 2] = 2*(x*z+y*w)
    R[:, 1, 0] = 2*(x*y+z*w)
    R[:, 1, 1] = 1-2*(x*x+z*z)
    R[:, 1, 2] = 2*(y*z-x*w)
    R[:, 2, 0] = 2*(x*z-y*w)
    R[:, 2, 1] = 2*(y*z+x*w)
    R[:, 2, 2] = 1-2*(x*x+y*y)
    return R


def load_frames(path):
    """One row per telemetry frame (the game timestamp changed) of a visual-runner CSV: phase, ts, position,
    velocity, attitude and the game's processed throttle (in_thr; processed_thr when absent)."""
    import pandas as pd
    d = pd.read_csv(path, low_memory=False)
    if not len(d):
        return None
    keep = np.r_[True, np.diff(d['ts'].to_numpy(float)) != 0]
    d = d[keep].reset_index(drop=True)
    thr = d['in_thr'] if 'in_thr' in d else d['processed_thr']
    out = dict(phase=d['phase'].to_numpy(float), ts=d['ts'].to_numpy(float),
               P=d[['x', 'y', 'z']].to_numpy(float), V=d[['vx', 'vy', 'vz']].to_numpy(float),
               Q=d[['qw', 'qx', 'qy', 'qz']].to_numpy(float), processed=thr.to_numpy(float),
               W=np.nan_to_num(d[['omega_x', 'omega_y', 'omega_z']].to_numpy(float)))
    ok = np.isfinite(out['ts']) & np.isfinite(out['P']).all(1) & np.isfinite(out['V']).all(1) & \
        np.isfinite(out['Q']).all(1) & np.isfinite(out['processed'])
    return {k: v[ok] for k, v in out.items()}


def external_force(frames, c):
    """Per-frame external specific force (m/s^2, world frame) and its parts: the measured acceleration, the thrust
    model's acceleration and the off-axis part (flightlog.collisions)."""
    t, V, R = frames['ts'], frames['V'], _rotations(frames['Q'])
    k = int(c['half_frames'])
    A = np.full_like(V, np.nan)
    if len(t) > 2*k:
        A[k:-k] = (V[2*k:]-V[:-2*k])/np.maximum(t[2*k:]-t[:-2*k], 1e-3)[:, None]
    drive = np.clip((frames['processed']+1)/2, 0, 1)
    up = R[:, :, 2]
    thrust = G*c['thrust_twr']*drive**c['thrust_exponent']
    # mean thrust over the same frames as the difference quotient
    if len(t) > 2*k:
        kernel = np.ones(2*k+1)/(2*k+1)
        thrust_m = np.convolve(thrust, kernel, mode='same')
    else:
        thrust_m = thrust
    body_v = np.einsum('nji,nj->ni', R, V)
    drag = np.einsum('nij,nj->ni', R, np.asarray(c['body_drag_s_inv'], float)*body_v)
    specific = A+np.array([0., 0., G])
    external = specific+drag-thrust_m[:, None]*up
    along = np.einsum('ni,ni->n', specific, up)
    offaxis = np.linalg.norm(specific-along[:, None]*up, axis=1)
    # 3-frame mean acceleration magnitude and off-axis force, as flightlog.collisions
    A1 = np.full_like(V, np.nan)
    A1[1:] = np.diff(V, axis=0)/np.maximum(np.diff(t), 1e-3)[:, None]
    A3 = np.stack([np.convolve(np.nan_to_num(A1[:, i]), np.ones(3)/3, mode='same') for i in range(3)], axis=1)
    spec3 = A3+np.array([0., 0., G])
    along3 = np.maximum(np.einsum('ni,ni->n', spec3, up), 0.)
    off3 = np.linalg.norm(spec3-along3[:, None]*up, axis=1)
    return dict(A=A, drive=drive, external=external, offaxis=offaxis, drag=drag, accel3=np.linalg.norm(A3, axis=1),
                offaxis3=off3, up=up)


def _centred_mean(x, t, window):
    """Mean of x over frames within +-window/2 of each frame (NaN-aware: NaN if any NaN inside)."""
    out = np.full(len(x), np.nan)
    lo = np.searchsorted(t, t-window/2, 'left')
    hi = np.searchsorted(t, t+window/2, 'right')
    csum = np.r_[0., np.cumsum(np.nan_to_num(x))]
    nsum = np.r_[0, np.cumsum(~np.isfinite(x))]
    n = hi-lo
    ok = (n > 0) & (nsum[hi]-nsum[lo] == 0)
    out[ok] = (csum[hi]-csum[lo])[ok]/n[ok]
    return out


def _window_all(mask, t, window, before=0.):
    """True where every frame from t-window/2-before to t+window/2 satisfies mask."""
    lo = np.searchsorted(t, t-window/2-before, 'left')
    hi = np.searchsorted(t, t+window/2, 'right')
    csum = np.r_[0, np.cumsum(~mask)]
    return (csum[hi]-csum[lo]) == 0


def _window_max(x, t, window):
    """Maximum of x over frames within +-window/2."""
    lo = np.searchsorted(t, t-window/2, 'left')
    hi = np.searchsorted(t, t+window/2, 'right')
    return np.array([x[a:b].max() if b > a else np.nan for a, b in zip(lo, hi)])


def audit_frames(frames, c, sidecar=None):
    """Contacts of one flight (see the module docstring) and the per-frame evidence."""
    c = dict(DEFAULTS, **(c or {}))
    t, P, V = frames['ts'], frames['P'], frames['V']
    phase = frames['phase']
    f = external_force(frames, c)
    ext_z = f['external'][:, 2]
    departed = np.maximum.accumulate(np.abs(P[:, 2]) > c['depart_m'])
    scored = departed & (phase >= c['arming_s'])
    valid = _window_all((f['drive'] >= c['drive_min']) & (f['drive'] <= c['drive_max']), t, c['window_s'])
    support_mean = _centred_mean(ext_z, t, c['window_s'])
    rate = _window_max(np.linalg.norm(frames['W'], axis=1), t, c['window_s'])
    threshold = c['support_on']+c['support_rate_gain']*np.maximum(0., rate-c['rate_free'])
    support = scored & valid & (support_mean >= threshold)
    # arrest with low throttle: sub-hover drive over the window, a descent at its start, upward acceleration after
    hover = (1/c['thrust_twr'])**(1/c['thrust_exponent'])
    sub_hover = _window_all(f['drive'] <= hover, t, c['window_s'], c['arrest_lookback_s'])
    net_up = _centred_mean(np.nan_to_num(f['A'][:, 2], nan=np.nan)+f['drag'][:, 2], t, c['window_s'])
    lo = np.searchsorted(t, t-c['window_s']/2, 'left')
    descending = V[lo, 2] <= -c['arrest_descent']
    arrest = scored & sub_hover & descending & (net_up >= c['arrest_accel'])
    impact = scored & (f['accel3'] > c['impact_accel']) & (f['offaxis3'] > c['impact_offaxis'])
    evidence = support | arrest | impact
    contacts = []
    idx = np.flatnonzero(evidence)
    groups = []
    for i in idx:
        if groups and t[i]-t[groups[-1][-1]] <= c['merge_s']:
            groups[-1].append(i)
        else:
            groups.append([i])
    launch_z = 0.
    for g in groups:
        g = np.asarray(g)
        a, b = g[0], g[-1]
        span = np.arange(a, b+1)
        sustained = (support[span] | arrest[span])
        sustained_s = float(np.sum(np.diff(t[span], append=t[b])*sustained))
        has_impact = bool(impact[span].any())
        if sustained_s < c['min_s'] and not has_impact:
            continue
        dts = np.diff(t[span], append=t[b])
        speed = np.hypot(V[span, 0], V[span, 1])
        kind = 'impact' if has_impact and sustained_s < c['min_s'] else (
            'slide' if sustained_s >= c['slide_s'] and np.median(speed) >= c['slide_speed'] else 'support')
        ext = np.linalg.norm(f['external'][span], axis=1)
        contacts.append(dict(
            t_start=round(float(phase[a]), 3), t_end=round(float(phase[b]), 3),
            duration_s=round(float(phase[b]-phase[a]), 3), kind=kind,
            pos=[round(float(x), 2) for x in P[a]], speed=round(float(np.hypot(*V[a, :2])), 2),
            vz=round(float(V[a, 2]), 2), min_z=round(float(P[span, 2].min()), 2),
            launch_plane=bool(P[span, 2].min()-launch_z <= c['launch_plane_m']),
            peak_external=round(float(np.nanmax(ext)), 2),
            mean_vertical=round(float(np.nanmean(ext_z[span])), 2),
            impulse_vertical=round(float(np.nansum(np.maximum(ext_z[span], 0.)*dts)), 3),
            evidence=dict(support_s=round(float(np.sum(dts*support[span])), 3),
                          arrest_s=round(float(np.sum(dts*arrest[span])), 3), impact_frames=int(impact[span].sum())),
            drive=[round(float(f['drive'][span].min()), 3), round(float(f['drive'][span].max()), 3)]))
    terminal = None
    if sidecar and isinstance(sidecar.get('impact'), dict):
        imp = sidecar['impact']
        stamp = imp.get('timestamp')
        if stamp is not None and np.isfinite(stamp) and len(t):
            # the sidecar stamps the game timestamp; the CSV ends at or just before it
            k = int(np.clip(np.searchsorted(t, stamp), 0, len(t)-1))
            pos = P[-1]+V[-1]*max(0., float(stamp)-float(t[-1]))
            terminal = dict(t=round(float(phase[k]+(stamp-t[k])), 3), kind='terminal',
                            pos=[round(float(x), 2) for x in pos], speed=round(float(np.linalg.norm(V[-1])), 2),
                            vz=round(float(V[-1, 2]), 2), acceleration=imp.get('acceleration_mps2'),
                            unexplained=imp.get('unexplained_mps2'))
    scored_s = float(np.sum(np.diff(t, append=t[-1])*scored)) if len(t) else 0.
    return dict(contacts=contacts, terminal_impact=terminal, scored_s=round(scored_s, 2),
                contacts_per_min=round(len(contacts)/max(scored_s/60, 1e-9), 3) if scored_s else None,
                frames=int(len(t))), dict(t=t, phase=phase, support_mean=support_mean, valid=valid, support=support,
                                          arrest=arrest, impact=impact, scored=scored, ext_z=ext_z,
                                          threshold=threshold, rate=rate)


def audit_log(path, config=None):
    """Audit one CSV (its .json sidecar, when present, adds the terminal impact)."""
    path = Path(path)
    frames = load_frames(path)
    sidecar_path = path.with_suffix('.json')
    sidecar = json.loads(sidecar_path.read_text(encoding='utf-8')) if sidecar_path.exists() else None
    if frames is None or len(frames['ts']) < 10:
        return dict(log=path.stem, contacts=[], terminal_impact=None, scored_s=0., frames=0), None
    result, detail = audit_frames(frames, config, sidecar)
    return dict(log=path.stem, **result), detail


def describe(result):
    lines = [f"{result['log']}: {len(result['contacts'])} contacts in {result['scored_s']/60:.1f} min scored"
             + (f"; terminal impact at {result['terminal_impact']['t']:.2f} s" if result.get('terminal_impact') else '')]
    for c in result['contacts']:
        lines.append(f"  {c['t_start']:8.2f}-{c['t_end']:8.2f} {c['kind']:8s} at ({c['pos'][0]:.1f}, {c['pos'][1]:.1f}, "
                     f"{c['pos'][2]:.2f}) {c['speed']:.1f} m/s vz {c['vz']:+.2f}; peak {c['peak_external']:.1f} m/s^2, "
                     f"vertical impulse {c['impulse_vertical']:.2f} m/s, min z {c['min_z']:.2f}"
                     + (' [launch plane]' if c['launch_plane'] else ''))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------------------
# Validation (frozen in the declaration's 'validation' block before any validation run)
# ---------------------------------------------------------------------------------------------
RUNS = Path('C:/DEV/Haltere/runs/fast-stack-20260923')


def logs_of(runs=RUNS, min_frames=100):
    """The log stems of a runs folder whose CSV has at least min_frames rows (sorted)."""
    out = []
    for path in sorted(Path(runs).glob('*.csv')):
        with open(path, encoding='utf-8', errors='replace') as f:
            rows = sum(1 for _ in f)-1
        if rows >= min_frames:
            out.append(path.stem)
    return out


def _overlaps(contact, lo, hi):
    return contact['t_start'] <= hi and contact['t_end'] >= lo


def _coverage(contacts, lo, hi):
    """Share of [lo, hi] covered by the union of the contacts' [t_start, t_end]."""
    spans = sorted((max(lo, c['t_start']), min(hi, c['t_end'])) for c in contacts if _overlaps(c, lo, hi))
    covered, end = 0., lo
    for a, b in spans:
        a = max(a, end)
        if b > a:
            covered, end = covered+b-a, b
    return covered/(hi-lo) if hi > lo else 0.


def _log_end(log, runs=RUNS):
    """(last logged phase, whether the run ended on an impact) of a log."""
    import pandas as pd
    d = pd.read_csv(Path(runs)/f'{log}.csv', usecols=['phase'], low_memory=False)
    side = Path(runs)/f'{log}.json'
    reason = json.loads(side.read_text(encoding='utf-8')).get('stop_reason') if side.exists() else ''
    return float(d['phase'].iloc[-1]), 'Impact' in str(reason or '')


def review_items(results, validation, runs=RUNS):
    """Contacts that need a video label for the false-positive count: outside every labelled window, not within
    terminal_s of the end of a run that ended on an impact, and (Minus Two) not confirmed by height. Returns (items,
    known) where known lists the contacts accounted for otherwise (with the reason)."""
    v = validation
    labelled = {}
    for log, onset in v['straw_downhill']['cases']:
        lo, hi = v['straw_downhill']['window_s']
        labelled.setdefault(log, []).append((onset+lo, onset+hi, 'straw_downhill'))
    s = v['slide']
    labelled.setdefault(s['log'], []).append((s['window'][0], s['window'][1], 'slide'))
    for f in v['floor']['windows']:
        labelled.setdefault(v['floor']['log'], []).append((f[0], f[1], 'floor'))
    items, known = [], []
    for r in results:
        if not r['contacts']:
            continue
        end, impact = _log_end(r['log'], runs)
        for c in r['contacts']:
            reason = next((kind for lo, hi, kind in labelled.get(r['log'], []) if _overlaps(c, lo, hi)), None)
            if reason is None and impact and c['t_end'] >= end-v['terminal_s']:
                reason = 'terminal'
            if (reason is None and any(r['log'].startswith(p) for p in v['height_confirmed_prefixes'])
                    and c['min_z'] <= v['height_confirmed_m']):
                reason = 'height'
            entry = dict(log=r['log'], t_start=c['t_start'], t_end=c['t_end'], kind=c['kind'], pos=c['pos'],
                         min_z=c['min_z'])
            if reason is None:
                items.append(entry)
            else:
                known.append(dict(entry, reason=reason))
    return items, known


def review_sheets(items, out_dir, runs=RUNS, frames=12, before=.4, after=.2, offset=.1):
    """One contact sheet per review item (game view crop, `frames` frames from t_start - before to t_end + after,
    evenly spaced; the video starts about `offset` s before the log's phase). Needs ffmpeg and OpenCV."""
    import subprocess
    import tempfile
    import cv2
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in items:
        times = np.linspace(item['t_start']-before, item['t_end']+after, frames)
        tiles = []
        with tempfile.TemporaryDirectory() as tmp:
            for t in times:
                dst = Path(tmp)/f'{t:.2f}.png'
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{max(0., t+offset):.3f}', '-i',
                                str(Path(runs)/f"{item['log']}.mp4"), '-frames:v', '1', '-vf',
                                'crop=1288:720:640:0,scale=400:224', str(dst)], check=False)
                img = cv2.imread(str(dst))
                img = np.zeros((224, 400, 3), np.uint8) if img is None else img
                cv2.rectangle(img, (0, 0), (120, 22), (0, 0, 0), -1)
                cv2.putText(img, f't={t:.2f}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1)
                tiles.append(img)
        rows = [np.hstack(tiles[i:i+6]) for i in range(0, len(tiles), 6)]
        path = out_dir/f"{item['log']}_{item['t_start']:.2f}.jpg"
        cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
        paths.append(str(path))
    return paths


def score_validation(results, validation, labels=None, runs=RUNS):
    """The frozen validation gates on audit results of every log (see the declaration's 'validation' block).
    `labels`: {log: {"t_start": "contact" | "none" | "ambiguous"}} video labels of the review items (a missing label
    counts as a false positive, as 'none' and 'ambiguous' do)."""
    v = validation
    by_log = {r['log']: r for r in results}
    out = {}
    # the labelled Straw Bale downhill contacts
    lo, hi = v['straw_downhill']['window_s']
    rows = []
    for log, onset in v['straw_downhill']['cases']:
        hits = [c for c in by_log[log]['contacts'] if _overlaps(c, onset+lo, onset+hi)]
        rows.append(dict(log=log, onset=onset, detected=bool(hits),
                         contacts=[dict(t_start=c['t_start'], t_end=c['t_end'], kind=c['kind'],
                                        peak_external=c['peak_external'], impulse_vertical=c['impulse_vertical'])
                                   for c in hits]))
    detected = sum(r['detected'] for r in rows)
    out['straw_downhill'] = dict(cases=rows, detected=detected, of=len(rows),
                                 passed=detected >= v['straw_downhill']['min_detected'])
    # the straw-brain08-01 slide
    s = v['slide']
    contacts = by_log[s['log']]['contacts']
    inside = [c for c in contacts if _overlaps(c, *s['window'])]
    coverage = _coverage(contacts, *s['window'])
    kinds = sorted({c['kind'] for c in inside})
    out['slide'] = dict(window=s['window'], coverage=round(coverage, 3), kinds=kinds,
                        contacts=[dict(t_start=c['t_start'], t_end=c['t_end'], kind=c['kind']) for c in inside],
                        passed=coverage >= s['min_coverage'] and s['kind'] in kinds)
    # the minus-fast6-r4-02 floor contacts (the first is the gate, the others are reported)
    f = v['floor']
    contacts = by_log[f['log']]['contacts']
    floors = []
    for window in f['windows']:
        hits = [c for c in contacts if _overlaps(c, *window)]
        floors.append(dict(window=window, detected=bool(hits),
                           contacts=[dict(t_start=c['t_start'], t_end=c['t_end'], kind=c['kind'], min_z=c['min_z'])
                                     for c in hits]))
    out['floor'] = dict(windows=floors, passed=floors[0]['detected'])
    # terminal impacts: every run that ended on the runner's impact is listed with its terminal record; whether the
    # CSV itself carries contact evidence in its last terminal_s is reported
    terminal = []
    for r in results:
        end, impact = _log_end(r['log'], runs)
        if not impact:
            continue
        evidence = [c for c in r['contacts'] if c['t_end'] >= end-v['terminal_s']]
        terminal.append(dict(log=r['log'], listed=r.get('terminal_impact') is not None,
                             terminal=r.get('terminal_impact'), csv_evidence=bool(evidence),
                             evidence=[dict(t_start=c['t_start'], t_end=c['t_end'], kind=c['kind']) for c in evidence]))
    out['terminal'] = dict(impacts=terminal, listed=sum(t['listed'] for t in terminal), of=len(terminal),
                           csv_evidence=sum(t['csv_evidence'] for t in terminal),
                           passed=bool(terminal) and all(t['listed'] for t in terminal))
    # false positives: every review item is video-labelled (or a seeded sample of them)
    items, known = review_items(results, v, runs)
    fp = v['false_positives']
    sample = items
    if len(items) > fp['max_review']:
        rng = np.random.default_rng(fp['seed'])
        pick = sorted(rng.choice(len(items), fp['max_review'], replace=False))
        sample = [items[i] for i in pick]
    labels = labels or {}
    judged = []
    for item in sample:
        label = labels.get(item['log'], {}).get(f"{item['t_start']:.2f}")
        judged.append(dict(item, label=label, false_positive=label != 'contact'))
    minutes = sum(r['scored_s'] for r in results)/60.
    false = sum(j['false_positive'] for j in judged)
    estimate = false*(len(items)/len(sample) if sample else 0.)
    rate = estimate/minutes if minutes else None
    out['false_positives'] = dict(review_items=len(items), reviewed=len(sample), labelled=sum(j['label'] is not None
                                                                                            for j in judged),
                                  false_positives=false, estimated_false_positives=round(estimate, 2),
                                  scored_minutes=round(minutes, 2),
                                  per_minute=None if rate is None else round(rate, 4), judged=judged, known=known,
                                  passed=rate is not None and rate <= fp['max_per_min'])
    out['contacts_total'] = sum(len(r['contacts']) for r in results)
    out['passed'] = {k: out[k]['passed'] for k in ('straw_downhill', 'slide', 'floor', 'terminal', 'false_positives')}
    return out


def support_recall(results, runs=RUNS, window=(-2., .3)):
    """Report: the logged support-climb onsets of every log (the pilot's own contact evidence, pilot-dependent) and
    whether an audit contact overlaps [onset + window[0], onset + window[1]]."""
    import pandas as pd
    rows = []
    for r in results:
        d = pd.read_csv(Path(runs)/f"{r['log']}.csv", usecols=['phase', 'pilot_state'], low_memory=False)
        s = (d['pilot_state'] == 'support_climb').to_numpy()
        onsets = np.flatnonzero(s & ~np.r_[False, s[:-1]])
        for i in onsets:
            t = float(d['phase'].iloc[i])
            hit = any(_overlaps(c, t+window[0], t+window[1]) for c in r['contacts'])
            rows.append(dict(log=r['log'], onset=round(t, 2), audit_contact=hit))
    return dict(onsets=len(rows), with_audit_contact=sum(r['audit_contact'] for r in rows), rows=rows)


def validate_main(argv):
    parser = argparse.ArgumentParser(description='Run the frozen contact-audit validation over every log')
    parser.add_argument('--config', default=str(CONFIG_PATH))
    parser.add_argument('--runs', default=str(RUNS))
    parser.add_argument('--out', required=True, help='output folder (audit results, review sheets, scores)')
    parser.add_argument('--labels', default=None, help='video labels of the review items (JSON); without it the '
                                                       'review items and their sheets are written and not scored')
    parser.add_argument('--sheets', action='store_true', help='write the contact sheets of the review items')
    args = parser.parse_args(argv)
    config, digest = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cached = out/'audit_results.json'
    if cached.exists() and json.loads(cached.read_text(encoding='utf-8')).get('config_sha256') == digest:
        results = json.loads(cached.read_text(encoding='utf-8'))['results']
    else:
        results = []
        for log in logs_of(args.runs, config['validation']['min_frames']):
            result, _ = audit_log(Path(args.runs)/f'{log}.csv', config['audit'])
            results.append(result)
            print(describe(result), flush=True)
        cached.write_text(json.dumps(dict(config=str(args.config), config_version=config['version'],
                                          config_sha256=digest, results=results), indent=1), encoding='utf-8')
    items, known = review_items(results, config['validation'], args.runs)
    (out/'review_items.json').write_text(json.dumps(dict(items=items, known=known), indent=1), encoding='utf-8')
    print(f'{len(items)} review items, {len(known)} accounted for by labels, terminal impacts or height', flush=True)
    if args.sheets:
        fp = config['validation']['false_positives']
        sample = items
        if len(items) > fp['max_review']:
            rng = np.random.default_rng(fp['seed'])
            sample = [items[i] for i in sorted(rng.choice(len(items), fp['max_review'], replace=False))]
        review_sheets(sample, out/'sheets', args.runs)
    if args.labels:
        labels = json.loads(Path(args.labels).read_text(encoding='utf-8'))
        scores = score_validation(results, config['validation'], labels, args.runs)
        scores.update(config_sha256=digest, config_version=config['version'], labels=str(args.labels),
                      support_recall_report=support_recall(results, args.runs))
        (out/'validation_scores.json').write_text(json.dumps(scores, indent=1), encoding='utf-8')
        print(json.dumps(scores['passed']), flush=True)


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == 'validate':
        return validate_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('logs', nargs='+')
    parser.add_argument('--config', default=str(CONFIG_PATH))
    parser.add_argument('--json', default=None)
    args = parser.parse_args(argv)
    config, digest = load_config(args.config)
    results = []
    for log in args.logs:
        result, _ = audit_log(log, config['audit'])
        results.append(result)
        print(describe(result), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(config=str(args.config), config_version=config['version'],
                                                   config_sha256=digest, results=results), indent=1),
                                   encoding='utf-8')


if __name__ == '__main__':
    main()
