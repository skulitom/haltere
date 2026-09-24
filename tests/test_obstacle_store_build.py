"""Frame-store builder: selection rules, clocks, planning and an end-to-end build on synthetic sources."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from haltere.obstacles import splits, store, store_build as sb
from haltere.obstacles.store import Flag, FrameStore, Grade, PoseMethod, Source

FFMPEG = shutil.which('ffmpeg')


# ----------------------------------------------------------------------------- unit rules

def test_video_new_frames_use_the_last_repeated_copy():
    dg = np.array([99, 0.0, 5, 5, 0.05, 0.0, 7], float)     # frames 1, 4, 5 repeat their predecessor
    idx, last, t = sb.video_new_frames(dg, fps=10.0)
    assert idx.tolist() == [0, 2, 3, 6]
    assert last.tolist() == [1, 2, 5, 6]
    assert np.allclose(t, [0.1, 0.2, 0.5, 0.6])


def test_select_keeps_stride_grid_dense_windows_and_flags_extras():
    t = np.arange(60) / 15.0                                 # 15 Hz source
    why = np.full(60, '', object)
    why[:5] = 'countdown'
    events = [dict(t_wall=float(t[50]), store_event_id=7, kind='contact')]
    windows = [dict(start_phase_s=1.0, end_phase_s=1.2)]
    sel = sb.select(why, t, t, events, windows, stride=3)
    assert sel['step'] == 3 and abs(sel['rate'] - 15) < 1e-6
    valid = np.flatnonzero(why == '')
    assert set(valid[::3]) <= set(np.flatnonzero(sel['keep']))
    assert not sel['keep'][:5].any()
    pre = (t <= t[50]) & (t[50] - t <= store.PRE_EVENT_DENSE_S) & (why == '')
    assert sel['keep'][pre].all()                             # every valid frame within 6 s before the event
    assert sel['keep'][(t >= 1.0) & (t <= 1.2)].all()         # every clean-window frame
    assert (sel['extra'] == (sel['keep'] & ~np.isin(np.arange(60), valid[::3]))).all()
    assert sel['eid'][50] == 7 and np.isinf(sel['tti'][51]) and sel['eid'][51] == -1


def test_grid_step_keeps_slow_sources_above_2_5_hz():
    assert sb.grid_step(np.arange(50) / 18.0, 3)[0] == 3
    assert sb.grid_step(np.arange(50) / 7.2, 3)[0] == 2
    assert sb.grid_step(np.arange(50) / 4.9, 3)[0] == 1


def _telemetry(n=500, dt=0.01, launch_at=1.0, speed=2.0, reset_at=None):
    t = 1000.0 + np.arange(n) * dt
    ts = 50.0 + np.arange(n) * dt
    x = np.where(t - t[0] > launch_at, speed * (t - t[0] - launch_at), 0.0)
    vx = np.where(t - t[0] > launch_at, speed, 0.0)
    if reset_at is not None:
        k = int(reset_at / dt)
        ts[k:] = 50.0 + (np.arange(n - k)) * dt
        x[k:] = 0.0
        vx[k:] = 0.0
    P = np.stack([x, np.zeros(n), np.zeros(n)], 1)
    V = np.stack([vx, np.zeros(n), np.zeros(n)], 1)
    Q = np.tile([1.0, 0, 0, 0], (n, 1))
    return sb.Telemetry(t, ts, P, Q, V, np.zeros((n, 3)), np.arange(n) * dt)


def test_drop_reasons_countdown_post_event_and_crash_reset():
    tel = _telemetry(n=1000, reset_at=6.0)
    t = 1000.0 + np.arange(0, 10.5, 0.1)
    events = [dict(kind='contact', t_wall=1004.0, store_event_id=0)]
    why = sb.drop_reasons(t, tel, events, repeat=np.zeros(len(t), bool), menu=np.zeros(len(t), bool),
                          hud=np.ones(len(t), bool))
    rel = t - 1000.0
    assert (why[rel < 0.85] == 'countdown').all()
    assert (why[(rel > 4.0) & (rel < 5.95)] == 'post_event').all()      # contact then a reset within 10 s
    assert (why[(rel >= 6.05) & (rel < 6.85)] == 'countdown').all()     # the post-reset countdown
    assert (why[(rel > 1.0) & (rel < 4.0)] == '').all()
    assert (why[rel > 10.0] == 'outside_telemetry').all() and (rel > 10.0).any()


def test_capture_telemetry_drops_results_screen_rows(tmp_path):
    rows = ['recv_time,timestamp,px,py,pz,qx,qy,qz,qw,vx,vy,vz']
    for i in range(60):
        live = i < 40                                        # then the results screen: all-zero position
        rows.append(f'{100 + i * 0.03:.3f},{5 + i * 0.03:.3f},{(1 + i * 0.1) if live else 0},{2 if live else 0},'
                    f'{3 if live else 0},0,0,0,1,0,0,{3 if live else 0}')
    (tmp_path / 'telemetry.csv').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    tel = sb.load_capture_telemetry(tmp_path, origin_sim=[3.0, -1.0, 2.0])
    assert len(tel) == 40
    s = tel.sample([100.0 + 45 * 0.03])
    assert s['outside'][0]                                  # frames over the results screen get no pose
    # Unity (x, y, z) = (1, 2, 3) is simulator (3, -1, 2): launch-relative zero at the origin
    assert np.allclose(tel.pos[0], [0, 0, 0], atol=1e-9)


def test_recorder_sets_without_telemetry_keep_every_attempt():
    # three attempts (game resets at frames 20 and 40), each launching 3 frames after its start
    n = 60
    t = 1000.0 + np.arange(n) * 0.14
    ts = np.concatenate([2 + np.arange(20) * 0.14] * 3)
    x = np.concatenate([np.maximum(np.arange(20) - 3, 0) * 0.5] * 3)
    pos = np.stack([x, np.zeros(n), np.zeros(n)], 1)
    why = sb._drops_without_telemetry(t, pos, ts, np.zeros(n, bool), np.zeros(n, bool))
    for a in (0, 20, 40):
        assert (why[a + 5:a + 19] == '').all()                         # every attempt keeps its flight
    assert why[19] == 'outside_telemetry' and why[20] == 'outside_telemetry'
    assert (why[21:24] == 'countdown').all() and (why[0:4] == 'countdown').all()


def test_telemetry_wall_at_ts_uses_the_attempt_of_the_frame():
    tel = _telemetry(n=1000, reset_at=5.0)
    # game time 51.0 exists in both attempts (1001.0 and 1006.0 on the wall clock)
    got = tel.wall_at_ts([51.0, 51.0], near_wall=[1001.2, 1006.3])
    assert np.allclose(got, [1001.0, 1006.0], atol=1e-6)


def test_load_run_csv_receipt_clock_and_first_row_wall(tmp_path):
    n = 300
    ft = 5000.0 + np.arange(n) * 0.01
    wall = 1.7e9 + ft - 5000.0 + 0.004 + 0.002 * (np.arange(n) % 3)     # write latency 4-8 ms
    rows = ['wall,ts,x,y,z,vx,vy,vz,qw,qx,qy,qz,omega_x,omega_y,omega_z,phase,frame_time,capture_time,cue_u,cue_v,cue_edge']
    for k in range(n):
        x = 'nan' if k == 0 else f'{k * 0.01:.4f}'           # first row has no pose: wall0 must still be row 0
        rows.append(f'{wall[k]:.6f},{10 + k * 0.01:.4f},{x},0,0,1,0,0,1,0,0,0,0,0,0,{k * 0.01:.3f},{ft[k]:.6f},'
                    f'{ft[k] - 0.02:.6f},{0.5 if k % 2 else -1},{0.25},False')
    p = tmp_path / 'flight.csv'
    p.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    wall_clock = sb.load_run_csv(p, clock='wall')
    assert wall_clock.clock == 'wall' and wall_clock.wall0 == pytest.approx(wall[0])
    rec = sb.load_run_csv(p, clock='receipt')
    assert rec.clock == 'receipt'
    C = rec.epoch_offset
    assert C == pytest.approx(1.7e9 - 5000.0 + 0.004, abs=1e-4)
    assert np.allclose(rec.wall, ft[1:] + C, rtol=0, atol=1e-6)
    s = rec.sample([ft[100] + C])
    assert s['pos'][0, 0] == pytest.approx(1.0, abs=1e-6)
    cue = rec.cue_at(np.array([ft[101] - 0.02 + C, ft[100] - 0.02 + C]))
    assert np.allclose(cue[0], [0.5, 0.25]) and np.isnan(cue[1]).all()   # -1 = no ring


def test_video_offset_prefers_a_set_used_offset():
    assert sb.video_offset({'used_offset_s': None, 'offset_after_first_row_s': -0.03}) == (-0.03, 'offset_after_first_row_s')
    assert sb.video_offset({'used_offset_s': 1.2, 'offset_after_first_row_s': -0.03}) == (1.2, 'used_offset_s')
    assert sb.video_offset({}) == (None, None)


def test_repose_rows_shifts_time_and_reinterpolates_pose():
    tel = _telemetry()
    rows = store.empty_index(3)
    rows['t_wall'] = [1002.0, 1002.5, 1003.0]
    rows['flags'] = int(Flag.HAS_CSV | Flag.DENSE_EXTRA)
    events = [dict(t_wall=1003.02, store_event_id=4)]
    out = sb.repose_rows(rows, tel, 0.03, events, [dict(start_phase_s=1.9, end_phase_s=2.6)])
    assert np.allclose(out['t_wall'], rows['t_wall'] + 0.03)
    assert np.allclose(out['pos'][:, 0], 2.0 * (out['t_wall'] - 1000.0 - 1.0), atol=1e-5)
    assert np.allclose(out['timing_delta_s'], 0.03)
    assert (out['flags'] & int(Flag.TIMING_REFINED)).all() and (out['flags'] & int(Flag.DENSE_EXTRA)).all()
    assert out['event_id'].tolist() == [4, 4, -1]
    assert ((out['flags'] & int(Flag.IN_CLEAN)) != 0).tolist() == [True, True, False]
    again = sb.repose_rows(out, tel, 0.01, events, [])                  # deltas replace, never accumulate
    assert np.allclose(again['t_wall'], rows['t_wall'] + 0.01)


def test_repose_capture_rows_compensates_the_logged_pose_lag(tmp_path):
    ds = tmp_path / 'old_set'
    ds.mkdir()
    rows = ['file,wall_time,t,dist,thr,roll,pitch,yaw,px,py,pz,tx,ty,tz,waypoint,n_waypoints,qw,qx,qy,qz,ts']
    for i in range(20):
        rows.append(f'{i:06d}.jpg,{1000 + i * 0.1:.4f},0,0,0,0,0,0,{i * 0.5:.3f},0,0,0,0,0,0,0,1,0,0,0,{3 + i * 0.1:.3f}')
    (ds / 'index.csv').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    tel, idx = sb.capture_index_telemetry(ds)
    assert len(idx) == 20 and tel.clock == 'index'
    stored = store.empty_index(2)
    stored['t_wall'] = [1000.5, 1001.0]
    stored['pos'] = [[2.5, 0, 0], [5.0, 0, 0]]
    stored['pose_method'] = int(PoseMethod.SOURCE_RAW)
    out = sb.repose_capture_rows(stored, tel, 0.06)
    assert np.allclose(out['pos'][:, 0], [2.8, 5.3])                   # 5 m/s logged motion, 60 ms later
    assert np.allclose(out['t_wall'], stored['t_wall'])                # the logged time is kept
    assert (out['pose_method'] == PoseMethod.SOURCE_COMPENSATED).all() and np.allclose(out['pose_lag_s'], 0.06)
    assert (out['flags'] & int(Flag.TIMING_REFINED)).all()
    assert np.allclose(out['t_game'], [3.56, 4.06])


# ----------------------------------------------------------------------------- end-to-end synthetic build

W0 = 1.79e9
FPS = 18.0
OFFSET = 0.25
IMPACT_S = 3.2


def _texture(seed, h, w):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    import cv2
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _write_video(path, n_frames, repeats):
    base = _texture(1, 720, 1600)
    frames, cap = [], 0
    for i in range(n_frames):
        if i not in repeats:
            cap += 1
        img = base[:, cap * 4:cap * 4 + 1280]
        frames.append(img)
    cmd = [FFMPEG, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', str(int(FPS)),
           '-i', '-', '-c:v', 'libx264rgb', '-qp', '0', str(path)]
    subprocess.run(cmd, input=b''.join(np.ascontiguousarray(f).tobytes() for f in frames), check=True)


def _write_flight_csv(path, seconds=4.0):
    n = int(seconds * 100)
    k = np.arange(n)
    ft = 7000.0 + k * 0.01
    wall = W0 + k * 0.01 + 0.003
    t = k * 0.01
    x = np.where(t > 1.0, 2.0 * (t - 1.0), 0.0)
    yaw = np.where(t > 1.0, 0.5 * (t - 1.0), 0.0)
    rows = ['wall,ts,x,y,z,vx,vy,vz,qw,qx,qy,qz,omega_x,omega_y,omega_z,phase,frame_time,capture_time,cue_u,cue_v,cue_edge']
    for i in range(n):
        rows.append(f'{wall[i]:.6f},{20 + t[i]:.5f},{x[i]:.5f},0,0.5,{2.0 if t[i] > 1 else 0},0,0,'
                    f'{np.cos(yaw[i] / 2):.8f},0,0,{np.sin(yaw[i] / 2):.8f},0,0,{0.5 if t[i] > 1 else 0},{t[i]:.3f},'
                    f'{ft[i]:.6f},{ft[i] - 0.015:.6f},0.4,0.6,False')
    path.write_text('\n'.join(rows) + '\n', encoding='utf-8')


def _write_capture(ds, n=72, rate=18.0, origin=(10.0, 20.0, 0.0)):
    import cv2
    (ds / 'frames').mkdir(parents=True)
    base = _texture(2, 360, 900)
    grab = W0 + 100.0 + np.arange(n) / rate
    tel_t = W0 + 100.0 - 0.2 + np.arange(int((n / rate + 0.5) * 50)) / 50.0
    rel_x = np.maximum(tel_t - (W0 + 100.5), 0.0) * 1.0             # launches 0.5 s after the first image
    tel_rows = ['recv_time,timestamp,px,py,pz,qx,qy,qz,qw,vx,vy,vz,gyro_pitch,gyro_roll,gyro_yaw,in_throttle,in_yaw,'
                'in_pitch,in_roll,batt_v,batt_pct,rpm_lf,rpm_rf,rpm_lb,rpm_rb']
    for i, tt in enumerate(tel_t):
        sim = np.array(origin) + [rel_x[i], 0.0, 0.0]
        unity = (-sim[1], sim[2], sim[0])                              # sim = M @ unity
        v = 1.0 if tt > W0 + 100.5 else 0.0
        tel_rows.append(f'{tt:.6f},{30 + i / 50:.5f},{unity[0]},{unity[1]},{unity[2]},0,0,0,1,0,0,{v},0,0,0,0,0,0,0,'
                        f'16,100,0,0,0,0')
    (ds / 'telemetry.csv').write_text('\n'.join(tel_rows) + '\n', encoding='utf-8')
    idx = ['file,wall_time,capture_end,telemetry_age_s,t,px,py,pz,qw,qx,qy,qz,ts,in_throttle,in_yaw,in_pitch,in_roll']
    for i in range(n):
        j = int(np.searchsorted(tel_t, grab[i]) - 1)
        name = f'{i:06d}.jpg'
        cv2.imwrite(str(ds / 'frames' / name), np.ascontiguousarray(base[:, i * 3:i * 3 + 640][..., ::-1]))
        idx.append(f'{name},{grab[i]:.6f},{grab[i] + 0.02:.6f},{grab[i] + 0.02 - tel_t[j]:.6f},{i / rate:.4f},'
                   f'{rel_x[j]:.5f},0,0,1,0,0,0,{30 + j / 50:.5f},0,0,0,0')
    (ds / 'index.csv').write_text('\n'.join(idx) + '\n', encoding='utf-8')
    (ds / 'capture.json').write_text(json.dumps(dict(origin_sim=list(origin), source='player_live', oracle_route=False,
                                                     pilot='human')), encoding='utf-8')


@pytest.mark.skipif(FFMPEG is None, reason='ffmpeg not installed')
def test_end_to_end_build_on_synthetic_sources(tmp_path):
    root = tmp_path / 'data'
    (root / 'runs' / 'rv').mkdir(parents=True)
    repeats = {3, 7, 8, 13, 20, 21, 22, 40}
    n_video = 63
    _write_video(root / 'runs' / 'rv' / 'flight.mp4', n_video, repeats)
    _write_flight_csv(root / 'runs' / 'rv' / 'flight.csv')
    (root / 'runs' / 'rv' / 'flight.json').write_text(json.dumps(dict(
        origin_sim=[1.0, 2.0, 3.0], impact=dict(timestamp=20 + IMPACT_S, acceleration_mps2=40.0), stop_reason='impact')))
    _write_capture(root / 'data' / 'vision' / 'cap1')
    inv = dict(
        root=str(root).replace('\\', '/'),
        flights=[dict(flight='rv/flight', environment='Minus Two', sources=[dict(kind='run_video', id='rv/flight')]),
                 dict(flight='vision:cap1', environment='Straw Bale',
                      sources=[dict(kind='capture_dataset', id='vision:cap1')]),
                 dict(flight='sealed/flight', environment='The Green',
                      sources=[dict(kind='run_video', id='sealed/flight')])],
        run_videos=[dict(id='rv/flight', video='runs/rv/flight.mp4', environment='Minus Two', usable=True, fps=FPS,
                         resolution='1280x720', files=dict(csv='runs/rv/flight.csv', json='runs/rv/flight.json'),
                         alignment=dict(quality='good', offset_after_first_row_s=OFFSET, used_offset_s=None),
                         controller=dict(control_mode='pilot-assisted visual fly brain')),
                    dict(id='sealed/flight', video='runs/sealed/flight.mp4', environment='The Green', usable=True,
                         files=dict(csv='runs/sealed/flight.csv'), alignment=dict(quality='good',
                                                                                 offset_after_first_row_s=0.0))],
        capture_datasets=[dict(id='vision:cap1', dataset='data/vision/cap1', environment='Straw Bale', usable=True,
                               counted_in_totals=True)],
        geometry_archives=[])
    inv_path = tmp_path / 'inventory.json'
    inv_path.write_text(json.dumps(inv), encoding='utf-8')
    folds = tmp_path / 'folds.json'
    splits.write_folds_config(inv_path, folds)
    out = tmp_path / 'store'
    lock = tmp_path / 'FLIGHT_LOCK'                                  # absent: no pause
    argv = ['build', '--inventory', str(inv_path), '--folds', str(folds), '--out', str(out), '--flight-lock', str(lock)]
    store.main(argv)
    s = FrameStore(out)
    assert [r['source_id'] for r in s.runs] == ['vision:cap1', 'rv/flight']       # env order; sealed skipped
    assert s.manifest['build']['data_root'] == str(root).replace('\\', '/')
    ix = s.index
    assert (ix['run_id'][:-1] <= ix['run_id'][1:]).all()

    # --- capture set: TELEMETRY_INTERP from the raw UDP log, stride 3 after the countdown
    cap = ix[ix['run_id'] == 0]
    assert (cap['source'] == Source.CAPTURE_DATASET).all() and (cap['grade'] == Grade.CAPTURE).all()
    assert (cap['pose_method'] == PoseMethod.TELEMETRY_INTERP).all()
    assert (cap['flags'] & int(Flag.HUMAN)).all()
    grab = W0 + 100.0 + np.arange(72) / 18.0
    launched = grab >= W0 + 100.5 - 0.1
    kept_t = np.sort(cap['t_wall'])
    assert np.allclose(kept_t, grab[launched][::3], rtol=0, atol=2e-6)
    assert np.allclose(cap['pos'][:, 0], np.maximum(kept_t - (W0 + 100.5), 0), atol=2e-3)
    assert s.runs[0]['grid_step'] == 3 and s.runs[0]['dropped']['countdown'] > 0
    assert np.allclose(s.absolute_pos(np.flatnonzero(ix['run_id'] == 0))[:, 1], 20.0)

    # --- video: t_wall = csv.wall[0] + last copy / fps + offset; pre-impact frames dense; post-impact dropped
    vid = ix[ix['run_id'] == 1]
    dg = np.full(n_video, 99.0)
    dg[list(repeats)] = 0.0
    idx, last, t_video = sb.video_new_frames(dg, FPS)
    t_all = (W0 + 0.003) + t_video + OFFSET
    launch = W0 + 0.003 + 1.01                                     # first CSV row with vx > 0.5
    impact = W0 + 0.003 + IMPACT_S
    expect = t_all[(t_all >= launch - 0.1 + 1e-6) & (t_all <= impact)]
    assert np.allclose(np.sort(vid['t_wall']), expect, atol=1e-6)
    assert (vid['pose_method'] == PoseMethod.TELEMETRY_INTERP).all() and (vid['grade'] == Grade.GOOD).all()
    assert np.allclose(vid['align_offset_s'], OFFSET)
    rel = vid['t_wall'] - (W0 + 0.003)
    assert np.allclose(vid['pos'][:, 0], np.maximum(2.0 * (rel - 1.0), 0), atol=1e-3)
    assert np.allclose(vid['t_phase'], rel, atol=1e-3)
    assert (vid['flags'] & int(Flag.PRE_EVENT)).all() and (vid['flags'] & int(Flag.HAS_CSV)).all()
    assert np.allclose(vid['tti_s'], impact - vid['t_wall'], atol=1e-4)
    ev = json.loads((out / 'events.json').read_text())['events']
    assert [e['kind'] for e in ev] == ['terminal_impact'] and ev[0]['run_id'] == 1
    assert (vid['event_id'] == ev[0]['store_event_id']).all()
    assert (vid['cue_src'] == store.CueSource.LOGGED).all() and np.allclose(vid['cue_uv'], [0.4, 0.6])
    assert s.runs[1]['dropped']['repeat'] == len(repeats) and s.runs[1]['dropped']['post_event'] > 0
    fr = np.load(out / 'parts' / 'r00001.frames.npz')
    assert np.allclose(fr['t_wall'], t_all) and fr['keep'].sum() == len(vid)

    # --- frames are the resized gameplay crop, bit-exact to an independent resize
    import cv2
    first = int(np.argmin(vid['t_wall']))
    k_new = int(np.argmin(np.abs(t_all - vid['t_wall'][first])))
    base = _texture(1, 720, 1600)
    cap_no = idx[k_new] + 1 - sum(1 for r in repeats if r <= idx[k_new])
    ref = cv2.resize(np.ascontiguousarray(base[:, cap_no * 4:cap_no * 4 + 1280]), (448, 252), interpolation=cv2.INTER_AREA)
    got = s.frame(np.flatnonzero(ix['run_id'] == 1)[first]).astype(int)
    assert np.abs(got - ref.astype(int)).mean() < 1.0

    rep = json.loads((out / 'build_report.json').read_text())['summary']
    assert rep['n_failed'] == 0 and rep['dense_incomplete_runs'] == []
    assert rep['folds']['F12']['test'] == len(vid)

    # --- resumable: a second build call finds every run done and rewrites the same index
    before = s.manifest['index_sha256']
    del s, ix                                                          # release the index memmap (Windows)
    import gc
    gc.collect()
    store.main(argv)
    assert FrameStore(out).manifest['index_sha256'] == before
    gc.collect()

    # --- repose: an accepted video delta shifts the clock; a run above the pose gate becomes UNRELIABLE
    vid_before = np.load(out / 'index.npy')
    vid_before = vid_before[vid_before['run_id'] == 1]
    (out / 'timing').mkdir()
    refined = dict(schema='haltere.obstacles.timing.v1', store_index_sha256=before,
                   runs={'1': dict(source_id='rv/flight', delta_s=0.02, accepted=True, residual_px_after=0.5)},
                   controls={'0': dict(source_id='vision:cap1', delta_s=0.0, accepted=False, residual_px_after=3.0)})
    (out / 'timing' / 'refined.json').write_text(json.dumps(refined), encoding='utf-8')
    store.main(['repose', '--store', str(out), '--flight-lock', str(lock)])
    s = FrameStore(out)
    assert s.manifest['index_sha256'] != before and (out / 'index.v1.npy').exists()
    v = s.index[s.index['run_id'] == 1]
    assert np.allclose(v['t_wall'], vid_before['t_wall'] + 0.02) and np.allclose(v['timing_delta_s'], 0.02)
    assert np.allclose(v['pos'][:, 0], np.maximum(2.0 * (v['t_wall'] - (W0 + 0.003) - 1.0), 0), atol=1e-3)
    assert (v['flags'] & int(Flag.TIMING_REFINED)).all() and (v['grade'] == Grade.GOOD).all()
    assert (s.index['grade'][s.index['run_id'] == 0] == Grade.UNRELIABLE).all()
    assert len(s.rows()) == len(v)                                     # UNRELIABLE rows are excluded by default
    assert s.runs[0]['pose_check']['status'] == 'failed' and s.runs[1]['pose_check']['status'] == 'passed'
    assert s.runs[1]['alignment']['refine_delta_s'] == 0.02
