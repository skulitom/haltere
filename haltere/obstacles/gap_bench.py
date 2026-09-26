"""Runtime bench G8 of the obstacle stack's gap cue (OFFLINE ONLY; never imported by the flight stack).

Runs the real flight camera process (`ProcessRetinaCamera`: GateNet on CUDA, retina, checkpoint-ring cue, looming,
and the gap cue in the declared placement) on a recorded 1280 x 720 flight video instead of the screen
(`haltere.liftoff.camera_replay`), with Liftoff idle. The bench process stands in for the controller: a
100 Hz loop publishes the recorded telemetry pose for the frame being served (the flight CSV, aligned by the
gap-cue evaluation's video offset), polls the camera like `visual_brain.run` and burns a declared brain-step
CPU load. Recorded telemetry only positions the frames; the camera sees nothing it would not see live.

Gates (configs/obstacles/gap_bench_gates.json, frozen before the first run): camera loop p95 and rate,
gap sample age p95 at the controller, checkpoint-cue latency increase over the matched baseline
(looming on, gap off) and fp16 vs fp32 depth.

Commands (``python -m haltere.obstacles.gap_bench <cmd> --out DIR --flight-lock PATH``):
  run      one condition (baseline | camera | process | camera2 | process2: placement and stride) -> DIR/<cond>.npz/.json
  fp16     fp16 vs fp32 relative disparity (336 x 602 input) on frames of the flight -> DIR/fp16.json
  parity   bench gap samples vs the offline gap-cue decision on the same recorded frames -> DIR/parity_<cond>.json
  score    the gates over the conditions run -> DIR/gates.json
GPU work is chunked (<= 300 s) behind ChunkGuard: pause above 70 C, hard stop at 80 C.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import threading
import time
from pathlib import Path

import numpy as np

from .thermal import ChunkGuard, limit_threads, require_flight_lock_path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATES_PATH = REPO_ROOT/'configs'/'obstacles'/'gap_bench_gates.json'
FLIGHT_DIR = Path('runs')/'fast-stack-20260923'
CONDITIONS = dict(baseline=None, camera=('camera', 1), process=('process', 1), camera2=('camera', 2),
                  process2=('process', 2))
PAUSE_C, RESUME_C, HARD_STOP_C, CHUNK_S = 70., 65., 80., 300.


def data_root():
    from .gap_cue_eval import data_root as root
    return Path(root())


def _log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_gates(path=GATES_PATH, require_frozen=True):
    from ..liftoff.gap_stack import config_sha256
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    sha = config_sha256(obj)
    if obj.get('frozen') and obj.get('sha256') != sha:
        raise RuntimeError(f'{path} changed after freezing')
    if require_frozen and not obj.get('frozen'):
        raise RuntimeError(f'{path} is not frozen')
    return obj, sha


class Telemetry:
    """Recorded pose by wall time (the flight CSV), for placing replayed frames."""

    def __init__(self, csv_path):
        import pandas as pd
        d = pd.read_csv(csv_path, usecols=['wall', 'ts', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz'])
        d = d.drop_duplicates('wall')
        self.wall = d['wall'].to_numpy(float)
        self.ts = d['ts'].to_numpy(float)
        self.pos = d[['x', 'y', 'z']].to_numpy(float)
        self.quat = d[['qw', 'qx', 'qy', 'qz']].to_numpy(float)
        self.vel = d[['vx', 'vy', 'vz']].to_numpy(float)

    def at(self, wall):
        i = int(np.clip(np.searchsorted(self.wall, wall), 1, len(self.wall)-1))
        a = float(np.clip((wall-self.wall[i-1])/max(self.wall[i]-self.wall[i-1], 1e-6), 0, 1))
        q0, q1 = self.quat[i-1], self.quat[i].copy()
        if q0 @ q1 < 0:
            q1 = -q1
        q = (1-a)*q0+a*q1
        q /= np.linalg.norm(q)
        lerp = lambda x: (1-a)*x[i-1]+a*x[i]
        return lerp(self.pos), q, lerp(self.vel), float(lerp(self.ts))


def flight_inputs(name, align_npz=None):
    root = data_root()
    base = root/FLIGHT_DIR
    sidecar = json.loads((base/f'{name}.json').read_text(encoding='utf-8'))
    sensor = dict(sidecar['gate_sensor'])
    sensor['checkpoint'] = str(root/Path(sensor['checkpoint'].replace('\\', '/')))
    contract = sidecar['motor_controller'].get('contract')
    offset, source = -0.03, 'default (gap_cue_eval fallback)'
    if align_npz and Path(align_npz).exists():
        z = np.load(align_npz, allow_pickle=True)
        offset, source = float(z['offset_s']), str(align_npz)
    return dict(video=str(base/f'{name}.mp4'), csv=str(base/f'{name}.csv'), sensor=sensor, contract=contract,
                offset=offset, offset_source=source, sidecar_camera=sidecar.get('camera_diagnostics', {}))


class BrainLoad:
    """A declared per-tick CPU load standing in for the controller's brain step (2 torch threads)."""

    def __init__(self, ms):
        import torch
        self.ms = float(ms)
        self.torch = torch
        self.a = torch.randn(256, 256)

    def __call__(self):
        end = time.perf_counter()+self.ms/1000
        while time.perf_counter() < end:
            self.a = self.torch.tanh(self.a @ self.a*1e-3)


BELOW_NORMAL_PRIORITY_CLASS = 0x4000


def priority_class(value=None):
    """This process's Windows priority class, set to `value` first when given (None elsewhere)."""
    import sys
    if sys.platform != 'win32':
        return None
    import ctypes
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetPriorityClass.argtypes = [ctypes.c_void_p]
    kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    handle = kernel.GetCurrentProcess()
    if value is not None and not kernel.SetPriorityClass(handle, value):
        raise ctypes.WinError()
    return int(kernel.GetPriorityClass(handle))


class GpuWatch(threading.Thread):
    """Polls the GPU temperature every 2 s off the timing loop; `hot` once it reaches the hard stop."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.hot, self.stop_flag = None, False, threading.Event()

    def run(self):
        from .thermal import gpu_temperature
        while not self.stop_flag.wait(2.):
            t = gpu_temperature()
            if t is not None:
                self.peak = t if self.peak is None else max(self.peak, t)
                self.hot = self.hot or t >= HARD_STOP_C


def cmd_run(args):
    from ..liftoff.camera_process import ProcessRetinaCamera
    from ..liftoff.gap_stack import GAP_PILOT_PATH, gap_spec, load_gap_pilot
    limit_threads(2, torch=True)
    guard = ChunkGuard(require_flight_lock_path(args.flight_lock), gpu=True, pause_c=PAUSE_C, resume_c=RESUME_C,
                       hard_stop_c=HARD_STOP_C, chunk_max_s=CHUNK_S, log=_log)
    gates, gates_sha = load_gates(require_frozen=True)
    protocol = gates['protocol']
    cond = args.condition
    placement = CONDITIONS[cond]
    inputs = flight_inputs(args.flight, args.align_npz)
    tel = Telemetry(inputs['csv'])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{args.flight}_{cond}' if args.tag is None else args.tag
    spec = None
    if placement is not None:
        declaration, digest = load_gap_pilot(GAP_PILOT_PATH, require_frozen=False)
        declaration = json.loads(json.dumps(declaration))
        declaration['runtime'].update(placement=placement[0], stride=placement[1])
        spec = gap_spec(declaration, inputs['contract'], inputs['sensor'], digest=digest)
    anchor = out/f'anchor_{tag}.txt'
    anchor.unlink(missing_ok=True)
    start, seconds = float(args.start), float(args.seconds)
    if seconds+protocol['warmup_s'] > CHUNK_S-60:
        raise SystemExit('Keep a run inside one GPU chunk')
    guard.before_chunk()
    watch = GpuWatch()
    watch.start()
    backend = f"replay:{inputs['video']}?start={start}&pad={protocol['capture_pad_ms']}&anchor={anchor}"
    launch = None
    if args.parent_priority == 'below-normal':
        # As visual_brain.run launched in Anode: the runner starts below normal, spawns the camera (and the depth
        # process) and only then raises its own class; each child sets its own.
        launch = dict(inherited=priority_class(), launched_at=priority_class(BELOW_NORMAL_PRIORITY_CLASS))
    t_launch = time.monotonic()
    camera = ProcessRetinaCamera(title='bench', fps=protocol['camera_fps'], gate_sensor=inputs['sensor'],
                                 backend=backend, race_cues=True, detector_device='cuda', looming=True, gap=spec)
    load = BrainLoad(protocol['brain_load_ms'])
    ticks, gaps, stages, cues = [], {}, {}, {}
    status = None
    try:
        camera.start()
        if launch is not None:
            from ..liftoff.scheduling import flight_process_priority
            launch['raised_after_spawn'] = flight_process_priority()
        camera.wait_ready(timeout=180.)
        while camera.latest is None:
            if camera.error:
                raise RuntimeError(camera.error)
            if time.monotonic()-t_launch > 180:
                raise RuntimeError('no camera frame')
            time.sleep(.05)
        ready_s = time.monotonic()-t_launch
        t0 = time.monotonic()
        anchor.write_text(repr(t0))
        wall0 = tel.wall[0]
        next_tick = t0
        while True:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(max(0., next_tick-now-.0005))
                continue
            next_tick += .01
            if now-t0 > seconds+protocol['warmup_s'] or watch.hot:
                break
            tv = start+(now-t0)
            pos, q, vel, ts = tel.at(wall0+tv+inputs['offset'])
            camera.motion.publish(now, now, ts, pos, q, vel, np.zeros(3))
            latest = camera.latest
            clearance = camera.clearance
            gap = camera.gap if spec else None
            st = camera.stages
            load()
            if latest is not None:
                cues.setdefault(latest[0], now)
            if gap is not None:
                gaps.setdefault((gap['time'], gap['seq']), dict(gap, received=now))
            if st is not None:
                stages.setdefault(st['frame_time'], st)
            ticks.append((now, latest[0] if latest else np.nan, gap['time'] if gap else np.nan,
                          (gap.get('depth_ms') if gap and gap.get('depth_ms') is not None else np.nan),
                          clearance['time'] if clearance else np.nan))
            if camera.error:
                raise RuntimeError(camera.error)
        status = camera.diagnostics()
    finally:
        camera.stop()
        watch.stop_flag.set()
    ticks = np.asarray(ticks, float)
    window = t0+protocol['warmup_s']
    sel = ticks[:, 0] >= window
    tick = ticks[sel]
    frames = np.array(sorted(k for k in cues if k >= window))
    stage_rows = [s for k, s in stages.items() if k >= window]
    gap_rows = [g for (k, _), g in gaps.items() if k >= window]
    summary = summarize(tick, frames, stage_rows, gap_rows, seconds)
    gap_status = (status or {}).get('gap') or {}
    priorities = dict(parent=launch if launch is not None else dict(current=priority_class()),
                      camera=(status or {}).get('priority'), depth_process=gap_status.get('priority'),
                      launch=args.parent_priority)
    result = dict(schema='haltere.obstacles.gap_bench_run.v1', condition=cond, placement=placement, flight=args.flight,
                  start_s=start, seconds=seconds, gates_sha256=gates_sha, protocol=protocol, gap_spec=spec,
                  priorities=priorities, camera_skips=gap_status.get('camera_skips'),
                  offset_s=inputs['offset'], offset_source=inputs['offset_source'], ready_s=round(ready_s, 1),
                  gpu_peak_c=watch.peak, gpu_hard_stop=watch.hot, chunk=guard.summary(),
                  camera_diagnostics=status, live_camera_reference=inputs['sidecar_camera'].get('stages_ms'),
                  finished_at=_dt.datetime.now().isoformat(timespec='seconds'), **summary)
    (out/f'{tag}.json').write_text(json.dumps(result, indent=1, default=float), encoding='utf-8')
    np.savez_compressed(out/f'{tag}.npz', ticks=ticks, t0=t0, anchor=t0, start=start,
                        frames=np.array(sorted(cues)), stages=np.array([[s[k] for k in s] for s in stages.values()]),
                        stage_fields=np.array(list(next(iter(stages.values())).keys())) if stages else np.array([]),
                        gaps=json.dumps([{k: v for k, v in g.items()} for g in gaps.values()], default=float))
    _log(f"{tag}: {json.dumps({k: result[k] for k in ('camera', 'gap', 'cue')}, default=float)[:1500]}")
    if watch.hot:
        raise SystemExit(f'GPU reached {watch.peak} C: stopped')


def _p(values, q):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    return float(np.percentile(values, q)) if len(values) else None


def summarize(tick, frames, stage_rows, gap_rows, seconds):
    s = lambda key: [r[key] for r in stage_rows]
    camera = dict(frames=int(len(frames)), rate_hz=float(len(frames)/seconds),
                  interval_ms=dict(p50=_p(np.diff(frames)*1000, 50), p95=_p(np.diff(frames)*1000, 95)),
                  **{f'{k}_ms': dict(p50=_p(s(k), 50), p95=_p(s(k), 95), max=_p(s(k), 100))
                     for k in ('capture', 'preprocess', 'inference', 'publish', 'looming', 'gap', 'total',
                               'cue_latency')})
    cue_age = (tick[:, 0]-tick[:, 1])*1000
    cue = dict(age_at_tick_ms=dict(p50=_p(cue_age, 50), p95=_p(cue_age, 95), max=_p(cue_age, 100)),
               latency_ms=camera['cue_latency_ms'])
    gap = None
    if gap_rows:
        age = (tick[:, 0]-tick[:, 2])*1000
        depth_ticks = np.isfinite(tick[:, 3])
        kinds = {}
        for g in gap_rows:
            kinds[g['kind']] = kinds.get(g['kind'], 0)+1
        depth = [g for g in gap_rows if g.get('depth_ms') is not None]
        gap = dict(samples=len(gap_rows), depth_samples=len(depth), rate_hz=len(gap_rows)/seconds,
                   depth_rate_hz=len(depth)/seconds, kinds=kinds,
                   age_at_tick_ms=dict(p50=_p(age, 50), p95=_p(age, 95), max=_p(age, 100)),
                   age_at_tick_depth_ms=dict(p50=_p(age[depth_ticks], 50), p95=_p(age[depth_ticks], 95),
                                             max=_p(age[depth_ticks], 100)),
                   latency_ms=dict(p50=_p([1000*g['age'] for g in depth], 50),
                                   p95=_p([1000*g['age'] for g in depth], 95)),
                   **{f'{k}_ms': dict(p50=_p([g[k] for g in depth], 50), p95=_p([g[k] for g in depth], 95))
                      for k in ('frame_ms', 'overlay_ms', 'depth_ms', 'decide_ms')},
                   active_share=float(np.mean([abs(g['shift']) >= 2 for g in depth])) if depth else None)
    return dict(camera=camera, cue=cue, gap=gap)


def cmd_fp16(args):
    import cv2
    from ..obstacles.overlays import to_model_frame
    from ..vision import gap_cue as gc
    from ..vision.relative_depth import RelativeDepth
    limit_threads(2, torch=True, cv2=True)
    guard = ChunkGuard(require_flight_lock_path(args.flight_lock), gpu=True, pause_c=PAUSE_C, resume_c=RESUME_C,
                       hard_stop_c=HARD_STOP_C, chunk_max_s=CHUNK_S, log=_log)
    gates, gates_sha = load_gates(require_frozen=True)
    params, cue_config, cue_sha = gc.load_config(require_frozen=True)
    hw = tuple(cue_config['relative_depth']['input_hw'])
    inputs = flight_inputs(args.flight, args.align_npz)
    video = cv2.VideoCapture(inputs['video'])
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = np.linspace(0, count-1, int(args.frames)).astype(int)
    frames = []
    for i in picks:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = video.read()
        if ok:
            frames.append(to_model_frame(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    video.release()
    guard.before_chunk()
    t0 = time.monotonic()
    half = RelativeDepth(fp16=True, input_hw=hw)
    full = RelativeDepth(fp16=False, input_hw=hw)
    rel, rel_max, corr = [], [], []
    for f in frames:
        a, b = half(f)[0].astype(np.float64), full(f)[0].astype(np.float64)
        rel.append(float(np.mean(np.abs(a-b))/np.mean(np.abs(b))))
        rel_max.append(float(np.max(np.abs(a-b))/np.mean(np.abs(b))))
        corr.append(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
        if time.monotonic()-t0 > CHUNK_S-30:
            break
    result = dict(schema='haltere.obstacles.gap_bench_fp16.v1', flight=args.flight, frames=len(rel), input_hw=list(hw),
                  gates_sha256=gates_sha, gap_cue_config_sha256=cue_sha,
                  relative_error=dict(definition='per frame mean|fp16 - fp32| / mean|fp32| over the 36 x 64 block means',
                                      p50=_p(rel, 50), p95=_p(rel, 95), max=_p(rel, 100)),
                  block_max_relative_error=dict(p50=_p(rel_max, 50), p95=_p(rel_max, 95), max=_p(rel_max, 100)),
                  correlation=dict(min=_p(corr, 0), p50=_p(corr, 50)), provenance=dict(fp16=half.provenance,
                                                                                      fp32=full.provenance),
                  chunk=guard.summary())
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out)/'fp16.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    _log(json.dumps(result['relative_error']))


def cmd_parity(args):
    """Bench gap samples of a run vs haltere.vision.gap_cue.decide on the recorded frame shown at that time,
    with the gap-cue evaluation's teacher-basis disparity, pose and logged cue for that frame."""
    from ..vision import gap_cue as gc
    from .gap_cue_eval import validity_from_fractions
    run = np.load(Path(args.out)/f'{args.tag}.npz', allow_pickle=True)
    gaps = json.loads(str(run['gaps']))
    t0, start = float(run['t0']), float(run['start'])
    rec = np.load(Path(args.eval_dir)/'flights'/f'{args.flight}.npz', allow_pickle=True)
    dep = np.load(Path(args.eval_dir)/'depth'/f'{args.flight}.npz')
    params, cfg, _ = gc.load_config(require_frozen=True)
    fps = float(rec['fps'])
    idx = rec['frame_idx']
    rows = []
    for g in gaps:
        if g.get('depth_ms') is None or not g.get('valid'):
            continue
        frame = int((start+g['time']-t0)*fps)
        j = int(np.searchsorted(idx, frame, side='right')-1)
        if j < 0 or not np.isfinite(rec['cue_uv'][j]).all():
            continue
        valid = validity_from_fractions(rec['maskfrac'][j], tuple(cfg['mask_layers']), params.block_mask_max_fraction)
        d = gc.decide(np.asarray(dep['teacher'][j], np.float32), rec['quat'][j], rec['cue_uv'][j], valid=valid,
                      params=params)
        if not d.valid:
            continue
        rows.append((g['shift'], d.shift_deg, g['ring_deg'], d.ring_bearing_deg, g['kind'] == d.kind))
    rows = np.asarray(rows, dtype=object)
    result = dict(schema='haltere.obstacles.gap_bench_parity.v1', flight=args.flight, run=args.tag, matched=len(rows))
    if len(rows):
        bench, offline = rows[:, 0].astype(float), rows[:, 1].astype(float)
        ring = np.abs((rows[:, 2].astype(float)-rows[:, 3].astype(float)+180) % 360-180)
        active = (np.abs(bench) >= 2) | (np.abs(offline) >= 2)
        result.update(ring_bearing_diff_deg=dict(p50=_p(ring, 50), p95=_p(ring, 95)),
                      shift_diff_deg=dict(p50=_p(np.abs(bench-offline), 50), p95=_p(np.abs(bench-offline), 95)),
                      kind_agreement=float(np.mean(rows[:, 4].astype(bool))),
                      active_frames=int(active.sum()),
                      active_side_agreement=float(np.mean(np.sign(bench[active]) == np.sign(offline[active])))
                      if active.any() else None,
                      note='the bench frame is the video frame shown at the capture time and its pose is the telemetry '
                           'at that time; the offline frame time is its last repeat in the video (up to a frame or '
                           'two later), the offline cue is the logged one and its disparity the 336 x 602 teacher '
                           'basis of the same recorded frame')
    (Path(args.out)/f'parity_{args.tag}.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    _log(json.dumps(result))


def cmd_score(args):
    gates, gates_sha = load_gates(require_frozen=True)
    g = gates['gates']
    out = Path(args.out)
    runs = {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in sorted(out.glob('*.json'))
            if not p.stem.startswith(('fp16', 'parity', 'gates'))}
    base = [r for r in runs.values() if r['condition'] == 'baseline']
    fp16 = json.loads((out/'fp16.json').read_text()) if (out/'fp16.json').exists() else None
    table = {}
    for name, r in runs.items():
        if r['condition'] == 'baseline':
            continue
        ref = [b for b in base if b['flight'] == r['flight']]
        ref = ref[-1] if ref else None
        cam, gap = r['camera'], r['gap'] or {}
        increase = (cam['cue_latency_ms']['p95']-ref['camera']['cue_latency_ms']['p95']) if ref else None
        checks = dict(
            camera_loop_p95_ms=(cam['total_ms']['p95'], cam['total_ms']['p95'] <= g['camera_loop_p95_ms']),
            camera_rate_hz=(cam['rate_hz'], cam['rate_hz'] >= g['camera_rate_hz_min']),
            gap_age_p95_ms=(gap.get('age_at_tick_depth_ms', {}).get('p95'),
                            gap.get('age_at_tick_depth_ms', {}).get('p95') is not None
                            and gap['age_at_tick_depth_ms']['p95'] <= g['gap_age_p95_ms']),
            cue_latency_increase_ms=(increase, increase is not None and increase <= g['cue_latency_increase_ms']),
            fp16_relative_error_p95=(fp16['relative_error']['p95'] if fp16 else None,
                                     bool(fp16) and fp16['relative_error']['p95'] <= g['fp16_rel_error_max']))
        table[name] = dict(condition=r['condition'], flight=r['flight'], baseline=ref['condition'] if ref else None,
                           checks={k: dict(value=v, passed=bool(ok)) for k, (v, ok) in checks.items()},
                           passed=all(ok for _, ok in checks.values()))
    result = dict(schema='haltere.obstacles.gap_bench_gates_result.v1', gates_sha256=gates_sha, gates=g,
                  baselines={k: dict(camera=r['camera'], cue=r['cue']) for k, r in runs.items()
                             if r['condition'] == 'baseline'}, conditions=table,
                  scored_at=_dt.datetime.now().isoformat(timespec='seconds'))
    (out/'gates.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    for k, v in table.items():
        _log(k, 'PASS' if v['passed'] else 'FAIL', json.dumps(v['checks']))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['run', 'fp16', 'parity', 'score'])
    p.add_argument('--out', required=True)
    p.add_argument('--flight-lock', default=None)
    p.add_argument('--flight', default='straw-brain08-06')
    p.add_argument('--condition', choices=sorted(CONDITIONS), default='baseline')
    p.add_argument('--start', type=float, default=30.)
    p.add_argument('--seconds', type=float, default=60.)
    p.add_argument('--frames', type=int, default=120)
    p.add_argument('--align-npz', default=None, help='gap-cue evaluation flight npz with the video offset_s')
    p.add_argument('--eval-dir', default=None, help='gap-cue evaluation output (flights/, depth/) for parity')
    p.add_argument('--tag', default=None)
    p.add_argument('--parent-priority', choices=['inherit', 'below-normal'], default='inherit',
                   help='below-normal: start the bench below normal and raise it after spawning the camera, as the '
                        'flight runner launched in Anode (recorded in the run with each process\'s class)')
    args = p.parse_args(argv)
    dict(run=cmd_run, fp16=cmd_fp16, parity=cmd_parity, score=cmd_score)[args.cmd](args)


if __name__ == '__main__':
    main()
