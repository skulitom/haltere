"""Command-line entry point: ``haltere <command>``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_fetch(a):
    from .connectome import sources
    paths = sources.fetch(version=a.version, force=a.force)
    for k, p in paths.items():
        print(f'{k:18s} {p}')


def cmd_build(a):
    from .config import load_yaml
    from .connectome.graph import GraphConfig, build_graph
    cfg = GraphConfig.from_dict(load_yaml(a.config)) if a.config else GraphConfig()
    if a.max_neurons:
        cfg.max_neurons = a.max_neurons
    if a.whole_cns:
        cfg.whole_cns = True
    tables = None
    if a.source == 'neuprint':
        from .connectome import neuprint_source
        print('fetching neuron table from neuPrint (this can take a long time for the full adjacency) ...')
        tables = neuprint_source.tables_from_neuprint(status=cfg.status, min_weight=cfg.min_synapses)
    g = build_graph(cfg, tables=tables)
    g.save(Path(a.out))
    print(f'saved graph to {a.out}.npz / .nodes.parquet / .meta.json')


def cmd_neuprint(a):
    import json
    from .connectome import neuprint_source
    if a.neuprint_cmd == 'check':
        print(json.dumps(neuprint_source.check(a.dataset), indent=2, default=str))
    elif a.neuprint_cmd == 'query':
        c = neuprint_source.client(a.dataset)
        print(c.fetch_custom(a.cypher).to_string())


def cmd_populations(a):
    from .config import load_yaml
    from .connectome import sources
    from .connectome.graph import GraphConfig
    from .connectome.populations import select_populations, describe_populations
    cfg = GraphConfig.from_dict(load_yaml(a.config)) if a.config else GraphConfig()
    ann = sources.load_annotations(version=cfg.version)
    ann = ann[ann['status'].isin(cfg.status)].reset_index(drop=True)
    print(describe_populations(ann, select_populations(ann, cfg.populations), top=a.top))


def cmd_train(a):
    from .config import load_yaml
    from .train.bptt import ExperimentConfig, train
    cfg = ExperimentConfig.from_dict(load_yaml(a.config))
    if a.run:
        cfg.train.run = a.run
    if a.iters:
        cfg.train.iters = a.iters
    if a.graph:
        cfg.train.graph = a.graph
    if a.batch:
        cfg.train.B = a.batch
    if a.device:
        cfg.train.device = a.device
    run_dir = train(cfg, resume=a.resume)
    print(f'run directory: {run_dir}')


def cmd_imitate(a):
    from .config import load_yaml, dataclass_from_dict
    from .train.bptt import ExperimentConfig
    from .train.imitate import ImitateConfig, imitate
    cfg = ExperimentConfig.from_dict(load_yaml(a.config))
    ic = dataclass_from_dict(ImitateConfig, load_yaml(a.imitate_config) if a.imitate_config else {})
    if a.teacher:
        ic.teacher = a.teacher
    if a.run:
        ic.run = a.run
    if a.iters:
        ic.iters = a.iters
    if a.motor:
        cfg.brain.motor = tuple(a.motor.split(','))
    if a.freeze_internal:
        ic.freeze_internal = True
    if a.student_frac_final is not None:
        ic.student_frac_final = a.student_frac_final
    if a.resume:
        ic.resume = a.resume
    run_dir = imitate(cfg, ic)
    print(f'run directory: {run_dir}')


def cmd_eval(a):
    import json
    import torch
    from .train.bptt import load_checkpoint, evaluate
    brain, cfg, graph = load_checkpoint(a.ckpt, a.device)
    device = brain.device
    m = evaluate(brain, cfg, a.steps, a.batch, device, difficulty=a.difficulty, record=bool(a.trajectory))
    traj = m.pop('trajectory', None)
    print(json.dumps(m, indent=2))
    if traj is not None:
        import numpy as np
        np.savez(a.trajectory, pos=np.stack([t['pos'] for t in traj]), target=np.stack([t['target'] for t in traj]),
                 act=np.stack([t['act'] for t in traj]), quat=np.stack([t['quat'] for t in traj]))
        print(f'trajectory of environment 0 saved to {a.trajectory}')


def cmd_summarize(a):
    import csv
    import json
    for run in a.runs:
        run = Path(run)
        print(f'== {run}')
        log = run / 'log.csv'
        if log.exists():
            rows = list(csv.DictReader(open(log, encoding='utf-8')))
            if rows:
                step = max(1, len(rows) // a.rows)
                sel = rows[::step] + ([rows[-1]] if (len(rows) - 1) % step else [])
                if 'mse' in rows[0]:   # imitation run
                    print('  iter      mse  R2 thr  R2 roll  R2 pitch  R2 yaw  student-driven')
                    for r in sel:
                        print(f'  {int(r["iter"]):5d} {float(r["mse"]):8.4f} {float(r["r2_thr"]):7.2f} {float(r["r2_roll"]):8.2f} '
                              f'{float(r["r2_pitch"]):9.2f} {float(r["r2_yaw"]):7.2f} {float(r["student_frac"]):8.2f}')
                else:
                    print('  iter    loss     cost   rate  dist(m)  crash/env  diff')
                    for r in sel:
                        print(f'  {int(r["iter"]):5d} {float(r["loss"]):8.3f} {float(r["cost"]):8.3f} {float(r["rate_mean"]):6.3f} '
                              f'{float(r["dist_mean"]):7.2f} {float(r["crashed"]):9.3f} {float(r["difficulty"]):6.2f}')
        ev = run / 'eval.jsonl'
        if ev.exists():
            print('  closed-loop evals (difficulty 1.0): iter  dist(m)  within0.5  crashed_ever  cost')
            for line in open(ev, encoding='utf-8'):
                e = json.loads(line)
                print(f'    {e["iter"]:5d}  {e["dist_mean"]:7.2f}  {e["within_0.5m"]:8.2f}  {e["crashed_ever"]:11.2f}  {e["cost"]:7.3f}')


def cmd_export(a):
    from .train.export import export_slim
    print(json.dumps(export_slim(a.ckpt, a.out), indent=1))


def cmd_publish_hf(a):
    """Upload the trained brains, the flight graph and the model card to a Hugging Face model repo."""
    from huggingface_hub import HfApi
    api = HfApi()
    me = api.whoami()
    role = (me.get('auth', {}).get('accessToken', {}) or {}).get('role')
    print(f'logged in as {me.get("name")} (token role: {role})')
    if role == 'read':
        raise SystemExit('the cached token is read-only: run `.venv\\Scripts\\hf.exe auth login` with a WRITE token '
                         '(https://huggingface.co/settings/tokens) and retry')
    repo = a.repo or f'{me["name"]}/haltere'
    api.create_repo(repo, repo_type='model', exist_ok=True, private=a.private)
    files = [('docs/hf_model_card.md', 'README.md'), ('docs/liftoff_hover.gif', 'liftoff_hover.gif'),
             ('docs/liftoff_square.gif', 'liftoff_square.gif'), ('docs/liftoff_orbit.gif', 'liftoff_orbit.gif'),
             ('docs/liftoff_climbdive.gif', 'liftoff_climbdive.gif'), ('docs/liftoff_race.gif', 'liftoff_race.gif'),
             ('docs/flight.gif', 'flight.gif'), ('docs/liftoff_sight.gif', 'liftoff_sight.gif'),
             ('docs/liftoff_lap4.gif', 'liftoff_lap4.gif'),
             ('configs/liftoff.yaml', 'liftoff.yaml'),
             ('data/built/flight.npz', 'flight.npz'), ('data/built/flight.nodes.parquet', 'flight.nodes.parquet'),
             ('data/built/flight.meta.json', 'flight.meta.json'), ('configs/flight.yaml', 'flight.yaml')]
    files += [(str(p), p.name) for p in sorted(Path('artifacts').glob('*.pt'))]
    for src, dst in files:
        if not Path(src).exists():
            print(f'  skip {src} (missing)')
            continue
        api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo, repo_type='model',
                        commit_message=f'add {dst}')
        print(f'  uploaded {src} -> {dst}')
    print(f'https://huggingface.co/{repo}')


def cmd_render(a):
    from .viz.render import record_flight, render_video, live_view
    rec = record_flight(a.ckpt, seconds=a.seconds, difficulty=a.difficulty, waypoint_every=a.waypoint_every,
                        stride=a.stride, device=a.device, seed=a.seed)
    print(f'recorded {len(rec["time"])} frames; mean distance {float(__import__("numpy").linalg.norm(rec["pos"] - rec["target"], axis=1).mean()):.2f} m; '
          f'crashed {rec["crashed"]}')
    if a.live:
        live_view(rec, speed=a.speed)
    else:
        meta = render_video(rec, a.out, fps=a.fps, gif_seconds=a.gif_seconds)
        print(json.dumps(meta, indent=2))


def cmd_liftoff(a):
    from .liftoff import commands
    getattr(commands, f'cmd_{a.liftoff_cmd.replace("-", "_")}')(a)


def cmd_vision(a):
    import yaml
    from .vision.camera import Camera
    if a.vision_cmd == 'calibrate':
        from .vision.calibrate import calibrate
        tilts = [float(x) for x in a.tilts.split(',')] if a.tilts else None
        r = calibrate(a.dataset, step=a.step, tilt_grid=tilts)
        cam = {'width': r['width'], 'height': r['height'], 'f': r['f'], 'tilt_deg': r['tilt_deg'], 'hfov_deg': r['hfov_deg'],
               'error_deg': r['error_deg'], 'pairs': r['pairs'], 'dataset': a.dataset}
        Path(a.out).write_text(yaml.safe_dump(cam, sort_keys=False), encoding='utf-8')
        print(f'camera written to {a.out}')
    elif a.vision_cmd == 'gates':
        from .vision.gates import passages_from_dataset, save_gates
        passages = passages_from_dataset(a.dataset, thresh=a.thresh, min_gap_m=a.min_gap)
        save_gates(passages, a.out)
        print(f'{len(passages)} gates written to {a.out}')
    elif a.vision_cmd == 'gates-from-frames':
        from .vision.gates import gates_from_passage_frames, save_gates
        frames = [int(x) for x in a.frames.split(',')]
        gates = gates_from_passage_frames(a.dataset, frames, a.track or None)
        save_gates(gates, a.out)
        print(f'{len(gates)} gates written to {a.out}')
    elif a.vision_cmd == 'sheet':
        from .vision.calibrate import load_index
        from .vision.triangulate import sheet
        rows = load_index(a.dataset)
        files = [r['file'] for r in rows][a.start:a.start + a.count * a.every:a.every]
        out = sheet(a.dataset, files, a.out, cols=a.cols, grid=a.grid)
        print(f'{len(files)} frames -> {out}: {", ".join(files)}')
    elif a.vision_cmd == 'triangulate':
        from .vision.gates import save_gates
        from .vision.triangulate import gates_from_observations, load_observations
        c = yaml.safe_load(Path(a.camera).read_text(encoding='utf-8'))
        cam = Camera(int(c['width']), int(c['height']), float(c['f']), float(c['tilt_deg']))
        gates = gates_from_observations(a.dataset, load_observations(a.observations), cam)
        save_gates(gates, a.out)
        print(f'{len(gates)} gates written to {a.out}')
    elif a.vision_cmd == 'inspect':
        from .vision.inspect import overlay
        out = overlay(a.dataset, a.out, every=a.every, count=a.count, cols=a.cols, ckpt=a.ckpt or None, device=a.device)
        print(f'overlay written to {out}')
    elif a.vision_cmd == 'label':
        from .vision.train import label_dataset
        c = yaml.safe_load(Path(a.camera).read_text(encoding='utf-8'))
        cam = Camera(int(c['width']), int(c['height']), float(c['f']), float(c['tilt_deg']))
        for d in a.datasets:
            label_dataset(d, a.gates, cam)
    elif a.vision_cmd == 'eval':
        from .vision.evaluate import evaluate
        evaluate(a.ckpt, a.datasets, device=a.device)
    elif a.vision_cmd == 'passes':
        from .vision.evaluate import gate_passes
        gate_passes(a.dataset, a.gates)
    elif a.vision_cmd == 'rehearse':
        from .vision.rehearse import DetectorModel, RehearsalOptions, rehearse
        det = DetectorModel.clean() if a.clean else DetectorModel()
        for k in ('rate_hz', 'latency_s', 'centre_px', 'width_frac', 'oblique', 'miss', 'burst_rate', 'flip', 'false_pos'):
            v = getattr(a, k)
            if v is not None:
                setattr(det, k, v)
        sets = {}
        for kv in a.set:
            k, _, v = kv.partition('=')
            sets[k.strip()] = yaml.safe_load(v)
        sg = [float(x) for x in str(a.stick_gain).split(',')]
        opts = RehearsalOptions(seconds=a.seconds, seed=a.seed, face_travel=a.face_travel, face_max=a.face_max,
                                stick_gain=sg[0] if len(sg) == 1 else sg, stick_lpf=a.stick_lpf,
                                delay_steps=a.delay_steps, start=tuple(float(x) for x in a.start.split(',')),
                                collide=not a.no_collide, pilot_set=sets, physics_jitter=a.physics_jitter,
                                max_gpu_temp=a.max_gpu_temp, burst_s=a.burst, cool_s=a.cool)
        rehearse(a.ckpt, a.camera, a.gates, a.log or None, a.track or None, a.device, opts, det, a.json or None)
    elif a.vision_cmd == 'train':
        from .vision.train import train
        out = train(a.datasets, out_dir=a.out, epochs=a.epochs, batch=a.batch, lr=a.lr, width=a.width, max_gpu_temp=a.max_gpu_temp, batch_sleep=a.batch_sleep, init=a.init, device=a.device)
        print(f'gate detector saved in {out}')


def main(argv=None):
    p = argparse.ArgumentParser(prog='haltere', description='A fly brain that flies a drone in Liftoff.')
    sp = p.add_subparsers(dest='cmd', required=True)

    s = sp.add_parser('fetch', help='download the male-CNS flat connectome (about 570 MB)')
    s.add_argument('--version', default='v1.0')
    s.add_argument('--force', action='store_true')
    s.set_defaults(fn=cmd_fetch)

    s = sp.add_parser('build', help='build the flight graph from the connectome')
    s.add_argument('--config', default='configs/flight.yaml')
    s.add_argument('--out', default='data/built/flight')
    s.add_argument('--max-neurons', type=int, default=0)
    s.add_argument('--whole-cns', action='store_true', help='keep every traced neuron (very large)')
    s.add_argument('--source', choices=['flat', 'neuprint'], default='flat',
                   help='flat = public bulk files (default); neuprint = live queries with your token')
    s.set_defaults(fn=cmd_build)

    s = sp.add_parser('neuprint', help='neuPrint access with your token (NEUPRINT_APPLICATION_CREDENTIALS)')
    ns = s.add_subparsers(dest='neuprint_cmd', required=True)
    q = ns.add_parser('check', help='verify the token and show dataset metadata')
    q.add_argument('--dataset', default='male-cns:v1.0')
    q = ns.add_parser('query', help='run a Cypher query')
    q.add_argument('cypher')
    q.add_argument('--dataset', default='male-cns:v1.0')
    s.set_defaults(fn=cmd_neuprint)

    s = sp.add_parser('populations', help='show which neurons each population selects')
    s.add_argument('--config', default='configs/flight.yaml')
    s.add_argument('--top', type=int, default=10)
    s.set_defaults(fn=cmd_populations)

    s = sp.add_parser('train', help='train the brain in the differentiable simulator')
    s.add_argument('--config', default='configs/train.yaml')
    s.add_argument('--run', default='')
    s.add_argument('--iters', type=int, default=0)
    s.add_argument('--graph', default='')
    s.add_argument('--batch', type=int, default=0)
    s.add_argument('--device', default='')
    s.add_argument('--resume', default=None)
    s.set_defaults(fn=cmd_train)

    s = sp.add_parser('imitate', help='train the brain to imitate a working controller (e.g. the MLP baseline)')
    s.add_argument('--config', default='configs/train.yaml')
    s.add_argument('--imitate-config', default='', help='YAML with ImitateConfig fields')
    s.add_argument('--teacher', default='', help='checkpoint of the controller to imitate')
    s.add_argument('--run', default='')
    s.add_argument('--iters', type=int, default=0)
    s.add_argument('--motor', default='', help='comma-separated readout populations, e.g. wing_mn,premotor')
    s.add_argument('--freeze-internal', action='store_true')
    s.add_argument('--student-frac-final', type=float, default=None)
    s.add_argument('--resume', default='', help='continue from a checkpoint (model + optimiser state)')
    s.set_defaults(fn=cmd_imitate)

    s = sp.add_parser('eval', help='evaluate a checkpoint in the simulator')
    s.add_argument('ckpt')
    s.add_argument('--steps', type=int, default=600)
    s.add_argument('--batch', type=int, default=256)
    s.add_argument('--difficulty', type=float, default=1.0)
    s.add_argument('--device', default='cuda')
    s.add_argument('--trajectory', default='', help='save env-0 trajectory to this .npz')
    s.set_defaults(fn=cmd_eval)

    s = sp.add_parser('publish-hf', help='upload artifacts/, the flight graph and the model card to Hugging Face')
    s.add_argument('--repo', default='', help='model repo id (default: <your account>/haltere)')
    s.add_argument('--private', action='store_true')
    s.set_defaults(fn=cmd_publish_hf)

    s = sp.add_parser('export', help='write a slim inference checkpoint (parameters only, ~12 MB) for publishing')
    s.add_argument('ckpt')
    s.add_argument('out')
    s.set_defaults(fn=cmd_export)

    s = sp.add_parser('render', help='record a flight and render neural activity + drone as MP4/GIF (or --live)')
    s.add_argument('ckpt')
    s.add_argument('--out', default='runs/video')
    s.add_argument('--seconds', type=float, default=20.0)
    s.add_argument('--difficulty', type=float, default=0.8)
    s.add_argument('--waypoint-every', type=float, default=4.0, help='seconds between target changes (0 = fixed)')
    s.add_argument('--stride', type=int, default=4, help='record every k-th control step (4 -> 25 fps)')
    s.add_argument('--fps', type=int, default=0)
    s.add_argument('--gif-seconds', type=float, default=12.0)
    s.add_argument('--live', action='store_true', help='show the animation in a window instead of writing files')
    s.add_argument('--speed', type=float, default=1.0)
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--device', default='cuda')
    s.set_defaults(fn=cmd_render)

    s = sp.add_parser('summarize', help='print the training and evaluation history of run directories')
    s.add_argument('runs', nargs='+')
    s.add_argument('--rows', type=int, default=12)
    s.set_defaults(fn=cmd_summarize)

    s = sp.add_parser('vision', help='gate vision: calibrate the FPV camera, find the gates, label frames, train GateNet')
    vs = s.add_subparsers(dest='vision_cmd', required=True)
    q = vs.add_parser('calibrate', help='focal length (and tilt) of the FPV camera from a flight dataset')
    q.add_argument('dataset')
    q.add_argument('--out', default='configs/camera.yaml')
    q.add_argument('--step', type=int, default=2, help='frames between the two images of a pair')
    q.add_argument('--tilts', default='', help='comma-separated camera tilts to try (deg); default: the 30 of the drone config')
    q = vs.add_parser('gates', help='recover the gate positions from the passages in a lap flight dataset')
    q.add_argument('dataset')
    q.add_argument('--out', default='configs/gates_strawbale.json')
    q.add_argument('--thresh', type=float, default=0.35, help='border-darkness threshold of a passage')
    q.add_argument('--min-gap', type=float, default=8.0, help='minimum distance between gates (m)')
    q = vs.add_parser('gates-from-frames', help='gate positions from the frames where the drone passes each gate')
    q.add_argument('dataset')
    q.add_argument('--frames', required=True, help='comma-separated frame numbers of the passages, in course order')
    q.add_argument('--track', default='configs/track_strawbale.yaml', help='taught path (gates snap to it); empty = none')
    q.add_argument('--out', default='configs/gates_strawbale.json')
    q = vs.add_parser('sheet', help='contact sheet of dataset frames with a pixel grid (to read gate positions off)')
    q.add_argument('dataset')
    q.add_argument('--out', default='data/vision/sheet.png')
    q.add_argument('--start', type=int, default=0)
    q.add_argument('--every', type=int, default=10)
    q.add_argument('--count', type=int, default=6)
    q.add_argument('--cols', type=int, default=3)
    q.add_argument('--grid', type=int, default=40)
    q = vs.add_parser('triangulate', help='gate positions from pixel observations {file, u, v, gate} in posed frames')
    q.add_argument('dataset')
    q.add_argument('--observations', required=True, help='JSON list of observations')
    q.add_argument('--camera', default='configs/camera.yaml')
    q.add_argument('--out', default='configs/gates_strawbale.json')
    q = vs.add_parser('inspect', help='overlay the gate labels (and a GateNet checkpoint) on dataset frames')
    q.add_argument('dataset')
    q.add_argument('--out', default='data/vision/inspect.png')
    q.add_argument('--every', type=int, default=25)
    q.add_argument('--count', type=int, default=12)
    q.add_argument('--cols', type=int, default=3)
    q.add_argument('--ckpt', default='')
    q.add_argument('--device', default='cuda')
    q = vs.add_parser('label', help='project the next gate into every frame of the datasets (labels.json)')
    q.add_argument('datasets', nargs='+')
    q.add_argument('--gates', default='configs/gates_strawbale.json')
    q.add_argument('--camera', default='configs/camera.yaml')
    q = vs.add_parser('eval', help='measure a GateNet checkpoint on labelled datasets (accuracy, false positives, bias per range)')
    q.add_argument('ckpt')
    q.add_argument('datasets', nargs='+')
    q.add_argument('--device', default='cuda')
    q = vs.add_parser('passes', help='which gates a recorded flight (dataset with poses) flew through')
    q.add_argument('dataset')
    q.add_argument('--gates', default='configs/gates_strawbale.json')
    q = vs.add_parser('rehearse', help='fly the by-sight pilot through the course in the simulator with a synthetic '
                                       'gate detector (no game): per-step log + score')
    q.add_argument('ckpt', help='brain checkpoint (e.g. runs/ftPath2/best.pt)')
    q.add_argument('--camera', default='configs/camera_seat.yaml')
    q.add_argument('--gates', default='configs/gates_strawbale.json')
    q.add_argument('--track', default='', help='taught track YAML: also report the distance to its line')
    q.add_argument('--seconds', type=float, default=120.0, help='simulated seconds')
    q.add_argument('--log', default='', help='per-step CSV in the `liftoff fly --log` format (scorable by `liftoff score`)')
    q.add_argument('--json', default='', help='write the score to this JSON file')
    q.add_argument('--seed', type=int, default=0)
    q.add_argument('--device', default='cuda')
    q.add_argument('--face-travel', type=float, default=0.8)
    q.add_argument('--face-max', type=float, default=0.25)
    q.add_argument('--stick-gain', default='1.0', help='one number or roll,pitch,yaw')
    q.add_argument('--stick-lpf', type=float, default=0.0)
    q.add_argument('--delay-steps', type=int, default=None, help='brain -> vehicle latency (default: the training value)')
    q.add_argument('--start', default='0,0,0,0', help='reset point x,y,z (m, gate-list frame) and yaw (deg)')
    q.add_argument('--no-collide', action='store_true', help='fly through the arch posts and top bars')
    q.add_argument('--physics-jitter', type=float, default=0.0, help='randomize mass/thrust/drag/motor lag by +- this fraction')
    q.add_argument('--set', action='append', default=[], metavar='ATTR=VALUE',
                   help='override a TelemetryPilot attribute, e.g. --set vision_speed=3 --set vision_lookahead=2.5')
    q.add_argument('--clean', action='store_true', help='perfect detector (no noise, misses, flips or phantoms; still 15 Hz and late)')
    q.add_argument('--rate-hz', type=float, default=None)
    q.add_argument('--latency-s', type=float, default=None)
    q.add_argument('--centre-px', type=float, default=None, help='centre noise std (px at 320x180)')
    q.add_argument('--width-frac', type=float, default=None, help='width noise (log-normal std)')
    q.add_argument('--oblique', type=float, default=None, help='width shrink of arches seen at an angle (0..1; GateNet showed none)')
    q.add_argument('--miss', type=float, default=None, help='per-frame miss probability')
    q.add_argument('--burst-rate', type=float, default=None, help='per-frame probability of a 0.3-1 s dropout')
    q.add_argument('--flip', type=float, default=None, help='probability of reporting the second arch in view')
    q.add_argument('--false-pos', type=float, default=None, help='per-frame phantom probability')
    q.add_argument('--max-gpu-temp', type=float, default=70.0, help='pause while the GPU is hotter than this (C); 0 = off')
    q.add_argument('--burst', type=float, default=100.0, help='wall seconds of GPU work between cool-down pauses (0 = none)')
    q.add_argument('--cool', type=float, default=20.0, help='length of a cool-down pause (s)')
    q = vs.add_parser('train', help='train GateNet on labelled datasets')
    q.add_argument('datasets', nargs='+')
    q.add_argument('--out', default='runs/gatenet')
    q.add_argument('--epochs', type=int, default=25)
    q.add_argument('--batch', type=int, default=64)
    q.add_argument('--lr', type=float, default=1e-3)
    q.add_argument('--width', type=int, default=32)
    q.add_argument('--device', default='cuda')
    q.add_argument('--max-gpu-temp', type=float, default=70.0, help='pause training while the GPU is hotter than this (C); 0 = off')
    q.add_argument('--batch-sleep', type=float, default=0.15, help='seconds to idle after every batch (keeps the machine cool)')
    q.add_argument('--init', default='', help='start from the weights of this checkpoint (a few epochs then suffice)')
    s.set_defaults(fn=cmd_vision)

    s = sp.add_parser('liftoff', help='Liftoff integration: setup, calibrate, record, fit, fly')
    ls = s.add_subparsers(dest='liftoff_cmd', required=True)
    q = ls.add_parser('setup', help='write Liftoff\'s TelemetryConfiguration.json and print the remaining manual steps')
    q.add_argument('--port', type=int, default=9001)
    q = ls.add_parser('doctor', help='check every prerequisite for flying in Liftoff and print the next step')
    q.add_argument('--port', type=int, default=9001)
    q.add_argument('--liftoff-config', default='configs/liftoff.yaml')
    q.add_argument('--ckpt', default='runs/ftRobust_best.pt')
    q = ls.add_parser('listen', help='print incoming telemetry frames (checks the UDP stream works)')
    q.add_argument('--port', type=int, default=9001)
    q.add_argument('--seconds', type=float, default=10.0)
    q = ls.add_parser('pad', help='keep the virtual Xbox pad alive with neutral sticks (to select it in Liftoff)')
    q.add_argument('--seconds', type=float, default=600.0)
    q.add_argument('--control-file', default='', help='execute pad commands written to this file (see cmd_pad)')
    q.add_argument('--udp-in', type=int, default=0, help='also accept stick packets on this UDP port (bridge mode)')
    q.add_argument('--log', default='', help='CSV log of every stick value sent, with wall-clock time')
    q = ls.add_parser('calibrate', help='drive the virtual gamepad axis by axis for Liftoff\'s controller wizard')
    q.add_argument('--auto', type=float, default=0.0, help='no keyboard: sweep each axis for this many seconds in turn')
    q.add_argument('--pause', type=float, default=4.0, help='auto mode: countdown before each axis (s)')
    q.add_argument('--rounds', type=int, default=2, help='auto mode: how many times to go through all axes')
    q.add_argument('--order', default='', help='axis order, e.g. throttle,yaw,pitch,roll')
    q.add_argument('--sweep', type=float, default=3.0, help='seconds per sweep in Enter mode')
    q = ls.add_parser('autotest', help='automated lift-off, axis pulses and landing flown by the virtual pad; infers the mapping')
    q.add_argument('--port', type=int, default=9001)
    q.add_argument('--out', default='data/liftoff/autotest.csv')
    q.add_argument('--max-throttle', type=float, default=0.25, help='throttle stick cap during the test (-1..1)')
    q.add_argument('--climb', type=float, default=2.5, help='metres to climb above the start')
    q.add_argument('--write-mapping', default='configs/liftoff.yaml', help='where to write the inferred mapping ("" = no)')
    q = ls.add_parser('stickcal', help='fit Liftoff\'s per-axis input curves from a pad log and a telemetry recording')
    q.add_argument('--pad-log', default='data/liftoff/pad_log.csv')
    q.add_argument('--csv', default='data/liftoff/stickcal.csv')
    q.add_argument('--latency', type=float, default=0.03, help='pad -> telemetry latency to assume (s)')
    q.add_argument('--throttle-points', default='', help='extra raw:processed throttle points measured in flight, e.g. 0.33:0.11,0.75:0.54')
    q.add_argument('--out', default='configs/liftoff.yaml')
    q = ls.add_parser('sticktest', help='measure the 2D stick processing of the game (radial deadzone) on the ground')
    q.add_argument('--port', type=int, default=9001)
    q.add_argument('--udp-out', default='127.0.0.1:9003', help='the pad bridge (`haltere liftoff pad --udp-in 9003`)')
    q.add_argument('--hold', type=float, default=0.25, help='seconds per stick combination')
    q.add_argument('--out', default='data/liftoff/sticktest.csv')
    q.add_argument('--fit', action='store_true', help='store the fitted radial model in --liftoff-config')
    q.add_argument('--fit-only', action='store_true', help='fit an existing --out recording without flying the test')
    q.add_argument('--liftoff-config', default='configs/liftoff.yaml')
    q = ls.add_parser('score', help='score flights from their `fly --log` CSVs: gates, speed, path error, wobble')
    q.add_argument('logs', nargs='+')
    q.add_argument('--gates', default='configs/gates_strawbale.json')
    q.add_argument('--track', default='', help='taught track YAML: also report the distance to its line')
    q.add_argument('--json', default='', help='write the scores to this JSON file')
    q = ls.add_parser('record', help='record telemetry (fly manually) to a CSV for system identification')
    q.add_argument('--port', type=int, default=9001)
    q.add_argument('--seconds', type=float, default=120.0)
    q.add_argument('--out', default='data/liftoff/recording.csv')
    q = ls.add_parser('fit', help='fit simulator parameters and stick signs to a recording')
    q.add_argument('--csv', default='data/liftoff/recording.csv')
    q.add_argument('--config', default='configs/train.yaml')
    q.add_argument('--out', default='configs/liftoff.yaml')
    q.add_argument('--iters', type=int, default=300)
    q = ls.add_parser('waypoints', help='turn a manually flown recording into a waypoint list (race track / freestyle line)')
    q.add_argument('--csv', default='data/liftoff/recording.csv')
    q.add_argument('--spacing', type=float, default=3.0, help='metres between waypoints along the flown path')
    q.add_argument('--min-alt', type=float, default=1.0, help='ignore samples below this altitude (take-off / landing)')
    q.add_argument('--min-z', type=float, default=1.0, help='clamp waypoint altitude to at least this (m above the reset point)')
    q.add_argument('--max-z', type=float, default=1e9, help='clamp waypoint altitude to at most this (m above the reset point)')
    q.add_argument('--out', default='configs/track.yaml')
    q = ls.add_parser('fly', help='let the trained brain fly the drone in Liftoff')
    q.add_argument('ckpt')
    q.add_argument('--port', type=int, default=9001)
    q.add_argument('--liftoff-config', default='configs/liftoff.yaml')
    q.add_argument('--offset', default='0,0,2', help='hover target relative to the reset position, sim frame x,y,z (m)')
    q.add_argument('--waypoints', default='', help='x,y,z;x,y,z;... targets relative to the reset point (sim frame)')
    q.add_argument('--waypoints-file', default='', help='YAML from `haltere liftoff waypoints` (a taught track)')
    q.add_argument('--dwell', type=float, default=4.0, help='seconds per waypoint (timer mode)')
    q.add_argument('--advance-radius', type=float, default=0.0,
                   help='advance to the next waypoint when this close (m); 0 = timer mode. Use ~1.0 for racing')
    q.add_argument('--no-loop', action='store_true', help='stop at the last waypoint instead of looping')
    q.add_argument('--path-speed', type=float, default=0.0,
                   help='follow the waypoint polyline as a moving target at this speed (m/s) instead of hopping')
    q.add_argument('--lookahead', type=float, default=1.5, help='path following: carrot distance ahead of the drone (m)')
    q.add_argument('--z-lead', type=float, default=-1.0,
                   help='path following: take the carrot height this far ahead instead of at the carrot (m; < 0: at the carrot)')
    q.add_argument('--pattern', default='', choices=['', 'orbit', 'climbdive', 'figure8'],
                   help='moving target pattern around the offset point (freestyle)')
    q.add_argument('--radius', type=float, default=3.0, help='pattern radius (m)')
    q.add_argument('--period', type=float, default=12.0, help='pattern period (s)')
    q.add_argument('--amplitude', type=float, default=1.5, help='climb/dive altitude swing (m)')
    q.add_argument('--stick-gain', type=str, default='1.0',
                   help='scale roll/pitch/yaw commands (smoothness): one number or roll,pitch,yaw')
    q.add_argument('--face-travel', type=float, default=0.0,
                   help='yaw the nose toward the target: stick per radian of heading error (0 = off; try 0.8). '
                        'The brain has no camera and no heading objective, so without this it flies sideways')
    q.add_argument('--face-max', type=float, default=0.25, help='cap on the facing yaw stick')
    q.add_argument('--face-ahead', type=float, default=0.0,
                   help='path mode: face the path this many metres beyond the carrot instead of the carrot itself '
                        '(keeps the camera on the course ahead)')
    q.add_argument('--face-wobble', type=float, default=0.0,
                   help='sweep the facing heading +- this many degrees (sinusoidal, --face-wobble-period): '
                        'viewpoint variety when collecting vision datasets')
    q.add_argument('--face-wobble-period', type=float, default=10.0)
    q.add_argument('--vision', default='', help='GateNet checkpoint: fly by sight (the goal comes from the gate detector '
                                                'on the game view instead of from telemetry positions)')
    q.add_argument('--camera', default='configs/camera.yaml', help='camera calibration for --vision')
    q.add_argument('--vision-fps', type=float, default=15.0)
    q.add_argument('--stick-lpf', type=float, default=0.0, help='low-pass time constant on the sticks (s)')
    q.add_argument('--gyro', choices=['quat', 'telemetry'], default='quat',
                   help='body rates from attitude differences (quat) or from Liftoff\'s Gyro field (telemetry)')
    q.add_argument('--throttle-scale', type=float, default=0.0, help='gain of the throttle remap around hover (0 = keep file value)')
    q.add_argument('--arm-hold', type=float, default=0.8, help='seconds of throttle-low before the brain gets control (arming)')
    q.add_argument('--arm-ramp', type=float, default=1.2, help='seconds over which the brain\'s sticks are ramped in')
    q.add_argument('--reset-button', default='', help='virtual pad button that resets the drone in Liftoff (e.g. A, Y, BACK); pressed after a crash')
    q.add_argument('--reset-key', default='', help='keyboard key that resets the drone in Liftoff (R by default in the game); sent to the game window after a crash')
    q.add_argument('--record', default='', help='write an MP4 of the Liftoff window + live brain activity')
    q.add_argument('--dataset', default='', help='vision dataset directory: save captured game frames (640x360 JPEG) '
                                                 'with the pose and target of each frame (index.csv)')
    q.add_argument('--dataset-every', type=int, default=2, help='save every Nth captured frame (20 fps capture)')
    q.add_argument('--capture', default='Liftoff', help='window title (substring) to capture for --record')
    q.add_argument('--capture-rect', default='', help='x,y,w,h screen region to capture instead of a window')
    q.add_argument('--fps', type=int, default=20)
    q.add_argument('--show', action='store_true', help='open a live window of the brain activity')
    q.add_argument('--dry-run', action='store_true', help='no gamepad output, just print what the brain would do')
    q.add_argument('--udp-out', default='', help='send sticks to host:port over UDP instead of the gamepad (fake Liftoff)')
    q.add_argument('--seconds', type=float, default=0.0, help='stop after this long (0 = until Ctrl+C)')
    q.add_argument('--device', default='cuda')
    q.add_argument('--log', default='', help='CSV of every telemetry frame: pose, processed input, brain output, sticks, goal')
    q.add_argument('--flow-gain', type=float, default=1.0,
                   help='scale on the horizontal speed written into the optic-flow and airflow senses (< 1: the brain flies faster)')
    q.add_argument('--stick-model', choices=['auto', 'curves'], default='auto',
                   help='auto: the radial stick model when the mapping file has one; curves: the per-axis curves')
    q = ls.add_parser('fake', help='run a stand-in for Liftoff (telemetry out, sticks in over UDP) to rehearse the loop')
    q.add_argument('--port', type=int, default=9001)
    q.add_argument('--stick-port', type=int, default=9002)
    q.add_argument('--seconds', type=float, default=60.0)
    q.add_argument('--twr', type=float, default=5.5)
    q.add_argument('--thrust-exp', type=float, default=1.3)
    q.add_argument('--no-autopilot', action='store_true')
    q.add_argument('--seed', type=int, default=0)
    s.set_defaults(fn=cmd_liftoff)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
