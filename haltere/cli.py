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
             ('docs/liftoff_square.gif', 'liftoff_square.gif'), ('docs/flight.gif', 'flight.gif'),
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
    q.add_argument('--pattern', default='', choices=['', 'orbit', 'climbdive', 'figure8'],
                   help='moving target pattern around the offset point (freestyle)')
    q.add_argument('--radius', type=float, default=3.0, help='pattern radius (m)')
    q.add_argument('--period', type=float, default=12.0, help='pattern period (s)')
    q.add_argument('--amplitude', type=float, default=1.5, help='climb/dive altitude swing (m)')
    q.add_argument('--stick-gain', type=float, default=1.0, help='scale roll/pitch/yaw commands (smoothness)')
    q.add_argument('--stick-lpf', type=float, default=0.0, help='low-pass time constant on the sticks (s)')
    q.add_argument('--gyro', choices=['quat', 'telemetry'], default='quat',
                   help='body rates from attitude differences (quat) or from Liftoff\'s Gyro field (telemetry)')
    q.add_argument('--throttle-scale', type=float, default=0.0, help='gain of the throttle remap around hover (0 = keep file value)')
    q.add_argument('--arm-hold', type=float, default=0.8, help='seconds of throttle-low before the brain gets control (arming)')
    q.add_argument('--arm-ramp', type=float, default=1.2, help='seconds over which the brain\'s sticks are ramped in')
    q.add_argument('--reset-button', default='', help='virtual pad button that resets the drone in Liftoff (e.g. A, Y, BACK); pressed after a crash')
    q.add_argument('--record', default='', help='write an MP4 of the Liftoff window + live brain activity')
    q.add_argument('--capture', default='Liftoff', help='window title (substring) to capture for --record')
    q.add_argument('--capture-rect', default='', help='x,y,w,h screen region to capture instead of a window')
    q.add_argument('--fps', type=int, default=20)
    q.add_argument('--show', action='store_true', help='open a live window of the brain activity')
    q.add_argument('--dry-run', action='store_true', help='no gamepad output, just print what the brain would do')
    q.add_argument('--udp-out', default='', help='send sticks to host:port over UDP instead of the gamepad (fake Liftoff)')
    q.add_argument('--seconds', type=float, default=0.0, help='stop after this long (0 = until Ctrl+C)')
    q.add_argument('--device', default='cuda')
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
