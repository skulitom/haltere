"""The odd-course bench: what does the by-sight pilot assume about the Straw Bale course?

The rabbit pilot (``haltere.liftoff.sightpilot``) flies clean 7/7 laps on Straw Bale. That course is one
shape: seven 4 m arches, the passage height only ever rising (1.2 m -> 13.0 m), gates 15-40 m apart,
turns anticlockwise, the first gate straight ahead of the spawn. Every one of those is an assumption,
and the pilot has constants that depend on them. This module generates synthetic gate lists (the shape
``gates.load_gate_file`` returns) for a suite of courses that break one assumption each, flies the real
pilot through each of them with the rehearsal of ``vision.rehearse`` (no game, no GateNet), and prints
what it passed and which of the pilot's own rejection counters fired.

It measures; it does not fix anything and it changes no pilot behaviour.

A caveat the bench cannot hide, and prints as a warning for every course whose width is not 4 m: the
nominal gate width is hard-coded in three places outside this module, so a course with another width is
only partly simulated.

  * ``vision.runtime.detection_geometry`` (runtime.py:48) turns the detected pixel width into a range
    with ``gates.GATE_WIDTH_M`` = 4.0, whatever the course says. The synthetic detector projects the
    course's real width, so the pilot's range is wrong by 4.0 / width (a 1.5 m arch reads 2.7x too far)
    and the arch drops below the detector's 11 px floor 2.7x nearer.
  * ``rehearse.arch_collision`` (rehearse.py:244) puts the posts at 1.7-2.3 m either side and the top
    bar at 3.2-3.8 m, a 4 m arch, for every course.
  * ``flightlog.gate_crossings`` (flightlog.py:56) scores a pass with a fixed 2.0 m half width; the
    bench therefore also reports a width-aware count (``through_w``) beside the project's own.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..liftoff.sightpilot import SightParams
from .gates import CENTRE_UP_M, GATE_WIDTH_M
from .rehearse import LOG_COLUMNS as REHEARSE_COLUMNS
from .rehearse import DetectorModel, RehearsalOptions, run_rehearsal, summarize

# the published by-sight preset (README: --sight-speed 3.5 --sight-gate-speed 3.2 --sight-turn-gate-speed 2.8
# --sight-flow-min 0.6 --sight-flow-alt ground), with the simulator's yaw rate as `vision rehearse` uses it
PRESET = dict(v_cruise=3.5, v_gate=3.2, v_gate_turn=2.8, flow_min=0.6, flow_alt='ground', yaw_rate=3.8)
FIXED_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)          # --seeds N takes the first N: the bench is reproducible
# A course's time budget: its length at the pace the pilot actually keeps on Straw Bale (7/7 in 125 s over 205 m,
# the first gate 18 s in), plus a margin, and never outside BUDGET_S.
CRUISE_GUESS = 1.3                              # m/s of course length per simulated second
BUDGET_MARGIN_S = 25.0
BUDGET_S = (150.0, 220.0)

# Straw Bale itself, copied from configs/gates_strawbale.json (x, y, passage height, heading in radians).
# tests/test_oddcourse.py checks that this copy still matches the file.
HOME_GATES = (
    (25.885, 0.670, 1.200, 0.0120995),
    (60.933, 1.575, 1.200, 0.1613049),
    (79.320, 16.480, 1.200, 1.5091967),
    (59.437, 44.003, 1.200, 2.3570509),
    (41.608, 66.938, 1.765, 1.9349965),
    (37.175, 99.100, 6.625, 1.6753389),
    (35.175, 123.968, 12.987, 1.6478858),
)


# ----------------------------------------------------------------------------- a course

@dataclass
class Course:
    """A synthetic course: the gate list plus what the bench needs to fly and read it."""
    name: str
    gates: list[dict]
    width_m: float = GATE_WIDTH_M
    up_m: float = CENTRE_UP_M
    start: tuple = (0.0, 0.0, 0.0, 0.0)          # spawn x, y, z (m) and yaw (deg), as `vision rehearse --start`
    breaks: str = ''                             # the Straw Bale assumption this course breaks
    seconds: float = 0.0                         # 0: derived from the course length

    def as_gate_file(self) -> dict:
        """The dict ``gates.load_gate_file`` reads (and ``gates.save_gates`` writes)."""
        return {'gate_width_m': self.width_m, 'centre_up_m': self.up_m, 'gates': self.gates}

    @property
    def length_m(self) -> float:
        p = [np.array(self.start[:3], dtype=np.float64)] + [np.asarray(g['pos'], dtype=np.float64) for g in self.gates]
        return float(sum(np.linalg.norm(b - a) for a, b in zip(p[:-1], p[1:])))

    @property
    def budget_s(self) -> float:
        if self.seconds > 0:
            return float(self.seconds)
        return float(min(max(self.length_m / CRUISE_GUESS + BUDGET_MARGIN_S, BUDGET_S[0]), BUDGET_S[1]))

    @property
    def legs_m(self) -> list[float]:
        p = [np.asarray(g['pos'], dtype=np.float64)[:2] for g in self.gates]
        return [float(np.linalg.norm(b - a)) for a, b in zip(p[:-1], p[1:])]

    def width_warning(self) -> str:
        if abs(self.width_m - GATE_WIDTH_M) < 1e-9:
            return ''
        k = GATE_WIDTH_M / self.width_m
        return (f'{self.name}: gate_width_m {self.width_m} != {GATE_WIDTH_M}. The pilot\'s range comes from '
                f'runtime.detection_geometry, which divides by the hard-coded gates.GATE_WIDTH_M = {GATE_WIDTH_M}: '
                f'every sighting reads {k:.2f}x its true range, and the 11 px detector floor cuts sight off at '
                f'{100.0 * self.width_m / 11.0:.0f} m instead of {100.0 * GATE_WIDTH_M / 11.0:.0f} m. '
                f'rehearse.arch_collision still builds a 4 m arch, and flightlog.gate_crossings still scores with a '
                f'2.0 m half width (the bench also reports through_w at {self.width_m / 2:.2f} m). Results on this '
                f'course measure the pilot AND that mismatch together.')


def _headings(points, start=(0.0, 0.0)) -> list[float]:
    """Gate facings from the course line, as ``gates.gates_from_passage_frames`` derives them from a flight:
    the direction of travel at the gate, i.e. the bisector of the leg in and the leg out."""
    pts = [(float(start[0]), float(start[1]))] + [(float(x), float(y)) for x, y in points]
    dirs = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(pts[:-1], pts[1:])]
    out = []
    for i in range(len(points)):
        d_in = dirs[i]
        d_out = dirs[i + 1] if i + 1 < len(dirs) else d_in
        cx, cy = math.cos(d_in) + math.cos(d_out), math.sin(d_in) + math.sin(d_out)
        out.append(math.atan2(cy, cx) if math.hypot(cx, cy) > 1e-6 else d_in)
    return out


def _gates(points, heights, headings=None, start=(0.0, 0.0)) -> list[dict]:
    h = list(headings) if headings is not None else _headings(points, start)
    return [{'pos': [round(float(x), 3), round(float(y), 3), round(float(z), 3)], 'heading': float(a), 'gate': i}
            for i, ((x, y), z, a) in enumerate(zip(points, heights, h))]


# ----------------------------------------------------------------------------- the suite

def home() -> Course:
    """Straw Bale itself: the regression control. Whatever else the bench says, this one must keep passing."""
    return Course('home', [{'pos': [x, y, z], 'heading': h, 'gate': i} for i, (x, y, z, h) in enumerate(HOME_GATES)],
                  breaks='nothing (the control)')


def _home_xy() -> list[tuple[float, float]]:
    return [(x, y) for x, y, _, _ in HOME_GATES]


def _home_z() -> list[float]:
    return [z for _, _, z, _ in HOME_GATES]


def _home_head() -> list[float]:
    return [h for _, _, _, h in HOME_GATES]


def narrow() -> Course:
    """Straw Bale with 1.5 m arches: the width the pilot's range conversion does not know about."""
    c = home()
    c.name, c.width_m, c.breaks = 'narrow', 1.5, 'gate width 1.5 m (Straw Bale is 4 m)'
    return c


def wide() -> Course:
    """Straw Bale with 8 m arches."""
    c = home()
    c.name, c.width_m, c.breaks = 'wide', 8.0, 'gate width 8 m (Straw Bale is 4 m)'
    return c


def descending() -> Course:
    """Straw Bale's layout, the heights running the other way: high at the first gate, down to the ground.

    9 m, not the 14 m a true mirror of home would want: the pilot's altitude reference climbs at most
    ``vz_max`` = 1.0 m/s (sightpilot.py:1147) and its target is clipped to ``z_aim_last + z_window[1]`` =
    1.5 + 12 m on the first gate (sightpilot.py:1138), so a 14 m first gate is out of reach for reasons that
    have nothing to do with descending and would mask what this course is for. Every drop here is larger than
    1.2 m, which is what it takes for the next arch's visual centre to sit below the last passage height."""
    z = [9.0, 7.0, 5.2, 3.6, 2.2, 1.4, 1.2]
    return Course('descending', _gates(_home_xy(), z, _home_head()),
                  breaks='passage height falls 9.0 -> 1.2 m (Straw Bale only ever climbs)')


def mixed_height() -> Course:
    """Straw Bale's layout with the heights up and down repeatedly."""
    z = [1.5, 9.0, 3.0, 12.0, 2.0, 8.0, 4.0]
    return Course('mixed_height', _gates(_home_xy(), z, _home_head()),
                  breaks='passage height up and down (1.5, 9, 3, 12, 2, 8, 4 m)')


def gate_pair() -> Course:
    """Two arches 6 m apart on the same line, twice in the course (gates 1-2 and 4-5)."""
    pts = [(26.0, 0.0), (52.0, 0.0), (58.0, 0.0), (84.0, 6.0), (104.0, 28.0), (108.243, 32.243), (128.0, 52.0)]
    hd = [0.0, 0.0, 0.0, math.radians(20.0), math.radians(45.0), math.radians(45.0), math.radians(45.0)]
    z = [1.2, 1.2, 1.2, 1.8, 2.6, 2.9, 3.8]
    return Course('gate_pair', _gates(pts, z, hd), breaks='two gates 6 m apart on one line, twice')


def hairpin() -> Course:
    """A course that doubles back: the leg into gate 2 and the leg out of gate 4 differ by about 178 deg."""
    pts = [(25.0, 0.0), (55.0, 0.0), (85.0, 5.0), (90.0, 20.0), (70.0, 26.0), (42.0, 20.0), (15.0, 12.0)]
    z = [1.2, 1.2, 1.5, 2.0, 2.6, 3.2, 4.0]
    return Course('hairpin', _gates(pts, z), breaks='a >150 deg direction reversal between two legs')


def clockwise() -> Course:
    """Straw Bale mirrored about its first leg: every turn is the other way."""
    pts = [(x, -y) for x, y in _home_xy()]
    hd = [-h for h in _home_head()]
    return Course('clockwise', _gates(pts, _home_z(), hd), breaks='the course turns clockwise')


def long_legs() -> Course:
    """Six gates 85-95 m apart: further than an arch can be seen (a 4 m arch falls under the detector's
    11 px floor beyond about 36 m), so nothing is ever in sight after a pass."""
    pts = [(30.0, 0.0), (115.0, 10.0), (205.0, 40.0), (270.0, 105.0), (290.0, 195.0), (250.0, 275.0)]
    z = [1.2, 2.0, 3.0, 4.0, 5.0, 6.0]
    # a fixed budget, not the length: whatever happens here happens on the first leg, and the rest of the
    # course would only buy more searching
    return Course('long_legs', _gates(pts, z), breaks='gates 85-95 m apart (Straw Bale is 15-40 m)', seconds=150.0)


def offaxis_start() -> Course:
    """Straw Bale, but the drone spawns facing 90 deg to the right of the first gate (which is outside the
    116 deg camera's view at the spawn)."""
    c = home()
    c.name, c.start = 'offaxis_start', (0.0, 0.0, 0.0, 90.0)
    c.breaks = 'the first gate is 90 deg to the side of the spawn heading'
    return c


BUILDERS = {'home': home, 'narrow': narrow, 'wide': wide, 'descending': descending, 'mixed_height': mixed_height,
            'gate_pair': gate_pair, 'hairpin': hairpin, 'clockwise': clockwise, 'long_legs': long_legs,
            'offaxis_start': offaxis_start}


def build(name: str) -> Course:
    if name not in BUILDERS:
        raise KeyError(f'no such course {name!r}; have {", ".join(BUILDERS)}')
    return BUILDERS[name]()


def all_courses(names=None) -> list[Course]:
    return [build(n) for n in (names or list(BUILDERS))]


# ----------------------------------------------------------------------------- flying one course

def preset_params() -> SightParams:
    """A fresh SightParams of the published by-sight preset (fresh: run_rehearsal writes flow_max into it)."""
    return SightParams(**PRESET)


def _columns(rows: list[list]) -> dict[str, np.ndarray]:
    return {c: np.array([r[j] for r in rows], dtype=np.float64)
            for j, c in enumerate(REHEARSE_COLUMNS) if c != 'status'}


def _width_aware_passes(rows: list[list], course: Course) -> list[int]:
    """The gates flown through when a pass is judged against the course's own half width instead of the
    2.0 m that ``flightlog.gate_crossings`` assumes. Best attempt."""
    from ..liftoff.flightlog import attempts, gate_crossings
    log = _columns(rows)
    best: list[int] = []
    for idx in attempts(log):
        P = np.c_[log['px'][idx], log['py'][idx], log['pz'][idx]]
        ts = log['ts'][idx] - log['ts'][idx][0]
        cr = gate_crossings(P, ts, course.gates, half_width=max(course.width_m / 2.0, 0.5))
        got = sorted({c['gate'] for c in cr if c['through']})
        if len(got) > len(best):
            best = got
    return best


def fly_course(course: Course, brain, cfg, cam, seed: int, seconds: float | None = None, verbose: bool = False,
               max_gpu_temp: float = 70.0, burst_s: float = 60.0, cool_s: float = 15.0,
               log_path: str | Path | None = None) -> dict:
    """One rehearsal of one course with one seed. Returns the row the table and the JSON are built from."""
    opts = RehearsalOptions(seconds=float(seconds or course.budget_s), seed=seed, start=tuple(course.start),
                            sight='rabbit', sight_params=preset_params(), flow_gain=1.0, verbose=verbose,
                            max_gpu_temp=max_gpu_temp, burst_s=burst_s, cool_s=cool_s)
    res = run_rehearsal(brain, cfg, course.gates, cam, log_path, opts, DetectorModel(), course.width_m, course.up_m)
    s = summarize(res, course.gates, None, course.up_m)
    best = max(s['attempts'], key=lambda r: len(r.get('gates_through', [])), default={})
    through = list(best.get('gates_through', []))
    passed_t = [c['t'] for c in best.get('crossings', []) if c['through']]
    sight = dict(best.get('sight', {}))
    return {
        'course': course.name, 'seed': seed, 'n_gates': len(course.gates), 'width_m': course.width_m,
        'seconds': opts.seconds, 'attempts': len(s['attempts']),
        'through': through, 'n_through': len(through),
        'through_w': _width_aware_passes(res['rows'], course),
        'first_gate_s': round(min(passed_t), 1) if passed_t else None,
        'last_gate_s': round(max(passed_t), 1) if passed_t else None,
        'lateral_max_m': best.get('lateral_max_through'),
        'crossings': [{'gate': c['gate'], 't': round(c['t'], 1), 'lat': round(c['lateral_m'], 2),
                       'dz': round(c['dz_m'], 2), 'through': bool(c['through'])} for c in best.get('crossings', [])],
        'counters': {k: (None if k not in sight else (sight[k] if isinstance(sight[k], dict) else float(sight[k])))
                     for k in ('n_passes', 'pass_kinds', 'ghosts', 'unpasses', 'reseeds', 'goal_clips', 'rej_elev',
                               'rej_stale', 'absorbed', 'low', 'behind', 'sight_errors', 'mode_s')},
        'gate_estimate': best.get('gate_estimate'),
        'detector': s['detector'],
        'crashes': [{'kind': c['kind'], 'gate': c.get('gate'), 'ts': round(c['ts'], 1)} for c in s['crashes']],
        'z_range': best.get('z_range'),
        'wall_s': round(res['wall_s'], 1),
    }


# ----------------------------------------------------------------------------- the bench

def _agg(runs: list[dict], key: str) -> float:
    vals = [r['counters'].get(key) for r in runs]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return float(np.median(vals)) if vals else float('nan')


def table(rows: list[dict], courses: dict[str, Course]) -> str:
    """One line per course: what it passed on each seed and the counters that explain it."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r['course'], []).append(r)
    head = (f'{"course":<14}{"w(m)":>5}{"gates":>6}  {"through/seed":<14}{"w-aware":>8}{"t_last":>8}'
            f'{"passes":>7}{"ghost":>6}{"low":>5}{"behind":>7}{"absorb":>7}{"unpass":>7}{"reseed":>7}'
            f'{"elev":>6}{"clips":>6}{"crash":>6}{"err":>4}')
    out = [head, '-' * len(head)]
    for name, runs in by.items():
        c = courses[name]
        thr = ','.join(str(r['n_through']) for r in runs)
        wa = ','.join(str(len(r['through_w'])) for r in runs)
        tl = [r['last_gate_s'] for r in runs if r['last_gate_s'] is not None]
        out.append(f'{name:<14}{c.width_m:>5.1f}{runs[0]["n_gates"]:>6}  {thr:<14}{wa:>8}'
                   f'{(f"{np.median(tl):.0f}" if tl else "-"):>8}'
                   f'{_agg(runs, "n_passes"):>7.0f}{_agg(runs, "ghosts"):>6.0f}{_agg(runs, "low"):>5.0f}'
                   f'{_agg(runs, "behind"):>7.0f}{_agg(runs, "absorbed"):>7.0f}{_agg(runs, "unpasses"):>7.0f}'
                   f'{_agg(runs, "reseeds"):>7.0f}{_agg(runs, "rej_elev"):>6.0f}{_agg(runs, "goal_clips"):>6.0f}'
                   f'{float(np.median([len(r["crashes"]) for r in runs])):>6.0f}'
                   f'{_agg(runs, "sight_errors"):>4.0f}')
    out.append('through/seed: gates flown through per seed (flightlog.gate_crossings, 2.0 m half width); '
               'w-aware: the same with the course\'s own half width; counters are the median over the seeds.')
    return '\n'.join(out)


def run_bench(ckpt: str, camera_yaml: str = 'configs/camera_seat.yaml', names=None, seeds: int = 3,
              seconds: float | None = None, device: str = 'cuda', json_out: str | None = None,
              verbose: bool = True) -> dict:
    """Fly every course in the suite with every seed and print the table. Returns the whole record."""
    import yaml

    from ..train.bptt import load_checkpoint
    from .camera import Camera
    cam_d = yaml.safe_load(Path(camera_yaml).read_text(encoding='utf-8'))
    cam = Camera(int(cam_d['width']), int(cam_d['height']), float(cam_d['f']), float(cam_d['tilt_deg']))
    courses = all_courses(names)
    seed_list = list(FIXED_SEEDS[:max(1, int(seeds))])
    brain, cfg, _ = load_checkpoint(ckpt, device)
    warnings = [w for w in (c.width_warning() for c in courses) if w]
    if verbose:
        print(f'odd-course bench: brain {ckpt} ({getattr(brain, "N", 0)} neurons on {brain.device}), '
              f'{len(courses)} courses x {len(seed_list)} seeds {seed_list}, camera {cam.hfov_deg:.0f} deg HFOV\n'
              f'  pilot: the published by-sight preset, {preset_params().describe()}', flush=True)
        for c in courses:
            legs = c.legs_m
            print(f'  {c.name:<14} {len(c.gates)} gates, width {c.width_m} m, legs '
                  f'{min(legs):.0f}-{max(legs):.0f} m, height {min(g["pos"][2] for g in c.gates):.1f}-'
                  f'{max(g["pos"][2] for g in c.gates):.1f} m, budget {c.budget_s:.0f} s  [{c.breaks}]', flush=True)
        for w in warnings:
            print(f'  WARNING  {w}', flush=True)
    rows = []
    t0 = time.time()
    for c in courses:
        for seed in seed_list:
            r = fly_course(c, brain, cfg, cam, seed, seconds)
            rows.append(r)
            if verbose:
                cn = r['counters']
                print(f'  {c.name:<14} seed {seed}: through {r["n_through"]}/{r["n_gates"]} {r["through"]} '
                      f'(width-aware {len(r["through_w"])}), passes {cn["n_passes"]:.0f} {cn["pass_kinds"]}, '
                      f'ghosts {cn["ghosts"]:.0f}, low {cn["low"]:.0f}, behind {cn["behind"]:.0f}, absorbed '
                      f'{cn["absorbed"]:.0f}, unpasses {cn["unpasses"]:.0f}, reseeds {cn["reseeds"]:.0f}, rej_elev '
                      f'{cn["rej_elev"]:.0f}, errors {cn["sight_errors"]:.0f}, modes {cn["mode_s"]}, crashes '
                      f'{len(r["crashes"])} [{r["wall_s"]:.0f} s wall]', flush=True)
    out = {'ckpt': ckpt, 'camera': camera_yaml, 'seeds': seed_list, 'preset': PRESET, 'warnings': warnings,
           'courses': {c.name: {'breaks': c.breaks, 'width_m': c.width_m, 'n_gates': len(c.gates),
                                'budget_s': c.budget_s, 'start': list(c.start), 'gates': c.as_gate_file()}
                       for c in courses},
           'runs': rows, 'wall_s': round(time.time() - t0, 1)}
    text = table(rows, {c.name: c for c in courses})
    print('\n' + text, flush=True)
    for w in warnings:
        print(f'WARNING  {w}', flush=True)
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(out, indent=1, default=float), encoding='utf-8')
        print(f'written to {json_out}', flush=True)
    return out
