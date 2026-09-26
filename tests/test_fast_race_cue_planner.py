"""Free-space corridor planner in the fast pilot (FastRaceCue plan=..., the runner's flag, logs and sidecar).

Pins, with synthetic planner samples (the perception module is built separately, interface fixed): the default
pilot and the stack without the planner are unchanged; the planner in shadow flies the stack with the planner off
and the gap aim unapplied bit for bit; a confirmed class rotates the ring ray about world z (left/right) or tilts it
up; a confirmed blocked or urgent path caps the speed along the applied aim (the lower of the planner and governor
caps wins); the planner displaces the TTC governor's terrain climb while its newest sample is fresh and valid, and
the governor climb comes back (levelled first while sinking) when it is not; the lag-turn lead is computed without
the planner offset; the runner's flag rules, declaration freeze check, CSV columns and sidecar. None of this is
flight evidence.
"""
import copy
import json

import numpy as np
import pytest
import torch

from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.corridor_aim import BLOCKED, CorridorAimConfig, VerticalGuardConfig
from haltere.liftoff.fast_race_cue import (PLAN_PILOT_COLUMNS, CeilingGuardConfig, FastRaceCue, LagTurnConfig,
                                           TurnFirstConfig, offset_ray)
from haltere.liftoff.gap_aim import GapAimConfig
from tests.test_corridor_aim import SPEC_V1, every, left, plan, right
from tests.test_fast_race_cue import cue_toward, single_thread  # noqa: F401 (fixture)
from tests.test_visual_assistance import SENSOR, senses

DT = .01
AIM = CorridorAimConfig()
GUARD = VerticalGuardConfig()
GAP = GapAimConfig()
PLANNER = dict(corridor_aim=AIM, vertical_guard=GUARD)


def heading(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def fly(pilot, history, *, seconds=1.5, start=10., plan_of=None, clearance_of=None, gap_of=None, speed=6.,
        vz=0., height=2., ring=(20., 0., 0.), plant=True, rate=17., latency=.1):
    """Straight flight at yaw 0 toward a ring ahead; planner/looming/gap samples every 1/rate s captured at the
    frame times and published `latency` later; plant=True reports the previous request as the measured motion
    (vz fixed when given as a number). Rows: dict per tick."""
    rows, measured = [], np.array([speed, 0., vz])
    captures = list(np.arange(start, start+seconds, 1/rate))
    latest = dict(plan=None, gap=None)
    last_clear = None
    for k in range(int(round(seconds/DT))):
        now = start+k*DT
        s = senses(position=(0., 0., height), velocity=tuple(measured.tolist()), yaw=0.)
        history.append(now, [0., 0., height], s['quat'][0].numpy())
        ready = [c for c in captures if c+latency <= now+1e-9]
        clearance = None
        if ready:
            c = float(ready[-1])
            if plan_of is not None:
                latest['plan'] = plan_of(c)
            if gap_of is not None:
                latest['gap'] = gap_of(c)
            if clearance_of is not None and c != last_clear:
                clearance = clearance_of(c)
                last_clear = c
        pilot.update(s, [0., 0., 0.], dict(race_cue=cue_toward(list(ring))), now-.05, now, clearance=clearance,
                     gap=latest['gap'], plan=latest['plan'])
        rows.append(dict(t=now, cmd=pilot.velocity_command.copy(), yaw=pilot.pilot.sight_yaw, state=pilot.state,
                         log=pilot.plan_log() if hasattr(pilot, 'plan_log') else None))
        if plant:
            measured = pilot.velocity_command.copy()
            if not callable(vz) and vz != 0.:
                measured[2] = vz
    return rows


def commands(rows):
    return np.array([np.r_[r['cmd'], r['yaw']] for r in rows])


def terrain(t):
    """A looming sample of ground below the path (the governor's terrain climb)."""
    return dict(time=t, ttc=.5, distance=3., below_fraction=.9, ttc_lower=.5)


# ---------------------------------------------------------------------------------------------
# Identity: off and shadow
# ---------------------------------------------------------------------------------------------
def test_the_default_pilot_ignores_planner_samples():
    runs = []
    for kwargs, plan_of in ((dict(), None), (dict(), lambda t: left(t, h_floor=.2)),
                            (dict(PLANNER, planner_apply=False), None)):
        history = CameraPoseHistory()
        pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **kwargs)
        runs.append(commands(fly(pilot, history, plan_of=plan_of, clearance_of=terrain)))
    np.testing.assert_array_equal(runs[0], runs[1])
    np.testing.assert_array_equal(runs[0], runs[2])          # a declared planner without samples changes nothing
    assert FastRaceCue(SENSOR, CameraPoseHistory(), 6.).metadata()['planner'] is None


def mixed_plan(t):
    """A busy planner stream: left shifts, then blocked with a cap, then an up class, a low floor, rising ground and
    a low ceiling, so every pilot rule has something to act on."""
    if t < 10.4:
        return left(t, az=10., el=3., h_floor=.4, h_ceil=1.2)
    if t < 10.7:
        return plan(t, 'blocked', v_cap=2., h_floor=.9, rise=1., rise_x=2.5)
    if t < 11.:
        return plan(t, 'shift', cls=2., el=9., v_el=9., v_ok=0., feasible=0., v_cap=3., h_floor=1., h_ceil=.7)
    return plan(t, 'clear', h_floor=.3)


def stack_kwargs(**extra):
    return dict(gap_aim=GAP, gap_apply=False, lag_turn=LagTurnConfig(), turn_first=TurnFirstConfig(),
                ceiling_guard=CeilingGuardConfig(), **extra)


@pytest.mark.parametrize('sink', [0., -.5])
def test_planner_shadow_flies_the_stack_without_it_bit_for_bit(sink):
    """Shadow (spec section 6): every planner quantity is computed and logged, the commands equal the same stack
    with the planner off and the gap aim unapplied, tick for tick (gap, looming, turn-first, ceiling guard, lag
    turns all active; also while sinking, where the applied planner would level off first)."""
    gap_of = lambda t: dict(time=t, shift=8., valid=True, ring_deg=0., lr=float('nan'), kind='gap')
    runs = {}
    for name, kwargs in (('off', stack_kwargs()), ('shadow', stack_kwargs(**PLANNER, planner_apply=False)),
                         ('on', stack_kwargs(**PLANNER))):
        history = CameraPoseHistory()
        pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **kwargs)
        rows = fly(pilot, history, seconds=1.6, plan_of=mixed_plan, clearance_of=terrain, gap_of=gap_of, vz=sink)
        runs[name] = (commands(rows), rows, pilot)
    np.testing.assert_array_equal(runs['off'][0], runs['shadow'][0])
    assert not np.array_equal(runs['off'][0], runs['on'][0])
    logs = [r['log'] for r in runs['shadow'][1]]
    assert max(l['plan_intended_az'] for l in logs) == 10. and all(l['plan_applied_az'] == 0. for l in logs)
    assert BLOCKED in [l['plan_cls_confirmed'] for l in logs] and 2 in [l['plan_cls_confirmed'] for l in logs]
    assert any(np.isfinite(l['plan_v_cap_intended']) for l in logs)
    assert all(np.isnan(l['plan_v_cap_applied']) for l in logs)
    assert any(np.isfinite(l['governor_climb_shadow']) and l['governor_climb_shadow'] > 0 for l in logs)
    meta = json.loads(json.dumps(runs['shadow'][2].metadata()))['planner']
    assert meta['applied'] is False and meta['version'] == 1
    assert meta['corridor_aim']['counts']['left_episodes'] == 1
    assert set(PLAN_PILOT_COLUMNS) == set(logs[0])


def test_the_planner_refuses_an_applied_gap_aim_and_needs_both_configs():
    with pytest.raises(ValueError, match='gap aim'):
        FastRaceCue(SENSOR, CameraPoseHistory(), 6., gap_aim=GAP, **PLANNER)
    with pytest.raises(ValueError, match='both'):
        FastRaceCue(SENSOR, CameraPoseHistory(), 6., corridor_aim=AIM)
    FastRaceCue(SENSOR, CameraPoseHistory(), 6., gap_aim=GAP, gap_apply=False, **PLANNER)


# ---------------------------------------------------------------------------------------------
# Aim, speed, vertical
# ---------------------------------------------------------------------------------------------
def test_offset_ray_rotates_left_and_tilts_up_and_inverts():
    v = np.array([1., 0., 0.])
    assert heading(offset_ray(v, 10., 0.)) == pytest.approx(10.)
    tilted = offset_ray(v, 0., 9.)
    assert np.degrees(np.arcsin(tilted[2])) == pytest.approx(9.) and np.linalg.norm(tilted) == pytest.approx(1.)
    w = offset_ray(offset_ray(np.array([.8, .3, -.2]), 7., 5.), 0., -5.)
    np.testing.assert_allclose(w, offset_ray(np.array([.8, .3, -.2]), 7., 0.), atol=1e-12)


def test_a_confirmed_left_class_rotates_the_aim_and_an_up_class_tilts_it():
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **PLANNER)
    rows = fly(pilot, history, plan_of=lambda t: left(t, az=8.))
    assert heading(rows[-1]['cmd']) == pytest.approx(8., abs=1.)
    assert np.linalg.norm(rows[-1]['cmd'][:2]) == pytest.approx(6., rel=.03)     # a feasible shift: no cap
    assert rows[-1]['log']['plan_applied_az'] == 8. and rows[-1]['log']['plan_cls_confirmed'] == 1
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **PLANNER)
    rows = fly(pilot, history, plan_of=lambda t: right(t, az=-6.))
    assert heading(rows[-1]['cmd']) == pytest.approx(-6., abs=1.)
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **PLANNER)
    up = lambda t: plan(t, 'shift', cls=2., el=9., v_el=9., v_ok=1., feasible=1.)
    rows = fly(pilot, history, plan_of=up, seconds=2.)
    cmd = rows[-1]['cmd']
    assert abs(heading(cmd)) < 1.
    assert np.degrees(np.arctan2(cmd[2], np.linalg.norm(cmd[:2]))) == pytest.approx(9., abs=1.5)
    assert cmd[2] <= 6.*np.tan(np.radians(12.))+1e-6                          # an up-shift is no hard climb


def test_a_confirmed_blocked_path_caps_the_speed_along_the_aim_and_the_lower_cap_wins():
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **PLANNER)
    rows = fly(pilot, history, plan_of=lambda t: plan(t, 'blocked', v_cap=2.), seconds=1.5)
    speeds = np.array([np.linalg.norm(r['cmd'][:2]) for r in rows])
    capped = next(i for i, r in enumerate(rows) if np.isfinite(r['log']['plan_v_cap_applied']))
    assert speeds[-1] == pytest.approx(2., abs=.05)
    assert np.all(np.diff(speeds[capped:]) >= -AIM.cap_slew_mps2*DT-1e-9)          # brought down at <= 15 m/s^2
    assert rows[-1]['log']['plan_cls_confirmed'] == BLOCKED
    # a lower governor cap (a wall ahead) wins; the planner's cap is then not applied
    wall = lambda t: dict(time=t, ttc=.45, distance=2.7, below_fraction=.5, ttc_lower=.45)
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **PLANNER)
    rows = fly(pilot, history, plan_of=lambda t: plan(t, 'blocked', v_cap=5.5), clearance_of=wall, seconds=1.5)
    late = rows[-1]
    assert pilot.clearance.cap < 5.5 and np.isnan(late['log']['plan_v_cap_applied'])
    assert late['log']['plan_v_cap_intended'] == 5.5


def test_the_planner_displaces_the_governor_climb_while_fresh_and_falls_back_when_stale():
    flat = lambda t: plan(t, 'clear', h_floor=1., h_ceil=1.5)
    # no planner: the governor's terrain climb (below-path looming) drives the request up to 3.5 m/s
    history = CameraPoseHistory()
    plain = fly(FastRaceCue(SENSOR, history, 6., reference_speed=6.), history, clearance_of=terrain)
    assert max(r['cmd'][2] for r in plain) > 3.
    # planner on with a fresh flat-floor sample: the climb is displaced (logged), the guard keeps the request level
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **PLANNER)
    rows = fly(pilot, history, clearance_of=terrain, plan_of=flat)
    fresh = [r for r in rows if r['log']['plan_fresh']]
    assert fresh and max(r['cmd'][2] for r in fresh) < .2
    assert max(r['log']['governor_climb_shadow'] for r in fresh) > 3.
    assert pilot.metadata()['planner']['counts']['displaced_climb_ticks'] > 0
    # the planner stale (not valid): the governor's climb applies as before
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **PLANNER)
    stale = lambda t: plan(t, 'no_scale', valid=False)
    rows = fly(pilot, history, clearance_of=terrain, plan_of=stale)
    np.testing.assert_array_equal(commands(rows), commands(plain))
    # ... but while sinking, the applied planner levels off first (a climb floor of 0), never a hard climb
    history = CameraPoseHistory()
    sinking = fly(FastRaceCue(SENSOR, history, 6., reference_speed=6.), history, clearance_of=terrain, vz=-.5)
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **PLANNER)
    rows = fly(pilot, history, clearance_of=terrain, plan_of=stale, vz=-.5)
    assert max(r['cmd'][2] for r in sinking) > 3.
    assert max(r['cmd'][2] for r in rows) < .1 and rows[-1]['cmd'][2] >= -1e-9
    assert pilot.metadata()['planner']['counts']['fallback_level_off_ticks'] > 0


def test_a_sink_toward_the_floor_is_stopped_gently_and_rising_ground_is_climbed_when_level():
    # sinking at 0.37 m/s from 0.7 m with the floor looming (minus-fast6-wall-01's pattern): no hard climb
    height = lambda t: .7-.37*(t-10.)
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **PLANNER)
    rows = fly(pilot, history, clearance_of=terrain, plan_of=lambda t: plan(t, 'clear', h_floor=height(t)),
               vz=-.37, seconds=1.2)
    assert max(r['cmd'][2] for r in rows) <= GUARD.gentle_up_mps+1e-9
    assert rows[-1]['log']['plan_vz_lo'] > -.37                               # the sink is bounded
    # level flight toward rising ground: the guard's terrain climb
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **PLANNER)
    rows = fly(pilot, history, plan_of=lambda t: plan(t, 'clear', h_floor=1., rise=1., rise_x=3.), seconds=1.)
    assert max(r['cmd'][2] for r in rows) > 3. and np.nanmax([r['log']['plan_climb'] for r in rows]) > 3.


def test_the_lag_turn_lead_is_computed_without_the_planner_offset():
    lag = LagTurnConfig()
    for apply in (True, False):
        pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., lag_turn=lag, lag_turn_apply=apply, **PLANNER)
        pilot.lag_turn_weight = 1.
        pilot.plan_offset_az = 12.
        ring = 12.7
        dh = np.array([np.cos(np.radians(ring+12.)), np.sin(np.radians(ring+12.))])
        goal = pilot._lead(dh, np.array([6., 0., 0.]))
        lead = float(np.clip(lag.course_lead*ring, -lag.course_lead_max_deg, lag.course_lead_max_deg))
        assert pilot.lag_turn_lead_deg == pytest.approx(lead)
        assert heading(goal) == pytest.approx(ring+12.+lead if apply else ring+12.)
        assert abs(heading(goal)-ring) <= lag.course_lead_max_deg+AIM.max_az_deg+1e-9


def test_another_ring_is_a_conflict_and_the_ring_aim_is_held():
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **PLANNER)
    fly(pilot, history, plan_of=lambda t: left(t, az=8.), seconds=.6)
    assert pilot.plan_offset_az > 0
    rows = fly(pilot, history, plan_of=lambda t: left(t, az=8., ring=25.), seconds=.5, start=10.6)
    assert 'ring' in [r['log']['plan_conflict'] for r in rows] and pilot.plan_offset_az == 0.
    assert pilot.corridor.counts['ring_conflicts'] == 1
    assert abs(heading(rows[-1]['cmd'])) < 2. and abs(heading(pilot.direction)) < .5      # the ring aim again


# ---------------------------------------------------------------------------------------------
# Runner: flag, declaration, logs
# ---------------------------------------------------------------------------------------------
def args(**kw):
    from types import SimpleNamespace
    base = dict(obstacle_stack=None, gap_cue=None, lag_turn=None, wall_pilot=None, pilot_profile='fast',
                looming_brake=True, obstacle_planner='off')
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_planner_flag_lives_inside_the_obstacle_stack():
    from haltere.liftoff.visual_brain import resolve_obstacle_planner, resolve_obstacle_stack
    resolve = lambda **kw: resolve_obstacle_planner(args(**kw), resolve_obstacle_stack(args(**kw)))
    assert resolve() is None and resolve(obstacle_planner=None) is None
    assert resolve(obstacle_stack='on') is None
    assert resolve(obstacle_stack='on', obstacle_planner='shadow') == 'shadow'
    assert resolve(obstacle_stack='on', obstacle_planner='on') == 'on'
    assert resolve(obstacle_stack='shadow', obstacle_planner='shadow') == 'shadow'
    with pytest.raises(ValueError, match='obstacle stack'):
        resolve(obstacle_planner='shadow')
    with pytest.raises(ValueError, match='stack on'):
        resolve(obstacle_stack='shadow', obstacle_planner='on')
    with pytest.raises(ValueError, match='gap'):
        resolve(obstacle_stack='on', gap_cue='off', obstacle_planner='on')
    with pytest.raises(ValueError, match='off, shadow or on'):
        resolve(obstacle_stack='on', obstacle_planner='maybe')
    # the stack's own resolution is unchanged by the new flag
    assert resolve_obstacle_stack(args(obstacle_stack='on', obstacle_planner='on'))['apply'] is True


def frozen(declaration):
    from haltere.liftoff.visual_brain import lag_turn_declaration_sha256
    obj = copy.deepcopy(declaration)
    obj.update(frozen=True, frozen_at='2026-09-26T12:00:00')
    obj['sha256'] = lag_turn_declaration_sha256(obj)
    return obj


def test_the_runner_flies_only_the_frozen_version_1_declaration(tmp_path):
    from haltere.liftoff.corridor_aim import planner_pilot_configs
    from haltere.liftoff.gap_stack import config_sha256
    from haltere.liftoff.visual_brain import load_free_space
    path = tmp_path/'free_space.json'
    good = frozen(SPEC_V1)
    assert good['sha256'] == config_sha256(good)                     # the gap_pilot.json canonical hash
    path.write_text(json.dumps(good), encoding='utf-8')
    declaration, digest = load_free_space(path)
    assert digest == good['sha256'] and planner_pilot_configs(declaration)['corridor'] == AIM
    edited = dict(good, pilot=dict(good['pilot'], confirm=1))
    path.write_text(json.dumps(edited), encoding='utf-8')
    with pytest.raises(ValueError, match='frozen'):
        load_free_space(path)
    path.write_text(json.dumps(SPEC_V1), encoding='utf-8')            # never frozen
    with pytest.raises(ValueError, match='frozen'):
        load_free_space(path)
    path.write_text(json.dumps(frozen(dict(SPEC_V1, version=2))), encoding='utf-8')
    with pytest.raises(ValueError, match='version'):
        load_free_space(path)
    with pytest.raises(ValueError, match='does not exist'):
        load_free_space(tmp_path/'missing.json')


def test_plan_samples_round_trip_through_the_shared_array_values():
    from haltere.liftoff.camera_process import PLAN_FIELDS, PLAN_KINDS, plan_sample, plan_values
    assert len(set(PLAN_FIELDS)) == len(PLAN_FIELDS) == 42 and PLAN_KINDS[0] == 'off'
    assert plan_sample([0.]*len(PLAN_FIELDS)) is None
    s = left(12.5, az=8., h_floor=.9, v_cap=None)
    s.update(seq=17., motor=1., age=.09, lk_ms=3.1)
    values = plan_values(s)
    assert len(values) == len(PLAN_FIELDS)
    back = plan_sample(values)
    assert back['kind'] == 'shift' and back['valid'] is True and back['time'] == 12.5 and back['seq'] == 17.
    assert back['l_az'] == 8. and back['h_floor'] == .9 and back['v_cap'] is None and back['r_az'] is None
    assert plan_values(back) == pytest.approx(values, nan_ok=True)
    invalid = plan_sample(plan_values(plan(3., 'no_scale', valid=False)))
    assert invalid['kind'] == 'no_scale' and invalid['valid'] is False


def test_the_camera_reads_plan_out_without_blocking():
    import multiprocessing as mp
    from haltere.liftoff.camera_process import PLAN_FIELDS, ProcessRetinaCamera, plan_values
    camera = ProcessRetinaCamera.__new__(ProcessRetinaCamera)
    context = mp.get_context('spawn')
    camera.data = context.Array('d', 1000, lock=True)
    camera.plan_out = context.Array('d', len(PLAN_FIELDS), lock=True)
    camera.queue, camera.done = context.Queue(), context.Event()
    camera.process = type('P', (), dict(exitcode=None, pid=None))()
    camera._latest = camera._error = None
    assert camera.plan is None
    np.frombuffer(camera.plan_out.get_obj())[:] = plan_values(dict(left(5., az=6.), seq=3.))
    assert camera.plan['l_az'] == 6. and camera.plan['seq'] == 3.
    import threading
    held, done = threading.Event(), threading.Event()

    def writer():                                                      # the depth process is writing
        with camera.plan_out.get_lock():
            np.frombuffer(camera.plan_out.get_obj())[:] = plan_values(dict(left(5.1, az=7.), seq=4.))
            held.set()
            done.wait(5.)
    thread = threading.Thread(target=writer)
    thread.start()
    held.wait(5.)
    assert camera.plan['seq'] == 3. and camera.plan_busy == 1           # skipped and counted, never waited
    done.set()
    thread.join()
    assert camera.plan['seq'] == 4.


def test_log_columns_rows_and_sidecar():
    from haltere.liftoff.camera_process import PLAN_FIELDS
    from haltere.liftoff.visual_brain import PLAN_COLUMNS, PlanStats, plan_row, planner_stack_metadata
    assert PLAN_COLUMNS[:len(PLAN_FIELDS)] == tuple(f'plan_{f}' for f in PLAN_FIELDS)
    assert set(PLAN_PILOT_COLUMNS) <= set(PLAN_COLUMNS) and len(PLAN_COLUMNS) == len(set(PLAN_COLUMNS))
    assert len(plan_row(None, None)) == len(PLAN_COLUMNS)
    off = dict(zip(PLAN_COLUMNS, plan_row(None, FastRaceCue(SENSOR, CameraPoseHistory(), 6.))))
    assert off['plan_kind'] == '' and np.isnan(off['plan_time']) and np.isnan(off['plan_fresh'])
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **PLANNER)
    fly(pilot, history, plan_of=lambda t: left(t, az=8., h_floor=1.), seconds=.8)
    row = dict(zip(PLAN_COLUMNS, plan_row(None, pilot)))
    assert row['plan_kind'] == 'shift' and row['plan_valid'] == 1 and row['plan_l_az'] == 8.
    assert row['plan_cls_confirmed'] == 1 and row['plan_applied_az'] == 8. and row['plan_fresh'] == 1
    stats = PlanStats()
    for k in range(5):
        stats.add(dict(left(1.+k*.06, age=.09+k*.01, plan_ms=2., lk_ms=3., depth_ms=20., scale_n=80.), seq=k))
        stats.add(dict(left(1.+k*.06), seq=k))                          # repeated seq: counted once
    summary = stats.summary()
    assert summary['samples'] == 5 and summary['kinds'] == {'shift': 5}
    assert summary['percentiles']['age']['p50'] == pytest.approx(.11)
    meta = planner_stack_metadata('shadow', ('configs/obstacles/gap_pilot.json', SPEC_V1, 'abc'),
                                  dict(motor_contract='fast_velocity_pd_v1', free_space=dict(motor='fast_pd')),
                                  dict(gap=dict(provenance=dict(weights_sha256='3152477cdead'))), stats)
    meta = json.loads(json.dumps(meta))
    assert meta['mode'] == 'shadow' and meta['applied'] is False and meta['free_space']['version'] == 1
    assert meta['motor_model'] == 'fast_pd' and meta['depth_weights_sha256_prefix'] == '3152477c'
    assert meta['depth_weights_sha256'].startswith('3152477c') and meta['received']['samples'] == 5
    assert meta['response_models_file_sha256'] is not None
    assert planner_stack_metadata(None, None, None, None, None) is None


def test_runner_refuses_the_planner_outside_the_fast_pilot():
    from haltere.liftoff.visual_brain import VisualController
    with pytest.raises(ValueError, match='fast pilot'):
        VisualController('missing.pt', 'missing.json', 'cpu', pilot_assistance='race-cue',
                         planner=dict(corridor=AIM, vertical=GUARD))
