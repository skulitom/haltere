"""FastRaceCue vertical guard (`VerticalGuardConfig`), declared in configs/obstacles/vertical_guard.json (obstacle
stack): a time margin to the ground below the path, descent first, and terrain climbs only for rising ground.

These tests pin the declared rules on synthetic samples (the sink margin and its ramp, descent first, confirmation,
the gentle bound and its escalation, the height bound, the kept horizontal speed), that the ceiling guard still cuts a
climb, that the default behaviour is unchanged and that the shadow control flies exactly the unguarded pilot. The
open-loop replays of the live logs are in haltere/obstacles/vertical_replay.py. None of this is flight evidence.
"""
import json
from dataclasses import asdict

import numpy as np
import pytest

from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import (VERTICAL_GUARD_VERSION, CeilingGuardConfig, ClearanceConfig, FastRaceCue,
                                           TtcClearanceConfig, TtcClearanceGovernor, VerticalGuardConfig,
                                           vertical_guard_config)
from tests.test_fast_race_cue import cue_toward, drive, single_thread  # noqa: F401 (fixture)
from tests.test_visual_assistance import SENSOR, senses

X = [1., 0., 0.]
FRAME = 1/18.
TTC = TtcClearanceConfig()               # the flown TTC policy
GUARD = CeilingGuardConfig()
VG = VerticalGuardConfig()
AHEAD = cue_toward([10., 0., 0.])
BELOW = cue_toward([10., 0., -1.5])      # a ring ahead and below: the pilot requests a sink of about 0.9 m/s


def run(gov, sample_of, seconds, *, start=0., speed=6., height=1., follow=False, delay=.08, rise=0.):
    """Drone along +x at `speed`; sample_of(t) -> None or dict(ttc, below, lower) at 18 Hz, `delay` late.
    `rise` is the measured vertical speed (m/s; a number or a function of time). follow=True flies the climb request
    (the height integrates it and the measured vertical speed is the request). Rows: dict(t, z, vz, climb, factor,
    arrest, stage)."""
    rows, pending, next_frame, z, vz = [], [], start, height, 0.
    for k in range(int(round(seconds/.01))):
        now = start+k*.01
        x = speed*now
        if now >= next_frame-1e-9:
            s = sample_of(now)
            if s is not None:
                pending.append((now+delay, now, x, z, s))
            next_frame += FRAME
        while pending and pending[0][0] <= now+1e-9:
            _, t, xc, zc, s = pending.pop(0)
            ttc = s.get('ttc')
            gov.ingest(t, ttc, None if ttc is None else ttc*speed, s.get('below'), [xc, 0., zc], X, speed,
                       received=now, ttc_lower=s.get('lower'))
        if not follow:
            vz = rise(now) if callable(rise) else rise
        _, _, climb = gov.limits([x, 0., z], [speed, 0., vz], now, .01, 3.5)
        if follow:
            vz = climb if climb > 0 else (rise(now) if callable(rise) else rise)
            z += vz*.01
        rows.append(dict(t=now, z=z, vz=vz, climb=climb, factor=gov.sink_factor, arrest=gov.arrest,
                         stage=0 if climb <= 0 else 2 if gov.escalated else 1))
    return rows


def series(rows, key):
    return np.array([np.nan if r[key] is None else r[key] for r in rows], float)


# ---------------------------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------------------------
def test_config_validation():
    for bad in (dict(margin_zero_s=2.), dict(climb_full_s=1.5), dict(confirm=1.5), dict(rising_confirm=0),
                dict(gentle_climb=float('inf')), dict(level_band=-.1), dict(memory_s=0.)):
        with pytest.raises(ValueError):
            VerticalGuardConfig(**bad)
    with pytest.raises(ValueError, match='VerticalGuardConfig'):
        TtcClearanceGovernor(TTC, vertical=GUARD)
    with pytest.raises(ValueError, match='TTC clearance policy'):
        FastRaceCue(SENSOR, CameraPoseHistory(), 6., clearance_config=ClearanceConfig(), vertical_guard=VG)
    with pytest.raises(ValueError, match='VerticalGuardConfig'):
        FastRaceCue(SENSOR, CameraPoseHistory(), 6., vertical_guard=GUARD)


def test_repository_declaration_is_frozen_and_declares_the_defaults(tmp_path):
    from haltere.liftoff.visual_brain import (VERTICAL_GUARD_DECLARATION, lag_turn_declaration_sha256,
                                              load_vertical_guard)
    declaration, digest = load_vertical_guard(VERTICAL_GUARD_DECLARATION)
    assert declaration['version'] == VERTICAL_GUARD_VERSION == 1 and declaration['frozen'] is True
    assert digest == declaration['sha256'] == lag_turn_declaration_sha256(declaration)
    assert vertical_guard_config(declaration) == VG                     # the declared values are the defaults
    # the declared values the task gave: full sink at 1.5 s, none at 0.6 s; about 1 m/s without rising ground
    assert (VG.margin_full_s, VG.margin_zero_s, VG.gentle_climb) == (1.5, .6, 1.)
    edited = dict(declaration, vertical_guard=dict(declaration['vertical_guard'], gentle_climb=2.))
    path = tmp_path/'edited.json'
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match='changed after the freeze'):
        load_vertical_guard(path)
    other = {k: v for k, v in declaration.items() if k not in ('frozen', 'frozen_at', 'sha256')}
    other['version'] = 2
    other.update(frozen=True, sha256=lag_turn_declaration_sha256(other))
    path.write_text(json.dumps(other))
    with pytest.raises(ValueError, match='version'):
        load_vertical_guard(path)
    with pytest.raises(ValueError, match='version'):
        vertical_guard_config(other)


# ---------------------------------------------------------------------------------------------
# Governor: sink margin, descent first, rising ground
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize('lower, expected', [(3., 1.), (1.5+.14, 1.), (1.05+.11, .5), (.6+.09, 0.), (.3, 0.)])
def test_sink_factor_follows_the_below_path_ttc_margin(lower, expected):
    """A below-path TTC kept fresh: the factor settles at clip((ttc_aged - 0.6)/0.9, 0, 1); samples arrive 0.08 s
    after capture and are aged to the tick (about 0.09 s on average at 18 Hz)."""
    gov = TtcClearanceGovernor(TTC, vertical=VG)
    rows = run(gov, lambda t: dict(ttc=5., below=.5, lower=lower), 2.)
    tail = series(rows, 'factor')[150:]
    assert tail.mean() == pytest.approx(expected, abs=.06)
    assert not any(r['arrest'] for r in rows) and max(r['climb'] for r in rows) == 0.


def test_sink_factor_ramps_and_releases_after_its_memory():
    gov = TtcClearanceGovernor(TTC, vertical=VG)
    rows = run(gov, lambda t: dict(ttc=.5, below=.5, lower=.4) if t < .3 else None, 4.)
    factor = series(rows, 'factor')
    steps = np.diff(factor)
    assert steps.min() >= -VG.factor_down_rate*.01-1e-12 and steps.max() <= VG.factor_up_rate*.01+1e-12
    assert factor[int(.08/.01)+int(1/(VG.factor_down_rate*.01))+2] == 0.       # down within 0.25 s of the sample
    last = 5*FRAME+.08+VG.memory_s                                               # the last sample's receipt + memory
    assert factor[int(last/.01)-2] == 0. and factor[int(last/.01)+5] > 0.       # kept memory_s, then released
    assert factor[int((last+1/VG.factor_up_rate)/.01)+3] == pytest.approx(1.)  # back to full at factor_up_rate


def test_descent_first_arrests_and_starts_no_climb_while_descending():
    """The minus-fast6-wall-01 shape: the floor looms below a sinking path (below_fraction 0.78, lower TTC 0.24 s,
    alarm 0.71 s). The policy climbs at 3.5 m/s at once; the guard arrests the descent and climbs nothing."""
    def floor(t):
        return dict(ttc=.71, below=.78, lower=.24)
    plain = run(TtcClearanceGovernor(TTC), floor, .6, rise=-.35)
    assert max(r['climb'] for r in plain) == pytest.approx(3.5)
    gov = TtcClearanceGovernor(TTC, vertical=VG)
    rows = run(gov, floor, .6, rise=-.35)
    assert max(r['climb'] for r in rows) == 0. and rows[-1]['arrest'] and rows[-1]['factor'] == 0.
    assert gov.vertical_counts['descent_first'] >= 2 and gov.vertical_counts['arrest_engagements'] == 1
    # the arrest holds arrest_hold_s after the last alarm received while descending
    rows = run(TtcClearanceGovernor(TTC, vertical=VG), lambda t: floor(t) if t < .2 else None, 1.2, rise=-.35)
    assert any(r['arrest'] for r in rows if r['t'] > .5) and not any(r['arrest'] for r in rows if r['t'] > .95)


def test_a_climb_needs_confirmation_and_is_gentle_until_rising_ground_is_confirmed():
    """Level flight over a below-path alarm (the gapon-02 arch below the path: below_fraction 1, lower TTC 0.3-0.7
    s): one sample climbs nothing, two confirm a gentle climb of at most 1 m/s, bounded to 1 m above its start;
    alarms that stop once the drone climbs (a thin structure below the path) never escalate it."""
    one = run(TtcClearanceGovernor(TTC, vertical=VG), lambda t: dict(ttc=.8, below=1., lower=.4) if t < FRAME else None,
              .8)
    assert max(r['climb'] for r in one) == 0.
    gov = TtcClearanceGovernor(TTC, vertical=VG)
    rows = run(gov, lambda t: dict(ttc=.8, below=1., lower=.4) if t < 2*FRAME else None, 2., follow=True)
    climb = series(rows, 'climb')
    assert climb.max() == pytest.approx(VG.gentle_climb) and not gov.escalated
    assert gov.vertical_counts['gentle_climbs'] == 1 and gov.vertical_counts['escalations'] == 0
    # alarms that keep arriving but never while the drone climbs faster than rising_min_rise (here: a guard that
    # would need 5 m/s): the gentle climb is bounded to gentle_max_m above the episode base, then released
    slow = VerticalGuardConfig(rising_min_rise=5.)
    gov = TtcClearanceGovernor(TTC, vertical=slow)
    rows = run(gov, lambda t: dict(ttc=.8, below=1., lower=.4), 4., follow=True)
    assert max(r['climb'] for r in rows) == pytest.approx(slow.gentle_climb) and not gov.escalated
    start = next(r['z'] for r in rows if r['climb'] > 0)
    top = max(r['z'] for r in rows)
    assert slow.gentle_max_m <= top-start <= slow.gentle_max_m+slow.gentle_climb**2/(2*TTC.climb_release)+.05
    assert rows[-1]['climb'] == 0. and gov.vertical_counts['topped'] > 0


def test_rising_ground_escalates_the_climb_graded_by_urgency():
    """A mound: below-path alarms keep arriving while the drone already climbs. The climb escalates beyond 1 m/s,
    graded by the below-path TTC (full 3.5 m/s at 0.6 s or less), up to the policy's climb_max_m."""
    def mound(t):
        return dict(ttc=.9, below=.95, lower=.3)
    gov = TtcClearanceGovernor(TTC, vertical=VG)
    rows = run(gov, mound, 3., follow=True)
    climb = series(rows, 'climb')
    first = next(i for i, c in enumerate(climb) if c > 0)
    assert climb[first] == pytest.approx(VG.gentle_climb)                       # gentle first
    assert climb.max() == pytest.approx(3.5) and gov.vertical_counts['escalations'] == 1
    up = next(i for i, c in enumerate(climb) if c > VG.gentle_climb)
    assert rows[up]['z']-rows[first]['z'] < VG.gentle_max_m
    assert rows[-1]['z']-rows[first]['z'] <= TTC.climb_max_m+TTC.climb_hold_s*3.5+3.5**2/(2*TTC.climb_release)+.05
    # graded: a longer below-path TTC asks for less (1.0 s -> 3.5 x 0.2/0.6)
    gov = TtcClearanceGovernor(TTC, vertical=VG)
    rows = run(gov, lambda t: dict(ttc=1.1, below=.95, lower=1.), 1.5, follow=True, rise=lambda t: .8)
    graded = series(rows, 'climb')
    aged = 1.-.08                                                                # received 0.08 s after capture
    assert gov.escalated and graded.max() == pytest.approx(3.5*(VG.climb_on_s-aged)/(VG.climb_on_s-VG.climb_full_s),
                                                           abs=.05)
    # without the guard the policy climbs on the first sample, at once and fast
    plain = run(TtcClearanceGovernor(TTC), mound, .1)
    assert series(plain, 'climb').max() > 3.


def test_alarms_while_descending_never_confirm_a_climb():
    gov = TtcClearanceGovernor(TTC, vertical=VG)
    rows = run(gov, lambda t: dict(ttc=.8, below=1., lower=.4), 1.5, rise=-.6)
    assert max(r['climb'] for r in rows) == 0. and gov.vertical_counts['gentle_climbs'] == 0
    assert gov.vertical_counts['unconfirmed_alarms'] == 0 and gov.vertical_counts['descent_first'] > 10


def test_the_ceiling_guard_still_cuts_a_guarded_climb():
    def mound_then_ceiling(t):
        if t < .4:
            return dict(ttc=.9, below=.95, lower=.3)
        return dict(ttc=.7, below=.1, lower=3.)
    gov = TtcClearanceGovernor(TTC, ceiling=GUARD, vertical=VG)
    rows = run(gov, mound_then_ceiling, 1.2, follow=True)
    assert max(r['climb'] for r in rows if r['t'] < .6) > 0 and gov.counts['overhead_engagements'] == 1
    assert all(r['climb'] == 0. for r in rows if r['t'] > .8)


def test_without_the_guard_the_governor_is_unchanged():
    """The vertical guard's state exists but nothing reads it: the same limits at every tick as the policy."""
    def mixed(t):
        if t < .5:
            return dict(ttc=.9, below=.9, lower=.4)
        return dict(ttc=1.1, lower=.6) if t < 1. else dict(ttc=.7, below=.1, lower=3.)
    for guard in (None, GUARD):
        a = run(TtcClearanceGovernor(TTC, ceiling=guard), mixed, 2., follow=True)
        b = run(TtcClearanceGovernor(TTC, guard), mixed, 2., follow=True)
        np.testing.assert_array_equal(series(a, 'climb'), series(b, 'climb'))
        assert all(r['factor'] == 1. and not r['arrest'] for r in a)


# ---------------------------------------------------------------------------------------------
# Pilot: applied and shadow
# ---------------------------------------------------------------------------------------------
def fly(sample_of, *, cue=BELOW, velocity=(6., 0., 0.), seconds=1.5, plant='static', **kw):
    """A pilot after a 2 s approach at `velocity`, then `seconds` with looming samples sample_of(k*0.01) every 5
    ticks (0.08 s late). plant 'static' keeps the measured velocity; 'perfect' flies the vertical request.
    Returns the pilot and rows (t, command, vertical_log, descent_scale, state)."""
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **kw)
    drive(pilot, history, cue, 200, velocity=(velocity[0], velocity[1], 0.), height=3.)
    now, rows, measured = 12., [], np.array(velocity, float)
    for k in range(int(round(seconds/.01))):
        t = now+k*.01
        clearance = None
        if k % 5 == 0:
            s = sample_of(k*.01)
            if s is not None:
                clearance = dict(time=t-.08, ttc=s.get('ttc'), distance=None if s.get('ttc') is None else s['ttc']*6.,
                                 below_fraction=s.get('below'), ttc_lower=s.get('lower'))
        state = senses(position=(0., 0., 3.), velocity=tuple(measured.tolist()))
        history.append(t, [0., 0., 3.], state['quat'][0].numpy())
        pilot.update(state, [0., 0., 0.], dict(race_cue=dict(cue)), t-.05, t, clearance=clearance)
        rows.append((t, pilot.velocity_command.copy(), pilot.vertical_log(), pilot.descent_scale, pilot.state))
        if plant == 'perfect':
            measured[2] = pilot.velocity_command[2]
    return pilot, rows


def test_the_pilot_keeps_a_time_margin_to_the_ground_below_a_descent():
    def ground(t):
        return dict(ttc=5., below=.5, lower=1.05+.09)                        # factor about 0.5
    _, plain = fly(ground, plant='perfect')
    guarded, rows = fly(ground, plant='perfect', vertical_guard=VG)
    pilot_sink = plain[-1][1][2]
    assert pilot_sink < -.6 and rows[-1][2]['vertical_pilot'] == pytest.approx(pilot_sink, abs=.05)
    assert rows[-1][1][2] == pytest.approx(.5*pilot_sink, abs=.1)              # half the sink kept
    assert all(r[1][2] <= 1e-9 for r in rows)                                  # never a climb
    vz = np.array([r[1][2] for r in rows])
    assert np.abs(np.diff(vz)).max() <= 5.*.01+1e-9                          # the pilot's own slew: no jump
    np.testing.assert_allclose([r[1][:2] for r in rows], [r[1][:2] for r in plain], atol=1e-9)   # same horizontal
    meta = json.loads(json.dumps(guarded.metadata()))['vertical_guard']
    assert meta['version'] == VERTICAL_GUARD_VERSION and meta['applied'] is True
    assert meta['parameters'] == asdict(VG) and meta['seconds']['limiting'] > 1.


def test_a_withheld_sink_does_not_cut_the_horizontal_speed():
    """Resting on a slope (the drone cannot sink) with ground close below: without the guard the descent-path
    governor cuts the horizontal request for the unachieved sink; with it the sink is withheld and the speed kept."""
    def slope(t):
        return dict(ttc=3., below=.6, lower=.5)
    _, plain = fly(slope, seconds=2.)
    _, rows = fly(slope, seconds=2., vertical_guard=VG)
    assert min(r[3] for r in plain) < .6
    assert min(np.linalg.norm(r[1][:2]) for r in plain) < .7*min(np.linalg.norm(r[1][:2]) for r in rows)
    assert min(r[3] for r in rows) == 1. and rows[-1][1][2] == pytest.approx(0., abs=1e-9)
    assert rows[-1][2]['vertical_factor'] == 0. and all(r[4] != 'support_climb' for r in rows)


def test_the_pilot_arrests_a_sink_instead_of_climbing_into_the_ceiling():
    """minus-fast6-wall-01: the pilot requests level, the drone sinks at 0.35 m/s and the floor looms below the
    path. Without the guard the governor requests 3.5 m/s; with it the request stays level (the pilot's own)."""
    def floor(t):
        return dict(ttc=.71, below=.78, lower=.24)
    _, plain = fly(floor, cue=AHEAD, velocity=(6., 0., -.35), seconds=1.)
    _, rows = fly(floor, cue=AHEAD, velocity=(6., 0., -.35), seconds=1., vertical_guard=VG)
    assert max(r[1][2] for r in plain) > 3.
    vz = np.array([r[1][2] for r in rows])
    assert vz.max() <= max(r[2]['vertical_pilot'] for r in rows)+1e-9 and vz.max() < .2
    assert vz[-1] >= -1e-9 and rows[-1][2]['vertical_arrest'] == 1. and rows[-1][2]['vertical_climb'] == 0.


def test_the_pilot_climbs_a_mound_once_rising_ground_is_confirmed():
    def mound(t):
        return dict(ttc=.9, below=.95, lower=.3)
    _, rows = fly(mound, cue=AHEAD, plant='perfect', seconds=1.5, vertical_guard=VG)
    vz = np.array([r[1][2] for r in rows])
    stages = np.array([r[2]['vertical_stage'] for r in rows])
    assert vz.max() > 3. and (stages == 1).any() and (stages == 2).any()
    gentle = vz[(stages == 1)]
    assert gentle.max() <= VG.gentle_climb+1e-9


def test_shadow_flies_the_unguarded_pilot_and_logs_the_guard():
    def floor(t):
        return dict(ttc=.71, below=.78, lower=.24) if t < .6 else dict(ttc=5., below=.5, lower=1.)
    _, plain = fly(floor, velocity=(6., 0., -.35), seconds=1.2)
    _, shadow = fly(floor, velocity=(6., 0., -.35), seconds=1.2, vertical_guard=VG, vertical_apply=False)
    np.testing.assert_array_equal(np.array([r[1] for r in shadow]), np.array([r[1] for r in plain]))
    np.testing.assert_array_equal([r[3] for r in shadow], [r[3] for r in plain])
    assert max(r[1][2] for r in plain) > 3.
    targets = np.array([r[2]['vertical_target'] for r in shadow])
    assert np.nanmax(targets) < .2 and any(r[2]['vertical_arrest'] == 1. for r in shadow)
    assert all(np.isnan(r[2]['vertical_target']) for r in plain)
    pilot, _ = fly(floor, velocity=(6., 0., -.35), seconds=.3, vertical_guard=VG, vertical_apply=False)
    meta = json.loads(json.dumps(pilot.metadata()))['vertical_guard']
    assert meta['applied'] is False and meta['governor'].startswith('shadow')
    assert json.loads(json.dumps(FastRaceCue(SENSOR, CameraPoseHistory(), 6.).metadata()))['vertical_guard'] is None


def test_shadow_with_the_wall_rules_still_flies_the_unguarded_pilot():
    def floor(t):
        return dict(ttc=.71, below=.78, lower=.24)
    _, plain = fly(floor, velocity=(6., 0., -.35), seconds=.8)
    pilot, shadow = fly(floor, velocity=(6., 0., -.35), seconds=.8, ceiling_guard=GUARD, wall_apply=False,
                        vertical_guard=VG, vertical_apply=False)
    np.testing.assert_array_equal(np.array([r[1] for r in shadow]), np.array([r[1] for r in plain]))
    assert pilot.clearance_shadow.ceiling == GUARD and pilot.clearance_shadow.vertical == VG
    assert pilot.clearance.ceiling is None and pilot.clearance.vertical is None


# ---------------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------------
def test_runner_columns_flags_and_refusals():
    from types import SimpleNamespace
    from haltere.liftoff.visual_brain import (VERTICAL_COLUMNS, VERTICAL_GUARD_DECLARATION, VisualController,
                                              obstacle_stack_metadata, resolve_obstacle_stack, vertical_row)
    assert len(vertical_row(None)) == len(VERTICAL_COLUMNS) == 6 and all(np.isnan(vertical_row(None)))
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., vertical_guard=VG)
    assert all(np.isnan(vertical_row(pilot)))                                  # before the first looming sample
    args = SimpleNamespace(obstacle_stack='on', pilot_profile='fast', looming_brake=True, gap_cue=None,
                           wall_pilot=None, vertical_guard=None, lag_turn=None)
    stack = resolve_obstacle_stack(args)
    assert stack['vertical_guard'] == str(VERTICAL_GUARD_DECLARATION) and stack['apply'] is True
    assert obstacle_stack_metadata(stack, None, None, None)['components']['vertical_guard'] is True
    args.vertical_guard = 'off'
    assert resolve_obstacle_stack(args)['vertical_guard'] is None
    args.obstacle_stack, args.vertical_guard = None, 'on'
    with pytest.raises(ValueError, match='vertical guard is part of the obstacle stack'):
        resolve_obstacle_stack(args)
    args.vertical_guard = None
    assert resolve_obstacle_stack(args)['vertical_guard'] is None
    with pytest.raises(ValueError, match='fast pilot'):
        VisualController('missing.pt', 'missing.json', 'cpu', pilot_assistance='race-cue', vertical_guard='x.json')
