"""The vertical-guard replay harness's scoring (haltere/obstacles/vertical_replay.py) on synthetic replay arrays, and
the frozen gates declaration. The replays of the live logs themselves are run outside the test suite."""
import json

import numpy as np
import pytest

from haltere.obstacles import vertical_replay as vr


def arrays(n=400, dt=.01, **columns):
    t = np.arange(n)*dt
    base = dict(t=t, now=t+100., x=np.zeros(n), y=np.zeros(n), z=np.ones(n), vx=np.full(n, 6.), vy=np.zeros(n),
                vz=np.zeros(n), cvx=np.full(n, 6.), cvy=np.zeros(n), cvz=np.zeros(n), yaw_cmd=np.zeros(n),
                state=np.array(['cue']*n), log_cvx=np.full(n, 6.), log_cvy=np.zeros(n), log_cvz=np.zeros(n),
                log_state=np.array(['cue']*n), log_climb=np.zeros(n), cap=np.full(n, np.nan), climb=np.zeros(n),
                descent_scale=np.ones(n), support_since=np.full(n, np.nan), slope_support_since=np.full(n, np.nan),
                new_sample=np.zeros(n, bool), ttc=np.full(n, np.nan), below=np.full(n, np.nan),
                lower=np.full(n, np.nan), vertical_pilot=np.zeros(n), vertical_target=np.zeros(n),
                vertical_factor=np.ones(n), vertical_arrest=np.zeros(n), vertical_stage=np.zeros(n),
                vertical_climb=np.zeros(n))
    for k, v in columns.items():
        base[k] = v
    return base


def test_gates_declaration_is_frozen_and_names_the_guard_version():
    gates, digest = vr.load_gates()
    assert gates['version'] == 3 and gates['frozen'] is True and digest == gates['sha256']
    from haltere.liftoff.visual_brain import VERTICAL_GUARD_DECLARATION, load_vertical_guard
    _, guard_digest = load_vertical_guard(VERTICAL_GUARD_DECLARATION)
    assert gates['vertical_guard']['sha256'] == guard_digest and gates['vertical_guard']['version'] == 3
    v2, v2_digest = vr.load_gates(vr.GATES_PATH.with_name('vertical_guard_gates_v2.json'))
    v1, v1_digest = vr.load_gates(vr.GATES_PATH.with_name('vertical_guard_gates_v1.json'))
    assert v2['version'] == 2 and v2['vertical_guard']['version'] == 2 and v2_digest.startswith('53926ceada10')
    assert v1['version'] == 1 and v1['vertical_guard']['version'] == 1 and v1_digest.startswith('977740fbc0f5')
    assert [p['sha256'] for p in gates['previous_versions']] == [v2_digest, v1_digest] and gates['change']
    g = gates['gates']
    # the version-2 definitions are kept and still scored (old_definitions); none is changed silently
    assert gates['old_definitions']['V_Straw'] == v2['gates']['V_Straw']
    assert gates['old_definitions']['V_Pine'] == v2['gates']['V_Pine']
    kept = {k: v for k, v in v2['gates']['V_Straw'].items() if k not in ('whole_lap', 'pass')}
    assert {k: g['V_Straw_downhill'][k] for k in kept} == kept
    assert {k: g['V_Minus'][k] for k in v2['gates']['V_Minus'] if k != 'pass'} == {
        k: v for k, v in v2['gates']['V_Minus'].items() if k != 'pass'}
    assert g['V_Minus']['no_escalation']['max_climb'] == 1. and len(g['V_Minus']['no_escalation']['flights']) == 9
    assert (g['V_Pine']['min_vz'], g['V_Pine']['climb_fraction'], g['V_Pine']['last_s']) == (1., .8, 2.)
    assert g['V_Pine']['mound_fraction'] == .9 and g['V_Straw_uphill']['max_escalated_per_min'] == .2
    assert len(g['V_Straw_uphill']['flights']) == 9 and len(g['Identity']['flights']) == 21
    assert set(gates['inputs']['stream_flights']) <= set(g['Identity']['flights'])
    assert set(gates['inputs']['looming_stream']['sha256']) == {f'stream_{f}.npz' for f in
                                                                gates['inputs']['stream_flights']}


def test_identity_compares_bitwise_with_nan():
    a = arrays(cap=np.r_[np.nan, np.ones(399)])
    b = arrays(cap=np.r_[np.nan, np.ones(399)])
    assert vr.identical(a, b)[0]
    b['cvz'] = b['cvz']+1e-15
    same, keys = vr.identical(a, b)
    assert not same and keys['cvz'] is False and keys['cap'] is True


def test_support_contacts_read_the_timer_before_the_climb():
    state = np.array(['cue']*400, dtype='<U16')
    state[200:260] = 'support_climb'
    slope = np.full(400, np.nan)
    slope[130:199] = 101.3                        # the timer runs, is reset on the tick that starts the climb
    flown = arrays(state=state, slope_support_since=slope)
    (onset_t, onset_now, contact), = vr.support_contacts(flown)
    assert onset_t == pytest.approx(2.) and onset_now == pytest.approx(102.) and contact == 101.3


def test_score_minus_window_climb_and_floor_sink():
    gate = dict(before_onset_s=1., max_climb=1., level_band=.3, max_height_loss_m=.3)
    n = 400
    vz = np.zeros(n)
    vz[150:230] = -.4                              # a sink under way at the logged climb onset (2.0 s)
    z = 1.+np.cumsum(vz)*.01
    log_climb = np.zeros(n)
    log_climb[200:] = 3.5
    cvz = np.zeros(n)
    cvz[150:175] = -.2                             # the request reaches level 0.25 s into the sink (0.1 m lost)
    cvz[300:] = .9
    side = dict(stop_reason='Impact detected from flight motion')
    r = vr.score_minus(arrays(vz=vz, z=z, cvz=cvz, log_climb=log_climb), side, gate, floor_sink=True)
    assert r['window'] == [1., 3.99] and r['max_requested_vz'] == .9 and r['passed']
    assert r['floor_sink']['height_loss_at_level_request'] == pytest.approx(.1, abs=.01)
    cvz[150:230] = -.2                             # never level during the sink: fails
    assert not vr.score_minus(arrays(vz=vz, z=z, cvz=cvz, log_climb=log_climb), side, gate, True)['passed']
    cvz[150:230] = 0.
    cvz[320] = 1.2                                 # one tick above 1 m/s in the window fails
    assert not vr.score_minus(arrays(vz=vz, z=z, cvz=cvz, log_climb=log_climb), side, gate, True)['passed']
    assert not vr.score_minus(arrays(), dict(stop_reason='done'), gate)['scored']


def test_score_pine_fraction_and_late_descent():
    gate = dict(min_vz=1., climb_fraction=.8, last_s=2.)
    log_climb = np.zeros(400)
    log_climb[50:150] = 3.
    cvz = np.zeros(400)
    cvz[60:150] = 1.                               # 90 of the 100 logged climb ticks
    side = dict(stop_reason='Impact detected from flight motion')
    r = vr.score_pine(arrays(cvz=cvz, log_climb=log_climb), side, gate)
    assert r['answered_fraction'] == pytest.approx(.9) and r['passed']
    cvz[350] = -.01                                # a requested descent in the last 2 s
    assert not vr.score_pine(arrays(cvz=cvz, log_climb=log_climb), side, gate)['passed']
    cvz[350] = 0.
    cvz[60:80] = .99
    assert not vr.score_pine(arrays(cvz=cvz, log_climb=log_climb), side, gate)['passed']


def test_score_straw_limited_raised_and_horizontal():
    gate = dict(level_band=.3, lookback_s=3., after_onset_s=.6, limited_fraction=.8, max_new_climb=1.)
    n = 600
    state = np.array(['cue']*n, dtype='<U16')
    state[400:460] = 'support_climb'
    slope = np.full(n, np.nan)
    slope[340:399] = 103.4                         # contact at 3.4 s (now 103.4), onset at 4.0 s
    flown = arrays(n, state=state, slope_support_since=slope)
    pilot = np.full(n, -1.)
    target = pilot.copy()
    target[300:340] = -.5                          # limited 0.4 s before contact
    vz = np.full(n, -.8)
    guard = arrays(n, vertical_pilot=pilot, vertical_target=target, vz=vz)
    control = arrays(n)
    r = vr.score_straw(guard, control, flown, gate)
    assert r['n_episodes'] == 1 and r['limited_fraction'] == 1. and r['passed']
    assert r['episodes'][0]['first_limit_t'] == 3. and r['episodes'][0]['contact_t'] == pytest.approx(3.4)
    target[350] = .2                               # a climb above the pilot's while descending
    assert vr.score_straw(arrays(n, vertical_pilot=pilot, vertical_target=target, vz=vz), control, flown,
                          gate)['ticks_raised_above_pilot'] == 1
    target[350] = -.5
    slower = arrays(n, vertical_pilot=pilot, vertical_target=target, vz=vz, cvx=np.full(n, 5.9))
    r = vr.score_straw(slower, control, flown, gate)
    assert not r['passed'] and r['ticks_horizontal_reduced'] > 0
    late = target.copy()
    late[300:340] = -1.                            # limited only after contact
    late[345:360] = -.5
    assert vr.score_straw(arrays(n, vertical_pilot=pilot, vertical_target=late, vz=vz), control, flown,
                          gate)['limited_fraction'] == 0.
    climbing = arrays(n, vertical_pilot=pilot, vertical_target=target, vz=vz, vertical_climb=np.r_[np.zeros(50), 1.5,
                                                                                                   np.zeros(n-51)])
    assert vr.score_straw(climbing, control, flown, gate)['whole_lap']['ticks_guard_climb_above_limit'] == 1


def test_score_pine_v3_counts_only_strong_logged_climb_and_the_mound():
    """Gates v3: the answered fraction is taken over the ticks in which the logged governor itself requested >= 1 m/s
    (its release tail and weak climbs excluded); the mound is the first logged climb episode's height request."""
    gate = dict(min_vz=1., climb_fraction=.8, last_s=2., mound_fraction=.9)
    log_climb = np.zeros(400)
    log_climb[50:150] = 3.
    log_climb[150:180] = .5                        # the release tail: v2 counted it, v3 does not
    log_cvz = np.zeros(400)
    log_cvz[55:150] = 3.
    cvz = np.zeros(400)
    cvz[60:150] = 3.                               # 90 of the 100 strong ticks; nothing during the tail
    side = dict(stop_reason='Impact detected from flight motion')
    guard = arrays(cvz=cvz, log_climb=log_climb, log_cvz=log_cvz)
    r = vr.score_pine_v3(guard, side, gate)
    assert r['answered_fraction'] == pytest.approx(.9) and r['climb_passed'] and r['no_descent_passed']
    assert r['mound_height_fraction'] == pytest.approx(90/95, abs=1e-4) and r['mound_passed'] and r['passed']
    assert vr.score_pine(guard, side, gate)['answered_fraction'] == pytest.approx(90/130, abs=1e-4)    # v2's
    cvz[60:80] = 2.                                # still answered (>= 1 m/s) but a smaller height request
    cvz[80:100] = .5
    r = vr.score_pine_v3(arrays(cvz=cvz, log_climb=log_climb, log_cvz=log_cvz), side, gate)
    assert not r['climb_passed'] and not r['mound_passed'] and not r['passed']


def test_score_straw_uphill_pools_escalated_seconds_per_minute():
    gate = dict(max_escalated_per_min=.2, max_new_climb=1.)
    n = 6000                                       # a minute at 100 Hz
    stage = np.zeros(n)
    stage[1000:1015] = 2                           # 0.15 s escalated
    climb = np.zeros(n)
    climb[990:1030] = 1.
    climb[1000:1015] = 2.
    lap = arrays(n, vertical_stage=stage, vertical_climb=climb)
    r = vr.score_straw_uphill(dict(a=lap, b=arrays(n)), gate)
    assert r['escalated_s'] == pytest.approx(.15) and r['minutes'] == pytest.approx(2., abs=.01) and r['passed']
    assert r['laps']['a']['escalations'] == 1 and r['laps']['a']['ticks_guard_climb_above_limit'] == 15
    assert r['laps']['a']['climb_onsets'] == 1 and r['laps']['b']['escalated_s'] == 0.
    stage[1000:1500] = 2
    assert not vr.score_straw_uphill(dict(a=arrays(n, vertical_stage=stage, vertical_climb=climb)), gate)['passed']


def test_score_minus_escalation():
    gate = dict(max_climb=1.)
    climb = np.zeros(400)
    climb[100:150] = 1.
    assert vr.score_minus_escalation(arrays(vertical_climb=climb), gate)['passed']
    climb[120] = 1.2
    r = vr.score_minus_escalation(arrays(vertical_climb=climb), gate)
    assert not r['passed'] and r['max_guard_climb'] == 1.2


def test_variant_resolution_and_tags():
    assert vr.resolve_vertical('none', 'on') == 'off'
    assert vr.resolve_vertical('flown', None) == 'off' and vr.resolve_vertical('flown', 'shadow') == 'shadow'
    assert vr.resolve_vertical('on', None) == 'on' and vr.resolve_vertical('on', 'off') == 'off'
    assert vr.resolve_vertical('shadow', None) == 'shadow' and vr.resolve_vertical('on', None, False) == 'off'
    assert vr.variant_tag('on', 'off', 'on', True) == 'on-won-von-stream'
    assert vr.variant_tag('flown', 'shadow', 'off', False) == 'flown-wshadow-voff'
    assert json.dumps(vr.variant_tag('none', 'on', 'off', False)) == '"none-woff-voff"'
