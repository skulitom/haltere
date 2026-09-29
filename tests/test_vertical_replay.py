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


def test_gates_v5_are_frozen_with_the_guard_and_keep_v4s_values():
    """Gates version 5 (scoring guard v5 against the m6 tree, guard v4): frozen, naming the guard's hash and version 4's
    kept gates; v4's V_R4, V_Minus, V_Straw_uphill and mound criteria keep their values; every flight is either
    development or held out; the idealised seeds are fresh."""
    gates, digest = vr.load_gates()
    assert gates['version'] == 5 and gates['frozen'] is True and digest == gates['sha256']
    from haltere.liftoff.visual_brain import VERTICAL_GUARD_DECLARATION, load_vertical_guard
    _, guard_digest = load_vertical_guard(VERTICAL_GUARD_DECLARATION)
    assert gates['vertical_guard']['sha256'] == guard_digest and gates['vertical_guard']['version'] == 5
    v4, v4_digest = vr.load_gates(vr.GATES_PATH.with_name('vertical_guard_gates_v4.json'))
    assert v4['version'] == 4 and v4_digest.startswith('7901b154abbe')
    assert [p['sha256'] for p in gates['previous_versions']][:2] == [v4_digest, v4['previous_versions'][0]['sha256']]
    g, g4 = gates['gates'], v4['gates']
    assert gates['baseline_tree']['guard_version'] == 4 and gates['baseline_tree']['commit'].startswith('c88bf73')
    assert {k: g['V_R4'][k] for k in ('flight', 'window', 'max_climb')} == {k: g4['V_R4'][k] for k in
                                                                            ('flight', 'window', 'max_climb')}
    assert g['V_Minus']['flights'] == g4['V_Minus']['flights'] and g['V_Minus']['no_escalation']['max_climb'] == 1.
    assert all(g['V_Minus'][k] == g4['V_Minus'][k] for k in ('before_onset_s', 'max_climb', 'level_band',
                                                              'max_height_loss_m'))
    dev_minus = set(g['V_Minus']['no_escalation']['development'])
    assert set(g4['V_Minus']['no_escalation']['flights']+g4['V_Minus']['no_escalation']['held_out_flights']
               + [g4['V_R4']['flight']]) == dev_minus
    assert g['V_Straw_uphill']['max_escalated_per_min'] == g4['V_Straw_uphill']['max_escalated_per_min'] == 0.
    assert set(g['V_Straw_uphill']['development']['plan']) == set(g4['V_Straw_uphill']['flights'])
    mound = g['V_Pine_hillsides']['episodes']['pine-fast6-ttc-01 mound']
    assert all(mound[k] == g4['V_Pine'][k] for k in ('mound_fraction', 'mound_escalated_by_s', 'min_vz'))
    assert g['V_Straw_downhill']['flights'] == g4['V_Straw_downhill']['flights']
    development, held_out = set(gates['flights']['development']), set(gates['flights']['held_out'])
    assert not development & held_out and development | held_out == set(g['Identity']['flights'])
    assert set(g4['Identity']['flights']) <= development and 'pine-fast6-r6-01' in development
    assert set(g['V_Minus']['no_escalation']['held_out']) | set(g['V_Straw_uphill']['held_out']['plan']) | set(
        g['V_Pine_heldout']['flights']) == held_out
    assert set(gates['inputs']['streams']['flights']) <= set(g['Identity']['flights'])
    assert {f[:-4] for f in gates['inputs']['log_sha256']} == set(g['Identity']['flights'])
    assert set(gates['inputs']['streams']['sha256']) == {f'stream_{f}.npz' for f in gates['inputs']['streams']['flights']}
    for name, digest4 in v4['inputs']['looming_stream']['sha256'].items():
        assert gates['inputs']['streams']['sha256'][name] == digest4          # gates v4's streams, unchanged
    ideal = g['Idealised']
    assert ideal['seeds'][0] >= 40 and ideal['baseline_version'] == 4 and ideal['baseline_sha256'].startswith('409d06f9')
    assert g['V_Pine_R6']['escalated_by_s'] == pytest.approx(g['V_Pine_R6']['impact_t']-.5)


def test_gates_v4_are_kept_verbatim():
    gates, digest = vr.load_gates(vr.GATES_PATH.with_name('vertical_guard_gates_v4.json'))
    assert gates['version'] == 4 and gates['frozen'] is True and digest == gates['sha256']
    assert gates['vertical_guard']['version'] == 4 and gates['vertical_guard']['sha256'].startswith('409d06f9')
    v3, v3_digest = vr.load_gates(vr.GATES_PATH.with_name('vertical_guard_gates_v3.json'))
    v2, v2_digest = vr.load_gates(vr.GATES_PATH.with_name('vertical_guard_gates_v2.json'))
    v1, v1_digest = vr.load_gates(vr.GATES_PATH.with_name('vertical_guard_gates_v1.json'))
    assert v3['version'] == 3 and v3['vertical_guard']['version'] == 3 and v3_digest.startswith('689635881467')
    assert v2['version'] == 2 and v2['vertical_guard']['version'] == 2 and v2_digest.startswith('53926ceada10')
    assert v1['version'] == 1 and v1['vertical_guard']['version'] == 1 and v1_digest.startswith('977740fbc0f5')
    assert [p['sha256'] for p in gates['previous_versions']] == [v3_digest, v2_digest, v1_digest] and gates['change']
    g, g3 = gates['gates'], v3['gates']
    # version 3's gates are kept with their values; the changed ones are also scored as v3 defined them
    assert g['V_Straw_downhill'] == g3['V_Straw_downhill']
    assert {k: g['V_Minus'][k] for k in g3['V_Minus'] if k not in ('pass', 'no_escalation')} == {
        k: v for k, v in g3['V_Minus'].items() if k not in ('pass', 'no_escalation')}
    assert {k: g['V_Minus']['no_escalation'][k] for k in g3['V_Minus']['no_escalation']} == g3['V_Minus'][
        'no_escalation']
    assert g['V_Minus']['no_escalation']['held_out_flights'] == ['minus-brain10b-r4-02', 'minus-brain09b-r4-01']
    assert {k: g['V_Pine'][k] for k in g3['V_Pine'] if k != 'pass'} == {k: v for k, v in g3['V_Pine'].items()
                                                                          if k != 'pass'}
    assert g['V_Pine']['mound_escalated_by_s'] == 4.65
    # tightened, never loosened: no escalated second on the Straw Bale uphills (v3 allowed 0.2 s/min)
    assert g['V_Straw_uphill']['max_escalated_per_min'] == 0. < g3['V_Straw_uphill']['max_escalated_per_min']
    assert {k: g['V_Straw_uphill'][k] for k in ('flights', 'max_new_climb')} == {
        k: g3['V_Straw_uphill'][k] for k in ('flights', 'max_new_climb')}
    for name in ('V_Straw_uphill', 'V_Pine', 'V_Minus'):
        assert gates['old_definitions'][name] == g3[name]
    assert g['V_R4']['flight'] == 'minus-fast6-r4-02' and g['V_R4']['development'] is True
    assert g['V_R4']['max_climb'] == 1. and g['V_R4']['window'] == [32.4, 33.4]
    assert g['Identity']['flights'] == g3['Identity']['flights']+['minus-fast6-r4-02', 'minus-brain10b-r4-02',
                                                                   'minus-brain09b-r4-01']
    assert gates['baseline_tree']['commit'].startswith('3decaac') and gates['baseline_tree']['guard_version'] == 3
    assert set(gates['inputs']['as_flown']) == {'minus-fast6-r4-02', 'minus-brain10b-r4-02', 'minus-brain09b-r4-01'}
    assert set(gates['inputs']['as_flown']) <= {f[:-4] for f in gates['inputs']['log_sha256']}
    assert set(gates['inputs']['stream_flights']) <= set(g['Identity']['flights'])
    assert set(gates['inputs']['looming_stream']['sha256']) == {f'stream_{f}.npz' for f in
                                                                gates['inputs']['stream_flights']}


def test_score_r4_mound_escalation_and_guard_report():
    """Gates v4: V_R4 (no escalated tick, guard climb and the window's issued request at most max_climb), the mound
    escalation (a stage-2 tick inside the first logged climb episode, by mound_escalated_by_s) and the keep report."""
    gate = dict(window=[1., 2.], max_climb=1.)
    climb = np.zeros(400)
    climb[100:300] = 1.
    stage = np.where(climb > 0, 1., 0.)
    cvz = np.where(climb > 0, .95, 0.)
    r = vr.score_r4(arrays(vertical_climb=climb, vertical_stage=stage, cvz=cvz, log_cvz=cvz), gate)
    assert r['passed'] and r['escalated_ticks'] == 0 and r['max_guard_climb'] == 1.
    stage[150] = 2
    assert not vr.score_r4(arrays(vertical_climb=climb, vertical_stage=stage, cvz=cvz, log_cvz=cvz), gate)['passed']
    stage[150], cvz[160] = 1, 1.2
    assert not vr.score_r4(arrays(vertical_climb=climb, vertical_stage=stage, cvz=cvz, log_cvz=cvz), gate)['passed']
    log_climb = np.zeros(400)
    log_climb[50:150] = 3.
    log_climb[250:300] = 3.
    stage = np.zeros(400)
    stage[70:120] = 2                              # escalated at 0.70 s inside the first logged episode (0.5-1.49 s)
    m = vr.mound_escalation(arrays(log_climb=log_climb, vertical_stage=stage), dict(mound_escalated_by_s=.75))
    assert m['passed'] and m['first_escalated_t'] == .7 and m['episode'] == [.5, 1.49]
    assert not vr.mound_escalation(arrays(log_climb=log_climb, vertical_stage=stage),
                                   dict(mound_escalated_by_s=.65))['passed']
    late = np.zeros(400)
    late[260:280] = 2                              # only in the second episode: the mound is not escalated
    assert not vr.mound_escalation(arrays(log_climb=log_climb, vertical_stage=late),
                                   dict(mound_escalated_by_s=4.))['passed']
    report = vr.guard_report(arrays(vertical_climb=np.where(stage > 0, 3.5, 0.), vertical_stage=stage),
                             arrays(vertical_climb=climb, vertical_stage=np.where(climb > 0, 1., 0.)))
    assert report['escalated_s'] == .5 and report['escalations'] == 1 and report['max_guard_climb'] == 3.5
    assert report['baseline'] == dict(climb_s=2., escalated_s=0., escalations=0, max_guard_climb=1.)


def test_gates_v5_scoring_functions():
    """Gates v5: an answered climb (a stage-2 tick by a time or inside a window, with an issued request), the held-out
    Pine criterion (the first escalation no later than the baseline's) and the changed-ticks report."""
    stage = np.zeros(400)
    stage[150:250] = 2                             # escalated from 1.5 s
    cvz = np.where(stage > 0, 2.9, .9)
    guard = arrays(vertical_stage=stage, cvz=cvz, vertical_climb=np.where(stage > 0, 2.9, 1.))
    r = vr.score_escalated_by(guard, 1.6, 2.5)
    assert r['passed'] and r['first_escalated_t'] == 1.5 and r['max_issued_vz_by'] == 2.9
    assert r['escalation_onsets'] == [1.5]
    assert not vr.score_escalated_by(guard, 1.4, 2.5)['passed']               # too late
    assert not vr.score_escalated_by(arrays(vertical_stage=stage, cvz=np.full(400, .9)), 1.6, 2.5)['passed']
    assert vr.score_escalated_by(guard, None, window=[1., 2.])['passed']
    assert not vr.score_escalated_by(guard, None, window=[2.6, 3.])['passed']
    later = np.zeros(400)
    later[200:250] = 2
    baseline = arrays(vertical_stage=later)
    r = vr.score_no_later(guard, baseline)
    assert r['passed'] and r['onsets'] == [1.5] and r['baseline_onsets'] == [2.] and r['new_onsets'] == [1.5]
    assert not vr.score_no_later(baseline, guard)['passed']
    assert vr.score_no_later(arrays(), arrays())['passed']                     # nobody escalates
    assert not vr.score_no_later(arrays(), guard)['passed']                    # v5 must escalate when v4 does
    c = vr.changed_ticks(guard, arrays(vertical_stage=stage, cvz=cvz))
    assert c['ticks'] == 0 and c['first_t'] is None
    other = arrays(vertical_stage=stage, cvz=np.where(np.arange(400) >= 300, 0., cvz))
    c = vr.changed_ticks(guard, other)
    assert c['ticks'] == 100 and c['first_t'] == 3.


def test_score_ideal_v5():
    gate = dict(identical_scenarios=['ramp', 'floor'], hill_scenarios=['hill'], tolerance_m=.02, improvement_m=.1,
                min_improved=1)

    def out(hill_median, ramp=(.3, .2)):
        def case(scenario, median, per_seed):
            return dict(scenario=scenario, metric='m', escalated_share=1., median=median, worst=min(per_seed),
                        per_seed=per_seed, per_seed_escalated=[True]*len(per_seed))
        return dict(seeds=[1000, 2], results={'ramp a': case('ramp', .25, list(ramp)),
                                              'hill a': case('hill', hill_median, [hill_median]*2),
                                              'floor_split a': case('floor_split', 2., [2., 2.])})
    r = vr.score_ideal_v5(out(.3), out(.1), gate)
    assert r['passed'] and r['hill_improved_cases'] == 1 and r['cases']['ramp a']['identical']
    assert not vr.score_ideal_v5(out(.3, ramp=(.3, .21)), out(.1), gate)['passed']     # not identical on a ramp
    assert not vr.score_ideal_v5(out(.07), out(.1), gate)['passed']                   # worse on the hill
    assert not vr.score_ideal_v5(out(.15), out(.1), gate)['passed']                   # not improved enough


def test_vertical_ideal_runs_the_declared_guards():
    """The idealised check (vertical_ideal.py) builds each guard from its declaration file: version 4 and version 5
    differ only through the clear-below switch, and on a ramp case (every sample below the path) they are identical."""
    from pathlib import Path
    from haltere.liftoff import fast_race_cue as frc
    from haltere.obstacles import vertical_ideal as vi
    ob = Path(vr.__file__).resolve().parents[2]/'configs'/'obstacles'
    guards = {v: frc.VerticalGuardConfig(**json.loads((ob/name).read_text(encoding='utf-8'))['vertical_guard'])
              for v, name in ((4, 'vertical_guard_v4.json'), (5, 'vertical_guard.json'))}
    assert not guards[4].clear_below_terrain and guards[5].clear_below_terrain
    ramp = [vi.fly(frc, guards[v], 'ramp', np.random.default_rng(7), slope=.35) for v in (4, 5)]
    assert ramp[0] == ramp[1]
    names = [name for name, _, _ in vi.cases()]
    assert len(names) == 16 and sum(n.startswith('hill') for n in names) == 6


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


def _side(stack_mode=None, contract='fast_velocity_pd_v1'):
    """A minimal flight sidecar for build(): the camera, motor contract and pilot speeds of a fast-stack flight."""
    from tests.test_visual_assistance import SENSOR
    return dict(gate_sensor=SENSOR, motor_controller=dict(contract=contract),
                obstacle_stack=dict(mode=stack_mode) if stack_mode else {},
                pilot_assistance=dict(nominal_speed_mps=6., trained_motor_reference_mps=6.))


def test_build_adds_the_descent_view_to_any_variant():
    from pathlib import Path
    from haltere.liftoff.fast_race_cue import DescentViewConfig, descent_view_config
    tree = Path(vr.__file__).resolve().parents[2]
    declaration = json.loads((tree/'configs'/'pilot'/'descent_view.json').read_text(encoding='utf-8'))
    config = descent_view_config(declaration)
    plain, info = vr.build(_side(), tree, 'none')
    assert plain.descent_view is None and info['descent_view'] is False
    pilot, info = vr.build(_side(), tree, 'none', descent_view=config)
    assert isinstance(pilot.descent_view, DescentViewConfig) and info['descent_view'] is True
    # the full round-4 stack: gap aim, wall rules, vertical guard and the view-keeping descent together
    stack, info = vr.build(_side('on', 'fast_velocity_brain_v1'), tree, 'on', vertical='on', descent_view=config)
    assert stack.descent_view == config and stack.vertical_guard is not None and stack.turn_first is not None
    assert stack.gap_aim is not None and info['wall'] == 'on' and info['vertical'] == 'on'


def test_main_refuses_an_edited_descent_view_declaration(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    monkeypatch.setattr(sys, 'path', list(sys.path))
    tree = Path(vr.__file__).resolve().parents[2]
    declaration = json.loads((tree/'configs'/'pilot'/'descent_view.json').read_text(encoding='utf-8'))
    declaration['descent_view']['margin_deg'] = 1.
    edited = tmp_path/'descent_view.json'
    edited.write_text(json.dumps(declaration), encoding='utf-8')
    with pytest.raises(SystemExit, match='not a frozen descent-view declaration'):
        vr.main(['no-such-flight', '--out', str(tmp_path/'x'), '--stack', 'none', '--descent-view', str(edited)])
