"""Pilot-independent contact audit (haltere/liftoff/contact_audit.py, configs/contact_audit.json): the frozen
declaration, the detector on synthetic telemetry (free flight, resting on a surface, an arrested descent below hover
thrust, an off-axis impact) and the validation scoring on synthetic audit results. The validation over the logged
flights is run outside the test suite."""
import json

import numpy as np
import pytest

from haltere.liftoff import contact_audit as ca

G = ca.G
C = dict(ca.DEFAULTS)
HOVER = float((1/C['thrust_twr'])**(1/C['thrust_exponent']))


def frames(seconds, *, drive_of, vz_of=None, vx=3., z0=5., dt=.01, start=0.):
    """Level attitude, velocity (vx, 0, vz); drive_of(t) the game's drive; vz_of(t, vz, drive) the next vertical
    speed (default: free air with the declared curve and drag). Returns the frame dict audit_frames reads."""
    n = int(round(seconds/dt))
    t = start+np.arange(n)*dt
    V = np.zeros((n, 3))
    P = np.zeros((n, 3))
    V[:, 0] = vx
    P[0, 2] = z0
    drive = np.array([drive_of(x) for x in t])
    for k in range(1, n):
        if vz_of is None:
            V[k, 2] = V[k-1, 2]+(G*C['thrust_twr']*drive[k-1]**C['thrust_exponent']-G
                                 -C['body_drag_s_inv'][2]*V[k-1, 2])*dt
        else:
            V[k, 2] = vz_of(t[k], V[k-1, 2], drive[k-1])
        P[k] = P[k-1]+V[k-1]*dt
    return dict(phase=t.copy(), ts=t.copy(), P=P, V=V, Q=np.tile([1., 0., 0., 0.], (n, 1)), processed=2*drive-1,
                W=np.zeros((n, 3)))


def test_declaration_is_frozen_and_declares_the_module_values(tmp_path):
    config, digest = ca.load_config()
    assert config['version'] == ca.AUDIT_VERSION == 1 and digest == config['sha256']
    assert config['audit'] == ca.DEFAULTS
    v = config['validation']
    assert len(v['straw_downhill']['cases']) == 10 and v['straw_downhill']['min_detected'] == 9
    assert v['floor']['windows'][0] == [27.4, 28.9] and v['false_positives']['max_per_min'] == .05
    edited = dict(config, audit=dict(config['audit'], support_on=.5))
    path = tmp_path/'edited.json'
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match='frozen'):
        ca.load_config(path)


def test_free_flight_has_no_contact_and_a_resting_drone_has_one():
    # hover, then drive 0.5 in free air: the drone sinks as the thrust explains
    free = frames(8., drive_of=lambda t: HOVER if t < 5. else .5)
    result, _ = ca.audit_frames(free, C)
    assert result['contacts'] == []
    # the same thrust while a surface holds the drone at vz 0 from 5 s
    rest = frames(8., drive_of=lambda t: HOVER if t < 5. else .5, vz_of=lambda t, vz, d: 0.)
    result, _ = ca.audit_frames(rest, C)
    assert len(result['contacts']) == 1
    c = result['contacts'][0]
    assert 4.8 <= c['t_start'] <= 5.3 and c['kind'] in ('support', 'slide') and c['evidence']['support_s'] > .5
    assert c['impulse_vertical'] > .5 and c['speed'] == pytest.approx(3.)


def test_an_arrested_descent_below_hover_thrust_and_an_impact():
    # sinking at 1.5 m/s on a drive of 0.35 (outside the curve's range), arrested within 0.25 s at 6 s
    def vz(t, v, d):
        return -1.5 if t < 6. else min(0., v+6.*.01)
    arrest = frames(8., drive_of=lambda t: HOVER if t < 5. else .35, vz_of=vz)
    result, _ = ca.audit_frames(arrest, C)
    assert any(c['evidence']['arrest_s'] > 0 and 5.8 <= c['t_start'] <= 6.3 for c in result['contacts'])
    # a hover with a sudden horizontal stop at 6 s (a wall): off the thrust axis
    hit = frames(8., drive_of=lambda t: HOVER)
    hit['V'][hit['ts'] >= 6., 0] = 0.
    result, _ = ca.audit_frames(hit, C)
    assert [c['kind'] for c in result['contacts']] == ['impact']


def test_not_scored_before_arming_or_departure():
    rest = frames(3., drive_of=lambda t: .5, vz_of=lambda t, vz, d: 0.)
    result, _ = ca.audit_frames(rest, C)
    assert result['contacts'] == [] and result['scored_s'] == 0.


def test_validation_scoring(tmp_path):
    for log, reason in (('straw-a', 'Telemetry stale'), ('minus-b', 'Impact detected from flight motion')):
        (tmp_path/f'{log}.csv').write_text('phase\n'+'\n'.join(str(x) for x in np.arange(0, 30.01, .01)))
        (tmp_path/f'{log}.json').write_text(json.dumps(dict(stop_reason=reason)))
    contact = lambda a, b, **kw: dict(dict(t_start=a, t_end=b, kind='support', pos=[0, 0, 5], min_z=5.,  # noqa: E731
                                           peak_external=2., impulse_vertical=1.), **kw)
    results = [dict(log='straw-a', scored_s=1200., terminal_impact=None,
                    contacts=[contact(9., 10.2), contact(20., 21.5, kind='slide'), contact(25., 25.3)]),
               dict(log='minus-b', scored_s=600., terminal_impact=dict(t=30.),
                    contacts=[contact(12., 12.5, min_z=.1), contact(15., 15.2, min_z=1.), contact(29.5, 30.)])]
    v = dict(straw_downhill=dict(cases=[['straw-a', 10.5], ['straw-a', 17.]], window_s=[-2., .3], min_detected=1),
             slide=dict(log='straw-a', window=[20., 23.], min_coverage=.5, kind='slide'),
             floor=dict(log='minus-b', windows=[[12., 12.4]]), terminal_s=1., height_confirmed_prefixes=['minus-'],
             height_confirmed_m=.3, false_positives=dict(max_review=80, seed=17, max_per_min=.05))
    items, known = ca.review_items(results, v, tmp_path)
    assert [(i['log'], i['t_start']) for i in items] == [('straw-a', 25.), ('minus-b', 15.)]
    assert sorted(k['reason'] for k in known) == ['floor', 'slide', 'straw_downhill', 'terminal']
    labels = {'straw-a': {'25.00': 'contact'}, 'minus-b': {'15.00': 'ambiguous'}}
    out = ca.score_validation(results, v, labels, tmp_path)
    assert out['straw_downhill']['detected'] == 1 and out['straw_downhill']['passed']
    assert out['slide']['coverage'] == pytest.approx(.5) and out['slide']['passed']
    assert out['floor']['passed'] and out['terminal']['of'] == 1 and out['terminal']['csv_evidence'] == 1
    fp = out['false_positives']
    assert fp['false_positives'] == 1 and fp['per_minute'] == pytest.approx(1/30., abs=1e-4) and fp['passed']
    assert ca._coverage([contact(1., 2.), contact(1.5, 3.)], 0., 4.) == pytest.approx(.5)
