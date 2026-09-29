"""FastRaceCue sighted descent (`SightedDescentConfig`), declared in configs/pilot/sighted_descent.json (off by
default).

These tests pin the rule on synthetic states: the ring's line of sight (two agreeing near-edge in-view sightings;
raised by later bottom clips; cleared by the reset conditions), the limit it puts on the view rule's steep late (never
below the in-view bound, never above the view rule's own bound: it only withholds sink), shadow mode (logged, commands
unchanged), the declaration, runner, deployed-pilot and replay wiring, and that the pilot with the rule off is
bit-identical to m5 (golden digests from a git archive of m5, bc7c71c). None of this is flight evidence.
"""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import (DescentViewConfig, FastRaceCue, SIGHTED_DESCENT_MODES,
                                           SIGHTED_DESCENT_VERSION, SightedDescentConfig, sighted_descent_config)
from haltere.vision.camera import Camera, quat_wxyz_to_mat
from tests.test_fast_race_cue import single_thread  # noqa: F401 (fixture)
from tests.test_fast_race_cue_descent_view import golden_scenario, pitched
from tests.test_visual_assistance import SENSOR

DV = DescentViewConfig()
SD = SightedDescentConfig()
CAMERA = Camera(320, 180, SENSOR['focal_320'], SENSOR['tilt_deg'])
CLIP = dict(u=.5, v=.97, edge=True)          # the marker clamped at the bottom edge, centred


def ring_cue(depression_deg, pitch_deg, azimuth_deg=0.):
    """The in-view HUD cue of a ring depression_deg below the horizon (world), azimuth_deg left, at a level-yaw attitude
    pitch_deg above the horizon."""
    d, a = np.radians(depression_deg), np.radians(azimuth_deg)
    world = np.array([np.cos(d)*np.cos(a), np.cos(d)*np.sin(a), -np.sin(d)])
    body = quat_wxyz_to_mat(pitched(pitch_deg)).T @ world
    px, ok = CAMERA.project_body(body[None])
    assert ok[0]
    u, v = px[0, 0]/320, px[0, 1]/180
    assert 0 <= u <= 1 and 0 <= v <= 1, (u, v)
    return dict(u=float(u), v=float(v), edge=False)


def fly(phases, *, velocity=(6., 0., -6.*np.tan(np.radians(15.))), **kw):
    """The pilot at a fixed measured velocity (default: a 15 degree descent at 6 m/s) through phases [(seconds,
    pitch_deg, cue or None or callable(t))]; a cue every tick captured 0.05 s earlier. Returns (pilot, rows): rows of
    (t, command, state, view bound, sighted log)."""
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **kw)
    t, rows = 10., []
    velocity = np.asarray(velocity, float)
    for seconds, pitch, cue in phases:
        q = pitched(pitch)
        for _ in range(int(round(seconds/.01))):
            f = lambda x: torch.tensor([x], dtype=torch.float32)  # noqa: E731
            s = dict(pos=f([0., 0., 20.]), quat=f(q.tolist()), vel_world=f(velocity.tolist()),
                     vel_body=f((quat_wxyz_to_mat(q).T @ velocity).tolist()), gyro=f([0., 0., 0.]))
            history.append(t, [0., 0., 20.], q)
            current = cue(t) if callable(cue) else cue
            pilot.update(s, [0., 0., 0.], dict(race_cue=dict(current)) if current else None, t-.05, t)
            rows.append((t, pilot.velocity_command.copy(), pilot.state, pilot.view_bound,
                         pilot.sighted_log() if pilot.sighted_descent is not None else None))
            t += .01
    return pilot, rows


# the attitude changes with no cue for 0.06 s, so no capture straddles it (captures are 0.05 s old)
SIGHTED_THEN_CLIPPED = [(.5, -6., ring_cue(14., -6.)), (.06, 0., None), (3.44, 0., CLIP)]


# ---------------------------------------------------------------------------------------------
# Configuration and declaration
# ---------------------------------------------------------------------------------------------
def test_config_validation():
    for bad in (dict(margin_deg=-1.), dict(margin_deg=30.), dict(edge_v=0.), dict(edge_v=1.), dict(agree_s=0.),
                dict(agree_s=3.), dict(switch_u=1.), dict(agree_deg=float('nan')), dict(version=2)):
        with pytest.raises(ValueError):
            SightedDescentConfig(**bad)
    with pytest.raises(ValueError, match='SightedDescentConfig'):
        FastRaceCue(SENSOR, CameraPoseHistory(), 6., descent_view=DV, sighted_descent=dict(margin_deg=1.))
    with pytest.raises(ValueError, match='declare both'):
        FastRaceCue(SENSOR, CameraPoseHistory(), 6., sighted_descent=SD)
    with pytest.raises(ValueError, match='version'):
        sighted_descent_config(dict(version=SIGHTED_DESCENT_VERSION+1, sighted_descent={}))
    assert SIGHTED_DESCENT_MODES == ('on', 'off', 'shadow')


def test_repository_declaration_and_gates_are_frozen():
    from haltere.liftoff.sighted_descent_gates import GATES_PATH, load_gates
    from haltere.liftoff.visual_brain import SIGHTED_DESCENT_DECLARATION, load_sighted_descent
    declaration, digest = load_sighted_descent(SIGHTED_DESCENT_DECLARATION)
    assert declaration['version'] == SIGHTED_DESCENT_VERSION == 1 and declaration['frozen'] is True
    assert digest == '49b8d7a79f32d5c5d4b63bdd815ba4092f9de35ce8c853a9580cb59f401424e6'
    assert sighted_descent_config(declaration) == SD                 # the declared values are the defaults
    assert SD.growth_range_m == 20. and SD.margin_deg == 1.
    gates, gates_digest = load_gates(GATES_PATH)
    assert gates['declaration']['sha256'] == digest and gates['version'] == 1
    assert gates_digest == '46deedeeea366a5204c19f053b3c71650867d71e3d73b5b9d7c9c525bbf0a3b1'
    # the development logs are the two brain-11 Straw laps; every other replayed log is held out
    assert set(gates['replays']['development']) == {'straw-brain11cw13-r5-noassist-04',
                                                    'straw-brain11cw13-r4b-noassist-02'}
    assert len(gates['replays']['flights']) == 31 and gates['surrogate']['sim_seed'] == 97


def test_loader_refuses_unfrozen_edited_and_other_versions(tmp_path):
    from haltere.liftoff.visual_brain import lag_turn_declaration_sha256, load_sighted_descent
    body = dict(schema='haltere.liftoff.sighted_descent.v1', version=1,
                sighted_descent=dict(margin_deg=1., edge_v=.85, agree_deg=1.5, agree_s=.5, switch_u=.1))
    path = tmp_path/'sd.json'
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='frozen'):
        load_sighted_descent(path)
    frozen = dict(body, frozen=True, sha256=lag_turn_declaration_sha256(body))
    path.write_text(json.dumps(frozen))
    declaration, digest = load_sighted_descent(path)
    assert digest == frozen['sha256'] and sighted_descent_config(declaration) == SD
    path.write_text(json.dumps(dict(frozen, sighted_descent=dict(body['sighted_descent'], margin_deg=0.))))
    with pytest.raises(ValueError, match='changed after the freeze'):
        load_sighted_descent(path)
    other = dict(body, version=2)
    path.write_text(json.dumps(dict(other, frozen=True, sha256=lag_turn_declaration_sha256(other))))
    with pytest.raises(ValueError, match='version'):
        load_sighted_descent(path)


# ---------------------------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------------------------
def test_off_by_default_and_the_m5_pilots_are_bit_identical():
    digest = lambda trace: hashlib.sha256(np.ascontiguousarray(trace.astype(np.float64)).tobytes()).hexdigest()  # noqa
    from tests.test_round5_safety import v3
    V3 = v3()
    # from a git archive of m5 (bc7c71c): the default pilot, the view rule, and the view rule with contact support v3
    assert digest(golden_scenario()[0]) == '67d31f2892c3f292871a40e6cb78e8a734ae540636aa8a3508a1ce3572ad3d1f'
    assert digest(golden_scenario(descent_view=DV)[0]) == \
        'eae7b39aeac9c76db695012782f1f6903d1ccf9ddbbe033c032dc9554dc7914a'
    m5 = 'efea6e68329d74a6d453a89e27b86226a49ba1d8c2766b8f5f2f846fdc55f110'
    assert digest(golden_scenario(descent_view=DV, contact_support=V3)[0]) == m5
    # shadow computes and logs the rule, and changes nothing
    shadow, pilot = golden_scenario(descent_view=DV, contact_support=V3, sighted_descent=SD, sighted_apply=False)
    assert digest(shadow) == m5 and pilot.sighted_summary()['seconds']['set'] > 0
    assert pilot.metadata()['sighted_descent']['applied'] is False
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., descent_view=DV)
    assert 'sighted_descent' not in pilot.metadata() and pilot.sighted_summary() is None
    assert all(np.isnan(v) for v in pilot.sighted_log().values())


def test_the_sighted_line_of_sight_caps_steep_late():
    """The Straw downhill in miniature: the ring seen at the bottom of the image 14 degrees below, then clipped with a
    level nose (the clamped ray 11.4 degrees down). Steep late takes the view bound far below 15 degrees; the rule holds
    the path 1 degree below the sighted line of sight, and keeps the horizontal request."""
    off, rows_off = fly(SIGHTED_THEN_CLIPPED, descent_view=DV)
    on, rows_on = fly(SIGHTED_THEN_CLIPPED, descent_view=DV, sighted_descent=SD)
    assert on.sighted_los == pytest.approx(14., abs=.05)
    cap = 6.*np.tan(np.radians(15.))
    late = [(a, b) for a, b in zip(rows_off, rows_on) if a[0] > 10.5+DV.late_after_s+1.]
    assert late and all(a[2] == b[2] == 'below' for a, b in late)
    assert max(a[3] for a, _ in late) > cap+.5                    # m5 steep late: 20 degrees, far below the ring
    assert all(b[3] <= cap+1e-6 for _, b in late)                  # the rule: 1 degree below the line of sight
    assert all(b[3] >= a[3]-1e-9 or b[3] == pytest.approx(cap) for a, b in late)
    # it only withholds sink and never lowers the horizontal request (keep speed)
    for (_, ca, *_), (_, cb, *_) in zip(rows_off, rows_on):
        assert cb[2] >= ca[2]-1e-9
        assert np.linalg.norm(cb[:2]) >= np.linalg.norm(ca[:2])-1e-9
    final = rows_on[-1][1]
    assert np.degrees(np.arctan2(-final[2], np.linalg.norm(final[:2]))) == pytest.approx(15., abs=.3)
    assert rows_on[-1][4]['sighted_withheld'] > 0 and on.sighted_summary()['seconds']['limiting'] > .8
    # before steep late (the first late_after_s of the clip) nothing differs
    early = [(a, b) for a, b in zip(rows_off, rows_on) if a[0] < 10.5+DV.late_after_s]
    assert all(np.array_equal(a[1], b[1]) for a, b in early)


def test_never_below_the_in_view_bound():
    """A sighting shallower than the view (a stale or low reading) never pulls the bound below the in-view bound."""
    low = SightedDescentConfig(margin_deg=0.)
    on, _ = fly(SIGHTED_THEN_CLIPPED, descent_view=DV, sighted_descent=low)
    on.sighted_los = 2.                   # as if sighted 2 degrees down (the path is steeper: no growth)
    speed, lowest = on._view_lowest(np.array([6., 0., -1.]), quat_wxyz_to_mat(pitched(0.)), DV.margin_deg, 0.)
    bound = on._sighted_limit(5., 5., 'below', DV.margin_deg, np.array([6., 0., -1.]),
                              quat_wxyz_to_mat(pitched(0.)), 0., .01)
    assert on.sighted_los == pytest.approx(2.)
    assert bound == pytest.approx(max(DV.free_sink, -speed*lowest)) and bound > 6.*np.tan(np.radians(3.))


def test_bottom_clips_raise_the_line_of_sight():
    """A clamped marker whose ray lies deeper than the sighting (a nose-down wobble with the ring still clipped) proves
    the ring lies at least that low."""
    pilot, _ = fly([(.5, -6., ring_cue(10., -6.)), (.06, -3., None), (.44, -3., CLIP)], descent_view=DV,
                   sighted_descent=SD)
    ray = quat_wxyz_to_mat(pitched(-3.)) @ CAMERA.unproject_body(np.array([[160., .97*180]]))[0]
    clamped = -np.degrees(np.arcsin(ray[2]))           # the clamped marker's ray: 13.2 degrees down
    assert 13. < clamped < 13.4 and pilot.sighted_counts['raised'] >= 1
    assert pilot.sighted_los == pytest.approx(clamped, abs=1e-6) == pytest.approx(pilot.edge_depression, abs=1e-6)


def test_what_sets_and_clears_the_line_of_sight():
    kw = dict(descent_view=DV, sighted_descent=SD)
    # one reading is not a sighting; two agreeing readings are
    pilot, _ = fly([(.01, -6., ring_cue(14., -6.))], **kw)
    assert pilot.sighted_los is None
    pilot, _ = fly([(.02, -6., ring_cue(14., -6.))], **kw)
    assert pilot.sighted_los == pytest.approx(14., abs=.05)
    # readings that disagree by more than agree_deg (a false marker, another ring) set nothing
    flip = lambda t: ring_cue(14. if round(t*100) % 2 else 17., -6.)  # noqa: E731
    pilot, _ = fly([(.3, -6., flip)], **kw)
    assert pilot.sighted_los is None
    # a ring well inside the view (v < edge_v) is the view rule's alone
    well_inside = ring_cue(2., -6.)
    assert well_inside['v'] < SD.edge_v
    pilot, _ = fly([(.5, -6., ring_cue(14., -6.)), (.1, -6., well_inside)], **kw)
    assert pilot.sighted_los is None and pilot.sighted_counts['resets'] == 1
    # a side clamp, a clamped marker that jumps along the edge, the pilot's own switch, and search clear it
    for clear in ([(.1, 0., dict(u=.99, v=.6, edge=True))], [(.1, 0., CLIP), (.1, 0., dict(u=.75, v=.97, edge=True))],
                  [(.1, 0., None), (1., 0., None)]):
        pilot, _ = fly([(.5, -6., ring_cue(14., -6.))]+clear, **kw)
        assert pilot.sighted_los is None, clear
    # the pilot's own checkpoint switch (a bearing jump beyond new_target_deg) clears it; the new ring sets its own
    pilot, _ = fly([(.5, -6., ring_cue(14., -6.)), (.1, -6., ring_cue(12., -6., azimuth_deg=36.))], **kw)
    assert pilot.target_switches == 1 and pilot.sighted_counts['resets'] == 1
    assert pilot.sighted_los == pytest.approx(12., abs=.05)
    # a short marker dropout keeps it
    pilot, _ = fly([(.5, -6., ring_cue(14., -6.)), (.3, 0., None), (.1, 0., CLIP)], **kw)
    assert pilot.sighted_los == pytest.approx(14., abs=.05)


def test_the_line_of_sight_turns_down_while_the_path_stays_above_it():
    """A drone flying shallower than the sighted line of sight passes above the ring: the estimate turns down at
    V sin(los - path)/growth_range_m, as for a ring growth_range_m away; a drone on or below the line keeps it."""
    level = (6., 0., 0.)
    pilot, rows = fly([(.5, -6., ring_cue(14., -6.)), (.06, 0., None), (1., 0., CLIP)], velocity=level,
                      descent_view=DV, sighted_descent=SD)
    los = np.array([r[4]['sighted_los'] for r in rows])
    t = np.array([r[0] for r in rows])
    after = (t > 10.6) & np.isfinite(los)
    rate = np.diff(los[after])/.01
    expected = np.degrees(6.*np.sin(np.radians(los[after][:-1]))/SD.growth_range_m)
    assert np.allclose(rate, expected, rtol=1e-6)
    assert los[after][-1] > 14.+.9*np.degrees(6.*np.sin(np.radians(14.))/SD.growth_range_m)
    steep, rows = fly([(.5, -6., ring_cue(14., -6.)), (.06, 0., None), (1., 0., CLIP)],
                      velocity=(6., 0., -6.*np.tan(np.radians(16.))), descent_view=DV, sighted_descent=SD)
    assert steep.sighted_los == pytest.approx(14., abs=.05)


def test_shadow_logs_and_changes_nothing():
    off, rows_off = fly(SIGHTED_THEN_CLIPPED, descent_view=DV)
    shadow, rows_shadow = fly(SIGHTED_THEN_CLIPPED, descent_view=DV, sighted_descent=SD, sighted_apply=False)
    assert all(np.array_equal(a[1], b[1]) and a[3] == b[3] for a, b in zip(rows_off, rows_shadow))
    assert max(r[4]['sighted_withheld'] for r in rows_shadow) > .5
    meta = shadow.metadata()['sighted_descent']
    assert meta['applied'] is False and meta['version'] == 1 and meta['seconds']['limiting'] > .8


# ---------------------------------------------------------------------------------------------
# Runner, deployed pilot, replay harness
# ---------------------------------------------------------------------------------------------
def test_runner_flag_columns_and_refusals():
    from haltere.liftoff.visual_brain import SIGHTED_COLUMNS, VisualController, resolve_sighted_descent, sighted_row
    assert resolve_sighted_descent(SimpleNamespace()) == 'off'
    assert resolve_sighted_descent(SimpleNamespace(sighted_descent='off')) == 'off'
    assert resolve_sighted_descent(SimpleNamespace(sighted_descent='on', descent_view='on')) == 'on'
    assert resolve_sighted_descent(SimpleNamespace(sighted_descent='shadow', descent_view='on')) == 'shadow'
    for bad in (SimpleNamespace(sighted_descent='on'), SimpleNamespace(sighted_descent='shadow', descent_view='off')):
        with pytest.raises(ValueError, match='needs --descent-view'):
            resolve_sighted_descent(bad)
    assert SIGHTED_COLUMNS == ('sighted_los', 'sighted_bound', 'sighted_withheld')
    assert len(sighted_row(None)) == 3 and all(np.isnan(sighted_row(None)))
    pilot, _ = fly(SIGHTED_THEN_CLIPPED, descent_view=DV, sighted_descent=SD)
    assert all(np.isfinite(sighted_row(pilot)))
    for mode, view in (('on', None), ('sideways', 'x.json')):
        with pytest.raises(ValueError, match='sighted-descent'):
            VisualController('missing.pt', 'missing.json', 'cpu', pilot_assistance='race-cue', pilot_profile='fast',
                             descent_view=view, sighted_descent=mode)


def test_runner_csv_tail_appends_the_sighted_columns_last():
    import inspect
    import re
    from haltere.liftoff import visual_brain
    source = re.sub(r'\s+', '', inspect.getsource(visual_brain.run))
    assert '*view_columns,*assist_columns,*stale_columns,*sighted_columns])' in source
    assert ('*(stale_row(controller.assistance)ifstale_columnselse()),'
            '*(sighted_row(controller.assistance)ifsighted_columnselse())])') in source
    assert "**({}ifsighted_mode=='off'elsedict(sighted_descent=sighted_mode))" in source


def test_deployed_pilot_and_replay_harness_wiring(monkeypatch, tmp_path):
    from haltere.liftoff import visual_brain as vb
    from haltere.liftoff.visual_brain import lag_turn_declaration_sha256
    from haltere.train.deployed_pilot import deployed_pilot_kwargs
    body = dict(schema='haltere.liftoff.sighted_descent.v1', version=1,
                sighted_descent=dict(margin_deg=1., edge_v=.85, agree_deg=1.5, agree_s=.5, switch_u=.1))
    path = tmp_path/'sighted_descent.json'
    path.write_text(json.dumps(dict(body, frozen=True, sha256=lag_turn_declaration_sha256(body))))
    monkeypatch.setattr(vb, 'SIGHTED_DESCENT_DECLARATION', path)
    monkeypatch.setattr(vb.load_sighted_descent, '__defaults__', (path,))
    off, record = deployed_pilot_kwargs('fast_velocity_brain_v1')
    assert 'sighted_descent' not in off and 'sighted_descent' not in record
    on, record = deployed_pilot_kwargs('fast_velocity_brain_v1', sighted_descent='on')
    assert on['sighted_descent'] == SD and 'sighted_apply' not in on and record['sighted_descent']['mode'] == 'on'
    shadow, _ = deployed_pilot_kwargs('fast_velocity_pd_v1', sighted_descent='shadow')
    assert shadow['sighted_apply'] is False
    with pytest.raises(ValueError, match='sighted-descent'):
        deployed_pilot_kwargs('fast_velocity_brain_v1', descent_view=False, sighted_descent='on')
    from haltere.obstacles.vertical_replay import build
    side = dict(motor_controller=dict(contract='fast_velocity_brain_v1'), obstacle_stack=dict(mode='on'),
                pilot_assistance=dict(nominal_speed_mps=6.), gate_sensor=SENSOR)
    from haltere.obstacles.vertical_replay import _modules  # noqa: F401 (the harness imports the pilot lazily)
    from pathlib import Path
    tree = str(Path(__file__).resolve().parents[1])
    pilot, info = build(side, tree, 'none', descent_view=DV, sighted_descent=SD, sighted_apply=False)
    assert pilot.sighted_descent == SD and pilot.sighted_apply is False and info['sighted_descent'] == 'shadow'
    pilot, info = build(side, tree, 'none', descent_view=DV)
    assert pilot.sighted_descent is None and 'sighted_descent' not in info
