"""FastRaceCue view-keeping descent (`DescentViewConfig`), declared in configs/pilot/descent_view.json (off by default).

These tests pin the rule on synthetic states: the view bound's geometry (the flight path kept inside the camera's lower
image edge at the measured attitude), keep speed for bottom-clipped rings and the descent-path governor's floor, more
speed rather than less, the gentle sink onset, the late steepening for rings that stay clipped, that climbs are never
changed, the declaration and runner wiring, and that the default pilot is bit-identical to m2-vertical (935cfdb) when
the rule is off. The surrogate gates are in haltere/liftoff/descent_gates.py. None of this is flight evidence.
"""
import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import (DESCENT_VIEW_VERSION, DescentViewConfig, FastRaceCue, descent_view_config)
from haltere.vision.camera import Camera, quat_wxyz_to_mat
from tests.test_fast_race_cue import BELOW, cue_toward, drive, single_thread  # noqa: F401 (fixture)
from tests.test_visual_assistance import SENSOR

DV = DescentViewConfig()
EXACT = DescentViewConfig(attitude_time_constant=0.)     # no attitude low-pass: the bound of the current attitude
CAMERA = Camera(320, 180, SENSOR['focal_320'], SENSOR['tilt_deg'])


def pitched(pitch_deg):
    """Quaternion (w, x, y, z) of a level-yaw attitude with the nose pitch_deg above the horizon (negative: down)."""
    theta = -np.radians(pitch_deg)
    return np.array([np.cos(theta/2), 0., np.sin(theta/2), 0.])


def state(velocity=(6., 0., 0.), pitch_deg=0., height=10.):
    q = pitched(pitch_deg)
    R = quat_wxyz_to_mat(q)
    t = lambda x: torch.tensor([x], dtype=torch.float32)
    return dict(pos=t([0., 0., height]), quat=t(q.tolist()), vel_world=t(list(velocity)),
                vel_body=t((R.T@np.asarray(velocity, float)).tolist()), gyro=t([0., 0., 0.]))


def run(cue, *, seconds=3., velocity=(6., 0., 0.), pitch_deg=0., perfect=False, **kw):
    """The pilot at a fixed measured attitude and velocity (perfect=True: the measured vertical speed follows the
    request), a cue every tick; rows (t, command, state, descent_scale, view log)."""
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **kw)
    measured = np.array(velocity, float)
    rows = []
    for k in range(int(round(seconds/.01))):
        t = 10.+k*.01
        s = state(tuple(measured.tolist()), pitch_deg)
        history.append(t, [0., 0., 10.], s['quat'][0].numpy())
        current = cue(t) if callable(cue) else cue
        pilot.update(s, [0., 0., 0.], dict(race_cue=dict(current)) if current else None, t-.05, t)
        rows.append((t, pilot.velocity_command.copy(), pilot.state, pilot.descent_scale, pilot.descent_view_log()))
        if perfect:
            measured[2] = pilot.velocity_command[2]
    return pilot, rows


# ---------------------------------------------------------------------------------------------
# Declaration and configuration
# ---------------------------------------------------------------------------------------------
def test_config_validation():
    for bad in (dict(margin_deg=-1.), dict(margin_deg=45.), dict(below_speed_fraction=0.), dict(descent_min_scale=1.5),
                dict(sink_acceleration=0.), dict(free_sink=float('nan')), dict(late_rate_deg_s=-1.)):
        with pytest.raises(ValueError):
            DescentViewConfig(**bad)
    with pytest.raises(ValueError, match='DescentViewConfig'):
        FastRaceCue(SENSOR, CameraPoseHistory(), 6., descent_view=dict(margin_deg=3.))
    with pytest.raises(ValueError, match='version'):
        descent_view_config(dict(version=DESCENT_VIEW_VERSION+1, descent_view={}))
    with pytest.raises(ValueError, match='version'):
        descent_view_config(dict(version=0, descent_view={}))


def test_repository_declaration_is_frozen_and_declares_the_defaults(tmp_path):
    from haltere.liftoff.visual_brain import DESCENT_VIEW_DECLARATION, lag_turn_declaration_sha256, load_descent_view
    declaration, digest = load_descent_view(DESCENT_VIEW_DECLARATION)
    # version 2 (round 4b) = version 1's view rule, unchanged, + contact support; version 1 is kept and refused
    assert declaration['version'] == DESCENT_VIEW_VERSION == 2 and declaration['frozen'] is True
    assert digest == declaration['sha256'] == lag_turn_declaration_sha256(declaration)
    assert descent_view_config(declaration) == DV                   # the declared values are the defaults
    kept_path = DESCENT_VIEW_DECLARATION.with_name('descent_view_v1.json')
    kept = json.loads(kept_path.read_text(encoding='utf-8'))
    assert kept['version'] == 1 and kept['sha256'] == '8afb64d730adcfc9ba7e502df38901a04ac42edc3e0011d15a7672019c79e33d'
    assert kept['descent_view'] == declaration['descent_view'] and descent_view_config(kept) == DV
    assert declaration['previous_versions'][0]['sha256'] == kept['sha256']
    with pytest.raises(ValueError, match='version'):
        load_descent_view(kept_path)
    edited = dict(declaration, descent_view=dict(declaration['descent_view'], margin_deg=0.))
    path = tmp_path/'edited.json'
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match='changed after the freeze'):
        load_descent_view(path)
    other = {k: v for k, v in declaration.items() if k not in ('frozen', 'frozen_at', 'sha256')}
    other['version'] = 3
    other.update(frozen=True, sha256=lag_turn_declaration_sha256(other))
    path.write_text(json.dumps(other))
    with pytest.raises(ValueError, match='version'):
        load_descent_view(path)


def test_gates_declaration_is_frozen_and_names_the_rule():
    from haltere.liftoff.descent_gates import GATES_PATH, content_sha256, load_gates
    gates, digest, path = load_gates(GATES_PATH)
    declaration = json.loads(path.read_text(encoding='utf-8'))
    assert digest == gates['sha256'] == content_sha256(gates)
    assert gates['descent_view']['sha256'] == declaration['sha256'] and gates['descent_view']['version'] == 1
    assert set(gates['controllers']) == {'pd', 'brain08', 'brain09b'}
    dev = set(gates['course_sets']['dev'])
    assert not dev & set(gates['course_sets']['gate'])                # the design seeds are not scored


# ---------------------------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------------------------
def lowest_path_deg(pilot, pitch_deg, speed=6.):
    bound = pilot._view_sink_bound(np.array([speed, 0., 0.]), quat_wxyz_to_mat(pitched(pitch_deg)))
    return -np.degrees(np.arctan2(bound, speed)), bound


@pytest.mark.parametrize('pitch_deg', [-15., -8., 0., 4.])
def test_the_bound_keeps_the_path_inside_the_lower_image_edge(pitch_deg):
    """At a level nose the lower image edge lies 12 degrees below the horizon (30 up, 42 half field of view): with a
    3 degree margin the path may point 9 degrees down; the nose pitch moves the edge one for one."""
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., descent_view=EXACT)
    edge = pitch_deg+30.-np.degrees(np.arctan(90./100.))
    path, bound = lowest_path_deg(pilot, pitch_deg)
    if edge+DV.margin_deg < -np.degrees(np.arctan2(DV.free_sink, 6.)):
        assert path == pytest.approx(edge+DV.margin_deg, abs=.05)
        # the bounded direction projects exactly margin_deg inside the image's lower edge
        body = np.array([6., 0., -bound]) @ quat_wxyz_to_mat(pitched(pitch_deg))
        px, ok = CAMERA.project_body(body[None])
        assert ok[0] and np.degrees(np.arctan((px[0, 1]-90.)/100.)) == pytest.approx(
            np.degrees(np.arctan(.9))-DV.margin_deg, abs=1e-6)
    else:
        assert bound == DV.free_sink                               # nose up (braking): no more than the free sink


def test_the_bound_scales_with_the_measured_horizontal_speed_and_allows_a_slow_free_sink():
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., descent_view=EXACT)
    fast = pilot._view_sink_bound(np.array([6., 0., 0.]), np.eye(3))
    slow = pilot._view_sink_bound(np.array([3., 0., 0.]), np.eye(3))
    hover = pilot._view_sink_bound(np.zeros(3), np.eye(3))
    assert fast == pytest.approx(2*slow, rel=1e-6) and fast == pytest.approx(6*np.tan(np.radians(9.)), abs=.02)
    assert hover == DV.free_sink


def test_a_bottom_clipped_ring_keeps_speed_and_the_sink_stays_in_view():
    """The Straw Bale pattern: the default pilot halves the speed for a bottom clip and asks for a descent 5-20
    degrees below the clipped edge ray; the rule keeps the speed schedule and bounds the sink to the view (here
    without the late steepening, which test_a_ring_that_stays_clipped_is_approached_steeply_late covers)."""
    strict = DescentViewConfig(late_max_deg=0.)
    _, plain = run(BELOW, perfect=True)
    pilot, rows = run(BELOW, perfect=True, descent_view=strict)
    assert {r[2] for r in rows[50:]} == {'below'} and {r[2] for r in plain[50:]} == {'below'}
    assert np.linalg.norm(plain[-1][1][:2]) < 3.5 < 5.5 < np.linalg.norm(rows[-1][1][:2])
    assert path_deg(plain[-1][1]) < -20.                              # the default: far below the image
    assert path_deg(rows[-1][1]) == pytest.approx(-9., abs=.5)        # the rule: 3 degrees inside a -12 degree edge
    assert rows[-1][4]['view_withheld'] > 0 and rows[-1][4]['view_sink_bound'] == pytest.approx(-rows[-1][1][2], abs=.02)
    meta = json.loads(json.dumps(pilot.metadata()))['descent_view']
    # the view rule alone is version 1's (version 2 adds contact support)
    assert meta['version'] == 1 and meta['parameters'] == asdict(strict)
    assert meta['seconds']['limiting'] > 2. and meta['withheld_m'] > 0


def path_deg(command):
    return float(np.degrees(np.arctan2(command[2], np.linalg.norm(command[:2]))))


def test_a_ring_that_stays_clipped_is_approached_steeply_late():
    """Steep late: after late_after_s of unbroken bottom clip the margin falls at late_rate_deg_s, to at most
    late_max_deg below the lower image edge; before that the path stays in view."""
    _, rows = run(BELOW, perfect=True, descent_view=DV, seconds=6.)
    t = np.array([r[0] for r in rows])-10.
    paths = np.array([path_deg(r[1]) for r in rows])
    early = paths[(t > .45) & (t < .7)]
    assert early.min() >= -9.5                                         # in view before late_after_s
    # 3 s in the margin is 3 - 6*(3-0.75) = -10.5 (a 22.5 degree path); the 1 s attitude low-pass of the bound lags it
    assert -24. < paths[np.argmin(abs(t-3.))] < -15.
    late = paths[(t > 1.) & (t < 4.5)]
    assert np.all(np.diff(late) <= 1e-6)                               # steepening
    assert paths.min() >= -12.-DV.late_max_deg-1. and paths[-1] < -26.  # toward late_max_deg below the edge, not beyond


def test_the_descent_path_governor_is_not_fed_while_the_bound_withholds_and_keeps_its_floor():
    """Resting on a slope (the drone cannot sink): the default governor cuts the horizontal request to 0.35x; with the
    rule it is never cut below descent_min_scale."""
    ahead_below = cue_toward([10., 0., -1.5])                   # in view, 8.5 degrees down: the pilot asks ~0.9 m/s
    _, plain = run(ahead_below, seconds=3.)
    _, rows = run(ahead_below, seconds=3., descent_view=DV)
    assert min(r[3] for r in plain) == pytest.approx(.35)
    assert min(r[3] for r in rows) >= DV.descent_min_scale


def test_the_sink_starts_gently():
    _, rows = run(BELOW, perfect=True, descent_view=DV, seconds=1.)
    vz = np.array([r[1][2] for r in rows])
    assert np.diff(vz).min() >= -DV.sink_acceleration*.01-1e-9
    _, plain = run(BELOW, perfect=True, seconds=1.)
    assert np.diff([r[1][2] for r in plain]).min() < -DV.sink_acceleration*.01-1e-3


def test_climbs_are_never_changed():
    above = cue_toward([10., 0., 4.])
    _, plain = run(above, perfect=True)
    _, rows = run(above, perfect=True, descent_view=DV)
    np.testing.assert_array_equal(np.array([r[1] for r in rows]), np.array([r[1] for r in plain]))


def test_off_by_default_logs_nan_and_adds_no_metadata():
    pilot, rows = run(BELOW)
    assert pilot.descent_view is None and 'descent_view' not in pilot.metadata()
    assert all(np.isnan(v) for v in rows[-1][4].values())


def golden_scenario(**kw):
    """A scripted descent, clip and climb sequence with a lagging plant that rests on a slope (the same scenario
    gave sha256 67d31f28... with the m2-vertical tree, 935cfdb)."""
    cal = dict(hover_processed=0.13566076755523682, throttle_scale=0.8, hover_stick_sim=-0.43019758448357537)
    history = CameraPoseHistory()
    pilot = FastRaceCue(dict(focal_320=100., tilt_deg=30.), history, 6., reference_speed=6., calibration=cal, **kw)
    position, velocity = np.array([0., 0., 20.]), np.zeros(3)
    ring = [np.array([40., 5., 12.]), np.array([70., -10., 2.]), np.array([90., 20., 9.])]
    target, out = 0, []
    for k in range(2400):
        now = k*.01
        pitch = np.radians(np.clip(2*(velocity[0]-4.), -20, 20))*.5
        q = np.array([np.cos(pitch/2), 0., np.sin(pitch/2), 0.])
        history.append(now, position.copy(), q)
        if np.linalg.norm(ring[target]-position) < 3 and target < 2:
            target += 1
        body = (ring[target]-position) @ quat_wxyz_to_mat(q)
        px, ok = CAMERA.project_body(body[None])
        u, v = px[0, 0]/320, px[0, 1]/180
        edge = not (ok[0] and 0 <= u <= 1 and 0 <= v <= 1)
        cue = dict(u=float(np.clip(u, 0, 1)) if ok[0] else .5, v=float(np.clip(v, 0, 1)) if ok[0] else .99, edge=edge)
        if edge and ok[0] and v > 1:
            cue['v'] = .99
        t = lambda x: torch.tensor([x], dtype=torch.float32)
        senses = dict(pos=t(position.tolist()), quat=t(q.tolist()), vel_world=t(velocity.tolist()),
                      vel_body=t((quat_wxyz_to_mat(q).T @ velocity).tolist()), gyro=t([0., 0., 0.]))
        pilot.update(senses, [0., 0., 0.], dict(race_cue=cue) if k % 7 else None, now-.05, now)
        pilot.command(np.array([-.45+.1*np.sin(k*.05), 0., 0., 0.]))
        out.append(np.r_[pilot.velocity_command, pilot.feedforward, pilot.pilot.sight_yaw, pilot.descent_scale])
        velocity = velocity+.08*(pilot.velocity_command-velocity)
        velocity[2] = max(velocity[2], -.8)
        position = position+velocity*.01
    return np.asarray(out), pilot


def test_default_pilot_is_bit_identical_to_m2_vertical():
    trace, pilot = golden_scenario()
    digest = hashlib.sha256(np.ascontiguousarray(trace.astype(np.float64)).tobytes()).hexdigest()
    assert {'below', 'cue'} <= set(pilot.state_time)
    assert digest == '67d31f2892c3f292871a40e6cb78e8a734ae540636aa8a3508a1ce3572ad3d1f'
    changed, _ = golden_scenario(descent_view=DV)
    assert not np.array_equal(changed, trace)                         # the scenario exercises the rule


# ---------------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------------
def test_runner_flag_columns_and_refusals():
    from types import SimpleNamespace
    from haltere.liftoff.visual_brain import (DESCENT_VIEW_COLUMNS, DESCENT_VIEW_DECLARATION, VisualController,
                                              descent_view_row, resolve_descent_view)
    assert resolve_descent_view(SimpleNamespace()) is None
    assert resolve_descent_view(SimpleNamespace(descent_view='off')) is None
    assert resolve_descent_view(SimpleNamespace(descent_view='on')) == str(DESCENT_VIEW_DECLARATION)
    assert resolve_descent_view(SimpleNamespace(descent_view='x.json')) == 'x.json'
    assert len(descent_view_row(None)) == len(DESCENT_VIEW_COLUMNS) == 3 and all(np.isnan(descent_view_row(None)))
    pilot, _ = run(BELOW, descent_view=DV, seconds=.5)
    assert all(np.isfinite(descent_view_row(pilot)))
    with pytest.raises(ValueError, match='fast pilot'):
        VisualController('missing.pt', 'missing.json', 'cpu', pilot_assistance='race-cue', descent_view='x.json')


def test_runner_csv_tail_keeps_gap_commit_before_the_view_columns():
    """Round-4 merge (m4): the header and every row end with the vertical-guard columns, gap_commit, then the three
    view columns only with --descent-view on, in the same order (round 4b: then the motor-assist columns only with
    --motor-assist on, tests/test_fast_race_cue_motor_assist.py)."""
    import inspect
    import re
    from haltere.liftoff import visual_brain
    source = re.sub(r'\s+', '', inspect.getsource(visual_brain.run))
    assert '*VERTICAL_COLUMNS,*COMMIT_COLUMNS,*view_columns,' in source
    assert ('*vertical_row(controller.assistance),*commit_row(controller.assistance),'
            '*(descent_view_row(controller.assistance)ifview_columnselse()),') in source
