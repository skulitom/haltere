"""FastRaceCue sighted descent version 2, the sighted line (`SightedDescentConfig(version=2)`), declared in
configs/pilot/sighted_descent.json (off by default; version 1 kept as sighted_descent_v1.json), and the offline arch-bar
tools of haltere.liftoff.gate_top.

These tests pin the rule on synthetic states: the line anchored by an in-view reading at the drone's position at
capture, the clamp lines that lower it, the sink it requests while the ring is clipped at the bottom edge (in place of
the view rule's bound; a floor under it with clamp lines only), the tracking bound, the resets, shadow mode, the
declaration, runner, deployed-pilot and replay wiring, and that the pilot with the rule off or in shadow is
bit-identical to m6 (the golden digests the m5 archive gave and m6 kept). None of this is flight evidence.
"""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import (DescentViewConfig, FastRaceCue, SIGHTED_DESCENT_FIELDS,
                                           SIGHTED_DESCENT_VERSION, SIGHTED_DESCENT_VERSIONS, SIGHTED_LINE_STATES,
                                           SightedDescentConfig, sighted_descent_config)
from haltere.vision.camera import Camera, quat_wxyz_to_mat
from tests.test_fast_race_cue import single_thread  # noqa: F401 (fixture)
from tests.test_fast_race_cue_descent_view import golden_scenario, pitched
from tests.test_visual_assistance import SENSOR

DV = DescentViewConfig()
SL = SightedDescentConfig(version=2, switch_u=.1, aim_above_m=0., band_m=.5, line_gain=1., correction_mps=.5)
CAMERA = Camera(320, 180, SENSOR['focal_320'], SENSOR['tilt_deg'])
CAL = dict(hover_processed=0.13566076755523682, throttle_scale=0.8, hover_stick_sim=-0.43019758448357537)


def cue_of(ring, position, q):
    """The HUD cue of a ring: in view, or clamped at the bottom edge (v 0.97) when it lies below the image."""
    body = (np.asarray(ring, float)-position) @ quat_wxyz_to_mat(q)
    px, ok = CAMERA.project_body(body[None])
    u, v = px[0, 0]/320, px[0, 1]/180
    if ok[0] and 0 <= u <= 1 and 0 <= v <= 1:
        return dict(u=float(u), v=float(v), edge=False)
    return dict(u=float(np.clip(u, .2, .8)) if ok[0] else .5, v=.97, edge=True)


def approach(ring=(60., 0., 5.), *, seconds=9., start=(0., 0., 20.), look_s=.3, pitch_down=-8., cues=None, **kw):
    """A kinematic drone at 6 m/s toward a ring below: the nose pitched pitch_deg down for look_s (the ring in view),
    then level (the ring clipped at the bottom edge). The measured velocity follows the pilot's request with a 0.125 s
    lag; a cue every 0.05 s, captured 0.05 s earlier (`cues(t, cue)` may replace it). Returns (pilot, rows) with rows
    (t, position, request, state)."""
    history = CameraPoseHistory(capacity=2000)
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., calibration=CAL, **kw)
    position, velocity = np.array(start, float), np.array([6., 0., 0.])
    rows, pending = [], []
    for k in range(int(round(seconds/.01))):
        t = 10.+k*.01
        q = pitched(pitch_down if t < 10.+look_s else 0.)
        history.append(t, position.copy(), q)
        cue = None
        if k % 5 == 0:
            cue = cue_of(ring, position, q)
            if cues is not None:
                cue = cues(t, cue)
            pending.append((t, cue))
        capture, current = pending[-1] if pending else (None, None)
        f = lambda x: torch.tensor([x], dtype=torch.float32)  # noqa: E731
        senses = dict(pos=f(position.tolist()), quat=f(q.tolist()), vel_world=f(velocity.tolist()),
                      vel_body=f((quat_wxyz_to_mat(q).T @ velocity).tolist()), gyro=f([0., 0., 0.]))
        fresh = k % 5 == 0
        pilot.update(senses, [0., 0., 0.], dict(race_cue=dict(current)) if fresh and current else None,
                     capture if fresh else None, t)
        pilot.command(np.array([-.45, 0., 0., 0.]))
        rows.append((t, position.copy(), pilot.velocity_command.copy(), pilot.state))
        velocity = velocity+.08*(pilot.velocity_command-velocity)
        position = position+velocity*.01
        if position[0] > ring[0]:
            rows.append((t+.01, position.copy(), pilot.velocity_command.copy(), pilot.state))
            break
    return pilot, rows


def height_at_ring(rows, ring=(60., 0., 5.)):
    xs = np.array([r[1][0] for r in rows])
    zs = np.array([r[1][2] for r in rows])
    return float(np.interp(ring[0], xs, zs)-ring[2]) if xs[-1] >= ring[0] else None


# ---------------------------------------------------------------------------------------------
# Configuration and declarations
# ---------------------------------------------------------------------------------------------
def test_version_two_config_and_declared_fields():
    assert SIGHTED_DESCENT_VERSION == 2 and SIGHTED_DESCENT_VERSIONS == (1, 2)
    assert SIGHTED_LINE_STATES == ('below', 'below_weak')
    assert SIGHTED_DESCENT_FIELDS[2] == ('switch_u', 'aim_above_m', 'band_m', 'line_gain', 'correction_mps')
    assert SightedDescentConfig().version == 1                     # the dataclass default stays version 1
    for bad in (dict(line_gain=0.), dict(line_gain=11.), dict(aim_above_m=2.5), dict(correction_mps=-.1),
                dict(aim_above_m=float('nan')), dict(version=3)):
        with pytest.raises(ValueError):
            SightedDescentConfig(**dict(dict(version=2), **bad))
    two = sighted_descent_config(dict(version=2, sighted_descent=dict(switch_u=.1, aim_above_m=0., band_m=.5,
                                                                       line_gain=1., correction_mps=.5)))
    assert two == SL
    with pytest.raises(ValueError, match='declares only'):
        sighted_descent_config(dict(version=2, sighted_descent=dict(margin_deg=1.)))
    with pytest.raises(ValueError, match='declares only'):
        sighted_descent_config(dict(version=1, sighted_descent=dict(line_gain=1.)))
    with pytest.raises(ValueError, match='versions'):
        sighted_descent_config(dict(version=3, sighted_descent={}))


def test_runner_loader_flies_version_two_and_refuses_the_kept_version_one(tmp_path):
    from haltere.liftoff.visual_brain import SIGHTED_DESCENT_DECLARATION, lag_turn_declaration_sha256, \
        load_sighted_descent
    kept = SIGHTED_DESCENT_DECLARATION.with_name('sighted_descent_v1.json')
    v1 = json.loads(kept.read_text(encoding='utf-8'))
    assert v1['version'] == 1 and v1['sha256'] == '49b8d7a79f32d5c5d4b63bdd815ba4092f9de35ce8c853a9580cb59f401424e6'
    assert lag_turn_declaration_sha256(v1) == v1['sha256']
    with pytest.raises(ValueError, match='version'):
        load_sighted_descent(kept)
    body = dict(schema='haltere.liftoff.sighted_descent.v2', version=2,
                sighted_descent=dict(switch_u=.1, aim_above_m=0., band_m=.5, line_gain=1., correction_mps=.5))
    path = tmp_path/'sd.json'
    path.write_text(json.dumps(dict(body, frozen=True, sha256=lag_turn_declaration_sha256(body))))
    declaration, _ = load_sighted_descent(path)
    assert sighted_descent_config(declaration) == SL


# ---------------------------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------------------------
def test_off_and_shadow_are_bit_identical_to_m6():
    digest = lambda trace: hashlib.sha256(np.ascontiguousarray(trace.astype(np.float64)).tobytes()).hexdigest()  # noqa
    from tests.test_round5_safety import v3
    V3 = v3()
    # the golden digests of the m5 archive (bc7c71c), which m6 kept (docs/sighted_descent.md, round-6 identity)
    m6 = 'efea6e68329d74a6d453a89e27b86226a49ba1d8c2766b8f5f2f846fdc55f110'
    assert digest(golden_scenario(descent_view=DV, contact_support=V3)[0]) == m6
    assert digest(golden_scenario(descent_view=DV)[0]) == 'eae7b39aeac9c76db695012782f1f6903d1ccf9ddbbe033c032dc9554dc7914a'
    shadow, pilot = golden_scenario(descent_view=DV, contact_support=V3, sighted_descent=SL, sighted_apply=False)
    assert digest(shadow) == m6
    summary = pilot.sighted_summary()
    assert summary['seconds']['acting'] > 0 and summary['counts']['sightings'] > 0 and summary['applied'] is False
    meta = pilot.metadata()['sighted_descent']
    assert meta['version'] == 2 and meta['applied'] is False and set(meta['parameters']) == set(SIGHTED_DESCENT_FIELDS[2])
    on, _ = golden_scenario(descent_view=DV, contact_support=V3, sighted_descent=SL)
    assert digest(on) != m6


def test_the_line_takes_a_clipped_ring_down_to_its_height():
    """A ring 15 m below and 60 m ahead (14 degrees), seen for 0.3 s with the nose down, then clipped with a level nose
    (the lower image edge 12 degrees down): the view rule's steep late dives up to 1.5 m below the line to the ring; the
    sighted line flies down it, aim_above_m above it, and reaches the ring's plane there."""
    ring = (60., 0., 5.)
    _, off = approach(ring, seconds=11., descent_view=DV)
    pilot, on = approach(ring, seconds=11., descent_view=DV, sighted_descent=SL)

    def below_line(rows):
        return np.array([20.-.25*r[1][0]-r[1][2] for r in rows])  # metres below the line from the start to the ring
    assert below_line(off).max() > 1.
    assert below_line(on).max() < .05
    assert -.1 < height_at_ring(on, ring) < SL.aim_above_m+SL.band_m+.1
    assert pilot.sighted_counts['sightings'] >= 1 and pilot.sighted_summary()['withheld_m'] > 1.
    assert 'below' in {r[3] for r in on}
    log = pilot.sighted_log()
    assert set(log) == {'sighted_los', 'sighted_bound', 'sighted_withheld', 'sighted_added', 'sighted_above'}


def test_the_bounded_correction_the_cue_lift_and_the_launch_plane_bound():
    pilot, _ = approach((60., 0., 5.), seconds=1., descent_view=DV, sighted_descent=SL)
    assert pilot.sighted_line is not None
    anchor, direction, slope = pilot.sighted_line
    level = np.array([6., 0., 0.])
    # on the line (at the target), flying along it: the line's own slope
    on_line = anchor+np.array([6., 0., -6.*slope+SL.aim_above_m])
    got = pilot._sighted_line_sink(5., 5., 'below', on_line, np.array([6., 0., -6.*slope]), .01)
    assert got == pytest.approx(6.*slope, abs=1e-6)
    # 3 m above the target: the height correction is bounded by correction_mps (no dive a lagging motor follows late)
    high = on_line+np.array([0., 0., 3.])
    assert pilot._sighted_line_sink(5., 5., 'below', high, level, .01) == pytest.approx(6.*slope+SL.correction_mps)
    low = on_line-np.array([0., 0., .1])
    assert pilot._sighted_line_sink(5., 5., 'below', low, level, .01) == \
        pytest.approx(6.*slope-SL.line_gain*.1, abs=1e-6)
    # within the corridor (between the line and band_m above it) the view rule's sink stands
    mid = on_line+np.array([0., 0., SL.band_m/2])
    lo, hi = 6.*slope-SL.line_gain*SL.band_m/2, 6.*slope+SL.line_gain*SL.band_m/2
    assert pilot._sighted_line_sink(1.1*lo+.0, 1.1*lo, 'below', mid, level, .01) == pytest.approx(1.1*lo)
    assert pilot._sighted_line_sink(5., 5., 'below', mid, level, .01) == pytest.approx(hi)
    assert pilot._sighted_line_sink(.1, .1, 'below', mid, level, .01) == pytest.approx(lo)
    # the ring in view: only the sink the view rule's cue margin withholds from the pilot's own request comes back,
    # and only while the drone is more than band_m above the line
    assert pilot._sighted_line_sink(1.5, .6, 'cue', high, level, .01) == pytest.approx(1.5)
    below_band = max(-SL.correction_mps, -SL.line_gain*(.1+SL.band_m))
    assert pilot._sighted_line_sink(1.5, .6, 'cue', low, level, .01) == pytest.approx(max(.6, 6.*slope+below_band))
    assert pilot._sighted_line_sink(1., 2., 'cue', high, level, .01) == pytest.approx(1.)      # nothing withheld
    assert pilot._sighted_line_sink(-1., 2., 'cue', high, level, .01) is None                   # a climb: untouched
    # other states: no request (the view rule alone), the target still logged
    assert pilot._sighted_line_sink(5., 5., 'coast', on_line, level, .01) is None
    assert np.isfinite(pilot.sighted_above)
    # near the launch plane the pilot's own launch-plane sink bound holds
    pilot.sighted_line = (np.array([0., 0., 10.]), np.array([1., 0.]), .5)
    pilot.sighted_clamps = []
    c = pilot.config
    assert pilot._sighted_line_sink(5., 5., 'below', np.array([6., 0., .4]), np.array([6., 0., -3.]), .01) <= \
        c.surface_sink+c.surface_sink_per_m*.4+1e-9


def test_clamp_lines_lower_the_target_and_clamps_alone_are_a_floor():
    pilot, _ = approach((60., 0., 5.), seconds=1., descent_view=DV, sighted_descent=SL)
    anchor, direction, slope = pilot.sighted_line
    here = anchor+np.r_[3.*direction, -3.*slope]
    # a clamp line from here, steeper than the sighted line: the ring lies below it, so the target follows it ahead
    steep = (here.copy(), direction.copy(), slope+.2)
    pilot.sighted_clamps = [steep]
    ahead = here+np.r_[5.*direction, 0.]
    heights = [pilot._line_height(line, ahead) for line in (pilot.sighted_line, steep)]
    assert heights[1] < heights[0]
    pilot._sighted_line_sink(5., 5., 'below', ahead, np.r_[6.*direction, -1.], .01)
    assert pilot.sighted_above == pytest.approx(ahead[2]-heights[1]-SL.aim_above_m)
    assert pilot.sighted_los == pytest.approx(np.degrees(np.arctan(slope+.2)))
    # clamp lines only (no sighting since the switch): a floor under the view rule's sink, never less than it
    pilot.sighted_line = None
    pilot.sighted_clamps = [(here.copy(), direction.copy(), .05)]  # a shallow clamp line: the view rule's sink stands
    got = pilot._sighted_line_sink(1.5, 1.2, 'below', here+np.r_[1.*direction, 0.], np.array([6., 0., -1.]), .01)
    assert got == pytest.approx(1.2) and pilot.sighted_withheld == 0.
    pilot.sighted_clamps = [(here+np.array([0., 0., -2.]), direction.copy(), .3)]  # a deep one: more sink, bounded
    got = pilot._sighted_line_sink(1.5, .5, 'below', here, np.array([6., 0., -1.]), .01)
    assert got == pytest.approx(6.*.3+SL.correction_mps) and pilot.sighted_added > 0


def test_what_anchors_and_clears_the_line():
    ring = (60., 0., 5.)
    pilot, rows = approach(ring, seconds=.2, descent_view=DV, sighted_descent=SL)
    assert pilot.sighted_line is not None and not pilot.sighted_clamps
    # the anchor is the drone's position at capture (the pose history) and the ring-centre ray
    anchor, direction, slope = pilot.sighted_line
    assert np.degrees(np.arctan(slope)) == pytest.approx(np.degrees(np.arctan2(15., 60.)), abs=1.)
    # clamps of the same ring add clamp lines (pruned to those that can be the lowest ahead)
    pilot, _ = approach(ring, seconds=2., descent_view=DV, sighted_descent=SL)
    assert pilot.sighted_counts['clamp_lines'] > 10 and len(pilot.sighted_clamps) <= 5
    # a side clamp clears it; a clamped marker that jumps along the edge clears it; search clears it
    side = lambda t, c: dict(u=.99, v=.6, edge=True) if t > 11. else c  # noqa: E731
    pilot, _ = approach(ring, seconds=1.2, cues=side, descent_view=DV, sighted_descent=SL)
    assert pilot.sighted_line is None and pilot.sighted_counts['resets'] >= 1
    jump = lambda t, c: dict(u=.8, v=.97, edge=True) if t > 11. else (dict(c, u=.5) if c['edge'] else c)  # noqa
    pilot, _ = approach(ring, seconds=1.2, cues=jump, descent_view=DV, sighted_descent=SL)
    assert pilot.sighted_line is None
    lost = lambda t, c: None if t > 10.5 else c  # noqa: E731
    pilot, _ = approach(ring, seconds=3., cues=lost, descent_view=DV, sighted_descent=SL)
    assert pilot.state == 'search' and pilot.sighted_line is None
    # the pilot's own checkpoint switch (a bearing jump beyond new_target_deg) clears it; the new ring anchors its own
    other = lambda t, c: cue_of((40., 40., 5.), np.zeros(3), pitched(-8.)) if 10.24 < t < 10.31 else c  # noqa: E731
    pilot, _ = approach(ring, seconds=.3, cues=other, descent_view=DV, sighted_descent=SL)
    assert pilot.target_switches >= 1 and pilot.sighted_counts['resets'] >= 1


def test_shadow_logs_and_changes_nothing():
    ring = (60., 0., 5.)
    _, off = approach(ring, descent_view=DV)
    shadow, rows = approach(ring, descent_view=DV, sighted_descent=SL, sighted_apply=False)
    assert all(np.array_equal(a[2], b[2]) for a, b in zip(off, rows))
    assert shadow.sighted_summary()['withheld_m'] > 1. and shadow.metadata()['sighted_descent']['applied'] is False


# ---------------------------------------------------------------------------------------------
# Runner, deployed pilot, replay harness
# ---------------------------------------------------------------------------------------------
def test_runner_columns_and_row():
    from haltere.liftoff.visual_brain import SIGHTED_COLUMNS, sighted_row
    assert SIGHTED_COLUMNS == ('sighted_los', 'sighted_bound', 'sighted_withheld', 'sighted_added', 'sighted_above')
    assert len(sighted_row(None)) == 5 and all(np.isnan(sighted_row(None)))
    pilot, _ = approach((60., 0., 5.), seconds=2., descent_view=DV, sighted_descent=SL)
    row = sighted_row(pilot)
    assert np.isfinite(row[0]) and np.isfinite(row[4])


def test_deployed_pilot_takes_the_runner_declaration_or_the_kept_version_one(monkeypatch, tmp_path):
    from haltere.liftoff import visual_brain as vb
    from haltere.liftoff.visual_brain import lag_turn_declaration_sha256
    from haltere.train.deployed_pilot import deployed_pilot_kwargs
    body = dict(schema='haltere.liftoff.sighted_descent.v2', version=2,
                sighted_descent=dict(switch_u=.1, aim_above_m=0., band_m=.5, line_gain=1., correction_mps=.5))
    path = tmp_path/'sighted_descent.json'
    path.write_text(json.dumps(dict(body, frozen=True, sha256=lag_turn_declaration_sha256(body))))
    kept = vb.SIGHTED_DESCENT_DECLARATION.with_name('sighted_descent_v1.json')
    (tmp_path/'sighted_descent_v1.json').write_bytes(kept.read_bytes())
    monkeypatch.setattr(vb, 'SIGHTED_DESCENT_DECLARATION', path)
    monkeypatch.setattr(vb.load_sighted_descent, '__defaults__', (path,))
    on, record = deployed_pilot_kwargs('fast_velocity_brain_v1', sighted_descent='on')
    assert on['sighted_descent'] == SL and record['sighted_descent']['version'] == 2
    one, record = deployed_pilot_kwargs('fast_velocity_brain_v1', sighted_descent='on', sighted_version=1)
    assert one['sighted_descent'] == SightedDescentConfig() and record['sighted_descent']['version'] == 1
    assert record['sighted_descent']['sha256'].startswith('49b8d7a79f32')
    shadow, _ = deployed_pilot_kwargs('fast_velocity_pd_v1', sighted_descent='shadow')
    assert shadow['sighted_apply'] is False
    from haltere.obstacles.vertical_replay import build
    from pathlib import Path
    side = dict(motor_controller=dict(contract='fast_velocity_brain_v1'), obstacle_stack=dict(mode='on'),
                pilot_assistance=dict(nominal_speed_mps=6.), gate_sensor=SENSOR)
    pilot, info = build(side, str(Path(__file__).resolve().parents[1]), 'none', descent_view=DV, sighted_descent=SL)
    assert pilot.sighted_descent == SL and info['sighted_descent'] == 'on'


# ---------------------------------------------------------------------------------------------
# Offline arch bars and the two-ring Straw downhill (haltere.liftoff.gate_top)
# ---------------------------------------------------------------------------------------------
def test_gate_top_plane_crossings_and_verdicts():
    from haltere.liftoff.gate_top import GateTop, GateTopConfig
    course = np.array([[10., 0., 5.], [20., 0., 5.], [30., 0., 5.]])
    top = GateTop(course, start=(0., 0., 5.))
    for x in np.arange(0., 35., .5):
        z = 5.+(1. if x < 15 else 0.)+(.4 if 25 < x else 0.)+(1.6 if x > 28 else 0.)
        top.step((x, .3, z))
    result = top.result()
    verdicts = [c['verdict'] for c in result['crossings']]
    assert verdicts == ['bar', 'through', 'over'] and result['top_bar_hits'] == 1 and result['over_arch'] == 1
    assert result['crossings'][0]['above_m'] == pytest.approx(1.) and result['crossings'][0]['lateral_m'] == \
        pytest.approx(.3)
    far = GateTop(course, start=(0., 0., 5.))
    for x in np.arange(0., 35., .5):
        far.step((x, 9., 6.))                                     # beyond reach_m sideways: no crossing counted
    assert far.result()['crossings'] == [None, None, None]
    with pytest.raises(ValueError):
        GateTopConfig(bar_low_m=1.5, bar_high_m=1.)


def test_the_two_ring_straw_downhill():
    from haltere.liftoff.gate_top import STRAW_ARCHES, StrawTwoRings
    logged = StrawTwoRings(None)
    assert np.allclose(logged.a, STRAW_ARCHES['A']['centre']) and np.allclose(logged.b, STRAW_ARCHES['B']['centre'])
    assert logged.course.shape == (4, 3) and logged.course[3, 2] == pytest.approx(6.)
    assert logged.ground_logged(-36.3, 120.8) < logged.a[2] < logged.ground_logged(-36.3, 140.)
    assert logged.b[2]-logged.ground_logged(*logged.b[:2]) == pytest.approx(1.7)
    for seed in (1, 2, 3):
        v = StrawTwoRings(seed)
        assert (v.course[:, 2] >= 6.-1e-9).all()
        assert 18. <= v.b_length <= 32. and 8. <= v.b_slope <= 20. and 1.2 <= v.b_clearance <= 2.2
        assert v.b[2]-v.ground_logged(*v.b[:2]) == pytest.approx(v.b_clearance)
        g = v.ground_at(3, None, 0., v.course[3][:2])
        assert g is not None and g[0] == pytest.approx(v.course[3][2]-v.b_clearance)
        assert v.ground_at(1, None, 0., v.course[1][:2]) is None
