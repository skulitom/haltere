"""FastRaceCue wall-pilot rules: turn first at a wall (`TurnFirstConfig`) and the ceiling guard of the TTC
governor's terrain climb (`CeilingGuardConfig`), declared in configs/obstacles/wall_pilot.json (obstacle stack).

These tests pin the declared rules (engagement, the bounded request, release, timeout; unexplained alarms as walls,
weak climbs, the overhead cut and hold), that below-path climbs and the default behaviour are unchanged, that the
shadow control flies exactly the unguarded pilot, and a closed-loop hairpin in the measured surrogate. The
open-loop replays of the live logs are outside the repository. None of this is flight evidence.
"""
import json
from collections import deque
from dataclasses import asdict

import numpy as np
import pytest
import torch

from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import (WALL_PILOT_VERSION, CeilingGuardConfig, ClearanceConfig, FastRaceCue,
                                           TtcClearanceConfig, TtcClearanceGovernor, TurnFirstConfig,
                                           wall_pilot_configs)
from tests.test_fast_race_cue import cue_toward, drive, single_thread  # noqa: F401 (fixture)
from tests.test_visual_assistance import SENSOR, senses

X = [1., 0., 0.]
FRAME = 1/18.
TTC = TtcClearanceConfig()               # the flown TTC policy
GUARD = CeilingGuardConfig()
TURN = TurnFirstConfig()
AHEAD = cue_toward([10., 0., 0.])


def run(gov, sample_of, seconds, *, start=0., speed=6., height=1., follow=False, delay=.08, rise=.5):
    """Drone along +x at `speed`; sample_of(t) -> None or dict(ttc, below, lower) at 18 Hz, `delay` late.
    `rise` is the measured vertical speed (m/s; a number or a function of time). follow=True integrates the climb
    request into the height. Rows: dict(t, z, cap, climb, vcap, status)."""
    rows, pending, next_frame, z = [], [], start, height
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
        vz = rise(now) if callable(rise) else rise
        cap, _, climb = gov.limits([x, 0., z], [speed, 0., vz], now, .01, 3.5)
        if follow:
            z += climb*.01
        rows.append(dict(t=now, z=z, cap=cap, climb=climb, vcap=gov.vertical_cap, status=gov.status))
    return rows


def series(rows, key):
    return np.array([np.nan if r[key] is None else r[key] for r in rows], float)


# ---------------------------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------------------------
def test_config_validation():
    for bad in (dict(release_deg=60.), dict(engage_deg=95., release_deg=30.), dict(creep_speed=0.),
                dict(max_s=float('nan')), dict(rearm_s=-1.)):
        with pytest.raises(ValueError):
            TurnFirstConfig(**bad)
    for bad in (dict(overhead_fraction=.6), dict(overhead_confirm=1.5), dict(vertical_cap=2.), dict(hold_s=0.),
                dict(weak_climb=float('inf')), dict(lower_ratio=-1.), dict(overhead_min_rise=0.),
                dict(overhead_positive=3), dict(overhead_positive=.5)):
        with pytest.raises(ValueError):
            CeilingGuardConfig(**bad)
    assert CeilingGuardConfig(vertical_cap=-.5).vertical_cap == -.5          # a slight descent may be declared
    with pytest.raises(ValueError, match='CeilingGuardConfig'):
        TtcClearanceGovernor(TTC, ceiling=TURN)
    with pytest.raises(ValueError, match='TTC clearance policy'):
        FastRaceCue(SENSOR, CameraPoseHistory(), 6., clearance_config=ClearanceConfig(), ceiling_guard=GUARD)
    with pytest.raises(ValueError, match='TurnFirstConfig'):
        FastRaceCue(SENSOR, CameraPoseHistory(), 6., turn_first=GUARD)


def test_repository_declaration_is_frozen_and_declares_the_defaults(tmp_path):
    from haltere.liftoff.visual_brain import WALL_PILOT_DECLARATION, lag_turn_declaration_sha256, load_wall_pilot
    declaration, digest = load_wall_pilot(WALL_PILOT_DECLARATION)
    assert declaration['version'] == WALL_PILOT_VERSION == 3 and declaration['frozen'] is True
    assert digest == declaration['sha256'] == lag_turn_declaration_sha256(declaration)
    configs = wall_pilot_configs(declaration)
    assert configs == dict(turn_first=TURN, ceiling_guard=GUARD)            # the declared values are the defaults
    edited = dict(declaration, turn_first=dict(declaration['turn_first'], creep_speed=2.))
    path = tmp_path/'edited.json'
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match='changed after the freeze'):
        load_wall_pilot(path)
    other = {k: v for k, v in declaration.items() if k not in ('frozen', 'frozen_at', 'sha256')}
    other['version'] = 4
    other.update(frozen=True, sha256=lag_turn_declaration_sha256(other))
    path.write_text(json.dumps(other))
    with pytest.raises(ValueError, match='version'):
        load_wall_pilot(path)
    with pytest.raises(ValueError, match='version'):
        wall_pilot_configs(other)


def test_earlier_versions_are_kept_verbatim_and_refused():
    """Versions 1 and 2 (replayed, never flown) are kept for provenance: version 2 added the ceiling guard's
    overhead_min_rise to version 1, version 3 its overhead_positive; nothing else changed."""
    from haltere.liftoff.visual_brain import (WALL_PILOT_DECLARATION, lag_turn_declaration_sha256,
                                              load_wall_pilot)
    current, _ = load_wall_pilot(WALL_PILOT_DECLARATION)
    older = {n: json.loads(WALL_PILOT_DECLARATION.with_name(f'wall_pilot_v{n}.json').read_text(encoding='utf-8'))
             for n in (1, 2)}
    assert older[1]['sha256'].startswith('17fecfb1fad7') and older[2]['sha256'].startswith('095addc577c0')
    added = {2: 'overhead_min_rise', 3: 'overhead_positive'}
    for n, newer in ((1, older[2]), (2, current)):
        old = older[n]
        assert old['version'] == n and old['frozen'] is True and lag_turn_declaration_sha256(old) == old['sha256']
        assert newer['previous_versions'][0]['sha256'] == old['sha256'] and newer['change']
        assert old['turn_first'] == newer['turn_first']
        assert {k: v for k, v in newer['ceiling_guard'].items() if k != added[n+1]} == old['ceiling_guard']
        with pytest.raises(ValueError, match='version'):
            load_wall_pilot(WALL_PILOT_DECLARATION.with_name(f'wall_pilot_v{n}.json'))
    assert [v['version'] for v in current['previous_versions']] == [2, 1]


# ---------------------------------------------------------------------------------------------
# Ceiling guard (governor)
# ---------------------------------------------------------------------------------------------
def ceiling_after_a_small_climb(t):
    """The minus-brain08-gapon-01 shape: a surface below a sinking path (small climb), then a ceiling ahead of the
    climbing path: alarms without a vertical fraction, the lower window seeing the floor farther than the alarm."""
    if t < .25:
        return dict(ttc=1.1, below=1., lower=.5)
    if t < .5:
        return dict(ttc=6.)
    return dict(ttc=max(.3, .95-.8*(t-.5)), lower=1.5)


def test_unexplained_alarms_during_a_climb_escalate_it_without_the_guard_and_cut_it_with_it():
    plain, guarded = TtcClearanceGovernor(TTC), TtcClearanceGovernor(TTC, ceiling=GUARD)
    a, b = run(plain, ceiling_after_a_small_climb, 1.2), run(guarded, ceiling_after_a_small_climb, 1.2)
    first = max(r['climb'] for r in a if r['t'] < .5)
    assert 0 < first < 2.5 and max(r['climb'] for r in b if r['t'] < .5) == pytest.approx(first)
    assert max(r['climb'] for r in a) == pytest.approx(3.5)                  # the old rule climbs into the ceiling
    late = [r for r in b if r['t'] > .75]
    assert max(r['climb'] for r in late) == 0. and all(r['vcap'] == GUARD.vertical_cap for r in late)
    assert late[-1]['status'] == 'overhead' and b[-1]['cap'] is not None       # braked for as a wall
    counts = guarded.counts
    assert counts['overhead_engagements'] == 1 and counts['overhead_samples'] >= GUARD.overhead_confirm
    assert counts['unexplained_walls'] >= 2 and counts['climb_engagements'] == 1


def test_an_alarm_ahead_of_a_sinking_path_does_not_cut_the_climb_that_arrests_the_sink():
    """The v1 replay of minus-brain08-gapon-01 at 17.7 s: below-path evidence (the floor under a sinking path) asks
    for a climb, then two unexplained alarms arrive while the drone still sinks. They are braked for as walls, but a
    ceiling cannot cross a sinking path: no overhead cut (version 2's overhead_min_rise)."""
    def floor_then_unexplained(t):
        return dict(ttc=1.1, below=1., lower=.3) if t < .2 else dict(ttc=1.1, lower=2.)
    sinking = TtcClearanceGovernor(TTC, ceiling=GUARD)
    rows = run(sinking, floor_then_unexplained, .6, rise=-.8)
    assert sinking.counts['overhead_samples'] == 0 and all(r['vcap'] is None for r in rows)
    assert max(r['climb'] for r in rows) > 0 and rows[30]['climb'] > 0            # the climb is kept (its hold)
    assert sinking.counts['unexplained_walls'] >= 2
    rising = TtcClearanceGovernor(TTC, ceiling=GUARD)
    rows = run(rising, floor_then_unexplained, .6, rise=.8)
    assert rising.counts['overhead_engagements'] == 1 and rows[-1]['vcap'] == GUARD.vertical_cap


def test_alarms_with_no_vertical_evidence_cannot_cut_a_hill_climb_alone():
    """The v2 Straw Bale replay: climbing a hill, both vertical windows lost the path (below_fraction and lower TTC
    None) while the alarm stayed short; such samples are walls but cannot confirm overhead evidence alone
    (version 3's overhead_positive). One positive sample (expansion above the path) confirms with them."""
    def hill_then_blind(t):
        return dict(ttc=1.1, below=.9, lower=.4) if t < .25 else dict(ttc=.9)
    blind = TtcClearanceGovernor(TTC, ceiling=GUARD)
    rows = run(blind, hill_then_blind, .7)
    assert blind.counts['overhead_samples'] >= 2 and blind.counts['overhead_engagements'] == 0
    assert all(r['vcap'] is None for r in rows)

    def hill_then_blind_and_above(t):
        if t < .25:
            return dict(ttc=1.1, below=.9, lower=.4)
        return dict(ttc=.9, below=.1, lower=3.) if t < .25+FRAME else dict(ttc=.9)
    seen = TtcClearanceGovernor(TTC, ceiling=GUARD)
    rows = run(seen, hill_then_blind_and_above, .7)
    assert seen.counts['overhead_engagements'] == 1 and rows[-1]['vcap'] == GUARD.vertical_cap


def test_explained_samples_hold_only_a_weak_climb_and_not_beyond_its_height_bound():
    def mound_then_lower_window_only(t):
        if t < .3:
            return dict(ttc=.85, below=.9, lower=.4)                          # a full climb from below the path
        return dict(ttc=1., lower=.6)                                          # no vertical fraction; lower explains it
    plain, guarded = TtcClearanceGovernor(TTC), TtcClearanceGovernor(TTC, ceiling=GUARD)
    a = run(plain, mound_then_lower_window_only, 2.5)
    b = run(guarded, mound_then_lower_window_only, 2.5)
    full = max(r['climb'] for r in a)
    assert full > 3. and max(r['climb'] for r in b) == pytest.approx(full)
    assert min(series(a, 'climb')[100:]) == pytest.approx(full)                # unguarded: held at the full climb
    weak = series(b, 'climb')[190:]             # after the hold and the release from the full climb
    assert max(weak) <= GUARD.weak_climb+1e-9 and min(weak) > .5              # guarded: decays to the weak level
    assert guarded.counts['weak_climb_samples'] > 0 and guarded.counts['overhead_engagements'] == 0
    # climbing for real: a small below-path climb, then only lower-window evidence asking for a fast climb. The weak
    # climb is bounded to weak_climb and ends weak_climb_max_m above the last below-path request (plus its hold and
    # release); without the guard the same samples climb at up to vertical_up until climb_max_m.
    def small_mound_then_lower_window_only(t):
        return dict(ttc=1.1, below=.9, lower=.5) if t < .2 else dict(ttc=.95, lower=.6)
    rows = {}
    for label, guard in (('plain', None), ('guarded', GUARD)):
        gov = TtcClearanceGovernor(TTC, ceiling=guard)
        rows[label] = run(gov, small_mound_then_lower_window_only, 4., follow=True)
    top = max(r['z'] for r in rows['guarded'] if r['t'] <= .2+.08+FRAME)     # where the last below-path sample acted
    bound = top+GUARD.weak_climb_max_m+GUARD.weak_climb*TTC.climb_hold_s+GUARD.weak_climb**2/(2*TTC.climb_release)
    assert max(r['climb'] for r in rows['guarded'] if r['t'] > 1.1) <= GUARD.weak_climb+1e-9    # after its hold
    assert rows['guarded'][-1]['climb'] == 0. and rows['guarded'][-1]['z'] <= bound+.05
    assert max(r['climb'] for r in rows['plain']) > 2.5 and rows['plain'][-1]['z'] > bound+.5


def test_overhead_evidence_needs_a_climb_and_confirmation():
    def level(t):
        return dict(ttc=.8, below=.1, lower=3.)
    gov = TtcClearanceGovernor(TTC, ceiling=GUARD)
    rows = run(gov, level, .8)
    assert all(r['vcap'] is None for r in rows) and gov.counts['overhead_samples'] == 0   # level flight: no guard

    def one_overhead(t):
        if t < .3:
            return dict(ttc=.9, below=.9, lower=.4)
        return dict(ttc=.8, below=.1, lower=3.) if t < .3+FRAME else dict(ttc=5., below=.5, lower=5.)
    gov = TtcClearanceGovernor(TTC, ceiling=GUARD)
    rows = run(gov, one_overhead, 1.)
    assert gov.counts['overhead_samples'] == 1 and gov.counts['overhead_engagements'] == 0
    assert all(r['vcap'] is None for r in rows)

    def two_overhead_then_terrain(t):
        if t < .3:
            return dict(ttc=.9, below=.9, lower=.4)
        if t < .3+2*FRAME:
            return dict(ttc=.8, below=.1, lower=3.)
        return dict(ttc=.7, below=.9, lower=.4)                                # below-path again, inside the hold
    gov = TtcClearanceGovernor(TTC, ceiling=GUARD)
    rows = run(gov, two_overhead_then_terrain, 2.)
    cut = next(r['t'] for r in rows if r['vcap'] is not None)
    assert .3+FRAME+.08 <= cut <= .3+2*FRAME+.1                               # at the second overhead sample
    assert all(r['climb'] == 0. for r in rows if cut <= r['t'] <= cut+GUARD.hold_s-.05)
    assert gov.counts['suppressed_climb_samples'] > 0 and gov.cap is not None   # braked for instead
    released = [r for r in rows if r['t'] > cut+GUARD.hold_s+.1]
    assert released[0]['vcap'] is None and max(r['climb'] for r in released) > 1.   # the hold ends


def test_below_path_climbs_are_unchanged_by_the_guard():
    """A Pine-mound shape (below-path samples with a short lower TTC, a long unexplained alarm while climbing, then
    an alarm-free crest with expansion above the path far away): the same climb and cap at every tick."""
    def mound(t):
        if t < .8:
            return dict(ttc=1.3-.6*t, below=.85+.1*np.sin(9*t), lower=.4)
        if t < 1.:
            return dict(ttc=2.)                                                # unexplained but long: no alarm
        return dict(ttc=4.5, below=0., lower=None)
    a = run(TtcClearanceGovernor(TTC), mound, 3., follow=True)
    b = run(TtcClearanceGovernor(TTC, ceiling=GUARD), mound, 3., follow=True)
    for key in ('climb', 'cap', 'z'):
        np.testing.assert_array_equal(series(a, key), series(b, key))
    assert max(series(a, 'climb')) > 3.


# ---------------------------------------------------------------------------------------------
# Ceiling guard in the pilot (applied and shadow)
# ---------------------------------------------------------------------------------------------
def climbing_pilot(**kw):
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **kw)
    drive(pilot, history, AHEAD, 200, velocity=(6., 0., 0.), height=1.)
    now, commands = 12., []
    for k in range(150):
        t = now+k*.01
        clearance = None
        if k % 5 == 0:
            s = ceiling_after_a_small_climb(k*.01) if k >= 50 else dict(ttc=.85, below=.9, lower=.4)
            clearance = dict(time=t-.08, ttc=s['ttc'], distance=s['ttc']*6., below_fraction=s.get('below'),
                             ttc_lower=s.get('lower'))
        state = senses(position=(0., 0., 1.), velocity=(6., 0., .8 if k >= 20 else 0.))
        history.append(t, [0., 0., 1.], state['quat'][0].numpy())
        pilot.update(state, [0., 0., 0.], dict(race_cue=dict(AHEAD)), t-.05, t, clearance=clearance)
        commands.append((t, pilot.velocity_command.copy(), pilot.wall_log()))
    return pilot, commands


def test_the_pilot_levels_off_fast_under_overhead_evidence_and_shadow_flies_the_unguarded_pilot():
    plain, flown = climbing_pilot()
    guarded, applied = climbing_pilot(ceiling_guard=GUARD)
    _, shadow = climbing_pilot(ceiling_guard=GUARD, wall_apply=False)
    vz_plain = np.array([c[1][2] for c in flown])
    vz = np.array([c[1][2] for c in applied])
    assert vz_plain.max() > 3. and vz_plain[-1] > 3.                            # the old rule keeps climbing
    cut = next(i for i, c in enumerate(applied) if c[2]['ceiling_status'] == 'overhead')
    assert vz[cut-1] > 1.
    assert vz[cut+int(np.ceil(vz[cut-1]/(GUARD.vertical_slew*.01)))+1] <= GUARD.vertical_cap+1e-9   # at 15 m/s^2
    assert np.all(np.diff(vz[cut:]) >= -GUARD.vertical_slew*.01-1e-9)
    assert np.all(vz[cut+30:] <= GUARD.vertical_cap+1e-9)
    # shadow: the flown commands are the unguarded pilot's, bit for bit; the log shows what the guard would do
    np.testing.assert_array_equal(np.array([c[1] for c in shadow]), np.array([c[1] for c in flown]))
    assert shadow[cut][2]['ceiling_status'] == 'overhead' and shadow[cut][2]['ceiling_vertical_cap'] == 0.
    assert np.isnan(flown[-1][2]['ceiling_climb']) and flown[-1][2]['ceiling_status'] == ''
    meta = json.loads(json.dumps(guarded.metadata()))['wall_pilot']
    assert meta['version'] == WALL_PILOT_VERSION and meta['applied'] is True and meta['turn_first'] is None
    assert meta['ceiling_guard']['parameters'] == asdict(GUARD)
    assert meta['ceiling_guard']['counts']['overhead_engagements'] == 1
    assert json.loads(json.dumps(plain.metadata()))['wall_pilot'] is None


# ---------------------------------------------------------------------------------------------
# Turn first (pilot)
# ---------------------------------------------------------------------------------------------
LEFT_EDGE = dict(u=.01, v=.5, edge=True, aim_u=.01)


def standoff_pilot(**kw):
    """A pilot that braked to a stand-off at a wall ahead (+x): wall samples at 18 Hz while slowing from 6 m/s."""
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **kw)
    drive(pilot, history, AHEAD, 200, velocity=(6., 0., 0.), height=2.)
    now, speed = 12., 6.
    for k in range(120):
        t = now+k*.01
        clearance = None
        if k % 5 == 0 and speed > 1.5:
            clearance = dict(time=t-.08, ttc=.45, distance=.45*speed, below_fraction=.5, ttc_lower=.45)
        state = senses(position=(0., 0., 2.), velocity=(speed, 0., 0.))
        history.append(t, [0., 0., 2.], state['quat'][0].numpy())
        pilot.update(state, [0., 0., 0.], dict(race_cue=dict(AHEAD)), t-.05, t, clearance=clearance)
        speed = max(.3, speed-8.*.01)
    return pilot, history, now+1.2, speed


def step(pilot, history, t, cue, *, velocity=(.3, 0., 0.), yaw=0.):
    state = senses(position=(0., 0., 2.), velocity=velocity, yaw=yaw)
    history.append(t, [0., 0., 2.], state['quat'][0].numpy())
    pilot.update(state, [0., 0., 0.], None if cue is None else dict(race_cue=dict(cue)), t-.05, t)
    return pilot.velocity_command.copy()


def test_side_clamp_at_a_stand_off_turns_before_translating():
    plain, history_plain, t0, _ = standoff_pilot()
    turn, history, _, _ = standoff_pilot(turn_first=TURN)
    assert plain.clearance.status == 'standoff' and turn.clearance.status == 'standoff'
    ray = np.asarray(turn.clearance.cap_ray)[:2]
    for k in range(40):
        a = step(plain, history_plain, t0+k*.01, LEFT_EDGE)
        b = step(turn, history, t0+k*.01, LEFT_EDGE)
    assert plain.state == turn.state == 'side' and turn.turn_first_active and not plain.turn_first_active
    assert np.linalg.norm(a[:2]) > 2. and a[:2] @ ray > .5                    # the side rule pushes toward the wall
    assert np.linalg.norm(b[:2]) <= TURN.creep_speed+1e-6 and b[:2] @ ray <= 1e-9
    assert turn.metadata()['wall_pilot']['turn_first']['counts']['episodes'] == 1
    # the yaw rule is unchanged: both turn toward the clamped edge at the same rate
    assert turn.pilot.sight_yaw == pytest.approx(plain.pilot.sight_yaw)


def test_turn_first_releases_inside_the_cone_and_times_out_otherwise():
    pilot, history, t0, _ = standoff_pilot(turn_first=TURN)
    for k in range(20):
        step(pilot, history, t0+k*.01, LEFT_EDGE)
    assert pilot.turn_first_active
    # the drone has yawed left by 60 deg: the ring (60 deg left of the old heading, 20 m away) is 0 deg off
    yaw = np.radians(60.)
    ring = cue_toward([20., 0., 0.])
    for k in range(20, 60):
        command = step(pilot, history, t0+k*.01, ring, yaw=yaw, velocity=(.3*np.cos(yaw), .3*np.sin(yaw), 0.))
    assert not pilot.turn_first_active and pilot.turn_first_counts['aligned'] == 1
    assert np.linalg.norm(command[:2]) > TURN.creep_speed                       # normal speed resumes

    pilot, history, t0, _ = standoff_pilot(turn_first=TURN)
    active = []
    for k in range(int((TURN.max_s+TURN.rearm_s+.5)/.01)):
        step(pilot, history, t0+k*.01, LEFT_EDGE)
        active.append(pilot.turn_first_active)
    active = np.array(active)
    assert active[:int(TURN.max_s/.01)-1].all() and not active[int(TURN.max_s/.01)+1]
    assert pilot.turn_first_counts['timeout'] == 1
    # the stand-off memory (standoff_s) has lapsed by the end of the rearm: no new episode without a new wall
    assert not active[int((TURN.max_s+TURN.rearm_s)/.01)+5:].any() and pilot.turn_first_counts['episodes'] == 1


def test_no_episode_without_a_wall_or_with_the_checkpoint_ahead():
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., turn_first=TURN)
    rows = drive(pilot, history, LEFT_EDGE, 150, velocity=(1., 0., 0.))            # no clearance evidence at all
    assert rows[-1][1] == 'side' and not pilot.turn_first_active
    pilot, history, t0, _ = standoff_pilot(turn_first=TURN)
    for k in range(30):
        step(pilot, history, t0+k*.01, AHEAD)                                      # at a wall, ring straight ahead
    assert pilot.turn_first_counts['episodes'] == 0


def test_shadow_turn_first_is_logged_and_not_flown():
    plain, history_plain, t0, _ = standoff_pilot()
    shadow, history, _, _ = standoff_pilot(turn_first=TURN, wall_apply=False)
    for k in range(40):
        a = step(plain, history_plain, t0+k*.01, LEFT_EDGE)
        b = step(shadow, history, t0+k*.01, LEFT_EDGE)
        np.testing.assert_array_equal(a, b)
    assert shadow.turn_first_active and shadow.wall_log()['turn_first'] == 1.
    meta = json.loads(json.dumps(shadow.metadata()))['wall_pilot']
    assert meta['applied'] is False and meta['turn_first']['counts']['episodes'] == 1
    assert meta['ceiling_guard'] is None


def test_runner_log_columns_and_refusals():
    from haltere.liftoff.visual_brain import WALL_COLUMNS, VisualController, wall_row
    assert len(wall_row(None)) == len(WALL_COLUMNS) == 4
    row = dict(zip(WALL_COLUMNS, wall_row(FastRaceCue(SENSOR, CameraPoseHistory(), 6., turn_first=TURN,
                                                       ceiling_guard=GUARD))))
    assert row['turn_first'] == 0. and row['ceiling_status'] == '' and np.isnan(row['ceiling_climb'])
    assert np.isnan(dict(zip(WALL_COLUMNS, wall_row(FastRaceCue(SENSOR, CameraPoseHistory(), 6.))))['turn_first'])
    with pytest.raises(ValueError, match='fast pilot'):
        VisualController('missing.pt', 'missing.json', 'cpu', pilot_assistance='race-cue', wall_pilot='x.json')


def surrogate_hairpin(turn_first, *, seconds=5., start=10., a_back=1.5, b_back=6., b_side=-3., speed=6.):
    """Measured surrogate + FastMotorPD at `speed` toward a wall `start` m ahead. Checkpoint A lies `a_back` m before
    the wall on the line; once the drone passes it the HUD marker switches to checkpoint B, `b_back` m before the
    wall and `b_side` m to the side (-3: right): a hairpin, B is behind the drone at the wall. Perfect TTC toward the
    wall (along +x) at 18 Hz, 0.1 s late; ring marker 0.06 s late. Returns the closest gap to the wall, the closest
    horizontal distance to B after the switch and when it came within 1 m, the along-wall (x) request while
    turn-first is active, and the pilot."""
    from haltere.brain.motor_baseline import FastMotorPD
    from haltere.liftoff.fast_rehearsal import hud_marker
    from haltere.sim.identified import IdentifiedSim
    from haltere.vision.camera import Camera
    from tests.test_fast_motor_pd import CAL, load_profile
    profile = load_profile()
    sim = IdentifiedSim(profile, CAL)
    state = sim.hover(1, 3.)
    state.quad.vel[0] = torch.tensor([speed, 0., 0.])
    camera = Camera(320, 180, SENSOR['focal_320'], SENSOR['tilt_deg'])
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., turn_first=turn_first)
    pilot.launching = False
    motor = FastMotorPD(profile, CAL)
    queue = deque(motor.command(sim.sensors(state), torch.zeros(1, 3)).clone() for _ in range(3))
    motor.reset()
    p0 = state.quad.pos[0].numpy().astype(float)
    x_wall = p0[0]+start
    ring_a = np.array([x_wall-a_back, p0[1], p0[2]])
    ring_b = np.array([x_wall-b_back, p0[1]+b_side, p0[2]])
    ring, pending, cues, next_frame, detection, capture = ring_a, deque(), deque(), 0., None, None
    gap, reached, along, t_ring = np.inf, np.inf, [], None
    for k in range(int(seconds/.01)):
        now = k*.01
        position = state.quad.pos[0].numpy().astype(float)
        velocity = state.quad.vel[0].numpy().astype(float)
        quaternion = state.quad.quat[0].numpy().astype(float)
        history.append(now, position, quaternion)
        if ring is ring_a and position[0] >= ring_a[0]:
            ring = ring_b                                                       # passed A: the HUD shows B
        if now >= next_frame:
            cue = hud_marker(camera, ring, position, quaternion)
            cue['aim_u'] = cue['u']
            cues.append((now+.06, now, cue))
            forward = velocity[0]
            ttc = (x_wall-position[0])/forward if forward > 1.5 else None
            pending.append((now+.1, dict(time=now, ttc=ttc, distance=None if ttc is None
                                         else ttc*np.linalg.norm(velocity), below_fraction=.5, ttc_lower=ttc)))
            next_frame = now+FRAME
        while cues and cues[0][0] <= now:
            _, capture, cue = cues.popleft()
            detection = dict(race_cue=cue)
        sample = None
        while pending and pending[0][0] <= now:
            _, sample = pending.popleft()
        sensors = sim.sensors(state)
        pilot.update(sensors, sensors['gyro'][0].numpy(), detection, capture, now, clearance=sample)
        if pilot.turn_first_active:
            along.append(float(pilot.velocity_command[0]))
        action = motor.command(sensors, torch.tensor(pilot.velocity_command, dtype=torch.float32)[None],
                               torch.tensor(pilot.feedforward, dtype=torch.float32)[None])
        queue.append(torch.tensor(pilot.command(action[0].numpy()), dtype=torch.float32)[None])
        state = sim.step(state, queue.popleft())
        gap = min(gap, x_wall-float(state.quad.pos[0, 0]))
        if ring is ring_b:
            reached = min(reached, float(np.linalg.norm(state.quad.pos[0, :2].numpy()-ring_b[:2])))
            t_ring = now if t_ring is None and reached < 1. else t_ring
    return dict(gap=gap, ring=reached, t_ring=t_ring, along=along, pilot=pilot)


def test_measured_surrogate_hairpin_turns_first_and_still_reaches_the_next_ring():
    """With a perfect TTC and the fast PD the surrogate does not reproduce the live side push into the wall (the
    plain pilot also stays off it here); this pins that turn-first engages, stops requesting speed toward the wall,
    releases inside its cone and costs little time. The live-log replay is the evidence for the push itself."""
    for geometry in (dict(), dict(a_back=1., b_back=5.), dict(a_back=2., b_side=3.)):
        plain, turn = surrogate_hairpin(None, **geometry), surrogate_hairpin(TURN, **geometry)
        counts = turn['pilot'].turn_first_counts
        assert counts['episodes'] >= 1 and counts['aligned'] >= 1 and counts['timeout'] == 0
        along = np.array(turn['along'])
        assert along[0] <= 2.5 and np.all(along[30:] <= 1e-6)                 # no speed toward the wall after 0.3 s
        assert turn['gap'] > .3 and turn['gap'] >= plain['gap']-.02            # no contact
        assert turn['ring'] < 1. and turn['t_ring']-plain['t_ring'] < 1.       # reaches B, at a bounded time cost
