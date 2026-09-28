"""Round 5 (safety): the ceiling guard's any_climb (wall-pilot declaration version 6) and contact support version 3
(descent-view declaration version 3) with the runner's --contact-support on|off|shadow.

These tests pin the declarations and their frozen gates (configs/pilot/safety_gates.json), the ceiling cut of every
climb source but the pilot's own climb toward the ring in view, the contact rule's manoeuvre exclusion (a hard pitch or
a hard horizontal brake is not a ground reaction), its gain learnt before arming (a drone 15% above the thrust curve does
not fire at arming), that its climb never ends turn-first, that shadow changes nothing, and that the version-2 and
default pilots stay bit-identical to m4b (golden digests from a git archive of 2a5bccb). The gates themselves are scored
outside the test suite. None of this is flight evidence.
"""
import hashlib
import json
from argparse import Namespace
from dataclasses import asdict, replace

import numpy as np
import pytest

from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import (DESCENT_VIEW_VERSION, WALL_PILOT_VERSION, CeilingGuardConfig,
                                           ContactSupportConfig, DescentViewConfig, FastRaceCue,
                                           contact_support_config, descent_view_config, wall_pilot_configs)
from tests.test_fast_race_cue import BELOW, cue_toward, single_thread  # noqa: F401 (fixture)
from tests.test_visual_assistance import SENSOR, senses

CAL = dict(hover_processed=0.13566076755523682, throttle_scale=0.8, hover_stick_sim=-0.43019758448357537)
DV = DescentViewConfig()
V2 = ContactSupportConfig()
AHEAD = cue_toward([20., 0., 0.])
G = 9.81


def declarations():
    from haltere.liftoff.visual_brain import (DESCENT_VIEW_DECLARATION, WALL_PILOT_DECLARATION, load_descent_view,
                                              load_wall_pilot)
    return load_wall_pilot(WALL_PILOT_DECLARATION)[0], load_descent_view(DESCENT_VIEW_DECLARATION)[0]


def v3():
    return contact_support_config(declarations()[1])


def brain_throttle(drive):
    return CAL['hover_stick_sim']+(2*drive-1-CAL['hover_processed'])/CAL['throttle_scale']


def thrust(drive, cs=V2):
    return G*cs.thrust_twr*drive**cs.thrust_exponent


HOVER_DRIVE = float((1/V2.thrust_twr)**(1/V2.thrust_exponent))


def fly(pilot, seconds, *, cue_of, drive_of, vz_of, speed_of=lambda t: 3., omega_of=lambda t: (0., 0., 0.), start=10.):
    """Level attitude; horizontal velocity (speed_of(t), 0, 0); drive_of(t) the drive the motor issues; vz_of(t, vz,
    drive) the measured vertical speed of the next tick; omega_of(t) the body rates the pilot receives. Rows: (t,
    state, command, fired)."""
    rows, vz, x = [], 0., 0.
    for k in range(int(round(seconds/.01))):
        t = start+k*.01
        speed = speed_of(t)
        x += speed*.01
        s = senses(position=(x, 0., 5.), velocity=(speed, 0., vz))
        pilot.pose_history.append(t, [x, 0., 5.], s['quat'][0].numpy())
        cue = cue_of(t)
        pilot.update(s, list(omega_of(t)), dict(race_cue=dict(cue)) if cue else None, t-.05, t)
        drive = drive_of(t)
        pilot.command(np.array([brain_throttle(drive), 0., 0., 0.]))
        rows.append((t, pilot.state, pilot.velocity_command.copy(), bool(pilot.contact_fired)))
        vz = vz_of(t, vz, drive)
    return rows


def pilot_with(contact, **kw):
    return FastRaceCue(SENSOR, CameraPoseHistory(), 6., reference_speed=6., calibration=CAL, descent_view=DV,
                       contact_support=contact, **kw)


def resting(t, vz, drive):
    return 0.


def free_air(t, vz, drive):
    return vz+(thrust(drive)-G-V2.body_drag_s_inv[2]*vz)*.01


# ---------------------------------------------------------------------------------------------
# Declarations and gates
# ---------------------------------------------------------------------------------------------
def test_round5_declarations_are_frozen_and_the_runner_flies_only_them():
    from haltere.liftoff.visual_brain import (DESCENT_VIEW_DECLARATION, WALL_PILOT_DECLARATION, load_descent_view,
                                              load_wall_pilot)
    wall, view = declarations()
    assert WALL_PILOT_VERSION == wall['version'] == 6 and DESCENT_VIEW_VERSION == view['version'] == 3
    assert wall_pilot_configs(wall)['ceiling_guard'] == CeilingGuardConfig(any_climb=True)
    cs = contact_support_config(view)
    assert cs.version == 3 and cs.turn_first_handoff is False and cs.learn_after_s == 3. and cs.arm_learn_s == .5
    assert cs.max_body_rate == 2.5 and cs.max_braking == 4.
    # version 3 keeps version 2's values and the view rule
    assert replace(cs, version=2, max_body_rate=None, max_braking=None, learn_after_s=None, arm_learn_s=0.,
                   turn_first_handoff=True) == V2
    assert descent_view_config(view) == DV
    for path, loader in ((WALL_PILOT_DECLARATION.with_name('wall_pilot_v5.json'), load_wall_pilot),
                         (DESCENT_VIEW_DECLARATION.with_name('descent_view_v2.json'), load_descent_view)):
        with pytest.raises(ValueError, match='version'):
            loader(path)
    kept = json.loads(DESCENT_VIEW_DECLARATION.with_name('descent_view_v2.json').read_text(encoding='utf-8'))
    assert contact_support_config(kept) == V2


def test_contact_support_versions_are_validated():
    for bad in (dict(max_body_rate=1.), dict(learn_after_s=3.), dict(arm_learn_s=.5), dict(turn_first_handoff=False),
                dict(version=4), dict(version=3), dict(version=3, max_body_rate=1., max_braking=1., learn_after_s=3.),
                dict(version=3, max_body_rate=-1., max_braking=1., learn_after_s=3., arm_learn_s=.5)):
        with pytest.raises(ValueError):
            ContactSupportConfig(**bad)
    _, view = declarations()
    missing = dict(view, contact_support={k: v for k, v in view['contact_support'].items() if k != 'max_braking'})
    with pytest.raises(ValueError, match='max_braking'):
        contact_support_config(missing)
    with pytest.raises(ValueError, match='version'):
        contact_support_config(dict(view, contact_support=dict(view['contact_support'], version=3)))
    with pytest.raises(ValueError, match='shadow'):
        pilot_with(V2, contact_apply=False)


def test_safety_gates_are_frozen_and_name_the_declarations():
    from haltere.liftoff.safety_gates import GATES_PATH, load_gates, replay_tag
    gates, digest = load_gates(GATES_PATH)
    wall, view = declarations()
    assert gates['version'] == 1 and digest == gates['sha256']
    assert gates['declarations']['wall_pilot']['sha256'] == wall['sha256']
    assert gates['declarations']['descent_view']['sha256'] == view['sha256']
    assert gates['held_out']['logs'] == ['minus-fast6-r4b-01', 'minus-brain11cw13-r4b-noassist-01',
                                         'straw-brain11cw13-r4b-noassist-02']
    g = gates['gates']
    assert g['SG_Contact_Arming']['sim_seeds'] == [101, 102] and g['SG_Contact_Rest']['max_delay_s'] == .8
    assert g['SG_Contact_Surrogate']['sets'] == ['flat:3100-3103', 'steep:3100-3103', 'hill:6200-6207']
    assert replay_tag(gates, 'stack_dv3_cthr', 'straw-brain08-04') == 'on-won-von-stream-nop-cthr-dv3'
    assert replay_tag(gates, 'stack_dv3off_cthr_wp5', 'minus-fast6-r4b-01') == 'on-won-von-nop-cthr-dv3-csoff-wp5'


# ---------------------------------------------------------------------------------------------
# Default behaviour and version 2 are m4b's
# ---------------------------------------------------------------------------------------------
def test_default_view_and_version_2_pilots_are_bit_identical_to_m4b():
    from tests.test_fast_race_cue_descent_view import golden_scenario
    digest = lambda trace: hashlib.sha256(np.ascontiguousarray(trace.astype(np.float64)).tobytes()).hexdigest()  # noqa
    # from a git archive of m4b (2a5bccb)
    assert digest(golden_scenario()[0]) == '67d31f2892c3f292871a40e6cb78e8a734ae540636aa8a3508a1ce3572ad3d1f'
    assert digest(golden_scenario(descent_view=DV)[0]) == \
        'eae7b39aeac9c76db695012782f1f6903d1ccf9ddbbe033c032dc9554dc7914a'
    assert digest(golden_scenario(descent_view=DV, contact_support=V2)[0]) == \
        'bd8ec8869a540c5a333a40d9c6bf27a5970ac248ab29273af46e191ccff3dd06'


# ---------------------------------------------------------------------------------------------
# Ceiling guard any_climb (wall pilot version 6)
# ---------------------------------------------------------------------------------------------
def test_overhead_evidence_cuts_a_climb_that_is_not_the_governors():
    """The round-4b review's case: something above a path that rises at 1 m/s with no governor climb (a support,
    contact, sag or search climb) is cut by version 6, not by version 5; the surface below (a sample the lower window
    explains) cuts nothing; during a governor climb both versions act alike."""
    from haltere.liftoff.safety_gates import ceiling_governor_case
    v5, v6 = CeilingGuardConfig(), CeilingGuardConfig(any_climb=True)
    for kw in (dict(below=.1), dict(below=None, lower=1.23)):
        assert ceiling_governor_case(v5, terrain_first=False, **kw)['engagements'] == 0
        cut = ceiling_governor_case(v6, terrain_first=False, **kw)
        assert cut['engagements'] == 1 and cut['first_bound_s'] <= .1
        assert ceiling_governor_case(v6, terrain_first=True, **kw) == ceiling_governor_case(v5, terrain_first=True, **kw)
    assert ceiling_governor_case(v6, terrain_first=False, below=None, lower=.2)['engagements'] == 0


def test_the_pilot_levels_off_every_climb_source_but_its_own_ring_climb():
    from haltere.liftoff.safety_gates import ceiling_pilot_case
    from haltere.liftoff.visual_brain import WALL_PILOT_DECLARATION
    wall, _ = declarations()
    v5 = json.loads(WALL_PILOT_DECLARATION.with_name('wall_pilot_v5.json').read_text(encoding='utf-8'))
    for source in ('support', 'contact', 'search'):
        cut, old = ceiling_pilot_case(wall, source), ceiling_pilot_case(v5, source)
        assert cut['engagements'] == 1 and cut['level_after_s'] <= .3 and cut['max_request_after_level'] <= .05
        assert old['engagements'] == 0 and old['max_request_before'] >= .5
        # version 5 keeps the climb until it ends by itself (the 0.6 s support climb, the 1.5 s search climb)
        assert old['level_after_s'] is None or old['level_after_s'] > cut['level_after_s']+.2
    own = ceiling_pilot_case(wall, 'pilot')
    assert own['engagements'] == 0 and own['level_after_s'] is None and own['max_request_before'] > .5


def test_the_ceiling_guard_metadata_names_the_version():
    configs = wall_pilot_configs(declarations()[0])
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., **configs)
    meta = pilot.metadata()['wall_pilot']
    assert meta['version'] == 6 and meta['ceiling_guard']['parameters']['any_climb'] is True
    assert 'any_climb' in meta['ceiling_guard']['rule']
    from haltere.liftoff.visual_brain import WALL_PILOT_DECLARATION
    v5 = json.loads(WALL_PILOT_DECLARATION.with_name('wall_pilot_v5.json').read_text(encoding='utf-8'))
    assert FastRaceCue(SENSOR, CameraPoseHistory(), 6., **wall_pilot_configs(v5)).metadata()['wall_pilot']['version'] == 5


# ---------------------------------------------------------------------------------------------
# Contact support version 3
# ---------------------------------------------------------------------------------------------
def rest_after(pilot, rest, *, omega_of=lambda t: (0., 0., 0.), speed_of=lambda t: 3., seconds=None):
    """Level free flight at hover thrust with the ring ahead until `rest`, then resting on a surface (vz 0) at a drive of
    0.5 with the ring clipped below: the review's resting case."""
    return fly(pilot, (rest-10.)+2. if seconds is None else seconds, cue_of=lambda t: AHEAD if t < rest else BELOW,
               drive_of=lambda t: HOVER_DRIVE if t < rest else .5,
               vz_of=lambda t, vz, drive: 0. if t >= rest else free_air(t, vz, drive), omega_of=omega_of,
               speed_of=speed_of)


def test_version_3_arms_after_its_gain_learning_and_detects_a_rest_within_0_8_s():
    pilot = pilot_with(v3())
    rows = rest_after(pilot, 15.)
    fired = [t for t, *_, f in rows if f]
    assert fired and 15. < fired[0] <= 15.8
    summary = pilot.contact_summary()
    assert summary['applied'] is True and 3.75 <= summary['armed_at_s'] <= 4.
    assert summary['prearm_gain'] == pytest.approx(1., abs=.02) and summary['seconds']['prearm_learning'] >= .5


def test_a_rest_before_arming_is_not_learnt_as_thrust_gain_while_the_drone_is_not_rising():
    """A rest that begins during the learning (3.5 s: 0.2 s of windows) is neither quiet nor rising: it is not learnt,
    the rule does not arm during it, and the gain stays the free-air one."""
    pilot = pilot_with(v3())
    rest_after(pilot, 13.5, seconds=5.)
    assert pilot.contact_armed_at is None and pilot.contact_gain == pytest.approx(1., abs=.02)


@pytest.mark.parametrize('rate', [3., 5.])
def test_a_hard_pitch_manoeuvre_is_not_read_as_ground_reaction(rate):
    """The same unexplained upward force while the body pitches at `rate` rad/s (a brake flare: the thrust of the mean
    drive underestimates a split motor set) starts no support climb in version 3; version 2 fires. At 1 rad/s (the
    minus-fast6-r4-02 floor touch) version 3 still fires."""
    rest = 15.
    pitch = lambda t: (0., rate, 0.) if t >= rest else (0., 0., 0.)  # noqa: E731
    old, new = pilot_with(V2), pilot_with(v3())
    assert any(f for *_, f in rest_after(old, rest, omega_of=pitch))
    assert not any(f for *_, f in rest_after(new, rest, omega_of=pitch))
    assert new.contact_time['excluded'] > 1.
    slow = pilot_with(v3())
    assert any(f for *_, f in rest_after(slow, rest, omega_of=lambda t: (1., 0., 0.) if t >= rest else (0., 0., 0.)))


def test_a_hard_horizontal_brake_is_not_read_as_ground_reaction():
    rest = 15.
    brake = lambda t: 6. if t < rest else max(.5, 6.-6.*(t-rest))  # noqa: E731   6 m/s^2 for about a second
    old, new = pilot_with(V2), pilot_with(v3())
    first_old = [t for t, *_, f in rest_after(old, rest, speed_of=brake) if f]
    first_new = [t for t, *_, f in rest_after(new, rest, speed_of=brake) if f]
    assert first_old and first_old[0] < rest+.5
    # version 3 waits until the brake has ended (about 0.9 s: the window's deceleration falls below 4 m/s^2)
    assert not first_new or first_new[0] > rest+.8


def test_a_drone_above_the_thrust_curve_does_not_fire_at_arming():
    """The round-4b integration case: a drone 15% above the curve, asked for a small sink while it still rises slowly
    after the launch, reads about 1.4 m/s^2 of unexplained force with the gain at 1 (the steady 0.2 m/s rise at that
    drive implies a gain of 1.16). Version 2 fires soon after arming (at 3.9 s here, once an older support climb from
    the sink request ends); version 3 learnt the gain from the rising windows before arming and does not."""
    strong_hover = float((1/(1.15*V2.thrust_twr))**(1/V2.thrust_exponent))
    rising = lambda t, vz, drive: .2  # noqa: E731
    runs = {}
    for name, cs in (('v2', V2), ('v3', v3())):
        pilot = pilot_with(cs)
        rows = fly(pilot, 6., cue_of=lambda t: BELOW, drive_of=lambda t: strong_hover, vz_of=rising)
        runs[name] = ([t-10. for t, *_, f in rows if f], pilot)
    assert runs['v2'][0] and runs['v2'][0][0] < 4.5
    assert not runs['v3'][0]
    implied = (G+V2.body_drag_s_inv[2]*.2)/thrust(strong_hover)
    assert runs['v3'][1].contact_gain == pytest.approx(implied, abs=.01) and runs['v3'][1].contact_armed_at >= 3.8


def test_shadow_changes_nothing_and_logs_where_the_rule_would_fire():
    rows_off = rest_after(pilot_with(None), 15.)
    shadow_pilot = pilot_with(v3(), contact_apply=False)
    rows_shadow = rest_after(shadow_pilot, 15.)
    rows_on = rest_after(pilot_with(v3()), 15.)
    np.testing.assert_array_equal(np.array([r[2] for r in rows_shadow]), np.array([r[2] for r in rows_off]))
    assert [r[1] for r in rows_shadow] == [r[1] for r in rows_off]
    marks = [t for t, *_, f in rows_shadow if f]
    fires = [t for t, *_, f in rows_on if f]
    assert marks and fires and marks[0] == fires[0] and 'support_climb' not in {r[1] for r in rows_shadow}
    assert shadow_pilot.contact_summary()['applied'] is False and shadow_pilot.climb_until is None
    log = shadow_pilot.contact_log()
    assert set(log) == {'contact_unexplained', 'contact_gain', 'contact_fire', 'contact_armed', 'contact_excluded'}


def test_a_contact_support_climb_never_ends_turn_first():
    from haltere.liftoff.safety_gates import turn_first_case
    wall, _ = declarations()
    new = turn_first_case(wall, v3(), 'contact')
    assert new['engaged_before'] and new['active_through_climb'] and new['max_toward_wall_mps'] <= .05
    assert new['ends']['handoff'] == 0
    for contact, source in ((v3(), 'support'), (V2, 'contact')):
        old = turn_first_case(wall, contact, source)
        assert old['ends']['handoff'] == 1 and not old['active_through_climb'] and old['max_toward_wall_mps'] > .5


# ---------------------------------------------------------------------------------------------
# Runner, deployed pilot and replay harness
# ---------------------------------------------------------------------------------------------
def test_the_runner_switch_columns_and_metadata():
    from haltere.liftoff.visual_brain import (CONTACT_COLUMNS, CONTACT_V3_COLUMNS, DESCENT_VIEW_COLUMNS,
                                              descent_view_columns, descent_view_row, resolve_contact_support)
    assert resolve_contact_support(Namespace(descent_view='on')) == 'on'
    assert resolve_contact_support(Namespace(descent_view='on', contact_support='shadow')) == 'shadow'
    with pytest.raises(ValueError, match='descent-view'):
        resolve_contact_support(Namespace(descent_view=None, contact_support='off'))
    pilot = pilot_with(v3())
    rest_after(pilot, 15., seconds=1.)
    assert descent_view_columns(pilot) == DESCENT_VIEW_COLUMNS+CONTACT_COLUMNS+CONTACT_V3_COLUMNS
    assert len(descent_view_row(pilot)) == 8
    assert descent_view_columns(pilot_with(V2)) == DESCENT_VIEW_COLUMNS+CONTACT_COLUMNS
    assert descent_view_columns(pilot_with(None)) == DESCENT_VIEW_COLUMNS
    meta = json.loads(json.dumps(pilot.metadata()))
    assert meta['descent_view']['version'] == 3 and meta['contact_support']['parameters']['version'] == 3
    assert 'version 3' in meta['contact_support']['rule'] and meta['contact_support']['applied'] is True
    assert pilot_with(V2).metadata()['descent_view']['version'] == 2


def test_the_deployed_pilot_follows_the_contact_support_switch():
    from haltere.train.deployed_pilot import deployed_pilot_kwargs
    on, record = deployed_pilot_kwargs('fast_velocity_pd_v1')
    assert on['contact_support'] == v3() and 'contact_apply' not in on and record['descent_view']['contact_support'] == 'on'
    assert on['ceiling_guard'].any_climb is True
    off, record = deployed_pilot_kwargs('fast_velocity_pd_v1', contact_support='off')
    assert 'contact_support' not in off and record['descent_view']['contact_support'] == 'off'
    shadow, _ = deployed_pilot_kwargs('fast_velocity_brain_v1', contact_support='shadow')
    assert shadow['contact_apply'] is False and shadow['contact_support'] == v3()
    with pytest.raises(ValueError, match='contact-support'):
        deployed_pilot_kwargs('fast_velocity_pd_v1', contact_support='maybe')


def test_the_replay_harness_builds_the_contact_modes():
    from pathlib import Path
    from haltere.obstacles.vertical_replay import build
    side = dict(motor_controller=dict(contract='fast_velocity_pd_v1'), gate_sensor=SENSOR,
                pilot_assistance=dict(nominal_speed_mps=6., trained_motor_reference_mps=6.))
    tree = str(Path(__file__).resolve().parents[1])
    pilot, info = build(side, tree, 'on', descent_view=DV, contact_support=v3(), contact_apply=False)
    assert pilot.contact_apply is False and info['contact_support'] and pilot.ceiling_guard.any_climb
    pilot, _ = build(side, tree, 'on', descent_view=DV, contact_support=v3())
    assert pilot.contact_apply is True
    assert asdict(pilot.contact_support) == asdict(v3())
