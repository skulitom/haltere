"""Round 4b pilot rules: contact support (descent-view declaration version 2, `ContactSupportConfig`) and the
clearance brake's sink floor (wall-pilot declaration version 5, `ClearanceBrakeConfig`).

These tests pin the declarations and their frozen gates, the contact rule on synthetic states (a drone resting on a
surface with a small sink request gets a support climb; the same thrust in free air, where the drone sinks as the
thrust explains, gets none; arming; the free-air thrust gain), the sink floor on a rising looming ray (no descent
the pilot did not ask for, the horizontal part unchanged, nothing applied in shadow), that both are off by default
and that the default and version-1 pilots are bit-identical to m4 (golden digests taken from the m4 tree, 3decaac).
The gates themselves are scored outside the test suite. None of this is flight evidence.
"""
import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest

from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import (DESCENT_VIEW_VERSION, WALL_PILOT_VERSION, ClearanceBrakeConfig,
                                           ContactSupportConfig, DescentViewConfig, FastRaceCue,
                                           contact_support_config, descent_view_config, wall_pilot_configs)
from tests.test_fast_race_cue import BELOW, cue_toward, single_thread  # noqa: F401 (fixture)
from tests.test_visual_assistance import SENSOR, senses

CAL = dict(hover_processed=0.13566076755523682, throttle_scale=0.8, hover_stick_sim=-0.43019758448357537)
CS = ContactSupportConfig()
DV = DescentViewConfig()
AHEAD = cue_toward([20., 0., 0.])
G = 9.81


def brain_throttle(drive):
    """The brain-order throttle whose processed value gives `drive` = (processed + 1)/2 (the pad calibration)."""
    return CAL['hover_stick_sim']+(2*drive-1-CAL['hover_processed'])/CAL['throttle_scale']


def thrust(drive, cs=CS):
    return G*cs.thrust_twr*drive**cs.thrust_exponent


HOVER_DRIVE = float((1/CS.thrust_twr)**(1/CS.thrust_exponent))


def fly(pilot, seconds, *, cue_of, drive_of, vz_of=None, speed=3., start=10.):
    """Level attitude, horizontal velocity (speed, 0, 0). drive_of(t) -> the drive the motor issues this tick;
    vz_of(t, vz, drive) -> the measured vertical speed of the next tick (default: free air with the declared thrust
    curve and body drag, so the thrust explains every change). Rows: (t, state, command vz, fired)."""
    rows, vz = [], 0.
    for k in range(int(round(seconds/.01))):
        t = start+k*.01
        s = senses(position=(speed*(t-start), 0., 5.), velocity=(speed, 0., vz))
        pilot.pose_history.append(t, [speed*(t-start), 0., 5.], s['quat'][0].numpy())
        cue = cue_of(t)
        pilot.update(s, [0., 0., 0.], dict(race_cue=dict(cue)) if cue else None, t-.05, t)
        drive = drive_of(t)
        pilot.command(np.array([brain_throttle(drive), 0., 0., 0.]))
        rows.append((t, pilot.state, float(pilot.velocity_command[2]), bool(pilot.contact_fired)))
        if vz_of is None:
            vz = vz+(thrust(drive)-G-CS.body_drag_s_inv[2]*vz)*.01
        else:
            vz = vz_of(t, vz, drive)
    return rows


def contact_pilot(**kw):
    return FastRaceCue(SENSOR, CameraPoseHistory(), 6., reference_speed=6., calibration=CAL, descent_view=DV,
                       contact_support=CS, **kw)


# ---------------------------------------------------------------------------------------------
# Declarations and gates
# ---------------------------------------------------------------------------------------------
def test_contact_support_config_validation():
    for bad in (dict(thrust_twr=.9), dict(window_s=0.), dict(drive_min=.9), dict(drive_max=1.2),
                dict(gain_min=1.1), dict(gain_max=.9), dict(body_drag_s_inv=(0., 0.)), dict(hold_s=float('nan')),
                dict(body_drag_s_inv=(0., -1., 0.)), dict(sink_min=-.1)):
        with pytest.raises(ValueError):
            ContactSupportConfig(**bad)
    with pytest.raises(ValueError, match='ContactSupportConfig'):
        FastRaceCue(SENSOR, CameraPoseHistory(), 6., calibration=CAL, contact_support=dict())
    with pytest.raises(ValueError, match='calibration'):
        FastRaceCue(SENSOR, CameraPoseHistory(), 6., contact_support=CS)


def test_declaration_version_2_declares_the_rule_and_the_measured_thrust_curve():
    from haltere.liftoff.visual_brain import DESCENT_VIEW_DECLARATION
    from haltere.liftoff.visual_brain import lag_turn_declaration_sha256
    # version 2 is kept beside version 3 (round 5), which the runner flies (tests/test_round5_safety.py)
    declaration = json.loads(DESCENT_VIEW_DECLARATION.with_name('descent_view_v2.json').read_text(encoding='utf-8'))
    assert declaration['version'] == 2 and DESCENT_VIEW_VERSION == 3
    assert declaration['sha256'] == lag_turn_declaration_sha256(declaration)
    assert contact_support_config(declaration) == CS                 # the declared values are the defaults
    kept = json.loads(DESCENT_VIEW_DECLARATION.with_name('descent_view_v1.json').read_text(encoding='utf-8'))
    assert contact_support_config(kept) is None and descent_view_config(kept) == descent_view_config(declaration)
    profile = DESCENT_VIEW_DECLARATION.parents[2]/'runs'/'measured-dynamics-low-speed-20260923'/'profile.json'
    if profile.exists():
        p = json.loads(profile.read_text(encoding='utf-8'))
        assert CS.thrust_twr == p['thrust']['full_input_twr'] and CS.thrust_exponent == p['thrust']['exponent']
        assert list(CS.body_drag_s_inv) == p['translation_drag_s_inv']


def test_gates_are_frozen_and_name_descent_view_version_2():
    from haltere.liftoff.contact_support_eval import GATES_PATH, identity_pairs, load_gates, replay_tag
    gates, digest = load_gates(GATES_PATH)
    assert gates['version'] == 1 and digest == gates['sha256'] and gates['descent_view']['version'] == 2
    g = gates['gates']
    assert g['CS_Rest']['max_delay_s'] == .8 and g['CS_Rest']['speeds'] == [0., 1., 2., 3., 4., 5., 6.]
    assert set(g['CS_Rest']['motors']) == set(g['CS_Surrogate']['motors']) == {'pd', 'brain08', 'brain09b',
                                                                                 'brain10b'}
    assert g['CS_R402']['window'] == [27.4, 28.9] and g['CS_Clean']['max_outside'] == 0
    assert g['CS_R402']['tag'] == replay_tag('on', False, dv=2, cthr=True, wp=4)
    # the surrogate runs only gate seeds, which the design never ran (development: 5000-5007 and 6100-6111)
    assert g['CS_Surrogate']['sets'] == ['flat:3000-3007', 'steep:3000-3007', 'hill:6000-6011']
    pairs = identity_pairs(['straw-brain08-04'], {'straw-brain08-04'})
    assert pairs[2] == ('straw-brain08-04', 'on-won-von-stream-nop-wp4', 'on-won-von-stream-nop')
    assert pairs[0][1] == 'none-woff-voff-stream'


def test_runner_flies_version_2_and_logs_the_contact_columns():
    from haltere.liftoff.visual_brain import (CONTACT_COLUMNS, DESCENT_VIEW_COLUMNS, descent_view_columns,
                                              descent_view_row)
    pilot = contact_pilot()
    fly(pilot, .5, cue_of=lambda t: BELOW, drive_of=lambda t: HOVER_DRIVE)
    assert descent_view_columns(pilot) == DESCENT_VIEW_COLUMNS+CONTACT_COLUMNS
    row = descent_view_row(pilot)
    assert len(row) == 6 and row[4] == pytest.approx(1.) and row[5] == 0.
    plain = FastRaceCue(SENSOR, CameraPoseHistory(), 6., descent_view=DV)
    assert descent_view_columns(plain) == DESCENT_VIEW_COLUMNS and len(descent_view_row(plain)) == 3
    meta = pilot.metadata()
    assert meta['descent_view']['version'] == 2 and meta['contact_support']['parameters'] == asdict(CS)
    assert plain.metadata()['descent_view']['version'] == 1 and 'contact_support' not in plain.metadata()


# ---------------------------------------------------------------------------------------------
# Contact support
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize('speed', [0., 3., 6.])
def test_a_drone_resting_on_a_surface_with_a_small_sink_request_gets_a_support_climb(speed):
    """3.5 s of level free flight at hover thrust (ring ahead), then the ring clips at the bottom edge and the drone
    rests on a surface: the motor issues the thrust of drive 0.5 (below hover, as a motor that asks for the bounded
    sink does) and the measured vertical speed stays 0. The view bound keeps the sink request small, so the older
    support rules cannot fire; the contact rule fires within 0.8 s."""
    pilot = contact_pilot()
    rest = 13.5
    rows = fly(pilot, 5., speed=speed, cue_of=lambda t: AHEAD if t < rest else BELOW,
               drive_of=lambda t: HOVER_DRIVE if t < rest else .5,
               vz_of=lambda t, vz, drive: 0. if t >= rest else vz+(thrust(drive)-G)*.01)
    fired = [t for t, _, _, f in rows if f]
    assert fired and not [t for t in fired if t < rest]
    assert rest < fired[0] <= rest+.8
    assert min(c for t, _, c, _ in rows if rest <= t < fired[0]) > -.8        # the older rules needed < -0.8
    assert pilot.contact_summary()['onsets'] >= 1
    after = [s for t, s, _, _ in rows if fired[0] < t < fired[0]+.5]
    assert after and all(s == 'support_climb' for s in after)


def test_the_same_thrust_in_free_air_starts_no_support_climb():
    """The same sub-hover thrust while the drone sinks as the thrust explains (free air): no onset, and the thrust
    gain stays at the measured vehicle's."""
    pilot = contact_pilot()
    rows = fly(pilot, 6., speed=3., cue_of=lambda t: AHEAD if t < 13.5 else BELOW,
               drive_of=lambda t: HOVER_DRIVE if t < 13.5 else .5)
    assert not any(f for *_, f in rows)
    assert pilot.contact_gain == pytest.approx(1., abs=.02)


def test_not_armed_while_launching_or_before_arm_after_s():
    pilot = contact_pilot()
    rows = fly(pilot, CS.arm_after_s-.1, speed=0., cue_of=lambda t: BELOW, drive_of=lambda t: .5,
               vz_of=lambda t, vz, drive: 0.)
    assert not any(f for *_, f in rows)


def test_the_thrust_gain_learns_a_stronger_drone_in_free_air():
    """A drone 15% stronger than the declared curve hovers at a lower drive: in level free flight it would read as a
    steady upward force; the gain learns it from quiet and rising windows and no support climb starts."""
    pilot = contact_pilot()
    stronger = lambda drive: 1.15*thrust(drive)                    # noqa: E731
    hover = float((1/(1.15*CS.thrust_twr))**(1/CS.thrust_exponent))
    t_climb = 12.
    rows = fly(pilot, 8., speed=3., cue_of=lambda t: AHEAD if t < 15. else BELOW,
               drive_of=lambda t: .65 if 10.5 <= t < t_climb else (hover if t < 15. else .5),
               vz_of=lambda t, vz, drive: vz+(stronger(drive)-G-CS.body_drag_s_inv[2]*vz)*.01)
    assert not any(f for *_, f in rows)
    assert pilot.contact_gain > 1.05


def test_off_by_default_and_the_default_and_version_1_pilots_are_bit_identical_to_m4():
    from tests.test_fast_race_cue_descent_view import golden_scenario
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., descent_view=DV)
    assert pilot.contact_support is None and pilot.contact_summary() is None
    assert all(np.isnan(v) for v in pilot.contact_log().values())
    digest = lambda trace: hashlib.sha256(np.ascontiguousarray(trace.astype(np.float64)).tobytes()).hexdigest()  # noqa
    assert digest(golden_scenario()[0]) == '67d31f2892c3f292871a40e6cb78e8a734ae540636aa8a3508a1ce3572ad3d1f'
    # the version-1 pilot (view rule only), from the m4 tree (3decaac)
    assert digest(golden_scenario(descent_view=DV)[0]) == \
        'eae7b39aeac9c76db695012782f1f6903d1ccf9ddbbe033c032dc9554dc7914a'


# ---------------------------------------------------------------------------------------------
# The clearance brake's sink floor (wall pilot version 5)
# ---------------------------------------------------------------------------------------------
def test_clearance_brake_config_and_declaration():
    with pytest.raises(ValueError):
        ClearanceBrakeConfig(max_added_sink=-.1)
    with pytest.raises(ValueError, match='ClearanceBrakeConfig'):
        FastRaceCue(SENSOR, CameraPoseHistory(), 6., clearance_brake=dict())
    from haltere.liftoff.visual_brain import WALL_PILOT_DECLARATION, load_wall_pilot
    declaration, _ = load_wall_pilot(WALL_PILOT_DECLARATION)
    # version 6 (round 5) keeps version 5's sink floor
    assert WALL_PILOT_VERSION == 6 and wall_pilot_configs(declaration)['clearance_brake'] == ClearanceBrakeConfig()
    assert declaration['clearance_brake'] == dict(max_added_sink=0.)
    v5 = json.loads(WALL_PILOT_DECLARATION.with_name('wall_pilot_v5.json').read_text(encoding='utf-8'))
    assert wall_pilot_configs(v5)['clearance_brake'] == ClearanceBrakeConfig()


def rising_wall(**kw):
    """The pilot cruising at 4 m/s toward a ring ahead while climbing 0.8 m/s (a travel ray about 11 deg up), with a
    wall 0.6 s ahead in every looming sample: the stand-off cap binds along that rising ray. Rows: (command, brake
    log, braking)."""
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **kw)
    rows = []
    for k in range(250):
        t = 10.+k*.01
        s = senses(position=(4.*k*.01, 0., 1.), velocity=(4., 0., .8))
        history.append(t, [4.*k*.01, 0., 1.], s['quat'][0].numpy())
        clearance = dict(time=t-.05, ttc=.6, distance=2.4, below_fraction=.5, ttc_lower=None) if k >= 50 and \
            k % 5 == 0 else None
        pilot.update(s, [0., 0., 0.], dict(race_cue=dict(AHEAD)), t-.05, t, clearance=clearance)
        rows.append((pilot.velocity_command.copy(), pilot.brake_log(), pilot.clearance_braking))
    return pilot, rows


def test_the_clearance_brake_commands_no_descent_the_pilot_did_not_ask_for():
    base, plain = rising_wall()
    floored, applied = rising_wall(clearance_brake=ClearanceBrakeConfig())
    shadow, shadowed = rising_wall(clearance_brake=ClearanceBrakeConfig(), wall_apply=False)
    vz_plain = np.array([r[0][2] for r in plain])
    vz = np.array([r[0][2] for r in applied])
    added = np.array([r[1]['brake_added_sink'] for r in plain])
    # without the floor the brake along the rising ray turns the level request into a descent
    assert any(r[2] for r in plain) and vz_plain.min() < -.1 and added.max() > .1
    # with it the request never goes below level (min(the pilot's request, 0)) and no sink is left
    assert vz.min() >= -1e-9 and max(r[1]['brake_sink_left'] for r in applied) <= 1e-9
    assert max(r[1]['brake_sink_withheld'] for r in applied) > .1
    # the horizontal part is at least as strongly braked along the ray (no speed toward the wall is added)
    ray = np.array([4., 0., .8])/np.hypot(4., .8)
    assert all(r[1]['brake_ray_x'] == pytest.approx(ray[0], abs=.05) for r in applied if r[2])
    along = np.array([r[0][:2] @ ray[:2] for r in applied])
    along_plain = np.array([r[0][:2] @ ray[:2] for r in plain])
    assert np.all(along <= along_plain+.02)
    # shadow: the flown commands are the unfloored pilot's, bit for bit; the log shows the sink the brake added
    np.testing.assert_array_equal(np.array([r[0] for r in shadowed]), np.array([r[0] for r in plain]))
    assert max(r[1]['brake_sink_withheld'] for r in shadowed) == 0.
    meta = floored.metadata()['wall_pilot']
    assert meta['version'] == 5 and meta['clearance_brake']['parameters'] == dict(max_added_sink=0.)
    assert meta['clearance_brake']['seconds']['withheld'] > 0 and meta['clearance_brake']['max_sink']['left'] == 0.
    assert base.metadata().get('wall_pilot') is None


def test_the_floor_keeps_the_pilots_own_sink_and_a_downward_ray_unchanged():
    """A sink the pilot asked for itself is kept (the floor is min(request, 0)), and the brake may still raise the
    request when the ray points down (terrain ahead and below)."""
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., clearance_brake=ClearanceBrakeConfig())
    assert pilot._brake_sink_floor(-.5, -.8) == -.5 and pilot.brake_sink_withheld == pytest.approx(.3)
    assert pilot._brake_sink_floor(-.5, -.3) == -.3
    assert pilot._brake_sink_floor(.4, .1) == .1 and pilot._brake_sink_floor(.4, -.2) == 0.
    free = FastRaceCue(SENSOR, CameraPoseHistory(), 6.)
    assert free._brake_sink_floor(.4, -.2) == -.2 and free.brake_added_sink == pytest.approx(.2)


def test_rising_ray_scenario_uses_the_tilted_travel_direction():
    _, rows = rising_wall(clearance_brake=ClearanceBrakeConfig())
    rays = [np.array([r[1]['brake_ray_x'], r[1]['brake_ray_y'], r[1]['brake_ray_z']]) for r in rows if r[2]]
    assert rays and all(ray[2] > .1 for ray in rays)
