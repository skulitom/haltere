"""Round 6 (brake): the looming governor's early brake per motor contract (configs/obstacles/early_brake.json version 1,
`EarlyBrakeConfig`) and motor assist version 4 (configs/pilot/motor_assist.json; no approach source, a tracking floor
with the ring in view, the ceiling cut bounding only the assist's share of a climb; versions 1-3 kept and refused by the
runner).

The governor tests feed a straight approach with perfect looming samples (the shape of the minus-brain11cw13-r4b-noassist-01
hairpin: a wall read at a TTC of about 1 s long before the governor's own 0.8 s engagement); the pilot tests use the
synthetic states of the motor-assist tests. Development cases only; none of this is flight evidence.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from haltere.liftoff import fast_race_cue as frc
from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import (EarlyBrakeConfig, FastRaceCue, MotorAssistConfig, TtcClearanceConfig,
                                           TtcClearanceGovernor, early_brake_for_contract, motor_assist_for_contract)
from tests.test_fast_race_cue import cue_toward, single_thread  # noqa: F401 (fixture)
from tests.test_visual_assistance import SENSOR, senses

ROOT = Path(__file__).resolve().parents[1]
BRAIN = EarlyBrakeConfig()                                             # the declared brain entry (floor while ring ahead)
BRAIN_ENGAGE = EarlyBrakeConfig(floor_until='engagement')              # the development alternative
PD_MODEL = EarlyBrakeConfig(stop_latency_s=.15, stop_deceleration=6.)  # the fast PD's model (the PD has no entry)


def canonical(obj):
    body = {k: v for k, v in obj.items() if k not in ('frozen', 'frozen_at', 'sha256')}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
                          .encode('utf-8')).hexdigest()


def approach(early=None, *, speed=4.5, wall_x=12., seconds=2.4, below=None, lower=None, climb=False, clear_from=None):
    """A drone flying +x at a constant speed toward a wall at wall_x; a looming sample every 0.055 s (received 0.085 s
    later) with the TTC of the wall (below_fraction/ttc_lower as given; lower 'same' puts ttc_lower = ttc). Rows per 10 ms
    tick: (t, x, cap, target, early_active, status)."""
    gov = TtcClearanceGovernor(TtcClearanceConfig(), early=early)
    if climb:
        gov.climb = 1.
        gov.climb_hold_until = np.inf
    rows, pending, next_sample, x = [], [], 0., 0.
    for k in range(int(round(seconds/.01))):
        t = k*.01
        if t >= next_sample:
            ttc = (wall_x-x)/speed if clear_from is None or x < clear_from else 5.
            pending.append((t+.085, t, ttc, x))
            next_sample += .055
        while pending and pending[0][0] <= t:
            received, time, ttc, where = pending.pop(0)
            gov.ingest(time, ttc, ttc*speed, below, [where, 0., 1.], [1., 0., 0.], speed, received=received,
                       ttc_lower=ttc if lower == 'same' else lower)
        cap, _, _ = gov.limits([x, 0., 1.], [speed, 0., 0.], t, .01, 3.5)
        rows.append((t, x, cap, gov.target, gov.early_active, gov.status))
        x += speed*.01
    return gov, rows


def first_cap(rows, below):
    return next((x for _, x, cap, *_ in rows if cap is not None and cap < below), None)


# ----------------------------------------------------------------------------------------------------------------
# Early brake: configuration
# ----------------------------------------------------------------------------------------------------------------
def test_config_validation_and_contract_entries():
    assert BRAIN.stopping_distance(4.5) == pytest.approx(4.5*.3+4.5**2/7.+.5)
    assert PD_MODEL.stopping_distance(6.) == pytest.approx(.9+3.+.5)
    for bad in (dict(stop_deceleration=0.), dict(floor_speed=-1.), dict(stop_latency_s=float('nan')),
                dict(lower_window=1), dict(no_climb='yes'), dict(floor_until='never')):
        with pytest.raises(ValueError):
            EarlyBrakeConfig(**bad)
    declaration = dict(version=frc.EARLY_BRAKE_VERSION, contracts=dict(fast_velocity_brain_v1=dict(floor_speed=2.),
                                                                        fast_velocity_pd_v1=None))
    assert early_brake_for_contract(declaration, 'fast_velocity_brain_v1') == EarlyBrakeConfig(floor_speed=2.)
    assert early_brake_for_contract(declaration, 'fast_velocity_pd_v1') is None
    assert early_brake_for_contract(declaration, 'motor_tracking_teacher_v1') is None
    with pytest.raises(ValueError, match='version'):
        early_brake_for_contract(dict(declaration, version=2), 'fast_velocity_brain_v1')
    with pytest.raises(ValueError, match='contracts'):
        early_brake_for_contract(dict(version=1), 'fast_velocity_brain_v1')
    with pytest.raises(ValueError):
        TtcClearanceGovernor(early=dict(floor_speed=2.))


# ----------------------------------------------------------------------------------------------------------------
# Early brake: the governor
# ----------------------------------------------------------------------------------------------------------------
def test_without_the_rule_the_governor_engages_at_its_own_ttc():
    gov, rows = approach(None)
    x = first_cap(rows, 4.5)
    # the governor's own engagement: three samples with TTC < 0.8 s (about 3.6 m before the wall at 4.5 m/s)
    assert x is not None and 12.-x < .8*4.5+.3
    assert 'early_votes' not in gov.counts


def test_the_brain_contract_engages_where_its_stopping_distance_reaches_the_wall():
    base, rows_base = approach(None)
    gov, rows = approach(BRAIN)
    x_base, x = first_cap(rows_base, 4.5), first_cap(rows, 4.5)
    assert x is not None and x_base is not None
    # the stopping distance at 4.5 m/s (4.74 m) is reached about 1.05 s (TTC) before the wall; three votes confirm it
    assert 12.-x > 4.2 and x_base-x > .9
    # (the older samples of the confirmation are aged to the present position, so they vote with the newest one)
    assert gov.counts['early_engagements'] == 1 and gov.counts['early_votes'] >= 1
    # the early episode's first target is the governor's own graded target (0.7 x the closing speed), above the floor
    engaged = next(r for r in rows if r[2] is not None and r[2] < 4.5)
    assert engaged[4] is True
    assert min(r[3] for r in rows if r[3] is not None) == pytest.approx(.7*4.5)


def test_the_engagement_alternative_is_floored_until_the_governors_own_condition_holds():
    # at 3.2 m/s the graded target (0.7 x 3.2 = 2.24) lies below the floor (2.5)
    gov, rows = approach(BRAIN_ENGAGE, speed=3.2, wall_x=8., seconds=2.4)
    early = [r for r in rows if r[4]]
    assert early and min(r[3] for r in early) == pytest.approx(2.5)
    handed = [r for r in rows if r[3] is not None and not r[4] and r[1] > early[0][1]]
    # once three samples read TTC < 0.8 s the governor brakes as without the rule (below the floor)
    assert handed and min(r[3] for r in handed) == pytest.approx(.7*3.2)
    assert gov.counts['early_handovers'] == 1 and gov.counts['early_floored'] > 0


def test_the_declared_floor_holds_while_the_ring_is_ahead_and_ends_with_a_wall_ahead_condition():
    gov, rows = approach(BRAIN, speed=3.2, wall_x=8., seconds=2.4)
    assert min(r[3] for r in rows if r[3] is not None) == pytest.approx(2.5)      # ring ahead: never below the floor
    assert gov.counts['early_handovers'] == 0 and gov.early_active
    # the pilot reports a wall-ahead condition (the next checkpoint beside the surface): the governor's own braking
    gov, rows = approach(BRAIN, speed=3.2, wall_x=8., seconds=1.2)
    gov.ring_ahead = False
    for k in range(60):
        t, x = 1.2+k*.01, 3.2*(1.2+k*.01)
        gov.ingest(t, (8.-x)/3.2, 8.-x, None, [x, 0., 1.], [1., 0., 0.], 3.2, received=t)
        gov.limits([x, 0., 1.], [3.2, 0., 0.], t, .01, 3.5)
    assert gov.target == pytest.approx(.7*3.2)


def test_the_pilot_reports_whether_its_checkpoint_is_ahead():
    from tests.test_fast_race_cue_motor_assist import SIDE
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., early_brake=BRAIN)
    pilot.launching = False
    assert pilot._checkpoint_beside('side', 0.) and pilot._checkpoint_beside('coast', 0.)
    pilot.direction = np.array([np.cos(np.radians(60.)), np.sin(np.radians(60.)), 0.])
    assert pilot._checkpoint_beside('cue', 0.) and not pilot._checkpoint_beside('cue', np.radians(40.))
    pilot.turn_first_active = True
    assert pilot._checkpoint_beside('cue', np.radians(40.))
    assert SIDE['edge'] is True


@pytest.mark.parametrize('below,lower,climb', [(.8, None, False), (None, 'same', False), (None, None, True)])
def test_terrain_lower_window_and_climb_samples_do_not_vote(below, lower, climb):
    gov, rows = approach(BRAIN, below=below, lower=lower, climb=climb)
    assert gov.counts['early_engagements'] == 0
    base, base_rows = approach(None, below=below, lower=lower, climb=climb)
    assert [r[2] for r in rows] == [r[2] for r in base_rows]


def test_the_fast_pd_model_would_engage_about_two_samples_before_the_governors_own_condition():
    # (the declaration gives the fast PD no entry; its stopping distance at 6 m/s, 4.4 m, is a TTC of 0.73 s aged to
    # the present, reached about two samples (0.11 s) before three raw TTCs fall below 0.8 s)
    gov, rows = approach(PD_MODEL, speed=6., wall_x=15.)
    base, base_rows = approach(None, speed=6., wall_x=15.)
    x, x_base = first_cap(rows, 6.), first_cap(base_rows, 6.)
    assert x is not None and x_base is not None and 0. <= x_base-x <= 6.*.12


def test_a_clear_view_ends_the_early_episode_when_the_cap_has_released():
    gov, rows = approach(BRAIN, wall_x=12., clear_from=8.2, seconds=5.)
    active = [r for r in rows if r[4]]
    assert active and not rows[-1][4]
    assert gov.counts['early_handovers'] == 0                # never reached the governor's own condition


# ----------------------------------------------------------------------------------------------------------------
# Early brake: the pilot
# ----------------------------------------------------------------------------------------------------------------
def pilot_run(**kw):
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **kw)
    x, rows, pending, next_sample, speed, wall_x = 0., [], [], 0., 4.5, 12.
    for k in range(200):
        t = 10.+k*.01
        s = senses(position=(x, 0., 5.), velocity=(speed, 0., 0.), yaw=0.)
        history.append(t, [x, 0., 5.], s['quat'][0].numpy())
        if k*.01 >= next_sample:
            ttc = (wall_x-x)/speed
            pending.append((t+.085, dict(time=t, ttc=ttc, distance=ttc*speed, below_fraction=None, ttc_lower=None)))
            next_sample += .055
        sample = None
        while pending and pending[0][0] <= t:
            sample = pending.pop(0)[1]
        extra = dict(clearance=sample) if sample is not None else {}
        pilot.update(s, [0., 0., 0.], dict(race_cue=dict(cue_toward([wall_x+10.-x, 0., 0.]))), t-.05, t, **extra)
        rows.append(np.r_[pilot.velocity_command, pilot.pilot.sight_yaw, pilot.early_log()['early_brake']])
        x += speed*.01
    return pilot, np.asarray(rows)


def test_off_by_default_and_in_shadow_the_requests_are_unchanged():
    base, rows = pilot_run()
    assert base.early_brake is None and 'early_brake' not in base.metadata()
    assert np.isnan(rows[:, -1]).all()
    shadow, shadow_rows = pilot_run(early_brake=BRAIN, early_apply=False)
    assert np.array_equal(shadow_rows[:, :4], rows[:, :4])
    assert shadow.clearance.early is None and shadow.clearance_shadow.early == BRAIN
    assert shadow_rows[:, -1].max() == 1.                     # the shadow copy's episode is logged
    meta = shadow.metadata()['early_brake']
    assert meta['applied'] is False and meta['counts']['early_engagements'] == 1
    on, on_rows = pilot_run(early_brake=BRAIN)
    # within these 2 s (the wall's TTC falls to 0.68 s) the governor's own condition has not braked; the early brake has
    assert np.hypot(rows[100:, 0], rows[100:, 1]).min() > 5.9
    assert np.hypot(on_rows[100:, 0], on_rows[100:, 1]).min() < 3.5
    assert on.metadata()['early_brake']['applied'] is True


def test_a_pilot_without_the_rule_builds_its_governor_exactly_as_before():
    pilot, _ = pilot_run()
    assert pilot.clearance.early is None and pilot.clearance_shadow is None


# ----------------------------------------------------------------------------------------------------------------
# Motor assist version 4
# ----------------------------------------------------------------------------------------------------------------
V4 = dict(cap_sources=('request', 'governor', 'turn_first', 'stopping'), track_floor=2.5, ceiling_share=True,
          version=4)


def test_version_4_validation_and_entries():
    assert MotorAssistConfig().version == 3 and MotorAssistConfig().track_floor == 0.
    MotorAssistConfig(**V4)
    for bad in (dict(V4, cap_sources=frc.MOTOR_ASSIST_SOURCES), dict(V4, stop_gate='any'),
                dict(track_floor=2.5), dict(ceiling_share=True), dict(V4, ceiling_share=1)):
        with pytest.raises(ValueError):
            MotorAssistConfig(**bad)
    entry = dict(cap_sources=list(V4['cap_sources']), track_floor=2.5, ceiling_share=True)
    config = motor_assist_for_contract(dict(version=4, contracts=dict(fast_velocity_brain_v1=entry)),
                                       'fast_velocity_brain_v1')
    assert config == MotorAssistConfig(**V4)
    with pytest.raises(ValueError, match='version-4 entry'):
        motor_assist_for_contract(dict(version=4, contracts=dict(fast_velocity_brain_v1=dict(track_floor=2.5))),
                                  'fast_velocity_brain_v1')
    with pytest.raises(ValueError, match='version-4 fields'):
        motor_assist_for_contract(dict(version=3, contracts=dict(fast_velocity_brain_v1=dict(track_floor=2.5))),
                                  'fast_velocity_brain_v1')


def wall_run(config, *, cue_fn=None, speed=5., wall_x=12., seconds=1.8):
    from tests.test_fast_race_cue_motor_assist import wall_run_v2
    return wall_run_v2(config, cue=cue_fn, speed=speed, wall_x=wall_x, seconds=seconds)


def test_version_4_never_plans_a_crawl_with_the_ring_in_view():
    """A brain flying 5 m/s at a surface with its ring straight ahead through it (a gate arch): version 3's approach
    plans down to its floor and its cap tracking below it; version 4 has no approach and keeps cap tracking at or above
    min(bound, track_floor) while no wall-ahead condition holds."""
    _, rows3 = wall_run(MotorAssistConfig())
    _, rows4 = wall_run(MotorAssistConfig(**V4))
    assert any('approach' in r['sources'] for r in rows3)
    assert not any('approach' in r['sources'] for r in rows4)
    ahead = [r for r in rows4 if not r['wall_ahead']]
    assert ahead
    for r in ahead:
        own, flown = np.linalg.norm(r['own'][:2]), np.linalg.norm(r['flown'][:2])
        assert flown >= min(own, 2.5)-1e-6


def test_version_4_keeps_the_stopping_source_under_wall_ahead_conditions():
    from tests.test_fast_race_cue_motor_assist import SIDE
    _, rows = wall_run(MotorAssistConfig(**V4), cue_fn=lambda x: dict(SIDE))
    stops = [r for r in rows if r['wall_ahead'] and 'stopping' in r['sources']]
    # the speed toward the wall (+x) is bounded by the stopping model below the pilot's own (no track floor there)
    assert stops and min(r['flown'][0] for r in stops) < min(r['own'][0] for r in stops) and         min(r['flown'][0] for r in stops) < 1.


def share_run(assist):
    """The pilot climbing toward a ring above in view (state cue) while it sinks below its request (a sag climb of the
    assist), under the ceiling guard (wall pilot v6: any_climb) with overhead evidence (expansion above the path)
    ahead: rows of (the pilot's own vertical request, the vertical request the motor receives, state)."""
    wall, _ = __import__('haltere.liftoff.visual_brain', fromlist=['load_wall_pilot']).load_wall_pilot()
    kw = frc.wall_pilot_configs(wall, 'fast_velocity_brain_v1')
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., motor_assist=assist, **kw)
    rows, pending, next_sample = [], [], 0.
    for k in range(150):
        t = 10.+k*.01
        position = (0., 0., 5.+.5*k*.01)
        s = senses(position=position, velocity=(3., 0., .5), yaw=0.)
        history.append(t, list(position), s['quat'][0].numpy())
        if k*.01 >= next_sample:
            pending.append((t+.085, dict(time=t, ttc=.8, distance=2.4, below_fraction=.2, ttc_lower=None)))
            next_sample += .055
        sample = None
        while pending and pending[0][0] <= t:
            sample = pending.pop(0)[1]
        extra = dict(clearance=sample) if sample is not None else {}
        pilot.update(s, [0., 0., 0.], dict(race_cue=dict(cue_toward([10., 0., 6.]))), t-.05, t, **extra)
        own = pilot.pilot_command if pilot.pilot_command is not None else pilot.velocity_command
        rows.append((float(own[2]), float(pilot.velocity_command[2]), pilot.state))
    return pilot, rows


def test_the_ceiling_cut_bounds_only_the_assists_share_of_a_climb():
    base, base_rows = share_run(None)
    v3, rows3 = share_run(MotorAssistConfig())
    v4, rows4 = share_run(MotorAssistConfig(**V4))
    assert all(r[2] == 'cue' for r in base_rows[20:])
    own_base = np.array([r[0] for r in base_rows])
    # without the assist the pilot's own climb toward the ring in view is exempt: no cut
    assert own_base[-50:].min() > 1.
    # version 3: the sag climb removes the exemption and the cut zeroes the pilot's own climb
    assert min(r[0] for r in rows3[-50:]) <= 1e-9 and v3.clearance.counts['overhead_engagements'] > 0
    # version 4: the pilot's own request is the unassisted pilot's; the assist's share is bounded to the cap
    assert np.allclose([r[0] for r in rows4], own_base, atol=1e-9)
    assert v4.clearance.counts.get('share_engagements', 0) > 0 and v4.clearance.counts['overhead_engagements'] == 0
    assert all(r[1] <= max(r[0], 0.)+1e-9 for r in rows4[-50:])
    assert v4.motor_assist_summary()['v4']['ceiling_share_seconds'] > 0


# ----------------------------------------------------------------------------------------------------------------
# Runner and replay harness
# ----------------------------------------------------------------------------------------------------------------
def test_runner_stack_wiring_columns_and_refusals(tmp_path):
    from types import SimpleNamespace
    from haltere.liftoff.visual_brain import (EARLY_BRAKE_DECLARATION, EARLY_COLUMNS, early_row, load_early_brake,
                                              resolve_obstacle_stack)
    base = dict(pilot_profile='fast', looming_brake=True)
    stack = resolve_obstacle_stack(SimpleNamespace(obstacle_stack='on', **base))
    assert stack['early_brake'] is None
    stack = resolve_obstacle_stack(SimpleNamespace(obstacle_stack='on', early_brake='on', **base))
    assert stack['early_brake'] == str(EARLY_BRAKE_DECLARATION)
    with pytest.raises(ValueError, match='obstacle stack'):
        resolve_obstacle_stack(SimpleNamespace(early_brake='on', **base))
    with pytest.raises(ValueError, match='on or off'):
        resolve_obstacle_stack(SimpleNamespace(obstacle_stack='on', early_brake='x', **base))
    assert EARLY_COLUMNS == ('early_brake',) and np.isnan(early_row(None)[0])
    unfrozen = tmp_path/'early.json'
    unfrozen.write_text(json.dumps(dict(version=1, contracts={})), encoding='utf-8')
    with pytest.raises(ValueError, match='frozen'):
        load_early_brake(unfrozen)


def test_runner_csv_tail_puts_the_early_brake_column_last():
    import inspect
    import re
    from haltere.liftoff import visual_brain
    source = re.sub(r'\s+', '', inspect.getsource(visual_brain.run))
    # m6 integration: the marker-jump columns (round 6, gates) come between the stale-evidence and early-brake columns,
    # and the sighted-descent columns (round 6, ground) after the early-brake column; round 7 appends the ring-lead columns
    assert ('*view_columns,*assist_columns,*stale_columns,*marker_columns,*early_columns,*sighted_columns,'
            '*ring_lead_columns])') in source
    assert ('*(stale_row(controller.assistance)ifstale_columnselse()),'
            '*(marker_jump_row(controller.assistance)ifmarker_columnselse()),'
            '*(early_row(controller.assistance)ifearly_columnselse()),'
            '*(sighted_row(controller.assistance)ifsighted_columnselse()),'
            '*(ring_lead_row(controller.assistance)ifring_lead_columnselse())])') in source


def test_the_replay_harness_adds_the_rule_for_the_brain_contract_only():
    from haltere.obstacles.vertical_replay import build
    declaration = dict(version=1, contracts=dict(fast_velocity_brain_v1={}, fast_velocity_pd_v1=None))
    side = dict(motor_controller=dict(contract='fast_velocity_pd_v1'), pilot_assistance=dict(nominal_speed_mps=6.),
                gate_sensor=dict(focal_320=100., tilt_deg=30.))
    pilot, info = build(side, ROOT, stack='on', early_brake=declaration)
    assert pilot.early_brake is None and info['early_brake'] is False
    side['motor_controller']['contract'] = 'fast_velocity_brain_v1'
    pilot, info = build(side, ROOT, stack='on', early_brake=declaration)
    assert pilot.early_brake == EarlyBrakeConfig() and info['early_brake'] is True
    pilot, info = build(side, ROOT, stack='shadow', early_brake=declaration)
    assert pilot.early_brake is not None and pilot.early_apply is False
    pilot, info = build(side, ROOT, stack='none', early_brake=declaration)
    assert pilot.early_brake is None


# ----------------------------------------------------------------------------------------------------------------
# Default off: the round-5 stack is bit-identical to m5
# ----------------------------------------------------------------------------------------------------------------
# the motor-assist tests' full-stack scenario (tests/test_fast_race_cue_motor_assist.stack_scenario) with the stale-evidence
# rule v2 and motor assist v3, computed with the same function on `git archive m5` (bc7c71c)
M5_STACK_SE_MA3_DIGEST = 'd1a5dc4a0bc6e5ed9aa73c93590e17b7c7600f39d181a3add03c2d001e524421'


def test_the_round5_stack_with_assist_v3_is_bit_identical_to_m5_and_the_new_rules_change_it():
    from tests.test_fast_race_cue_motor_assist import stack_scenario
    se = frc.stale_evidence_configs(json.loads((ROOT/'configs/obstacles/stale_evidence.json').read_text(encoding='utf-8')))
    v3 = motor_assist_for_contract(json.loads((ROOT/'configs/pilot/motor_assist_v3.json').read_text(encoding='utf-8')),
                                   'fast_velocity_brain_v1')
    trace, _ = stack_scenario(**se, motor_assist=v3)
    digest = hashlib.sha256(np.ascontiguousarray(trace.astype(np.float64)).tobytes()).hexdigest()
    assert digest == M5_STACK_SE_MA3_DIGEST
    # (the scenario's wall samples carry ttc_lower = ttc, which the declared rule leaves to the lower window)
    same, pilot = stack_scenario(**se, motor_assist=v3, early_brake=BRAIN)
    assert pilot.clearance.counts['early_engagements'] == 0 and np.array_equal(same, trace)
    changed, pilot = stack_scenario(**se, motor_assist=v3, early_brake=EarlyBrakeConfig(lower_window=False))
    assert pilot.clearance.counts['early_engagements'] > 0 and not np.array_equal(changed, trace)


# ----------------------------------------------------------------------------------------------------------------
# Frozen declarations and gates
# ----------------------------------------------------------------------------------------------------------------
def test_the_declarations_are_frozen_and_the_runner_flies_only_them(tmp_path):
    from haltere.liftoff.visual_brain import load_early_brake, load_motor_assist
    early, digest = load_early_brake()
    assert early['version'] == 1 and digest == early['sha256'] == canonical(early)
    assert early_brake_for_contract(early, 'fast_velocity_brain_v1') == EarlyBrakeConfig()
    assert early_brake_for_contract(early, 'fast_velocity_pd_v1') is None and 'fast_velocity_pd_v1' in early['contracts']
    assist, digest = load_motor_assist()
    assert assist['version'] == 4 and digest == assist['sha256'] == canonical(assist)
    assert motor_assist_for_contract(assist, 'fast_velocity_brain_v1') == MotorAssistConfig(**V4)
    assert motor_assist_for_contract(assist, 'fast_velocity_pd_v1') is None
    kept = json.loads((ROOT/'configs/pilot/motor_assist_v3.json').read_text(encoding='utf-8'))
    assert kept['sha256'] == '7c3b49e7bcc70ece18b00e68285c427ea430016fc52392122663205fd6be8772' == canonical(kept)
    with pytest.raises(ValueError, match='version 3'):
        load_motor_assist(ROOT/'configs/pilot/motor_assist_v3.json')
    edited = dict(early, contracts=dict(early['contracts'], fast_velocity_brain_v1=dict(
        early['contracts']['fast_velocity_brain_v1'], floor_speed=2.)))
    path = tmp_path/'edited.json'
    path.write_text(json.dumps(edited), encoding='utf-8')
    with pytest.raises(ValueError, match='frozen'):
        load_early_brake(path)


def test_the_gates_are_frozen_name_the_declarations_and_hold_out_fresh_sets():
    from haltere.obstacles.early_brake_gates import load_gates
    gates, digest = load_gates()
    assert gates['version'] == 1 and digest == gates['sha256']
    names = {k: v['file'] for k, v in gates['declarations'].items()}
    assert names == dict(early_brake='configs/obstacles/early_brake.json', motor_assist='configs/pilot/motor_assist.json',
                         motor_assist_v3='configs/pilot/motor_assist_v3.json')
    logs = gates['logs']
    assert len(logs['development']) == 11 and len(logs['held_out']) == 20
    assert not set(logs['development']) & set(logs['held_out'])
    parts = gates['harness']['parts']
    assert parts['hairpin']['sim_seed'] not in (17, 23, 31, 41, 67) and parts['hairpin']['live_wall_samples'] is True
    used = dict(turn_deg={20., 60., 90., 45., 75., 105., 35., 85., 115., 50., 80., 110., 55., 120.},
                arch_m={7., 11., 8., 13., 9., 12., 10., 14., 9.5, 13.5},
                wall_m={2.1, 2.6, 2.3, 3., 2.4, 2.8, 2.5, 2.9, 2.2, 2.65})
    for key, values in used.items():
        assert not set(parts['hairpin']['set'][key]) & values
    assert not set(parts['passthrough']['set']['turn_deg']) & {0., 45., 90., 30., 80.}
    assert parts['hill']['set'] == 'hill:6500-6511'


def test_gate_metrics_on_synthetic_replays():
    from haltere.obstacles import early_brake_gates as eg
    n = 400
    t = np.arange(n)*.01
    base = dict(t=t, cvx=np.full(n, 5.), cvy=np.zeros(n), cvz=np.zeros(n), braking=np.zeros(n),
                state=np.array(['cue']*n))
    var = {k: v.copy() for k, v in base.items()}
    var['braking'][250:] = 1.
    assert eg.brake_warning_s(var) == pytest.approx(t[-1]-t[250])
    var['cvx'][300:] = 3.5
    var['cvx'][320:325] = 4.5                          # a 0.05 s gap is bridged
    assert eg.final_cut_s(base, var) == pytest.approx(t[-1]-t[300])
    var['cvx'][100:130] = 1.5
    assert eg.crawl_s(base, var) == pytest.approx(.3)
    var['state'][100:130] = 'side'
    assert eg.crawl_s(base, var) == 0.
    a = dict(t=t, cvx=np.full(n, 2.), cvy=np.zeros(n), assist_pilot_vx=np.full(n, 4.), assist_pilot_vy=np.zeros(n),
             assist_wall_ahead=np.zeros(n))
    assert eg.floor_violation_s(a, 2.5) == pytest.approx(t[-1])
    a['assist_wall_ahead'][:] = 1.
    assert eg.floor_violation_s(a, 2.5) == 0.
    stack = dict(t=t, cvz=np.full(n, 1.5))
    v4 = dict(state=np.array(['cue']*n), assist_vertical=np.full(n, .2), assist_pilot_vz=np.full(n, 1.5))
    assert eg.share_loss_s(stack, v4) == 0.
    v4['assist_pilot_vz'][200:250] = 0.
    assert eg.share_loss_s(stack, v4) == pytest.approx(.5)
