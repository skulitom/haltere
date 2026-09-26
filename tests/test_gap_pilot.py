"""Gap cue wired into the fast pilot (haltere.liftoff.gap_aim, FastRaceCue gap=..., camera slots, runner flags).

Pins: the default pilot is unchanged (bit-identical with no gap aim, or a gap aim without samples); the
confirmation (2 of 3 same-sign samples within 0.25 s), stale-sample rejection, slew, decay, side latch and the
two conflicts; terrain side steer only while the looming governor requests a climb; the shift rotates the ring
ray about world z (positive = left) and never changes the requested speed schedule beyond its own bearing;
shadow computes and logs without applying; lag-turn triggers ignore the applied shift; the camera's shared
slots and frame slot carry the samples; the runner's flag resolution. None of it is flight evidence.
"""
from dataclasses import asdict, replace
import json
import multiprocessing as mp
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import FastRaceCue, LagTurnConfig, TtcClearanceGovernor
from haltere.liftoff.gap_aim import GapAim, GapAimConfig, direction_offset, rotate_z
from tests.test_fast_race_cue import cue_toward, drive
from tests.test_visual_assistance import SENSOR, senses

CFG = GapAimConfig()
DT = .01


@pytest.fixture(autouse=True, scope='module')
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def sample(t, shift, ring=0., valid=True, lr=float('nan'), kind='gap'):
    return dict(time=t, shift=shift, valid=valid, ring_deg=ring, lr=lr, kind=kind)


def run_aim(aim, samples, until, start=0., latency=.03, terrain=False):
    """Feed samples (each published `latency` after its capture) through 100 Hz ticks; rows (now, target, applied)."""
    rows = []
    pending = sorted(samples, key=lambda s: s['time'])
    latest = None
    n = int(round((until-start)/DT))
    for k in range(n+1):
        now = start+k*DT
        while pending and pending[0]['time']+latency <= now+1e-9:
            latest = pending.pop(0)
        aim.ingest(latest, now, terrain=terrain)
        aim.step(now, DT)
        rows.append((now, aim.target, aim.applied))
    return rows


def at(rows, t):
    return min(rows, key=lambda r: abs(r[0]-t))


# ---------------------------------------------------------------------------------------------
# GapAim
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize('overrides', [dict(confirm=4), dict(window=2.5), dict(slew_deg_s=0.), dict(decay_s=-1.),
                                       dict(max_shift_deg=50.), dict(terrain_side_deg=20.),
                                       dict(max_age_s=float('nan'))])
def test_gap_aim_config_validates(overrides):
    with pytest.raises(ValueError):
        GapAimConfig(**overrides)
    with pytest.raises(ValueError):
        GapAimConfig.from_dict(dict(asdict(CFG), route_gate=3))


def test_two_of_three_same_side_samples_confirm_and_the_shift_slews():
    aim = GapAim(CFG)
    rows = run_aim(aim, [sample(1.0, 8.), sample(1.066, 8.), sample(1.133, 8.)], 1.5, start=.95)
    assert at(rows, 1.05)[1] == 0.                      # one sample: not confirmed
    assert at(rows, 1.1)[1] == 8.                       # second sample (published at 1.096) confirms
    applied = np.array([r[2] for r in rows])
    assert np.all(np.abs(np.diff(applied)) <= CFG.slew_deg_s*DT+1e-9)
    assert at(rows, 1.3)[2] == pytest.approx(8.)
    assert aim.counts['obstacle_episodes'] == 1


def test_a_single_flicker_or_opposite_votes_do_not_confirm():
    aim = GapAim(CFG)
    rows = run_aim(aim, [sample(1.0, 8.), sample(1.066, 0.), sample(1.133, 0.), sample(1.2, -8.), sample(1.266, 1.)],
                   1.4, start=.95)
    assert all(r[1] == 0. and r[2] == 0. for r in rows)
    aim = GapAim(CFG)
    rows = run_aim(aim, [sample(1.0, 8.), sample(1.066, 0.), sample(1.133, 6.)], 1.3, start=.95)
    assert at(rows, 1.2)[1] == 6.                       # 2 of 3: the newest confirming shift


def test_samples_further_apart_than_the_window_do_not_confirm():
    aim = GapAim(CFG)
    rows = run_aim(aim, [sample(1.0, 8.), sample(1.3, 8.)], 1.45, start=.95)
    assert all(r[1] == 0. for r in rows)


def test_stale_samples_are_rejected():
    aim = GapAim(CFG)
    rows = run_aim(aim, [sample(1.0, 8.), sample(1.066, 8.), sample(1.133, 8.)], 1.6, start=.95, latency=.25)
    assert all(r[1] == 0. for r in rows) and aim.counts['stale'] == 3 and aim.counts['samples'] == 0


def test_without_fresh_samples_the_shift_decays_to_zero_over_decay_s():
    aim = GapAim(CFG)
    rows = run_aim(aim, [sample(1.0 + k*.066, 12.) for k in range(6)], 2.2, start=.95)
    last = 1.0+5*.066
    held = at(rows, last+CFG.max_age_s-.02)
    assert held[1] == 12. and held[2] == pytest.approx(12.)
    released = last+CFG.max_age_s+.01
    assert at(rows, released)[1] == 0.
    assert at(rows, released+CFG.decay_s/2)[2] == pytest.approx(6., abs=.5)
    assert at(rows, released+CFG.decay_s+.02)[2] == pytest.approx(0., abs=1e-9)


def test_side_latch_blocks_the_other_side_until_it_expires():
    aim = GapAim(CFG)
    left = [sample(1.0 + k*.066, 8.) for k in range(3)]
    right = [sample(1.3 + k*.066, -8.) for k in range(12)]
    rows = run_aim(aim, left+right, 2.3, start=.95)
    last_left = max(r[0] for r in rows if r[1] > 0)
    assert all(r[1] >= 0 for r in rows if r[0] < last_left+CFG.side_latch_s-.01)
    first_right = min(r[0] for r in rows if r[1] < 0)
    assert first_right >= last_left+CFG.side_latch_s-.011
    assert aim.counts['latch_blocks'] == 1 and aim.counts['obstacle_episodes'] == 2


def test_ring_and_flag_conflicts_drop_the_evidence_and_hold_the_ring_aim():
    aim = GapAim(CFG)
    run_aim(aim, [sample(1.0 + k*.066, 8., ring=0.) for k in range(4)], 1.4, start=.95)
    assert aim.applied > 0
    assert not aim.reconcile_ring(3., 1.4)              # the same ring, a few degrees on
    assert aim.reconcile_ring(25., 1.4)                 # another ring: conflict
    assert aim.applied == 0. and aim.target == 0. and not aim.samples and aim.counts['ring_conflicts'] == 1
    rows = run_aim(aim, [sample(1.45 + k*.066, 8., ring=25.) for k in range(12)], 2.4, start=1.41)
    assert all(r[1] == 0. for r in rows if r[0] < 1.4+CFG.side_latch_s)
    assert at(rows, 2.3)[1] == 8.
    assert not aim.flag_conflict(3., 2.4)               # the flag clearance points to the same side
    assert aim.flag_conflict(-3., 2.4) and aim.applied == 0. and aim.counts['flag_conflicts'] == 1
    assert not GapAim(CFG).reconcile_ring(40., 0.)      # nothing engaged: no conflict


def test_samples_of_another_ring_are_forgotten_without_a_conflict_when_idle():
    aim = GapAim(CFG)
    run_aim(aim, [sample(1.0, 8., ring=30.)], 1.05, start=.95)
    assert len(aim.samples) == 1
    assert not aim.reconcile_ring(0., 1.05) and not aim.samples


def test_terrain_side_steer_only_while_terrain_is_reported():
    lr = [sample(1.0 + k*.066, 0., lr=.8, kind='clear') for k in range(4)]
    aim = GapAim(CFG)
    assert all(r[1] == 0. for r in run_aim(aim, lr, 1.4, start=.95, terrain=False))
    aim = GapAim(CFG)
    rows = run_aim(aim, lr, 1.4, start=.95, terrain=True)
    assert at(rows, 1.3)[1] == -CFG.terrain_side_deg    # left nearer: steer right
    assert aim.counts['terrain_episodes'] == 1
    aim = GapAim(CFG)
    mixed = [sample(1.0 + k*.066, 7., lr=.8) for k in range(4)]
    assert at(run_aim(aim, mixed, 1.4, start=.95, terrain=True), 1.3)[1] == 7.   # obstacle votes first
    aim = GapAim(replace(CFG, terrain=False))
    assert all(r[1] == 0. for r in run_aim(aim, lr, 1.4, start=.95, terrain=True))


def test_direction_offset_combines_with_the_flag_clearance():
    assert direction_offset(0., 5., CFG) == 0.
    assert direction_offset(8., 0.2, CFG) == 8.
    assert direction_offset(8., 3., CFG) == pytest.approx(5.)     # total 8 from the centre
    assert direction_offset(2., 5., CFG) == 0.                    # the flag already aims further
    assert direction_offset(-8., 3., CFG) == 0.                   # opposite: a conflict, not a blend
    v = rotate_z(np.array([1., 0., .3]), 90.)
    assert v == pytest.approx([0., 1., .3])                       # positive = left (FLU)


# ---------------------------------------------------------------------------------------------
# FastRaceCue with the gap aim
# ---------------------------------------------------------------------------------------------
def fly_with_gap(pilot, history, shift, *, ring_deg=None, seconds=1.5, start=10., rate=15., latency=.05,
                 speed=6., ring=(20., 0., 0.)):
    """Straight flight at yaw 0 toward a ring ahead; gap samples every 1/rate s, published `latency` after capture."""
    rows = []
    measured = np.array([speed, 0., 0.])
    ring_az = float(np.degrees(np.arctan2(ring[1], ring[0])))
    latest = None
    next_capture = start
    captures = []
    for k in range(int(round(seconds/DT))):
        now = start+k*DT
        s = senses(position=(0., 0., 5.), velocity=tuple(measured.tolist()), yaw=0.)
        history.append(now, [0., 0., 5.], s['quat'][0].numpy())
        while next_capture <= now:
            captures.append(next_capture)
            next_capture += 1/rate
        ready = [c for c in captures if c+latency <= now+1e-9]
        if ready:
            latest = sample(ready[-1], shift(ready[-1]) if callable(shift) else shift,
                            ring=ring_az if ring_deg is None else ring_deg)
        detection = dict(race_cue=dict(cue_toward(list(ring))))
        pilot.update(s, [0., 0., 0.], detection, now-.05, now, gap=latest)
        rows.append((now, pilot.velocity_command.copy(), pilot.gap_offset_deg, pilot.gap_aim and pilot.gap_aim.applied,
                     pilot.gap_conflict, pilot.state))
        measured = pilot.velocity_command.copy()
    return rows


def heading(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def test_default_pilot_is_unchanged_by_gap_samples_and_by_an_idle_gap_aim():
    runs = []
    for kwargs, gap in ((dict(), None), (dict(), 8.), (dict(gap_aim=CFG), None)):
        history = CameraPoseHistory()
        pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **kwargs)
        if gap is None:
            rows = drive(pilot, history, cue_toward([20., 3., 0.]), 120, velocity=(6., 0., 0.))
            runs.append(np.array([r[2] for r in rows]))
        else:
            rows = []
            for k in range(120):
                now = 10.+k*DT
                s = senses(position=(0., 0., 5.), velocity=(6., 0., 0.) if not rows else tuple(rows[-1].tolist()))
                history.append(now, [0., 0., 5.], s['quat'][0].numpy())
                pilot.update(s, [0., 0., 0.], dict(race_cue=cue_toward([20., 3., 0.])), now-.05, now,
                             gap=sample(now-.05, gap))
                rows.append(pilot.velocity_command.copy())
            runs.append(np.array(rows))
    assert np.array_equal(runs[0], runs[1]) and np.array_equal(runs[0], runs[2])
    assert FastRaceCue(SENSOR, CameraPoseHistory(), 6.).metadata()['gap_aim'] is None


def test_a_confirmed_shift_rotates_the_aim_left_and_keeps_the_speed():
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., gap_aim=CFG)
    rows = fly_with_gap(pilot, history, 8.)
    final = rows[-1]
    assert final[2] == pytest.approx(8.) and final[3] == pytest.approx(8.)
    assert heading(final[1]) == pytest.approx(8., abs=1.)
    assert np.linalg.norm(final[1][:2]) == pytest.approx(6., rel=.03)      # no speed cap
    history = CameraPoseHistory()
    right = FastRaceCue(SENSOR, history, 6., reference_speed=6., gap_aim=CFG)
    assert heading(fly_with_gap(right, history, -8.)[-1][1]) == pytest.approx(-8., abs=1.)
    meta = json.loads(json.dumps(pilot.metadata()))['gap_aim']
    assert meta['applied'] is True and meta['counts']['obstacle_episodes'] == 1 and meta['parameters'] == asdict(CFG)


def test_shadow_computes_the_shift_but_flies_the_ring():
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., gap_aim=CFG, gap_apply=False)
    rows = fly_with_gap(pilot, history, 8.)
    assert rows[-1][3] == pytest.approx(8.) and all(r[2] == 0. for r in rows)
    assert abs(heading(rows[-1][1])) < .5
    assert pilot.metadata()['gap_aim']['applied'] is False


def test_a_sample_of_another_ring_is_a_conflict_and_the_ring_aim_is_held():
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., gap_aim=CFG)
    rows = fly_with_gap(pilot, history, lambda t: 8., ring_deg=None, seconds=.6)
    assert rows[-1][2] > 0
    rows = fly_with_gap(pilot, history, 8., ring_deg=25., seconds=.6, start=10.6)
    assert 'ring' in [r[4] for r in rows] and rows[-1][2] == 0.
    assert pilot.gap_aim.counts['ring_conflicts'] == 1
    assert abs(heading(rows[-1][1])) < 1.


def test_the_applied_shift_does_not_trigger_lag_turns():
    lag = LagTurnConfig()
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., gap_aim=CFG, lag_turn=lag)
    fly_with_gap(pilot, history, 12., seconds=2.)
    assert pilot.gap_offset_deg == pytest.approx(12.)
    assert pilot.lag_turn_triggers == 0


def test_lag_turn_shadow_logs_the_lead_without_flying_it():
    runs = {}
    for name, kwargs in (('off', {}), ('shadow', dict(lag_turn=LagTurnConfig(), lag_turn_apply=False)),
                         ('on', dict(lag_turn=LagTurnConfig()))):
        history = CameraPoseHistory()
        pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **kwargs)
        cue = lambda now: cue_toward([20., 0., 0.]) if now < 10.3 else cue_toward([20., 9., 0.])
        weights = []
        rows = []
        for k in range(100):
            now = 10.+k*DT
            v = (6., 0., 0.) if not rows else tuple(rows[-1].tolist())
            s = senses(position=(0., 0., 5.), velocity=v)
            history.append(now, [0., 0., 5.], s['quat'][0].numpy())
            pilot.update(s, [0., 0., 0.], dict(race_cue=cue(now)), now-.05, now)
            rows.append(pilot.velocity_command.copy())
            weights.append(pilot.lag_turn_weight)
        runs[name] = (np.array(rows), max(weights), pilot)
    assert np.array_equal(runs['off'][0], runs['shadow'][0])
    assert runs['shadow'][1] > 0 and runs['shadow'][2].lag_turn_triggers == runs['on'][2].lag_turn_triggers >= 1
    assert not np.array_equal(runs['off'][0], runs['on'][0])
    assert runs['shadow'][2].metadata()['lag_turn']['applied'] is False


@pytest.mark.parametrize('ring, shift', [(12.7, 12.), (12.7, -12.), (30., 12.), (-5., 8.), (0., 12.), (40., -3.)])
@pytest.mark.parametrize('apply', [True, False])
def test_the_lag_turn_lead_is_computed_without_the_gap_shift_and_the_shift_added_after_it(ring, shift, apply):
    """Declared interaction (lag_turn.json and gap_pilot.json version 2): the lead is course_lead x the angle from
    the flown course to the ring cue's bearing WITHOUT the applied gap shift (clipped), and the shift is added after
    it, so the lead never amplifies the shift. In shadow the same lead is logged and the bearing flown unchanged."""
    lag = LagTurnConfig()
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., gap_aim=CFG, lag_turn=lag, lag_turn_apply=apply)
    pilot.lag_turn_weight = 1.
    pilot.gap_offset_deg = shift
    dh = np.array([np.cos(np.radians(ring+shift)), np.sin(np.radians(ring+shift))])
    goal = pilot._lead(dh, np.array([6., 0., 0.]))                  # flown course 0 deg
    lead = float(np.clip(lag.course_lead*ring, -lag.course_lead_max_deg, lag.course_lead_max_deg))
    assert pilot.lag_turn_lead_deg == pytest.approx(lead)
    assert heading(goal) == pytest.approx(ring+shift+lead if apply else ring+shift)
    assert abs(heading(goal)-ring) <= lag.course_lead_max_deg+CFG.max_shift_deg+1e-9


def ring_switch_with_gap(gap, lag, apply=True, seconds=1.2):
    """Static course 0 deg (a lagging motor that has not turned yet): the ring marker jumps from 0 to 12.7 deg left
    at 10.3 s (as at Minus Two pillar A in minus-brain08-01) and gap samples of +12 deg for that ring arrive from
    10.45 s. Rows per tick: (now, goal heading or None, lead, gap offset, command heading, lag weight)."""
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., gap_aim=CFG if gap else None, gap_apply=apply,
                        lag_turn=LagTurnConfig() if lag else None, lag_turn_apply=apply)
    goals = []
    lead = pilot._lead
    pilot._lead = lambda dh, v: goals.append(lead(dh, v)) or goals[-1]
    ring = lambda t: 0. if t < 10.3 else 12.7
    rows, latest, captures = [], None, np.arange(10.45, 10.+seconds, 1/15)
    for k in range(int(round(seconds/DT))):
        now = 10.+k*DT
        s = senses(position=(0., 0., 5.), velocity=(6., 0., 0.), yaw=0.)
        history.append(now, [0., 0., 5.], s['quat'][0].numpy())
        ready = captures[captures+.05 <= now+1e-9]
        if len(ready):
            latest = sample(float(ready[-1]), 12., ring=12.7)
        a = np.radians(ring(now-.05))
        detection = dict(race_cue=cue_toward([20.*np.cos(a), 20.*np.sin(a), 0.]))
        count = len(goals)
        pilot.update(s, [0., 0., 0.], detection, now-.05, now, gap=latest if gap else None)
        rows.append((now, heading(goals[-1]) if len(goals) > count else None, pilot.lag_turn_lead_deg,
                     pilot.gap_offset_deg, heading(pilot.velocity_command), pilot.lag_turn_weight))
    return pilot, rows


def test_with_both_components_the_lead_equals_the_lag_turn_alone_and_the_shift_is_added():
    """Both stack components on (the --obstacle-stack on default): per tick the lead equals the lag turn's alone and
    the goal is the lag-turn goal rotated by the applied gap shift; the flown aim stays within course_lead_max_deg +
    max_shift_deg of the ring. Shadow logs the same lead. (Version 1 led the shifted bearing: about 15 deg of lead
    and 24 deg beyond the ring in this case.)"""
    lag_pilot, lag = ring_switch_with_gap(gap=False, lag=True)
    both_pilot, both = ring_switch_with_gap(gap=True, lag=True)
    shadow_pilot, shadow = ring_switch_with_gap(gap=True, lag=True, apply=False)
    assert lag_pilot.lag_turn_triggers == both_pilot.lag_turn_triggers == shadow_pilot.lag_turn_triggers >= 1
    assert max(r[3] for r in both) == pytest.approx(12.) and all(r[3] == 0. for r in shadow)
    assert both_pilot.gap_aim.counts['ring_conflicts'] == 0 and both_pilot.gap_aim.counts['flag_conflicts'] == 0
    led = 0
    for a, b, c in zip(lag, both, shadow):
        assert b[2] == pytest.approx(a[2], abs=1e-9) and c[2] == pytest.approx(a[2], abs=1e-9)
        if a[1] is not None and b[1] is not None:
            assert b[1] == pytest.approx(a[1]+b[3], abs=1e-6)
            assert b[1]-12.7 <= LagTurnConfig().course_lead_max_deg+CFG.max_shift_deg+1e-6
            led += b[3] > 11.9 and b[2] > 5.
    assert led > 10, 'the window and the full shift overlap'
    check = [r for r in both if r[0] >= 11.1-1e-9][0]
    assert check[1] == pytest.approx(12.7+check[2]+12., abs=1e-6) and check[2] < 12.7*.6+1e-6


def test_terrain_side_steer_follows_the_looming_governor():
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., gap_aim=CFG)
    pilot.clearance = TtcClearanceGovernor()
    pilot.clearance.limits = lambda *a, **k: (None, None, 0.)
    lr_sample = lambda t: dict(sample(t, 0., lr=.9, kind='clear'))
    history.append(9.99, [0., 0., 5.], np.array([1., 0., 0., 0.]))
    for k in range(60):
        now = 10.+k*DT
        s = senses(position=(0., 0., 5.), velocity=(6., 0., 0.))
        history.append(now, [0., 0., 5.], s['quat'][0].numpy())
        pilot.clearance.climb = 1. if k >= 20 else 0.
        pilot.update(s, [0., 0., 0.], dict(race_cue=cue_toward([20., 0., 0.])), now-.05, now,
                     gap=lr_sample(round(now-.04, 3)) if k % 6 == 0 else None)
    assert pilot.gap_aim.counts['terrain_votes'] >= 2 and pilot.gap_aim.target == -CFG.terrain_side_deg


# ---------------------------------------------------------------------------------------------
# Camera slots, frame slot and worker
# ---------------------------------------------------------------------------------------------
def test_shared_slots_are_contiguous_and_carry_gap_and_stage_samples():
    from haltere.liftoff import camera_process as cp
    assert cp.LOOMING_SLOTS.stop == cp.GAP_SLOTS.start and cp.GAP_SLOTS.stop == cp.STAGE_SLOTS.start
    assert cp.SHARED_SIZE == cp.STAGE_SLOTS.stop and cp.LOOMING_SLOTS.start == 732
    values = [12.5, 7.5, 5., 3.1, 1.2, 1., .4, .031, 17., 1., 0., 9., 1.1, 2.2, 8.8, .9]
    s = cp.gap_sample(values)
    assert s['time'] == 12.5 and s['shift'] == 7.5 and s['kind'] == 'gap' and s['near_on_path'] is True
    assert s['valid'] is True and s['confirmed'] is False and s['ring_deg'] == 17. and s['age'] == pytest.approx(.031)
    none = cp.gap_sample([12.6, 0., 1.]+[float('nan')]*5+[float('nan'), 0., 0., 10.]+[float('nan')]*4)
    assert none['kind'] == 'no_ring' and none['r_peak'] is None and none['near_on_path'] is None
    assert cp.gap_sample([0.]*len(cp.GAP_FIELDS)) is None
    camera = cp.ProcessRetinaCamera.__new__(cp.ProcessRetinaCamera)
    camera.queue, camera.process, camera.looming = Queue(maxsize=2), SimpleNamespace(exitcode=None), False
    camera.gap_spec = dict(placement='camera', stride=1)
    camera.gap_process = camera.gap_queue = None
    camera.data = mp.get_context('spawn').Array('d', cp.SHARED_SIZE, lock=True)
    camera._latest = camera._error = camera._clearance = camera._gap = camera._stages = None
    camera._diagnostics, camera._gap_status = {}, {}
    camera.done = SimpleNamespace(is_set=lambda: False)
    shared = np.frombuffer(camera.data.get_obj(), dtype=np.float64)
    shared[cp.GAP_SLOTS] = values
    shared[cp.STAGE_SLOTS] = [12.5, 20., 5., 30., .2, 10., 1.5, 67., 55.]
    assert camera.gap == s
    assert camera.stages['total'] == 67. and camera.stages['cue_latency'] == 55.


def test_frame_slot_hands_over_the_newest_frame_and_the_cue_follows():
    from haltere.liftoff.gap_stack import FrameSlot, wait_for_cue
    slot = FrameSlot()
    assert slot.read(0) is None and not slot.wait_frame(.001)
    frame = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    assert slot.write(frame, 3.25, now=3.27)
    assert slot.wait_frame(.01) and not slot.wait_frame(.001)       # one signal per written frame, then cleared
    seq, copy, stamp, written = slot.read(0)
    assert seq == 1 and stamp == 3.25 and written == 3.27
    assert np.array_equal(copy, frame)
    assert slot.read(seq) is None
    small = frame[:252, :448].copy()
    slot.write(small, 3.3)
    assert np.array_equal(slot.read(seq)[1], small)
    with pytest.raises(ValueError):
        slot.write(np.zeros((1200, 1280, 3), np.uint8), 3.4)
    assert not slot.fits(np.zeros((1440, 2560, 3), np.uint8)) and slot.fits(frame)
    bgra = np.random.default_rng(1).integers(0, 255, (720, 1280, 4), dtype=np.uint8)
    view = bgra[:, :, :3][:, :, ::-1]                      # as a screen grab arrives: a strided view
    slot.write(view, 3.5)
    assert np.array_equal(slot.read(0)[1], view)
    stop = SimpleNamespace(is_set=lambda: False)
    assert slot.cue_for(3.3) == ('pending', None)
    assert wait_for_cue(slot, 3.3, stop, timeout=.01) == (False, None)
    assert slot.publish_cue(3.3, dict(u=.4, v=.5, edge=False, aim_u=.45))
    assert wait_for_cue(slot, 3.3, stop) == (True, dict(u=.4, v=.5, edge=False, aim_u=.45))
    assert slot.cue_for(3.25) == ('replaced', None)
    slot.publish_cue(3.3, None)
    assert slot.cue_for(3.3) == ('published', None)
    assert slot.skips() == dict(frames=0, cues=0)


def test_the_camera_side_of_the_slot_never_waits_for_the_depth_process():
    """A depth process that holds a slot lock and is not scheduled (a starved low-priority process) must not stall
    the camera: its writes skip at once (counted) and its signals are semaphore releases, which never block."""
    import threading
    import time as _time
    from haltere.liftoff.gap_stack import FrameSlot, wait_for_cue
    slot = FrameSlot()
    frame = np.zeros((252, 448, 3), np.uint8)
    for lock, call in ((slot.lock, lambda: slot.write(frame, 1.)),
                       (slot.cue_lock, lambda: slot.publish_cue(1., dict(u=.5, v=.5, edge=False)))):
        assert lock.acquire(False)                  # the depth side holds it
        try:
            begin = _time.perf_counter()
            assert call() is False
            assert _time.perf_counter()-begin < .002
        finally:
            lock.release()
    assert slot.skips() == dict(frames=1, cues=1)
    for _ in range(50):                             # nobody waiting: signals accumulate without blocking
        assert slot.write(frame, 2.) and slot.publish_cue(2., None)
    assert slot.wait_frame(.01) and not slot.wait_frame(.001)
    # a depth side waiting for a cue wakes on the camera's signal
    result = {}
    waiter = threading.Thread(target=lambda: result.update(
        cue=wait_for_cue(slot, 3., SimpleNamespace(is_set=lambda: False), timeout=2.)))
    waiter.start()
    _time.sleep(.05)
    slot.publish_cue(3., dict(u=.25, v=.5, edge=True))
    waiter.join(2.)
    assert result['cue'] == (True, dict(u=.25, v=.5, edge=True, aim_u=.25))


def test_the_depth_process_shares_no_lock_or_event_with_the_camera():
    """Process placement: the depth process gets the slot, its own sample array and its own stop event, never the
    camera's shared data (whose lock the camera takes blocking) or the camera's done event; the controller reads
    its samples without blocking."""
    from haltere.liftoff import camera_process as cp
    spec = dict(placement='process', stride=1)
    camera = cp.ProcessRetinaCamera(gate_sensor=dict(focal_320=100., tilt_deg=30.), race_cues=True, gap=spec)
    args = camera.gap_process._args
    assert args[0] is camera.frame_slot and args[1] is camera.gap_out and args[3] is camera.gap_stop
    assert all(a is not camera.data and a is not camera.done for a in args)
    assert camera.frame_slot in camera.process._args          # the camera writes frames and cues into the slot
    values = [12.5, 7.5, 5., 3.1, 1.2, 1., .4, .031, 17., 1., 0., 9., 1.1, 2.2, 8.8, .9]
    np.frombuffer(camera.gap_out.get_obj(), dtype=np.float64)[:] = values
    assert camera.gap == cp.gap_sample(values)
    import threading
    held, release = threading.Event(), threading.Event()

    def hold():                                             # the depth process writing (another thread: an RLock)
        with camera.gap_out.get_lock():
            held.set()
            release.wait(2.)
    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert held.wait(2.)
        camera._gap = None
        assert camera.gap is None                           # a held lock skips this poll, it never waits
    finally:
        release.set()
        holder.join(2.)
    assert camera.gap == cp.gap_sample(values)
    camera.stop()
    assert camera.done.is_set() and camera.gap_stop.is_set()


def test_depth_worker_raises_its_priority_and_publishes_to_its_own_array(monkeypatch):
    """gap_process_worker end to end in this process, with a fake frame worker: it sets its priority class first
    (spawned from a below-normal runner it would inherit below normal), reads the slot's frame and cue, and
    writes its sample to its own array; its status reports the priority."""
    import queue
    import threading
    import time as _time
    import cv2
    from haltere.liftoff import camera_process as cp
    from haltere.liftoff import gap_stack, scheduling
    from haltere.liftoff.geometry_shadow import MotionBuffer
    calls = []
    monkeypatch.setattr(scheduling, 'flight_process_priority',
                        lambda: calls.append(1) or dict(applied=True, previous=0x4000, current=0x8000))

    class FakeWorker:
        def __init__(self, spec):
            self.spec = spec

        def perceive(self, frame):
            assert frame.shape == (252, 448, 3)
            return dict(overlay_ms=1., depth_ms=2.)

        def process(self, frame, capture_time, cue, pose, frame_ms, seq, perceived):
            return [capture_time, 7.5 if cue else 0., 5., 3., 1., 1., .4, .03, 17., 1., 0., seq, frame_ms, 1., 2.,
                    .5 if pose is not None else -1.]

        def status(self):
            return dict(ready=True)

    monkeypatch.setattr(gap_stack, 'GapFrameWorker', FakeWorker)

    class Status(queue.Queue):
        def cancel_join_thread(self):
            pass
    threads, cv2_threads = torch.get_num_threads(), cv2.getNumThreads()
    slot, motion, status = gap_stack.FrameSlot(), MotionBuffer(), Status(maxsize=2)
    gap_out = mp.get_context('spawn').Array('d', len(cp.GAP_FIELDS), lock=True)
    stop = mp.get_context('spawn').Event()
    now = _time.monotonic()
    for k in range(3):       # observed poses around the capture time (marked available later: never stale here)
        motion.publish(now+60., now-.02+.01*k, 0., [0., 0., 5.], [1., 0., 0., 0.], [6., 0., 0.], [0., 0., 0.])
    worker = threading.Thread(target=gap_stack.gap_process_worker,
                              args=(slot, gap_out, motion, stop, status, dict(stride=1)))
    worker.start()
    try:
        slot.write(np.zeros((720, 1280, 3), np.uint8), now-.01)
        slot.publish_cue(now-.01, dict(u=.5, v=.5, edge=False, aim_u=.5))
        shared = np.frombuffer(gap_out.get_obj(), dtype=np.float64)
        deadline = _time.monotonic()+5.
        while not shared[0] and _time.monotonic() < deadline:
            _time.sleep(.01)
    finally:
        stop.set()
        worker.join(5.)
        torch.set_num_threads(threads)
        cv2.setNumThreads(cv2_threads)
    assert not worker.is_alive() and calls == [1]
    sample = cp.gap_sample(shared.copy())
    assert sample['time'] == pytest.approx(now-.01) and sample['shift'] == 7.5 and sample['seq'] == 1.
    assert sample['decide_ms'] == .5                                 # the pose at the capture time was found
    packets = []
    while not status.empty():
        packets.append(status.get_nowait())
    assert packets and packets[-1]['gap']['priority'] == dict(applied=True, previous=0x4000, current=0x8000)
    assert packets[-1]['gap']['camera_skips'] == dict(frames=0, cues=0) and 'error' not in packets[-1]


class FakeDepth:
    """Relative disparity with a near column at a given image column band (a pillar)."""
    def __init__(self, columns):
        self.columns = columns
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        d = np.ones((1, 36, 64), np.float32)
        d[0, :, self.columns[0]:self.columns[1]] = 4.
        return d


def worker_spec():
    from haltere.liftoff.gap_stack import REPO_ROOT
    return dict(gap_cue_config=str(REPO_ROOT/'configs'/'obstacles'/'gap_cue.json'),
                response_models=str(REPO_ROOT/'configs'/'obstacles'/'response_models.json'),
                motor='brain08', focal_320=100., tilt_deg=30.)


def level_pose():
    return np.array([np.cos(np.radians(15)), 0., np.sin(np.radians(15)), 0.]), np.array([6., 0., 0.])


def test_worker_turns_a_near_column_right_of_the_ring_into_a_left_shift():
    from haltere.liftoff import camera_process as cp
    from haltere.liftoff.gap_stack import GapFrameWorker
    depth = FakeDepth((33, 38))      # just right of the image centre
    worker = GapFrameWorker(worker_spec(), depth=depth)
    frame = np.full((252, 448, 3), 90, np.uint8)
    q, v = level_pose()
    # nose pitched down 30 deg: the camera axis is level, the ring at the image centre is at the horizon
    clock = lambda: 5.02
    values = worker.process(frame, 5., dict(u=.5, v=.5, edge=False), (q, v), frame_ms=1., clock=clock)
    s = cp.gap_sample(values)
    assert s['valid'] and s['kind'] in ('gap', 'occluded') and s['shift'] > 2.
    assert s['ring_deg'] == pytest.approx(0., abs=1.) and s['depth_ms'] >= 0 and s['frame_ms'] == 1.
    assert depth.calls == 1
    edge = cp.gap_sample(worker.process(frame, 5.01, dict(u=.99, v=.5, edge=True), (q, v), clock=clock))
    assert edge['kind'] == 'no_ring' and not edge['valid'] and depth.calls == 1      # no depth without a ring
    nopose = cp.gap_sample(worker.process(frame, 5.015, dict(u=.5, v=.5, edge=False), None, clock=clock))
    assert nopose['kind'] == 'no_pose' and depth.calls == 1
    status = worker.status()
    assert status['frames'] == 3 and status['depth_frames'] == 1 and status['ready']


# ---------------------------------------------------------------------------------------------
# Runner flags, logs and declarations
# ---------------------------------------------------------------------------------------------
def args(**kw):
    base = dict(obstacle_stack=None, gap_cue=None, lag_turn=None, wall_pilot=None, pilot_profile='fast',
                looming_brake=True)
    base.update(kw)
    return SimpleNamespace(**base)


def test_obstacle_stack_flags_resolve_to_off_by_default_components():
    from haltere.liftoff.visual_brain import LAG_TURN_DECLARATION, WALL_PILOT_DECLARATION, resolve_obstacle_stack
    default, wall = str(LAG_TURN_DECLARATION), str(WALL_PILOT_DECLARATION)
    assert resolve_obstacle_stack(args()) == dict(mode=None, gap=False, lag_turn=None, apply=True, wall_pilot=None)
    assert resolve_obstacle_stack(args(lag_turn='on'))['lag_turn'] == default
    assert resolve_obstacle_stack(args(lag_turn='x.json'))['lag_turn'] == 'x.json'
    assert resolve_obstacle_stack(args(obstacle_stack='on')) == dict(mode='on', gap=True, lag_turn=default, apply=True,
                                                                     wall_pilot=wall)
    assert resolve_obstacle_stack(args(obstacle_stack='shadow')) == dict(mode='shadow', gap=True, lag_turn=default,
                                                                         apply=False, wall_pilot=wall)
    assert resolve_obstacle_stack(args(obstacle_stack='on', gap_cue='off'))['gap'] is False
    assert resolve_obstacle_stack(args(obstacle_stack='on', lag_turn='off'))['lag_turn'] is None
    assert resolve_obstacle_stack(args(obstacle_stack='on', wall_pilot='off'))['wall_pilot'] is None
    assert resolve_obstacle_stack(args(obstacle_stack='shadow', wall_pilot='on'))['wall_pilot'] == wall
    with pytest.raises(ValueError, match='obstacle stack'):
        resolve_obstacle_stack(args(gap_cue='on'))
    with pytest.raises(ValueError, match='obstacle stack'):
        resolve_obstacle_stack(args(wall_pilot='on'))
    with pytest.raises(ValueError, match='on or off'):
        resolve_obstacle_stack(args(obstacle_stack='on', wall_pilot='maybe'))
    with pytest.raises(ValueError, match='looming'):
        resolve_obstacle_stack(args(obstacle_stack='on', looming_brake=False))
    with pytest.raises(ValueError, match='fast'):
        resolve_obstacle_stack(args(obstacle_stack='shadow', pilot_profile='standard'))


def test_runner_refuses_the_gap_aim_outside_the_fast_pilot():
    from haltere.liftoff.visual_brain import VisualController
    with pytest.raises(ValueError, match='fast pilot'):
        VisualController('missing.pt', 'missing.json', 'cpu', pilot_assistance='race-cue', gap_pilot=CFG)


def test_log_columns_and_rows():
    from haltere.liftoff.visual_brain import GAP_COLUMNS, STAGE_COLUMNS, gap_row, stage_row
    required = {'gap_shift', 'gap_kind', 'gap_r_peak', 'gap_r_ring', 'gap_age', 'near_on_path', 'gap_applied',
                'gap_lr', 'gap_conflict'}
    assert required <= set(GAP_COLUMNS) and 'cam_cue_latency_ms' in STAGE_COLUMNS and 'cam_gap_ms' in STAGE_COLUMNS
    assert len(gap_row(None, None, 1.)) == len(GAP_COLUMNS) and len(stage_row(None)) == len(STAGE_COLUMNS)
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., gap_aim=CFG)
    s = dict(sample(9.9, 5.), r_peak=3., r_ring=1., near_on_path=True, confirmed=True, seq=4., overlay_ms=2.,
             depth_ms=9., decide_ms=1., age=.03)
    row = dict(zip(GAP_COLUMNS, gap_row(s, pilot, 10.)))
    assert row['gap_shift'] == 5. and row['gap_kind'] == 'gap' and row['gap_age'] == pytest.approx(.1)
    assert row['near_on_path'] == 1 and row['gap_applied'] == 0. and row['gap_latency_ms'] == pytest.approx(30.)
    stages = dict(frame_time=9.9, capture=20., preprocess=5., inference=30., publish=.2, looming=10., gap=1.,
                  total=66., cue_latency=55.)
    assert dict(zip(STAGE_COLUMNS, stage_row(stages)))['cam_cue_latency_ms'] == 55.


def test_repository_gap_pilot_declaration_parses_and_points_at_frozen_configs():
    from haltere.liftoff.gap_stack import GAP_PILOT_PATH, REPO_ROOT, config_sha256, load_gap_pilot
    from haltere.vision import gap_cue as gc
    declaration, digest = load_gap_pilot(GAP_PILOT_PATH)          # flights refuse an unfrozen or edited file
    assert GapAimConfig.from_dict(declaration['pilot']) == CFG      # the declared values are the defaults
    runtime = declaration['runtime']
    assert runtime['placement'] in ('camera', 'process') and runtime['stride'] in (1, 2)
    _, cue_config, _ = gc.load_config(REPO_ROOT/runtime['gap_cue_config'], require_frozen=True)
    assert cue_config['relative_depth']['input_hw'] == [336, 602] and cue_config['version'] == 2
    models = gc.load_response_models(REPO_ROOT/runtime['response_models'])
    assert set(runtime['response_model_for_contract'].values()) <= set(models)
    assert config_sha256(declaration) == digest and declaration['frozen'] is True and declaration['sha256'] == digest
    assert runtime['placement'] == 'process' and runtime['stride'] == 1       # chosen by the G8 bench


def test_gap_pilot_declaration_edits_are_refused(tmp_path):
    from haltere.liftoff.gap_stack import GAP_PILOT_PATH, load_gap_pilot
    declaration = json.loads(GAP_PILOT_PATH.read_text(encoding='utf-8'))
    declaration['pilot']['slew_deg_s'] = 80.
    path = tmp_path/'edited.json'
    path.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match='changed'):
        load_gap_pilot(path)
    for key in ('frozen', 'frozen_at', 'sha256'):
        declaration.pop(key)
    path.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match='not frozen'):
        load_gap_pilot(path)


def test_version_1_declarations_are_kept_verbatim_and_refused_at_runtime():
    """gap_pilot.json and lag_turn.json version 2 keep version 1's values; the v1 files are kept verbatim for
    provenance (their scored results) and the runtime refuses them (their rules are no longer the code's)."""
    from haltere.liftoff.fast_race_cue import LAG_TURN_VERSION
    from haltere.liftoff.gap_stack import GAP_PILOT_PATH, GAP_PILOT_VERSION, REPO_ROOT, config_sha256, load_gap_pilot
    from haltere.liftoff.visual_brain import LAG_TURN_DECLARATION, load_lag_turn_declaration
    assert GAP_PILOT_VERSION == LAG_TURN_VERSION == 2
    ob = REPO_ROOT/'configs'/'obstacles'
    for current, old, loader, first in ((GAP_PILOT_PATH, ob/'gap_pilot_v1.json', load_gap_pilot, 'e704a3ba0d3d'),
                                        (LAG_TURN_DECLARATION, ob/'lag_turn_v1.json', load_lag_turn_declaration,
                                         '94315b4ddc4a')):
        v1 = json.loads(old.read_text(encoding='utf-8'))
        v2, digest = loader(current)
        assert v1['version'] == 1 and v1['frozen'] is True and config_sha256(v1) == v1['sha256']
        assert v1['sha256'].startswith(first) and v2['version'] == 2 and v2['frozen'] is True
        assert v2['previous_versions'][0]['sha256'] == v1['sha256'] and v2['previous_versions'][0]['version'] == 1
        assert v2['change'] and digest == v2['sha256']
        for key in ('pilot', 'runtime', 'contracts', 'response_models'):
            assert v1.get(key) == v2.get(key)                       # values unchanged
        with pytest.raises(ValueError, match='version'):
            loader(old)
    assert 'interaction_with_gap_aim' in load_lag_turn_declaration(LAG_TURN_DECLARATION)[0]
    notes = load_gap_pilot()[0]['runtime_notes']['placement']
    assert 'above normal' in notes and 'normal priority' not in notes


def test_gap_spec_maps_each_fast_contract_to_its_response_model():
    from haltere.liftoff.gap_stack import GAP_PILOT_PATH, gap_spec, load_gap_pilot
    declaration, digest = load_gap_pilot(GAP_PILOT_PATH)
    sensor = dict(focal_320=100., tilt_deg=30.)
    brain = gap_spec(declaration, 'fast_velocity_brain_v1', sensor, digest=digest)
    pd = gap_spec(declaration, 'fast_velocity_pd_v1', sensor, digest=digest)
    assert (brain['motor'], pd['motor']) == ('brain08', 'fast_pd') and brain['input_hw'] == [336, 602]
    assert brain['gap_cue_version'] == 2 and brain['placement'] == 'process' and brain['gap_pilot_sha256'] == digest
    json.dumps(brain)                                                   # picklable, loggable
    with pytest.raises(ValueError, match='response model'):
        gap_spec(declaration, 'motor_tracking_teacher_v1', sensor)
    with pytest.raises(ValueError, match='calibrated camera'):
        gap_spec(declaration, 'fast_velocity_pd_v1', None)


def test_gap_cue_v2_only_changes_the_depth_input_of_v1():
    from haltere.liftoff.gap_stack import REPO_ROOT
    from haltere.vision import gap_cue as gc
    p1, v1, sha1 = gc.load_config(REPO_ROOT/'configs'/'obstacles'/'gap_cue_v1.json', require_frozen=True)
    p2, v2, _ = gc.load_config(require_frozen=True)
    assert sha1.startswith('26ae946693a2') and v2['previous_versions'][0]['sha256'] == sha1
    assert p1 == p2 and v1['mask_layers'] == v2['mask_layers']
    r1, r2 = dict(v1['relative_depth']), dict(v2['relative_depth'])
    assert (r1.pop('input_hw'), r2.pop('input_hw')) == ([252, 448], [336, 602])
    assert {k: v for k, v in r1.items() if k != 'precision'} == \
        {k: v for k, v in r2.items() if k not in ('precision', 'preprocessing')}


def test_replay_capture_serves_the_gameplay_crop_in_real_time(tmp_path):
    import cv2
    from haltere.liftoff.camera_replay import ReplayCapture
    path = tmp_path/'clip.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 18., (1928, 720))
    for k in range(20):
        frame = np.zeros((720, 1928, 3), np.uint8)
        frame[:, 648:] = 10*k              # gameplay part changes per frame; the panel stays black
        writer.write(frame)
    writer.release()
    anchor = tmp_path/'anchor.txt'
    capture = ReplayCapture.from_spec(f'{path}?start=0.5&pad=0&anchor={anchor}')
    with capture:
        t, first = capture.read()
        assert first.shape == (720, 1280, 3) and capture.video_time(t) == .5    # no anchor yet: the start frame
        anchor.write_text(repr(t-.3))
        t2, later = capture.read()
        assert capture.video_time(t2) >= .8 and later.mean() > first.mean()
