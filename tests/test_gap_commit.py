"""Gap pilot versions 4 and 5: side commitment near an obstacle and the terrain-vote rules (haltere.liftoff.gap_aim).

Pins: every new rule is off by default and the version 2 gap aim is unchanged by the new fields; a close obstacle
confirmation commits, holds its side and largest shift through flicker and 'clear' frames, is refreshed by close
samples and by any obstacle vote (the obstacle is still ahead), switches only on much stronger opposite evidence
early enough, and is released by the hold timeout, its maximum, and ring or flag conflicts; version 5 (the declared
one) starts commitments only on one-sided evidence (not on 'occluded' decisions, as at Straw Bale gate arches) and
holds 0.3 s; a terrain episode yields to obstacle evidence; terrain votes only for confirmed rising ground with a
vertical guard; shadow never flies it; the pillar C sample stream of minus-fast6-vg-02 (development case) is held on
its free side under versions 4 and 5 and latched out under version 2; the offline evaluation's scoring helpers. None
of it is flight evidence.
"""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import FastRaceCue, TtcClearanceGovernor, VerticalGuardConfig
from haltere.liftoff.gap_aim import GapAim, GapAimConfig
from tests.test_fast_race_cue import cue_toward
from tests.test_gap_pilot import at, fly_with_gap, run_aim, sample
from tests.test_visual_assistance import SENSOR, senses

V2 = GapAimConfig()
V4 = replace(V2, commit=True, terrain_yields=True, terrain_rising_only=True)
V5 = replace(V4, commit_hold_s=.3, commit_occluded=False)
DT = .01


def close(t, shift, ring=0., kind='gap', near=True, lr=float('nan')):
    return dict(sample(t, shift, ring=ring, kind=kind, lr=lr), near_on_path=near)


def stream(times, shifts, **kw):
    times = list(times)
    shifts = list(shifts)*len(times) if len(shifts) == 1 else list(shifts)
    return [close(t, s, **kw) for t, s in zip(times, shifts)]


# ---------------------------------------------------------------------------------------------
# configuration and default behaviour
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize('overrides', [dict(commit=1), dict(terrain_yields='yes'), dict(switch_votes=2.5),
                                       dict(switch_min_deg=20.), dict(commit_hold_s=3.), dict(switch_window_s=0.),
                                       dict(commit_max_s=float('nan')), dict(commit_occluded=0)])
def test_new_parameters_validate(overrides):
    with pytest.raises(ValueError):
        GapAimConfig(**overrides)


def test_every_new_rule_is_off_by_default_and_each_declaration_reproduces_its_version():
    from haltere.liftoff.gap_stack import REPO_ROOT
    assert not (V2.commit or V2.terrain_yields or V2.terrain_rising_only)
    ob = REPO_ROOT/'configs'/'obstacles'
    load = lambda name: GapAimConfig.from_dict(json.loads((ob/name).read_text(encoding='utf-8'))['pilot'])
    assert load('gap_pilot_v2.json') == V2 and load('gap_pilot_v4.json') == V4 and load('gap_pilot.json') == V5


def test_the_version_2_rule_ignores_the_version_4_parameters_and_near_on_path():
    """With the switches off, the commitment parameters and the samples' near_on_path change nothing."""
    other = replace(V2, commit_hold_s=.1, commit_max_s=.2, switch_votes=1, switch_min_deg=2., switch_window_s=5.)
    samples = (stream([1.0, 1.066], [8., 8.]) + [close(1.133, 0., kind='clear', near=False)]
               + stream([1.2, 1.266, 1.333, 1.4], [-8., -8., -9., -9.]) + stream([2.2, 2.266], [-3., -4.], near=False))
    rows = [run_aim(GapAim(c), s, 3., start=.95)
            for c, s in ((V2, samples), (other, samples),
                         (V2, [dict(x, near_on_path=None) for x in samples]))]
    assert rows[0] == rows[1] == rows[2]
    assert 'commits' not in GapAim(V2).counts and 'commit_seconds' not in GapAim(V2).metadata()


# ---------------------------------------------------------------------------------------------
# side commitment
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize('config', [V4, V5], ids=['v4', 'v5'])
def test_a_close_confirmation_commits_and_holds_the_largest_shift_through_flicker(config):
    aim = GapAim(config)
    samples = (stream([1.0, 1.066], [-3., -2.5]) + [close(1.133, 0., kind='clear', near=False)]
               + stream([1.2], [-6.]) + [close(1.266, 0., kind='clear', near=False), close(1.333, 0., kind='clear',
                                                                                          near=False)]
               + stream([1.4], [-2.2]))
    rows = run_aim(aim, samples, 1.6, start=.95)
    assert at(rows, 1.1)[1] == -3.                        # committed: the larger confirming shift, not the newest
    assert aim.counts['commits'] == 1
    assert all(r[1] < 0 for r in rows if r[0] >= 1.1)     # never released or decayed through the clear frames
    assert at(rows, 1.3)[1] == -6. and at(rows, 1.5)[1] == -6.    # grows to the largest, never shrinks
    v2 = run_aim(GapAim(V2), samples, 1.6, start=.95)
    assert at(v2, 1.3)[1] == 0. and at(v2, 1.5)[1] == 0.  # version 2 released on the clear frames


def test_without_close_evidence_the_confirmation_does_not_commit():
    samples = [dict(sample(1.0+k*.066, 8.), near_on_path=False) for k in range(3)]
    for kind in ('gap', 'clear'):
        aim = GapAim(V4)
        rows = run_aim(aim, [dict(s, kind=kind) for s in samples], 1.8, start=.95)
        assert aim.counts['commits'] == 0 and not aim.committed
        assert rows == run_aim(GapAim(V2), [dict(s, kind=kind) for s in samples], 1.8, start=.95)


def test_an_occluded_decision_starts_a_commitment_only_in_version_4_and_refreshes_one_in_both():
    occluded = [dict(sample(1.0+k*.066, -9., kind='occluded'), near_on_path=True) for k in range(2)]
    aim = GapAim(V4)
    run_aim(aim, occluded, 1.2, start=.95)
    assert aim.committed and aim.commit_side == -1
    aim = GapAim(V5)
    rows = run_aim(aim, occluded, 1.2, start=.95)
    assert not aim.committed and aim.counts['commits'] == 0
    assert rows == run_aim(GapAim(V2), occluded, 1.2, start=.95)          # version 2's confirmation, no hold
    # started by one-sided 'gap' evidence, a commitment is refreshed and grown by same-side occluded votes
    aim = GapAim(V5)
    rows = run_aim(aim, stream([1.0, 1.066], [-3., -3.]) + [dict(sample(1.2+k*.25, -9., kind='occluded'),
                                                                 near_on_path=False) for k in range(4)],
                   2.2, start=.95)
    assert aim.counts['commits'] == 1 and all(r[1] < 0 for r in rows if 1.1 <= r[0] <= 2.05)
    assert at(rows, 1.4)[1] == -9.


def test_version_5_does_not_commit_on_a_gate_arch_flicker():
    """Straw Bale gates (straw-brain08-04, 1.5 s before a checkpoint): the inflatable arch's top crosses the band at
    the ring and the decision flickers 'occluded' +-12 deg with near_on_path. Version 4 committed and held 12 deg
    into the gate; version 5 does not commit and flies version 2's confirmation."""
    shifts = [-12., -12., -12., 12., -12., -12., -12., -12., -2.5, 0., .5, 2.5, .5, .5]
    kinds = ['occluded']*9+['gap']*5
    t = 1.0+np.arange(len(shifts))*.056
    flicker = [dict(sample(float(tk), s, kind=k, ring=-91.), near_on_path=k == 'occluded')
               for tk, s, k in zip(t, shifts, kinds)]
    v4 = GapAim(V4)
    rows4 = run_aim(v4, flicker, 2.2, start=.95)
    v5 = GapAim(V5)
    rows5 = run_aim(v5, flicker, 2.2, start=.95)
    assert v4.counts['commits'] == 1 and at(rows4, 1.9)[1] == -12.       # still held 0.5 s after the last evidence
    assert v5.counts['commits'] == 0 and rows5 == run_aim(GapAim(V2), flicker, 2.2, start=.95)
    assert at(rows5, 1.9)[2] == 0.


def test_the_hold_expires_without_refreshing_evidence_and_the_shift_decays():
    aim = GapAim(V4)
    rows = run_aim(aim, stream([1.0, 1.066], [5., 5.]), 2.5, start=.95)
    refresh = 1.066+.03                                   # the second sample arrives 0.03 s after capture
    assert all(r[1] == 5. for r in rows if refresh <= r[0] <= refresh+V4.commit_hold_s-.01)
    released = refresh+V4.commit_hold_s+.02
    assert at(rows, released)[1] == 0. and aim.counts['commit_hold_releases'] == 1
    assert at(rows, released+V4.decay_s+.03)[2] == pytest.approx(0.)


def test_close_samples_without_a_vote_refresh_the_hold_and_the_maximum_ends_it():
    aim = GapAim(V4)
    samples = stream([1.0, 1.066], [5., 5.]) + stream(np.arange(1.13, 4.5, .066), [1.])  # close, below active_deg
    rows = run_aim(aim, samples, 4.5, start=.95)
    start = 1.066+.03
    assert all(r[1] == 5. for r in rows if start <= r[0] <= start+V4.commit_max_s-.01)
    assert at(rows, start+V4.commit_max_s+.02)[1] == 0. and aim.counts['commit_max_releases'] == 1
    # a weak sample that is not close refreshes nothing
    aim = GapAim(V4)
    rows = run_aim(aim, stream([1.0, 1.066], [5., 5.]) + stream(np.arange(1.13, 2.5, .066), [1.], near=False),
                   2.5, start=.95)
    assert at(rows, 1.1+V4.commit_hold_s+.02)[1] == 0.


def test_a_switch_needs_much_stronger_opposite_evidence_early_enough():
    # early: three consecutive opposite votes of >= switch_min_deg inside switch_window_s
    aim = GapAim(V4)
    rows = run_aim(aim, stream([1.0, 1.05, 1.1, 1.15, 1.2], [3., 3., -8., -9., -10.]), 1.5, start=.95)
    assert aim.counts['commit_switches'] == 1 and at(rows, 1.3)[1] == -10.
    # weak opposite votes (below switch_min_deg) never switch: they are blocked while committed and, since the
    # obstacle is still ahead, keep the commitment alive (version 3 released it after commit_hold_s)
    aim = GapAim(V4)
    rows = run_aim(aim, stream([1.0, 1.05] + [1.1+.05*k for k in range(30)], [3., 3.]+[-4.]*30), 2.6, start=.95)
    assert aim.counts['commit_switches'] == 0 and all(r[1] == 3. for r in rows if r[0] > 1.1)
    assert aim.counts['commit_blocks'] == 1 and aim.counts['commit_hold_releases'] == 0
    # strong but late (the vehicle has already responded): blocked, side held
    aim = GapAim(V4)
    late = stream([1.0, 1.05, 1.1, 1.15, 1.2, 1.25], [3.]*6) + stream([1.3+.05*k for k in range(6)], [-10.]*6)
    rows = run_aim(aim, late, 1.7, start=.95)
    assert aim.counts['commit_switches'] == 0 and all(r[1] > 0 for r in rows if r[0] > 1.1)
    # an interrupted run of opposite votes does not count as consecutive
    aim = GapAim(V4)
    run_aim(aim, stream([1.0, 1.05, 1.1, 1.13, 1.16, 1.19], [3., 3., -8., 0., -9., -9.]), 1.4, start=.95)
    assert aim.counts['commit_switches'] == 0


def test_ring_and_flag_conflicts_release_the_commitment():
    aim = GapAim(V4)
    run_aim(aim, stream([1.0, 1.066], [5., 5.]), 1.2, start=.95)
    assert aim.committed and aim.reconcile_ring(30., 1.2)      # the checkpoint switched: the ring was passed
    assert not aim.committed and aim.target == aim.applied == 0.
    assert aim.counts['commit_conflict_releases'] == 1
    aim = GapAim(V4)
    run_aim(aim, stream([1.0, 1.066], [5., 5.]), 1.2, start=.95)
    assert aim.flag_conflict(-3., 1.2) and not aim.committed


def test_terrain_votes_are_ignored_while_committed_and_a_terrain_episode_yields():
    lr = lambda t: dict(sample(t, 0., lr=.9, kind='clear'), near_on_path=False)     # left nearer: terrain right
    # a terrain episode (right) and then obstacle evidence (left): version 2 latches it out, version 4 takes it
    samples = [lr(1.0+k*.066) for k in range(4)] + stream([1.3, 1.366, 1.433, 1.5], [6., 6., 7., 7.])
    v2 = run_aim(GapAim(V2), samples, 1.7, start=.95, terrain=True)
    aim = GapAim(V4)
    v3 = run_aim(aim, samples, 1.7, start=.95, terrain=True)
    assert at(v2, 1.2)[1] == at(v3, 1.2)[1] == -V2.terrain_side_deg
    assert at(v2, 1.45)[1] == 0.                          # version 2: the terrain latch blocks the obstacle side
    assert at(v3, 1.45)[1] == 6. and aim.counts['terrain_yields'] == 1 and aim.committed
    # committed: later terrain votes do nothing
    aim = GapAim(V4)
    rows = run_aim(aim, stream([1.0, 1.066], [5., 5.]) + [dict(lr(1.1+k*.066), near_on_path=True) for k in range(5)],
                   1.5, start=.95, terrain=True)
    assert all(r[1] == 5. for r in rows if r[0] >= 1.1)


# ---------------------------------------------------------------------------------------------
# pillar C (minus-fast6-vg-02, the development case): the live sample stream, capture and receipt times
# ---------------------------------------------------------------------------------------------
# (capture, receipt, shift, kind, near_on_path, lr); the governor's (gentle) terrain climb ran 21.60-22.60 s
PILLAR_C = [
    (21.537, 21.598, 0., 'gap', 0, -.65), (21.600, 21.658, 0., 'gap', 0, -1.04), (21.657, 21.718, 0., 'gap', 0, -1.52),
    (21.724, 21.778, 0., 'gap', 0, -2.17), (21.782, 21.838, 0., 'gap', 0, -2.1), (21.843, 21.898, 0., 'gap', 0, -1.88),
    (21.903, 21.958, 0., 'gap', 0, -2.3), (21.965, 22.018, 0., 'gap', 0, -2.2), (22.027, 22.088, 0., 'gap', 0, -2.27),
    (22.075, 22.138, 0., 'gap', 0, -1.88), (22.142, 22.198, 0., 'gap', 0, -1.98), (22.205, 22.258, 0., 'gap', 0, -2.11),
    (22.267, 22.328, -.29, 'gap', 1, -1.89), (22.331, 22.388, -3.24, 'gap', 1, -1.86),
    (22.388, 22.448, -2.35, 'gap', 1, -1.34), (22.441, 22.498, -3.27, 'gap', 1, -1.04),
    (22.504, 22.558, -4.21, 'gap', 1, -.77), (22.574, 22.628, 0., 'clear', 0, -1.61),
    (22.629, 22.678, -5.82, 'gap', 1, -.83), (22.685, 22.738, 0., 'clear', 0, -1.19),
    (22.751, 22.808, 0., 'clear', 0, -1.2), (22.815, 22.868, -8.15, 'occluded', 1, -.89),
    (22.877, 22.938, -9.54, 'occluded', 1, -2.06), (22.923, 22.988, -9.53, 'occluded', 1, -2.),
    (22.995, 23.058, -5.11, 'gap', 1, -2.5), (23.053, 23.118, -12., 'occluded', 1, -1.31),
    (23.121, 23.198, -12., 'occluded', 1, -2.24), (23.199, 23.278, -12., 'occluded', 1, -.9),
    (23.273, 23.348, 0., 'clear', 0, .66), (23.347, 23.418, 0., 'clear', 0, .29)]
IMPACT = 23.438


def replay_pillar_c(config, terrain):
    """GapAim over the logged pillar C samples at 100 Hz; `terrain(now)` is the pilot's terrain flag."""
    aim = GapAim(config)
    rows, k, latest = [], 0, None
    for i in range(int(round((IMPACT-21.5)/DT))+1):
        now = 21.5+i*DT
        while k < len(PILLAR_C) and PILLAR_C[k][1] <= now+1e-9:
            c, _, shift, kind, near, lr = PILLAR_C[k]
            latest = dict(time=c, shift=shift, kind=kind, valid=True, near_on_path=bool(near), lr=lr, ring_deg=85.)
            k += 1
        aim.ingest(latest, now, terrain=terrain(now))
        aim.step(now, DT)
        rows.append((now, aim.target, aim.applied))
    return aim, rows


def test_pillar_c_version_2_steers_toward_the_pillar_and_latches_out_the_free_side():
    aim, rows = replay_pillar_c(V2, lambda now: 21.6 <= now <= 22.6)
    assert max(r[2] for r in rows) == pytest.approx(V2.terrain_side_deg)          # left, toward the pillar
    first_right = min(r[0] for r in rows if r[1] < 0)
    assert IMPACT-first_right < .45 and aim.counts['latch_blocks'] >= 1           # 0.39 s before the impact


@pytest.mark.parametrize('version', [V4, V5], ids=['v4', 'v5'])
@pytest.mark.parametrize('terrain', ['gentle climb suppressed', 'terrain votes as version 2'])
def test_pillar_c_versions_4_and_5_hold_the_free_side_from_confirmation_to_the_impact(terrain, version):
    flag = (lambda now: False) if terrain.startswith('gentle') else (lambda now: 21.6 <= now <= 22.6)
    config = version if terrain.startswith('gentle') else replace(version, terrain_rising_only=False)
    aim, rows = replay_pillar_c(config, flag)
    first_right = min(r[0] for r in rows if r[1] < 0)
    assert IMPACT-first_right >= .9                                               # about 1 s before the impact
    after = [r for r in rows if r[0] >= first_right]
    assert all(r[1] < 0 for r in after)                                           # held to the impact
    assert all(abs(b[1]) >= abs(a[1]) for a, b in zip(after, after[1:]))          # never shrinks
    assert after[-1][1] == -12. and after[-1][2] == pytest.approx(-12.)
    if terrain.startswith('gentle'):
        assert max(r[2] for r in rows) == 0.                                      # never toward the pillar
    else:
        assert aim.counts['terrain_yields'] == 1


# ---------------------------------------------------------------------------------------------
# FastRaceCue: terrain votes for rising ground only, shadow
# ---------------------------------------------------------------------------------------------
def terrain_pilot(config, *, vertical, escalated):
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., gap_aim=config,
                        vertical_guard=VerticalGuardConfig() if vertical else None)
    pilot.clearance = TtcClearanceGovernor(vertical=VerticalGuardConfig() if vertical else None)
    pilot.clearance.limits = lambda *a, **k: (None, None, 0.)
    history.append(9.99, [0., 0., 5.], np.array([1., 0., 0., 0.]))
    for k in range(60):
        now = 10.+k*DT
        s = senses(position=(0., 0., 5.), velocity=(6., 0., 0.))
        history.append(now, [0., 0., 5.], s['quat'][0].numpy())
        pilot.clearance.climb = 1.
        pilot.clearance.escalated = escalated
        pilot.update(s, [0., 0., 0.], dict(race_cue=cue_toward([20., 0., 0.])), now-.05, now,
                     gap=dict(sample(round(now-.04, 3), 0., lr=.9, kind='clear'), near_on_path=False)
                     if k % 6 == 0 else None)
    return pilot


def test_terrain_votes_count_only_for_confirmed_rising_ground_with_a_vertical_guard():
    assert terrain_pilot(V2, vertical=True, escalated=False).gap_aim.counts['terrain_votes'] >= 2   # version 2
    assert terrain_pilot(V4, vertical=True, escalated=False).gap_aim.counts['terrain_votes'] == 0   # gentle climb
    rising = terrain_pilot(V4, vertical=True, escalated=True)
    assert rising.gap_aim.counts['terrain_votes'] >= 2 and rising.gap_aim.target == -V4.terrain_side_deg
    assert terrain_pilot(V4, vertical=False, escalated=False).gap_aim.counts['terrain_votes'] >= 2  # no guard


def test_shadow_computes_the_commitment_and_flies_the_ring():
    runs = {}
    for name, kw in (('none', {}), ('v2 shadow', dict(gap_aim=V2, gap_apply=False)),
                     ('v3 shadow', dict(gap_aim=V4, gap_apply=False))):
        history = CameraPoseHistory()
        pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., **kw)
        runs[name] = (np.array([r[1] for r in fly_with_gap(pilot, history, -8.)]), pilot)
    assert np.array_equal(runs['none'][0], runs['v2 shadow'][0]) and np.array_equal(runs['none'][0], runs['v3 shadow'][0])
    assert runs['v3 shadow'][1].gap_aim.applied == pytest.approx(-8.)
    assert runs['v3 shadow'][1].gap_offset_deg == 0.


def test_the_runner_logs_the_committed_side():
    from haltere.liftoff.visual_brain import COMMIT_COLUMNS, commit_row
    assert COMMIT_COLUMNS == ('gap_commit',) and np.isnan(commit_row(None)[0])
    pilot = FastRaceCue(SENSOR, CameraPoseHistory(), 6., gap_aim=V4)
    assert commit_row(pilot) == (0.,)
    pilot.gap_aim.commit_side = -1
    assert commit_row(pilot) == (-1.,)


# ---------------------------------------------------------------------------------------------
# offline evaluation helpers (haltere.obstacles.gap_commit_eval)
# ---------------------------------------------------------------------------------------------
def test_pillar_c_metrics_and_the_offline_pilot_replay():
    from haltere.obstacles import gap_commit_eval as ev
    t = np.arange(20.7, 23.44, .01)
    target = np.where(t >= 22.45, -np.minimum(3.+(t-22.45)*10, 12.), 0.)
    ticks = dict(t=t, target=target, applied=target, mode=np.where(target != 0, 'obstacle', ''))
    m = ev.pillar_c_metrics(ticks, 23.438, dict(window_start_t=20.7))
    assert m['left_applied_ticks'] == 0 and m['lead_s'] == pytest.approx(.988, abs=.011)
    assert m['held'] and m['non_decreasing']
    bad = dict(ticks, target=np.where((t > 22.8) & (t < 22.9), 0., target))
    assert not ev.pillar_c_metrics(bad, 23.438, dict(window_start_t=20.7))['held']
    # the offline replay: a close right obstacle 12 frames long at 18 Hz, then clear
    n = 40
    tf = 10.+np.arange(n)/18.
    seq = dict(t=tf, raw=np.where(np.arange(n) < 12, -5., 0.), valid=np.ones(n, bool),
               kind=np.array(['gap']*12+['clear']*(n-12)), ring=np.full(n, 30.), lr=np.full(n, np.nan),
               near_on_path=np.arange(n) < 12, pos=np.zeros((n, 3)))
    v2 = ev.pilot_replay(seq, V2, gap_latency_s=.09, cue_latency_s=.06)
    v3 = ev.pilot_replay(seq, V4, gap_latency_s=.09, cue_latency_s=.06)
    assert v2['counts']['obstacle_episodes'] == v3['counts']['obstacle_episodes'] == 1
    assert v3['counts']['commits'] == 1 and np.all(v3['target'][v3['commit'] != 0] == -5.)
    last = tf[11]+.09
    assert np.all(v3['target'][(v3['now'] > last) & (v3['now'] < last+V4.commit_hold_s-.01)] == -5.)
    assert v2['target'][np.searchsorted(v2['now'], last+.25)] == 0.


def test_approach_and_straw_metrics_count_ticks_and_episodes():
    from haltere.obstacles import gap_commit_eval as ev
    g = dict(approach_region=dict(x=[0., 10.], y=[-1., 1.]), distance_to=[10., 0.],
             pillar_box_xy=dict(x=[10., 10.5], y=[.2, .6]), switch_window_s=.5, switch_abs_shift_max_deg=4.)
    n = 100
    x = np.linspace(0., 9.9, n)
    target = np.where(np.arange(n) >= 40, 6., 0.)
    ticks = dict(pos=np.stack([x, np.zeros(n), np.ones(n)], 1), target=target, applied=target,
                 mode=np.where(target != 0, 'obstacle', ''), ring=np.zeros(n))
    m = ev.approach_metrics(ticks, g)
    assert m['first_left_d_m'] == pytest.approx(10.-x[40], abs=1e-3) and m['right_ticks'] == 0
    assert m['left_share_after_first'] == 1.
    seq = dict(t=np.arange(0., 10., .05), vel=np.tile([5., 0., 0.], (200, 1)), ring=np.zeros(200))
    now = np.arange(0., 10., .01)
    tgt = np.where(((now > 2) & (now < 3)) | ((now > 5) & (now < 5.5)), 3., 0.)
    tgt[(now > 5.2) & (now < 5.3)] = -3.
    s = ev.straw_metrics(dict(now=now, target=tgt, applied=tgt), seq, g, .06)
    assert s['episodes'] == 4 and s['minutes'] == pytest.approx(9.95/60, abs=1e-3)
