"""FastRaceCue TTC-graded clearance policy (`TtcClearanceConfig` / `TtcClearanceGovernor`).

These tests pin the declared pilot-side law (confirmation, the graded cap
v * clip((ttc - ttc_min) / (ttc_target - ttc_min), floor_fraction, 1) >= min_speed,
rate limit, hold while a wall stays in view, release, terrain climb with reduced
braking, stand-off memory, no-evidence handling) and check in the measured
surrogate that a delayed perfect TTC slows the drone before a wall. The offline
replay on recorded looming streams is outside the repository. None of this is
flight evidence.
"""
import json

import numpy as np
import pytest

from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import (ClearanceConfig, ClearanceGovernor, FastCueConfig, FastRaceCue,
                                           TtcClearanceConfig, TtcClearanceGovernor, clearance_governor)
from tests.test_fast_race_cue import cue_toward, drive, single_thread  # noqa: F401 (fixture)
from tests.test_fast_race_cue_clearance import surrogate_wall
from tests.test_visual_assistance import SENSOR, senses

AHEAD = cue_toward([10., 0., 0.])
X = [1., 0., 0.]
FRAME = 1/18.
# explicit parameters, so the declared law is tested independently of the defaults
CFG = TtcClearanceConfig(ttc_on=1.2, ttc_target=1.5, ttc_min=.4, floor_fraction=.3, min_speed=1., stop_ttc_s=.2,
                         confirm=2, confirm_window_s=.25, urgent_ttc_s=.4, brake_rate=8., hold_s=.5, hold_ttc_s=2.5,
                         release=3., memory_s=.3, terrain_fraction=.7, climb_on_s=1.5, climb_full_s=.8,
                         terrain_confirm=1, climb_ttc_source='alarm', climb_hold_s=.5, climb_release=3.,
                         climb_max_m=2.5, terrain_brake=.3, standoff_speed=1.5, standoff_s=2.)


def fraction(ttc, c=CFG):
    return float(np.clip((ttc-c.ttc_min)/(c.ttc_target-c.ttc_min), c.floor_fraction, 1.))


def fly(gov, ttc_of, seconds, *, speed=6., below=None, delay=.1, x0=0., start=0.):
    """Constant-speed drone along +x; a sample every frame with ttc_of(x) (None = no evidence), `delay` late."""
    pending, rows, next_frame = [], [], start
    for k in range(int(round(seconds/.01))):
        now = start+k*.01
        x = x0+speed*(now-start)
        if now >= next_frame-1e-9:
            ttc = ttc_of(x)
            pending.append((now+delay, now, x, ttc))
            next_frame += FRAME
        while pending and pending[0][0] <= now+1e-9:
            _, t, xc, ttc = pending.pop(0)
            gov.ingest(t, ttc, None if ttc is None else ttc*speed, below, [xc, 0., 5.], X, speed, received=now)
        rows.append((now, x, *gov.limits([x, 0., 5.], [speed, 0., 0.], now, .01, 3.5)))
    return rows


def test_config_validation():
    for bad in (dict(ttc_min=1.3), dict(ttc_target=1.1), dict(hold_ttc_s=1.4), dict(floor_fraction=0.),
                dict(floor_fraction=1.2), dict(terrain_brake=1.5), dict(terrain_brake=-.1), dict(confirm=1.5),
                dict(terrain_confirm=0), dict(climb_full_s=1.6), dict(brake_rate=float('nan')),
                dict(climb_ttc_source='lower'), dict(climb_max_m=0.)):
        with pytest.raises(ValueError):
            TtcClearanceConfig(**{**{k: getattr(CFG, k) for k in ('ttc_min', 'ttc_on', 'ttc_target', 'hold_ttc_s')},
                                  **bad})
    assert TtcClearanceConfig(terrain_brake=0.).terrain_brake == 0.


def test_the_fast_pilot_uses_the_ttc_policy_by_default_and_keeps_the_distance_policy():
    assert isinstance(clearance_governor(TtcClearanceConfig()), TtcClearanceGovernor)
    assert isinstance(clearance_governor(ClearanceConfig()), ClearanceGovernor)
    for config, kind, policy in ((None, TtcClearanceGovernor, 'ttc-graded'),
                                 (ClearanceConfig(), ClearanceGovernor, 'stopping-distance')):
        history = CameraPoseHistory()
        pilot = FastRaceCue(SENSOR, history, 6., reference_speed=6., clearance_config=config)
        drive(pilot, history, AHEAD, 50, velocity=(6., 0., 0.), height=5.)
        now = 10.5
        s = senses(position=(0., 0., 5.), velocity=(6., 0., 0.))
        history.append(now, [0., 0., 5.], s['quat'][0].numpy())
        pilot.update(s, [0., 0., 0.], dict(race_cue=dict(AHEAD)), now-.05, now,
                     clearance=dict(time=now-.05, ttc=3., distance=18., below_fraction=.5, ttc_lower=2.))
        assert isinstance(pilot.clearance, kind)
        assert json.loads(json.dumps(pilot.metadata()))['clearance_response']['policy'] == policy


def test_one_short_ttc_waits_for_confirmation_and_then_caps_by_the_graded_law():
    gov = TtcClearanceGovernor(CFG)
    gov.ingest(0., 1.0, 6., .5, [0., 0., 5.], X, 6., received=.1)
    assert gov.limits([.6, 0., 5.], [6., 0., 0.], .1, .01, 3.5)[0] is None        # one vote of two
    gov.ingest(FRAME, 1.0-FRAME, 6.*(1.-FRAME), .5, [6.*FRAME, 0., 5.], X, 6., received=FRAME+.1)
    x = 6.*(FRAME+.1)
    cap, ray, climb = gov.limits([x, 0., 5.], [6., 0., 0.], FRAME+.1, .01, 3.5)
    aged = 1.-FRAME-.1                                                              # odometry since capture
    target = gov.target
    assert target == pytest.approx(6.*fraction(aged))
    assert cap == pytest.approx(6.-CFG.brake_rate*.01) and climb == 0.            # rate-limited from the speed
    np.testing.assert_allclose(ray, X)
    caps = [gov.limits([x, 0., 5.], [6., 0., 0.], FRAME+.1+k*.01, .01, 3.5)[0] for k in range(1, 200)]
    assert np.all(np.diff(caps[:40]) >= -CFG.brake_rate*.01-1e-9)                # falls at most brake_rate
    assert min(caps) == pytest.approx(target) and caps[-1] > target               # reaches it, later released
    urgent = TtcClearanceGovernor(CFG)
    urgent.ingest(0., .3, 1.8, .5, [0., 0., 5.], X, 6., received=.05)
    assert urgent.limits([.3, 0., 5.], [6., 0., 0.], .05, .01, 3.5)[0] is not None  # one sample below urgent_ttc_s


def test_a_constant_short_ttc_slows_until_the_ttc_recovers_to_the_target():
    # A surface whose image expansion says 1.0 s at 6 m/s and does not come closer (position frozen): measured
    # TTC scales with 1/speed, so the cap settles at or below the speed where it reads ttc_target, then holds
    # (it holds while TTC < hold_ttc_s). The delayed samples taken while still fast make it settle lower.
    gov = TtcClearanceGovernor(CFG)
    speed, rows, pending, next_frame = 6., [], [], 0.
    for k in range(400):
        now = k*.01
        if now >= next_frame-1e-9:
            pending.append((now+.1, now, 6./speed))
            next_frame += FRAME
        while pending and pending[0][0] <= now+1e-9:
            _, t, ttc = pending.pop(0)
            gov.ingest(t, ttc, ttc*speed, .5, [0., 0., 5.], X, speed, received=now)
        cap, _, _ = gov.limits([0., 0., 5.], [speed, 0., 0.], now, .01, 3.5)   # position frozen: pure TTC
        speed = speed if cap is None else speed+(min(6., cap)-speed)*min(1., .01/.15)
        rows.append(speed)
    settled = 6./CFG.ttc_target                                                  # reads ttc_target here
    assert .6*settled <= rows[-1] <= settled+.05
    assert np.all(np.diff(rows) <= 1e-9)                                           # no stop-and-go
    assert gov.counts['brake_engagements'] == 1


def test_min_speed_floor_holds_until_a_wall_reads_below_stop_ttc():
    floor = TtcClearanceGovernor(CFG)
    fly(floor, lambda x: .5, .6, speed=2.)                      # aged TTC 0.4 s > stop_ttc_s: floor at min_speed
    assert floor.target == pytest.approx(CFG.min_speed)
    stop = TtcClearanceGovernor(TtcClearanceConfig(**{**CFG.__dict__, 'stop_ttc_s': .5}))
    fly(stop, lambda x: .5, .6, speed=2.)
    assert stop.target == pytest.approx(2.*CFG.floor_fraction)  # below min_speed: floor_fraction of the speed
    terrain = TtcClearanceGovernor(TtcClearanceConfig(**{**CFG.__dict__, 'stop_ttc_s': .5}))
    fly(terrain, lambda x: .5, .6, speed=2., below=.9)             # terrain keeps the floor (and climbs)
    assert terrain.target >= CFG.min_speed and terrain.climb > 0.


def test_hold_while_a_wall_is_in_view_then_release_at_the_declared_rate():
    gov = TtcClearanceGovernor(CFG)
    fly(gov, lambda x: .8, .5)
    low = gov.target
    rows = fly(gov, lambda x: 2., 1.5, start=.5, x0=3.)       # TTC recovered, but the wall is still in view
    assert gov.target == pytest.approx(low)
    rows = fly(gov, lambda x: None, 2., start=2., x0=12.)     # no evidence: hold_s, then release
    caps = np.array([r[2] for r in rows if r[2] is not None])
    times = np.array([r[0] for r in rows if r[2] is not None])
    last_wall = gov.lowered_at                                 # the last wall sample in view
    assert 1.9 <= last_wall <= 2.
    assert np.all(caps[times < last_wall+CFG.hold_s-.005] == pytest.approx(low))
    rise = np.diff(caps[times > last_wall+CFG.hold_s+.005])
    assert rise.max() == pytest.approx(CFG.release*.01) and (rise <= CFG.release*.01+1e-9).all()


def test_terrain_climbs_and_brakes_less_than_a_wall():
    ttc = 1.1
    wall, terrain = TtcClearanceGovernor(CFG), TtcClearanceGovernor(CFG)
    fly(wall, lambda x: ttc, .4, below=.5)
    rows = fly(terrain, lambda x: ttc, .4, below=.9)
    assert wall.climb == 0. and wall.counts['climb_engagements'] == 0
    up = 3.5*np.clip((CFG.climb_on_s-(ttc-.1))/(CFG.climb_on_s-CFG.climb_full_s), 0, 1)
    assert terrain.climb == pytest.approx(up, rel=.05) and rows[-1][4] == terrain.climb
    assert terrain.target > wall.target
    assert terrain.target == pytest.approx(6.*(1.-CFG.terrain_brake*(1.-fraction(ttc-.1))), rel=.05)
    free = TtcClearanceGovernor(TtcClearanceConfig(**{**CFG.__dict__, 'terrain_brake': 0.}))
    fly(free, lambda x: ttc, .4, below=.9)
    assert free.cap is None and free.climb > 0.


def test_the_lower_surface_ttc_and_the_climb_height_bound():
    def climb(source, ttc, lower):
        gov = TtcClearanceGovernor(TtcClearanceConfig(**{**CFG.__dict__, 'climb_ttc_source': source}))
        gov.ingest(0., ttc, ttc*6., .9, [0., 0., 5.], X, 6., received=.1, ttc_lower=lower)
        gov.limits([.6, 0., 5.], [6., 0., 0.], .1, .01, 3.5)
        return gov.climb
    assert climb('alarm', 1.4, .9) == climb('alarm', 1.4, None) == pytest.approx(3.5*(1.5-1.3)/.7)
    assert climb('either', 1.4, .9) == pytest.approx(3.5*(1.5-.8)/.7)            # the shorter of the two
    assert climb('both', .9, 1.4) == climb('alarm', 1.4, None) and climb('both', .9, None) == 0.
    # a climb ends once the drone is climb_max_m above where it began, even while terrain is still reported
    gov = TtcClearanceGovernor(CFG)
    z, climbs = 5., []
    for k in range(300):
        now = k*.01
        if k % 5 == 0:
            gov.ingest(now, 1., 6., .9, [6.*now, 0., z], X, 6., received=now)
        climb = gov.limits([6.*now, 0., z], [6., 0., 0.], now, .01, 3.5)[2]
        z += climb*.01
        climbs.append(climb)
    assert max(climbs) > 2. and climbs[-1] == 0.
    assert CFG.climb_max_m <= z-5. <= CFG.climb_max_m+3.5**2/(2*CFG.climb_release)+.05


def test_missing_vertical_evidence_is_a_wall_unless_already_climbing():
    unknown = TtcClearanceGovernor(CFG)
    fly(unknown, lambda x: 1., .4, below=None)
    wall = TtcClearanceGovernor(CFG)
    fly(wall, lambda x: 1., .4, below=.5)
    assert unknown.target == pytest.approx(wall.target) and unknown.climb == 0.
    climbing = TtcClearanceGovernor(CFG)
    fly(climbing, lambda x: 1., .2, below=.9)
    fly(climbing, lambda x: 1., .3, below=None, start=.2, x0=1.2)
    assert climbing.climb > 0. and climbing.target > wall.target


def test_stand_off_memory_keeps_a_low_cap_after_the_evidence_is_lost():
    gov = TtcClearanceGovernor(CFG)
    fly(gov, lambda x: .5, .6, speed=2.)                        # a wall that caps the drone at <= standoff_speed
    assert gov.target <= CFG.standoff_speed and gov.counts['standoff_engagements'] == 1
    held = gov.target
    rows = fly(gov, lambda x: None, 3., start=.6, x0=1.2, speed=.5)   # too slow for looming: no evidence
    caps = {round(r[0], 2): r[2] for r in rows}
    assert caps[round(.6+CFG.standoff_s-.1, 2)] == pytest.approx(held)
    assert caps[3.59] > held
    assert gov.status in ('armed', 'no_evidence', 'clear')


def test_no_evidence_changes_nothing_and_clear_samples_do_not_brake():
    gov = TtcClearanceGovernor(CFG)
    fly(gov, lambda x: None, 1.)
    fly(gov, lambda x: 5., 1., start=1., x0=6.)
    assert gov.cap is None and gov.climb == 0. and gov.counts['brake_engagements'] == 0
    assert gov.counts['no_evidence'] > 10 and gov.status == 'clear'


def cruising(config, speed=6., steps=300):
    history = CameraPoseHistory()
    pilot = FastRaceCue(SENSOR, history, speed, reference_speed=speed, clearance_config=config)
    drive(pilot, history, AHEAD, steps, velocity=(speed, 0., 0.), height=5.)
    return pilot, history, 10.+steps*.01


def test_pilot_brakes_along_the_ray_climbs_for_terrain_and_respects_command_rates():
    rng = np.random.default_rng(4)
    pilot, history, now = cruising(CFG)
    previous = pilot.velocity_command.copy()
    top_h = max(FastCueConfig().command_acceleration, CFG.brake_slew)
    top_v = max(FastCueConfig().vertical_command_acceleration, CFG.terrain_climb_acceleration)
    for k in range(400):
        t = now+k*.01
        clearance = None
        if k % 5 == 0:
            ttc = [None, .3, .7, 1., 3.][int(rng.integers(5))]
            clearance = dict(time=t-.05, ttc=ttc, distance=None if ttc is None else ttc*6.,
                             below_fraction=[None, .2, .5, .9][int(rng.integers(4))],
                             ttc_lower=None if ttc is None else ttc*.8)
        s = senses(position=(0., 0., 5.), velocity=tuple(previous+rng.normal(0, .2, 3)))
        history.append(t, [0., 0., 5.], s['quat'][0].numpy())
        pilot.update(s, [0., 0., 0.], dict(race_cue=dict(AHEAD)), t-.05, t, clearance=clearance)
        command = pilot.velocity_command.copy()
        assert np.linalg.norm(command[:2]-previous[:2]) <= top_h*.01+1e-6
        assert abs(command[2]-previous[2]) <= top_v*.01+1e-6
        previous = command
    counts = pilot.metadata()['clearance_response']['counts']
    assert counts['brake_engagements'] >= 1 and counts['climb_engagements'] >= 1
    with pytest.raises(ValueError):
        pilot.update(senses(position=(0., 0., 5.), velocity=(6., 0., 0.)), [0., 0., 0.], dict(race_cue=dict(AHEAD)),
                     now+4.05, now+4.1, clearance=dict(time=now+4.08, ttc=1., distance=6., ttc_lower=-1.))


def test_terrain_request_raises_the_vertical_command_without_stopping():
    pilot, history, now = cruising(CFG)
    for k in range(60):
        t = now+k*.01
        clearance = dict(time=t-.05, ttc=1.0, distance=6., below_fraction=.9, ttc_lower=.9) if k % 5 == 0 else None
        s = senses(position=(0., 0., 5.), velocity=(6., 0., 0.))
        history.append(t, [0., 0., 5.], s['quat'][0].numpy())
        pilot.update(s, [0., 0., 0.], dict(race_cue=dict(AHEAD)), t-.05, t, clearance=clearance)
    assert pilot.clearance.climb > 1. and pilot.velocity_command[2] > 1.
    assert pilot.velocity_command[0] > 6.*(1.-CFG.terrain_brake)-.1
    assert pilot.clearance.status == 'climb'


@pytest.mark.parametrize('delay', [.1, .2])
def test_measured_surrogate_default_policy_touches_a_wall_softly_and_clears_the_hairpin(delay):
    # With a perfect TTC the graded law, unlike the stopping-distance cap, may still touch a wall head-on at the
    # stand-off speed (looming is blind below ~1 m/s); the recorded looming TTC is not perfect (see module doc).
    result = surrogate_wall(10., delay=delay, clearance_config=TtcClearanceConfig())
    assert not result['contact'] or result['speed'] <= 1.
    assert not surrogate_wall(10., delay=delay, side_at=1.56, clearance_config=TtcClearanceConfig())['contact']


@pytest.mark.parametrize('delay', [.1, .2])
def test_measured_surrogate_slows_before_a_wall_with_delayed_perfect_ttc(delay):
    free = surrogate_wall(10., delay=delay, clearance=False, clearance_config=CFG)
    assert free['contact'] and free['speed'] > 5.
    result = surrogate_wall(10., delay=delay, clearance_config=CFG)
    assert not result['contact'] or result['speed'] <= 1.5          # a soft touch at about min_speed
    assert result['pilot'].clearance.counts['brake_engagements'] >= 1
