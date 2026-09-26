"""Pilot side of the free-space corridor planner (haltere.liftoff.corridor_aim): CorridorAim and VerticalGuard.

Pins the declared rules with synthetic planner samples (the perception module is built separately): confirmation
(2 of 3 shift samples of one class within 0.25 s), the latch and its flip-after-3 rule, slew, release and decay,
stale samples, the ring and flag conflicts, turn-first and suspension; the vertical guard's rule order (floor bound,
descent first, terrain climb, ceiling bound, squeeze), its hold/release/height bound and ramps; the declaration
parser. None of this is flight evidence.
"""
from dataclasses import asdict
import json

import numpy as np
import pytest

from haltere.liftoff.corridor_aim import (BLOCKED, FREE_SPACE_SCHEMA, LEFT, RIGHT, VERTICAL, CorridorAim,
                                          CorridorAimConfig, VerticalGuard, VerticalGuardConfig,
                                          planner_pilot_configs)

DT = .01
NAN = float('nan')
CFG = CorridorAimConfig()
VCFG = VerticalGuardConfig()

# configs/obstacles/free_space.json v1 as fixed by the planner specification (section 3); the file itself is frozen
# by the perception side. The pilot's defaults must equal it.
SPEC_V1 = json.loads('''
{"schema":"haltere.obstacles.free_space.v1","version":1,"enabled":false,
 "camera":{"width":448,"height":252,"focal_px":140.0,"cx":224.0,"cy":126.0,"uptilt_deg":30.0,"blocks":[36,64],"block_px":7},
 "depth":{"model":"DA-V2-Small","weights":"runs/dense-depth-probe-20260923/model","weights_sha256_prefix":"3152477c","input_hw":[336,602],"dtype":"fp16","pool":"block_mean"},
 "masks":{"layers":["hud","ring","ghost","propeller"],"block_mask_max_fraction":0.0},
 "tracks":{"max_corners":600,"quality":0.005,"min_distance_px":5,"lk_window_px":21,"lk_levels":4,"fb_max_px":0.5,
           "perp_max_px":1.0,"min_foe_px":30.0,"min_tz_mps":1.0,"inv_z_range":[0.0333,2.0],"pair_dt_s":[0.03,0.15]},
 "scale":{"huber_k":1.345,"irls_iters":4,"sigma_floor":0.01,"forget":0.85,"reset_gap_s":0.3,"reset_speed_mps":1.0,
          "min_anchors":60,"min_spread":0.05,"spread_pct":[10,90],"hold_s":0.3,"max_range_m":30.0},
 "plan":{"min_speed_mps":1.5,"az_offsets_deg":[-30,30,2],"el_offsets_deg":[0,12,3],"cost_el_weight":2.0,
         "max_az_deg":20.0,"max_el_deg":12.0,"path_step_m":0.25,"horizon_lateral_m":1.5,"horizon_extra_s":0.3,
         "min_horizon_m":2.0,"a_vert_mps2":5.0,"r_lat_m":0.75,"r_vert_m":0.35,"x_min_m":0.3,"min_points":2,
         "footprint_x_m":4.0,"footprint_valid":0.6,"image_margin_deg":2.0,"aperture_ratio":1.3,"aperture_gap_m":0.7,
         "aperture_edge_m":0.25,"brake_mps2":8.0,"stop_margin_m":0.5,"cloud_beyond_horizon_m":1.5},
 "profile":{"bin_m":0.5,"near_x_m":[0.5,3.0],"lateral_m":0.75,"min_bins":2,"ceil_min_m":0.3,"rise_level_m":0.35,
            "rise_bins":2,"rise_max_step_m":0.5},
 "response_models":"configs/obstacles/response_models.json",
 "response_model_for_contract":{"fast_velocity_brain_v1":"brain08","fast_velocity_pd_v1":"fast_pd"},
 "pilot":{"max_age_s":0.2,"confirm":2,"window":3,"confirm_window_s":0.25,"side_latch_s":0.6,"flip_after":3,
          "release_after":2,"slew_az_deg_s":40.0,"slew_el_deg_s":20.0,"decay_s":0.3,"conflict_deg":6.0,
          "flag_conflict_deg":1.0,"conflict_hold_s":0.6,"v_cap_floor_mps":1.0,"cap_slew_mps2":15.0},
 "vertical":{"floor_clear_m":0.5,"ceil_clear_m":0.5,"tau_s":0.5,"gentle_up_mps":1.0,"gentle_down_mps":1.0,
             "level_band_mps":0.3,"rise_lag_s":0.3,"rise_min_t_s":0.3,"climb_max_mps":3.5,"climb_hold_s":0.5,
             "climb_release_mps2":3.0,"climb_max_rise_m":2.5,"climb_accel_mps2":10.0,"floor_accel_mps2":5.0,
             "ceil_accel_mps2":15.0,"fallback_descent_first":true},
 "runtime":{"placement":"process","stride":1,"device":"cuda","cue_timeout_s":0.15,"pose_max_lag_s":0.05,"offline_age_s":0.10}}
''')


def plan(t, kind='clear', seq=None, ring=0., **kw):
    """A synthetic planner sample (camera_process.plan_sample layout: None = not applicable)."""
    s = dict(time=t, seq=seq if seq is not None else round(t*1000), kind=kind,
             valid=kind in ('clear', 'aperture', 'shift', 'blocked'), ring_az=ring, cls=0., az=0., el=0.,
             feasible=None, l_az=None, l_el=None, l_ok=None, r_az=None, r_el=None, r_ok=None, v_el=None, v_ok=None,
             v_cap=None, h_floor=None, h_ceil=None, rise=None, rise_x=None, speed=6.)
    s.update(kw)
    return s


def left(t, az=8., el=0., ok=1., **kw):
    return plan(t, 'shift', cls=1., az=az, el=el, feasible=ok, l_az=az, l_el=el, l_ok=ok, **kw)


def right(t, az=-8., el=0., ok=1., **kw):
    return plan(t, 'shift', cls=-1., az=az, el=el, feasible=ok, r_az=az, r_el=el, r_ok=ok, **kw)


def run(aim, samples, until, start=0., latency=.1, **step_kw):
    """Feed samples (published `latency` after capture) through 100 Hz ticks; rows (now, step output)."""
    rows, pending, latest = [], sorted(samples, key=lambda s: s['time']), None
    for k in range(int(round((until-start)/DT))+1):
        now = start+k*DT
        while pending and pending[0]['time']+latency <= now+1e-9:
            latest = pending.pop(0)
        aim.ingest(latest, now)
        kw = {key: (v(now) if callable(v) else v) for key, v in step_kw.items()}
        rows.append((now, aim.step(now, DT, **kw)))
    return rows


def at(rows, t):
    return min(rows, key=lambda r: abs(r[0]-t))[1]


def every(t0, t1, rate=17.):
    return list(np.arange(t0, t1, 1/rate))


# ---------------------------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------------------------
def test_the_spec_declaration_parses_to_the_defaults():
    parsed = planner_pilot_configs(SPEC_V1, 'fast_velocity_brain_v1')
    assert parsed['corridor'] == CFG and parsed['vertical'] == VCFG and parsed['motor'] == 'brain08'
    assert planner_pilot_configs(SPEC_V1, 'fast_velocity_pd_v1')['motor'] == 'fast_pd'
    assert asdict(CFG)['stale_release_s'] == .3              # the spec's value; not a key of the v1 declaration
    with pytest.raises(ValueError, match='version'):
        planner_pilot_configs(dict(SPEC_V1, version=2))
    with pytest.raises(ValueError, match='schema'):
        planner_pilot_configs(dict(SPEC_V1, schema='x'))
    with pytest.raises(ValueError, match='unknown'):
        planner_pilot_configs(dict(SPEC_V1, pilot=dict(SPEC_V1['pilot'], extra=1.)))
    with pytest.raises(ValueError, match='response model'):
        planner_pilot_configs(SPEC_V1, 'motor_tracking_teacher_v1')
    assert SPEC_V1['schema'] == FREE_SPACE_SCHEMA


@pytest.mark.parametrize('overrides', [dict(confirm=4), dict(window=2.5), dict(slew_az_deg_s=0.), dict(decay_s=-1.),
                                       dict(max_az_deg=50.), dict(max_age_s=float('nan'))])
def test_corridor_config_validates(overrides):
    with pytest.raises(ValueError):
        CorridorAimConfig(**overrides)


@pytest.mark.parametrize('overrides', [dict(tau_s=0.), dict(climb_max_mps=float('inf')),
                                       dict(fallback_descent_first=1)])
def test_vertical_config_validates(overrides):
    with pytest.raises(ValueError):
        VerticalGuardConfig(**overrides)


# ---------------------------------------------------------------------------------------------
# CorridorAim
# ---------------------------------------------------------------------------------------------
def test_two_of_three_shift_samples_of_one_class_confirm_and_the_offset_slews():
    aim = CorridorAim()
    rows = run(aim, [left(t) for t in every(0., 1.)], 1.1)
    first = next(t for t, r in rows if r.cls == LEFT)
    assert first == pytest.approx(1/17.+.1, abs=.011)                 # the second sample, as published
    reached = next(t for t, r in rows if r.az >= 8.-1e-9)
    assert reached-first == pytest.approx(8./CFG.slew_az_deg_s, abs=.02)
    assert at(rows, 1.).target_az == 8. and at(rows, 1.).el == 0.
    assert aim.counts['left_episodes'] == 1 and aim.episode == 1


def test_a_single_sample_or_samples_further_apart_than_the_window_do_not_confirm():
    aim = CorridorAim()
    rows = run(aim, [left(0.)], .5)
    assert all(r.cls == 0 and r.az == 0. for _, r in rows)
    aim = CorridorAim()
    rows = run(aim, [left(0.), plan(.1), left(.35), plan(.45)], 1.)
    assert all(r.cls == 0 and r.az == 0. for _, r in rows)


def test_stale_and_invalid_samples_are_dropped():
    aim = CorridorAim()
    rows = run(aim, [left(t) for t in every(0., .5)], .8, latency=.25)       # every sample older than 0.2 s
    assert all(r.cls == 0 for _, r in rows) and aim.counts['stale'] > 5 and aim.counts['samples'] == 0
    aim = CorridorAim()
    rows = run(aim, [plan(t, 'no_scale', cls=1., l_az=8.) for t in every(0., .5)], .8)
    assert all(r.cls == 0 for _, r in rows) and aim.counts['invalid'] > 5
    # each seq is taken once however often the runner passes the newest sample
    aim = CorridorAim()
    s = left(0.)
    for k in range(10):
        aim.ingest(s, .1+k*DT)
    assert aim.counts['samples'] == 1


def test_the_latch_holds_its_side_while_its_option_exists_and_flips_after_three_samples_without():
    # latched left; the planner then proposes right (lower cost) but the left option stays open: no swap
    samples = [left(t) for t in every(0., .3)]
    samples += [right(t, l_az=10., l_el=0., l_ok=1.) for t in every(.3, .8)]
    aim = CorridorAim()
    rows = run(aim, samples, .95)
    assert all(r.cls in (0, LEFT) for _, r in rows) and at(rows, .9).target_az == 10.
    assert aim.counts['flips'] == 0 and aim.counts['latch_blocks'] > 0
    # the left option disappears: the target is held, and after three samples without it the latch flips right
    samples = [left(t) for t in every(0., .3)]+[right(t) for t in every(.3, .8)]
    aim = CorridorAim()
    rows = run(aim, samples, .95)
    no_left = [s['time'] for s in samples if s['time'] >= .3]
    flip_at = no_left[2]+.1
    assert at(rows, flip_at-.02).cls == LEFT and at(rows, flip_at-.02).target_az == 8.
    assert at(rows, flip_at+.01).cls == RIGHT and at(rows, flip_at+.01).target_az == -8.
    assert aim.counts['flips'] == 1
    assert any(r.flip for _, r in rows)


def test_release_after_two_clear_samples_once_the_minimum_latch_passed_then_linear_decay():
    samples = [left(t) for t in every(0., .12)]+[plan(t) for t in every(.12, 1.5)]
    aim = CorridorAim()
    rows = run(aim, samples, 1.5)
    latched = next(t for t, r in rows if r.cls == LEFT)
    released = next(t for t, r in rows if t > latched and r.cls == 0)
    assert released >= latched+CFG.side_latch_s-1e-9                   # clear samples came earlier: latch minimum
    start = at(rows, released-DT).az
    assert start > 0
    gone = next(t for t, r in rows if t > released and r.az == 0.)
    assert gone-released == pytest.approx(CFG.decay_s, abs=.02)        # linear over decay_s from any value
    half = at(rows, released+CFG.decay_s/2).az
    assert half == pytest.approx(start/2, abs=start*.1)


def test_without_fresh_samples_the_offset_is_released_after_stale_release_s():
    aim = CorridorAim()
    rows = run(aim, [left(t) for t in every(0., .5)], 1.6)
    last = max(every(0., .5))+.1                                      # last sample received
    assert at(rows, last+CFG.stale_release_s-.02).cls == LEFT
    assert at(rows, last+CFG.stale_release_s+.02).cls == 0
    assert at(rows, last+CFG.stale_release_s+CFG.decay_s+.03).az == 0.
    assert aim.counts['releases_stale'] == 1


def test_offsets_are_bounded_and_the_vertical_class_raises_el_only():
    aim = CorridorAim()
    rows = run(aim, [left(t, az=35., el=20.) for t in every(0., 1.5)], 1.5)
    assert at(rows, 1.5).az == CFG.max_az_deg and at(rows, 1.5).el == CFG.max_el_deg
    up = [plan(t, 'shift', cls=2., el=9., v_el=9., v_ok=1.) for t in every(0., 1.5)]
    aim = CorridorAim()
    rows = run(aim, up, 1.5)
    reached = next(t for t, r in rows if r.el >= 9.-1e-9)
    first = next(t for t, r in rows if r.cls == VERTICAL)
    assert reached-first == pytest.approx(9./CFG.slew_el_deg_s, abs=.02) and at(rows, 1.5).az == 0.


def test_blocked_is_confirmed_with_a_capped_speed_and_an_urgent_shift_caps_too():
    samples = [plan(t, 'blocked', v_cap=2.5) for t in every(0., .6)]
    aim = CorridorAim()
    rows = run(aim, samples, .6)
    assert at(rows, .5).cls == BLOCKED and at(rows, .5).v_cap == 2.5 and at(rows, .5).az == 0.
    assert aim.counts['blocked_episodes'] == 1
    low = [plan(t, 'blocked', v_cap=.2) for t in every(0., .6)]
    assert at(run(CorridorAim(), low, .6), .5).v_cap == CFG.v_cap_floor_mps       # the floor
    urgent = [left(t, az=20., ok=0., v_cap=3.) for t in every(0., .6)]
    rows = run(CorridorAim(), urgent, .6)
    assert at(rows, .5).cls == LEFT and at(rows, .5).v_cap == 3.
    feasible = [left(t, az=8., ok=1., v_cap=None) for t in every(0., .6)]
    assert at(run(CorridorAim(), feasible, .6), .5).v_cap is None


def test_ring_and_flag_conflicts_drop_the_evidence_and_hold_the_ring_aim():
    aim = CorridorAim()
    run(aim, [left(t) for t in every(0., .5)], .6)
    assert aim.az > 0
    assert aim.reconcile_ring(25., .61) and aim.az == 0. and aim.cls == 0
    assert aim.counts['ring_conflicts'] == 1
    rows = run(aim, [left(t, ring=25.) for t in every(.62, 1.5)], 1.5, start=.62)
    held = [r for t, r in rows if t < .61+CFG.conflict_hold_s]
    assert all(r.cls == 0 for r in held) and at(rows, 1.4).cls == LEFT     # confirmed again after the hold
    aim = CorridorAim()
    run(aim, [left(t) for t in every(0., .5)], .6)
    assert not aim.flag_conflict(3., .61)                                 # the flag on the same side
    assert aim.flag_conflict(-3., .62) and aim.az == 0. and aim.counts['flag_conflicts'] == 1


def test_samples_of_another_ring_are_forgotten_without_a_conflict_when_idle():
    aim = CorridorAim()
    aim.ingest(left(0.), .1)
    assert not aim.reconcile_ring(30., .1) and len(aim.samples) == 0 and aim.counts['ring_conflicts'] == 0


def test_turn_first_holds_az_at_zero_with_decay_and_suspension_resets():
    aim = CorridorAim()
    turn = lambda now: now >= .6
    rows = run(aim, [left(t, el=6.) for t in every(0., 1.2)], 1.2, turn_first=turn)
    before = at(rows, .59)
    assert before.az == 8.
    assert at(rows, .6+CFG.decay_s+.02).az == 0. and at(rows, .6+CFG.decay_s/2).az == pytest.approx(4., abs=.5)
    assert at(rows, 1.1).el == 6. and at(rows, 1.1).cls == LEFT          # the vertical option stays
    aim = CorridorAim()
    rows = run(aim, [left(t) for t in every(0., 1.)], 1., suspended=lambda now: now >= .7)
    assert at(rows, .69).az == 8. and all(r.az == 0. and r.cls == 0 for t, r in rows if t >= .7)
    assert aim.counts['resets'] == 1


# ---------------------------------------------------------------------------------------------
# VerticalGuard
# ---------------------------------------------------------------------------------------------
def guard_run(samples, until, z_request=0., vz=0., vh=6., height=1., start=0., latency=.1, follow=False):
    """VerticalGuard over 100 Hz ticks; vz/height may be functions of time; follow integrates the output into the
    measured vz (a plant that tracks). Rows (now, GuardStep, measured vz, height)."""
    guard = VerticalGuard()
    rows, pending, latest, z, v = [], sorted(samples, key=lambda s: s['time']), None, height, None
    for k in range(int(round((until-start)/DT))+1):
        now = start+k*DT
        while pending and pending[0]['time']+latency <= now+1e-9:
            latest = pending.pop(0)
        guard.ingest(latest, now)
        measured = (v if follow and v is not None else vz(now) if callable(vz) else vz)
        h = height(now) if callable(height) else z
        out = guard.apply(z_request(now) if callable(z_request) else z_request, now, DT, h, [vh, 0., measured])
        if follow:
            v = out.z
            z += out.z*DT
        rows.append((now, out, measured, h))
    return guard, rows


def test_floor_bound_slows_and_stops_a_sink_and_never_climbs_hard():
    # sinking at 0.4 m/s toward a floor 0.7 m below: the request is bounded to -(h-0.5)/0.5, then level
    height = lambda t: .7-.4*t
    samples = [plan(t, h_floor=height(t)) for t in every(0., 1.5)]
    guard, rows = guard_run(samples, 1.5, z_request=-.4, vz=-.4, height=height)
    early = at(rows, .3)
    assert early.active and early.floor_bound
    newest = max(t for t in every(0., 1.5) if t+.1 <= .3)                # the newest published sample
    expected = -(height(newest)-.4*(.3-newest)-.5)/.5                    # its height dead-reckoned by vz*age
    assert early.z == pytest.approx(expected) and -.4 < early.z < 0
    late = at(rows, 1.2)                                                   # below 0.5 m: a gentle climb only
    assert late.z > 0 and late.z <= VCFG.gentle_up_mps
    assert max(r[1].z for r in rows) <= VCFG.gentle_up_mps
    assert guard.counts['floor_bound_activations'] == 1 and at(rows, .3).up_rate == VCFG.floor_accel_mps2


def test_terrain_climb_needs_confirmed_rising_ground_is_held_released_and_bounded():
    samples = [plan(t, h_floor=1., rise=1., rise_x=3.) for t in every(0., .6)]
    samples += [plan(t, h_floor=1.) for t in every(.6, 2.5)]              # level ground after it
    guard, rows = guard_run(samples, 2.5)
    on = next(t for t, r, *_ in rows if r.climb > 0)
    assert on == pytest.approx(1/17.+.1, abs=.011)                         # the second rising sample
    expected = min(VCFG.climb_max_mps, 1./max(3./6.-VCFG.rise_lag_s, VCFG.rise_min_t_s))
    assert at(rows, .3).climb == pytest.approx(expected) and at(rows, .3).z == pytest.approx(expected)
    assert at(rows, .3).up_rate == VCFG.climb_accel_mps2
    last = max(every(0., .6))+.1                                        # the last confirming sample
    assert at(rows, last+VCFG.climb_hold_s-.02).climb == pytest.approx(expected)
    released = at(rows, last+VCFG.climb_hold_s+.5)
    assert released.climb == pytest.approx(expected-VCFG.climb_release_mps2*.5, abs=.05)
    assert at(rows, 2.4).climb == 0.
    assert guard.counts['terrain_climbs'] == 1
    # a single rising sample does not climb
    _, rows = guard_run([plan(0., h_floor=1., rise=1., rise_x=3.)]+[plan(t, h_floor=1.) for t in every(.1, .6)], .6)
    assert all(r.climb == 0 for _, r, *_ in rows)
    # the climb ends climb_max_rise_m above where it started
    samples = [plan(t, rise=3., rise_x=2.) for t in every(0., 3.)]
    guard, rows = guard_run(samples, 3., follow=True, height=0.)
    topped = next(t for t, r, v, h in rows if h >= VCFG.climb_max_rise_m)
    assert guard.counts['topped'] >= 1
    after = at(rows, topped+.3)
    assert after.climb < at(rows, topped-.01).climb


def test_descent_first_withholds_the_terrain_climb_until_level():
    samples = [plan(t, h_floor=1.5, rise=1., rise_x=3.) for t in every(0., 1.)]
    vz = lambda t: -.6 if t < .5 else 0.
    guard, rows = guard_run(samples, 1., vz=vz)
    assert all(r.climb == 0. for t, r, *_ in rows if t < .5)
    assert any(r.descent_first for t, r, *_ in rows if t < .5)
    assert at(rows, .7).climb > 1.
    assert guard.counts['descent_first_holds'] == 1


def test_the_ceiling_bound_is_applied_last_and_a_squeeze_takes_the_middle():
    samples = [plan(t, h_floor=1., h_ceil=1., rise=1., rise_x=3.) for t in every(0., .6)]
    _, rows = guard_run(samples, .6, z_request=.2)
    r = at(rows, .4)
    assert r.climb > 3. and r.hi == pytest.approx(1.) and r.z == pytest.approx(1.) and r.ceil_bound
    assert r.down_rate == VCFG.ceil_accel_mps2
    samples = [plan(t, h_ceil=.3) for t in every(0., .6)]
    _, rows = guard_run(samples, .6, z_request=.5)
    assert at(rows, .4).z == pytest.approx(-.4)
    samples = [plan(t, h_ceil=0.) for t in every(0., .6)]
    _, rows = guard_run(samples, .6, z_request=.5)
    assert at(rows, .4).z == pytest.approx(-VCFG.gentle_down_mps)          # never faster down than gentle_down
    samples = [plan(t, h_floor=.3, h_ceil=.4) for t in every(0., .6)]
    guard, rows = guard_run(samples, .6, z_request=.5)
    r = at(rows, .4)
    assert r.squeeze and r.lo == pytest.approx(.4) and r.hi == pytest.approx(-.2) and r.z == pytest.approx(.1)
    assert guard.counts['squeezes'] == 1


def test_the_guard_is_inactive_without_a_fresh_valid_newest_sample():
    samples = [plan(t, h_floor=.1) for t in every(0., .3)]+[plan(t, 'no_scale', valid=False) for t in every(.3, .6)]
    _, rows = guard_run(samples, .8, z_request=-1.)
    assert at(rows, .3).active and at(rows, .3).z > 0
    assert not at(rows, .5).active and at(rows, .5).z == -1.
    _, rows = guard_run([plan(0., h_floor=.1)], .5, z_request=-1.)
    assert at(rows, .15).active and not at(rows, .25).active                # older than max_age_s
