"""The rabbit by-sight pilot (haltere.liftoff.sightpilot) on a fake clock with scripted detections and a kinematic drone:
bounded goal and yaw under perception upsets, intake bookkeeping, track separation and absorption, the pass rules."""
import math
import time

import numpy as np
import pytest
import torch

from haltere.liftoff.pilot import LiftoffMapping, TelemetryPilot
from haltere.liftoff.sightpilot import LOG_COLUMNS, SightParams, SightPilot, Track, add_cli_args, params_from_args
from haltere.sim.tasks import HoverTaskConfig
from haltere.vision.camera import Camera, quat_wxyz_to_mat
from haltere.vision.model import IN_H, IN_W
from haltere.vision.rehearse import SimClock, frame_from_sim, quat_from_yaw
from haltere.vision.runtime import Detection, detection_geometry

CAM = Camera(640, 360, 200.0, 30.0)
DT = 0.01


class DummyBrain:
    def weight_matrix(self):
        return torch.zeros(1)

    def init_state(self, B):
        return {}


class ScriptVision:
    """Detections of world points picked by a script: grabbed at 15 Hz with the drone's pose, delivered 80 ms later,
    stamped with the grab time; a frame whose point is not in the image holds no arch (p 0.01), like the detector's."""

    def __init__(self, where, rate=15.0, latency=0.08):
        self.cam = CAM.scaled(IN_W, IN_H)
        self.latest = Detection()
        self.where = where
        self.rate, self.latency = rate, latency
        self.queue = []
        self.next_t = None
        self.n = 0

    def get(self):
        return self.latest

    def update(self, now, pos, quat):
        if self.next_t is None:
            self.next_t = now
        if now >= self.next_t - 1e-9:
            self.next_t += 1.0 / self.rate
            c = self.where(now)
            det = None
            if c is not None:
                rel = quat_wxyz_to_mat(quat).T @ (np.asarray(c, dtype=float) - pos)
                d = float(np.linalg.norm(rel))
                px, ok = self.cam.project_body(rel[None])
                u, v = px[0]
                if ok[0] and 0 <= u <= self.cam.width and 0 <= v <= self.cam.height and d < 45.0:
                    f = self.cam.f
                    du, dv = u - self.cam.width / 2, v - self.cam.height / 2
                    stretch = np.sqrt(f * f + du * du) * np.sqrt(f * f + du * du + dv * dv) / (f * f)
                    width = f * 4.0 * stretch / d
                    direction, dist = detection_geometry(self.cam, u, v, width)
                    det = Detection(now, 0.99, u, v, width, direction, dist)
            if det is None:
                w, h = self.cam.width, self.cam.height
                direction, dist = detection_geometry(self.cam, w / 2, h / 2, 30.0)
                det = Detection(now, 0.01, w / 2, h / 2, 30.0, direction, dist)
            self.n += 1
            det.frames = self.n
            self.queue.append((now + self.latency, det))
        while self.queue and self.queue[0][0] <= now + 1e-9:
            self.latest = self.queue.pop(0)[1]


class Rig:
    """A TelemetryPilot without a brain, fed telemetry of a kinematic drone that flies toward the pilot's goal
    (speed cap, first-order response) and yaws with the pilot's yaw stick."""

    def __init__(self, vision, params=None, vmax=2.5, yaw_rate=3.8):
        self.pilot = TelemetryPilot(DummyBrain(), HoverTaskConfig(), LiftoffMapping(), 'cpu')
        self.clock = SimClock()
        self.pilot.clock = self.clock
        self.pilot.vision = vision
        self.pilot.sight = 'rabbit'
        self.pilot.sight_params = params or SightParams(yaw_rate=yaw_rate)
        self.pos, self.vel, self.yaw = np.zeros(3), np.zeros(3), 0.0
        self.vmax, self.yaw_rate = vmax, yaw_rate
        self.frozen = False
        self.k = 0
        self.goal = np.zeros(3)

    @property
    def sp(self) -> SightPilot:
        return self.pilot.sightpilot

    def tick(self):
        now = self.clock()
        q = quat_from_yaw(self.yaw)
        if hasattr(self.pilot.vision, 'update'):
            self.pilot.vision.update(now, self.pos, q)
        fr = frame_from_sim(self.k * DT, self.pos, self.vel, q, np.zeros(3), np.full(4, 0.5), np.zeros(4), now)
        s = self.pilot.sensors(fr)
        p = s['pos'][0].numpy().astype(np.float64)
        self.pilot.last_pos = p
        self.goal = self.pilot.vision_goal(p)
        if not self.frozen:
            gw = self.pilot.last_R @ self.goal
            n = float(np.linalg.norm(gw))
            v_des = gw / n * min(n, self.vmax) if n > 1e-6 else np.zeros(3)
            self.vel += (v_des - self.vel) * (DT / 0.5)
            self.pos = self.pos + self.vel * DT
            self.pos[2] = max(self.pos[2], 0.0)
            self.yaw += -self.sp.sight_yaw * self.yaw_rate * DT
        self.clock.t += DT
        self.k += 1
        return self.goal

    def hover_at(self, pos, yaw=0.0):
        """One tick on the ground at the origin (the reset point), then hold still at pos."""
        self.frozen = True
        self.tick()
        self.pos, self.yaw = np.asarray(pos, dtype=float), yaw
        for _ in range(3):
            self.tick()


def test_goal_and_yaw_stay_bounded_under_perception_upsets():
    rng = np.random.default_rng(0)

    def where(t):
        t -= 1000.0
        if t < 8.0:
            return (25.0, 0.0, 2.7)
        if t < 12.0:
            return (25.0, 10.0, 2.7)                                  # the arch teleports 10 m sideways
        if t < 16.0:
            return (30.0, 10.0, 2.7) if int(t / 0.5) % 2 else (30.0, -6.0, 2.7)   # flips between two arches
        if t < 22.0:
            if int(t / 0.5) % 2:                                       # phantom bursts
                return (rig.pos[0] + rng.uniform(4, 25), rng.uniform(-12, 12), rng.uniform(0.0, 3.0))
            return None
        return (45.0, 5.0, 2.7)

    rig = Rig(ScriptVision(where), vmax=3.0)
    prev_c = prev_yaw = None
    worst = {'carrot': 0.0, 'goal_h': 0.0, 'goal_z': 0.0, 'yaw': 0.0}
    ticks = []
    travelled = 0.0
    for _ in range(3000):
        p0 = rig.pos.copy()
        a = time.perf_counter()
        g = rig.tick()
        ticks.append(time.perf_counter() - a)
        travelled += float(np.linalg.norm(rig.pos - p0))
        sp = rig.sp
        worst['goal_h'] = max(worst['goal_h'], math.hypot(g[0], g[1]))
        worst['goal_z'] = max(worst['goal_z'], abs(float((rig.pilot.last_R @ g)[2])))
        if prev_yaw is not None:
            worst['yaw'] = max(worst['yaw'], abs(sp.sight_yaw - prev_yaw))
        prev_yaw = sp.sight_yaw
        if sp.t_air is not None:
            if prev_c is not None:
                worst['carrot'] = max(worst['carrot'], float(np.linalg.norm(sp.carrot - prev_c)))
            prev_c = sp.carrot.copy()
    assert worst['carrot'] <= 0.06, worst
    assert worst['goal_h'] <= 5.0 + 1e-9 and worst['goal_z'] <= 1.2 + 1e-9, worst
    assert worst['yaw'] <= 0.03 + 1e-9, worst
    assert rig.sp.errors == 0 and np.isfinite(rig.goal).all()
    assert travelled > 40.0                                             # it kept flying through all of it
    assert np.mean(ticks) < 2e-3                                        # sensors excluded; the 100 Hz budget


def test_a_detection_read_every_tick_updates_a_track_once():
    det = Detection()
    vis = type('V', (), {'cam': CAM.scaled(IN_W, IN_H), 'get': lambda self: det})()
    rig = Rig(vis)
    rig.hover_at((0.0, 0.0, 1.5))
    grab = rig.clock() - 0.02                                           # inside the pose history
    c = np.array([15.0, 0.0, 2.7]) - np.array([0.0, 0.0, 1.5])
    px, _ = vis.cam.project_body(c[None])
    direction, dist = detection_geometry(vis.cam, px[0, 0], px[0, 1], 30.0)
    det = Detection(grab, 0.99, px[0, 0], px[0, 1], 30.0, direction, dist, 1)
    for _ in range(50):
        rig.tick()
    assert len(rig.sp.tracks) == 1 and rig.sp.tracks[0].hits == 1


def test_two_lined_up_arches_make_two_tracks():
    near, far = np.array([12.0, 0.0, 2.7]), np.array([30.0, 0.0, 2.7])
    flip = {'n': 0}

    def where(t):
        flip['n'] += 1
        return near if flip['n'] % 2 else far

    rig = Rig(ScriptVision(where), SightParams(yaw_rate=3.8, range_corr=None))
    rig.hover_at((0.0, 0.0, 2.7))
    for _ in range(300):
        rig.tick()
    conf = [t for t in rig.sp.tracks if t.confirmed and not t.passed]
    assert len(conf) == 2
    ms = sorted((t.m for t in conf), key=lambda m: m[0])
    assert np.linalg.norm(ms[0] - near) < 1.0 and np.linalg.norm(ms[1] - far) < 3.0
    assert rig.sp.target is not None and np.linalg.norm(rig.sp.target.m - near) < 1.0      # the nearer one first


def test_one_pass_and_the_arch_seen_from_behind_is_absorbed():
    arch = np.array([20.0, 0.0, 2.7])
    rig = Rig(ScriptVision(lambda t: arch))
    passes_at = []
    for _ in range(4000):
        rig.tick()
        if rig.sp.pass_kind in (1, 2, 3):
            passes_at.append(rig.pos.copy())
    sp = rig.sp
    assert sp.n_passes == 1 and len(passes_at) == 1
    assert abs(passes_at[0][0] - 20.0) < 3.0                            # passed at the arch
    assert sp.absorbed > 0                                              # it came round and saw the arch from behind
    assert not any(not t.passed and t.confirmed and np.linalg.norm((t.m - arch)[:2]) < 4.0 for t in sp.tracks)


def _crafted_target(rig, m, hits=10, min_a=-5.0):
    sp = rig.sp
    now = rig.clock()
    m = np.asarray(m, dtype=float)
    r = np.array([1.0, 0.0, 0.0])
    T = Track(99, now, m, np.eye(3) * 0.1, r, rig.pilot.last_pos.copy(), 5.0)
    T.confirmed, T.hits, T.min_a = True, hits, min_a
    sp.tracks.append(T)
    sp.target = T
    sp.n_ang = 0.0
    sp.t_air = now - 10.0
    return T


def test_seen_ahead_vetoes_a_pass_until_the_arch_is_no_longer_seen_ahead():
    arch = np.array([20.0, 0.0, 2.7])
    showing = {'on': True}
    rig = Rig(ScriptVision(lambda t: arch if showing['on'] else None), SightParams(yaw_rate=3.8, range_corr=None))
    pos = np.array([20.5, 3.5, 2.0])
    rig.hover_at(pos, yaw=math.atan2(arch[1] - pos[1], arch[0] - pos[0]))
    T = _crafted_target(rig, arch)
    for _ in range(100):                                                 # crossed its plane 3.5 m to the side ...
        rig.tick()
    assert rig.sp.n_passes == 0 and rig.sp.target is T                   # ... but the arch is still seen ahead
    showing['on'] = False
    for _ in range(100):
        rig.tick()
    assert rig.sp.n_passes == 1 and T.passed


def test_an_early_pass_is_undone_when_the_arch_is_seen_ahead():
    arch = np.array([20.0, 0.0, 2.7])
    rig = Rig(ScriptVision(lambda t: arch), SightParams(yaw_rate=3.8, range_corr=None))
    rig.pilot.vision.next_t = 1e9                                        # no detections until the pass is made
    rig.hover_at((17.0, 0.0, 2.0))
    T = _crafted_target(rig, arch)
    rig.sp._pass(T, rig.clock(), 'travel', (1.0, 0.0))
    assert T.passed and rig.sp.n_passes == 1 and rig.sp.target is None
    rig.pilot.vision.next_t = rig.clock()
    for _ in range(50):
        rig.tick()
    sp = rig.sp
    assert not T.passed and sp.unpasses == 1 and sp.n_passes == 0 and sp.target is T


def test_a_bale_height_phantom_never_confirms():
    rng = np.random.default_rng(1)
    rig = Rig(ScriptVision(lambda t: np.array([12.0, 1.0, 0.7]) + rng.normal(0, 0.05, 3)))
    rig.hover_at((0.0, 0.0, 1.5))
    for _ in range(400):
        rig.tick()
        assert not any(t.confirmed for t in rig.sp.tracks) and rig.sp.target is None
    assert rig.sp.tracks                                                  # it was seen, as a tentative track


def test_pose_at_interpolates_and_refuses_times_outside_the_history():
    rig = Rig(ScriptVision(lambda t: None))
    rig.vel = np.array([2.0, 0.0, 0.0])
    for _ in range(150):
        rig.tick()
    h = rig.pilot._pose_hist
    assert h[-1][0] - h[0][0] <= 1.0 + 1e-9
    t_mid = 0.5 * (h[40][0] + h[41][0])
    p, R = rig.pilot.pose_at(t_mid)
    assert np.allclose(p, 0.5 * (h[40][1] + h[41][1]))
    assert rig.pilot.pose_at(h[0][0] - 0.5) is None and rig.pilot.pose_at(h[-1][0] + 0.5) is None
    rig.pilot.reset(None)
    assert rig.pilot._pose_hist == [] and rig.pilot._yaw_override is None and rig.sp.tracks == []


def test_the_pilot_never_raises():
    class Broken:
        cam = CAM.scaled(IN_W, IN_H)

        def get(self):
            raise RuntimeError('detector thread died')

    rig = Rig(Broken())
    for _ in range(300):
        g = rig.tick()
        assert np.isfinite(g).all() and math.hypot(g[0], g[1]) <= 5.0 + 1e-9
    assert rig.sp.errors == 300


def test_speed_sense_follows_the_cruise_speed():
    P = SightParams(v_cruise=4.0, v_gate=4.0)
    lo, hi = P.flow_bounds()
    assert hi == 1.0 and lo == pytest.approx(0.9 * 2.45 / 4.0)
    assert SightParams().flow_bounds() == (0.7, 1.0)
    host = type('H', (), {'flow_gain': 1.0})()
    sp = SightPilot(host, P)
    sp.v_nom = sp.v_des = 4.0
    for _ in range(20):
        sp._flow(0.0, 1.0, np.zeros(3))
    assert host.flow_gain == pytest.approx(2.45 / 4.0, abs=1e-3)


def test_cli_parameters():
    import argparse
    from haltere.liftoff.sightpilot import add_cli_args
    q = argparse.ArgumentParser()
    add_cli_args(q)
    a = q.parse_args(['--sight', 'rabbit', '--sight-speed', '4', '--sight-elev=-30,40', '--sight-range-corr', 'none',
                      '--sight-turn-hints', '0,45', '--sight-set', 'v_blind=2.5', '--sight-search-side', 'right'])
    P = params_from_args(a, flow_gain=0.8)
    assert P.v_cruise == 4.0 and P.elev_deg == (-30.0, 40.0) and P.range_corr is None and P.turn_hints == (0.0, 45.0)
    assert P.v_blind == 2.5 and P.search_side == -1.0 and P.flow_max == 0.8 and P.yaw_rate == 2.3
    assert len(SightPilot(type('H', (), {})(), P).log_values()) == len(LOG_COLUMNS)


def test_lined_up_arches_inside_the_old_merge_window_are_not_fused():
    # 20 m and 36 m on one bearing (ln 0.59: apart for the association gate, inside the old merge window of 0.7)
    near, far = np.array([20.0, 0.0, 2.7]), np.array([36.0, 0.0, 2.7])
    count = {'n': 0}

    def where(t):
        count['n'] += 1
        return far if count['n'] % 3 == 0 else near

    rig = Rig(ScriptVision(where), SightParams(yaw_rate=3.8, range_corr=None))
    rig.hover_at((0.0, 0.0, 2.7))
    for _ in range(400):
        rig.tick()
    conf = sorted((t for t in rig.sp.tracks if t.confirmed and not t.passed), key=lambda t: t.m[0])
    assert len(conf) == 2 and np.linalg.norm(conf[0].m - near) < 1.0 and np.linalg.norm(conf[1].m - far) < 3.0
    assert rig.sp.target is conf[0]


def test_a_sharp_turn_keeps_the_next_gate_for_the_bisector():
    # the strawbale course at gate 2: gate 3 is 34 m away but only 1.9 m along the gate 1 -> 2 direction (87 deg turn)
    g1, g2, g3 = np.array([60.93, 1.575, 2.7]), np.array([79.32, 16.48, 2.7]), np.array([59.44, 44.0, 2.7])
    sp = SightPilot(type('H', (), {})(), SightParams())
    tracks = []
    for i, m in enumerate((g2, g3)):
        T = Track(i, 0.0, m, np.eye(3) * 0.2, np.array([1.0, 0.0, 0.0]), g1, 10.0)
        T.confirmed, T.hits = True, 20
        tracks.append(T)
    sp.tracks = tracks
    sp.last_pass = {'t': 0.0, 'm': tuple(g1), 'n': (1.0, 0.0), 's': 0.0, 'kind': 'cross', 'track': None}
    c = (g2 - g1)[:2] / np.linalg.norm((g2 - g1)[:2])
    p = np.r_[g2[:2] - 15.0 * c, 2.0]
    sp._axis(tracks[0], g2, p, 1.0, 0.01)
    assert sp.next is tracks[1] and sp.turn_gate
    assert abs(math.degrees(sp.n_ang) - 86.5) < 6.0                   # the arch's heading (it was 54: 32 deg off)


def _move_to(rig, pos, ticks=3):
    """Teleport the (already hovering, frozen) drone and let a few ticks run there."""
    rig.pos = np.asarray(pos, dtype=float)
    for _ in range(ticks):
        rig.tick()


def test_the_current_target_is_not_ghosted_inside_twelve_metres_while_a_neighbour_is_fed():
    # the detector reports one arch per frame: feeding only the neighbour used to delete the gate being flown at
    # (the neighbour sits 15 deg off the nose: the 30 deg camera uptilt already spends 23 deg of the off-axis budget)
    near, far = np.array([9.0, 0.0, 2.7]), np.array([24.15, 6.47, 2.7])

    def rig_for(keep_d):
        rig = Rig(ScriptVision(lambda t: far), SightParams(yaw_rate=3.8, range_corr=None, ghost_keep_d=keep_d))
        rig.hover_at((0.0, 0.0, 2.0))
        T = _crafted_target(rig, near, min_a=math.inf)
        for _ in range(500):                                   # 5 s, well past ghost_s
            rig.tick()
        return rig, T

    rig, T = rig_for(12.0)
    assert rig.sp.target is T and T in rig.sp.tracks and rig.sp.ghosts == 0
    assert T.unseen_in_view <= rig.sp._ghost_s(T)
    rig0, T0 = rig_for(0.0)                                    # ghost_keep_d 0 = the behaviour that lost the gate
    assert rig0.sp.ghosts >= 1 and T0 not in rig0.sp.tracks


def test_a_frame_that_showed_another_arch_on_the_same_bearing_still_counts_against_an_estimate():
    # ghost_evidence 'other': the timer only runs on frames whose arch is somewhere else in the image
    near, far = np.array([20.0, 0.0, 2.7]), np.array([24.15, 6.47, 2.7])         # 15 deg apart, both in view
    out = {}
    for name, fed in (('off_bearing', far), ('same_bearing', np.array([34.0, 0.0, 2.7]))):
        rig = Rig(ScriptVision(lambda t, f=fed: f), SightParams(yaw_rate=3.8, range_corr=None, ghost_keep_d=0.0))
        rig.hover_at((0.0, 0.0, 2.0))
        T = _crafted_target(rig, near, min_a=math.inf)
        for _ in range(400):
            rig.tick()
        out[name] = (rig.sp, T)
    assert out['off_bearing'][0].ghosts >= 1                   # the detector looked elsewhere: charge it
    assert out['same_bearing'][0].ghosts == 0                  # it was busy with an arch on this very bearing


def test_a_target_nothing_sees_any_more_is_dropped_but_not_while_the_detector_is_stalled():
    # ghost_keep_d resets the ghost timer and the target is exempt from conf_life, so without target_life nothing
    # retires an estimate that is simply never seen again (in replay the pilot flew at one for 45 s)
    far = np.array([40.0, 0.0, 2.7])                       # beyond ghost_range: the ghost rule cannot touch it

    def run(target_life, feed):
        rig = Rig(ScriptVision(feed), SightParams(yaw_rate=3.8, range_corr=None, target_life=target_life))
        rig.hover_at((0.0, 0.0, 2.0))
        T = _crafted_target(rig, far, min_a=math.inf)
        for _ in range(2500):                              # 25 s
            rig.tick()
        return rig.sp, T

    live = lambda t: np.array([9.0, 9.0, 2.7])             # noqa: E731  the detector is alive and sees another arch
    sp, T = run(20.0, live)
    assert sp.target is not T and T not in sp.tracks and sp.stale_targets == 1
    sp, T = run(0.0, live)                                 # 0 = off: the old behaviour, nothing retires it
    assert sp.target is T and sp.stale_targets == 0
    sp, T = run(20.0, lambda t: None)                      # ScriptVision still delivers empty frames: alive
    assert sp.target is not T and sp.stale_targets == 1
    rig = Rig(ScriptVision(lambda t: None), SightParams(yaw_rate=3.8, range_corr=None, target_life=20.0))
    rig.hover_at((0.0, 0.0, 2.0))
    T = _crafted_target(rig, far, min_a=math.inf)
    rig.pilot.vision.next_t = 1e9                          # the detector stops sending frames altogether
    for _ in range(2500):
        rig.tick()
    assert rig.sp.stalled and rig.sp.target is T and rig.sp.stale_targets == 0   # blind flight is not a lost gate


def test_the_same_arch_passed_twice_under_two_ids_counts_one_gate():
    # an orphan books the arch from its (short) estimate and the surviving track books it again a moment later
    arch, far = np.array([22.0, 0.0, 2.7]), np.array([60.0, 0.0, 6.7])
    rig = Rig(ScriptVision(lambda t: None), SightParams(yaw_rate=3.8, range_corr=None, ghost_keep_d=0.0))
    rig.hover_at((2.0, 0.0, 2.0))
    sp = rig.sp
    short = _crafted_target(rig, np.array([12.0, 0.0, 2.7]), min_a=math.inf)     # the same arch, 10 m short
    short.min_d = 1.0
    short.unseen_in_view = 99.0
    rig.tick()
    assert short not in sp.tracks and sp.orphans
    T = _crafted_target(rig, arch, min_a=math.inf)                               # the good estimate of that arch
    _move_to(rig, (14.0, 0.0, 2.0), ticks=40)                                    # past the orphan's plane
    assert sp.n_passes == 1 and abs(sp.z_pass_last - (2.7 - 1.5)) < 1e-6
    _move_to(rig, (23.0, 0.0, 2.0), ticks=40)                                    # and now past the real one
    assert T.passed and sp.n_passes == 1 and sp.dup_passes == 1                  # one arch, one gate
    assert abs(sp.last_pass['m'][0] - 22.0) < 1e-6                               # the nearer pass is the one kept
    nxt = _crafted_target(rig, far, min_a=math.inf)                              # a real neighbour, 38 m on
    _move_to(rig, (61.0, 0.0, 2.0), ticks=40)
    assert nxt.passed and sp.n_passes == 2 and sp.dup_passes == 1                # ... is still its own gate


def test_a_dropped_then_passed_track_registers_a_travel_pass_without_taking_the_target():
    arch, nxt = np.array([10.0, 0.0, 2.7]), np.array([40.0, 0.0, 2.7])
    rig = Rig(ScriptVision(lambda t: None), SightParams(yaw_rate=3.8, range_corr=None, ghost_keep_d=0.0))
    rig.hover_at((2.0, 0.0, 2.0))
    sp = rig.sp
    T = _crafted_target(rig, arch, min_a=math.inf)
    T.min_d = 1.0                                              # the drone came close to it before it was dropped
    T.unseen_in_view = 99.0                                    # ... and then it was ghosted
    rig.tick()
    assert T not in sp.tracks and sp.target is None and sp.orphans
    T2 = _crafted_target(rig, nxt, min_a=math.inf)             # already flying at the next gate
    _move_to(rig, (14.0, 0.0, 2.0))                           # the drone travels past the dropped arch
    assert sp.n_passes == 1 and sp.orphans_registered == 1
    assert sp.last_pass is not None and sp.last_pass['kind'] == 'travel' and abs(sp.last_pass['m'][0] - 10.0) < 1e-6
    assert abs(sp.z_pass_last - (2.7 - 1.5)) < 1e-6 and abs(sp.z_aim_last - 1.2) < 1e-6
    assert sp.target is T2                                     # the gate now being flown at is untouched


def test_the_altitude_reference_stays_within_the_grade_band_when_a_wild_estimate_arrives():
    tall = np.array([20.0, 0.0, 30.0])                         # an estimate 27 m too high
    for ref, ceiling in ((True, 1.2 + 2.0), (False, math.inf)):
        P = SightParams(yaw_rate=3.8, range_corr=None, z_window=(1.5, 2.0), z_window_ref=ref)
        if not ref:
            P.z_window = (3.0, 12.0)
        rig = Rig(ScriptVision(lambda t: None), P)
        rig.hover_at((0.0, 0.0, 2.0))
        sp = rig.sp
        sp.z_aim_last, sp.grade_last = 1.2, 0.0
        sp.last_pass = {'t': rig.clock(), 'm': (0.0, 0.0, 2.7), 'n': (1.0, 0.0), 's': sp.s, 'kind': 'cross',
                        'track': None}
        _crafted_target(rig, tall, min_a=math.inf)
        top = 0.0
        for _ in range(600):
            rig.tick()
            top = max(top, sp.z_c)
        if math.isfinite(ceiling):
            assert top <= ceiling + 0.02, top                   # bounded by the grade line + z_window[1]
        else:
            assert top > 5.0, top                               # the old window let the estimate lift the approach


def test_the_altitude_floor_stays_on_the_last_passage_height():
    # only the ceiling rides the grade line: a floor that climbed with it would push the approach UP (+1.9 m on w21),
    # which is the failure the ceiling was written against
    low = np.array([20.0, 0.0, 2.7])                                    # an arch whose passage point is at 1.2 m
    P = SightParams(yaw_rate=3.8, range_corr=None)
    rig = Rig(ScriptVision(lambda t: None), P)
    rig.hover_at((0.0, 0.0, 2.0))
    sp = rig.sp
    sp.z_pass_last, sp.z_aim_last, sp.grade_last = 1.2, 1.2, P.grade_max
    sp.last_pass = {'t': rig.clock(), 'm': (-20.0, 0.0, 2.7), 'n': (1.0, 0.0), 's': sp.s - 100.0,
                    'kind': 'cross', 'track': None}                     # a steep leg, its grade line fully run out
    _crafted_target(rig, low, min_a=math.inf)
    for _ in range(600):
        rig.tick()
    assert sp._z_ref() > 6.0                                            # the grade line has climbed 5 m ...
    assert sp.z_c < 1.6, sp.z_c                                         # ... and the approach has stayed at the arch


def test_a_fragment_beyond_the_target_is_neither_selected_nor_used_as_the_next_gate():
    near, frag = np.array([15.0, 0.0, 2.7]), np.array([18.0, 0.0, 2.7])         # 3 m further along the same ray

    def build(**kw):
        sp = SightPilot(type('H', (), {})(), SightParams(**kw))
        p = np.array([0.0, 0.0, 2.0])
        ts = []
        for i, m in enumerate((near, frag)):
            T = Track(i, 0.0, m, np.eye(3) * 0.2, np.array([1.0, 0.0, 0.0]), p, float(m[0]))
            T.confirmed, T.hits, T.t_last = True, 30, 0.0
            ts.append(T)
        sp.tracks, sp.target, sp.t_air = ts, ts[0], -10.0
        sp.last_pass = {'t': 0.0, 'm': (-20.0, 0.0, 2.7), 'n': (1.0, 0.0), 's': 0.0, 'kind': 'cross', 'track': None}
        sp._p = p
        sp._axis(ts[0], near, p, 0.5, 0.01)
        return sp, ts, p

    sp, ts, p = build()
    assert sp.next is None                                     # not a gate of its own: it is 3 m, not 24-35 m, away
    assert not sp._ahead_fragment(ts[0], ts[1], p) and sp._ahead_fragment(ts[1], ts[0], p)
    ts[0].t_last = -5.0                                        # the target goes stale, the fragment is seen now
    sp._select(0.5, p)
    assert sp.target is ts[0] and sp.pending is None            # ... and still does not win the target
    sp2, ts2, p2 = build(frag_gap=0.0, next_min_sep=2.0)        # both rules off: the old behaviour
    assert sp2.next is ts2[1]
    ts2[0].t_last = -5.0
    sp2._select(0.5, p2)
    assert sp2.pending is ts2[1]


def test_a_detection_far_off_the_optical_axis_does_not_move_the_target():
    cam = CAM.scaled(IN_W, IN_H)

    def deliver(sp, off_deg, pos, R, now):
        d_c = np.array([math.tan(math.radians(off_deg)), 0.0, 1.0])
        d_b = (d_c / np.linalg.norm(d_c)) @ cam.body_to_cam()
        px, ok = cam.project_body(d_b[None])
        det = Detection(now, 0.99, float(px[0, 0]), float(px[0, 1]), 30.0, d_b, 15.0, sp.det_seen + 1)
        sp._intake(now, det, (pos, R), pos)

    for off, moves in ((50.0, False), (10.0, True)):
        sp = SightPilot(type('H', (), {'vision': type('V', (), {'cam': cam})()})(), SightParams(range_corr=None))
        sp._full_init(np.zeros(3), 0.0, 0.0)
        pos, R = np.array([0.0, 0.0, 2.0]), np.eye(3)
        before = sp.rej_offaxis
        deliver(sp, off, pos, R, 1.0)
        assert bool(sp.tracks) is moves
        assert (sp.rej_offaxis == before + 1) is not moves


def test_an_arch_level_with_the_drone_dead_ahead_is_not_cut_off_axis():
    # the off-axis angle is the box's image radius, and the camera's 30 deg uptilt already spends 30 deg of it on an
    # arch at the drone's own height straight ahead: the cut has to sit above that, or the commonest geometry is lost
    cam = CAM.scaled(IN_W, IN_H)
    P = SightParams(range_corr=None)
    assert P.offaxis_max > 30.0 + 5.0, P.offaxis_max
    for dx, dy in ((15.0, 0.0), (15.0, 4.0), (8.0, 0.0)):                 # dead ahead, 15 deg off, and closer in
        sp = SightPilot(type('H', (), {'vision': type('V', (), {'cam': cam})()})(), P)
        sp._full_init(np.zeros(3), 0.0, 0.0)
        pos, R = np.array([0.0, 0.0, 2.0]), np.eye(3)
        rel = np.array([dx, dy, 0.0])                                    # the arch centre at the drone's height
        px, ok = cam.project_body(rel[None])
        d_b = cam.unproject_body(px)[0]
        det = Detection(1.0, 0.99, float(px[0, 0]), float(px[0, 1]), 30.0, d_b, float(np.linalg.norm(rel)), 1)
        assert ok[0] and SightPilot._off_axis(det, cam) > 25.0            # the uptilt alone puts it out this far
        sp._intake(1.0, det, (pos, R), pos)
        assert sp.tracks and sp.rej_offaxis == 0, (dx, dy, SightPilot._off_axis(det, cam))


def test_the_speed_sense_trim_does_not_wind_up_while_its_output_is_clipped():
    # the game's brain holds a sensed 2.45 m/s: true speed = 2.45 / flow_gain, never below 2.45 with flow_max 1
    P = SightParams(v_cruise=3.0, v_gate=3.0)
    host = type('H', (), {'flow_gain': 1.0})()
    sp = SightPilot(host, P)
    sp.t_air = 0.0
    now, dt = 5.0, 0.01
    speed = lambda: 2.45 / host.flow_gain                                # noqa: E731
    sp.v_nom = sp.v_des = 2.0                                           # 15 s of search at 2 m/s
    for _ in range(1500):
        sp._flow(now, dt, np.array([speed(), 0.0, 0.0]))
        now += dt
    assert sp.flow_trim <= 1.0 + 1e-9
    sp.v_nom = sp.v_des = 3.0
    for _ in range(400):
        sp._flow(now, dt, np.array([speed(), 0.0, 0.0]))
        now += dt
    assert abs(speed() - 3.0) < 0.15                                    # at cruise within 4 s (14 s with the windup)


def test_a_stalled_detector_keeps_the_target_and_holds_without_one():
    arch = np.array([30.0, 0.0, 2.7])
    vis = ScriptVision(lambda t: arch)
    rig = Rig(vis)
    for _ in range(600):
        rig.tick()
    sp = rig.sp
    T = sp.target
    assert T is not None and sp.mode == 2 and not sp.stalled
    x_stall = rig.pos[0]
    assert 30.0 - x_stall > 12.0                                         # the arch is well ahead and in view
    vis.next_t, vis.queue = 1e9, []                                      # the capture thread dies
    for _ in range(300):
        rig.tick()
    assert sp.stalled and sp.ghosts == 0 and sp.target is T and sp.mode == 2   # not a ghost: nothing looked
    assert sp.v_des <= sp.params.v_blind + 1e-9 and 'STALLED' in rig.pilot.vision_status
    for _ in range(1500):                                                # flown through blind, then no target: hold
        rig.tick()
    assert sp.target is None and sp.mode == 4
    p0 = rig.pos.copy()
    for _ in range(500):
        rig.tick()
    assert np.linalg.norm((rig.pos - p0)[:2]) < 0.5 and sp.mode == 4
    # no frame ever: take off and hold, no launch leg and no search
    vis2 = ScriptVision(lambda t: arch)
    vis2.next_t = 1e9
    rig2 = Rig(vis2)
    far = 0.0
    for _ in range(3000):
        rig2.tick()
        far = max(far, float(np.linalg.norm(rig2.pos[:2])))
    assert rig2.sp.mode == 4 and far < 4.0 and rig2.pos[2] > 1.0
    assert rig2.sp.log_values()[LOG_COLUMNS.index('mode')] == 4


def test_a_failing_tick_holds_the_drone_instead_of_flying_on():
    rig = Rig(ScriptVision(lambda t: np.array([60.0, 0.0, 2.7])))
    for _ in range(500):
        rig.tick()
    sp = rig.sp
    assert sp.mode in (1, 2) and np.linalg.norm(rig.vel[:2]) > 1.0

    def broken(*args, **kwargs):
        raise ZeroDivisionError('late in the tick')

    sp._yaw = broken                                                     # after the rabbit, altitude and goal steps
    p0 = rig.pos.copy()
    for _ in range(1200):
        g = rig.tick()
        assert np.isfinite(g).all()
    assert sp.errors == 1200
    assert np.linalg.norm((rig.pos - p0)[:2]) < 4.0                      # it crept 28 m before
    p1 = rig.pos.copy()
    for _ in range(500):
        rig.tick()
    assert np.linalg.norm((rig.pos - p1)[:2]) < 0.3
    n_err = sp.errors
    del sp._yaw                                                      # recovers: flies again, the goal moves smoothly
    prev = None
    for _ in range(600):
        rig.tick()
        if prev is not None:
            assert np.linalg.norm(sp.carrot - prev) < 0.1
        prev = sp.carrot.copy()
    assert sp._hold is None and sp.errors == n_err and sp.mode in (1, 2, 3)


def test_parameters_the_pilot_cannot_fly_with_are_refused():
    q = __import__('argparse').ArgumentParser()
    add_cli_args(q)
    for argv in (['--sight-yaw-rate', '0'], ['--sight-set', 'elev_deg=5'], ['--sight-set', 'flow_ref=0'],
                 ['--sight-set', 'yaw_db=0'], ['--sight-range-corr', '10:1,5:1'], ['--sight-set', 'look_tau=0']):
        with pytest.raises(SystemExit):
            params_from_args(q.parse_args(['--sight', 'rabbit', *argv]))
    params_from_args(q.parse_args(['--sight', 'rabbit', '--sight-set', 'v_blind=2.5']))


def test_fly_log_header_and_rows_have_the_same_columns():
    from haltere.liftoff.commands import _fly_log_header, _fly_log_row
    for sight in ('legacy', 'rabbit'):
        rig = Rig(ScriptVision(lambda t: np.array([20.0, 0.0, 2.7])))
        for _ in range(50 if sight == 'rabbit' else 0):
            rig.tick()
        rig.pilot.sight = sight
        q = quat_from_yaw(rig.yaw)
        fr = frame_from_sim(1.0, rig.pos, rig.vel, q, np.zeros(3), np.full(4, 0.5), np.zeros(4), rig.clock())
        rig.pilot.sensors(fr)
        pl = rig.pilot                                                   # what step() leaves behind
        pl.last_brain, pl.last_cmd, pl.last_rel_b, pl.last_target = np.zeros(4), np.zeros(4), np.zeros(3), np.zeros(3)
        pl.last_quat = q
        head = _fly_log_header(rig.pilot)
        row = _fly_log_row(rig.pilot, fr, rig.clock(), np.zeros(4), 1.0, False)
        assert len(head) == len(row) and head[-1] == 'status'
        assert ('sight_yaw' in head) == (sight == 'rabbit')


def test_a_gate_below_the_last_one_is_not_ground_clutter():
    """Straw Bale only ever climbs. Two rules took that for a law of courses, and a descending course lost
    every gate to them: the next gate was refused confirmation and then deleted as clutter."""
    from haltere.liftoff.sightpilot import SightParams, SightPilot

    P = SightParams()
    sp = SightPilot.__new__(SightPilot)
    sp.params = P
    sp.z_pass_last = 9.0
    floor = sp._z_floor()
    assert floor == max(P.z_min, 9.0 - P.z_drop_max)
    assert floor < 8.5, 'a gate 0.5 m below the last one must still be a gate'
    assert 7.0 >= floor, 'a 2 m descent between gates is an ordinary course, not clutter'
    # ... but something on the ground under a high gate still is clutter
    sp.z_pass_last = 20.0
    assert sp._z_floor() > 1.5

    # and the floor never goes under the absolute minimum height
    sp.z_pass_last = 1.3
    assert sp._z_floor() == P.z_min
