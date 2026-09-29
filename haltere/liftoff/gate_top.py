"""Gate tops in the surrogate: scoring-only arch bars, the two Straw Bale downhill arches, closed-loop windows from logged
states, and a two-ring Straw downhill (offline development and gate tool; never used at runtime).

The pilot and the motor never see anything here except the synthetic HUD marker of each checkpoint's centre
(`fast_rehearsal.hud_marker`); the arch bars and the ground are scoring-only (the drone flies through them). Course
geometry and logged positions are used offline for scoring and for the synthetic marker only, never by a rule. Nothing
here is flight evidence.

**The arch bar model (`GateTopConfig`).** A checkpoint is a ring whose arch has a top bar across the plane through the
ring centre normal to the approach (the horizontal direction from the previous checkpoint): a drone that crosses that
plane with its centre bar_low_m to bar_high_m above the ring centre, within bar_half_width_m of the centre sideways,
hits the bar; above bar_high_m it passes over the arch. The values come from the logged crossings of the two Straw Bale
downhill arches (offline; `STRAW_ARCHES` below, drone centre above the ring centre): passes at up to +0.67 m (arch A)
and +0.92 m (arch B, `straw-brain11cw13-r4b-noassist-02`, grazing), top-bar impacts at +0.83 to +1.09 m (A:
`straw-brain04-01`, `-05-01`, `-03-01`, `-05-trim-01`, `straw-fast6-arc-01` lap 3, `straw-brain11cw13-r5-noassist-04`;
B: `straw-brain07-steep-02`, `-07-nosweep-01` lap 2, `straw-brain11cw13-r6-01`) and one pass over arch B at +1.33 m
(`straw-brain07-nosweep-01` lap 1).

**The two Straw arches (`STRAW_ARCHES`).** Ring centres from the in-view ring-marker rays of every Straw Bale log (the
pilot's own ray computation: the pose at capture, the calibrated camera), each ray's height where it meets the arch
plane (the y of the audited top-bar impacts); the rays taken 3-10 m before the plane (646 for A, 208 for B; the least
range-sensitive) give 12.46 m (IQR 12.44-12.48) for A (ImmersionRC, y 120.7) and 6.05 m (IQR 6.02-6.09) for B (the
next arch, y 95.9). A plain least-squares triangulation of the same rays is ill-conditioned along the approach (the
rays are nearly parallel: its per-log estimates spread 116-152 m in y for A), which is why the plane heights are used.

**Closed-loop windows (`window`).** The drone of a logged flight from its logged state at t0 in the identified
surrogate (`brake_gates`: the logged position, velocity, attitude, rates and drive; the brain warmed on its recorded
inputs), the pilot warmed on the log up to t0 (`vertical_replay.replay(until=t0)`: the same rows, cues, looming and
gap samples as the open-loop replays), then both in closed loop with the synthetic marker of the remaining arches (no
looming or gap samples, 10% seeded marker dropout, the rehearsal's camera period and latency). It records the height
above each ring centre where the drone crosses its arch plane, the arch-bar verdict and, with a ground function, the
ground clearance.

**The two-ring Straw downhill (`StrawTwoRings`).** `straw_downhill.StrawVariant` with the next arch (B) after the
ImmersionRC ring and the ground under that leg: the logged geometry (A and B above, the ground under A->B a straight
line from the downhill profile's ground under A to 1.7 m below B, flat beyond B) and seeded variations (the downhill
as StrawVariant's; B U(18, 32) m beyond A, turned U(-20, 20) degrees, on a U(8, 20) degree slope, U(1.2, 2.2) m above
the ground there). The launch plane (the identified simulator's ground) lies 6 m below the lower of StrawVariant's
ring and B.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

RUNS = Path('C:/DEV/Haltere/runs/fast-stack-20260923')


@dataclass(frozen=True)
class GateTopConfig:
    """Scoring-only arch bars (see the module doc). pass_on_plane: descent_rehearsal.run_batch passes a checkpoint
    where the drone crosses its arch plane within its radius of the centre, as a gate is flown through (instead of
    within the radius of the centre point)."""
    bar_low_m: float = .85
    bar_high_m: float = 1.35
    bar_half_width_m: float = 2.
    pass_on_plane: bool = True

    def __post_init__(self):
        if not 0 < self.bar_low_m < self.bar_high_m or not self.bar_half_width_m > 0:
            raise ValueError('Use 0 < bar_low_m < bar_high_m and a positive bar half width')


# The two Straw Bale downhill arches (offline, see the module doc): ring centre, approach direction (the logged
# approaches fly south, -y), and the source.
STRAW_ARCHES = {
    'A': dict(centre=(-36.34, 120.7, 12.46), approach=(0., -1., 0.), name='ImmersionRC arch'),
    'B': dict(centre=(-36.2, 95.9, 6.05), approach=(0., -1., 0.), name='the next arch (white top banner)'),
}


class GateTop:
    """Plane crossings of a course's checkpoints and their arch-bar verdicts. `course` (N, 3) checkpoint centres,
    `start` the point the first leg starts from; `normals` (optional) the approach direction of each checkpoint
    (default: from the previous checkpoint). Each checkpoint's plane is crossed at most once (the first crossing
    within reach_m sideways)."""

    def __init__(self, course, start=(0., 0., 0.), config=None, normals=None, reach_m=6.):
        self.config = config or GateTopConfig()
        self.course = np.asarray(course, float)
        self.reach_m = float(reach_m)
        if normals is None:
            normals = []
            previous = np.asarray(start, float)
            for c in self.course:
                d = (c-previous)[:2]
                n = d/max(np.linalg.norm(d), 1e-9)
                normals.append(np.r_[n, 0.])
                previous = c
        self.normals = np.asarray(normals, float)
        self.crossings = [None]*len(self.course)
        self.previous = None

    def step(self, position):
        """Call once per tick with the drone's position; returns the indices crossed this tick."""
        position = np.asarray(position, float)
        crossed = []
        if self.previous is not None:
            for i, (c, n) in enumerate(zip(self.course, self.normals)):
                if self.crossings[i] is not None:
                    continue
                a, b = float((self.previous-c) @ n), float((position-c) @ n)
                if a < 0 <= b:
                    f = -a/max(b-a, 1e-12)
                    p = self.previous+f*(position-self.previous)
                    side = np.array([-n[1], n[0], 0.])
                    lateral = float((p-c) @ side)
                    if abs(lateral) > self.reach_m:
                        continue
                    above = float(p[2]-c[2])
                    g = self.config
                    verdict = ('bar' if g.bar_low_m <= above <= g.bar_high_m and abs(lateral) <= g.bar_half_width_m
                               else 'over' if above > g.bar_high_m and abs(lateral) <= g.bar_half_width_m else 'through')
                    self.crossings[i] = dict(above_m=round(above, 3), lateral_m=round(lateral, 3), verdict=verdict)
                    crossed.append(i)
        self.previous = position.copy()
        return crossed

    def through(self, index, previous, position, radius):
        """Whether the segment previous -> position crosses checkpoint `index`'s plane within `radius` of its centre
        (sideways and vertically): a gate flown through."""
        c, n = self.course[index], self.normals[index]
        previous, position = np.asarray(previous, float), np.asarray(position, float)
        a, b = float((previous-c) @ n), float((position-c) @ n)
        if not a < 0 <= b:
            return False
        p = previous+(-a/max(b-a, 1e-12))*(position-previous)
        side = np.array([-n[1], n[0], 0.])
        return abs(float((p-c) @ side)) <= radius and abs(float(p[2]-c[2])) <= radius

    def result(self):
        rows = [c for c in self.crossings if c is not None]
        return dict(top_bar_hits=sum(c['verdict'] == 'bar' for c in rows),
                    over_arch=sum(c['verdict'] == 'over' for c in rows),
                    crossings=self.crossings)


# ---------------------------------------------------------------------------------------------------------------
# Closed-loop windows from a logged state
# ---------------------------------------------------------------------------------------------------------------
def window(flight, t0, arches, pilot_kwargs, *, controller, profile, seconds=12., seed=0, dropout=.1,
           camera_period=.055, camera_latency=.06, warm_s=8., ground=None, tree=None, runs=RUNS, config=None,
           record=False, inject=()):
    """One closed-loop window (see the module doc). `arches`: [(centre, approach unit vector)] of the checkpoints still
    ahead at t0, in order; `pilot_kwargs`: the replay harness's `replay` keyword arguments of the pilot variant (stack,
    descent_view, ...); `controller`: a brake_gates.Controller (the brain) whose checkpoint the flight flew; `ground`:
    an optional callable (x, y) -> ground height (scoring only); `inject`: [(logged phase, cue dict or None)] readings
    of the live reader that replace the synthetic marker of the first capture at or after that time (e.g. a logged false
    marker or unread captures, so that a window keeps the live reader's errors). Returns a dict (crossings, verdicts,
    clearance)."""
    import torch
    from collections import deque
    from ..obstacles.vertical_replay import replay
    from ..sim.identified import IdentifiedSim
    from ..train import brake_gates as bg
    from ..train.fast_motor_tracking import brain_observation
    from ..brain.retina import RETINA_DIM
    from ..vision.camera import Camera
    from .fast_rehearsal import hud_marker
    tree = tree or str(Path(__file__).resolve().parents[2])
    _, pilot, info = replay(flight, tree, runs, until=t0, **pilot_kwargs)
    k0 = info['next_row']
    f = bg.Flight(runs, flight)
    row = f.d.iloc[k0]
    now0 = float(row.capture_time)+float(row.image_age)
    sim = IdentifiedSim(profile, controller.calibration, 'cpu', bg.DT)
    sim.randomize(1, 0.)
    s = f.state_fn(k0)(sim, 1)
    q = deque(torch.tensor(f.cmds[k0-3+i], dtype=torch.float32)[None] for i in range(3))
    state = f.warm(controller, k0, warm_s)
    camera = Camera(320, 180, pilot.camera.f, pilot.camera.tilt_deg)
    rng = np.random.default_rng([int(seed), 7127])
    centres = np.array([a[0] for a in arches], float)
    top = GateTop(centres, normals=[np.asarray(a[1], float) for a in arches], config=config)
    target, pending, detection, capture = 0, deque(), None, None
    next_capture = now0
    phase0 = float(row.phase)
    injected = sorted((float(t), c) for t, c in inject)
    min_clearance, contact_s = np.inf, 0.
    support, last_state = 0, pilot.state
    trace = [] if record else None
    for j in range(int(round(seconds/bg.DT))):
        now = now0+j*bg.DT
        position = s.quad.pos[0].numpy().astype(float)
        quaternion = s.quad.quat[0].numpy().astype(float)
        for i in top.step(position):
            if i == target:
                target += 1
        if target >= len(centres):
            break
        if ground is not None:
            clearance = float(position[2]-ground(position[0], position[1]))
            min_clearance = min(min_clearance, clearance)
            contact_s += bg.DT*(clearance < 0)
        pilot.pose_history.append(now, position, quaternion)
        if now >= next_capture:
            cue = hud_marker(camera, centres[target], position, quaternion) if rng.random() > dropout else None
            if injected and phase0+j*bg.DT >= injected[0][0]:
                _, forced = injected.pop(0)
                cue = None if forced is None else dict(forced)
            if cue is not None:
                cue.setdefault('aim_u', cue['u'])
            pending.append((now, now+camera_latency, cue))
            next_capture = now+camera_period
        while pending and pending[0][1] <= now:
            capture, _, cue = pending.popleft()
            detection = dict(race_cue=cue) if cue is not None else None
        senses = sim.sensors(s)
        pilot.update(senses, senses['gyro'][0].numpy(), detection, capture, now)
        support += int(pilot.state == 'support_climb' and last_state != 'support_climb')
        last_state = pilot.state
        request = torch.tensor(np.asarray(pilot.velocity_command, float)[None], dtype=torch.float32)
        obs = brain_observation(controller.meta, senses, s.quad.motor.mean(-1, keepdim=True), controller.cfg.task,
                                torch.zeros(1, RETINA_DIM), request, controller.contract)
        with torch.no_grad():
            action, state, _ = controller.brain(obs, state, controller.W)
        command = torch.as_tensor(pilot.command(action[0].numpy()), dtype=torch.float32)[None]
        q.append(command)
        s = sim.step(s, q.popleft())
        v = s.quad.vel
        s.quad.vel = v-.0075*v.norm(dim=-1, keepdim=True)*v*bg.DT
        s.quad.crashed[:] = False
        if record:
            trace.append(dict(t=round(float(row.phase)+j*bg.DT, 3), pos=np.round(position, 3).tolist(),
                              vel=np.round(s.quad.vel[0].numpy().astype(float), 3).tolist(),
                              req=np.round(np.asarray(pilot.velocity_command, float), 3).tolist(), state=pilot.state,
                              target=int(target)))
    out = dict(flight=flight, t0=t0, seed=int(seed), start_row=int(k0), **top.result(),
               min_ground_clearance_m=None if not np.isfinite(min_clearance) else round(min_clearance, 3),
               ground_contact_s=round(contact_s, 3), support_climbs=int(support), sighted=pilot.sighted_summary())
    if record:
        out['trace'] = trace
    return out


# ---------------------------------------------------------------------------------------------------------------
# The two-ring Straw downhill
# ---------------------------------------------------------------------------------------------------------------
B_CLEARANCE_LOGGED_M = 1.7


class StrawTwoRings:
    """straw_downhill.StrawVariant (the hilltop approach, the hilltop checkpoint and the ImmersionRC ring A) with the
    next arch B after A and the ground under the A->B leg (see the module doc); seed None is the logged geometry, with
    A and B at STRAW_ARCHES. `course` is the four checkpoints in the scenario frame; the ground and the arch bars are
    scoring-only."""

    def __init__(self, seed=None):
        from .straw_downhill import HILLTOP, RING_ABOVE_LAUNCH_M, StrawVariant
        self.base = base = StrawVariant(seed)
        self.seed = seed
        if seed is None:
            # the logged downhill with ring A where its arch plane is (on the ray of StrawVariant's triangulated ring)
            a = np.array(STRAW_ARCHES['A']['centre'])
            b = np.array(STRAW_ARCHES['B']['centre'])
            self.a_ground = float(base.ground(a[1]))
            self.b_clearance = B_CLEARANCE_LOGGED_M
            self.b_ground = float(b[2]-self.b_clearance)
            self.b_length = float(np.linalg.norm((b-a)[:2]))
            self.b_turn = 0.
            self.b_slope = float(np.degrees(np.arctan2(self.a_ground-self.b_ground, self.b_length)))
        else:
            rng = np.random.default_rng([int(seed), 3571])
            a = base.ring.copy()
            self.a_ground = float(base.ground(a[1]))
            self.b_length = float(rng.uniform(18., 32.))
            self.b_turn = float(rng.uniform(-20., 20.))
            self.b_slope = float(rng.uniform(8., 20.))
            self.b_clearance = float(rng.uniform(1.2, 2.2))
            leg = (a-HILLTOP)[:2]
            heading = float(np.arctan2(leg[1], leg[0]))+np.radians(self.b_turn)
            b_xy = a[:2]+self.b_length*np.array([np.cos(heading), np.sin(heading)])
            self.b_ground = self.a_ground-self.b_length*float(np.tan(np.radians(self.b_slope)))
            b = np.r_[b_xy, self.b_ground+self.b_clearance]
        self.a, self.b = a, b
        # the identified simulator's ground is its launch plane: B (the lowest ring) at least RING_ABOVE_LAUNCH_M above it
        self.z0 = float(min(base.z0, b[2]-RING_ABOVE_LAUNCH_M))
        self.course = np.array([self.to_scenario(p) for p in (base.approach, HILLTOP, a, b)])
        self.config = base.config
        e2 = (self.course[2]-self.course[1])[:2]
        e3 = (self.course[3]-self.course[2])[:2]
        self.legs = {2: dict(e=e2/np.linalg.norm(e2), slope_deg=base.slope_deg),
                     3: dict(e=e3/np.linalg.norm(e3), slope_deg=self.b_slope)}

    def to_scenario(self, p):
        p = np.asarray(p, float)
        return np.r_[self.base.rotation @ (p[:2]-self.base.launch), p[2]-self.z0]

    def ground_logged(self, x, y):
        """Ground height (logged frame) at (x, y): the downhill profile before A's plane (along A->B), the straight
        A->B ground after it, flat beyond B (scoring only)."""
        ab = (self.b-self.a)[:2]
        s = float((np.array([x, y], float)-self.a[:2]) @ ab)/float(ab @ ab)
        if s <= 0:
            return float(self.base.ground(y))
        return float(self.a_ground+min(s, 1.)*(self.b_ground-self.a_ground))

    def ground_at(self, target, last_pass, now, xy):
        """(ground height in the scenario frame, dh/ds along the leg, leg) while the drone flies the downhill or the
        A->B leg, else None."""
        if target not in (2, 3):
            return None
        xy = np.asarray(xy, float)[:2]
        e = self.legs[target]['e']
        g = self.ground_logged(*self.base.to_logged_xy(xy))
        ahead = self.ground_logged(*self.base.to_logged_xy(xy+.5*e))
        behind = self.ground_logged(*self.base.to_logged_xy(xy-.5*e))
        return (g-self.z0, ahead-behind, target)

    def describe(self):
        return dict(**self.base.describe(), b_length_m=round(self.b_length, 2), b_turn_deg=round(self.b_turn, 1),
                    b_slope_deg=round(self.b_slope, 2), b_clearance_m=round(self.b_clearance, 2))


class _TwoRingTerrain:
    """DescentScore's view of a StrawTwoRings (ground(target, last_pass, now, xy), legs, config)."""

    def __init__(self, variant):
        self.variant, self.legs, self.config = variant, variant.legs, variant.config

    def ground(self, target, last_pass, now, xy):
        return self.variant.ground_at(target, last_pass, now, xy)


def gate_top_record(config=None):
    return asdict(config or GateTopConfig())
