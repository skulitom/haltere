"""Straw Bale downhill scenarios in the surrogate (offline development and gate tool; never used at runtime).

The Straw Bale hilltop and downhill, rebuilt from logged flights for the descent surrogate
(`descent_rehearsal.run_batch`), and a seeded family of Straw-like variations of it. The pilot and the motor see only
the synthetic HUD marker of the checkpoints; the ground is scoring-only (`fast_rehearsal.DescentScore`: the drone flies
through it), as in `fast_rehearsal.CourseTerrain`. Nothing here is flight evidence.

Where the geometry comes from (offline; course geometry is allowed in offline scoring, never at runtime):
- the approach point: the position of straw-brain11cw13-r5-noassist-04 at 66.0 s (the hilltop approach);
- the hilltop checkpoint: where that flight passed it (-34.0, 178.6, 25.7);
- the downhill ring: the in-view ring-marker rays of straw-brain11cw13-r4b-noassist-02 (72.7-81.6 s, 122 rays) and
  -r5-noassist-04 (72.6-81.7 s, 174 rays) triangulate it at (-36.34, 117.5, 12.13) and (-36.34, 115.9, 11.76)
  (median ray residuals 0.13 and 0.08 m); their mean, rounded, (-36.34, 116.7, 11.95) is used;
- the ground under the downhill leg: the median recorded height of the drone during the 46 audited Straw downhill
  contacts (contact audit v1 contacts of every Straw log, x -41..-33, y 118-180, 7101 samples) per 1 m of y, minus
  0.1 m (the drone centre above its skids), linear between bins, flat above the top bin and on the lowest four bins'
  slope below the lowest one. It falls 9 degrees at the crest and 14-15 degrees near y 130-145: a convex hill.
Both flights are development logs of the sighted descent (SightedDescentConfig).

The scenario frame turns the approach onto +x and places the launch 45 m before the approach point with the downhill
ring 6 m above the launch plane (the identified simulator's ground is its launch plane). A run launches, climbs to the
approach point, passes the hilltop checkpoint (a left turn of about 53 degrees in the logged geometry) and flies the
downhill leg to the ring; scoring (contacts, clearance, pass heights) covers the downhill leg.

`StrawVariant(seed)` draws, per seed: the hill's slope scaled by U(0.8, 1.25) about the hilltop (its convex shape kept),
the ring U(50, 70) m beyond the hilltop checkpoint along the hill (logged: 61.9 m), U(1.2, 2.2) m above the ground
there (logged: 1.7 m), U(-3, 3) m to the side, and the approach turned by U(-40, 40) degrees about the hilltop.
`StrawVariant(None)` is the logged geometry.

usage: python -m haltere.liftoff.straw_downhill run --controller pd|brain --checkpoint CKPT --seeds 1-8|logged \
           [--n 16] [--sighted-descent on|off] --sim-seed 73 --out OUT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .fast_rehearsal import TerrainConfig

APPROACH = np.array([-13.25, 193.01, 27.37])
HILLTOP = np.array([-34.0, 178.6, 25.7])
RING = np.array([-36.34, 116.7, 11.95])
# (y, ground height) of the Straw Bale downhill from the audited contacts (see the module doc)
PROFILE = np.array([
    [129.5, 13.603], [130.5, 13.725], [131.5, 14.083], [132.5, 14.33], [133.5, 14.571], [134.5, 14.862],
    [135.5, 15.157], [136.5, 15.45], [137.5, 15.746], [138.5, 16.027], [139.5, 16.287], [140.5, 16.526],
    [141.5, 16.732], [142.5, 17.059], [143.5, 17.236], [144.5, 17.508], [145.5, 17.764], [146.5, 17.974],
    [147.5, 18.222], [148.5, 18.453], [149.5, 18.718], [150.5, 18.985], [151.5, 19.157], [152.5, 19.377],
    [153.5, 19.606], [154.5, 19.902], [155.5, 20.246], [156.5, 20.455], [157.5, 20.674], [158.5, 20.889],
    [159.5, 21.088], [160.5, 21.335], [161.5, 21.514], [162.5, 21.668], [163.5, 21.878], [164.5, 22.057],
    [165.5, 22.258], [166.5, 22.451], [167.5, 22.608], [168.5, 22.792], [169.5, 22.958], [170.5, 23.125],
    [171.5, 23.254], [172.5, 23.392], [173.5, 23.518], [174.5, 23.699], [175.5, 23.889]])
LAUNCH_BEFORE_M = 45.
RING_ABOVE_LAUNCH_M = 6.


def logged_ground(y):
    """Ground height of the logged Straw downhill at y (see the module doc)."""
    ys, zs = PROFILE[:, 0], PROFILE[:, 1]
    if y >= ys[-1]:
        return float(zs[-1])
    if y <= ys[0]:
        return float(zs[0]+(y-ys[0])*(zs[3]-zs[0])/(ys[3]-ys[0]))
    return float(np.interp(y, ys, zs))


class StrawVariant:
    """One Straw-like course: the logged geometry (seed None) or a seeded variation (see the module doc). `course` is
    the three checkpoints in the scenario frame; the ground is scoring-only (`terrain()`)."""

    def __init__(self, seed=None):
        self.seed = seed
        top = logged_ground(HILLTOP[1])
        if seed is None:
            self.scale, self.length, self.side, self.turn = 1., float(HILLTOP[1]-RING[1]), 0., 0.
            self.clearance = float(RING[2]-logged_ground(RING[1]))
        else:
            rng = np.random.default_rng([int(seed), 881])
            self.scale = float(rng.uniform(.8, 1.25))
            self.length = float(rng.uniform(50., 70.))
            self.clearance = float(rng.uniform(1.2, 2.2))
            self.side = float(rng.uniform(-3., 3.))
            self.turn = float(np.radians(rng.uniform(-40., 40.)))
        self.top = top
        ring_y = HILLTOP[1]-self.length
        self.ring = np.array([RING[0]+self.side, ring_y, self.ground(ring_y)+self.clearance])
        if seed is None:
            self.ring = RING.copy()
        c, s = np.cos(self.turn), np.sin(self.turn)
        d = (APPROACH-HILLTOP)[:2]
        self.approach = np.r_[HILLTOP[:2]+np.array([c*d[0]-s*d[1], s*d[0]+c*d[1]]), APPROACH[2]]
        a = (HILLTOP-self.approach)[:2]/np.linalg.norm((HILLTOP-self.approach)[:2])
        angle = float(np.arctan2(a[1], a[0]))
        self.rotation = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        self.launch = self.approach[:2]-LAUNCH_BEFORE_M*a
        self.z0 = float(self.ring[2]-RING_ABOVE_LAUNCH_M)
        self.course = np.array([self.to_scenario(p) for p in (self.approach, HILLTOP, self.ring)])
        e = (self.course[2]-self.course[1])[:2]
        self.slope_deg = float(np.degrees(np.arctan2(self.ground(HILLTOP[1])-self.ground(self.ring[1]),
                                                     np.linalg.norm((HILLTOP-self.ring)[:2]))))
        self.legs = {2: dict(e=e/np.linalg.norm(e), slope_deg=self.slope_deg)}
        self.config = TerrainConfig()

    def ground(self, y):
        return float(self.top+self.scale*(logged_ground(y)-self.top))

    def to_scenario(self, p):
        p = np.asarray(p, float)
        return np.r_[self.rotation @ (p[:2]-self.launch), p[2]-self.z0]

    def to_logged_xy(self, q):
        return self.rotation.T @ np.asarray(q, float)[:2]+self.launch

    def ground_at(self, target, last_pass, now, xy):
        if target != 2:
            return None
        y = self.to_logged_xy(xy)[1]
        return (self.ground(y)-self.z0, -(self.ground(y+.5)-self.ground(y-.5)), 2)

    def describe(self):
        return dict(seed=self.seed, scale=round(self.scale, 3), length_m=round(self.length, 2),
                    ring_clearance_m=round(self.clearance, 2), side_m=round(self.side, 2),
                    turn_deg=round(float(np.degrees(self.turn)), 1), leg_slope_deg=round(self.slope_deg, 2))


class _Terrain:
    """DescentScore's view of a StrawVariant (ground(target, last_pass, now, xy), legs, config)."""

    def __init__(self, variant):
        self.variant, self.legs, self.config = variant, variant.legs, variant.config

    def ground(self, target, last_pass, now, xy):
        return self.variant.ground_at(target, last_pass, now, xy)


def variants(spec):
    """'logged' (with n copies given separately) or seeds 'a-b' / 'a,b' -> [StrawVariant]."""
    if spec == 'logged':
        return [StrawVariant(None)]
    out = []
    for part in str(spec).split(','):
        lo, _, hi = part.partition('-')
        out += [StrawVariant(s) for s in range(int(lo), int(hi or lo)+1)]
    return out


def run(controller, profile, courses, *, pilot_kwargs, seconds=60., sim_seed=73):
    """descent_rehearsal.run_batch over StrawVariants (one per drone); rows with each variant's description and the
    ring pass height (None when the ring was not passed)."""
    from .descent_rehearsal import run_batch
    rows, _ = run_batch(controller, profile, [v.course for v in courses], [_Terrain(v) for v in courses],
                        pilot_kwargs=pilot_kwargs, seconds=seconds, seed=sim_seed)
    for v, r in zip(courses, rows):
        r.update(variant=v.describe(),
                 ring_pass_height_m=r['pass_heights_m'][-1] if len(r['pass_heights_m']) == 3 else None)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--controller', choices=('pd', 'brain'), required=True)
    r.add_argument('--checkpoint', required=True)
    r.add_argument('--profile', default='runs/measured-dynamics-low-speed-20260923/profile.json')
    r.add_argument('--seeds', default='logged', help="'logged' or seeds a-b")
    r.add_argument('--n', type=int, default=16, help='copies of the logged geometry (with --seeds logged)')
    r.add_argument('--sighted-descent', choices=('on', 'off'), default='off')
    r.add_argument('--seconds', type=float, default=60.)
    r.add_argument('--sim-seed', type=int, default=73)
    r.add_argument('--out', required=True)
    args = parser.parse_args()
    import torch
    from .descent_rehearsal import _summary, load_controller
    from ..train.deployed_pilot import deployed_pilot_kwargs
    torch.set_num_threads(2)
    controller = load_controller(args.controller, args.checkpoint)
    contract = 'fast_velocity_brain_v1' if args.controller == 'brain' else 'fast_velocity_pd_v1'
    kw, record = deployed_pilot_kwargs(contract, stale_evidence=True, sighted_descent=args.sighted_descent)
    courses = variants(args.seeds)*(args.n if args.seeds == 'logged' else 1)
    begin = time.time()
    rows = run(controller, json.loads(Path(args.profile).read_text()), courses, pilot_kwargs=kw,
               seconds=args.seconds, sim_seed=args.sim_seed)
    out = dict(controller=args.controller, checkpoint=args.checkpoint,
               checkpoint_sha256=controller['checkpoint_sha256'], seeds=args.seeds, n=len(courses),
               sim_seed=args.sim_seed, seconds=args.seconds, sighted_descent=args.sighted_descent, declarations=record,
               elapsed_s=round(time.time()-begin, 1), summary=_summary(rows), courses=rows,
               scope='surrogate Straw-like downhills (development geometry from logged flights); not flight evidence')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(dict(out=args.out, **out['summary'])), flush=True)


if __name__ == '__main__':
    main()
