"""Idealised closed-loop checks of the vertical guard (offline; report and gate input only, not flight evidence).

The verifier model of rounds 3-4b (session scratchpads m4/vguard/sim.py and m4b/guard/sim_noise.py), committed so that the
held-out checks of guard version 5 can be rerun: a point drone flies +x at `speed`; its vertical speed follows the issued
vertical request with a first-order lag `tau`; looming samples arrive at 18 Hz, 0.08 s after capture; the vertical request
is the pilot's own passed through FastRaceCue's vertical composition (`_guard_vertical`, the ceiling guard's bound) and
the command slew (5 m/s^2, 10 m/s^2 for a terrain climb or an arrest, 15 m/s^2 down under an overhead bound). The
governor is the TTC policy with the ceiling guard declared (as flown) and the vertical guard under test.

Scenarios (all noisy: multiplicative log-normal noise of `sigma` on each TTC of each sample, seeded):
  ramp   terrain rising at `slope` from 12 m ahead; a sample's alarm TTC and lower TTC are the flight path's true
         crossing time with the terrain (x noise), below_fraction 0.9 (the round-4b model);
  hill   the ramp with split readings: each sample is, with probability `split`, a reading of the slope's face at the
         flight path (below_fraction uniform in [0.45, 0.69], the lower TTC lengthened by a factor uniform in [1.5, 3.0];
         the alarm TTC unchanged). The split model was set from the development log pine-fast6-r6-01 (during the gentle
         climb 7 of 13 samples had below_fraction 0.05-0.64 with lower TTCs of 1.1-2.7 s, while the flight path's own
         crossing read 0.3-1.3 s); the seeds are fresh;
  floor  the r4-02 onset shape (round 4b): a flat floor under a 2.2 m ceiling, the drone at 0.45 m climbing 0.45 m/s at
         4.3 m/s, the pilot asking +0.45 m/s for 0.3 s and level after; the lower window misreads the floor as a crossing
         at ttc_lower = z / (0.22 speed) (x noise), alarm uniform in [1.1, 1.7] s, below_fraction 1.0; the ceiling's own
         crossing is sampled (below_fraction 0.1) when it comes first;
  floor_split  the floor with split readings as in hill (a structure ahead at path height), probability `split`.
Reported per case: the share of seeds that escalate (vertical_stage 2), the minimum ground clearance (ramp, hill) or the
maximum height (floor, floor_split; the ceiling is at 2.2 m).

usage: python haltere/obstacles/vertical_ideal.py --tree TREE --declaration JSON --seeds FIRST COUNT --out OUT.json
(``--declaration``: a vertical-guard declaration file, parsed without the runner's version check, so that version 4 and
version 5 can be run in the same tree.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SLOPES = (.2, .35, .5)
STARTS = ((1.5, 0., 0.), (1., -.5, -.8))          # (height, vertical speed, pilot's request)


def crossing(x, z, vx, vz, surface, horizon=3.):
    for k in range(1, int(horizon/.01)+1):
        tt = k*.01
        s = surface(x+vx*tt)
        if s is not None and z+vz*tt <= s:
            return tt
    return None


def fly(frc, guard, scenario, rng, *, slope=.35, speed=6., z0=1.5, vz0=0., pilot_vz=0., tau=.25, seconds=4.,
        sigma=.25, split=0., pilot_for=None):
    gov = frc.TtcClearanceGovernor(frc.TtcClearanceConfig(), ceiling=frc.CeilingGuardConfig(), vertical=guard)
    z, vz, cmd = z0, vz0, vz0
    pending, next_frame = [], 0.
    min_ground, max_z, escalated = np.inf, -np.inf, False
    ramp = scenario in ('ramp', 'hill')
    terrain = (lambda x: max(0., (x-12.)*slope)) if ramp else (lambda x: 0.)
    for k in range(int(seconds/.01)):
        now = k*.01
        x = speed*now
        if now >= next_frame-1e-9:
            s = None
            if ramp:
                tg = crossing(x, z, speed, vz, terrain)
                if tg is not None:
                    s = dict(ttc=tg*np.exp(rng.normal(0, sigma)), below=.9, lower=tg*np.exp(rng.normal(0, sigma)))
            else:
                lower = max(z, .02)/(.22*speed)*np.exp(rng.normal(0, sigma))
                tc = None if vz <= 0 else (2.2-z)/vz*np.exp(rng.normal(0, sigma))
                if tc is not None and tc < min(lower, 3.):
                    s = dict(ttc=tc, below=.1, lower=lower)
                else:
                    s = dict(ttc=rng.uniform(1.1, 1.7), below=1., lower=lower)
            if s is not None and split > 0 and scenario in ('hill', 'floor_split') and s['below'] > .5:
                if rng.uniform() < split:
                    s = dict(ttc=s['ttc'], below=rng.uniform(.45, .69), lower=s['lower']*rng.uniform(1.5, 3.))
            if s is not None:
                pending.append((now+.08, now, x, z, s))
            next_frame += 1/18.
        while pending and pending[0][0] <= now+1e-9:
            _, t, xc, zc, s = pending.pop(0)
            gov.ingest(t, s['ttc'], s['ttc']*speed, s['below'], [xc, 0., zc], [1., 0., 0.], speed, received=now,
                       ttc_lower=s['lower'])
        pilot = pilot_vz if pilot_for is None or now < pilot_for else 0.
        gov.pilot_vertical = pilot
        _, _, climb = gov.limits([x, 0., z], [speed, 0., vz], now, .01, 3.5)
        escalated |= bool(gov.escalated and climb > 0)
        want = frc.FastRaceCue._guard_vertical(gov, pilot, climb, np.array([speed, 0., vz]))
        if gov.vertical_cap is not None:
            want = min(want, gov.vertical_cap)
        up = max(5., 10. if climb > 0 else 0., 10. if gov.arrest else 0.)
        down = 5. if gov.vertical_cap is None else 15.
        cmd += float(np.clip(want-cmd, -down*.01, up*.01))
        vz += (cmd-vz)*.01/tau
        z += vz*.01
        min_ground = min(min_ground, z-terrain(x+speed*.01))
        max_z = max(max_z, z)
    return escalated, min_ground, max_z


def cases():
    """Every case of the check: (name, scenario, kwargs)."""
    out = []
    for scenario, split in (('ramp', 0.), ('hill', .5)):
        for slope in SLOPES:
            for z0, vz0, pilot in STARTS:
                name = f'{scenario} slope {slope:.2f} z0 {z0} vz0 {vz0:+.1f} pilot {pilot:+.1f}'
                out.append((name, scenario, dict(slope=slope, z0=z0, vz0=vz0, pilot_vz=pilot, split=split)))
    for scenario, split in (('floor', 0.), ('floor_split', .5)):
        for sigma in (.25, .5):
            out.append((f'{scenario} sigma {sigma}', scenario,
                        dict(speed=4.3, z0=.45, vz0=.45, pilot_vz=.45, seconds=3., sigma=sigma, pilot_for=.3,
                             split=split)))
    return out


def run(frc, guard, first, count):
    results = {}
    for name, scenario, kw in cases():
        per_seed = [fly(frc, guard, scenario, np.random.default_rng(seed), **kw) for seed in range(first, first+count)]
        escalated = [bool(o[0]) for o in per_seed]
        value = np.array([o[1] if scenario in ('ramp', 'hill') else o[2] for o in per_seed])
        results[name] = dict(scenario=scenario, escalated_share=round(float(np.mean(escalated)), 4),
                             metric='min_ground_clearance_m' if scenario in ('ramp', 'hill') else 'max_height_m',
                             median=round(float(np.median(value)), 4),
                             worst=round(float(value.min() if scenario in ('ramp', 'hill') else value.max()), 4),
                             per_seed=[round(float(v), 4) for v in value], per_seed_escalated=escalated)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tree', required=True)
    parser.add_argument('--declaration', required=True)
    parser.add_argument('--seeds', nargs=2, type=int, required=True, metavar=('FIRST', 'COUNT'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(args.tree).resolve()))
    from haltere.liftoff import fast_race_cue as frc
    declaration = json.loads(Path(args.declaration).read_text(encoding='utf-8'))
    guard = frc.VerticalGuardConfig(**declaration['vertical_guard'])
    results = run(frc, guard, *args.seeds)
    Path(args.out).write_text(json.dumps(dict(declaration=str(args.declaration), version=declaration['version'],
                                              sha256=declaration.get('sha256'), seeds=list(args.seeds),
                                              results=results), indent=1), encoding='utf-8')
    for name, r in results.items():
        print(f"{name}: escalated {r['escalated_share']:.0%}, {r['metric']} median {r['median']:.2f} worst "
              f"{r['worst']:.2f}")


if __name__ == '__main__':
    main()
