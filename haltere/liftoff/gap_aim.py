"""Pilot side of the gap cue: when the fast pilot shifts its aim beside the ring, and by how much (runtime).

Off unless a runner passes a `GapAimConfig` to `FastRaceCue` (``--obstacle-stack``). Numpy only.

Input: causal gap samples from the camera stack (`haltere.liftoff.gap_stack`), one per processed camera frame:
``dict(time=capture time, shift=deg, valid=bool, kind=str, ring_deg=deg, lr=float, ...)``. ``shift`` is the
per-frame free-interval aim shift of `haltere.vision.gap_cue.decide` relative to the ring bearing seen in that
frame (positive = LEFT, the FLU yaw sense), NOT confirmed; ``ring_deg`` is that ring's world azimuth and ``lr``
the terrain side statistic ln(median band disparity left / right) (> 0: the left side is nearer).

Rule (`GapAimConfig`, declared in configs/obstacles/gap_pilot.json):

- A sample is accepted once (capture times strictly increase) and only while it is at most ``max_age_s`` old.
- Obstacle vote: a valid sample with |shift| >= ``active_deg`` votes for sign(shift).
- Terrain vote, only while the looming TTC governor reports terrain (expansion below the flight path) when the
  sample arrives: |lr| >= ``terrain_lr`` votes for the farther side, -sign(lr), with ``terrain_side_deg``.
- Confirmed: at least ``confirm`` of the last ``window`` accepted samples, captured within
  ``confirm_window_s`` of the newest, vote for the same side. Obstacle votes take priority over terrain
  votes. Target: the newest confirming obstacle sample's shift (clipped to +-``max_shift_deg``), or
  +-``terrain_side_deg``.
- The newest sample older than ``max_age_s`` at the tick releases the target (0).
- Side latch: after a side is confirmed, the other side cannot be confirmed until ``side_latch_s`` after the
  last confirmation of the first (the target is 0 meanwhile).
- The applied shift moves toward a confirmed target at <= ``slew_deg_s``; released, it returns to 0
  linearly over ``decay_s`` (never faster than ``slew_deg_s``).
- Conflict (`reconcile_ring`, `flag_conflict`): the gap evidence describes another ring bearing than the
  pilot's ring cue (e.g. the checkpoint switched), or the ring cue's own flag clearance points to the other
  side. The evidence is dropped, the applied shift returns to 0 at once (the pilot holds the ring cue's own
  aim) and nothing is confirmed for ``side_latch_s``.

The module outputs a bearing offset only: it never changes the requested speed.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GapAimConfig:
    active_deg: float = 2.
    confirm: int = 2
    window: int = 3
    confirm_window_s: float = .25
    side_latch_s: float = .6
    slew_deg_s: float = 40.
    max_age_s: float = .2
    decay_s: float = .3
    max_shift_deg: float = 12.
    # A gap sample whose ring azimuth differs from the ring cue's by more than this describes another ring.
    conflict_deg: float = 6.
    # The ring cue's flag clearance (aim_u beside the ring centre) counts from this offset.
    flag_conflict_deg: float = 1.
    terrain: bool = True
    terrain_side_deg: float = 6.
    terrain_lr: float = .405

    def __post_init__(self):
        values = asdict(self)
        values.pop('terrain')
        if not np.isfinite(list(values.values())).all() or min(values.values()) <= 0:
            raise ValueError('Use finite positive gap aim parameters')
        for name in ('confirm', 'window'):
            if int(getattr(self, name)) != getattr(self, name):
                raise ValueError(f'{name} counts samples')
        if self.confirm > self.window:
            raise ValueError('confirm must not exceed window')
        if not self.max_shift_deg < 45 or not self.terrain_side_deg <= self.max_shift_deg:
            raise ValueError('Keep the shift below 45 degrees and the terrain side shift within it')

    @classmethod
    def from_dict(cls, d):
        names = set(cls.__dataclass_fields__)
        unknown = set(d)-names
        if unknown:
            raise ValueError(f'unknown gap aim parameters: {sorted(unknown)}')
        return cls(**d)


def wrap_deg(a):
    return float((a+180.) % 360.-180.)


def rotate_z(vector, degrees):
    """Rotate a world vector about +z (FLU: positive = counter-clockwise = towards the left)."""
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    v = np.array(vector, dtype=float, copy=True)
    v[0], v[1] = c*vector[0]-s*vector[1], s*vector[0]+c*vector[1]
    return v


def direction_offset(applied, flag_deg, config):
    """Rotation (deg) of the pilot's aim ray (the ring cue's own aim point) for an applied gap shift.

    The gap shift is relative to the ring centre; the ring cue may already aim ``flag_deg`` beside it (flag
    clearance). Same side: the larger of the two offsets from the centre; no flag offset: the gap shift;
    opposite sides: none (a conflict, see `GapAim.flag_conflict`)."""
    if abs(applied) < 1e-9:
        return 0.
    if abs(flag_deg) < config.flag_conflict_deg:
        return float(applied)
    if applied*flag_deg > 0:
        return float(np.sign(applied)*max(abs(applied), abs(flag_deg))-flag_deg)
    return 0.


class GapAim:
    """Confirmation, side latch, slew and decay of the gap aim shift (see the module docstring)."""

    def __init__(self, config=None):
        self.config = config or GapAimConfig()
        self.samples = deque(maxlen=int(self.config.window))
        self.last_time = None
        self.target = self.applied = 0.
        self.side = 0
        self.side_until = self.hold_until = -np.inf
        self.mode = None
        self.decay_from = None
        self.blocked = False
        self.counts = dict(samples=0, stale=0, repeated=0, obstacle_votes=0, terrain_votes=0, obstacle_episodes=0,
                           terrain_episodes=0, latch_blocks=0, ring_conflicts=0, flag_conflicts=0)
        self.engaged_seconds = 0.
        self.max_applied_deg = 0.

    @property
    def engaged(self):
        return self.target != 0. or abs(self.applied) > 1e-6

    def ingest(self, sample, now, terrain=False):
        """Accept one gap sample (see the module docstring); returns whether it was new and fresh."""
        c = self.config
        stamp = None if sample is None else sample.get('time')
        if stamp is None or not np.isfinite(stamp):
            return False
        stamp = float(stamp)
        if self.last_time is not None and stamp <= self.last_time:
            self.counts['repeated'] += int(stamp < self.last_time)
            return False
        self.last_time = stamp
        if not 0 <= now-stamp <= c.max_age_s:
            self.counts['stale'] += 1
            return False
        valid = bool(sample.get('valid'))
        shift = float(sample.get('shift') or 0.)
        obstacle = int(np.sign(shift)) if valid and np.isfinite(shift) and abs(shift) >= c.active_deg else 0
        lr = sample.get('lr')
        side = 0
        if c.terrain and terrain and valid and lr is not None and np.isfinite(lr) and abs(lr) >= c.terrain_lr:
            side = -int(np.sign(lr))
        ring = sample.get('ring_deg')
        self.samples.append(dict(time=stamp, shift=shift, obstacle=obstacle, terrain=side, valid=valid,
                                 ring_deg=float(ring) if valid and ring is not None and np.isfinite(ring) else None))
        self.counts['samples'] += 1
        self.counts['obstacle_votes'] += int(obstacle != 0)
        self.counts['terrain_votes'] += int(side != 0)
        return True

    def _candidate(self, now):
        c = self.config
        if not self.samples or not 0 <= now-self.samples[-1]['time'] <= c.max_age_s or now < self.hold_until:
            return None
        newest = self.samples[-1]['time']
        recent = [s for s in self.samples if newest-s['time'] <= c.confirm_window_s]
        for key in ('obstacle', 'terrain'):
            voted = [s for s in recent if s[key]]
            # the side whose latest vote is newest first
            for sign in dict.fromkeys(s[key] for s in reversed(voted)):
                votes = [s for s in voted if s[key] == sign]
                if len(votes) >= c.confirm:
                    shift = votes[-1]['shift'] if key == 'obstacle' else sign*c.terrain_side_deg
                    return sign, float(np.clip(shift, -c.max_shift_deg, c.max_shift_deg)), key
        return None

    def step(self, now, dt):
        """Advance the target and the applied shift by one control tick; returns the applied shift (deg)."""
        c = self.config
        candidate = self._candidate(now)
        if candidate is not None and self.side and candidate[0] != self.side and now < self.side_until:
            if not self.blocked:
                self.counts['latch_blocks'] += 1
            self.blocked = True
            candidate = None
        else:
            self.blocked = False
        if candidate is not None:
            sign, shift, mode = candidate
            if self.target == 0. or sign != self.side or mode != self.mode:
                self.counts[f'{mode}_episodes'] += 1
            self.side, self.side_until, self.mode = sign, now+c.side_latch_s, mode
            self.target, self.decay_from = shift, None
            room = c.slew_deg_s*dt
            self.applied += float(np.clip(self.target-self.applied, -room, room))
        else:
            self.target = 0.
            if now >= self.side_until:
                self.side, self.mode = 0, None
            if self.decay_from is None:
                self.decay_from = abs(self.applied)
            room = min(c.slew_deg_s, self.decay_from/c.decay_s)*dt
            self.applied -= float(np.clip(self.applied, -room, room))
        if self.engaged:
            self.engaged_seconds += dt
        self.max_applied_deg = max(self.max_applied_deg, abs(self.applied))
        return self.applied

    def conflict(self, now, kind):
        """The gap evidence and the ring cue disagree: drop the evidence and hold the ring cue's own aim."""
        self.samples.clear()
        self.target = self.applied = 0.
        self.decay_from = None
        self.side, self.mode = 0, None
        self.side_until = -np.inf
        self.hold_until = now+self.config.side_latch_s
        self.counts[f'{kind}_conflicts'] += 1

    def reconcile_ring(self, ring_deg, now):
        """Forget samples of another ring bearing; a conflict when the newest valid sample (the evidence the
        current shift rests on) describes another ring while engaged. Returns whether it was a conflict."""
        c = self.config
        rings = [s for s in self.samples if s['ring_deg'] is not None]
        newest_other = bool(rings) and abs(wrap_deg(rings[-1]['ring_deg']-ring_deg)) > c.conflict_deg
        kept = [s for s in self.samples
                if s['ring_deg'] is None or abs(wrap_deg(s['ring_deg']-ring_deg)) <= c.conflict_deg]
        if len(kept) != len(self.samples):
            self.samples = deque(kept, maxlen=int(c.window))
        if newest_other and self.engaged:
            self.conflict(now, 'ring')
            return True
        return False

    def flag_conflict(self, flag_deg, now):
        """A conflict when the ring cue's flag clearance points to the other side of the ring than the shift."""
        shift = self.target if self.target != 0. else self.applied
        if abs(flag_deg) >= self.config.flag_conflict_deg and abs(shift) > 1e-6 and shift*flag_deg < 0:
            self.conflict(now, 'flag')
            return True
        return False

    def metadata(self):
        return dict(parameters=asdict(self.config), counts=dict(self.counts),
                    engaged_seconds=round(self.engaged_seconds, 3), max_applied_deg=round(self.max_applied_deg, 3))
