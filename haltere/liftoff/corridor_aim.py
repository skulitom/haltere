"""Pilot side of the free-space corridor planner (runtime): the aim offset beside the ring and the graded vertical guard.

Off unless a runner passes the configs to `FastRaceCue` (``--obstacle-planner shadow|on`` inside
``--obstacle-stack``). Numpy only; no course geometry, routes, per-course parameters or offline labels (covered by
tests/test_obstacle_label_isolation.py).

Input: causal planner samples from the depth process (``camera_process.plan_sample``: one dict per processed frame
with the ``camera_process.PLAN_FIELDS``; ``kind`` is a ``PLAN_KINDS`` name, ``valid`` a bool, None = not
applicable). ``az``/``el`` and the per-class options ``l_*``/``r_*``/``v_*`` are offsets from the ring cue ray of
the same frame (degrees; +az = left, +el = up), NOT confirmed; ``*_ok`` 0 marks an option whose lag-aware path
cannot clear (urgent); ``v_cap`` (m/s) is published for ``blocked`` and for an urgent ``shift``; ``h_floor``,
``h_ceil``, ``rise`` and ``rise_x`` are metres from the camera at capture.

`CorridorAim` (the free-space declaration's ``pilot`` section, bounds ``plan.max_az_deg``/``plan.max_el_deg``):

- A sample is taken once (by seq and capture time) and only while it is valid and at most ``max_age_s`` old.
- Confirmation: ``confirm`` of the last ``window`` accepted samples, captured within ``confirm_window_s`` of the
  newest, are kind ``shift`` with the same class (+1 left, -1 right, +2 vertical); blocked: ``confirm`` of them are
  kind ``blocked``.
- Latch: the confirmed class is held for at least ``side_latch_s`` and as long as the ring path stays blocked
  (kind shift or blocked). Its target is that class's option in the newest accepted sample (a missing option holds
  the previous target). The latch flips to another confirmed class only after the latched class had no option in
  ``flip_after`` consecutive accepted samples: it never swaps to the far side while its own side is open.
- Release: ``release_after`` consecutive clear/aperture samples (after the minimum latch), or ``stale_release_s``
  without an accepted sample; the offset then decays linearly to 0 over ``decay_s``. Otherwise the offset moves to
  the target at <= ``slew_az_deg_s`` / ``slew_el_deg_s``, bounded to |az| <= ``max_az_deg``, 0 <= el <= ``max_el_deg``.
- Conflicts (as in `gap_aim.GapAim`): a sample whose ring azimuth differs from the ring cue's by more than
  ``conflict_deg``, or a ring-cue flag clearance of at least ``flag_conflict_deg`` on the other side of the offset,
  drops the evidence; the offset returns to 0 at once and nothing is confirmed for ``conflict_hold_s``.
- Turn-first active: the az target is 0 (decaying); the vertical option, the speed cap and the vertical guard stay.
  The search, launch, wait and support-climb states reset it.
- Speed: on a confirmed blocked state, or a confirmed class whose newest sample is an urgent shift (feasible 0),
  the sample's ``v_cap`` (at least ``v_cap_floor_mps``) caps the request along the applied aim (FastRaceCue, at
  ``cap_slew_mps2``).

`VerticalGuard` (the declaration's ``vertical`` section) from the newest sample while it is valid and fresh; heights are
dead-reckoned with the measured vertical speed. In order, on the vertical request:
1. floor bound: vz >= lo = -(h_floor_now - floor_clear_m)/tau_s, a positive lo capped at gentle_up_mps (a sink
   toward the floor is slowed and stopped, never answered with a hard climb);
2. descent first: while the measured vz < -level_band_mps rule 3 is not applied;
3. terrain climb (rising ground while level): once ``rise`` is finite in ``confirm`` of the last ``window`` accepted
   samples, vz >= min(climb_max_mps, v_h tan(gamma+) + rise/max(rise_x/v_h - rise_lag_s, rise_min_t_s)), held
   climb_hold_s after the last confirming sample, then released at climb_release_mps2, ended climb_max_rise_m above
   the episode start (gamma+ is the non-negative flight-path angle at the sample's capture, the path the rise is
   measured from);
4. ceiling bound (last): vz <= hi = max((h_ceil_now - ceil_clear_m)/tau_s, -gentle_down_mps); hi < lo requests
   (lo + hi)/2 (a squeeze).
Ramps: rule 1 at floor_accel_mps2, rule 3 at climb_accel_mps2, rule 4 lowers at ceil_accel_mps2.

The modules output offsets, a speed cap and a vertical request only; FastRaceCue decides what is applied (nothing in
shadow).
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields

import numpy as np

FREE_SPACE_SCHEMA = 'haltere.obstacles.free_space.v1'
# The free-space declaration version whose pilot rules this code implements; runners refuse others.
FREE_SPACE_VERSION = 1
# Kinds on which the pilot may act (camera_process.PLAN_KINDS); every other kind is not a usable sample.
PLAN_VALID_KINDS = ('clear', 'aperture', 'shift', 'blocked')
CLEAR_KINDS = ('clear', 'aperture')
LEFT, RIGHT, VERTICAL = 1, -1, 2
BLOCKED = 9                      # plan_cls_confirmed log value of a confirmed blocked ring path
# Pilot states that own the request and reset the corridor aim.
PLAN_SUSPEND_STATES = ('search', 'launch', 'wait', 'support_climb')


def _finite(value):
    return value is not None and np.isfinite(value)


def _value(sample, key):
    """A sample field as float, NaN when missing or None."""
    v = sample.get(key) if sample else None
    return float(v) if v is not None and np.isfinite(v) else float('nan')


def wrap_deg(a):
    return float((a+180.) % 360.-180.)


@dataclass(frozen=True)
class CorridorAimConfig:
    max_age_s: float = .2
    confirm: int = 2
    window: int = 3
    confirm_window_s: float = .25
    side_latch_s: float = .6
    flip_after: int = 3
    release_after: int = 2
    slew_az_deg_s: float = 40.
    slew_el_deg_s: float = 20.
    decay_s: float = .3
    conflict_deg: float = 6.
    flag_conflict_deg: float = 1.
    conflict_hold_s: float = .6
    v_cap_floor_mps: float = 1.
    cap_slew_mps2: float = 15.
    # plan.max_az_deg / plan.max_el_deg of the declaration (the planner's own option bounds)
    max_az_deg: float = 20.
    max_el_deg: float = 12.
    # Release after this long without an accepted sample (spec section 6; declaration v1 lists no key for it,
    # so it is this declared constant and is written to the sidecar with the other parameters).
    stale_release_s: float = .3

    def __post_init__(self):
        values = asdict(self)
        if not np.isfinite(list(values.values())).all() or min(values.values()) <= 0:
            raise ValueError('Use finite positive corridor aim parameters')
        for name in ('confirm', 'window', 'flip_after', 'release_after'):
            if int(getattr(self, name)) != getattr(self, name):
                raise ValueError(f'{name} counts samples')
        if self.confirm > self.window:
            raise ValueError('confirm must not exceed window')
        if not self.max_az_deg < 45 or not self.max_el_deg < 45:
            raise ValueError('Keep the corridor offsets below 45 degrees')


@dataclass(frozen=True)
class VerticalGuardConfig:
    floor_clear_m: float = .5
    ceil_clear_m: float = .5
    tau_s: float = .5
    gentle_up_mps: float = 1.
    gentle_down_mps: float = 1.
    level_band_mps: float = .3
    rise_lag_s: float = .3
    rise_min_t_s: float = .3
    climb_max_mps: float = 3.5
    climb_hold_s: float = .5
    climb_release_mps2: float = 3.
    climb_max_rise_m: float = 2.5
    climb_accel_mps2: float = 10.
    floor_accel_mps2: float = 5.
    ceil_accel_mps2: float = 15.
    fallback_descent_first: bool = True

    def __post_init__(self):
        values = asdict(self)
        values.pop('fallback_descent_first')
        if not np.isfinite(list(values.values())).all() or min(values.values()) <= 0:
            raise ValueError('Use finite positive vertical guard parameters')
        if not isinstance(self.fallback_descent_first, bool):
            raise ValueError('fallback_descent_first is true or false')


def _from_section(cls, section, extra=None):
    names = {f.name for f in fields(cls)}
    values = dict(section)
    values.update(extra or {})
    unknown = set(values)-names
    if unknown:
        raise ValueError(f'unknown {cls.__name__} parameters: {sorted(unknown)}')
    return cls(**values)


def planner_pilot_configs(declaration, contract=None):
    """dict(corridor=CorridorAimConfig, vertical=VerticalGuardConfig, motor=response model or None) from a
    free-space declaration already loaded (and hash-checked) by the runner. This module reads no files."""
    if (declaration or {}).get('schema') != FREE_SPACE_SCHEMA:
        raise ValueError(f'The free-space declaration has schema {(declaration or {}).get("schema")}; the pilot '
                         f'implements {FREE_SPACE_SCHEMA}')
    if (declaration or {}).get('version') != FREE_SPACE_VERSION:
        raise ValueError(f'The free-space declaration is version {(declaration or {}).get("version")}; the pilot '
                         f'implements version {FREE_SPACE_VERSION}')
    plan = declaration['plan']
    corridor = _from_section(CorridorAimConfig, declaration['pilot'],
                             dict(max_az_deg=float(plan['max_az_deg']), max_el_deg=float(plan['max_el_deg'])))
    vertical = _from_section(VerticalGuardConfig, declaration['vertical'])
    motor = None
    if contract is not None:
        motor = (declaration.get('response_model_for_contract') or {}).get(contract)
        if motor is None:
            raise ValueError(f'The free-space declaration names no response model for the {contract} motor contract')
    return dict(corridor=corridor, vertical=vertical, motor=motor)


def _record(sample, now):
    """The fields the pilot uses from one accepted sample (NaN for None)."""
    r = {k: _value(sample, k) for k in ('time', 'seq', 'cls', 'az', 'el', 'feasible', 'l_az', 'l_el', 'l_ok', 'r_az',
                                          'r_el', 'r_ok', 'v_el', 'v_ok', 'v_cap', 'ring_az', 'h_floor', 'h_ceil',
                                          'rise', 'rise_x', 'speed')}
    r.update(kind=str(sample.get('kind') or ''), received=float(now))
    return r


def _identity(sample):
    seq = _value(sample, 'seq')
    return (None if not np.isfinite(seq) else seq, float(sample['time']))


class _SampleGate:
    """Each sample once (by seq and capture time); the newest received sample and whether it is fresh and valid."""

    def __init__(self, max_age_s):
        self.max_age_s = float(max_age_s)
        self.latest = None
        self.latest_accepted = False
        self.last_id = None
        self.last_time = None

    def take(self, sample, now):
        """'new' (fresh and valid), 'stale', 'invalid', or None (nothing new)."""
        if sample is None or not _finite(sample.get('time')):
            return None
        ident, stamp = _identity(sample), float(sample['time'])
        if ident == self.last_id or (self.last_time is not None and stamp < self.last_time):
            return None
        self.last_id, self.last_time = ident, stamp
        self.latest, self.latest_accepted = sample, False
        if not 0 <= now-stamp <= self.max_age_s:
            return 'stale'
        if not sample.get('valid') or sample.get('kind') not in PLAN_VALID_KINDS:
            return 'invalid'
        self.latest_accepted = True
        return 'new'

    def fresh_valid(self, now):
        """The newest received sample is valid and at most max_age_s old at `now`."""
        return bool(self.latest_accepted and 0 <= now-float(self.latest['time']) <= self.max_age_s)


class CorridorAim:
    """Confirmation, latch, flip, slew, release and conflicts of the planner's aim offset (module docstring)."""

    def __init__(self, config=None):
        self.config = config or CorridorAimConfig()
        self.gate = _SampleGate(self.config.max_age_s)
        self.samples = deque(maxlen=int(self.config.window))
        self.pending = False
        self.last_accepted_at = -np.inf
        self.cls = 0                     # latched class (0: none)
        self.latch_until = -np.inf
        self.blocked = False             # blocked ring path confirmed
        self.no_option = 0               # consecutive accepted samples without an option for the latched class
        self.clear_run = 0               # consecutive accepted clear/aperture samples
        self.target_az = self.target_el = 0.
        self.az = self.el = 0.
        self.decay_az = self.decay_el = None     # |offset| a decay started from (None: not decaying)
        self.hold_until = -np.inf        # conflict hold
        self.v_cap = None
        self.episode = 0
        self.was_engaged = False
        self.flip = False
        self.conflict_kind = ''
        self.counts = dict(samples=0, stale=0, invalid=0, left_episodes=0, right_episodes=0, vertical_episodes=0,
                           blocked_episodes=0, flips=0, latch_blocks=0, ring_conflicts=0, flag_conflicts=0,
                           releases_clear=0, releases_stale=0, resets=0)
        self.engaged_seconds = 0.
        self.max_abs_az = self.max_el = 0.

    # -- samples -----------------------------------------------------------------------------------------------
    @property
    def latest(self):
        return self.gate.latest

    def fresh(self, now):
        return self.gate.fresh_valid(now)

    def ingest(self, sample, now):
        """Take one planner sample (each seq once; stale or invalid samples are counted and dropped). Returns whether
        it was accepted."""
        state = self.gate.take(sample, now)
        if state is None:
            return False
        if state != 'new':
            self.counts[state] += 1
            return False
        record = _record(sample, now)
        self.samples.append(record)
        self.counts['samples'] += 1
        self.last_accepted_at = now
        self.pending = True
        self.clear_run = self.clear_run+1 if record['kind'] in CLEAR_KINDS else 0
        if self.cls:
            option = self.option(record, self.cls)
            if option is None:
                self.no_option += 1
            else:
                self.no_option = 0
                self.target_az, self.target_el = option
        return True

    def option(self, record, cls):
        """(az, el) of a class's option in a sample, bounded, or None when that class has no option."""
        c = self.config
        if cls == VERTICAL:
            az, el = 0., record['v_el']
            if not np.isfinite(el):
                return None
        else:
            az, el = (record['l_az'], record['l_el']) if cls == LEFT else (record['r_az'], record['r_el'])
            if not np.isfinite(az):
                return None
            el = el if np.isfinite(el) else 0.
        return float(np.clip(az, -c.max_az_deg, c.max_az_deg)), float(np.clip(el, 0., c.max_el_deg))

    @property
    def engaged(self):
        return bool(self.cls or self.blocked or abs(self.az) > 1e-9 or abs(self.el) > 1e-9)

    # -- conflicts ---------------------------------------------------------------------------------------------
    def conflict(self, now, kind):
        """The planner evidence and the ring cue disagree: drop the evidence and hold the ring cue's own aim."""
        self.samples.clear()
        self.pending = False
        self.cls, self.blocked, self.no_option, self.clear_run = 0, False, 0, 0
        self.latch_until = -np.inf
        self.target_az = self.target_el = self.az = self.el = 0.
        self.decay_az = self.decay_el = None
        self.v_cap = None
        self.was_engaged = False
        self.hold_until = now+self.config.conflict_hold_s
        self.conflict_kind = kind
        self.counts[f'{kind}_conflicts'] += 1

    def reconcile_ring(self, ring_az, now):
        """Forget samples of another ring bearing; a conflict when the newest sample (the evidence the offset rests
        on) describes another ring while engaged. Returns whether it was a conflict."""
        c = self.config
        rings = [s for s in self.samples if np.isfinite(s['ring_az'])]
        newest_other = bool(rings) and abs(wrap_deg(rings[-1]['ring_az']-ring_az)) > c.conflict_deg
        kept = [s for s in self.samples
                if not np.isfinite(s['ring_az']) or abs(wrap_deg(s['ring_az']-ring_az)) <= c.conflict_deg]
        if len(kept) != len(self.samples):
            self.samples = deque(kept, maxlen=int(c.window))
        if newest_other and self.engaged:
            self.conflict(now, 'ring')
            return True
        return False

    def flag_conflict(self, flag_deg, now):
        """A conflict when the ring cue's flag clearance points to the other side of the ring than the offset."""
        az = self.target_az if self.target_az != 0. else self.az
        if abs(flag_deg) >= self.config.flag_conflict_deg and abs(az) > 1e-9 and az*flag_deg < 0:
            self.conflict(now, 'flag')
            return True
        return False

    # -- per tick ----------------------------------------------------------------------------------------------
    def reset(self):
        """A state that owns the request (search, launch, wait, support climb): no offset, no evidence."""
        if self.engaged:
            self.counts['resets'] += 1
        self.samples.clear()
        self.pending = False
        self.cls, self.blocked, self.no_option, self.clear_run = 0, False, 0, 0
        self.latch_until = -np.inf
        self.target_az = self.target_el = self.az = self.el = 0.
        self.decay_az = self.decay_el = None
        self.v_cap = None
        self.was_engaged = False

    def _release(self, reason):
        self.cls, self.no_option = 0, 0
        self.latch_until = -np.inf
        self.target_az = self.target_el = 0.
        self.counts[f'releases_{reason}'] += 1

    def _evaluate(self, now):
        """Confirmation, latch, flip and release on the newest accepted sample."""
        c = self.config
        newest = self.samples[-1]
        recent = [s for s in self.samples if newest['time']-s['time'] <= c.confirm_window_s]
        held = now < self.hold_until
        blocked = not held and sum(s['kind'] == 'blocked' for s in recent) >= c.confirm
        if blocked and not self.blocked:
            self.counts['blocked_episodes'] += 1
        self.blocked = blocked
        candidate = None
        if not held:
            votes = [s for s in recent if s['kind'] == 'shift' and s['cls'] in (LEFT, RIGHT, VERTICAL)]
            for cls in dict.fromkeys(int(s['cls']) for s in reversed(votes)):   # newest vote first
                if sum(int(s['cls']) == cls for s in votes) >= c.confirm:
                    candidate = cls
                    break
        if candidate is not None:
            if not self.cls or (candidate != self.cls and self.no_option >= c.flip_after):
                if self.cls:
                    self.counts['flips'] += 1
                    self.flip = True
                self.cls, self.no_option = candidate, 0
                self.counts[{LEFT: 'left', RIGHT: 'right', VERTICAL: 'vertical'}[candidate]+'_episodes'] += 1
                self.latch_until = now+c.side_latch_s
                option = next((o for o in (self.option(s, candidate) for s in reversed(self.samples))
                               if o is not None), None)
                if option is not None:
                    self.target_az, self.target_el = option
            elif candidate == self.cls:
                self.latch_until = max(self.latch_until, now+c.side_latch_s)
            else:
                self.counts['latch_blocks'] += 1
        if self.cls and self.clear_run >= c.release_after and now >= self.latch_until:
            self._release('clear')

    def step(self, now, dt, ring_az_cue=None, flag_deg=None, turn_first=False, suspended=False):
        """Advance by one control tick; returns this tick's `CorridorStep`. ``ring_az_cue``/``flag_deg`` (the ring
        cue's ring azimuth and flag-clearance offset, deg) check the two conflicts first when given."""
        c = self.config
        self.flip = False
        self.conflict_kind = ''
        if suspended:
            self.reset()
            return self.output(now)
        if ring_az_cue is not None:
            self.reconcile_ring(ring_az_cue, now)
        if flag_deg is not None and not self.conflict_kind:
            self.flag_conflict(flag_deg, now)
        if self.pending and self.samples:
            self._evaluate(now)
        self.pending = False
        if self.engaged and now-self.last_accepted_at >= c.stale_release_s:
            # no evidence: release everything, the offset decays
            if self.cls:
                self._release('stale')
            self.blocked = False
            self.samples.clear()
        if not self.cls:
            self.target_az = self.target_el = 0.
        target_az = 0. if turn_first else self.target_az
        self.az, self.decay_az = self._move(self.az, target_az, self.decay_az, c.slew_az_deg_s, dt,
                                            release=not self.cls or turn_first)
        self.el, self.decay_el = self._move(self.el, self.target_el, self.decay_el, c.slew_el_deg_s, dt,
                                            release=not self.cls)
        self.az = float(np.clip(self.az, -c.max_az_deg, c.max_az_deg))
        self.el = float(np.clip(self.el, 0., c.max_el_deg))
        self.v_cap = self._cap(now)
        if self.engaged and not self.was_engaged:
            self.episode += 1              # a new episode: an offset, a latched class or a blocked path from none
        self.was_engaged = self.engaged
        if self.engaged:
            self.engaged_seconds += dt
        self.max_abs_az = max(self.max_abs_az, abs(self.az))
        self.max_el = max(self.max_el, self.el)
        return self.output(now)

    def _move(self, value, target, decay_from, slew, dt, release):
        """Slew toward a latched target; a released (or suspended) offset decays linearly over decay_s."""
        if release and target == 0.:
            if abs(value) <= 1e-12:
                return 0., None
            decay_from = abs(value) if decay_from is None else decay_from
            room = decay_from/self.config.decay_s*dt
            return value-float(np.clip(value, -room, room)), decay_from
        room = slew*dt
        return value+float(np.clip(target-value, -room, room)), None

    def _cap(self, now):
        """The speed cap (m/s, at least v_cap_floor_mps) on a confirmed blocked path or a confirmed class whose newest
        sample is an urgent shift; None otherwise or without a fresh newest sample."""
        c = self.config
        if not self.samples or not (self.blocked or self.cls):
            return None
        newest = self.samples[-1]
        if not 0 <= now-newest['time'] <= c.max_age_s or not np.isfinite(newest['v_cap']):
            return None
        urgent = newest['kind'] == 'shift' and newest['feasible'] == 0.
        if self.blocked or (self.cls and urgent):
            return max(float(newest['v_cap']), c.v_cap_floor_mps)
        return None

    def confirmed_class(self):
        """plan_cls_confirmed: BLOCKED (9) on a confirmed blocked ring path, else the latched class (+1/-1/+2) or 0."""
        return BLOCKED if self.blocked else int(self.cls)

    def output(self, now):
        return CorridorStep(az=self.az, el=self.el, target_az=0. if not self.cls else self.target_az,
                            target_el=0. if not self.cls else self.target_el, cls=self.confirmed_class(),
                            v_cap=self.v_cap, fresh=self.fresh(now), episode=self.episode if self.engaged else 0,
                            flip=self.flip, conflict=self.conflict_kind)

    def metadata(self):
        return dict(parameters=asdict(self.config), counts=dict(self.counts), episodes=self.episode,
                    engaged_seconds=round(self.engaged_seconds, 3), max_abs_az_deg=round(self.max_abs_az, 3),
                    max_el_deg=round(self.max_el, 3))


@dataclass(frozen=True)
class CorridorStep:
    az: float                 # offset the corridor aim would apply this tick (slewed; deg, + left)
    el: float                 # (deg, + up)
    target_az: float          # the latched class's option (intended; 0 when nothing is latched)
    target_el: float
    cls: int                  # plan_cls_confirmed (-1/+1/+2/0, 9 = blocked)
    v_cap: float | None       # speed cap along the applied aim (m/s) or None
    fresh: bool               # the newest sample is valid and fresh
    episode: int              # running episode number while engaged, else 0
    flip: bool                # the latch flipped this tick
    conflict: str             # 'ring', 'flag' or ''


@dataclass(frozen=True)
class GuardStep:
    z: float                  # vertical request after the rules (the input when inactive)
    active: bool              # a fresh valid sample drives the guard
    lo: float                 # floor bound (-inf: none)
    hi: float                 # ceiling bound (+inf: none)
    climb: float              # applied terrain climb (0: none)
    up_rate: float | None     # vertical command ramp up (m/s^2) the applied rules ask for
    down_rate: float | None   # vertical command ramp down (m/s^2) the applied rules ask for
    floor_bound: bool
    ceil_bound: bool
    squeeze: bool
    descent_first: bool       # a terrain climb was withheld while descending


class VerticalGuard:
    """The graded vertical rules 1-4 from planner samples (module docstring). Samples come in through `ingest`;
    `apply` runs every control tick (it also keeps the measured motion used for the flight-path angle at capture)."""

    def __init__(self, config=None, pilot=None):
        self.config = config or VerticalGuardConfig()
        self.pilot = pilot or CorridorAimConfig()
        self.gate = _SampleGate(self.pilot.max_age_s)
        self.samples = deque(maxlen=int(self.pilot.window))
        self.motion = deque(maxlen=128)          # (tick time, vz, horizontal speed)
        self.pending = False
        self.climb = 0.
        self.hold_until = -np.inf
        self.base = None
        self.flags = dict(floor=False, ceil=False, squeeze=False, descent=False)
        self.rise_confirmed = False       # rising ground confirmed in the newest fresh window (this tick)
        self.counts = dict(samples=0, floor_bound_activations=0, descent_first_holds=0, terrain_climbs=0,
                           ceiling_bound_activations=0, squeezes=0, topped=0)
        self.active_seconds = 0.

    def fresh(self, now):
        return self.gate.fresh_valid(now)

    def _tan_gamma_at(self, stamp):
        """tan of the non-negative flight-path angle measured at (or just before) a capture time."""
        rows = [m for m in self.motion if m[0] <= stamp]
        _, vz, vh = rows[-1] if rows else (self.motion[0] if self.motion else (0., 0., 0.))
        return max(vz, 0.)/vh if vh > .5 else 0.

    def ingest(self, sample, now):
        if self.gate.take(sample, now) != 'new':
            return False
        record = _record(sample, now)
        record['tan_gamma'] = self._tan_gamma_at(record['time'])
        self.samples.append(record)
        self.counts['samples'] += 1
        self.pending = True
        return True

    def _count(self, key, on, counter):
        if on and not self.flags[key]:
            self.counts[counter] += 1
        self.flags[key] = bool(on)

    def apply(self, z, now, dt, height, velocity):
        """Rules 1-4 on the vertical request `z` (m/s); `height` is the measured height (only differences are used)
        and `velocity` the measured world velocity. Returns a `GuardStep` (z unchanged when inactive)."""
        c, p = self.config, self.pilot
        velocity = np.asarray(velocity, float)
        vz, vh = float(velocity[2]), float(np.hypot(velocity[0], velocity[1]))
        self.motion.append((float(now), vz, vh))
        descending = vz < -c.level_band_mps
        topped = self.base is not None and height-self.base >= c.climb_max_rise_m
        confirmed, rising = False, []
        if self.samples and 0 <= now-self.samples[-1]['time'] <= p.max_age_s:
            newest = self.samples[-1]
            recent = [s for s in self.samples if newest['time']-s['time'] <= p.confirm_window_s]
            rising = [s for s in recent if np.isfinite(s['rise']) and np.isfinite(s['rise_x'])]
            confirmed = len(rising) >= p.confirm
        self.rise_confirmed = confirmed
        if self.pending:
            # a new sample: a confirmed rise refreshes the climb while level (rule 2: never while descending)
            self.pending = False
            if confirmed and not descending and not topped:
                r = rising[-1]
                speed = max(vh, .1)
                reach = max(r['rise_x']/speed-c.rise_lag_s, c.rise_min_t_s)
                request = float(min(c.climb_max_mps, vh*r['tan_gamma']+max(r['rise'], 0.)/reach))
                if request > 0:
                    if self.base is None:
                        self.base = float(height)
                        self.counts['terrain_climbs'] += 1
                    self.climb = request
                    self.hold_until = now+c.climb_hold_s
        if topped and self.climb > 0 and now <= self.hold_until:
            self.counts['topped'] += 1
            self.hold_until = -np.inf
        if now > self.hold_until:
            self.climb = max(0., self.climb-c.climb_release_mps2*dt)
            if self.climb == 0.:
                self.base = None
        withheld = descending and (confirmed or self.climb > 0)
        self._count('descent', withheld, 'descent_first_holds')
        active = self.fresh(now)
        nan = float('nan')
        if not active:
            for key in ('floor', 'ceil', 'squeeze'):
                self.flags[key] = False
            return GuardStep(z=z, active=False, lo=nan, hi=nan, climb=0., up_rate=None, down_rate=None,
                             floor_bound=False, ceil_bound=False, squeeze=False, descent_first=withheld)
        self.active_seconds += dt
        s = self.samples[-1]
        age = now-s['time']
        lo, hi = -np.inf, np.inf
        if np.isfinite(s['h_floor']):
            lo = -(s['h_floor']+vz*age-c.floor_clear_m)/c.tau_s
            if lo > 0:
                lo = min(lo, c.gentle_up_mps)
        if np.isfinite(s['h_ceil']):
            hi = max((s['h_ceil']-vz*age-c.ceil_clear_m)/c.tau_s, -c.gentle_down_mps)
        out = float(z)
        floor_bound = lo > out
        if floor_bound:
            out = float(lo)
        climb = self.climb if (self.climb > 0 and not descending) else 0.
        if climb > 0 and climb > out:
            out = climb
        ceil_bound = hi < out
        if ceil_bound:
            out = float(hi)
        squeeze = hi < lo
        if squeeze:
            out = float((lo+hi)/2)
        self._count('floor', floor_bound, 'floor_bound_activations')
        self._count('ceil', ceil_bound, 'ceiling_bound_activations')
        self._count('squeeze', squeeze, 'squeezes')
        up = c.climb_accel_mps2 if climb > 0 else c.floor_accel_mps2 if floor_bound else None
        down = c.ceil_accel_mps2 if ceil_bound or (squeeze and out < z) else None
        return GuardStep(z=out, active=True, lo=float(lo), hi=float(hi), climb=float(climb), up_rate=up,
                         down_rate=down, floor_bound=bool(floor_bound), ceil_bound=bool(ceil_bound),
                         squeeze=bool(squeeze), descent_first=withheld)

    def metadata(self):
        return dict(parameters=asdict(self.config), counts=dict(self.counts),
                    active_seconds=round(self.active_seconds, 3))
