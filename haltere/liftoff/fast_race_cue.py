"""Velocity-level visible-checkpoint guidance for faster, smoother races.

A separate, explicitly declared profile beside `RaceCueAssistance`, whose
behaviour is unchanged. It still reads only the game's visible next-checkpoint
ring through the camera, the drone's own telemetry and elapsed time. It loads
no route, course file, checkpoint list or per-course parameter.

The ring is a fixed-size HUD marker, so it provides a bearing and no range.
Rather than chase a short goal point, this profile requests a world velocity
along the filtered bearing. Speed falls continuously with the turn still
required, commands are acceleration-limited and change direction as a
coordinated turn (an arc, not a chord through low speeds), a ring clamped
at a side edge is followed just beyond that edge at a moderate speed, and a
brief cue dropout coasts on the previous request instead of stopping. Clipped
markers set bounded climb or descent, and a descent that the vehicle cannot
achieve is treated as support by terrain and answered with a short climb.
These are generic heuristics, not a completed-lap estimate.

Optionally, `update(..., clearance=...)` accepts a causal forward-clearance
sample (e.g. fly-style looming time-to-contact from `vision.looming2`). By
default (`TtcClearanceConfig`) a sustained short time-to-contact lowers the
speed along the looming ray in proportion until the measured TTC recovers;
expansion that lies below the flight path (terrain) requests a bounded climb
and brakes less. `ClearanceConfig` selects the earlier stopping-distance cap.
Without that input the behaviour is unchanged. Missing evidence is not free
space, but it is not an obstacle either: recent evidence is dead-reckoned for
a short memory and nothing else is inferred.

Optionally (``gap_aim=GapAimConfig``, off by default), `update(..., gap=...)`
accepts causal gap-cue samples (relative-depth free interval beside the ring,
`haltere.liftoff.gap_stack`); `haltere.liftoff.gap_aim` confirms them and the
confirmed shift rotates the ring ray about world z in `_ingest`. ``gap_apply``
and ``lag_turn_apply`` False compute and log everything without applying it
(the obstacle stack's matched shadow control).

Optionally (``turn_first=TurnFirstConfig``, ``ceiling_guard=CeilingGuardConfig``,
off by default; the obstacle stack's wall-pilot declaration), two wall rules:
at a wall with the checkpoint far off the heading the pilot turns before it
translates (no request toward the wall, a creep speed until the bearing is
inside a cone, bounded in time), and the TTC governor's terrain climb is kept
out of ceilings (unexplained alarms during a climb are walls, weak climbs are
bounded, overhead evidence cuts the climb and bounds the vertical request).
``wall_apply`` False computes and logs them without applying them.
"""
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np
import torch

from ..brain.motor_baseline import measured_inverse_rate
from ..vision.camera import Camera, quat_wxyz_to_mat
from .gap_aim import GapAim, GapAimConfig, direction_offset, rotate_z, wrap_deg

# Measured original-drone yaw curve (runs/measured-dynamics-low-speed-20260923):
# coefficient deg/s, super rate applied after expo, expo. A loaded dynamics
# profile replaces this declared default.
DEFAULT_YAW_CURVE = (180.17256995580442, .73, .3)


@dataclass(frozen=True)
class FastCueConfig:
    min_speed_fraction: float = .3
    command_acceleration: float = 10.
    vertical_command_acceleration: float = 5.
    vertical_up: float = 3.5
    vertical_down: float = 3.
    edge_speed: float = 1.2
    below_weak_deg: float = 2.
    below_full_deg: float = 10.
    below_slope_margin_deg: float = 5.
    # A ring that stays clipped below while the drone follows that slope must lie
    # steeper still: the margin grows with the clip's duration up to a bound, so a
    # long downhill approach does not reach a lower ring from above (and clip the
    # top of its arch).
    below_slope_growth_deg_s: float = 4.
    below_slope_margin_max_deg: float = 20.
    # The clip's age restarts after a gap in bottom-clip frames or on a new ring.
    below_gap_s: float = .25
    below_speed_fraction: float = .5
    edge_sweep_after_s: float = 1.5
    edge_sweep_rate: float = 1.
    command_time_constant: float = .25
    surface_sink: float = 1.
    surface_sink_per_m: float = .5
    surface_release_m: float = .5
    # Coordinated turn: the horizontal request changes as a heading rotation
    # (centripetal share at most turn_acceleration, i.e. turn_acceleration/max(|v|, 1)
    # rad/s) plus a speed change that uses the rest of command_acceleration, so a
    # new bearing is flown as an arc instead of a chord through low speeds.
    # turn_acceleration < command_acceleration keeps sqrt(10^2-8^2) = 6 m/s^2 for
    # speed changes while turning: with no such room a turn that never converges
    # (a close checkpoint off to the side) cannot slow down and circles it.
    # Below turn_min_speed (request or goal) the heading is undefined and the
    # request slews along the straight line as before.
    turn_acceleration: float = 8.
    turn_min_speed: float = .5
    # A ring clamped at the left/right edge lies beyond the field of view: request
    # side_speed_fraction of the nominal speed toward side_margin_deg beyond the
    # clamped edge ray's bearing, level: the clamped marker's height on the edge
    # is not reliable vertical evidence in Liftoff.
    side_speed_fraction: float = .65
    side_margin_deg: float = 10.
    coast_s: float = .6
    coast_distance_m: float = 4.
    search_yaw_rate: float = 1.2
    # A lost checkpoint is searched for while slowing gently and rising for a moment:
    # a hard stop pitches the camera up (the ring then reappears clamped to the bottom
    # edge) and, close to the ground, the brake-and-turn that follows can touch it.
    search_deceleration: float = 3.
    search_climb: float = .5
    search_climb_s: float = 1.5
    yaw_gain: float = 3.
    yaw_damping: float = .25
    max_yaw_rate: float = 3.
    yaw_slew: float = 8.
    direction_blend: float = .5
    new_target_deg: float = 30.
    feedforward_time_constant: float = .05
    support_after_s: float = .4
    support_climb_s: float = .6
    # Contact on a slope: sliding down a hillside still sinks at the hill's slope, so
    # a requested descent short by this much while the issued throttle sits this far
    # (brain units) below hover for support_slope_after_s also means support.
    support_slope_shortfall: float = .4
    support_thrust_margin: float = .2
    support_slope_after_s: float = .6
    launch_height: float = .6
    # Descent path angle: while a requested descent (sink > descent_sink) is
    # not achieved, the filtered shortfall (time constant descent_time_constant)
    # beyond descent_free scales the horizontal request down, reaching
    # descent_min_scale at descent_free + descent_span, so the path keeps the
    # requested slope instead of passing above a lower checkpoint.
    descent_sink: float = .3
    descent_time_constant: float = .5
    descent_free: float = .2
    descent_span: float = .5
    descent_min_scale: float = .35

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Use finite positive fast cue parameters')
        if not self.min_speed_fraction < 1 or not self.direction_blend <= 1 or not self.descent_min_scale <= 1:
            raise ValueError('Fractions must stay below one')
        if not self.below_slope_margin_max_deg >= self.below_slope_margin_deg:
            raise ValueError('The bottom-edge slope margin bound must not be below its start')
        if not self.below_full_deg > self.below_weak_deg:
            raise ValueError('Bottom-edge evidence needs an increasing depression range')
        if not self.turn_acceleration <= self.command_acceleration:
            raise ValueError('The turn share must fit within command_acceleration')
        if not self.side_speed_fraction <= 1 or not self.side_margin_deg < 90:
            raise ValueError('Use a side speed fraction <= 1 and a side margin below 90 degrees')


# The state in which lag-aware turns act: the ring is in view (not clamped to an edge).
LAG_TURN_STATES = ('cue',)
# The lag-turn declaration version whose rule this code implements (see LagTurnConfig); runners refuse others.
LAG_TURN_VERSION = 2


@dataclass(frozen=True)
class LagTurnConfig:
    """Faster convergence onto a new checkpoint bearing for a motor that lags its request.

    Off unless a runner passes it (declared per motor contract). The motors follow a
    velocity request late (brain-08 by 0.3-0.4 s, the fast PD by 0.1-0.15 s), so after
    a checkpoint switch the flown path swings outside the line to the new ring.

    Trigger (a new checkpoint): a fresh in-view (not edge-clamped) cue whose ring-centre
    azimuth (marker u, v) differs by at least trigger_deg from the filtered ring-centre
    bearing or from the ring centre of any fresh in-view cue captured within the last
    trigger_span_s (the HUD marker can take two frames to move to the new ring). The flown
    aim beside the ring (aim_u, the flag clearance) and an applied gap shift are not part
    of the trigger: a flag clearance that appears, disappears or flickers is no switch.
    Clamped markers never trigger: their azimuth is not reliable. For window_s after the
    triggering capture, while the ring is in view:
    - the horizontal goal aims beyond the bearing by course_lead times the angle from
      the flown course (measured horizontal velocity) to the bearing, clipped to
      course_lead_max_deg; no lead below min_course_speed or beyond max_lead_angle_deg.
      The bearing is the ring cue's filtered aim bearing; an applied gap shift is removed
      before the lead is computed and added after it (never amplified by the lead);
    - the request heading rotates toward the goal with heading_time_constant instead
      of command_time_constant (turn_acceleration still bounds the rotation rate, and
      the speed change keeps command_time_constant).
    Both fade out linearly over the last fade_s of the window. The speed schedule
    still uses the angle to the bearing itself, so no extra braking is requested.
    Lag-turn declaration version 2 (read by the runner from configs/obstacles); version 1
    triggered on the flown aim ray and led the gap-shifted bearing.
    """
    course_lead: float = .6
    course_lead_max_deg: float = 15.
    heading_time_constant: float = .1
    window_s: float = 1.
    fade_s: float = .25
    trigger_deg: float = 10.
    trigger_span_s: float = .25
    min_course_speed: float = 1.
    max_lead_angle_deg: float = 90.

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Use finite positive lag-turn parameters')
        if not self.course_lead_max_deg < 90 or not self.trigger_deg < 90 or not self.max_lead_angle_deg <= 180:
            raise ValueError('Use a lead clip and trigger below 90 degrees and a lead range within 180 degrees')
        if not self.fade_s <= self.window_s:
            raise ValueError('The lag-turn fade must fit inside its window')


def lag_turn_for_contract(declaration, contract):
    """The `LagTurnConfig` that a declaration already parsed by the runner (a dict with
    'contracts': {motor contract: parameters or None}) assigns to a motor contract, or
    None when that contract has no lag-turn entry. This module reads no files."""
    contracts = (declaration or {}).get('contracts')
    if not isinstance(contracts, dict):
        raise ValueError('A lag-turn declaration lists its motor contracts')
    entry = contracts.get(contract)
    if entry is None:
        return None
    return LagTurnConfig(**entry)


# The pilot states in which the checkpoint bearing is measured from a fresh marker (in view or clamped to the
# bottom/top edge away from the corners); 'side' (clamped at a side edge or a corner) lies beyond the field of view.
TURN_FIRST_BEARING_STATES = ('cue', 'below', 'below_weak', 'above')
# States that end a turn-first episode: other rules own the request there.
TURN_FIRST_HANDOFF_STATES = ('search', 'launch', 'wait', 'support_climb')
# The wall-pilot declaration version whose rules this code implements (TurnFirstConfig, CeilingGuardConfig);
# version 2 adds the ceiling guard's overhead_min_rise (version 1 is kept for provenance and refused).
WALL_PILOT_VERSION = 2


@dataclass(frozen=True)
class TurnFirstConfig:
    """Turn before translating at a wall (a hairpin): what a pilot does when the next gate lies behind a wall
    beside it. Off unless a runner passes it (the wall-pilot declaration in configs/obstacles; obstacle stack only).

    Engage when both hold:
    - near a wall: the clearance governor holds a stand-off (a wall that capped the request at its
      standoff_speed or less is remembered), or its wall brake capped the request within the last
      brake_recent_s while the horizontal speed is at most slow_speed;
    - the checkpoint is far off the heading: its marker is clamped at a side edge or a corner (pilot state
      'side'), or its bearing (the filtered aim bearing of a marker in view or clamped at the bottom/top edge)
      lies engage_deg or more from the heading.
    While engaged, the horizontal request loses any component toward the wall (along the looming ray that
    capped it, taken at engagement) and is bounded to creep_speed, and the request's speed toward the wall is
    removed at the clearance brake_slew; the vertical request and the yaw rule are unchanged, so the assisted
    yaw keeps turning toward the checkpoint (the clamped edge ray turns with the camera). The episode ends when
    the bearing of a marker in view (or bottom/top clamped) comes within release_deg of the heading ('aligned';
    the ordinary speed schedule and acceleration limits then resume), when a state that owns the request takes
    over (search, launch, support climb: 'handoff'), or after max_s ('timeout'), after which it cannot engage
    again for rearm_s: the drone never hovers at a wall indefinitely.
    """
    slow_speed: float = 1.5
    brake_recent_s: float = 1.
    engage_deg: float = 50.
    release_deg: float = 30.
    creep_speed: float = .8
    max_s: float = 2.
    rearm_s: float = 2.

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Use finite positive turn-first parameters')
        if not self.release_deg < self.engage_deg < 90:
            raise ValueError('Use release_deg < engage_deg < 90 degrees')


def stopping_speed(distance, deceleration, latency, margin):
    """Largest speed v with v*latency + v^2/(2*deceleration) + margin <= distance."""
    room = max(0., float(distance)-margin)
    return float(-deceleration*latency+np.sqrt((deceleration*latency)**2+2*deceleration*room))


@dataclass(frozen=True)
class ClearanceConfig:
    """Pilot response to forward clearance samples (looming time-to-contact).

    Each sample is dead-reckoned along its ray with odometry, which removes the
    perception delay; `latency` covers only the velocity loop (measured fast-PD
    onset 0.13 s, equivalent delay 0.14 s). A wall is braked for when at least
    `confirm` of the last `window` samples, taken within `confirm_window_s`,
    place it ahead; the distance used is their median (the nearer of two). A
    single sample below `urgent_ttc_s` acts at once. Samples are forgotten
    `memory_s` after they arrive; once a wall has been seen in `window` samples
    they are kept up to `memory_max_s` while the drone is too slow for looming
    (`slow_speed`), so a drone halted at a wall stays halted. The cap holds its
    lowest value for `hold_s`, then rises at `release` m/s^2. Samples whose
    expansion lies mostly below the path (`below_fraction` >= `terrain_fraction`)
    add a climb floor instead, unless their TTC is below `terrain_brake_s`.
    Missing evidence changes nothing unless `blind_after_s` is finite (off by
    default: 65% of frames at speed on the clean Straw Bale run had none).
    """
    deceleration: float = 10.
    latency: float = .15
    margin: float = .5
    brake_slew: float = 15.
    hold_s: float = .15
    release: float = 10.
    max_age_s: float = .25
    memory_s: float = .3
    slow_speed: float = 1.5
    memory_max_s: float = 2.
    standoff_speed: float = 1.
    confirm: int = 2
    window: int = 3
    confirm_window_s: float = .2
    urgent_ttc_s: float = .25
    terrain_fraction: float = .7
    terrain_on_s: float = 1.
    terrain_full_s: float = .6
    terrain_climb_acceleration: float = 10.
    terrain_brake_s: float = .3
    terrain_release: float = 3.
    blind_after_s: float = float('inf')
    blind_speed: float = 4.

    def __post_init__(self):
        values = asdict(self)
        blind_after = values.pop('blind_after_s')
        if not np.isfinite(list(values.values())).all() or min(values.values()) <= 0 or not blind_after > 0:
            raise ValueError('Use finite positive clearance parameters')
        if int(self.confirm) != self.confirm or int(self.window) != self.window or self.confirm > self.window:
            raise ValueError('confirm and window count samples, confirm <= window')
        if not self.terrain_fraction <= 1 or not self.terrain_on_s > self.terrain_full_s:
            raise ValueError('Terrain fraction must be <= 1 and terrain_on_s > terrain_full_s')


class ClearanceGovernor:
    """Speed cap along a looming ray and a terrain climb floor from clearance samples.

    Pure and causal: samples carry their capture time, the capture position and
    the travel direction (ray); limits are evaluated at the current position.
    """

    def __init__(self, config=None):
        self.config = config or ClearanceConfig()
        self.samples = []
        self.last_time = self.last_evidence = self.first_input = None
        self.cap = self.cap_ray = None
        self.cap_hold_until = self.climb_hold_until = -np.inf
        self.climb = 0.
        self.status = 'none'
        self.blind = self.sustained = False
        self.counts = dict(samples=0, no_evidence=0, brake_engagements=0, climb_engagements=0, blind_engagements=0)

    def ingest(self, time, ttc, distance, below_fraction, position, ray, closing_speed, received=None):
        """Add one fresh sample captured at `time` at `position` and received at `received`
        (default: `time`); `ray` is the unit travel direction, closing_speed the speed along it."""
        self.first_input = time if self.first_input is None else self.first_input
        self.last_time = time
        if ttc is None and distance is None:
            self.counts['no_evidence'] += 1
            return
        if distance is None:
            distance = ttc*max(closing_speed, 0.)
        if ttc is None:
            ttc = distance/max(closing_speed, 1e-3)
        b = .5 if below_fraction is None else float(below_fraction)
        keep = int(self.config.window)-1
        self.samples = (self.samples[-keep:] if keep else [])+[
            (float(time), float(distance), float(ttc), b, np.array(position, float), np.array(ray, float),
             float(time if received is None else received))]
        self.last_evidence = time
        self.counts['samples'] += 1

    def _recent(self, position, velocity, now):
        """Remembered samples at the current position: (time, distance, ttc, is_wall, ray)."""
        c = self.config
        out = []
        position, velocity = np.asarray(position, float), np.asarray(velocity, float)
        for time, distance, ttc, b, where, ray, received in self.samples:
            closing = float(velocity @ ray)
            age = now-received
            # Too slow for looming: a wall seen in `window` samples is still there.
            slow_hold = self.sustained and closing < c.slow_speed
            if age > c.memory_max_s or (age > c.memory_s and not slow_hold):
                continue
            # Classified on the measurement itself; ground under the path is not on the
            # ray, so terrain samples age in time instead of being dead-reckoned along it.
            wall = b < c.terrain_fraction or ttc < c.terrain_brake_s
            d = distance-float((position-where) @ ray)
            out.append((time, d, d/max(closing, .3) if wall else ttc-(now-time), wall, ray))
        return [s for s in out if out[-1][0]-s[0] <= c.confirm_window_s]

    def limits(self, position, velocity, now, dt, vertical_up):
        """Return (cap or None, ray or None, climb floor) for the current tick."""
        c = self.config
        recent = self._recent(position, velocity, now)
        wall = [s for s in recent if s[3]]
        terrain = [s for s in recent if not s[3]]
        cap_now, ray = np.inf, None
        if len(wall) >= c.confirm:
            # median of three rejects one outlier; with two, the nearer one is used
            d = float(np.median([s[1] for s in wall])) if len(wall) >= 3 else min(s[1] for s in wall)
            cap_now, ray = stopping_speed(d, c.deceleration, c.latency, c.margin), wall[-1][4]
            self.sustained = self.sustained or len(wall) >= c.window
        elif wall and wall[-1] is recent[-1] and wall[-1][2] < c.urgent_ttc_s:
            cap_now, ray = stopping_speed(wall[-1][1], c.deceleration, c.latency, c.margin), wall[-1][4]
        blind = (np.isfinite(c.blind_after_s) and self.first_input is not None
                 and now-(self.last_evidence if self.last_evidence is not None else self.first_input) > c.blind_after_s)
        speed = float(np.linalg.norm(velocity))
        if blind and speed > 1e-3 and c.blind_speed < cap_now:
            cap_now, ray = c.blind_speed, np.asarray(velocity, float)/speed
        if self.cap is None or cap_now <= self.cap:
            if np.isfinite(cap_now):
                self.cap, self.cap_ray, self.cap_hold_until = cap_now, ray, now+c.hold_s
        elif self.sustained and self.cap < c.standoff_speed and wall:
            pass  # stand-off: never creep back toward a confirmed, still remembered wall
        elif now >= self.cap_hold_until:
            self.cap = min(cap_now, self.cap+c.release*dt)
            if np.isfinite(cap_now) and ray is not None:
                self.cap_ray = ray
        if self.cap is not None and self.cap > 25.:
            self.cap = self.cap_ray = None
        if self.cap is None:
            self.sustained = False
        # Terrain under the path: the same confirmation, a climb floor instead of a brake.
        floor = 0.
        if len(terrain) >= c.confirm:
            ttc = float(np.median([s[2] for s in terrain]))
            floor = vertical_up*float(np.clip((c.terrain_on_s-ttc)/(c.terrain_on_s-c.terrain_full_s), 0, 1))
        if floor >= self.climb:
            if floor > 0 and self.climb == 0:
                self.counts['climb_engagements'] += 1
            self.climb = floor
            if floor > 0:
                self.climb_hold_until = now+c.hold_s
        elif now >= self.climb_hold_until:
            self.climb = max(floor, self.climb-c.terrain_release*dt)
        self.blind = bool(blind and self.cap == c.blind_speed)
        self.status = ('climb' if self.climb > 0 else 'armed' if self.cap is not None
                       else 'clear' if recent else 'no_evidence')
        return self.cap, self.cap_ray, self.climb


@dataclass(frozen=True)
class TtcClearanceConfig:
    """Graded pilot response to looming time-to-contact (TTC); no metric stopping distance.

    Looming measures TTC from image expansion. Its metric distance (TTC x speed)
    read long in the first live test (4-9 m before a mound), so a stopping-
    distance cap braked 0.3 s before contact although TTC had stayed below 1.5 s
    for 2 s. This policy acts on TTC alone.

    Slow-down: `confirm` samples with TTC < `ttc_on` within `confirm_window_s`
    (or one below `urgent_ttc_s`) engage it; while engaged, each new sample
    with TTC < `ttc_target` lowers the cap on the speed along its ray to
        v * clip((ttc - ttc_min) / (ttc_target - ttc_min), floor_fraction, 1),
    never below `min_speed` (unless a wall sample reads TTC < `stop_ttc_s`),
    where v is the speed along the ray and ttc is aged to the present by
    odometry. The cap settles where the measured TTC has
    recovered to `ttc_target` (> ttc_on: hysteresis). It falls at most
    `brake_rate` m/s^2 and holds its lowest value while wall samples keep TTC
    below `hold_ttc_s`, and for `hold_s` after the last one; then it rises at
    `release` m/s^2 (a slower drone reads a longer TTC from the same wall, so
    recovery alone must not re-accelerate it into the wall).
    Terrain: a sample whose expansion lies below the path (below_fraction >=
    `terrain_fraction`, `terrain_confirm` such samples within
    `confirm_window_s`) requests a climb of
        vertical_up * clip((climb_on_s - ttc) / (climb_on_s - climb_full_s), 0, 1),
    held `climb_hold_s`, then released at `climb_release` m/s^2, and at the
    latest once the drone is `climb_max_m` above where the episode began (it
    ends `climb_hold_s` after the last terrain request): a bound on false
    climbs, e.g. under a ceiling. It lowers the cap only `terrain_brake` as
    much as a wall sample would. ttc is the alarm TTC (`climb_ttc_source`
    'alarm'), or also the lower surface's TTC: 'either' uses the shorter,
    'both' the longer (both must be short). While a climb is active, samples
    without vertical evidence (below_fraction None) count as terrain;
    otherwise None counts as a wall (braking is the safe default).
    Stand-off: a wall sample that leaves the cap at `standoff_speed` or less is
    remembered for `standoff_s`: the cap along that ray does not rise in that
    time, also without evidence (looming needs forward speed), so the drone
    does not creep back toward the wall.
    Missing evidence changes nothing.
    """
    # Defaults: selected by a closed-loop replay of recorded looming streams (17 impacts, 9.3 min of clean 6 m/s
    # flight; declared rule: most impacts avoided, then soft contacts, subject to <= 3 s/min lost and <= 3
    # climbs/min on the clean flights). climb_max_m is a declared safety bound, not tuned. Not flight evidence.
    ttc_on: float = .8
    ttc_target: float = 1.3
    ttc_min: float = .4
    floor_fraction: float = .7
    min_speed: float = 2.
    stop_ttc_s: float = 1.2
    confirm: int = 3
    confirm_window_s: float = .25
    urgent_ttc_s: float = .4
    brake_rate: float = 8.
    hold_s: float = .6
    hold_ttc_s: float = 1.3
    release: float = 3.
    memory_s: float = .3
    max_age_s: float = .25
    brake_slew: float = 15.
    terrain_fraction: float = .7
    climb_on_s: float = 1.2
    climb_full_s: float = .8
    terrain_confirm: int = 1
    climb_ttc_source: str = 'alarm'
    climb_hold_s: float = .5
    climb_release: float = 3.
    climb_max_m: float = 2.5
    terrain_brake: float = 0.
    terrain_climb_acceleration: float = 10.
    standoff_speed: float = 2.
    standoff_s: float = 2.

    def __post_init__(self):
        values = asdict(self)
        terrain_brake, source = values.pop('terrain_brake'), values.pop('climb_ttc_source')
        if not np.isfinite(list(values.values())+[terrain_brake]).all() or min(values.values()) <= 0:
            raise ValueError('Use finite positive TTC clearance parameters')
        if source not in ('alarm', 'either', 'both'):
            raise ValueError('climb_ttc_source is alarm, either or both')
        if not 0 <= terrain_brake <= 1 or not self.floor_fraction <= 1 or not self.terrain_fraction <= 1:
            raise ValueError('terrain_brake, floor_fraction and terrain_fraction are fractions')
        if not self.ttc_min < self.ttc_on <= self.ttc_target <= self.hold_ttc_s or not self.climb_full_s < self.climb_on_s:
            raise ValueError('Use ttc_min < ttc_on <= ttc_target <= hold_ttc_s and climb_full_s < climb_on_s')
        for name in ('confirm', 'terrain_confirm'):
            if int(getattr(self, name)) != getattr(self, name):
                raise ValueError(f'{name} counts samples')


@dataclass(frozen=True)
class CeilingGuardConfig:
    """Keep the TTC governor's terrain climb out of ceilings and overhangs. Off unless a runner passes it
    (the wall-pilot declaration in configs/obstacles; obstacle stack only); TTC policy only.

    Without it, a sample without vertical evidence (below_fraction None) counts as terrain while a climb is
    active, so an alarm from a ceiling ahead of a climbing path raised the climb to vertical_up. With it:
    - Unexplained alarms are walls: during a climb such a sample counts as terrain only when the lower window
      explains it (its lower-surface TTC is known and at most lower_ratio x its alarm TTC); otherwise it is a
      wall sample (it may brake, it never climbs).
    - Weak climbs are bounded: a climb request from such an explained sample (terrain seen by the lower window
      only) is at most weak_climb, it refreshes the climb hold only at that level (a stronger climb decays after
      its own hold), and it is ignored once the drone is weak_climb_max_m above where the last below-path
      (below_fraction >= terrain_fraction) climb request was accepted.
    - Overhead cut: during a climb (or an overhead hold) while the drone rises faster than overhead_min_rise
      (a ceiling can only cross a rising path; an alarm ahead of a sinking path is no reason to stop arresting
      the sink), a sample whose alarm TTC is below overhead_ttc_s and whose expansion lies above the path
      (below_fraction <= overhead_fraction) or is unexplained (as above) is overhead evidence. overhead_confirm
      such samples within the governor's confirm_window_s cut the climb to 0 at once and start an overhead hold
      of hold_s (renewed by further overhead evidence): no terrain climb is requested, below-path samples brake
      like walls, and the whole vertical request is bounded to vertical_cap, brought down at up to
      vertical_slew m/s^2.
    Below-path climbs (below_fraction >= terrain_fraction) are otherwise unchanged. Declaration version 2
    (version 1 had no overhead_min_rise).
    """
    lower_ratio: float = 1.
    weak_climb: float = 1.
    weak_climb_max_m: float = 1.
    overhead_fraction: float = .3
    overhead_ttc_s: float = 1.2
    overhead_confirm: int = 2
    overhead_min_rise: float = .3
    hold_s: float = 1.
    vertical_cap: float = 0.
    vertical_slew: float = 15.

    def __post_init__(self):
        values = asdict(self)
        cap = values.pop('vertical_cap')
        if not np.isfinite(list(values.values())+[cap]).all() or min(values.values()) <= 0:
            raise ValueError('Use finite positive ceiling-guard parameters (vertical_cap may be <= 0)')
        if int(self.overhead_confirm) != self.overhead_confirm:
            raise ValueError('overhead_confirm counts samples')
        if not self.overhead_fraction < .5 or not cap <= self.weak_climb:
            raise ValueError('Overhead evidence lies above the path (overhead_fraction < 0.5); vertical_cap <= weak_climb')


def wall_pilot_configs(declaration):
    """dict(turn_first=TurnFirstConfig, ceiling_guard=CeilingGuardConfig) from a wall-pilot declaration already
    parsed (and hash-checked) by the runner; refuses another rule version. This module reads no files."""
    if (declaration or {}).get('version') != WALL_PILOT_VERSION:
        raise ValueError(f'The wall-pilot declaration is version {(declaration or {}).get("version")}; the fast pilot '
                         f'implements version {WALL_PILOT_VERSION}')
    return dict(turn_first=TurnFirstConfig(**declaration['turn_first']),
                ceiling_guard=CeilingGuardConfig(**declaration['ceiling_guard']))


class TtcClearanceGovernor:
    """Graded speed cap along the looming ray and a terrain climb from TTC samples.

    Same interface as `ClearanceGovernor`. Pure and causal: each sample carries
    its capture time, capture position and ray; it acts once, when it first
    reaches `limits`, with the TTC aged to that moment by odometry.
    `ceiling` (a `CeilingGuardConfig`, off by default) adds the ceiling guard;
    `vertical_cap` is then the bound on the whole vertical request (None: none).
    """

    def __init__(self, config=None, ceiling=None):
        self.config = config or TtcClearanceConfig()
        if ceiling is not None and not isinstance(ceiling, CeilingGuardConfig):
            raise ValueError('Pass a CeilingGuardConfig (or None) for the ceiling guard')
        self.ceiling = ceiling
        self.samples = []
        self.last_time = self.last_evidence = self.first_input = None
        self.cap = self.cap_ray = self.target = None
        self.lowered_at = self.climb_hold_until = self.standoff_until = -np.inf
        self.climb = 0.
        self.climb_base, self.terrain_at = None, -np.inf
        self.status = 'none'
        self.blind = False
        self.counts = dict(samples=0, no_evidence=0, brake_engagements=0, climb_engagements=0, blind_engagements=0,
                           standoff_engagements=0)
        # Ceiling guard state (unused without it)
        self.overhead_until = -np.inf
        self.overhead_times = []        # receipt times of recent overhead samples
        self.strong_height = None       # height where the last below-path climb request was accepted
        self.vertical_cap = None
        if ceiling is not None:
            self.counts.update(overhead_samples=0, overhead_engagements=0, unexplained_walls=0, weak_climb_samples=0,
                               suppressed_climb_samples=0)

    def ingest(self, time, ttc, distance, below_fraction, position, ray, closing_speed, received=None, ttc_lower=None):
        """Add one fresh sample captured at `time` at `position` and received at `received`
        (default: `time`); `ray` is the unit travel direction, closing_speed the speed along it.
        `distance` (TTC x capture speed) only ages the TTC by odometry; `ttc_lower` optionally
        gives the TTC of the surface below the path for the climb."""
        self.first_input = time if self.first_input is None else self.first_input
        self.last_time = time
        if ttc is None and distance is None:
            self.counts['no_evidence'] += 1
            return
        closing = max(float(closing_speed), 1e-3)
        ttc = float(distance)/closing if ttc is None else float(ttc)
        reach = float(distance) if distance is not None else ttc*closing
        self.samples.append(dict(time=float(time), ttc=ttc, reach=reach,
                                 below=None if below_fraction is None else float(below_fraction),
                                 ttc_lower=None if ttc_lower is None else float(ttc_lower),
                                 position=np.array(position, float), ray=np.array(ray, float),
                                 received=float(time if received is None else received), new=True))
        self.last_evidence = time
        self.counts['samples'] += 1

    @staticmethod
    def _aged(sample, reach, position, closing):
        """TTC now: the capture-time reach along the ray minus the odometry travelled along it."""
        left = reach-float((np.asarray(position, float)-sample['position']) @ sample['ray'])
        return max(0., left)/max(closing, .3)

    def limits(self, position, velocity, now, dt, vertical_up):
        """Return (cap or None, ray or None, climb request) for the current tick."""
        c = self.config
        g = self.ceiling
        position, velocity = np.asarray(position, float), np.asarray(velocity, float)
        keep = max(c.memory_s, c.confirm_window_s)
        self.samples = [s for s in self.samples if now-s['received'] <= keep]
        climbing = self.climb > 0
        height = float(position[2]) if position.size > 2 else 0.
        rise = float(velocity[2]) if velocity.size > 2 else 0.
        if not climbing and now-self.terrain_at > c.climb_hold_s:
            self.climb_base = None                  # a new climb episode may start from the present height
            self.strong_height = None
        topped = self.climb_base is not None and height-self.climb_base >= c.climb_max_m
        overhead = g is not None and now <= self.overhead_until
        for s in self.samples:
            if not s['new']:
                continue
            s['new'] = False
            if now-s['received'] > c.memory_s:
                continue
            closing = float(velocity @ s['ray'])
            ttc = self._aged(s, s['reach'], position, closing)
            weak = unexplained = False
            if s['below'] is not None:
                terrain = s['below'] >= c.terrain_fraction
            elif g is None:
                terrain = climbing
            else:
                # ceiling guard: without vertical evidence a climb continues only on what the lower window explains
                weak = terrain = (climbing and s['ttc_lower'] is not None
                                  and s['ttc_lower'] <= g.lower_ratio*max(s['ttc'], 1e-3))
                unexplained = not terrain
                self.counts['unexplained_walls'] += int(climbing and unexplained)
            recent = [r for r in self.samples if s['received']-r['received'] <= c.confirm_window_s
                      and r['received'] <= s['received']]
            if (g is not None and (climbing or overhead) and rise > g.overhead_min_rise and ttc < g.overhead_ttc_s
                    and (unexplained or (s['below'] is not None and s['below'] <= g.overhead_fraction))):
                # the alarm lies above a rising path (or is not explained by the surface below it)
                self.counts['overhead_samples'] += 1
                self.overhead_times = [t for t in self.overhead_times
                                       if s['received']-t <= c.confirm_window_s]+[s['received']]
                if len(self.overhead_times) >= g.overhead_confirm:
                    if now > self.overhead_until:
                        self.counts['overhead_engagements'] += 1
                    self.overhead_until, overhead = now+g.hold_s, True
                    self.climb, self.climb_hold_until = 0., -np.inf
            # climb: expansion below the path
            if terrain:
                lower = None if s['ttc_lower'] is None else self._aged(
                    s, s['ttc_lower']*s['reach']/max(s['ttc'], 1e-3), position, closing)
                climb_ttc = (ttc if c.climb_ttc_source == 'alarm' or (lower is None and c.climb_ttc_source == 'either')
                             else np.inf if lower is None else
                             min(ttc, lower) if c.climb_ttc_source == 'either' else max(ttc, lower))
                votes = sum(r['below'] is not None and r['below'] >= c.terrain_fraction for r in recent)
                request = vertical_up*float(np.clip((c.climb_on_s-climb_ttc)/(c.climb_on_s-c.climb_full_s), 0, 1))
                if weak and request > 0:
                    self.counts['weak_climb_samples'] += 1
                    request = min(request, g.weak_climb)
                    if self.strong_height is not None and height-self.strong_height >= g.weak_climb_max_m:
                        request = 0.
                if overhead and request > 0:
                    self.counts['suppressed_climb_samples'] += 1
                    request = 0.
                if request > 0 and (votes >= c.terrain_confirm or climbing):
                    self.terrain_at = now
                    self.climb_base = height if self.climb_base is None else self.climb_base
                if request > 0 and (votes >= c.terrain_confirm or climbing) and not topped:
                    if self.climb == 0:
                        self.counts['climb_engagements'] += 1
                    if not weak or request >= self.climb:
                        self.climb_hold_until = now+c.climb_hold_s
                    self.climb = max(self.climb, request)
                    if g is not None and not weak:
                        self.strong_height = height
            # graded slow-down; during an overhead hold a surface below the path is braked for like a wall
            brake_terrain = terrain and not overhead
            active = self.cap is not None and self.cap < max(closing, 0.)+1.
            if self.target is not None and not brake_terrain and ttc < c.hold_ttc_s:
                self.lowered_at = now               # a wall still in view: hold the cap
            threshold = c.ttc_target if active else c.ttc_on
            votes = sum(r['ttc'] < threshold for r in recent)
            if not (votes >= c.confirm or ttc < c.urgent_ttc_s) or closing <= 0:
                continue
            fraction = float(np.clip((ttc-c.ttc_min)/(c.ttc_target-c.ttc_min), c.floor_fraction, 1.))
            if brake_terrain:
                fraction = 1.-c.terrain_brake*(1.-fraction)
            if fraction >= 1.:
                continue
            target = (closing*fraction if (ttc < c.stop_ttc_s and not brake_terrain)
                      else max(c.min_speed, closing*fraction))
            self.lowered_at = now                   # TTC has not recovered to ttc_target: keep holding
            if self.target is None or target < self.target:
                if self.cap is None or not active:
                    self.counts['brake_engagements'] += 1
                self.target, self.cap_ray = target, s['ray']
                self.cap = closing if self.cap is None else min(self.cap, max(closing, target))
            if not brake_terrain and self.target <= c.standoff_speed:
                if now > self.standoff_until:
                    self.counts['standoff_engagements'] += 1
                self.standoff_until = now+c.standoff_s
        if self.target is not None:
            standoff = now <= self.standoff_until
            if now-self.lowered_at > c.hold_s and not standoff:
                self.target += c.release*dt
            # the cap follows the target down at brake_rate and up with it
            self.cap = self.target if self.cap is None or self.target >= self.cap else max(self.target, self.cap-c.brake_rate*dt)
            if self.target > 25.:
                self.cap = self.cap_ray = self.target = None
        if now > self.climb_hold_until or topped:
            self.climb = max(0., self.climb-c.climb_release*dt)
        if overhead:
            self.climb = 0.
        self.vertical_cap = g.vertical_cap if overhead else None
        fresh = any(now-s['received'] <= c.memory_s for s in self.samples)
        self.status = ('overhead' if overhead else 'climb' if self.climb > 0
                       else 'standoff' if now <= self.standoff_until and self.cap is not None
                       else 'armed' if self.cap is not None else 'clear' if fresh else 'no_evidence')
        return self.cap, self.cap_ray, self.climb


def clearance_governor(config):
    """The governor for a clearance config: TTC-graded (`TtcClearanceConfig`) or stopping-distance."""
    return TtcClearanceGovernor(config) if isinstance(config, TtcClearanceConfig) else ClearanceGovernor(config)


class FastRaceCue:
    """Follow the visible checkpoint bearing with a continuous velocity request."""

    profile = 'fast-v1'

    def __init__(self, sensor, pose_history, speed=6., *, reference_speed=2., config=None,
                 yaw_curve=DEFAULT_YAW_CURVE, calibration=None, velocity_scale=None, clearance_config=None,
                 lag_turn=None, lag_turn_apply=True, gap_aim=None, gap_apply=True, turn_first=None,
                 ceiling_guard=None, wall_apply=True):
        if not sensor:
            raise ValueError('Race cue assistance requires a calibrated camera')
        if not np.isfinite(speed) or not 0 < speed <= 20:
            raise ValueError('Fast cue speed must be finite and in (0, 20] m/s')
        if not np.isfinite(reference_speed) or not 0 < reference_speed <= 20:
            raise ValueError('Invalid motor reference speed')
        self.config = config or FastCueConfig()
        self.yaw_curve = tuple(float(v) for v in yaw_curve)
        if len(self.yaw_curve) != 3 or not np.isfinite(self.yaw_curve).all() or min(self.yaw_curve) <= 0:
            raise ValueError('Use a measured (coefficient, super rate, expo) yaw curve')
        # Throttle priority on the shared left stick needs the brain->processed map.
        self.calibration = None if calibration is None else tuple(
            float(calibration[k]) for k in ('hover_processed', 'throttle_scale', 'hover_stick_sim'))
        self.below_weight = 0.
        self.edge_depression = 0.
        self.below_since = self.below_last = None
        self.vertical_clip_since = None
        self.sweep_side = None
        # A fast-contract brain senses horizontal velocity scaled by a declared
        # factor (below one at race speed); other motor contracts keep >= 1.
        if velocity_scale is not None and (not np.isfinite(velocity_scale) or velocity_scale <= 0):
            raise ValueError('Use a finite positive velocity scale')
        self.velocity_scale = velocity_scale
        self.camera = Camera(320, 180, sensor['focal_320'], sensor['tilt_deg'])
        self.pose_history, self.speed, self.reference_speed = pose_history, float(speed), float(reference_speed)
        self.host = SimpleNamespace(flow_gain=1.)
        self.pilot = SimpleNamespace(carrot=np.zeros(3), target=None, mode=4, n_passes=0, sight_yaw=0.)
        self.last_capture = self.last_seen = self.last_time = None
        self.direction = None
        self.edge = self.below = self.above = False
        self.side = 1.
        self.cue = None
        self.frames = 0
        self.target_switches = 0
        self.velocity_command = None
        self.feedforward = np.zeros(3)
        self.search_since = None
        self.support_since = self.slope_support_since = None
        self.issued_throttle = None
        self.climb_until = None
        self.descent_shortfall = 0.
        self.descent_scale = 1.
        self.launching = True
        self.state = 'launch'
        self.state_time = {}
        # Created on the first clearance sample: without that input nothing changes.
        self.clearance_config = clearance_config or TtcClearanceConfig()
        self.clearance = None
        self.clearance_braking = False
        self.clearance_time = {}
        # Off (None) unless declared for the motor contract; see LagTurnConfig.
        if lag_turn is not None and not isinstance(lag_turn, LagTurnConfig):
            raise ValueError('Pass a LagTurnConfig (or None) for lag-aware turns')
        self.lag_turn = lag_turn
        # False: the trigger, weight and lead are computed and logged but not applied (shadow control).
        self.lag_turn_apply = bool(lag_turn_apply)
        self.lag_turn_recent = []      # (capture time, ring-centre azimuth) of recent fresh in-view cues
        self.lag_turn_centre = None    # filtered ring-centre bearing (the trigger's reference; never flown)
        self.lag_turn_since = None
        self.lag_turn_weight = 0.
        self.lag_turn_lead_deg = 0.
        self.lag_turn_triggers = 0
        self.lag_turn_time = 0.
        # Gap aim (off unless declared): see haltere.liftoff.gap_aim.
        if gap_aim is not None and not isinstance(gap_aim, GapAimConfig):
            raise ValueError('Pass a GapAimConfig (or None) for the gap aim')
        self.gap_aim = GapAim(gap_aim) if gap_aim is not None else None
        self.gap_apply = bool(gap_apply)
        self.gap_offset_deg = 0.       # rotation currently applied to the aim ray and the filtered direction
        self.gap_flag_deg = 0.         # the last in-view ring cue's flag-clearance offset from its centre
        self.gap_conflict = ''         # conflict found this tick ('ring', 'flag' or '')
        self.gap_ring_deg = float('nan')
        # Wall-pilot rules (off unless declared; obstacle stack only): see TurnFirstConfig and CeilingGuardConfig.
        # wall_apply False computes and logs them without applying them (the stack's shadow control): the flown
        # governor then has no ceiling guard and a guarded copy fed the same samples reports what it would do.
        if turn_first is not None and not isinstance(turn_first, TurnFirstConfig):
            raise ValueError('Pass a TurnFirstConfig (or None) for the turn-first rule')
        if ceiling_guard is not None and not isinstance(ceiling_guard, CeilingGuardConfig):
            raise ValueError('Pass a CeilingGuardConfig (or None) for the ceiling guard')
        if ceiling_guard is not None and not isinstance(self.clearance_config, TtcClearanceConfig):
            raise ValueError('The ceiling guard is part of the TTC clearance policy')
        self.turn_first, self.ceiling_guard, self.wall_apply = turn_first, ceiling_guard, bool(wall_apply)
        self.clearance_shadow = None   # the guarded governor copy in shadow
        self.wall_brake_at = -np.inf   # last tick at which the clearance cap bound the request (a wall brake)
        self.turn_first_since = self.turn_first_ray = None
        self.turn_first_block_until = -np.inf
        self.turn_first_active = False
        self.turn_first_counts = dict(episodes=0, aligned=0, handoff=0, timeout=0)
        self.turn_first_time = 0.

    def _ingest_clearance(self, clearance, velocity, yaw, now):
        c = self.clearance_config
        stamp = clearance.get('time')
        if stamp is None or not np.isfinite(stamp):
            raise ValueError('A clearance sample needs its capture time')
        if self.clearance is None:
            guard = self.ceiling_guard
            if guard is not None and self.wall_apply:
                self.clearance = TtcClearanceGovernor(c, ceiling=guard)
            else:
                self.clearance = clearance_governor(c)
                if guard is not None:
                    self.clearance_shadow = TtcClearanceGovernor(c, ceiling=guard)
        if not (0 <= now-stamp <= c.max_age_s
                and (self.clearance.last_time is None or stamp > self.clearance.last_time)):
            return
        ttc, distance, below, lower = (clearance.get(k) for k in ('ttc', 'distance', 'below_fraction', 'ttc_lower'))
        for name, value in (('ttc', ttc), ('distance', distance), ('ttc_lower', lower)):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f'Invalid clearance {name}')
        if below is not None and not (np.isfinite(below) and 0 <= below <= 1):
            raise ValueError('Invalid clearance below_fraction')
        position, _ = self.pose_history.at(stamp)
        speed = float(np.linalg.norm(velocity))
        # Looming is measured around the focus of expansion: the travel direction.
        ray = velocity/speed if speed > .5 else np.array([np.cos(yaw), np.sin(yaw), 0.])
        extra = dict(ttc_lower=lower) if isinstance(self.clearance, TtcClearanceGovernor) else {}
        self.clearance.ingest(float(stamp), ttc, distance, below, position, ray, speed, received=now, **extra)
        if self.clearance_shadow is not None:
            self.clearance_shadow.ingest(float(stamp), ttc, distance, below, position, ray, speed, received=now,
                                         ttc_lower=lower)

    def _ingest(self, detection, capture_time, now):
        cue = detection.get('race_cue') if detection else None
        if not (capture_time is not None and 0 <= now-capture_time <= .12
                and (self.last_capture is None or capture_time > self.last_capture)):
            return
        self.last_capture = capture_time
        self.cue = cue
        if cue is None:
            return
        if not np.isfinite([cue['u'], cue['v']]).all() or not (0 <= cue['u'] <= 1 and 0 <= cue['v'] <= 1):
            raise ValueError('Invalid race cue image position')
        aim_u = cue.get('aim_u', cue['u'])
        if not np.isfinite(aim_u) or not 0 <= aim_u <= 1:
            raise ValueError('Invalid race cue clearance position')
        _, q = self.pose_history.at(capture_time)
        ray = self.camera.unproject_body(np.array([[aim_u*320, cue['v']*180]]))[0]
        ray = quat_wxyz_to_mat(q) @ ray
        ray = ray/max(np.linalg.norm(ray), 1e-9)
        if self.lag_turn is not None:
            # A new checkpoint moves the ring marker itself: the trigger reads the ring-centre ray (u, v), never the
            # flown aim beside it (aim_u, the flag clearance), whose appearance or one-frame flicker is no switch.
            centre = self.camera.unproject_body(np.array([[cue['u']*320, cue['v']*180]]))[0]
            centre = quat_wxyz_to_mat(q) @ centre
            centre = centre/max(np.linalg.norm(centre), 1e-9)
            self._lag_turn_trigger(centre, bool(cue['edge']), capture_time)
            self.lag_turn_centre, _ = self._blend(self.lag_turn_centre, centre)
        if self.gap_aim is not None:
            ray = self._gap_ray(ray, cue, q, now)
        self.direction, switched = self._blend(self.direction, ray)
        self.target_switches += int(switched)
        self.last_seen, self.edge = capture_time, bool(cue['edge'])
        # Corner clamps stay lateral: Liftoff clamps a marker behind the drone
        # to a top corner, so a corner is not evidence of a target above/below.
        self.below = self.edge and cue['v'] > .96 and .1 < cue['u'] < .9
        self.above = self.edge and cue['v'] < .04 and .1 < cue['u'] < .9
        c = self.config
        if self.below:
            # A bottom clip only bounds the target below the clamped edge ray.
            # Nose-up braking lifts that ray toward the horizon, so weight the
            # descent by how far below the horizon it points; latch within one
            # clip episode so the response does not flicker as attitude changes.
            elevation = np.degrees(np.arcsin(np.clip(ray[2], -1, 1)))
            weight = float(np.clip((-elevation-c.below_weak_deg)/(c.below_full_deg-c.below_weak_deg), 0, 1))
            self.below_weight = max(self.below_weight, weight)
            self.edge_depression = float(max(0., -elevation))
            # Only an unbroken clip of the same ring is evidence that the slope is too shallow.
            if (self.below_since is None or switched or self.below_last is None
                    or capture_time-self.below_last > c.below_gap_s):
                self.below_since = capture_time
            self.below_last = capture_time
        else:
            self.below_weight = 0.
            self.edge_depression = 0.
            self.below_since = self.below_last = None
        if not ((self.below or self.above) and abs(cue['u']-.5) < .1):
            self.vertical_clip_since = None
            self.sweep_side = None
        elif self.vertical_clip_since is None:
            self.vertical_clip_since = capture_time
        self.frames += 1

    def _blend(self, previous, ray):
        """(filtered bearing after one more cue ray, whether it was a new target): a ray more than new_target_deg
        from the previous bearing is adopted as is (a checkpoint switch), a closer one is blended in."""
        if previous is None or np.degrees(np.arccos(np.clip(ray @ previous, -1, 1))) > self.config.new_target_deg:
            return ray, previous is not None
        blended = previous+self.config.direction_blend*(ray-previous)
        return blended/max(np.linalg.norm(blended), 1e-9), False

    def _desired(self, position, velocity, yaw, now):
        c = self.config
        horizontal_speed = np.linalg.norm(velocity[:2])
        if self.direction is None:
            return np.array([0., 0., 1.5 if self.launching else 0.]), 'wait'
        seen_age = now-self.last_seen
        coast = min(c.coast_s, max(.25, c.coast_distance_m/max(horizontal_speed, 1e-6)))
        if seen_age > coast:
            return np.zeros(3), 'search'
        if seen_age > .25 and self.velocity_command is not None:
            return self.velocity_command.copy(), 'coast'
        d = self.direction
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        dh = d[:2]/max(np.linalg.norm(d[:2]), 1e-9)
        if self.edge and not (self.below or self.above):
            # Beyond the horizontal field of view: keep a moderate speed and head
            # just beyond the clamped edge ray, which turns with the camera.
            angle = np.arctan2(heading[0]*dh[1]-heading[1]*dh[0], heading @ dh)
            self.side = np.sign(angle) if abs(angle) > .02 else self.side
            turn = np.arctan2(dh[1], dh[0])+self.side*np.radians(c.side_margin_deg)
            # Height is left to the in-view states: Liftoff places side-clamped
            # markers near the lower corners even for rings that turn out to be
            # above, so the clamped ray's elevation is not vertical evidence.
            horizontal = self.speed*c.side_speed_fraction
            return np.r_[np.array([np.cos(turn), np.sin(turn)])*horizontal, 0.], 'side'
        reference = velocity[:2]/horizontal_speed if horizontal_speed > 1. else heading
        alignment = max(0., float(reference @ dh))
        fraction = c.min_speed_fraction+(1-c.min_speed_fraction)*alignment**2
        speed = self.speed*fraction
        slope = d[2]/max(np.linalg.norm(d[:2]), 1e-6)
        if self.below:
            # The target lies below the lower image edge, i.e. at least as steep
            # as the clamped edge ray. Descend along a slope only slightly
            # steeper than that bound, rather than diving: racing lines often
            # follow terrain down a hill, and forward pitch soon brings the
            # marker back into view. Weight the response by the evidence.
            w = self.below_weight
            horizontal = speed*(1-w)+min(speed, max(c.edge_speed, c.below_speed_fraction*speed))*w
            clipped_s = max(0., now-self.below_since) if self.below_since is not None else 0.
            margin = min(c.below_slope_margin_max_deg, c.below_slope_margin_deg+c.below_slope_growth_deg_s*clipped_s)
            sink = min(c.vertical_down, horizontal*np.tan(np.radians(min(self.edge_depression+margin, 80.))))
            return np.r_[dh*horizontal, -sink*w], 'below' if w > 0 else 'below_weak'
        if self.above:
            # The clipped elevation is only a lower bound: preserve that slope.
            speed = min(speed, c.vertical_up/max(slope, .2))
            return np.r_[dh*speed, c.vertical_up], 'above'
        vertical = speed*slope
        limit = c.vertical_up if vertical > 0 else c.vertical_down
        if abs(vertical) > limit:
            speed *= limit/abs(vertical)
            vertical = np.sign(vertical)*limit
        # Lag-aware turns (off by default): the horizontal goal leads the in-view bearing.
        return np.r_[self._lead(dh, velocity)*speed, vertical], 'cue'

    def _gap_ray(self, ray, cue, quaternion, now):
        """The aim ray rotated by the applied gap shift; reconciles the gap evidence with this ring cue.

        The ring centre's world azimuth is compared with the ring bearing the gap samples describe
        (another ring: a conflict), and the ring cue's own flag-clearance offset (aim_u beside the centre)
        with the side of the shift (the other side: a conflict). The filtered direction is brought to the
        same offset before it is blended with the new ray."""
        edge = bool(cue['edge'])
        if not edge:
            centre = quat_wxyz_to_mat(quaternion) @ self.camera.unproject_body(
                np.array([[cue['u']*320, cue['v']*180]]))[0]
            if np.linalg.norm(centre[:2]) > 1e-6 and np.linalg.norm(ray[:2]) > 1e-6:
                ring = float(np.degrees(np.arctan2(centre[1], centre[0])))
                self.gap_ring_deg = ring
                self.gap_flag_deg = wrap_deg(float(np.degrees(np.arctan2(ray[1], ray[0])))-ring)
                if self.gap_aim.reconcile_ring(ring, now):
                    self.gap_conflict = 'ring'
                elif self.gap_aim.flag_conflict(self.gap_flag_deg, now):
                    self.gap_conflict = 'flag'
        else:
            self.gap_flag_deg = 0.
        self._set_gap_offset()
        return rotate_z(ray, self.gap_offset_deg)

    def _set_gap_offset(self):
        """Rotate the filtered direction to the offset the applied gap shift calls for (none in shadow)."""
        offset = (direction_offset(self.gap_aim.applied, self.gap_flag_deg, self.gap_aim.config)
                  if self.gap_apply else 0.)
        if self.direction is not None and offset != self.gap_offset_deg:
            self.direction = rotate_z(self.direction, offset-self.gap_offset_deg)
        self.gap_offset_deg = offset

    def _lag_turn_trigger(self, centre, edge, capture_time):
        """Open a lag-turn window when a fresh in-view ring-centre bearing jumps (see LagTurnConfig).

        `centre` is the world ray to the ring marker's centre; the references are the in-view centre azimuths
        of the last trigger_span_s and the filtered centre bearing (`lag_turn_centre`, filtered like the flown
        direction but from centre rays), so neither the flag-clearance aim nor an applied gap shift can look
        like a new checkpoint."""
        lt = self.lag_turn
        if edge or np.linalg.norm(centre[:2]) < 1e-6:
            return
        azimuth = float(np.arctan2(centre[1], centre[0]))
        recent = [(t, a) for t, a in self.lag_turn_recent if capture_time-t <= lt.trigger_span_s]
        references = [a for _, a in recent]
        filtered = self.lag_turn_centre
        if filtered is not None and np.linalg.norm(filtered[:2]) > 1e-6:
            references.append(float(np.arctan2(filtered[1], filtered[0])))
        jump = max((abs((azimuth-a+np.pi) % (2*np.pi)-np.pi) for a in references), default=0.)
        if jump >= np.radians(lt.trigger_deg):
            self.lag_turn_since = capture_time
            self.lag_turn_triggers += 1
            recent = []                # later frames compare with the new bearing
        self.lag_turn_recent = recent+[(capture_time, azimuth)]

    def _lag_turn_weight_at(self, now):
        """1 during a lag-turn window, falling linearly to 0 over its last fade_s; 0 when off."""
        lt = self.lag_turn
        if lt is None or self.lag_turn_since is None:
            return 0.
        age = now-self.lag_turn_since
        if not 0 <= age < lt.window_s:
            return 0.
        return float(min(1., (lt.window_s-age)/lt.fade_s))

    def _lead(self, dh, velocity):
        """Horizontal goal direction: the bearing, led beyond it during a lag-turn window.

        The lead is course_lead times the angle from the flown course to the bearing,
        clipped, so it shrinks to zero as the lagging motor's course catches up and turns
        back if the course overshoots. With the gap aim, `dh` carries the applied gap shift:
        the lead is computed on the ring cue's own bearing (the shift removed) and the shift
        is added after it, so the lead never amplifies the shift and each keeps its own
        declared bound (course_lead_max_deg, max_shift_deg); in shadow the logged lead is the
        one this rule would fly."""
        lt = self.lag_turn
        if lt is None or self.lag_turn_weight <= 0:
            return dh
        horizontal_speed = float(np.linalg.norm(velocity[:2]))
        if horizontal_speed < lt.min_course_speed:
            return dh
        course = velocity[:2]/horizontal_speed
        shift = np.radians(self.gap_offset_deg)
        bearing = dh if shift == 0. else np.array([np.cos(-shift)*dh[0]-np.sin(-shift)*dh[1],
                                                   np.sin(-shift)*dh[0]+np.cos(-shift)*dh[1]])
        angle = float(np.arctan2(course[0]*bearing[1]-course[1]*bearing[0], course @ bearing))
        if abs(angle) > np.radians(lt.max_lead_angle_deg):
            return dh
        limit = np.radians(lt.course_lead_max_deg)
        lead = self.lag_turn_weight*float(np.clip(lt.course_lead*angle, -limit, limit))
        self.lag_turn_lead_deg = float(np.degrees(lead))
        if not self.lag_turn_apply:
            return dh                  # shadow: the lead is logged, not flown
        turn = lead+shift
        cos, sin = np.cos(turn), np.sin(turn)
        return np.array([cos*bearing[0]-sin*bearing[1], sin*bearing[0]+cos*bearing[1]])

    def _bearing_off_deg(self, yaw):
        """Horizontal angle (deg) between the filtered checkpoint bearing and the heading, or None."""
        if self.direction is None or np.linalg.norm(self.direction[:2]) < 1e-6:
            return None
        bearing = float(np.arctan2(self.direction[1], self.direction[0]))
        return float(abs(np.degrees((bearing-yaw+np.pi) % (2*np.pi)-np.pi)))

    def _near_wall(self, speed, now):
        """The turn-first wall condition (see TurnFirstConfig): a stand-off, or a recent wall brake at low speed."""
        gov, tf = self.clearance, self.turn_first
        if gov is None or gov.cap_ray is None:
            return False
        if isinstance(gov, TtcClearanceGovernor):
            standoff = now <= gov.standoff_until and gov.cap is not None
        else:
            standoff = bool(gov.sustained and gov.cap is not None and gov.cap < gov.config.standoff_speed)
        return standoff or (now-self.wall_brake_at <= tf.brake_recent_s and speed <= tf.slow_speed)

    def _turn_first(self, state, desired, velocity, yaw, now):
        """Turn before translating at a wall (TurnFirstConfig): the limited horizontal request while an episode
        is active (also computed in shadow), else None. Updates the episode state and its counts."""
        tf = self.turn_first
        off = self._bearing_off_deg(yaw)
        if self.turn_first_since is not None:
            end = ('handoff' if state in TURN_FIRST_HANDOFF_STATES or self.launching else
                   'aligned' if state in TURN_FIRST_BEARING_STATES and off is not None and off <= tf.release_deg else
                   'timeout' if now-self.turn_first_since >= tf.max_s else None)
            if end is not None:
                self.turn_first_counts[end] += 1
                self.turn_first_since = self.turn_first_ray = None
                if end == 'timeout':
                    self.turn_first_block_until = now+tf.rearm_s
        speed = float(np.linalg.norm(velocity[:2]))
        if (self.turn_first_since is None and not self.launching and now >= self.turn_first_block_until
                and state not in TURN_FIRST_HANDOFF_STATES and self._near_wall(speed, now)
                and (state == 'side' or (state in TURN_FIRST_BEARING_STATES and off is not None
                                         and off >= tf.engage_deg))):
            ray = np.asarray(self.clearance.cap_ray, float)[:2]
            if np.linalg.norm(ray) > 1e-6:
                self.turn_first_since, self.turn_first_ray = now, ray/np.linalg.norm(ray)
                self.turn_first_counts['episodes'] += 1
        self.turn_first_active = self.turn_first_since is not None
        if not self.turn_first_active:
            return None
        horizontal = np.array(desired[:2], dtype=float)
        horizontal -= self.turn_first_ray*max(0., float(horizontal @ self.turn_first_ray))
        norm = float(np.linalg.norm(horizontal))
        if norm > tf.creep_speed:
            horizontal *= tf.creep_speed/norm
        return horizontal

    @staticmethod
    def _cap_command(previous, command, cap, ray, dt, rate, top, vertical_limits):
        """The command with its component along `ray` brought down to `cap` at up to `rate` m/s^2 (beyond the
        taper), the total change bounded as the clearance cap bounds it."""
        before, after = float(previous @ ray), float(command @ ray)
        if after <= cap:
            return command
        room = max(0., rate*dt-max(0., before-after))
        command = command-ray*min(after-cap, room)
        change = command-previous
        norm = float(np.linalg.norm(change[:2]))
        if norm > top:
            change[:2] *= top/norm
        change[2] = np.clip(change[2], *vertical_limits)
        return previous+change

    def _horizontal_step(self, current, goal, dt, heading_time_constant=None):
        """Change of the horizontal request this tick: a coordinated turn.

        The heading rotates toward the goal's at most turn_acceleration/max(|v|, 1)
        rad/s (centripetal share <= turn_acceleration) and the speed changes with the
        rest of the command_acceleration budget, so a new bearing is flown as an arc
        at nearly constant speed instead of a chord through lower speeds. Both parts
        taper with command_time_constant near the goal, which to first order equals
        the straight-line taper for small corrections. Without a defined heading
        (request or goal slower than turn_min_speed) the straight-line slew applies.
        `heading_time_constant`, when given, replaces command_time_constant in the
        heading taper only (a lag-aware turn); the default keeps the rule above.
        """
        c = self.config
        heading_tc = c.command_time_constant if heading_time_constant is None else heading_time_constant
        chord = goal-current
        norm = float(np.linalg.norm(chord))
        top = c.command_acceleration*dt
        # Taper the requested acceleration near the goal, so feedforward ends
        # smoothly instead of overshooting into a nose-up brake.
        limit = min(top, norm*dt/c.command_time_constant)
        speed, target = float(np.linalg.norm(current)), float(np.linalg.norm(goal))
        if min(speed, target) < c.turn_min_speed:
            return chord*(limit/norm) if norm > limit else chord
        angle = float(np.arctan2(current[0]*goal[1]-current[1]*goal[0], current @ goal))
        rotation = np.sign(angle)*min(abs(angle), abs(angle)*dt/heading_tc,
                                      c.turn_acceleration/max(speed, 1.)*dt)
        normal = speed*abs(rotation)/max(dt, 1e-9)
        room = float(np.sqrt(max(c.command_acceleration**2-normal**2, 0.)))*dt
        change = target-speed
        change = np.sign(change)*min(abs(change)*dt/c.command_time_constant, room)
        heading = float(np.arctan2(current[1], current[0]))+rotation
        step = (speed+change)*np.array([np.cos(heading), np.sin(heading)])-current
        size = float(np.linalg.norm(step))
        return step*(top/size) if size > top else step

    def update(self, senses, omega, detection, capture_time, now, clearance=None, gap=None):
        """One control tick. `clearance`, when given, is a causal forward-clearance sample:
        dict(time=capture time, ttc=s or None, distance=m or None, below_fraction=0..1 or None,
        optional ttc_lower=s or None), where below_fraction is the share of the image expansion
        below the flight path (about 0.5 for a wall facing the drone, towards 1 for ground under
        the path) and ttc_lower the TTC of the surface fitted below the path.
        ttc and distance both None means no evidence (e.g. low texture).
        `gap`, when given and a gap aim is declared, is the latest causal gap-cue sample
        (haltere.liftoff.camera_process.gap_sample); without a declared gap aim it is ignored."""
        c = self.config
        position = senses['pos'][0].cpu().numpy().astype(float)
        velocity = senses['vel_world'][0].cpu().numpy().astype(float)
        rotation = quat_wxyz_to_mat(senses['quat'][0].cpu().numpy())
        dt = .01 if self.last_time is None else float(np.clip(now-self.last_time, 0., .1))
        self.last_time = now
        if position[2] >= c.launch_height:
            self.launching = False
        if self.gap_aim is not None:
            self.gap_conflict = ''
            # Terrain side steer only while the looming governor reports terrain (a climb request).
            terrain = self.clearance is not None and self.clearance.climb > 0
            self.gap_aim.ingest(gap, now, terrain=terrain)
            self.gap_aim.step(now, dt)
            self._set_gap_offset()
        self._ingest(detection, capture_time, now)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        self.lag_turn_weight = 0. if self.launching else self._lag_turn_weight_at(now)
        self.lag_turn_lead_deg = 0.
        desired, state = self._desired(position, velocity, yaw, now)
        in_view_goal = state in LAG_TURN_STATES
        if state == 'search':
            self.search_since = now if self.search_since is None else self.search_since
            if now-self.search_since < c.search_climb_s:
                desired[2] = c.search_climb
        else:
            self.search_since = None
        if self.launching:
            desired[:2] *= min(1., 1.5/max(np.linalg.norm(desired[:2]), 1e-9))
            desired[2] = max(desired[2], 1.5)
            state = 'launch'
        elif position[2] > -c.surface_release_m:
            # Height is measured from the launch point, which may be a rooftop
            # or hill. Sink gently near and above that plane; the support rule
            # below still recognises contact because the cap exceeds 0.8 m/s.
            desired[2] = max(desired[2], -min(c.vertical_down, c.surface_sink+c.surface_sink_per_m*max(0., position[2])))
        # Descent path angle: a vehicle that keeps falling short of the requested
        # sink rate would pass above a lower checkpoint. Slow horizontally in
        # proportion so the flight path keeps the requested slope. A vehicle
        # that tracks its descents is unaffected.
        shortfall = 0.
        if self.velocity_command is not None and self.velocity_command[2] < -c.descent_sink and not self.launching:
            shortfall = max(0., float(velocity[2]-self.velocity_command[2]))
        self.descent_shortfall += (1-np.exp(-dt/c.descent_time_constant))*(shortfall-self.descent_shortfall)
        self.descent_scale = float(np.clip(1-(self.descent_shortfall-c.descent_free)/c.descent_span,
                                           c.descent_min_scale, 1.))
        desired[:2] *= self.descent_scale
        # Support: a requested descent the vehicle cannot achieve means contact
        # below (terrain or an object), not a controller fault. Climb briefly.
        if self.climb_until is not None and now < self.climb_until:
            desired[2] = max(desired[2], 1.)
            state = 'support_climb'
        elif (self.velocity_command is not None and self.velocity_command[2] < -.8
              and velocity[2] > max(-.25, self.velocity_command[2]+.6)):
            self.support_since = now if self.support_since is None else self.support_since
            self.slope_support_since = None
            if now-self.support_since > c.support_after_s:
                self.climb_until, self.support_since = now+c.support_climb_s, None
        elif (self.velocity_command is not None and self.velocity_command[2] < -.8
              and velocity[2] > self.velocity_command[2]+c.support_slope_shortfall
              and self.calibration is not None and self.issued_throttle is not None
              and self.issued_throttle < self.calibration[2]-c.support_thrust_margin):
            # Thrust well below hover would reach the requested sink within a
            # fraction of a second in free air; a persistent shortfall means the
            # vehicle is resting on something, e.g. sliding down a hillside.
            self.support_since = None
            self.slope_support_since = now if self.slope_support_since is None else self.slope_support_since
            if now-self.slope_support_since > c.support_slope_after_s:
                self.climb_until, self.slope_support_since = now+c.support_climb_s, None
        else:
            self.support_since = self.slope_support_since = None
        cap = ray = vertical_cap = None
        climb = 0.
        if clearance is not None and not self.launching:
            self._ingest_clearance(clearance, velocity, yaw, now)
        if self.clearance is not None:
            cap, ray, climb = self.clearance.limits(position, velocity, now, dt, c.vertical_up)
            if self.clearance_shadow is not None:
                self.clearance_shadow.limits(position, velocity, now, dt, c.vertical_up)
            if climb > 0:
                # Expansion below the flight path: rise over it rather than stop.
                desired[2] = max(desired[2], climb)
            vertical_cap = getattr(self.clearance, 'vertical_cap', None)
            if vertical_cap is not None:
                # Ceiling guard: overhead evidence during a climb bounds the whole vertical request.
                desired[2] = min(desired[2], vertical_cap)
            along = float(desired @ ray) if cap is not None else 0.
            braking = cap is not None and along > cap
            if braking:
                desired = desired-ray*(along-cap)
                if not self.clearance_braking:
                    self.clearance.counts['blind_engagements' if self.clearance.blind else 'brake_engagements'] += 1
                self.wall_brake_at = now
            self.clearance_braking = braking
            status = ('blind' if self.clearance.blind else 'brake') if braking else self.clearance.status
            self.clearance_time[status] = self.clearance_time.get(status, 0.)+dt
        turn_first = None
        if self.turn_first is not None:
            # Turn before translating at a wall (computed in shadow too, applied only with wall_apply).
            turn_first = self._turn_first(state, desired, velocity, yaw, now)
            if turn_first is not None:
                self.turn_first_time += dt
                if self.wall_apply:
                    desired[:2] = turn_first
                else:
                    turn_first = None
        if self.velocity_command is None:
            self.velocity_command = velocity.copy()
        step = desired-self.velocity_command
        heading_tc = None
        if self.lag_turn is not None and self.lag_turn_weight > 0 and in_view_goal:
            if self.lag_turn_apply:
                heading_tc = c.command_time_constant+self.lag_turn_weight*(
                    self.lag_turn.heading_time_constant-c.command_time_constant)
            self.lag_turn_time += dt
        step[:2] = self._horizontal_step(self.velocity_command[:2], desired[:2], dt, heading_tc)
        if state == 'search':
            norm = float(np.linalg.norm(step[:2]))
            if norm > c.search_deceleration*dt:
                step[:2] *= c.search_deceleration*dt/norm
        up = max(c.vertical_command_acceleration, self.clearance_config.terrain_climb_acceleration if climb > 0 else 0.)
        down = (c.vertical_command_acceleration if vertical_cap is None
                else max(c.vertical_command_acceleration, self.clearance.ceiling.vertical_slew))
        step[2] = np.clip(step[2], -down*dt, up*dt)
        previous = self.velocity_command.copy()
        self.velocity_command = self.velocity_command+step
        slew = self.clearance_config.brake_slew
        top, vertical_limits = max(c.command_acceleration, slew)*dt, (-down*dt, up*dt)
        if cap is not None:
            # The cap acts on the request itself, without the taper, at up to brake_slew.
            self.velocity_command = self._cap_command(previous, self.velocity_command, cap, ray, dt, slew, top,
                                                      vertical_limits)
        if turn_first is not None:
            # ... and so does turn-first: no speed toward the wall that capped it.
            self.velocity_command = self._cap_command(previous, self.velocity_command, 0.,
                                                      np.r_[self.turn_first_ray, 0.], dt, slew, top, vertical_limits)
        raw_ff = (self.velocity_command-previous)/max(dt, 1e-3)
        alpha = 1-np.exp(-dt/c.feedforward_time_constant)
        self.feedforward = self.feedforward+alpha*(raw_ff-self.feedforward)
        self.state = state
        self.state_time[state] = self.state_time.get(state, 0.)+dt
        # Yaw faces the observed checkpoint; searching turns toward its last side.
        world_rate = float((rotation @ np.asarray(omega))[2])
        if state == 'search' or self.direction is None:
            yaw_rate = c.search_yaw_rate*self.side if (capture_time is not None and 0 <= now-capture_time <= .25) else 0.
        else:
            angle = (np.arctan2(self.direction[1], self.direction[0])-yaw+np.pi) % (2*np.pi)-np.pi
            if abs(angle) > .08:
                self.side = np.sign(angle)
            yaw_rate = float(np.clip(c.yaw_gain*angle-c.yaw_damping*world_rate, -c.max_yaw_rate, c.max_yaw_rate))
            if (state == 'above' and self.vertical_clip_since is not None
                    and now-self.vertical_clip_since > c.edge_sweep_after_s):
                # A centred top clip cannot separate overhead from behind; a slow
                # yaw moves a target behind off the centre. Its direction is latched
                # for the episode: re-deciding it from the bearing flips it every few
                # degrees. A bottom clip is never behind (Liftoff clamps rings behind
                # to the top), so a descent keeps facing the ring instead of weaving.
                if self.sweep_side is None:
                    self.sweep_side = self.side
                yaw_rate = c.edge_sweep_rate*self.sweep_side
        # Invert the measured post-expo yaw curve so max_yaw_rate is honoured.
        desired_yaw = -float(np.clip(measured_inverse_rate(
            torch.tensor([np.degrees(yaw_rate)], dtype=torch.float64), *self.yaw_curve)[0], -1., 1.))
        self.pilot.sight_yaw += float(np.clip(desired_yaw-self.pilot.sight_yaw, -c.yaw_slew*dt, c.yaw_slew*dt))
        self.pilot.mode = {'cue': 2, 'launch': 2, 'below_weak': 2, 'coast': 3, 'side': 5, 'below': 6, 'above': 7,
                           'support_climb': 8}.get(state, 4)
        fresh = self.last_seen is not None and now-self.last_seen < .25
        self.pilot.target = SimpleNamespace(t_last=self.last_seen) if fresh else None
        self.pilot.carrot = position+self.velocity_command
        self.host.flow_gain = (self.velocity_scale if self.velocity_scale is not None
                               else max(1., self.reference_speed/self.speed))
        scaled_velocity = senses['vel_world']*senses['vel_world'].new_tensor(
            [self.host.flow_gain, self.host.flow_gain, 1.])
        modified = {**senses, 'vel_world': scaled_velocity,
                    'vel_body': scaled_velocity @ torch.as_tensor(rotation, dtype=scaled_velocity.dtype,
                                                                 device=scaled_velocity.device)}
        # Body-frame one-second lead for goal-point motor contracts and logs.
        return rotation.T @ self.velocity_command, modified

    def _guarded_governor(self):
        """The governor that runs the ceiling guard: the flown one, or its shadow copy; None without the guard."""
        if self.clearance_shadow is not None:
            return self.clearance_shadow
        return self.clearance if getattr(self.clearance, 'ceiling', None) is not None else None

    def wall_log(self):
        """Per-tick wall-pilot values for logs: turn_first (1 while an episode is active, also in shadow; NaN when
        not declared) and the ceiling guard's governor status, climb request and vertical bound (the guarded copy's
        in shadow; NaN / '' without the guard or before the first clearance sample)."""
        nan = float('nan')
        guard = self._guarded_governor()
        return dict(turn_first=float(self.turn_first_active) if self.turn_first is not None else nan,
                    ceiling_status=guard.status if guard is not None else '',
                    ceiling_climb=float(guard.climb) if guard is not None else nan,
                    ceiling_vertical_cap=nan if guard is None or guard.vertical_cap is None else float(guard.vertical_cap))

    def _wall_metadata(self):
        if self.turn_first is None and self.ceiling_guard is None:
            return None
        guard = self._guarded_governor()
        keys = ('overhead_samples', 'overhead_engagements', 'unexplained_walls', 'weak_climb_samples',
                'suppressed_climb_samples', 'climb_engagements')
        return dict(
            version=WALL_PILOT_VERSION, applied=self.wall_apply,
            turn_first=None if self.turn_first is None else dict(
                rule='near a wall (a clearance stand-off, or a wall brake within brake_recent_s at <= slow_speed) with '
                     'the checkpoint clamped at a side edge/corner or its bearing >= engage_deg off the heading: the '
                     'horizontal request loses its component toward the wall (the capping looming ray) and is bounded '
                     'to creep_speed, the speed toward the wall is removed at brake_slew, yaw keeps turning to the '
                     'checkpoint; ends when an in-view (or bottom/top clamped) bearing is within release_deg '
                     '(aligned), on search/launch/support climb (handoff) or after max_s (timeout, then rearm_s '
                     'without an episode)',
                parameters=asdict(self.turn_first), counts=dict(self.turn_first_counts),
                active_seconds=round(self.turn_first_time, 3)),
            ceiling_guard=None if self.ceiling_guard is None else dict(
                rule='during a climb a sample without vertical evidence is terrain only if its lower-surface TTC <= '
                     'lower_ratio x its alarm TTC (else a wall); such weak terrain climbs <= weak_climb and not beyond '
                     'weak_climb_max_m above the last below-path climb request; while climbing faster than '
                     'overhead_min_rise, overhead_confirm samples with TTC < overhead_ttc_s whose expansion lies above '
                     'the path (below_fraction <= overhead_fraction) or is unexplained cut the climb and start an '
                     'overhead hold of hold_s (no climb, below-path samples brake, vertical request <= vertical_cap, '
                     'brought down at vertical_slew)',
                parameters=asdict(self.ceiling_guard),
                governor='flown' if self.wall_apply else 'shadow copy fed the same samples',
                counts=None if guard is None else {k: guard.counts.get(k, 0) for k in keys}))

    def command(self, action):
        result = np.array(action, copy=True)
        self.issued_throttle = float(result[0])   # motor throttle (brain units), read by the support rule
        result[3] = float(np.clip(self.pilot.sight_yaw, -1., 1.))
        if self.calibration is not None:
            # Throttle and yaw share one pad stick clamped to the unit circle:
            # give throttle priority instead of silently losing thrust.
            hover, scale, hover_stick = self.calibration
            throttle = hover+scale*(float(result[0])-hover_stick)
            room = float(np.sqrt(max(.97**2-min(throttle*throttle, .97**2), 0.)))
            result[3] = float(np.clip(result[3], -room, room))
        return result

    def _clearance_policy(self):
        if isinstance(self.clearance_config, TtcClearanceConfig):
            return dict(policy='ttc-graded',
                        input='causal looming samples (time, ttc, distance, below_fraction, ttc_lower)',
                        wall='after confirmation each sample caps the speed along the looming ray at '
                             'v*clip((ttc-ttc_min)/(ttc_target-ttc_min), floor_fraction, 1) >= min_speed, falling at '
                             'brake_rate, held while a wall sample has ttc < hold_ttc_s (+hold_s), then released at '
                             'release; ttc aged by odometry; no metric stopping distance',
                        terrain='below_fraction >= terrain_fraction requests a climb up to vertical_up and brakes '
                                'terrain_brake as much as a wall',
                        standoff='a wall that capped the drone at <= standoff_speed holds the cap for standoff_s',
                        no_evidence='no constraint beyond the hold and the stand-off memory')
        return dict(policy='stopping-distance',
                    input='causal forward clearance samples (time, ttc, distance, below_fraction)',
                    wall='speed along the looming ray capped at the stopping speed '
                         '-aL + sqrt((aL)^2 + 2a(d - margin)); sample age removed by odometry dead reckoning',
                    terrain='below_fraction >= terrain_fraction adds a climb floor instead of braking',
                    no_evidence='no constraint beyond dead-reckoned memory unless blind_after_s is finite')

    def metadata(self):
        return dict(mode='race-cue', profile=self.profile,
                    goal_source='visible next-checkpoint ring with local flag clearance',
                    visible_race_cues=True, runtime_route_oracle=False,
                    local_flag_clearance=True, visible_route_arrows_for_clearance_side=True,
                    guidance='world velocity along the filtered cue bearing, acceleration-limited with feedforward',
                    speed_schedule='min_speed_fraction + (1-min)*cos^2(angle between velocity and bearing)',
                    turn='coordinated: heading rotation with centripetal share <= turn_acceleration '
                         '(turn_acceleration/max(|v|, 1) rad/s), speed change within the rest of '
                         'command_acceleration; straight-line slew below turn_min_speed',
                    side_edge='side_speed_fraction of nominal speed toward side_margin_deg beyond the '
                              'clamped edge ray bearing, level (edge height is not used)',
                    bottom_edge='descent bounded by the clamped edge ray depression plus a margin that grows '
                                'with the clip duration (below_slope_growth_deg_s, up to '
                                'below_slope_margin_max_deg; the duration restarts after a below_gap_s gap in '
                                'bottom-clip frames or on a new ring), weighted by that depression and latched per clip',
                    top_edge='climb while preserving the clipped slope bound',
                    centred_vertical_clip='centred top clip: slow yaw sweep after edge_sweep_after_s, direction '
                                          'latched per episode; a centred bottom clip keeps facing the ring',
                    launch_surface='sink rate limited near and above the launch plane',
                    yaw_mapping='measured post-expo yaw curve inverse, throttle priority on the shared stick',
                    yaw_curve=list(self.yaw_curve),
                    support='requested descent not achieved for support_after_s, or short by support_slope_shortfall with the '
                            'issued throttle support_thrust_margin below hover for support_slope_after_s (a slope) '
                            '-> short climb',
                    cue_dropout='coast on the previous request, then search: slow at <= search_deceleration while rising at '
                                'search_climb for search_climb_s',
                    parameters=asdict(self.config), yaw_assistance=True, speed_assistance=True,
                    nominal_speed_mps=self.speed, trained_motor_reference_mps=self.reference_speed,
                    cue_frames=self.frames, target_switches_observed=self.target_switches,
                    state_seconds={k: round(v, 3) for k, v in self.state_time.items()},
                    estimated_passages=None,
                    lag_turn=None if self.lag_turn is None else dict(
                        rule='for window_s after a fresh in-view cue\'s ring-centre azimuth (marker u, v; not the '
                             'flag-clearance aim) differs >= trigger_deg from the filtered ring-centre bearing or from '
                             'the ring centre of an in-view cue of the last trigger_span_s (clamped markers never '
                             'trigger): with the ring in view (state cue) aim course_lead*(bearing - flown course) '
                             'beyond the bearing (clipped to course_lead_max_deg; none below min_course_speed or beyond '
                             'max_lead_angle_deg; an applied gap shift is removed before and added after the lead) and '
                             'taper the request heading with heading_time_constant; both fade out over the last fade_s; '
                             'the speed schedule keeps the bearing itself',
                        version=LAG_TURN_VERSION,
                        parameters=asdict(self.lag_turn), triggers=self.lag_turn_triggers,
                        active_seconds=round(self.lag_turn_time, 3), applied=self.lag_turn_apply),
                    gap_aim=None if self.gap_aim is None else dict(
                        self.gap_aim.metadata(), applied=self.gap_apply,
                        input='causal gap-cue samples: per-frame relative-depth free-interval shift beside the ring '
                              '(haltere.vision.gap_cue.decide in the camera stack), ring azimuth, terrain side '
                              'statistic',
                        rule='accept samples at most max_age_s old; confirm when confirm of the last window samples '
                             'within confirm_window_s vote for one side (obstacle: |shift| >= active_deg; terrain, '
                             'only while the looming governor requests a climb: |ln(L/R)| >= terrain_lr, '
                             'terrain_side_deg to the farther side); side latch side_latch_s; applied shift slews at '
                             '<= slew_deg_s and decays to 0 over decay_s; rotates the ring ray about world z in '
                             '_ingest; gap evidence of another ring bearing or against the ring cue\'s flag '
                             'clearance is a conflict: dropped, the ring cue\'s own aim is held for side_latch_s; '
                             'never changes the requested speed; with lag-aware turns the lead is computed on the '
                             'bearing without the shift and the shift is added after it (not amplified)'),
                    wall_pilot=self._wall_metadata(),
                    clearance_response=None if self.clearance is None else dict(
                        self._clearance_policy(),
                        parameters={k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                    for k, v in asdict(self.clearance_config).items()},
                        counts=dict(self.clearance.counts),
                        status_seconds={k: round(v, 3) for k, v in self.clearance_time.items()}),
                    limitations='Race guidance only; no freestyle objective, obstacle model or completed-lap inference')
