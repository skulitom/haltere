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
at a wall it cannot stop before, with the checkpoint far off the heading or
its marker lost, the pilot turns before it translates (no request toward the
wall, a creep speed until the bearing is inside a cone, bounded in time), and
the TTC governor's terrain climb is kept
out of ceilings (unexplained alarms during a climb are walls, weak climbs are
bounded, overhead evidence cuts the climb and bounds the vertical request).
``wall_apply`` False computes and logs them without applying them.

Optionally (``vertical_guard=VerticalGuardConfig``, off by default; the obstacle stack's vertical-guard
declaration), a scale-free graded vertical guard on the same looming samples: a time margin to the ground below
the path scales the pilot's own sink, a descent is stopped before any terrain climb, a terrain climb needs
confirmation and stays gentle until rising ground is confirmed, and the descent-path governor does not cut speed
for a sink the guard withheld or that contact prevents. ``vertical_apply`` False computes and logs it without
applying it.

Optionally (``motor_assist=MotorAssistConfig``, off by default; declared per motor contract in
configs/pilot, for the brain contract only), the request a lagging brain receives is adjusted after
every other rule: speed it flies beyond a binding cap lowers that cap in proportion (and a stopping model bounds the
speed toward a confirmed wall), and a bounded climb bias is added while it sinks below its vertical request or is asked
to accelerate hard from low speed. The pilot keeps its own request as its state.
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
# version 2 added the ceiling guard's overhead_min_rise, version 3 its overhead_positive, version 4 replaced
# turn-first's fixed slow_speed engagement with the stopping-distance one, version 5 added the clearance brake's
# sink floor (ClearanceBrakeConfig). Runners fly only WALL_PILOT_VERSION (versions 1-4 are kept for provenance and
# refused); WALL_PILOT_VERSIONS are the versions this code can rebuild for replays (version 4: the same rules without
# the sink floor).
WALL_PILOT_VERSION = 5
WALL_PILOT_VERSIONS = (4, 5)
# Pilot state in which the checkpoint's bearing is unknown (its marker was lost for more than 0.25 s and the pilot
# repeats its last request): turn-first may engage there too (version 4), with the horizontal request bounded to
# coast_creep_speed, because the direction to creep toward is not known.
TURN_FIRST_UNKNOWN_STATES = ('coast',)


@dataclass(frozen=True)
class TurnFirstConfig:
    """Turn before translating at a wall (a hairpin): what a pilot does when the next gate lies behind a wall
    beside it. Off unless a runner passes it (the wall-pilot declaration in configs/obstacles; obstacle stack only).

    Engage when both hold:
    - near a wall, at a horizontal speed of at most max_speed: the clearance governor holds a stand-off (a wall
      that capped the request at its standoff_speed or less is remembered) and the wall lies within the stopping
      distance from that stand-off speed (or from the closing speed, if higher: a drone that has flown away from
      the wall is no longer at it), or its wall brake capped the request within the last brake_recent_s and the
      drone could not stop before the wall:
      the remaining distance to the latest wall sample (TTC policy: its capture-time reach along its looming ray
      minus the odometry travelled along that ray since, 0 once reached or passed; 0 without such a sample, e.g.
      with the stopping-distance policy, which keeps none) is at most the stopping distance at the measured
      closing speed v along
      that ray, v*stop_latency_s + v^2/(2*stop_deceleration) + stop_margin_m (the motor's measured braking: its
      delay behind the request and its deceleration, declared per motor contract);
    - the checkpoint is far off the heading or unknown: its marker is clamped at a side edge or a corner (pilot
      state 'side'), or its bearing (the filtered aim bearing of a marker in view or clamped at the bottom/top
      edge) lies engage_deg or more from the heading, or the marker is lost and the pilot coasts on its last
      request (state 'coast': after a checkpoint passage in a confined space the next marker can take 0.2-0.5 s
      to appear; coasting on the old request carries the drone into the wall it just braked for).
    While engaged, the horizontal request loses any component toward the wall (along the looming ray that
    capped it, taken at engagement) and is bounded to creep_speed (coast_creep_speed while coasting: the
    direction to the checkpoint is not known), and the request's speed toward the wall is removed, and its
    horizontal speed brought to that bound, at the clearance brake_slew (beyond the usual taper); the vertical
    request and the yaw rule are unchanged, so the assisted yaw keeps
    turning toward the checkpoint (the clamped edge ray turns with the camera). The episode ends when the
    bearing of a marker in view (or bottom/top clamped) comes within release_deg of the heading ('aligned';
    the ordinary speed schedule and acceleration limits then resume), when a state that owns the request takes
    over (search, launch, support climb: 'handoff'), or after max_s ('timeout'), after which it cannot engage
    again for rearm_s: the drone never hovers at a wall indefinitely. During that rearm time the side rule still
    never requests speed toward a wall it is near (side guard: the component along the governor's current wall
    ray is removed, without the creep bound).
    Declaration version 4; versions 1-3 engaged at a wall brake only at or below a fixed slow_speed (1.5 m/s),
    not in 'coast', and had no side guard.
    """
    brake_recent_s: float = 1.
    engage_deg: float = 50.
    release_deg: float = 30.
    creep_speed: float = .8
    coast_creep_speed: float = 0.
    max_speed: float = 3.5
    stop_latency_s: float = .3
    stop_deceleration: float = 3.5
    stop_margin_m: float = .5
    max_s: float = 2.
    rearm_s: float = 2.

    def __post_init__(self):
        values = asdict(self)
        coast = values.pop('coast_creep_speed')
        if not np.isfinite(list(values.values())+[coast]).all() or min(values.values()) <= 0:
            raise ValueError('Use finite positive turn-first parameters (coast_creep_speed may be 0)')
        if not 0 <= coast <= self.creep_speed:
            raise ValueError('Use 0 <= coast_creep_speed <= creep_speed')
        if not self.release_deg < self.engage_deg < 90:
            raise ValueError('Use release_deg < engage_deg < 90 degrees')

    def stopping_distance(self, closing_speed):
        """Distance (m) the declared motor needs to stop from `closing_speed` (m/s along the wall ray), with the
        margin: v*stop_latency_s + v^2/(2*stop_deceleration) + stop_margin_m (v clipped at 0)."""
        v = max(0., float(closing_speed))
        return v*self.stop_latency_s+v*v/(2*self.stop_deceleration)+self.stop_margin_m


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
      (below_fraction <= overhead_fraction) or is unexplained (as above) is overhead evidence. It is positive
      evidence when something shows the alarm is not the surface below: below_fraction <= overhead_fraction, or
      a lower-surface TTC that is known and longer than lower_ratio x the alarm TTC (a sample where neither
      vertical window crosses the path says nothing either way: climbing a hill, the windows often lose it).
      overhead_confirm overhead samples within the governor's confirm_window_s, at least overhead_positive of
      them positive, cut the climb to 0 at once and start an overhead hold of hold_s (renewed by further
      confirmed evidence): no terrain climb is requested, below-path samples brake like walls, and the whole
      vertical request is bounded to vertical_cap, brought down at up to vertical_slew m/s^2.
    Below-path climbs (below_fraction >= terrain_fraction) are otherwise unchanged. Declaration version 3
    (version 1 had no overhead_min_rise, version 2 no overhead_positive).
    """
    lower_ratio: float = 1.
    weak_climb: float = 1.
    weak_climb_max_m: float = 1.
    overhead_fraction: float = .3
    overhead_ttc_s: float = 1.2
    overhead_confirm: int = 2
    overhead_positive: int = 1
    overhead_min_rise: float = .3
    hold_s: float = 1.
    vertical_cap: float = 0.
    vertical_slew: float = 15.

    def __post_init__(self):
        values = asdict(self)
        cap = values.pop('vertical_cap')
        if not np.isfinite(list(values.values())+[cap]).all() or min(values.values()) <= 0:
            raise ValueError('Use finite positive ceiling-guard parameters (vertical_cap may be <= 0)')
        for name in ('overhead_confirm', 'overhead_positive'):
            if int(getattr(self, name)) != getattr(self, name):
                raise ValueError(f'{name} counts samples')
        if not self.overhead_positive <= self.overhead_confirm:
            raise ValueError('overhead_positive must not exceed overhead_confirm')
        if not self.overhead_fraction < .5 or not cap <= self.weak_climb:
            raise ValueError('Overhead evidence lies above the path (overhead_fraction < 0.5); '
                             'vertical_cap <= weak_climb')


@dataclass(frozen=True)
class ClearanceBrakeConfig:
    """The clearance brake commands no descent the pilot did not ask for (wall-pilot declaration version 5; obstacle
    stack only, off unless a runner passes it; computed but not applied in shadow).

    The TTC governor's wall cap bounds the request's component along the looming ray, the travel direction at the
    sample's capture: the brake subtracts ray x (along - cap) from the request (and brings the command's component
    along the ray down to the cap at brake_slew). The ray tilts with the flight path, so on a path that rises (a
    drone climbing out of a floor dip, a hop over a ring) the brake also lowers the vertical request. On
    minus-fast6-r4-02 the stand-off cap (0.53-0.62 m/s along a ray tilted about 11 deg up) turned the pilot's +0.1 to
    +0.23 m/s into -0.3 to -0.5 m/s for 3 s (26.5-29.6 s) and drove the drone onto the garage floor.
    With this rule the brake's vertical part is floored, on both steps: after the brake the vertical request (and the
    command) is at least min(its value before the brake, 0) - max_added_sink. The brake may still reduce a climb to
    level, and raise the request when the ray points down; its horizontal part is unchanged. The guard's and the
    pilot's own sink are never changed.
    """
    max_added_sink: float = 0.

    def __post_init__(self):
        if not np.isfinite(self.max_added_sink) or self.max_added_sink < 0:
            raise ValueError('Use a finite max_added_sink >= 0')


def wall_pilot_configs(declaration, contract=None):
    """dict(turn_first=TurnFirstConfig, ceiling_guard=CeilingGuardConfig[, clearance_brake=ClearanceBrakeConfig]) from
    a wall-pilot declaration already parsed (and hash-checked) by the runner; refuses a rule version this code does
    not implement (WALL_PILOT_VERSIONS; the runner itself flies only WALL_PILOT_VERSION). Version 5 adds the clearance
    brake's sink floor. Turn-first's stopping model (the motor's measured braking) is declared per motor contract
    under 'turn_first_stopping'; a contract it does not list (or None) gets its 'default' entry. This module reads no
    files."""
    version = (declaration or {}).get('version')
    if version not in WALL_PILOT_VERSIONS:
        raise ValueError(f'The wall-pilot declaration is version {version}; the fast pilot implements versions '
                         f'{WALL_PILOT_VERSIONS}')
    stopping = declaration.get('turn_first_stopping')
    if not isinstance(stopping, dict) or not isinstance(stopping.get('default'), dict):
        raise ValueError('A wall-pilot declaration lists turn-first stopping models per motor contract and a default')
    model = stopping.get(contract) if isinstance(stopping.get(contract), dict) else stopping['default']
    out = dict(turn_first=TurnFirstConfig(**declaration['turn_first'], **model),
               ceiling_guard=CeilingGuardConfig(**declaration['ceiling_guard']))
    if version >= 5:
        out['clearance_brake'] = ClearanceBrakeConfig(**declaration['clearance_brake'])
    return out


# The vertical-guard declaration version whose rules this code implements (VerticalGuardConfig); runners refuse others.
VERTICAL_GUARD_VERSION = 4


@dataclass(frozen=True)
class VerticalGuardConfig:
    """Scale-free graded vertical guard of the TTC governor and the fast pilot. Off unless a runner passes it (the
    vertical-guard declaration in configs/obstacles; obstacle stack only); TTC policy only.

    It reads only the looming samples the governor already receives (below_fraction; ttc_lower, the time to contact
    with the surface fitted below the flight path; the alarm ttc) and the measured vertical speed: no height above
    ground, no metric distance. Descending means a measured vertical speed below -level_band, climbing above
    +level_band, level in between.
    1. Sink margin: the path's crossing TTC with the surface below it, max(alarm ttc, ttc_lower) of the latest sample
       that has a ttc_lower (aged by the time since its capture, kept memory_s after receipt), scales the pilot's own
       requested sink by clip((ttc - margin_zero_s) / (margin_full_s - margin_zero_s), 0, 1): all of it at
       margin_full_s or more, none at margin_zero_s or less. Both TTCs must be short: a surface below the path that
       the path itself does not head into (a long alarm, e.g. a ring or arch opening ahead of a descent) limits
       nothing. The factor moves at up to factor_down_rate (falling) and factor_up_rate (recovering) per second, so
       the request changes without a jump.
    Keep speed: while the guard withholds part of the pilot's sink, arrests a descent or withholds a climb because
       the drone descends, and while the pilot's support timers run (the requested sink is not achieved: contact with
       the terrain below), the descent-path shortfall is not fed: the horizontal request is not cut for a sink that
       was withheld or that the terrain prevents; the drone keeps its speed instead of sinking into the hill.
    2. Descent first: a below-path alarm (below_fraction >= the policy's terrain_fraction, with the path's crossing
       TTC, max(alarm ttc, ttc_lower) aged by odometry, under climb_on_s) received while descending arrests the
       descent: the vertical request is at least level (never a climb the pilot did not ask for) for arrest_hold_s,
       brought there at up to arrest_acceleration m/s^2. No alarm received while descending starts a climb, and while
       the drone descends no terrain climb is applied at all.
    3. Terrain climb only for rising ground: a climb needs `confirm` below-path alarms (below_fraction >=
       terrain_fraction, below-path TTC = ttc_lower aged by odometry under climb_on_s) received while level or
       climbing within confirm_window_s, and a climb that is not already running starts only if at least one of them
       is a path alarm: the path's crossing TTC, max(alarm ttc, ttc_lower) aged by odometry, is also under
       climb_on_s (the flight path itself heads into the surface below it, as in rules 1 and 2). A surface below the
       path that the path does not head into (a short ttc_lower with a long alarm: a crest or slope below the pilot's
       line) starts no climb; it only sustains a running one. Its rate is graded by urgency, vertical_up *
       clip((climb_on_s - ttc) / (climb_on_s - climb_full_s), 0, 1) on the below-path TTC, and bounded to gentle_climb
       (and to gentle_max_m above where the episode began) until rising ground is confirmed: `rising_confirm`
       below-path alarms received within rising_window_s while the drone already climbs faster than rising_min_rise
       and the guard's own climb binds: it exceeds every vertical request the pilot made itself in the last
       rising_window_s by at least rising_min_rise (the drone climbs that much faster because of the guard, and the
       ground keeps looming; a pilot that follows a ring up a hill climbs by its own request, and a brief dip of that
       request does not make the guard's climb the reason). Then the graded rate applies up to vertical_up, bounded by
       the policy's climb_max_m. The policy's climb hold and release are unchanged. The ceiling guard, when declared,
       still cuts climbs under overhead evidence.
       Version 4: the ground must keep looming throughout the rising window. No looming sample received in the
       rising_window_s before the confirming alarm may have seen the surface below the path farther than climb_on_s
       (a lower-surface TTC, aged by odometry, of climb_on_s or more, or no lower-surface TTC although the vertical
       windows had evidence). A floor close below a climbing path (its lower-window reading grows with the height and
       flickers with the attitude) or a structure the climb is already clearing gives such readings while the gentle
       climb runs: the climb was enough at that moment, so it is not escalated. Rising ground that the gentle climb
       does not clear keeps every reading under climb_on_s.
    Declaration version 4 (version 1 used ttc_lower alone for rules 1 and 2, counted any climb as rising-ground
    evidence and had no contact rule; version 2 started climbs on the lower window alone and compared the guard's
    climb only with the pilot's request of the same tick; version 3 escalated on alarms of the rising window although
    other samples of it saw the surface below farther than climb_on_s; all are kept and refused).
    """
    margin_full_s: float = 1.5
    margin_zero_s: float = .6
    memory_s: float = 1.
    factor_down_rate: float = 4.
    factor_up_rate: float = 1.
    level_band: float = .3
    arrest_hold_s: float = .5
    arrest_acceleration: float = 10.
    climb_on_s: float = 1.2
    climb_full_s: float = .6
    confirm: int = 2
    confirm_window_s: float = .3
    gentle_climb: float = 1.
    gentle_max_m: float = 1.
    rising_confirm: int = 2
    rising_window_s: float = .5
    rising_min_rise: float = .5

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Use finite positive vertical-guard parameters')
        for name in ('confirm', 'rising_confirm'):
            if int(getattr(self, name)) != getattr(self, name):
                raise ValueError(f'{name} counts samples')
        if not self.margin_zero_s < self.margin_full_s or not self.climb_full_s < self.climb_on_s:
            raise ValueError('Use margin_zero_s < margin_full_s and climb_full_s < climb_on_s')


def vertical_guard_config(declaration):
    """The VerticalGuardConfig of a vertical-guard declaration already parsed (and hash-checked) by the runner;
    refuses another rule version. This module reads no files."""
    if (declaration or {}).get('version') != VERTICAL_GUARD_VERSION:
        raise ValueError(f'The vertical-guard declaration is version {(declaration or {}).get("version")}; the fast '
                         f'pilot implements version {VERTICAL_GUARD_VERSION}')
    return VerticalGuardConfig(**declaration['vertical_guard'])


# The descent-view declaration version runners fly (DescentViewConfig and, from version 2, ContactSupportConfig);
# runners refuse others. DESCENT_VIEW_VERSIONS are the versions this code can rebuild (for replays and the surrogate).
DESCENT_VIEW_VERSION = 2
DESCENT_VIEW_VERSIONS = (1, 2)
# States whose horizontal request follows the speed schedule toward the ring (the view rule may restore it).
DESCENT_VIEW_BOOST_STATES = ('cue', 'below', 'below_weak')


@dataclass(frozen=True)
class DescentViewConfig:
    """View-keeping descent of the fast pilot. Off unless a runner passes it (the descent-view declaration in configs/pilot).

    The camera looks 30 degrees up with a 42 degree vertical half field of view, so its lower image edge lies about
    12 degrees below the body x axis. A drag-light quadrotor flies level at race speed with the nose within a few
    degrees of level, so in steady flight a descent steeper than about 12-14 degrees points below the image: no camera
    cue sees the ground the drone descends toward. On the Straw Bale downhill the default pilot slowed to half speed
    for every bottom-clipped ring (the brake pitched the nose 10-20 degrees up, which lifted the lower image edge and
    clipped rings that lay only 5-10 degrees down), then asked for 20-30 degree descents at 3 m/s and sank into the
    ~12 degree hillside. This rule keeps the flight path in view instead, and steepens beyond it only late:
    1. View bound: the pilot's own requested sink is bounded so that the flight path, made of the MEASURED horizontal
       velocity and the requested vertical speed, points at least margin_deg (cue_margin_deg while the ring is in
       view) inside the camera's lower image edge at the MEASURED attitude (the exact projection through the
       calibrated camera, roll included). The lowest in-view vertical speed per 1 m/s of horizontal speed is low-passed
       with attitude_time_constant: the pitch of a lagging motor oscillates, and a bound that followed it would drive
       the oscillation. A slower or nose-up (braking) drone may sink less; a hovering drone sinks at most free_sink.
       A climb is never changed.
    2. Keep speed: a ring clipped at the bottom edge does not lower the horizontal request (below_speed_fraction of
       the schedule instead of FastCueConfig.below_speed_fraction; 1 keeps the schedule), and the descent-path
       governor is not fed while the view bound withholds sink and never cuts below descent_min_scale.
    3. More speed, not less: while the bound withholds sink toward a ring ahead (states cue, below, below_weak), the
       horizontal request rises toward the speed schedule's (at most the declared speed), all the way once boost_sink
       is withheld: a faster drone descends further in view, and accelerating pitches the nose down.
    4. Throttle up: the requested sink grows at up to sink_acceleration m/s^2 (instead of the pilot's 5 m/s^2), so the
       motors are not asked for deep throttle cuts to start a descent.
    5. Steep late: a ring that stays clipped below (the pilot's unbroken bottom clip) for more than late_after_s lies
       more steeply below than the view allows; the margin then falls at late_rate_deg_s, down to late_max_deg below
       the lower image edge, so a steep leg is flown shallow first (in view, where a convex crest follows the upper
       checkpoint) and steep late (toward the lower checkpoint) instead of steep from the start.
    Reads only the measured attitude, velocity, the ring cue and the camera calibration: no height above ground, no
    course geometry. Declaration version 1.
    """
    margin_deg: float = 3.
    free_sink: float = .3
    below_speed_fraction: float = 1.
    descent_min_scale: float = .75
    boost_sink: float = .3
    sink_acceleration: float = 2.5
    attitude_time_constant: float = 1.
    cue_margin_deg: float = 3.
    late_after_s: float = .75
    late_rate_deg_s: float = 6.
    late_max_deg: float = 20.

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError('Use finite non-negative descent-view parameters')
        if not 0 < self.below_speed_fraction <= 1 or not 0 < self.descent_min_scale <= 1:
            raise ValueError('below_speed_fraction and descent_min_scale are fractions in (0, 1]')
        if not max(self.margin_deg, self.cue_margin_deg, self.late_max_deg) < 40 or not self.sink_acceleration > 0:
            raise ValueError('Use view margins below 40 degrees and a positive sink acceleration')


def descent_view_config(declaration):
    """The DescentViewConfig of a descent-view declaration already parsed (and hash-checked) by the runner; refuses a
    rule version this code does not implement (DESCENT_VIEW_VERSIONS; the runner itself flies only
    DESCENT_VIEW_VERSION). The view rule's values are the same in versions 1 and 2. This module reads no files."""
    if (declaration or {}).get('version') not in DESCENT_VIEW_VERSIONS:
        raise ValueError(f'The descent-view declaration is version {(declaration or {}).get("version")}; the fast '
                         f'pilot implements versions {DESCENT_VIEW_VERSIONS}')
    return DescentViewConfig(**declaration['descent_view'])


# Gravity of the contact rule's thrust model (the fast PD's value).
CONTACT_GRAVITY = 9.81


@dataclass(frozen=True)
class ContactSupportConfig:
    """Contact support of the fast pilot: descent-view declaration version 2 (off unless a runner passes it).

    Why: the view bound (DescentViewConfig) keeps the pilot's sink request above about -0.8 m/s below about 5 m/s, and
    both older support rules (FastCueConfig.support_*) need a command below -0.8 m/s, so a drone resting on a floor or a
    hill at low speed got no support climb (minus-fast6-r4-02 lay on the garage floor at 27.4-28.9 s with the sink
    request at -0.3..-0.5 m/s). This rule reads the force the ground exerts instead of the size of the request.

    Physics: the propellers push only along the drone's own up axis, with a thrust the measured curve gives for the
    issued throttle (g * thrust_twr * drive^thrust_exponent, drive = (processed + 1)/2, processed from the pad
    calibration, acting throttle_delay_s after it was issued), less the measured body-frame linear drag. Over the last
    window_s the measured change of the vertical velocity is compared with what that thrust, gravity and drag explain:
       unexplained = dvz/dt - (gain * mean(thrust * up_z) - g - mean(drag_z))      [m/s^2, upward positive]
    In free air it is near zero (live logs: median -0.03..-0.08 m/s^2, p10..p90 within about +-0.2 m/s^2 for a drive of
    0.5-0.7); a surface below can only push, so a sustained positive value is a ground reaction. Below drive_min or above
    drive_max the curve is not reliable (median residual +0.4..+3.7 m/s^2 at a drive of 0.3-0.4: motor idle thrust and
    spin-down), and a window with any such tick is not used.
    Contact support fires (a support climb of FastCueConfig.support_climb_s, as the older rules) when, continuously for
    hold_s: the command asks to sink (velocity command <= -sink_min), the measured vertical speed does not follow it
    (vz >= command + shortfall), the drone is not climbing away (vz <= rest_vz) and the unexplained upward specific
    force is at least unexplained_on. It is armed arm_after_s after the pilot's first tick (the runner holds the
    throttle for 1 s and ramps it until 3 s, so the issued throttle is not the game's before) and never while launching.
    Thrust gain: `gain` (starts at 1, the measured vehicle) follows the observed gain (dvz/dt + g + drag) / thrust with
    gain_time_constant (each step's difference clipped to +-gain_step), bounded to [gain_min, gain_max], only on armed,
    valid windows outside a contact episode and a support climb, and only when the window's unexplained force is at most
    gain_quiet (a negative one is never a ground reaction) or the drone rises (the window's lowest vz >= gain_rising: a
    surface below cannot hold up a drone that moves away from it; an uphill scrape can, which the clipped step bounds).
    On the logged flights the gain stays within 0.95-1.10 (the measured curve), except an uphill scrape
    (straw-brain6-02, 1.195), after which quiet windows bring it back at up to about 0.15 per second; the surrogate
    randomises the thrust by up to about +-20%, which the gain absorbs.
    Reads only the measured velocity and attitude, the throttle the motor issued and the pad calibration: no height above
    ground, no course geometry.
    """
    thrust_twr: float = 3.1378033647887618
    thrust_exponent: float = 1.9728633605611887
    body_drag_s_inv: tuple = (0.02745813096840542, 0., 0.3490431637001165)
    throttle_delay_s: float = .03
    window_s: float = .3
    drive_min: float = .4
    drive_max: float = .8
    unexplained_on: float = .8
    sink_min: float = .1
    shortfall: float = .1
    rest_vz: float = .5
    hold_s: float = .15
    arm_after_s: float = 3.2
    gain_time_constant: float = 1.
    gain_quiet: float = .4
    gain_rising: float = .05
    gain_step: float = .15
    gain_min: float = .75
    gain_max: float = 1.3

    def __post_init__(self):
        drag = tuple(float(v) for v in self.body_drag_s_inv)
        object.__setattr__(self, 'body_drag_s_inv', drag)
        values = [v for k, v in asdict(self).items() if k != 'body_drag_s_inv']
        if len(drag) != 3 or not np.isfinite(values+list(drag)).all() or min(drag) < 0:
            raise ValueError('Use finite contact-support parameters and three non-negative drag coefficients')
        if min(self.thrust_twr-1, self.thrust_exponent, self.window_s, self.hold_s, self.unexplained_on,
               self.gain_time_constant, self.gain_step, self.gain_min) <= 0 or min(
                   self.throttle_delay_s, self.arm_after_s, self.sink_min, self.gain_quiet) < 0:
            raise ValueError('Use a thrust-to-weight above one and positive windows, thresholds and gains')
        if not 0 <= self.drive_min < self.drive_max <= 1 or not self.gain_min <= 1 <= self.gain_max:
            raise ValueError('Use 0 <= drive_min < drive_max <= 1 and gain bounds around one')


def contact_support_config(declaration):
    """The ContactSupportConfig of a descent-view declaration (version 2 and later; None for version 1, which has no
    contact rule). This module reads no files."""
    descent_view_config(declaration)            # the same version check
    if declaration['version'] < 2:
        return None
    return ContactSupportConfig(**declaration['contact_support'])
# The motor-assist declaration version whose rule this code implements (MotorAssistConfig); runners fly only this one.
# MOTOR_ASSIST_VERSIONS are the versions this code can rebuild for replays: version 1 is kept for provenance
# (configs/pilot/motor_assist_v1.json) and refused by the runner; its entries get MOTOR_ASSIST_V1_FIELDS.
MOTOR_ASSIST_VERSION = 2
MOTOR_ASSIST_VERSIONS = (1, 2)
# The binding caps whose direction cap tracking can follow (MotorAssistConfig.cap_sources); 'approach' is version 2's.
MOTOR_ASSIST_SOURCES = ('request', 'governor', 'turn_first', 'stopping', 'approach')
# Pilot states whose own vertical request the sag compensation leaves alone (they own the vertical request).
MOTOR_ASSIST_SAG_EXCLUDED = ('launch', 'support_climb')
# Version 2's wall-ahead condition: states in which the checkpoint's bearing is unknown (its marker is lost).
MOTOR_ASSIST_UNKNOWN_STATES = ('coast', 'search')
# Version 2: sources whose cap tracking shares another source's extra reduction (one stopping model, two conditions).
MOTOR_ASSIST_EXTRA_KEY = {'approach': 'stopping'}
# The version-2 fields as version 1 flew them (no slew bound, the stopping source at any wall sample, no floor): a
# version-1 declaration entry is rebuilt with these values (replays only; the runner refuses version 1).
MOTOR_ASSIST_V1_FIELDS = dict(slew=0., stop_gate='any', floor_speed=0., stop_memory_s=0., wall_ahead_deg=50.,
                              wall_ahead_standoff=True, standoff_tracking=True, approach_climb_max=float('inf'))


@dataclass(frozen=True)
class MotorAssistConfig:
    """Pilot-level help for a lagging brain motor contract (fast_velocity_brain_v1). Off unless a runner passes it (the
    motor-assist declaration in configs/pilot, declared per motor contract; the fast PD contract has no
    entry and flies unchanged).

    The fast brains follow the pilot's velocity request late and weakly: fast-brain-10b flew 5.1-5.2 m/s under 3.6 m/s
    governor caps into the Minus Two hairpin wall (minus-brain10b-r4-02), and fast-brain-09b sank 0.7 m to the floor while
    it accelerated out of a turn-first stop toward a 4.9 m/s request with a level vertical request
    (minus-brain09b-r4-01). The surrogate reproduces both from the logged requests. Both rules act on the pilot's final
    request (after every other rule and the command slews): the motor receives the assisted request, while the pilot's
    own request stays its internal state (FastRaceCue.pilot_command) and is what its rules read on the next tick. Only
    the support rule and the descent-path shortfall compare the measured vertical speed with the vertical request the
    motor received (a sink the assist withheld is not the vehicle failing to descend).
    1. Cap tracking: for each binding cap of a declared source (cap_sources), with a horizontal unit direction h and a
       bound c on the request's speed along h, the measured horizontal speed along h beyond c + cap_deadband lowers that
       bound by cap_gain x the excess beyond the deadband (at most cap_max), so a brain that lags converges to the cap
       instead of flying through it. The extra reduction of each source moves at up to cap_rise (growing) and cap_fall
       (shrinking) m/s^2 and acts only while its source binds. Sources:
       - 'request': the pilot's final horizontal request itself (h its direction, c its magnitude) in the pilot states
         request_states: the speed schedule, the side and bottom-edge speeds and every cap the pilot already applied;
       - 'governor': the looming governor's cap along the horizontal part of its ray, while it bounds the request;
       - 'turn_first': a turn-first episode or side guard: no speed toward the wall (c = 0), and its creep bound along
         the request;
       - 'stopping': the motor's stopping model on the governor's latest wall sample: once stop_confirm wall samples
         (not below-path terrain) with a TTC under stop_ttc_s arrived within stop_window_s, the newest of them being the
         newest looming sample with evidence, the speed along that sample's horizontal ray is bounded (directly, not only
         beyond the deadband) by the speed from which the motor stops within the remaining distance,
         v*stop_latency_s + v^2/(2*stop_deceleration) + stop_margin_m. The governor's TTC thresholds suit the fast PD;
         a brain that follows 0.3 s late must start braking earlier. A clear sample (a gate arch flown through) ends it.
       The request never reverses along its own direction ('request' stops at 0) and points away from a wall (the
       other sources) by at most cap_reverse m/s: a hard reversal pitches a brain back so far that it climbs.
    2. Sag compensation: the vertical request rises by
       - sag_gain x the shortfall beyond sag_deadband while the measured vertical speed lies more than sag_deadband
         below the pilot's final vertical request (the brain sinks faster than asked), plus
       - sag_lead_gain x the horizontal velocity change the motor is asked for beyond sag_lead_deadband (|assisted
         horizontal request - measured horizontal velocity|), while the measured horizontal speed is at most
         sag_lead_max_speed (0: at any speed): a brain pitches hard to accelerate from low speed and its sink starts
         before any feedback sees it (live logs: at < 2.5 m/s the vertical shortfall grows 0.2-0.65 m/s per 1 m/s of
         lead; brain-09b tilted 57 deg out of the turn-first stop),
       at most sag_max, moving at up to sag_rise (growing) and sag_fall (shrinking) m/s^2; nothing while the drone
       already climbs more than sag_climb_margin faster than the pilot's request, while launching or in a support
       climb, and never above the pilot's vertical_up nor the ceiling guard's vertical bound during an overhead hold.
    Reads only the measured velocity, the pilot's own rule states and the governor's looming samples: no course
    geometry, route or per-course value.

    Version 2 (declaration version 2; the round-4b review of version 1: its assisted request stepped by up to 4.5 m/s
    per tick, and its stopping source asked brains for 0.01-0.4 m/s at the Minus Two first arch, before pillar A and at
    the 90-degree arch, where the pilot flies on through a gate arch):
    3. Slew: the assist's change of the request (assisted - the pilot's own) moves per tick by at most slew x dt (slew:
       the clearance brake_slew), as a horizontal vector and vertically each, as the clearance brake bounds its own
       steps, on onsets and on releases; the motor's request then changes per tick by at most the pilot's own change
       plus slew x dt in each. slew 0: unbounded (version 1).
    4. Wall ahead (stop_gate 'wall_ahead'; 'any' is version 1): the stopping source acts only while a wall-ahead
       condition holds, the conditions turn-first reads: the checkpoint beyond the view to the side (pilot state
       'side'), its bearing wall_ahead_deg or more off the heading (a marker in view or clamped at the bottom/top edge),
       its marker lost ('coast', 'search'), a turn-first episode or its side guard, or (wall_ahead_standoff) the looming
       governor's stand-off. There the next checkpoint does not lie through the surface ahead. The stop then keeps the
       latest wall sample's dead-reckoned distance for stop_memory_s after the last confirmation (a wall at arm's length
       stops looming: the live samples read no evidence within ~2 m) unless a newer looming sample with evidence is not
       a wall sample (a clear view ends it).
    5. Approach: outside a wall-ahead condition the confirmed wall samples of the stopping source feed the 'approach'
       source instead: the same stopping model on the same distance, its bound never below floor_speed, and none while
       the pilot's own vertical request before the looming governor and the vertical guard exceeds approach_climb_max
       (a surface looming while the pilot follows its checkpoint up is rising ground, the vertical guard's). The assist
       thus plans a lagging brain down toward a surface ahead that may be a gate arch it will fly through, to a speed
       from which it can still stop in the room behind that arch, and never plans it to a stop there. Cap tracking of
       the two bounds shares one extra reduction (MOTOR_ASSIST_EXTRA_KEY), so a wall-ahead condition that starts
       mid-approach (the arch passed, the next marker to the side or lost) carries it over.
    6. standoff_tracking False: no cap tracking of the governor's cap while the governor holds a stand-off (it holds the
       drone at or below its stand-off speed by itself; tracking beyond that asks a slow brain to back away).
    """
    cap_sources: tuple = ('request', 'governor', 'turn_first', 'stopping', 'approach')
    request_states: tuple = ('cue', 'below', 'below_weak', 'side')
    cap_deadband: float = .3
    cap_gain: float = 2.
    cap_max: float = 3.
    cap_reverse: float = 1.
    cap_rise: float = 8.
    cap_fall: float = 8.
    stop_ttc_s: float = 1.3
    stop_confirm: int = 2
    stop_window_s: float = .25
    stop_latency_s: float = .3
    stop_deceleration: float = 3.5
    stop_margin_m: float = .5
    sag_deadband: float = .3
    sag_gain: float = 1.
    sag_lead_gain: float = .3
    sag_lead_deadband: float = .5
    sag_lead_max_speed: float = 2.5
    sag_climb_margin: float = .3
    sag_max: float = 1.
    sag_rise: float = 5.
    sag_fall: float = 2.
    slew: float = 15.
    stop_gate: str = 'wall_ahead'
    wall_ahead_deg: float = 50.
    wall_ahead_standoff: bool = False
    standoff_tracking: bool = False
    stop_memory_s: float = 1.
    floor_speed: float = 2.5
    approach_climb_max: float = .3
    version: int = MOTOR_ASSIST_VERSION

    def __post_init__(self):
        values = {k: v for k, v in asdict(self).items() if k not in ('cap_sources', 'request_states', 'stop_gate',
                                                                    'approach_climb_max')}
        if not np.isfinite(list(values.values())).all() or min(values.values()) < 0:
            raise ValueError('Use finite non-negative motor-assist parameters')
        if np.isnan(self.approach_climb_max):
            raise ValueError('approach_climb_max is a vertical speed (inf: no bound)')
        if self.version not in MOTOR_ASSIST_VERSIONS:
            raise ValueError(f'The fast pilot implements motor-assist versions {MOTOR_ASSIST_VERSIONS}')
        if self.stop_gate not in ('any', 'wall_ahead'):
            raise ValueError("stop_gate is 'any' (version 1) or 'wall_ahead'")
        if not 0 < self.wall_ahead_deg < 180:
            raise ValueError('Use 0 < wall_ahead_deg < 180 degrees')
        if self.version == 1 and ('approach' in self.cap_sources or any(
                getattr(self, k) != v for k, v in MOTOR_ASSIST_V1_FIELDS.items())):
            raise ValueError('A version-1 motor assist has the version-1 fields (MOTOR_ASSIST_V1_FIELDS) and no '
                             "'approach' source")
        for name in ('cap_gain', 'cap_rise', 'cap_fall', 'sag_rise', 'sag_fall', 'stop_ttc_s', 'stop_window_s',
                     'stop_deceleration'):
            if not getattr(self, name) > 0:
                raise ValueError(f'{name} must be positive')
        if not set(self.cap_sources) <= set(MOTOR_ASSIST_SOURCES) or len(set(self.cap_sources)) != len(self.cap_sources):
            raise ValueError(f'cap_sources are distinct names from {MOTOR_ASSIST_SOURCES}')
        if not all(isinstance(s, str) for s in self.request_states):
            raise ValueError('request_states are pilot state names')
        if int(self.stop_confirm) != self.stop_confirm or self.stop_confirm < 1:
            raise ValueError('stop_confirm counts samples')


def motor_assist_for_contract(declaration, contract):
    """The `MotorAssistConfig` that a motor-assist declaration already parsed (and hash-checked) by the runner assigns
    to a motor contract ('contracts': {contract: parameters or None}), or None when the contract has none (the fast PD);
    refuses a rule version this code does not implement (MOTOR_ASSIST_VERSIONS; the runner itself flies only
    MOTOR_ASSIST_VERSION). A version-1 entry is rebuilt as version 1 flew (MOTOR_ASSIST_V1_FIELDS). This module reads
    no files."""
    version = (declaration or {}).get('version')
    if version not in MOTOR_ASSIST_VERSIONS:
        raise ValueError(f'The motor-assist declaration is version {version}; the fast pilot implements versions '
                         f'{MOTOR_ASSIST_VERSIONS}')
    contracts = declaration.get('contracts')
    if not isinstance(contracts, dict):
        raise ValueError('A motor-assist declaration lists its motor contracts')
    entry = contracts.get(contract)
    if entry is None:
        return None
    entry = dict(entry)
    if 'version' in entry:
        raise ValueError('The rule version is the declaration\'s, not a contract entry\'s')
    if version == 1:
        if set(entry) & set(MOTOR_ASSIST_V1_FIELDS):
            raise ValueError('A version-1 declaration has no version-2 fields')
        entry.update(MOTOR_ASSIST_V1_FIELDS)
        entry.setdefault('cap_sources', MOTOR_ASSIST_SOURCES[:4])
    for key in ('cap_sources', 'request_states'):
        if key in entry:
            entry[key] = tuple(entry[key])
    return MotorAssistConfig(**entry, version=version)


class TtcClearanceGovernor:
    """Graded speed cap along the looming ray and a terrain climb from TTC samples.

    Same interface as `ClearanceGovernor`. Pure and causal: each sample carries
    its capture time, capture position and ray; it acts once, when it first
    reaches `limits`, with the TTC aged to that moment by odometry.
    `ceiling` (a `CeilingGuardConfig`, off by default) adds the ceiling guard;
    `vertical_cap` is then the bound on the whole vertical request (None: none).
    `vertical` (a `VerticalGuardConfig`, off by default) adds the graded vertical guard: its terrain climb replaces
    the policy's, and `sink_factor` / `arrest` tell the pilot how much of its own sink to keep and whether to hold
    at least level (1. / False without it).
    """

    def __init__(self, config=None, ceiling=None, vertical=None):
        self.config = config or TtcClearanceConfig()
        if ceiling is not None and not isinstance(ceiling, CeilingGuardConfig):
            raise ValueError('Pass a CeilingGuardConfig (or None) for the ceiling guard')
        if vertical is not None and not isinstance(vertical, VerticalGuardConfig):
            raise ValueError('Pass a VerticalGuardConfig (or None) for the vertical guard')
        self.ceiling = ceiling
        self.vertical = vertical
        # Vertical guard state (unused without it)
        self.sink_factor = 1.
        self.arrest = False
        self.arrest_until = -np.inf
        self.lower_evidence = None      # (capture time, ttc_lower, receipt time) of the latest known below-path TTC
        self.alarms = []                # (receipt, measured vz, guard climb binding, path alarm) of below-path alarms,
                                        # not descending
        self.escalated = False          # rising ground confirmed in this climb episode
        self.clear_below_at = -np.inf   # v4: receipt of the latest sample that saw no surface looming below the path
                                        # within climb_on_s
        self.pilot_vertical = -np.inf   # the pilot's own vertical request this tick (set by the pilot)
        self.pilot_history = []         # (time, pilot's own vertical request) of the last rising_window_s
        if vertical is not None:
            self.vertical_counts = dict(sink_limited_samples=0, descent_first=0, arrest_engagements=0,
                                        unconfirmed_alarms=0, no_path_onsets=0, gentle_climbs=0, escalations=0,
                                        topped=0, clear_below_samples=0, clear_below_blocks=0)
        self.samples = []
        self.last_time = self.last_evidence = self.first_input = None
        # The latest wall sample (not braked for as terrain, aged TTC < hold_ttc_s): its capture-time reach along its
        # ray and capture position (wall_distance). Read by the pilot's turn-first rule only; the governor ignores it.
        self.wall_evidence = None
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
        self.overhead_times = []        # (receipt time, positive) of recent overhead samples
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

    def wall_distance(self, position):
        """(remaining distance along its ray to the latest wall sample, that ray) at `position`: the capture-time
        reach minus the odometry travelled along the ray since, at least 0 (reached or passed); (None, None) before
        any wall sample."""
        e = self.wall_evidence
        if e is None:
            return None, None
        left = e['reach']-float((np.asarray(position, float)-e['position']) @ e['ray'])
        return max(0., left), e['ray']

    def _climb_bound(self):
        """Height above the episode base at which the terrain climb is topped (the vertical guard's gentle bound
        until rising ground is confirmed)."""
        v = self.vertical
        return self.config.climb_max_m if v is None or self.escalated else v.gentle_max_m

    def _guard_request(self, s, lower, ttc, rise, now, vertical_up):
        """The vertical guard's climb request for one terrain sample (VerticalGuardConfig rules 2 and 3)."""
        v, c = self.vertical, self.config
        below_path = s['below'] is not None and s['below'] >= c.terrain_fraction
        if rise < -v.level_band:
            # descent first: the ground looms because the drone descends (the path itself heads into the surface
            # below it: both the alarm and the lower-surface TTC are short); stop the descent, start no climb
            path_ttc = ttc if lower is None else max(ttc, lower)
            if below_path and path_ttc < v.climb_on_s:
                if now > self.arrest_until:
                    self.vertical_counts['arrest_engagements'] += 1
                self.vertical_counts['descent_first'] += 1
                self.arrest_until = now+v.arrest_hold_s
            return 0.
        below_ttc = lower if lower is not None else ttc
        graded = vertical_up*float(np.clip((v.climb_on_s-below_ttc)/(v.climb_on_s-v.climb_full_s), 0, 1))
        if graded <= 0:
            return 0.
        if not below_path:
            # no below-path evidence (the ceiling guard's explained alarms, or none): sustains a climb, gently
            return min(graded, v.gentle_climb) if self.climb > 0 else 0.
        # a path alarm: the flight path itself heads into the surface below it (both TTCs short, as in rules 1 and 2)
        path = (ttc if lower is None else max(ttc, lower)) < v.climb_on_s
        # rising-ground evidence counts only while the guard's own climb binds: it exceeds every vertical request the
        # pilot made itself in the last rising_window_s by rising_min_rise (the drone climbs because of the guard, not
        # because the pilot follows a ring up a hill)
        pilot = max((p for _, p in self.pilot_history), default=self.pilot_vertical)
        binding = self.climb > 0 and self.climb-pilot >= v.rising_min_rise
        horizon = max(v.confirm_window_s, v.rising_window_s)
        self.alarms = [a for a in self.alarms if s['received']-a[0] <= horizon]+[(s['received'], rise, binding, path)]
        window = [a for a in self.alarms if s['received']-a[0] <= v.confirm_window_s]
        # a climb starts only with a path alarm among the confirming alarms; the lower window alone sustains one
        confirmed = len(window) >= v.confirm and (self.climb > 0 or any(a[3] for a in window))
        rising = sum(s['received']-t <= v.rising_window_s and r > v.rising_min_rise and b
                     for t, r, b, _ in self.alarms) >= v.rising_confirm
        if confirmed and rising and not self.escalated:
            if s['received']-self.clear_below_at <= v.rising_window_s:
                # v4: a sample of the rising window saw the surface below the path farther than climb_on_s: the climb
                # was enough at that moment, so the ground did not keep looming
                self.vertical_counts['clear_below_blocks'] += 1
            else:
                self.escalated = True           # the ground keeps looming although the drone climbs: rising ground
                self.vertical_counts['escalations'] += 1
        if self.escalated:
            return graded
        if confirmed:
            if self.climb == 0:
                self.vertical_counts['gentle_climbs'] += 1
            return min(graded, v.gentle_climb)
        # enough alarms but none on the path (the lower window alone): no climb starts
        self.vertical_counts['unconfirmed_alarms' if len(window) < v.confirm else 'no_path_onsets'] += 1
        return 0.

    def limits(self, position, velocity, now, dt, vertical_up):
        """Return (cap or None, ray or None, climb request) for the current tick."""
        c = self.config
        g = self.ceiling
        v = self.vertical
        position, velocity = np.asarray(position, float), np.asarray(velocity, float)
        keep = max(c.memory_s, c.confirm_window_s)
        self.samples = [s for s in self.samples if now-s['received'] <= keep]
        climbing = self.climb > 0
        height = float(position[2]) if position.size > 2 else 0.
        rise = float(velocity[2]) if velocity.size > 2 else 0.
        if not climbing and now-self.terrain_at > c.climb_hold_s:
            self.climb_base = None                  # a new climb episode may start from the present height
            self.strong_height = None
            self.escalated = False
        topped = self.climb_base is not None and height-self.climb_base >= (
            c.climb_max_m if v is None else self._climb_bound())
        overhead = g is not None and now <= self.overhead_until
        if v is not None:
            # the pilot's own vertical requests of the last rising_window_s (the binding test of rule 3)
            self.pilot_history = [(t, p) for t, p in self.pilot_history
                                  if now-t <= v.rising_window_s]+[(now, float(self.pilot_vertical))]
        for s in self.samples:
            if not s['new']:
                continue
            s['new'] = False
            if v is not None and s['ttc_lower'] is not None:
                # the path's own crossing with the surface below it: both the alarm and the lower TTC
                crossing = max(s['ttc'], s['ttc_lower'])
                self.lower_evidence = (s['time'], crossing, s['received'])
                self.vertical_counts['sink_limited_samples'] += int(crossing-(now-s['time']) < v.margin_full_s)
            if now-s['received'] > c.memory_s:
                continue
            closing = float(velocity @ s['ray'])
            ttc = self._aged(s, s['reach'], position, closing)
            if v is not None and (s['below'] is not None if s['ttc_lower'] is None else self._aged(
                    s, s['ttc_lower']*s['reach']/max(s['ttc'], 1e-3), position, closing) >= v.climb_on_s):
                # vertical guard v4: the surface below the path does not loom within climb_on_s (a lower-surface TTC at
                # least climb_on_s, or none although the vertical windows had evidence): no rising ground is confirmed
                # for rising_window_s after its receipt
                self.vertical_counts['clear_below_samples'] += 1
                self.clear_below_at = s['received']
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
                # the alarm lies above a rising path (or is not explained by the surface below it); positive when
                # something shows it is not the surface below (expansion above the path, or a farther lower surface)
                positive = s['below'] is not None or (s['ttc_lower'] is not None
                                                      and s['ttc_lower'] > g.lower_ratio*max(s['ttc'], 1e-3))
                self.counts['overhead_samples'] += 1
                self.overhead_times = [(t, p) for t, p in self.overhead_times
                                       if s['received']-t <= c.confirm_window_s]+[(s['received'], positive)]
                if (len(self.overhead_times) >= g.overhead_confirm
                        and sum(p for _, p in self.overhead_times) >= g.overhead_positive):
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
                confirmed = votes >= c.terrain_confirm or climbing
                if v is not None:
                    # vertical guard: descent first, confirmed and graded climbs on the below-path TTC
                    request = self._guard_request(s, lower, ttc, rise, now, vertical_up)
                    confirmed = True                    # the guard confirms on its own
                    topped = self.climb_base is not None and height-self.climb_base >= self._climb_bound()
                if weak and request > 0:
                    self.counts['weak_climb_samples'] += 1
                    request = min(request, g.weak_climb)
                    if self.strong_height is not None and height-self.strong_height >= g.weak_climb_max_m:
                        request = 0.
                if overhead and request > 0:
                    self.counts['suppressed_climb_samples'] += 1
                    request = 0.
                if request > 0 and confirmed:
                    self.terrain_at = now
                    self.climb_base = height if self.climb_base is None else self.climb_base
                if v is not None and request > 0 and topped:
                    self.vertical_counts['topped'] += 1
                if request > 0 and confirmed and not topped:
                    if self.climb == 0:
                        self.counts['climb_engagements'] += 1
                    if not weak or request >= self.climb:
                        self.climb_hold_until = now+c.climb_hold_s
                    self.climb = max(self.climb, request)
                    if g is not None and not weak:
                        self.strong_height = height
            # graded slow-down; during an overhead hold a surface below the path is braked for like a wall
            brake_terrain = terrain and not overhead
            if not brake_terrain and ttc < c.hold_ttc_s:
                self.wall_evidence = dict(time=s['time'], reach=s['reach'], position=s['position'], ray=s['ray'])
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
        if v is not None:
            # sink margin: the latest known below-path TTC, aged since its capture, ramps the allowed sink
            target = 1.
            if self.lower_evidence is not None and now-self.lower_evidence[2] <= v.memory_s:
                aged = self.lower_evidence[1]-(now-self.lower_evidence[0])
                target = float(np.clip((aged-v.margin_zero_s)/(v.margin_full_s-v.margin_zero_s), 0, 1))
            rate = v.factor_down_rate if target < self.sink_factor else v.factor_up_rate
            self.sink_factor = float(self.sink_factor+np.clip(target-self.sink_factor, -rate*dt, rate*dt))
            self.arrest = now <= self.arrest_until
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
                 ceiling_guard=None, wall_apply=True, vertical_guard=None, vertical_apply=True, descent_view=None,
                 contact_support=None, clearance_brake=None, motor_assist=None):
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
        # what engaged each episode: the checkpoint trigger (side, bearing, coast) and the wall condition (standoff,
        # stopping: a recent wall brake within the stopping distance)
        self.turn_first_triggers = dict(side=0, bearing=0, coast=0, standoff=0, stopping=0)
        self.turn_first_time = 0.
        self.side_guard_time = 0.       # seconds the side guard removed speed toward a wall outside an episode
        self.side_guard_active = False
        # The clearance brake's sink floor (wall-pilot version 5; off unless declared, applied only with wall_apply):
        # see ClearanceBrakeConfig. brake_added_sink is measured on every tick whether or not the rule is declared (a
        # diagnostic that changes nothing): the sink the clearance brake added this tick below min(the request before
        # the brake, 0), on the request and on the command (m/s, the larger of the two).
        if clearance_brake is not None and not isinstance(clearance_brake, ClearanceBrakeConfig):
            raise ValueError('Pass a ClearanceBrakeConfig (or None) for the clearance brake\'s sink floor')
        self.clearance_brake = clearance_brake
        self.brake_added_sink = 0.
        self.brake_sink_withheld = 0.   # the sink the floor withheld this tick (0 in shadow or without the rule)
        self.brake_sink_left = 0.       # the sink the brake left after the floor this tick
        self.brake_reference = float('nan')    # the vertical request before the brake this tick
        self.brake_ray = np.full(3, np.nan)    # the clearance cap's looming ray this tick (NaN without a cap)
        self.brake_sink_time = dict(added=0., withheld=0., left=0.)
        self.brake_sink_max = dict(added=0., withheld=0., left=0.)
        # Vertical guard (off unless declared; obstacle stack only): see VerticalGuardConfig. vertical_apply False
        # computes and logs it without applying it: the flown governor then has no vertical guard and a guarded copy
        # fed the same samples reports the vertical request it would make (vertical_target).
        if vertical_guard is not None and not isinstance(vertical_guard, VerticalGuardConfig):
            raise ValueError('Pass a VerticalGuardConfig (or None) for the vertical guard')
        if vertical_guard is not None and not isinstance(self.clearance_config, TtcClearanceConfig):
            raise ValueError('The vertical guard is part of the TTC clearance policy')
        self.vertical_guard, self.vertical_apply = vertical_guard, bool(vertical_apply)
        self.pilot_vertical = float('nan')     # the pilot's own vertical request this tick (before governor and guard)
        self.vertical_target = float('nan')    # the vertical request the guard makes (applied or, in shadow, intended)
        self.vertical_limiting = False         # the applied guard withheld part of the pilot's sink this tick
        self.vertical_time = dict(limiting=0., arrest=0., climb=0.)
        # View-keeping descent (off unless declared): see DescentViewConfig.
        if descent_view is not None and not isinstance(descent_view, DescentViewConfig):
            raise ValueError('Pass a DescentViewConfig (or None) for the view-keeping descent')
        self.descent_view = descent_view
        self.schedule_speed = float('nan')     # the speed schedule's horizontal speed toward the ring this tick
        self.view_bound = float('nan')         # the largest sink keeping the path in view this tick
        self.view_withheld = 0.                # sink the view bound withheld from the pilot's own request this tick
        self.view_boost = False                # the horizontal request was raised to the speed schedule this tick
        self.view_time = dict(limiting=0., boost=0.)
        self.view_withheld_integral = 0.       # metres of sink withheld (integral of view_withheld)
        self.view_slope = None                 # low-passed lowest vertical speed per 1 m/s of horizontal speed in view
        # Contact support (off unless declared; descent-view declaration version 2): see ContactSupportConfig.
        if contact_support is not None and not isinstance(contact_support, ContactSupportConfig):
            raise ValueError('Pass a ContactSupportConfig (or None) for the contact support')
        if contact_support is not None and self.calibration is None:
            raise ValueError('Contact support needs the pad throttle calibration (the issued throttle\'s thrust)')
        self.contact_support = contact_support
        self.contact_samples = []              # (time, measured vz, nominal thrust along world z, drag z, drive)
        self.contact_throttle = []             # (issue time, issued throttle) of recent motor commands
        self.contact_gain = 1.                 # thrust gain against the declared curve (learnt in free air)
        self.contact_first = None              # the pilot's first tick (arming)
        self.contact_since = None              # start of the current run of contact conditions
        self.contact_unexplained = float('nan')    # unexplained upward specific force of the window (m/s^2)
        self.contact_fired = False             # the rule started a support climb this tick
        self.contact_time = dict(valid=0., suspected=0., gain_updates=0.)
        self.contact_onsets = 0
        # Motor assist (off unless declared for the motor contract): see MotorAssistConfig. With it, velocity_command
        # after update() is the assisted request the motor receives; pilot_command keeps the pilot's own request, which
        # is restored as the pilot's state at the start of the next update().
        if motor_assist is not None and not isinstance(motor_assist, MotorAssistConfig):
            raise ValueError('Pass a MotorAssistConfig (or None) for the motor assist')
        self.motor_assist = motor_assist
        self.pilot_command = None
        self.assist_extra = {name: 0. for name in MOTOR_ASSIST_SOURCES}   # current extra reduction per source (m/s)
        self.assist_sag = 0.                   # current climb added by the sag compensation (m/s)
        self.assist_horizontal = 0.            # horizontal request removed this tick (m/s)
        self.assist_vertical = 0.              # vertical request added this tick (m/s)
        self.assist_source = ''                # the source whose cap tracking removed the most this tick
        self.assist_sources = []               # binding caps of this tick: (source, horizontal unit ray, bound)
        self.assist_time = {name: 0. for name in MOTOR_ASSIST_SOURCES+('sag',)}
        self.assist_removed = 0.               # metres of horizontal request travel removed (integral)
        self.assist_added = 0.                 # metres of climb requested by the sag compensation (integral)
        # Version 2 (see MotorAssistConfig): the slewed change of the request (assisted - own), the wall-ahead condition
        # of this tick, and the stopping source's memory (receipt times of the newest wall sample, of the latest
        # confirmation and of the newest looming sample with evidence that is no wall sample).
        self.assist_delta = np.zeros(3)
        self.assist_wall_ahead = False
        self.assist_plan = float('nan')        # the stopping model's bound this tick (approach or stopping; NaN: none)
        self.assist_seen_at = -np.inf
        self.assist_wall_at = self.assist_confirmed_at = self.assist_clear_at = -np.inf
        self.assist_v2_time = dict(wall_ahead=0., slew_limited=0., approach=0., stopping=0.)

    def _ingest_clearance(self, clearance, velocity, yaw, now):
        c = self.clearance_config
        stamp = clearance.get('time')
        if stamp is None or not np.isfinite(stamp):
            raise ValueError('A clearance sample needs its capture time')
        if self.clearance is None:
            guard, vertical = self.ceiling_guard, self.vertical_guard
            flown = dict(ceiling=guard if self.wall_apply else None,
                         vertical=vertical if self.vertical_apply else None)
            if flown['ceiling'] is not None or flown['vertical'] is not None:
                self.clearance = TtcClearanceGovernor(c, **flown)
            else:
                self.clearance = clearance_governor(c)
            if (guard is not None and not self.wall_apply) or (vertical is not None and not self.vertical_apply):
                # shadow: a copy with every declared rule, fed the same samples, reports what the rules would do
                self.clearance_shadow = TtcClearanceGovernor(c, ceiling=guard, vertical=vertical)
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
        self.schedule_speed = float('nan')
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
        self.schedule_speed = float(speed)
        slope = d[2]/max(np.linalg.norm(d[:2]), 1e-6)
        if self.below:
            # The target lies below the lower image edge, i.e. at least as steep
            # as the clamped edge ray. Descend along a slope only slightly
            # steeper than that bound, rather than diving: racing lines often
            # follow terrain down a hill, and forward pitch soon brings the
            # marker back into view. Weight the response by the evidence.
            w = self.below_weight
            below_fraction = (c.below_speed_fraction if self.descent_view is None
                              else self.descent_view.below_speed_fraction)
            horizontal = speed*(1-w)+min(speed, max(c.edge_speed, below_fraction*speed))*w
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

    def _near_wall(self, position, velocity, now):
        """The turn-first wall condition (see TurnFirstConfig): 'standoff' (a clearance stand-off, the wall within
        the stopping distance from the stand-off speed or the closing speed), 'stopping' (a recent wall brake at <=
        max_speed with the latest wall sample within the stopping distance at the closing speed), or None."""
        gov, tf = self.clearance, self.turn_first
        if gov is None or gov.cap_ray is None or float(np.linalg.norm(velocity[:2])) > tf.max_speed:
            return None
        # the latest wall sample, dead-reckoned; without one (the stopping-distance policy keeps none) it counts as
        # reached
        distance, ray = gov.wall_distance(position) if hasattr(gov, 'wall_distance') else (None, None)
        if distance is None:
            distance, ray = 0., gov.cap_ray
        closing = float(np.asarray(velocity, float) @ np.asarray(ray, float))
        if isinstance(gov, TtcClearanceGovernor):
            standoff = now <= gov.standoff_until and gov.cap is not None
        else:
            standoff = bool(gov.sustained and gov.cap is not None and gov.cap < gov.config.standoff_speed)
        if standoff and distance <= tf.stopping_distance(max(closing, gov.config.standoff_speed)):
            return 'standoff'
        if now-self.wall_brake_at > tf.brake_recent_s:
            return None
        return 'stopping' if distance <= tf.stopping_distance(closing) else None

    def _turn_first(self, state, desired, position, velocity, yaw, now):
        """Turn before translating at a wall (TurnFirstConfig): (the limited horizontal request, the horizontal unit
        wall ray whose speed is removed at the brake slew, the bound on the horizontal speed that is also reached at
        the brake slew) while an episode or the side guard (no bound: inf) is active (also computed in shadow), else
        None. Updates the episode state and its counts."""
        tf = self.turn_first
        off = self._bearing_off_deg(yaw)
        self.side_guard_active = False
        if self.turn_first_since is not None:
            end = ('handoff' if state in TURN_FIRST_HANDOFF_STATES or self.launching else
                   'aligned' if state in TURN_FIRST_BEARING_STATES and off is not None and off <= tf.release_deg else
                   'timeout' if now-self.turn_first_since >= tf.max_s else None)
            if end is not None:
                self.turn_first_counts[end] += 1
                self.turn_first_since = self.turn_first_ray = None
                if end == 'timeout':
                    self.turn_first_block_until = now+tf.rearm_s
        trigger = ('side' if state == 'side' else
                   'bearing' if state in TURN_FIRST_BEARING_STATES and off is not None and off >= tf.engage_deg else
                   'coast' if state in TURN_FIRST_UNKNOWN_STATES else None)
        near = None
        if (self.turn_first_since is None and not self.launching and trigger is not None
                and state not in TURN_FIRST_HANDOFF_STATES):
            near = self._near_wall(position, velocity, now)
        ray = None if self.clearance is None or self.clearance.cap_ray is None else \
            np.asarray(self.clearance.cap_ray, float)[:2]
        if ray is None or np.linalg.norm(ray) <= 1e-6:
            ray = None
        else:
            ray = ray/np.linalg.norm(ray)
        if near is not None and ray is not None and now >= self.turn_first_block_until:
            self.turn_first_since, self.turn_first_ray = now, ray
            self.turn_first_counts['episodes'] += 1
            self.turn_first_triggers[trigger] += 1
            self.turn_first_triggers[near] += 1
        self.turn_first_active = self.turn_first_since is not None
        horizontal = np.array(desired[:2], dtype=float)
        if not self.turn_first_active:
            if trigger == 'side' and near is not None and ray is not None:
                # side guard (rearm after a timeout): the side rule never requests speed toward a wall it is near
                self.side_guard_active = True
                horizontal -= ray*max(0., float(horizontal @ ray))
                return horizontal, ray, np.inf
            return None
        horizontal -= self.turn_first_ray*max(0., float(horizontal @ self.turn_first_ray))
        creep = tf.coast_creep_speed if state in TURN_FIRST_UNKNOWN_STATES else tf.creep_speed
        norm = float(np.linalg.norm(horizontal))
        if norm > creep:
            horizontal *= creep/norm if norm > 0 else 0.
        return horizontal, self.turn_first_ray, creep

    @staticmethod
    def _bound_speed(previous, command, bound, dt, rate):
        """The command with its horizontal speed brought down to `bound` at up to `rate` m/s^2 (beyond the taper),
        its direction kept."""
        speed = float(np.linalg.norm(command[:2]))
        if speed <= bound:
            return command
        allowed = max(bound, float(np.linalg.norm(previous[:2]))-rate*dt)
        if speed > allowed:
            command = command.copy()
            command[:2] *= allowed/speed
        return command

    def _brake_sink_floor(self, before, after):
        """The vertical value after one clearance-brake step (`after`; `before`: the value the step started from)
        under the clearance brake's sink floor (ClearanceBrakeConfig; applied only with wall_apply): at least
        min(before, 0) - max_added_sink. Records the sink the step added below min(before, 0) (brake_added_sink,
        measured with or without the rule), the sink it left after the floor (brake_sink_left: equal to
        brake_added_sink without the rule or in shadow) and the sink the floor withheld (brake_sink_withheld)."""
        reference = min(before, 0.)
        self.brake_added_sink = max(self.brake_added_sink, reference-after)
        cb = self.clearance_brake
        if cb is not None and self.wall_apply and after < reference-cb.max_added_sink:
            self.brake_sink_withheld = max(self.brake_sink_withheld, reference-cb.max_added_sink-after)
            after = reference-cb.max_added_sink
        self.brake_sink_left = max(self.brake_sink_left, reference-after)
        return after

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

    def _assist_sources(self, state, cap, ray, turn_first, turn_first_bound, position=None, now=None):
        """The binding caps of this tick for cap tracking (MotorAssistConfig): (source, horizontal unit direction, bound
        on the request's speed along it), from the pilot's own final request (pilot_command)."""
        a = self.motor_assist
        command = self.pilot_command
        out = []
        norm = float(np.linalg.norm(command[:2]))
        if 'request' in a.cap_sources and state in a.request_states and not self.launching and norm > 1e-6:
            out.append(('request', command[:2]/norm, norm))
        if 'governor' in a.cap_sources and cap is not None and ray is not None:
            flat = np.asarray(ray, float)[:2]
            size = float(np.linalg.norm(flat))
            # the governor bounds the request this tick (it braked, or the request sits at the cap)
            binds = self.clearance_braking or float(command @ np.asarray(ray, float)) >= cap-.05
            gov = self.clearance
            standoff = (not a.standoff_tracking and isinstance(gov, TtcClearanceGovernor) and now is not None
                        and now <= gov.standoff_until)
            if size >= .5 and binds and not standoff:
                out.append(('governor', flat/size, float(cap)))
        if 'turn_first' in a.cap_sources and turn_first is not None:
            out.append(('turn_first', np.asarray(turn_first, float), 0.))
            if np.isfinite(turn_first_bound) and norm > 1e-6:
                out.append(('turn_first', command[:2]/norm, float(turn_first_bound)))
        gov = self.clearance
        if a.stop_gate != 'any':
            return out+self._assist_stop_sources(position, now)
        if ('stopping' in a.cap_sources and isinstance(gov, TtcClearanceGovernor) and position is not None
                and gov.samples):
            # the motor's stopping model on the latest wall sample, while recent short-TTC wall samples confirm it and
            # the newest looming sample with evidence is one of them (a clear sample, e.g. past a gate arch, ends it)
            terrain = gov.config.terrain_fraction
            recent = [x for x in gov.samples if now-x['received'] <= a.stop_window_s and x['ttc'] < a.stop_ttc_s
                      and (x['below'] is None or x['below'] < terrain)]
            distance, wall = gov.wall_distance(position)
            if len(recent) >= a.stop_confirm and recent[-1] is gov.samples[-1] and distance is not None:
                flat = np.asarray(wall, float)[:2]
                size = float(np.linalg.norm(flat))
                if size >= .5:
                    bound = stopping_speed(distance, a.stop_deceleration, a.stop_latency_s, a.stop_margin_m)
                    out.append(('stopping', flat/size, float(bound)))
        return out

    def _assist_wall_sample(self, sample):
        """A looming sample the stopping source counts as a wall: TTC under stop_ttc_s, not below-path terrain."""
        terrain = self.clearance.config.terrain_fraction
        return sample['ttc'] < self.motor_assist.stop_ttc_s and (sample['below'] is None or sample['below'] < terrain)

    def _assist_wall_ahead(self, state, yaw, now):
        """Version 2's wall-ahead condition (MotorAssistConfig): the next checkpoint does not lie through the surface
        ahead. The checkpoint's marker is clamped at a side edge, lies wall_ahead_deg or more off the heading, or is lost;
        or a turn-first episode or its side guard is active; or the looming governor holds a stand-off."""
        a = self.motor_assist
        if self.launching:
            return False
        if state == 'side' or state in MOTOR_ASSIST_UNKNOWN_STATES:
            return True
        off = self._bearing_off_deg(yaw)
        if state in TURN_FIRST_BEARING_STATES and off is not None and off >= a.wall_ahead_deg:
            return True
        if self.turn_first_active or self.side_guard_active:
            return True
        gov = self.clearance
        return bool(a.wall_ahead_standoff and isinstance(gov, TtcClearanceGovernor) and gov.cap is not None
                    and now <= gov.standoff_until)

    def _assist_stop_sources(self, position, now):
        """Version 2's stopping-model sources (MotorAssistConfig 4): 'stopping' while a wall-ahead condition holds (with
        its memory), else 'approach' (floored at floor_speed) while recent short-TTC wall samples confirm a wall and the
        newest looming sample with evidence is one of them. Updates the memory from the samples received since the
        previous tick."""
        a, gov = self.motor_assist, self.clearance
        if not isinstance(gov, TtcClearanceGovernor) or position is None:
            return []
        for x in gov.samples:
            if x['received'] > self.assist_seen_at:
                if self._assist_wall_sample(x):
                    self.assist_wall_at = max(self.assist_wall_at, x['received'])
                else:
                    self.assist_clear_at = max(self.assist_clear_at, x['received'])
        if gov.samples:
            self.assist_seen_at = max(self.assist_seen_at, max(x['received'] for x in gov.samples))
        recent = [x for x in gov.samples if now-x['received'] <= a.stop_window_s and self._assist_wall_sample(x)]
        confirmed = bool(gov.samples) and len(recent) >= a.stop_confirm and recent[-1] is gov.samples[-1]
        if confirmed:
            self.assist_confirmed_at = now
        distance, wall = gov.wall_distance(position)
        if distance is None:
            return []
        flat = np.asarray(wall, float)[:2]
        size = float(np.linalg.norm(flat))
        if size < .5:
            return []
        bound = stopping_speed(distance, a.stop_deceleration, a.stop_latency_s, a.stop_margin_m)
        if self.assist_wall_ahead:
            remembered = (now-self.assist_confirmed_at <= a.stop_memory_s
                          and self.assist_wall_at > self.assist_clear_at)
            if 'stopping' in a.cap_sources and (confirmed or remembered):
                return [('stopping', flat/size, float(bound))]
            return []
        if 'approach' not in a.cap_sources or not confirmed:
            return []
        # rising ground is the vertical guard's, not a surface to brake for: no approach while the pilot's own path
        # toward its checkpoint climbs (its vertical request before the looming governor and the vertical guard, whose
        # terrain climb can read a wall ahead as rising ground: minus-brain10b-r4-02's hairpin)
        if np.isfinite(self.pilot_vertical) and self.pilot_vertical > a.approach_climb_max:
            return []
        return [('approach', flat/size, float(max(bound, a.floor_speed)))]

    def _motor_assist(self, command, velocity, dt, state, sources, vertical_cap=None):
        """The assisted request (MotorAssistConfig) for the pilot's own final request `command`; updates the rule's
        state, per-tick log values and seconds."""
        a, c = self.motor_assist, self.config
        out = np.array(command, dtype=float)
        vh = np.asarray(velocity[:2], float)
        self.assist_sources = sources
        # 1. cap tracking: per source, the extra reduction follows gain x (excess - deadband), rate-limited; version 2's
        # approach and stopping bounds are one stopping model and share one extra reduction (a wall-ahead condition
        # that starts mid-approach carries it over)
        key = lambda name: MOTOR_ASSIST_EXTRA_KEY.get(name, name)
        targets = {name: 0. for name in MOTOR_ASSIST_SOURCES}
        for name, h, bound in sources:
            excess = float(vh @ h)-bound
            targets[key(name)] = max(targets[key(name)],
                                     float(np.clip(a.cap_gain*(excess-a.cap_deadband), 0., a.cap_max)))
        for name in MOTOR_ASSIST_SOURCES:
            x = self.assist_extra[name]
            self.assist_extra[name] = float(x+np.clip(targets[name]-x, -a.cap_fall*dt, a.cap_rise*dt))
        removed, top = {}, 0.
        before = out[:2].copy()
        plans = [bound for name, _, bound in sources if name in ('stopping', 'approach')]
        self.assist_plan = float(min(plans)) if plans else float('nan')
        if a.stop_gate != 'any':
            for name in ('approach', 'stopping'):
                self.assist_v2_time[name] += dt*any(n == name for n, _, _ in sources)
        for name, h, bound in sources:
            x = self.assist_extra[key(name)]
            if x <= 0 and name not in ('stopping', 'approach'):
                continue                       # the stopping model's bounds apply as caps of their own
            along = float(out[:2] @ h)
            limit = max(0., bound-x) if name == 'request' else max(-a.cap_reverse, bound-x)
            if along > limit:
                out[:2] -= h*(along-limit)
                removed[name] = removed.get(name, 0.)+along-limit
        self.assist_source = ''
        for name, amount in removed.items():
            self.assist_time[name] += dt
            if amount > top:
                top, self.assist_source = amount, name
        # 2. sag compensation: climb while the measured vertical speed lies below the request, and while the motor is
        # asked to accelerate hard from low speed
        target = 0.
        if state not in MOTOR_ASSIST_SAG_EXCLUDED and not self.launching:
            shortfall = float(out[2]-velocity[2])
            if shortfall > a.sag_deadband:
                target = a.sag_gain*(shortfall-a.sag_deadband)
            if a.sag_lead_gain > 0 and (a.sag_lead_max_speed <= 0 or float(np.linalg.norm(vh)) <= a.sag_lead_max_speed):
                lead = float(np.linalg.norm(out[:2]-vh))
                target += a.sag_lead_gain*max(0., lead-a.sag_lead_deadband)
            if -shortfall > a.sag_climb_margin:
                target = 0.                    # already climbing faster than asked
            target = float(np.clip(target, 0., a.sag_max))
        self.assist_sag = float(self.assist_sag+np.clip(target-self.assist_sag, -a.sag_fall*dt, a.sag_rise*dt))
        vertical = out[2]
        if self.assist_sag > 0:
            ceiling = c.vertical_up if vertical_cap is None else min(c.vertical_up, vertical_cap)
            out[2] = max(out[2], min(out[2]+self.assist_sag, ceiling))
        if a.slew > 0:
            # 3. slew (version 2): the assist's change of the request moves by at most slew x dt per tick
            # (the horizontal change as a vector and the vertical change each, as the clearance brake bounds its own
            # horizontal and vertical steps)
            command = np.asarray(command, dtype=float)
            step = (out-command)-self.assist_delta
            size, room = float(np.linalg.norm(step[:2])), a.slew*dt
            limited = size > room or abs(float(step[2])) > room
            if size > room:
                step[:2] *= room/size
            step[2] = float(np.clip(step[2], -room, room))
            self.assist_v2_time['slew_limited'] += dt*limited
            self.assist_delta = self.assist_delta+step
            out = command+self.assist_delta
        self.assist_horizontal = float(np.linalg.norm(before-out[:2]))
        self.assist_removed += dt*max(0., float(np.linalg.norm(before))-float(np.linalg.norm(out[:2])))
        self.assist_vertical = float(out[2]-vertical)
        self.assist_time['sag'] += dt*(self.assist_vertical > 0)
        self.assist_added += dt*self.assist_vertical
        return out

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
        # The request the motor received on the previous tick: the support rule and the descent-path shortfall compare
        # the measured vertical speed with it. It is the pilot's own request unless a motor assist changed it.
        issued = self.velocity_command
        if self.motor_assist is not None and self.pilot_command is not None:
            # the pilot's own request is its state; the assisted one only went to the motor
            self.velocity_command = self.pilot_command
        position = senses['pos'][0].cpu().numpy().astype(float)
        velocity = senses['vel_world'][0].cpu().numpy().astype(float)
        rotation = quat_wxyz_to_mat(senses['quat'][0].cpu().numpy())
        issued_at = self.last_time             # the previous tick, when self.issued_throttle was issued
        dt = .01 if self.last_time is None else float(np.clip(now-self.last_time, 0., .1))
        self.last_time = now
        if position[2] >= c.launch_height:
            self.launching = False
        if self.gap_aim is not None:
            self.gap_conflict = ''
            # Terrain side steer only while the looming governor reports terrain (a climb request); with
            # terrain_rising_only (gap pilot version 3) and a vertical guard, only while its climb is for confirmed
            # rising ground (the guard's own governor: the flown one, or its shadow copy in shadow).
            terrain = self.clearance is not None and self.clearance.climb > 0
            if self.gap_aim.config.terrain_rising_only:
                guard = self._vertical_governor()
                if guard is not None:
                    terrain = guard.climb > 0 and bool(guard.escalated)
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
        dv = self.descent_view
        if dv is not None:
            # View-keeping descent (DescentViewConfig): bound the pilot's own sink so that the flight path stays inside
            # the camera's lower field of view, given the measured attitude and horizontal velocity.
            margin = dv.cue_margin_deg if state == 'cue' else dv.margin_deg
            if dv.late_max_deg > 0 and state == 'below' and self.below_since is not None:
                late = dv.late_rate_deg_s*max(0., now-self.below_since-dv.late_after_s)
                margin = max(-dv.late_max_deg, margin-late)
            self.view_bound = self._view_sink_bound(velocity, rotation, margin, yaw, dt)
            self.view_withheld = max(0., float(-desired[2])-self.view_bound)
            if self.view_withheld > 0:
                desired[2] = -self.view_bound
            self.view_withheld_integral += dt*self.view_withheld
            self.view_time['limiting'] += dt*(self.view_withheld > 0)
        # Descent path angle: a vehicle that keeps falling short of the requested
        # sink rate would pass above a lower checkpoint. Slow horizontally in
        # proportion so the flight path keeps the requested slope. A vehicle
        # that tracks its descents is unaffected.
        shortfall = 0.
        # With the vertical guard applied, a sink it withheld and a sink the terrain below prevents (the support
        # timers run: contact) are no shortfall: the horizontal request is not cut for them (keep speed).
        contact = (self.vertical_guard is not None and self.vertical_apply
                   and (self.support_since is not None or self.slope_support_since is not None))
        if (issued is not None and issued[2] < -c.descent_sink and not self.launching
                and not self.vertical_limiting and not contact and not (dv is not None and self.view_withheld > 0)):
            shortfall = max(0., float(velocity[2]-issued[2]))
        self.descent_shortfall += (1-np.exp(-dt/c.descent_time_constant))*(shortfall-self.descent_shortfall)
        self.descent_scale = float(np.clip(1-(self.descent_shortfall-c.descent_free)/c.descent_span,
                                           c.descent_min_scale if dv is None else max(c.descent_min_scale,
                                                                                      dv.descent_min_scale), 1.))
        desired[:2] *= self.descent_scale
        self.view_boost = False
        if (dv is not None and self.view_withheld > 0 and state in DESCENT_VIEW_BOOST_STATES
                and not self.launching and np.isfinite(self.schedule_speed)):
            # More speed, not less: a ring below needs a steeper path than the view allows at this speed. The horizontal
            # request rises toward the speed schedule's in proportion to the withheld sink (all of it at boost_sink).
            norm = float(np.linalg.norm(desired[:2]))
            if 1e-6 < norm < self.schedule_speed:
                share = min(1., self.view_withheld/max(dv.boost_sink, 1e-9))
                desired[:2] *= (norm+share*(self.schedule_speed-norm))/norm
                self.view_boost = True
                self.view_time['boost'] += dt
        if self.contact_support is not None:
            # Contact support (ContactSupportConfig): a ground reaction the thrust cannot explain starts the climb.
            # It compares the measured vertical speed with the request the motor received (`issued`), as the older
            # support rules below do; without a motor assist that is the pilot's own request (unchanged).
            self._contact_step(now, issued_at, velocity, rotation, dt, issued)
        # Support: a requested descent the vehicle cannot achieve means contact
        # below (terrain or an object), not a controller fault. Climb briefly.
        if self.climb_until is not None and now < self.climb_until:
            desired[2] = max(desired[2], 1.)
            state = 'support_climb'
        elif (issued is not None and issued[2] < -.8
              and velocity[2] > max(-.25, issued[2]+.6)):
            self.support_since = now if self.support_since is None else self.support_since
            self.slope_support_since = None
            if now-self.support_since > c.support_after_s:
                self.climb_until, self.support_since = now+c.support_climb_s, None
        elif (issued is not None and issued[2] < -.8
              and velocity[2] > issued[2]+c.support_slope_shortfall
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
        self.brake_added_sink = self.brake_sink_withheld = self.brake_sink_left = 0.
        self.brake_reference = float('nan')
        self.brake_ray = np.full(3, np.nan)
        self.pilot_vertical = float(desired[2])
        self.vertical_limiting = False
        arrest = False
        if clearance is not None and not self.launching:
            self._ingest_clearance(clearance, velocity, yaw, now)
        if self.clearance is not None:
            for governor in (self.clearance, self.clearance_shadow):
                if getattr(governor, 'vertical', None) is not None:
                    governor.pilot_vertical = self.pilot_vertical     # the binding test of its rising-ground rule
            cap, ray, climb = self.clearance.limits(position, velocity, now, dt, c.vertical_up)
            shadow_climb = 0.
            if self.clearance_shadow is not None:
                shadow_climb = self.clearance_shadow.limits(position, velocity, now, dt, c.vertical_up)[2]
            if getattr(self.clearance, 'vertical', None) is not None:
                # Vertical guard: sink margin, descent first, terrain climb only for rising ground.
                desired[2] = self._guard_vertical(self.clearance, desired[2], climb, velocity)
                arrest = self.clearance.arrest
                # the guard withholds sink, arrests a descent or withholds a climb while descending (ground below)
                withheld = climb > 0 and float(velocity[2]) < -self.clearance.vertical.level_band
                self.vertical_limiting = bool((self.pilot_vertical < 0 and desired[2] > self.pilot_vertical)
                                              or arrest or withheld)
            elif climb > 0:
                # Expansion below the flight path: rise over it rather than stop.
                desired[2] = max(desired[2], climb)
            vertical_cap = getattr(self.clearance, 'vertical_cap', None)
            if vertical_cap is not None:
                # Ceiling guard: overhead evidence during a climb bounds the whole vertical request.
                desired[2] = min(desired[2], vertical_cap)
            guard = self._vertical_governor()
            if guard is not None:
                if guard is self.clearance:
                    self.vertical_target = float(desired[2])
                else:
                    # shadow: the request the guard would make, logged and not applied
                    intended = self._guard_vertical(guard, self.pilot_vertical, shadow_climb, velocity)
                    self.vertical_target = float(intended if guard.vertical_cap is None
                                                 else min(intended, guard.vertical_cap))
                self.vertical_time['limiting'] += dt*(self.pilot_vertical < 0
                                                      and self.vertical_target > self.pilot_vertical)
                self.vertical_time['arrest'] += dt*guard.arrest
                self.vertical_time['climb'] += dt*(guard.climb > 0)
            self.brake_reference = float(desired[2])
            if cap is not None:
                self.brake_ray = np.array(ray, float)
            along = float(desired @ ray) if cap is not None else 0.
            braking = cap is not None and along > cap
            if braking:
                before = float(desired[2])
                desired = desired-ray*(along-cap)
                # the sink the brake added below min(the request before it, 0): measured always, floored with the
                # clearance brake's sink floor (wall-pilot version 5) when applied
                desired[2] = self._brake_sink_floor(before, float(desired[2]))
                if not self.clearance_braking:
                    self.clearance.counts['blind_engagements' if self.clearance.blind else 'brake_engagements'] += 1
                self.wall_brake_at = now
            self.clearance_braking = braking
            status = ('blind' if self.clearance.blind else 'brake') if braking else self.clearance.status
            self.clearance_time[status] = self.clearance_time.get(status, 0.)+dt
        turn_first = None               # the horizontal wall ray whose request speed turn-first removes this tick
        turn_first_bound = np.inf       # and the bound on the horizontal request it brings the request to
        if self.turn_first is not None:
            # Turn before translating at a wall (computed in shadow too, applied only with wall_apply).
            result = self._turn_first(state, desired, position, velocity, yaw, now)
            if result is not None:
                if self.turn_first_active:
                    self.turn_first_time += dt
                else:
                    self.side_guard_time += dt
                if self.wall_apply:
                    desired[:2], turn_first, turn_first_bound = result
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
        if arrest:
            up = max(up, self.clearance.vertical.arrest_acceleration)
        down = (c.vertical_command_acceleration if vertical_cap is None
                else max(c.vertical_command_acceleration, self.clearance.ceiling.vertical_slew))
        if dv is not None and vertical_cap is None and self.velocity_command[2] <= 0:
            # Throttle up: a descent starts at up to sink_acceleration (no deep throttle cut).
            down = min(down, dv.sink_acceleration)
        step[2] = np.clip(step[2], -down*dt, up*dt)
        previous = self.velocity_command.copy()
        self.velocity_command = self.velocity_command+step
        slew = self.clearance_config.brake_slew
        top, vertical_limits = max(c.command_acceleration, slew)*dt, (-down*dt, up*dt)
        if cap is not None:
            # The cap acts on the request itself, without the taper, at up to brake_slew.
            before = float(self.velocity_command[2])
            self.velocity_command = self._cap_command(previous, self.velocity_command, cap, ray, dt, slew, top,
                                                      vertical_limits)
            floored = self._brake_sink_floor(before, float(self.velocity_command[2]))
            if floored != self.velocity_command[2]:
                self.velocity_command = self.velocity_command.copy()
                self.velocity_command[2] = floored
        for key, value in (('added', self.brake_added_sink), ('withheld', self.brake_sink_withheld),
                           ('left', self.brake_sink_left)):
            if value > 0:
                self.brake_sink_time[key] += dt
                self.brake_sink_max[key] = max(self.brake_sink_max[key], value)
        if turn_first is not None:
            # ... and so does turn-first: no speed toward the wall that capped it, and the creep bound (0 while the
            # marker is lost) reached at the same slew.
            self.velocity_command = self._cap_command(previous, self.velocity_command, 0.,
                                                      np.r_[turn_first, 0.], dt, slew, top, vertical_limits)
            if np.isfinite(turn_first_bound):
                self.velocity_command = self._bound_speed(previous, self.velocity_command, turn_first_bound, dt,
                                                          slew)
        raw_ff = (self.velocity_command-previous)/max(dt, 1e-3)
        alpha = 1-np.exp(-dt/c.feedforward_time_constant)
        self.feedforward = self.feedforward+alpha*(raw_ff-self.feedforward)
        if self.motor_assist is not None:
            # Motor assist (MotorAssistConfig): the motor receives the assisted request, the pilot keeps its own.
            self.pilot_command = self.velocity_command.copy()
            if self.motor_assist.stop_gate != 'any':
                self.assist_wall_ahead = self._assist_wall_ahead(state, yaw, now)
                self.assist_v2_time['wall_ahead'] += dt*self.assist_wall_ahead
            sources = self._assist_sources(state, cap, ray, turn_first, turn_first_bound, position, now)
            self.velocity_command = self._motor_assist(self.pilot_command, velocity, dt, state, sources, vertical_cap)
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
        if getattr(self.clearance, 'ceiling', None) is not None:
            return self.clearance
        if self.clearance_shadow is not None and self.clearance_shadow.ceiling is not None:
            return self.clearance_shadow
        return None

    def _vertical_governor(self):
        """The governor that runs the vertical guard: the flown one, or its shadow copy; None without the guard."""
        if getattr(self.clearance, 'vertical', None) is not None:
            return self.clearance
        if self.clearance_shadow is not None and self.clearance_shadow.vertical is not None:
            return self.clearance_shadow
        return None

    def _view_sink_bound(self, velocity, rotation, margin_deg=None, yaw=0., dt=0.):
        """Largest sink (m/s, at least free_sink) at which the flight path made of the measured horizontal velocity and
        that sink points margin_deg inside the camera's lower image edge, at the measured attitude (DescentViewConfig).

        In camera coordinates (y down, z along the optical axis) the path's direction per 1 m/s of horizontal speed is
        a + vz*b, with a the measured horizontal direction and b the world vertical; it lies inside the lower edge by
        the margin when y/z <= kappa = tan(atan(cy/f) - margin), which is linear in vz. The lowest vz per 1 m/s of
        horizontal speed is low-passed with attitude_time_constant (the attitude of a lagging motor oscillates), then
        scaled by the measured horizontal speed. Below 0.5 m/s the heading replaces the velocity direction."""
        dv = self.descent_view
        margin = dv.margin_deg if margin_deg is None else margin_deg
        speed = float(np.hypot(velocity[0], velocity[1]))
        direction = (np.array([velocity[0], velocity[1], 0.])/speed if speed > .5
                     else np.array([np.cos(yaw), np.sin(yaw), 0.]))
        to_cam = self.camera.body_to_cam() @ np.asarray(rotation, float).T
        a, b = to_cam @ direction, to_cam @ np.array([0., 0., 1.])
        kappa = float(np.tan(np.arctan(self.camera.cy/self.camera.f)-np.radians(margin)))
        den = kappa*float(b[2])-float(b[1])
        # the lowest vertical speed per 1 m/s of horizontal speed with the path in view (0 when the camera cannot help)
        lowest = (float(a[1])-kappa*float(a[2]))/den if den > 1e-6 else 0.
        if self.view_slope is None or dv.attitude_time_constant <= 0:
            self.view_slope = lowest
        else:
            self.view_slope += (1-np.exp(-dt/dv.attitude_time_constant))*(lowest-self.view_slope)
        return float(max(dv.free_sink, -speed*self.view_slope))

    def _contact_step(self, now, issued_at, velocity, rotation, dt, issued=None):
        """One tick of contact support (ContactSupportConfig): the window's unexplained upward specific force, the
        thrust gain in free air, and a support climb (climb_until) after hold_s of contact conditions. `issued`: the
        velocity request the motor received on the previous tick (update() passes it; it differs from the pilot's own
        request, self.velocity_command at this point, only with a motor assist, whose declaration has the support rules
        compare the measured vertical speed with the request the motor received); None reads self.velocity_command."""
        if issued is None:
            issued = self.velocity_command
        cs, c = self.contact_support, self.config
        g = CONTACT_GRAVITY
        self.contact_fired = False
        self.contact_first = now if self.contact_first is None else self.contact_first
        if self.issued_throttle is not None and issued_at is not None:
            self.contact_throttle.append((float(issued_at), float(self.issued_throttle)))
        # the throttle acting now: the latest one issued at least throttle_delay_s ago
        while len(self.contact_throttle) > 1 and self.contact_throttle[1][0] <= now-cs.throttle_delay_s:
            self.contact_throttle.pop(0)
        drive = float('nan')
        if self.contact_throttle and self.contact_throttle[0][0] <= now-cs.throttle_delay_s:
            hover, scale, hover_stick = self.calibration
            drive = float(np.clip((hover+scale*(self.contact_throttle[0][1]-hover_stick)+1)/2, 0., 1.))
        thrust_z = (g*cs.thrust_twr*drive**cs.thrust_exponent*float(rotation[2, 2]) if np.isfinite(drive)
                    else float('nan'))
        drag_z = float((rotation @ (np.asarray(cs.body_drag_s_inv)*(rotation.T @ velocity)))[2])
        self.contact_samples.append((float(now), float(velocity[2]), thrust_z, drag_z, drive))
        while self.contact_samples[0][0] < now-cs.window_s-1e-9:
            self.contact_samples.pop(0)
        samples = self.contact_samples
        span = float(now-samples[0][0])
        valid = span >= .8*cs.window_s and all(np.isfinite(s[4]) and cs.drive_min <= s[4] <= cs.drive_max
                                               for s in samples)
        if not valid:
            self.contact_unexplained = float('nan')
            self.contact_since = None
            return
        self.contact_time['valid'] += dt
        thrust = float(np.mean([s[2] for s in samples]))
        drag = float(np.mean([s[3] for s in samples]))
        observed = ((samples[-1][1]-samples[0][1])/span+g+drag)/max(thrust, 1e-3)
        unexplained = (observed-self.contact_gain)*thrust
        self.contact_unexplained = float(unexplained)
        armed = not self.launching and now-self.contact_first >= cs.arm_after_s
        command = float(issued[2]) if issued is not None else float('nan')
        vz = float(velocity[2])
        climbing = self.climb_until is not None and now < self.climb_until
        suspect = (armed and not climbing and command <= -cs.sink_min and vz >= command+cs.shortfall
                   and vz <= cs.rest_vz and unexplained >= cs.unexplained_on)
        if armed and not climbing and not suspect and self.contact_since is None:
            # thrust gain in free air: quiet windows, or a rising drone (nothing below holds up a drone moving away)
            if unexplained <= cs.gain_quiet or min(s[1] for s in samples) >= cs.gain_rising:
                step = float(np.clip(observed-self.contact_gain, -cs.gain_step, cs.gain_step))
                self.contact_gain = float(np.clip(
                    self.contact_gain+(1-np.exp(-dt/cs.gain_time_constant))*step, cs.gain_min, cs.gain_max))
                self.contact_time['gain_updates'] += dt
        if not suspect:
            self.contact_since = None
            return
        self.contact_time['suspected'] += dt
        self.contact_since = now if self.contact_since is None else self.contact_since
        if now-self.contact_since >= cs.hold_s:
            self.climb_until, self.contact_since = now+c.support_climb_s, None
            self.support_since = self.slope_support_since = None
            self.contact_fired = True
            self.contact_onsets += 1

    def contact_log(self):
        """Per-tick contact-support values for logs (NaN when the rule is not declared): the window's unexplained
        upward specific force (NaN when the window is not valid), the thrust gain, and 1 on a tick the rule fired."""
        if self.contact_support is None:
            nan = float('nan')
            return dict(contact_unexplained=nan, contact_gain=nan, contact_fire=nan)
        return dict(contact_unexplained=float(self.contact_unexplained), contact_gain=float(self.contact_gain),
                    contact_fire=float(self.contact_fired))

    def contact_summary(self):
        """Support-climb onsets of the contact rule, seconds valid / suspected / learning the gain and the final gain
        (None when the rule is not declared)."""
        if self.contact_support is None:
            return None
        return dict(onsets=self.contact_onsets, seconds={k: round(v, 3) for k, v in self.contact_time.items()},
                    gain=round(self.contact_gain, 4))

    def descent_view_log(self):
        """Per-tick view-keeping values for logs (NaN when the rule is not declared): the sink bound, the sink withheld
        from the pilot's own request, and whether the horizontal request was raised (1/0)."""
        if self.descent_view is None:
            nan = float('nan')
            return dict(view_sink_bound=nan, view_withheld=nan, view_boost=nan)
        return dict(view_sink_bound=float(self.view_bound), view_withheld=float(self.view_withheld),
                    view_boost=float(self.view_boost))

    def descent_view_summary(self):
        """Seconds limiting and boosting and metres of sink withheld (None when the rule is not declared)."""
        if self.descent_view is None:
            return None
        return dict(seconds={k: round(v, 3) for k, v in self.view_time.items()},
                    withheld_m=round(self.view_withheld_integral, 3))

    def motor_assist_log(self):
        """Per-tick motor-assist values for logs (NaN / '' when the rule is not declared): the pilot's own request, the
        horizontal request removed and the climb added this tick, and the cap-tracking source that removed the most."""
        if self.motor_assist is None:
            nan = float('nan')
            return dict(assist_pilot_vx=nan, assist_pilot_vy=nan, assist_pilot_vz=nan, assist_horizontal=nan,
                        assist_vertical=nan, assist_source='', assist_plan=nan, assist_wall_ahead=nan)
        own = self.pilot_command if self.pilot_command is not None else self.velocity_command
        own = np.full(3, np.nan) if own is None else own
        return dict(assist_pilot_vx=float(own[0]), assist_pilot_vy=float(own[1]), assist_pilot_vz=float(own[2]),
                    assist_horizontal=float(self.assist_horizontal), assist_vertical=float(self.assist_vertical),
                    assist_source=self.assist_source, assist_plan=float(self.assist_plan),
                    assist_wall_ahead=float(self.assist_wall_ahead))

    def motor_assist_summary(self):
        """Seconds each part acted and metres of request changed (None when the rule is not declared); version 2 adds
        the seconds with a wall-ahead condition, with the slew bound limiting and with a floor raising a limit."""
        if self.motor_assist is None:
            return None
        out = dict(seconds={k: round(v, 3) for k, v in self.assist_time.items()},
                   removed_m=round(self.assist_removed, 3), climb_added_m=round(self.assist_added, 3))
        if self.motor_assist.version >= 2:
            out['v2_seconds'] = {k: round(v, 3) for k, v in self.assist_v2_time.items()}
        return out

    def _motor_assist_metadata(self):
        if self.motor_assist is None:
            return None
        v2 = ('; version 2: the assist\'s change of the request (assisted - own) moves per tick by at most slew x dt, '
              'horizontally (vector) and vertically each; with stop_gate wall_ahead the stopping source acts only while '
              'a wall-ahead condition holds (the checkpoint marker side-clamped, wall_ahead_deg or more off the heading '
              'or lost, a turn-first episode or side guard, or with wall_ahead_standoff the governor\'s stand-off), '
              'keeping the latest wall sample for stop_memory_s after its last confirmation unless a newer non-wall '
              'sample arrives; otherwise its confirmed samples feed the approach source, the same stopping model with its '
              'bound never below floor_speed and none while the pilot\'s vertical request before the governor and the '
              'guard exceeds approach_climb_max (cap tracking of the two shares one extra reduction); without '
              'standoff_tracking no cap tracking of the governor\'s cap during its stand-off'
              if self.motor_assist.version >= 2 else '')
        return dict(
            version=self.motor_assist.version,
            rule='cap tracking: for each binding cap of cap_sources (request: the pilot\'s final horizontal request in '
                 'request_states, bound its own magnitude; governor: the looming cap along the horizontal part of its '
                 'ray while it bounds the request; turn_first: no speed toward the wall and the creep bound; stopping: '
                 'while stop_confirm wall samples with TTC < stop_ttc_s arrived within stop_window_s and the newest '
                 'looming sample is one of them, the speed from which the motor stops within the remaining distance to '
                 'the latest wall sample, v*stop_latency_s + v^2/(2*stop_deceleration) + stop_margin_m, applied as a '
                 'bound itself), the measured horizontal speed along it beyond bound + cap_deadband lowers the bound by '
                 'cap_gain x the excess beyond the deadband (at most cap_max; rate cap_rise/cap_fall m/s^2; the request '
                 'never reverses along its own direction and points away from a wall by at most cap_reverse); sag '
                 'compensation: sag_gain x (the final vertical request - the measured vertical speed - sag_deadband) '
                 'when positive, plus sag_lead_gain x (|assisted horizontal request - measured horizontal velocity| - '
                 'sag_lead_deadband) when positive at a horizontal speed <= sag_lead_max_speed (0: any), none while '
                 'climbing more than sag_climb_margin faster than the request, at most sag_max, rate sag_rise/sag_fall, '
                 'not while launching or in a support climb, never above vertical_up or an overhead bound; the pilot '
                 'keeps its own request as its state, and the support rule and the descent-path shortfall compare the '
                 'measured vertical speed with the vertical request the motor received'+v2,
            input='measured velocity, the pilot\'s own rule states and the looming governor\'s samples; no course '
                  'geometry',
            parameters={k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self.motor_assist).items()},
            **self.motor_assist_summary())

    def _descent_view_metadata(self):
        if self.descent_view is None:
            return None
        return dict(
            version=1 if self.contact_support is None else 2,
            rule='view bound: the pilot\'s own requested sink is bounded so that the flight path (measured horizontal '
                 'velocity, requested vertical speed) points margin_deg (cue_margin_deg with the ring in view) inside '
                 'the camera\'s lower image edge at the measured attitude (exact projection, roll included; the lowest '
                 'in-view vertical speed per 1 m/s low-passed with attitude_time_constant), at least free_sink; keep '
                 'speed: a bottom-clipped ring keeps below_speed_fraction of the speed schedule, the descent-path '
                 'governor is not fed while the bound withholds sink and never cuts below descent_min_scale; more speed: '
                 'while the bound withholds sink toward a ring ahead (cue, below, below_weak) the horizontal request '
                 'rises toward the speed schedule\'s (fully at boost_sink); throttle up: the requested sink grows at up '
                 'to sink_acceleration; steep late: after late_after_s of unbroken bottom clip the margin falls at '
                 'late_rate_deg_s down to late_max_deg below the lower image edge',
            input='measured attitude and velocity and the camera calibration; no height above ground, no course geometry',
            parameters=asdict(self.descent_view), **self.descent_view_summary())

    @staticmethod
    def _guard_vertical(governor, pilot, climb, velocity):
        """The vertical request under the vertical guard (VerticalGuardConfig): the pilot's own sink scaled by the
        sink margin, at least level during an arrest, and the terrain climb only while the drone is not descending
        (descent first)."""
        v = governor.vertical
        z = pilot*governor.sink_factor if pilot < 0 else pilot
        if governor.arrest:
            z = max(z, 0.)
        if climb > 0 and float(velocity[2]) >= -v.level_band:
            z = max(z, climb)
        return float(z)

    def vertical_log(self):
        """Per-tick vertical-guard values for logs (NaN without the guard or before the first clearance sample):
        the pilot's own vertical request, the guard's vertical request (applied, or intended in shadow), the sink
        factor, arrest (1/0), the climb stage (0 none, 1 gentle, 2 rising ground confirmed) and the guard's climb."""
        nan = float('nan')
        guard = self._vertical_governor()
        if guard is None:
            return dict(vertical_pilot=nan, vertical_target=nan, vertical_factor=nan, vertical_arrest=nan,
                        vertical_stage=nan, vertical_climb=nan)
        return dict(vertical_pilot=self.pilot_vertical, vertical_target=self.vertical_target,
                    vertical_factor=float(guard.sink_factor), vertical_arrest=float(guard.arrest),
                    vertical_stage=float(0 if guard.climb <= 0 else 2 if guard.escalated else 1),
                    vertical_climb=float(guard.climb))

    def _vertical_metadata(self):
        if self.vertical_guard is None:
            return None
        guard = self._vertical_governor()
        return dict(
            version=VERTICAL_GUARD_VERSION, applied=self.vertical_apply,
            rule='sink margin: the pilot\'s own requested sink x clip((crossing_aged - margin_zero_s)/(margin_full_s - '
                 'margin_zero_s), 0, 1), crossing = max(alarm ttc, ttc_lower) of the latest sample with a ttc_lower, '
                 'the factor ramped at factor_down_rate/factor_up_rate per s, kept memory_s; keep speed: while it '
                 'withholds sink, arrests or withholds a climb while descending, or the pilot\'s support timers run '
                 '(contact), the descent-path shortfall is not fed (no horizontal cut); descent first: a below-path '
                 'alarm (below_fraction >= terrain_fraction, crossing < climb_on_s) while descending (vz < '
                 '-level_band) holds the request at least level for arrest_hold_s (slew arrest_acceleration) and starts '
                 'no climb, and no terrain climb is applied while descending; terrain climb: confirm below-path alarms '
                 '(ttc_lower < climb_on_s) while level or climbing within confirm_window_s, a new climb only with a '
                 'path alarm among them (crossing = max(alarm ttc, ttc_lower) < climb_on_s; the lower window alone '
                 'only sustains a running climb), rate graded by ttc_lower (climb_on_s -> 0, climb_full_s -> '
                 'vertical_up), bounded to gentle_climb and gentle_max_m until rising_confirm alarms arrive within '
                 'rising_window_s while climbing faster than rising_min_rise with the guard\'s climb at least '
                 'rising_min_rise above every vertical request the pilot made itself in the last rising_window_s '
                 '(rising ground; v4: only if no looming sample of the last rising_window_s saw the surface below the '
                 'path farther than climb_on_s), then up to vertical_up and the policy\'s climb_max_m; the policy\'s '
                 'climb hold and release; the ceiling guard still cuts climbs',
            input='causal looming samples (below_fraction, ttc_lower, ttc) and the measured vertical speed; no height '
                  'above ground, no metric distance',
            parameters=asdict(self.vertical_guard),
            governor='flown' if self.vertical_apply else 'shadow copy fed the same samples',
            counts=None if guard is None else dict(guard.vertical_counts),
            seconds={k: round(v, 3) for k, v in self.vertical_time.items()})

    def wall_log(self):
        """Per-tick wall-pilot values for logs: turn_first (1 while an episode is active, also in shadow; NaN when
        not declared) and the ceiling guard's governor status, climb request and vertical bound (the guarded copy's
        in shadow; NaN / '' without the guard or before the first clearance sample)."""
        nan = float('nan')
        guard = self._guarded_governor()
        cap = None if guard is None else guard.vertical_cap
        return dict(turn_first=float(self.turn_first_active) if self.turn_first is not None else nan,
                    ceiling_status=guard.status if guard is not None else '',
                    ceiling_climb=float(guard.climb) if guard is not None else nan,
                    ceiling_vertical_cap=nan if cap is None else float(cap))

    def brake_log(self):
        """Per-tick clearance-brake sink values (m/s; logged by the replay harness): the sink the brake added this
        tick below min(the request before it, 0) (measured with or without the sink floor), the sink it left after
        the floor, the sink the floor withheld (0 without the rule or in shadow), and the vertical request before the
        brake (after the vertical guard and the ceiling cap; NaN before the first clearance sample) and the clearance
        cap's looming ray (NaN without a cap)."""
        return dict(brake_added_sink=float(self.brake_added_sink), brake_sink_left=float(self.brake_sink_left),
                    brake_sink_withheld=float(self.brake_sink_withheld), brake_reference=float(self.brake_reference),
                    brake_ray_x=float(self.brake_ray[0]), brake_ray_y=float(self.brake_ray[1]),
                    brake_ray_z=float(self.brake_ray[2]))

    def _wall_metadata(self):
        if self.turn_first is None and self.ceiling_guard is None and self.clearance_brake is None:
            return None
        guard = self._guarded_governor()
        keys = ('overhead_samples', 'overhead_engagements', 'unexplained_walls', 'weak_climb_samples',
                'suppressed_climb_samples', 'climb_engagements')
        extra = {}
        if self.clearance_brake is not None:
            # wall-pilot version 5 (absent in version 4 metadata: unchanged)
            extra['clearance_brake'] = dict(
                rule='the clearance brake (the TTC wall cap along the looming ray, on the request and on the command) '
                     'leaves the vertical request at least min(its value before the brake, 0) - max_added_sink: no '
                     'descent the pilot did not ask for; its horizontal part is unchanged',
                parameters=asdict(self.clearance_brake), applied=self.wall_apply,
                seconds={k: round(v, 3) for k, v in self.brake_sink_time.items()},
                max_sink={k: round(v, 3) for k, v in self.brake_sink_max.items()})
        return dict(
            version=WALL_PILOT_VERSION if self.clearance_brake is not None else 4, applied=self.wall_apply,
            **extra,
            turn_first=None if self.turn_first is None else dict(
                rule='near a wall (a clearance stand-off, or a wall brake within brake_recent_s at a horizontal speed '
                     '<= max_speed with the latest wall sample, dead-reckoned along its looming ray and 0 once reached, '
                     'within the stopping distance v*stop_latency_s + v^2/(2*stop_deceleration) + stop_margin_m at the '
                     'closing speed v along that ray) with the checkpoint clamped at a side edge/corner, its bearing '
                     '>= engage_deg off the heading, or its marker lost (coast): the horizontal request loses its '
                     'component toward the wall (the capping looming ray) and is bounded to creep_speed '
                     '(coast_creep_speed while coasting), the speed toward the wall is removed at brake_slew, yaw keeps '
                     'turning to the checkpoint; ends when an in-view (or bottom/top clamped) bearing is within '
                     'release_deg (aligned), on search/launch/support climb (handoff) or after max_s (timeout, then '
                     'rearm_s without an episode, during which the side rule near a wall still loses its component '
                     'toward the wall: side guard)',
                version_note='version 4: stopping-distance engagement up to max_speed, coast trigger and side guard '
                             '(versions 1-3: a wall brake only at <= slow_speed 1.5 m/s)',
                parameters=asdict(self.turn_first), counts=dict(self.turn_first_counts),
                triggers=dict(self.turn_first_triggers), active_seconds=round(self.turn_first_time, 3),
                side_guard_seconds=round(self.side_guard_time, 3)),
            ceiling_guard=None if self.ceiling_guard is None else dict(
                rule='during a climb a sample without vertical evidence is terrain only if its lower-surface TTC <= '
                     'lower_ratio x its alarm TTC (else a wall); such weak terrain climbs <= weak_climb and not beyond '
                     'weak_climb_max_m above the last below-path climb request; while climbing faster than '
                     'overhead_min_rise, overhead_confirm samples with TTC < overhead_ttc_s whose expansion lies above '
                     'the path (below_fraction <= overhead_fraction) or is unexplained, at least overhead_positive of '
                     'them positive (expansion above the path, or a lower-surface TTC longer than lower_ratio x the '
                     'alarm), cut the climb and start an overhead hold of hold_s (no climb, below-path samples brake, '
                     'vertical request <= vertical_cap, brought down at vertical_slew)',
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
        out = dict(mode='race-cue', profile=self.profile,
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
                             'bearing without the shift and the shift is added after it (not amplified)'
                             + ('; side commitment (commit): a confirmation with one-sided close evidence '
                                '(near_on_path, not occluded unless commit_occluded) holds its side and largest shift '
                                'while obstacle votes or close samples keep arriving (commit_hold_s), switches only on '
                                'switch_votes consecutive opposite votes >= switch_min_deg within switch_window_s, '
                                'ends after commit_max_s or on a ring/flag conflict' if self.gap_aim.config.commit
                                else '')
                             + ('; terrain_yields: a terrain episode never latches out an obstacle confirmation'
                                if self.gap_aim.config.terrain_yields else '')
                             + ('; terrain_rising_only: with a vertical guard, terrain votes only while its climb is '
                                'for confirmed rising ground' if self.gap_aim.config.terrain_rising_only else '')),
                    wall_pilot=self._wall_metadata(),
                    vertical_guard=self._vertical_metadata(),
                    clearance_response=None if self.clearance is None else dict(
                        self._clearance_policy(),
                        parameters={k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                    for k, v in asdict(self.clearance_config).items()},
                        counts=dict(self.clearance.counts),
                        status_seconds={k: round(v, 3) for k, v in self.clearance_time.items()}),
                    limitations='Race guidance only; no freestyle objective, obstacle model or completed-lap inference')
        if self.descent_view is not None:
            out['descent_view'] = self._descent_view_metadata()     # absent when the rule is off (default unchanged)
        if self.contact_support is not None:
            out['contact_support'] = dict(
                rule='over the last window_s: unexplained = dvz/dt - (gain * mean(g*thrust_twr*drive^thrust_exponent*up_z) '
                     '- g - mean(body drag z)), drive = (processed + 1)/2 of the throttle issued throttle_delay_s '
                     'earlier (pad calibration), windows with any drive outside [drive_min, drive_max] unused; a support '
                     'climb (support_climb_s) after hold_s of: velocity command <= -sink_min, vz >= command + shortfall, '
                     'vz <= rest_vz, unexplained >= unexplained_on; armed arm_after_s after the first tick and never while '
                     'launching; gain (from 1) follows the observed thrust gain with gain_time_constant within '
                     '[gain_min, gain_max], each step clipped to +-gain_step, on armed quiet windows (unexplained <= '
                     'gain_quiet) or rising ones (lowest vz >= gain_rising), never during a contact episode or a support '
                     'climb',
                input='measured velocity and attitude, the issued throttle and the pad calibration; no height above '
                      'ground, no course geometry',
                parameters=asdict(self.contact_support), **self.contact_summary())
        if self.motor_assist is not None:
            out['motor_assist'] = self._motor_assist_metadata()     # absent when the rule is off (default unchanged)
        return out
