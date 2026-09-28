"""Distil the fast velocity-command PD into the connectome's motor readout.

Offline measured-drone surrogate only (`IdentifiedSim`, synthetic checkpoint
courses and the synthetic HUD marker from `liftoff.fast_rehearsal`). The fast
race-cue pilot supplies world velocity requests; the brain sees them as a
body-frame goal and its horizontal velocity senses in a declared speed-scaled
contract, so a fast flight looks like its familiar 3 m/s regime. Only the
throttle/roll/pitch readout rows and biases change: connectome wiring,
transmitter signs, encoders and recurrent weights are untouched and verified.
The PD teacher is never saved into or loaded by the exported brain.

Data are gathered DAgger-style: first under the teacher, then under the
brain's own control with teacher labels on the states it visits. The readout
is a parent-centred ridge fit, optionally with steep-sink samples up-weighted
and a smoothness penalty on the command change between consecutive 10 ms ticks
(low ridge alone makes the sticks chatter). ``--resolve RUN`` refits a saved
run's data without new rollouts. Rehearsal results are development checks,
never Liftoff flight evidence.

Braking data (all off by default). ``--synthetic-caps F`` gives a share F of the
drones in every collection round a `SyntheticCaps` object in the place of the
pilot's looming governor (``pilot.clearance``): it lowers a speed cap along a ray
near the travel direction as the TTC governor does, holds it and releases it,
so the pilot's own cap path (brake_slew, no taper) shapes the request and the
teacher labels the braking. ``--slow-legs F`` flies a share F of the collection
courses at a sustained pilot speed drawn from ``--slow-leg-speed`` (at or below
the nominal contract speed). ``--brake-weight W`` up-weights samples that are
aligned, level, at speed and over-speed along the track (`brake_weights`).
Synthetic caps are training data only; nothing at runtime reads them.

Label teacher (brain-10, all off by default). ``--teacher-gains k=v ...`` overrides FastPDConfig fields of the
teacher that labels every sample and flies the first round (a slower attitude loop the connectome's latency can
follow; the deployed FastMotorPD is unchanged). ``--turn-relief F`` shapes the request the teacher sees in capped
turns (`capped_turn_relief`): F x the along-track braking that comes only from the turn geometry is removed, so
the lagging student is not taught to brake below the requested speed while it turns. ``--caps-config JSON``
overrides SyntheticCapsConfig fields (e.g. longer holds like the live governor's). ``--smooth-rows T R P``
scales the smoothness penalty per readout row. Labels and caps are training data only.
"""
from __future__ import annotations

import argparse
import copy
from collections import deque
from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .bptt import load_checkpoint
from .human_brain import export
from .motor_tracking import load_recorded_retina, motor_features, retina_sequence
from .thermal import wait_if_hot
from ..brain.gate_senses import gate_observation
from ..brain.motor_baseline import FastMotorPD, FastPDConfig
from ..brain.retina import RETINA_DIM
from ..liftoff.camera_pose import CameraPoseHistory
from ..liftoff.fast_race_cue import FastRaceCue
from ..liftoff.fast_rehearsal import hill_course, hud_marker, synthetic_course
from ..sim.identified import IdentifiedSim
from ..sim.quad import quat_to_mat
from ..vision.camera import Camera
from ..vision.datasets import sha256


def fast_contract(speed, vertical_goal_seconds=1., scaled_speed=3.):
    """Declared sensory scaling: a nominal-speed request looks like `scaled_speed`.

    The brain's velocity senses saturate (tanh) above about 3 m/s, so a smaller
    scaled speed keeps requests between half and full speed distinguishable.
    """
    if not 0 < speed <= 20 or not 0 < scaled_speed <= 3:
        raise ValueError('Use a nominal speed in (0, 20] m/s and a scaled speed in (0, 3]')
    return dict(nominal_speed_mps=float(speed), goal_seconds=scaled_speed/speed, velocity_scale=scaled_speed/speed,
                scaled_speed_mps=float(scaled_speed),
                vertical_goal_seconds=float(vertical_goal_seconds),
                encoding='horizontal goal = request*goal_seconds, vertical goal = request*vertical_goal_seconds; '
                         'horizontal velocity senses scaled by velocity_scale, vertical unscaled')


def brain_observation(meta, senses, motor, task, retina, request, contract):
    """Exactly the runtime sensory path for brain motors under the fast pilot."""
    rotation = quat_to_mat(senses['quat'])
    scale = contract['velocity_scale']
    velocity = senses['vel_world']*senses['vel_world'].new_tensor([scale, scale, 1.])
    scaled = {**senses, 'vel_world': velocity, 'vel_body': torch.einsum('bji,bj->bi', rotation, velocity)}
    goal = request*request.new_tensor([contract['goal_seconds'], contract['goal_seconds'],
                                       contract['vertical_goal_seconds']])
    relative = torch.einsum('bji,bj->bi', rotation, goal)
    sensor = meta['gate_sensor']
    height = torch.zeros(len(request), 1, device=request.device) if sensor.get('search_height_anchor') else None
    return gate_observation(scaled, motor, task, retina, relative,
                            height_invariant=sensor.get('height_invariant', False),
                            gravity_aligned_height=sensor.get('gravity_aligned_height', False),
                            search_height_error=height, raw_retina_active=sensor.get('raw_retina_active', False))


@dataclass(frozen=True)
class SyntheticCapsConfig:
    """Synthetic looming-governor caps for DAgger rollouts (training data only; never used at runtime).

    While the pilot is not launching, its state is one of `states` and it flies at least `min_speed`
    horizontally, a cap event starts at `rate_per_min` per minute (Poisson; no new event while one is
    active). Its ray is the horizontal velocity direction turned by U(-ray_jitter_deg, ray_jitter_deg); its
    target is, with probability `absolute_share`, U(absolute) m/s, else U(relative) x the horizontal speed.
    As in `TtcClearanceGovernor` the cap starts at the speed along the ray and falls at `brake_rate` m/s^2 to
    the target; it holds U(hold_s) s from reaching it, then rises at `release` m/s^2 and ends once it exceeds
    the pilot speed + `end_margin`. brake_rate and release are the TTC governor's defaults.

    Climbing caps (brain-11; off at climb_share 0): with probability `climb_share` an event also requests a climb of
    U(climb) m/s from its onset for U(climb_s) s (at most until its release starts), as the looming governor's terrain
    climb (or the vertical guard's) does beside its cap: the pilot raises its vertical request to it. The extra draws
    are made only when climb_share > 0, so the default event stream is unchanged."""
    rate_per_min: float = 8.
    min_speed: float = 3.
    states: tuple = ('cue', 'side', 'coast')
    absolute_share: float = .5
    absolute: tuple = (1., 4.)
    relative: tuple = (.2, .7)
    hold_s: tuple = (.5, 3.)
    ray_jitter_deg: float = 15.
    brake_rate: float = 8.
    release: float = 3.
    end_margin: float = 1.
    climb_share: float = 0.
    climb: tuple = (.5, 1.5)
    climb_s: tuple = (.5, 2.)

    def __post_init__(self):
        values = [self.rate_per_min, self.min_speed, self.absolute_share, *self.absolute, *self.relative,
                  *self.hold_s, self.ray_jitter_deg, self.brake_rate, self.release, self.end_margin,
                  self.climb_share, *self.climb, *self.climb_s]
        if not np.isfinite(values).all() or min(values) < 0 or self.rate_per_min <= 0 or self.brake_rate <= 0 \
                or self.release <= 0:
            raise ValueError('Use finite non-negative synthetic cap parameters (positive rate, brake rate, release)')
        for name in ('absolute', 'relative', 'hold_s', 'climb', 'climb_s'):
            low, high = getattr(self, name)
            if not 0 <= low <= high:
                raise ValueError(f'{name} is a (low, high) range')
        if not self.absolute_share <= 1 or not self.relative[1] <= 1 or not self.ray_jitter_deg < 90:
            raise ValueError('absolute_share and relative are fractions; ray_jitter_deg < 90')
        if not self.climb_share <= 1 or not self.climb[1] <= 3.5:
            raise ValueError('climb_share is a fraction and climbs are at most the pilot vertical_up (3.5 m/s)')
        if not set(self.states):
            raise ValueError('Name the pilot states in which caps may start')


def caps_record(caps):
    """JSON record of a SyntheticCapsConfig (None for None). The climbing-cap fields are recorded only when climbing
    caps are on, so runs made before they existed (and default runs) keep their record and can be resolved."""
    if caps is None:
        return None
    record = json.loads(json.dumps(asdict(caps)))
    if caps.climb_share == 0:
        for key in ('climb_share', 'climb', 'climb_s'):
            record.pop(key)
    return record


class SyntheticCaps:
    """Stand-in for the pilot's looming governor (``FastRaceCue.clearance``) that issues synthetic caps.

    It has the interface FastRaceCue reads from a governor (limits(), cap, cap_ray, climb, vertical_cap, blind,
    status, counts, standoff_until); it climbs only with climbing caps and never holds a stand-off. Assign it as
    ``pilot.clearance`` before the first tick; rollouts pass no clearance samples, so the pilot never builds a
    looming governor of its own. Nothing happens while the pilot is launching (the pilot calls limits() then
    too). `events` logs every cap: onset, target, kind, hold, ray, closing (speed along the ray at onset) and
    the times it reached the target, started its release and ended (None while pending)."""

    def __init__(self, config, rng, pilot):
        self.config, self.rng, self.pilot = config, rng, pilot
        self.cap = self.cap_ray = None
        self.climb = 0.
        self.vertical_cap = None
        self.blind = self.sustained = False
        self.standoff_until = -np.inf
        self.last_time = None
        self.status = 'none'
        self.counts = dict(samples=0, no_evidence=0, brake_engagements=0, climb_engagements=0, blind_engagements=0,
                           synthetic_events=0)
        self.events = []
        self.event = None

    def _start(self, velocity, speed, now):
        c = self.config
        turn = np.radians(self.rng.uniform(-c.ray_jitter_deg, c.ray_jitter_deg))
        heading = velocity[:2]/speed
        ray = np.array([np.cos(turn)*heading[0]-np.sin(turn)*heading[1],
                        np.sin(turn)*heading[0]+np.cos(turn)*heading[1], 0.])
        absolute = bool(self.rng.random() < c.absolute_share)
        target = float(self.rng.uniform(*c.absolute) if absolute else self.rng.uniform(*c.relative)*speed)
        closing = float(velocity @ ray)
        self.event = dict(onset=float(now), target=target, kind='absolute' if absolute else 'relative',
                          hold=float(self.rng.uniform(*c.hold_s)), ray=ray, closing=closing,
                          reached=None, release=None, end=None)
        if c.climb_share > 0:
            climbing = bool(self.rng.random() < c.climb_share)
            rate, duration = float(self.rng.uniform(*c.climb)), float(self.rng.uniform(*c.climb_s))
            self.event.update(climb=rate if climbing else 0., climb_until=float(now)+duration)
        self.events.append(self.event)
        self.counts['synthetic_events'] += 1
        self.cap, self.cap_ray = max(closing, target), ray

    def limits(self, position, velocity, now, dt, vertical_up):
        """(cap or None, ray or None, climb: 0 unless a climbing cap is active) for this tick."""
        c = self.config
        if self.pilot.launching:
            self.status = 'none'
            return None, None, 0.
        velocity = np.asarray(velocity, float)
        speed = float(np.linalg.norm(velocity[:2]))
        if (self.event is None and speed >= max(c.min_speed, 1e-3) and self.pilot.state in c.states
                and self.rng.random() < c.rate_per_min/60.*dt):
            self._start(velocity, speed, now)
        e = self.event
        if e is not None:
            if e['reached'] is None:
                self.cap = max(e['target'], self.cap-c.brake_rate*dt)
                if self.cap <= e['target']:
                    e['reached'] = float(now)
            elif e['release'] is None and now-e['reached'] >= e['hold']:
                e['release'] = float(now)
            if e['release'] is not None:
                self.cap += c.release*dt
                if self.cap > self.pilot.speed+c.end_margin:
                    e['end'] = float(now)
                    self.event = self.cap = self.cap_ray = None
        self.status = 'armed' if self.cap is not None else 'none'
        if c.climb_share > 0:
            # a climbing cap requests its climb from the onset for its duration, at most until the release starts
            e = self.event
            self.climb = e['climb'] if e is not None and e['release'] is None and now < e['climb_until'] else 0.
            return self.cap, self.cap_ray, self.climb
        return self.cap, self.cap_ray, 0.


@dataclass(frozen=True)
class YawHoldConfig:
    """Yaw holds for DAgger rollouts (brain-11; training data only, never used at runtime): on each drone, while its
    pilot is not launching, a hold starts at `rate_per_min` per minute (Poisson; none while one is active) and holds the
    yaw stick at 0 for U(hold_s) s. The pilot keeps requesting its world velocity toward the checkpoint, so the request
    turns in the body frame (lateral and diagonal requests, as in a sharp turn faster than the yaw or after a turn-first
    stop), and the teacher labels them."""
    rate_per_min: float = 4.
    hold_s: tuple = (.5, 2.)

    def __post_init__(self):
        if not np.isfinite([self.rate_per_min, *self.hold_s]).all() or self.rate_per_min <= 0 \
                or not 0 < self.hold_s[0] <= self.hold_s[1]:
            raise ValueError('Use a positive yaw-hold rate and a (low, high) hold range in seconds')


class YawHolds:
    """Per-drone yaw-hold state of a rollout (YawHoldConfig), drawn with rng([seed, 5])."""

    def __init__(self, config, batch, seed):
        self.config, self.rng = config, np.random.default_rng([seed, 5])
        self.until = np.full(batch, -np.inf)
        self.count = 0

    def step(self, now, dt, launching):
        """Mask (B,) of drones whose yaw stick is held this tick."""
        c = self.config
        free = (self.until <= now) & ~np.asarray(launching, bool)
        start = free & (self.rng.random(len(self.until)) < c.rate_per_min/60.*dt)
        if start.any():
            self.until[start] = now+self.rng.uniform(*c.hold_s, size=int(start.sum()))
            self.count += int(start.sum())
        return (self.until > now) & ~np.asarray(launching, bool)


def capped_turn_relief(request, velocity, relief, *, full_deg=45., zero_deg=90., over_low=.3, over_high=1.,
                       min_speed=1.5):
    """Request seen by the label teacher in a capped turn (training only; the pilot's request is unchanged).

    Horizontal request r (magnitude R), horizontal velocity v (speed V, unit u). The Cartesian teacher brakes along
    the track by V - r.u; the speed magnitude alone asks V - min(R, V). The difference g = min(R, V) - r.u >= 0 is
    the along-track braking that comes only from the turn geometry. The shaped request adds
    relief * w_turn * w_cap * g along u, where w_turn = 1 up to `full_deg` between r and v and falls linearly to 0
    at `zero_deg`, and w_cap (request side: the request magnitude below the speed, as under a governor cap) rises
    linearly from 0 at V - R = `over_low` to 1 at `over_high` m/s. The vertical request and the lateral part are
    unchanged; aligned requests, turns at or above the flown speed, hairpins beyond `zero_deg` and V < `min_speed`
    are not shaped. request, velocity: (B, 3) world m/s."""
    if relief <= 0:
        return request
    rh, vh = request[:, :2], velocity[:, :2].to(request.dtype)
    speed, magnitude = vh.norm(dim=-1), rh.norm(dim=-1)
    unit = vh/speed.clamp_min(1e-6)[:, None]
    along = (rh*unit).sum(-1)
    cosine = (along/magnitude.clamp_min(1e-6)).clamp(-1., 1.)
    angle = torch.rad2deg(torch.arccos(cosine))
    w_turn = ((zero_deg-angle)/(zero_deg-full_deg)).clamp(0., 1.)
    w_cap = ((speed-magnitude-over_low)/(over_high-over_low)).clamp(0., 1.)
    gap = (torch.minimum(magnitude, speed)-along).clamp_min(0.)
    shift = float(relief)*w_turn*w_cap*gap*(speed >= min_speed)
    shaped = request.clone()
    shaped[:, :2] = rh+shift[:, None]*unit
    return shaped


class LabelTeacher:
    """FastMotorPD with declared FastPDConfig overrides and optional capped-turn relief (training labels only).

    With no overrides and no relief it is FastMotorPD. The teacher is never saved into or loaded by a brain."""

    def __init__(self, profile, calibration, gains=None, turn_relief=0.):
        config = replace(FastPDConfig(), **gains) if gains else None
        self.pd = FastMotorPD(profile, calibration, config)
        self.turn_relief = float(turn_relief)

    def command(self, sensors, velocity, feedforward=None, dt=.01):
        if self.turn_relief > 0:
            velocity = capped_turn_relief(torch.as_tensor(velocity, dtype=torch.float32), sensors['vel_world'],
                                          self.turn_relief)
        return self.pd.command(sensors, velocity, feedforward, dt)


def parse_gains(items):
    """'name=value' strings -> FastPDConfig overrides (validated)."""
    known = {f.name for f in fields(FastPDConfig)}
    out = {}
    for item in items or []:
        key, _, value = item.partition('=')
        if key not in known or not value:
            raise ValueError(f'Unknown teacher gain {item!r}; known: {sorted(known)}')
        out[key] = float(value)
    replace(FastPDConfig(), **out)  # validates
    return out


def brake_metrics(t, request, velocity, active, events, speeds, nominal, *, settle=3., onset_skip=.5,
                  within=.5, aligned_deg=20., level=.5, cruise_fraction=.8, slow_fraction=.7):
    """Braking metrics of one rollout; every class is chosen from the request side, never by the excess.

    t (T,) s; request, velocity (T, B, 3) world m/s (velocity after the tick's step, as the rollout's other
    metrics); active (T, B); events: per drone a list of SyntheticCaps events; speeds (B,) pilot speeds.
    - cruise: t > settle, no cap event, |request_h| >= cruise_fraction x the drone's pilot speed, request_h
      within aligned_deg of velocity_h, |request_z| < level. Excess = |v_h| - request_h . v_h/|v_h| (speed
      along the track beyond the request). Reported for all drones, nominal-speed drones and slower ones.
    - cap events: excess = (v_h - request_h) . ray from onset + onset_skip to the start of the release (or the
      event's end), tick-weighted over events; time until v_h . ray <= target + within for events that start
      more than `within` above the target (censored at the release: 'not reached').
    - slow-request ticks (t > settle, |request_h| < slow_fraction x nominal): mean |v_h| - |request_h| split
      into ticks inside a cap event, aligned ticks and turn ticks (request_h more than aligned_deg off v_h)."""
    t = np.asarray(t, float)
    request, velocity = np.asarray(request, float), np.asarray(velocity, float)
    active = np.asarray(active, bool)
    speeds = np.asarray(speeds, float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else .01
    rh, vh = request[..., :2], velocity[..., :2]
    req_speed, speed = np.linalg.norm(rh, axis=-1), np.linalg.norm(vh, axis=-1)
    along = (rh*vh).sum(-1)/np.maximum(speed, 1e-6)
    aligned = (speed > 1e-6) & (req_speed > 1e-6) & (along >= np.cos(np.radians(aligned_deg))*req_speed)
    level_mask = np.abs(request[..., 2]) < level
    in_cap = np.zeros(active.shape, bool)
    for i, drone_events in enumerate(events):
        for e in drone_events:
            end = e['end'] if e['end'] is not None else np.inf
            in_cap[(t >= e['onset']) & (t <= end), i] = True
    ok = active & (t[:, None] > settle)

    def mean(values, mask):
        return round(float(values[mask].mean()), 3) if mask.any() else None

    track = speed-along
    cruise = ok & ~in_cap & (req_speed >= cruise_fraction*speeds[None, :]) & aligned & level_mask
    slow_drone = (speeds < nominal-1e-6)[None, :]
    out = dict(cruise_excess_mps=mean(track, cruise), cruise_s=round(float(cruise.sum())*dt, 2),
               cruise_excess_nominal_mps=mean(track, cruise & ~slow_drone),
               cruise_excess_slow_mps=mean(track, cruise & slow_drone))
    slow = ok & (req_speed < slow_fraction*nominal)
    gap = speed-req_speed
    for name, mask in (('cap', slow & in_cap), ('aligned', slow & ~in_cap & aligned), ('turn', slow & ~in_cap & ~aligned)):
        out[f'slow_excess_{name}_mps'] = mean(gap, mask)
        out[f'slow_{name}_s'] = round(float(mask.sum())*dt, 2)
    excess, per_event, times, reachable = [], [], [], 0
    for i, drone_events in enumerate(events):
        for e in drone_events:
            ray = np.asarray(e['ray'], float)[:2]
            stop = e['release'] if e['release'] is not None else e['end'] if e['end'] is not None else np.inf
            window = (t >= e['onset']+onset_skip) & (t < stop) & active[:, i]
            if window.any():
                values = (vh[window, i]-rh[window, i]) @ ray
                excess.append(values)
                per_event.append(float(values.mean()))
            if e['closing'] > e['target']+within:
                reachable += 1
                span = np.flatnonzero((t >= e['onset']) & (t < stop) & active[:, i])
                hit = span[vh[span, i] @ ray <= e['target']+within] if len(span) else span
                times.append(float(t[hit[0]]-e['onset']) if len(hit) else np.inf)
    times = np.asarray(times, float)
    out.update(cap_events=int(sum(len(e) for e in events)),
               cap_excess_mps=round(float(np.concatenate(excess).mean()), 3) if excess else None,
               cap_excess_event_mean_mps=round(float(np.mean(per_event)), 3) if per_event else None,
               cap_excess_s=round(float(sum(len(v) for v in excess))*dt, 2),
               cap_within_events=reachable,
               cap_within_reached=round(float(np.isfinite(times).mean()), 3) if reachable else None,
               cap_within_1s=round(float((times <= 1.).mean()), 3) if reachable else None,
               cap_within_median_s=(None if not reachable or not np.isfinite(np.median(times))
                                    else round(float(np.median(times)), 2)))
    return out


@torch.no_grad()
def rollout(brain, cfg, meta, profile, contract, courses, *, controller='pd', speed=None, seconds=150.,
            seed=0, randomize=.1, collect=False, retina_stream=None, retina_dropout=.25, dropout=.1,
            camera_period=.055, camera_latency=.06, delay_steps=3, radius=3., quadratic_drag=.0075,
            pilot_speeds=None, caps=None, cap_fraction=0., cap_seed=None, record_brake=False, teacher_factory=None,
            label_lead=0, pilot_kwargs=None, yaw_holds=None, pilot_share=1.):
    """Batched closed loop: one synthetic course per drone; controller 'pd' or 'brain'.

    Off by default: `pilot_speeds` (one pilot speed per course; default `speed`), `caps` (a SyntheticCapsConfig
    given to a share `cap_fraction` of the drones, drawn with `cap_seed`, default `seed`), `record_brake`
    (adds `brake_metrics` to the result), `teacher_factory` (profile, calibration -> the label teacher that
    also flies controller 'pd'; default FastMotorPD) and `label_lead` (collect: each sample's features are paired
    with the teacher's label `label_lead` ticks later, on the same drone if it is still flying, so the student
    learns to anticipate its own latency; the sample's request and velocity stay those of the feature tick) and
    `pilot_kwargs` (FastRaceCue keyword arguments for every drone's pilot, e.g. the deployed pilot of
    haltere.train.deployed_pilot; with `pilot_share` < 1 only that share of the drones, drawn with seed [seed, 6], get
    them) and `yaw_holds` (a YawHoldConfig: training data only, the pilot's yaw stick is held at 0 now and then so the
    request turns in the body frame). Without them the rollout is unchanged."""
    speed = contract['nominal_speed_mps'] if speed is None else speed
    batch = len(courses)
    device = brain.device
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    calibration = meta['calibration']
    sim = IdentifiedSim(profile, calibration, 'cpu', cfg.brain.dt)
    sim.randomize(batch, randomize, generator)
    state = sim.hover(batch, 0.)
    state.quad.pos[:] = 0.
    state.quad.vel[:] = 0.
    sensor = meta['gate_sensor']
    camera = Camera(320, 180, sensor['focal_320'], sensor['tilt_deg'])
    yaw_axis = profile['axes']['yaw']
    curve = (yaw_axis['coefficient_deg_s'], yaw_axis['super_rate'], yaw_axis['expo'])
    histories = [CameraPoseHistory() for _ in range(batch)]
    course_speeds = [speed]*batch if pilot_speeds is None else [float(s) for s in pilot_speeds]
    if len(course_speeds) != batch or not all(np.isfinite(s) and 0 < s <= 20 for s in course_speeds):
        raise ValueError('Give one finite pilot speed in (0, 20] m/s per course')
    kwargs_for = set(range(batch))
    if pilot_kwargs and pilot_share < 1:
        kwargs_for = {int(i) for i in np.random.default_rng([seed, 6]).permutation(batch)[:int(round(pilot_share*batch))]}
    pilots = [FastRaceCue(sensor, histories[i], course_speeds[i], reference_speed=course_speeds[i], yaw_curve=curve,
                          calibration=calibration, **((pilot_kwargs or {}) if i in kwargs_for else {}))
              for i in range(batch)]
    capped = []
    if caps is not None and cap_fraction > 0:
        base = seed if cap_seed is None else cap_seed
        capped = sorted(int(i) for i in np.random.default_rng([base, 1]).permutation(batch)[:int(round(cap_fraction*batch))])
        for i in capped:
            pilots[i].clearance = SyntheticCaps(caps, np.random.default_rng([base, 2, i]), pilots[i])
    teacher = (FastMotorPD if teacher_factory is None else teacher_factory)(profile, calibration)
    holds = None if yaw_holds is None else YawHolds(yaw_holds, batch, seed)
    idle = torch.tensor([[-1., 0., 0., 0.]]).repeat(batch, 1)
    queue = deque(idle.clone() for _ in range(delay_steps))
    pending = [deque() for _ in range(batch)]
    detections, captures = [None]*batch, [None]*batch
    next_capture = np.zeros(batch)
    targets = np.zeros(batch, int)
    finish = np.full(batch, np.nan)
    crashed = np.zeros(batch, bool)
    brain_state = brain.init_state(batch)
    W = brain.inference_matrix() if device.type == 'cpu' else brain.weight_matrix().detach()
    steps = int(seconds/cfg.brain.dt)
    retinal = (retina_sequence(retina_stream, steps, batch, seed, retina_dropout)
               if retina_stream is not None else None)
    features, labels, requested, requests_3d, velocities, steps_diff = [], [], [], [], [], []
    before = None  # motor features one tick before a sample, for the command-smoothness penalty
    if label_lead < 0 or int(label_lead) != label_lead:
        raise ValueError('label_lead is a whole number of ticks >= 0')
    leading = deque()  # samples waiting for their label (label_lead > 0)
    chatter, speeds, previous = [], [], None
    slow_excess = []  # measured minus requested horizontal speed when the request is below 70% of nominal
    sink_shortfall, climb_shortfall = [], []  # requested minus achieved vertical speed on descents/climbs
    tracking_error = []  # |measured - requested| velocity, all three axes
    steep_shortfall = []  # achieved minus requested vertical speed while the pilot asks for >= 1.5 m/s sink
    overspeed_s = 0.  # drone-seconds above 3 m/s horizontal while a steep, slow (<= 1.5 m/s) descent is requested
    trace = dict(t=[], request=[], velocity=[], active=[]) if record_brake else None
    for k in range(steps):
        now = k*cfg.brain.dt
        positions = state.quad.pos.numpy().astype(float)
        quaternions = state.quad.quat.numpy().astype(float)
        active = ~crashed & np.isnan(finish)
        if not active.any():
            break
        for i in range(batch):
            histories[i].append(now, positions[i], quaternions[i])
            if active[i] and np.linalg.norm(courses[i][targets[i]]-positions[i]) < radius:
                targets[i] += 1
                if targets[i] == len(courses[i]):
                    finish[i] = now
                    continue
            if active[i] and now >= next_capture[i]:
                cue = hud_marker(camera, courses[i][targets[i]], positions[i], quaternions[i]) if rng.random() > dropout else None
                if cue is not None:
                    cue['aim_u'] = cue['u']
                pending[i].append((now, now+camera_latency, cue))
                next_capture[i] = now+camera_period
            while pending[i] and pending[i][0][1] <= now:
                captures[i], _, cue = pending[i].popleft()
                detections[i] = dict(race_cue=cue) if cue is not None else None
        senses = sim.sensors(state)
        requests, feedforward = [], []
        for i in range(batch):
            single = {key: value[i:i+1] for key, value in senses.items()}
            pilots[i].update(single, senses['gyro'][i].numpy(), detections[i], captures[i], now)
            requests.append(pilots[i].velocity_command)
            feedforward.append(pilots[i].feedforward)
        request = torch.tensor(np.asarray(requests), dtype=torch.float32)
        target_action = teacher.command(senses, request, torch.tensor(np.asarray(feedforward), dtype=torch.float32))
        while leading and leading[0]['due'] == k:
            entry = leading.popleft()
            keep = entry['mask'] & torch.as_tensor(active)
            if keep.any():
                features.append(entry['m'][keep])
                labels.append(target_action[keep, :3].clamp(-.97, .97).atanh())
                requested.append(entry['request'][keep, :2].norm(dim=-1))
                requests_3d.append(entry['request'][keep])
                velocities.append(entry['velocity'][keep])
                if entry['step'] is not None:
                    steps_diff.append(entry['step'][keep])
        motor = state.quad.motor.mean(-1, keepdim=True)
        retina = retinal[k] if retinal is not None else torch.zeros(batch, RETINA_DIM)
        obs = brain_observation(meta, {key: value.to(device) for key, value in senses.items()}, motor.to(device),
                                cfg.task, retina.to(device), request.to(device), contract)
        if k == 0:
            for _ in range(50):
                _, brain_state, _ = brain(obs, brain_state, W)
        action, brain_state, _ = brain(obs, brain_state, W)
        action = action.cpu()
        command = (target_action if controller == 'pd' else action).clone()
        for i in range(batch):
            command[i] = torch.as_tensor(pilots[i].command(command[i].numpy()), dtype=torch.float32)
        if holds is not None:
            held = holds.step(now, cfg.brain.dt, [p.launching for p in pilots])
            command[torch.as_tensor(held), 3] = 0.
        if now < 1.:
            command = idle.clone()
        if collect and k > 50 and k % 5 == 4:
            before = motor_features(brain, brain_state).cpu()
        if collect and k > 50 and k % 5 == 0 and now >= 1. and active.any():
            m = motor_features(brain, brain_state).cpu()
            mask = torch.as_tensor(active)
            if label_lead:
                leading.append(dict(due=k+int(label_lead), m=m, mask=mask, request=request,
                                    velocity=state.quad.vel.clone(), step=None if before is None else m-before))
            else:
                features.append(m[mask])
                labels.append(target_action[mask, :3].clamp(-.97, .97).atanh())
                requested.append(request[mask, :2].norm(dim=-1))
                requests_3d.append(request[mask])
                velocities.append(state.quad.vel[mask].clone())
                if before is not None:
                    steps_diff.append((m-before)[mask])
        if previous is not None:
            chatter.append(float((command[:, 1:3]-previous[:, 1:3]).abs().mean()))
        previous = command
        queue.append(command)
        state = sim.step(state, queue.popleft())
        velocity = state.quad.vel
        state.quad.vel = velocity-quadratic_drag*velocity.norm(dim=-1, keepdim=True)*velocity*cfg.brain.dt
        speeds.append(float(velocity[torch.as_tensor(active)].norm(dim=-1).mean()) if active.any() else 0.)
        if trace is not None:
            trace['t'].append(now); trace['request'].append(request.numpy().copy())
            trace['velocity'].append(velocity.numpy().copy()); trace['active'].append(active.copy())
        slow = torch.as_tensor(active) & (request[:, :2].norm(dim=-1) < .7*speed) & torch.tensor(now > 3.)
        if slow.any():
            slow_excess.append(float((velocity[slow, :2].norm(dim=-1)-request[slow, :2].norm(dim=-1)).mean()))
        flying = torch.as_tensor(active) & torch.tensor(now > 3.)
        if flying.any():
            tracking_error.append(float((velocity[flying]-request[flying]).norm(dim=-1).mean()))
        steep = flying & (request[:, 2] <= -1.5)
        if steep.any():
            steep_shortfall.append(float((velocity[steep, 2]-request[steep, 2]).mean()))
            overspeed_s += float((steep & (request[:, :2].norm(dim=-1) <= 1.5)
                                  & (velocity[:, :2].norm(dim=-1) >= 3.)).sum())*cfg.brain.dt
        for bucket, mask, sign in ((sink_shortfall, flying & (request[:, 2] < -.3), 1.),
                                   (climb_shortfall, flying & (request[:, 2] > .3), -1.)):
            if mask.any():
                bucket.append(float(sign*(velocity[mask, 2]-request[mask, 2]).mean()))
        if now <= 1.5:
            state.quad.pos[:, 2] = state.quad.pos[:, 2].clamp_min(0.)
            state.quad.vel[:, 2] = state.quad.vel[:, 2].clamp_min(0.)
            state.quad.crashed[:] = False
        crashed |= state.quad.crashed.numpy() & np.isnan(finish)
        if k % 500 == 0 and device.type == 'cuda':
            wait_if_hot(68.)
    result = dict(controller=controller, speed=speed, seed=seed, drones=batch,
                  finished=int(np.isfinite(finish).sum()), crashed=int(crashed.sum()),
                  finish_s=[None if not np.isfinite(t) else round(float(t), 2) for t in finish],
                  gates=targets.tolist(), mean_speed=round(float(np.mean(speeds)), 2),
                  stick_chatter=round(float(np.mean(chatter)), 5),
                  slow_request_excess_mps=round(float(np.mean(slow_excess)), 3) if slow_excess else None,
                  sink_shortfall_mps=round(float(np.mean(sink_shortfall)), 3) if sink_shortfall else None,
                  climb_shortfall_mps=round(float(np.mean(climb_shortfall)), 3) if climb_shortfall else None,
                  velocity_error_mps=round(float(np.mean(tracking_error)), 3) if tracking_error else None,
                  steep_sink_shortfall_mps=round(float(np.mean(steep_shortfall)), 3) if steep_shortfall else None,
                  steep_slow_overspeed_s=round(overspeed_s, 2))
    if pilot_speeds is not None:
        result['pilot_speeds'] = [round(s, 3) for s in course_speeds]
    if capped:
        result['capped_drones'] = capped
        result['synthetic_cap_events'] = int(sum(len(pilots[i].clearance.events) for i in capped))
    if holds is not None:
        result['yaw_holds'] = holds.count
    if pilot_kwargs and pilot_share < 1:
        result['deployed_pilot_drones'] = sorted(kwargs_for)
    if trace is not None and trace['t']:
        events = [pilots[i].clearance.events if i in capped else [] for i in range(batch)]
        result['brake'] = brake_metrics(np.asarray(trace['t']), np.stack(trace['request']), np.stack(trace['velocity']),
                                        np.stack(trace['active']), events, course_speeds, float(speed))
    data = None
    if collect and features:
        data = dict(features=torch.cat(features), labels=torch.cat(labels), requested_speed=torch.cat(requested),
                    request=torch.cat(requests_3d), velocity=torch.cat(velocities),
                    **step_gram(torch.cat(steps_diff) if steps_diff else torch.zeros(0, features[0].shape[1])))
    return result, data


def step_gram(steps, chunk=4096):
    """Sum of outer products of the readout input's change over one 10 ms tick (bias column: zero)."""
    width = steps.shape[1]+1
    gram = torch.zeros(width, width, dtype=torch.float64)
    for start in range(0, len(steps), chunk):
        x = steps[start:start+chunk].double()
        gram[:-1, :-1] += x.T@x
    return dict(step_gram=gram, step_count=len(steps))


def speed_balance_weights(requested_speed, nominal, bins=6):
    """Equal total weight per requested-speed bin, so slow requests (descents,
    turns, braking) are not swamped by cruise samples. Mean weight is one."""
    index = (requested_speed/max(nominal, 1e-6)*bins).long().clamp(0, bins-1)
    counts = torch.bincount(index, minlength=bins).float()
    weights = 1./counts[index].clamp_min(1.)
    return weights*len(weights)/weights.sum()


def sink_weights(request, weight, threshold=-1.5):
    """Up-weight samples whose pilot request sinks at `threshold` m/s or faster (mean weight one)."""
    weights = torch.where(request[:, 2] <= threshold, float(weight), 1.)
    return weights*len(weights)/weights.sum()


def brake_mask(request, velocity, *, aligned_deg=20., level=.5, min_speed=3., min_excess=.8, turn_deg=None):
    """Samples that call for a level brake: the horizontal request within `aligned_deg` of the horizontal
    velocity, |vertical request| < `level`, |v_h| >= `min_speed` and |v_h| - request_h . v_h/|v_h| >= `min_excess`
    (over-speed along the track). With `turn_deg` (brain-11; default off) also capped turns: the request within
    `turn_deg` of the velocity and its magnitude at least `min_excess` below the speed (|v_h| - |request_h|; the turn
    geometry alone does not count), same level and speed conditions."""
    rh, vh = request[:, :2], velocity[:, :2]
    speed = vh.norm(dim=-1)
    along = (rh*vh).sum(-1)/speed.clamp_min(1e-6)
    aligned = (rh.norm(dim=-1) > 1e-6) & (along >= np.cos(np.radians(aligned_deg))*rh.norm(dim=-1))
    mask = aligned & (request[:, 2].abs() < level) & (speed >= min_speed) & (speed-along >= min_excess)
    if turn_deg is None:
        return mask
    magnitude = rh.norm(dim=-1)
    within = (magnitude > 1e-6) & (along >= np.cos(np.radians(turn_deg))*magnitude)
    return mask | (within & (request[:, 2].abs() < level) & (speed >= min_speed) & (speed-magnitude >= min_excess))


def cruise_mask(request, velocity, nominal, *, aligned_deg=20., level=.5, min_speed=3., fraction=.8, min_excess=.8):
    """Samples of cruise at the contract's speed (brain-11): the horizontal request at least `fraction` x `nominal`,
    within `aligned_deg` of the horizontal velocity, |vertical request| < `level`, |v_h| >= `min_speed` and not a brake
    sample (speed along the track over the request by less than `min_excess`)."""
    rh, vh = request[:, :2], velocity[:, :2]
    speed, magnitude = vh.norm(dim=-1), rh.norm(dim=-1)
    along = (rh*vh).sum(-1)/speed.clamp_min(1e-6)
    aligned = (magnitude > 1e-6) & (along >= np.cos(np.radians(aligned_deg))*magnitude)
    return (aligned & (magnitude >= fraction*nominal) & (request[:, 2].abs() < level) & (speed >= min_speed)
            & (speed-along < min_excess))


def cruise_weights(request, velocity, nominal, weight, **selection):
    """Up-weight `cruise_mask` samples by `weight` (mean weight one)."""
    weights = torch.where(cruise_mask(request, velocity, nominal, **selection), float(weight), 1.)
    return weights*len(weights)/weights.sum()


def brake_weights(request, velocity, weight, **selection):
    """Up-weight `brake_mask` samples by `weight` (mean weight one)."""
    weights = torch.where(brake_mask(request, velocity, **selection), float(weight), 1.)
    return weights*len(weights)/weights.sum()


def sag_mask(request, velocity, *, min_sag=1., max_descent=.5):
    """Samples where the drone sinks below its vertical request by >= `min_sag` m/s while the request does not ask for
    a descent faster than `max_descent` m/s (losing height it was asked to keep: the throttle must rise)."""
    return (request[:, 2]-velocity[:, 2] >= min_sag) & (request[:, 2] > -max_descent)


def sag_weights(request, velocity, weight, **selection):
    """Up-weight `sag_mask` samples by `weight` (mean weight one)."""
    weights = torch.where(sag_mask(request, velocity, **selection), float(weight), 1.)
    return weights*len(weights)/weights.sum()


def fit_readout(brain, features, labels, ridge, weights=None, smooth=0., step_gram=None, step_count=0,
                smooth_rows=None):
    """Parent-centred (optionally weighted) ridge on throttle/roll/pitch rows; nothing else may change.

    With `smooth` > 0 the fit also penalises the mean squared change of the (pre-tanh) command over one
    10 ms tick, from the collected feature steps: minimise weighted error + ridge*|delta|^2
    + smooth*mean|step @ (parent + delta)|^2. `smooth_rows` (three multipliers for throttle, roll, pitch; default
    all one) scales that penalty per row, each row then solved on its own."""
    x = torch.cat((features, torch.ones(len(features), 1)), -1).to(brain.readout.weight.device)
    y = labels.to(x.device)
    w = (torch.ones(len(x)) if weights is None else weights).to(x.device)[:, None]
    parent = torch.cat((brain.readout.weight[:3].detach(), brain.readout.bias[:3, None].detach()), -1)
    gram = (x*w).T@x/w.sum()
    system = gram+ridge*torch.eye(x.shape[1], device=x.device)
    rhs = (x*w).T@(y-x@parent.T)/w.sum()
    rows = None if smooth_rows is None or all(float(m) == 1. for m in smooth_rows) else [float(m) for m in smooth_rows]
    if rows is not None and (len(rows) != 3 or min(rows) < 0 or smooth <= 0):
        raise ValueError('smooth_rows takes three non-negative multipliers of a positive smooth')
    if smooth > 0:
        if step_gram is None or step_count < 1:
            raise ValueError('The smoothness penalty needs collected feature steps')
        steps = (step_gram/step_count).to(x.device, x.dtype)
        if rows is None:
            system = system+smooth*steps
            rhs = rhs-smooth*steps@parent.T
    if rows is None:
        delta = torch.linalg.solve(system, rhs).T
    else:
        delta = torch.stack([torch.linalg.solve(system+smooth*m*steps, rhs[:, i]-smooth*m*steps@parent[i])
                             for i, m in enumerate(rows)])
    before = {n: p.detach().cpu().clone() for n, p in brain.named_parameters()}
    with torch.no_grad():
        brain.readout.weight[:3].add_(delta[:, :-1])
        brain.readout.bias[:3].add_(delta[:, -1])
    changed = {n: float((p.detach().cpu()-before[n]).abs().max()) for n, p in brain.named_parameters()
               if not torch.equal(p.detach().cpu(), before[n])}
    if set(changed)-{'readout.weight', 'readout.bias'}:
        raise RuntimeError('Readout distillation changed another brain parameter')
    return changed


def slow_leg_speeds(courses, share, low, high, nominal, seed):
    """Pilot speed per collection course: a share `share` of them (drawn with `seed`) at U(low, high) m/s,
    the rest at `nominal`; None when share is 0 (all nominal)."""
    if share <= 0:
        return None
    rng = np.random.default_rng([seed, 3])
    speeds = np.full(courses, float(nominal))
    chosen = rng.permutation(courses)[:int(round(share*courses))]
    speeds[chosen] = rng.uniform(low, high, size=len(chosen))
    return speeds.tolist()


# collection settings a --resolve must repeat, with their value for runs made before they existed
COLLECTION_DEFAULTS = dict(synthetic_caps=0., slow_legs=0., slow_leg_speed=[2.5, 4.5], teacher_gains=[], turn_relief=0.,
                           label_lead=0., pilot='default', hill_share=0., yaw_holds=0., pilot_share=1.)


def collection_courses(round_index, courses, steep, hill_share):
    """The seeded collection courses of a DAgger round: synthetic_course(1000 x round + i, steep); with `hill_share` > 0 a
    seeded share of them (drawn with seed [round, 4]) are hill_course(1000 x round + i) instead (long descents off a
    crest; seeds below 5000, disjoint from the descent gates' hill seeds 6000-6011). Returns (courses, hill indices)."""
    seeds = [1000*round_index+s for s in range(courses)]
    hills = set()
    if hill_share > 0:
        order = np.random.default_rng([round_index, 4]).permutation(courses)
        hills = {int(i) for i in order[:int(round(hill_share*courses))]}
    return [hill_course(s) if i in hills else synthetic_course(s, steep=steep) for i, s in enumerate(seeds)], sorted(hills)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--out', required=True)
    parser.add_argument('--profile', default='runs/measured-dynamics-low-speed-20260923/profile.json')
    parser.add_argument('--speed', type=float, default=8.)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--ridge', type=float, default=10.)
    parser.add_argument('--rounds', type=int, default=3, help='1 teacher round + brain-controlled DAgger rounds')
    parser.add_argument('--courses', type=int, default=8)
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--retina-data', default='data/vision/observed_scene_v1/train_continuous.npz')
    parser.add_argument('--validation-retina-data', default='data/vision/observed_scene_v1/validation_continuous.npz')
    parser.add_argument('--retina-dropout', type=float, default=.25)
    parser.add_argument('--evaluation-seeds', type=int, nargs='+', default=[900, 901, 902, 903, 904, 905, 906, 907])
    parser.add_argument('--rest', type=float, default=5., help='seconds of rest between rollouts (thermal duty cycle)')
    parser.add_argument('--steep', type=float, default=0., help='probability of a 15-35 degree climbing/descending leg')
    parser.add_argument('--scaled-speed', type=float, default=3., help='apparent speed of a nominal request in the brain senses')
    parser.add_argument('--balance-speed', action='store_true', help='weight samples so each requested-speed bin counts equally')
    parser.add_argument('--vertical-goal-seconds', type=float, default=1.,
                        help='vertical goal = vertical request * this (the goal neurons saturate beyond about 1 m)')
    parser.add_argument('--sink-weight', type=float, default=1., help='weight of samples requesting >= 1.5 m/s sink')
    parser.add_argument('--smooth', type=float, default=0., help='penalty on the command change per 10 ms tick')
    parser.add_argument('--resolve', default='', help='refit the training.pt of this run (no new rollouts)')
    parser.add_argument('--synthetic-caps', type=float, default=0.,
                        help='share of the drones in every collection round given synthetic governor caps (0: off)')
    parser.add_argument('--slow-legs', type=float, default=0.,
                        help='share of the collection courses flown at a sustained slower pilot speed (0: off)')
    parser.add_argument('--slow-leg-speed', type=float, nargs=2, default=[2.5, 4.5], metavar=('LOW', 'HIGH'),
                        help='range of the slow-leg pilot speed in m/s (at most the nominal speed)')
    parser.add_argument('--brake-weight', type=float, default=1.,
                        help='weight of aligned, level, at-speed samples that are over-speed along the track')
    parser.add_argument('--sag-weight', type=float, default=1.,
                        help='weight of samples sinking >= 1 m/s below a vertical request that is not a fast descent')
    parser.add_argument('--teacher-gains', nargs='*', default=[], metavar='NAME=VALUE',
                        help='FastPDConfig overrides of the label teacher (training only), e.g. attitude_gain=4')
    parser.add_argument('--turn-relief', type=float, default=0.,
                        help='share of the turn-geometry braking removed from the labels in capped turns (0: off)')
    parser.add_argument('--caps-config', dest='caps_source', default='',
                        help='SyntheticCapsConfig overrides: a JSON object or a .json file (keys starting with _ '
                             'are comments), e.g. {"hold_s": [1, 8]}')
    parser.add_argument('--label-lead', type=float, default=0.,
                        help='pair each sample with the teacher label this many seconds later (whole ticks; 0: off)')
    parser.add_argument('--smooth-rows', type=float, nargs=3, default=[1., 1., 1.], metavar=('THR', 'ROLL', 'PITCH'),
                        help='per-row multipliers of --smooth (refit only)')
    parser.add_argument('--pilot', choices=['default', 'deployed'], default='default',
                        help='pilot of every rollout: the default fast race-cue pilot, or the deployed one (the runner\'s '
                             '--obstacle-stack on --descent-view on for the brain contract; haltere.train.deployed_pilot)')
    parser.add_argument('--pilot-share', type=float, default=1.,
                        help='with --pilot deployed: share of the drones of each collection round flying it (the rest '
                             'fly the default pilot; evaluations fly it on every drone)')
    parser.add_argument('--hill-share', type=float, default=0.,
                        help='share of the collection courses that are hill courses (long descents; 0: off)')
    parser.add_argument('--brake-level', type=float, default=.5,
                        help='largest |vertical request| of a --brake-weight sample (refit only; default 0.5: level)')
    parser.add_argument('--yaw-holds', type=float, default=0.,
                        help='yaw holds per minute per drone in the collection rounds (training data: the yaw stick held '
                             'at 0 for 0.5-2 s so the request turns in the body frame; 0: off)')
    parser.add_argument('--cruise-weight', type=float, default=1.,
                        help='weight of aligned, level samples cruising at >= 0.8 x the nominal speed that are not '
                             'over-speed (refit only; counters the brake weights\' pull on the cruise speed)')
    parser.add_argument('--brake-turn-deg', type=float, default=0.,
                        help='also up-weight capped turns with the request within this angle of the velocity and its '
                             'magnitude over-speed (refit only; 0: off)')
    args = parser.parse_args()
    if (args.ridge <= 0 or args.rounds < 1 or args.sink_weight <= 0 or args.smooth < 0 or args.brake_weight <= 0
            or args.sag_weight <= 0 or args.cruise_weight <= 0):
        raise ValueError('Use positive ridge, sink and brake weights, non-negative smoothing and at least one round')
    if not 0 < args.vertical_goal_seconds <= 2:
        raise ValueError('Use a vertical goal time in (0, 2] s')
    if not 0 <= args.synthetic_caps <= 1 or not 0 <= args.slow_legs <= 1:
        raise ValueError('--synthetic-caps and --slow-legs are shares in [0, 1]')
    if not 0 < args.slow_leg_speed[0] <= args.slow_leg_speed[1] <= args.speed:
        raise ValueError('The slow-leg speed range must lie in (0, nominal speed]')
    teacher_gains = parse_gains(args.teacher_gains)
    if not 0 <= args.turn_relief <= 1:
        raise ValueError('--turn-relief is a share in [0, 1]')
    if not 0 <= args.label_lead <= .5 or abs(args.label_lead*100-round(args.label_lead*100)) > 1e-6:
        raise ValueError('--label-lead is 0-0.5 s in whole 10 ms ticks')
    if args.caps_source and args.synthetic_caps <= 0:
        raise ValueError('--caps-config needs --synthetic-caps')
    if min(args.smooth_rows) < 0 or (args.smooth_rows != [1., 1., 1.] and args.smooth <= 0):
        raise ValueError('--smooth-rows takes non-negative multipliers of a positive --smooth')
    if not 0 <= args.yaw_holds <= 60:
        raise ValueError('--yaw-holds is 0-60 per minute')
    if not 0 < args.pilot_share <= 1 or (args.pilot_share < 1 and args.pilot != 'deployed'):
        raise ValueError('--pilot-share is a share in (0, 1] of the drones flying --pilot deployed')
    if not 0 <= args.hill_share <= 1 or not 0 < args.brake_level <= 3.5 or not 0 <= args.brake_turn_deg < 180:
        raise ValueError('--hill-share is a share, --brake-level in (0, 3.5] m/s, --brake-turn-deg in [0, 180)')
    if not args.retina_data and args.validation_retina_data:
        raise ValueError('A readout fitted without scene currents is blanked at runtime; '
                         'evaluate it that way too (--validation-retina-data "")')
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/'training-source.py')
    brain, cfg, _ = load_checkpoint(args.checkpoint, args.device)
    meta = copy.deepcopy(torch.load(args.checkpoint, map_location='cpu', weights_only=True)['visual_brain'])
    profile = json.loads(Path(args.profile).read_text())
    caps_text = (Path(args.caps_source).read_text(encoding='utf-8') if args.caps_source.endswith('.json')
                 else args.caps_source)
    caps_overrides = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(caps_text).items()
                      if not k.startswith('_')} if args.caps_source else {}
    caps = SyntheticCapsConfig(**caps_overrides) if args.synthetic_caps > 0 else None
    teacher_factory = None
    if teacher_gains or args.turn_relief > 0:
        def teacher_factory(profile, calibration):
            return LabelTeacher(profile, calibration, teacher_gains, args.turn_relief)
    caps_config = caps_record(caps)
    pilot_kwargs, pilot_record = None, None
    if args.pilot == 'deployed':
        from .deployed_pilot import deployed_pilot_kwargs
        pilot_kwargs, pilot_record = deployed_pilot_kwargs('fast_velocity_brain_v1')
    source = None
    if args.resolve:
        source = torch.load(Path(args.resolve)/'training.pt', map_location='cpu', weights_only=False)
        if (args.smooth > 0 and 'step_gram' not in source or args.sink_weight != 1 and 'request' not in source
                or (args.brake_weight != 1 or args.sag_weight != 1 or args.cruise_weight != 1)
                and ('request' not in source or 'velocity' not in source)):
            raise ValueError('That run did not save feature steps / 3D requests / velocities')
        for key in ('checkpoint', 'profile', 'speed', 'scaled_speed', 'vertical_goal_seconds', 'steep', 'seconds',
                    'courses', 'rounds', 'retina_data', 'validation_retina_data', 'retina_dropout', 'evaluation_seeds',
                    *COLLECTION_DEFAULTS):
            if source['config'].get(key, COLLECTION_DEFAULTS.get(key, getattr(args, key))) != getattr(args, key):
                raise ValueError(f'--{key.replace("_", "-")} differs from the resolved run: {source["config"][key]!r}')
        if source['config'].get('caps_config') != caps_config:
            raise ValueError(f'The resolved run used other synthetic caps: {source["config"].get("caps_config")}')
        # The same paths must still hold the same files the data were collected with.
        for key, path in (('parent_sha256', args.checkpoint), ('profile_sha256', args.profile),
                          ('retina_data_sha256', args.retina_data),
                          ('validation_retina_data_sha256', args.validation_retina_data)):
            if source['config'].get(key) != (sha256(path) if path else None):
                raise ValueError(f'{key} differs from the resolved run: the file at {path!r} was replaced')
    contract = fast_contract(args.speed, args.vertical_goal_seconds, scaled_speed=args.scaled_speed)
    if source is not None and source['config']['contract'] != contract:
        raise ValueError(f'The resolved run used another contract: {source["config"]["contract"]}')
    config = dict(**vars(args), parent_sha256=sha256(args.checkpoint), source_sha256=sha256(__file__),
                  profile_sha256=sha256(args.profile), contract=contract, caps_config=caps_config,
                  retina_data_sha256=sha256(args.retina_data) if args.retina_data else None,
                  validation_retina_data_sha256=sha256(args.validation_retina_data) if args.validation_retina_data else None,
                  resolved_training_sha256=sha256(Path(args.resolve)/'training.pt') if args.resolve else None,
                  scope='offline measured-drone surrogate on synthetic courses; not Liftoff qualification')
    if pilot_record is not None:
        config['pilot_declarations'] = pilot_record
    if source is not None and source['config'].get('pilot_declarations') != (pilot_record or None):
        raise ValueError('The resolved run was collected under other pilot declarations')
    (out/'config.json').write_text(json.dumps(config, indent=2))
    training_retina = load_recorded_retina(args.retina_data, meta['gate_sensor'])
    evaluation_retina = load_recorded_retina(args.validation_retina_data, meta['gate_sensor'])
    evaluation_courses = [synthetic_course(s, steep=args.steep) for s in args.evaluation_seeds]
    log = open(out/'log.jsonl', 'w')

    def record(entry):
        entry = dict(time=round(time.time(), 1), **entry)
        log.write(json.dumps(entry)+'\n'); log.flush(); print(json.dumps(entry), flush=True)

    for controller in ('pd', 'brain') if source is None else ():
        row, _ = rollout(brain, cfg, meta, profile, contract, evaluation_courses, controller=controller,
                         seconds=args.seconds, seed=17, retina_stream=evaluation_retina, retina_dropout=args.retina_dropout,
                         record_brake=True, pilot_kwargs=pilot_kwargs)
        record(dict(stage='baseline', **row))
    rows, targets, requests, requests_3d, velocities, history = [], [], [], [], [], []
    steps, step_count = None, 0

    def refit():
        """Refit from the parent on all data gathered so far."""
        fresh, _, _ = load_checkpoint(args.checkpoint, args.device)
        brain.load_state_dict(fresh.state_dict())
        weights = speed_balance_weights(torch.cat(requests), args.speed) if args.balance_speed else None
        if args.sink_weight != 1:
            sink = sink_weights(torch.cat(requests_3d), args.sink_weight)
            weights = sink if weights is None else weights*sink/(weights*sink).mean()
        if args.sag_weight != 1:
            sag = sag_weights(torch.cat(requests_3d), torch.cat(velocities), args.sag_weight)
            weights = sag if weights is None else weights*sag/(weights*sag).mean()
        if args.brake_weight != 1:
            brake = brake_weights(torch.cat(requests_3d), torch.cat(velocities), args.brake_weight, level=args.brake_level,
                                  turn_deg=args.brake_turn_deg or None)
            weights = brake if weights is None else weights*brake/(weights*brake).mean()
        if args.cruise_weight != 1:
            cruise = cruise_weights(torch.cat(requests_3d), torch.cat(velocities), args.speed, args.cruise_weight)
            weights = cruise if weights is None else weights*cruise/(weights*cruise).mean()
        return fit_readout(brain, torch.cat(rows), torch.cat(targets), args.ridge, weights, args.smooth,
                           steps, step_count, smooth_rows=args.smooth_rows)

    if source is not None:
        rows, targets, requests = [source['features']], [source['labels']], [source['requested_speed']]
        requests_3d = [source['request']] if 'request' in source else []
        velocities = [source['velocity']] if 'velocity' in source else []
        steps, step_count = source.get('step_gram'), source.get('step_count', 0)
        changed = refit()
        row, _ = rollout(brain, cfg, meta, profile, contract, evaluation_courses, controller='brain',
                         seconds=args.seconds, seed=17, retina_stream=evaluation_retina, retina_dropout=args.retina_dropout,
                         record_brake=True, pilot_kwargs=pilot_kwargs)
        record(dict(stage='evaluate-resolved', **row))
        history.append(dict(round=args.rounds-1, evaluation=row, changed=changed))
    for round_index in range(0 if source is None else args.rounds, args.rounds):
        controller = 'pd' if round_index == 0 else 'brain'
        courses, hills = collection_courses(round_index, args.courses, args.steep, args.hill_share)
        legs = slow_leg_speeds(args.courses, args.slow_legs, *args.slow_leg_speed, args.speed, 100+round_index)
        time.sleep(args.rest)
        row, data = rollout(brain, cfg, meta, profile, contract, courses,
                            controller=controller, seconds=args.seconds, seed=100+round_index, collect=True,
                            retina_stream=training_retina, retina_dropout=args.retina_dropout,
                            pilot_speeds=legs, caps=caps, cap_fraction=args.synthetic_caps,
                            record_brake=caps is not None or legs is not None, teacher_factory=teacher_factory,
                            label_lead=int(round(args.label_lead/cfg.brain.dt)), pilot_kwargs=pilot_kwargs,
                            yaw_holds=YawHoldConfig(rate_per_min=args.yaw_holds) if args.yaw_holds > 0 else None,
                            pilot_share=args.pilot_share)
        if hills:
            row['hill_courses'] = hills
        record(dict(stage=f'collect-{round_index}', **row, samples=len(data['labels']) if data else 0))
        if data is None:
            break
        rows.append(data['features']); targets.append(data['labels']); requests.append(data['requested_speed'])
        requests_3d.append(data['request']); velocities.append(data['velocity'])
        steps = data['step_gram'] if steps is None else steps+data['step_gram']
        step_count += data['step_count']
        changed = refit()
        time.sleep(args.rest)
        row, _ = rollout(brain, cfg, meta, profile, contract, evaluation_courses, controller='brain',
                         seconds=args.seconds, seed=17, retina_stream=evaluation_retina, retina_dropout=args.retina_dropout,
                         record_brake=True, pilot_kwargs=pilot_kwargs)
        record(dict(stage=f'evaluate-{round_index}', **row))
        history.append(dict(round=round_index, evaluation=row, changed=changed))
    if source is None:
        torch.save(dict(features=torch.cat(rows), labels=torch.cat(targets), requested_speed=torch.cat(requests),
                        request=torch.cat(requests_3d), velocity=torch.cat(velocities), step_gram=steps,
                        step_count=step_count, config=config), out/'training.pt')
    meta.pop('schema', None)
    braking = dict(synthetic_caps=args.synthetic_caps, caps_config=caps_config, slow_legs=args.slow_legs,
                   slow_leg_speed=args.slow_leg_speed, brake_weight=args.brake_weight)
    meta.update(qualified=False, fast_motor_tracking=dict(
        **contract, teacher='FastMotorPD in the measured surrogate; offline only, never loaded at runtime',
        dynamics_profile_sha256=config['profile_sha256'], parent_sha256=config['parent_sha256'],
        source_sha256=config['source_sha256'], rounds=args.rounds, courses_per_round=args.courses,
        ridge=args.ridge, sink_weight=args.sink_weight, smooth=args.smooth, balance_speed=args.balance_speed,
        **({'braking_data': braking} if braking != dict(synthetic_caps=0., caps_config=None, slow_legs=0.,
                                                         slow_leg_speed=[2.5, 4.5], brake_weight=1.) else {}),
        **({'label_teacher': dict(pd_config_overrides=teacher_gains, turn_relief=args.turn_relief,
                                  label_lead_s=args.label_lead,
                                  note='training labels only; the deployed FastMotorPD and the pilot are unchanged')}
           if teacher_factory is not None or args.label_lead > 0 else {}),
        **({'smooth_rows': args.smooth_rows} if args.smooth_rows != [1., 1., 1.] else {}),
        **({'sag_weight': args.sag_weight} if args.sag_weight != 1 else {}),
        **({'collection_pilot': dict(kind='deployed', declarations=pilot_record, share=args.pilot_share,
                                     note='DAgger rollouts and evaluations flew the deployed pilot (--obstacle-stack on '
                                          '--descent-view on for the brain contract); training data only')}
           if pilot_record is not None else {}),
        **({'hill_share': args.hill_share} if args.hill_share > 0 else {}),
        **({'yaw_holds': asdict(YawHoldConfig(rate_per_min=args.yaw_holds))} if args.yaw_holds > 0 else {}),
        **({'brake_selection': dict(level=args.brake_level, turn_deg=args.brake_turn_deg or None)}
           if args.brake_level != .5 or args.brake_turn_deg else {}),
        **({'cruise_weight': args.cruise_weight} if args.cruise_weight != 1 else {}),
        resolved_training_sha256=config['resolved_training_sha256'],
        resolved_from=None if source is None else dict(
            run=args.resolve, source_sha256=source['config']['source_sha256'],
            **{k: source['config'].get(k) for k in ('ridge', 'sink_weight', 'smooth', 'balance_speed', 'brake_weight',
                                                    'smooth_rows', 'sag_weight')
               if k not in ('brake_weight', 'smooth_rows', 'sag_weight') or k in source['config']}),
        changed_parameters=history[-1]['changed'], runtime_requires_teacher=False,
        recorded_scene_currents=training_retina is not None, evaluation=history[-1]['evaluation']))
    export(out/'candidate.pt', brain, cfg, meta, 1)
    (out/'evaluation.json').write_text(json.dumps(history, indent=2))
    record(dict(stage='exported', candidate=str(out/'candidate.pt'), sha256=sha256(out/'candidate.pt')))


if __name__ == '__main__':
    main()
