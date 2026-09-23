"""Explicit experimental boundary between visual proposals and local guidance.

No control process is created here. The caller must opt in and retain its own
image, telemetry, flight-limit and impact guards. Unknown space is never certified.
This module contains no course-specific inputs, route access or motor policy.
"""
from __future__ import annotations

from collections import Counter
import multiprocessing as mp

import numpy as np


PROPOSAL_STATUSES = ('nominal_unverified','observed_obstacle_detour','observed_obstacle_brake',
                    'no_observed_clear_path_brake','stale_geometry_brake','observed_obstacle_escape')


class ProposalBuffer:
    """Latest-only shared values; both writer and reader always try-lock."""
    def __init__(self):
        context=mp.get_context('spawn')
        self.data=context.RawArray('d',15)
        self.lock=context.Lock()

    def publish(self, row, available_at):
        if row['status'] not in PROPOSAL_STATUSES:
            return False
        values=np.asarray([available_at,row['capture_time'],row['requested_goal_time'],
                           PROPOSAL_STATUSES.index(row['status']),*row['position'],
                           *row['requested_velocity'],*row['proposal_velocity'],
                           float(row['changed']),1.],float)
        if not np.isfinite(values).all():
            raise ValueError('Proposal transport requires finite observations')
        if not self.lock.acquire(False):
            return False
        try:
            np.frombuffer(self.data)[:]=values
        finally:
            self.lock.release()
        return True

    def read(self):
        if not self.lock.acquire(False):
            return None
        try:
            values=np.frombuffer(self.data).copy()
        finally:
            self.lock.release()
        if not values[14]:
            return None
        return dict(available_at=values[0],capture_time=values[1],requested_goal_time=values[2],
                    status=PROPOSAL_STATUSES[int(values[3])],position=values[4:7],
                    requested_velocity=values[7:10],proposal_velocity=values[10:13],changed=bool(values[13]))


class GeometryControlGate:
    """Reject stale or mismatched proposals; velocity-zero is a braking request.

    A valid nominal proposal leaves the current visual pilot request untouched.
    Detours remain experimental, based on incomplete observed geometry and a
    surrogate motion model. This gate is not a collision-avoidance guarantee.
    """
    def __init__(self, *, max_age=.4, speed=2.5, vertical_speed=1.2):
        if not np.isfinite([max_age,speed,vertical_speed]).all() or min(max_age,speed,vertical_speed)<=0:
            raise ValueError('Use finite positive proposal limits')
        self.max_age,self.speed,self.vertical_speed=max_age,speed,vertical_speed
        self.first_time=None
        self.last_valid=None
        self.last_time=None
        self.status='waiting'
        self.counts=Counter()

    def resolve(self, requested, position, now, proposal):
        requested,position=np.asarray(requested,float),np.asarray(position,float)
        if (requested.shape!=(3,) or position.shape!=(3,) or not np.isfinite(requested).all()
                or not np.isfinite(position).all() or not np.isfinite(now)
                or self.last_time is not None and now<self.last_time):
            raise ValueError('Use finite causal state and task velocity')
        self.first_time=now if self.first_time is None else self.first_time
        self.last_time=now
        rejection='waiting_for_proposal'
        if proposal is not None:
            fields=np.r_[proposal['available_at'],proposal['capture_time'],proposal['requested_goal_time'],
                         proposal['position'],proposal['requested_velocity'],proposal['proposal_velocity']]
            if not np.isfinite(fields).all():
                rejection='invalid_proposal'
            elif (proposal['available_at']>now or proposal['capture_time']>proposal['available_at']
                  or proposal['requested_goal_time']>proposal['available_at']):
                rejection='future_proposal'
            elif max(now-proposal['capture_time'],now-proposal['requested_goal_time'])>self.max_age:
                rejection='stale_proposal'
            elif np.linalg.norm(position-np.asarray(proposal['position']))>self.speed*(now-proposal['requested_goal_time'])+.5:
                rejection='position_mismatch'
            elif np.linalg.norm(requested-np.asarray(proposal['requested_velocity']))>1.:
                rejection='task_changed'
            elif proposal['status'] not in PROPOSAL_STATUSES or proposal['status']=='stale_geometry_brake':
                rejection='geometry_unavailable'
            else:
                self.last_valid=now
                selected=np.array(proposal['proposal_velocity'] if proposal['changed'] else requested,dtype=float,copy=True)
                selected[:2] *= min(1.,self.speed/max(1e-9,np.linalg.norm(selected[:2])))
                selected[2]=np.clip(selected[2],-self.vertical_speed,self.vertical_speed)
                self.status='escape' if proposal['status']=='observed_obstacle_escape' else (
                    'detour' if proposal['changed'] and np.linalg.norm(selected)>.01 else (
                        'obstacle_brake' if proposal['changed'] else 'pilot_unchanged'))
                self.counts[self.status]+=1
                return selected
        self.status=rejection
        self.counts[rejection]+=1
        allowed=5. if self.last_valid is None else 1.
        since=self.first_time if self.last_valid is None else self.last_valid
        if now-since>allowed:
            raise RuntimeError('Visual geometry unavailable; pause this attempt')
        return np.zeros(3)

    def metadata(self):
        return dict(mode='experimental visual geometry guidance',live_authority=True,
                    runtime_course_geometry=False,max_proposal_age_s=self.max_age,
                    startup_grace_s=5.,stale_stop_s=1.,counts=dict(self.counts),
                    horizontal_speed_mps=self.speed,vertical_speed_mps=self.vertical_speed,
                    coverage_certified=False,
                    recovery='Initially violated surfaces must recede within 1 cm slack; all other margins and braking-endpoint clearance remain required',
                    limits='Incomplete image geometry and unvalidated point-mass response; not a collision-avoidance guarantee')
