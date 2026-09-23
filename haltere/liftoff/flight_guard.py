"""Stop criteria for experiments; these guards never steer the drone."""
from collections import deque

import numpy as np

from ..vision.camera import quat_wxyz_to_mat


class ImpactMonitor:
    def __init__(self):
        self.history=deque(maxlen=4)
        self.impact=None
        self.departed_launch_plane=False

    def update(self, timestamp, position, velocity, quaternion):
        if self.history and timestamp<=self.history[-1][0]:
            return False
        up=quat_wxyz_to_mat(quaternion)[:,2]
        self.history.append((float(timestamp),np.array(velocity,copy=True),up))
        # Position is relative to launch, not height above local ground.
        # Once the drone leaves that plane, descending below it must not
        # disable collision detection. A drop from a raised start also arms it.
        self.departed_launch_plane |= abs(float(position[2]))>=.4
        if len(self.history)<4 or not self.departed_launch_plane:
            return False
        first,last=self.history[0],self.history[-1]
        dt=last[0]-first[0]
        if not .015<=dt<=.12:
            return False
        acceleration=(last[1]-first[1])/dt
        specific=acceleration+np.array([0.,0.,9.80665])
        axis=first[2]+last[2]
        axis=axis/max(np.linalg.norm(axis),1e-6)
        unexplained=np.linalg.norm(specific-max(0.,specific@axis)*axis)
        if np.linalg.norm(acceleration)>30 and unexplained>8:
            self.impact=dict(timestamp=timestamp,acceleration_mps2=float(np.linalg.norm(acceleration)),
                             unexplained_mps2=float(unexplained))
            return True
        return False
