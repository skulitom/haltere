"""Align delayed camera measurements with already observed telemetry poses."""
from collections import deque

import numpy as np


class CameraPoseHistory:
    def __init__(self, capacity=256):
        self.samples=deque(maxlen=capacity)

    def append(self, received_at, position, quaternion):
        self.samples.append((float(received_at),np.array(position,copy=True),np.array(quaternion,copy=True)))

    def at(self, captured_at):
        if not self.samples:
            raise ValueError('No observed pose for camera measurement')
        # The first startup frame can precede the first telemetry receipt.
        # Later frames interpolate only poses already observed by the pilot.
        if captured_at <= self.samples[0][0]:
            return self.samples[0][1:]
        for previous,current in zip(self.samples,list(self.samples)[1:]):
            if current[0]>=captured_at:
                a=np.clip((captured_at-previous[0])/max(current[0]-previous[0],1e-9),0,1)
                position=(1-a)*previous[1]+a*current[1]
                q0,q1=previous[2],current[2]
                if q0@q1<0:
                    q1=-q1
                quaternion=(1-a)*q0+a*q1
                quaternion=quaternion/max(np.linalg.norm(quaternion),1e-9)
                return position,quaternion
        return self.samples[-1][1:]
