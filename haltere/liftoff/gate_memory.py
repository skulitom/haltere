"""Short sensory inhibition of recently passed, camera-estimated landmarks.

This knows no track, checkpoint IDs, lap count or route. Estimated passages
are not scoring evidence. All motor commands still come from the brain.
"""
import numpy as np


class PassedGateMemory:
    contract = 'visual_refractory_v1'
    radius_m = 2.
    reverse_radius_m = 5.
    lifetime_s = 30.

    def __init__(self):
        self.active = None
        self.recent = []
        self.passages = 0
        self.rejected_detections = 0

    def advance(self, position, now):
        self.recent = [(p,d,t) for p,d,t in self.recent if 0 <= now-t <= self.lifetime_s]
        a = self.active
        if a is None:
            return None
        if not 0 <= now-a['stamp'] <= 6.:
            self.active = None
            return None
        offset = position-a['point']
        along = float(offset @ a['direction'])
        travel = float((position-a['start']) @ a['direction'])
        lateral = np.linalg.norm(offset-along*a['direction'])
        if .6 < along < 2.5 and travel > 2. and lateral < 1.6:
            point = a['point'].copy()
            self.recent.append((point,a['direction'].copy(),now));self.recent = self.recent[-16:]
            self.passages += 1
            self.active = None
            return point
        return None

    def rejects(self, point, now, position):
        # Range errors can put opposite views several metres apart. Allow that
        # uncertainty only for a measurement behind the original approach,
        # keeping a closely spaced following arch in front eligible.
        rejected = any(0 <= now-t <= self.lifetime_s and
                       (np.linalg.norm(point-p) < self.radius_m or
                        (np.linalg.norm(point-p) < self.reverse_radius_m and (point-position)@direction < -.5))
                       for p,direction,t in self.recent)
        self.rejected_detections += int(rejected)
        return rejected

    def observe(self, point, stamp, position, now):
        if point is None or stamp is None or not 0 <= now-stamp < .25:
            return
        if self.active is not None and np.linalg.norm(point-self.active['point']) < 2.:
            if stamp > self.active['stamp']:
                self.active.update(point=np.array(point,copy=True),stamp=stamp)
            return
        relative = point-position
        distance = np.linalg.norm(relative)
        if 2. <= distance <= 6.:
            self.active = dict(point=np.array(point,copy=True),stamp=stamp,
                               direction=relative/distance,start=np.array(position,copy=True))
