"""Time-to-contact from image expansion: a fly-style looming cue.

Flies brake and turn from looming: an approaching surface expands in the eye,
and the expansion rate gives time-to-contact without knowing distance. Here
dense optical flow between consecutive camera frames is corrected for the
measured body rotation (rotational flow does not depend on depth), and the
remaining expansion is fitted around the focus of expansion, i.e. where the
drone's velocity points in the image. For a surface facing the direction of
travel, divergence = 2/TTC.

It needs image texture and only sees what the camera sees; missing texture or
occlusion gives no evidence rather than free space. HUD overlays are masked.
It reads only causal frames and telemetry, never a course, route or map.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .camera import Camera


@dataclass(frozen=True)
class LoomingConfig:
    width: int = 160
    height: int = 90
    window: float = .22          # half-size of the corridor window, as a fraction of image width
    min_speed: float = 1.5       # m/s along the optical axis; below this looming is not evaluated
    min_texture: float = 4.      # mean gradient magnitude (grey levels/pixel) required in the window
    min_valid: float = .35       # fraction of unmasked, textured pixels required
    max_ttc: float = 10.

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Use finite positive looming parameters')


def hud_mask(rgb_small):
    """True where the pixel is usable: excludes white/grey HUD graphics and fixed HUD panels."""
    rgb = np.asarray(rgb_small, dtype=np.int16)
    h, w = rgb.shape[:2]
    lo, hi = rgb.min(-1), rgb.max(-1)
    white = (lo > 185) & (hi-lo < 60)
    import cv2
    white = cv2.dilate(white.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    usable = ~white
    usable[:int(.2*h)] = False                                   # timer, lap, compass
    usable[int(.3*h):int(.8*h), int(.82*w):] = False            # standings panel
    usable[int(.85*h):, int(.41*w):int(.59*w)] = False          # stick boxes
    usable[int(.87*h):, :int(.1*w)] = False                     # record/status marks
    return usable


class LoomingEstimator:
    """Stateful: feed consecutive frames with their capture times, gyro and velocity."""

    def __init__(self, sensor, config=None):
        self.config = config or LoomingConfig()
        c = self.config
        self.camera = Camera(c.width, c.height, sensor['focal_320']*c.width/320, sensor['tilt_deg'])
        self.previous = None
        self.last = None
        ys, xs = np.mgrid[0:c.height, 0:c.width].astype(np.float32)
        self.x = (xs-self.camera.cx)/self.camera.f
        self.y = (ys-self.camera.cy)/self.camera.f

    def rotational_flow(self, omega_body, dt):
        """Pixel displacement from body rotation over dt, in the camera's (right, down, forward) frame."""
        w = self.camera.body_to_cam() @ np.asarray(omega_body, float)
        x, y, f = self.x, self.y, self.camera.f
        # Scene points appear to rotate opposite to the camera.
        u = (-w[1]*(1+x*x)+w[0]*x*y+w[2]*y)*f*dt
        v = (w[0]*(1+y*y)-w[1]*x*y-w[2]*x)*f*dt
        return u, v

    def update(self, rgb, capture_time, omega_body, velocity_body):
        import cv2
        c = self.config
        small = cv2.resize(np.asarray(rgb), (c.width, c.height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        usable = hud_mask(small)
        previous, self.previous = self.previous, (gray, usable, float(capture_time))
        if previous is None:
            return None
        dt = float(capture_time)-previous[2]
        if not .005 < dt < .25:
            return None
        velocity_cam = self.camera.body_to_cam() @ np.asarray(velocity_body, float)
        forward = velocity_cam[2]
        result = dict(time=float(capture_time), dt=dt, ttc=None, divergence=None, valid=0., reason=None)
        if forward < c.min_speed:
            result['reason'] = 'slow'
            self.last = result
            return result
        flow = cv2.calcOpticalFlowFarneback(previous[0], gray, None, .5, 2, 9, 3, 5, 1.1, 0)
        ru, rv = self.rotational_flow(omega_body, dt)
        u, v = flow[..., 0]-ru, flow[..., 1]-rv
        # Focus of expansion: the direction of travel in the image.
        fx = self.camera.cx+self.camera.f*velocity_cam[0]/forward
        fy = self.camera.cy+self.camera.f*velocity_cam[1]/forward
        half = c.window*c.width
        x0, x1 = int(max(0, fx-half)), int(min(c.width, fx+half))
        y0, y1 = int(max(0, fy-.8*half)), int(min(c.height, fy+.8*half))
        if x1-x0 < 8 or y1-y0 < 6:
            result['reason'] = 'focus_outside_image'
            self.last = result
            return result
        gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        texture = np.hypot(gx, gy)[y0:y1, x0:x1]/8.
        mask = usable[y0:y1, x0:x1] & previous[1][y0:y1, x0:x1] & (texture > c.min_texture)
        valid = float(mask.mean())
        result['valid'] = valid
        if valid < c.min_valid:
            result['reason'] = 'low_texture'
            self.last = result
            return result
        ys, xs = np.nonzero(mask)
        px, py = xs+x0-fx, ys+y0-fy
        du, dv = u[y0:y1, x0:x1][mask], v[y0:y1, x0:x1][mask]
        # Least-squares affine flow; divergence = a_xx + a_yy (per frame).
        A = np.stack((np.ones_like(px), px, py), 1).astype(np.float64)
        cu, *_ = np.linalg.lstsq(A, du, rcond=None)
        cv, *_ = np.linalg.lstsq(A, dv, rcond=None)
        divergence = float(cu[1]+cv[2])/dt
        result['divergence'] = divergence
        result['ttc'] = float(min(c.max_ttc, 2./divergence)) if divergence > 2./c.max_ttc else c.max_ttc
        result['distance'] = result['ttc']*forward
        self.last = result
        return result

    def metadata(self):
        return dict(cue='image expansion (looming) after gyro de-rotation, affine fit around the focus of expansion',
                    parameters=asdict(self.config), establishes_free_space=False,
                    limitations='Needs texture; fronto-parallel TTC model; no evidence is not free space')
