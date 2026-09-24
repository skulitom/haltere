"""Keep camera decoding/detection outside the real-time controller process."""
from __future__ import annotations

import multiprocessing as mp
import gc
from queue import Empty, Full
import time
import numpy as np

PHASES = ('starting','capture','preprocess','inference','publish','wait','stopped')


def put_latest(queue, packet):
    try:
        queue.put_nowait(packet)
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            pass
        try:
            queue.put_nowait(packet)
        except Full:
            pass  # the next camera frame will replace it; never block capture


# capture time, time-to-contact (s), distance (m), evidence flag, below_fraction (0..1), lower-surface TTC (s);
# NaN = none. below_fraction is the share of the expansion below the flight path (vision.looming2; None unless
# both the upper and the lower window have evidence).
LOOMING_FIELDS = ('time', 'ttc', 'distance', 'evidence', 'below_fraction', 'ttc_lower')
LOOMING_SLOTS = slice(732, 732+len(LOOMING_FIELDS))
SHARED_SIZE = LOOMING_SLOTS.stop


def looming_values(capture_time, result):
    """Shared-slot values for one looming result (NaN for None)."""
    value = lambda key: np.nan if result.get(key) is None else float(result[key])
    return [capture_time, value('ttc'), value('distance'), float(result['evidence']),
            value('below_fraction'), value('ttc_lower')]


def clearance_sample(values):
    """The FastRaceCue clearance sample from shared-slot values, or None when nothing was published."""
    stamp, ttc, distance, evidence, below, lower = (float(v) for v in values)
    if not stamp:
        return None
    finite = lambda v: float(v) if np.isfinite(v) else None
    return dict(time=stamp, ttc=finite(ttc), distance=finite(distance), evidence=bool(evidence),
                below_fraction=finite(below), ttc_lower=finite(lower))


def pose_at(rows, capture_time, now, max_age=.12):
    """Attitude and world velocity at a capture time from already observed motion rows.

    Rows are MotionBuffer rows: available, pose receipt, game time, position,
    quaternion (wxyz), velocity, request. Returns None when the motion is stale
    or the capture time lies outside the observed interval (no extrapolation
    beyond 40 ms).
    """
    if rows is None or len(rows) < 2 or now-rows[-1, 0] > max_age:
        return None
    times = rows[:, 1]
    if capture_time < times[0] or capture_time > times[-1]+.04:
        return None
    index = int(np.clip(np.searchsorted(times, capture_time), 1, len(rows)-1))
    t0, t1 = times[index-1], times[index]
    a = float(np.clip((capture_time-t0)/max(t1-t0, 1e-6), 0., 1.))
    q0, q1 = rows[index-1, 6:10], rows[index, 6:10]
    if q0 @ q1 < 0:
        q1 = -q1
    quaternion = (1-a)*q0+a*q1
    quaternion /= max(np.linalg.norm(quaternion), 1e-9)
    velocity = (1-a)*rows[index-1, 10:13]+a*rows[index, 10:13]
    return quaternion, velocity


def camera_worker(queue, data, done, phase, title, fps, gate_sensor, backend, race_cues=False,detector_device='cpu',
                  motion=None, looming=False):
    import torch
    import cv2
    from .visual_brain import RetinaCamera
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    queue.cancel_join_thread()
    camera = None
    priority = None
    last_diagnostics = 0.
    def publish(frame):
        nonlocal last_diagnostics
        stamp,retina,detection = frame
        with data.get_lock():
            shared = np.frombuffer(data.get_obj(),dtype=np.float64)
            shared[:7] = [stamp,detection is not None,detection['p'] if detection else 0.,
                          detection['width'] if detection else 0.,*(detection['point'] if detection else [0.,0.,0.])]
            shared[7:727] = retina.numpy().ravel()
            cue = detection.get('race_cue') if detection else None
            shared[727:732] = [cue is not None, cue['u'] if cue else 0.,
                            cue['v'] if cue else 0., cue['edge'] if cue else 0.,
                            cue.get('aim_u',cue['u']) if cue else 0.]
        if time.monotonic()-last_diagnostics>.5:
            put_latest(queue,dict(diagnostics=dict(camera.diagnostics(),priority=priority),error=camera.error))
            last_diagnostics = time.monotonic()
    estimator = None
    if looming:
        from ..vision.looming2 import LoomingEstimator2, Looming2Config
        estimator = LoomingEstimator2(Looming2Config(width=240, height=135, focal_320=gate_sensor['focal_320'],
                                                     tilt_deg=gate_sensor['tilt_deg']))

    def measure_looming(capture_time, image):
        # Runs after the cue is published, so it never delays checkpoint guidance.
        sample = pose_at(motion.read(), capture_time, time.monotonic())
        if sample is None:
            estimator.reset()
            return
        result = estimator.update(image, capture_time, *sample)
        if result is None:
            return
        values = looming_values(capture_time, result)
        with data.get_lock():
            np.frombuffer(data.get_obj(),dtype=np.float64)[LOOMING_SLOTS] = values
    try:
        from .scheduling import flight_process_priority
        priority = flight_process_priority()
        camera = RetinaCamera(title,fps,gate_sensor,backend,phase_status=phase,on_frame=publish,
                              race_cues=race_cues,detector_device=detector_device,
                              on_image=measure_looming if estimator is not None else None)
        camera.done = done
        gc.collect()
        gc.disable()
        camera.run()
        put_latest(queue,dict(diagnostics=dict(camera.diagnostics(),priority=priority),error=camera.error))
    except Exception as e:
        put_latest(queue,dict(error=repr(e)))
    finally:
        if camera is not None:
            if camera.capture is not None:
                camera.capture.close()


class ProcessRetinaCamera:
    def __init__(self,title='Liftoff',fps=24,gate_sensor=None,backend='mss',race_cues=False,detector_device='cpu',
                 looming=False):
        context = mp.get_context('spawn')
        self.queue = context.Queue(maxsize=2)
        self.data = context.Array('d',SHARED_SIZE,lock=True)
        self.done = context.Event()
        self.phase = context.Array('d',[0.,time.monotonic()],lock=False)
        self.looming = bool(looming)
        self.motion = None
        if self.looming:
            if not gate_sensor:
                raise ValueError('Looming needs the calibrated camera')
            from .geometry_shadow import MotionBuffer
            self.motion = MotionBuffer()
        self._clearance = None
        self.process = context.Process(target=camera_worker,
            args=(self.queue,self.data,self.done,self.phase,title,fps,gate_sensor,backend,race_cues,detector_device,
                  self.motion,self.looming),daemon=True)
        self.fps, self.backend = fps,backend
        self._latest = None
        self._error = None
        self._diagnostics = {}

    def start(self):
        self.process.start()
        return self

    def _poll(self):
        import torch
        # A single tiny, nonblocking snapshot replaces the queue/feeder relay.
        # Contention retains the previous timestamp; it never delays control.
        lock = self.data.get_lock()
        if lock.acquire(False):
            try:
                shared = np.frombuffer(self.data.get_obj(),dtype=np.float64)
                if shared[0] and (self._latest is None or shared[0]!=self._latest[0]):
                    snapshot = shared.copy()
                    detection = dict(p=snapshot[2],width=snapshot[3],point=snapshot[4:7]) if snapshot[1] else None
                    if len(snapshot) > 727 and snapshot[727] and detection is not None:
                        detection['race_cue'] = dict(u=snapshot[728],v=snapshot[729],edge=bool(snapshot[730]))
                        if len(snapshot) > 731:
                            detection['race_cue']['aim_u'] = snapshot[731]
                    self._latest = snapshot[0],torch.from_numpy(snapshot[7:727].astype(np.float32)[None]),detection
                if getattr(self,'looming',False) and len(shared) >= SHARED_SIZE:
                    sample = clearance_sample(shared[LOOMING_SLOTS])
                    if sample is not None and (self._clearance is None or sample['time'] != self._clearance['time']):
                        self._clearance = sample
            finally:
                lock.release()
        while True:
            try:
                packet = self.queue.get_nowait()
            except Empty:
                break
            if packet.get('diagnostics') is not None:
                self._diagnostics = packet['diagnostics']
            if packet.get('error'):
                self._error = packet['error']
        if self.process.exitcode is not None and not self.done.is_set() and self._error is None:
            self._error = f'Camera process exited ({self.process.exitcode})'

    @property
    def clearance(self):
        """Latest looming sample (capture time, ttc, distance, evidence, below_fraction, ttc_lower) or None."""
        self._poll()
        return self._clearance

    @property
    def latest(self):
        self._poll()
        return self._latest

    @property
    def error(self):
        self._poll()
        return self._error

    def diagnostics(self):
        self._poll()
        return dict(self._diagnostics,phase=PHASES[int(self.phase[0])],
                    phase_age_ms=1000*(time.monotonic()-self.phase[1]),
                    error=self._error,process_isolated=True,worker_pid=self.process.pid)

    def stop(self):
        self.done.set()
        self.process.join(timeout=4.)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2.)
        self.queue.close()
        self.queue.cancel_join_thread()
