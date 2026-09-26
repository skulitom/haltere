"""Keep camera decoding/detection outside the real-time controller process."""
from __future__ import annotations

import multiprocessing as mp
import gc
from queue import Empty, Full
import time
import numpy as np

PHASES = ('starting','capture','preprocess','inference','publish','wait','stopped','looming','gap')


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
# Gap cue (haltere.liftoff.gap_stack), one sample per processed frame: capture time, per-frame aim shift (deg,
# + = left of the ring; NOT confirmed), kind (gap_stack.kind_name), r_peak, r_ring, near_on_path (1/0/NaN),
# terrain side statistic lr, age (capture to publication, s), world azimuth of the ring (deg), valid, the
# cue's own two-frame confirmation (diagnostic), frame sequence number and its stage timings (ms).
GAP_FIELDS = ('time', 'shift', 'kind', 'r_peak', 'r_ring', 'near_on_path', 'lr', 'age', 'ring_deg', 'valid',
              'confirmed', 'seq', 'frame_ms', 'overlay_ms', 'depth_ms', 'decide_ms')
GAP_SLOTS = slice(LOOMING_SLOTS.stop, LOOMING_SLOTS.stop+len(GAP_FIELDS))
# Camera loop stage timings of the latest frame (ms): written at the end of every camera iteration.
# cue_latency = capture start to the checkpoint cue in shared memory; gap = gap work inside the camera loop.
STAGE_FIELDS = ('frame_time', 'capture', 'preprocess', 'inference', 'publish', 'looming', 'gap', 'total',
                'cue_latency')
STAGE_SLOTS = slice(GAP_SLOTS.stop, GAP_SLOTS.stop+len(STAGE_FIELDS))
SHARED_SIZE = STAGE_SLOTS.stop


def gap_sample(values):
    """The pilot's gap sample from shared-slot values, or None when nothing was published."""
    from .gap_stack import kind_name
    raw = dict(zip(GAP_FIELDS, (float(v) for v in values)))
    if not raw['time']:
        return None
    finite = lambda v: v if np.isfinite(v) else None
    sample = {k: finite(v) for k, v in raw.items()}
    sample.update(time=raw['time'], shift=raw['shift'] if np.isfinite(raw['shift']) else 0.,
                  kind=kind_name(raw['kind']), valid=bool(raw['valid']), confirmed=bool(raw['confirmed']),
                  near_on_path=None if not np.isfinite(raw['near_on_path']) else bool(raw['near_on_path']))
    return sample


def cue_for(data, capture_time):
    """The camera's published race cue for one capture time: ('published', cue dict or None) once the camera
    process has published that frame, ('pending', None) before, ('replaced', None) when a later frame is there."""
    lock = data.get_lock()
    if not lock.acquire(timeout=.002):
        return 'pending', None
    try:
        shared = np.frombuffer(data.get_obj(), dtype=np.float64)
        stamp, row = float(shared[0]), shared[727:732].copy()
    finally:
        lock.release()
    if stamp < capture_time:
        return 'pending', None
    if stamp > capture_time:
        return 'replaced', None
    return 'published', (dict(u=float(row[1]), v=float(row[2]), edge=bool(row[3]), aim_u=float(row[4]))
                         if row[0] else None)


def stage_sample(values):
    """Latest camera stage timings (ms) with the capture time they belong to, or None."""
    raw = dict(zip(STAGE_FIELDS, (float(v) for v in values)))
    return raw if raw['frame_time'] else None


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
                  motion=None, looming=False, gap=None, frame_slot=None):
    import torch
    import cv2
    from .visual_brain import RetinaCamera
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    queue.cancel_join_thread()
    camera = None
    priority = None
    last_diagnostics = 0.
    gap_worker = None
    gap_frames = 0
    def diagnostics():
        result = dict(camera.diagnostics(),priority=priority)
        if gap_worker is not None:
            result['gap'] = dict(gap_worker.status(),placement='camera')
        return result
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
            put_latest(queue,dict(diagnostics=diagnostics(),error=camera.error))
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

    def hand_off(capture_time, rgb):
        # process placement: the depth process gets the frame right after capture (a copy, no resize here;
        # a window larger than the slot is resized to the 448 x 252 model frame first)
        if not frame_slot.fits(rgb):
            from ..obstacles.overlays import to_model_frame
            rgb = to_model_frame(rgb)
        frame_slot.write(rgb, capture_time)

    def measure_gap(capture_time, rgb, detection):
        # camera placement: runs after the cue (and looming) are published, so it never delays checkpoint guidance.
        nonlocal gap_frames
        from ..obstacles.overlays import to_model_frame
        from .gap_stack import read_motion, write_gap
        gap_frames += 1
        if gap['stride'] > 1 and gap_frames % gap['stride']:
            return
        begin = time.monotonic()
        frame = to_model_frame(rgb)
        cue = detection.get('race_cue') if detection else None
        pose = pose_at(read_motion(motion), capture_time, time.monotonic())
        write_gap(data, gap_worker.process(frame, capture_time, cue, pose, frame_ms=1000*(time.monotonic()-begin),
                                           seq=gap_frames))

    def stages(capture_time, values):
        with data.get_lock():
            np.frombuffer(data.get_obj(),dtype=np.float64)[STAGE_SLOTS] = [capture_time,*(1000*v for v in values)]
    try:
        from .scheduling import flight_process_priority
        priority = flight_process_priority()
        if gap is not None and frame_slot is None:
            from .gap_stack import GapFrameWorker
            gap_worker = GapFrameWorker(gap)
            put_latest(queue,dict(gap=dict(gap_worker.status(),placement='camera')))
        camera = RetinaCamera(title,fps,gate_sensor,backend,phase_status=phase,on_frame=publish,
                              race_cues=race_cues,detector_device=detector_device,
                              on_image=measure_looming if estimator is not None else None,
                              on_capture=measure_gap if gap is not None and frame_slot is None else None,
                              on_raw=hand_off if frame_slot is not None else None,on_stages=stages)
        camera.done = done
        gc.collect()
        gc.disable()
        camera.run()
        put_latest(queue,dict(diagnostics=diagnostics(),error=camera.error))
    except Exception as e:
        put_latest(queue,dict(error=repr(e)))
    finally:
        if camera is not None:
            if camera.capture is not None:
                camera.capture.close()


class ProcessRetinaCamera:
    def __init__(self,title='Liftoff',fps=24,gate_sensor=None,backend='mss',race_cues=False,detector_device='cpu',
                 looming=False,gap=None):
        context = mp.get_context('spawn')
        self.queue = context.Queue(maxsize=2)
        self.data = context.Array('d',SHARED_SIZE,lock=True)
        self.done = context.Event()
        self.phase = context.Array('d',[0.,time.monotonic()],lock=False)
        self.looming = bool(looming)
        # gap: a haltere.liftoff.gap_stack.gap_spec dict (None: off)
        self.gap_spec = dict(gap) if gap else None
        self.motion = None
        if self.looming or self.gap_spec:
            if not gate_sensor:
                raise ValueError('Looming and the gap cue need the calibrated camera')
            if self.gap_spec and not race_cues:
                raise ValueError('The gap cue needs the checkpoint-ring cue')
            from .geometry_shadow import MotionBuffer
            self.motion = MotionBuffer()
        self._clearance = None
        self._gap = None
        self._stages = None
        self._gap_status = {}
        self.frame_slot = self.gap_process = self.gap_queue = None
        if self.gap_spec and self.gap_spec['placement'] == 'process':
            from .gap_stack import FrameSlot, gap_process_worker
            self.frame_slot = FrameSlot()
            self.gap_queue = context.Queue(maxsize=2)
            self.gap_process = context.Process(target=gap_process_worker,
                args=(self.frame_slot,self.data,self.motion,self.done,self.gap_queue,self.gap_spec),daemon=True)
        self.process = context.Process(target=camera_worker,
            args=(self.queue,self.data,self.done,self.phase,title,fps,gate_sensor,backend,race_cues,detector_device,
                  self.motion,self.looming,self.gap_spec,self.frame_slot),daemon=True)
        self.fps, self.backend = fps,backend
        self._latest = None
        self._error = None
        self._diagnostics = {}

    def start(self):
        if self.gap_process is not None:
            self.gap_process.start()
        self.process.start()
        return self

    def wait_ready(self, timeout=120.):
        """Wait until the gap cue has loaded and warmed its depth model (nothing to wait for without it)."""
        end = time.monotonic()+timeout
        while self.gap_spec and not self._gap_status.get('ready'):
            self._poll()
            if self._error:
                raise RuntimeError(f'Camera or gap cue failed to start: {self._error}')
            if time.monotonic() > end:
                raise RuntimeError('Gap cue depth model not ready in time')
            time.sleep(.05)
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
                if len(shared) >= SHARED_SIZE:
                    if getattr(self,'gap_spec',None) and shared[GAP_SLOTS.start] and (
                            self._gap is None or shared[GAP_SLOTS.start+GAP_FIELDS.index('seq')] != self._gap['seq']
                            or shared[GAP_SLOTS.start] != self._gap['time']):
                        self._gap = gap_sample(shared[GAP_SLOTS])
                    if shared[STAGE_SLOTS.start] and (getattr(self,'_stages',None) is None
                                                      or shared[STAGE_SLOTS.start] != self._stages['frame_time']):
                        self._stages = stage_sample(shared[STAGE_SLOTS])
            finally:
                lock.release()
        for queue in (self.queue, getattr(self,'gap_queue',None)):
            while queue is not None:
                try:
                    packet = queue.get_nowait()
                except Empty:
                    break
                if packet.get('diagnostics') is not None:
                    self._diagnostics = packet['diagnostics']
                    if packet['diagnostics'].get('gap'):
                        self._gap_status = packet['diagnostics']['gap']
                if packet.get('gap') is not None:
                    self._gap_status = packet['gap']
                if packet.get('error'):
                    self._error = packet['error']
        if self.process.exitcode is not None and not self.done.is_set() and self._error is None:
            self._error = f'Camera process exited ({self.process.exitcode})'
        gap_process = getattr(self,'gap_process',None)
        if (gap_process is not None and gap_process.exitcode is not None and not self.done.is_set()
                and self._error is None):
            self._error = f'Gap depth process exited ({gap_process.exitcode})'

    @property
    def gap(self):
        """Latest gap-cue sample (see gap_sample) or None."""
        self._poll()
        return self._gap

    @property
    def stages(self):
        """Camera stage timings (ms) of the latest frame, or None."""
        self._poll()
        return self._stages

    def gap_status(self):
        self._poll()
        return dict(self._gap_status)

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
        result = dict(self._diagnostics,phase=PHASES[int(self.phase[0])],
                      phase_age_ms=1000*(time.monotonic()-self.phase[1]),
                      error=self._error,process_isolated=True,worker_pid=self.process.pid)
        if getattr(self,'gap_spec',None):
            result['gap'] = dict(self._gap_status, placement=self.gap_spec['placement'], stride=self.gap_spec['stride'],
                                 depth_worker_pid=self.gap_process.pid if self.gap_process is not None else None)
        return result

    def stop(self):
        self.done.set()
        for process in (self.process, getattr(self,'gap_process',None)):
            if process is None or process.pid is None:
                continue
            process.join(timeout=4.)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.)
        for queue in (self.queue, getattr(self,'gap_queue',None)):
            if queue is not None:
                queue.close()
                queue.cancel_join_thread()
