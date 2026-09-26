"""Gap cue in the flight stack (runtime): frozen relative depth + `haltere.vision.gap_cue` on camera frames.

Off unless a runner passes a gap spec to `ProcessRetinaCamera` (``--obstacle-stack``). Everything here is causal:
the current frame, the checkpoint-ring cue detected in that frame, and the pose at its capture time interpolated
from telemetry the controller has already observed (`MotionBuffer`). No course geometry, routes, per-course
parameters or offline labels (this module is covered by tests/test_obstacle_label_isolation.py).

Per processed frame:
1. the captured gameplay image resized to 448 x 252 with cv2.INTER_AREA (`overlays.to_model_frame`, as the
   obstacle store and the gap-cue gates);
2. with an in-view ring cue (centre u, v; not edge-clamped; the camera placement runs depth only then): overlay masks (the ``mask_layers`` of
   configs/obstacles/gap_cue.json) -> block validity; relative disparity from Depth-Anything-V2-Small
   (`haltere.vision.relative_depth`, frozen weights, fp16 on CUDA, input size from gap_cue.json
   ``relative_depth.input_hw``); `GapCue.update` with the capture attitude and world velocity (its own two-frame
   confirmation is logged as a diagnostic; the pilot confirms the per-frame shift itself, `gap_aim`);
3. one sample of `camera_process.GAP_FIELDS` written into the camera's shared slots.

Placement (configs/obstacles/gap_pilot.json ``runtime.placement``):
- ``camera``: in the camera process, after the checkpoint cue and looming (the `RetinaCamera.on_capture` hook);
- ``process``: a separate depth process fed by a one-frame slot (`FrameSlot`). The camera process only copies
  the captured frame into the slot right after capture; the resize, masks and depth run while the camera
  process computes the checkpoint cue, and the (cheap) decision waits for that frame's cue, which the camera
  also copies into the slot once published. Depth then runs on every frame, also without an in-view ring.
  The depth process sets its own priority class to above normal (`scheduling.flight_process_priority`, as the
  camera process does): spawned from a runner launched at below normal (Anode), it would otherwise inherit
  below normal. The camera never waits for it: the depth process takes no lock or event of the camera's (it
  writes its samples to its own shared array, `gap_out`, read by the controller without blocking, and stops on
  its own event), and the camera's side of the slot uses non-blocking lock attempts and semaphore signals only.
``runtime.stride`` 2 processes every other frame (by frame sequence number).
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GAP_PILOT_PATH = REPO_ROOT/'configs'/'obstacles'/'gap_pilot.json'
FRAME_SHAPE = (252, 448, 3)
GRID = (36, 64)
CONFIG_META_KEYS = ('frozen', 'frozen_at', 'sha256')
# The gap pilot declaration version this code implements (version 2: the lag-turn lead is computed on the ring
# cue's bearing with the gap shift removed and the shift added after it; the depth process runs above normal and
# the camera never waits for it). Other versions are refused.
GAP_PILOT_VERSION = 2


def _kinds():
    from ..vision.gap_cue import KINDS
    return KINDS+('no_pose',)


def kind_index(kind):
    return float(_kinds().index(kind))


def kind_name(index):
    kinds = _kinds()
    if index is None or not np.isfinite(index) or not 0 <= int(index) < len(kinds):
        return ''
    return kinds[int(index)]


def config_sha256(obj):
    body = {k: v for k, v in obj.items() if k not in CONFIG_META_KEYS}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode('utf-8')).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_gap_pilot(path=GAP_PILOT_PATH, *, require_frozen=True):
    """The gap pilot declaration and its content sha256; refuses an unfrozen or edited file (for flights) and a
    declaration version other than the one this code implements (`GAP_PILOT_VERSION`)."""
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = config_sha256(obj)
    if obj.get('frozen') and obj.get('sha256') != digest:
        raise ValueError(f'{path} changed after the freeze (sha256 mismatch)')
    if require_frozen and obj.get('frozen') is not True:
        raise ValueError(f'{path} is not frozen')
    if obj.get('version') != GAP_PILOT_VERSION:
        raise ValueError(f'{path} declares gap pilot version {obj.get("version")}; this code implements version '
                         f'{GAP_PILOT_VERSION}')
    return obj, digest


def gap_spec(declaration, contract, sensor, *, path=GAP_PILOT_PATH, digest=None):
    """The camera-side spec (picklable dict) for a motor contract from a gap pilot declaration."""
    from ..vision import gap_cue as gc
    runtime = declaration['runtime']
    motor = runtime['response_model_for_contract'].get(contract)
    if motor is None:
        raise ValueError(f'The gap pilot declares no response model for the {contract} motor contract')
    if runtime['placement'] not in ('camera', 'process') or int(runtime['stride']) not in (1, 2):
        raise ValueError('Gap placement is camera or process, stride 1 or 2')
    cue_path = REPO_ROOT/runtime['gap_cue_config']
    _, cue_config, cue_sha = gc.load_config(cue_path, require_frozen=True)
    models_path = REPO_ROOT/runtime['response_models']
    if motor not in gc.load_response_models(models_path):
        raise ValueError(f'{models_path} has no {motor} response model')
    if not sensor or not {'focal_320', 'tilt_deg'} <= set(sensor):
        raise ValueError('The gap cue needs the calibrated camera')
    return dict(placement=runtime['placement'], stride=int(runtime['stride']), device=runtime.get('device', 'cuda'),
                gap_cue_config=str(cue_path), gap_cue_sha256=cue_sha, gap_cue_version=cue_config.get('version'),
                gap_cue_file_sha256=file_sha256(cue_path), response_models=str(models_path),
                response_models_file_sha256=file_sha256(models_path), motor=motor, motor_contract=contract,
                input_hw=[int(v) for v in cue_config['relative_depth']['input_hw']],
                focal_320=float(sensor['focal_320']), tilt_deg=float(sensor['tilt_deg']),
                gap_pilot_config=str(path), gap_pilot_sha256=digest, gap_pilot_version=declaration.get('version'))


def gap_camera(spec):
    from ..vision.camera import Camera
    return Camera(FRAME_SHAPE[1], FRAME_SHAPE[0], spec['focal_320']*FRAME_SHAPE[1]/320., spec['tilt_deg'])


class GapFrameWorker:
    """One frame (448 x 252 RGB, capture time, ring cue, pose) -> one gap sample (`camera_process.GAP_FIELDS`)."""

    def __init__(self, spec, *, depth=None, warmup=3):
        from ..vision import gap_cue as gc
        params, cue_config, sha = gc.load_config(spec['gap_cue_config'], require_frozen=depth is None)
        self.params = replace(params, enabled=True, motor=spec['motor'])
        self.layers = tuple(cue_config['mask_layers'])
        models = gc.load_response_models(spec['response_models'])
        self.cue = gc.GapCue(self.params, gap_camera(spec), models[spec['motor']])
        if depth is None:
            from ..vision.relative_depth import RelativeDepth
            depth = RelativeDepth(device=spec.get('device', 'cuda'), fp16=True, input_hw=tuple(spec['input_hw']))
            if depth.device.type != 'cuda':
                raise RuntimeError('The gap cue needs CUDA: on the CPU (about 0.3 s per frame) its samples go stale')
            blank = np.full(FRAME_SHAPE, 128, np.uint8)
            for _ in range(int(warmup)):
                depth(blank)
            depth.torch.cuda.synchronize()
        self.depth = depth
        self.provenance = dict(getattr(depth, 'provenance', {}), gap_cue_config_sha256=sha,
                               mask_layers=list(self.layers), motor=spec['motor'])
        self.frames = self.depth_frames = 0
        self.timings = deque(maxlen=4096)

    def perceive(self, frame, clock=time.monotonic):
        """Overlay-mask block validity and block relative disparity of one 448 x 252 frame (no cue needed)."""
        from ..obstacles import overlays as ov
        from ..vision import gap_cue as gc
        t0 = clock()
        masks = ov.overlay_masks(frame)
        masked = np.zeros(frame.shape[:2], bool)
        for name in self.layers:
            masked |= getattr(masks, name)
        valid = gc.block_validity(masked, GRID, self.params.block_mask_max_fraction)
        t1 = clock()
        disparity = np.asarray(self.depth(frame)[0], np.float32)
        t2 = clock()
        self.depth_frames += 1
        return dict(valid=valid, disparity=disparity, overlay_ms=1000*(t1-t0), depth_ms=1000*(t2-t1))

    def in_view(self, cue, pose):
        """The ring cue as (u, v) when it is in view and the pose gives it a bearing, else None."""
        from ..vision import gap_cue as gc
        if pose is None or not cue or cue.get('edge'):
            return None
        cue_uv = (float(cue['u']), float(cue['v']))
        if gc.ring_bearing_deg(cue_uv, pose[0], self.cue.camera, self.params.cue_edge_margin) is None:
            return None
        return cue_uv

    def process(self, frame, capture_time, cue, pose, *, frame_ms=float('nan'), seq=0, clock=time.monotonic,
                perceived=None):
        """Gap values for one frame; ``cue`` is the race-cue dict of the same frame (or None), ``pose``
        (quaternion wxyz, world velocity) at the capture time or None. Depth runs only for an in-view ring
        unless ``perceived`` (from `perceive`, run before the cue was known) is given."""
        from .camera_process import GAP_FIELDS
        begin = clock()
        inner = 0.
        cue_uv = self.in_view(cue, pose)
        if pose is None:
            values = self._values(capture_time, None, None, 'no_pose')
        else:
            quaternion, velocity = pose
            if cue_uv is not None and perceived is None:
                perceived = self.perceive(frame, clock)
                inner = perceived['overlay_ms']+perceived['depth_ms']
            use = perceived if cue_uv is not None else None
            out = self.cue.update(None if use is None else use['disparity'], quaternion, cue_uv, float(capture_time),
                                  clock(), velocity=velocity, valid=None if use is None else use['valid'])
            values = self._values(capture_time, out.decision, out, out.kind)
        end = clock()
        overlay_ms = perceived['overlay_ms'] if perceived else float('nan')
        depth_ms = perceived['depth_ms'] if perceived else float('nan')
        values.update(seq=float(seq), frame_ms=float(frame_ms), overlay_ms=overlay_ms, depth_ms=depth_ms,
                      decide_ms=1000*(end-begin)-inner, age=end-float(capture_time))   # age: capture to publication
        self.frames += 1
        self.timings.append((frame_ms, overlay_ms, depth_ms, values['decide_ms'], 1000*values['age']))
        return [values[k] for k in GAP_FIELDS]

    @staticmethod
    def _values(capture_time, decision, output, kind):
        nan = float('nan')
        d = decision
        valid = d is not None and d.valid
        near = nan if d is None or d.near_on_path is None else float(d.near_on_path)
        return dict(time=float(capture_time), shift=float(d.shift_deg) if valid else 0., kind=kind_index(kind),
                    r_peak=float(d.r_peak) if valid else nan, r_ring=float(d.r_ring) if valid else nan,
                    near_on_path=near, lr=float(d.lr) if valid else nan,
                    ring_deg=nan if d is None or d.ring_bearing_deg is None else float(d.ring_bearing_deg),
                    valid=float(valid), confirmed=float(bool(output is not None and output.confirmed)))

    def status(self):
        rows = np.asarray(list(self.timings), float)
        out = dict(ready=True, frames=self.frames, depth_frames=self.depth_frames, provenance=self.provenance)
        if len(rows):
            out['timings_ms'] = {name: dict(p50=_pct(rows[:, i], 50), p95=_pct(rows[:, i], 95), max=_pct(rows[:, i], 100))
                                 for i, name in enumerate(('frame', 'overlay', 'depth', 'decide', 'latency'))}
        return out


def _pct(values, q):
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if len(values) else None


def _bgra_base(frame):
    """The contiguous BGRA buffer behind an mss screen grab's RGB view (``grab[:, :, :3][:, :, ::-1]``), or None.

    That view reads each 4-byte pixel backwards, so a plain copy is a slow strided one; reversing it back gives
    the B byte of pixel 0 at the buffer start, and the (h, w, 4) layout with 4-byte pixels is the grab itself."""
    h, w, _ = frame.shape
    if frame.strides != (4*w, 4, -1):
        return None
    from numpy.lib.stride_tricks import as_strided
    bgra = as_strided(frame[:, :, ::-1], shape=(h, w, 4), strides=(4*w, 4, 1), writeable=False)
    return bgra if bgra.flags.c_contiguous else None


class FrameSlot:
    """Camera -> depth process hand-off of the ``process`` placement: the newest captured frame (any size up to
    1920 x 1080 RGB), written right after capture, and that frame's checkpoint-ring cue, written once the camera
    has published it (`publish_cue`).

    The camera side never waits for the depth process. It takes the slot's locks without blocking (a lock the
    depth process holds skips that frame or cue, counted in ``skipped``) and signals with semaphore releases,
    which never block. A multiprocessing Event is not used for the signals: its set() takes a lock that the
    waiting process also takes, then waits until each sleeping waiter has woken, so a starved depth process
    would stall the camera. The depth process waits on the signals with timeouts, and takes the locks with
    short timeouts (the camera holds them only while it copies)."""
    HEADER = ('seq', 'capture_time', 'height', 'width', 'written_at')
    CUE = ('capture_time', 'present', 'u', 'v', 'edge', 'aim_u')
    MAX_SHAPE = (1080, 1920, 3)

    def __init__(self):
        context = mp.get_context('spawn')
        self.pixels = context.RawArray('B', int(np.prod(self.MAX_SHAPE)))
        self.header = context.RawArray('d', len(self.HEADER))
        self.lock = context.Lock()
        self.frame_signal = context.Semaphore(0)
        self.cue = context.RawArray('d', len(self.CUE))
        self.cue_lock = context.Lock()
        self.cue_signal = context.Semaphore(0)
        self.skipped = context.RawArray('q', 2)     # frames, cues the camera skipped: the depth side held the lock

    def fits(self, frame):
        frame = np.asarray(frame)
        return (frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
                and frame.shape[0] <= self.MAX_SHAPE[0] and frame.shape[1] <= self.MAX_SHAPE[1])

    def write(self, frame, capture_time, now=None):
        """Camera side: copy a captured frame into the slot (never waits; False when the slot was busy)."""
        frame = np.asarray(frame)
        if not self.fits(frame):
            raise ValueError('expected an RGB uint8 frame of at most 1920 x 1080')
        if not self.lock.acquire(False):
            self.skipped[0] += 1
            return False
        try:
            target = np.frombuffer(self.pixels, np.uint8, count=frame.size).reshape(frame.shape)
            bgra = _bgra_base(frame)
            if bgra is not None:
                import cv2
                cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB, dst=target)      # 0.04 ms instead of a 2 ms strided copy
            else:
                target[:] = frame
            header = np.frombuffer(self.header)
            header[1:] = [capture_time, frame.shape[0], frame.shape[1], time.monotonic() if now is None else now]
            header[0] += 1
        finally:
            self.lock.release()
        self.frame_signal.release()
        return True

    def publish_cue(self, capture_time, cue):
        """Camera side: the checkpoint-ring cue (dict or None) published for a capture time (never waits; False
        when the depth side held the cue lock)."""
        if not self.cue_lock.acquire(False):
            self.skipped[1] += 1
            return False
        try:
            np.frombuffer(self.cue)[:] = [capture_time, cue is not None, cue['u'] if cue else 0.,
                                          cue['v'] if cue else 0., cue['edge'] if cue else 0.,
                                          cue.get('aim_u', cue['u']) if cue else 0.]
        finally:
            self.cue_lock.release()
        self.cue_signal.release()
        return True

    def wait_frame(self, timeout):
        """Depth side: wait up to `timeout` s for a frame signal (then clear any further signals)."""
        if not self.frame_signal.acquire(True, timeout):
            return False
        while self.frame_signal.acquire(False):
            pass
        return True

    def read(self, last_seq=0):
        """Depth side: (seq, frame copy, capture time, written_at) for a newer frame, else None."""
        if not self.lock.acquire(timeout=.005):
            return None
        try:
            header = np.frombuffer(self.header).copy()
            if header[0] <= last_seq:
                return None
            shape = (int(header[2]), int(header[3]), 3)
            frame = np.frombuffer(self.pixels, np.uint8, count=int(np.prod(shape))).reshape(shape).copy()
        finally:
            self.lock.release()
        return int(header[0]), frame, float(header[1]), float(header[4])

    def cue_for(self, capture_time):
        """Depth side: ('published', cue dict or None) once the camera has published the cue of this capture time,
        ('pending', None) before, ('replaced', None) when a later frame's cue is there."""
        if not self.cue_lock.acquire(timeout=.002):
            return 'pending', None
        try:
            row = np.frombuffer(self.cue).copy()
        finally:
            self.cue_lock.release()
        if row[0] < capture_time:
            return 'pending', None
        if row[0] > capture_time:
            return 'replaced', None
        return 'published', (dict(u=float(row[2]), v=float(row[3]), edge=bool(row[4]), aim_u=float(row[5]))
                             if row[1] else None)

    def skips(self):
        return dict(frames=int(self.skipped[0]), cues=int(self.skipped[1]))


def read_motion(motion, tries=4):
    """MotionBuffer rows; a read that meets the writer's lock is retried after 0.5 ms (None after `tries`)."""
    for attempt in range(tries):
        rows = motion.read()
        if rows is not None:
            return rows
        time.sleep(.0005)
    return None


def write_gap(data, values, slots=None):
    """One gap sample into a shared array: the camera's GAP_SLOTS (camera placement) or all of the depth
    process's own `gap_out` array (``slots=slice(None)``)."""
    from .camera_process import GAP_SLOTS
    with data.get_lock():
        np.frombuffer(data.get_obj(), dtype=np.float64)[GAP_SLOTS if slots is None else slots] = values


def wait_for_cue(slot, capture_time, stop, timeout=.15):
    """Depth side: the race cue the camera published into the slot for this capture time: (True, cue or None) once
    published, (False, None) when a later frame replaced it first, the timeout passed or the stack stopped. Waits
    on the slot's cue signal (a semaphore), never on a lock the camera needs."""
    end = time.monotonic()+timeout
    while slot.cue_signal.acquire(False):
        pass                       # stale signals; the state is checked before every wait
    while not stop.is_set():
        state, cue = slot.cue_for(capture_time)
        if state == 'published':
            return True, cue
        left = end-time.monotonic()
        if state == 'replaced' or left <= 0:
            return False, None
        slot.cue_signal.acquire(True, min(left, .02))
    return False, None


def gap_process_worker(slot, gap_out, motion, stop, status, spec):
    """The separate depth process of the ``process`` placement: the frame arrives right after capture, so the
    resize, overlay masks and depth run while the camera process computes the checkpoint cue; the cheap
    decision waits for that cue. It raises its own priority class (as the camera process does), writes its
    samples to its own shared array and stops on its own event: no lock or event of the camera's is taken."""
    import cv2
    import torch
    from ..obstacles.overlays import to_model_frame
    from .camera_process import pose_at, put_latest
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    status.cancel_join_thread()
    worker = None
    priority = None
    last = 0
    counts = dict(skipped_frames=0, cue_missed=0, cue_wait_ms=deque(maxlen=4096))

    def report():
        waits = np.asarray(counts['cue_wait_ms'], float)
        return dict(gap=dict(worker.status() if worker is not None else dict(ready=False), placement='process',
                             priority=priority, camera_skips=slot.skips(),
                             skipped_frames=counts['skipped_frames'], cue_missed=counts['cue_missed'],
                             cue_wait_ms=dict(p50=_pct(waits, 50), p95=_pct(waits, 95)) if len(waits) else None))
    last_status = 0.
    try:
        from .scheduling import flight_process_priority
        # Spawned before the runner raises its own class, this process starts with the runner's inherited class
        # (below normal when launched in Anode). Set it explicitly, as the camera process does.
        priority = flight_process_priority()
        worker = GapFrameWorker(spec)
        put_latest(status, report())
        while not stop.is_set():
            if not slot.wait_frame(.1):
                continue
            item = slot.read(last)
            if item is None:
                continue
            seq, rgb, capture_time, written_at = item
            counts['skipped_frames'] += max(0, seq-last-1) if last else 0
            last = seq
            if spec['stride'] > 1 and seq % spec['stride']:
                continue
            t0 = time.monotonic()
            frame = to_model_frame(rgb)
            frame_ms = 1000*(time.monotonic()-t0)
            perceived = worker.perceive(frame)
            t1 = time.monotonic()
            published, cue = wait_for_cue(slot, capture_time, stop)
            counts['cue_wait_ms'].append(1000*(time.monotonic()-t1))
            if not published:
                counts['cue_missed'] += 1
            pose = pose_at(read_motion(motion), capture_time, time.monotonic())
            values = worker.process(frame, capture_time, cue, pose, frame_ms=frame_ms, seq=seq, perceived=perceived)
            write_gap(gap_out, values, slice(None))
            if time.monotonic()-last_status > .5:
                put_latest(status, report())
                last_status = time.monotonic()
        put_latest(status, report())
    except Exception as e:
        put_latest(status, dict(report(), error=repr(e)))
