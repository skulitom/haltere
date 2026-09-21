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


def camera_worker(queue, data, done, phase, title, fps, gate_sensor, backend):
    import torch
    from .visual_brain import RetinaCamera
    torch.set_num_threads(2)
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
            shared[7:] = retina.numpy().ravel()
        if time.monotonic()-last_diagnostics>.5:
            put_latest(queue,dict(diagnostics=dict(camera.diagnostics(),priority=priority),error=camera.error))
            last_diagnostics = time.monotonic()
    try:
        from .scheduling import flight_process_priority
        priority = flight_process_priority()
        camera = RetinaCamera(title,fps,gate_sensor,backend,phase_status=phase,on_frame=publish)
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
    def __init__(self,title='Liftoff',fps=24,gate_sensor=None,backend='mss'):
        context = mp.get_context('spawn')
        self.queue = context.Queue(maxsize=2)
        self.data = context.Array('d',727,lock=True)
        self.done = context.Event()
        self.phase = context.Array('d',[0.,time.monotonic()],lock=False)
        self.process = context.Process(target=camera_worker,
            args=(self.queue,self.data,self.done,self.phase,title,fps,gate_sensor,backend),daemon=True)
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
                    self._latest = snapshot[0],torch.from_numpy(snapshot[7:].astype(np.float32)[None]),detection
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
