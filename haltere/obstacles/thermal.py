"""Flight lock, GPU temperature and thread limits for heavy offline obstacle jobs.

Every heavy job (video decoding, triangulation, teacher inference, training, long
evaluation loops) must be chunked and resumable and must call
``ChunkGuard.before_chunk()`` before each chunk:

1. Flight lock: while the lock file exists (the operator is flying a live test),
   sleep in 20 s steps without using CPU or GPU, then resume. The job never
   creates or deletes the lock file. Its path comes from ``--flight-lock`` or the
   ``HALTERE_FLIGHT_LOCK`` environment variable; heavy CLIs refuse to start without one
   (``require_flight_lock_path``) so a forgotten lock cannot silently disable the pause.
2. GPU jobs only: ``haltere.train.thermal.wait_if_hot(75, 65)`` (pause at > 75 C until
   <= 65 C), then a hard stop (``GpuTooHot``) if the GPU still reads >= 80 C.
3. Chunk length: GPU chunks are at most 10 minutes; ``chunk_expired()`` tells the
   loop when to checkpoint and call ``before_chunk()`` again.

``limit_threads(2)`` sets OMP/MKL/OpenBLAS, torch and OpenCV to two threads. Run at
most one heavy process per agent.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

FLIGHT_LOCK_ENV = 'HALTERE_FLIGHT_LOCK'
FLIGHT_LOCK_POLL_S = 20.0
GPU_PAUSE_C = 75.0
GPU_RESUME_C = 65.0
GPU_HARD_STOP_C = 80.0
GPU_CHUNK_MAX_S = 600.0
THREADS = 2


class GpuTooHot(RuntimeError):
    """The GPU is at or above the hard-stop temperature: checkpoint and stop the job."""


def limit_threads(n: int = THREADS, *, torch: bool = False, cv2: bool = False) -> None:
    """Limit BLAS/OpenMP (environment), and optionally torch and OpenCV, to ``n`` threads."""
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(n)
    if torch:
        import torch as _torch
        _torch.set_num_threads(n)
        try:
            _torch.set_num_interop_threads(n)
        except RuntimeError:
            pass   # already set once in this process
    if cv2:
        import cv2 as _cv2
        _cv2.setNumThreads(n)


def flight_lock_path(explicit: str | os.PathLike | None = None) -> Path | None:
    """The lock file to honour: the explicit argument, else $HALTERE_FLIGHT_LOCK, else None."""
    if explicit:
        return Path(explicit)
    env = os.environ.get(FLIGHT_LOCK_ENV)
    return Path(env) if env else None


def require_flight_lock_path(explicit: str | os.PathLike | None = None) -> Path:
    path = flight_lock_path(explicit)
    if path is None:
        raise SystemExit(f'Heavy obstacle jobs need a flight lock path: pass --flight-lock PATH or set '
                         f'{FLIGHT_LOCK_ENV}')
    return path


def wait_for_flight_lock(path: str | os.PathLike | None, poll_s: float = FLIGHT_LOCK_POLL_S,
                         log=print, sleep=time.sleep) -> float:
    """Sleep in ``poll_s`` steps while the lock file exists; return the seconds waited. Never touches the file."""
    if path is None:
        return 0.0
    path = Path(path)
    waited = 0.0
    announced = False
    while path.exists():
        if not announced:
            log(f'flight lock {path} present: pausing (polling every {poll_s:.0f} s)')
            announced = True
        sleep(poll_s)
        waited += poll_s
    if announced:
        log(f'flight lock released after {waited:.0f} s: resuming')
    return waited


def gpu_temperature() -> float | None:
    from ..train.thermal import gpu_temperature as _t
    return _t()


class ChunkGuard:
    """Call ``before_chunk()`` before every chunk of a heavy job (see module docstring)."""

    def __init__(self, flight_lock: str | os.PathLike | None, *, gpu: bool = False,
                 pause_c: float = GPU_PAUSE_C, resume_c: float = GPU_RESUME_C,
                 hard_stop_c: float = GPU_HARD_STOP_C, chunk_max_s: float | None = None,
                 log=print, sleep=time.sleep, temperature=None):
        self.flight_lock = None if flight_lock is None else Path(flight_lock)
        self.gpu = gpu
        self.pause_c, self.resume_c, self.hard_stop_c = pause_c, resume_c, hard_stop_c
        self.chunk_max_s = (GPU_CHUNK_MAX_S if gpu else None) if chunk_max_s is None else chunk_max_s
        if gpu and self.chunk_max_s > GPU_CHUNK_MAX_S:
            raise ValueError(f'GPU chunks are limited to {GPU_CHUNK_MAX_S:.0f} s')
        self.log, self.sleep = log, sleep
        self._temperature = temperature or gpu_temperature
        self.chunk_start = None
        self.lock_wait_s = 0.0
        self.heat_wait_s = 0.0
        self.chunks = 0

    def before_chunk(self) -> None:
        self.lock_wait_s += wait_for_flight_lock(self.flight_lock, log=self.log, sleep=self.sleep)
        if self.gpu:
            from ..train.thermal import wait_if_hot
            t0 = time.monotonic()
            wait_if_hot(self.pause_c, self.resume_c)
            self.heat_wait_s += time.monotonic() - t0
            t = self._temperature()
            if t is not None and t >= self.hard_stop_c:
                raise GpuTooHot(f'GPU at {t:.0f} C >= {self.hard_stop_c:.0f} C hard stop')
            # A flight may have started while cooling down.
            self.lock_wait_s += wait_for_flight_lock(self.flight_lock, log=self.log, sleep=self.sleep)
        self.chunk_start = time.monotonic()
        self.chunks += 1

    def chunk_expired(self) -> bool:
        if self.chunk_max_s is None or self.chunk_start is None:
            return False
        return time.monotonic() - self.chunk_start >= self.chunk_max_s

    def summary(self) -> dict:
        return dict(chunks=self.chunks, flight_lock_wait_s=round(self.lock_wait_s, 1),
                    gpu_heat_wait_s=round(self.heat_wait_s, 1), gpu=self.gpu,
                    flight_lock=None if self.flight_lock is None else str(self.flight_lock))
