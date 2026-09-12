"""Keep the GPU cool: poll nvidia-smi and pause while the temperature is above a limit."""
from __future__ import annotations

import subprocess
import time


def gpu_temperature() -> float | None:
    """Current GPU temperature in C from nvidia-smi (None when unavailable)."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def wait_if_hot(limit: float, resume_below: float | None = None, poll: float = 5.0) -> float:
    """Block while the GPU is hotter than `limit`, until it cools below `resume_below` (limit - 8 by default).
    Returns the seconds spent waiting."""
    t = gpu_temperature()
    if t is None or t <= limit:
        return 0.0
    resume_below = limit - 8.0 if resume_below is None else resume_below
    t0 = time.time()
    print(f'GPU at {t:.0f} C > {limit:.0f} C: pausing until it is below {resume_below:.0f} C', flush=True)
    while t is not None and t > resume_below:
        time.sleep(poll)
        t = gpu_temperature()
    waited = time.time() - t0
    print(f'GPU at {t} C: resuming after {waited:.0f} s', flush=True)
    return waited
