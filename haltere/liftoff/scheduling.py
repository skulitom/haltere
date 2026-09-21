"""Explicit Windows scheduling for bounded controller/camera processes."""
import sys


def flight_process_priority():
    """Undo inherited background priority for time-sensitive flight work.

    Above-normal leaves Windows' high/realtime classes unused. This affects
    only the calling process; training, recording and unrelated apps keep their
    own priorities. Children call this separately when they own a camera loop.
    """
    if sys.platform!='win32':
        return dict(applied=False,platform=sys.platform)
    import ctypes
    kernel=ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.GetPriorityClass.argtypes=[ctypes.c_void_p]
    kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    handle=kernel.GetCurrentProcess()
    previous=kernel.GetPriorityClass(handle)
    if not kernel.SetPriorityClass(handle,0x8000):
        raise ctypes.WinError()
    current=kernel.GetPriorityClass(handle)
    if current!=0x8000:
        raise RuntimeError('Flight process is still subject to background scheduling')
    return dict(applied=True,previous=previous,current=current)
