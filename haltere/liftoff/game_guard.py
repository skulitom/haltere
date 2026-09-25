"""Keep the virtual gamepad away from the user's own games.

The ViGEm virtual Xbox pad is machine-wide: every XInput game sees it, not only
Liftoff in the Anode seat, so flight commands reach whatever game the user is
playing. A game here is a process whose image lives in a game-library folder
(Steam, Epic, Xbox, GOG, EA, Ubisoft, Riot). Liftoff counts only outside the
pad's own Windows session: our Liftoff runs in the seat beside the pad, a
Liftoff on another desktop is the user's. Games installed elsewhere (emulators,
standalone installs) are not recognised.
"""
from __future__ import annotations

import ntpath
import os
import sys

GAME_DIRS = ('\\steamapps\\common\\', '\\epic games\\', '\\xboxgames\\', '\\gog galaxy\\games\\',
             '\\gog games\\', '\\ea games\\', '\\ubisoft game launcher\\games\\', '\\riot games\\')
# Launchers, shared runtimes and installers that live inside those folders.
NOT_GAMES = ('\\steamapps\\common\\steamworks shared\\', '\\_commonredist\\', '\\epic games\\launcher\\',
             '\\riot games\\riot client\\', '\\steamapps\\common\\wallpaper_engine\\')
LIFTOFF_DIR = '\\steamapps\\common\\liftoff\\'


class GameDetected(RuntimeError):
    pass


def _norm(path):
    return (path or '').replace('/', '\\').lower()


def is_game_path(path):
    p = _norm(path)
    return any(d in p for d in GAME_DIRS) and not any(d in p for d in NOT_GAMES)


def game_processes(rows, own_session):
    """Game processes among ``rows`` (dicts with ProcessId, SessionId, ExecutablePath)."""
    games = []
    for row in rows:
        path = row.get('ExecutablePath') or ''
        if not is_game_path(path):
            continue
        if LIFTOFF_DIR in _norm(path) and row.get('SessionId') == own_session:
            continue
        games.append(dict(pid=int(row['ProcessId']), session=row.get('SessionId'),
                          executable=ntpath.basename(path)))
    return games


_K32 = None


def _kernel32():
    global _K32
    if _K32 is None:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.K32EnumProcesses.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        k32.K32EnumProcesses.restype = wintypes.BOOL
        k32.ProcessIdToSessionId.argtypes = [wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        k32.ProcessIdToSessionId.restype = wintypes.BOOL
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                   ctypes.POINTER(wintypes.DWORD)]
        k32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL
        _K32 = k32
    return _K32


def snapshot():
    """Every process this user can query, in all Windows sessions (a few ms, no subprocess)."""
    if sys.platform != 'win32':
        return []
    import ctypes
    from ctypes import wintypes
    k32 = _kernel32()
    size = 2048
    while True:
        pids = (wintypes.DWORD*size)()
        needed = wintypes.DWORD()
        if not k32.K32EnumProcesses(pids, ctypes.sizeof(pids), ctypes.byref(needed)):
            raise ctypes.WinError(ctypes.get_last_error())
        if needed.value < ctypes.sizeof(pids):
            break
        size *= 2
    rows = []
    buf = ctypes.create_unicode_buffer(32768)
    for pid in pids[:needed.value//ctypes.sizeof(wintypes.DWORD)]:
        session = wintypes.DWORD()
        if not pid or not k32.ProcessIdToSessionId(pid, ctypes.byref(session)):
            continue
        handle = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            continue
        try:
            n = wintypes.DWORD(len(buf))
            if k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(n)):
                rows.append(dict(ProcessId=int(pid), SessionId=int(session.value), ExecutablePath=buf.value))
        finally:
            k32.CloseHandle(handle)
    return rows


def own_session():
    if sys.platform != 'win32':
        return None
    import ctypes
    from ctypes import wintypes
    session = wintypes.DWORD()
    if not _kernel32().ProcessIdToSessionId(os.getpid(), ctypes.byref(session)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(session.value)


def running_games():
    return game_processes(snapshot(), own_session())


def describe(games):
    return ', '.join(f"{g['executable']} (PID {g['pid']}, session {g['session']})" for g in games)


def require_no_games(detector=running_games):
    games = detector()
    if games:
        raise GameDetected('A game is running and the virtual gamepad is machine-wide, so flight commands '
                           f'would reach it: {describe(games)}. Close it or wait until it has closed.')
