"""Keep the virtual gamepad away from the user's own games.

The ViGEm virtual Xbox pad is machine-wide: every XInput or DirectInput game sees
it, not only Liftoff in the Anode seat, so flight commands reach whatever game
the user is playing. Two signals, either one blocks:

- the image path: a game-library folder (Steam, Epic, C:\\XboxGames, GOG, EA,
  Ubisoft, Riot), a Blizzard/Battle.net install (a ``.build.info`` marker in an
  ancestor folder), a Store package with ``MicrosoftGame.config``, or a known
  emulator;
- outside the pad's own Windows session, a process that has a controller input
  library loaded (XInput, GameInput, DirectInput), except known non-game apps
  (Explorer, browsers, Steam, overlays) and Windows itself.

Liftoff counts only outside the pad's own session: our Liftoff runs in the seat
beside the pad, a Liftoff on another desktop is the user's. This is a heuristic
that fails safe for unknown controller users (they block) but can miss a game
that reads the pad through another API before it loads one of those libraries.
"""
from __future__ import annotations

import ntpath
import os
import sys
import time

GAME_DIRS = ('\\steamapps\\common\\', '\\epic games\\', '\\xboxgames\\', '\\gog galaxy\\games\\',
             '\\gog games\\', '\\ea games\\', '\\ubisoft game launcher\\games\\', '\\riot games\\')
# Launchers, shared runtimes and installers that live inside those folders.
NOT_GAMES = ('\\steamapps\\common\\steamworks shared\\', '\\_commonredist\\', '\\epic games\\launcher\\',
             '\\riot games\\riot client\\', '\\steamapps\\common\\wallpaper_engine\\', '\\battle.net\\')
STORE_DIRS = ('\\windowsapps\\', '\\modifiablewindowsapps\\')
STORE_GAMES = ('microsoft.minecraftuwp_',)
EMULATORS = ('retroarch', 'dolphin', 'pcsx2', 'yuzu', 'ryujinx', 'cemu', 'rpcs3', 'duckstation', 'ppsspp',
             'xemu', 'xenia', 'nox', 'noxvmhandle', 'hd-player')
LIFTOFF_DIR = '\\steamapps\\common\\liftoff\\'
INPUT_LIBRARIES = frozenset({'xinput1_3.dll', 'xinput1_4.dll', 'xinput9_1_0.dll', 'xinputuap.dll',
                             'gameinput.dll', 'dinput8.dll', 'dinput.dll'})
# Seen with controller libraries loaded on this machine during earlier flights; not games.
NOT_GAME_NAMES = frozenset({'explorer.exe', 'chrome.exe', 'msedge.exe', 'msedgewebview2.exe', 'firefox.exe',
                            'brave.exe', 'opera.exe', 'steam.exe', 'steamwebhelper.exe', 'nvidia overlay.exe',
                            'discord.exe', 'claude.exe', 'anode.exe'})
# Windows' own gaming overlay and Xbox app (Game Bar starts them when a controller connects).
NOT_GAME_PACKAGES = ('\\windowsapps\\microsoft.xboxgamingoverlay_', '\\windowsapps\\microsoft.gamingapp_',
                     '\\windowsapps\\microsoft.gamingservices', '\\windowsapps\\microsoft.xboxidentityprovider_')
RECHECK_AGES =(2., 5., 10., 20., 40.)  # seconds after a process is first seen; then every RECHECK_PERIOD
RECHECK_PERIOD = 60.


class GameDetected(RuntimeError):
    pass


def _norm(path):
    return (path or '').replace('/', '\\').lower()


_MARKERS = {}


def _install_marker(path, levels=3):
    """A Blizzard CASC install (.build.info) in one of the first `levels` ancestor folders (cached)."""
    folder = ntpath.dirname(path.replace('/', '\\'))
    for _ in range(levels):
        if folder not in _MARKERS:
            _MARKERS[folder] = os.path.exists(ntpath.join(folder, '.build.info'))
        if _MARKERS[folder]:
            return True
        parent = ntpath.dirname(folder)
        if parent == folder:
            break
        folder = parent
    return False


def _store_game(p, path):
    """A Store / Game Pass package that is a game: known package names or a GDK MicrosoftGame.config."""
    for d in STORE_DIRS:
        if d in p:
            start = p.index(d)+len(d)
            package = p[start:].split('\\', 1)[0]
            if package.startswith(STORE_GAMES):
                return True
            root = path.replace('/', '\\')[:start+len(package)]
            return os.path.exists(ntpath.join(root, 'MicrosoftGame.config'))
    return False


def is_game_path(path):
    p = _norm(path)
    if not p or any(d in p for d in NOT_GAMES):
        return False
    if any(d in p for d in GAME_DIRS):
        return True
    stem = ntpath.splitext(ntpath.basename(p))[0]
    if stem.startswith(EMULATORS):
        return True
    return _store_game(p, path) or _install_marker(path)


def game_processes(rows, own_session):
    """Game processes among ``rows`` (dicts with ProcessId, SessionId, ExecutablePath) by image path."""
    games = []
    for row in rows:
        path = row.get('ExecutablePath') or ''
        if not is_game_path(path):
            continue
        if LIFTOFF_DIR in _norm(path) and row.get('SessionId') == own_session:
            continue
        games.append(dict(pid=int(row['ProcessId']), session=row.get('SessionId'),
                          executable=ntpath.basename(path), reason='game install'))
    return games


def library_candidate(row, own_session):
    """Processes whose controller libraries count: other sessions, not Windows, not a known non-game app."""
    path = _norm(row.get('ExecutablePath'))
    windows = _norm(os.environ.get('SystemRoot', 'C:\\Windows'))+'\\'
    return (bool(path) and row.get('SessionId') != own_session and not path.startswith(windows)
            and ntpath.basename(path) not in NOT_GAME_NAMES and not any(p in path for p in NOT_GAME_PACKAGES))


_K32 = _PSAPI = None


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


def _psapi():
    global _PSAPI
    if _PSAPI is None:
        import ctypes
        from ctypes import wintypes
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.EnumProcessModulesEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD,
                                               ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
        psapi.EnumProcessModulesEx.restype = wintypes.BOOL
        psapi.GetModuleBaseNameW.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        psapi.GetModuleBaseNameW.restype = wintypes.DWORD
        _PSAPI = psapi
    return _PSAPI


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


def input_libraries(pid):
    """Controller input libraries loaded by `pid` (empty if it cannot be read)."""
    if sys.platform != 'win32':
        return set()
    import ctypes
    from ctypes import wintypes
    k32, psapi = _kernel32(), _psapi()
    handle = k32.OpenProcess(0x1000 | 0x0010, False, pid)   # QUERY_LIMITED_INFORMATION | VM_READ
    if not handle:
        return set()
    try:
        modules = (wintypes.HMODULE*2048)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModulesEx(handle, modules, ctypes.sizeof(modules), ctypes.byref(needed), 3):
            return set()
        buf = ctypes.create_unicode_buffer(260)
        found = set()
        for i in range(min(needed.value//ctypes.sizeof(wintypes.HMODULE), len(modules))):
            if psapi.GetModuleBaseNameW(handle, modules[i], buf, 260) and buf.value.lower() in INPUT_LIBRARIES:
                found.add(buf.value.lower())
        return found
    finally:
        k32.CloseHandle(handle)


def own_session():
    if sys.platform != 'win32':
        return None
    import ctypes
    from ctypes import wintypes
    session = wintypes.DWORD()
    if not _kernel32().ProcessIdToSessionId(os.getpid(), ctypes.byref(session)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(session.value)


def library_games(rows, own, libraries=input_libraries):
    """One full controller-library pass over `rows` (about 2 s for every process on this machine)."""
    return [dict(pid=int(r['ProcessId']), session=r.get('SessionId'), executable=ntpath.basename(r['ExecutablePath']),
                 reason='controller input: ' + ', '.join(sorted(found)))
            for r in rows if library_candidate(r, own) and (found := libraries(int(r['ProcessId'])))]


class GameWatch:
    """Detector for the pad watchdog: image paths on every call; controller libraries of processes outside the
    pad's session when first seen, again at RECHECK_AGES and then every RECHECK_PERIOD seconds, so a poll costs
    tens of milliseconds instead of a full module scan."""

    def __init__(self, own=None, snapshot=snapshot, libraries=input_libraries, clock=time.monotonic):
        self.own = own_session() if own is None else own
        self.snapshot, self.libraries, self.clock = snapshot, libraries, clock
        self.seen = {}      # (pid, path) -> (first seen, next check)
        self.flagged = {}   # (pid, path) -> game entry, until the process exits

    def __call__(self):
        rows = self.snapshot()
        games = game_processes(rows, self.own)
        now = self.clock()
        alive = set()
        for row in rows:
            key = (int(row['ProcessId']), row.get('ExecutablePath'))
            alive.add(key)
            if key in self.flagged or not library_candidate(row, self.own):
                continue
            first, due = self.seen.get(key, (now, now))
            if now < due:
                continue
            age = now-first
            later = [a for a in RECHECK_AGES if a > age]
            self.seen[key] = (first, first+later[0] if later else now+RECHECK_PERIOD)
            found = self.libraries(key[0])
            if found:
                self.flagged[key] = dict(pid=key[0], session=row.get('SessionId'), executable=ntpath.basename(key[1]),
                                         reason='controller input: ' + ', '.join(sorted(found)))
        for key in set(self.seen)-alive:
            del self.seen[key]
        for key in set(self.flagged)-alive:
            del self.flagged[key]
        return games+list(self.flagged.values())


def running_games():
    """A full check: image paths and every process's controller libraries."""
    rows, own = snapshot(), own_session()
    return game_processes(rows, own)+library_games(rows, own)


def describe(games):
    return ', '.join(f"{g['executable']} (PID {g['pid']}, session {g['session']}, {g.get('reason', 'game')})"
                     for g in games)


def require_no_games(detector=running_games):
    games = detector()
    if games:
        raise GameDetected('A game is running and the virtual gamepad is machine-wide, so flight commands '
                           f'would reach it: {describe(games)}. Close it or wait until it has closed.')
