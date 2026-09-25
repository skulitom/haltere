import sys
import threading
import types

import pytest

from haltere.liftoff.game_guard import GameDetected, game_processes


def row(pid, path, session=1):
    return dict(ProcessId=pid, SessionId=session, ExecutablePath=path)


def test_games_are_recognised_by_library_folder_and_liftoff_only_outside_the_pad_session():
    rows = [row(1, r"C:\SteamLibrary\steamapps\common\No Man's Sky\Binaries\NMS.exe"),
            row(2, r'C:\SteamLibrary\steamapps\common\Liftoff\Liftoff.exe', session=2),
            row(3, r'C:\SteamLibrary\steamapps\common\Liftoff\Liftoff.exe', session=1),
            row(4, r'C:\Program Files (x86)\Steam\steam.exe'),
            row(5, r'C:\Program Files (x86)\Steam\steamapps\common\Steamworks Shared\_CommonRedist\x.exe'),
            row(6, r'C:\Program Files (x86)\Epic Games\Launcher\Portal\Binaries\Win64\EpicGamesLauncher.exe'),
            row(7, 'D:/Games/Epic Games/Fortnite/FortniteClient.exe'),
            row(8, r'C:\XboxGames\Forza\Content\forza.exe'),
            row(9, r'C:\Program Files\Mozilla Firefox\firefox.exe'),
            row(10, None)]
    assert [g['pid'] for g in game_processes(rows, own_session=2)] == [1, 3, 7, 8]
    assert game_processes(rows, own_session=2)[0]['executable'] == 'NMS.exe'


class FakeTarget:
    def __init__(self, log):
        self.log = log
        log.append('plug')

    def left_joystick_float(self, **kw):
        self.log.append(('left', kw))

    def right_joystick_float(self, **kw):
        self.log.append(('right', kw))

    def update(self):
        self.log.append('update')

    def reset(self):
        self.log.append('reset')

    def __del__(self):
        self.log.append('unplug')


@pytest.fixture
def fake_vgamepad(monkeypatch):
    log = []
    module = types.SimpleNamespace(VX360Gamepad=lambda: FakeTarget(log))
    monkeypatch.setitem(sys.modules, 'vgamepad', module)
    return log


def test_pad_refuses_to_plug_in_while_a_game_runs(fake_vgamepad):
    from haltere.liftoff.gamepad import VirtualPad
    with pytest.raises(GameDetected, match='NMS.exe'):
        VirtualPad(detector=lambda: [dict(pid=1, session=1, executable='NMS.exe')])
    assert 'plug' not in fake_vgamepad


def test_pad_unplugs_itself_when_a_game_starts(fake_vgamepad):
    from haltere.liftoff.gamepad import VirtualPad
    games = []
    checked = threading.Event()

    def detector():
        checked.set()
        return list(games)

    pad = VirtualPad(detector=detector, poll_seconds=.01)
    pad.send(0., .1, .2, .3)
    games.append(dict(pid=7, session=1, executable='NMS.exe'))
    pad._watch.join(2.)
    assert not pad._watch.is_alive() and checked.is_set()
    assert fake_vgamepad[-2:] == ['reset', 'unplug'] or fake_vgamepad[-3:] == ['reset', 'update', 'unplug']
    with pytest.raises(GameDetected):
        pad.send(-1., 0., 0., 0.)
    with pytest.raises(GameDetected):
        pad.reconnect(pause=0.)
    pad.close()
    assert fake_vgamepad.count('plug') == 1


def test_a_failing_game_check_also_unplugs(fake_vgamepad):
    from haltere.liftoff.gamepad import VirtualPad
    calls = []

    def detector():
        calls.append(1)
        if len(calls) > 1:
            raise OSError('access denied')
        return []

    pad = VirtualPad(detector=detector, poll_seconds=.01)
    pad._watch.join(2.)
    with pytest.raises(GameDetected, match='game check failed'):
        pad.send(-1., 0., 0., 0.)


def test_preflight_blocks_a_game_but_not_our_seat_liftoff(monkeypatch):
    from haltere.liftoff import preflight
    rows = [dict(ProcessId=42, ParentProcessId=0, SessionId=2, Name='Liftoff.exe', CommandLine='',
                 ExecutablePath=r'C:\SteamLibrary\steamapps\common\Liftoff\Liftoff.exe',
                 UserModeTime=0, KernelModeTime=0)]
    monkeypatch.setattr(preflight, 'inventory', lambda: rows)
    monkeypatch.setattr(preflight.time, 'sleep', lambda _: None)
    monkeypatch.setattr(preflight, 'own_session', lambda: 2)
    monkeypatch.setattr(preflight, 'library_games', lambda rows, own: [])
    assert preflight.check_workloads()['passed']
    rows.append(dict(rows[0], ProcessId=43, SessionId=1, Name='NMS.exe',
                     ExecutablePath=r"C:\SteamLibrary\steamapps\common\No Man's Sky\Binaries\NMS.exe"))
    report = preflight.check_workloads([43])
    assert not report['passed'] and [b['pid'] for b in report['blockers']] == [43]


def test_battle_net_store_and_emulator_games_are_recognised(tmp_path):
    from haltere.liftoff.game_guard import is_game_path
    diablo = tmp_path/'GAMES'/'Diablo IV'
    (diablo/'x64').mkdir(parents=True)
    (diablo/'.build.info').write_text('')
    store = tmp_path/'WindowsApps'/'Contoso.Game_1.0_x64__abc'
    store.mkdir(parents=True)
    (store/'MicrosoftGame.config').write_text('')
    (tmp_path/'WindowsApps'/'Microsoft.WindowsTerminal_1_x64__8we').mkdir()
    assert is_game_path(str(diablo/'Diablo IV.exe')) and is_game_path(str(diablo/'x64'/'Diablo IV64.exe'))
    assert is_game_path(str(store/'Game.exe'))
    assert is_game_path(r'C:\Program Files\WindowsApps\Microsoft.MinecraftUWP_1.21_x64__8we\Minecraft.Windows.exe')
    assert is_game_path('C:/RetroArch-Win64/retroarch.exe')
    assert not is_game_path(str(tmp_path/'WindowsApps'/'Microsoft.WindowsTerminal_1_x64__8we'/'WindowsTerminal.exe'))
    assert not is_game_path(r'C:\Program Files (x86)\Battle.net\Battle.net.exe')
    assert not is_game_path(str(tmp_path/'Tools'/'editor.exe'))


def test_controller_library_users_outside_the_pad_session_block_except_known_apps():
    from haltere.liftoff.game_guard import library_games
    rows = [row(1, r'D:\Games\Standalone\game.exe', session=1),
            row(2, r'C:\Program Files\Google\Chrome\Application\chrome.exe', session=1),
            row(3, r'C:\Windows\explorer.exe', session=1),
            row(4, r'C:\SteamLibrary\steamapps\common\Liftoff\Liftoff.exe', session=2),
            row(5, r'D:\Tools\quiet.exe', session=1)]
    libraries = {1: {'xinput1_4.dll'}, 2: {'xinput1_4.dll'}, 3: {'xinput1_4.dll'}, 4: {'xinput1_3.dll'}}
    games = library_games(rows, 2, libraries=lambda pid: libraries.get(pid, set()))
    assert [g['pid'] for g in games] == [1] and 'xinput1_4.dll' in games[0]['reason']


def test_game_watch_rechecks_new_processes_until_their_controller_library_loads():
    from haltere.liftoff.game_guard import GameWatch
    clock, loaded, calls = [0.], set(), []
    rows = [row(1, r'D:\Games\Standalone\game.exe', session=1)]

    def libraries(pid):
        calls.append(clock[0])
        return {'xinput1_4.dll'} if pid in loaded else set()

    watch = GameWatch(own=2, snapshot=lambda: rows, libraries=libraries, clock=lambda: clock[0])
    assert watch() == [] and calls == [0.]
    clock[0] = 1.
    assert watch() == [] and calls == [0.]          # not due yet: polls stay cheap
    loaded.add(1)
    clock[0] = 2.
    assert [g['pid'] for g in watch()] == [1]
    clock[0] = 3.
    assert [g['pid'] for g in watch()] == [1] and calls == [0., 2.]   # stays flagged without a rescan
    rows.clear()
    watch()
    assert watch.seen == {} and watch.flagged == {}


def test_game_watch_rescans_quiet_processes_once_a_minute():
    from haltere.liftoff.game_guard import GameWatch
    clock, calls = [0.], []
    rows = [row(1, r'D:\Tools\quiet.exe', session=1)]
    watch = GameWatch(own=2, snapshot=lambda: rows, libraries=lambda pid: calls.append(clock[0]) or set(),
                      clock=lambda: clock[0])
    for t in range(0, 200):
        clock[0] = float(t)
        watch()
    assert calls == [0., 2., 5., 10., 20., 40., 100., 160.]
