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
    assert preflight.check_workloads()['passed']
    rows.append(dict(rows[0], ProcessId=43, SessionId=1, Name='NMS.exe',
                     ExecutablePath=r"C:\SteamLibrary\steamapps\common\No Man's Sky\Binaries\NMS.exe"))
    report = preflight.check_workloads([43])
    assert not report['passed'] and [b['pid'] for b in report['blockers']] == [43]
