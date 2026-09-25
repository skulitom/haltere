"""Virtual Xbox 360 controller (ViGEmBus via ``vgamepad``) used to fly Liftoff.

Mode-2 layout, which is what Liftoff's controller wizard expects from a gamepad:
left stick x = yaw, left stick y = throttle, right stick x = roll, right stick y = pitch.
"""
from __future__ import annotations

import math
import sys
import threading
import time

from .game_guard import (GameDetected, GameWatch, anode_isolation, describe, require_no_games, running_games,
                         seat_keeps_new_pads, seat_only_pads)

INSTALL_HELP = """
vgamepad / ViGEmBus is not available. To drive Liftoff you need the ViGEmBus driver (admin install):
  1. Download and run ViGEmBus_1.22.0_x64_x86_arm64.exe from
     https://github.com/nefarius/ViGEmBus/releases/tag/v1.22.0
  2. Then, in this project's environment:
       set VGAMEPAD_SKIP_VIGEMBUS_INSTALL=true
       uv pip install --python .venv/Scripts/python.exe vgamepad
  3. In Steam, open Liftoff's properties -> Controller and set 'Override for Liftoff' to
     'Disable Steam Input' (Liftoff's input library does not work behind Steam Input).
"""


class UdpSticks:
    """Drop-in replacement for VirtualPad that sends sticks as 4 float32 (Liftoff Input order:
    throttle, yaw, pitch, roll) to a UDP port - used with ``haltere liftoff fake``."""

    def __init__(self, host: str = '127.0.0.1', port: int = 9002):
        import socket
        import struct
        self._struct = struct
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (host, port)

    def send(self, throttle: float, roll: float, pitch: float, yaw: float) -> None:
        self.sock.sendto(self._struct.pack('<4f', throttle, yaw, pitch, roll), self.addr)

    def neutral(self) -> None:
        self.send(-1.0, 0.0, 0.0, 0.0)

    def press(self, button: str = 'A', seconds: float = 0.15) -> None:
        self.sock.sendto(b'PRESS ' + button.encode(), self.addr)

    def reconnect(self) -> None:
        """Ask the bridge to unplug and re-plug its virtual pad (Liftoff sometimes drops the binding)."""
        self.sock.sendto(b'RECONNECT', self.addr)

    def close(self) -> None:
        self.sock.close()


class VirtualPad:
    """While Anode keeps the pad inside its seat (HidHide), the user's games cannot open it and it just flies.
    Otherwise the pad reaches every game on the machine, so it refuses to plug in while a game is running and
    unplugs itself, without pausing Liftoff, as soon as one starts (see game_guard)."""

    def __init__(self, detector=None, poll_seconds: float = 1.0, isolation=None, confirm_seconds: float = 5.):
        """`detector` returns running games; by default a full check before plugging in, then a GameWatch.
        `isolation` returns Anode's report on virtual pads (None without Anode); by default ``anode gamepad state``."""
        isolation = isolation or anode_isolation
        expect_seat_only = seat_keeps_new_pads(isolation())
        if not expect_seat_only:
            require_no_games(detector or running_games)
        try:
            import vgamepad as vg
        except Exception as e:  # ImportError or ViGEm client errors
            raise RuntimeError(INSTALL_HELP) from e
        self._vg = vg
        self._detector, self._poll = detector, poll_seconds
        self._isolation, self._confirm_seconds = isolation, confirm_seconds
        self._lock = threading.RLock()
        self._tripped = ''
        self._stop = threading.Event()
        self._watch = None
        self.pad = vg.VX360Gamepad()
        self.neutral()
        self.seat_only = expect_seat_only and self._confirm_seat_only()
        if self.seat_only:
            print(f'virtual pad {self.device} stays inside the Anode seat; the game guard stands down', file=sys.stderr, flush=True)
        else:
            self._guard(check_now=expect_seat_only)

    @property
    def device(self) -> str:
        """The pad's Plug and Play name, as Anode reports it: ViGEm names a pad after its serial."""
        return 'USB\\VID_045E&PID_028E\\' + f'{self.pad.get_index():02d}'

    def _confirm_seat_only(self) -> bool:
        """Waits for Anode to report this pad kept inside the seat and checked from the user's desktop."""
        deadline = time.monotonic()+self._confirm_seconds
        while True:
            state = self._isolation()
            if seat_keeps_new_pads(state) and self.device in seat_only_pads(state):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(.25)

    def _guard(self, check_now: bool = False) -> None:
        """The pad reaches every game on the machine: unplug now if one is running, and watch for one starting."""
        if check_now and (games := (self._detector or running_games)()):
            self.trip(describe(games))
            raise GameDetected(f'Anode could not confirm the virtual pad stays inside its seat, and a game is running: {describe(games)}')
        if self._watch is None or not self._watch.is_alive():
            self._watch = threading.Thread(target=self._watch_games, args=(self._detector or GameWatch(), self._poll),
                                           name='game-guard', daemon=True)
            self._watch.start()

    def _watch_games(self, detector, poll_seconds):
        while not self._stop.wait(poll_seconds):
            try:
                games = detector()
            except Exception as e:  # a failed check must not leave the pad attached unguarded
                games = [dict(pid=-1, session=None, executable=f'game check failed: {e}')]
            if games:
                self.trip(describe(games))
                return

    def trip(self, reason: str) -> None:
        """Release all inputs and unplug the pad for good."""
        with self._lock:
            if self._tripped:
                return
            self._tripped = reason
            try:
                self.pad.reset()
                self.pad.update()
            except Exception:
                pass
            self.pad = None               # vgamepad removes the ViGEm target on destruction
        print(f'virtual pad unplugged, a game is running: {reason}', file=sys.stderr, flush=True)

    def _require_live(self):
        if self._tripped:
            raise GameDetected(f'virtual pad unplugged because a game is running: {self._tripped}')

    @staticmethod
    def _clip(x: float) -> float:
        return max(-1.0, min(1.0, float(x)))

    def send(self, throttle: float, roll: float, pitch: float, yaw: float) -> None:
        """All values in [-1, 1]; throttle -1 = idle."""
        with self._lock:
            self._require_live()
            self.pad.left_joystick_float(x_value_float=self._clip(yaw), y_value_float=self._clip(throttle))
            self.pad.right_joystick_float(x_value_float=self._clip(roll), y_value_float=self._clip(pitch))
            self.pad.update()

    def neutral(self) -> None:
        self.send(-1.0, 0.0, 0.0, 0.0)

    def press(self, button: str = 'A', seconds: float = 0.1) -> None:
        b = getattr(self._vg.XUSB_BUTTON, f'XUSB_GAMEPAD_{button.upper()}')
        with self._lock:
            self._require_live()
            self.pad.press_button(button=b)
            self.pad.update()
            time.sleep(seconds)
            self.pad.release_button(button=b)
            self.pad.update()

    def sweep(self, axis: str, seconds: float = 3.0, hz: float = 100.0) -> None:
        """Move one axis through its full range (sine), keeping the others neutral (for the wizard)."""
        t0 = time.time()
        while time.time() - t0 < seconds:
            x = math.sin(2 * math.pi * (time.time() - t0) / 1.5)
            vals = {'throttle': -1.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
            vals[axis] = x
            self.send(**vals)
            time.sleep(1.0 / hz)
        self.neutral()

    def reconnect(self, pause: float = 1.0) -> None:
        """Unplug the virtual pad and plug a fresh one in. Liftoff drops its binding to the pad now and then
        (after the window lost the focus, it seems); a re-plug makes the game pick it up again."""
        with self._lock:
            self._require_live()
            try:
                self.pad.reset()
                self.pad.update()
            except Exception:
                pass
            self.pad = None               # vgamepad removes the ViGEm target on destruction
            time.sleep(pause)
            self.pad = self._vg.VX360Gamepad()
            self.neutral()
            if self.seat_only and not self._confirm_seat_only():
                self.seat_only = False
                self._guard(check_now=True)

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            if self._tripped:
                return
            try:
                self.neutral()
                self.pad.reset()
                self.pad.update()
            except Exception:
                pass
