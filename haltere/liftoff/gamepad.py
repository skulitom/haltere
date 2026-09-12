"""Virtual Xbox 360 controller (ViGEmBus via ``vgamepad``) used to fly Liftoff.

Mode-2 layout, which is what Liftoff's controller wizard expects from a gamepad:
left stick x = yaw, left stick y = throttle, right stick x = roll, right stick y = pitch.
"""
from __future__ import annotations

import math
import time

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
    def __init__(self):
        try:
            import vgamepad as vg
        except Exception as e:  # ImportError or ViGEm client errors
            raise RuntimeError(INSTALL_HELP) from e
        self._vg = vg
        self.pad = vg.VX360Gamepad()
        self.neutral()

    @staticmethod
    def _clip(x: float) -> float:
        return max(-1.0, min(1.0, float(x)))

    def send(self, throttle: float, roll: float, pitch: float, yaw: float) -> None:
        """All values in [-1, 1]; throttle -1 = idle."""
        self.pad.left_joystick_float(x_value_float=self._clip(yaw), y_value_float=self._clip(throttle))
        self.pad.right_joystick_float(x_value_float=self._clip(roll), y_value_float=self._clip(pitch))
        self.pad.update()

    def neutral(self) -> None:
        self.send(-1.0, 0.0, 0.0, 0.0)

    def press(self, button: str = 'A', seconds: float = 0.1) -> None:
        b = getattr(self._vg.XUSB_BUTTON, f'XUSB_GAMEPAD_{button.upper()}')
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
        try:
            self.pad.reset()
            self.pad.update()
        except Exception:
            pass
        del self.pad                      # vgamepad removes the ViGEm target on destruction
        time.sleep(pause)
        self.pad = self._vg.VX360Gamepad()
        self.neutral()

    def close(self) -> None:
        try:
            self.neutral()
            self.pad.reset()
            self.pad.update()
        except Exception:
            pass
