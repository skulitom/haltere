"""Timestamped DXGI capture of the foreground game inside the current session."""
import threading
import time

from .commands import find_game_window, game_window_active
from .recorder import find_window_rect


class DxGameCapture:
    def __init__(self,title='Liftoff',fps=60,camera=None):
        self.title=title
        self.hwnd=find_game_window(title)
        self.rect=find_window_rect(title)
        if not game_window_active(self.hwnd) or self.rect is None:
            raise RuntimeError('Game must be foreground before starting DXGI capture')
        self.closed=threading.Event()
        self.close_lock=threading.Lock()
        if camera is None:
            import dxcam
            camera=dxcam.create(output_color='RGB',max_buffer_len=2,backend='dxgi')
        self.camera=camera
        left,top,width,height=self.rect
        try:
            camera.start(region=(left,top,left+width,top+height),target_fps=fps,video_mode=False)
        except Exception:
            camera.release()
            raise

    def _valid_window(self):
        return (game_window_active(self.hwnd) and find_game_window(self.title)==self.hwnd
                and find_window_rect(self.title)==self.rect)

    def read(self):
        if not self._valid_window():
            self.close()
            raise RuntimeError('Game hidden or moved during DXGI capture')
        verified_at=time.monotonic()
        while not self.closed.is_set():
            result=self.camera.get_latest_frame(with_timestamp=True)
            if result is None:
                return None
            if not self._valid_window():
                self.close()
                raise RuntimeError('Game hidden or moved during DXGI capture')
            rgb,captured_at=result
            # Never consume a queued frame from before the foreground check.
            # Use the original Windows presentation time, never delivery time.
            if verified_at<=captured_at<=time.monotonic():
                return captured_at,rgb
        return None

    def close(self):
        with self.close_lock:
            if not self.closed.is_set():
                self.closed.set()
                self.camera.release()  # wakes a reader blocked waiting for a frame

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()
