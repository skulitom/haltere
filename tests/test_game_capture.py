from unittest.mock import Mock

import numpy as np
import pytest

from haltere.liftoff import game_capture


def setup_window(monkeypatch):
    monkeypatch.setattr(game_capture,'find_game_window',lambda title:123)
    monkeypatch.setattr(game_capture,'find_window_rect',lambda title:(0,0,1280,720))
    monkeypatch.setattr(game_capture,'game_window_active',lambda hwnd:True)


def test_dxgi_discards_buffered_old_frames_and_preserves_presentation_timestamp(monkeypatch):
    setup_window(monkeypatch)
    monkeypatch.setattr(game_capture.time,'monotonic',Mock(side_effect=[10.,10.1]))
    camera=Mock()
    rgb=np.zeros((4,4,3),dtype=np.uint8)
    camera.get_latest_frame.side_effect=[(rgb,9.9),(rgb,10.05)]
    with game_capture.DxGameCapture(camera=camera) as capture:
        stamp,image=capture.read()
        assert stamp==10.05 and image is rgb
        assert camera.get_latest_frame.call_count==2
    camera.start.assert_called_once_with(region=(0,0,1280,720),target_fps=60,video_mode=False)
    camera.release.assert_called_once()


def test_dxgi_discards_focus_changes_and_closes_capture(monkeypatch):
    setup_window(monkeypatch)
    active=Mock(side_effect=[True,True,False])
    monkeypatch.setattr(game_capture,'game_window_active',active)
    camera=Mock()
    camera.get_latest_frame.return_value=(np.ones((4,4,3)),100.)
    capture=game_capture.DxGameCapture(camera=camera)
    with pytest.raises(RuntimeError,match='hidden or moved'):
        capture.read()
    capture.close()
    camera.release.assert_called_once()


def test_dxgi_never_starts_when_game_is_hidden(monkeypatch):
    setup_window(monkeypatch)
    monkeypatch.setattr(game_capture,'game_window_active',lambda hwnd:False)
    camera=Mock()
    with pytest.raises(RuntimeError,match='foreground'):
        game_capture.DxGameCapture(camera=camera)
    camera.start.assert_not_called()
