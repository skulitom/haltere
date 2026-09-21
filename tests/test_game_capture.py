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


def test_camera_process_keeps_original_frame_time_and_drops_old_backlog():
    import multiprocessing as mp
    from queue import Queue
    from threading import Event
    from types import SimpleNamespace
    from haltere.liftoff.camera_process import ProcessRetinaCamera,put_latest
    camera=ProcessRetinaCamera.__new__(ProcessRetinaCamera)
    camera.queue=Queue(maxsize=2)
    camera.data=mp.get_context('spawn').Array('d',727,lock=True)
    camera.done=Event()
    camera.process=SimpleNamespace(exitcode=None)
    camera._latest=camera._error=None
    camera._diagnostics={}
    for stamp in [1.,2.,3.]:
        put_latest(camera.queue,{'diagnostics':{'frames':stamp}})
        shared=np.frombuffer(camera.data.get_obj(),dtype=np.float64)
        shared[0]=stamp
        shared[7:]=stamp
    assert camera.queue.qsize()==2
    stamp,retina,detection=camera.latest
    assert stamp==3. and (retina==3.).all() and detection is None
    assert camera._diagnostics['frames']==3.
    # Reading a cached packet does not give it a new receipt-time timestamp.
    assert camera.latest[0]==3.
    camera.process.exitcode=1
    assert 'exited (1)' in camera.error
