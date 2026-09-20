from unittest.mock import Mock

import numpy as np

from haltere.liftoff import commands, recorder


def test_recorder_captures_only_foreground_game_and_discards_focus_changes(monkeypatch):
    screen = Mock()
    screen.grab.return_value = np.zeros((100, 200, 4), dtype=np.uint8)
    screen.grab.return_value[..., :3] = [10, 20, 30]
    monkeypatch.setattr(commands, 'find_game_window', lambda title: 123)
    active = Mock(return_value=False)
    monkeypatch.setattr(commands, 'game_window_active', active)
    monkeypatch.setattr(recorder, 'find_window_rect', lambda title: (10, 20, 200, 100))
    assert recorder._capture_game_frame(screen, 'Liftoff') is None
    screen.grab.assert_not_called()
    active.return_value = True
    image = recorder._capture_game_frame(screen, 'Liftoff')
    screen.grab.assert_called_once_with(dict(left=10, top=20, width=200, height=100))
    np.testing.assert_array_equal(image[0, 0], [30, 20, 10])
    active.side_effect = [True, False]
    assert recorder._capture_game_frame(screen, 'Liftoff') is None


def test_recorder_missing_window_never_captures_desktop(monkeypatch):
    screen = Mock()
    monkeypatch.setattr(commands, 'find_game_window', lambda title: 0)
    assert recorder._capture_game_frame(screen, 'Liftoff') is None
    screen.grab.assert_not_called()
