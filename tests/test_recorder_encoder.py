from types import SimpleNamespace

import pytest

from haltere.liftoff import recorder


def test_recording_cannot_succeed_without_ffmpeg(monkeypatch):
    monkeypatch.setattr(recorder.shutil, 'which', lambda _: None)
    with pytest.raises(RuntimeError, match='unavailable'):
        recorder.validate_encoder('h264_nvenc')


def test_nvenc_failure_does_not_fall_back_to_cpu(monkeypatch):
    monkeypatch.setattr(recorder.shutil, 'which', lambda _: 'ffmpeg')
    calls = []
    def fail(command, **kw):
        calls.append(command)
        return SimpleNamespace(returncode=1, stderr='No capable devices found')
    monkeypatch.setattr(recorder.subprocess, 'run', fail)
    with pytest.raises(RuntimeError, match='No capable devices'):
        recorder.validate_encoder('h264_nvenc')
    assert len(calls) == 1
    assert 'h264_nvenc' in calls[0] and 'libx264' not in calls[0]


def test_unsupported_encoder_rejected_before_start():
    with pytest.raises(ValueError, match='Unsupported'):
        recorder.encoder_options('typo')
