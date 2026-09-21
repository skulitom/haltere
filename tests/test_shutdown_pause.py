from types import SimpleNamespace
import numpy as np
import pytest
from haltere.liftoff.visual_brain import telemetry_still_progressing


@pytest.mark.parametrize('after_time,after_live,expected', [
    (1.2,True,True), (1.,True,False), (.1,True,False), (1.2,False,False), (None,True,False),
])
def test_pause_confirmation_uses_fresh_progress_not_the_cached_live_frame(monkeypatch, after_time, after_live, expected):
    def frame(t, live=True):
        return SimpleNamespace(timestamp=t, position=np.array([1.,2.,3.]) if live else np.zeros(3),
                               attitude=np.array([0.,0.,0.,1.]),input=np.zeros(4))
    before=frame(1.);after=frame(after_time,after_live) if after_time is not None else None
    frames=iter([before,after]);receiver=SimpleNamespace(poll=lambda:next(frames),last=before)
    waited=[]
    monkeypatch.setattr('haltere.liftoff.visual_brain.time.sleep',waited.append)
    assert telemetry_still_progressing(receiver) is expected
    assert waited==[.2]
