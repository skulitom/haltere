import pytest


@pytest.fixture(autouse=True)
def no_anode(monkeypatch):
    """Tests never ask the real Anode about virtual pads; a test that needs Anode's report passes or patches one."""
    from haltere.liftoff import gamepad, preflight
    monkeypatch.setattr(gamepad, 'anode_isolation', lambda: None)
    monkeypatch.setattr(preflight, 'anode_isolation', lambda: None)
