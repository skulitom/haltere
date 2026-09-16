"""The by-sight pilot must decide where to fly from what it can see, and nothing else.

That claim is the whole point of the project, and it is the easy one to lose by accident: a gate list
loaded "just for the altitude", a track file read "only to seed the search", the game's own
next-checkpoint marker wired in "temporarily". Each would make every by-sight number a lie, and none
would fail a test that only checks gates flown through. So the claim is a test.

What the pilot legitimately takes from telemetry is its own pose, velocity and body rates - what a quad
has from its IMU and baro and a fly from its halteres. That is self-knowledge, not course knowledge, and
it is not what this file guards.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'haltere'

# Modules that run inside the by-sight flight loop. pilot.py is deliberately absent: it also hosts the
# taught-line --follow pilot, which is allowed to read a track file because it is not flying by sight.
BY_SIGHT_MODULES = [ROOT / 'liftoff' / 'sightpilot.py', ROOT / 'vision' / 'runtime.py']

COURSE_FILES = ['gates_strawbale', 'track_strawbale', 'obstacles_strawbale', 'gates_pinevalley',
                'load_gate_file', 'gate_passes']


@pytest.mark.parametrize('path', BY_SIGHT_MODULES, ids=lambda p: p.name)
def test_the_by_sight_loop_never_reads_the_course(path):
    src = path.read_text(encoding='utf-8')
    found = [name for name in COURSE_FILES if name in src]
    assert not found, (
        f'{path.name} reaches for {found}. A pilot that reads the course it is meant to be seeing is not '
        f'flying by sight, and every number measured with it would be wrong.')


@pytest.mark.parametrize('path', BY_SIGHT_MODULES, ids=lambda p: p.name)
def test_the_by_sight_loop_never_reads_the_games_marker(path):
    """Liftoff draws a beacon on the next checkpoint. Steering to it is reading the answer off the screen."""
    src = path.read_text(encoding='utf-8')
    assert 'beacon' not in src.lower(), (
        f'{path.name} mentions the beacon. It is ground truth for building training labels, never a flight cue.')


def test_the_pilot_does_not_open_files_while_flying():
    """Whatever it needs, it has before take-off: nothing is read from disk once it is airborne.

    Only the SightPilot class is checked. Parsing --sight-set at start-up is allowed to read YAML; the
    flight loop is not.
    """
    import ast
    src = (ROOT / 'liftoff' / 'sightpilot.py').read_text(encoding='utf-8')
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == 'SightPilot')
    called = set()
    for node in ast.walk(cls):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                called.add(f.id)
            elif isinstance(f, ast.Attribute):
                called.add(f.attr)
    forbidden = called & {'open', 'load', 'safe_load', 'loads', 'read_text', 'read_bytes'}
    assert not forbidden, f'SightPilot calls {forbidden}: the flight loop must not read anything from disk'
