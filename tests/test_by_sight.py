"""The by-sight pilot must decide where to fly from what it can see, and nothing else.

That is the declared input contract of these particular modules, not a ban on
other explicitly labelled helpers or generic visible race cues. A gate list
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
BY_SIGHT_MODULES = [ROOT / 'liftoff' / 'sightpilot.py', ROOT / 'vision' / 'runtime.py',
                   ROOT / 'liftoff' / 'visual_assistance.py']

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
    """These runners do not explicitly consume markers; keep that stated contract accurate."""
    src = path.read_text(encoding='utf-8')
    assert 'beacon' not in src.lower(), (
        f'{path.name} mentions the beacon. Declare a new cue-assisted mode before changing this input contract.')


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


def test_gate_clearance_is_measured_on_the_course_being_flown():
    """The optic-flow speed sense needs height above terrain, and used to get it from a Straw Bale constant.

    The brain flies on that sense, so a wrong clearance is not a navigation shortcut - it corrupts the input.
    It must come from the course under the drone: the height at which it crosses the first gate.
    """
    from types import SimpleNamespace

    from haltere.liftoff.pilot import TelemetryPilot

    p = TelemetryPilot.__new__(TelemetryPilot)
    p._gate_clearance = None
    sp = SimpleNamespace(n_passes=0, z_pass_last=1.2, params=SimpleNamespace(z_pass0=1.2))

    assert p.gate_clearance(sp) == 1.2, 'before any gate, the prior stands in'

    sp.n_passes, sp.z_pass_last = 1, 4.6            # this course's gates stand 4.6 m up, not 1.2
    assert p.gate_clearance(sp) == 4.6, 'the first gate flown through sets the clearance'

    sp.z_pass_last = 11.0                            # a later, higher gate must not move it
    assert p.gate_clearance(sp) == 4.6, 'the clearance is latched at the first gate, not the last'
